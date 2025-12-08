//! The [STIR](Stir) polynomial low-degree test over the
//! [extension field](XFieldElement).

use itertools::Itertools;
use num_traits::ConstOne;
use num_traits::One;
use twenty_first::math::traits::FiniteField;
use twenty_first::prelude::*;

use crate::arithmetic_domain::ArithmeticDomain;
use crate::error::StirParameterError;
use crate::error::StirProvingError;
use crate::error::StirVerificationError;
use crate::error::U32_TO_USIZE_ERR;
use crate::error::USIZE_TO_U64_ERR;
use crate::proof_item::ProofItem;
use crate::proof_item::StirResponse;
use crate::proof_stream::ProofStream;
use crate::table::master_table::BfeSlice;

pub type AuthenticationStructure = Vec<Digest>;

type SetupResult<T> = Result<T, StirParameterError>;
type ProverResult<T> = Result<T, StirProvingError>;
type VerifierResult<T> = Result<T, StirVerificationError>;

/// An [`ArithmeticDomain`] can have at most 2^32 elements. Converting a usize
/// that represents a (valid) domain index to a u32 can never fail.
const DOMAIN_INDEX_TO_U32_ERR: &str = "internal error: domain index should be a valid u32";

/// The quotient set used in any round of STIR must never be larger than or
/// equal to the degree of the polynomial of that round (see also
/// [Stir::round_params]).
/// The polynomial degree, in turn, is upper bounded by the length of the
/// [domain](ArithmeticDomain) for that round, which is at most 2^32.
/// Consequently, converting the length of the quotient set to a u32 can never
/// fail.
const QUOTIENT_SET_LEN_TO_U32_ERR: &str =
    "internal error: length of quotient set should be a valid u32";

/// The [STIR] (Shift To Improve Rate) low-degree test.
///
/// STIR is an Interactive Oracle Proof of Proximity for Reed-Solomon codes.
/// When combined with the [BCS] transform and the [Fiat-Shamir] heuristic,
/// the result is a non-interactive system for proving that the initial input
/// codeword corresponds to a polynomial of low degree.
///
/// This struct defines the protocol parameters. The central methods are
/// [prove](Self::prove) and [verify](Self::verify).
///
/// [STIR]: https://eprint.iacr.org/2024/390.pdf
/// [BCS]: https://eprint.iacr.org/2016/116
/// [Fiat-Shamir]: https://link.springer.com/chapter/10.1007/3-540-47721-7_12
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
pub struct Stir {
    /// The desired security level in bits.
    ///
    /// Concretely, the system
    /// - is perfectly complete, and
    /// - has soundness error 2^(-security_level).
    #[cfg_attr(test, strategy(16_usize..=192))]
    security_level: usize,

    soundness: SoundnessAssumption,

    /// Corresponds to the (log₂ of the) paper’s folding factor `k`.
    ///
    /// Must be greater than or equal to 2, _i.e._, `k` must be greater than or
    /// equal to 4.
    //
    // The paper allows for this to change between rounds. This (current)
    // implementation does not.
    #[cfg_attr(test, strategy(2_usize..=5))]
    log2_folding_factor: usize,

    /// The amount of “redundancy” in the [prover](Self::prove)’s input.
    ///
    /// In particular, the Reed-Solomon code’s rate is the reciprocal of 2
    /// raised to this value. In other words, the initial rate equals
    /// `1 / 2^initial_log2_expansion_factor`.
    ///
    /// Must be greater than 0.
    #[cfg_attr(test, strategy(1_usize..=6))]
    log2_initial_expansion_factor: usize,

    /// The (log₂ of the) polynomial degree that is considered “high” for
    /// this STIR instance.
    ///
    /// In other words, the low-degreeness of polynomials with degree
    /// 2^log2_high_degree and higher cannot be [proven](Self::prove) (in a way
    /// that the [verifier](Self::verify) accepts).
    ///
    /// Must be greater than or equal to the (log₂ of the)
    /// [folding factor](Self::log2_folding_factor).
    #[cfg_attr(test, strategy(#log2_folding_factor..=15))]
    log2_high_degree: usize,
}

/// The type of soundness assumption you are willing to make for [STIR](Stir).
///
/// The choice influences the derivation of additional parameters used in STIR,
/// like the [number of queries](NumQueries) per round (called “t_i” in the
/// paper). Generally, the more “daring” the assumption, the lower the runtime
/// cost and proof size.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
pub enum SoundnessAssumption {
    /// Only use provable results.
    ///
    /// In particular, assume that the distance of each oracle is within the
    /// Johnson bound, (1 - √ρ).
    #[default]
    Provable,

    /// Use the conjecture that Reed-Solomon codes are list-decodable up to
    /// capacity and have correlated agreement up to capacity.
    ///
    /// In particular, assume that the distance of each oracle is within the
    /// capacity bound, (1 - ρ).
    Conjectured,
}

/// The parameters derived from a [STIR](Stir) instance.
#[derive(Debug, Clone, Eq, PartialEq)]
struct RoundParams {
    /// The number of queries for each of the full rounds.
    ///
    /// The length of the vector corresponds to the number of full rounds.
    ///
    /// Note that there is an additional, final round. The final round is not
    /// a full round since it doesn’t contain a quotienting step.
    round_queries: Vec<NumQueries>,

    /// The number of in-domain queries for the final round.
    ///
    /// Corresponds to the paper’s “repetition parameter” t_M.
    final_num_in_domain_queries: usize,

    /// The degree of the final polynomial.
    ///
    /// Because folding can never (guarantee to) produce the zero-polynomial,
    /// this is not of type [isize].
    final_degree: usize,
}

/// The number of oracle queries in a single, full round of [STIR](Stir).
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct NumQueries {
    /// Corresponds to the paper’s “repetition parameter” t_i.
    in_domain: usize,

    /// Corresponds to the paper’s “repetition parameter” `s_i`.
    out_of_domain: usize,
}

/// A Merkle tree where the leafs are “stacks” of values.
///
/// Allows shorter Merkle inclusion proofs in [STIR](Stir): since the verifier
/// requires all entries in a stack if it requires any entry in the stack, the
/// entire stack might as well be the Merkle leaf’s underlying value.
struct LeafStackMerkleTree {
    stacked_leafs: Vec<Vec<XFieldElement>>,
    tree: MerkleTree,
}

/// A Merkle tree inclusion proof for all queries to the folding polynomial
/// of one verifier round.
///
/// This is an extended version of a [`MerkleTreeInclusionProof`] that also
/// tracks the actual queries to the folding polynomial the prover has
/// commited to.
struct PolyFoldQueriesInclusionProof {
    tree_height: u32,
    queries: Vec<FoldingPolynomialQuery>,
    indexed_leaf_digests: Vec<(usize, Digest)>,
    auth_structure: AuthenticationStructure,
}

/// The result of a query to the folding polynomial, yielding `k`-many
/// (i.e., `folding_factor`-many) answers from the oracle.
struct FoldingPolynomialQuery {
    /// The index of the [point](Self::point) within the query domain.
    index: usize,

    /// Corresponds to r_{i,j}^{shift} in the STIR paper.
    point: BFieldElement,

    /// A `k`-th root of [`Self::point`].
    root: BFieldElement,

    /// The distance between any two `k`-th roots of [`Self::point`].
    ///
    /// In particular, the following gives all `k` such `k`-th roots:
    ///
    /// ```no_compile
    /// (0..k).map(|j| root * root_distance.mod_pow(j))
    /// ```
    //
    // While this is the same for all queries of one round, it wastes memory on
    // the order of `num_queries * BFieldElement::BYTES`, which is about 1KiB.
    // For Triton VM, that’s on the very low end of things.
    root_distance: BFieldElement,

    /// Corresponds to the evaluation of f_i at all `y` with
    /// `y^k == point`.
    values: Vec<XFieldElement>,
}

/// The required data to perform a quotienting step.
///
/// To compute the folding oracle for the next round, this data is needed in
/// addition to the data dequeued from the proof stream.
///
/// The exception from this rule is the initial round, which does not require
/// this data. Note that the initial round might coincide with the final round,
/// in which case this struct is not created.
struct QuotientingData {
    quotient_set: Vec<XFieldElement>,
    quotient_answers: Vec<XFieldElement>,
    degree_correction_randomness: XFieldElement,
}

impl Stir {
    /// The number of bits required to represent an [XFieldElement].
    ///
    /// While the value computed here is slightly higher than the more correct
    /// log₂([BFieldElement::P] · [x_field_element::EXTENSION_DEGREE]), the
    /// difference is about 1.0e-9, while the rounding error for the “actual”
    /// log₂(|𝔽|) in double-precision IEEE 754 floating point format (i.e.,
    /// rust’s [f64]) is about 1.2e-7.
    const LOG2_FIELD_SIZE: usize = BFieldElement::BYTES * 8 * x_field_element::EXTENSION_DEGREE;

    /// The (log₂ of the) relative length difference of the evaluation
    /// domains of two consecutive rounds.
    //
    // In this current implementation, the Reed-Solomon code’s domain shrinks
    // by the same amount between any two rounds. While the STIR paper does not
    // mention that this could be variable, it seems plausible that it could.
    // A future change to this implementation might make this variable.
    const LOG2_DOMAIN_SHRINKAGE: usize = 1;

    /// The highest polynomial degree for which low-degreeness can be proven
    /// with this STIR instance.
    pub fn max_degree(&self) -> usize {
        (1 << self.log2_high_degree) - 1
    }

    /// Derive the round parameters.
    fn round_params(&self) -> SetupResult<RoundParams> {
        if self.log2_folding_factor < 2 {
            return Err(StirParameterError::TooSmallLog2FoldingFactor(
                self.log2_folding_factor,
            ));
        }
        if self.log2_initial_expansion_factor == 0 {
            return Err(StirParameterError::TooSmallInitialExpansionFactor);
        }
        if self.log2_high_degree < self.log2_folding_factor {
            return Err(StirParameterError::TooLowDegreeOfHighDegreePolynomials);
        }

        let Ok(log2_folding_factor) = u32::try_from(self.log2_folding_factor) else {
            return Err(StirParameterError::TooBigLog2FoldingFactor(
                self.log2_folding_factor,
            ));
        };
        let Some(folding_factor) = 1_usize.checked_shl(log2_folding_factor) else {
            return Err(StirParameterError::TooBigLog2FoldingFactor(
                self.log2_folding_factor,
            ));
        };

        // the initial folding happens before any full round
        let mut poly_degree = self.max_degree() / folding_factor;
        let mut log2_expansion_factor = self.log2_initial_expansion_factor;
        let mut round_queries = Vec::new();

        // Folding lowers the current degree to ⌊poly_degree / folding_factor⌋.
        // Since this “bottoms out” at 0, folding too often can lead to
        // soundness problems.
        // For example, take a current degree of 2 and a folding factor of 4.
        // Folding results in a polynomial of degree 0. However, folding a
        // polynomial of degree 3 _also_ results in a polynomial of degree 0,
        // even though the degree-3 polynomial should be rejected by the
        // verifier. The solution is to stop folding before the process
        // “bottoms out.” Since there is one more folding step applied in the
        // final round, no full round must reduce the degree to be less than or
        // equal to the folding factor.
        //
        // Also note that a STIR instance with a “high degree” of less than the
        // folding factor is degenerate and will have been rejected earlier in
        // this method.
        while poly_degree > folding_factor {
            // Because the out-of-domain queries are made to the folded
            // polynomial, the number of out-of-domain queries are set with
            // respect to new rate.
            let log2_next_expansion_factor =
                log2_expansion_factor + self.log2_folding_factor - Self::LOG2_DOMAIN_SHRINKAGE;
            let log2_poly_degree = poly_degree.ilog2().try_into().expect(U32_TO_USIZE_ERR);
            let out_of_domain = self.num_ood_queries(log2_poly_degree, log2_next_expansion_factor);

            let in_domain = self.num_in_domain_queries(log2_expansion_factor)?;
            let num_queries = NumQueries {
                in_domain,
                out_of_domain,
            };

            let folded_poly_degree = poly_degree / folding_factor;
            if num_queries.total() > folded_poly_degree {
                // In STIR’s quotienting step, the so-called “answer”
                // polynomial is subtracted from the folded polynomial.
                // Both prover and verifier compute the “answer” polynomial by
                // interpolating over all queries and their – well – answers.
                //
                // If the “answer” polynomial _equals_ the folded polynomial,
                // the subsequent quotient is the zero-polynomial. This
                // equality happens if the total number of queries exceeds the
                // folded polynomial’s degree.
                //
                // It has the following consequences:
                // 1. Even though the next round’s polynomial is the degree-
                //    corrected quotient polynomial, no amount of degree-
                //    correction can turn the zero-polynomial into anything but
                //    the zero-polynomial. If the folded polynomial has a degree
                //    that is too high, but the quotient ends up being the zero-
                //    polynomial, then degree-correction cannot recover the
                //    too-high-degree polynomial. In other words, the verifier
                //    will incorrectly accept a polynomial of high degree.
                // 2. The verifier is doing more work than is optimal, because
                //    it just recovered the entirety of the folded polynomial.
                //    At that point, the protocol might as well stop.
                //
                // Note that the first consequence does not apply to the final
                // round, since the final round has no quotienting step. If the
                // second consequence applies, it does not break any soundness
                // guarantees, the protocol only becomes less efficient.
                break;
            }

            round_queries.push(num_queries);
            poly_degree = folded_poly_degree;
            log2_expansion_factor = log2_next_expansion_factor;
        }

        let final_num_in_domain_queries = self.num_in_domain_queries(log2_expansion_factor)?;
        let final_degree = poly_degree;
        let round_params = RoundParams {
            round_queries,
            final_num_in_domain_queries,
            final_degree,
        };

        Ok(round_params)
    }

    /// The domain for the initial codeword of the [prover](Stir::prove).
    //
    // The first part of this method essentially only computes:
    // 1 << (log2_high_degree + log2_initial_expansion_factor)
    //
    // However, because the input parameters can be waaaayy too big, a bunch of
    // input validation is… necessary? Let’s go with “beneficial.” Whether the
    // removal of this would be a DOS against Triton VM is somewhat doubtful,
    // but hey, better safe than sorry.
    pub(crate) fn initial_domain(&self) -> SetupResult<ArithmeticDomain> {
        let as_u64 = |int| u64::try_from(int).expect(USIZE_TO_U64_ERR);
        let error = |x| Err(StirParameterError::InitialDomainTooBig(x));

        let log2_high_degree = as_u64(self.log2_high_degree);
        let log2_expansion_factor = as_u64(self.log2_initial_expansion_factor);
        let Some(log2_domain_len) = log2_high_degree.checked_add(log2_expansion_factor) else {
            return error(u64::MAX);
        };
        let Ok(log2_domain_len) = u32::try_from(log2_domain_len) else {
            return error(log2_domain_len);
        };
        let Some(domain_len) = 1_usize.checked_shl(log2_domain_len) else {
            return error(log2_domain_len.into());
        };

        let Ok(domain) = ArithmeticDomain::of_length(domain_len) else {
            return error(log2_domain_len.into());
        };
        let domain = domain.with_offset(BFieldElement::generator());

        Ok(domain)
    }

    /// The number of in-domain queries for the Reed-Solomon code implied by
    /// the parameters.
    //
    // The formula used to derive this method is
    //
    //   Pr[δ-far-from-code codeword is not exposed as inconsistent] ⩽ (1-δ)^t
    //
    // where δ is the proximity parameter of the used Reed-Solomon code.
    // Additionally, we have the requirement that the magnitude of this
    // probability is acceptable:
    //
    //   (1-δ)^t ⩽ 2^-security_level
    //
    // Under provable list-decoding bounds, we have δ = 1 - √ρ - η,
    // under conjectured list-decoding bounds, we have δ = 1 - ρ - η,
    // where η is the gap between the chosen bound and δ, and is defined in
    // the corresponding method.
    //
    // With all of these ingredients, we solve for repetition parameter t by
    // taking the log₂ on both sides:
    //   t·log₂(1-δ) ⩽ -security_level
    //
    // Since 0 < δ < 1, we know that log₂(1 - δ) < 0:
    //             t ⩾ -security_level / log₂(1-δ)
    fn num_in_domain_queries(&self, log2_expansion_factor: usize) -> SetupResult<usize> {
        let delta = self.delta(log2_expansion_factor)?;
        let num_queries = -(self.security_level as f64) / (1.0 - delta).log2();

        Ok(num_queries.ceil() as usize)
    }

    /// The proximity parameter of the Reed-Solomon code.
    ///
    /// This is the δ in a (δ, ℓ)-list-decodable Reed-Solomon code. It is the
    /// threshold for which words are considered “far” from the code.
    ///
    /// Note that there are a range of valid values for δ. This method makes a
    /// concrete choice; see also [η](Self::log2_eta).
    fn delta(&self, log2_expansion_factor: usize) -> SetupResult<f64> {
        let log2_eta = self.log2_eta(log2_expansion_factor);
        let eta = 2_f64.powf(log2_eta);

        let Ok(log2_expansion_factor) = u32::try_from(log2_expansion_factor) else {
            return Err(StirParameterError::TooBigInitialExpansionFactor);
        };
        let Some(expansion_factor) = 1_u32.checked_shl(log2_expansion_factor) else {
            return Err(StirParameterError::TooBigInitialExpansionFactor);
        };
        let rate = 1.0 / f64::from(expansion_factor);
        let delta = match self.soundness {
            SoundnessAssumption::Provable => 1.0 - rate.sqrt() - eta,
            SoundnessAssumption::Conjectured => 1.0 - rate - eta,
        };

        Ok(delta)
    }

    /// The (log₂ of the) distance between the proximity parameter δ and the
    /// [chosen bound](SoundnessAssumption), called η.
    ///
    /// No matter the soundness assumption, the upper bound on the
    /// [list size](Self::log2_list_size) only holds for proximity parameters δ
    /// that are slightly below the bound. This function defines just how big
    /// the distance between the proximity parameter and the bound is.
    ///
    /// For example, if the chosen bound is the
    /// [Johnson bound](SoundnessAssumption::Provable), then δ ∈ (0, 1 - √ρ),
    /// and we chose δ = 1 - √ρ - η.
    /// For the [conjectured bound](SoundnessAssumption::Conjectured),
    /// δ ∈ (0, 1 - ρ), and we chose δ = 1 - ρ - η. In either case, η is defined
    /// in this method.
    //
    // TODO: Giacomo thinks this can be set in a better way. Maybe it makes
    //       more sense if it’s multiplicative?
    // TODO: This is based on some heuristic by Giacomo. Ferdinand doesn’t
    //       fully understand how this comes to be. Explore different things.
    // TODO: According to Giacomo, “Funnily enough, [η = ρ/10 or η = √ρ/10]
    //       avoids all the new attacks on proximity gaps 😉”
    //       Does a division by 20 also work? Need to investigate.
    // TODO: Nicolas Mohnblatt has a recent write-up that’s nice:
    //       https://blog.zksecurity.xyz/posts/proximity-conjecture/
    fn log2_eta(&self, log2_expansion_factor: usize) -> f64 {
        let log2_expansion_factor = log2_expansion_factor as f64;
        let log2_rho_or_sqrt_rho = match self.soundness {
            SoundnessAssumption::Provable => 0.5 * -log2_expansion_factor,
            SoundnessAssumption::Conjectured => -log2_expansion_factor,
        };

        // This is where the heuristic kicks in:
        // log₂(ρ/20) or log₂(√ρ/20)
        log2_rho_or_sqrt_rho - core::f64::consts::LOG2_10 - 1.
    }

    /// The number of out-of-domain queries for the Reed-Solomon code implied
    /// by the parameters.
    //
    // This method uses Lemma 4.5 from the STIR paper, requiring that the
    // probability that there are two distinct codewords in the list-decoding
    // set of our function (i.e., the folded polynomial) that both agree on a
    // random point is smaller than (or equal to) the soundness level:
    //
    //   (ℓ^2 / 2)·(d / (|𝔽| - |D|))^s ⩽ 2^-security_level
    //
    // where ℓ is the list size of the Reed-Solomon code, d is the (max) degree
    // of this round’s polynomial, 𝔽 is the field the Reed-Solomon code (and by
    // extension, STIR) is defined over (in our case always the extension
    // field), and D is this round’s domain over which the polynomial is
    // evaluated.
    // We start to solve for the repetition parameter s:
    //
    //             (d / (|𝔽| - |D|))^s ⩽ 2^-security_level · (2 / ℓ^2)
    //
    // Taking the log₂ on both sides:
    //   s·(log₂(d) - log₂(|𝔽| - |D|)) ⩽ -security_level + 1 - 2·log₂(ℓ)
    //
    // Under the assumption that d, the degree of the polynomial, is (much!)
    // smaller than the size of the field, we know that
    // (log₂(d) - log₂(|𝔽| - |D|)) is negative.
    // The assumption is reasonable (or even necessary) because polynomials
    // with degree as big (or bigger) than the field start behaving weirdly.
    // For example, the polynomial X^p over field 𝔽_p is functionally
    // equivalent to the polynomial 1. While distinct as polynomials, it’s
    // impossible to differentiate between the two in evaluation form.
    // It’s also hard to argue that such polynomials are of “low degree”.
    // And as a final nail in the coffin, the largest possible
    // `ArithmeticDomain` we currently support has size 2^32. Polynomials of
    // higher degree cannot even be passed in to STIR.
    // Anyway, on with the derivation:
    //   s ⩾ (-security_level + 1 - 2·log₂(ℓ)) / (log₂(d) - log₂(|𝔽| - |D|))
    //   s ⩾ (security_level - 1 + 2·log₂(ℓ)) / (log₂(|𝔽| - |D|) - log₂(d))
    //
    // The largest possible size of the `ArithmeticDomain` D is 2^32. Because
    // this is vanishingly small compared to the size of the field 𝔽, we ignore
    // the size of D. In fact, we even use only an approximation for |𝔽|:
    // Instead of log₂((2^64 - 2^32 + 1)^3), we use log₂((2^64)^3).
    // The difference between log₂((2^64)^3) and log₂(|𝔽| - |D|) is less than
    // 1.1e-9.
    // Representing log₂((2^64 - 2^32 + 1)^3) using double-precision IEEE 754
    // floating point format (i.e., rust’s `f64`) gives a rounding error of
    // about 1.2e-7. In other words, accurately computing log₂(|𝔽| - |D|)
    // requires a specialized, high-precision floating point library. We’re
    // not doing that.
    //
    // Long story short:
    //   s ⩾ (security_level - 1 + 2·log₂(ℓ)) / (log₂(|𝔽|) - log₂(d))
    fn num_ood_queries(&self, log2_poly_degree: usize, log2_expansion_factor: usize) -> usize {
        let log2_list_size = self.log2_list_size(log2_poly_degree, log2_expansion_factor);
        let num_ood_queries = (self.security_level as f64 - 1.0 + 2.0 * log2_list_size)
            / (Self::LOG2_FIELD_SIZE - log2_poly_degree) as f64;

        num_ood_queries.ceil() as usize
    }

    /// The list size of the Reed-Solomon code.
    ///
    /// The Reed-Solomon code in question is defined by the given polynomial
    /// degree and the given rate. The proximity parameter δ is implied by
    /// [η](Self::log2_eta).
    ///
    /// For the [Johnson bound](SoundnessAssumption::Provable), it is shown that
    /// the Reed-Solomon codes are (1-√ρ-η, 1/(2·√ρ·η))-list decodable. In
    /// other words, the list size for δ = 1-√ρ-η is
    /// ℓ = √(extension_factor)/(2·η).
    ///
    /// For the [conjectured bound](SoundnessAssumption::Conjectured), it is
    /// assumed that Reed-Solomon codes are (1-ρ-η, d/(ρ·η))-list decodable. In
    /// other words, the list size for δ = 1-ρ-η is
    /// ℓ = degree·extension_factor/η.
    /// See also Conjecture 5.6 in the STIR paper.
    fn log2_list_size(&self, log2_poly_degree: usize, log2_expansion_factor: usize) -> f64 {
        let log2_eta = self.log2_eta(log2_expansion_factor);

        match self.soundness {
            SoundnessAssumption::Provable => log2_expansion_factor as f64 / 2. - 1. - log2_eta,
            SoundnessAssumption::Conjectured => {
                (log2_poly_degree + log2_expansion_factor) as f64 - log2_eta
            }
        }
    }

    //                                                           in DEEP commitment
    //
    // - commit to f                                                    x
    // - sample folding randomness                    (fold_rand)
    //
    // fold
    // - decompose initial polynomial (of degree d)
    //   into k polynomials (of degree d/k)
    // - randomly combine using fold_rand             (g_poly)
    //
    // maybe terminate:
    //   - if deg(g_poly) <= final_degree, send g_poly
    //
    // - halve domain, shift by ω                     (new_dom)
    // - commit to g_poly evaluated over new_dom
    //
    // - sample ood-randomness (more than one point?) (αs)              x
    // - evaluate g_poly over αs                      (βs)              x
    // - send βs
    // - sample shift-queries from domain^k           (v_i)_i           x
    // - send evaluations of f in all v_i             (y_i)_i
    // - send authentication paths of those evaluations
    //
    // quotient
    // - interpolate (α_i, β_i)_i and (v_i, y_i)_i    (p)
    // - quotient g_poly with respect to p:
    //   - compute zerofier of α_i and v_i            (z_S)
    //   - (g_poly - p) / z_S                         (f')
    // - sample degree-correction randomness          (deg_rand)
    // - degree-correct quotient                      (new f)
    //
    // # Todo
    //
    // - Figure out use of “shake polynomial”. I think this is Giacomo’s trick.
    //   This makes the interpolation in the verifier superfluous.
    // - Include an ergonomic interface to extract all the data used by the
    //   verifier. For example, the verifier could always return the entire
    //   transcript, including sampled indices and other randomness.
    pub fn prove(
        &self,
        codeword: &[XFieldElement],
        proof_stream: &mut ProofStream,
    ) -> ProverResult<Vec<usize>> {
        let round_params = self.round_params()?;

        let mut domain = self.initial_domain()?;
        if domain.len() != codeword.len() {
            return Err(StirProvingError::InitialCodewordMismatch {
                domain_len: domain.len(),
                codeword_len: codeword.len(),
            });
        }

        let mut commitment = LeafStackMerkleTree::new(codeword, 1 << self.log2_folding_factor);
        proof_stream.enqueue(ProofItem::MerkleRoot(commitment.tree.root()));

        let mut poly = domain.interpolate(codeword);
        let mut first_round_queried_indices = None;
        for num_queries in round_params.round_queries {
            let fold_randomness = proof_stream.sample_scalars(1)[0];
            let folded_poly =
                Self::fold_polynomial(&poly, 1 << self.log2_folding_factor, fold_randomness);
            let small_domain = domain.pow(2).unwrap().with_offset(domain.generator()); // TODO: this offset is wrong
            let folded_evaluations = small_domain.evaluate(&folded_poly);
            let folded_poly_commitment =
                LeafStackMerkleTree::new(&folded_evaluations, 1 << self.log2_folding_factor);
            proof_stream.enqueue(ProofItem::MerkleRoot(folded_poly_commitment.tree.root()));

            let ood_queries = proof_stream.sample_scalars(num_queries.out_of_domain);
            let ood_values = ood_queries
                .iter()
                .map(|&x| folded_poly.evaluate_in_same_field(x))
                .collect_vec();
            proof_stream.enqueue(ProofItem::StirOutOfDomainValues(ood_values.clone()));

            let query_domain = domain.pow(1 << self.log2_folding_factor).unwrap();
            let queried_indices = proof_stream
                .sample_indices(query_domain.len(), num_queries.in_domain)
                .into_iter() // TODO: over / rejection sample to ensure safe minimum?
                .unique() // TODO: <-- avoid this if possible, but it’s probably not
                .collect_vec();
            let inclusion_proof = commitment.inclusion_proof(&queried_indices);
            proof_stream.enqueue(ProofItem::StirResponse(inclusion_proof));

            // construct the witness polynomial for the next round
            let queried_domain_values = queried_indices
                .iter()
                .map(|&i| u32::try_from(i).expect(DOMAIN_INDEX_TO_U32_ERR))
                .map(|i| query_domain.value(i))
                .collect_vec();
            let points_to_quotient_out = queried_domain_values
                .iter()
                .map(|&x| folded_poly.evaluate(x))
                .chain(ood_values)
                .collect_vec();
            let domain_values_to_quotient_out = queried_domain_values
                .into_iter()
                .map(|bfe| bfe.lift())
                .chain(ood_queries)
                .collect_vec();
            let answer_poly =
                Polynomial::interpolate(&domain_values_to_quotient_out, &points_to_quotient_out);
            let zerofier = Polynomial::zerofier(&domain_values_to_quotient_out);
            let quotient = (folded_poly - answer_poly) / zerofier;

            let degree_correction_randomness = proof_stream.sample_scalars(1)[0];
            let degree_correction_coefficients = (0..=points_to_quotient_out.len())
                .map(|i| u32::try_from(i).expect(QUOTIENT_SET_LEN_TO_U32_ERR))
                .map(|i| degree_correction_randomness.mod_pow_u32(i))
                .collect();
            let degree_correction_poly = Polynomial::new(degree_correction_coefficients);

            // carry state to next round
            poly = quotient * degree_correction_poly;
            domain = small_domain;
            commitment = folded_poly_commitment;

            first_round_queried_indices.get_or_insert(queried_indices);
        }

        // final round is special because there is no quotienting
        let fold_randomness = proof_stream.sample_scalars(1)[0];
        let final_poly =
            Self::fold_polynomial(&poly, 1 << self.log2_folding_factor, fold_randomness);
        proof_stream.enqueue(ProofItem::StirPolynomial(final_poly));

        let final_query_domain = domain.pow(1 << self.log2_folding_factor).unwrap();
        let final_queried_indices = proof_stream
            .sample_indices(
                final_query_domain.len(),
                round_params.final_num_in_domain_queries,
            )
            .into_iter()
            .unique()
            .collect_vec();
        let final_inclusion_proof = commitment.inclusion_proof(&final_queried_indices);
        proof_stream.enqueue(ProofItem::StirResponse(final_inclusion_proof));

        Ok(first_round_queried_indices.unwrap_or(final_queried_indices))
    }

    /// # Panics
    ///
    /// Panics if the folding factor is 0.
    fn fold_polynomial<FF>(
        poly: &Polynomial<FF>,
        folding_factor: usize,
        fold_randomness: FF,
    ) -> Polynomial<'static, FF>
    where
        FF: FiniteField,
    {
        let folded_coefficients = poly
            .coefficients()
            .chunks(folding_factor)
            .map(|chunk| Polynomial::new_borrowed(chunk).evaluate(fold_randomness))
            .collect();

        Polynomial::new(folded_coefficients)
    }

    /// Verify low-degreeness of the polynomial on the proof stream.
    ///
    /// Returns the indices and revealed elements of the codeword at the top
    /// level of the STIR proof.
    pub fn verify(
        &self,
        proof_stream: &mut ProofStream,
    ) -> VerifierResult<Vec<(usize, XFieldElement)>> {
        let round_params = self.round_params()?;

        let mut first_round_queries = None;
        let mut previous_quotienting_data = None;
        let mut domain = self.initial_domain()?;
        let mut commitment_to_previous_polynomial =
            proof_stream.dequeue()?.try_into_merkle_root()?;
        for num_queries in round_params.round_queries {
            let fold_randomness = proof_stream.sample_scalars(1)[0];
            let commitment_to_current_polynomial =
                proof_stream.dequeue()?.try_into_merkle_root()?;
            let ood_queries = proof_stream.sample_scalars(num_queries.out_of_domain);
            let ood_answers = proof_stream.dequeue()?.try_into_stir_ood_values()?;
            let queries = self
                .extract_merkle_tree_inclusion_proof(proof_stream, domain, num_queries.in_domain)?
                .authenticated_queries(commitment_to_previous_polynomial)?;
            first_round_queries
                .get_or_insert(queries.iter().map(|q| (q.index, q.values[0])).collect());

            let in_domain_answers = match previous_quotienting_data {
                None => Self::initial_in_domain_answers(&queries, fold_randomness),
                Some(data) => Self::subsequent_in_domain_answers(data, &queries, fold_randomness),
            };

            let quotient_set = queries
                .into_iter()
                .map(|q| q.point.lift())
                .chain(ood_queries)
                .collect_vec();
            let quotient_answers = in_domain_answers
                .into_iter()
                .chain(ood_answers)
                .collect_vec();
            let degree_correction_randomness = proof_stream.sample_scalars(1)[0];
            let quotienting_data = QuotientingData {
                quotient_set,
                quotient_answers,
                degree_correction_randomness,
            };

            previous_quotienting_data = Some(quotienting_data);
            domain = domain.pow(2).unwrap().with_offset(domain.generator()); // TODO: this offset is wrong
            commitment_to_previous_polynomial = commitment_to_current_polynomial;
        }

        // final round
        let folding_rand = proof_stream.sample_scalars(1)[0];
        let final_poly = proof_stream.dequeue()?.try_into_stir_polynomial()?;

        // for the low, low chance that the final polynomial is the zero
        // polynomial, we treat it as if it was a constant polynomial when
        // checking the degree
        let poly_degree = final_poly.degree().try_into().unwrap_or(0);
        if poly_degree > round_params.final_degree {
            return Err(StirVerificationError::LastRoundPolynomialHasTooHighDegree);
        }

        let final_num_queries = round_params.final_num_in_domain_queries;
        let queries = self
            .extract_merkle_tree_inclusion_proof(proof_stream, domain, final_num_queries)?
            .authenticated_queries(commitment_to_previous_polynomial)?;

        let final_folds = match previous_quotienting_data {
            None => Self::initial_in_domain_answers(&queries, folding_rand),
            Some(data) => Self::subsequent_in_domain_answers(data, &queries, folding_rand),
        };
        let final_evaluations = queries.iter().map(|query| final_poly.evaluate(query.point));
        if final_folds
            .into_iter()
            .zip(final_evaluations)
            .any(|(fold, eval)| fold != eval)
        {
            return Err(StirVerificationError::LastRoundPolynomialEvaluationMismatch);
        }

        let first_round_queries = first_round_queries
            .unwrap_or_else(|| queries.iter().map(|q| (q.index, q.values[0])).collect());

        Ok(first_round_queries)
    }

    /// The evaluations of the initial (virtual) function “f”.
    fn initial_in_domain_answers(
        queries: &[FoldingPolynomialQuery],
        fold_randomness: XFieldElement,
    ) -> Vec<XFieldElement> {
        queries
            .iter()
            .map(|query| Polynomial::fast_coset_interpolate(query.root, &query.values))
            .map(|poly| poly.evaluate(fold_randomness))
            .collect()
    }

    /// Turn evaluations of (previous) committed function “g” into evaluations
    /// of (current) (virtual) function “f”.
    fn subsequent_in_domain_answers(
        quotienting_data: QuotientingData,
        queries: &[FoldingPolynomialQuery],
        fold_randomness: XFieldElement,
    ) -> Vec<XFieldElement> {
        const ONE: XFieldElement = XFieldElement::ONE;

        let QuotientingData {
            quotient_set,
            quotient_answers,
            degree_correction_randomness,
        } = quotienting_data;

        let answer_poly = Polynomial::interpolate(&quotient_set, &quotient_answers);
        let zerofier = Polynomial::zerofier(&quotient_set);
        let zerofier_degree = u32::try_from(zerofier.degree()).expect(QUOTIENT_SET_LEN_TO_U32_ERR);

        let mut in_domain_answers = Vec::with_capacity(queries.len());
        let mut coset_evaluations = Vec::new();
        for query in queries {
            coset_evaluations.clear(); // re-use the small allocation
            let mut current_root_distance = BFieldElement::ONE;
            for &evaluation in &query.values {
                // quotienting
                let current_root = query.root * current_root_distance;
                let answer_evaluation = answer_poly.evaluate::<_, XFieldElement>(current_root);
                let quotient = (evaluation - answer_evaluation) / zerofier.evaluate(current_root);

                // degree correction
                let common_factor = current_root * degree_correction_randomness;
                let degree_correction_factor = if common_factor == ONE {
                    xfe!(zerofier_degree)
                } else {
                    (ONE - common_factor.mod_pow_u32(zerofier_degree)) / (ONE - common_factor)
                };

                coset_evaluations.push(degree_correction_factor * quotient);
                current_root_distance *= query.root_distance;
            }

            let poly = Polynomial::fast_coset_interpolate(query.root, &coset_evaluations);
            in_domain_answers.push(poly.evaluate(fold_randomness));
        }

        in_domain_answers
    }

    fn extract_merkle_tree_inclusion_proof(
        &self,
        proof_stream: &mut ProofStream,
        poly_domain: ArithmeticDomain,
        num_id_queries: usize,
    ) -> VerifierResult<PolyFoldQueriesInclusionProof> {
        let query_domain = poly_domain.pow(1 << self.log2_folding_factor).unwrap();
        let queried_indices = proof_stream
            .sample_indices(query_domain.len(), num_id_queries)
            .into_iter()
            .unique()
            .collect_vec();
        let inclusion_proof = proof_stream.dequeue()?.try_into_stir_response()?;

        // TODO: turn this into `… != num_id_queries` once over- / rejection
        //       sampling is figured out
        if inclusion_proof.queried_leafs.len() != queried_indices.len() {
            return Err(StirVerificationError::IncorrectNumberOfRevealedLeaves);
        }

        let leaf_digests = inclusion_proof
            .queried_leafs
            .iter()
            .map(|slice| XFieldElement::bfe_slice(slice))
            .map(Tip5::hash_varlen);
        let indexed_leaf_digests = queried_indices.iter().copied().zip(leaf_digests).collect();

        let query_domain_len = u64::try_from(query_domain.len()).expect(USIZE_TO_U64_ERR);
        let root_distance = poly_domain.generator().mod_pow(query_domain_len);
        let queries = queried_indices
            .into_iter()
            .zip(inclusion_proof.queried_leafs)
            .map(|(index, values)| {
                let query_index = u32::try_from(index).expect(DOMAIN_INDEX_TO_U32_ERR);
                FoldingPolynomialQuery {
                    index,
                    point: query_domain.value(query_index),
                    root: poly_domain.value(query_index),
                    root_distance,
                    values,
                }
            })
            .collect();

        let inclusion_proof = PolyFoldQueriesInclusionProof {
            tree_height: query_domain.len().ilog2(),
            queries,
            indexed_leaf_digests,
            auth_structure: inclusion_proof.auth_structure,
        };

        Ok(inclusion_proof)
    }
}

impl NumQueries {
    /// The total number of queries, both in- and out-of-domain.
    fn total(&self) -> usize {
        self.in_domain + self.out_of_domain
    }
}

impl LeafStackMerkleTree {
    /// # Panics
    ///
    /// - if the `stack_height` is not a power of two
    /// - if the codeword’s length is not a power of two
    /// - if the codeword’s length is smaller than the `stack_height`
    fn new(codeword: &[XFieldElement], stack_height: usize) -> Self {
        let stacked_leafs = Self::stack(codeword, stack_height);
        let leaf_digests = stacked_leafs
            .iter()
            .map(|stack| XFieldElement::bfe_slice(stack))
            .map(Tip5::hash_varlen)
            .collect_vec();
        let tree = MerkleTree::par_new(&leaf_digests).unwrap();

        Self {
            stacked_leafs,
            tree,
        }
    }

    /// Re-organize the passed-in vector.
    ///
    /// If the requested `stack_height` divides the length of the input vector
    /// cleanly, all the returned vectors have the same length.
    ///
    /// # Examples
    ///
    /// On input `vec![0, 1, 2, 3, 4, 5, 6, 7]` with a requested stack height
    /// of 4, this produces `[[0, 2, 4, 6], [1, 3, 5, 7]]` (as vectors).
    ///
    /// On the same input vector but a requested stack height of 2, produces
    /// `[[0, 4], [1, 5], [2, 6], [3, 7]]` (as vectors).
    fn stack(codeword: &[XFieldElement], stack_height: usize) -> Vec<Vec<XFieldElement>> {
        let element_distance = codeword.len().div_ceil(stack_height);
        (0..element_distance)
            .map(|initial_skip| {
                codeword
                    .iter()
                    .skip(initial_skip)
                    .step_by(element_distance)
                    .copied()
                    .collect()
            })
            .collect()
    }

    fn inclusion_proof(&self, indices: &[usize]) -> StirResponse {
        let queried_leafs = indices
            .iter()
            .map(|&i| self.stacked_leafs[i].clone())
            .collect_vec();
        let auth_structure = self.tree.authentication_structure(indices).unwrap();

        StirResponse {
            auth_structure,
            queried_leafs,
        }
    }
}

impl PolyFoldQueriesInclusionProof {
    fn authenticated_queries(
        self,
        merkle_root: Digest,
    ) -> VerifierResult<Vec<FoldingPolynomialQuery>> {
        let inclusion_proof = MerkleTreeInclusionProof {
            tree_height: self.tree_height,
            indexed_leafs: self.indexed_leaf_digests,
            authentication_structure: self.auth_structure,
        };

        if inclusion_proof.verify(merkle_root) {
            Ok(self.queries)
        } else {
            Err(StirVerificationError::BadMerkleAuthenticationPath)
        }
    }
}

#[cfg(test)]
mod tests {
    use assert2::let_assert;
    use proptest::prelude::*;
    use proptest_arbitrary_interop::arb;
    use test_strategy::proptest;

    use crate::arithmetic_domain::tests::arbitrary_domain;
    use crate::shared_tests::DigestCorruptor;
    use crate::shared_tests::arbitrary_polynomial_of_degree;

    use super::*;

    /// A type alias exclusive to this test module.
    type XfePoly = Polynomial<'static, XFieldElement>;

    #[test]
    fn stacking() {
        let evaluations = xfe_array![0, 1, 2, 3, 4, 5, 6, 7];
        let stacks_4 = LeafStackMerkleTree::stack(&evaluations, 4);
        let expected_4 = vec![xfe_vec![0, 2, 4, 6], xfe_vec![1, 3, 5, 7]];
        assert_eq!(expected_4, stacks_4);

        let stacks_2 = LeafStackMerkleTree::stack(&evaluations, 2);
        let expected_2 = [[0, 4], [1, 5], [2, 6], [3, 7]]
            .map(|v| v.map(|x| xfe!(x)).to_vec())
            .to_vec();
        assert_eq!(expected_2, stacks_2);
    }

    #[proptest]
    fn roots_of_domain_points(
        #[strategy(0..=3)]
        #[map(|x| 1_usize << x)]
        folding_factor: usize,
        #[strategy(arbitrary_domain())]
        #[filter(#old_domain.len() >= #folding_factor)]
        old_domain: ArithmeticDomain,
        #[strategy(0..#old_domain.len() as u32)] index: u32,
    ) {
        let new_domain = old_domain.pow(folding_factor)?;

        let folding_factor = folding_factor.try_into()?;
        let old_domain_value = old_domain.value(index);
        let root_distance = old_domain.generator().mod_pow(new_domain.len().try_into()?);
        let roots = (0..folding_factor).map(|i| old_domain_value * root_distance.mod_pow(i));

        let new_domain_value = new_domain.value(index);
        for (i, root) in roots.enumerate() {
            let raised_root = root.mod_pow(folding_factor);
            prop_assert_eq!(new_domain_value, raised_root, "{}", i);
        }
    }

    #[test]
    fn folding_polynomial_gives_expected_coefficients() {
        let folding_factor = 4;
        let fold_randomness = bfe!(10);
        let poly = Polynomial::new(bfe_vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        //                                  ╰────┬────╯ ╰──┬──────╯ ╰─┬──╯
        //                                       │     ╭───╯ ╭────────╯
        let expected = Polynomial::new(bfe_vec![4321, 8765, 109]);
        let actual = Stir::fold_polynomial(&poly, folding_factor, fold_randomness);

        assert_eq!(expected, actual);
    }

    #[proptest]
    fn evaluation_of_folded_polynomial_corresponds_to_folding_of_evaluated_polynomial(
        #[strategy(arb())] poly: Polynomial<'static, BFieldElement>,
        #[strategy(1..=5)]
        #[map(|exp| 1_usize << exp)]
        folding_factor: usize,
        #[strategy(arb())] fold_randomness: BFieldElement,
        #[strategy(arbitrary_domain())]
        #[filter(#old_domain.len() >= #folding_factor)]
        old_domain: ArithmeticDomain,
        #[strategy(0..#old_domain.len() as u32)] evaluation_index: u32,
    ) {
        // Ideally, the proptest would sample an evaluation index in range
        // 0..(old_domain.length / folding_factor). However, the corresponding
        // strategy is instantiated before the filter
        // (old_domain.length >= folding_factor) is applied. This means there
        // are cases (that _would_ be rejected) where the strategy is
        // instantiated with range 0..0, causing an immediate test failure.
        let evaluation_index = evaluation_index % u32::try_from(folding_factor)?;

        let folded_poly = Stir::fold_polynomial(&poly, folding_factor, fold_randomness);
        let new_domain = old_domain.pow(folding_factor)?;
        let evaluation_point = new_domain.value(evaluation_index);
        let evaluation_of_folded_poly = folded_poly.evaluate::<_, BFieldElement>(evaluation_point);

        let root_of_evaluation_point = old_domain.value(evaluation_index);
        let root_distance = old_domain.generator().mod_pow(new_domain.len().try_into()?);
        let roots = (0..folding_factor.try_into()?)
            .map(|i| root_of_evaluation_point * root_distance.mod_pow(i));
        let pre_folded_points = roots
            .map(|r| poly.evaluate::<_, BFieldElement>(r))
            .collect_vec();
        let folding_of_evaluated_poly =
            Polynomial::fast_coset_interpolate(root_of_evaluation_point, &pre_folded_points)
                .evaluate::<_, BFieldElement>(fold_randomness);

        prop_assert_eq!(evaluation_of_folded_poly, folding_of_evaluated_poly);
    }

    #[proptest]
    fn invalid_initial_codeword_is_rejected(
        stir: Stir,
        mut proof_stream: ProofStream,
        #[strategy(Just(#stir.initial_domain().unwrap().len() as isize))] domain_length: isize,
        #[strategy(-#domain_length..=#domain_length)]
        #[filter(#delta != &0)]
        delta: isize,
    ) {
        let wrong_length = (domain_length + delta) as usize;
        let codeword = xfe_vec![1; wrong_length];

        let_assert!(Err(err) = stir.prove(&codeword, &mut proof_stream));
        let_assert!(
            StirProvingError::InitialCodewordMismatch {
                domain_len,
                codeword_len
            } = err
        );
        prop_assert_eq!(domain_length as usize, domain_len);
        prop_assert_eq!(wrong_length, codeword_len);
    }

    #[proptest]
    fn invalid_parameter_initial_expansion_factor_1_is_rejected(
        mut stir: Stir,
        mut proof_stream: ProofStream,
    ) {
        let codeword = xfe_vec![1; stir.initial_domain()?.len()];

        stir.log2_initial_expansion_factor = 0;
        let_assert!(Err(err) = stir.prove(&codeword, &mut proof_stream));
        let_assert!(StirProvingError::ParameterError(err) = err);
        prop_assert_eq!(StirParameterError::TooSmallInitialExpansionFactor, err);
    }

    #[proptest]
    fn invalid_parameter_big_initial_expansion_factor_is_rejected(
        mut stir: Stir,
        mut proof_stream: ProofStream,
        #[strategy(32_usize..)] bad_log2_initial_expansion_factor: usize,
    ) {
        let codeword = xfe_vec![1; stir.initial_domain()?.len()];

        stir.log2_initial_expansion_factor = bad_log2_initial_expansion_factor;
        let_assert!(Err(err) = stir.prove(&codeword, &mut proof_stream));
        let_assert!(StirProvingError::ParameterError(err) = err);
        prop_assert_eq!(StirParameterError::TooBigInitialExpansionFactor, err);
    }

    #[proptest]
    fn invalid_parameter_small_folding_factor_is_rejected(
        mut stir: Stir,
        mut proof_stream: ProofStream,
        #[strategy(0_usize..=1)] bad_log2_folding_factor: usize,
    ) {
        let codeword = xfe_vec![1; stir.initial_domain()?.len()];

        stir.log2_folding_factor = bad_log2_folding_factor;
        let_assert!(Err(err) = stir.prove(&codeword, &mut proof_stream));
        let_assert!(StirProvingError::ParameterError(err) = err);
        let_assert!(StirParameterError::TooSmallLog2FoldingFactor(f) = err);
        prop_assert_eq!(bad_log2_folding_factor, f);
    }

    #[proptest]
    fn invalid_parameter_big_folding_factor_is_rejected(
        mut stir: Stir,
        mut proof_stream: ProofStream,
        #[strategy(64_usize..)] bad_log2_folding_factor: usize,
    ) {
        let codeword = xfe_vec![1; stir.initial_domain()?.len()];

        stir.log2_folding_factor = bad_log2_folding_factor;
        stir.log2_high_degree = bad_log2_folding_factor;
        let_assert!(Err(err) = stir.prove(&codeword, &mut proof_stream));
        let_assert!(StirProvingError::ParameterError(err) = err);
        let_assert!(StirParameterError::TooBigLog2FoldingFactor(f) = err);
        prop_assert_eq!(bad_log2_folding_factor, f);
    }

    #[proptest]
    fn invalid_parameter_small_high_degree_is_rejected(
        mut stir: Stir,
        mut proof_stream: ProofStream,
        #[strategy(..#stir.log2_folding_factor)] bad_log2_high_degree: usize,
    ) {
        let codeword = xfe_vec![1; stir.initial_domain()?.len()];

        stir.log2_high_degree = bad_log2_high_degree;
        let_assert!(Err(err) = stir.prove(&codeword, &mut proof_stream));
        let_assert!(StirProvingError::ParameterError(err) = err);
        prop_assert_eq!(StirParameterError::TooLowDegreeOfHighDegreePolynomials, err);
    }

    #[proptest]
    fn too_big_initial_domain_doesnt_cause_crash(
        mut stir: Stir,
        #[strategy(33 - #stir.log2_initial_expansion_factor..=64)] log2_high_degree: usize,
    ) {
        stir.log2_high_degree = log2_high_degree;
        let_assert!(Err(err) = stir.initial_domain());
        let_assert!(StirParameterError::InitialDomainTooBig(_) = err);
    }

    #[proptest]
    fn prove_and_verify_zero_polynomial(stir: Stir) {
        let zero_poly = xfe_vec![0; stir.initial_domain()?.len()];

        let mut proof_stream = ProofStream::new();
        stir.prove(&zero_poly, &mut proof_stream)?;

        proof_stream.reset_sponge();
        stir.verify(&mut proof_stream)?;
    }

    #[proptest]
    fn prove_and_verify_low_degree_polynomial(
        stir: Stir,
        #[strategy(-1..=#stir.max_degree() as i64)] _d: i64,
        #[strategy(arbitrary_polynomial_of_degree(#_d))] poly: XfePoly,
    ) {
        let codeword = stir.initial_domain()?.evaluate(&poly);

        let mut proof_stream = ProofStream::new();
        let prover_indices = stir.prove(&codeword, &mut proof_stream)?;
        let prover_sponge = proof_stream.sponge.clone();

        proof_stream.reset_sponge();
        let queries = stir.verify(&mut proof_stream)?;
        let verifier_sponge = proof_stream.sponge;

        prop_assert_eq!(prover_sponge, verifier_sponge);
        prop_assert_eq!(proof_stream.items.len(), proof_stream.items_index);

        let verifier_indices = queries.iter().map(|(idx, _)| idx).copied().collect_vec();
        prop_assert_eq!(prover_indices, verifier_indices);

        prop_assert!(queries.into_iter().all(|(idx, v)| codeword[idx] == v));
    }

    #[proptest]
    fn prove_and_fail_to_verify_high_degree_polynomial(
        stir: Stir,
        #[strategy(Just(1 << #stir.log2_high_degree))] _too_high_degree: i64,
        #[strategy(#_too_high_degree..2 * #_too_high_degree)] _d: i64,
        #[strategy(arbitrary_polynomial_of_degree(#_d))] poly: XfePoly,
    ) {
        let codeword = stir.initial_domain()?.evaluate(&poly);
        let mut proof_stream = ProofStream::new();
        stir.prove(&codeword, &mut proof_stream)?;

        proof_stream.reset_sponge();
        let verdict = stir.verify(&mut proof_stream);
        prop_assert!(verdict.is_err());
    }

    #[proptest(cases = 100)]
    fn proof_stream_serialization(
        stir: Stir,
        #[strategy(-1..=#stir.max_degree() as i64)] _d: i64,
        #[strategy(arbitrary_polynomial_of_degree(#_d))] poly: XfePoly,
    ) {
        let codeword = stir.initial_domain()?.evaluate(&poly);
        let mut proof_stream = ProofStream::new();
        stir.prove(&codeword, &mut proof_stream)?;

        let proof = (&proof_stream).into();
        let deserialized_proof_stream = ProofStream::try_from(&proof)?;

        let prover_items = proof_stream.items.iter();
        let verifier_items = deserialized_proof_stream.items.iter();
        for (prover_item, verifier_item) in prover_items.zip_eq(verifier_items) {
            use ProofItem::*;
            match (prover_item, verifier_item) {
                (MerkleRoot(p), MerkleRoot(v)) => prop_assert_eq!(p, v),
                (StirOutOfDomainValues(p), StirOutOfDomainValues(v)) => prop_assert_eq!(p, v),
                (StirResponse(p), StirResponse(v)) => prop_assert_eq!(p, v),
                (StirPolynomial(p), StirPolynomial(v)) => prop_assert_eq!(p, v),
                _ => panic!("Unknown items.\nProver: {prover_item:?}\nVerifier: {verifier_item:?}"),
            }
        }
    }

    #[proptest]
    fn verifying_arbitrary_proof_does_not_panic(stir: Stir, mut proof_stream: ProofStream) {
        let _verdict = stir.verify(&mut proof_stream);
    }

    #[proptest(cases = 100)]
    fn modified_proof_stream_results_in_verification_failure(
        stir: Stir,
        #[strategy(-1..=#stir.max_degree() as i64)] _d: i64,
        #[strategy(arbitrary_polynomial_of_degree(#_d))] poly: XfePoly,
        item_index: usize,
        vec_index: usize,
        digest_corruptor: DigestCorruptor,
        #[strategy(arb())] random_xfe: XFieldElement,
        corrupt_auth_structure: bool,
    ) {
        let codeword = stir.initial_domain()?.evaluate(&poly);
        let mut proof_stream = ProofStream::new();
        stir.prove(&codeword, &mut proof_stream)?;
        proof_stream.reset_sponge();

        let proof_item = select_element(&mut proof_stream.items, item_index)?;
        match proof_item {
            ProofItem::MerkleRoot(d) => digest_corruptor.corrupt(d)?,
            ProofItem::StirOutOfDomainValues(xfes) => corrupt_slice(xfes, vec_index, random_xfe)?,
            ProofItem::StirResponse(stir_response) => {
                if corrupt_auth_structure {
                    let chosen_digest =
                        select_element(&mut stir_response.auth_structure, vec_index)?;
                    digest_corruptor.corrupt(chosen_digest)?;
                } else {
                    let num_lists = stir_response.queried_leafs.len();
                    let chosen_list = select_element(&mut stir_response.queried_leafs, vec_index)?;
                    corrupt_slice(chosen_list, vec_index / num_lists, random_xfe)?;
                }
            }
            ProofItem::StirPolynomial(poly) => {
                let mut coefficients = poly.coefficients().to_vec();
                corrupt_slice(&mut coefficients, vec_index, random_xfe)?;
                *poly = Polynomial::new(coefficients);
            }
            _ => panic!("cannot modify non-STIR proof item"),
        }

        let verdict = stir.verify(&mut proof_stream);
        prop_assert!(verdict.is_err());
    }

    fn corrupt_slice<T>(slice: &mut [T], index: usize, new_elem: T) -> Result<(), TestCaseError>
    where
        T: PartialEq,
    {
        let chosen_elem = select_element(slice, index)?;
        if *chosen_elem == new_elem {
            let reason = "corruption must change the Vec".into();
            return Err(TestCaseError::Reject(reason));
        }
        *chosen_elem = new_elem;

        Ok(())
    }

    fn select_element<T>(slice: &mut [T], index: usize) -> Result<&mut T, TestCaseError> {
        let len = slice.len();
        if len == 0 {
            let reason = "cannot modify empty list".into();
            return Err(TestCaseError::Reject(reason));
        }

        Ok(&mut slice[index % len])
    }
}
