//! The [STIR](Stir) polynomial low-degree test over the
//! [extension field](XFieldElement).
//
// Future enhancements may include:
//
// - A folding factor that varies between rounds.
// - “Domain shrinkage” that varies between rounds.
// - Giacomo et al.: “Interactive Proofs for Batch Polynomial Evaluation”. This
//   makes interpolation in the verifier superfluous.

use std::collections::HashMap;

use arbitrary::Arbitrary;
use itertools::Itertools;
use num_traits::ConstOne;
use twenty_first::math::traits::FiniteField;
use twenty_first::prelude::*;

use crate::arithmetic_domain::ArithmeticDomain;
use crate::error::LdtParameterError;
use crate::error::LdtProvingError;
use crate::error::LdtVerificationError;
use crate::error::USIZE_TO_U64_ERR;
use crate::low_degree_test::LowDegreeTest;
use crate::low_degree_test::ProverResult;
use crate::low_degree_test::ProximityRegime;
use crate::low_degree_test::ReedSolomonCode;
use crate::low_degree_test::SetupResult;
use crate::low_degree_test::VerifierPostscript;
use crate::low_degree_test::VerifierResult;
use crate::profiler::profiler;
use crate::proof_item::AuthenticationStructure;
use crate::proof_item::ProofItem;
use crate::proof_stream::ProofStream;
use crate::table::master_table::BfeSlice;

/// An [`ArithmeticDomain`] can have at most 2^32 elements. Converting a usize
/// that represents a (valid) domain index to a u32 can never fail.
const DOMAIN_INDEX_TO_U32_ERR: &str = "internal error: domain index should be a valid u32";

/// The quotient set used in any round of STIR must never be larger than or
/// equal to the degree of the polynomial of that round.
/// The polynomial degree, in turn, is upper bounded by the length of the
/// [domain](ArithmeticDomain) for that round, which is at most 2^32.
/// Consequently, converting the length of the quotient set to a u32 can never
/// fail.
const QUOTIENT_SET_LEN_TO_U32_ERR: &str =
    "internal error: length of quotient set should be a valid u32";

/// The initial parameters from which to derive a [STIR](Stir) instance.
///
/// This struct captures the defining protocol parameters. It can be used to
/// [create an instance of STIR](Stir::new).
/// Note that this is a fallible operation because there are invalid parameter
/// combinations. The documentation on this struct's fields informs about the
/// legal parameter space.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
pub struct StirParameters {
    /// The desired security level in bits.
    ///
    /// See also: [`Stark::security_level`](crate::Stark::security_level).
    #[cfg_attr(test, strategy(16_usize..=192))]
    pub security_level: usize,

    /// The soundness-influencing assumption (or lack thereof) you are willing
    /// to make.
    pub soundness: ProximityRegime,

    /// Corresponds to the (log₂ of the) paper's folding factor `k`.
    ///
    /// Must be greater than or equal to 2, _i.e._, `k` must be greater than or
    /// equal to 4.
    //
    // The paper allows for this to change between rounds. This (current)
    // implementation does not.
    #[cfg_attr(test, strategy(2_usize..=5))]
    pub log2_folding_factor: usize,

    /// The amount of “redundancy” in the [prover](Stir::prove)'s input.
    ///
    /// In particular, the Reed-Solomon code's rate is the reciprocal of 2
    /// raised to this value. In other words, the initial rate equals
    /// `1 / 2^initial_log2_expansion_factor`.
    ///
    /// Must be greater than 0.
    #[cfg_attr(test, strategy(1_usize..=6))]
    pub log2_initial_expansion_factor: usize,

    /// The low-degree test's degree bound.
    ///
    /// In particular, the (log₂ of the) polynomial degree that is considered
    /// “high” (_i.e._, “not low”) for this STIR instance.
    ///
    /// In other words, the low-degreeness of polynomials with degree
    /// `2^log2_high_degree_bound` (and higher) cannot be [proven](Stir::prove)
    /// (in a way that the [verifier](Stir::verify) accepts with high
    /// probability). On the other hand, the low-degreeness of polynomials with
    /// degree `2^log2_high_degree_bound - 1` (and lower) _can_ be proven.
    ///
    /// Must be greater than or equal to the (log₂ of the)
    /// [folding factor](Self::log2_folding_factor).
    #[cfg_attr(test, strategy(#log2_folding_factor..=15))]
    pub log2_high_degree_bound: usize,
}

/// The “Shift to Improve Rate” (“[STIR][stir]”) low-degree test
/// (“[LDT](LowDegreeTest)”).
///
/// [stir]: https://eprint.iacr.org/2024/390.pdf
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Stir {
    /// The domain for the initial codeword of the [prover](Stir::prove).
    ///
    /// See [StirParameters::initial_domain] for more details.
    initial_domain: ArithmeticDomain,

    /// The (actual) folding factor.
    ///
    /// Corresponds to the paper's “k”. Guaranteed to be a power of 2.
    ///
    /// See also: [StirParameters::log2_folding_factor].
    folding_factor: usize,

    /// The number of queries for each of the full rounds.
    ///
    /// The length of the vector corresponds to the number of full rounds.
    ///
    /// Note that there is an additional, final round. The final round is not
    /// a full round since it doesn't contain a quotienting step.
    round_queries: Vec<NumQueries>,

    /// The number of in-domain queries for the final round.
    ///
    /// Corresponds to the paper's “repetition parameter” t_M.
    ///
    /// See also: [NumQueries::in_domain].
    final_num_in_domain_queries: usize,

    /// The degree of the final polynomial.
    ///
    /// Because folding can never (guarantee to) produce the zero-polynomial,
    /// this is not of type [isize].
    final_degree: usize,
}

/// A [STIR](Stir) round's revealed values together with an authentication
/// structure.
#[derive(Debug, Clone, Eq, PartialEq, Hash, BFieldCodec, Arbitrary)]
pub struct StirResponse {
    /// The revealed values of the round polynomial `f_i` for all verifier
    /// queries.
    ///
    /// Each element is the answer to one query `q` to the `k`-wise folded
    /// polynomial. This means (in case the prover is honest) the element
    /// contains the evaluations of the round polynomial `f_i` in each of the
    /// k-many k-th roots of `q`. In other words, each element corresponds to
    /// the evaluation of `f_i` at all `y` with `y^k == q`.
    pub queried_leafs: Vec<Vec<XFieldElement>>,

    /// The cryptographic data proving that the revealed leafs are included
    /// in some Merkle tree.
    ///
    /// Only useful in combination with a Merkle root and additional metadata,
    /// like the height of the tree. See also:
    /// [`MerkleTreeInclusionProof::authentication_structure`].
    pub auth_structure: AuthenticationStructure,
}

/// A postscript of a [STIR verification](Stir::verify).
///
/// For additional details, see [`VerifierPostscript`].
//
// Marked `#[non_exhaustive]` because
// 1. additional fields might be added in the future and I don't want that to be
//    a breaking change, and
// 2. this type is not intended to be constructed anywhere but in this module.
//
// Also applies to the other “Postscript” structs.
#[non_exhaustive]
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Postscript {
    /// The (partial) codeword used in the first round.
    pub partial_first_codeword: Vec<XFieldElement>,

    /// One round-postscript per full round of STIR.
    pub rounds: Vec<RoundPostscript>,

    /// The postscript of the final round of STIR.
    pub final_round: FinalRoundPostscript,
}

/// A postscript of a single full round of a [STIR verification](Stir::verify).
///
/// See [`Postscript`] for a full explanation.
#[non_exhaustive]
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RoundPostscript {
    /// The (unfolded) domain used in this round.
    pub domain: ArithmeticDomain,

    /// The folding factor used in this round.
    pub folding_factor: usize,

    /// The randomness used for folding the round's polynomial.
    pub folding_randomness: XFieldElement,

    /// The indices of the domain points that were queried.
    ///
    /// Note that these indices serve the same two purposes as the [Postscript]
    /// itself. For the secondary purpose, the indices must be post-processed,
    /// since the queried indices recorded here are with respect to the round's
    /// domain to serve the primary purpose. However, the queries the verifier
    /// actually makes are with respect to the round's _folded_ domain. The
    /// method [`folded_queried_indices`](Self::folded_queried_indices) applies
    /// the necessary post-processing.
    pub queried_indices: Vec<usize>,

    /// The out-of-domain points that were queried.
    pub queried_points: Vec<XFieldElement>,

    /// The randomness used to correct the polynomial's degree after
    /// quotienting.
    pub degree_correction_randomness: XFieldElement,

    /// The authentication structure proving the inclusion of the polynomial's
    /// evaluations at the [queried indices](Self::queried_indices).
    //
    // While this does not fit the theme of “sampled randomness”, it's useful
    // to have direct access to anything that will end up in Triton VM's source
    // for non-deterministically guessed digests.
    pub auth_structure: AuthenticationStructure,
}

/// A postscript of the final round of a [STIR verification](Stir::verify).
///
/// See [`Postscript`] for a full explanation.
#[non_exhaustive]
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct FinalRoundPostscript {
    /// See [RoundPostscript::domain].
    pub domain: ArithmeticDomain,

    /// See [RoundPostscript::folding_factor].
    pub folding_factor: usize,

    /// See [RoundPostscript::folding_randomness].
    pub folding_randomness: XFieldElement,

    /// See [RoundPostscript::queried_indices].
    pub queried_indices: Vec<usize>,

    /// See [RoundPostscript::auth_structure].
    pub auth_structure: AuthenticationStructure,
}

/// The number of oracle queries in a single, full round of [STIR](Stir).
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
struct NumQueries {
    /// Corresponds to the paper's “repetition parameter” t_i.
    ///
    /// Does include an error margin, or “oversampling amount”, to ensure that
    /// enough unique indices are sampled with overwhelming (in the security
    /// parameter) probability.
    /// See also [StirParameters::num_total_in_domain_queries].
    in_domain: usize,

    /// Corresponds to the paper's “repetition parameter” `s_i`.
    out_of_domain: usize,
}

/// A Merkle tree where the leafs are “stacks” of values.
///
/// In particular, records the pre-image for each leaf in the Merkle tree in a
/// way that is helpful for [STIR](Stir): Each such pre-image is a “stack” of
/// values. This allows shorter Merkle inclusion proofs in STIR, since the
/// verifier requires all entries in a stack if it requires any entry in the
/// stack. Therefore, the entire stack might as well be the Merkle tree's leaf's
/// underlying value.
///
/// # Example
///
/// Say you want to build a Merkle tree that commits to the values
/// `[0, 1, 2, 3, 4, 5, 6, 7]`. Usually, the Merkle tree would look like this:
///
/// ```markdown
///         ──── _ ────
///        ╱           ╲
///       _             _
///      ╱  ╲          ╱  ╲
///    _      _      _      _
///   ╱ ╲    ╱ ╲    ╱ ╲    ╱ ╲
///  _   _  _   _  _   _  _   _
///  ↑   ↑  ↑   ↑  ↑   ↑  ↑   ↑
///  0   1  2   3  4   5  6   7
/// ```
///
/// A Merkle tree with a stack height of 2 instead looks like this:
///
/// ```markdown
///         ──── _ ────
///        ╱           ╲
///       _             _
///      ╱  ╲          ╱  ╲
///    _      _      _      _
///    ↑      ↑      ↑      ↑
/// ╭──┴─╮ ╭──┴─╮ ╭──┴─╮ ╭──┴─╮
/// [0, 4] [1, 5] [2, 6] [3, 7]
/// ```
///
/// A Merkle tree with a stack height of 4 instead looks like this:
///
/// ```markdown
///             _
///           ╱   ╲
///          _     _
///          ↑     ↑
/// ╭────────┴─╮ ╭─┴────────╮
/// [0, 2, 4, 6] [1, 3, 5, 7]
/// ```
struct StirMerkleTree {
    stacked_leafs: Vec<Vec<XFieldElement>>,
    tree: MerkleTree,
}

/// All answered queries to the folded polynomial of one round, including the
/// cryptographic data to prove that the answers had been committed to.
///
/// Conceptually, this is an extended [`MerkleTreeInclusionProof`] that also
/// tracks the actual queries to the folding polynomial the prover has commited
/// to.
///
/// The most important functionality of this struct is to provide the
/// [authenticated queries](Self::authenticated_queries).
struct FoldingPolynomialQueriesInclusionProof {
    queries: Vec<FoldingPolynomialQuery>,
    auth_structure: AuthenticationStructure,
    folded_domain_len: usize,
}

/// The result of a query to the folding polynomial, yielding `k`-many
/// (i.e., `folding_factor`-many) answers from the oracle.
struct FoldingPolynomialQuery {
    /// The index of the [point](Self::point) within that round's domain.
    index: usize,

    /// Corresponds to r_{i,j}^{shift} in the STIR paper.
    point: BFieldElement,

    /// A `k`-th root of [`Self::point`].
    ///
    /// In other words, [`Self::root`]^`k` = [`Self::point`].
    root: BFieldElement,

    /// A primitive `k`-th root of unity.
    ///
    /// In some sense, the “distance” between any two `k`-th [roots](Self::root)
    /// of [`Self::point`]. In particular, the following gives all `k` such
    /// `k`-th roots:
    ///
    /// ```no_compile
    /// (0..k).map(|j| root * primitive_kth_root_of_unity.mod_pow(j))
    /// ```
    //
    // This is the same for all queries within one round. This duplication
    // wastes memory on the order of `num_queries * BFieldElement::BYTES`,
    // which is about 1KiB. For Triton VM, that's next to nothing.
    //
    // In the current implementation, this could probably be removed from this
    // struct and replaced by
    // `BFieldElement::primitive_root_of_unity(self.folding_factor)`
    // where needed. With an eye to future changes, in particular, a variable
    // folding factor, this seems more future-proof.
    primitive_kth_root_of_unity: BFieldElement,

    /// Corresponds to the evaluation of f_i at all `y` with
    /// `y^k == point`.
    values: Vec<XFieldElement>,
}

/// The required data to perform a quotienting step.
///
/// To compute the folding oracle for the next round, this data is needed in
/// addition to the data dequeued from the proof stream.
///
/// The exception to this rule is the initial round, which does not require
/// this data. Note that the initial round might coincide with the final round,
/// in which case this struct is not created.
struct QuotientingData {
    quotient_set: Vec<XFieldElement>,
    quotient_answers: Vec<XFieldElement>,
    degree_correction_randomness: XFieldElement,
}

impl TryFrom<StirParameters> for Stir {
    type Error = LdtParameterError;

    fn try_from(params: StirParameters) -> SetupResult<Self> {
        params.try_into_stir()
    }
}

impl StirParameters {
    /// The number of bits required to represent an [XFieldElement].
    ///
    /// While the value computed here is slightly higher than the more correct
    /// log₂([BFieldElement::P] · [x_field_element::EXTENSION_DEGREE]), the
    /// difference is about 1.0e-9, while the rounding error for the “actual”
    /// log₂(|𝔽|) in double-precision IEEE 754 floating point format (i.e.,
    /// rust's [f64]) is about 1.2e-7.
    const LOG2_FIELD_SIZE: u32 =
        (BFieldElement::BYTES * 8 * x_field_element::EXTENSION_DEGREE) as u32;

    /// The (log₂ of the) relative length difference of the evaluation
    /// domains of two consecutive rounds.
    //
    // In this current implementation, the Reed-Solomon code's domain shrinks
    // by the same amount between any two rounds. While the STIR paper does not
    // mention that this could be variable, it seems plausible that it could.
    // A future change to this implementation might make this variable.
    const LOG2_DOMAIN_SHRINKAGE: usize = 1;

    /// The highest polynomial degree for which low-degreeness can be proven
    /// with the STIR instance corresponding to these parameters.
    pub fn max_degree(&self) -> usize {
        (1 << self.log2_high_degree_bound) - 1
    }

    pub(crate) fn expansion_factor(&self) -> usize {
        let err = "internal error: log₂(expansion factor) exceeds expected maximum";
        let log2_expansion_factor = u32::try_from(self.log2_initial_expansion_factor).expect(err);

        1_usize.checked_shl(log2_expansion_factor).expect(err)
    }

    /// Create a new STIR instance by deriving the round parameters.
    fn try_into_stir(&self) -> SetupResult<Stir> {
        if self.log2_folding_factor < 2 {
            return Err(LdtParameterError::TooSmallLog2FoldingFactor(
                self.log2_folding_factor,
            ));
        }
        if self.log2_initial_expansion_factor == 0 {
            return Err(LdtParameterError::TooSmallInitialExpansionFactor);
        }
        if self.log2_high_degree_bound < self.log2_folding_factor {
            return Err(LdtParameterError::TooLowDegreeOfHighDegreePolynomials);
        }

        let Ok(log2_folding_factor) = u32::try_from(self.log2_folding_factor) else {
            return Err(LdtParameterError::TooBigLog2FoldingFactor(
                self.log2_folding_factor,
            ));
        };
        let Some(folding_factor) = 1_usize.checked_shl(log2_folding_factor) else {
            return Err(LdtParameterError::TooBigLog2FoldingFactor(
                self.log2_folding_factor,
            ));
        };

        // none of the round parameters depend on the non-folded polynomial
        let mut folded_poly_degree = self.max_degree() / folding_factor;
        let mut log2_expansion_factor = self.log2_initial_expansion_factor;
        let initial_domain = self.initial_domain()?;
        let mut log2_folded_domain_size = initial_domain.len().ilog2() - log2_folding_factor;
        let mut round_queries = Vec::new();

        // Folding lowers the current degree to ⌊poly_degree / folding_factor⌋.
        // Since this “bottoms out” at 0, folding too often can lead to
        // soundness problems.
        // For example, assume that the highest legal polynomial degree for the
        // current round is 2 and the folding factor is 4. Folding results in a
        // polynomial of degree 0. However, folding a polynomial of degree 3
        // _also_ results in a polynomial of degree 0, even though the degree-3
        // polynomial should be rejected by the verifier. The solution is to
        // stop folding before the process “bottoms out.” Since there is one
        // more folding step applied in the final round, no full round must
        // reduce the degree to be less than or equal to the folding factor.
        //
        // Also note that a STIR instance with a “high degree” of less than the
        // folding factor is degenerate and will have been rejected earlier in
        // this method.
        while folded_poly_degree > folding_factor {
            let in_domain =
                self.num_in_domain_queries(log2_folded_domain_size, log2_expansion_factor)?;

            // Because the out-of-domain queries are made to the folded
            // polynomial, the number of out-of-domain queries are set with
            // respect to the new rate.
            let log2_next_expansion_factor =
                log2_expansion_factor + self.log2_folding_factor - Self::LOG2_DOMAIN_SHRINKAGE;
            let log2_folded_poly_degree = folded_poly_degree.ilog2();
            let out_of_domain =
                self.num_ood_queries(log2_folded_poly_degree, log2_next_expansion_factor)?;

            let num_queries = NumQueries {
                in_domain,
                out_of_domain,
            };

            // In STIR's quotienting step, the so-called “Ans” (probably meaning
            // “answer”) polynomial is subtracted from the folded polynomial.
            // Both prover and verifier compute the “Ans” polynomial by
            // interpolating over all queries and their – well – answers.
            //
            // If the “answer” polynomial _equals_ the folded polynomial, the
            // subsequent quotient is the zero-polynomial. This equality happens
            // if the total number of queries exceeds the folded polynomial's
            // degree.
            //
            // It has the following consequences:
            // 1. Even though the next round's polynomial is the
            //    degree-corrected quotient polynomial, no amount of
            //    degree-correction can turn the zero-polynomial into anything
            //    but the zero-polynomial. If the folded polynomial has a degree
            //    that is too high, but the quotient ends up being the
            //    zero-polynomial, then degree-correction cannot recover the
            //    too-high-degree polynomial. In other words, the verifier will
            //    incorrectly accept a polynomial of high degree.
            // 2. The verifier is doing more work than is optimal, because it
            //    just recovered the entirety of the folded polynomial. At that
            //    point, the protocol might as well stop.
            //
            // Note that the first consequence does not apply to the final
            // round, since the final round has no quotienting step. If the
            // second consequence applies, it does not break any soundness
            // guarantees, the protocol only becomes less efficient.
            //
            // Note also that it's possible that some indices are queried
            // multiple times. Technically, the following check only applies if
            // the total number of _unique_ queries exceeds the polynomial's
            // degree. However, there is no way of knowing the number of unique
            // indices ahead of time. Consequently, this check is rather
            // conservative and always assumes that all sampled queries are
            // unique.
            let next_folded_poly_deg = folded_poly_degree / folding_factor;
            if num_queries.total() > next_folded_poly_deg {
                break;
            }

            round_queries.push(num_queries);
            folded_poly_degree = next_folded_poly_deg;
            log2_expansion_factor = log2_next_expansion_factor;
            log2_folded_domain_size -= Self::LOG2_DOMAIN_SHRINKAGE as u32;
        }

        let final_num_in_domain_queries =
            self.num_in_domain_queries(log2_folded_domain_size, log2_expansion_factor)?;
        let final_degree = folded_poly_degree;
        let stir = Stir {
            initial_domain,
            folding_factor,
            round_queries,
            final_num_in_domain_queries,
            final_degree,
        };

        Ok(stir)
    }

    /// The domain for the initial codeword of the [prover](Stir::prove).
    //
    // The first part of this method essentially only computes:
    // 1 << (log2_high_degree_bound + log2_initial_expansion_factor)
    //
    // However, because the input parameters can be waaaayy too big, a bunch of
    // input validation is… necessary? Let's go with “beneficial.” Whether the
    // removal of this would be a DOS against Triton VM is somewhat doubtful,
    // but hey, better safe than sorry.
    fn initial_domain(&self) -> SetupResult<ArithmeticDomain> {
        let as_u64 = |int| u64::try_from(int).expect(USIZE_TO_U64_ERR);
        let error = |x| Err(LdtParameterError::InitialDomainTooBig(x));

        let log2_high_degree_bound = as_u64(self.log2_high_degree_bound);
        let log2_expansion_factor = as_u64(self.log2_initial_expansion_factor);
        let Some(log2_domain_len) = log2_high_degree_bound.checked_add(log2_expansion_factor)
        else {
            return error(u64::MAX);
        };
        let Ok(log2_domain_len) = u32::try_from(log2_domain_len) else {
            return error(log2_domain_len);
        };
        let Some(domain_len) = 1_usize.checked_shl(log2_domain_len) else {
            return error(log2_domain_len.into());
        };

        let domain = ArithmeticDomain::of_length(domain_len)
            .map_err(|_| LdtParameterError::InitialDomainTooBig(log2_domain_len.into()))?
            .with_offset(BFieldElement::generator());

        Ok(domain)
    }

    /// The total number of in-domain queries to make, including error margin.
    ///
    /// For further details, see [Self::num_total_in_domain_queries].
    fn num_in_domain_queries(
        &self,
        log2_domain_size: u32,
        log2_expansion_factor: usize,
    ) -> Result<usize, LdtParameterError> {
        let num_uniques = self.num_unique_in_domain_queries(log2_expansion_factor)?;

        // it doesn't make sense to request more unique indices than there are
        let num_uniques = num_uniques.min(1 << log2_domain_size);
        let num_total = self.num_total_in_domain_queries(log2_domain_size, num_uniques);

        Ok(num_total)
    }

    /// The number of required (unique) in-domain queries for the Reed-Solomon
    /// code implied by the parameters.
    ///
    /// Because any two queries might be identical, this number by itself is
    /// usually not enough. See also [Self::num_total_in_domain_queries].
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
    // With all of these ingredients, we solve for repetition parameter t by
    // taking the log₂ on both sides:
    //   t·log₂(1-δ) ⩽ -security_level
    //
    // Since 0 < δ < 1, we know that log₂(1 - δ) < 0:
    //             t ⩾ -security_level / log₂(1-δ)
    fn num_unique_in_domain_queries(&self, log2_expansion_factor: usize) -> SetupResult<usize> {
        let rs_code = ReedSolomonCode::new(log2_expansion_factor).with_soundness(self.soundness);
        let proximity_parameter = rs_code.proximity_parameter()?;
        let num_queries = -(self.security_level as f64) / (1.0 - proximity_parameter).log2();

        Ok(num_queries.ceil() as usize)
    }

    /// The total amount of in-domain queries to make, including error margin.
    ///
    /// Since STIR cannot tolerate re-use of any in-domain index, duplicates
    /// have to be rejected. In order to sample unique indices efficiently,
    /// all indices are sampled in one go and duplicates are removed afterward.
    /// This requires a certain error margin, which is added here.
    ///
    /// Note that the parameter `log2_domain_len` is that of the _folded_
    /// domain: even though indices are sampled from the round's domain, they
    /// have to be unique on the folded domain. See also:
    /// [RoundPostscript::queried_indices].
    //
    // The goal of this method is to figure out the smallest total number of
    // queries that will yield the required number of unique indices with a high
    // enough probability. The chosen probability is based on the requested
    // soundness level; in particular, the probability that the number of total
    // samples does _not_ include enough unique values must be less than or
    // equal to 2^-λ. This is in line with all other probabilities used for
    // failure events throughout this codebase.
    //
    // The combinatorics involved in this function are somewhat involved. In
    // order to decrease the space required for the various equations and
    // formulas below, we use the following notation:
    // - U = `domain.len()` is the size of the universe,
    // - k = `num_in_domain_queries` is the number of unique samples we need,
    // - D is the actual number of unique samples, and
    // - n is the total number of samples. This method returns n.
    //
    // The failure event is D < k, and so we are interested in the probability
    // Pr[D < k] = Σ_(d=1)^(k-1) Pr[D = d].
    //
    // To identify Pr[D = d]:
    // 1. Choose d values from U, for which there are (U choose d) options.
    // 2. For a fixed set of d labeled values, count sequences of length n
    //    that use only those values and use each value at least once. View such
    //    a sequence as a surjective function f: [n] → [d].
    //    The number of such functions almost coincides with the Stirling number
    //    of the second kind, S(n, d), which counts the number of ways to
    //    partition a set of n labeled objects into d nonempty unlabelled
    //    subsets. However, we must account for the fact that our subsets are
    //    labeled. The number of labellings is d!.
    // 3. Dividing the product of the terms identified above by the total number
    //    of ordered sequences, U^n, gives the sought-after probability:
    //
    //    Pr[D = d] = ((U choose d)·d!·S(n, d)) / U^n
    //              = …
    //              = (U choose d) · Σ_(i=0)^d (-1)^(d-i)·(d choose i)·(i/U)^n
    //
    // This is complicated enough that I don't know how to solve
    // Pr[D < k] ⩽ 2^-λ for n. Numerical problems prevented me from directly
    // using the probability formula in a simple search for the smallest n. In
    // the end, I think it's best to bite the bullet of upper-bounding the
    // probability.
    //
    // As a first step, note that the sum in Pr[D = d] is the number of
    // surjective functions from a set of size n onto a fixed set of size d.
    // If we remove the constraint that those functions have to be surjective
    // and take all functions instead, the bound becomes simpler:
    //
    //    Pr[D = d] ⩽ (U choose d) · (d/U)^n
    //
    // Using this to upper-bound the probability Pr[D < k] still has a somewhat
    // unwieldy summation going on:
    //
    //    Pr[D < k] ⩽ Σ_(d=1)^(k-1) (U choose d) · (d/U)^n
    //
    // We upper-bound this further by replacing each summand with the largest
    // of them all. The factor `(d/U)^n` can easily be upper bounded by
    // `((k-1)/U)^n`. The binomial coefficient requires a little more care. For
    // a given `U`, the largest binomial coefficient is `(U choose U//2)`. If
    // `k - 1 > U // 2`, then that largest binomial coefficient should be used.
    // Otherwise, the largest binomial coefficient in any of the summands is
    // `(U choose k-1)`. To finish simplifying the sum, define
    // `l = min(k - 1, U // 2)` and use it in the binomial coefficient:
    //
    //    Pr[D < k] ⩽ (k-1) · (U choose l) · ((k-1)/U)^n
    //
    // Finally, the expression starts to approach something workable. Now, we
    // can use the upper bound by the security level to solve for n:
    //
    //                         (k-1) · (U choose l) · ((k-1)/U)^n ⩽ 2^-λ
    //  ↔  log₂(k-1) + log₂(U choose l) + n·(log₂(k-1) - log₂(U)) ⩽ -λ
    //  ↔  n·(log₂(k-1) - log₂(U)) ⩽ -λ - log₂(k-1) - log₂(U choose l)
    //  ↔  n·(log₂(U) - log₂(k-1)) ⩾ λ + log₂(k-1) + log₂(U choose l)
    //
    //         λ + log₂(k-1) + log₂(U choose l)
    //  ↔  n ⩾ ────────────────────────────────
    //               log₂(U) - log₂(k-1)
    //
    // That equation is what we use in the method body. A few points of
    // discussion remain:
    //
    // 1. It's quite possible that the bounds used in the above derivation can
    //    be improved upon. If you find better bounds, feel free to change
    //    things around.
    // 2. How bad are the effects of the upper bounds? Good question. Short
    //    answer: for the parameter ranges we are interested in, the bounds in
    //    use lead to a 2-10% increase in indices to sample over what would be
    //    optimal. For example, for λ=160, u=2^23, and k=160, the optimal n is
    //    173 (error margin 13), while this method indicates n=184 (error
    //    margin 24). The 11 additional indices are superfluous overhead of
    //    about ~6.4%.
    //    This is not to say that the error margin is always that small. In
    //    particular, as k approaches u, the (optimal!) error margin can exceed
    //    k by multiple factors. For example, take λ=160, u=2^8, and k=160. Then
    //    the optimal n is 574, an error margin of 414! This method indicates
    //    n=610, and the 36 superfluous indices are an overhead of ~6.3%.
    // 3. Choosing rejection sampling over oversampling has pros and cons:
    //    + This method with all its complexity becomes superfluous.
    //    + The number of additional indices is as small as possible since it
    //      doesn't rely on probabilistic arguments.
    //    - The logic for sampling indices becomes significantly more complex,
    //      particularly in a Triton assembly context.
    //    We decided that the runtime complexity of rejection sampling in Triton
    //    assembly (O(num_unique_indices²)) dominates the other points.
    //    Tracking the actually sampled indices and the folded & de-duplicated
    //    indices is required in either case.
    fn num_total_in_domain_queries(
        &self,
        log2_domain_len: u32,
        num_in_domain_queries: usize,
    ) -> usize {
        let k = u64::try_from(num_in_domain_queries).expect(USIZE_TO_U64_ERR);
        let k_minus_1 = k.checked_sub(1).expect("internal error: too few queries");
        let domain_len = 1 << log2_domain_len;
        let l = k_minus_1.min(domain_len / 2);
        let log2_u_choose_l = log2_binomial_coefficient(domain_len, l);

        // for edge case `k == 1`, clamp log to 0
        let log2_k_minus_1 = (k_minus_1 as f64).log2().max(0.0);
        let security_level = self.security_level as f64;
        let log2_domain_len = f64::from(log2_domain_len);
        let num_total_queries = (security_level + log2_k_minus_1 + log2_u_choose_l)
            / (log2_domain_len - log2_k_minus_1);

        num_total_queries.ceil() as usize
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
    // of this round's polynomial, 𝔽 is the field the Reed-Solomon code (and by
    // extension, STIR) is defined over (in our case always the extension
    // field), and D is this round's domain over which the polynomial is
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
    // equivalent to the polynomial X. While distinct as polynomials, it's
    // impossible to differentiate between the two in evaluation form.
    // It's also hard to argue that such polynomials are of “low degree”.
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
    // floating point format (i.e., rust's `f64`) gives a rounding error of
    // about 1.2e-7. In other words, accurately computing log₂(|𝔽| - |D|)
    // requires a specialized, high-precision floating point library. We're
    // not doing that.
    //
    // Long story short:
    //   s ⩾ (security_level - 1 + 2·log₂(ℓ)) / (log₂(|𝔽|) - log₂(d))
    fn num_ood_queries(
        &self,
        log2_poly_degree: u32,
        log2_expansion_factor: usize,
    ) -> SetupResult<usize> {
        let rs_code = ReedSolomonCode::new(log2_expansion_factor).with_soundness(self.soundness);
        let log2_list_size = rs_code.log2_list_size(log2_poly_degree)?;
        let num_ood_queries = (self.security_level as f64 - 1.0 + 2.0 * log2_list_size)
            / f64::from(Self::LOG2_FIELD_SIZE - log2_poly_degree);

        Ok(num_ood_queries.ceil() as usize)
    }
}

/// The log₂ of the binomial coefficient (a choose b).
///
/// # Panics
///
/// Panics if `b` is smaller than `a`.
//
// While it is valid to define the binomial coefficient (a choose b) as 0
// if `b < a`, we don't need this functionality. I'd rather things fail
// early and hard instead.
fn log2_binomial_coefficient(a: u64, b: u64) -> f64 {
    assert!(a >= b, "internal error: binomial coefficient with b < a");

    // Kahan-Babuška summation for better numerical stability
    let mut log2_binom = 0.0;
    let mut compensation = 0.0;
    for i in 0..b.min(a - b) {
        let summand = ((a - i) as f64).log2() - ((i + 1) as f64).log2();
        let corrected_summand = summand - compensation;
        let next_log2_binom = log2_binom + corrected_summand;
        compensation = (next_log2_binom - log2_binom) - corrected_summand;
        log2_binom = next_log2_binom;
    }

    log2_binom
}

impl super::private::Seal for Stir {}

impl LowDegreeTest for Stir {
    fn initial_domain(&self) -> ArithmeticDomain {
        self.initial_domain
    }

    fn num_first_round_queries(&self) -> usize {
        self.round_queries
            .first()
            .map_or(self.final_num_in_domain_queries, |query| query.in_domain)
    }

    fn prove(
        &self,
        codeword: &[XFieldElement],
        proof_stream: &mut ProofStream,
    ) -> ProverResult<Vec<usize>> {
        profiler!(start "initialize");
        let mut domain = self.initial_domain;
        if domain.len() != codeword.len() {
            return Err(LdtProvingError::InitialCodewordMismatch {
                domain_len: domain.len(),
                codeword_len: codeword.len(),
            });
        }

        let folding_factor = self.folding_factor;
        let mut commitment = StirMerkleTree::new(codeword, folding_factor);
        proof_stream.enqueue(ProofItem::MerkleRoot(commitment.tree.root()));

        let mut poly = domain.interpolate(codeword);
        let mut first_round_queried_indices = None;
        profiler!(stop "initialize");

        for num_queries in &self.round_queries {
            profiler!(start "full rounds");
            let folding_randomness = proof_stream.sample_scalars(1)[0];
            let folded_poly = Self::fold_polynomial(&poly, folding_factor, folding_randomness);
            let next_round_domain = Self::next_round_domain(domain);

            let folded_evaluations = next_round_domain.evaluate(&folded_poly);
            let folded_poly_commitment = StirMerkleTree::new(&folded_evaluations, folding_factor);
            proof_stream.enqueue(ProofItem::MerkleRoot(folded_poly_commitment.tree.root()));

            let ood_queries = proof_stream.sample_scalars(num_queries.out_of_domain);
            let ood_values = ood_queries
                .iter()
                .map(|&x| folded_poly.evaluate_in_same_field(x))
                .collect_vec();
            proof_stream.enqueue(ProofItem::StirOutOfDomainValues(ood_values.clone()));

            // See [RoundPostscript::queried_indices] for an explanation why
            // the queries are sampled from the full domain but used with
            // respect to the folded domain most of the time.
            let queried_indices = proof_stream.sample_indices(domain.len(), num_queries.in_domain);
            let folded_domain = domain.pow(folding_factor).unwrap();
            let folded_queried_indices = queried_indices
                .iter()
                .map(|&idx| idx % folded_domain.len())
                .unique()
                .collect_vec();
            let inclusion_proof = commitment.inclusion_proof(&folded_queried_indices);
            proof_stream.enqueue(ProofItem::StirResponse(inclusion_proof));

            // construct the witness polynomial for the next round
            let queried_domain_values = folded_queried_indices
                .iter()
                .map(|&i| u32::try_from(i).expect(DOMAIN_INDEX_TO_U32_ERR))
                .map(|i| folded_domain.value(i))
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
            domain = next_round_domain;
            commitment = folded_poly_commitment;

            first_round_queried_indices.get_or_insert(queried_indices);
            profiler!(stop "full rounds");
        }

        // final round is special because there is no quotienting
        profiler!(start "final round");
        let folding_randomness = proof_stream.sample_scalars(1)[0];
        let poly = Self::fold_polynomial(&poly, folding_factor, folding_randomness);
        proof_stream.enqueue(ProofItem::Polynomial(poly));

        let folded_domain = domain.pow(folding_factor).unwrap();
        let queried_indices =
            proof_stream.sample_indices(domain.len(), self.final_num_in_domain_queries);
        let folded_poly_queried_indices = queried_indices
            .iter()
            .map(|&idx| idx % folded_domain.len())
            .unique()
            .collect_vec();
        let inclusion_proof = commitment.inclusion_proof(&folded_poly_queried_indices);
        proof_stream.enqueue(ProofItem::StirResponse(inclusion_proof));
        profiler!(stop "final round");

        Ok(first_round_queried_indices.unwrap_or(queried_indices))
    }

    fn verify(&self, proof_stream: &mut ProofStream) -> VerifierResult<VerifierPostscript> {
        profiler!(start "initialize");
        let mut partial_first_codeword = None;
        let mut previous_quotienting_data = None;
        let mut round_postscripts = Vec::with_capacity(self.round_queries.len());
        let mut domain = self.initial_domain;
        let mut commitment_to_previous_polynomial =
            proof_stream.dequeue()?.try_into_merkle_root()?;
        profiler!(stop "initialize");

        for num_queries in &self.round_queries {
            profiler!(start "full rounds");
            let folding_randomness = proof_stream.sample_scalars(1)[0];
            let commitment_to_current_polynomial =
                proof_stream.dequeue()?.try_into_merkle_root()?;
            let ood_queries = proof_stream.sample_scalars(num_queries.out_of_domain);
            let ood_answers = proof_stream.dequeue()?.try_into_stir_ood_values()?;
            let inclusion_proof =
                self.extract_inclusion_proof(proof_stream, domain, num_queries.in_domain)?;
            let queried_indices = inclusion_proof.queried_indices().collect();
            let auth_structure = inclusion_proof.auth_structure.clone();
            let queries =
                inclusion_proof.authenticated_queries(commitment_to_previous_polynomial)?;
            partial_first_codeword.get_or_insert_with(|| self.partial_codeword(domain, &queries));

            let in_domain_answers = match previous_quotienting_data {
                None => Self::initial_in_domain_answers(&queries, folding_randomness),
                Some(qd) => Self::subsequent_in_domain_answers(qd, &queries, folding_randomness),
            };

            // queried indices are not generally unique, but have to be for
            // interpolation (in the next round) to work
            let (quotient_set, quotient_answers) = queries
                .into_iter()
                .map(|q| q.point.lift())
                .zip(in_domain_answers)
                .chain(ood_queries.iter().copied().zip(ood_answers))
                .unique_by(|(query_point, _)| *query_point)
                .unzip();
            let degree_correction_randomness = proof_stream.sample_scalars(1)[0];
            let quotienting_data = QuotientingData {
                quotient_set,
                quotient_answers,
                degree_correction_randomness,
            };

            // record all sampled randomness as well as other helpful data
            round_postscripts.push(RoundPostscript {
                domain,
                folding_factor: self.folding_factor,
                folding_randomness,
                queried_indices,
                queried_points: ood_queries,
                degree_correction_randomness,
                auth_structure,
            });

            // prepare next (or final) round
            previous_quotienting_data = Some(quotienting_data);
            domain = Self::next_round_domain(domain);
            commitment_to_previous_polynomial = commitment_to_current_polynomial;
            profiler!(stop "full rounds");
        }

        profiler!(start "final round");
        let folding_randomness = proof_stream.sample_scalars(1)[0];
        let poly = proof_stream.dequeue()?.try_into_polynomial()?;

        // for the low, low chance that the final polynomial is the zero
        // polynomial, we treat it as if it was a constant polynomial when
        // checking the degree
        let poly_degree = poly.degree().try_into().unwrap_or(0);
        if poly_degree > self.final_degree {
            return Err(LdtVerificationError::LastRoundPolynomialHasTooHighDegree);
        }

        let num_queries = self.final_num_in_domain_queries;
        let inclusion_proof = self.extract_inclusion_proof(proof_stream, domain, num_queries)?;
        let queried_indices = inclusion_proof.queried_indices().collect();
        let auth_structure = inclusion_proof.auth_structure.clone();
        let queries = inclusion_proof.authenticated_queries(commitment_to_previous_polynomial)?;

        let final_answers = match previous_quotienting_data {
            None => Self::initial_in_domain_answers(&queries, folding_randomness),
            Some(qd) => Self::subsequent_in_domain_answers(qd, &queries, folding_randomness),
        };
        let final_evaluations = queries.iter().map(|query| poly.evaluate(query.point));
        if final_answers
            .into_iter()
            .zip(final_evaluations)
            .any(|(answer, evaluation)| answer != evaluation)
        {
            return Err(LdtVerificationError::LastRoundPolynomialEvaluationMismatch);
        }

        let final_round_postscript = FinalRoundPostscript {
            domain,
            folding_factor: self.folding_factor,
            folding_randomness,
            queried_indices,
            auth_structure,
        };
        let partial_first_codeword =
            partial_first_codeword.unwrap_or_else(|| self.partial_codeword(domain, &queries));
        let postscript = Postscript {
            partial_first_codeword,
            rounds: round_postscripts,
            final_round: final_round_postscript,
        };
        profiler!(stop "final round");

        Ok(VerifierPostscript::Stir(postscript))
    }

    #[cfg(test)]
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl Stir {
    /// (Try to) construct a new STIR instance for the given initial parameters.
    ///
    /// It is equivalent to use the provided trait implementation of
    /// [`TryFrom<StirParameters> for Stir`](TryFrom).
    ///
    /// # Errors
    ///
    /// Errors if any of the pre-conditions documented in the [`StirParameters`]
    /// are violated.
    pub fn new(params: StirParameters) -> SetupResult<Self> {
        params.try_into()
    }

    /// # Panics
    ///
    /// Panics if the folding factor is 0.
    fn fold_polynomial<FF>(
        poly: &Polynomial<FF>,
        folding_factor: usize,
        folding_randomness: FF,
    ) -> Polynomial<'static, FF>
    where
        FF: FiniteField,
    {
        let folded_coefficients = poly
            .coefficients()
            .chunks(folding_factor)
            .map(|chunk| Polynomial::new_borrowed(chunk).evaluate(folding_randomness))
            .collect();

        Polynomial::new(folded_coefficients)
    }

    fn next_round_domain(domain: ArithmeticDomain) -> ArithmeticDomain {
        let next_domain = domain
            .pow(1 << StirParameters::LOG2_DOMAIN_SHRINKAGE)
            .unwrap();

        next_domain.with_offset(next_domain.offset() * domain.offset())
    }

    fn extract_inclusion_proof(
        &self,
        proof_stream: &mut ProofStream,
        round_domain: ArithmeticDomain,
        num_id_queries: usize,
    ) -> VerifierResult<FoldingPolynomialQueriesInclusionProof> {
        let queried_indices = proof_stream.sample_indices(round_domain.len(), num_id_queries);
        let StirResponse {
            queried_leafs,
            auth_structure,
        } = proof_stream.dequeue()?.try_into_stir_response()?;

        let folded_domain = round_domain.pow(self.folding_factor).unwrap();
        let folded_indices = queried_indices
            .iter()
            .map(|&index| index % folded_domain.len())
            .unique();
        if queried_leafs.len() != folded_indices.clone().count() {
            return Err(LdtVerificationError::IncorrectNumberOfRevealedLeaves);
        }

        // Because of above length check, we know that the hash map contains
        // exactly one entry per folded-domain index. This makes it safe to
        // directly index into the map using the folded-domain indices as keys.
        //
        // The reason this setup is seemingly convoluted is due to the need of
        // 1. de-duplication of folded-domain indices, and
        // 2. preservation of all sampled indices.
        //
        // In particular, if two sampled indices give the same folded-domain
        // index, both original indices must be recorded, and the corresponding
        // (identical) revealed leaf must be associated with both queries. This
        // requires some form of data structure from which the revealed leaf can
        // be fetched based on the folded-domain index – a map.
        let indexed_queried_leafs = folded_indices.zip(queried_leafs).collect::<HashMap<_, _>>();

        let folded_domain_len = u64::try_from(folded_domain.len()).expect(USIZE_TO_U64_ERR);
        let primitive_kth_root_of_unity = round_domain.generator().mod_pow(folded_domain_len);
        let folded_domain_len = usize::try_from(folded_domain_len).unwrap(); // ugh…

        let queries = queried_indices
            .into_iter()
            .map(|index| {
                let query_index = index % folded_domain_len;
                let values = indexed_queried_leafs[&query_index].clone();
                let query_index = u32::try_from(query_index).expect(DOMAIN_INDEX_TO_U32_ERR);
                let point = folded_domain.value(query_index);
                let root = round_domain.value(query_index);
                FoldingPolynomialQuery {
                    index,
                    point,
                    root,
                    primitive_kth_root_of_unity,
                    values,
                }
            })
            .collect();

        let inclusion_proof = FoldingPolynomialQueriesInclusionProof {
            queries,
            auth_structure,
            folded_domain_len,
        };

        Ok(inclusion_proof)
    }

    /// Get the query indices of the first round as well as the values of the
    /// initial codeword at those indices.
    ///
    /// # Panics
    ///
    /// Panics if the queries' answers are invalid, in particular, if there are
    /// fewer than `folding_factor` many answers.
    //
    // In the paper, the verifier queries indices in the _folded_ domain, and
    // the prover reveals k elements per such query. In order to link the
    // initial codeword of STIR into the greater STARK context, one of the k
    // revealed elements per query of the first round is used. But which one?
    //
    // We want the distribution of the linking indices to be uniformly random
    // over the range `0..initial_codeword.len()`.
    // Therefore, the query indices are sampled from this range and then mapped
    // to the range [0; folded_codeword.len()) simply by computing the modulus
    // with the folded domain's length. Per query, the prover answers with
    // k points. To fetch the desired point, the query index has to be mapped
    // into the range `0..k`. This is achieved through integer division by the
    // folded domain's length.
    fn partial_codeword(
        &self,
        domain: ArithmeticDomain,
        queries: &[FoldingPolynomialQuery],
    ) -> Vec<XFieldElement> {
        let folded_domain_len = domain.pow(self.folding_factor).unwrap().len();

        queries
            .iter()
            .map(|q| q.values[q.index / folded_domain_len])
            .collect()
    }

    /// The evaluations of the initial (virtual) function “f”.
    fn initial_in_domain_answers(
        queries: &[FoldingPolynomialQuery],
        folding_randomness: XFieldElement,
    ) -> Vec<XFieldElement> {
        queries
            .iter()
            .map(|query| Polynomial::fast_coset_interpolate(query.root, &query.values))
            .map(|poly| poly.evaluate(folding_randomness))
            .collect()
    }

    /// Turn evaluations of (previous) committed function “g” into evaluations
    /// of (current) (virtual) function “f”.
    ///
    /// See also the paper's construction 5.2, bullet point “verifier decision
    /// phase”, subpoint 1 “main loop”, subsubpoints (b) and (c).
    fn subsequent_in_domain_answers(
        quotienting_data: QuotientingData,
        queries: &[FoldingPolynomialQuery],
        folding_randomness: XFieldElement,
    ) -> Vec<XFieldElement> {
        const ONE: XFieldElement = XFieldElement::ONE;

        let QuotientingData {
            quotient_set,
            quotient_answers,
            degree_correction_randomness,
        } = quotienting_data;

        let answer_poly = Polynomial::interpolate(&quotient_set, &quotient_answers);
        let zerofier = Polynomial::zerofier(&quotient_set);

        // This is the paper's `e := d* - d` from section 2.3. It is the
        // difference of the target degree and the degree after quotienting.
        let degree_difference =
            u32::try_from(quotient_set.len() + 1).expect(QUOTIENT_SET_LEN_TO_U32_ERR);

        let mut in_domain_answers = Vec::with_capacity(queries.len());
        let mut coset_evaluations = Vec::new();
        for query in queries {
            coset_evaluations.clear(); // re-use the small allocation
            let mut current_root = query.root;
            for &evaluation in &query.values {
                // quotienting
                let answer_evaluation = answer_poly.evaluate::<_, XFieldElement>(current_root);
                let quotient = (evaluation - answer_evaluation) / zerofier.evaluate(current_root);

                // degree correction
                let common_factor = current_root * degree_correction_randomness;
                let degree_correction_factor = if common_factor == ONE {
                    xfe!(degree_difference)
                } else {
                    (ONE - common_factor.mod_pow_u32(degree_difference)) / (ONE - common_factor)
                };

                coset_evaluations.push(degree_correction_factor * quotient);
                current_root *= query.primitive_kth_root_of_unity;
            }

            let poly = Polynomial::fast_coset_interpolate(query.root, &coset_evaluations);
            in_domain_answers.push(poly.evaluate(folding_randomness));
        }

        in_domain_answers
    }
}

impl Postscript {
    /// The indices of the domain points that were queried in the first round
    /// of [STIR](Stir).
    pub fn first_round_indices(&self) -> &[usize] {
        self.rounds
            .first()
            .map_or(&self.final_round.queried_indices, |r| &r.queried_indices)
    }

    /// Helper function for [RoundPostscript::folded_queried_indices] and
    /// [FinalRoundPostscript::folded_queried_indices].
    fn folded_queried_indices(
        domain: ArithmeticDomain,
        folding_factor: usize,
        queried_indices: &[usize],
    ) -> impl Iterator<Item = usize> {
        let folded_domain_len = domain.pow(folding_factor).unwrap().len();

        queried_indices
            .iter()
            .map(move |&idx| idx % folded_domain_len)
            .unique()
    }
}

impl RoundPostscript {
    /// The _actual_ in-domain queries made in this round. See
    /// [RoundPostscript::queried_indices] for more details.
    pub fn folded_queried_indices(&self) -> impl Iterator<Item = usize> {
        Postscript::folded_queried_indices(self.domain, self.folding_factor, &self.queried_indices)
    }
}

impl FinalRoundPostscript {
    /// See [RoundPostscript::folded_queried_indices].
    pub fn folded_queried_indices(&self) -> impl Iterator<Item = usize> {
        Postscript::folded_queried_indices(self.domain, self.folding_factor, &self.queried_indices)
    }
}

impl NumQueries {
    /// The total number of queries, both in- and out-of-domain.
    fn total(&self) -> usize {
        self.in_domain + self.out_of_domain
    }
}

impl StirMerkleTree {
    /// # Panics
    ///
    /// - if the `stack_height` is not a power of two
    /// - if the codeword's length is not a power of two
    /// - if the codeword's length is smaller than the `stack_height`
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
            queried_leafs,
            auth_structure,
        }
    }
}

impl FoldingPolynomialQueriesInclusionProof {
    fn authenticated_queries(
        self,
        merkle_root: Digest,
    ) -> VerifierResult<Vec<FoldingPolynomialQuery>> {
        // Even though these indices into the folded domain must be unique for
        // the quotienting step, Merkle tree inclusion proofs can handle
        // duplicate entries.
        let folded_queried_indices = self
            .queries
            .iter()
            .map(|q| q.index % self.folded_domain_len);
        let leaf_digests = self
            .queries
            .iter()
            .map(|q| q.values.as_slice())
            .map(XFieldElement::bfe_slice)
            .map(Tip5::hash_varlen);

        let inclusion_proof = MerkleTreeInclusionProof {
            tree_height: self.folded_domain_len.ilog2(),
            indexed_leafs: folded_queried_indices.zip(leaf_digests).collect(),
            authentication_structure: self.auth_structure,
        };

        inclusion_proof
            .verify(merkle_root)
            .then_some(self.queries)
            .ok_or(LdtVerificationError::BadMerkleAuthenticationPath)
    }

    fn queried_indices(&self) -> impl Iterator<Item = usize> + '_ {
        self.queries.iter().map(|q| q.index)
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use assert2::assert;
    use assert2::let_assert;
    use proptest::prelude::*;
    use proptest_arbitrary_adapter::arb;

    use super::*;
    use crate::error::U32_TO_USIZE_ERR;
    use crate::low_degree_test::tests::LdtStats;
    use crate::shared_tests::DigestCorruptor;
    use crate::shared_tests::arbitrary_polynomial_of_degree;
    use crate::tests::proptest;
    use crate::tests::test;

    /// A type alias exclusive to this test module.
    type XfePoly = Polynomial<'static, XFieldElement>;

    impl proptest::arbitrary::Arbitrary for Stir {
        type Parameters = ();

        fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
            any::<StirParameters>()
                .prop_map(|params| params.try_into().unwrap())
                .boxed()
        }

        type Strategy = BoxedStrategy<Self>;
    }

    impl LdtStats for Stir {
        fn num_rounds(&self) -> usize {
            self.round_queries.len()
        }

        fn num_total_queries(&self) -> usize {
            self.round_queries.iter().map(|q| q.total()).sum::<usize>()
                + self.final_num_in_domain_queries
        }

        fn log2_initial_domain_len(&self) -> usize {
            self.initial_domain
                .len()
                .ilog2()
                .try_into()
                .expect(U32_TO_USIZE_ERR)
        }

        fn log2_final_degree_plus_1(&self) -> usize {
            (self.final_degree + 1)
                .ilog2()
                .try_into()
                .expect(U32_TO_USIZE_ERR)
        }
    }

    #[macro_rules_attr::apply(test)]
    fn stacking() {
        let evaluations = xfe_array![0, 1, 2, 3, 4, 5, 6, 7];
        let stacks_4 = StirMerkleTree::stack(&evaluations, 4);
        let expected_4 = vec![xfe_vec![0, 2, 4, 6], xfe_vec![1, 3, 5, 7]];
        assert!(expected_4 == stacks_4);

        let stacks_2 = StirMerkleTree::stack(&evaluations, 2);
        let expected_2 = [[0, 4], [1, 5], [2, 6], [3, 7]]
            .map(|v| v.map(|x| xfe!(x)).to_vec())
            .to_vec();
        assert!(expected_2 == stacks_2);
    }

    #[macro_rules_attr::apply(test)]
    fn log2_binomial_coefficient_is_close_to_precomputed_result() {
        let assert_are_close = |expected: f64, (a, b)| {
            let log2_bin_coeff = log2_binomial_coefficient(a, b);
            let are_close = (expected - log2_bin_coeff).abs() < 1.0e-3;
            assert!(are_close, "{expected} ≠ {log2_bin_coeff}");
        };

        assert_are_close(0.000, (10, 0));
        assert_are_close(3.322, (10, 1));
        assert_are_close(5.492, (10, 2));
        assert_are_close(6.907, (10, 3));
        assert_are_close(7.714, (10, 4));
        assert_are_close(7.977, (10, 5));
        assert_are_close(7.714, (10, 6));
        assert_are_close(6.907, (10, 7));
        assert_are_close(5.492, (10, 8));
        assert_are_close(3.322, (10, 9));
        assert_are_close(0.000, (10, 10));

        assert_are_close(0.00000, (500, 0));
        assert_are_close(230.424, (500, 50));
        assert_are_close(356.476, (500, 100));
        assert_are_close(435.962, (500, 150));
        assert_are_close(480.695, (500, 200));
        assert_are_close(495.191, (500, 250));
        assert_are_close(480.695, (500, 300));
        assert_are_close(435.962, (500, 350));
        assert_are_close(356.476, (500, 400));
        assert_are_close(230.424, (500, 450));
        assert_are_close(0.00000, (500, 500));

        assert_are_close(4446.650, (1 << 13, 0b001 << 10));
        assert_are_close(6639.372, (1 << 13, 0b010 << 10));
        assert_are_close(7811.944, (1 << 13, 0b011 << 10));
        assert_are_close(8185.174, (1 << 13, 0b100 << 10));
        assert_are_close(7811.944, (1 << 13, 0b101 << 10));
        assert_are_close(6639.372, (1 << 13, 0b110 << 10));
        assert_are_close(4446.650, (1 << 13, 0b111 << 10));
    }

    #[macro_rules_attr::apply(proptest(cases = 6))]
    fn error_margin_is_stochastically_correct(
        params: StirParameters,
        #[strategy(arb().no_shrink())] mut tip5: Tip5,
        #[strategy(3..32_u32)] log2_domain_len: u32,
        #[strategy(1..=320.min(1_usize << #log2_domain_len))] num_uniques: usize,
    ) {
        const NUM_TRIES: usize = 100_000;

        tip5.permutation(); // garble
        let num_samples = params.num_total_in_domain_queries(log2_domain_len, num_uniques);

        let mut too_few_uniques_count = 0;
        for _ in 0..NUM_TRIES {
            let samples = tip5.sample_indices(1 << log2_domain_len, num_samples);
            if samples.into_iter().unique().count() < num_uniques {
                too_few_uniques_count += 1;
            }
        }

        let prob_not_enough_uniques = f64::from(too_few_uniques_count) / NUM_TRIES as f64;
        let max_allowed_prob = 2.0_f64.powf(params.security_level as f64).recip();

        prop_assert!(
            prob_not_enough_uniques < max_allowed_prob,
            "required: {max_allowed_prob} actual: {prob_not_enough_uniques}"
        );
    }

    #[macro_rules_attr::apply(proptest)]
    fn roots_of_domain_points(
        #[strategy(0..=3)]
        #[map(|x| 1_usize << x)]
        folding_factor: usize,
        #[filter(#old_domain.len() >= #folding_factor)] old_domain: ArithmeticDomain,
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

    #[macro_rules_attr::apply(test)]
    fn folding_polynomial_gives_expected_coefficients() {
        let folding_factor = 4;
        let folding_randomness = bfe!(10);
        let poly = Polynomial::new(bfe_vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10]);
        //                                  ╰────┬───╯  ╰─┬──────╯  ╰─┬─╯
        //                                       │     ╭──╯  ╭────────╯
        let expected = Polynomial::new(bfe_vec![4321, 8765, 109]);
        let actual = Stir::fold_polynomial(&poly, folding_factor, folding_randomness);

        assert!(expected == actual);
    }

    #[macro_rules_attr::apply(proptest)]
    fn evaluation_of_folded_polynomial_corresponds_to_folding_of_evaluated_polynomial(
        #[strategy(arb())] poly: Polynomial<'static, BFieldElement>,
        #[strategy(1..=5)]
        #[map(|exp| 1_usize << exp)]
        folding_factor: usize,
        #[strategy(arb())] folding_randomness: BFieldElement,
        #[filter(#old_domain.len() >= #folding_factor)] old_domain: ArithmeticDomain,
        #[strategy(0..#old_domain.len() as u32)] evaluation_index: u32,
    ) {
        // Ideally, the proptest would sample an evaluation index in range
        // `0..(old_domain.len() / folding_factor)`. However, the corresponding
        // strategy is instantiated before the filter
        // (old_domain.length >= folding_factor) is applied. This means there
        // are cases (that _would_ be rejected) where the strategy is
        // instantiated with range 0..0, causing an immediate test failure.
        let evaluation_index = evaluation_index % u32::try_from(folding_factor)?;

        let folded_poly = Stir::fold_polynomial(&poly, folding_factor, folding_randomness);
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
                .evaluate::<_, BFieldElement>(folding_randomness);

        prop_assert_eq!(evaluation_of_folded_poly, folding_of_evaluated_poly);
    }

    #[macro_rules_attr::apply(proptest)]
    fn invalid_initial_codeword_is_rejected(
        stir: Stir,
        mut proof_stream: ProofStream,
        #[strategy(Just(#stir.initial_domain.len() as isize))] domain_length: isize,
        #[strategy(-#domain_length..=#domain_length)]
        #[filter(#length_difference != &0)]
        length_difference: isize,
    ) {
        let wrong_length = (domain_length + length_difference) as usize;
        let codeword = xfe_vec![1; wrong_length];

        let_assert!(Err(err) = stir.prove(&codeword, &mut proof_stream));
        let_assert!(
            LdtProvingError::InitialCodewordMismatch {
                domain_len,
                codeword_len
            } = err
        );
        prop_assert_eq!(domain_length as usize, domain_len);
        prop_assert_eq!(wrong_length, codeword_len);
    }

    #[macro_rules_attr::apply(proptest)]
    fn invalid_parameter_initial_expansion_factor_1_is_rejected(mut params: StirParameters) {
        params.log2_initial_expansion_factor = 0;
        let_assert!(Err(err) = Stir::try_from(params));
        prop_assert_eq!(LdtParameterError::TooSmallInitialExpansionFactor, err);
    }

    #[macro_rules_attr::apply(proptest)]
    fn invalid_parameter_big_initial_expansion_factor_is_rejected(
        mut params: StirParameters,
        #[strategy(32_usize..)] bad_log2_initial_expansion_factor: usize,
    ) {
        params.log2_initial_expansion_factor = bad_log2_initial_expansion_factor;
        let_assert!(Err(err) = Stir::try_from(params));
        let_assert!(LdtParameterError::InitialDomainTooBig(_) = err);
    }

    #[macro_rules_attr::apply(proptest)]
    fn invalid_parameter_small_folding_factor_is_rejected(
        mut params: StirParameters,
        #[strategy(0_usize..=1)] bad_log2_folding_factor: usize,
    ) {
        params.log2_folding_factor = bad_log2_folding_factor;
        let_assert!(Err(err) = Stir::try_from(params));
        let_assert!(LdtParameterError::TooSmallLog2FoldingFactor(f) = err);
        prop_assert_eq!(bad_log2_folding_factor, f);
    }

    #[macro_rules_attr::apply(proptest)]
    fn invalid_parameter_big_folding_factor_is_rejected(
        mut params: StirParameters,
        #[strategy(64_usize..)] bad_log2_folding_factor: usize,
    ) {
        params.log2_folding_factor = bad_log2_folding_factor;
        params.log2_high_degree_bound = bad_log2_folding_factor;
        let_assert!(Err(err) = Stir::try_from(params));
        let_assert!(LdtParameterError::TooBigLog2FoldingFactor(f) = err);
        prop_assert_eq!(bad_log2_folding_factor, f);
    }

    /// The proptest [`invalid_parameter_big_folding_factor_is_rejected`] does
    /// not cover all failure paths reliably.
    #[macro_rules_attr::apply(test)]
    fn concrete_invalid_big_parameter_folding_factor_are_rejected() {
        fn assert_too_big_folding_factor_is_rejected(factor: usize) {
            let params = StirParameters {
                security_level: 42,
                soundness: ProximityRegime::default(),
                log2_folding_factor: factor,
                log2_initial_expansion_factor: 1,
                log2_high_degree_bound: factor,
            };
            let err = Stir::try_from(params).unwrap_err();

            assert!(LdtParameterError::TooBigLog2FoldingFactor(factor) == err);
        }

        assert_too_big_folding_factor_is_rejected(u32::MAX as usize);
        assert_too_big_folding_factor_is_rejected(usize::MAX);
    }

    #[macro_rules_attr::apply(proptest)]
    fn invalid_parameter_small_high_degree_bound_is_rejected(
        mut params: StirParameters,
        #[strategy(..#params.log2_folding_factor)] bad_log2_high_degree_bound: usize,
    ) {
        params.log2_high_degree_bound = bad_log2_high_degree_bound;
        let_assert!(Err(err) = Stir::try_from(params));
        prop_assert_eq!(LdtParameterError::TooLowDegreeOfHighDegreePolynomials, err);
    }

    #[macro_rules_attr::apply(proptest)]
    fn too_big_initial_domain_doesnt_cause_crash(
        mut params: StirParameters,
        #[strategy(33 - #params.log2_initial_expansion_factor..=64)] log2_high_degree_bound: usize,
    ) {
        params.log2_high_degree_bound = log2_high_degree_bound;
        let_assert!(Err(err) = params.initial_domain());
        let_assert!(LdtParameterError::InitialDomainTooBig(_) = err);
    }

    /// The proptest [`too_big_initial_domain_doesnt_cause_crash`] does not
    /// cover all failure paths reliably.
    //
    // Don't `#[macro_rules_attr::apply(test)]`:
    // 32-bit architectures cannot trigger this failure path since `u32` and
    // `usize` have equal size.
    #[test]
    fn concrete_too_big_initial_domain_doesnt_cause_crash() {
        let two_thirds_u64_max = (u64::MAX as usize / 3) * 2;
        let params = StirParameters {
            security_level: 42,
            soundness: ProximityRegime::default(),
            log2_folding_factor: two_thirds_u64_max,
            log2_initial_expansion_factor: two_thirds_u64_max,
            log2_high_degree_bound: two_thirds_u64_max,
        };
        let_assert!(Err(err) = params.initial_domain());
        assert!(LdtParameterError::InitialDomainTooBig(u64::MAX) == err);
    }

    #[macro_rules_attr::apply(proptest)]
    fn try_from_and_new_correspond(params: StirParameters) {
        prop_assert_eq!(Stir::try_from(params), Stir::new(params));
    }

    #[macro_rules_attr::apply(proptest)]
    fn prove_and_verify_zero_polynomial(stir: Stir) {
        let zero_poly = xfe_vec![0; stir.initial_domain.len()];

        let mut proof_stream = ProofStream::new();
        stir.prove(&zero_poly, &mut proof_stream)?;

        proof_stream.reset_sponge();
        stir.verify(&mut proof_stream)?;
    }

    // It's quite difficult to meaningfully check that the folded query indices
    // are _correct_. (In particular, re-implementing the method is not
    // meaningful.) Hence, this is only a sanity check.
    #[macro_rules_attr::apply(test)]
    fn folded_query_indices_look_sane() {
        let params = StirParameters {
            security_level: 42,
            soundness: ProximityRegime::default(),
            log2_folding_factor: 2,
            log2_initial_expansion_factor: 2,
            log2_high_degree_bound: 11,
        };
        let stir = Stir::new(params).unwrap();
        let zero_poly = xfe_vec![0; stir.initial_domain().len()];
        let mut proof_stream = ProofStream::new();
        stir.prove(&zero_poly, &mut proof_stream).unwrap();
        proof_stream.reset_sponge();
        let_assert!(VerifierPostscript::Stir(postscript) = stir.verify(&mut proof_stream).unwrap());

        // the sanity check
        assert!(postscript.rounds.len() > 0);
        let domain_shrinkage = 1 << StirParameters::LOG2_DOMAIN_SHRINKAGE;
        let mut domain = stir.initial_domain();
        for round in &postscript.rounds {
            domain = domain.pow(domain_shrinkage).unwrap();
            for i in round.folded_queried_indices() {
                assert!(i < domain.len());
            }
        }
        domain = domain.pow(domain_shrinkage).unwrap();
        for i in postscript.final_round.folded_queried_indices() {
            assert!(i < domain.len());
        }
    }

    #[macro_rules_attr::apply(proptest)]
    fn prove_and_verify_low_degree_polynomial(
        params: StirParameters,
        #[strategy(-1..=#params.max_degree() as i64)] _d: i64,
        #[strategy(arbitrary_polynomial_of_degree(#_d).no_shrink())] poly: XfePoly,
    ) {
        let stir = Stir::try_from(params)?;
        let codeword = stir.initial_domain.evaluate(&poly);

        let mut proof_stream = ProofStream::new();
        let prover_indices = stir.prove(&codeword, &mut proof_stream)?;
        let prover_sponge = proof_stream.sponge.clone();

        proof_stream.reset_sponge();
        let_assert!(VerifierPostscript::Stir(postscript) = stir.verify(&mut proof_stream)?);
        let verifier_sponge = proof_stream.sponge;

        prop_assert_eq!(prover_sponge, verifier_sponge);
        prop_assert_eq!(proof_stream.items.len(), proof_stream.items_index);
        prop_assert_eq!(&prover_indices, postscript.first_round_indices());

        for (idx, value) in prover_indices
            .into_iter()
            .zip_eq(postscript.partial_first_codeword)
        {
            prop_assert_eq!(codeword[idx], value);
        }
    }

    #[macro_rules_attr::apply(proptest)]
    fn prove_and_fail_to_verify_high_degree_polynomial(
        params: StirParameters,
        #[strategy(Just(1 << #params.log2_high_degree_bound))] _too_high_degree: i64,
        #[strategy(#_too_high_degree..2 * #_too_high_degree)] _d: i64,
        #[strategy(arbitrary_polynomial_of_degree(#_d).no_shrink())] poly: XfePoly,
    ) {
        let stir = Stir::try_from(params)?;
        let codeword = stir.initial_domain.evaluate(&poly);
        let mut proof_stream = ProofStream::new();
        stir.prove(&codeword, &mut proof_stream)?;

        proof_stream.reset_sponge();
        let verdict = stir.verify(&mut proof_stream);
        prop_assert!(verdict.is_err());
    }

    #[macro_rules_attr::apply(proptest(cases = 100))]
    fn proof_stream_serialization(
        params: StirParameters,
        #[strategy(-1..=#params.max_degree() as i64)] _d: i64,
        #[strategy(arbitrary_polynomial_of_degree(#_d))] poly: XfePoly,
    ) {
        let stir = Stir::try_from(params)?;
        let codeword = stir.initial_domain.evaluate(&poly);
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
                (Polynomial(p), Polynomial(v)) => prop_assert_eq!(p, v),
                _ => panic!("Unknown items.\nProver: {prover_item:?}\nVerifier: {verifier_item:?}"),
            }
        }
    }

    #[macro_rules_attr::apply(proptest)]
    fn verifying_arbitrary_proof_does_not_panic(stir: Stir, mut proof_stream: ProofStream) {
        let _verdict = stir.verify(&mut proof_stream);
    }

    #[macro_rules_attr::apply(proptest(cases = 100))]
    fn modified_proof_stream_results_in_verification_failure(
        params: StirParameters,
        #[strategy(-1..=#params.max_degree() as i64)] _d: i64,
        #[strategy(arbitrary_polynomial_of_degree(#_d))] poly: XfePoly,
        item_index: usize,
        vec_index: usize,
        digest_corruptor: DigestCorruptor,
        #[strategy(arb())] random_xfe: XFieldElement,
        corrupt_auth_structure: bool,
    ) {
        let stir = Stir::try_from(params)?;
        let codeword = stir.initial_domain.evaluate(&poly);
        let mut proof_stream = ProofStream::new();
        stir.prove(&codeword, &mut proof_stream)?;
        proof_stream.reset_sponge();

        let proof_item = select_element(&mut proof_stream.items, item_index)?;
        match proof_item {
            ProofItem::MerkleRoot(root) => digest_corruptor.corrupt(root)?,
            ProofItem::StirOutOfDomainValues(xfes) => corrupt_slice(xfes, vec_index, random_xfe)?,
            ProofItem::StirResponse(stir_response) => {
                if corrupt_auth_structure {
                    let digest = select_element(&mut stir_response.auth_structure, vec_index)?;
                    digest_corruptor.corrupt(digest)?;
                } else {
                    let num_lists = stir_response.queried_leafs.len();
                    let list = select_element(&mut stir_response.queried_leafs, vec_index)?;
                    corrupt_slice(list, vec_index / num_lists, random_xfe)?;
                }
            }
            ProofItem::Polynomial(poly) => {
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
