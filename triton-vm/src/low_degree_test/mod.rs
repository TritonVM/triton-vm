//! Low-degree tests (“LDTs”) for polynomials over finite fields.
//!
//! For more information, see the central trait, [`LowDegreeTest`].

use std::fmt::Debug;

use arbitrary::Arbitrary;
use serde::Deserialize;
use serde::Serialize;
use strum::EnumIter;
use twenty_first::prelude::XFieldElement;

use crate::arithmetic_domain::ArithmeticDomain;
use crate::error::LdtParameterError;
use crate::error::LdtProvingError;
use crate::error::LdtVerificationError;
use crate::proof_stream::ProofStream;

pub mod fri;
pub mod stir;

type SetupResult<T> = Result<T, LdtParameterError>;
type ProverResult<T> = Result<T, LdtProvingError>;
type VerifierResult<T> = Result<T, LdtVerificationError>;

mod private {
    /// A public but un-nameable type for sealing traits.
    pub trait Seal {}
}

/// A trait to describe proof systems that establish the low-degreeness of
/// some polynomial.
///
/// The two low-degree tests (“LDTs”) implemented in this codebase are
/// [FRI](fri::Fri) and [STIR](stir::Stir). Both are Interactive Oracle Proof of
/// Proximity for Reed-Solomon codes.
/// When combined with the [BCS] transform and the [Fiat-Shamir] heuristic (as
/// is done here), the result is a non-interactive system for proving that the
/// initial input codeword corresponds to a polynomial of low degree.
///
/// [BCS]: https://eprint.iacr.org/2016/116
/// [Fiat-Shamir]: https://link.springer.com/chapter/10.1007/3-540-47721-7_12
///
/// The central methods are [prove](Self::prove) and [verify](Self::verify).
///
/// This is a sealed trait; it cannot be implemented outside of this crate.
pub trait LowDegreeTest: private::Seal + Debug {
    /// The domain for the initial codeword of the [prover](Self::prove).
    fn initial_domain(&self) -> ArithmeticDomain;

    /// The number of in-domain queries made in the first round.
    ///
    /// This value is particularly relevant when using the low-degree test as a
    /// sub-protocol in a Zero-Knowledge protocol: the number of “masking” or
    /// “randomizing” values must be at least as big as this number to uphold
    /// the Zero-Knowledge guarantees.
    fn num_first_round_queries(&self) -> usize;

    /// Prove that the given codeword corresponds to a polynomial of low degree
    /// and populate the proof stream with the generated proof artifacts.
    ///
    /// Returns the indices the verifier queried in the first round. If the
    /// low-degree test is used in a larger context (like a STARK), then these
    /// indices can be used to prove that the codeword of low degree actually
    /// corresponds to some codeword of interest, not just any codeword.
    fn prove(
        &self,
        codeword: &[XFieldElement],
        proof_stream: &mut ProofStream,
    ) -> ProverResult<Vec<usize>>;

    /// Verify that the polynomial committed to via the proof stream is of low
    /// degree.
    ///
    /// Returns a [postscript](VerifierPostscript), from which the partial
    /// codeword of the first round as well as their corresponding indices can
    /// be fetched. These revealed elements are relevant when using the
    /// low-degree test in a bigger context.
    fn verify(&self, proof_stream: &mut ProofStream) -> VerifierResult<VerifierPostscript>;

    /// Help inspecting which concrete LDT a trait object is.
    ///
    /// This is a test-only helper method.
    ///
    /// # Example
    ///
    /// ```
    /// # use triton_vm::low_degree_test::LowDegreeTest;  
    /// # use triton_vm::low_degree_test::fri::Fri;
    /// # use triton_vm::low_degree_test::stir::Stir;
    /// #
    /// # fn takes_ldt(ldt: Box<dyn LowDegreeTest>) {
    /// if ldt.as_any().is::<Fri>() { /* … */ }
    /// if let Some(stir) = ldt.as_any().downcast_ref::<Stir>() { /* … */ }
    /// # }
    /// ```
    #[cfg(test)]
    fn as_any(&self) -> &dyn std::any::Any;
}

/// The assumption (or lack thereof) you are willing to make about the proximity
/// gap of Reed-Solomon codes used in the [low-degree test](LowDegreeTest).
///
/// This choice might influence the soundness of the low-degree test. In
/// particular, it influences the number of queries per round. Generally, the
/// more “daring” the assumption, the lower the runtime cost and proof size, but
/// the higher the risk that the resulting system is unsound due to as-of-yet
/// undiscovered attacks.
///
/// The [`Proven`](Self::Proven) variant is generally recommended.
#[derive(
    Debug, Default, Copy, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, EnumIter, Arbitrary,
)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
pub enum ProximityRegime {
    /// Only use proven results.
    ///
    /// In particular, assume that the distance of each oracle is within the
    /// Johnson bound, (1 - √ρ).
    #[default]
    Proven,

    /// Use the conjecture that Reed-Solomon codes are list-decodable up to
    /// list-decoding capacity and have correlated agreement up to list-decoding
    /// capacity.
    ///
    /// In particular, assume that the distance of each oracle is within the
    /// capacity bound, (1 - ρ).
    Conjectured,
}

/// A “P.S.” of the [verification](LowDegreeTest::verify) of a
/// [low-degree test](LowDegreeTest).
///
/// Includes the following data used or generated by the verifier:
/// - the [first (partially revealed) codeword][first_codeword] and the
///   [corresponding indices][first_queries]
/// - sampled randomness (any [indices](ProofStream::sample_indices) and
///   [scalars](ProofStream::sample_scalars))
/// - [authentication structures][auth_struct]
///
/// The postscript serves one main and one secondary purpose.
/// 1. Primarily, the partially revealed first codeword in combination with
///    the indices queried in the first round help link the polynomial of
///    which the low-degree test proves the low degree into a greater
///    context, like a STARK.
/// 2. Secondarily, the sampled randomness and used authentication structures
///    help operate, develop, and test other implementations of this
///    low-degree test that try to mimic this implementation exactly.
///
/// Note that the verifier uses and computes additional intermediate artifacts
/// as part of the verification process. If those serve neither the primary nor
/// the secondary purpose of the postscript, they are not included.
///
/// ### Why “Postscript”?
///
/// Much like the “post scriptum” (“P.S.”) in a letter, the postscript
/// 1. comes after the main content, _i.e._, after the verifier's verdict, and
/// 2. contains additional information recorded (“written down”) by the
///    verifier.
///
/// [first_codeword]: Self::partial_first_codeword
/// [first_queries]: Self::first_round_indices
/// [auth_struct]: twenty_first::prelude::MerkleTreeInclusionProof::authentication_structure
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum VerifierPostscript {
    Stir(stir::Postscript),
    Fri(fri::Postscript),
}

impl VerifierPostscript {
    pub fn partial_first_codeword(&self) -> &[XFieldElement] {
        match self {
            Self::Stir(t) => &t.partial_first_codeword,
            Self::Fri(t) => &t.initial_round.partial_codeword,
        }
    }

    pub fn first_round_indices(&self) -> &[usize] {
        match self {
            Self::Stir(t) => t.first_round_indices(),
            Self::Fri(t) => &t.initial_round.indices,
        }
    }
}

/// Helper struct to derive relevant quantities for Reed-Solomon codes.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
struct ReedSolomonCode {
    /// Informs over the bound to use when deriving additional parameters.
    soundness: ProximityRegime,

    /// The (log₂ of the) reciprocal of the rate ρ.
    log2_expansion_factor: usize,
}

impl ReedSolomonCode {
    /// The proximity parameter of this Reed-Solomon code.
    ///
    /// This is the δ in a (δ, ℓ)-list-decodable Reed-Solomon code as well as
    /// the δ in the (δ, ε)-proximity gap. It is the threshold for which words
    /// are considered “far” from the code.
    ///
    /// Note that there are a range of valid values for δ. This method makes a
    /// concrete choice; see also [`Self::log2_slackness_factor`].
    fn proximity_parameter(&self) -> SetupResult<f64> {
        let log2_slackness_factor = self.log2_slackness_factor();
        let slackness_factor = 2_f64.powf(log2_slackness_factor);

        let Ok(log2_expansion_factor) = u32::try_from(self.log2_expansion_factor) else {
            return Err(LdtParameterError::TooBigInitialExpansionFactor);
        };
        let Some(expansion_factor) = 1_u32.checked_shl(log2_expansion_factor) else {
            return Err(LdtParameterError::TooBigInitialExpansionFactor);
        };
        let rate = 1.0 / f64::from(expansion_factor);
        let proximity_parameter = match self.soundness {
            ProximityRegime::Proven => 1.0 - rate.sqrt() - slackness_factor,
            ProximityRegime::Conjectured => 1.0 - rate - slackness_factor,
        };

        Ok(proximity_parameter)
    }

    /// The (log₂ of the) distance between the
    /// [proximity parameter](Self::proximity_parameter) δ and the
    /// [chosen bound](Self::soundness).
    ///
    /// This parameter is often called η in the context of Reed-Solomon codes.
    /// In this implementation, it is not a parameter; the choice has been
    /// fixed.
    ///
    /// No matter the proximity regime, the upper bound on the
    /// [list size](Self::log2_list_size) only holds for proximity parameters δ
    /// that are slightly below the bound. This function defines just how big
    /// the distance between the proximity parameter and the bound is.
    ///
    /// For example, if the chosen bound is the
    /// [Johnson bound](ProximityRegime::Proven), then δ ∈ (0, 1 - √ρ),
    /// and we chose δ = 1 - √ρ - η.
    /// For the [conjectured bound](ProximityRegime::Conjectured),
    /// δ ∈ (0, 1 - ρ), and we chose δ = 1 - ρ - η. In either case, η is defined
    /// in this method.
    //
    // Giacomo thinks this can be set in a better way. Maybe it makes more sense
    // if it's multiplicative? In either case, it's based on some heuristic by
    // Giacomo that Ferdinand doesn't fully understand. It might be interesting
    // to explore different things at some point.
    //
    // According to Giacomo: “Funnily enough, [η = ρ/10 or η = √ρ/10] avoids
    // all the new attacks on proximity gaps 😉”. Presumably, division by 20
    // also works.
    //
    // A high-level overview of the new attacks mentioned above:
    // https://blog.zksecurity.xyz/posts/proximity-conjecture/
    fn log2_slackness_factor(&self) -> f64 {
        let log2_expansion_factor = self.log2_expansion_factor as f64;
        let log2_rho_or_sqrt_rho = match self.soundness {
            ProximityRegime::Proven => 0.5 * -log2_expansion_factor,
            ProximityRegime::Conjectured => -log2_expansion_factor,
        };

        // This is where the heuristic kicks in:
        // log₂(ρ/20) or log₂(√ρ/20)
        log2_rho_or_sqrt_rho - core::f64::consts::LOG2_10 - 1.0
    }

    /// The list size of the Reed-Solomon code.
    ///
    /// This is the ℓ in a (δ, ℓ)-list-decodable Reed-Solomon code.
    ///
    /// The Reed-Solomon code in question is defined by the given polynomial
    /// degree and the given rate. The [proximity
    /// parameter](Self::proximity_parameter) δ is implied by the [slackness
    /// factor](Self::log2_slackness_factor) η.
    ///
    /// For the [Johnson bound](ProximityRegime::Proven), it is shown that
    /// the Reed-Solomon codes are (1-√ρ-η, 1/(2·√ρ·η))-list decodable. In
    /// other words, the list size for δ = 1-√ρ-η is
    /// ℓ = √(expansion_factor)/(2·η).
    ///
    /// For the [conjectured bound](ProximityRegime::Conjectured), it is
    /// assumed that Reed-Solomon codes are (1-ρ-η, d/(ρ·η))-list decodable. In
    /// other words, the list size for δ = 1-ρ-η is
    /// ℓ = degree·expansion_factor/η.
    /// See also Conjecture 5.6 in the STIR paper.
    fn log2_list_size(&self, log2_poly_degree: usize) -> f64 {
        let log2_slackness_factor = self.log2_slackness_factor();

        match self.soundness {
            ProximityRegime::Proven => {
                self.log2_expansion_factor as f64 / 2. - 1. - log2_slackness_factor
            }
            ProximityRegime::Conjectured => {
                (log2_poly_degree + self.log2_expansion_factor) as f64 - log2_slackness_factor
            }
        }
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
pub(crate) mod tests {
    use assert2::let_assert;
    use test_strategy::proptest;
    use twenty_first::prelude::BFieldCodec;
    use twenty_first::prelude::BFieldElement;
    use twenty_first::xfe_vec;

    use super::*;
    use crate::prelude::Proof;

    /// Test-only trait to gather statistics of low-degree tests.
    ///
    /// The statistics in question help inform about the choice of the concrete
    /// low-degree test to use.
    pub(crate) trait LdtStats: LowDegreeTest {
        fn num_rounds(&self) -> usize;
        fn num_total_queries(&self) -> usize;
        fn log2_initial_domain_len(&self) -> usize;
        fn log2_final_degree_plus_1(&self) -> usize;

        /// In KiB.
        fn proof_size(&self) -> usize {
            let codeword = xfe_vec![1; self.initial_domain().len()];
            let mut proof_stream = ProofStream::new();
            self.prove(&codeword, &mut proof_stream).unwrap();
            let proof_len_in_bfes = Proof::from(proof_stream).encode().len();
            let proof_size_bytes = proof_len_in_bfes * BFieldElement::BYTES;

            proof_size_bytes.div_ceil(1024)
        }
    }

    #[proptest]
    fn too_big_expansion_factor_is_rejected(mut code: ReedSolomonCode) {
        type Err = LdtParameterError;

        code.log2_expansion_factor = usize::MAX;
        let_assert!(Err(Err::TooBigInitialExpansionFactor) = code.proximity_parameter());

        code.log2_expansion_factor = 32;
        let_assert!(Err(Err::TooBigInitialExpansionFactor) = code.proximity_parameter());
    }
}
