use twenty_first::prelude::XFieldElement;

use crate::arithmetic_domain::ArithmeticDomain;
use crate::error::LdtProvingError;
use crate::error::LdtVerificationError;
use crate::proof_stream::ProofStream;

pub mod fri;
pub mod stir;

mod private {
    /// A public but un-nameable type for sealing traits.
    pub trait Seal {}
}

/// A trait to describe proof systems that establish the low-degreeness of
/// some polynomial.
///
/// This is a sealed trait; it cannot be implemented outside of this crate.
/// For more details on the concrete low-degree tests, see [`stir`] and [`fri`].
pub trait LowDegreeTest: private::Seal {
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
    ) -> Result<Vec<usize>, LdtProvingError>;

    /// Verify that the polynomial committed to via the proof stream is of low
    /// degree.
    ///
    /// Returns a [transcript](VerificationTranscript), from which the partial
    /// codeword of the first round as well as their corresponding indices can
    /// be fetched. These revealed elements are relevant when using STIR in a
    /// bigger context.
    fn verify(
        &self,
        proof_stream: &mut ProofStream,
    ) -> Result<VerificationTranscript, LdtVerificationError>;
}

/// The type of soundness assumption (or lack thereof) you are willing to make
/// for the [low-degree test](LowDegreeTest).
///
/// The choice influences the derivation of additional parameters, like the
/// number of queries per round. Generally, the more “daring” the assumption,
/// the lower the runtime cost and proof size, but the higher the risk that the
/// resulting system is unsound due to as-of-yet undiscovered attacks.
///
/// The [`Provable`](Self::Provable) variant is generally recommended.
#[derive(Debug, Default, Copy, Clone, Eq, PartialEq)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
pub enum SoundnessType {
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

/// A transcript of the [verification](LowDegreeTest::verify) of a
/// [low-degree test](LowDegreeTest).
///
/// This transcript serves one main and one secondary purpose.
/// 1. Primarily, the partially revealed first codeword in combination with
///    the indices queried in the first round help link the polynomial of
///    which the low-degree test proves the low degree into a greater
///    context, like a STARK.
/// 2. Secondarily, the sampled randomness and used authentication structures
///    help operate, develop, and test other implementations of this
///    low-degree test that try to mimic this implementation exactly.
#[derive(Debug, Clone, Eq, PartialEq)]
pub enum VerificationTranscript {
    Stir(stir::Transcript),
    Fri,
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;

    #[test]
    fn low_degree_test_trait_is_dyn_compatible() {
        fn _foo() -> Option<Box<dyn LowDegreeTest>> {
            None
        }
    }
}
