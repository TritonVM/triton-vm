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
    /// list-decoding capacity bound, (1 - H_q(ρ)), where H_q is the q-ary
    /// entropy function and q is the number of elements in the
    /// [extension field](XFieldElement), _i.e._,
    /// [`BFieldElement::P`](crate::prelude::BFieldElement::P) to the power
    /// [3](twenty_first::math::x_field_element::EXTENSION_DEGREE).
    ///
    /// This soundness assumption takes into account the results of 2025.
    /// See, for example,
    /// “[On Reed–Solomon Proximity Gaps Conjectures][crites_stewart]”,
    /// or [this blog post][blog] for a more high-level overview of those
    /// results.
    ///
    /// [crites_stewart]: https://eprint.iacr.org/2025/2046.pdf
    /// [blog]: https://blog.zksecurity.xyz/posts/proximity-conjecture/
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
    /// The base-2 logarithm of the size of the
    /// [extension field](XFieldElement).
    const LOG2_FIELD_SIZE: f64 = 191.99999999899228;

    #[must_use]
    fn new(log2_expansion_factor: usize) -> Self {
        ReedSolomonCode {
            soundness: ProximityRegime::default(),
            log2_expansion_factor,
        }
    }

    #[must_use]
    fn with_soundness(mut self, soundness: ProximityRegime) -> Self {
        self.soundness = soundness;
        self
    }

    /// The proximity parameter of this Reed-Solomon code.
    ///
    /// This is the δ in a (δ, ℓ)-list-decodable Reed-Solomon code as well as
    /// the δ in the (δ, ε)-proximity gap. It is the threshold for which words
    /// are considered “far” from the code.
    ///
    /// Note that there are a range of valid values for δ. This method makes a
    /// concrete choice; see also [`Self::slackness_factor`].
    fn proximity_parameter(&self) -> SetupResult<f64> {
        Ok(1.0 - self.proximity_margin()? - self.slackness_factor()?)
    }

    /// The distance between the
    /// [proximity parameter](Self::proximity_parameter) δ and the
    /// [proximity bound](Self::soundness).
    ///
    /// This parameter is often called η in the context of Reed-Solomon codes.
    /// In this implementation, it is not a parameter; the choice has been
    /// fixed.
    ///
    /// No matter the proximity regime, the upper bound on the
    /// [list size](Self::log2_list_size) only holds for proximity parameters δ
    /// that are slightly below the bound. This function defines just how big
    /// the distance between the proximity parameter and the bound is.
    //
    // Giacomo thinks this can be set in a better way. Maybe it makes more sense
    // if it's multiplicative? In either case, it's based on some heuristic by
    // Giacomo that Ferdinand doesn't fully understand the reasons for. It might
    // be interesting to explore different things at some point.
    fn slackness_factor(&self) -> SetupResult<f64> {
        // this is the heuristic
        Ok(self.proximity_margin()? / 20.0)
    }

    /// The margin (distance from 1) of the
    /// [proximity parameter](Self::proximity_parameter)'s upper bound.
    fn proximity_margin(&self) -> SetupResult<f64> {
        let margin = match self.soundness {
            ProximityRegime::Proven => self.rate()?.sqrt(),
            ProximityRegime::Conjectured => self.q_ary_entropy()?,
        };

        Ok(margin)
    }

    /// The q-ary entropy function evaluated at the rate, ρ (see also
    /// [`Self::log2_expansion_factor`]).
    ///
    /// “Parameter” q is hardcoded to the size of the extension field; see also
    /// [ProximityRegime::Conjectured].
    //
    // The hardcoded nature of `q` allows some approximations to gracefully
    // work around potential loss of precision arising from floating point
    // operations.
    //
    // Start with the definition of h_q(ρ):
    //
    //     h_q(ρ) = ρ·log_q(q-1) - ρ·log_q(ρ) - (1-ρ)·log_q(1-ρ).
    //
    // As a first simplification, approximate log_q(q-1) with 1. (The error in
    // this approximation is about 1e-60.)
    // Next, move to base 2 as much as possible:
    //
    //     h_q(ρ) = ρ -  ρ·log₂(ρ)/log₂(q) - (1-ρ)·log₂(1-ρ)/log₂(q)
    //            = ρ - (ρ·log₂(ρ)         + (1-ρ)·log₂(1-ρ)) / log₂(q)
    //
    // Finally, observe that log₂(ρ) = -log₂(expansion_factor).
    fn q_ary_entropy(&self) -> SetupResult<f64> {
        let rate = self.rate()?;
        let rate_log_rate = rate * -(self.log2_expansion_factor as f64);
        let one_m_rate_log_one_m_rate = (1.0 - rate) * (1.0 - rate).log2();

        Ok(rate - (rate_log_rate + one_m_rate_log_one_m_rate) / Self::LOG2_FIELD_SIZE)
    }

    /// The code's rate, ρ.
    fn rate(&self) -> SetupResult<f64> {
        let err = LdtParameterError::TooBigInitialExpansionFactor;
        let log2_expansion_factor = u32::try_from(self.log2_expansion_factor).map_err(|_| err)?;
        let expansion_factor = 1_u32.checked_shl(log2_expansion_factor).ok_or(err)?;
        let rate = 1.0 / f64::from(expansion_factor);

        Ok(rate)
    }

    /// The (log₂ of the) list size of the Reed-Solomon code.
    ///
    /// This is the ℓ in a (δ, ℓ)-list-decodable Reed-Solomon code.
    ///
    /// The Reed-Solomon code in question is defined by the given polynomial
    /// degree and the given rate. The [proximity
    /// parameter](Self::proximity_parameter) δ is implied by the [slackness
    /// factor](Self::slackness_factor) η.
    ///
    /// For the [Johnson bound](ProximityRegime::Proven), it is shown that
    /// the Reed-Solomon codes are (1-√ρ-η, 1/(2·√ρ·η))-list decodable, _i.e._,
    /// ℓ = 1/(2·√ρ·η).
    ///
    /// For the [conjectured bound](ProximityRegime::Conjectured), it is
    /// assumed that Reed-Solomon codes are (1-H_q(ρ)-η, d/(H_q(ρ)·η))-list
    /// decodable, _i.e._, ℓ = d/(H_q(ρ)·η), where H_q(ρ) is the
    /// [q-ary entropy function](Self::q_ary_entropy).
    /// Beware that there's a _lot_ of uncertainty in this bound.
    /// See also Conjecture 5.6 in the STIR paper (which uses out-of-date
    /// assumptions but is informative regardless).
    fn log2_list_size(&self, log2_poly_degree: u32) -> SetupResult<f64> {
        let list_size = match self.soundness {
            ProximityRegime::Proven => {
                1.0 / (2.0 * self.rate()?.sqrt() * self.slackness_factor()?)
            }
            ProximityRegime::Conjectured => {
                1_f64.powf(f64::from(log2_poly_degree))
                    / (self.q_ary_entropy()? * self.slackness_factor()?)
            }
        };

        Ok(list_size.log2())
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
pub(crate) mod tests {
    use assert2::let_assert;
    use test_strategy::proptest;
    use twenty_first::prelude::BFieldCodec;
    use twenty_first::prelude::BFieldElement;
    use twenty_first::prelude::x_field_element::EXTENSION_DEGREE;
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

    #[test]
    fn log2_extension_field_size_is_correct() {
        let log2_field_size = (BFieldElement::P as f64)
            .powf(EXTENSION_DEGREE as f64)
            .log2();
        assert!((ReedSolomonCode::LOG2_FIELD_SIZE - log2_field_size).abs() < 0.0001);
    }

    #[test]
    fn q_ary_entropy_fn_is_correct() {
        const DELTA: f64 = 0.0001;

        let assert_are_close = |l: f64, r: f64| assert!((l - r).abs() < DELTA, "{l} != {r}");
        let entropy_of_log2_expansion_factor =
            |l2_exp_factor| ReedSolomonCode::new(l2_exp_factor).q_ary_entropy().unwrap();

        // references computed with sage
        assert_are_close(0.505208333333361, entropy_of_log2_expansion_factor(1));
        assert_are_close(0.254225406898247, entropy_of_log2_expansion_factor(2));
        assert_are_close(0.127831064808346, entropy_of_log2_expansion_factor(3));
        assert_are_close(0.064256719096972, entropy_of_log2_expansion_factor(4));
        assert_are_close(0.032294907939134, entropy_of_log2_expansion_factor(5));
        assert_are_close(0.016229766017215, entropy_of_log2_expansion_factor(6));
        assert_are_close(0.008155804230956, entropy_of_log2_expansion_factor(7));
        assert_are_close(0.004098304720073, entropy_of_log2_expansion_factor(8));
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
