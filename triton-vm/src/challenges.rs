//! Challenges are needed for the [cross-table arguments](CrossTableArg),
//! _i.e._, [Permutation Arguments](air::cross_table_argument::PermArg),
//! [Evaluation Arguments](EvalArg), and
//! [Lookup Arguments](air::cross_table_argument::LookupArg),
//! as well as for the RAM Table's Contiguity Argument.
//!
//! There are three types of challenges:
//! - **Weights**. Weights are used to linearly combine multiple elements into
//!   one element. The resulting single element can then be used in a
//!   cross-table argument.
//! - **Indeterminates**. All cross-table arguments work by checking the
//!   equality of polynomials (or rational functions). Through the
//!   Schwartz-Zippel lemma, this equality check can be performed by evaluating
//!   the polynomials (or rational functions) in a single point. The challenges
//!   that are indeterminates are exactly this evaluation point. The polynomials
//!   (or rational functions) are never stored explicitly. Instead, they are
//!   directly evaluated at the point indicated by a challenge of “type”
//!   `Indeterminate`, giving rise to “running products”, “running evaluations”,
//!   _et cetera_.
//! - **Terminals**. The public input (respectively output) of the program is
//!   not stored in any table. Instead, the terminal of the Evaluation Argument
//!   is computed directly from the public input (respectively output) and the
//!   indeterminate.

use arbitrary::Arbitrary;
use std::ops::Index;
use std::ops::Range;
use std::ops::RangeInclusive;
use strum::EnumCount;
use twenty_first::prelude::*;
use twenty_first::tip5;

use air::challenge_id::ChallengeId;
use air::cross_table_argument::CrossTableArg;
use air::cross_table_argument::EvalArg;

use crate::prelude::Claim;

/// The `Challenges` struct holds the challenges used in Triton VM. The concrete
/// challenges are known only at runtime. The challenges are indexed using enum
/// [`ChallengeId`]. The `Challenges` struct is essentially a thin wrapper
/// around an array of [`XFieldElement`]s, providing convenience methods.
#[derive(Debug, Clone, Arbitrary)]
pub struct Challenges {
    pub challenges: [XFieldElement; Self::COUNT],
}

impl Challenges {
    /// The total number of challenges used in Triton VM.
    pub const COUNT: usize = ChallengeId::COUNT;

    /// The number of weights to sample using the Fiat-Shamir heuristic. This
    /// number is lower than the number of challenges because several
    /// challenges are not sampled, but computed from publicly known values
    /// and other, sampled challenges.
    ///
    /// Concretely:
    /// - The [`StandardInputTerminal`][std_in_term] is computed from Triton
    ///   VM's public input and the sampled indeterminate
    ///   [`StandardInputIndeterminate`][std_in_ind].
    /// - The [`StandardOutputTerminal`][std_out_term] is computed from Triton
    ///   VM's public output and the sampled indeterminate
    ///   [`StandardOutputIndeterminate`][std_out_ind].
    /// - The [`LookupTablePublicTerminal`][lk_up_term] is computed from the
    ///   publicly known and constant lookup table and the sampled indeterminate
    ///   [`LookupTablePublicIndeterminate`][lk_up_ind].
    /// - The [`CompressedProgramDigest`][program_digest] is computed from the
    ///   program to be executed and the sampled indeterminate
    ///   [`CompressProgramDigestIndeterminate`][program_digest_ind].
    ///
    /// [std_in_term]: ChallengeId::StandardInputTerminal
    /// [std_in_ind]: ChallengeId::StandardInputIndeterminate
    /// [std_out_term]: ChallengeId::StandardOutputTerminal
    /// [std_out_ind]: ChallengeId::StandardOutputIndeterminate
    /// [lk_up_term]: ChallengeId::LookupTablePublicTerminal
    /// [lk_up_ind]: ChallengeId::LookupTablePublicIndeterminate
    /// [program_digest]: ChallengeId::CompressedProgramDigest
    /// [program_digest_ind]: ChallengeId::CompressProgramDigestIndeterminate
    pub const SAMPLE_COUNT: usize = Self::COUNT - ChallengeId::NUM_DERIVED_CHALLENGES;

    pub fn new(mut challenges: Vec<XFieldElement>, claim: &Claim) -> Self {
        assert_eq!(Self::SAMPLE_COUNT, challenges.len());

        let compressed_digest = EvalArg::compute_terminal(
            &claim.program_digest.values(),
            EvalArg::default_initial(),
            challenges[ChallengeId::CompressProgramDigestIndeterminate.index()],
        );
        let input_terminal = EvalArg::compute_terminal(
            &claim.input,
            EvalArg::default_initial(),
            challenges[ChallengeId::StandardInputIndeterminate.index()],
        );
        let output_terminal = EvalArg::compute_terminal(
            &claim.output,
            EvalArg::default_initial(),
            challenges[ChallengeId::StandardOutputIndeterminate.index()],
        );
        let lookup_terminal = EvalArg::compute_terminal(
            &tip5::LOOKUP_TABLE.map(BFieldElement::from),
            EvalArg::default_initial(),
            challenges[ChallengeId::LookupTablePublicIndeterminate.index()],
        );

        challenges.insert(ChallengeId::StandardInputTerminal.index(), input_terminal);
        challenges.insert(ChallengeId::StandardOutputTerminal.index(), output_terminal);
        challenges.insert(
            ChallengeId::LookupTablePublicTerminal.index(),
            lookup_terminal,
        );
        challenges.insert(
            ChallengeId::CompressedProgramDigest.index(),
            compressed_digest,
        );
        assert_eq!(Self::COUNT, challenges.len());
        let challenges = challenges.try_into().unwrap();

        Self { challenges }
    }
}

impl Index<usize> for Challenges {
    type Output = XFieldElement;

    fn index(&self, id: usize) -> &Self::Output {
        &self.challenges[id]
    }
}

impl Index<Range<usize>> for Challenges {
    type Output = [XFieldElement];

    fn index(&self, indices: Range<usize>) -> &Self::Output {
        &self.challenges[indices.start..indices.end]
    }
}

impl Index<RangeInclusive<usize>> for Challenges {
    type Output = [XFieldElement];

    fn index(&self, indices: RangeInclusive<usize>) -> &Self::Output {
        &self.challenges[*indices.start()..=*indices.end()]
    }
}

impl Index<ChallengeId> for Challenges {
    type Output = XFieldElement;

    fn index(&self, id: ChallengeId) -> &Self::Output {
        &self[id.index()]
    }
}

impl Index<Range<ChallengeId>> for Challenges {
    type Output = [XFieldElement];

    fn index(&self, indices: Range<ChallengeId>) -> &Self::Output {
        &self[indices.start.index()..indices.end.index()]
    }
}

impl Index<RangeInclusive<ChallengeId>> for Challenges {
    type Output = [XFieldElement];

    fn index(&self, indices: RangeInclusive<ChallengeId>) -> &Self::Output {
        &self[indices.start().index()..=indices.end().index()]
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use super::*;
    use crate::prelude::Claim;
    use twenty_first::xfe;

    // For testing purposes only.
    impl Default for Challenges {
        fn default() -> Self {
            Self::placeholder(&Claim::default())
        }
    }

    impl Challenges {
        /// Stand-in challenges for use in tests. For non-interactive STARKs,
        /// use the Fiat-Shamir heuristic to derive the actual
        /// challenges.
        pub fn placeholder(claim: &Claim) -> Self {
            let stand_in_challenges = (1..=Self::SAMPLE_COUNT)
                .map(|i| xfe!([42, i as u64, 24]))
                .collect();
            Self::new(stand_in_challenges, claim)
        }
    }

    #[test]
    fn various_challenge_indexing_operations_are_possible() {
        let challenges = Challenges::default();
        let _ = challenges[ChallengeId::StackWeight0];
        let _ = challenges[ChallengeId::StackWeight0..ChallengeId::StackWeight8];
        let _ = challenges[ChallengeId::StackWeight0..=ChallengeId::StackWeight8];
        let _ = challenges[0];
        let _ = challenges[0..8];
        let _ = challenges[0..=8];
    }
}
