//! Challenges are needed for the [cross-table arguments](CrossTableArg), _i.e._,
//! [Permutation Arguments](crate::table::cross_table_argument::PermArg),
//! [Evaluation Arguments](crate::table::cross_table_argument::EvalArg), and
//! [Lookup Arguments](crate::table::cross_table_argument::LookupArg),
//! as well as for the RAM Table's Contiguity Argument.
//!
//! There are three types of challenges:
//! - **Weights**. Weights are used to linearly combine multiple elements into one element. The
//! resulting single element can then be used in a cross-table argument.
//! - **Indeterminates**. All cross-table arguments work by checking the equality of polynomials (or
//! rational functions). Through the Schwartz-Zippel lemma, this equality check can be performed
//! by evaluating the polynomials (or rational functions) in a single point. The challenges that
//! are indeterminates are exactly this evaluation point. The polynomials (or rational functions)
//! are never stored explicitly. Instead, they are directly evaluated at the point indicated by a
//! challenge of “type” `Indeterminate`, giving rise to “running products”, “running
//! evaluations”, _et cetera_.
//! - **Terminals**. The public input (respectively output) of the program is not stored in any
//! table. Instead, the terminal of the Evaluation Argument is computed directly from the
//! public input (respectively output) and the indeterminate.

use std::fmt::Debug;
use std::hash::Hash;
use std::ops::Index;
use std::ops::Range;
use std::ops::RangeInclusive;

use arbitrary::Arbitrary;
use strum::Display;
use strum::EnumCount;
use strum::EnumIter;
use twenty_first::prelude::*;

use crate::table::challenges::ChallengeId::*;
use crate::table::cross_table_argument::CrossTableArg;
use crate::table::cross_table_argument::EvalArg;
use crate::Claim;

/// A `ChallengeId` is a unique, symbolic identifier for a challenge used in Triton VM. The
/// `ChallengeId` enum works in tandem with the struct [`Challenges`], which can be
/// instantiated to hold actual challenges that can be indexed by some `ChallengeId`.
///
/// Since almost all challenges relate to the Processor Table in some form, the words “Processor
/// Table” are usually omitted from the `ChallengeId`'s name.
#[repr(usize)]
#[derive(Debug, Display, Copy, Clone, Eq, PartialEq, Hash, EnumCount, EnumIter, Arbitrary)]
pub enum ChallengeId {
    /// The indeterminate for the [Evaluation Argument](EvalArg) compressing the program digest
    /// into a single extension field element, _i.e._, [`CompressedProgramDigest`].
    /// Relates to program attestation.
    CompressProgramDigestIndeterminate,

    /// The indeterminate for the [Evaluation Argument](EvalArg) with standard input.
    StandardInputIndeterminate,

    /// The indeterminate for the [Evaluation Argument](EvalArg) with standard output.
    StandardOutputIndeterminate,

    /// The indeterminate for the instruction
    /// [Lookup Argument](crate::table::cross_table_argument::LookupArg)
    /// between the [Processor Table](crate::table::processor_table) and the
    /// [Program Table](crate::table::program_table) guaranteeing that the instructions and their
    /// arguments are copied correctly.
    InstructionLookupIndeterminate,

    HashInputIndeterminate,
    HashDigestIndeterminate,
    SpongeIndeterminate,

    OpStackIndeterminate,
    RamIndeterminate,
    JumpStackIndeterminate,

    U32Indeterminate,

    /// The indeterminate for the Lookup Argument between the Processor Table and all memory-like
    /// tables, _i.e._, the OpStack Table, the Ram Table, and the JumpStack Table, guaranteeing
    /// that all clock jump differences are directed forward.
    ClockJumpDifferenceLookupIndeterminate,

    /// The indeterminate for the Contiguity Argument within the Ram Table.
    RamTableBezoutRelationIndeterminate,

    /// A weight for linearly combining multiple elements. Applies to
    /// - `Address` in the Program Table
    /// - `IP` in the Processor Table
    ProgramAddressWeight,

    /// A weight for linearly combining multiple elements. Applies to
    /// - `Instruction` in the Program Table
    /// - `CI` in the Processor Table
    ProgramInstructionWeight,

    /// A weight for linearly combining multiple elements. Applies to
    /// - `Instruction'` (_i.e._, in the next row) in the Program Table
    /// - `NIA` in the Processor Table
    ProgramNextInstructionWeight,

    OpStackClkWeight,
    OpStackIb1Weight,
    OpStackPointerWeight,
    OpStackFirstUnderflowElementWeight,

    RamClkWeight,
    RamPointerWeight,
    RamValueWeight,
    RamInstructionTypeWeight,

    JumpStackClkWeight,
    JumpStackCiWeight,
    JumpStackJspWeight,
    JumpStackJsoWeight,
    JumpStackJsdWeight,

    /// The indeterminate for compressing a [`RATE`][rate]-sized chunk of instructions into a
    /// single extension field element.
    /// Relates to program attestation.
    ///
    /// Used by the evaluation argument [`PrepareChunkEvalArg`][prep] and in the Hash Table.
    ///
    /// [rate]: twenty_first::shared_math::tip5::RATE
    /// [prep]: crate::table::table_column::ProgramExtTableColumn::PrepareChunkRunningEvaluation
    ProgramAttestationPrepareChunkIndeterminate,

    /// The indeterminate for the bus over which the [`RATE`][rate]-sized chunks of instructions
    /// are sent. Relates to program attestation.
    /// Used by the evaluation arguments [`SendChunkEvalArg`][send] and
    /// [`ReceiveChunkEvalArg`][recv]. See also: [`ProgramAttestationPrepareChunkIndeterminate`].
    ///
    /// [rate]: twenty_first::shared_math::tip5::RATE
    /// [send]: crate::table::table_column::ProgramExtTableColumn::SendChunkRunningEvaluation
    /// [recv]: crate::table::table_column::HashExtTableColumn::ReceiveChunkRunningEvaluation
    ProgramAttestationSendChunkIndeterminate,

    HashCIWeight,
    HashStateWeight0,
    HashStateWeight1,
    HashStateWeight2,
    HashStateWeight3,
    HashStateWeight4,
    HashStateWeight5,
    HashStateWeight6,
    HashStateWeight7,
    HashStateWeight8,
    HashStateWeight9,
    HashStateWeight10,
    HashStateWeight11,
    HashStateWeight12,
    HashStateWeight13,
    HashStateWeight14,
    HashStateWeight15,

    /// The indeterminate for the Lookup Argument between the Hash Table and the Cascade Table.
    HashCascadeLookupIndeterminate,

    /// A weight for linearly combining multiple elements. Applies to
    /// - `*LkIn` in the Hash Table, and
    /// - `2^16·LookInHi + LookInLo` in the Cascade Table.
    HashCascadeLookInWeight,

    /// A weight for linearly combining multiple elements. Applies to
    /// - `*LkOut` in the Hash Table, and
    /// - `2^16·LookOutHi + LookOutLo` in the Cascade Table.
    HashCascadeLookOutWeight,

    /// The indeterminate for the Lookup Argument between the Cascade Table and the Lookup Table.
    CascadeLookupIndeterminate,

    /// A weight for linearly combining multiple elements. Applies to
    /// - `LkIn*` in the Cascade Table, and
    /// - `LookIn` in the Lookup Table.
    LookupTableInputWeight,

    /// A weight for linearly combining multiple elements. Applies to
    /// - `LkOut*` in the Cascade Table, and
    /// - `LookOut` in the Lookup Table.
    LookupTableOutputWeight,

    /// The indeterminate for the public Evaluation Argument establishing correctness of the
    /// Lookup Table.
    LookupTablePublicIndeterminate,

    U32LhsWeight,
    U32RhsWeight,
    U32CiWeight,
    U32ResultWeight,

    /// The terminal for the [`EvaluationArgument`](EvalArg) with standard input.
    /// Makes use of challenge [`StandardInputIndeterminate`].
    StandardInputTerminal,

    /// The terminal for the [`EvaluationArgument`](EvalArg) with standard output.
    /// Makes use of challenge [`StandardOutputIndeterminate`].
    StandardOutputTerminal,

    /// The terminal for the [`EvaluationArgument`](EvalArg) establishing correctness of the
    /// [Lookup Table](crate::table::lookup_table::LookupTable).
    /// Makes use of challenge [`LookupTablePublicIndeterminate`].
    LookupTablePublicTerminal,

    /// The digest of the program to be executed, compressed into a single extension field element.
    /// The compression happens using an [`EvaluationArgument`](EvalArg) under challenge
    /// [`CompressProgramDigestIndeterminate`].
    /// Relates to program attestation.
    CompressedProgramDigest,
}

impl ChallengeId {
    pub const fn index(&self) -> usize {
        *self as usize
    }
}

impl From<ChallengeId> for usize {
    fn from(id: ChallengeId) -> Self {
        id.index()
    }
}

/// The `Challenges` struct holds the challenges used in Triton VM. The concrete challenges are
/// known only at runtime. The challenges are indexed using enum [`ChallengeId`]. The `Challenges`
/// struct is essentially a thin wrapper around an array of [`XFieldElement`]s, providing
/// convenience methods.
#[derive(Debug, Clone, Arbitrary)]
pub struct Challenges {
    pub challenges: [XFieldElement; Self::COUNT],
}

impl Challenges {
    /// The total number of challenges used in Triton VM.
    pub const COUNT: usize = ChallengeId::COUNT;

    /// The number of weights to sample using the Fiat-Shamir heuristic. This number is lower
    /// than the number of challenges because several challenges are not sampled, but computed
    /// from publicly known values and other, sampled challenges.
    ///
    /// Concretely:
    /// - The [`StandardInputTerminal`] is computed from Triton VM's public input and the sampled
    /// indeterminate [`StandardInputIndeterminate`].
    /// - The [`StandardOutputTerminal`] is computed from Triton VM's public output and the sampled
    /// indeterminate [`StandardOutputIndeterminate`].
    /// - The [`LookupTablePublicTerminal`] is computed from the publicly known and constant
    /// lookup table and the sampled indeterminate [`LookupTablePublicIndeterminate`].
    /// - The [`CompressedProgramDigest`] is computed from the program to be executed and the
    /// sampled indeterminate [`CompressProgramDigestIndeterminate`].
    // When modifying this, be sure to add to the compile-time assertions in the
    // `#[test] const fn compile_time_index_assertions() { … }`
    // at the end of this file.
    pub const SAMPLE_COUNT: usize = Self::COUNT - 4;

    #[deprecated(since = "0.39.0", note = "Use `Self::COUNT` instead")]
    pub const fn count() -> usize {
        Self::COUNT
    }

    #[deprecated(since = "0.39.0", note = "Use `Self::SAMPLE_COUNT` instead")]
    pub const fn num_challenges_to_sample() -> usize {
        Self::SAMPLE_COUNT
    }

    pub fn new(mut challenges: Vec<XFieldElement>, claim: &Claim) -> Self {
        assert_eq!(Self::SAMPLE_COUNT, challenges.len());

        let compressed_digest = EvalArg::compute_terminal(
            &claim.program_digest.values(),
            EvalArg::default_initial(),
            challenges[CompressProgramDigestIndeterminate.index()],
        );
        let input_terminal = EvalArg::compute_terminal(
            &claim.input,
            EvalArg::default_initial(),
            challenges[StandardInputIndeterminate.index()],
        );
        let output_terminal = EvalArg::compute_terminal(
            &claim.output,
            EvalArg::default_initial(),
            challenges[StandardOutputIndeterminate.index()],
        );
        let lookup_terminal = EvalArg::compute_terminal(
            &tip5::LOOKUP_TABLE.map(BFieldElement::from),
            EvalArg::default_initial(),
            challenges[LookupTablePublicIndeterminate.index()],
        );

        challenges.insert(StandardInputTerminal.index(), input_terminal);
        challenges.insert(StandardOutputTerminal.index(), output_terminal);
        challenges.insert(LookupTablePublicTerminal.index(), lookup_terminal);
        challenges.insert(CompressedProgramDigest.index(), compressed_digest);
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
pub(crate) mod tests {
    use super::*;

    // For testing purposes only.
    impl Default for Challenges {
        fn default() -> Self {
            Self::placeholder(&Claim::default())
        }
    }

    impl Challenges {
        /// Stand-in challenges for use in tests. For non-interactive STARKs, use the
        /// Fiat-Shamir heuristic to derive the actual challenges.
        pub fn placeholder(claim: &Claim) -> Self {
            let stand_in_challenges = (1..=Self::SAMPLE_COUNT)
                .map(|i| xfe!([42, i as u64, 24]))
                .collect();
            Self::new(stand_in_challenges, claim)
        }
    }

    #[test]
    const fn compile_time_index_assertions() {
        // Terminal challenges are computed from public information, such as public input or
        // public output, and other challenges. Because these other challenges are used to compute
        // the terminal challenges, the terminal challenges must be inserted into the challenges
        // vector after the used challenges.
        assert!(StandardInputIndeterminate.index() < StandardInputTerminal.index());
        assert!(StandardInputIndeterminate.index() < StandardOutputTerminal.index());
        assert!(StandardInputIndeterminate.index() < LookupTablePublicTerminal.index());
        assert!(StandardInputIndeterminate.index() < CompressedProgramDigest.index());

        assert!(StandardOutputIndeterminate.index() < StandardInputTerminal.index());
        assert!(StandardOutputIndeterminate.index() < StandardOutputTerminal.index());
        assert!(StandardOutputIndeterminate.index() < LookupTablePublicTerminal.index());
        assert!(StandardOutputIndeterminate.index() < CompressedProgramDigest.index());

        assert!(CompressProgramDigestIndeterminate.index() < StandardInputTerminal.index());
        assert!(CompressProgramDigestIndeterminate.index() < StandardOutputTerminal.index());
        assert!(CompressProgramDigestIndeterminate.index() < LookupTablePublicTerminal.index());
        assert!(CompressProgramDigestIndeterminate.index() < CompressedProgramDigest.index());

        assert!(LookupTablePublicIndeterminate.index() < StandardInputTerminal.index());
        assert!(LookupTablePublicIndeterminate.index() < StandardOutputTerminal.index());
        assert!(LookupTablePublicIndeterminate.index() < LookupTablePublicTerminal.index());
        assert!(LookupTablePublicIndeterminate.index() < CompressedProgramDigest.index());
    }

    // Ensure the compile-time assertions are actually executed by the compiler.
    const _: () = compile_time_index_assertions();

    #[test]
    fn various_challenge_indexing_operations_are_possible() {
        let challenges = Challenges::placeholder(&Claim::default());
        let _ = challenges[HashStateWeight0];
        let _ = challenges[HashStateWeight0..HashStateWeight8];
        let _ = challenges[HashStateWeight0..=HashStateWeight8];
        let _ = challenges[0];
        let _ = challenges[0..8];
        let _ = challenges[0..=8];
    }
}
