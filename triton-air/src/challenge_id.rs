use std::fmt::Debug;
use std::hash::Hash;

use arbitrary::Arbitrary;
use strum::Display;
use strum::EnumCount;
use strum::EnumIter;

/// A `ChallengeId` is a unique, symbolic identifier for a challenge used in
/// Triton VM.
///
/// Since almost all challenges relate to the Processor Table in some form, the
/// words “Processor Table” are usually omitted from the `ChallengeId`'s name.
#[repr(usize)]
#[derive(Debug, Display, Copy, Clone, Eq, PartialEq, Hash, EnumCount, EnumIter, Arbitrary)]
pub enum ChallengeId {
    /// The indeterminate for the [Evaluation Argument][eval] compressing the
    /// program digest into a single extension field element, _i.e._,
    /// [`CompressedProgramDigest`][Self::CompressedProgramDigest].
    /// Relates to program attestation.
    ///
    /// [eval]: crate::cross_table_argument::EvalArg
    CompressProgramDigestIndeterminate,

    /// The indeterminate for the [Evaluation Argument][eval] with standard
    /// input.
    ///
    /// [eval]: crate::cross_table_argument::EvalArg
    StandardInputIndeterminate,

    /// The indeterminate for the [Evaluation Argument][eval] with standard
    /// output.
    ///
    /// [eval]: crate::cross_table_argument::EvalArg
    StandardOutputIndeterminate,

    /// The indeterminate for the instruction
    /// [Lookup Argument](crate::cross_table_argument::LookupArg)
    /// between the [Processor Table](crate::table::processor) and the
    /// [Program Table](crate::table::program) guaranteeing that the
    /// instructions and their arguments are copied correctly.
    InstructionLookupIndeterminate,

    HashInputIndeterminate,
    HashDigestIndeterminate,
    SpongeIndeterminate,

    OpStackIndeterminate,
    RamIndeterminate,
    JumpStackIndeterminate,

    U32Indeterminate,

    /// The indeterminate for the Lookup Argument between the Processor Table
    /// and all memory-like tables, _i.e._, the OpStack Table, the Ram
    /// Table, and the JumpStack Table, guaranteeing that all clock jump
    /// differences are directed forward.
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
    /// - `Instruction` in the next row in the Program Table
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

    /// The indeterminate for compressing a [`RATE`][rate]-sized chunk of
    /// instructions into a single extension field element.
    /// Relates to program attestation.
    ///
    /// Used by the evaluation argument [`PrepareChunkEvalArg`][prep] and in the
    /// Hash Table.
    ///
    /// [rate]: twenty_first::prelude::tip5::RATE
    /// [prep]: crate::table_column::ProgramAuxColumn::PrepareChunkRunningEvaluation
    ProgramAttestationPrepareChunkIndeterminate,

    /// The indeterminate for the bus over which the [`RATE`][rate]-sized chunks
    /// of instructions are sent. Relates to program attestation.
    /// Used by the evaluation arguments [`SendChunkEvalArg`][send] and
    /// [`ReceiveChunkEvalArg`][recv]. See also:
    /// [`ProgramAttestationPrepareChunkIndeterminate`][ind].
    ///
    /// [rate]: twenty_first::prelude::tip5::RATE
    /// [send]: crate::table_column::ProgramAuxColumn::SendChunkRunningEvaluation
    /// [recv]: crate::table_column::HashAuxColumn::ReceiveChunkRunningEvaluation
    /// [ind]: ChallengeId::ProgramAttestationPrepareChunkIndeterminate
    ProgramAttestationSendChunkIndeterminate,

    HashCIWeight,

    StackWeight0,
    StackWeight1,
    StackWeight2,
    StackWeight3,
    StackWeight4,
    StackWeight5,
    StackWeight6,
    StackWeight7,
    StackWeight8,
    StackWeight9,
    StackWeight10,
    StackWeight11,
    StackWeight12,
    StackWeight13,
    StackWeight14,
    StackWeight15,

    /// The indeterminate for the Lookup Argument between the Hash Table and the
    /// Cascade Table.
    HashCascadeLookupIndeterminate,

    /// A weight for linearly combining multiple elements. Applies to
    /// - `*LkIn` in the Hash Table, and
    /// - `2^16·LookInHi + LookInLo` in the Cascade Table.
    HashCascadeLookInWeight,

    /// A weight for linearly combining multiple elements. Applies to
    /// - `*LkOut` in the Hash Table, and
    /// - `2^16·LookOutHi + LookOutLo` in the Cascade Table.
    HashCascadeLookOutWeight,

    /// The indeterminate for the Lookup Argument between the Cascade Table and
    /// the Lookup Table.
    CascadeLookupIndeterminate,

    /// A weight for linearly combining multiple elements. Applies to
    /// - `LkIn*` in the Cascade Table, and
    /// - `LookIn` in the Lookup Table.
    LookupTableInputWeight,

    /// A weight for linearly combining multiple elements. Applies to
    /// - `LkOut*` in the Cascade Table, and
    /// - `LookOut` in the Lookup Table.
    LookupTableOutputWeight,

    /// The indeterminate for the public Evaluation Argument establishing
    /// correctness of the Lookup Table.
    LookupTablePublicIndeterminate,

    U32LhsWeight,
    U32RhsWeight,
    U32CiWeight,
    U32ResultWeight,

    // Derived challenges.
    //
    // When modifying this, be sure to add to the compile-time assertions in the
    // `#[test] const fn compile_time_index_assertions() { … }`
    // at the end of this file.
    /// The terminal for the [`EvaluationArgument`][eval] with standard input.
    /// Makes use of challenge
    /// [`StandardInputIndeterminate`][Self::StandardInputIndeterminate].
    ///
    /// [eval]: crate::cross_table_argument::EvalArg
    StandardInputTerminal,

    /// The terminal for the [`EvaluationArgument`][eval] with standard output.
    /// Makes use of challenge
    /// [`StandardOutputIndeterminate`][Self::StandardOutputIndeterminate].
    ///
    /// [eval]: crate::cross_table_argument::EvalArg
    StandardOutputTerminal,

    /// The terminal for the [`EvaluationArgument`][eval] establishing
    /// correctness of the
    /// [Lookup Table](crate::table::lookup::LookupTable).
    /// Makes use of challenge
    /// [`LookupTablePublicIndeterminate`][Self::LookupTablePublicIndeterminate].
    ///
    /// [eval]: crate::cross_table_argument::EvalArg
    LookupTablePublicTerminal,

    /// The digest of the program to be executed, compressed into a single
    /// extension field element. The compression happens using an
    /// [`EvaluationArgument`][eval] under challenge
    /// [`CompressProgramDigestIndeterminate`][ind].
    ///
    /// Relates to program attestation.
    ///
    /// [eval]: crate::cross_table_argument::EvalArg
    /// [ind]: Self::CompressProgramDigestIndeterminate
    CompressedProgramDigest,
}

impl ChallengeId {
    /// The number of challenges derived from other challenges.
    ///
    /// The IDs of the derived challenges are guaranteed to be larger than the
    /// challenges they are derived from.
    pub const NUM_DERIVED_CHALLENGES: usize = 4;

    pub const fn index(&self) -> usize {
        *self as usize
    }
}

impl From<ChallengeId> for usize {
    fn from(id: ChallengeId) -> Self {
        id.index()
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
pub(crate) mod tests {
    use super::*;

    /// Terminal challenges are computed from public information, such as public
    /// input or public output, and other challenges. Because these other
    /// challenges are used to compute the terminal challenges, the terminal
    /// challenges must be inserted into the challenges vector after the
    /// used challenges.
    #[test]
    const fn compile_time_index_assertions() {
        const DERIVED: [ChallengeId; ChallengeId::NUM_DERIVED_CHALLENGES] = [
            ChallengeId::StandardInputTerminal,
            ChallengeId::StandardOutputTerminal,
            ChallengeId::LookupTablePublicTerminal,
            ChallengeId::CompressedProgramDigest,
        ];

        assert!(ChallengeId::StandardInputIndeterminate.index() < DERIVED[0].index());
        assert!(ChallengeId::StandardInputIndeterminate.index() < DERIVED[1].index());
        assert!(ChallengeId::StandardInputIndeterminate.index() < DERIVED[2].index());
        assert!(ChallengeId::StandardInputIndeterminate.index() < DERIVED[3].index());

        assert!(ChallengeId::StandardOutputIndeterminate.index() < DERIVED[0].index());
        assert!(ChallengeId::StandardOutputIndeterminate.index() < DERIVED[1].index());
        assert!(ChallengeId::StandardOutputIndeterminate.index() < DERIVED[2].index());
        assert!(ChallengeId::StandardOutputIndeterminate.index() < DERIVED[3].index());

        assert!(ChallengeId::CompressProgramDigestIndeterminate.index() < DERIVED[0].index());
        assert!(ChallengeId::CompressProgramDigestIndeterminate.index() < DERIVED[1].index());
        assert!(ChallengeId::CompressProgramDigestIndeterminate.index() < DERIVED[2].index());
        assert!(ChallengeId::CompressProgramDigestIndeterminate.index() < DERIVED[3].index());

        assert!(ChallengeId::LookupTablePublicIndeterminate.index() < DERIVED[0].index());
        assert!(ChallengeId::LookupTablePublicIndeterminate.index() < DERIVED[1].index());
        assert!(ChallengeId::LookupTablePublicIndeterminate.index() < DERIVED[2].index());
        assert!(ChallengeId::LookupTablePublicIndeterminate.index() < DERIVED[3].index());
    }

    // Ensure the compile-time assertions are actually executed by the compiler.
    const _: () = compile_time_index_assertions();
}
