//! Enums that convert table column names into `usize` indices. Allows
//! addressing columns by name rather than their hard-to-remember index.

use std::hash::Hash;

use strum::Display;
use strum::EnumCount;
use strum::EnumIter;

use crate::table::AUX_CASCADE_TABLE_START;
use crate::table::AUX_HASH_TABLE_START;
use crate::table::AUX_JUMP_STACK_TABLE_START;
use crate::table::AUX_LOOKUP_TABLE_START;
use crate::table::AUX_OP_STACK_TABLE_START;
use crate::table::AUX_PROCESSOR_TABLE_START;
use crate::table::AUX_PROGRAM_TABLE_START;
use crate::table::AUX_RAM_TABLE_START;
use crate::table::AUX_U32_TABLE_START;
use crate::table::CASCADE_TABLE_START;
use crate::table::HASH_TABLE_START;
use crate::table::JUMP_STACK_TABLE_START;
use crate::table::LOOKUP_TABLE_START;
use crate::table::OP_STACK_TABLE_START;
use crate::table::PROCESSOR_TABLE_START;
use crate::table::PROGRAM_TABLE_START;
use crate::table::RAM_TABLE_START;
use crate::table::U32_TABLE_START;

#[repr(usize)]
#[derive(Debug, Display, Copy, Clone, Eq, PartialEq, Hash, EnumCount, EnumIter)]
pub enum ProgramMainColumn {
    /// An instruction's address.
    Address,

    /// The (opcode of the) instruction.
    Instruction,

    /// How often an instruction has been executed.
    LookupMultiplicity,

    /// The index in the vector of length [`Rate`] that is to be absorbed in the
    /// Sponge in order to compute the program's digest.
    /// In other words:
    /// [`Address`] modulo [`Rate`].
    ///
    /// [`Address`]: ProgramMainColumn::Address
    /// [`Rate`]: twenty_first::tip5::RATE
    IndexInChunk,

    /// The inverse-or-zero of [`Rate`] - 1 - [`IndexInChunk`].
    /// Helper variable to guarantee [`IndexInChunk`]'s correct transition.
    ///
    /// [`IndexInChunk`]: ProgramMainColumn::IndexInChunk
    /// [`Rate`]: twenty_first::tip5::RATE
    MaxMinusIndexInChunkInv,

    /// Padding indicator for absorbing the program into the Sponge.
    IsHashInputPadding,

    /// Padding indicator for rows only required due to the dominating length of
    /// some other table.
    IsTablePadding,
}

#[repr(usize)]
#[derive(Debug, Display, Copy, Clone, Eq, PartialEq, Hash, EnumCount, EnumIter)]
pub enum ProgramAuxColumn {
    /// The server part of the instruction lookup.
    ///
    /// The counterpart to [`InstructionLookupClientLogDerivative`][client].
    ///
    /// [client]: ProcessorAuxColumn::InstructionLookupClientLogDerivative
    InstructionLookupServerLogDerivative,

    /// An evaluation argument accumulating [`RATE`][rate] many instructions
    /// before they are sent using
    /// [`SendChunkEvalArg`](ProgramAuxColumn::SendChunkRunningEvaluation).
    /// Resets to zero after each chunk.
    /// Relevant for program attestation.
    ///
    /// [rate]: twenty_first::tip5::RATE
    PrepareChunkRunningEvaluation,

    /// An evaluation argument over all [`RATE`][rate]-sized chunks of
    /// instructions, which are prepared in [`PrepareChunkEvalArg`][prep].
    /// This bus is used for sending those chunks to the Hash Table.
    /// Relevant for program attestation.
    ///
    /// The counterpart to
    /// [`RcvChunkEvalArg`](HashAuxColumn::ReceiveChunkRunningEvaluation).
    ///
    /// [rate]: twenty_first::tip5::RATE
    /// [prep]: ProgramAuxColumn::PrepareChunkRunningEvaluation
    SendChunkRunningEvaluation,
}

#[repr(usize)]
#[derive(Debug, Display, Copy, Clone, Eq, PartialEq, Hash, EnumCount, EnumIter)]
pub enum ProcessorMainColumn {
    CLK,
    IsPadding,
    IP,
    CI,
    NIA,
    IB0,
    IB1,
    IB2,
    IB3,
    IB4,
    IB5,
    IB6,
    JSP,
    JSO,
    JSD,
    ST0,
    ST1,
    ST2,
    ST3,
    ST4,
    ST5,
    ST6,
    ST7,
    ST8,
    ST9,
    ST10,
    ST11,
    ST12,
    ST13,
    ST14,
    ST15,
    OpStackPointer,
    HV0,
    HV1,
    HV2,
    HV3,
    HV4,
    HV5,
    /// The number of clock jump differences of magnitude `CLK` in all
    /// memory-like tables.
    ClockJumpDifferenceLookupMultiplicity,
}

#[repr(usize)]
#[derive(Debug, Display, Copy, Clone, Eq, PartialEq, Hash, EnumCount, EnumIter)]
pub enum ProcessorAuxColumn {
    InputTableEvalArg,
    OutputTableEvalArg,
    InstructionLookupClientLogDerivative,
    OpStackTablePermArg,
    RamTablePermArg,
    JumpStackTablePermArg,

    /// For copying the hash function's input to the hash coprocessor.
    HashInputEvalArg,
    /// For copying the hash digest from the hash coprocessor.
    HashDigestEvalArg,
    /// For copying the RATE next to-be-absorbed to the hash coprocessor and the
    /// RATE squeezed elements from the hash coprocessor, depending on the
    /// executed instruction.
    SpongeEvalArg,

    /// The (running sum of the) logarithmic derivative for the Lookup Argument
    /// with the U32 Table.
    U32LookupClientLogDerivative,

    /// The (running sum of the) logarithmic derivative for the clock jump
    /// difference Lookup Argument with the memory-like tables.
    ClockJumpDifferenceLookupServerLogDerivative,
}

#[repr(usize)]
#[derive(Debug, Display, Copy, Clone, Eq, PartialEq, Hash, EnumCount, EnumIter)]
pub enum OpStackMainColumn {
    CLK,
    IB1ShrinkStack,
    StackPointer,
    FirstUnderflowElement,
}

#[repr(usize)]
#[derive(Debug, Display, Copy, Clone, Eq, PartialEq, Hash, EnumCount, EnumIter)]
pub enum OpStackAuxColumn {
    RunningProductPermArg,
    /// The (running sum of the) logarithmic derivative for the clock jump
    /// difference Lookup Argument with the Processor Table.
    ClockJumpDifferenceLookupClientLogDerivative,
}

#[repr(usize)]
#[derive(Debug, Display, Copy, Clone, Eq, PartialEq, Hash, EnumCount, EnumIter)]
pub enum RamMainColumn {
    CLK,

    /// Is [`INSTRUCTION_TYPE_READ`] for instruction `read_mem` and
    /// [`INSTRUCTION_TYPE_WRITE`] for instruction `write_mem`. For padding
    /// rows, this is set to [`PADDING_INDICATOR`].
    ///
    /// [`INSTRUCTION_TYPE_READ`]: crate::table::ram::INSTRUCTION_TYPE_READ
    /// [`INSTRUCTION_TYPE_WRITE`]: crate::table::ram::INSTRUCTION_TYPE_WRITE
    /// [`PADDING_INDICATOR`]: crate::table::ram::PADDING_INDICATOR
    InstructionType,
    RamPointer,
    RamValue,
    InverseOfRampDifference,
    BezoutCoefficientPolynomialCoefficient0,
    BezoutCoefficientPolynomialCoefficient1,
}

#[repr(usize)]
#[derive(Debug, Display, Copy, Clone, Eq, PartialEq, Hash, EnumCount, EnumIter)]
pub enum RamAuxColumn {
    RunningProductOfRAMP,
    FormalDerivative,
    BezoutCoefficient0,
    BezoutCoefficient1,
    RunningProductPermArg,
    /// The (running sum of the) logarithmic derivative for the clock jump
    /// difference Lookup Argument with the Processor Table.
    ClockJumpDifferenceLookupClientLogDerivative,
}

#[repr(usize)]
#[derive(Debug, Display, Copy, Clone, Eq, PartialEq, Hash, EnumCount, EnumIter)]
pub enum JumpStackMainColumn {
    CLK,
    CI,
    JSP,
    JSO,
    JSD,
}

#[repr(usize)]
#[derive(Debug, Display, Copy, Clone, Eq, PartialEq, Hash, EnumCount, EnumIter)]
pub enum JumpStackAuxColumn {
    RunningProductPermArg,
    /// The (running sum of the) logarithmic derivative for the clock jump
    /// difference Lookup Argument with the Processor Table.
    ClockJumpDifferenceLookupClientLogDerivative,
}

#[repr(usize)]
#[derive(Debug, Display, Copy, Clone, Eq, PartialEq, Hash, EnumCount, EnumIter)]
pub enum HashMainColumn {
    /// The indicator for the [`HashTableMode`][mode].
    ///
    /// [mode]: crate::table::hash::HashTableMode
    Mode,

    /// The current instruction. Only relevant for [`Mode`][mode]
    /// [`Sponge`][mode_sponge] in order to distinguish between the
    /// different Sponge instructions.
    ///
    /// [mode]: HashMainColumn::Mode
    /// [mode_sponge]: crate::table::hash::HashTableMode::Sponge
    CI,

    /// The number of the current round in the permutation. The round number
    /// evolves as
    /// - 0 → 1 → 2 → 3 → 4 → 5 (→ 0) in [`Mode`][mode]s
    ///   [`ProgramHashing`][mode_prog_hash], [`Sponge`][mode_sponge] and
    ///   [`Hash`][mode_hash],
    /// - 0 → 0 in [`Mode`][mode] [`Sponge`][mode_sponge] if the current
    ///   instruction [`CI`][ci] is `sponge_init`, as an exception to above
    ///   rule, and
    /// - 0 → 0 in [`Mode`][mode] [`Pad`][mode_pad].
    ///
    /// [ci]: HashMainColumn::CI
    /// [mode]: HashMainColumn::Mode
    /// [mode_prog_hash]: crate::table::hash::HashTableMode::ProgramHashing
    /// [mode_sponge]: crate::table::hash::HashTableMode::Sponge
    /// [mode_hash]: crate::table::hash::HashTableMode::Hash
    /// [mode_pad]: crate::table::hash::HashTableMode::Pad
    RoundNumber,

    State0HighestLkIn,
    State0MidHighLkIn,
    State0MidLowLkIn,
    State0LowestLkIn,
    State1HighestLkIn,
    State1MidHighLkIn,
    State1MidLowLkIn,
    State1LowestLkIn,
    State2HighestLkIn,
    State2MidHighLkIn,
    State2MidLowLkIn,
    State2LowestLkIn,
    State3HighestLkIn,
    State3MidHighLkIn,
    State3MidLowLkIn,
    State3LowestLkIn,
    State0HighestLkOut,
    State0MidHighLkOut,
    State0MidLowLkOut,
    State0LowestLkOut,
    State1HighestLkOut,
    State1MidHighLkOut,
    State1MidLowLkOut,
    State1LowestLkOut,
    State2HighestLkOut,
    State2MidHighLkOut,
    State2MidLowLkOut,
    State2LowestLkOut,
    State3HighestLkOut,
    State3MidHighLkOut,
    State3MidLowLkOut,
    State3LowestLkOut,
    State4,
    State5,
    State6,
    State7,
    State8,
    State9,
    State10,
    State11,
    State12,
    State13,
    State14,
    State15,

    State0Inv,
    State1Inv,
    State2Inv,
    State3Inv,

    Constant0,
    Constant1,
    Constant2,
    Constant3,
    Constant4,
    Constant5,
    Constant6,
    Constant7,
    Constant8,
    Constant9,
    Constant10,
    Constant11,
    Constant12,
    Constant13,
    Constant14,
    Constant15,
}

#[repr(usize)]
#[derive(Debug, Display, Copy, Clone, Eq, PartialEq, Hash, EnumCount, EnumIter)]
pub enum HashAuxColumn {
    /// The evaluation argument corresponding to receiving instructions in
    /// chunks of size [`RATE`][rate]. The chunks are hashed in Sponge mode.
    /// This allows program attestation.
    ///
    /// The counterpart to
    /// [`SendChunkEvalArg`](ProgramAuxColumn::SendChunkRunningEvaluation).
    ///
    /// [rate]: twenty_first::tip5::RATE
    ReceiveChunkRunningEvaluation,

    HashInputRunningEvaluation,
    HashDigestRunningEvaluation,

    SpongeRunningEvaluation,

    CascadeState0HighestClientLogDerivative,
    CascadeState0MidHighClientLogDerivative,
    CascadeState0MidLowClientLogDerivative,
    CascadeState0LowestClientLogDerivative,

    CascadeState1HighestClientLogDerivative,
    CascadeState1MidHighClientLogDerivative,
    CascadeState1MidLowClientLogDerivative,
    CascadeState1LowestClientLogDerivative,

    CascadeState2HighestClientLogDerivative,
    CascadeState2MidHighClientLogDerivative,
    CascadeState2MidLowClientLogDerivative,
    CascadeState2LowestClientLogDerivative,

    CascadeState3HighestClientLogDerivative,
    CascadeState3MidHighClientLogDerivative,
    CascadeState3MidLowClientLogDerivative,
    CascadeState3LowestClientLogDerivative,
}

#[repr(usize)]
#[derive(Debug, Display, Copy, Clone, Eq, PartialEq, Hash, EnumCount, EnumIter)]
pub enum CascadeMainColumn {
    /// Indicator for padding rows.
    IsPadding,

    /// The more significant bits of the lookup input.
    LookInHi,

    /// The less significant bits of the lookup input.
    LookInLo,

    /// The more significant bits of the lookup output.
    LookOutHi,

    /// The less significant bits of the lookup output.
    LookOutLo,

    /// The number of times the S-Box is evaluated, _i.e._, the value is looked
    /// up.
    LookupMultiplicity,
}

#[repr(usize)]
#[derive(Debug, Display, Copy, Clone, Eq, PartialEq, Hash, EnumCount, EnumIter)]
pub enum CascadeAuxColumn {
    /// The (running sum of the) logarithmic derivative for the Lookup Argument
    /// with the Hash Table. In every row, the sum accumulates
    /// `LookupMultiplicity / (X - Combo)` where `X` is a verifier-supplied
    /// challenge and `Combo` is the weighted sum of
    /// - `2^8·LookInHi + LookInLo`, and
    /// - `2^8·LookOutHi + LookOutLo` with weights supplied by the verifier.
    HashTableServerLogDerivative,

    /// The (running sum of the) logarithmic derivative for the Lookup Argument
    /// with the Lookup Table. In every row, accumulates the two summands
    /// - `1 / combo_hi` where `combo_hi` is the verifier-weighted combination
    ///   of `LookInHi` and `LookOutHi`, and
    /// - `1 / combo_lo` where `combo_lo` is the verifier-weighted combination
    ///   of `LookInLo` and `LookOutLo`.
    LookupTableClientLogDerivative,
}

#[repr(usize)]
#[derive(Debug, Display, Copy, Clone, Eq, PartialEq, Hash, EnumCount, EnumIter)]
pub enum LookupMainColumn {
    /// Indicator for padding rows.
    IsPadding,

    /// The lookup input.
    LookIn,

    /// The lookup output.
    LookOut,

    /// The number of times the value is looked up.
    LookupMultiplicity,
}

#[repr(usize)]
#[derive(Debug, Display, Copy, Clone, Eq, PartialEq, Hash, EnumCount, EnumIter)]
pub enum LookupAuxColumn {
    /// The (running sum of the) logarithmic derivative for the Lookup Argument
    /// with the Cascade Table. In every row, accumulates the summand
    /// `LookupMultiplicity / Combo` where `Combo` is the verifier-weighted
    /// combination of `LookIn` and `LookOut`.
    CascadeTableServerLogDerivative,

    /// The running sum for the public evaluation argument of the Lookup Table.
    /// In every row, accumulates `LookOut`.
    PublicEvaluationArgument,
}

#[repr(usize)]
#[derive(Debug, Display, Copy, Clone, Eq, PartialEq, Hash, EnumCount, EnumIter)]
pub enum U32MainColumn {
    /// Marks the beginning of an independent section within the U32 table.
    CopyFlag,

    /// The number of bits that LHS and RHS have already been shifted by.
    Bits,

    /// The inverse-or-zero of the difference between
    /// 1. the first disallowed number of bits to shift LHS and RHS by, _i.e.,_
    ///    33, and
    /// 2. the number of bits that LHS and RHS have already been shifted by.
    BitsMinus33Inv,

    /// Current Instruction, the instruction the processor is currently
    /// executing.
    CI,

    /// Left-hand side of the operation.
    LHS,

    /// The inverse-or-zero of LHS. Needed to check whether `LHS` is unequal to
    /// 0.
    LhsInv,

    /// Right-hand side of the operation.
    RHS,

    /// The inverse-or-zero of RHS. Needed to check whether `RHS` is unequal to
    /// 0.
    RhsInv,

    /// The result (or intermediate result) of the instruction requested by the
    /// processor.
    Result,

    /// The number of times the processor has executed the current instruction
    /// with the same arguments.
    LookupMultiplicity,
}

#[repr(usize)]
#[derive(Debug, Display, Copy, Clone, Eq, PartialEq, Hash, EnumCount, EnumIter)]
pub enum U32AuxColumn {
    /// The (running sum of the) logarithmic derivative for the Lookup Argument
    /// with the Processor Table.
    LookupServerLogDerivative,
}

/// A trait for the columns of the master main table. This trait is implemented
/// for all enums relating to the main tables. This trait provides two methods:
/// - one to get the index of the column in the “local” main table, _i.e., not
///   the master base table, and
/// - one to get the index of the column in the master main table.
pub trait MasterMainColumn {
    /// The index of the column in the “local” main table, _i.e., not the master
    /// base table.
    fn main_index(&self) -> usize;

    /// The index of the column in the master main table.
    fn master_main_index(&self) -> usize;
}

impl MasterMainColumn for ProgramMainColumn {
    #[inline]
    fn main_index(&self) -> usize {
        (*self) as usize
    }

    #[inline]
    fn master_main_index(&self) -> usize {
        PROGRAM_TABLE_START + self.main_index()
    }
}

impl MasterMainColumn for ProcessorMainColumn {
    #[inline]
    fn main_index(&self) -> usize {
        (*self) as usize
    }

    #[inline]
    fn master_main_index(&self) -> usize {
        PROCESSOR_TABLE_START + self.main_index()
    }
}

impl MasterMainColumn for OpStackMainColumn {
    #[inline]
    fn main_index(&self) -> usize {
        (*self) as usize
    }

    #[inline]
    fn master_main_index(&self) -> usize {
        OP_STACK_TABLE_START + self.main_index()
    }
}

impl MasterMainColumn for RamMainColumn {
    #[inline]
    fn main_index(&self) -> usize {
        (*self) as usize
    }

    #[inline]
    fn master_main_index(&self) -> usize {
        RAM_TABLE_START + self.main_index()
    }
}

impl MasterMainColumn for JumpStackMainColumn {
    #[inline]
    fn main_index(&self) -> usize {
        (*self) as usize
    }

    #[inline]
    fn master_main_index(&self) -> usize {
        JUMP_STACK_TABLE_START + self.main_index()
    }
}

impl MasterMainColumn for HashMainColumn {
    #[inline]
    fn main_index(&self) -> usize {
        (*self) as usize
    }

    #[inline]
    fn master_main_index(&self) -> usize {
        HASH_TABLE_START + self.main_index()
    }
}

impl MasterMainColumn for CascadeMainColumn {
    #[inline]
    fn main_index(&self) -> usize {
        (*self) as usize
    }

    #[inline]
    fn master_main_index(&self) -> usize {
        CASCADE_TABLE_START + self.main_index()
    }
}

impl MasterMainColumn for LookupMainColumn {
    #[inline]
    fn main_index(&self) -> usize {
        (*self) as usize
    }

    #[inline]
    fn master_main_index(&self) -> usize {
        LOOKUP_TABLE_START + self.main_index()
    }
}

impl MasterMainColumn for U32MainColumn {
    #[inline]
    fn main_index(&self) -> usize {
        (*self) as usize
    }

    #[inline]
    fn master_main_index(&self) -> usize {
        U32_TABLE_START + self.main_index()
    }
}

/// A trait for the columns in the master auxiliary table. This trait is
/// implemented for all enums relating to the auxiliary tables. The trait
/// provides two methods:
/// - one to get the index of the column in the “local” auxiliary table, _i.e._,
///   not the master auxiliary table, and
/// - one to get the index of the column in the master auxiliary table.
pub trait MasterAuxColumn {
    /// The index of the column in the “local” auxiliary table, _i.e._, not the
    /// master extension table.
    fn aux_index(&self) -> usize;

    /// The index of the column in the master auxiliary table.
    fn master_aux_index(&self) -> usize;
}

impl MasterAuxColumn for ProgramAuxColumn {
    #[inline]
    fn aux_index(&self) -> usize {
        (*self) as usize
    }

    #[inline]
    fn master_aux_index(&self) -> usize {
        AUX_PROGRAM_TABLE_START + self.aux_index()
    }
}

impl MasterAuxColumn for ProcessorAuxColumn {
    #[inline]
    fn aux_index(&self) -> usize {
        (*self) as usize
    }

    #[inline]
    fn master_aux_index(&self) -> usize {
        AUX_PROCESSOR_TABLE_START + self.aux_index()
    }
}

impl MasterAuxColumn for OpStackAuxColumn {
    #[inline]
    fn aux_index(&self) -> usize {
        (*self) as usize
    }

    #[inline]
    fn master_aux_index(&self) -> usize {
        AUX_OP_STACK_TABLE_START + self.aux_index()
    }
}

impl MasterAuxColumn for RamAuxColumn {
    #[inline]
    fn aux_index(&self) -> usize {
        (*self) as usize
    }

    #[inline]
    fn master_aux_index(&self) -> usize {
        AUX_RAM_TABLE_START + self.aux_index()
    }
}

impl MasterAuxColumn for JumpStackAuxColumn {
    #[inline]
    fn aux_index(&self) -> usize {
        (*self) as usize
    }

    #[inline]
    fn master_aux_index(&self) -> usize {
        AUX_JUMP_STACK_TABLE_START + self.aux_index()
    }
}

impl MasterAuxColumn for HashAuxColumn {
    #[inline]
    fn aux_index(&self) -> usize {
        (*self) as usize
    }

    #[inline]
    fn master_aux_index(&self) -> usize {
        AUX_HASH_TABLE_START + self.aux_index()
    }
}

impl MasterAuxColumn for CascadeAuxColumn {
    #[inline]
    fn aux_index(&self) -> usize {
        (*self) as usize
    }

    #[inline]
    fn master_aux_index(&self) -> usize {
        AUX_CASCADE_TABLE_START + self.aux_index()
    }
}

impl MasterAuxColumn for LookupAuxColumn {
    #[inline]
    fn aux_index(&self) -> usize {
        (*self) as usize
    }

    #[inline]
    fn master_aux_index(&self) -> usize {
        AUX_LOOKUP_TABLE_START + self.aux_index()
    }
}

impl MasterAuxColumn for U32AuxColumn {
    #[inline]
    fn aux_index(&self) -> usize {
        (*self) as usize
    }

    #[inline]
    fn master_aux_index(&self) -> usize {
        AUX_U32_TABLE_START + self.aux_index()
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use strum::IntoEnumIterator;

    use super::*;

    #[test]
    fn master_main_table_is_contiguous() {
        let mut expected_column_index = 0;
        for column in ProgramMainColumn::iter() {
            assert_eq!(expected_column_index, column.master_main_index());
            expected_column_index += 1;
        }
        for column in ProcessorMainColumn::iter() {
            assert_eq!(expected_column_index, column.master_main_index());
            expected_column_index += 1;
        }
        for column in OpStackMainColumn::iter() {
            assert_eq!(expected_column_index, column.master_main_index());
            expected_column_index += 1;
        }
        for column in RamMainColumn::iter() {
            assert_eq!(expected_column_index, column.master_main_index());
            expected_column_index += 1;
        }
        for column in JumpStackMainColumn::iter() {
            assert_eq!(expected_column_index, column.master_main_index());
            expected_column_index += 1;
        }
        for column in HashMainColumn::iter() {
            assert_eq!(expected_column_index, column.master_main_index());
            expected_column_index += 1;
        }
        for column in CascadeMainColumn::iter() {
            assert_eq!(expected_column_index, column.master_main_index());
            expected_column_index += 1;
        }
        for column in LookupMainColumn::iter() {
            assert_eq!(expected_column_index, column.master_main_index());
            expected_column_index += 1;
        }
        for column in U32MainColumn::iter() {
            assert_eq!(expected_column_index, column.master_main_index());
            expected_column_index += 1;
        }
    }

    #[test]
    fn master_aux_table_is_contiguous() {
        let mut expected_column_index = 0;
        for column in ProgramAuxColumn::iter() {
            assert_eq!(expected_column_index, column.master_aux_index());
            expected_column_index += 1;
        }
        for column in ProcessorAuxColumn::iter() {
            assert_eq!(expected_column_index, column.master_aux_index());
            expected_column_index += 1;
        }
        for column in OpStackAuxColumn::iter() {
            assert_eq!(expected_column_index, column.master_aux_index());
            expected_column_index += 1;
        }
        for column in RamAuxColumn::iter() {
            assert_eq!(expected_column_index, column.master_aux_index());
            expected_column_index += 1;
        }
        for column in JumpStackAuxColumn::iter() {
            assert_eq!(expected_column_index, column.master_aux_index());
            expected_column_index += 1;
        }
        for column in HashAuxColumn::iter() {
            assert_eq!(expected_column_index, column.master_aux_index());
            expected_column_index += 1;
        }
        for column in CascadeAuxColumn::iter() {
            assert_eq!(expected_column_index, column.master_aux_index());
            expected_column_index += 1;
        }
        for column in LookupAuxColumn::iter() {
            assert_eq!(expected_column_index, column.master_aux_index());
            expected_column_index += 1;
        }
        for column in U32AuxColumn::iter() {
            assert_eq!(expected_column_index, column.master_aux_index());
            expected_column_index += 1;
        }
    }
}
