//! Enums that convert table column names into `usize` indices
//!
//! Allows addressing columns by name rather than their hard-to-remember index.

use std::fmt::Display;
use std::hash::Hash;

use strum::{EnumCount, IntoEnumIterator};
use strum_macros::{Display, EnumCount as EnumCountMacro, EnumIter};

use crate::table::table_collection::EXT_HASH_TABLE_START;
use crate::table::table_collection::EXT_INSTRUCTION_TABLE_START;
use crate::table::table_collection::EXT_JUMP_STACK_TABLE_START;
use crate::table::table_collection::EXT_OP_STACK_TABLE_START;
use crate::table::table_collection::EXT_PROCESSOR_TABLE_START;
use crate::table::table_collection::EXT_PROGRAM_TABLE_START;
use crate::table::table_collection::EXT_RAM_TABLE_START;
use crate::table::table_collection::HASH_TABLE_START;
use crate::table::table_collection::INSTRUCTION_TABLE_START;
use crate::table::table_collection::JUMP_STACK_TABLE_START;
use crate::table::table_collection::OP_STACK_TABLE_START;
use crate::table::table_collection::PROCESSOR_TABLE_START;
use crate::table::table_collection::PROGRAM_TABLE_START;
use crate::table::table_collection::RAM_TABLE_START;

// -------- Program Table --------

#[derive(Display, Debug, Clone, Copy, PartialEq, Eq, EnumIter, EnumCountMacro, Hash)]
pub enum ProgramBaseTableColumn {
    Address,
    Instruction,
    IsPadding,
}

#[derive(Display, Debug, Clone, Copy, PartialEq, Eq, EnumIter, EnumCountMacro, Hash)]
pub enum ProgramExtTableColumn {
    RunningEvaluation,
}

// -------- Instruction Table --------

#[derive(Display, Debug, Clone, Copy, PartialEq, Eq, EnumIter, EnumCountMacro, Hash)]
pub enum InstructionBaseTableColumn {
    Address,
    CI,
    NIA,
    IsPadding,
}

#[derive(Display, Debug, Clone, Copy, PartialEq, Eq, EnumIter, EnumCountMacro, Hash)]
pub enum InstructionExtTableColumn {
    RunningProductPermArg,
    RunningEvaluation,
}

// -------- Processor Table --------

#[derive(Display, Debug, Clone, Copy, PartialEq, Eq, EnumIter, EnumCountMacro, Hash)]
pub enum ProcessorBaseTableColumn {
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
    OSP,
    OSV,
    HV0,
    HV1,
    HV2,
    HV3,
    ClockJumpDifference,
    ClockJumpDifferenceInverse,
    UniqueClockJumpDiffDiffInverse,
    RAMP,
    RAMV,
}

#[derive(Display, Debug, Clone, Copy, PartialEq, Eq, EnumIter, EnumCountMacro, Hash)]
pub enum ProcessorExtTableColumn {
    InputTableEvalArg,
    OutputTableEvalArg,
    InstructionTablePermArg,
    OpStackTablePermArg,
    RamTablePermArg,
    JumpStackTablePermArg,

    ToHashTableEvalArg,
    FromHashTableEvalArg,

    SelectedClockCyclesEvalArg,
    UniqueClockJumpDifferencesEvalArg,
    AllClockJumpDifferencesPermArg,
}

// -------- OpStack Table --------

#[derive(Display, Debug, Clone, Copy, PartialEq, Eq, EnumIter, EnumCountMacro, Hash)]
pub enum OpStackBaseTableColumn {
    CLK,
    InverseOfClkDiffMinusOne,
    IB1ShrinkStack,
    OSP,
    OSV,
}

#[derive(Display, Debug, Clone, Copy, PartialEq, Eq, EnumIter, EnumCountMacro, Hash)]
pub enum OpStackExtTableColumn {
    RunningProductPermArg,
    AllClockJumpDifferencesPermArg,
}

// -------- RAM Table --------

#[derive(Display, Debug, Clone, Copy, PartialEq, Eq, EnumIter, EnumCountMacro, Hash)]
pub enum RamBaseTableColumn {
    CLK,
    InverseOfClkDiffMinusOne,
    RAMP,
    RAMV,
    InverseOfRampDifference,
    BezoutCoefficientPolynomialCoefficient0,
    BezoutCoefficientPolynomialCoefficient1,
}

#[derive(Display, Debug, Clone, Copy, PartialEq, Eq, EnumIter, EnumCountMacro, Hash)]
pub enum RamExtTableColumn {
    RunningProductOfRAMP,
    FormalDerivative,
    BezoutCoefficient0,
    BezoutCoefficient1,
    RunningProductPermArg,
    AllClockJumpDifferencesPermArg,
}

// -------- JumpStack Table --------

#[derive(Display, Debug, Clone, Copy, PartialEq, Eq, EnumIter, EnumCountMacro, Hash)]
pub enum JumpStackBaseTableColumn {
    CLK,
    InverseOfClkDiffMinusOne,
    CI,
    JSP,
    JSO,
    JSD,
}

#[derive(Display, Debug, Clone, Copy, PartialEq, Eq, EnumIter, EnumCountMacro, Hash)]
pub enum JumpStackExtTableColumn {
    RunningProductPermArg,
    AllClockJumpDifferencesPermArg,
}

// -------- Hash Table --------

#[derive(Display, Debug, Clone, Copy, PartialEq, Eq, EnumIter, EnumCountMacro, Hash)]
pub enum HashBaseTableColumn {
    ROUNDNUMBER,
    STATE0,
    STATE1,
    STATE2,
    STATE3,
    STATE4,
    STATE5,
    STATE6,
    STATE7,
    STATE8,
    STATE9,
    STATE10,
    STATE11,
    STATE12,
    STATE13,
    STATE14,
    STATE15,
    CONSTANT0A,
    CONSTANT1A,
    CONSTANT2A,
    CONSTANT3A,
    CONSTANT4A,
    CONSTANT5A,
    CONSTANT6A,
    CONSTANT7A,
    CONSTANT8A,
    CONSTANT9A,
    CONSTANT10A,
    CONSTANT11A,
    CONSTANT12A,
    CONSTANT13A,
    CONSTANT14A,
    CONSTANT15A,
    CONSTANT0B,
    CONSTANT1B,
    CONSTANT2B,
    CONSTANT3B,
    CONSTANT4B,
    CONSTANT5B,
    CONSTANT6B,
    CONSTANT7B,
    CONSTANT8B,
    CONSTANT9B,
    CONSTANT10B,
    CONSTANT11B,
    CONSTANT12B,
    CONSTANT13B,
    CONSTANT14B,
    CONSTANT15B,
}

#[derive(Display, Debug, Clone, Copy, PartialEq, Eq, EnumIter, EnumCountMacro, Hash)]
pub enum HashExtTableColumn {
    ToProcessorRunningEvaluation,
    FromProcessorRunningEvaluation,
}

// --------------------------------------------------------------------

impl From<ProgramBaseTableColumn> for usize {
    fn from(column: ProgramBaseTableColumn) -> Self {
        ProgramBaseTableColumn::iter()
            .enumerate()
            .find(|&(_n, col)| column == col)
            .map(|(n, _col)| n)
            .unwrap()
    }
}

impl From<ProgramExtTableColumn> for usize {
    fn from(column: ProgramExtTableColumn) -> Self {
        ProgramExtTableColumn::iter()
            .enumerate()
            .find(|&(_n, col)| column == col)
            .map(|(n, _col)| n)
            .unwrap()
    }
}

impl From<InstructionBaseTableColumn> for usize {
    fn from(column: InstructionBaseTableColumn) -> Self {
        InstructionBaseTableColumn::iter()
            .enumerate()
            .find(|&(_n, col)| column == col)
            .map(|(n, _col)| n)
            .unwrap()
    }
}

impl From<InstructionExtTableColumn> for usize {
    fn from(column: InstructionExtTableColumn) -> Self {
        InstructionExtTableColumn::iter()
            .enumerate()
            .find(|&(_n, col)| column == col)
            .map(|(n, _col)| n)
            .unwrap()
    }
}

impl From<ProcessorBaseTableColumn> for usize {
    fn from(column: ProcessorBaseTableColumn) -> Self {
        ProcessorBaseTableColumn::iter()
            .enumerate()
            .find(|&(_n, col)| column == col)
            .map(|(n, _col)| n)
            .unwrap()
    }
}

impl From<ProcessorExtTableColumn> for usize {
    fn from(column: ProcessorExtTableColumn) -> Self {
        ProcessorExtTableColumn::iter()
            .enumerate()
            .find(|&(_n, col)| column == col)
            .map(|(n, _col)| n)
            .unwrap()
    }
}

impl From<OpStackBaseTableColumn> for usize {
    fn from(column: OpStackBaseTableColumn) -> Self {
        OpStackBaseTableColumn::iter()
            .enumerate()
            .find(|&(_n, col)| column == col)
            .map(|(n, _col)| n)
            .unwrap()
    }
}

impl From<OpStackExtTableColumn> for usize {
    fn from(column: OpStackExtTableColumn) -> Self {
        OpStackExtTableColumn::iter()
            .enumerate()
            .find(|&(_n, col)| column == col)
            .map(|(n, _col)| n)
            .unwrap()
    }
}

impl From<RamBaseTableColumn> for usize {
    fn from(column: RamBaseTableColumn) -> Self {
        RamBaseTableColumn::iter()
            .enumerate()
            .find(|&(_n, col)| column == col)
            .map(|(n, _col)| n)
            .unwrap()
    }
}

impl From<RamExtTableColumn> for usize {
    fn from(column: RamExtTableColumn) -> Self {
        RamExtTableColumn::iter()
            .enumerate()
            .find(|&(_n, col)| column == col)
            .map(|(n, _col)| n)
            .unwrap()
    }
}

impl From<JumpStackBaseTableColumn> for usize {
    fn from(column: JumpStackBaseTableColumn) -> Self {
        JumpStackBaseTableColumn::iter()
            .enumerate()
            .find(|&(_n, col)| column == col)
            .map(|(n, _col)| n)
            .unwrap()
    }
}

impl From<JumpStackExtTableColumn> for usize {
    fn from(column: JumpStackExtTableColumn) -> Self {
        JumpStackExtTableColumn::iter()
            .enumerate()
            .find(|&(_n, col)| column == col)
            .map(|(n, _col)| n)
            .unwrap()
    }
}

impl From<HashBaseTableColumn> for usize {
    fn from(column: HashBaseTableColumn) -> Self {
        HashBaseTableColumn::iter()
            .enumerate()
            .find(|&(_n, col)| column == col)
            .map(|(n, _col)| n)
            .unwrap()
    }
}

impl From<HashExtTableColumn> for usize {
    fn from(column: HashExtTableColumn) -> Self {
        HashExtTableColumn::iter()
            .enumerate()
            .find(|&(_n, col)| column == col)
            .map(|(n, _col)| n)
            .unwrap()
    }
}

// --------------------------------------------------------------------

pub trait MasterBaseTableColumn:
    Into<usize> + EnumCount + IntoEnumIterator + Hash + Copy + Eq + Display
{
    fn master_table_index(&self) -> usize;
}

pub trait MasterExtTableColumn:
    Into<usize> + EnumCount + IntoEnumIterator + Hash + Copy + Eq + Display
{
    fn master_table_index(&self) -> usize;
}

impl MasterBaseTableColumn for ProgramBaseTableColumn {
    fn master_table_index(&self) -> usize {
        PROGRAM_TABLE_START + usize::from(*self)
    }
}

impl MasterBaseTableColumn for InstructionBaseTableColumn {
    fn master_table_index(&self) -> usize {
        INSTRUCTION_TABLE_START + usize::from(*self)
    }
}

impl MasterBaseTableColumn for ProcessorBaseTableColumn {
    fn master_table_index(&self) -> usize {
        PROCESSOR_TABLE_START + usize::from(*self)
    }
}

impl MasterBaseTableColumn for OpStackBaseTableColumn {
    fn master_table_index(&self) -> usize {
        OP_STACK_TABLE_START + usize::from(*self)
    }
}

impl MasterBaseTableColumn for RamBaseTableColumn {
    fn master_table_index(&self) -> usize {
        RAM_TABLE_START + usize::from(*self)
    }
}

impl MasterBaseTableColumn for JumpStackBaseTableColumn {
    fn master_table_index(&self) -> usize {
        JUMP_STACK_TABLE_START + usize::from(*self)
    }
}

impl MasterBaseTableColumn for HashBaseTableColumn {
    fn master_table_index(&self) -> usize {
        HASH_TABLE_START + usize::from(*self)
    }
}

impl MasterExtTableColumn for ProgramExtTableColumn {
    fn master_table_index(&self) -> usize {
        EXT_PROGRAM_TABLE_START + usize::from(*self)
    }
}

impl MasterExtTableColumn for InstructionExtTableColumn {
    fn master_table_index(&self) -> usize {
        EXT_INSTRUCTION_TABLE_START + usize::from(*self)
    }
}

impl MasterExtTableColumn for ProcessorExtTableColumn {
    fn master_table_index(&self) -> usize {
        EXT_PROCESSOR_TABLE_START + usize::from(*self)
    }
}

impl MasterExtTableColumn for OpStackExtTableColumn {
    fn master_table_index(&self) -> usize {
        EXT_OP_STACK_TABLE_START + usize::from(*self)
    }
}

impl MasterExtTableColumn for RamExtTableColumn {
    fn master_table_index(&self) -> usize {
        EXT_RAM_TABLE_START + usize::from(*self)
    }
}

impl MasterExtTableColumn for JumpStackExtTableColumn {
    fn master_table_index(&self) -> usize {
        EXT_JUMP_STACK_TABLE_START + usize::from(*self)
    }
}

impl MasterExtTableColumn for HashExtTableColumn {
    fn master_table_index(&self) -> usize {
        EXT_HASH_TABLE_START + usize::from(*self)
    }
}

// --------------------------------------------------------------------
/*
// todo What's this associated type?  v
//  alternatively: make a thin wrapper
impl<C: BaseTableColumn> From<C> for usize {
    fn from(column: C) -> Self {
        C::iter()
            .enumerate()
            .find(|&(_n, col)| column == col)
            .map(|(n, _col)| n)
            .unwrap()
    }
}
 */
// --------------------------------------------------------------------

#[cfg(test)]
mod table_column_tests {
    use crate::table::hash_table;
    use crate::table::instruction_table;
    use crate::table::jump_stack_table;
    use crate::table::op_stack_table;
    use crate::table::processor_table;
    use crate::table::program_table;
    use crate::table::ram_table;

    use super::*;

    #[test]
    fn column_max_bound_matches_table_width() {
        assert_eq!(
            program_table::BASE_WIDTH,
            usize::from(ProgramBaseTableColumn::iter().last().unwrap()) + 1,
            "ProgramTable's BASE_WIDTH is 1 + its max column index",
        );
        assert_eq!(
            instruction_table::BASE_WIDTH,
            usize::from(InstructionBaseTableColumn::iter().last().unwrap()) + 1,
            "InstructionTable's BASE_WIDTH is 1 + its max column index",
        );
        assert_eq!(
            processor_table::BASE_WIDTH,
            usize::from(ProcessorBaseTableColumn::iter().last().unwrap()) + 1,
            "ProcessorTable's BASE_WIDTH is 1 + its max column index",
        );
        assert_eq!(
            op_stack_table::BASE_WIDTH,
            usize::from(OpStackBaseTableColumn::iter().last().unwrap()) + 1,
            "OpStackTable's BASE_WIDTH is 1 + its max column index",
        );
        assert_eq!(
            ram_table::BASE_WIDTH,
            usize::from(RamBaseTableColumn::iter().last().unwrap()) + 1,
            "RamTable's BASE_WIDTH is 1 + its max column index",
        );
        assert_eq!(
            jump_stack_table::BASE_WIDTH,
            usize::from(JumpStackBaseTableColumn::iter().last().unwrap()) + 1,
            "JumpStackTable's BASE_WIDTH is 1 + its max column index",
        );
        assert_eq!(
            hash_table::BASE_WIDTH,
            usize::from(HashBaseTableColumn::iter().last().unwrap()) + 1,
            "HashTable's BASE_WIDTH is 1 + its max column index",
        );

        assert_eq!(
            program_table::EXT_WIDTH,
            usize::from(ProgramExtTableColumn::iter().last().unwrap()) + 1,
            "ProgramTable's EXT_WIDTH is 1 + its max column index",
        );
        assert_eq!(
            instruction_table::EXT_WIDTH,
            usize::from(InstructionExtTableColumn::iter().last().unwrap()) + 1,
            "InstructionTable's EXT_WIDTH is 1 + its max column index",
        );
        assert_eq!(
            processor_table::EXT_WIDTH,
            usize::from(ProcessorExtTableColumn::iter().last().unwrap()) + 1,
            "ProcessorTable's EXT_WIDTH is 1 + its max column index",
        );
        assert_eq!(
            op_stack_table::EXT_WIDTH,
            usize::from(OpStackExtTableColumn::iter().last().unwrap()) + 1,
            "OpStack:Table's EXT_WIDTH is 1 + its max column index",
        );
        assert_eq!(
            ram_table::EXT_WIDTH,
            usize::from(RamExtTableColumn::iter().last().unwrap()) + 1,
            "RamTable's EXT_WIDTH is 1 + its max column index",
        );
        assert_eq!(
            jump_stack_table::EXT_WIDTH,
            usize::from(JumpStackExtTableColumn::iter().last().unwrap()) + 1,
            "JumpStack:Table's EXT_WIDTH is 1 + its max column index",
        );
        assert_eq!(
            hash_table::EXT_WIDTH,
            usize::from(HashExtTableColumn::iter().last().unwrap()) + 1,
            "HashTable's EXT_WIDTH is 1 + its max column index",
        );
    }

    #[test]
    fn master_base_table_is_contiguous() {
        let mut expected_column_index = 0;
        for column in ProgramBaseTableColumn::iter() {
            assert_eq!(expected_column_index, column.master_table_index());
            expected_column_index += 1;
        }
        for column in InstructionBaseTableColumn::iter() {
            assert_eq!(expected_column_index, column.master_table_index());
            expected_column_index += 1;
        }
        for column in ProcessorBaseTableColumn::iter() {
            assert_eq!(expected_column_index, column.master_table_index());
            expected_column_index += 1;
        }
        for column in OpStackBaseTableColumn::iter() {
            assert_eq!(expected_column_index, column.master_table_index());
            expected_column_index += 1;
        }
        for column in RamBaseTableColumn::iter() {
            assert_eq!(expected_column_index, column.master_table_index());
            expected_column_index += 1;
        }
        for column in JumpStackBaseTableColumn::iter() {
            assert_eq!(expected_column_index, column.master_table_index());
            expected_column_index += 1;
        }
        for column in HashBaseTableColumn::iter() {
            assert_eq!(expected_column_index, column.master_table_index());
            expected_column_index += 1;
        }
    }

    #[test]
    fn master_ext_table_is_contiguous() {
        let mut expected_column_index = 0;
        for column in ProgramExtTableColumn::iter() {
            assert_eq!(expected_column_index, column.master_table_index());
            expected_column_index += 1;
        }
        for column in InstructionExtTableColumn::iter() {
            assert_eq!(expected_column_index, column.master_table_index());
            expected_column_index += 1;
        }
        for column in ProcessorExtTableColumn::iter() {
            assert_eq!(expected_column_index, column.master_table_index());
            expected_column_index += 1;
        }
        for column in OpStackExtTableColumn::iter() {
            assert_eq!(expected_column_index, column.master_table_index());
            expected_column_index += 1;
        }
        for column in RamExtTableColumn::iter() {
            assert_eq!(expected_column_index, column.master_table_index());
            expected_column_index += 1;
        }
        for column in JumpStackExtTableColumn::iter() {
            assert_eq!(expected_column_index, column.master_table_index());
            expected_column_index += 1;
        }
        for column in HashExtTableColumn::iter() {
            assert_eq!(expected_column_index, column.master_table_index());
            expected_column_index += 1;
        }
    }
}
