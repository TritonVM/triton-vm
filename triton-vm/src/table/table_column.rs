//! Enums that convert table column names into `usize` indices
//!
//! These let one address a given column by its name rather than its arbitrary index.

// --------------------------------------------------------------------

use num_traits::Bounded;
use strum::{EnumCount, IntoEnumIterator};
use strum_macros::{Display, EnumCount as EnumCountMacro, EnumIter};

#[derive(Display, Debug, Clone, Copy, PartialEq, Eq, EnumIter, EnumCountMacro)]
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
    RAMV,
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

impl Bounded for ProcessorBaseTableColumn {
    fn min_value() -> Self {
        ProcessorBaseTableColumn::iter().next().unwrap()
    }

    fn max_value() -> Self {
        ProcessorBaseTableColumn::iter().last().unwrap()
    }
}

#[derive(Display, Debug, Clone, Copy, PartialEq, Eq, EnumIter, EnumCountMacro)]
pub enum ProcessorExtTableColumn {
    InputTableEvalArg,
    OutputTableEvalArg,
    InstructionTablePermArg,
    OpStackTablePermArg,
    RamTablePermArg,
    JumpStackTablePermArg,

    ToHashTableEvalArg,
    FromHashTableEvalArg,
}

impl From<ProcessorExtTableColumn> for usize {
    fn from(column: ProcessorExtTableColumn) -> Self {
        ProcessorExtTableColumn::iter()
            .enumerate()
            .find(|&(_n, col)| column == col)
            .map(|(n, _col)| n + ProcessorBaseTableColumn::COUNT)
            .unwrap()
    }
}

impl Bounded for ProcessorExtTableColumn {
    fn min_value() -> Self {
        ProcessorExtTableColumn::iter().next().unwrap()
    }

    fn max_value() -> Self {
        ProcessorExtTableColumn::iter().last().unwrap()
    }
}

// --------------------------------------------------------------------

#[derive(Display, Debug, Clone, Copy, PartialEq, Eq, EnumIter, EnumCountMacro)]
pub enum ProgramBaseTableColumn {
    Address,
    Instruction,
    IsPadding,
}

impl From<ProgramBaseTableColumn> for usize {
    fn from(column: ProgramBaseTableColumn) -> Self {
        ProgramBaseTableColumn::iter()
            .enumerate()
            .find(|&(_n, col)| column == col)
            .map(|(n, _col)| n)
            .unwrap()
    }
}

impl Bounded for ProgramBaseTableColumn {
    fn min_value() -> Self {
        ProgramBaseTableColumn::iter().next().unwrap()
    }

    fn max_value() -> Self {
        ProgramBaseTableColumn::iter().last().unwrap()
    }
}

#[derive(Display, Debug, Clone, Copy, PartialEq, Eq, EnumIter, EnumCountMacro)]
pub enum ProgramExtTableColumn {
    RunningEvaluation,
}

impl From<ProgramExtTableColumn> for usize {
    fn from(column: ProgramExtTableColumn) -> Self {
        ProgramExtTableColumn::iter()
            .enumerate()
            .find(|&(_n, col)| column == col)
            .map(|(n, _col)| n + ProgramBaseTableColumn::COUNT)
            .unwrap()
    }
}

impl Bounded for ProgramExtTableColumn {
    fn min_value() -> Self {
        ProgramExtTableColumn::iter().next().unwrap()
    }

    fn max_value() -> Self {
        ProgramExtTableColumn::iter().last().unwrap()
    }
}

// --------------------------------------------------------------------

#[derive(Display, Debug, Clone, Copy, PartialEq, Eq, EnumIter, EnumCountMacro)]
pub enum InstructionBaseTableColumn {
    Address,
    CI,
    NIA,
    IsPadding,
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

impl Bounded for InstructionBaseTableColumn {
    fn min_value() -> Self {
        InstructionBaseTableColumn::iter().next().unwrap()
    }

    fn max_value() -> Self {
        InstructionBaseTableColumn::iter().last().unwrap()
    }
}

#[derive(Display, Debug, Clone, Copy, PartialEq, Eq, EnumIter, EnumCountMacro)]
pub enum InstructionExtTableColumn {
    RunningProductPermArg,
    RunningEvaluation,
}

impl From<InstructionExtTableColumn> for usize {
    fn from(column: InstructionExtTableColumn) -> Self {
        InstructionExtTableColumn::iter()
            .enumerate()
            .find(|&(_n, col)| column == col)
            .map(|(n, _col)| n + InstructionBaseTableColumn::COUNT)
            .unwrap()
    }
}

impl Bounded for InstructionExtTableColumn {
    fn min_value() -> Self {
        InstructionExtTableColumn::iter().next().unwrap()
    }

    fn max_value() -> Self {
        InstructionExtTableColumn::iter().last().unwrap()
    }
}

// --------------------------------------------------------------------

#[derive(Display, Debug, Clone, Copy, PartialEq, Eq, EnumIter, EnumCountMacro)]
pub enum OpStackBaseTableColumn {
    CLK,
    IB1ShrinkStack,
    OSP,
    OSV,
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

impl Bounded for OpStackBaseTableColumn {
    fn min_value() -> Self {
        OpStackBaseTableColumn::iter().next().unwrap()
    }

    fn max_value() -> Self {
        OpStackBaseTableColumn::iter().last().unwrap()
    }
}

#[derive(Display, Debug, Clone, Copy, PartialEq, Eq, EnumIter, EnumCountMacro)]
pub enum OpStackExtTableColumn {
    RunningProductPermArg,
}

impl From<OpStackExtTableColumn> for usize {
    fn from(column: OpStackExtTableColumn) -> Self {
        OpStackExtTableColumn::iter()
            .enumerate()
            .find(|&(_n, col)| column == col)
            .map(|(n, _col)| n + OpStackBaseTableColumn::COUNT)
            .unwrap()
    }
}

impl Bounded for OpStackExtTableColumn {
    fn min_value() -> Self {
        OpStackExtTableColumn::iter().next().unwrap()
    }

    fn max_value() -> Self {
        OpStackExtTableColumn::iter().last().unwrap()
    }
}

// --------------------------------------------------------------------

#[derive(Display, Debug, Clone, Copy, PartialEq, Eq, EnumIter, EnumCountMacro)]
pub enum RamBaseTableColumn {
    CLK,
    RAMP,
    RAMV,
    InverseOfRampDifference,
    BezoutCoefficientPolynomialCoefficient0,
    BezoutCoefficientPolynomialCoefficient1,
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

impl Bounded for RamBaseTableColumn {
    fn min_value() -> Self {
        RamBaseTableColumn::iter().next().unwrap()
    }

    fn max_value() -> Self {
        RamBaseTableColumn::iter().last().unwrap()
    }
}

#[derive(Display, Debug, Clone, Copy, PartialEq, Eq, EnumIter, EnumCountMacro)]
pub enum RamExtTableColumn {
    RunningProductOfRAMP,
    FormalDerivative,
    BezoutCoefficient0,
    BezoutCoefficient1,
    RunningProductPermArg,
}

impl From<RamExtTableColumn> for usize {
    fn from(column: RamExtTableColumn) -> Self {
        RamExtTableColumn::iter()
            .enumerate()
            .find(|&(_n, col)| column == col)
            .map(|(n, _col)| n + RamBaseTableColumn::COUNT)
            .unwrap()
    }
}

impl Bounded for RamExtTableColumn {
    fn min_value() -> Self {
        RamExtTableColumn::iter().next().unwrap()
    }

    fn max_value() -> Self {
        RamExtTableColumn::iter().last().unwrap()
    }
}

// --------------------------------------------------------------------

#[derive(Display, Debug, Clone, Copy, PartialEq, Eq, EnumIter, EnumCountMacro)]
pub enum JumpStackBaseTableColumn {
    CLK,
    CI,
    JSP,
    JSO,
    JSD,
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

impl Bounded for JumpStackBaseTableColumn {
    fn min_value() -> Self {
        JumpStackBaseTableColumn::iter().next().unwrap()
    }

    fn max_value() -> Self {
        JumpStackBaseTableColumn::iter().last().unwrap()
    }
}

#[derive(Display, Debug, Clone, Copy, PartialEq, Eq, EnumIter, EnumCountMacro)]
pub enum JumpStackExtTableColumn {
    RunningProductPermArg,
}

impl From<JumpStackExtTableColumn> for usize {
    fn from(column: JumpStackExtTableColumn) -> Self {
        JumpStackExtTableColumn::iter()
            .enumerate()
            .find(|&(_n, col)| column == col)
            .map(|(n, _col)| n + JumpStackBaseTableColumn::COUNT)
            .unwrap()
    }
}

impl Bounded for JumpStackExtTableColumn {
    fn min_value() -> Self {
        JumpStackExtTableColumn::iter().next().unwrap()
    }

    fn max_value() -> Self {
        JumpStackExtTableColumn::iter().last().unwrap()
    }
}

// --------------------------------------------------------------------

#[derive(Display, Debug, Clone, Copy, PartialEq, Eq, EnumIter, EnumCountMacro)]
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

impl From<HashBaseTableColumn> for usize {
    fn from(column: HashBaseTableColumn) -> Self {
        HashBaseTableColumn::iter()
            .enumerate()
            .find(|&(_n, col)| column == col)
            .map(|(n, _col)| n)
            .unwrap()
    }
}

impl TryFrom<usize> for HashBaseTableColumn {
    type Error = String;

    fn try_from(idx: usize) -> Result<Self, Self::Error> {
        HashBaseTableColumn::iter()
            .get(idx)
            .ok_or_else(|| format!("Column index {} out of bounds", idx))
    }
}

impl Bounded for HashBaseTableColumn {
    fn min_value() -> Self {
        HashBaseTableColumn::iter().next().unwrap()
    }

    fn max_value() -> Self {
        HashBaseTableColumn::iter().last().unwrap()
    }
}

#[derive(Display, Debug, Clone, Copy, PartialEq, Eq, EnumIter, EnumCountMacro)]
pub enum HashExtTableColumn {
    ToProcessorRunningEvaluation,
    FromProcessorRunningEvaluation,
}

impl From<HashExtTableColumn> for usize {
    fn from(column: HashExtTableColumn) -> Self {
        HashExtTableColumn::iter()
            .enumerate()
            .find(|&(_n, col)| column == col)
            .map(|(n, _col)| n + HashBaseTableColumn::COUNT)
            .unwrap()
    }
}

impl Bounded for HashExtTableColumn {
    fn min_value() -> Self {
        HashExtTableColumn::iter().next().unwrap()
    }

    fn max_value() -> Self {
        HashExtTableColumn::iter().last().unwrap()
    }
}

#[cfg(test)]
mod table_column_tests {
    use crate::table::{
        hash_table, instruction_table, jump_stack_table, op_stack_table, processor_table,
        program_table, ram_table,
    };

    use super::*;

    struct TestCase<'a> {
        base_width: usize,
        full_width: usize,
        max_base_column: usize,
        max_ext_column: usize,
        table_name: &'a str,
    }

    impl<'a> TestCase<'a> {
        pub fn new(
            base_width: usize,
            full_width: usize,
            max_base_column: usize,
            max_ext_column: usize,
            table_name: &'a str,
        ) -> Self {
            TestCase {
                base_width,
                full_width,
                max_base_column,
                max_ext_column,
                table_name,
            }
        }
    }

    #[test]
    fn column_max_bound_matches_table_width() {
        let cases: Vec<TestCase> = vec![
            TestCase::new(
                program_table::BASE_WIDTH,
                program_table::FULL_WIDTH,
                ProgramBaseTableColumn::max_value().into(),
                ProgramExtTableColumn::max_value().into(),
                "ProgramTable",
            ),
            TestCase::new(
                instruction_table::BASE_WIDTH,
                instruction_table::FULL_WIDTH,
                InstructionBaseTableColumn::max_value().into(),
                InstructionExtTableColumn::max_value().into(),
                "InstructionTable",
            ),
            TestCase::new(
                processor_table::BASE_WIDTH,
                processor_table::FULL_WIDTH,
                ProcessorBaseTableColumn::max_value().into(),
                ProcessorExtTableColumn::max_value().into(),
                "ProcessorTable",
            ),
            TestCase::new(
                op_stack_table::BASE_WIDTH,
                op_stack_table::FULL_WIDTH,
                OpStackBaseTableColumn::max_value().into(),
                OpStackExtTableColumn::max_value().into(),
                "OpStackTable",
            ),
            TestCase::new(
                ram_table::BASE_WIDTH,
                ram_table::FULL_WIDTH,
                RamBaseTableColumn::max_value().into(),
                RamExtTableColumn::max_value().into(),
                "RamTable",
            ),
            TestCase::new(
                jump_stack_table::BASE_WIDTH,
                jump_stack_table::FULL_WIDTH,
                JumpStackBaseTableColumn::max_value().into(),
                JumpStackExtTableColumn::max_value().into(),
                "JumpStackTable",
            ),
            TestCase::new(
                hash_table::BASE_WIDTH,
                hash_table::FULL_WIDTH,
                HashBaseTableColumn::max_value().into(),
                HashExtTableColumn::max_value().into(),
                "HashTable",
            ),
        ];

        for case in cases.iter() {
            assert_eq!(
                case.base_width,
                case.max_base_column + 1,
                "{}'s BASE_WIDTH is 1 + its max column index",
                case.table_name
            );

            assert_eq!(
                case.full_width,
                case.max_ext_column + 1,
                "Ext{}'s FULL_WIDTH is 1 + its max ext column index",
                case.table_name
            );
        }
    }
}
