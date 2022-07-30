//! Enums that convert table column names into `usize` indices
//!
//! These let one address a given column by its name rather than its arbitrary index.

// --------------------------------------------------------------------

use num_traits::Bounded;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, Copy)]
pub enum ProcessorTableColumn {
    CLK,
    IP,
    CI,
    NIA,
    IB0,
    IB1,
    IB2,
    IB3,
    IB4,
    IB5,
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

impl Display for ProcessorTableColumn {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        use ProcessorTableColumn::*;

        match self {
            CLK => write!(f, "CLK"),
            IP => write!(f, "IP"),
            CI => write!(f, "CI"),
            NIA => write!(f, "NIA"),
            IB0 => write!(f, "IB0"),
            IB1 => write!(f, "IB1"),
            IB2 => write!(f, "IB2"),
            IB3 => write!(f, "IB3"),
            IB4 => write!(f, "IB4"),
            IB5 => write!(f, "IB5"),
            JSP => write!(f, "JSP"),
            JSO => write!(f, "JSO"),
            JSD => write!(f, "JSD"),
            ST0 => write!(f, "ST0"),
            ST1 => write!(f, "ST1"),
            ST2 => write!(f, "ST2"),
            ST3 => write!(f, "ST3"),
            ST4 => write!(f, "ST4"),
            ST5 => write!(f, "ST5"),
            ST6 => write!(f, "ST6"),
            ST7 => write!(f, "ST7"),
            ST8 => write!(f, "ST8"),
            ST9 => write!(f, "ST9"),
            ST10 => write!(f, "ST10"),
            ST11 => write!(f, "ST11"),
            ST12 => write!(f, "ST12"),
            ST13 => write!(f, "ST13"),
            ST14 => write!(f, "ST14"),
            ST15 => write!(f, "ST15"),
            OSP => write!(f, "OSP"),
            OSV => write!(f, "OSV"),
            HV0 => write!(f, "HV0"),
            HV1 => write!(f, "HV1"),
            HV2 => write!(f, "HV2"),
            HV3 => write!(f, "HV3"),
            RAMV => write!(f, "RAMV"),
        }
    }
}

impl From<ProcessorTableColumn> for usize {
    fn from(c: ProcessorTableColumn) -> Self {
        use ProcessorTableColumn::*;

        match c {
            CLK => 0,
            IP => 1,
            CI => 2,
            NIA => 3,
            IB0 => 4,
            IB1 => 5,
            IB2 => 6,
            IB3 => 7,
            IB4 => 8,
            IB5 => 9,
            JSP => 10,
            JSO => 11,
            JSD => 12,
            ST0 => 13,
            ST1 => 14,
            ST2 => 15,
            ST3 => 16,
            ST4 => 17,
            ST5 => 18,
            ST6 => 19,
            ST7 => 20,
            ST8 => 21,
            ST9 => 22,
            ST10 => 23,
            ST11 => 24,
            ST12 => 25,
            ST13 => 26,
            ST14 => 27,
            ST15 => 28,
            OSP => 29,
            OSV => 30,
            HV0 => 31,
            HV1 => 32,
            HV2 => 33,
            HV3 => 34,
            RAMV => 35,
        }
    }
}

impl Bounded for ProcessorTableColumn {
    fn min_value() -> Self {
        ProcessorTableColumn::CLK
    }

    fn max_value() -> Self {
        ProcessorTableColumn::RAMV
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ExtProcessorTableColumn {
    BaseColumn(ProcessorTableColumn),

    InputTableEvalArg,
    OutputTableEvalArg,
    CompressedRowInstructionTable,
    InstructionTablePermArg,
    CompressedRowOpStackTable,
    OpStackTablePermArg,
    CompressedRowRamTable,
    RamTablePermArg,
    CompressedRowJumpStackTable,
    JumpStackTablePermArg,

    CompressedRowForHashInput,
    ToHashTableEvalArg,
    CompressedRowForHashDigest,
    FromHashTableEvalArg,

    CompressedRowLtU32Op,
    LtU32OpTablePermArg,
    CompressedRowAndU32Op,
    AndU32OpTablePermArg,
    CompressedRowXorU32Op,
    XorU32OpTablePermArg,
    CompressedRowReverseU32Op,
    ReverseU32OpTablePermArg,
    CompressedRowDivU32Op,
    DivU32OpTablePermArg,
}

impl std::fmt::Display for ExtProcessorTableColumn {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use ExtProcessorTableColumn::*;

        match self {
            BaseColumn(base_column) => write!(f, "{}", *base_column),
            InputTableEvalArg => write!(f, "InputTableEvalArg"),
            OutputTableEvalArg => write!(f, "OutputTableEvalArg"),
            CompressedRowInstructionTable => write!(f, "CompressedRowInstructionTable"),
            InstructionTablePermArg => write!(f, "InstructionTablePermArg"),
            CompressedRowOpStackTable => write!(f, "CompressedRowOpStackTable"),
            OpStackTablePermArg => write!(f, "OpStackTablePermArg"),
            CompressedRowRamTable => write!(f, "CompressedRowRamTable"),
            RamTablePermArg => write!(f, "RamTablePermArg"),
            CompressedRowJumpStackTable => write!(f, "CompressedRowJumpStackTable"),
            JumpStackTablePermArg => write!(f, "JumpStackTablePermArg"),
            CompressedRowForHashInput => write!(f, "CompressedRowForHashInput"),
            ToHashTableEvalArg => write!(f, "ToHashTableEvalArg"),
            CompressedRowForHashDigest => write!(f, "CompressedRowForHashDigest"),
            FromHashTableEvalArg => write!(f, "FromHashTableEvalArg"),
            CompressedRowLtU32Op => write!(f, "CompressedRowLtU32Op"),
            LtU32OpTablePermArg => write!(f, "LtU32OpTablePermArg"),
            CompressedRowAndU32Op => write!(f, "CompressedRowAndU32Op"),
            AndU32OpTablePermArg => write!(f, "AndU32OpTablePermArg"),
            CompressedRowXorU32Op => write!(f, "CompressedRowXorU32Op"),
            XorU32OpTablePermArg => write!(f, "XorU32OpTablePermArg"),
            CompressedRowReverseU32Op => write!(f, "CompressedRowReverseU32Op"),
            ReverseU32OpTablePermArg => write!(f, "ReverseU32OpTablePermArg"),
            CompressedRowDivU32Op => write!(f, "CompressedRowDivU32Op"),
            DivU32OpTablePermArg => write!(f, "DivU32OpTablePermArg"),
        }
    }
}

impl From<ExtProcessorTableColumn> for usize {
    fn from(c: ExtProcessorTableColumn) -> Self {
        use ExtProcessorTableColumn::*;

        match c {
            BaseColumn(base_column) => base_column.into(),
            InputTableEvalArg => 36,
            OutputTableEvalArg => 37,
            CompressedRowInstructionTable => 38,
            InstructionTablePermArg => 39,
            CompressedRowOpStackTable => 40,
            OpStackTablePermArg => 41,
            CompressedRowRamTable => 42,
            RamTablePermArg => 43,
            CompressedRowJumpStackTable => 44,
            JumpStackTablePermArg => 45,
            CompressedRowForHashInput => 46,
            ToHashTableEvalArg => 47,
            CompressedRowForHashDigest => 48,
            FromHashTableEvalArg => 49,
            CompressedRowLtU32Op => 50,
            LtU32OpTablePermArg => 51,
            CompressedRowAndU32Op => 52,
            AndU32OpTablePermArg => 53,
            CompressedRowXorU32Op => 54,
            XorU32OpTablePermArg => 55,
            CompressedRowReverseU32Op => 56,
            ReverseU32OpTablePermArg => 57,
            CompressedRowDivU32Op => 58,
            DivU32OpTablePermArg => 59,
        }
    }
}

impl Bounded for ExtProcessorTableColumn {
    fn min_value() -> Self {
        ExtProcessorTableColumn::BaseColumn(ProcessorTableColumn::min_value())
    }

    fn max_value() -> Self {
        ExtProcessorTableColumn::DivU32OpTablePermArg
    }
}

// --------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub enum ProgramTableColumn {
    Address,
    Instruction,
}

impl From<ProgramTableColumn> for usize {
    fn from(c: ProgramTableColumn) -> Self {
        use ProgramTableColumn::*;

        match c {
            Address => 0,
            Instruction => 1,
        }
    }
}

impl Bounded for ProgramTableColumn {
    fn min_value() -> Self {
        ProgramTableColumn::Address
    }

    fn max_value() -> Self {
        ProgramTableColumn::Instruction
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ExtProgramTableColumn {
    BaseColumn(ProgramTableColumn),
    EvalArgCompressedRow,
    EvalArgRunningSum,
}

impl From<ExtProgramTableColumn> for usize {
    fn from(c: ExtProgramTableColumn) -> Self {
        use ExtProgramTableColumn::*;

        match c {
            BaseColumn(base_column) => base_column.into(),
            EvalArgCompressedRow => 2,
            EvalArgRunningSum => 3,
        }
    }
}

impl Bounded for ExtProgramTableColumn {
    fn min_value() -> Self {
        ExtProgramTableColumn::BaseColumn(ProgramTableColumn::min_value())
    }

    fn max_value() -> Self {
        ExtProgramTableColumn::EvalArgRunningSum
    }
}

// --------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub enum InstructionTableColumn {
    Address,
    CI,
    NIA,
}

impl From<InstructionTableColumn> for usize {
    fn from(c: InstructionTableColumn) -> Self {
        use InstructionTableColumn::*;

        match c {
            Address => 0,
            CI => 1,
            NIA => 2,
        }
    }
}

impl Bounded for InstructionTableColumn {
    fn min_value() -> Self {
        InstructionTableColumn::Address
    }

    fn max_value() -> Self {
        InstructionTableColumn::NIA
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ExtInstructionTableColumn {
    BaseColumn(InstructionTableColumn),
    CompressedRowPermArg,
    RunningProductPermArg,
    CompressedRowEvalArg,
    RunningSumEvalArg,
}

impl From<ExtInstructionTableColumn> for usize {
    fn from(c: ExtInstructionTableColumn) -> Self {
        use ExtInstructionTableColumn::*;

        match c {
            BaseColumn(base_column) => base_column.into(),
            CompressedRowPermArg => 3,
            RunningProductPermArg => 4,
            CompressedRowEvalArg => 5,
            RunningSumEvalArg => 6,
        }
    }
}

impl Bounded for ExtInstructionTableColumn {
    fn min_value() -> Self {
        ExtInstructionTableColumn::BaseColumn(InstructionTableColumn::min_value())
    }

    fn max_value() -> Self {
        ExtInstructionTableColumn::RunningSumEvalArg
    }
}

// --------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub enum OpStackTableColumn {
    CLK,
    CI,
    OSV,
    OSP,
}

impl From<OpStackTableColumn> for usize {
    fn from(c: OpStackTableColumn) -> Self {
        use OpStackTableColumn::*;

        match c {
            CLK => 0,
            CI => 1,
            OSV => 2,
            OSP => 3,
        }
    }
}

impl Bounded for OpStackTableColumn {
    fn min_value() -> Self {
        OpStackTableColumn::CLK
    }

    fn max_value() -> Self {
        OpStackTableColumn::OSP
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ExtOpStackTableColumn {
    BaseColumn(OpStackTableColumn),
    PermArgCompressedRow,
    RunningProductPermArg,
}

impl From<ExtOpStackTableColumn> for usize {
    fn from(c: ExtOpStackTableColumn) -> Self {
        use ExtOpStackTableColumn::*;

        match c {
            BaseColumn(base_column) => base_column.into(),
            PermArgCompressedRow => 4,
            RunningProductPermArg => 5,
        }
    }
}

impl Bounded for ExtOpStackTableColumn {
    fn min_value() -> Self {
        ExtOpStackTableColumn::BaseColumn(OpStackTableColumn::min_value())
    }

    fn max_value() -> Self {
        ExtOpStackTableColumn::RunningProductPermArg
    }
}

// --------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub enum RamTableColumn {
    CLK,
    RAMP,
    RAMV,
    InverseOfRampDifference,
}

impl From<RamTableColumn> for usize {
    fn from(c: RamTableColumn) -> Self {
        use RamTableColumn::*;

        match c {
            CLK => 0,
            RAMP => 1,
            RAMV => 2,
            InverseOfRampDifference => 3,
        }
    }
}

impl Bounded for RamTableColumn {
    fn min_value() -> Self {
        RamTableColumn::CLK
    }

    fn max_value() -> Self {
        RamTableColumn::InverseOfRampDifference
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ExtRamTableColumn {
    BaseColumn(RamTableColumn),
    PermArgCompressedRow,
    RunningProductPermArg,
}

impl From<ExtRamTableColumn> for usize {
    fn from(c: ExtRamTableColumn) -> Self {
        use ExtRamTableColumn::*;

        match c {
            BaseColumn(base_column) => base_column.into(),
            PermArgCompressedRow => 4,
            RunningProductPermArg => 5,
        }
    }
}

impl Bounded for ExtRamTableColumn {
    fn min_value() -> Self {
        ExtRamTableColumn::BaseColumn(RamTableColumn::min_value())
    }

    fn max_value() -> Self {
        ExtRamTableColumn::RunningProductPermArg
    }
}

// --------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub enum JumpStackTableColumn {
    CLK,
    CI,
    JSP,
    JSO,
    JSD,
}

impl From<JumpStackTableColumn> for usize {
    fn from(c: JumpStackTableColumn) -> Self {
        use JumpStackTableColumn::*;

        match c {
            CLK => 0,
            CI => 1,
            JSP => 2,
            JSO => 3,
            JSD => 4,
        }
    }
}

impl Bounded for JumpStackTableColumn {
    fn min_value() -> Self {
        JumpStackTableColumn::CLK
    }

    fn max_value() -> Self {
        JumpStackTableColumn::JSD
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ExtJumpStackTableColumn {
    BaseColumn(JumpStackTableColumn),
    PermArgCompressedRow,
    RunningProductPermArg,
}

impl From<ExtJumpStackTableColumn> for usize {
    fn from(c: ExtJumpStackTableColumn) -> Self {
        use ExtJumpStackTableColumn::*;

        match c {
            BaseColumn(base_column) => base_column.into(),
            PermArgCompressedRow => 5,
            RunningProductPermArg => 6,
        }
    }
}

impl Bounded for ExtJumpStackTableColumn {
    fn min_value() -> Self {
        ExtJumpStackTableColumn::BaseColumn(JumpStackTableColumn::min_value())
    }

    fn max_value() -> Self {
        ExtJumpStackTableColumn::RunningProductPermArg
    }
}

// --------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub enum HashTableColumn {
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
}

impl From<HashTableColumn> for usize {
    fn from(c: HashTableColumn) -> Self {
        use HashTableColumn::*;

        match c {
            ROUNDNUMBER => 0,
            STATE0 => 1,
            STATE1 => 2,
            STATE2 => 3,
            STATE3 => 4,
            STATE4 => 5,
            STATE5 => 6,
            STATE6 => 7,
            STATE7 => 8,
            STATE8 => 9,
            STATE9 => 10,
            STATE10 => 11,
            STATE11 => 12,
            STATE12 => 13,
            STATE13 => 14,
            STATE14 => 15,
            STATE15 => 16,
        }
    }
}

impl Bounded for HashTableColumn {
    fn min_value() -> Self {
        HashTableColumn::ROUNDNUMBER
    }

    fn max_value() -> Self {
        HashTableColumn::STATE15
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ExtHashTableColumn {
    BaseColumn(HashTableColumn),
    CompressedStateForInput,
    ToProcessorRunningSum,
    CompressedStateForOutput,
    FromProcessorRunningSum,
}

impl From<ExtHashTableColumn> for usize {
    fn from(c: ExtHashTableColumn) -> Self {
        use ExtHashTableColumn::*;

        match c {
            BaseColumn(base_column) => base_column.into(),
            CompressedStateForInput => 17,
            ToProcessorRunningSum => 18,
            CompressedStateForOutput => 19,
            FromProcessorRunningSum => 20,
        }
    }
}

impl Bounded for ExtHashTableColumn {
    fn min_value() -> Self {
        ExtHashTableColumn::BaseColumn(HashTableColumn::min_value())
    }

    fn max_value() -> Self {
        ExtHashTableColumn::FromProcessorRunningSum
    }
}

// --------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub enum U32OpTableColumn {
    IDC,
    LHS,
    RHS,
    LT,
    AND,
    XOR,
    REV,
}

impl From<U32OpTableColumn> for usize {
    fn from(c: U32OpTableColumn) -> Self {
        use U32OpTableColumn::*;

        match c {
            IDC => 0,
            LHS => 1,
            RHS => 2,
            LT => 3,
            AND => 4,
            XOR => 5,
            REV => 6,
        }
    }
}

impl Bounded for U32OpTableColumn {
    fn min_value() -> Self {
        U32OpTableColumn::IDC
    }

    fn max_value() -> Self {
        U32OpTableColumn::REV
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ExtU32OpTableColumn {
    BaseColumn(U32OpTableColumn),
    LtCompressedRow,
    LtRunningProductPermArg,
    AndCompressedRow,
    AndRunningProductPermArg,
    XorCompressedRow,
    XorRunningProductPermArg,
    ReverseCompressedRow,
    ReverseRunningProductPermArg,
    DivCompressedRow,
    DivRunningProductPermArg,
}

impl From<ExtU32OpTableColumn> for usize {
    fn from(c: ExtU32OpTableColumn) -> Self {
        use ExtU32OpTableColumn::*;

        match c {
            BaseColumn(base_column) => base_column.into(),
            LtCompressedRow => 7,
            LtRunningProductPermArg => 8,
            AndCompressedRow => 9,
            AndRunningProductPermArg => 10,
            XorCompressedRow => 11,
            XorRunningProductPermArg => 12,
            ReverseCompressedRow => 13,
            ReverseRunningProductPermArg => 14,
            DivCompressedRow => 15,
            DivRunningProductPermArg => 16,
        }
    }
}

impl Bounded for ExtU32OpTableColumn {
    fn min_value() -> Self {
        ExtU32OpTableColumn::BaseColumn(U32OpTableColumn::min_value())
    }

    fn max_value() -> Self {
        ExtU32OpTableColumn::DivRunningProductPermArg
    }
}

#[cfg(test)]
mod table_column_tests {
    use crate::shared_math::stark::triton::table::{
        hash_table, instruction_table, jump_stack_table, op_stack_table, processor_table,
        program_table, ram_table, u32_op_table,
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
                ProgramTableColumn::max_value().into(),
                ExtProgramTableColumn::max_value().into(),
                "ProgramTable",
            ),
            TestCase::new(
                instruction_table::BASE_WIDTH,
                instruction_table::FULL_WIDTH,
                InstructionTableColumn::max_value().into(),
                ExtInstructionTableColumn::max_value().into(),
                "InstructionTable",
            ),
            TestCase::new(
                processor_table::BASE_WIDTH,
                processor_table::FULL_WIDTH,
                ProcessorTableColumn::max_value().into(),
                ExtProcessorTableColumn::max_value().into(),
                "ProcessorTable",
            ),
            TestCase::new(
                op_stack_table::BASE_WIDTH,
                op_stack_table::FULL_WIDTH,
                OpStackTableColumn::max_value().into(),
                ExtOpStackTableColumn::max_value().into(),
                "OpStackTable",
            ),
            TestCase::new(
                ram_table::BASE_WIDTH,
                ram_table::FULL_WIDTH,
                RamTableColumn::max_value().into(),
                ExtRamTableColumn::max_value().into(),
                "RamTable",
            ),
            TestCase::new(
                jump_stack_table::BASE_WIDTH,
                jump_stack_table::FULL_WIDTH,
                JumpStackTableColumn::max_value().into(),
                ExtJumpStackTableColumn::max_value().into(),
                "JumpStackTable",
            ),
            TestCase::new(
                hash_table::BASE_WIDTH,
                hash_table::FULL_WIDTH,
                HashTableColumn::max_value().into(),
                ExtHashTableColumn::max_value().into(),
                "HashTable",
            ),
            TestCase::new(
                u32_op_table::BASE_WIDTH,
                u32_op_table::FULL_WIDTH,
                U32OpTableColumn::max_value().into(),
                ExtU32OpTableColumn::max_value().into(),
                "U32OpTable",
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
