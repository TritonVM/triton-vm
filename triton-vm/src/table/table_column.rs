//! Enums that convert table column names into `usize` indices
//!
//! These let one address a given column by its name rather than its arbitrary index.

// --------------------------------------------------------------------

use num_traits::Bounded;
use std::fmt::{Display, Formatter};
use HashTableColumn::*;

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
            IB6 => write!(f, "IB6"),
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
            IB6 => 10,
            JSP => 11,
            JSO => 12,
            JSD => 13,
            ST0 => 14,
            ST1 => 15,
            ST2 => 16,
            ST3 => 17,
            ST4 => 18,
            ST5 => 19,
            ST6 => 20,
            ST7 => 21,
            ST8 => 22,
            ST9 => 23,
            ST10 => 24,
            ST11 => 25,
            ST12 => 26,
            ST13 => 27,
            ST14 => 28,
            ST15 => 29,
            OSP => 30,
            OSV => 31,
            HV0 => 32,
            HV1 => 33,
            HV2 => 34,
            HV3 => 35,
            RAMV => 36,
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

    CompressedRowForU32Op,
    U32OpTablePermArg,
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
            CompressedRowForU32Op => write!(f, "CompressedRowForU32Op"),
            U32OpTablePermArg => write!(f, "U32OpTablePermArg"),
        }
    }
}

impl From<ExtProcessorTableColumn> for usize {
    fn from(c: ExtProcessorTableColumn) -> Self {
        use ExtProcessorTableColumn::*;

        match c {
            BaseColumn(base_column) => base_column.into(),
            InputTableEvalArg => 37,
            OutputTableEvalArg => 38,
            CompressedRowInstructionTable => 39,
            InstructionTablePermArg => 40,
            CompressedRowOpStackTable => 41,
            OpStackTablePermArg => 42,
            CompressedRowRamTable => 43,
            RamTablePermArg => 44,
            CompressedRowJumpStackTable => 45,
            JumpStackTablePermArg => 46,
            CompressedRowForHashInput => 47,
            ToHashTableEvalArg => 48,
            CompressedRowForHashDigest => 49,
            FromHashTableEvalArg => 50,
            CompressedRowForU32Op => 51,
            U32OpTablePermArg => 52,
        }
    }
}

impl Bounded for ExtProcessorTableColumn {
    fn min_value() -> Self {
        ExtProcessorTableColumn::BaseColumn(ProcessorTableColumn::min_value())
    }

    fn max_value() -> Self {
        ExtProcessorTableColumn::U32OpTablePermArg
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
    IB1ShrinkStack,
    OSP,
    OSV,
}

impl From<OpStackTableColumn> for usize {
    fn from(c: OpStackTableColumn) -> Self {
        use OpStackTableColumn::*;

        match c {
            CLK => 0,
            IB1ShrinkStack => 1,
            OSP => 2,
            OSV => 3,
        }
    }
}

impl Bounded for OpStackTableColumn {
    fn min_value() -> Self {
        OpStackTableColumn::CLK
    }

    fn max_value() -> Self {
        OpStackTableColumn::OSV
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

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
            CONSTANT0A => 17,
            CONSTANT1A => 18,
            CONSTANT2A => 19,
            CONSTANT3A => 20,
            CONSTANT4A => 21,
            CONSTANT5A => 22,
            CONSTANT6A => 23,
            CONSTANT7A => 24,
            CONSTANT8A => 25,
            CONSTANT9A => 26,
            CONSTANT10A => 27,
            CONSTANT11A => 28,
            CONSTANT12A => 29,
            CONSTANT13A => 30,
            CONSTANT14A => 31,
            CONSTANT15A => 32,
            CONSTANT0B => 33,
            CONSTANT1B => 34,
            CONSTANT2B => 35,
            CONSTANT3B => 36,
            CONSTANT4B => 37,
            CONSTANT5B => 38,
            CONSTANT6B => 39,
            CONSTANT7B => 40,
            CONSTANT8B => 41,
            CONSTANT9B => 42,
            CONSTANT10B => 43,
            CONSTANT11B => 44,
            CONSTANT12B => 45,
            CONSTANT13B => 46,
            CONSTANT14B => 47,
            CONSTANT15B => 48,
        }
    }
}

impl From<usize> for HashTableColumn {
    fn from(idx: usize) -> Self {
        match idx {
            0 => ROUNDNUMBER,
            1 => STATE0,
            2 => STATE1,
            3 => STATE2,
            4 => STATE3,
            5 => STATE4,
            6 => STATE5,
            7 => STATE6,
            8 => STATE7,
            9 => STATE8,
            10 => STATE9,
            11 => STATE10,
            12 => STATE11,
            13 => STATE12,
            14 => STATE13,
            15 => STATE14,
            16 => STATE15,
            17 => CONSTANT0A,
            18 => CONSTANT1A,
            19 => CONSTANT2A,
            20 => CONSTANT3A,
            21 => CONSTANT4A,
            22 => CONSTANT5A,
            23 => CONSTANT6A,
            24 => CONSTANT7A,
            25 => CONSTANT8A,
            26 => CONSTANT9A,
            27 => CONSTANT10A,
            28 => CONSTANT11A,
            29 => CONSTANT12A,
            30 => CONSTANT13A,
            31 => CONSTANT14A,
            32 => CONSTANT15A,
            33 => CONSTANT0B,
            34 => CONSTANT1B,
            35 => CONSTANT2B,
            36 => CONSTANT3B,
            37 => CONSTANT4B,
            38 => CONSTANT5B,
            39 => CONSTANT6B,
            40 => CONSTANT7B,
            41 => CONSTANT8B,
            42 => CONSTANT9B,
            43 => CONSTANT10B,
            44 => CONSTANT11B,
            45 => CONSTANT12B,
            46 => CONSTANT13B,
            47 => CONSTANT14B,
            48 => CONSTANT15B,
            _ => panic!("No Hash Table column with index {idx} exists."),
        }
    }
}

impl Bounded for HashTableColumn {
    fn min_value() -> Self {
        HashTableColumn::ROUNDNUMBER
    }

    fn max_value() -> Self {
        HashTableColumn::CONSTANT15B
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
            CompressedStateForInput => 49,
            FromProcessorRunningSum => 50,
            CompressedStateForOutput => 51,
            ToProcessorRunningSum => 52,
        }
    }
}

impl Bounded for ExtHashTableColumn {
    fn min_value() -> Self {
        ExtHashTableColumn::BaseColumn(HashTableColumn::min_value())
    }

    fn max_value() -> Self {
        ExtHashTableColumn::ToProcessorRunningSum
    }
}

// --------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
pub enum U32OpTableColumn {
    IDC,
    Bits,
    Inv33MinusBits,
    CI,
    LHS,
    RHS,
    LT,
    AND,
    XOR,
    REV,
    LHSInv,
    RHSInv,
}

impl From<U32OpTableColumn> for usize {
    fn from(c: U32OpTableColumn) -> Self {
        use U32OpTableColumn::*;

        match c {
            IDC => 0,
            Bits => 1,
            Inv33MinusBits => 2,
            CI => 3,
            LHS => 4,
            RHS => 5,
            LT => 6,
            AND => 7,
            XOR => 8,
            REV => 9,
            LHSInv => 10,
            RHSInv => 11,
        }
    }
}

impl Bounded for U32OpTableColumn {
    fn min_value() -> Self {
        U32OpTableColumn::IDC
    }

    fn max_value() -> Self {
        U32OpTableColumn::RHSInv
    }
}

#[derive(Debug, Clone, Copy)]
pub enum ExtU32OpTableColumn {
    BaseColumn(U32OpTableColumn),
    CompressedRow,
    RunningProductPermArg,
}

impl From<ExtU32OpTableColumn> for usize {
    fn from(c: ExtU32OpTableColumn) -> Self {
        use ExtU32OpTableColumn::*;

        match c {
            BaseColumn(base_column) => base_column.into(),
            CompressedRow => 12,
            RunningProductPermArg => 13,
        }
    }
}

impl Bounded for ExtU32OpTableColumn {
    fn min_value() -> Self {
        ExtU32OpTableColumn::BaseColumn(U32OpTableColumn::min_value())
    }

    fn max_value() -> Self {
        ExtU32OpTableColumn::RunningProductPermArg
    }
}

#[cfg(test)]
mod table_column_tests {
    use crate::table::{
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
