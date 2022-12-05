use std::fmt::Display;
use std::fmt::Formatter;

use itertools::Itertools;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::instruction::AnInstruction::*;
use crate::table::hash_table;
use crate::table::jump_stack_table;
use crate::table::processor_table;
use crate::table::table_column::JumpStackBaseTableColumn;
use crate::table::table_column::ProcessorBaseTableColumn::*;
use crate::table::table_column::ProcessorExtTableColumn;
use crate::table::table_column::ProcessorExtTableColumn::*;

// todo: clean up this file – AET might belong in a separate file, the Display impl's probably also.

#[derive(Debug, Clone, Default)]
pub struct AlgebraicExecutionTrace {
    pub processor_matrix: Vec<[BFieldElement; processor_table::BASE_WIDTH]>,
    pub hash_matrix: Vec<[BFieldElement; hash_table::BASE_WIDTH]>,
}

pub struct ProcessorMatrixRow {
    pub row: [BFieldElement; processor_table::BASE_WIDTH],
}

impl Display for ProcessorMatrixRow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        fn row(f: &mut std::fmt::Formatter<'_>, s: String) -> std::fmt::Result {
            writeln!(f, "│ {: <103} │", s)
        }

        fn row_blank(f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            row(f, "".into())
        }

        let instruction = self.row[usize::from(CI)].value().try_into().unwrap();
        let instruction_with_arg = match instruction {
            Push(_) => Push(self.row[usize::from(NIA)]),
            Call(_) => Call(self.row[usize::from(NIA)]),
            Dup(_) => Dup((self.row[usize::from(NIA)].value() as u32)
                .try_into()
                .unwrap()),
            Swap(_) => Swap(
                (self.row[usize::from(NIA)].value() as u32)
                    .try_into()
                    .unwrap(),
            ),
            _ => instruction,
        };

        writeln!(f, " ╭───────────────────────────╮")?;
        writeln!(f, " │ {: <25} │", format!("{}", instruction_with_arg))?;
        writeln!(
            f,
            "╭┴───────────────────────────┴────────────────────────────────────\
            ────────────────────┬───────────────────╮"
        )?;

        let width = 20;
        row(
            f,
            format!(
                "ip:   {:>width$} ╷ ci:   {:>width$} ╷ nia: {:>width$} │ {:>17}",
                self.row[usize::from(IP)].value(),
                self.row[usize::from(CI)].value(),
                self.row[usize::from(NIA)].value(),
                self.row[usize::from(CLK)].value(),
            ),
        )?;

        writeln!(
            f,
            "│ jsp:  {:>width$} │ jso:  {:>width$} │ jsd: {:>width$} ╰───────────────────┤",
            self.row[usize::from(JSP)].value(),
            self.row[usize::from(JSO)].value(),
            self.row[usize::from(JSD)].value(),
        )?;
        row(
            f,
            format!(
                "ramp: {:>width$} │ ramv: {:>width$} │",
                self.row[usize::from(RAMP)].value(),
                self.row[usize::from(RAMV)].value(),
            ),
        )?;
        row(
            f,
            format!(
                "osp:  {:>width$} │ osv:  {:>width$} ╵",
                self.row[usize::from(OSP)].value(),
                self.row[usize::from(OSV)].value(),
            ),
        )?;

        row_blank(f)?;

        row(
            f,
            format!(
                "st0-3:    [ {:>width$} | {:>width$} | {:>width$} | {:>width$} ]",
                self.row[usize::from(ST0)].value(),
                self.row[usize::from(ST1)].value(),
                self.row[usize::from(ST2)].value(),
                self.row[usize::from(ST3)].value(),
            ),
        )?;
        row(
            f,
            format!(
                "st4-7:    [ {:>width$} | {:>width$} | {:>width$} | {:>width$} ]",
                self.row[usize::from(ST4)].value(),
                self.row[usize::from(ST5)].value(),
                self.row[usize::from(ST6)].value(),
                self.row[usize::from(ST7)].value(),
            ),
        )?;
        row(
            f,
            format!(
                "st8-11:   [ {:>width$} | {:>width$} | {:>width$} | {:>width$} ]",
                self.row[usize::from(ST8)].value(),
                self.row[usize::from(ST9)].value(),
                self.row[usize::from(ST10)].value(),
                self.row[usize::from(ST11)].value(),
            ),
        )?;
        row(
            f,
            format!(
                "st12-15:  [ {:>width$} | {:>width$} | {:>width$} | {:>width$} ]",
                self.row[usize::from(ST12)].value(),
                self.row[usize::from(ST13)].value(),
                self.row[usize::from(ST14)].value(),
                self.row[usize::from(ST15)].value(),
            ),
        )?;

        row_blank(f)?;

        row(
            f,
            format!(
                "hv0-3:    [ {:>width$} | {:>width$} | {:>width$} | {:>width$} ]",
                self.row[usize::from(HV0)].value(),
                self.row[usize::from(HV1)].value(),
                self.row[usize::from(HV2)].value(),
                self.row[usize::from(HV3)].value(),
            ),
        )?;
        let w = 2;
        row(
            f,
            format!(
                "ib0-6:    \
                [ {:>w$} | {:>w$} | {:>w$} | {:>w$} | {:>w$} | {:>w$} | {:>w$} ]",
                self.row[usize::from(IB0)].value(),
                self.row[usize::from(IB1)].value(),
                self.row[usize::from(IB2)].value(),
                self.row[usize::from(IB3)].value(),
                self.row[usize::from(IB4)].value(),
                self.row[usize::from(IB5)].value(),
                self.row[usize::from(IB6)].value(),
            ),
        )?;
        write!(
            f,
            "╰─────────────────────────────────────────────────────────────────\
            ────────────────────────────────────────╯"
        )
    }
}

pub struct ExtProcessorMatrixRow {
    pub row: [XFieldElement; processor_table::FULL_WIDTH],
}

impl Display for ExtProcessorMatrixRow {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let base_row = self.row[0..processor_table::BASE_WIDTH]
            .iter()
            .map(|xfe| xfe.unlift().unwrap())
            .collect_vec()
            .try_into()
            .unwrap();
        let base_row = ProcessorMatrixRow { row: base_row };

        let row = |form: &mut std::fmt::Formatter<'_>,
                   desc: &str,
                   col: ProcessorExtTableColumn|
         -> std::fmt::Result {
            // without the extra `format!()`, alignment in `writeln!()` fails
            let formatted_col_elem = format!("{}", self.row[usize::from(col)]);
            writeln!(form, "     │ {: <18}  {:>73} │", desc, formatted_col_elem,)
        };

        writeln!(f, "{}", base_row)?;
        writeln!(
            f,
            "     ╭───────────────────────────────────────────────────────\
            ────────────────────────────────────────╮"
        )?;
        row(f, "input_table_ea", InputTableEvalArg)?;
        row(f, "output_table_ea", OutputTableEvalArg)?;
        row(f, "instr_table_pa", InstructionTablePermArg)?;
        row(f, "opstack_table_pa", OpStackTablePermArg)?;
        row(f, "ram_table_pa", RamTablePermArg)?;
        row(f, "jumpstack_table_pa", JumpStackTablePermArg)?;
        row(f, "to_hash_table_ea", ToHashTableEvalArg)?;
        row(f, "from_hash_table_ea", FromHashTableEvalArg)?;
        write!(
            f,
            "     ╰───────────────────────────────────────────────────────\
            ────────────────────────────────────────╯"
        )
    }
}

pub struct JumpStackMatrixRow {
    pub row: [BFieldElement; jump_stack_table::BASE_WIDTH],
}

impl Display for JumpStackMatrixRow {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let width = 5;
        write!(
            f,
            "│ CLK: {:>width$} │ CI:  {:>width$} │ \
            JSP: {:>width$} │ JSO: {:>width$} │ JSD: {:>width$} │",
            self.row[usize::from(JumpStackBaseTableColumn::CLK)].value(),
            self.row[usize::from(JumpStackBaseTableColumn::CI)].value(),
            self.row[usize::from(JumpStackBaseTableColumn::JSP)].value(),
            self.row[usize::from(JumpStackBaseTableColumn::JSO)].value(),
            self.row[usize::from(JumpStackBaseTableColumn::JSD)].value(),
        )
    }
}
