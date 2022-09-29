use std::fmt::{Display, Formatter};

use itertools::Itertools;
use num_traits::Zero;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::traits::FiniteField;
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::instruction::AnInstruction::*;
use crate::table::table_column::ProcessorExtTableColumn::*;
use crate::table::table_column::{
    InstructionBaseTableColumn, OpStackBaseTableColumn, ProcessorBaseTableColumn,
    ProgramBaseTableColumn, RamBaseTableColumn,
};
use crate::vm::Program;

use super::table_column::{
    JumpStackBaseTableColumn, ProcessorBaseTableColumn::*, ProcessorExtTableColumn,
};
use super::{
    hash_table, instruction_table, jump_stack_table, op_stack_table, processor_table,
    program_table, ram_table,
};

#[derive(Debug, Clone, Default)]
pub struct AlgebraicExecutionTrace {
    pub processor_matrix: Vec<[BFieldElement; processor_table::BASE_WIDTH]>,
    pub hash_matrix: Vec<[BFieldElement; hash_table::BASE_WIDTH]>,
}

#[derive(Debug, Clone, Default)]
pub struct BaseMatrices {
    pub program_matrix: Vec<[BFieldElement; program_table::BASE_WIDTH]>,
    pub instruction_matrix: Vec<[BFieldElement; instruction_table::BASE_WIDTH]>,
    pub processor_matrix: Vec<[BFieldElement; processor_table::BASE_WIDTH]>,
    pub op_stack_matrix: Vec<[BFieldElement; op_stack_table::BASE_WIDTH]>,
    pub ram_matrix: Vec<[BFieldElement; ram_table::BASE_WIDTH]>,
    pub jump_stack_matrix: Vec<[BFieldElement; jump_stack_table::BASE_WIDTH]>,
    pub hash_matrix: Vec<[BFieldElement; hash_table::BASE_WIDTH]>,
}

impl BaseMatrices {
    pub fn new(aet: AlgebraicExecutionTrace, program: &Program) -> Self {
        Self {
            program_matrix: Self::derive_program_matrix(program),
            instruction_matrix: Self::derive_instruction_matrix(&aet, program),
            op_stack_matrix: Self::derive_op_stack_matrix(&aet),
            ram_matrix: Self::derive_ram_matrix(&aet),
            jump_stack_matrix: Self::derive_jump_stack_matrix(&aet),
            processor_matrix: aet.processor_matrix,
            hash_matrix: aet.hash_matrix,
        }
    }

    fn derive_program_matrix(program: &Program) -> Vec<[BFieldElement; program_table::BASE_WIDTH]> {
        program
            .to_bwords()
            .into_iter()
            .enumerate()
            .map(|(idx, instruction)| {
                let mut derived_row = [BFieldElement::zero(); program_table::BASE_WIDTH];
                derived_row[usize::from(ProgramBaseTableColumn::Address)] = (idx as u32).into();
                derived_row[usize::from(ProgramBaseTableColumn::Instruction)] = instruction;
                derived_row
            })
            .collect_vec()
    }

    fn derive_instruction_matrix(
        aet: &AlgebraicExecutionTrace,
        program: &Program,
    ) -> Vec<[BFieldElement; instruction_table::BASE_WIDTH]> {
        use InstructionBaseTableColumn::*;

        let program_append_0 = [program.to_bwords(), vec![BFieldElement::zero()]].concat();
        let program_part = program_append_0
            .into_iter()
            .tuple_windows()
            .enumerate()
            .map(|(idx, (instruction, next_instruction))| {
                let mut derived_row = [BFieldElement::zero(); instruction_table::BASE_WIDTH];
                derived_row[usize::from(Address)] = (idx as u32).into();
                derived_row[usize::from(CI)] = instruction;
                derived_row[usize::from(NIA)] = next_instruction;
                derived_row
            })
            .collect_vec();
        let processor_part = aet
            .processor_matrix
            .iter()
            .map(|&row| {
                let mut derived_row = [BFieldElement::zero(); instruction_table::BASE_WIDTH];
                derived_row[usize::from(Address)] = row[usize::from(ProcessorBaseTableColumn::IP)];
                derived_row[usize::from(CI)] = row[usize::from(ProcessorBaseTableColumn::CI)];
                derived_row[usize::from(NIA)] = row[usize::from(ProcessorBaseTableColumn::NIA)];
                derived_row
            })
            .collect_vec();
        let mut instruction_matrix = [program_part, processor_part].concat();
        instruction_matrix.sort_by_key(|row| row[usize::from(Address)].value());
        instruction_matrix
    }

    fn derive_op_stack_matrix(
        aet: &AlgebraicExecutionTrace,
    ) -> Vec<[BFieldElement; op_stack_table::BASE_WIDTH]> {
        use OpStackBaseTableColumn::*;

        let mut op_stack_matrix = aet
            .processor_matrix
            .iter()
            .map(|&row| {
                let mut derived_row = [BFieldElement::zero(); op_stack_table::BASE_WIDTH];
                derived_row[usize::from(CLK)] = row[usize::from(ProcessorBaseTableColumn::CLK)];
                derived_row[usize::from(IB1ShrinkStack)] =
                    row[usize::from(ProcessorBaseTableColumn::IB1)];
                derived_row[usize::from(OSP)] = row[usize::from(ProcessorBaseTableColumn::OSP)];
                derived_row[usize::from(OSV)] = row[usize::from(ProcessorBaseTableColumn::OSV)];
                derived_row
            })
            .collect_vec();
        op_stack_matrix
            .sort_by_key(|row| (row[usize::from(OSP)].value(), row[usize::from(CLK)].value()));
        op_stack_matrix
    }

    fn derive_ram_matrix(
        aet: &AlgebraicExecutionTrace,
    ) -> Vec<[BFieldElement; ram_table::BASE_WIDTH]> {
        let mut ram_matrix = aet
            .processor_matrix
            .iter()
            .map(|&row| {
                let mut derived_row = [BFieldElement::zero(); ram_table::BASE_WIDTH];
                derived_row[usize::from(RamBaseTableColumn::CLK)] = row[usize::from(CLK)];
                derived_row[usize::from(RamBaseTableColumn::RAMP)] = row[usize::from(ST1)];
                derived_row[usize::from(RamBaseTableColumn::RAMV)] = row[usize::from(RAMV)];
                derived_row[usize::from(RamBaseTableColumn::InverseOfRampDifference)] =
                    BFieldElement::zero();
                derived_row
            })
            .collect_vec();
        ram_matrix.sort_by_key(|row| {
            (
                row[usize::from(RamBaseTableColumn::RAMP)].value(),
                row[usize::from(RamBaseTableColumn::CLK)].value(),
            )
        });

        // calculate inverse of ramp difference
        let indexed_non_zero_differences = ram_matrix
            .iter()
            .tuple_windows()
            .enumerate()
            .map(|(idx, (curr_row, next_row))| {
                (
                    idx,
                    next_row[usize::from(RamBaseTableColumn::RAMP)]
                        - curr_row[usize::from(RamBaseTableColumn::RAMP)],
                )
            })
            .filter(|(_, x)| !BFieldElement::is_zero(x))
            .collect_vec();
        let inverses = BFieldElement::batch_inversion(
            indexed_non_zero_differences
                .iter()
                .map(|&(_, x)| x)
                .collect_vec(),
        );
        for ((idx, _), inverse) in indexed_non_zero_differences
            .into_iter()
            .zip_eq(inverses.into_iter())
        {
            ram_matrix[idx][usize::from(RamBaseTableColumn::InverseOfRampDifference)] = inverse;
        }
        ram_matrix
    }

    fn derive_jump_stack_matrix(
        aet: &AlgebraicExecutionTrace,
    ) -> Vec<[BFieldElement; jump_stack_table::BASE_WIDTH]> {
        let mut jump_stack_matrix = aet
            .processor_matrix
            .iter()
            .map(|&row| {
                let mut derived_row = [BFieldElement::zero(); jump_stack_table::BASE_WIDTH];
                derived_row[usize::from(JumpStackBaseTableColumn::CLK)] = row[usize::from(CLK)];
                derived_row[usize::from(JumpStackBaseTableColumn::CI)] = row[usize::from(CI)];
                derived_row[usize::from(JumpStackBaseTableColumn::JSP)] = row[usize::from(JSP)];
                derived_row[usize::from(JumpStackBaseTableColumn::JSO)] = row[usize::from(JSO)];
                derived_row[usize::from(JumpStackBaseTableColumn::JSD)] = row[usize::from(JSD)];
                derived_row
            })
            .collect_vec();
        jump_stack_matrix.sort_by_key(|row| {
            (
                row[usize::from(JumpStackBaseTableColumn::JSP)].value(),
                row[usize::from(JumpStackBaseTableColumn::CLK)].value(),
            )
        });
        jump_stack_matrix
    }
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
                self.row[usize::from(ST1)].value(),
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
                "st3-0:    [ {:>width$} | {:>width$} | {:>width$} | {:>width$} ]",
                self.row[usize::from(ST3)].value(),
                self.row[usize::from(ST2)].value(),
                self.row[usize::from(ST1)].value(),
                self.row[usize::from(ST0)].value(),
            ),
        )?;
        row(
            f,
            format!(
                "st7-4:    [ {:>width$} | {:>width$} | {:>width$} | {:>width$} ]",
                self.row[usize::from(ST7)].value(),
                self.row[usize::from(ST6)].value(),
                self.row[usize::from(ST5)].value(),
                self.row[usize::from(ST4)].value(),
            ),
        )?;
        row(
            f,
            format!(
                "st11-8:   [ {:>width$} | {:>width$} | {:>width$} | {:>width$} ]",
                self.row[usize::from(ST11)].value(),
                self.row[usize::from(ST10)].value(),
                self.row[usize::from(ST9)].value(),
                self.row[usize::from(ST8)].value(),
            ),
        )?;
        row(
            f,
            format!(
                "st15-12:  [ {:>width$} | {:>width$} | {:>width$} | {:>width$} ]",
                self.row[usize::from(ST15)].value(),
                self.row[usize::from(ST14)].value(),
                self.row[usize::from(ST13)].value(),
                self.row[usize::from(ST12)].value(),
            ),
        )?;

        row_blank(f)?;

        row(
            f,
            format!(
                "hv3-0:    [ {:>width$} | {:>width$} | {:>width$} | {:>width$} ]",
                self.row[usize::from(HV3)].value(),
                self.row[usize::from(HV2)].value(),
                self.row[usize::from(HV1)].value(),
                self.row[usize::from(HV0)].value(),
            ),
        )?;
        row(
            f,
            format!(
                "ib5-0: [ {:>12} | {:>13} | {:>13} | {:>13} | {:>13} | {:>13} ]",
                self.row[usize::from(IB5)].value(),
                self.row[usize::from(IB4)].value(),
                self.row[usize::from(IB3)].value(),
                self.row[usize::from(IB2)].value(),
                self.row[usize::from(IB1)].value(),
                self.row[usize::from(IB0)].value(),
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
