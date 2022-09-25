use std::fmt::{Display, Formatter};

use itertools::Itertools;
use num_traits::Zero;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::traits::FiniteField;
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::instruction::AnInstruction::*;
use crate::table::table_column::ExtProcessorTableColumn::*;
use crate::table::table_column::{
    InstructionTableColumn, OpStackTableColumn, ProgramTableColumn, RamTableColumn,
};
use crate::vm::Program;

use super::table_column::{ExtProcessorTableColumn, JumpStackTableColumn, ProcessorTableColumn::*};
use super::{
    hash_table, instruction_table, jump_stack_table, op_stack_table, processor_table,
    program_table, ram_table, u32_op_table,
};

#[derive(Debug, Clone, Default)]
pub struct AlgebraicExecutionTrace {
    pub processor_matrix: Vec<[BFieldElement; processor_table::BASE_WIDTH]>,
    pub hash_matrix: Vec<[BFieldElement; hash_table::BASE_WIDTH]>,
    pub u32_op_matrix: Vec<[BFieldElement; u32_op_table::BASE_WIDTH]>,
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
    pub u32_op_matrix: Vec<[BFieldElement; u32_op_table::BASE_WIDTH]>,
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
            u32_op_matrix: aet.u32_op_matrix,
        }
    }

    fn derive_program_matrix(program: &Program) -> Vec<[BFieldElement; program_table::BASE_WIDTH]> {
        program
            .to_bwords()
            .into_iter()
            .enumerate()
            .map(|(idx, instruction)| {
                let mut derived_row = [BFieldElement::zero(); program_table::BASE_WIDTH];
                derived_row[ProgramTableColumn::Address as usize] = (idx as u32).into();
                derived_row[ProgramTableColumn::Instruction as usize] = instruction;
                derived_row
            })
            .collect_vec()
    }

    fn derive_instruction_matrix(
        aet: &AlgebraicExecutionTrace,
        program: &Program,
    ) -> Vec<[BFieldElement; instruction_table::BASE_WIDTH]> {
        let program_append_0 = [program.to_bwords(), vec![BFieldElement::zero()]].concat();
        let program_part = program_append_0
            .into_iter()
            .tuple_windows()
            .enumerate()
            .map(|(idx, (instruction, next_instruction))| {
                let mut derived_row = [BFieldElement::zero(); instruction_table::BASE_WIDTH];
                derived_row[InstructionTableColumn::Address as usize] = (idx as u32).into();
                derived_row[InstructionTableColumn::CI as usize] = instruction;
                derived_row[InstructionTableColumn::NIA as usize] = next_instruction;
                derived_row
            })
            .collect_vec();
        let processor_part = aet
            .processor_matrix
            .iter()
            .map(|&row| {
                let mut derived_row = [BFieldElement::zero(); instruction_table::BASE_WIDTH];
                derived_row[InstructionTableColumn::Address as usize] = row[IP as usize];
                derived_row[InstructionTableColumn::CI as usize] = row[CI as usize];
                derived_row[InstructionTableColumn::NIA as usize] = row[NIA as usize];
                derived_row
            })
            .collect_vec();
        let mut instruction_matrix = [program_part, processor_part].concat();
        instruction_matrix.sort_by_key(|row| row[InstructionTableColumn::Address as usize].value());
        instruction_matrix
    }

    fn derive_op_stack_matrix(
        aet: &AlgebraicExecutionTrace,
    ) -> Vec<[BFieldElement; op_stack_table::BASE_WIDTH]> {
        let mut op_stack_matrix = aet
            .processor_matrix
            .iter()
            .map(|&row| {
                let mut derived_row = [BFieldElement::zero(); op_stack_table::BASE_WIDTH];
                derived_row[OpStackTableColumn::CLK as usize] = row[CLK as usize];
                derived_row[OpStackTableColumn::IB1ShrinkStack as usize] = row[IB1 as usize];
                derived_row[OpStackTableColumn::OSP as usize] = row[OSP as usize];
                derived_row[OpStackTableColumn::OSV as usize] = row[OSV as usize];
                derived_row
            })
            .collect_vec();
        op_stack_matrix.sort_by_key(|row| {
            (
                row[OpStackTableColumn::OSP as usize].value(),
                row[OpStackTableColumn::CLK as usize].value(),
            )
        });
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
                derived_row[RamTableColumn::CLK as usize] = row[CLK as usize];
                derived_row[RamTableColumn::RAMP as usize] = row[ST1 as usize];
                derived_row[RamTableColumn::RAMV as usize] = row[RAMV as usize];
                derived_row[RamTableColumn::InverseOfRampDifference as usize] =
                    BFieldElement::zero();
                derived_row
            })
            .collect_vec();
        ram_matrix.sort_by_key(|row| {
            (
                row[RamTableColumn::RAMP as usize].value(),
                row[RamTableColumn::CLK as usize].value(),
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
                    next_row[RamTableColumn::RAMP as usize]
                        - curr_row[RamTableColumn::RAMP as usize],
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
            ram_matrix[idx][RamTableColumn::InverseOfRampDifference as usize] = inverse;
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
                derived_row[JumpStackTableColumn::CLK as usize] = row[CLK as usize];
                derived_row[JumpStackTableColumn::CI as usize] = row[CI as usize];
                derived_row[JumpStackTableColumn::JSP as usize] = row[JSP as usize];
                derived_row[JumpStackTableColumn::JSO as usize] = row[JSO as usize];
                derived_row[JumpStackTableColumn::JSD as usize] = row[JSD as usize];
                derived_row
            })
            .collect_vec();
        jump_stack_matrix.sort_by_key(|row| {
            (
                row[JumpStackTableColumn::JSP as usize].value(),
                row[JumpStackTableColumn::CLK as usize].value(),
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

        let instruction = self.row[CI as usize].value().try_into().unwrap();
        let instruction_with_arg = match instruction {
            Push(_) => Push(self.row[NIA as usize]),
            Call(_) => Call(self.row[NIA as usize]),
            Dup(_) => Dup((self.row[NIA as usize].value() as u32).try_into().unwrap()),
            Swap(_) => Swap((self.row[NIA as usize].value() as u32).try_into().unwrap()),
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
                self.row[IP as usize].value(),
                self.row[CI as usize].value(),
                self.row[NIA as usize].value(),
                self.row[CLK as usize].value(),
            ),
        )?;

        writeln!(
            f,
            "│ jsp:  {:>width$} │ jso:  {:>width$} │ jsd: {:>width$} ╰───────────────────┤",
            self.row[JSP as usize].value(),
            self.row[JSO as usize].value(),
            self.row[JSD as usize].value(),
        )?;
        row(
            f,
            format!(
                "ramp: {:>width$} │ ramv: {:>width$} │",
                self.row[ST1 as usize].value(),
                self.row[RAMV as usize].value(),
            ),
        )?;
        row(
            f,
            format!(
                "osp:  {:>width$} │ osv:  {:>width$} ╵",
                self.row[OSP as usize].value(),
                self.row[OSV as usize].value(),
            ),
        )?;

        row_blank(f)?;

        row(
            f,
            format!(
                "st3-0:    [ {:>width$} | {:>width$} | {:>width$} | {:>width$} ]",
                self.row[ST3 as usize].value(),
                self.row[ST2 as usize].value(),
                self.row[ST1 as usize].value(),
                self.row[ST0 as usize].value(),
            ),
        )?;
        row(
            f,
            format!(
                "st7-4:    [ {:>width$} | {:>width$} | {:>width$} | {:>width$} ]",
                self.row[ST7 as usize].value(),
                self.row[ST6 as usize].value(),
                self.row[ST5 as usize].value(),
                self.row[ST4 as usize].value(),
            ),
        )?;
        row(
            f,
            format!(
                "st11-8:   [ {:>width$} | {:>width$} | {:>width$} | {:>width$} ]",
                self.row[ST11 as usize].value(),
                self.row[ST10 as usize].value(),
                self.row[ST9 as usize].value(),
                self.row[ST8 as usize].value(),
            ),
        )?;
        row(
            f,
            format!(
                "st15-12:  [ {:>width$} | {:>width$} | {:>width$} | {:>width$} ]",
                self.row[ST15 as usize].value(),
                self.row[ST14 as usize].value(),
                self.row[ST13 as usize].value(),
                self.row[ST12 as usize].value(),
            ),
        )?;

        row_blank(f)?;

        row(
            f,
            format!(
                "hv3-0:    [ {:>width$} | {:>width$} | {:>width$} | {:>width$} ]",
                self.row[HV3 as usize].value(),
                self.row[HV2 as usize].value(),
                self.row[HV1 as usize].value(),
                self.row[HV0 as usize].value(),
            ),
        )?;
        row(
            f,
            format!(
                "ib5-0: [ {:>12} | {:>13} | {:>13} | {:>13} | {:>13} | {:>13} ]",
                self.row[IB5 as usize].value(),
                self.row[IB4 as usize].value(),
                self.row[IB3 as usize].value(),
                self.row[IB2 as usize].value(),
                self.row[IB1 as usize].value(),
                self.row[IB0 as usize].value(),
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
                   col: ExtProcessorTableColumn|
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
        row(f, "u32_lt_pa", U32OpTablePermArg)?;
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
            self.row[JumpStackTableColumn::CLK as usize].value(),
            self.row[JumpStackTableColumn::CI as usize].value(),
            self.row[JumpStackTableColumn::JSP as usize].value(),
            self.row[JumpStackTableColumn::JSO as usize].value(),
            self.row[JumpStackTableColumn::JSD as usize].value(),
        )
    }
}
