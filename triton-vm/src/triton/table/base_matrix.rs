use super::table_column::{
    ExtProcessorTableColumn, InstructionTableColumn, JumpStackTableColumn, OpStackTableColumn,
    ProcessorTableColumn::*, RamTableColumn,
};
use super::{
    hash_table, instruction_table, jump_stack_table, op_stack_table, processor_table,
    program_table, ram_table, u32_op_table,
};
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::stark::triton::instruction::AnInstruction::*;
use crate::shared_math::stark::triton::instruction::Instruction;
use crate::shared_math::stark::triton::state::{VMOutput, VMState};
use crate::shared_math::stark::triton::table::table_column::ExtProcessorTableColumn::*;
use crate::shared_math::stark::triton::table::table_column::RamTableColumn::{
    InverseOfRampDifference, RAMP,
};
use crate::shared_math::stark::triton::vm::Program;
use crate::shared_math::traits::IdentityValues;
use crate::shared_math::traits::Inverse;
use crate::shared_math::x_field_element::XFieldElement;
use itertools::Itertools;
use std::fmt::{Display, Formatter};

#[derive(Debug, Clone, Default)]
pub struct BaseMatrices {
    pub program_matrix: Vec<[BFieldElement; program_table::BASE_WIDTH]>,
    pub processor_matrix: Vec<[BFieldElement; processor_table::BASE_WIDTH]>,
    pub instruction_matrix: Vec<[BFieldElement; instruction_table::BASE_WIDTH]>,
    pub op_stack_matrix: Vec<[BFieldElement; op_stack_table::BASE_WIDTH]>,
    pub ram_matrix: Vec<[BFieldElement; ram_table::BASE_WIDTH]>,
    pub jump_stack_matrix: Vec<[BFieldElement; jump_stack_table::BASE_WIDTH]>,
    pub hash_matrix: Vec<[BFieldElement; hash_table::BASE_WIDTH]>,
    pub u32_op_matrix: Vec<[BFieldElement; u32_op_table::BASE_WIDTH]>,
}

impl BaseMatrices {
    /// Initialize `program_matrix` and `instruction_matrix` so that both contain one row per word
    /// in the program. Note that this does not mean “one row per instruction:” instructions that
    /// take two words (e.g. `push N`) add two rows.
    pub fn initialize(&mut self, program: &Program) {
        let mut words_with_0 = program.to_bwords();
        words_with_0.push(0.into());

        for (i, (word, next_word)) in words_with_0.into_iter().tuple_windows().enumerate() {
            let index = (i as u32).into();
            self.program_matrix.push([index, word]);
            self.instruction_matrix.push([index, word, next_word]);
        }

        debug_assert_eq!(program.len(), self.instruction_matrix.len());
    }

    pub fn sort_instruction_matrix(&mut self) {
        self.instruction_matrix
            .sort_by_key(|row| row[InstructionTableColumn::Address as usize].value());
    }

    pub fn sort_op_stack_matrix(&mut self) {
        self.op_stack_matrix.sort_by_key(|row| {
            (
                row[OpStackTableColumn::OSP as usize].value(),
                row[OpStackTableColumn::CLK as usize].value(),
            )
        })
    }

    pub fn sort_ram_matrix(&mut self) {
        self.ram_matrix.sort_by_key(|row| {
            (
                row[RamTableColumn::RAMP as usize].value(),
                row[RamTableColumn::CLK as usize].value(),
            )
        })
    }

    pub fn sort_jump_stack_matrix(&mut self) {
        self.jump_stack_matrix.sort_by_key(|row| {
            (
                row[JumpStackTableColumn::JSP as usize].value(),
                row[JumpStackTableColumn::CLK as usize].value(),
            )
        })
    }

    pub fn set_ram_matrix_inverse_of_ramp_diff(&mut self) {
        let mut iord_column = Vec::with_capacity(self.ram_matrix.len());

        for (curr_row, next_row) in self.ram_matrix.iter().tuple_windows() {
            let ramp_difference = next_row[RAMP as usize] - curr_row[RAMP as usize];
            let inverse_of_ramp_difference = if ramp_difference.is_zero() {
                ramp_difference
            } else {
                ramp_difference.inverse()
            };
            iord_column.push(inverse_of_ramp_difference);
        }

        // fill in last row, for which there is no next row, with default value
        iord_column.push(0.into());

        debug_assert_eq!(self.ram_matrix.len(), iord_column.len());

        for (ram_row, iord) in self.ram_matrix.iter_mut().zip(iord_column) {
            ram_row[InverseOfRampDifference as usize] = iord;
        }
    }

    pub fn append(
        &mut self,
        state: &VMState,
        vm_output: Option<VMOutput>,
        current_instruction: Instruction,
    ) {
        self.processor_matrix
            .push(state.to_processor_row(current_instruction));

        self.instruction_matrix
            .push(state.to_instruction_row(current_instruction));

        self.op_stack_matrix
            .push(state.to_op_stack_row(current_instruction));

        self.ram_matrix.push(state.to_ram_row());

        self.jump_stack_matrix
            .push(state.to_jump_stack_row(current_instruction));

        match vm_output {
            Some(VMOutput::WriteOutputSymbol(_)) => (),
            Some(VMOutput::XlixTrace(mut hash_trace)) => self.hash_matrix.append(&mut hash_trace),
            Some(VMOutput::U32OpTrace(mut trace)) => self.u32_op_matrix.append(&mut trace),
            None => (),
        }
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
        row(f, "cr_instr_table", CompressedRowInstructionTable)?;
        row(f, "instr_table_pa", InstructionTablePermArg)?;
        row(f, "cr_opstack_table", CompressedRowOpStackTable)?;
        row(f, "opstack_table_pa", OpStackTablePermArg)?;
        row(f, "cr_ram_table", CompressedRowRamTable)?;
        row(f, "ram_table_pa", RamTablePermArg)?;
        row(f, "cr_jumpstack_table", CompressedRowJumpStackTable)?;
        row(f, "jumpstack_table_pa", JumpStackTablePermArg)?;
        row(f, "cr_to_hash_table", CompressedRowForHashInput)?;
        row(f, "to_hash_table_ea", ToHashTableEvalArg)?;
        row(f, "cr_from_hash_table", CompressedRowForHashDigest)?;
        row(f, "from_hash_table_ea", FromHashTableEvalArg)?;
        row(f, "cr_u32_lt", CompressedRowLtU32Op)?;
        row(f, "u32_lt_pa", LtU32OpTablePermArg)?;
        row(f, "cr_u32_and", CompressedRowAndU32Op)?;
        row(f, "u32_and_pa", AndU32OpTablePermArg)?;
        row(f, "cr_u32_xor", CompressedRowXorU32Op)?;
        row(f, "u32_xor_pa", XorU32OpTablePermArg)?;
        row(f, "cr_u32_rev", CompressedRowReverseU32Op)?;
        row(f, "u32_rev_pa", ReverseU32OpTablePermArg)?;
        row(f, "cr_u32_div", CompressedRowDivU32Op)?;
        row(f, "u32_div_pa", DivU32OpTablePermArg)?;
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
