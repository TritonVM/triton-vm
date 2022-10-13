use std::fmt::{Display, Formatter};
use std::ops::Mul;

use itertools::Itertools;
use num_traits::{One, Zero};
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::polynomial::Polynomial;
use twenty_first::shared_math::traits::{FiniteField, Inverse};
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::instruction::AnInstruction::*;
use crate::table::table_column::ProcessorExtTableColumn::*;
use crate::table::table_column::RamBaseTableColumn::{
    BezoutCoefficientPolynomialCoefficient0, BezoutCoefficientPolynomialCoefficient1, RAMP,
};
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
        let program_matrix = Self::derive_program_matrix(program);
        let instruction_matrix = Self::derive_instruction_matrix(&aet, program);
        let op_stack_matrix = Self::derive_op_stack_matrix(&aet);
        let ram_matrix = Self::derive_ram_matrix(&aet);
        let jump_stack_matrix = Self::derive_jump_stack_matrix(&aet);

        let processor_matrix = Self::add_clock_jump_differences_to_processor_matrix(
            aet.processor_matrix,
            &op_stack_matrix,
            &ram_matrix,
            &jump_stack_matrix,
        );

        Self {
            program_matrix,
            instruction_matrix,
            processor_matrix,
            op_stack_matrix,
            ram_matrix,
            jump_stack_matrix,
            hash_matrix: aet.hash_matrix,
        }
    }

    fn add_clock_jump_differences_to_processor_matrix(
        mut processor_matrix: Vec<[BFieldElement; processor_table::BASE_WIDTH]>,
        op_stack_matrix: &[[BFieldElement; op_stack_table::BASE_WIDTH]],
        ram_matrix: &[[BFieldElement; ram_table::BASE_WIDTH]],
        jump_stack_matrix: &[[BFieldElement; jump_stack_table::BASE_WIDTH]],
    ) -> Vec<[BFieldElement; processor_table::BASE_WIDTH]> {
        let one = BFieldElement::one();

        // get clock jump differences for all 3 memory-like tables
        let op_stack_mp_column = usize::from(OpStackBaseTableColumn::OSP);
        let op_stack_clk_column = usize::from(OpStackBaseTableColumn::CLK);
        let mut all_clk_jump_differences = vec![];
        for (curr_row, next_row) in op_stack_matrix.iter().tuple_windows() {
            if next_row[op_stack_mp_column] == curr_row[op_stack_mp_column] {
                let clock_jump_diff = next_row[op_stack_clk_column] - curr_row[op_stack_clk_column];
                if clock_jump_diff != one {
                    all_clk_jump_differences.push(clock_jump_diff);
                }
            }
        }

        let ramp_mp_column = usize::from(RamBaseTableColumn::RAMP);
        let ram_clk_column = usize::from(RamBaseTableColumn::CLK);
        for (curr_row, next_row) in ram_matrix.iter().tuple_windows() {
            if next_row[ramp_mp_column] == curr_row[ramp_mp_column] {
                let clock_jump_diff = next_row[ram_clk_column] - curr_row[ram_clk_column];
                if clock_jump_diff != one {
                    all_clk_jump_differences.push(clock_jump_diff);
                }
            }
        }

        let jump_stack_mp_column = usize::from(JumpStackBaseTableColumn::JSP);
        let jump_stack_clk_column = usize::from(JumpStackBaseTableColumn::CLK);
        for (curr_row, next_row) in jump_stack_matrix.iter().tuple_windows() {
            if next_row[jump_stack_mp_column] == curr_row[jump_stack_mp_column] {
                let clock_jump_diff =
                    next_row[jump_stack_clk_column] - curr_row[jump_stack_clk_column];
                if clock_jump_diff != one {
                    all_clk_jump_differences.push(clock_jump_diff);
                }
            }
        }
        all_clk_jump_differences.sort_by_key(|bfe| std::cmp::Reverse(bfe.value())); // todo: might not need reversal

        // add all clock jump differences and their inverses, as well as inverses of uniques
        let zero = BFieldElement::zero();
        let mut previous_row: Option<[BFieldElement; processor_table::BASE_WIDTH]> = None;
        for row in processor_matrix.iter_mut() {
            let clk_jump_difference = all_clk_jump_differences.pop().unwrap_or(zero);
            let clk_jump_difference_inv = if clk_jump_difference.is_zero() {
                0_u64.into()
            } else {
                clk_jump_difference.inverse()
            };
            row[usize::from(ClockJumpDifference)] = clk_jump_difference;
            row[usize::from(ClockJumpDifferenceInverse)] = clk_jump_difference_inv;

            if let Some(prow) = previous_row {
                let previous_clock_jump_difference = prow[usize::from(ClockJumpDifference)];
                if previous_clock_jump_difference != clk_jump_difference {
                    row[usize::from(UniqueClockJumpDifferenceInverse)] =
                        (previous_clock_jump_difference - clk_jump_difference).inverse();
                }
            }

            previous_row = Some(*row);
        }
        assert!(
            all_clk_jump_differences.is_empty(),
            "Processor Table must record all clock jump differences, \
            but didn't have enough space for remaining {}.",
            all_clk_jump_differences.len()
        );

        processor_matrix
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

        // set inverse of clock difference - 1
        let matrix_len = op_stack_matrix.len();
        let &last_op_stack_matrix_row = op_stack_matrix.last().unwrap();
        let mut new_op_stack_matrix = vec![];
        for (mut current_row, next_row) in op_stack_matrix.into_iter().tuple_windows() {
            current_row[usize::from(OpStackBaseTableColumn::InverseOfClkDiffMinusOne)] = next_row
                [usize::from(OpStackBaseTableColumn::CLK)]
                - current_row[usize::from(OpStackBaseTableColumn::CLK)];
            new_op_stack_matrix.push(current_row);
        }
        new_op_stack_matrix.push(last_op_stack_matrix_row);
        assert_eq!(matrix_len, new_op_stack_matrix.len());

        new_op_stack_matrix
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

        // set inverse of clock difference - 1
        let matrix_len = ram_matrix.len();
        let &last_ram_matrix_row = ram_matrix.last().unwrap();
        let mut new_ram_matrix = vec![];
        for (mut current_row, next_row) in ram_matrix.into_iter().tuple_windows() {
            current_row[usize::from(RamBaseTableColumn::InverseOfClkDiffMinusOne)] = next_row
                [usize::from(RamBaseTableColumn::CLK)]
                - current_row[usize::from(RamBaseTableColumn::CLK)];
            new_ram_matrix.push(current_row);
        }
        new_ram_matrix.push(last_ram_matrix_row);
        let mut ram_matrix = new_ram_matrix;
        assert_eq!(matrix_len, ram_matrix.len());

        // compute Bézout coefficient polynomials
        let all_ramps = ram_matrix
            .iter()
            .map(|row| row[usize::from(RAMP)])
            .dedup()
            .collect_vec();
        let num_of_different_ramps = all_ramps.len();
        let polynomial_with_ramps_as_roots = all_ramps
            .into_iter()
            .map(|ramp| Polynomial::new(vec![-ramp, BFieldElement::one()]))
            .fold(
                Polynomial::from_constant(BFieldElement::one()),
                Polynomial::mul,
            );

        let formal_derivative_coefficients = polynomial_with_ramps_as_roots
            .coefficients
            .iter()
            .enumerate()
            .map(|(i, &coefficient)| BFieldElement::new(i as u64) * coefficient)
            .skip(1)
            .collect_vec();
        let formal_derivative = Polynomial::new(formal_derivative_coefficients);

        let (gcd, bezout_0, bezout_1) =
            XFieldElement::xgcd(polynomial_with_ramps_as_roots, formal_derivative);
        assert!(gcd.is_one());
        assert!(
            bezout_0.degree() < num_of_different_ramps as isize,
            "The Bézout coefficient must be of degree at most {}.",
            num_of_different_ramps - 1
        );
        assert!(
            bezout_1.degree() <= num_of_different_ramps as isize,
            "The Bézout coefficient must be of degree at most {num_of_different_ramps}."
        );

        let mut bezout_coefficient_polynomial_coefficient_0 = bezout_0.coefficients;
        bezout_coefficient_polynomial_coefficient_0
            .resize(num_of_different_ramps, BFieldElement::zero());

        let mut bezout_coefficient_polynomial_coefficient_1 = bezout_1.coefficients;
        bezout_coefficient_polynomial_coefficient_1
            .resize(num_of_different_ramps, BFieldElement::zero());

        // first row
        let mut current_bcpc_0 = bezout_coefficient_polynomial_coefficient_0.pop().unwrap();
        let mut current_bcpc_1 = bezout_coefficient_polynomial_coefficient_1.pop().unwrap();
        ram_matrix.first_mut().unwrap()[usize::from(BezoutCoefficientPolynomialCoefficient0)] =
            current_bcpc_0;
        ram_matrix.first_mut().unwrap()[usize::from(BezoutCoefficientPolynomialCoefficient1)] =
            current_bcpc_1;

        // rest of table
        let mut previous_ramp = ram_matrix.first().unwrap()[usize::from(RAMP)];
        for row in ram_matrix.iter_mut().skip(1) {
            if previous_ramp != row[usize::from(RAMP)] {
                current_bcpc_0 = bezout_coefficient_polynomial_coefficient_0.pop().unwrap();
                current_bcpc_1 = bezout_coefficient_polynomial_coefficient_1.pop().unwrap();
                previous_ramp = row[usize::from(RAMP)]
            }
            row[usize::from(BezoutCoefficientPolynomialCoefficient0)] = current_bcpc_0;
            row[usize::from(BezoutCoefficientPolynomialCoefficient1)] = current_bcpc_1;
        }

        assert_eq!(0, bezout_coefficient_polynomial_coefficient_0.len());
        assert_eq!(0, bezout_coefficient_polynomial_coefficient_1.len());

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

        // set inverse of clock difference - 1
        let matrix_len = jump_stack_matrix.len();
        let &last_op_stack_matrix_row = jump_stack_matrix.last().unwrap();
        let mut new_jump_stack_matrix = vec![];
        for (mut current_row, next_row) in jump_stack_matrix.into_iter().tuple_windows() {
            current_row[usize::from(JumpStackBaseTableColumn::InverseOfClkDiffMinusOne)] = next_row
                [usize::from(JumpStackBaseTableColumn::CLK)]
                - current_row[usize::from(JumpStackBaseTableColumn::CLK)];
            new_jump_stack_matrix.push(current_row);
        }
        new_jump_stack_matrix.push(last_op_stack_matrix_row);
        assert_eq!(matrix_len, new_jump_stack_matrix.len());

        new_jump_stack_matrix
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
