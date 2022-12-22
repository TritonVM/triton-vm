use std::collections::HashMap;

use ndarray::parallel::prelude::*;
use ndarray::s;
use ndarray::Array1;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use ndarray::ArrayViewMut2;
use ndarray::Axis;
use num_traits::One;
use num_traits::Zero;
use strum::EnumCount;
use strum_macros::Display;
use strum_macros::EnumCount as EnumCountMacro;
use strum_macros::EnumIter;
use triton_opcodes::instruction::Instruction;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::polynomial::Polynomial;
use twenty_first::shared_math::traits::Inverse;
use twenty_first::shared_math::x_field_element::XFieldElement;

use RamTableChallengeId::*;

use crate::table::challenges::TableChallenges;
use crate::table::constraint_circuit::ConstraintCircuit;
use crate::table::constraint_circuit::ConstraintCircuitBuilder;
use crate::table::constraint_circuit::DualRowIndicator;
use crate::table::constraint_circuit::DualRowIndicator::*;
use crate::table::constraint_circuit::SingleRowIndicator;
use crate::table::constraint_circuit::SingleRowIndicator::*;
use crate::table::cross_table_argument::CrossTableArg;
use crate::table::cross_table_argument::PermArg;
use crate::table::master_table::NUM_BASE_COLUMNS;
use crate::table::master_table::NUM_EXT_COLUMNS;
use crate::table::table_column::BaseTableColumn;
use crate::table::table_column::ExtTableColumn;
use crate::table::table_column::MasterBaseTableColumn;
use crate::table::table_column::MasterExtTableColumn;
use crate::table::table_column::ProcessorBaseTableColumn;
use crate::table::table_column::RamBaseTableColumn;
use crate::table::table_column::RamBaseTableColumn::*;
use crate::table::table_column::RamExtTableColumn;
use crate::table::table_column::RamExtTableColumn::*;
use crate::vm::AlgebraicExecutionTrace;

pub const RAM_TABLE_NUM_PERMUTATION_ARGUMENTS: usize = 1;
pub const RAM_TABLE_NUM_EVALUATION_ARGUMENTS: usize = 0;
pub const RAM_TABLE_NUM_EXTENSION_CHALLENGES: usize = RamTableChallengeId::COUNT;

pub const BASE_WIDTH: usize = RamBaseTableColumn::COUNT;
pub const EXT_WIDTH: usize = RamExtTableColumn::COUNT;
pub const FULL_WIDTH: usize = BASE_WIDTH + EXT_WIDTH;

#[derive(Debug, Clone)]
pub struct RamTable {}

#[derive(Debug, Clone)]
pub struct ExtRamTable {}

impl RamTable {
    /// Fills the trace table in-place and returns all clock jump differences greater than 1.
    pub fn fill_trace(
        ram_table: &mut ArrayViewMut2<BFieldElement>,
        aet: &AlgebraicExecutionTrace,
    ) -> Vec<BFieldElement> {
        // Store the registers relevant for the Ram Table, i.e., CLK, RAMP, RAMV, and
        // PreviousInstruction, with RAMP as the key. Preserves, thus allows reusing, the order
        // of the processor's rows, which are sorted by CLK. Note that the Ram Table must not be
        // sorted by RAMP, but must form contiguous regions of RAMP values.
        let mut pre_processed_ram_table: HashMap<_, Vec<_>> = HashMap::new();
        for processor_row in aet.processor_matrix.rows() {
            let clk = processor_row[ProcessorBaseTableColumn::CLK.base_table_index()];
            let ramp = processor_row[ProcessorBaseTableColumn::RAMP.base_table_index()];
            let ramv = processor_row[ProcessorBaseTableColumn::RAMV.base_table_index()];
            let previous_instruction =
                processor_row[ProcessorBaseTableColumn::PreviousInstruction.base_table_index()];
            let ram_row = (clk, previous_instruction, ramv);
            pre_processed_ram_table
                .entry(ramp)
                .and_modify(|v| v.push(ram_row))
                .or_insert_with(|| vec![ram_row]);
        }

        // Compute Bézout coefficient polynomials.
        let num_of_ramps = pre_processed_ram_table.keys().len();
        let polynomial_with_ramps_as_roots = pre_processed_ram_table.keys().fold(
            Polynomial::from_constant(BFieldElement::one()),
            |acc, &ramp| acc * Polynomial::new(vec![-ramp, BFieldElement::one()]), // acc·(x - ramp)
        );
        let formal_derivative = polynomial_with_ramps_as_roots.formal_derivative();
        let (gcd, bezout_0, bezout_1) =
            Polynomial::xgcd(polynomial_with_ramps_as_roots, formal_derivative);
        assert!(gcd.is_one(), "Each RAMP value must occur at most once.");
        assert!(
            bezout_0.degree() < num_of_ramps as isize,
            "The Bézout coefficient 0 must be of degree at most {}.",
            num_of_ramps - 1
        );
        assert!(
            bezout_1.degree() <= num_of_ramps as isize,
            "The Bézout coefficient 1 must be of degree at most {num_of_ramps}."
        );
        let mut bezout_coefficient_polynomial_coefficients_0 = bezout_0.coefficients;
        let mut bezout_coefficient_polynomial_coefficients_1 = bezout_1.coefficients;
        bezout_coefficient_polynomial_coefficients_0.resize(num_of_ramps, BFieldElement::zero());
        bezout_coefficient_polynomial_coefficients_1.resize(num_of_ramps, BFieldElement::zero());
        let mut current_bcpc_0 = bezout_coefficient_polynomial_coefficients_0.pop().unwrap();
        let mut current_bcpc_1 = bezout_coefficient_polynomial_coefficients_1.pop().unwrap();
        ram_table[[
            0,
            BezoutCoefficientPolynomialCoefficient0.base_table_index(),
        ]] = current_bcpc_0;
        ram_table[[
            0,
            BezoutCoefficientPolynomialCoefficient1.base_table_index(),
        ]] = current_bcpc_1;

        // Move the rows into the Ram Table as contiguous regions of RAMP values. Each such
        // contiguous region is sorted by CLK by virtue of the order of the processor's rows.
        let mut ram_table_row_idx = 0;
        for (ramp, ram_table_rows) in pre_processed_ram_table {
            for (clk, previous_instruction, ramv) in ram_table_rows {
                let mut ram_table_row = ram_table.row_mut(ram_table_row_idx);
                ram_table_row[CLK.base_table_index()] = clk;
                ram_table_row[RAMP.base_table_index()] = ramp;
                ram_table_row[RAMV.base_table_index()] = ramv;
                ram_table_row[PreviousInstruction.base_table_index()] = previous_instruction;
                ram_table_row_idx += 1;
            }
        }
        assert_eq!(aet.processor_matrix.nrows(), ram_table_row_idx);

        // - Set inverse of clock difference - 1.
        // - Set inverse of RAMP difference.
        // - Fill in the Bézout coefficients if the RAMP has changed.
        // - Collect all clock jump differences greater than 1.
        // The Ram Table and the Processor Table have the same length.
        let mut clock_jump_differences_greater_than_1 = vec![];
        for row_idx in 0..aet.processor_matrix.nrows() - 1 {
            let (mut curr_row, mut next_row) =
                ram_table.multi_slice_mut((s![row_idx, ..], s![row_idx + 1, ..]));

            let clk_diff = next_row[CLK.base_table_index()] - curr_row[CLK.base_table_index()];
            let clk_diff_minus_1 = clk_diff - BFieldElement::one();
            let clk_diff_minus_1_inverse = clk_diff_minus_1.inverse_or_zero();
            curr_row[InverseOfClkDiffMinusOne.base_table_index()] = clk_diff_minus_1_inverse;

            let ramp_diff = next_row[RAMP.base_table_index()] - curr_row[RAMP.base_table_index()];
            let ramp_diff_inverse = ramp_diff.inverse_or_zero();
            curr_row[InverseOfRampDifference.base_table_index()] = ramp_diff_inverse;

            if !ramp_diff.is_zero() {
                current_bcpc_0 = bezout_coefficient_polynomial_coefficients_0.pop().unwrap();
                current_bcpc_1 = bezout_coefficient_polynomial_coefficients_1.pop().unwrap();
            }
            next_row[BezoutCoefficientPolynomialCoefficient0.base_table_index()] = current_bcpc_0;
            next_row[BezoutCoefficientPolynomialCoefficient1.base_table_index()] = current_bcpc_1;

            if ramp_diff.is_zero() && !clk_diff.is_zero() && !clk_diff.is_one() {
                clock_jump_differences_greater_than_1.push(clk_diff);
            }
        }

        assert_eq!(0, bezout_coefficient_polynomial_coefficients_0.len());
        assert_eq!(0, bezout_coefficient_polynomial_coefficients_1.len());

        clock_jump_differences_greater_than_1
    }

    pub fn pad_trace(ram_table: &mut ArrayViewMut2<BFieldElement>, processor_table_len: usize) {
        assert!(
            processor_table_len > 0,
            "Processor Table must have at least 1 row."
        );

        // Set up indices for relevant sections of the table.
        let padded_height = ram_table.nrows();
        let num_padding_rows = padded_height - processor_table_len;
        let max_clk_before_padding = processor_table_len - 1;
        let max_clk_before_padding_row_idx = ram_table
            .rows()
            .into_iter()
            .enumerate()
            .find(|(_, row)| row[CLK.base_table_index()].value() as usize == max_clk_before_padding)
            .map(|(idx, _)| idx)
            .expect("Ram Table must contain row with clock cycle equal to max cycle.");
        let rows_to_move_source_section_start = max_clk_before_padding_row_idx + 1;
        let rows_to_move_source_section_end = processor_table_len;
        let num_rows_to_move = rows_to_move_source_section_end - rows_to_move_source_section_start;
        let rows_to_move_dest_section_start = rows_to_move_source_section_start + num_padding_rows;
        let rows_to_move_dest_section_end = rows_to_move_dest_section_start + num_rows_to_move;
        let padding_section_start = rows_to_move_source_section_start;
        let padding_section_end = padding_section_start + num_padding_rows;
        assert_eq!(padded_height, rows_to_move_dest_section_end);

        // Move all rows below the row with highest CLK to the end of the table – if they exist.
        if num_rows_to_move > 0 {
            let rows_to_move_source_range =
                rows_to_move_source_section_start..rows_to_move_source_section_end;
            let rows_to_move_dest_range =
                rows_to_move_dest_section_start..rows_to_move_dest_section_end;
            let rows_to_move = ram_table
                .slice(s![rows_to_move_source_range, ..])
                .to_owned();
            rows_to_move.move_into(&mut ram_table.slice_mut(s![rows_to_move_dest_range, ..]));
        }

        // Fill the created gap with padding rows, i.e., with (adjusted) copies of the last row
        // before the gap. This is the padding section.
        let mut padding_row_template = ram_table.row(max_clk_before_padding_row_idx).to_owned();
        let ramp_difference_inverse =
            padding_row_template[InverseOfRampDifference.base_table_index()];
        padding_row_template[InverseOfRampDifference.base_table_index()] = BFieldElement::zero();
        padding_row_template[InverseOfClkDiffMinusOne.base_table_index()] = BFieldElement::zero();
        let mut padding_section =
            ram_table.slice_mut(s![padding_section_start..padding_section_end, ..]);
        padding_section
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|padding_row| padding_row_template.clone().move_into(padding_row));

        // CLK keeps increasing by 1 also in the padding section.
        let clk_range = processor_table_len..padded_height;
        let clk_col = Array1::from_iter(clk_range.map(|clk| BFieldElement::new(clk as u64)));
        clk_col.move_into(padding_section.slice_mut(s![.., CLK.base_table_index()]));

        // InverseOfRampDifference and InverseOfClkDiffMinusOne must be consistent at the padding
        // section's boundaries.
        ram_table[[
            max_clk_before_padding_row_idx,
            InverseOfRampDifference.base_table_index(),
        ]] = BFieldElement::zero();
        ram_table[[
            max_clk_before_padding_row_idx,
            InverseOfClkDiffMinusOne.base_table_index(),
        ]] = BFieldElement::zero();
        if num_rows_to_move > 0 && rows_to_move_dest_section_start > 0 {
            let max_clk_after_padding = padded_height - 1;
            let clk_diff_minus_one_at_padding_section_lower_boundary = ram_table
                [[rows_to_move_dest_section_start, CLK.base_table_index()]]
                - BFieldElement::new(max_clk_after_padding as u64)
                - BFieldElement::one();
            let last_row_in_padding_section_idx = rows_to_move_dest_section_start - 1;
            ram_table[[
                last_row_in_padding_section_idx,
                InverseOfRampDifference.base_table_index(),
            ]] = ramp_difference_inverse;
            ram_table[[
                last_row_in_padding_section_idx,
                InverseOfClkDiffMinusOne.base_table_index(),
            ]] = clk_diff_minus_one_at_padding_section_lower_boundary.inverse_or_zero();
        }
    }

    pub fn extend(
        base_table: ArrayView2<BFieldElement>,
        mut ext_table: ArrayViewMut2<XFieldElement>,
        challenges: &RamTableChallenges,
    ) {
        assert_eq!(BASE_WIDTH, base_table.ncols());
        assert_eq!(EXT_WIDTH, ext_table.ncols());
        assert_eq!(base_table.nrows(), ext_table.nrows());
        let mut running_product_for_perm_arg = PermArg::default_initial();
        let mut all_clock_jump_differences_running_product = PermArg::default_initial();

        // initialize columns establishing Bézout relation
        let mut running_product_of_ramp =
            challenges.bezout_relation_indeterminate - base_table.row(0)[RAMP.base_table_index()];
        let mut formal_derivative = XFieldElement::one();
        let mut bezout_coefficient_0 =
            base_table.row(0)[BezoutCoefficientPolynomialCoefficient0.base_table_index()].lift();
        let mut bezout_coefficient_1 =
            base_table.row(0)[BezoutCoefficientPolynomialCoefficient1.base_table_index()].lift();

        let mut previous_row: Option<ArrayView1<BFieldElement>> = None;
        for row_idx in 0..base_table.nrows() {
            let current_row = base_table.row(row_idx);
            let clk = current_row[CLK.base_table_index()];
            let ramp = current_row[RAMP.base_table_index()];
            let ramv = current_row[RAMV.base_table_index()];
            let previous_instruction = current_row[PreviousInstruction.base_table_index()];

            if let Some(prev_row) = previous_row {
                if prev_row[RAMP.base_table_index()] != current_row[RAMP.base_table_index()] {
                    // accumulate coefficient for Bézout relation, proving new RAMP is unique
                    let bcpc0 =
                        current_row[BezoutCoefficientPolynomialCoefficient0.base_table_index()];
                    let bcpc1 =
                        current_row[BezoutCoefficientPolynomialCoefficient1.base_table_index()];
                    let bezout_challenge = challenges.bezout_relation_indeterminate;

                    formal_derivative =
                        (bezout_challenge - ramp) * formal_derivative + running_product_of_ramp;
                    running_product_of_ramp *= bezout_challenge - ramp;
                    bezout_coefficient_0 = bezout_coefficient_0 * bezout_challenge + bcpc0;
                    bezout_coefficient_1 = bezout_coefficient_1 * bezout_challenge + bcpc1;
                } else {
                    // prove that clock jump is directed forward
                    let clock_jump_difference =
                        current_row[CLK.base_table_index()] - prev_row[CLK.base_table_index()];
                    if !clock_jump_difference.is_one() {
                        all_clock_jump_differences_running_product *= challenges
                            .all_clock_jump_differences_multi_perm_indeterminate
                            - clock_jump_difference;
                    }
                }
            }

            // permutation argument to Processor Table
            let compressed_row_for_permutation_argument = clk * challenges.clk_weight
                + ramp * challenges.ramp_weight
                + ramv * challenges.ramv_weight
                + previous_instruction * challenges.previous_instruction_weight;
            running_product_for_perm_arg *=
                challenges.processor_perm_indeterminate - compressed_row_for_permutation_argument;

            let mut extension_row = ext_table.row_mut(row_idx);
            extension_row[RunningProductPermArg.ext_table_index()] = running_product_for_perm_arg;
            extension_row[RunningProductOfRAMP.ext_table_index()] = running_product_of_ramp;
            extension_row[FormalDerivative.ext_table_index()] = formal_derivative;
            extension_row[BezoutCoefficient0.ext_table_index()] = bezout_coefficient_0;
            extension_row[BezoutCoefficient1.ext_table_index()] = bezout_coefficient_1;
            extension_row[AllClockJumpDifferencesPermArg.ext_table_index()] =
                all_clock_jump_differences_running_product;
            previous_row = Some(current_row);
        }
    }
}

impl ExtRamTable {
    pub fn ext_initial_constraints_as_circuits() -> Vec<
        ConstraintCircuit<
            RamTableChallenges,
            SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        use RamTableChallengeId::*;

        let circuit_builder = ConstraintCircuitBuilder::new();
        let one = circuit_builder.b_constant(1_u32.into());

        let bezout_challenge = circuit_builder.challenge(BezoutRelationIndeterminate);
        let rppa_challenge = circuit_builder.challenge(ProcessorPermIndeterminate);

        let clk = circuit_builder.input(BaseRow(CLK.master_base_table_index()));
        let ramp = circuit_builder.input(BaseRow(RAMP.master_base_table_index()));
        let ramv = circuit_builder.input(BaseRow(RAMV.master_base_table_index()));
        let previous_instruction =
            circuit_builder.input(BaseRow(PreviousInstruction.master_base_table_index()));
        let bcpc0 = circuit_builder.input(BaseRow(
            BezoutCoefficientPolynomialCoefficient0.master_base_table_index(),
        ));
        let bcpc1 = circuit_builder.input(BaseRow(
            BezoutCoefficientPolynomialCoefficient1.master_base_table_index(),
        ));
        let rp = circuit_builder.input(ExtRow(RunningProductOfRAMP.master_ext_table_index()));
        let fd = circuit_builder.input(ExtRow(FormalDerivative.master_ext_table_index()));
        let bc0 = circuit_builder.input(ExtRow(BezoutCoefficient0.master_ext_table_index()));
        let bc1 = circuit_builder.input(ExtRow(BezoutCoefficient1.master_ext_table_index()));
        let rpcjd = circuit_builder.input(ExtRow(
            AllClockJumpDifferencesPermArg.master_ext_table_index(),
        ));
        let rppa = circuit_builder.input(ExtRow(RunningProductPermArg.master_ext_table_index()));

        let write_mem_opcode = circuit_builder.b_constant(Instruction::WriteMem.opcode_b());
        let ramv_is_0_or_was_written_to =
            ramv.clone() * (write_mem_opcode - previous_instruction.clone());
        let bezout_coefficient_polynomial_coefficient_0_is_0 = bcpc0;
        let bezout_coefficient_0_is_0 = bc0;
        let bezout_coefficient_1_is_bezout_coefficient_polynomial_coefficient_1 = bc1 - bcpc1;
        let formal_derivative_is_1 = fd - one.clone();
        let running_product_polynomial_is_initialized_correctly =
            rp - (bezout_challenge - ramp.clone());

        let running_product_for_clock_jump_differences_is_initialized_to_1 = rpcjd - one;

        let clk_weight = circuit_builder.challenge(ClkWeight);
        let ramp_weight = circuit_builder.challenge(RampWeight);
        let ramv_weight = circuit_builder.challenge(RamvWeight);
        let previous_instruction_weight = circuit_builder.challenge(PreviousInstructionWeight);
        let compressed_row_for_permutation_argument = clk * clk_weight
            + ramp * ramp_weight
            + ramv * ramv_weight
            + previous_instruction * previous_instruction_weight;
        let running_product_permutation_argument_is_initialized_correctly =
            rppa - (rppa_challenge - compressed_row_for_permutation_argument);

        [
            ramv_is_0_or_was_written_to,
            bezout_coefficient_polynomial_coefficient_0_is_0,
            bezout_coefficient_0_is_0,
            bezout_coefficient_1_is_bezout_coefficient_polynomial_coefficient_1,
            running_product_polynomial_is_initialized_correctly,
            formal_derivative_is_1,
            running_product_for_clock_jump_differences_is_initialized_to_1,
            running_product_permutation_argument_is_initialized_correctly,
        ]
        .map(|circuit| circuit.consume())
        .to_vec()
    }

    pub fn ext_consistency_constraints_as_circuits() -> Vec<
        ConstraintCircuit<
            RamTableChallenges,
            SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        // no further constraints
        vec![]
    }

    pub fn ext_transition_constraints_as_circuits() -> Vec<
        ConstraintCircuit<RamTableChallenges, DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>>,
    > {
        let circuit_builder = ConstraintCircuitBuilder::new();
        let one = circuit_builder.b_constant(1u32.into());

        let bezout_challenge = circuit_builder.challenge(BezoutRelationIndeterminate);
        let cjd_challenge =
            circuit_builder.challenge(AllClockJumpDifferencesMultiPermIndeterminate);
        let rppa_challenge = circuit_builder.challenge(ProcessorPermIndeterminate);
        let clk_weight = circuit_builder.challenge(ClkWeight);
        let ramp_weight = circuit_builder.challenge(RampWeight);
        let ramv_weight = circuit_builder.challenge(RamvWeight);
        let previous_instruction_weight = circuit_builder.challenge(PreviousInstructionWeight);

        let clk = circuit_builder.input(CurrentBaseRow(CLK.master_base_table_index()));
        let ramp = circuit_builder.input(CurrentBaseRow(RAMP.master_base_table_index()));
        let ramv = circuit_builder.input(CurrentBaseRow(RAMV.master_base_table_index()));
        let iord = circuit_builder.input(CurrentBaseRow(
            InverseOfRampDifference.master_base_table_index(),
        ));
        let bcpc0 = circuit_builder.input(CurrentBaseRow(
            BezoutCoefficientPolynomialCoefficient0.master_base_table_index(),
        ));
        let bcpc1 = circuit_builder.input(CurrentBaseRow(
            BezoutCoefficientPolynomialCoefficient1.master_base_table_index(),
        ));
        let clk_diff_minus_one_inv = circuit_builder.input(CurrentBaseRow(
            InverseOfClkDiffMinusOne.master_base_table_index(),
        ));
        let rp =
            circuit_builder.input(CurrentExtRow(RunningProductOfRAMP.master_ext_table_index()));
        let fd = circuit_builder.input(CurrentExtRow(FormalDerivative.master_ext_table_index()));
        let bc0 = circuit_builder.input(CurrentExtRow(BezoutCoefficient0.master_ext_table_index()));
        let bc1 = circuit_builder.input(CurrentExtRow(BezoutCoefficient1.master_ext_table_index()));
        let rpcjd = circuit_builder.input(CurrentExtRow(
            AllClockJumpDifferencesPermArg.master_ext_table_index(),
        ));
        let rppa = circuit_builder.input(CurrentExtRow(
            RunningProductPermArg.master_ext_table_index(),
        ));

        let clk_next = circuit_builder.input(NextBaseRow(CLK.master_base_table_index()));
        let ramp_next = circuit_builder.input(NextBaseRow(RAMP.master_base_table_index()));
        let ramv_next = circuit_builder.input(NextBaseRow(RAMV.master_base_table_index()));
        let previous_instruction_next =
            circuit_builder.input(NextBaseRow(PreviousInstruction.master_base_table_index()));
        let bcpc0_next = circuit_builder.input(NextBaseRow(
            BezoutCoefficientPolynomialCoefficient0.master_base_table_index(),
        ));
        let bcpc1_next = circuit_builder.input(NextBaseRow(
            BezoutCoefficientPolynomialCoefficient1.master_base_table_index(),
        ));
        let rp_next =
            circuit_builder.input(NextExtRow(RunningProductOfRAMP.master_ext_table_index()));
        let fd_next = circuit_builder.input(NextExtRow(FormalDerivative.master_ext_table_index()));
        let bc0_next =
            circuit_builder.input(NextExtRow(BezoutCoefficient0.master_ext_table_index()));
        let bc1_next =
            circuit_builder.input(NextExtRow(BezoutCoefficient1.master_ext_table_index()));
        let rpcjd_next = circuit_builder.input(NextExtRow(
            AllClockJumpDifferencesPermArg.master_ext_table_index(),
        ));
        let rppa_next =
            circuit_builder.input(NextExtRow(RunningProductPermArg.master_ext_table_index()));

        let ramp_diff = ramp_next.clone() - ramp;
        let ramp_changes = ramp_diff.clone() * iord.clone();

        // iord is 0 or iord is the inverse of (ramp' - ramp)
        let iord_is_0_or_iord_is_inverse_of_ramp_diff = iord * (ramp_changes.clone() - one.clone());

        // (ramp' - ramp) is zero or iord is the inverse of (ramp' - ramp)
        let ramp_diff_is_0_or_iord_is_inverse_of_ramp_diff =
            ramp_diff.clone() * (ramp_changes.clone() - one.clone());

        // (ramp doesn't change) and (previous instruction is not write_mem)
        //      implies the ramv doesn't change
        let op_code_write_mem = circuit_builder.b_constant(Instruction::WriteMem.opcode_b());
        let ramp_changes_or_write_mem_or_ramv_stays = (one.clone() - ramp_changes.clone())
            * (op_code_write_mem.clone() - previous_instruction_next.clone())
            * (ramv_next.clone() - ramv);

        // (ramp changes) and (previous instruction is not write_mem)
        //      implies the next ramv is 0
        let ramp_stays_or_write_mem_or_ramv_next_is_0 = ramp_diff.clone()
            * (op_code_write_mem - previous_instruction_next.clone())
            * ramv_next.clone();

        let bcbp0_only_changes_if_ramp_changes =
            (one.clone() - ramp_changes.clone()) * (bcpc0_next.clone() - bcpc0);

        let bcbp1_only_changes_if_ramp_changes =
            (one.clone() - ramp_changes.clone()) * (bcpc1_next.clone() - bcpc1);

        let running_product_ramp_updates_correctly = ramp_diff.clone()
            * (rp_next.clone() - rp.clone() * (bezout_challenge.clone() - ramp_next.clone()))
            + (one.clone() - ramp_changes.clone()) * (rp_next - rp.clone());

        let formal_derivative_updates_correctly = ramp_diff.clone()
            * (fd_next.clone() - rp - (bezout_challenge.clone() - ramp_next.clone()) * fd.clone())
            + (one.clone() - ramp_changes.clone()) * (fd_next - fd);

        let bezout_coefficient_0_is_constructed_correctly = ramp_diff.clone()
            * (bc0_next.clone() - bezout_challenge.clone() * bc0.clone() - bcpc0_next)
            + (one.clone() - ramp_changes.clone()) * (bc0_next - bc0);

        let bezout_coefficient_1_is_constructed_correctly = ramp_diff.clone()
            * (bc1_next.clone() - bezout_challenge * bc1.clone() - bcpc1_next)
            + (one.clone() - ramp_changes.clone()) * (bc1_next - bc1);

        let clk_diff_minus_one = clk_next.clone() - clk.clone() - one.clone();
        let clk_di_is_inverse_of_clk_diff =
            clk_diff_minus_one_inv.clone() * clk_diff_minus_one.clone();
        let clk_di_is_zero_or_inverse_of_clkd =
            clk_diff_minus_one_inv.clone() * (clk_di_is_inverse_of_clk_diff.clone() - one.clone());
        let clkd_is_zero_or_inverse_of_clk_di =
            clk_diff_minus_one.clone() * (clk_di_is_inverse_of_clk_diff - one.clone());

        // Running product of clock jump differences (“rpcjd”) updates iff
        //  - the RAMP remains the same, and
        //  - the clock difference is greater than 1.
        let clk_diff = clk_next.clone() - clk;
        let clk_diff_eq_one = one.clone() - clk_diff.clone();
        let clk_diff_gt_one = one.clone() - clk_diff_minus_one * clk_diff_minus_one_inv;
        let rpcjd_remains = rpcjd_next.clone() - rpcjd.clone();
        let rpcjd_absorbs_clk_diff = rpcjd_next - rpcjd * (cjd_challenge - clk_diff);
        let rpcjd_updates_correctly =
            (one - ramp_changes) * clk_diff_eq_one * rpcjd_absorbs_clk_diff
                + ramp_diff * rpcjd_remains.clone()
                + clk_diff_gt_one * rpcjd_remains;

        let compressed_row_for_permutation_argument = clk_next * clk_weight
            + ramp_next * ramp_weight
            + ramv_next * ramv_weight
            + previous_instruction_next * previous_instruction_weight;
        let rppa_updates_correctly =
            rppa_next - rppa * (rppa_challenge - compressed_row_for_permutation_argument);

        [
            iord_is_0_or_iord_is_inverse_of_ramp_diff,
            ramp_diff_is_0_or_iord_is_inverse_of_ramp_diff,
            ramp_changes_or_write_mem_or_ramv_stays,
            ramp_stays_or_write_mem_or_ramv_next_is_0,
            bcbp0_only_changes_if_ramp_changes,
            bcbp1_only_changes_if_ramp_changes,
            running_product_ramp_updates_correctly,
            formal_derivative_updates_correctly,
            bezout_coefficient_0_is_constructed_correctly,
            bezout_coefficient_1_is_constructed_correctly,
            clk_di_is_zero_or_inverse_of_clkd,
            clkd_is_zero_or_inverse_of_clk_di,
            rpcjd_updates_correctly,
            rppa_updates_correctly,
        ]
        .map(|circuit| circuit.consume())
        .to_vec()
    }

    pub fn ext_terminal_constraints_as_circuits() -> Vec<
        ConstraintCircuit<
            RamTableChallenges,
            SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        let circuit_builder = ConstraintCircuitBuilder::new();
        let one = circuit_builder.b_constant(1_u32.into());

        let rp = circuit_builder.input(ExtRow(RunningProductOfRAMP.master_ext_table_index()));
        let fd = circuit_builder.input(ExtRow(FormalDerivative.master_ext_table_index()));
        let bc0 = circuit_builder.input(ExtRow(BezoutCoefficient0.master_ext_table_index()));
        let bc1 = circuit_builder.input(ExtRow(BezoutCoefficient1.master_ext_table_index()));

        let bezout_relation_holds = bc0 * rp + bc1 * fd - one;

        vec![bezout_relation_holds.consume()]
    }
}

#[derive(Debug, Copy, Clone, Display, EnumCountMacro, EnumIter, PartialEq, Eq, Hash)]
pub enum RamTableChallengeId {
    BezoutRelationIndeterminate,
    ProcessorPermIndeterminate,
    ClkWeight,
    RamvWeight,
    RampWeight,
    PreviousInstructionWeight,
    AllClockJumpDifferencesMultiPermIndeterminate,
}

impl From<RamTableChallengeId> for usize {
    fn from(val: RamTableChallengeId) -> Self {
        val as usize
    }
}

#[derive(Debug, Clone)]
pub struct RamTableChallenges {
    /// The point in which the Bézout relation establishing contiguous memory regions is queried.
    pub bezout_relation_indeterminate: XFieldElement,

    /// Point of evaluation for the row set equality argument between RAM and Processor Tables
    pub processor_perm_indeterminate: XFieldElement,

    /// Weights for condensing part of a row into a single column. (Related to processor table.)
    pub clk_weight: XFieldElement,
    pub ramp_weight: XFieldElement,
    pub ramv_weight: XFieldElement,
    pub previous_instruction_weight: XFieldElement,

    /// Point of evaluation for accumulating all clock jump differences into a running product
    pub all_clock_jump_differences_multi_perm_indeterminate: XFieldElement,
}

impl TableChallenges for RamTableChallenges {
    type Id = RamTableChallengeId;

    #[inline]
    fn get_challenge(&self, id: Self::Id) -> XFieldElement {
        match id {
            BezoutRelationIndeterminate => self.bezout_relation_indeterminate,
            ProcessorPermIndeterminate => self.processor_perm_indeterminate,
            ClkWeight => self.clk_weight,
            RamvWeight => self.ramv_weight,
            RampWeight => self.ramp_weight,
            PreviousInstructionWeight => self.previous_instruction_weight,
            AllClockJumpDifferencesMultiPermIndeterminate => {
                self.all_clock_jump_differences_multi_perm_indeterminate
            }
        }
    }
}
