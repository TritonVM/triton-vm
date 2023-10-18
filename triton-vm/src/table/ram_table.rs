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
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::polynomial::Polynomial;
use twenty_first::shared_math::traits::Inverse;
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::aet::AlgebraicExecutionTrace;
use crate::instruction::Instruction;
use crate::table::challenges::ChallengeId::*;
use crate::table::challenges::Challenges;
use crate::table::constraint_circuit::DualRowIndicator::*;
use crate::table::constraint_circuit::SingleRowIndicator::*;
use crate::table::constraint_circuit::*;
use crate::table::cross_table_argument::*;
use crate::table::table_column::RamBaseTableColumn::*;
use crate::table::table_column::RamExtTableColumn::*;
use crate::table::table_column::*;

pub const BASE_WIDTH: usize = RamBaseTableColumn::COUNT;
pub const EXT_WIDTH: usize = RamExtTableColumn::COUNT;
pub const FULL_WIDTH: usize = BASE_WIDTH + EXT_WIDTH;

#[derive(Debug, Clone)]
pub struct RamTable {}

#[derive(Debug, Clone)]
pub struct ExtRamTable {}

impl RamTable {
    /// Fills the trace table in-place and returns all clock jump differences.
    pub fn fill_trace(
        ram_table: &mut ArrayViewMut2<BFieldElement>,
        aet: &AlgebraicExecutionTrace,
    ) -> Vec<BFieldElement> {
        // Store the registers relevant for the Ram Table, i.e., CLK, RAMP, RAMV, and
        // PreviousInstruction, with RAMP as the key. Preserves, thus allows reusing, the order
        // of the processor's rows, which are sorted by CLK. Note that the Ram Table does not
        // have to be sorted by RAMP, but must form contiguous regions of RAMP values.
        let mut pre_processed_ram_table: HashMap<_, Vec<_>> = HashMap::new();
        for processor_row in aet.processor_trace.rows() {
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
        assert_eq!(aet.processor_trace.nrows(), ram_table_row_idx);

        // - Set inverse of RAMP difference.
        // - Fill in the Bézout coefficients if the RAMP has changed.
        // - Collect all clock jump differences.
        // The Ram Table and the Processor Table have the same length.
        let mut clock_jump_differences = vec![];
        for row_idx in 0..aet.processor_trace.nrows() - 1 {
            let (mut curr_row, mut next_row) =
                ram_table.multi_slice_mut((s![row_idx, ..], s![row_idx + 1, ..]));

            let ramp_diff = next_row[RAMP.base_table_index()] - curr_row[RAMP.base_table_index()];
            let ramp_diff_inverse = ramp_diff.inverse_or_zero();
            curr_row[InverseOfRampDifference.base_table_index()] = ramp_diff_inverse;

            if !ramp_diff.is_zero() {
                current_bcpc_0 = bezout_coefficient_polynomial_coefficients_0.pop().unwrap();
                current_bcpc_1 = bezout_coefficient_polynomial_coefficients_1.pop().unwrap();
            }
            next_row[BezoutCoefficientPolynomialCoefficient0.base_table_index()] = current_bcpc_0;
            next_row[BezoutCoefficientPolynomialCoefficient1.base_table_index()] = current_bcpc_1;

            let clk_diff = next_row[CLK.base_table_index()] - curr_row[CLK.base_table_index()];
            if ramp_diff.is_zero() {
                assert!(
                    !clk_diff.is_zero(),
                    "All rows must have distinct CLK values, but don't on row with index {row_idx}."
                );
                clock_jump_differences.push(clk_diff);
            }
        }

        assert_eq!(0, bezout_coefficient_polynomial_coefficients_0.len());
        assert_eq!(0, bezout_coefficient_polynomial_coefficients_1.len());

        clock_jump_differences
    }

    pub fn pad_trace(mut ram_table: ArrayViewMut2<BFieldElement>, processor_table_len: usize) {
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

        // InverseOfRampDifference must be consistent at the padding section's boundaries.
        ram_table[[
            max_clk_before_padding_row_idx,
            InverseOfRampDifference.base_table_index(),
        ]] = BFieldElement::zero();
        if num_rows_to_move > 0 && rows_to_move_dest_section_start > 0 {
            let last_row_in_padding_section_idx = rows_to_move_dest_section_start - 1;
            ram_table[[
                last_row_in_padding_section_idx,
                InverseOfRampDifference.base_table_index(),
            ]] = ramp_difference_inverse;
        }
    }

    pub fn extend(
        base_table: ArrayView2<BFieldElement>,
        mut ext_table: ArrayViewMut2<XFieldElement>,
        challenges: &Challenges,
    ) {
        assert_eq!(BASE_WIDTH, base_table.ncols());
        assert_eq!(EXT_WIDTH, ext_table.ncols());
        assert_eq!(base_table.nrows(), ext_table.nrows());

        let clk_weight = challenges[RamClkWeight];
        let ramp_weight = challenges[RamRampWeight];
        let ramv_weight = challenges[RamRamvWeight];
        let previous_instruction_weight = challenges[RamPreviousInstructionWeight];
        let processor_perm_indeterminate = challenges[RamIndeterminate];
        let bezout_relation_indeterminate = challenges[RamTableBezoutRelationIndeterminate];
        let clock_jump_difference_lookup_indeterminate =
            challenges[ClockJumpDifferenceLookupIndeterminate];

        let mut running_product_for_perm_arg = PermArg::default_initial();
        let mut clock_jump_diff_lookup_log_derivative = LookupArg::default_initial();

        // initialize columns establishing Bézout relation
        let mut running_product_of_ramp =
            bezout_relation_indeterminate - base_table.row(0)[RAMP.base_table_index()];
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

                    formal_derivative = (bezout_relation_indeterminate - ramp) * formal_derivative
                        + running_product_of_ramp;
                    running_product_of_ramp *= bezout_relation_indeterminate - ramp;
                    bezout_coefficient_0 =
                        bezout_coefficient_0 * bezout_relation_indeterminate + bcpc0;
                    bezout_coefficient_1 =
                        bezout_coefficient_1 * bezout_relation_indeterminate + bcpc1;
                } else {
                    // prove that clock jump is directed forward
                    let clock_jump_difference =
                        current_row[CLK.base_table_index()] - prev_row[CLK.base_table_index()];
                    clock_jump_diff_lookup_log_derivative +=
                        (clock_jump_difference_lookup_indeterminate - clock_jump_difference)
                            .inverse();
                }
            }

            // permutation argument to Processor Table
            let compressed_row_for_permutation_argument = clk * clk_weight
                + ramp * ramp_weight
                + ramv * ramv_weight
                + previous_instruction * previous_instruction_weight;
            running_product_for_perm_arg *=
                processor_perm_indeterminate - compressed_row_for_permutation_argument;

            let mut extension_row = ext_table.row_mut(row_idx);
            extension_row[RunningProductPermArg.ext_table_index()] = running_product_for_perm_arg;
            extension_row[RunningProductOfRAMP.ext_table_index()] = running_product_of_ramp;
            extension_row[FormalDerivative.ext_table_index()] = formal_derivative;
            extension_row[BezoutCoefficient0.ext_table_index()] = bezout_coefficient_0;
            extension_row[BezoutCoefficient1.ext_table_index()] = bezout_coefficient_1;
            extension_row[ClockJumpDifferenceLookupClientLogDerivative.ext_table_index()] =
                clock_jump_diff_lookup_log_derivative;
            previous_row = Some(current_row);
        }
    }
}

impl ExtRamTable {
    pub fn initial_constraints(
        circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        let one = circuit_builder.b_constant(1_u32.into());

        let bezout_challenge = circuit_builder.challenge(RamTableBezoutRelationIndeterminate);
        let rppa_challenge = circuit_builder.challenge(RamIndeterminate);

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
        let rppa = circuit_builder.input(ExtRow(RunningProductPermArg.master_ext_table_index()));
        let clock_jump_diff_log_derivative = circuit_builder.input(ExtRow(
            ClockJumpDifferenceLookupClientLogDerivative.master_ext_table_index(),
        ));

        let bezout_coefficient_polynomial_coefficient_0_is_0 = bcpc0;
        let bezout_coefficient_0_is_0 = bc0;
        let bezout_coefficient_1_is_bezout_coefficient_polynomial_coefficient_1 = bc1 - bcpc1;
        let formal_derivative_is_1 = fd - one;
        let running_product_polynomial_is_initialized_correctly =
            rp - (bezout_challenge - ramp.clone());

        let clock_jump_diff_log_derivative_is_initialized_correctly = clock_jump_diff_log_derivative
            - circuit_builder.x_constant(LookupArg::default_initial());

        let clk_weight = circuit_builder.challenge(RamClkWeight);
        let ramp_weight = circuit_builder.challenge(RamRampWeight);
        let ramv_weight = circuit_builder.challenge(RamRamvWeight);
        let previous_instruction_weight = circuit_builder.challenge(RamPreviousInstructionWeight);
        let compressed_row_for_permutation_argument = clk * clk_weight
            + ramp * ramp_weight
            + ramv * ramv_weight
            + previous_instruction * previous_instruction_weight;
        let running_product_permutation_argument_is_initialized_correctly =
            rppa - (rppa_challenge - compressed_row_for_permutation_argument);

        vec![
            bezout_coefficient_polynomial_coefficient_0_is_0,
            bezout_coefficient_0_is_0,
            bezout_coefficient_1_is_bezout_coefficient_polynomial_coefficient_1,
            running_product_polynomial_is_initialized_correctly,
            formal_derivative_is_1,
            running_product_permutation_argument_is_initialized_correctly,
            clock_jump_diff_log_derivative_is_initialized_correctly,
        ]
    }

    pub fn consistency_constraints(
        _circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        // no further constraints
        vec![]
    }

    pub fn transition_constraints(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let one = circuit_builder.b_constant(1u32.into());

        let bezout_challenge = circuit_builder.challenge(RamTableBezoutRelationIndeterminate);
        let rppa_challenge = circuit_builder.challenge(RamIndeterminate);
        let clk_weight = circuit_builder.challenge(RamClkWeight);
        let ramp_weight = circuit_builder.challenge(RamRampWeight);
        let ramv_weight = circuit_builder.challenge(RamRamvWeight);
        let previous_instruction_weight = circuit_builder.challenge(RamPreviousInstructionWeight);

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
        let rp =
            circuit_builder.input(CurrentExtRow(RunningProductOfRAMP.master_ext_table_index()));
        let fd = circuit_builder.input(CurrentExtRow(FormalDerivative.master_ext_table_index()));
        let bc0 = circuit_builder.input(CurrentExtRow(BezoutCoefficient0.master_ext_table_index()));
        let bc1 = circuit_builder.input(CurrentExtRow(BezoutCoefficient1.master_ext_table_index()));
        let rppa = circuit_builder.input(CurrentExtRow(
            RunningProductPermArg.master_ext_table_index(),
        ));
        let clock_jump_diff_log_derivative = circuit_builder.input(CurrentExtRow(
            ClockJumpDifferenceLookupClientLogDerivative.master_ext_table_index(),
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
        let rppa_next =
            circuit_builder.input(NextExtRow(RunningProductPermArg.master_ext_table_index()));
        let clock_jump_diff_log_derivative_next = circuit_builder.input(NextExtRow(
            ClockJumpDifferenceLookupClientLogDerivative.master_ext_table_index(),
        ));

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
            * (op_code_write_mem - previous_instruction_next.clone())
            * (ramv_next.clone() - ramv);

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

        let compressed_row_for_permutation_argument = clk_next.clone() * clk_weight
            + ramp_next * ramp_weight
            + ramv_next * ramv_weight
            + previous_instruction_next * previous_instruction_weight;
        let rppa_updates_correctly =
            rppa_next - rppa * (rppa_challenge - compressed_row_for_permutation_argument);

        // The running sum of the logarithmic derivative for the clock jump difference Lookup
        // Argument accumulates a summand of `clk_diff` if and only if the `ramp` does not change.
        // Expressed differently:
        // - the `ramp` changes or the log derivative accumulates a summand, and
        // - the `ramp` does not change or the log derivative does not change.
        let log_derivative_remains =
            clock_jump_diff_log_derivative_next.clone() - clock_jump_diff_log_derivative.clone();
        let clk_diff = clk_next - clk;
        let log_derivative_accumulates = (clock_jump_diff_log_derivative_next
            - clock_jump_diff_log_derivative)
            * (circuit_builder.challenge(ClockJumpDifferenceLookupIndeterminate) - clk_diff)
            - one.clone();
        let log_derivative_updates_correctly =
            (one - ramp_changes) * log_derivative_accumulates + ramp_diff * log_derivative_remains;

        vec![
            iord_is_0_or_iord_is_inverse_of_ramp_diff,
            ramp_diff_is_0_or_iord_is_inverse_of_ramp_diff,
            ramp_changes_or_write_mem_or_ramv_stays,
            bcbp0_only_changes_if_ramp_changes,
            bcbp1_only_changes_if_ramp_changes,
            running_product_ramp_updates_correctly,
            formal_derivative_updates_correctly,
            bezout_coefficient_0_is_constructed_correctly,
            bezout_coefficient_1_is_constructed_correctly,
            rppa_updates_correctly,
            log_derivative_updates_correctly,
        ]
    }

    pub fn terminal_constraints(
        circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        let one = circuit_builder.b_constant(1_u32.into());

        let rp = circuit_builder.input(ExtRow(RunningProductOfRAMP.master_ext_table_index()));
        let fd = circuit_builder.input(ExtRow(FormalDerivative.master_ext_table_index()));
        let bc0 = circuit_builder.input(ExtRow(BezoutCoefficient0.master_ext_table_index()));
        let bc1 = circuit_builder.input(ExtRow(BezoutCoefficient1.master_ext_table_index()));

        let bezout_relation_holds = bc0 * rp + bc1 * fd - one;

        vec![bezout_relation_holds]
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;

    pub fn constraints_evaluate_to_zero(
        master_base_trace_table: ArrayView2<BFieldElement>,
        master_ext_trace_table: ArrayView2<XFieldElement>,
        challenges: &Challenges,
    ) -> bool {
        let zero = XFieldElement::zero();
        assert_eq!(
            master_base_trace_table.nrows(),
            master_ext_trace_table.nrows()
        );

        let circuit_builder = ConstraintCircuitBuilder::new();
        for (constraint_idx, constraint) in ExtRamTable::initial_constraints(&circuit_builder)
            .into_iter()
            .map(|constraint_monad| constraint_monad.consume())
            .enumerate()
        {
            let evaluated_constraint = constraint.evaluate(
                master_base_trace_table.slice(s![..1, ..]),
                master_ext_trace_table.slice(s![..1, ..]),
                challenges,
            );
            assert_eq!(
                zero, evaluated_constraint,
                "Initial constraint {constraint_idx} failed."
            );
        }

        let circuit_builder = ConstraintCircuitBuilder::new();
        for (constraint_idx, constraint) in ExtRamTable::consistency_constraints(&circuit_builder)
            .into_iter()
            .map(|constraint_monad| constraint_monad.consume())
            .enumerate()
        {
            for row_idx in 0..master_base_trace_table.nrows() {
                let evaluated_constraint = constraint.evaluate(
                    master_base_trace_table.slice(s![row_idx..row_idx + 1, ..]),
                    master_ext_trace_table.slice(s![row_idx..row_idx + 1, ..]),
                    challenges,
                );
                assert_eq!(
                    zero, evaluated_constraint,
                    "Consistency constraint {constraint_idx} failed on row {row_idx}."
                );
            }
        }

        let circuit_builder = ConstraintCircuitBuilder::new();
        for (constraint_idx, constraint) in ExtRamTable::transition_constraints(&circuit_builder)
            .into_iter()
            .map(|constraint_monad| constraint_monad.consume())
            .enumerate()
        {
            for row_idx in 0..master_base_trace_table.nrows() - 1 {
                let evaluated_constraint = constraint.evaluate(
                    master_base_trace_table.slice(s![row_idx..row_idx + 2, ..]),
                    master_ext_trace_table.slice(s![row_idx..row_idx + 2, ..]),
                    challenges,
                );
                assert_eq!(
                    zero, evaluated_constraint,
                    "Transition constraint {constraint_idx} failed on row {row_idx}."
                );
            }
        }

        let circuit_builder = ConstraintCircuitBuilder::new();
        for (constraint_idx, constraint) in ExtRamTable::terminal_constraints(&circuit_builder)
            .into_iter()
            .map(|constraint_monad| constraint_monad.consume())
            .enumerate()
        {
            let evaluated_constraint = constraint.evaluate(
                master_base_trace_table.slice(s![-1.., ..]),
                master_ext_trace_table.slice(s![-1.., ..]),
                challenges,
            );
            assert_eq!(
                zero, evaluated_constraint,
                "Terminal constraint {constraint_idx} failed."
            );
        }

        true
    }
}
