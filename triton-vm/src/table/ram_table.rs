use std::cmp::Ordering;

use arbitrary::Arbitrary;
use itertools::Itertools;
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
use twenty_first::shared_math::b_field_element::BFIELD_ONE;
use twenty_first::shared_math::b_field_element::BFIELD_ZERO;
use twenty_first::shared_math::polynomial::Polynomial;
use twenty_first::shared_math::traits::Inverse;
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::aet::AlgebraicExecutionTrace;
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

pub const INSTRUCTION_TYPE_WRITE: BFieldElement = BFIELD_ZERO;
pub const INSTRUCTION_TYPE_READ: BFieldElement = BFIELD_ONE;
pub const PADDING_INDICATOR: BFieldElement = BFieldElement::new(2);

#[derive(Debug, Clone, PartialEq, Eq, Hash, Arbitrary)]
pub struct RamTableCall {
    pub clk: u32,
    pub ram_pointer: BFieldElement,
    pub ram_value: BFieldElement,
    pub is_write: bool,
}

impl RamTableCall {
    pub fn to_table_row(self) -> Array1<BFieldElement> {
        let instruction_type = match self.is_write {
            true => INSTRUCTION_TYPE_WRITE,
            false => INSTRUCTION_TYPE_READ,
        };

        let mut row = Array1::zeros(BASE_WIDTH);
        row[CLK.base_table_index()] = self.clk.into();
        row[InstructionType.base_table_index()] = instruction_type;
        row[RamPointer.base_table_index()] = self.ram_pointer;
        row[RamValue.base_table_index()] = self.ram_value;
        row
    }
}

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
        let mut ram_table = ram_table.slice_mut(s![0..aet.ram_table_length(), ..]);
        let trace_iter = aet.ram_trace.rows().into_iter();

        let sorted_rows =
            trace_iter.sorted_by(|row_0, row_1| Self::compare_rows(row_0.view(), row_1.view()));
        for (row_index, row) in sorted_rows.enumerate() {
            ram_table.row_mut(row_index).assign(&row);
        }

        let (bezout_0, bezout_1) =
            Self::bezout_coefficient_polynomials_coefficients(ram_table.view());

        Self::make_ram_table_consistent(&mut ram_table, bezout_0, bezout_1)
    }

    fn compare_rows(
        row_0: ArrayView1<BFieldElement>,
        row_1: ArrayView1<BFieldElement>,
    ) -> Ordering {
        let ram_pointer_0 = row_0[RamPointer.base_table_index()].value();
        let ram_pointer_1 = row_1[RamPointer.base_table_index()].value();
        let compare_ram_pointers = ram_pointer_0.cmp(&ram_pointer_1);

        let clk_0 = row_0[CLK.base_table_index()].value();
        let clk_1 = row_1[CLK.base_table_index()].value();
        let compare_clocks = clk_0.cmp(&clk_1);

        compare_ram_pointers.then(compare_clocks)
    }

    fn bezout_coefficient_polynomials_coefficients(
        ram_table: ArrayView2<BFieldElement>,
    ) -> (Vec<BFieldElement>, Vec<BFieldElement>) {
        if ram_table.nrows() == 0 {
            return (vec![], vec![]);
        }

        let linear_poly_with_root = |&r: &BFieldElement| Polynomial::new(vec![-r, BFIELD_ONE]);

        let all_ram_pointers = ram_table.column(RamPointer.base_table_index());
        let unique_ram_pointers = all_ram_pointers.iter().unique();
        let num_unique_ram_pointers = unique_ram_pointers.clone().count();

        let polynomial_with_ram_pointers_as_roots = unique_ram_pointers
            .map(linear_poly_with_root)
            .reduce(|accumulator, linear_poly| accumulator * linear_poly)
            .unwrap_or_else(Polynomial::zero);
        let formal_derivative = polynomial_with_ram_pointers_as_roots.formal_derivative();

        let (gcd, bezout_poly_0, bezout_poly_1) =
            Polynomial::xgcd(polynomial_with_ram_pointers_as_roots, formal_derivative);

        assert!(gcd.is_one());
        assert!(bezout_poly_0.degree() < num_unique_ram_pointers as isize);
        assert!(bezout_poly_1.degree() <= num_unique_ram_pointers as isize);

        let mut coefficients_0 = bezout_poly_0.coefficients;
        let mut coefficients_1 = bezout_poly_1.coefficients;
        coefficients_0.resize(num_unique_ram_pointers, BFIELD_ZERO);
        coefficients_1.resize(num_unique_ram_pointers, BFIELD_ZERO);
        (coefficients_0, coefficients_1)
    }

    /// - Set inverse of RAM pointer difference
    /// - Fill in the Bézout coefficients if the RAM pointer changes between two consecutive rows
    /// - Collect and return all clock jump differences
    fn make_ram_table_consistent(
        ram_table: &mut ArrayViewMut2<BFieldElement>,
        mut bezout_coefficient_polynomial_coefficients_0: Vec<BFieldElement>,
        mut bezout_coefficient_polynomial_coefficients_1: Vec<BFieldElement>,
    ) -> Vec<BFieldElement> {
        if ram_table.nrows() == 0 {
            assert_eq!(0, bezout_coefficient_polynomial_coefficients_0.len());
            assert_eq!(0, bezout_coefficient_polynomial_coefficients_1.len());
            return vec![];
        }

        let mut current_bcpc_0 = bezout_coefficient_polynomial_coefficients_0.pop().unwrap();
        let mut current_bcpc_1 = bezout_coefficient_polynomial_coefficients_1.pop().unwrap();
        ram_table.row_mut(0)[BezoutCoefficientPolynomialCoefficient0.base_table_index()] =
            current_bcpc_0;
        ram_table.row_mut(0)[BezoutCoefficientPolynomialCoefficient1.base_table_index()] =
            current_bcpc_1;

        let mut clock_jump_differences = vec![];
        for row_idx in 0..ram_table.nrows() - 1 {
            let (mut curr_row, mut next_row) =
                ram_table.multi_slice_mut((s![row_idx, ..], s![row_idx + 1, ..]));

            let ramp_diff =
                next_row[RamPointer.base_table_index()] - curr_row[RamPointer.base_table_index()];
            let clk_diff = next_row[CLK.base_table_index()] - curr_row[CLK.base_table_index()];

            if ramp_diff.is_zero() {
                assert!(!clk_diff.is_zero(), "row_idx = {row_idx}");
                clock_jump_differences.push(clk_diff);
            } else {
                current_bcpc_0 = bezout_coefficient_polynomial_coefficients_0.pop().unwrap();
                current_bcpc_1 = bezout_coefficient_polynomial_coefficients_1.pop().unwrap();
            }

            curr_row[InverseOfRampDifference.base_table_index()] = ramp_diff.inverse_or_zero();
            next_row[BezoutCoefficientPolynomialCoefficient0.base_table_index()] = current_bcpc_0;
            next_row[BezoutCoefficientPolynomialCoefficient1.base_table_index()] = current_bcpc_1;
        }

        assert_eq!(0, bezout_coefficient_polynomial_coefficients_0.len());
        assert_eq!(0, bezout_coefficient_polynomial_coefficients_1.len());
        clock_jump_differences
    }

    pub fn pad_trace(mut ram_table: ArrayViewMut2<BFieldElement>, ram_table_len: usize) {
        let last_row_index = ram_table_len.saturating_sub(1);
        let mut padding_row = ram_table.row(last_row_index).to_owned();
        padding_row[InstructionType.base_table_index()] = PADDING_INDICATOR;
        if ram_table_len == 0 {
            padding_row[BezoutCoefficientPolynomialCoefficient1.base_table_index()] = BFIELD_ONE;
        }

        let mut padding_section = ram_table.slice_mut(s![ram_table_len.., ..]);
        padding_section
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut row| row.assign(&padding_row));
    }

    pub fn extend(
        base_table: ArrayView2<BFieldElement>,
        mut ext_table: ArrayViewMut2<XFieldElement>,
        challenges: &Challenges,
    ) {
        assert_eq!(BASE_WIDTH, base_table.ncols());
        assert_eq!(EXT_WIDTH, ext_table.ncols());
        assert_eq!(base_table.nrows(), ext_table.nrows());

        let mut running_product_for_perm_arg = PermArg::default_initial();
        let mut clock_jump_diff_lookup_log_derivative = LookupArg::default_initial();

        // initialize columns establishing Bézout relation
        let bezout_indeterminate = challenges[RamTableBezoutRelationIndeterminate];
        let clock_jump_difference_lookup_indeterminate =
            challenges[ClockJumpDifferenceLookupIndeterminate];
        let mut running_product_ram_pointer =
            bezout_indeterminate - base_table.row(0)[RamPointer.base_table_index()];
        let mut formal_derivative = XFieldElement::one();
        let mut bezout_coefficient_0 =
            base_table.row(0)[BezoutCoefficientPolynomialCoefficient0.base_table_index()].lift();
        let mut bezout_coefficient_1 =
            base_table.row(0)[BezoutCoefficientPolynomialCoefficient1.base_table_index()].lift();

        let mut previous_row: Option<ArrayView1<BFieldElement>> = None;
        for row_idx in 0..base_table.nrows() {
            let current_row = base_table.row(row_idx);
            let clk = current_row[CLK.base_table_index()];
            let instruction_type = current_row[InstructionType.base_table_index()];
            let current_ram_pointer = current_row[RamPointer.base_table_index()];
            let ram_value = current_row[RamValue.base_table_index()];

            let is_no_padding_row = instruction_type != PADDING_INDICATOR;

            if is_no_padding_row {
                if let Some(previous_row) = previous_row {
                    let previous_ram_pointer = previous_row[RamPointer.base_table_index()];
                    if previous_ram_pointer != current_ram_pointer {
                        // accumulate coefficient for Bézout relation, proving new RAMP is unique
                        let bcpc0 =
                            current_row[BezoutCoefficientPolynomialCoefficient0.base_table_index()];
                        let bcpc1 =
                            current_row[BezoutCoefficientPolynomialCoefficient1.base_table_index()];

                        formal_derivative = (bezout_indeterminate - current_ram_pointer)
                            * formal_derivative
                            + running_product_ram_pointer;
                        running_product_ram_pointer *= bezout_indeterminate - current_ram_pointer;
                        bezout_coefficient_0 = bezout_coefficient_0 * bezout_indeterminate + bcpc0;
                        bezout_coefficient_1 = bezout_coefficient_1 * bezout_indeterminate + bcpc1;
                    } else {
                        let previous_clock = previous_row[CLK.base_table_index()];
                        let current_clock = current_row[CLK.base_table_index()];
                        let clock_jump_difference = current_clock - previous_clock;
                        let log_derivative_summand =
                            clock_jump_difference_lookup_indeterminate - clock_jump_difference;
                        clock_jump_diff_lookup_log_derivative += log_derivative_summand.inverse();
                    }
                }

                // permutation argument to Processor Table
                let compressed_row = clk * challenges[RamClkWeight]
                    + instruction_type * challenges[RamInstructionTypeWeight]
                    + current_ram_pointer * challenges[RamPointerWeight]
                    + ram_value * challenges[RamValueWeight];
                running_product_for_perm_arg *= challenges[RamIndeterminate] - compressed_row;
            }

            let mut extension_row = ext_table.row_mut(row_idx);
            extension_row[RunningProductPermArg.ext_table_index()] = running_product_for_perm_arg;
            extension_row[RunningProductOfRAMP.ext_table_index()] = running_product_ram_pointer;
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
        let challenge = |c| circuit_builder.challenge(c);
        let constant = |c| circuit_builder.b_constant(c);
        let x_constant = |c| circuit_builder.x_constant(c);
        let base_row = |column: RamBaseTableColumn| {
            circuit_builder.input(BaseRow(column.master_base_table_index()))
        };
        let ext_row = |column: RamExtTableColumn| {
            circuit_builder.input(ExtRow(column.master_ext_table_index()))
        };

        let first_row_is_padding_row = base_row(InstructionType) - constant(PADDING_INDICATOR);
        let first_row_is_not_padding_row = (base_row(InstructionType)
            - constant(INSTRUCTION_TYPE_READ))
            * (base_row(InstructionType) - constant(INSTRUCTION_TYPE_WRITE));

        let bezout_coefficient_polynomial_coefficient_0_is_0 =
            base_row(BezoutCoefficientPolynomialCoefficient0);
        let bezout_coefficient_0_is_0 = ext_row(BezoutCoefficient0);
        let bezout_coefficient_1_is_bezout_coefficient_polynomial_coefficient_1 =
            ext_row(BezoutCoefficient1) - base_row(BezoutCoefficientPolynomialCoefficient1);
        let formal_derivative_is_1 = ext_row(FormalDerivative) - constant(1_u32.into());
        let running_product_polynomial_is_initialized_correctly = ext_row(RunningProductOfRAMP)
            - challenge(RamTableBezoutRelationIndeterminate)
            + base_row(RamPointer);

        let clock_jump_diff_log_derivative_is_default_initial =
            ext_row(ClockJumpDifferenceLookupClientLogDerivative)
                - x_constant(LookupArg::default_initial());

        let compressed_row_for_permutation_argument = base_row(CLK) * challenge(RamClkWeight)
            + base_row(InstructionType) * challenge(RamInstructionTypeWeight)
            + base_row(RamPointer) * challenge(RamPointerWeight)
            + base_row(RamValue) * challenge(RamValueWeight);
        let running_product_permutation_argument_has_accumulated_first_row =
            ext_row(RunningProductPermArg) - challenge(RamIndeterminate)
                + compressed_row_for_permutation_argument;
        let running_product_permutation_argument_is_default_initial =
            ext_row(RunningProductPermArg) - x_constant(PermArg::default_initial());

        let running_product_permutation_argument_starts_correctly =
            running_product_permutation_argument_has_accumulated_first_row
                * first_row_is_padding_row
                + running_product_permutation_argument_is_default_initial
                    * first_row_is_not_padding_row;

        vec![
            bezout_coefficient_polynomial_coefficient_0_is_0,
            bezout_coefficient_0_is_0,
            bezout_coefficient_1_is_bezout_coefficient_polynomial_coefficient_1,
            running_product_polynomial_is_initialized_correctly,
            formal_derivative_is_1,
            running_product_permutation_argument_starts_correctly,
            clock_jump_diff_log_derivative_is_default_initial,
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
        let constant = |c| circuit_builder.b_constant(c);
        let challenge = |c| circuit_builder.challenge(c);
        let curr_base_row = |column: RamBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(column.master_base_table_index()))
        };
        let curr_ext_row = |column: RamExtTableColumn| {
            circuit_builder.input(CurrentExtRow(column.master_ext_table_index()))
        };
        let next_base_row = |column: RamBaseTableColumn| {
            circuit_builder.input(NextBaseRow(column.master_base_table_index()))
        };
        let next_ext_row = |column: RamExtTableColumn| {
            circuit_builder.input(NextExtRow(column.master_ext_table_index()))
        };

        let one = constant(1_u32.into());

        let bezout_challenge = challenge(RamTableBezoutRelationIndeterminate);

        let clock = curr_base_row(CLK);
        let ram_pointer = curr_base_row(RamPointer);
        let ram_value = curr_base_row(RamValue);
        let instruction_type = curr_base_row(InstructionType);
        let inverse_of_ram_pointer_difference = curr_base_row(InverseOfRampDifference);
        let bcpc0 = curr_base_row(BezoutCoefficientPolynomialCoefficient0);
        let bcpc1 = curr_base_row(BezoutCoefficientPolynomialCoefficient1);

        let running_product_ram_pointer = curr_ext_row(RunningProductOfRAMP);
        let fd = curr_ext_row(FormalDerivative);
        let bc0 = curr_ext_row(BezoutCoefficient0);
        let bc1 = curr_ext_row(BezoutCoefficient1);
        let rppa = curr_ext_row(RunningProductPermArg);
        let clock_jump_diff_log_derivative =
            curr_ext_row(ClockJumpDifferenceLookupClientLogDerivative);

        let clock_next = next_base_row(CLK);
        let ram_pointer_next = next_base_row(RamPointer);
        let ram_value_next = next_base_row(RamValue);
        let instruction_type_next = next_base_row(InstructionType);
        let bcpc0_next = next_base_row(BezoutCoefficientPolynomialCoefficient0);
        let bcpc1_next = next_base_row(BezoutCoefficientPolynomialCoefficient1);

        let running_product_ram_pointer_next = next_ext_row(RunningProductOfRAMP);
        let fd_next = next_ext_row(FormalDerivative);
        let bc0_next = next_ext_row(BezoutCoefficient0);
        let bc1_next = next_ext_row(BezoutCoefficient1);
        let rppa_next = next_ext_row(RunningProductPermArg);
        let clock_jump_diff_log_derivative_next =
            next_ext_row(ClockJumpDifferenceLookupClientLogDerivative);

        let next_row_is_padding_row =
            instruction_type_next.clone() - constant(PADDING_INDICATOR).clone();
        let if_current_row_is_padding_row_then_next_row_is_padding_row = (instruction_type.clone()
            - constant(INSTRUCTION_TYPE_READ))
            * (instruction_type - constant(INSTRUCTION_TYPE_WRITE))
            * next_row_is_padding_row.clone();

        let ram_pointer_difference = ram_pointer_next.clone() - ram_pointer;
        let ram_pointer_changes = one.clone()
            - ram_pointer_difference.clone() * inverse_of_ram_pointer_difference.clone();

        let iord_is_0_or_iord_is_inverse_of_ram_pointer_difference =
            inverse_of_ram_pointer_difference * ram_pointer_changes.clone();

        let ram_pointer_difference_is_0_or_iord_is_inverse_of_ram_pointer_difference =
            ram_pointer_difference.clone() * ram_pointer_changes.clone();

        let ram_pointer_changes_or_write_mem_or_ram_value_stays = ram_pointer_changes.clone()
            * (constant(INSTRUCTION_TYPE_WRITE) - instruction_type_next.clone())
            * (ram_value_next.clone() - ram_value);

        let bcbp0_only_changes_if_ram_pointer_changes =
            ram_pointer_changes.clone() * (bcpc0_next.clone() - bcpc0);

        let bcbp1_only_changes_if_ram_pointer_changes =
            ram_pointer_changes.clone() * (bcpc1_next.clone() - bcpc1);

        let running_product_ram_pointer_updates_correctly = ram_pointer_difference.clone()
            * (running_product_ram_pointer_next.clone()
                - running_product_ram_pointer.clone()
                    * (bezout_challenge.clone() - ram_pointer_next.clone()))
            + ram_pointer_changes.clone()
                * (running_product_ram_pointer_next - running_product_ram_pointer.clone());

        let formal_derivative_updates_correctly = ram_pointer_difference.clone()
            * (fd_next.clone()
                - running_product_ram_pointer
                - (bezout_challenge.clone() - ram_pointer_next.clone()) * fd.clone())
            + ram_pointer_changes.clone() * (fd_next - fd);

        let bezout_coefficient_0_is_constructed_correctly = ram_pointer_difference.clone()
            * (bc0_next.clone() - bezout_challenge.clone() * bc0.clone() - bcpc0_next)
            + ram_pointer_changes.clone() * (bc0_next - bc0);

        let bezout_coefficient_1_is_constructed_correctly = ram_pointer_difference.clone()
            * (bc1_next.clone() - bezout_challenge * bc1.clone() - bcpc1_next)
            + ram_pointer_changes.clone() * (bc1_next - bc1);

        let compressed_row = clock_next.clone() * challenge(RamClkWeight)
            + ram_pointer_next * challenge(RamPointerWeight)
            + ram_value_next * challenge(RamValueWeight)
            + instruction_type_next.clone() * challenge(RamInstructionTypeWeight);
        let rppa_accumulates_next_row =
            rppa_next.clone() - rppa.clone() * (challenge(RamIndeterminate) - compressed_row);

        let next_row_is_not_padding_row = (instruction_type_next.clone()
            - constant(INSTRUCTION_TYPE_READ))
            * (instruction_type_next - constant(INSTRUCTION_TYPE_WRITE));
        let rppa_remains_unchanged = rppa_next - rppa;

        let rppa_updates_correctly = rppa_accumulates_next_row * next_row_is_padding_row.clone()
            + rppa_remains_unchanged * next_row_is_not_padding_row.clone();

        let clock_difference = clock_next - clock;
        let log_derivative_accumulates = (clock_jump_diff_log_derivative_next.clone()
            - clock_jump_diff_log_derivative.clone())
            * (challenge(ClockJumpDifferenceLookupIndeterminate) - clock_difference)
            - one.clone();
        let log_derivative_remains =
            clock_jump_diff_log_derivative_next - clock_jump_diff_log_derivative.clone();

        let log_derivative_accumulates_or_ram_pointer_changes_or_next_row_is_padding_row =
            log_derivative_accumulates * ram_pointer_changes.clone() * next_row_is_padding_row;
        let log_derivative_remains_or_ram_pointer_doesnt_change =
            log_derivative_remains.clone() * ram_pointer_difference.clone();
        let log_derivative_remains_or_next_row_is_not_padding_row =
            log_derivative_remains * next_row_is_not_padding_row;

        let log_derivative_updates_correctly =
            log_derivative_accumulates_or_ram_pointer_changes_or_next_row_is_padding_row
                + log_derivative_remains_or_ram_pointer_doesnt_change
                + log_derivative_remains_or_next_row_is_not_padding_row;

        vec![
            if_current_row_is_padding_row_then_next_row_is_padding_row,
            iord_is_0_or_iord_is_inverse_of_ram_pointer_difference,
            ram_pointer_difference_is_0_or_iord_is_inverse_of_ram_pointer_difference,
            ram_pointer_changes_or_write_mem_or_ram_value_stays,
            bcbp0_only_changes_if_ram_pointer_changes,
            bcbp1_only_changes_if_ram_pointer_changes,
            running_product_ram_pointer_updates_correctly,
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
        let constant = |c: u32| circuit_builder.b_constant(c.into());
        let ext_row = |column: RamExtTableColumn| {
            circuit_builder.input(ExtRow(column.master_ext_table_index()))
        };

        let bezout_relation_holds = ext_row(BezoutCoefficient0) * ext_row(RunningProductOfRAMP)
            + ext_row(BezoutCoefficient1) * ext_row(FormalDerivative)
            - constant(1);

        vec![bezout_relation_holds]
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use assert2::assert;
    use assert2::check;
    use proptest_arbitrary_interop::arb;
    use test_strategy::proptest;

    use super::*;

    #[proptest]
    fn ram_table_call_can_be_converted_to_table_row(
        #[strategy(arb())] ram_table_call: RamTableCall,
    ) {
        let _ = ram_table_call.to_table_row();
    }

    pub fn check_constraints(
        master_base_trace_table: ArrayView2<BFieldElement>,
        master_ext_trace_table: ArrayView2<XFieldElement>,
        challenges: &Challenges,
    ) {
        assert!(master_base_trace_table.nrows() == master_ext_trace_table.nrows());

        let zero = XFieldElement::zero();
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
            check!(
                zero == evaluated_constraint,
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
                check!(
                    zero == evaluated_constraint,
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
                check!(
                    zero == evaluated_constraint,
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
            check!(
                zero == evaluated_constraint,
                "Terminal constraint {constraint_idx} failed."
            );
        }
    }
}
