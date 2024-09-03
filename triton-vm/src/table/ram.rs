use std::cmp::Ordering;

use air::challenge_id::ChallengeId::*;
use air::cross_table_argument::*;
use air::table::ram::RamTable;
use air::table::ram::PADDING_INDICATOR;
use air::table::TableId;
use air::table_column::RamBaseTableColumn::*;
use air::table_column::*;
use air::AIR;
use arbitrary::Arbitrary;
use itertools::Itertools;
use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use num_traits::ConstOne;
use num_traits::One;
use num_traits::Zero;
use serde::Deserialize;
use serde::Serialize;
use strum::EnumCount;
use strum::IntoEnumIterator;
use twenty_first::math::traits::FiniteField;
use twenty_first::prelude::*;

use crate::aet::AlgebraicExecutionTrace;
use crate::challenges::Challenges;
use crate::ndarray_helper::contiguous_column_slices;
use crate::ndarray_helper::horizontal_multi_slice_mut;
use crate::profiler::profiler;
use crate::table::TraceTable;

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, Arbitrary)]
pub struct RamTableCall {
    pub clk: u32,
    pub ram_pointer: BFieldElement,
    pub ram_value: BFieldElement,
    pub is_write: bool,
}

impl RamTableCall {
    pub fn to_table_row(self) -> Array1<BFieldElement> {
        let instruction_type = if self.is_write {
            air::table::ram::INSTRUCTION_TYPE_WRITE
        } else {
            air::table::ram::INSTRUCTION_TYPE_READ
        };

        let mut row = Array1::zeros(<RamTable as AIR>::MainColumn::COUNT);
        row[CLK.base_table_index()] = self.clk.into();
        row[InstructionType.base_table_index()] = instruction_type;
        row[RamPointer.base_table_index()] = self.ram_pointer;
        row[RamValue.base_table_index()] = self.ram_value;
        row
    }
}

impl TraceTable for RamTable {
    type FillParam = ();
    type FillReturnInfo = Vec<BFieldElement>;

    fn fill(
        mut ram_table: ArrayViewMut2<BFieldElement>,
        aet: &AlgebraicExecutionTrace,
        _: Self::FillParam,
    ) -> Self::FillReturnInfo {
        let mut ram_table = ram_table.slice_mut(s![0..aet.height_of_table(TableId::Ram), ..]);
        let trace_iter = aet.ram_trace.rows().into_iter();

        let sorted_rows =
            trace_iter.sorted_by(|row_0, row_1| compare_rows(row_0.view(), row_1.view()));
        for (row_index, row) in sorted_rows.enumerate() {
            ram_table.row_mut(row_index).assign(&row);
        }

        let all_ram_pointers = ram_table.column(RamPointer.base_table_index());
        let unique_ram_pointers = all_ram_pointers.iter().unique().copied().collect_vec();
        let (bezout_0, bezout_1) =
            bezout_coefficient_polynomials_coefficients(&unique_ram_pointers);

        make_ram_table_consistent(&mut ram_table, bezout_0, bezout_1)
    }

    fn pad(mut main_table: ArrayViewMut2<BFieldElement>, table_len: usize) {
        let last_row_index = table_len.saturating_sub(1);
        let mut padding_row = main_table.row(last_row_index).to_owned();
        padding_row[InstructionType.base_table_index()] = PADDING_INDICATOR;
        if table_len == 0 {
            padding_row[BezoutCoefficientPolynomialCoefficient1.base_table_index()] =
                BFieldElement::ONE;
        }

        let mut padding_section = main_table.slice_mut(s![table_len.., ..]);
        padding_section
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut row| row.assign(&padding_row));
    }

    fn extend(
        base_table: ArrayView2<BFieldElement>,
        mut ext_table: ArrayViewMut2<XFieldElement>,
        challenges: &Challenges,
    ) {
        profiler!(start "ram table");
        assert_eq!(Self::MainColumn::COUNT, base_table.ncols());
        assert_eq!(Self::AuxColumn::COUNT, ext_table.ncols());
        assert_eq!(base_table.nrows(), ext_table.nrows());

        let extension_column_indices = RamExtTableColumn::iter()
            // RunningProductOfRAMP + FormalDerivative are constitute one
            // slice and are populated by the same function
            .filter(|column| *column != RamExtTableColumn::FormalDerivative)
            .map(|column| column.ext_table_index())
            .collect_vec();
        let extension_column_slices = horizontal_multi_slice_mut(
            ext_table.view_mut(),
            &contiguous_column_slices(&extension_column_indices),
        );
        let extension_functions = [
            extension_column_running_product_of_ramp_and_formal_derivative,
            extension_column_bezout_coefficient_0,
            extension_column_bezout_coefficient_1,
            extension_column_running_product_perm_arg,
            extension_column_clock_jump_difference_lookup_log_derivative,
        ];
        extension_functions
            .into_par_iter()
            .zip_eq(extension_column_slices)
            .for_each(|(generator, slice)| {
                generator(base_table, challenges).move_into(slice);
            });

        profiler!(stop "ram table");
    }
}

fn compare_rows(row_0: ArrayView1<BFieldElement>, row_1: ArrayView1<BFieldElement>) -> Ordering {
    let ram_pointer_0 = row_0[RamPointer.base_table_index()].value();
    let ram_pointer_1 = row_1[RamPointer.base_table_index()].value();
    let compare_ram_pointers = ram_pointer_0.cmp(&ram_pointer_1);

    let clk_0 = row_0[CLK.base_table_index()].value();
    let clk_1 = row_1[CLK.base_table_index()].value();
    let compare_clocks = clk_0.cmp(&clk_1);

    compare_ram_pointers.then(compare_clocks)
}

/// Compute the [Bézout coefficients](https://en.wikipedia.org/wiki/B%C3%A9zout%27s_identity)
/// of the polynomial with the given roots and its formal derivative.
///
/// All roots _must_ be unique. That is, the corresponding polynomial must be square free.
#[doc(hidden)] // public for benchmarking purposes only
pub fn bezout_coefficient_polynomials_coefficients(
    unique_roots: &[BFieldElement],
) -> (Vec<BFieldElement>, Vec<BFieldElement>) {
    if unique_roots.is_empty() {
        return (vec![], vec![]);
    }

    // The structure of the problem is exploited heavily to compute the Bézout coefficients
    // as fast as possible. In the following paragraphs, let `rp` denote the polynomial with the
    // given `unique_roots` as its roots, and `fd` the formal derivative of `rp`.
    //
    // The naïve approach is to perform the extended Euclidean algorithm (xgcd) on `rp` and
    // `fd`. This has a time complexity in O(n^2) where `n` is the number of roots: for the
    // given problem shape, the degrees `rp` and `fd` are `n` and `n-1`, respectively. Each step
    // of the (x)gcd takes O(n) time and reduces the degree of the polynomials by one.
    // For programs with a large number of different RAM accesses, `n` is large.
    //
    // The approach taken here is to exploit the structure of the problem. Concretely, since all
    // roots of `rp` are unique, _i.e._, `rp` is square free, the gcd of `rp` and `fd` is 1.
    // This implies `∀ r ∈ unique_roots: fd(r)·b(r) = 1`, where `b` is one of the Bézout
    // coefficients. In other words, the evaluation of `fd` in `unique_roots` is the inverse of
    // the evaluation of `b` in `unique_roots`. Furthermore, `b` is a polynomial of degree `n`,
    // and therefore fully determined by the evaluations in `unique_roots`. Finally, the other
    // Bézout coefficient `a` is determined by `a = (1 - fd·b) / rp`.
    // In total, this allows computing the Bézout coefficients in O(n·(log n)^2) time.

    debug_assert!(unique_roots.iter().all_unique());
    let rp = Polynomial::par_zerofier(unique_roots);
    let fd = rp.formal_derivative();
    let fd_in_roots = fd.par_batch_evaluate(unique_roots);
    let b_in_roots = BFieldElement::batch_inversion(fd_in_roots);
    let b = Polynomial::par_interpolate(unique_roots, &b_in_roots);
    let one_minus_fd_b = Polynomial::one() - fd.multiply(&b);
    let a = one_minus_fd_b.clean_divide(rp);

    let mut coefficients_0 = a.coefficients;
    let mut coefficients_1 = b.coefficients;
    coefficients_0.resize(unique_roots.len(), bfe!(0));
    coefficients_1.resize(unique_roots.len(), bfe!(0));
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

fn extension_column_running_product_of_ramp_and_formal_derivative(
    base_table: ArrayView2<BFieldElement>,
    challenges: &Challenges,
) -> Array2<XFieldElement> {
    let bezout_indeterminate = challenges[RamTableBezoutRelationIndeterminate];

    let mut extension_columns = Vec::with_capacity(2 * base_table.nrows());
    let mut running_product_ram_pointer =
        bezout_indeterminate - base_table.row(0)[RamPointer.base_table_index()];
    let mut formal_derivative = xfe!(1);

    extension_columns.push(running_product_ram_pointer);
    extension_columns.push(formal_derivative);

    for (previous_row, current_row) in base_table.rows().into_iter().tuple_windows() {
        let instruction_type = current_row[InstructionType.base_table_index()];
        let is_no_padding_row = instruction_type != PADDING_INDICATOR;

        if is_no_padding_row {
            let current_ram_pointer = current_row[RamPointer.base_table_index()];
            let previous_ram_pointer = previous_row[RamPointer.base_table_index()];
            if previous_ram_pointer != current_ram_pointer {
                formal_derivative = (bezout_indeterminate - current_ram_pointer)
                    * formal_derivative
                    + running_product_ram_pointer;
                running_product_ram_pointer *= bezout_indeterminate - current_ram_pointer;
            }
        }

        extension_columns.push(running_product_ram_pointer);
        extension_columns.push(formal_derivative);
    }

    Array2::from_shape_vec((base_table.nrows(), 2), extension_columns).unwrap()
}

fn extension_column_bezout_coefficient_0(
    base_table: ArrayView2<BFieldElement>,
    challenges: &Challenges,
) -> Array2<XFieldElement> {
    extension_column_bezout_coefficient(
        base_table,
        challenges,
        BezoutCoefficientPolynomialCoefficient0,
    )
}

fn extension_column_bezout_coefficient_1(
    base_table: ArrayView2<BFieldElement>,
    challenges: &Challenges,
) -> Array2<XFieldElement> {
    extension_column_bezout_coefficient(
        base_table,
        challenges,
        BezoutCoefficientPolynomialCoefficient1,
    )
}

fn extension_column_bezout_coefficient(
    base_table: ArrayView2<BFieldElement>,
    challenges: &Challenges,
    bezout_cefficient_column: RamBaseTableColumn,
) -> Array2<XFieldElement> {
    let bezout_indeterminate = challenges[RamTableBezoutRelationIndeterminate];

    let mut bezout_coefficient =
        base_table.row(0)[bezout_cefficient_column.base_table_index()].lift();
    let mut extension_column = Vec::with_capacity(base_table.nrows());
    extension_column.push(bezout_coefficient);

    for (previous_row, current_row) in base_table.rows().into_iter().tuple_windows() {
        if current_row[InstructionType.base_table_index()] == PADDING_INDICATOR {
            break; // padding marks the end of the trace
        }

        let previous_ram_pointer = previous_row[RamPointer.base_table_index()];
        let current_ram_pointer = current_row[RamPointer.base_table_index()];
        if previous_ram_pointer != current_ram_pointer {
            bezout_coefficient *= bezout_indeterminate;
            bezout_coefficient += current_row[bezout_cefficient_column.base_table_index()];
        }
        extension_column.push(bezout_coefficient);
    }

    // fill padding section
    extension_column.resize(base_table.nrows(), bezout_coefficient);
    Array2::from_shape_vec((base_table.nrows(), 1), extension_column).unwrap()
}

fn extension_column_running_product_perm_arg(
    base_table: ArrayView2<BFieldElement>,
    challenges: &Challenges,
) -> Array2<XFieldElement> {
    let mut running_product_for_perm_arg = PermArg::default_initial();
    let mut extension_column = Vec::with_capacity(base_table.nrows());
    for row in base_table.rows() {
        let instruction_type = row[InstructionType.base_table_index()];
        if instruction_type == PADDING_INDICATOR {
            break; // padding marks the end of the trace
        }

        let clk = row[CLK.base_table_index()];
        let current_ram_pointer = row[RamPointer.base_table_index()];
        let ram_value = row[RamValue.base_table_index()];
        let compressed_row = clk * challenges[RamClkWeight]
            + instruction_type * challenges[RamInstructionTypeWeight]
            + current_ram_pointer * challenges[RamPointerWeight]
            + ram_value * challenges[RamValueWeight];
        running_product_for_perm_arg *= challenges[RamIndeterminate] - compressed_row;
        extension_column.push(running_product_for_perm_arg);
    }

    // fill padding section
    extension_column.resize(base_table.nrows(), running_product_for_perm_arg);
    Array2::from_shape_vec((base_table.nrows(), 1), extension_column).unwrap()
}

fn extension_column_clock_jump_difference_lookup_log_derivative(
    base_table: ArrayView2<BFieldElement>,
    challenges: &Challenges,
) -> Array2<XFieldElement> {
    let indeterminate = challenges[ClockJumpDifferenceLookupIndeterminate];

    let mut cjd_lookup_log_derivative = LookupArg::default_initial();
    let mut extension_column = Vec::with_capacity(base_table.nrows());
    extension_column.push(cjd_lookup_log_derivative);

    for (previous_row, current_row) in base_table.rows().into_iter().tuple_windows() {
        if current_row[InstructionType.base_table_index()] == PADDING_INDICATOR {
            break; // padding marks the end of the trace
        }

        let previous_ram_pointer = previous_row[RamPointer.base_table_index()];
        let current_ram_pointer = current_row[RamPointer.base_table_index()];
        if previous_ram_pointer == current_ram_pointer {
            let previous_clock = previous_row[CLK.base_table_index()];
            let current_clock = current_row[CLK.base_table_index()];
            let clock_jump_difference = current_clock - previous_clock;
            let log_derivative_summand = (indeterminate - clock_jump_difference).inverse();
            cjd_lookup_log_derivative += log_derivative_summand;
        }
        extension_column.push(cjd_lookup_log_derivative);
    }

    // fill padding section
    extension_column.resize(base_table.nrows(), cjd_lookup_log_derivative);
    Array2::from_shape_vec((base_table.nrows(), 1), extension_column).unwrap()
}

#[cfg(test)]
pub(crate) mod tests {
    use proptest::prelude::*;
    use proptest_arbitrary_interop::arb;
    use test_strategy::proptest;

    use super::*;

    #[proptest]
    fn ram_table_call_can_be_converted_to_table_row(
        #[strategy(arb())] ram_table_call: RamTableCall,
    ) {
        ram_table_call.to_table_row();
    }

    #[test]
    fn bezout_coefficient_polynomials_of_empty_ram_table_are_default() {
        let (a, b) = bezout_coefficient_polynomials_coefficients(&[]);
        assert_eq!(a, vec![]);
        assert_eq!(b, vec![]);
    }

    #[test]
    fn bezout_coefficient_polynomials_are_as_expected() {
        let rp = bfe_array![1, 2, 3];
        let (a, b) = bezout_coefficient_polynomials_coefficients(&rp);

        let expected_a = bfe_array![9, 0x7fff_ffff_7fff_fffc_u64, 0];
        let expected_b = bfe_array![5, 0xffff_fffe_ffff_fffb_u64, 0x7fff_ffff_8000_0002_u64];

        assert_eq!(expected_a, *a);
        assert_eq!(expected_b, *b);
    }

    #[proptest]
    fn bezout_coefficient_polynomials_agree_with_xgcd(
        #[strategy(arb())]
        #[filter(#ram_pointers.iter().all_unique())]
        ram_pointers: Vec<BFieldElement>,
    ) {
        let (a, b) = bezout_coefficient_polynomials_coefficients(&ram_pointers);

        let rp = Polynomial::zerofier(&ram_pointers);
        let fd = rp.formal_derivative();
        let (_, a_xgcd, b_xgcd) = Polynomial::xgcd(rp, fd);

        let mut a_xgcd = a_xgcd.coefficients;
        let mut b_xgcd = b_xgcd.coefficients;

        a_xgcd.resize(ram_pointers.len(), bfe!(0));
        b_xgcd.resize(ram_pointers.len(), bfe!(0));

        prop_assert_eq!(a, a_xgcd);
        prop_assert_eq!(b, b_xgcd);
    }

    #[proptest]
    fn bezout_coefficients_are_actually_bezout_coefficients(
        #[strategy(arb())]
        #[filter(!#ram_pointers.is_empty())]
        #[filter(#ram_pointers.iter().all_unique())]
        ram_pointers: Vec<BFieldElement>,
    ) {
        let (a, b) = bezout_coefficient_polynomials_coefficients(&ram_pointers);

        let rp = Polynomial::zerofier(&ram_pointers);
        let fd = rp.formal_derivative();

        let [a, b] = [a, b].map(Polynomial::new);
        let gcd = rp * a + fd * b;
        prop_assert_eq!(Polynomial::one(), gcd);
    }
}
