use std::cmp::Ordering;

use air::challenge_id::ChallengeId;
use air::cross_table_argument::CrossTableArg;
use air::cross_table_argument::LookupArg;
use air::cross_table_argument::PermArg;
use air::table::TableId;
use air::table::ram::PADDING_INDICATOR;
use air::table::ram::RamTable;
use air::table_column::MasterAuxColumn;
use air::table_column::MasterMainColumn;
use arbitrary::Arbitrary;
use itertools::Itertools;
use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use num_traits::ConstOne;
use num_traits::ConstZero;
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
use crate::ndarray_helper::ROW_AXIS;
use crate::ndarray_helper::contiguous_column_slices;
use crate::ndarray_helper::horizontal_multi_slice_mut;
use crate::profiler::profiler;
use crate::table::TraceTable;

type MainColumn = <RamTable as air::AIR>::MainColumn;
type AuxColumn = <RamTable as air::AIR>::AuxColumn;

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

        let mut row = Array1::zeros(MainColumn::COUNT);
        row[MainColumn::CLK.main_index()] = self.clk.into();
        row[MainColumn::InstructionType.main_index()] = instruction_type;
        row[MainColumn::RamPointer.main_index()] = self.ram_pointer;
        row[MainColumn::RamValue.main_index()] = self.ram_value;
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

        let all_ram_pointers = ram_table.column(MainColumn::RamPointer.main_index());
        let unique_ram_pointers = all_ram_pointers.iter().unique().copied().collect_vec();
        let (bezout_0, bezout_1) =
            bezout_coefficient_polynomials_coefficients(&unique_ram_pointers);

        make_ram_table_consistent(&mut ram_table, bezout_0, bezout_1)
    }

    fn pad(mut main_table: ArrayViewMut2<BFieldElement>, table_len: usize) {
        let last_row_index = table_len.saturating_sub(1);
        let mut padding_row = main_table.row(last_row_index).to_owned();
        padding_row[MainColumn::InstructionType.main_index()] = PADDING_INDICATOR;
        if table_len == 0 {
            padding_row[MainColumn::BezoutCoefficientPolynomialCoefficient1.main_index()] =
                BFieldElement::ONE;
        }

        let mut padding_section = main_table.slice_mut(s![table_len.., ..]);
        padding_section
            .axis_iter_mut(ROW_AXIS)
            .into_par_iter()
            .for_each(|mut row| row.assign(&padding_row));
    }

    fn extend(
        main_table: ArrayView2<BFieldElement>,
        mut aux_table: ArrayViewMut2<XFieldElement>,
        challenges: &Challenges,
    ) {
        profiler!(start "ram table");
        assert_eq!(MainColumn::COUNT, main_table.ncols());
        assert_eq!(AuxColumn::COUNT, aux_table.ncols());
        assert_eq!(main_table.nrows(), aux_table.nrows());

        let auxiliary_column_indices = AuxColumn::iter()
            // RunningProductOfRAMP + FormalDerivative are constitute one
            // slice and are populated by the same function
            .filter(|column| *column != AuxColumn::FormalDerivative)
            .map(|column| column.aux_index())
            .collect_vec();
        let auxiliary_column_slices = horizontal_multi_slice_mut(
            aux_table.view_mut(),
            &contiguous_column_slices(&auxiliary_column_indices),
        );
        let extension_functions = [
            auxiliary_column_running_product_of_ramp_and_formal_derivative,
            auxiliary_column_bezout_coefficient_0,
            auxiliary_column_bezout_coefficient_1,
            auxiliary_column_running_product_perm_arg,
            auxiliary_column_clock_jump_difference_lookup_log_derivative,
        ];
        extension_functions
            .into_par_iter()
            .zip_eq(auxiliary_column_slices)
            .for_each(|(generator, slice)| {
                generator(main_table, challenges).move_into(slice);
            });

        profiler!(stop "ram table");
    }
}

fn compare_rows(row_0: ArrayView1<BFieldElement>, row_1: ArrayView1<BFieldElement>) -> Ordering {
    let ram_pointer_0 = row_0[MainColumn::RamPointer.main_index()].value();
    let ram_pointer_1 = row_1[MainColumn::RamPointer.main_index()].value();
    let compare_ram_pointers = ram_pointer_0.cmp(&ram_pointer_1);

    let clk_0 = row_0[MainColumn::CLK.main_index()].value();
    let clk_1 = row_1[MainColumn::CLK.main_index()].value();
    let compare_clocks = clk_0.cmp(&clk_1);

    compare_ram_pointers.then(compare_clocks)
}

/// Compute the
/// [Bézout coefficients](https://en.wikipedia.org/wiki/B%C3%A9zout%27s_identity)
/// of the polynomial with the given roots and its formal derivative.
///
/// All roots _must_ be unique. That is, the corresponding polynomial must be
/// square free.
#[doc(hidden)] // public for benchmarking purposes only
pub fn bezout_coefficient_polynomials_coefficients(
    unique_roots: &[BFieldElement],
) -> (Vec<BFieldElement>, Vec<BFieldElement>) {
    if unique_roots.is_empty() {
        return (vec![], vec![]);
    }

    // The structure of the problem is exploited heavily to compute the Bézout
    // coefficients as fast as possible. In the following paragraphs, let `rp`
    // denote the polynomial with the given `unique_roots` as its roots, and
    // `fd` the formal derivative of `rp`.
    //
    // The naïve approach is to perform the extended Euclidean algorithm (xgcd)
    // on `rp` and `fd`. This has a time complexity in O(n^2) where `n` is the
    // number of roots: for the given problem shape, the degrees `rp` and `fd`
    // are `n` and `n-1`, respectively. Each step of the (x)gcd takes O(n) time
    // and reduces the degree of the polynomials by one. For programs with a
    // large number of different RAM accesses, `n` is large.
    //
    // The approach taken here is to exploit the structure of the problem.
    // Concretely, since all roots of `rp` are unique, _i.e._, `rp` is square
    // free, the gcd of `rp` and `fd` is 1. This implies `∀ r ∈ unique_roots:
    // fd(r)·b(r) = 1`, where `b` is one of the Bézout coefficients. In other
    // words, the evaluation of `fd` in `unique_roots` is the inverse of
    // the evaluation of `b` in `unique_roots`. Furthermore, `b` is a polynomial
    // of degree `n`, and therefore fully determined by the evaluations in
    // `unique_roots`. Finally, the other Bézout coefficient `a` is determined
    // by `a = (1 - fd·b) / rp`. In total, this allows computing the Bézout
    // coefficients in O(n·(log n)^2) time.

    debug_assert!(unique_roots.iter().all_unique());
    let rp = Polynomial::par_zerofier(unique_roots);
    let fd = rp.formal_derivative();
    let fd_in_roots = fd.par_batch_evaluate(unique_roots);
    let b_in_roots = BFieldElement::batch_inversion(fd_in_roots);
    let b = Polynomial::par_interpolate(unique_roots, &b_in_roots);
    let one_minus_fd_b = Polynomial::one() - fd.multiply(&b);
    let a = one_minus_fd_b.clean_divide(rp);

    let mut coefficients_0 = a.into_coefficients();
    let mut coefficients_1 = b.into_coefficients();
    coefficients_0.resize(unique_roots.len(), BFieldElement::ZERO);
    coefficients_1.resize(unique_roots.len(), BFieldElement::ZERO);
    (coefficients_0, coefficients_1)
}

/// - Set inverse of RAM pointer difference
/// - Fill in the Bézout coefficients if the RAM pointer changes between two
///   consecutive rows
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
    ram_table.row_mut(0)[MainColumn::BezoutCoefficientPolynomialCoefficient0.main_index()] =
        current_bcpc_0;
    ram_table.row_mut(0)[MainColumn::BezoutCoefficientPolynomialCoefficient1.main_index()] =
        current_bcpc_1;

    let mut clock_jump_differences = vec![];
    for row_idx in 0..ram_table.nrows() - 1 {
        let (mut curr_row, mut next_row) =
            ram_table.multi_slice_mut((s![row_idx, ..], s![row_idx + 1, ..]));

        let ramp_diff = next_row[MainColumn::RamPointer.main_index()]
            - curr_row[MainColumn::RamPointer.main_index()];
        let clk_diff =
            next_row[MainColumn::CLK.main_index()] - curr_row[MainColumn::CLK.main_index()];

        if ramp_diff.is_zero() {
            clock_jump_differences.push(clk_diff);
        } else {
            current_bcpc_0 = bezout_coefficient_polynomial_coefficients_0.pop().unwrap();
            current_bcpc_1 = bezout_coefficient_polynomial_coefficients_1.pop().unwrap();
        }

        curr_row[MainColumn::InverseOfRampDifference.main_index()] = ramp_diff.inverse_or_zero();
        next_row[MainColumn::BezoutCoefficientPolynomialCoefficient0.main_index()] = current_bcpc_0;
        next_row[MainColumn::BezoutCoefficientPolynomialCoefficient1.main_index()] = current_bcpc_1;
    }

    assert_eq!(0, bezout_coefficient_polynomial_coefficients_0.len());
    assert_eq!(0, bezout_coefficient_polynomial_coefficients_1.len());
    clock_jump_differences
}

fn auxiliary_column_running_product_of_ramp_and_formal_derivative(
    main_table: ArrayView2<BFieldElement>,
    challenges: &Challenges,
) -> Array2<XFieldElement> {
    let bezout_indeterminate = challenges[ChallengeId::RamTableBezoutRelationIndeterminate];

    let mut auxiliary_columns = Vec::with_capacity(2 * main_table.nrows());
    let mut running_product_ram_pointer =
        bezout_indeterminate - main_table.row(0)[MainColumn::RamPointer.main_index()];
    let mut formal_derivative = xfe!(1);

    auxiliary_columns.push(running_product_ram_pointer);
    auxiliary_columns.push(formal_derivative);

    for (previous_row, current_row) in main_table.rows().into_iter().tuple_windows() {
        let instruction_type = current_row[MainColumn::InstructionType.main_index()];
        let is_no_padding_row = instruction_type != PADDING_INDICATOR;

        if is_no_padding_row {
            let current_ram_pointer = current_row[MainColumn::RamPointer.main_index()];
            let previous_ram_pointer = previous_row[MainColumn::RamPointer.main_index()];
            if previous_ram_pointer != current_ram_pointer {
                formal_derivative = (bezout_indeterminate - current_ram_pointer)
                    * formal_derivative
                    + running_product_ram_pointer;
                running_product_ram_pointer *= bezout_indeterminate - current_ram_pointer;
            }
        }

        auxiliary_columns.push(running_product_ram_pointer);
        auxiliary_columns.push(formal_derivative);
    }

    Array2::from_shape_vec((main_table.nrows(), 2), auxiliary_columns).unwrap()
}

fn auxiliary_column_bezout_coefficient_0(
    main_table: ArrayView2<BFieldElement>,
    challenges: &Challenges,
) -> Array2<XFieldElement> {
    auxiliary_column_bezout_coefficient(
        main_table,
        challenges,
        MainColumn::BezoutCoefficientPolynomialCoefficient0,
    )
}

fn auxiliary_column_bezout_coefficient_1(
    main_table: ArrayView2<BFieldElement>,
    challenges: &Challenges,
) -> Array2<XFieldElement> {
    auxiliary_column_bezout_coefficient(
        main_table,
        challenges,
        MainColumn::BezoutCoefficientPolynomialCoefficient1,
    )
}

fn auxiliary_column_bezout_coefficient(
    main_table: ArrayView2<BFieldElement>,
    challenges: &Challenges,
    bezout_cefficient_column: MainColumn,
) -> Array2<XFieldElement> {
    let bezout_indeterminate = challenges[ChallengeId::RamTableBezoutRelationIndeterminate];

    let mut bezout_coefficient = main_table.row(0)[bezout_cefficient_column.main_index()].lift();
    let mut auxiliary_column = Vec::with_capacity(main_table.nrows());
    auxiliary_column.push(bezout_coefficient);

    for (previous_row, current_row) in main_table.rows().into_iter().tuple_windows() {
        if current_row[MainColumn::InstructionType.main_index()] == PADDING_INDICATOR {
            break; // padding marks the end of the trace
        }

        let previous_ram_pointer = previous_row[MainColumn::RamPointer.main_index()];
        let current_ram_pointer = current_row[MainColumn::RamPointer.main_index()];
        if previous_ram_pointer != current_ram_pointer {
            bezout_coefficient *= bezout_indeterminate;
            bezout_coefficient += current_row[bezout_cefficient_column.main_index()];
        }
        auxiliary_column.push(bezout_coefficient);
    }

    // fill padding section
    auxiliary_column.resize(main_table.nrows(), bezout_coefficient);
    Array2::from_shape_vec((main_table.nrows(), 1), auxiliary_column).unwrap()
}

fn auxiliary_column_running_product_perm_arg(
    main_table: ArrayView2<BFieldElement>,
    challenges: &Challenges,
) -> Array2<XFieldElement> {
    let mut running_product_for_perm_arg = PermArg::default_initial();
    let mut auxiliary_column = Vec::with_capacity(main_table.nrows());
    for row in main_table.rows() {
        let instruction_type = row[MainColumn::InstructionType.main_index()];
        if instruction_type == PADDING_INDICATOR {
            break; // padding marks the end of the trace
        }

        let clk = row[MainColumn::CLK.main_index()];
        let current_ram_pointer = row[MainColumn::RamPointer.main_index()];
        let ram_value = row[MainColumn::RamValue.main_index()];
        let compressed_row = clk * challenges[ChallengeId::RamClkWeight]
            + instruction_type * challenges[ChallengeId::RamInstructionTypeWeight]
            + current_ram_pointer * challenges[ChallengeId::RamPointerWeight]
            + ram_value * challenges[ChallengeId::RamValueWeight];
        running_product_for_perm_arg *= challenges[ChallengeId::RamIndeterminate] - compressed_row;
        auxiliary_column.push(running_product_for_perm_arg);
    }

    // fill padding section
    auxiliary_column.resize(main_table.nrows(), running_product_for_perm_arg);
    Array2::from_shape_vec((main_table.nrows(), 1), auxiliary_column).unwrap()
}

fn auxiliary_column_clock_jump_difference_lookup_log_derivative(
    main_table: ArrayView2<BFieldElement>,
    challenges: &Challenges,
) -> Array2<XFieldElement> {
    let indeterminate = challenges[ChallengeId::ClockJumpDifferenceLookupIndeterminate];

    let mut cjd_lookup_log_derivative = LookupArg::default_initial();
    let mut auxiliary_column = Vec::with_capacity(main_table.nrows());
    auxiliary_column.push(cjd_lookup_log_derivative);

    for (previous_row, current_row) in main_table.rows().into_iter().tuple_windows() {
        if current_row[MainColumn::InstructionType.main_index()] == PADDING_INDICATOR {
            break; // padding marks the end of the trace
        }

        let previous_ram_pointer = previous_row[MainColumn::RamPointer.main_index()];
        let current_ram_pointer = current_row[MainColumn::RamPointer.main_index()];
        if previous_ram_pointer == current_ram_pointer {
            let previous_clock = previous_row[MainColumn::CLK.main_index()];
            let current_clock = current_row[MainColumn::CLK.main_index()];
            let clock_jump_difference = current_clock - previous_clock;
            let log_derivative_summand = (indeterminate - clock_jump_difference).inverse();
            cjd_lookup_log_derivative += log_derivative_summand;
        }
        auxiliary_column.push(cjd_lookup_log_derivative);
    }

    // fill padding section
    auxiliary_column.resize(main_table.nrows(), cjd_lookup_log_derivative);
    Array2::from_shape_vec((main_table.nrows(), 1), auxiliary_column).unwrap()
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
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

        let mut a_xgcd = a_xgcd.into_coefficients();
        let mut b_xgcd = b_xgcd.into_coefficients();

        a_xgcd.resize(ram_pointers.len(), BFieldElement::ZERO);
        b_xgcd.resize(ram_pointers.len(), BFieldElement::ZERO);

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
