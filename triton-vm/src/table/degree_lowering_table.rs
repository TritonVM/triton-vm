use ndarray::s;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use ndarray::ArrayViewMut2;
use ndarray::Axis;
use ndarray::Zip;
use strum::EnumCount;
use strum_macros::Display;
use strum_macros::EnumCount as EnumCountMacro;
use strum_macros::EnumIter;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::mpolynomial::Degree;
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::table::challenges::Challenges;
use crate::table::extension_table::Evaluable;
use crate::table::extension_table::Quotientable;
use crate::table::master_table::NUM_BASE_COLUMNS;
use crate::table::master_table::NUM_EXT_COLUMNS;

pub const BASE_WIDTH: usize = DegreeLoweringBaseTableColumn::COUNT;
pub const EXT_WIDTH: usize = DegreeLoweringExtTableColumn::COUNT;
pub const FULL_WIDTH: usize = BASE_WIDTH + EXT_WIDTH;

#[repr(usize)]
#[derive(Display, Debug, Clone, Copy, PartialEq, Eq, EnumIter, EnumCountMacro, Hash)]
pub enum DegreeLoweringBaseTableColumn {
    Col1,
}

#[repr(usize)]
#[derive(Display, Debug, Clone, Copy, PartialEq, Eq, EnumIter, EnumCountMacro, Hash)]
pub enum DegreeLoweringExtTableColumn {
    XCol1,
}

#[derive(Debug, Clone)]
pub struct DegreeLoweringTable {}

#[derive(Debug, Clone)]
pub struct ExtDegreeLoweringTable {}

impl DegreeLoweringTable {
    pub fn fill_deterministic_base_columns(master_base_table: &mut ArrayViewMut2<BFieldElement>) {
        assert_eq!(NUM_BASE_COLUMNS, master_base_table.ncols());

        let main_trace_section_start = 0;
        let main_trace_section_end = main_trace_section_start + NUM_BASE_COLUMNS - BASE_WIDTH;
        let deterministic_section_start = main_trace_section_end;
        let deterministic_section_end = deterministic_section_start + BASE_WIDTH;

        let (main_trace_section, mut deterministic_section) = master_base_table.multi_slice_mut((
            s![.., main_trace_section_start..main_trace_section_end],
            s![.., deterministic_section_start..deterministic_section_end],
        ));

        Zip::from(main_trace_section.axis_iter(Axis(0)))
            .and(deterministic_section.axis_iter_mut(Axis(0)))
            .par_for_each(|main_trace_row, mut deterministic_row| {
                deterministic_row[0] = main_trace_row[0] * main_trace_row[1];
            });
    }

    pub fn fill_deterministic_ext_columns(
        master_base_table: ArrayView2<BFieldElement>,
        master_ext_table: &mut ArrayViewMut2<XFieldElement>,
    ) {
        assert_eq!(NUM_BASE_COLUMNS, master_base_table.ncols());
        assert_eq!(NUM_EXT_COLUMNS, master_ext_table.ncols());

        let main_ext_section_start = 0;
        let main_ext_section_end = main_ext_section_start + NUM_EXT_COLUMNS - EXT_WIDTH;
        let det_ext_section_start = main_ext_section_end;
        let det_ext_section_end = det_ext_section_start + EXT_WIDTH;

        let (main_ext_section, mut deterministic_section) = master_ext_table.multi_slice_mut((
            s![.., main_ext_section_start..main_ext_section_end],
            s![.., det_ext_section_start..det_ext_section_end],
        ));

        Zip::from(master_base_table.axis_iter(Axis(0)))
            .and(main_ext_section.axis_iter(Axis(0)))
            .and(deterministic_section.axis_iter_mut(Axis(0)))
            .par_for_each(|base_row, main_ext_row, mut det_ext_row| {
                det_ext_row[0] = base_row[0] * main_ext_row[0];
            });
    }
}

impl Evaluable<BFieldElement> for ExtDegreeLoweringTable {
    fn evaluate_initial_constraints(
        _base_row: ArrayView1<BFieldElement>,
        _ext_row: ArrayView1<XFieldElement>,
        _challenges: &Challenges,
    ) -> Vec<XFieldElement> {
        vec![]
    }

    fn evaluate_consistency_constraints(
        base_row: ArrayView1<BFieldElement>,
        ext_row: ArrayView1<XFieldElement>,
        _challenges: &Challenges,
    ) -> Vec<XFieldElement> {
        let deterministic_base_section_start = NUM_BASE_COLUMNS - BASE_WIDTH;
        let deterministic_ext_section_start = NUM_EXT_COLUMNS - EXT_WIDTH;

        let b_dummy_0 = base_row[deterministic_base_section_start] - base_row[0] * base_row[1];
        let x_dummy_0 = ext_row[deterministic_ext_section_start] - base_row[0] * ext_row[0];

        let base_constraints = [b_dummy_0];
        let ext_constraints = [x_dummy_0];

        base_constraints
            .map(|b| b.lift())
            .into_iter()
            .chain(ext_constraints)
            .collect()
    }

    fn evaluate_transition_constraints(
        _current_base_row: ArrayView1<BFieldElement>,
        _current_ext_row: ArrayView1<XFieldElement>,
        _next_base_row: ArrayView1<BFieldElement>,
        _next_ext_row: ArrayView1<XFieldElement>,
        _challenges: &Challenges,
    ) -> Vec<XFieldElement> {
        vec![]
    }

    fn evaluate_terminal_constraints(
        _base_row: ArrayView1<BFieldElement>,
        _ext_row: ArrayView1<XFieldElement>,
        _challenges: &Challenges,
    ) -> Vec<XFieldElement> {
        vec![]
    }
}

impl Evaluable<XFieldElement> for ExtDegreeLoweringTable {
    fn evaluate_initial_constraints(
        _base_row: ArrayView1<XFieldElement>,
        _ext_row: ArrayView1<XFieldElement>,
        _challenges: &Challenges,
    ) -> Vec<XFieldElement> {
        vec![]
    }

    fn evaluate_consistency_constraints(
        base_row: ArrayView1<XFieldElement>,
        ext_row: ArrayView1<XFieldElement>,
        _challenges: &Challenges,
    ) -> Vec<XFieldElement> {
        let deterministic_base_section_start = NUM_BASE_COLUMNS - BASE_WIDTH;
        let deterministic_ext_section_start = NUM_EXT_COLUMNS - EXT_WIDTH;

        let b_dummy_0 = base_row[deterministic_base_section_start] - base_row[0] * base_row[1];
        let x_dummy_0 = ext_row[deterministic_ext_section_start] - base_row[0] * ext_row[0];

        let base_constraints = [b_dummy_0];
        let ext_constraints = [x_dummy_0];

        base_constraints
            .into_iter()
            .chain(ext_constraints)
            .collect()
    }

    fn evaluate_transition_constraints(
        _current_base_row: ArrayView1<XFieldElement>,
        _current_ext_row: ArrayView1<XFieldElement>,
        _next_base_row: ArrayView1<XFieldElement>,
        _next_ext_row: ArrayView1<XFieldElement>,
        _challenges: &Challenges,
    ) -> Vec<XFieldElement> {
        vec![]
    }

    fn evaluate_terminal_constraints(
        _base_row: ArrayView1<XFieldElement>,
        _ext_row: ArrayView1<XFieldElement>,
        _challenges: &Challenges,
    ) -> Vec<XFieldElement> {
        vec![]
    }
}

impl Quotientable for ExtDegreeLoweringTable {
    fn num_initial_quotients() -> usize {
        0
    }

    fn num_consistency_quotients() -> usize {
        2
    }

    fn num_transition_quotients() -> usize {
        0
    }

    fn num_terminal_quotients() -> usize {
        0
    }

    fn initial_quotient_degree_bounds(_interpolant_degree: Degree) -> Vec<Degree> {
        vec![]
    }

    fn consistency_quotient_degree_bounds(
        interpolant_degree: Degree,
        padded_height: usize,
    ) -> Vec<Degree> {
        let zerofier_degree = padded_height as Degree;
        [
            interpolant_degree * 2 - zerofier_degree,
            interpolant_degree * 2 - zerofier_degree,
        ]
        .to_vec()
    }

    fn transition_quotient_degree_bounds(
        _interpolant_degree: Degree,
        _padded_height: usize,
    ) -> Vec<Degree> {
        vec![]
    }

    fn terminal_quotient_degree_bounds(_interpolant_degree: Degree) -> Vec<Degree> {
        vec![]
    }
}
