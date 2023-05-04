use ndarray::s;
use ndarray::ArrayView2;
use ndarray::ArrayViewMut2;
use ndarray::Axis;
use ndarray::Zip;
use strum::EnumCount;
use strum_macros::Display;
use strum_macros::EnumCount as EnumCountMacro;
use strum_macros::EnumIter;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::x_field_element::XFieldElement;

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

impl Evaluable<BFieldElement> for ExtDegreeLoweringTable {}
impl Evaluable<XFieldElement> for ExtDegreeLoweringTable {}
impl Quotientable for ExtDegreeLoweringTable {}
