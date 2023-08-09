use ndarray::ArrayView2;
use ndarray::ArrayViewMut2;
use strum::EnumCount;
use strum_macros::Display;
use strum_macros::EnumCount as EnumCountMacro;
use strum_macros::EnumIter;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::table::challenges::Challenges;

pub const BASE_WIDTH: usize = DegreeLoweringBaseTableColumn::COUNT;
pub const EXT_WIDTH: usize = DegreeLoweringExtTableColumn::COUNT;
pub const FULL_WIDTH: usize = BASE_WIDTH + EXT_WIDTH;

// This file is a placeholder for auto-generated code.
// Run `cargo run --bin constraint-evaluation-generator` to generate the actual code.

#[derive(Display, Debug, Clone, Copy, PartialEq, Eq, EnumIter, EnumCountMacro, Hash)]
pub enum DegreeLoweringBaseTableColumn {}

#[derive(Display, Debug, Clone, Copy, PartialEq, Eq, EnumIter, EnumCountMacro, Hash)]
pub enum DegreeLoweringExtTableColumn {}

#[derive(Debug, Clone)]
pub struct DegreeLoweringTable {}

impl DegreeLoweringTable {
    pub fn fill_derived_base_columns(mut _master_base_table: ArrayViewMut2<BFieldElement>) {
        // to be filled by generated code
    }

    pub fn fill_derived_ext_columns(
        _master_base_table: ArrayView2<BFieldElement>,
        mut _master_ext_table: ArrayViewMut2<XFieldElement>,
        _challenges: &Challenges,
    ) {
        // to be filled by generated code
    }
}
