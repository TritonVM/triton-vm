//! This file is a placeholder for auto-generated code.
//! Run `cargo run --bin constraint-evaluation-generator` to generate the actual code.

use ndarray::ArrayView2;
use ndarray::ArrayViewMut2;
use strum::Display;
use strum::EnumCount;
use strum::EnumIter;
use twenty_first::prelude::BFieldElement;
use twenty_first::prelude::XFieldElement;

use crate::table::challenges::Challenges;

pub const BASE_WIDTH: usize = DegreeLoweringBaseTableColumn::COUNT;
pub const EXT_WIDTH: usize = DegreeLoweringExtTableColumn::COUNT;
pub const FULL_WIDTH: usize = BASE_WIDTH + EXT_WIDTH;

#[derive(Debug, Display, Copy, Clone, Eq, PartialEq, Hash, EnumCount, EnumIter)]
pub enum DegreeLoweringBaseTableColumn {}

#[derive(Debug, Display, Copy, Clone, Eq, PartialEq, Hash, EnumCount, EnumIter)]
pub enum DegreeLoweringExtTableColumn {}

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct DegreeLoweringTable;

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
