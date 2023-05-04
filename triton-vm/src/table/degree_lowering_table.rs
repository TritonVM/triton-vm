use strum::EnumCount;
use strum_macros::Display;
use strum_macros::EnumCount as EnumCountMacro;
use strum_macros::EnumIter;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::table::extension_table::Evaluable;
use crate::table::extension_table::Quotientable;

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

impl Evaluable<BFieldElement> for ExtDegreeLoweringTable {}
impl Evaluable<XFieldElement> for ExtDegreeLoweringTable {}
impl Quotientable for ExtDegreeLoweringTable {}
