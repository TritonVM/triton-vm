use ndarray::ArrayView2;
use ndarray::ArrayViewMut2;
use strum::EnumCount;
use strum_macros::Display;
use strum_macros::EnumCount as EnumCountMacro;
use strum_macros::EnumIter;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::x_field_element::XFieldElement;

use U32TableChallengeId::*;

use crate::table::challenges::TableChallenges;
use crate::table::constraint_circuit::ConstraintCircuit;
use crate::table::constraint_circuit::DualRowIndicator;
use crate::table::constraint_circuit::SingleRowIndicator;
use crate::table::master_table::NUM_BASE_COLUMNS;
use crate::table::master_table::NUM_EXT_COLUMNS;
use crate::table::table_column::U32BaseTableColumn;
use crate::table::table_column::U32ExtTableColumn;
use crate::vm::AlgebraicExecutionTrace;

pub const U32_TABLE_NUM_PERMUTATION_ARGUMENTS: usize = 2;
pub const U32_TABLE_NUM_EVALUATION_ARGUMENTS: usize = 0;
pub const U32_TABLE_NUM_EXTENSION_CHALLENGES: usize = U32TableChallengeId::COUNT;

pub const BASE_WIDTH: usize = U32BaseTableColumn::COUNT;
pub const EXT_WIDTH: usize = U32ExtTableColumn::COUNT;
pub const FULL_WIDTH: usize = BASE_WIDTH + EXT_WIDTH;

pub struct U32Table {}

pub struct ExtU32Table {}

impl ExtU32Table {
    pub fn ext_initial_constraints_as_circuits() -> Vec<
        ConstraintCircuit<
            U32TableChallenges,
            SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        todo!()
    }

    pub fn ext_consistency_constraints_as_circuits() -> Vec<
        ConstraintCircuit<
            U32TableChallenges,
            SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        todo!()
    }

    pub fn ext_transition_constraints_as_circuits() -> Vec<
        ConstraintCircuit<U32TableChallenges, DualRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>>,
    > {
        todo!()
    }

    pub fn ext_terminal_constraints_as_circuits() -> Vec<
        ConstraintCircuit<
            U32TableChallenges,
            SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        todo!()
    }
}

impl U32Table {
    pub fn fill_trace(
        _u32_table: &mut ArrayViewMut2<BFieldElement>,
        _aet: &AlgebraicExecutionTrace,
    ) {
        todo!()
    }

    pub fn pad_trace(_u32_table: &mut ArrayViewMut2<BFieldElement>, _u32_table_len: usize) {
        todo!()
    }

    pub fn extend(
        base_table: ArrayView2<BFieldElement>,
        ext_table: ArrayViewMut2<XFieldElement>,
        _challenges: &U32TableChallenges,
    ) {
        assert_eq!(BASE_WIDTH, base_table.ncols());
        assert_eq!(EXT_WIDTH, ext_table.ncols());
        assert_eq!(base_table.nrows(), ext_table.nrows());
        todo!()
    }
}

#[repr(usize)]
#[derive(Debug, Copy, Clone, Display, EnumCountMacro, EnumIter, PartialEq, Eq, Hash)]
pub enum U32TableChallengeId {
    LhsWeight,
    RhsWeight,
    CIWeight,
    ResultWeight,
    ProcessorPermIndeterminate,
}

impl From<U32TableChallengeId> for usize {
    fn from(val: U32TableChallengeId) -> Self {
        val as usize
    }
}

impl TableChallenges for U32TableChallenges {
    type Id = U32TableChallengeId;

    fn get_challenge(&self, id: Self::Id) -> XFieldElement {
        match id {
            LhsWeight => self.lhs_weight,
            RhsWeight => self.rhs_weight,
            CIWeight => self.ci_weight,
            ResultWeight => self.result_weight,
            ProcessorPermIndeterminate => self.processor_perm_indeterminate,
        }
    }
}

#[derive(Debug, Clone)]
pub struct U32TableChallenges {
    pub lhs_weight: XFieldElement,
    pub rhs_weight: XFieldElement,
    pub ci_weight: XFieldElement,
    pub result_weight: XFieldElement,
    pub processor_perm_indeterminate: XFieldElement,
}
