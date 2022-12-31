use ndarray::ArrayView2;
use ndarray::ArrayViewMut2;
use std::ops::Mul;
use strum::EnumCount;
use strum_macros::Display;
use strum_macros::EnumCount as EnumCountMacro;
use strum_macros::EnumIter;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::x_field_element::XFieldElement;

use triton_opcodes::instruction::Instruction;
use U32TableChallengeId::*;

use crate::table::challenges::TableChallenges;
use crate::table::constraint_circuit::ConstraintCircuit;
use crate::table::constraint_circuit::ConstraintCircuitBuilder;
use crate::table::constraint_circuit::ConstraintCircuitMonad;
use crate::table::constraint_circuit::DualRowIndicator;
use crate::table::constraint_circuit::SingleRowIndicator;
use crate::table::constraint_circuit::SingleRowIndicator::*;
use crate::table::cross_table_argument::CrossTableArg;
use crate::table::cross_table_argument::PermArg;
use crate::table::master_table::NUM_BASE_COLUMNS;
use crate::table::master_table::NUM_EXT_COLUMNS;
use crate::table::table_column::MasterBaseTableColumn;
use crate::table::table_column::MasterExtTableColumn;
use crate::table::table_column::U32BaseTableColumn;
use crate::table::table_column::U32BaseTableColumn::*;
use crate::table::table_column::U32ExtTableColumn;
use crate::table::table_column::U32ExtTableColumn::*;
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
        let circuit_builder = ConstraintCircuitBuilder::new();
        let challenge = |c| circuit_builder.challenge(c);
        let one = circuit_builder.b_constant(1_u32.into());

        let copy_flag = circuit_builder.input(BaseRow(CopyFlag.master_base_table_index()));
        let lhs = circuit_builder.input(BaseRow(LHS.master_base_table_index()));
        let rhs = circuit_builder.input(BaseRow(RHS.master_base_table_index()));
        let ci = circuit_builder.input(BaseRow(CI.master_base_table_index()));

        let rp = circuit_builder.input(ExtRow(ProcessorPermArg.master_ext_table_index()));

        let deselect_instructions = |instructions: &[Instruction]| {
            instructions
                .iter()
                .map(|&instr| ci.clone() - circuit_builder.b_constant(instr.opcode_b()))
                .fold(one.clone(), ConstraintCircuitMonad::mul)
                * ci.clone()
        };
        let lt_div_deselector = deselect_instructions(&[
            Instruction::And,
            Instruction::Xor,
            Instruction::Log2Floor,
            Instruction::Pow,
        ]);
        let and_deselector = deselect_instructions(&[
            Instruction::Lt,
            Instruction::Div,
            Instruction::Xor,
            Instruction::Log2Floor,
            Instruction::Pow,
        ]);
        let xor_deselector = deselect_instructions(&[
            Instruction::Lt,
            Instruction::Div,
            Instruction::And,
            Instruction::Log2Floor,
            Instruction::Pow,
        ]);
        let log2floor_deselector = deselect_instructions(&[
            Instruction::Lt,
            Instruction::Div,
            Instruction::And,
            Instruction::Xor,
            Instruction::Pow,
        ]);
        let pow_deselector = deselect_instructions(&[
            Instruction::Lt,
            Instruction::Div,
            Instruction::And,
            Instruction::Xor,
            Instruction::Log2Floor,
        ]);
        let result = lt_div_deselector
            * circuit_builder.input(BaseRow(LT.master_base_table_index()))
            + and_deselector * circuit_builder.input(BaseRow(AND.master_base_table_index()))
            + xor_deselector * circuit_builder.input(BaseRow(XOR.master_base_table_index()))
            + log2floor_deselector
                * circuit_builder.input(BaseRow(Log2Floor.master_base_table_index()))
            + pow_deselector * circuit_builder.input(BaseRow(Pow.master_base_table_index()));

        let initial_factor = challenge(ProcessorPermIndeterminate)
            - challenge(LhsWeight) * lhs
            - challenge(RhsWeight) * rhs
            - challenge(CIWeight) * ci
            - challenge(ResultWeight) * result;
        let copy_flag_is_0_or_rp_has_accumulated_first_row =
            copy_flag.clone() * (rp.clone() - initial_factor);

        let default_initial = circuit_builder.x_constant(PermArg::default_initial());
        let copy_flag_is_1_or_rp_is_default_initial = (one - copy_flag) * (default_initial - rp);

        [
            copy_flag_is_1_or_rp_is_default_initial,
            copy_flag_is_0_or_rp_has_accumulated_first_row,
        ]
        .map(|circuit| circuit.consume())
        .to_vec()
    }

    pub fn ext_consistency_constraints_as_circuits() -> Vec<
        ConstraintCircuit<
            U32TableChallenges,
            SingleRowIndicator<NUM_BASE_COLUMNS, NUM_EXT_COLUMNS>,
        >,
    > {
        let circuit_builder = ConstraintCircuitBuilder::new();
        let one = circuit_builder.b_constant(1_u32.into());

        let copy_flag = circuit_builder.input(BaseRow(CopyFlag.master_base_table_index()));
        let bits = circuit_builder.input(BaseRow(Bits.master_base_table_index()));
        let bits_minus_33_inv =
            circuit_builder.input(BaseRow(BitsMinus33Inv.master_base_table_index()));
        let lhs = circuit_builder.input(BaseRow(LHS.master_base_table_index()));
        let rhs = circuit_builder.input(BaseRow(RHS.master_base_table_index()));
        let lt = circuit_builder.input(BaseRow(LT.master_base_table_index()));
        let and = circuit_builder.input(BaseRow(AND.master_base_table_index()));
        let xor = circuit_builder.input(BaseRow(XOR.master_base_table_index()));
        let log2floor = circuit_builder.input(BaseRow(Log2Floor.master_base_table_index()));
        let lhs_copy = circuit_builder.input(BaseRow(LhsCopy.master_base_table_index()));
        let pow = circuit_builder.input(BaseRow(Pow.master_base_table_index()));
        let lhs_inv = circuit_builder.input(BaseRow(LhsInv.master_base_table_index()));
        let rhs_inv = circuit_builder.input(BaseRow(RhsInv.master_base_table_index()));

        let copy_flag_is_bit = copy_flag.clone() * (one.clone() - copy_flag.clone());
        let copy_flag_is_0_or_bits_is_0 = copy_flag.clone() * bits.clone();
        let bits_minus_33_inv_is_inverse_of_bits_minus_33 = one.clone()
            - bits_minus_33_inv * (bits - circuit_builder.b_constant(BFieldElement::new(33)));
        let lhs_inv_is_0_or_the_inverse_of_lhs =
            lhs_inv.clone() * (one.clone() - lhs.clone() * lhs_inv.clone());
        let lhs_is_0_or_lhs_inverse_is_the_inverse_of_lhs =
            lhs.clone() * (one.clone() - lhs.clone() * lhs_inv.clone());
        let rhs_inv_is_0_or_the_inverse_of_rhs =
            rhs_inv.clone() * (one.clone() - rhs.clone() * rhs_inv.clone());
        let rhs_is_0_or_rhs_inverse_is_the_inverse_of_rhs =
            rhs.clone() * (one.clone() - rhs.clone() * rhs_inv.clone());
        let copy_flag_is_0_or_lhs_copy_is_lhs = copy_flag.clone() * (lhs_copy - lhs.clone());
        let padding_row = (one.clone() - copy_flag)
            * (one.clone() - lhs.clone() * lhs_inv.clone())
            * (one.clone() - rhs * rhs_inv);
        let padding_row_or_lt_is_2 =
            padding_row.clone() * (lt - circuit_builder.b_constant(BFieldElement::new(2)));
        let padding_row_or_and_is_0 = padding_row.clone() * and;
        let padding_row_or_xor_is_0 = padding_row.clone() * xor;
        let padding_row_or_pow_is_1 = padding_row * (one.clone() - pow);
        let lhs_is_not_zero_or_log2floor_is_negative_1 =
            (one.clone() - lhs * lhs_inv) * (log2floor + one);

        [
            copy_flag_is_bit,
            copy_flag_is_0_or_bits_is_0,
            bits_minus_33_inv_is_inverse_of_bits_minus_33,
            lhs_inv_is_0_or_the_inverse_of_lhs,
            lhs_is_0_or_lhs_inverse_is_the_inverse_of_lhs,
            rhs_inv_is_0_or_the_inverse_of_rhs,
            rhs_is_0_or_rhs_inverse_is_the_inverse_of_rhs,
            copy_flag_is_0_or_lhs_copy_is_lhs,
            padding_row_or_lt_is_2,
            padding_row_or_and_is_0,
            padding_row_or_xor_is_0,
            padding_row_or_pow_is_1,
            lhs_is_not_zero_or_log2floor_is_negative_1,
        ]
        .map(|circuit| circuit.consume())
        .to_vec()
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
