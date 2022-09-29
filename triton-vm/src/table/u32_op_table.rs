use crate::cross_table_arguments::{CrossTableArg, PermArg};
use itertools::Itertools;
use num_traits::{One, Zero};
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::mpolynomial::{Degree, MPolynomial};
use twenty_first::shared_math::traits::Inverse;
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::fri_domain::FriDomain;
use crate::instruction::Instruction;
use crate::table::base_table::Extendable;
use crate::table::extension_table::Evaluable;
use crate::table::table_column::U32OpBaseTableColumn::*;
use crate::table::table_column::U32OpExtTableColumn::*;

use super::base_table::{InheritsFromTable, Table, TableLike};
use super::challenges::AllChallenges;
use super::extension_table::{ExtensionTable, Quotientable, QuotientableExtensionTable};
use super::table_column::U32OpBaseTableColumn;

pub const U32_OP_TABLE_NUM_PERMUTATION_ARGUMENTS: usize = 1;
pub const U32_OP_TABLE_NUM_EVALUATION_ARGUMENTS: usize = 0;

/// This is 4 because it combines: ci, lhs, rhs, result
pub const U32_OP_TABLE_NUM_EXTENSION_CHALLENGES: usize = 4;

pub const BASE_WIDTH: usize = 12;
pub const FULL_WIDTH: usize =
    BASE_WIDTH + U32_OP_TABLE_NUM_PERMUTATION_ARGUMENTS + U32_OP_TABLE_NUM_EVALUATION_ARGUMENTS;

#[derive(Debug, Clone)]
pub struct U32OpTable {
    inherited_table: Table<BFieldElement>,
}

impl InheritsFromTable<BFieldElement> for U32OpTable {
    fn inherited_table(&self) -> &Table<BFieldElement> {
        &self.inherited_table
    }

    fn mut_inherited_table(&mut self) -> &mut Table<BFieldElement> {
        &mut self.inherited_table
    }
}

#[derive(Debug, Clone)]
pub struct ExtU32OpTable {
    inherited_table: Table<XFieldElement>,
}

impl Default for ExtU32OpTable {
    fn default() -> Self {
        Self {
            inherited_table: Table::new(
                BASE_WIDTH,
                FULL_WIDTH,
                vec![],
                "EmptyExtU32OpTable".to_string(),
            ),
        }
    }
}

impl TableLike<BFieldElement> for U32OpTable {}
impl Evaluable for ExtU32OpTable {}
impl Quotientable for ExtU32OpTable {}
impl QuotientableExtensionTable for ExtU32OpTable {}

impl InheritsFromTable<XFieldElement> for ExtU32OpTable {
    fn inherited_table(&self) -> &Table<XFieldElement> {
        &self.inherited_table
    }

    fn mut_inherited_table(&mut self) -> &mut Table<XFieldElement> {
        &mut self.inherited_table
    }
}

impl Extendable for U32OpTable {
    fn get_padding_rows(&self) -> (Option<usize>, Vec<Vec<BFieldElement>>) {
        let mut padding_row = vec![BFieldElement::zero(); BASE_WIDTH];
        padding_row[usize::from(LT)] = BFieldElement::new(2);
        padding_row[usize::from(Inv33MinusBits)] = BFieldElement::new(33).inverse();
        if let Some(row) = self.data().last() {
            padding_row[usize::from(CI)] = row[usize::from(CI)];
        }
        (None, vec![padding_row])
    }
}

impl TableLike<XFieldElement> for ExtU32OpTable {}

impl ExtU32OpTable {
    fn ext_initial_constraints(
        _challenges: &U32OpTableChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        // todo: check that running product starts with correct initial. Need challenges for that.
        vec![]
    }

    fn ext_consistency_constraints(
        _challenges: &U32OpTableChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        let one = MPolynomial::from_constant(1.into(), FULL_WIDTH);
        let two = MPolynomial::from_constant(2.into(), FULL_WIDTH);
        let thirty_three = MPolynomial::from_constant(33.into(), FULL_WIDTH);
        let variables = MPolynomial::variables(FULL_WIDTH, 1.into());

        let idc = variables[usize::from(IDC)].clone();
        let bits = variables[usize::from(Bits)].clone();
        let inv = variables[usize::from(Inv33MinusBits)].clone();
        let lhs = variables[usize::from(LHS)].clone();
        let lhs_inv = variables[usize::from(LHSInv)].clone();
        let rhs = variables[usize::from(RHS)].clone();
        let rhs_inv = variables[usize::from(RHSInv)].clone();
        let lt = variables[usize::from(LT)].clone();
        let and = variables[usize::from(AND)].clone();
        let xor = variables[usize::from(XOR)].clone();
        let rev = variables[usize::from(REV)].clone();

        let idc_is_0_or_1 = idc.clone() * (idc.clone() - one.clone());
        let idc_is_zero_or_bits_is_zero = idc.clone() * bits.clone();

        let bits_is_not_33 = one.clone() - (thirty_three - bits) * inv;

        let lhs_is_zero_or_lhs_inv_is_inverse_of_lhs =
            lhs.clone() * (one.clone() - lhs.clone() * lhs_inv.clone());
        let lhs_inv_is_zero_or_lhs_inv_is_inverse_of_lhs =
            lhs_inv.clone() * (one.clone() - lhs.clone() * lhs_inv.clone());

        let rhs_is_zero_or_rhs_inv_is_inverse_of_rhs =
            rhs.clone() * (rhs.clone() * rhs_inv.clone() - one.clone());
        let rhs_inv_is_zero_or_rhs_inv_is_inverse_of_rhs =
            rhs_inv.clone() * (rhs.clone() * rhs_inv.clone() - one.clone());

        let idc_is_one_or_lhs_is_nonzero_or_rhs_is_nonzero =
            (one.clone() - idc) * (one.clone() - lhs * lhs_inv) * (one - rhs * rhs_inv);
        let lt_is_two_after_u32_op =
            idc_is_one_or_lhs_is_nonzero_or_rhs_is_nonzero.clone() * (lt - two);
        let and_is_zero_after_u32_op = idc_is_one_or_lhs_is_nonzero_or_rhs_is_nonzero.clone() * and;
        let xor_is_zero_after_u32_op = idc_is_one_or_lhs_is_nonzero_or_rhs_is_nonzero.clone() * xor;
        let rev_is_zero_after_u32_op = idc_is_one_or_lhs_is_nonzero_or_rhs_is_nonzero * rev;

        vec![
            idc_is_0_or_1,
            idc_is_zero_or_bits_is_zero,
            bits_is_not_33,
            lhs_is_zero_or_lhs_inv_is_inverse_of_lhs,
            lhs_inv_is_zero_or_lhs_inv_is_inverse_of_lhs,
            rhs_is_zero_or_rhs_inv_is_inverse_of_rhs,
            rhs_inv_is_zero_or_rhs_inv_is_inverse_of_rhs,
            lt_is_two_after_u32_op,
            and_is_zero_after_u32_op,
            xor_is_zero_after_u32_op,
            rev_is_zero_after_u32_op,
        ]
    }

    fn ext_transition_constraints(
        _challenges: &U32OpTableChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        let half = MPolynomial::from_constant(
            XFieldElement::new_const(BFieldElement::new(9223372034707292161)),
            2 * FULL_WIDTH,
        );
        let one = MPolynomial::from_constant(1.into(), 2 * FULL_WIDTH);
        let two = MPolynomial::from_constant(2.into(), 2 * FULL_WIDTH);
        let rev_shift = MPolynomial::from_constant(2_u32.pow(31).into(), 2 * FULL_WIDTH);
        let variables = MPolynomial::variables(2 * FULL_WIDTH, 1.into());
        assert_eq!(one, half.clone() * two.clone());

        let idc = variables[usize::from(IDC)].clone();
        let bits = variables[usize::from(Bits)].clone();
        let ci = variables[usize::from(CI)].clone();
        let lhs = variables[usize::from(LHS)].clone();
        let rhs = variables[usize::from(RHS)].clone();
        let lt = variables[usize::from(LT)].clone();
        let and = variables[usize::from(AND)].clone();
        let xor = variables[usize::from(XOR)].clone();
        let rev = variables[usize::from(REV)].clone();
        let idc_next = variables[FULL_WIDTH + usize::from(IDC)].clone();
        let bits_next = variables[FULL_WIDTH + usize::from(Bits)].clone();
        let ci_next = variables[FULL_WIDTH + usize::from(CI)].clone();
        let lhs_next = variables[FULL_WIDTH + usize::from(LHS)].clone();
        let rhs_next = variables[FULL_WIDTH + usize::from(RHS)].clone();
        let lt_next = variables[FULL_WIDTH + usize::from(LT)].clone();
        let and_next = variables[FULL_WIDTH + usize::from(AND)].clone();
        let xor_next = variables[FULL_WIDTH + usize::from(XOR)].clone();
        let rev_next = variables[FULL_WIDTH + usize::from(REV)].clone();

        let lhs_lsb = lhs.clone() - two.clone() * lhs_next;
        let rhs_lsb = rhs.clone() - two.clone() * rhs_next;
        let idc_next_is_1 = idc_next.clone() - one.clone();

        let lhs_is_0_or_idc_next_is_0 = lhs.clone() * idc_next.clone();
        let rhs_is_0_or_idc_next_is_0 = rhs.clone() * idc_next;
        let idc_next_is_1_or_ci_stays = idc_next_is_1.clone() * (ci - ci_next);
        let idc_next_is_1_or_lhs_is_0_or_bits_increases =
            idc_next_is_1.clone() * lhs * (bits_next.clone() - (bits.clone() + one.clone()));
        let idc_next_is_1_or_rhs_is_0_or_bits_increases =
            idc_next_is_1.clone() * rhs * (bits_next - (bits + one.clone()));
        let idc_next_is_1_or_lsb_of_lhs_is_0_or_1 =
            idc_next_is_1.clone() * lhs_lsb.clone() * (lhs_lsb.clone() - one.clone());
        let idc_next_is_1_or_lsb_of_rhs_is_0_or_1 =
            idc_next_is_1.clone() * rhs_lsb.clone() * (rhs_lsb.clone() - one.clone());

        // LT
        let idc_next_is_1_or_lt_next_is_1_or_2_or_lt_is_0 = idc_next_is_1.clone()
            * (lt_next.clone() - one.clone())
            * (lt_next.clone() - two.clone())
            * lt.clone();
        let idc_next_is_1_or_lt_next_is_0_or_2_or_lt_is_1 = idc_next_is_1.clone()
            * lt_next.clone()
            * (lt_next.clone() - two.clone())
            * (lt.clone() - one.clone());

        let idc_next_is_1_or_lt_next_is_0_or_1 =
            idc_next_is_1.clone() * lt_next.clone() * (lt_next - one.clone());
        let idc_next_is_1_or_lt_next_is_0_or_1_or_lhs_lsb_is_1_or_rhs_lsb_is_0_or_lt_is_1 =
            idc_next_is_1_or_lt_next_is_0_or_1.clone()
                * (lhs_lsb.clone() - one.clone())
                * rhs_lsb.clone()
                * (lt.clone() - one.clone());
        let idc_next_is_1_or_lt_next_is_0_or_1_or_lhs_lbs_is_0_or_rhs_lsb_is_1_or_lt_is_0 =
            idc_next_is_1_or_lt_next_is_0_or_1.clone()
                * lhs_lsb.clone()
                * (rhs_lsb.clone() - one.clone())
                * lt.clone();

        let idc_next_is_1_or_lt_next_is_0_or_1_or_lhs_lsb_uneq_rhs_lsb =
            idc_next_is_1_or_lt_next_is_0_or_1 * (one.clone() - lhs_lsb.clone() - rhs_lsb.clone());
        let idc_next_is_1_or_lt_next_is_0_or_1_or_lhs_lsb_uneq_rhs_lsb_or_idc_is_1_or_lt_is_2 =
            idc_next_is_1_or_lt_next_is_0_or_1_or_lhs_lsb_uneq_rhs_lsb.clone()
                * (idc.clone() - one)
                * (lt.clone() - two.clone());
        let idc_next_is_1_or_lt_next_is_0_or_1_or_lhs_lsb_uneq_rhs_lsb_or_idc_is_0_or_lt_is_0 =
            idc_next_is_1_or_lt_next_is_0_or_1_or_lhs_lsb_uneq_rhs_lsb * idc * lt;

        // AND, XOR, REV
        let idc_next_is_1_or_and_eq_twice_and_next_plus_and_of_lsbs = idc_next_is_1.clone()
            * (and - (two.clone() * and_next + lhs_lsb.clone() * rhs_lsb.clone()));
        let idc_next_is_1_or_xor_eq_twice_xor_next_plus_xor_of_lsbs = idc_next_is_1.clone()
            * (xor
                - (two.clone() * xor_next + lhs_lsb.clone() + rhs_lsb.clone()
                    - two * lhs_lsb.clone() * rhs_lsb));
        let idc_next_is_1_or_rev_eq_half_rev_next_plus_shifted_lhs_lsb =
            idc_next_is_1 * (rev - (half * rev_next + rev_shift * lhs_lsb));

        vec![
            lhs_is_0_or_idc_next_is_0,
            rhs_is_0_or_idc_next_is_0,
            idc_next_is_1_or_ci_stays,
            idc_next_is_1_or_lhs_is_0_or_bits_increases,
            idc_next_is_1_or_rhs_is_0_or_bits_increases,
            idc_next_is_1_or_lsb_of_lhs_is_0_or_1,
            idc_next_is_1_or_lsb_of_rhs_is_0_or_1,
            idc_next_is_1_or_lt_next_is_1_or_2_or_lt_is_0,
            idc_next_is_1_or_lt_next_is_0_or_2_or_lt_is_1,
            idc_next_is_1_or_lt_next_is_0_or_1_or_lhs_lsb_is_1_or_rhs_lsb_is_0_or_lt_is_1,
            idc_next_is_1_or_lt_next_is_0_or_1_or_lhs_lbs_is_0_or_rhs_lsb_is_1_or_lt_is_0,
            idc_next_is_1_or_lt_next_is_0_or_1_or_lhs_lsb_uneq_rhs_lsb_or_idc_is_1_or_lt_is_2,
            idc_next_is_1_or_lt_next_is_0_or_1_or_lhs_lsb_uneq_rhs_lsb_or_idc_is_0_or_lt_is_0,
            idc_next_is_1_or_and_eq_twice_and_next_plus_and_of_lsbs,
            idc_next_is_1_or_xor_eq_twice_xor_next_plus_xor_of_lsbs,
            idc_next_is_1_or_rev_eq_half_rev_next_plus_shifted_lhs_lsb,
        ]
    }

    fn ext_terminal_constraints(
        _challenges: &U32OpTableChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        let variables = MPolynomial::variables(FULL_WIDTH, 1.into());
        let idc = variables[usize::from(IDC)].clone();
        let lhs = variables[usize::from(LHS)].clone();
        let rhs = variables[usize::from(RHS)].clone();
        vec![idc, lhs, rhs]
    }
}

impl U32OpTable {
    pub fn new_prover(matrix: Vec<Vec<BFieldElement>>) -> Self {
        let inherited_table = Table::new(BASE_WIDTH, FULL_WIDTH, matrix, "U32OpTable".to_string());
        Self { inherited_table }
    }

    pub fn codeword_table(
        &self,
        fri_domain: &FriDomain<BFieldElement>,
        omicron: BFieldElement,
        padded_height: usize,
        num_trace_randomizers: usize,
    ) -> Self {
        let base_columns = 0..self.base_width();
        let codewords = self.low_degree_extension(
            fri_domain,
            omicron,
            padded_height,
            num_trace_randomizers,
            base_columns,
        );
        let inherited_table = self.inherited_table.with_data(codewords);
        Self { inherited_table }
    }

    pub fn extend(
        &self,
        challenges: &U32OpTableChallenges,
        interpolant_degree: Degree,
    ) -> ExtU32OpTable {
        let mut extension_matrix: Vec<Vec<XFieldElement>> = Vec::with_capacity(self.data().len());
        let mut running_product = PermArg::default_initial();

        for row in self.data().iter() {
            let mut extension_row = [0.into(); FULL_WIDTH];
            extension_row[..BASE_WIDTH]
                .copy_from_slice(&row.iter().map(|elem| elem.lift()).collect_vec());

            let current_instruction: Instruction = row[usize::from(U32OpBaseTableColumn::CI)]
                .value()
                .try_into()
                .expect("CI does not correspond to any instruction.");

            if row[usize::from(U32OpBaseTableColumn::IDC)].is_one() {
                let ci = extension_row[usize::from(U32OpBaseTableColumn::CI)];
                let lhs = extension_row[usize::from(U32OpBaseTableColumn::LHS)];
                let rhs = extension_row[usize::from(U32OpBaseTableColumn::RHS)];
                let result = match current_instruction {
                    // halt is used for padding
                    Instruction::Halt => XFieldElement::zero(),
                    x => panic!("Unknown instruction '{x}' in the U32 Table."),
                };

                // Compress (ci, lhs, rhs, result) into single value
                let compressed_row = ci * challenges.ci_weight
                    + lhs * challenges.lhs_weight
                    + rhs * challenges.rhs_weight
                    + result * challenges.result_weight;

                running_product *= challenges.processor_perm_row_weight - compressed_row;
            }
            extension_row[usize::from(RunningProductPermArg)] = running_product;

            extension_matrix.push(extension_row.to_vec());
        }

        let inherited_table = self.extension(
            extension_matrix,
            interpolant_degree,
            ExtU32OpTable::ext_initial_constraints(challenges),
            ExtU32OpTable::ext_consistency_constraints(challenges),
            ExtU32OpTable::ext_transition_constraints(challenges),
            ExtU32OpTable::ext_terminal_constraints(challenges),
        );
        ExtU32OpTable { inherited_table }
    }

    pub fn for_verifier(
        interpolant_degree: Degree,
        all_challenges: &AllChallenges,
    ) -> ExtU32OpTable {
        let inherited_table = Table::<BFieldElement>::new(
            BASE_WIDTH,
            FULL_WIDTH,
            vec![],
            "ExtU32OpTable".to_string(),
        );
        let base_table = Self { inherited_table };
        let empty_matrix: Vec<Vec<XFieldElement>> = vec![];
        let extension_table = base_table.extension(
            empty_matrix,
            interpolant_degree,
            ExtU32OpTable::ext_initial_constraints(&all_challenges.u32_op_table_challenges),
            ExtU32OpTable::ext_consistency_constraints(&all_challenges.u32_op_table_challenges),
            ExtU32OpTable::ext_transition_constraints(&all_challenges.u32_op_table_challenges),
            ExtU32OpTable::ext_terminal_constraints(&all_challenges.u32_op_table_challenges),
        );

        ExtU32OpTable {
            inherited_table: extension_table,
        }
    }
}

impl ExtU32OpTable {
    pub fn ext_codeword_table(
        &self,
        fri_domain: &FriDomain<XFieldElement>,
        omicron: XFieldElement,
        padded_height: usize,
        num_trace_randomizers: usize,
        base_codewords: &[Vec<BFieldElement>],
    ) -> Self {
        let ext_columns = self.base_width()..self.full_width();
        let ext_codewords = self.low_degree_extension(
            fri_domain,
            omicron,
            padded_height,
            num_trace_randomizers,
            ext_columns,
        );

        let lifted_base_codewords = base_codewords
            .iter()
            .map(|base_codeword| base_codeword.iter().map(|bfe| bfe.lift()).collect_vec())
            .collect_vec();
        let all_codewords = vec![lifted_base_codewords, ext_codewords].concat();
        assert_eq!(self.full_width(), all_codewords.len());

        let inherited_table = self.inherited_table.with_data(all_codewords);
        ExtU32OpTable { inherited_table }
    }
}

#[derive(Debug, Clone)]
pub struct U32OpTableChallenges {
    /// The weight that combines two consecutive rows in the
    /// permutation/evaluation column of the u32 op table.
    pub processor_perm_row_weight: XFieldElement,

    /// Weights for condensing part of a row into a single column. (Related to processor table.)
    pub lhs_weight: XFieldElement,
    pub rhs_weight: XFieldElement,
    pub ci_weight: XFieldElement,
    pub result_weight: XFieldElement,
}

impl ExtensionTable for ExtU32OpTable {
    fn dynamic_initial_constraints(
        &self,
        challenges: &AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtU32OpTable::ext_initial_constraints(&challenges.u32_op_table_challenges)
    }

    fn dynamic_consistency_constraints(
        &self,
        challenges: &AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtU32OpTable::ext_consistency_constraints(&challenges.u32_op_table_challenges)
    }

    fn dynamic_transition_constraints(
        &self,
        challenges: &super::challenges::AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtU32OpTable::ext_transition_constraints(&challenges.u32_op_table_challenges)
    }

    fn dynamic_terminal_constraints(
        &self,
        challenges: &AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtU32OpTable::ext_terminal_constraints(&challenges.u32_op_table_challenges)
    }
}
