use itertools::Itertools;
use num_traits::{One, Zero};
use strum::EnumCount;
use strum_macros::{Display, EnumCount as EnumCountMacro, EnumIter};
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::traits::Inverse;
use twenty_first::shared_math::x_field_element::XFieldElement;

use RamTableChallengeId::*;

use crate::arithmetic_domain::ArithmeticDomain;
use crate::cross_table_arguments::{CrossTableArg, PermArg};
use crate::table::base_table::Extendable;
use crate::table::base_table::{InheritsFromTable, Table, TableLike};
use crate::table::challenges::TableChallenges;
use crate::table::constraint_circuit::SingleRowIndicator;
use crate::table::constraint_circuit::SingleRowIndicator::Row;
use crate::table::constraint_circuit::{
    ConstraintCircuit, ConstraintCircuitBuilder, DualRowIndicator,
};
use crate::table::extension_table::{ExtensionTable, QuotientableExtensionTable};
use crate::table::table_column::RamBaseTableColumn::{self, *};
use crate::table::table_column::RamExtTableColumn::{self, *};

use super::constraint_circuit::DualRowIndicator::*;

pub const RAM_TABLE_NUM_PERMUTATION_ARGUMENTS: usize = 1;
pub const RAM_TABLE_NUM_EVALUATION_ARGUMENTS: usize = 0;

/// This is 3 because it combines: clk, ramv, ramp
pub const RAM_TABLE_NUM_EXTENSION_CHALLENGES: usize = 3;

pub const BASE_WIDTH: usize = RamBaseTableColumn::COUNT;
pub const FULL_WIDTH: usize = BASE_WIDTH + RamExtTableColumn::COUNT;

#[derive(Debug, Clone)]
pub struct RamTable {
    inherited_table: Table<BFieldElement>,
}

impl InheritsFromTable<BFieldElement> for RamTable {
    fn inherited_table(&self) -> &Table<BFieldElement> {
        &self.inherited_table
    }

    fn mut_inherited_table(&mut self) -> &mut Table<BFieldElement> {
        &mut self.inherited_table
    }
}

#[derive(Debug, Clone)]
pub struct ExtRamTable {
    pub(crate) inherited_table: Table<XFieldElement>,
}

impl Default for ExtRamTable {
    fn default() -> Self {
        Self {
            inherited_table: Table::new(
                BASE_WIDTH,
                FULL_WIDTH,
                vec![],
                "EmptyExtRamTable".to_string(),
            ),
        }
    }
}

impl QuotientableExtensionTable for ExtRamTable {}

impl InheritsFromTable<XFieldElement> for ExtRamTable {
    fn inherited_table(&self) -> &Table<XFieldElement> {
        &self.inherited_table
    }

    fn mut_inherited_table(&mut self) -> &mut Table<XFieldElement> {
        &mut self.inherited_table
    }
}

impl RamTable {
    pub fn new(inherited_table: Table<BFieldElement>) -> Self {
        Self { inherited_table }
    }

    pub fn new_prover(matrix: Vec<Vec<BFieldElement>>) -> Self {
        let inherited_table = Table::new(BASE_WIDTH, FULL_WIDTH, matrix, "RamTable".to_string());
        Self { inherited_table }
    }

    pub fn to_quotient_and_fri_domain_table(
        &self,
        quotient_domain: &ArithmeticDomain<BFieldElement>,
        fri_domain: &ArithmeticDomain<BFieldElement>,
        num_trace_randomizers: usize,
    ) -> (Self, Self) {
        let base_columns = 0..self.base_width();
        let (quotient_domain_table, fri_domain_table) = self.dual_low_degree_extension(
            quotient_domain,
            fri_domain,
            num_trace_randomizers,
            base_columns,
        );

        (
            Self::new(quotient_domain_table),
            Self::new(fri_domain_table),
        )
    }

    pub fn extend(&self, challenges: &RamTableChallenges) -> ExtRamTable {
        let mut extension_matrix: Vec<Vec<XFieldElement>> = Vec::with_capacity(self.data().len());
        let mut running_product_for_perm_arg = PermArg::default_initial();
        let mut all_clock_jump_differences_running_product = PermArg::default_initial();

        // initialize columns establishing Bézout relation
        let ramp_first_row = self.data().first().unwrap()[usize::from(RAMP)];
        let mut running_product_of_ramp = challenges.bezout_relation_indeterminate - ramp_first_row;
        let mut formal_derivative = XFieldElement::one();
        let mut bezout_coefficient_0 = XFieldElement::zero();
        let bcpc_first_row =
            self.data().first().unwrap()[usize::from(BezoutCoefficientPolynomialCoefficient1)];
        let mut bezout_coefficient_1 = bcpc_first_row.lift();

        let mut previous_row: Option<Vec<BFieldElement>> = None;
        for row in self.data().iter() {
            let mut extension_row = [0.into(); FULL_WIDTH];
            extension_row[..BASE_WIDTH]
                .copy_from_slice(&row.iter().map(|elem| elem.lift()).collect_vec());

            let clk = extension_row[usize::from(CLK)];
            let ramp = extension_row[usize::from(RAMP)];
            let ramv = extension_row[usize::from(RAMV)];

            if let Some(prow) = previous_row {
                if prow[usize::from(RAMP)] != row[usize::from(RAMP)] {
                    // accumulate coefficient for Bézout relation, proving new RAMP is unique
                    let bcpc0 = extension_row[usize::from(BezoutCoefficientPolynomialCoefficient0)];
                    let bcpc1 = extension_row[usize::from(BezoutCoefficientPolynomialCoefficient1)];
                    let bezout_challenge = challenges.bezout_relation_indeterminate;

                    formal_derivative =
                        (bezout_challenge - ramp) * formal_derivative + running_product_of_ramp;
                    running_product_of_ramp *= bezout_challenge - ramp;
                    bezout_coefficient_0 = bezout_coefficient_0 * bezout_challenge + bcpc0;
                    bezout_coefficient_1 = bezout_coefficient_1 * bezout_challenge + bcpc1;
                } else {
                    // prove that clock jump is directed forward
                    let clock_jump_difference =
                        (row[usize::from(CLK)] - prow[usize::from(CLK)]).lift();
                    if clock_jump_difference != XFieldElement::one() {
                        all_clock_jump_differences_running_product *= challenges
                            .all_clock_jump_differences_multi_perm_indeterminate
                            - clock_jump_difference;
                    }
                }
            }

            extension_row[usize::from(RunningProductOfRAMP)] = running_product_of_ramp;
            extension_row[usize::from(FormalDerivative)] = formal_derivative;
            extension_row[usize::from(BezoutCoefficient0)] = bezout_coefficient_0;
            extension_row[usize::from(BezoutCoefficient1)] = bezout_coefficient_1;
            extension_row[usize::from(AllClockJumpDifferencesPermArg)] =
                all_clock_jump_differences_running_product;

            // permutation argument to Processor Table
            let clk_w = challenges.clk_weight;
            let ramp_w = challenges.ramp_weight;
            let ramv_w = challenges.ramv_weight;

            // compress multiple values within one row so they become one value
            let compressed_row_for_permutation_argument =
                clk * clk_w + ramp * ramp_w + ramv * ramv_w;

            // compute the running product of the compressed column for permutation argument
            running_product_for_perm_arg *=
                challenges.processor_perm_indeterminate - compressed_row_for_permutation_argument;
            extension_row[usize::from(RunningProductPermArg)] = running_product_for_perm_arg;

            previous_row = Some(row.clone());
            extension_matrix.push(extension_row.to_vec());
        }

        assert_eq!(self.data().len(), extension_matrix.len());
        let inherited_table = self.new_from_lifted_matrix(extension_matrix);
        ExtRamTable { inherited_table }
    }

    pub fn for_verifier() -> ExtRamTable {
        let inherited_table = Table::new(BASE_WIDTH, FULL_WIDTH, vec![], "ExtRamTable".to_string());
        let base_table = Self { inherited_table };
        let empty_matrix: Vec<Vec<XFieldElement>> = vec![];
        let extension_table = base_table.new_from_lifted_matrix(empty_matrix);

        ExtRamTable {
            inherited_table: extension_table,
        }
    }
}

impl ExtRamTable {
    pub fn new(inherited_table: Table<XFieldElement>) -> Self {
        Self { inherited_table }
    }

    pub fn to_quotient_and_fri_domain_table(
        &self,
        quotient_domain: &ArithmeticDomain<BFieldElement>,
        fri_domain: &ArithmeticDomain<BFieldElement>,
        num_trace_randomizers: usize,
    ) -> (Self, Self) {
        let ext_columns = self.base_width()..self.full_width();
        let (quotient_domain_table, fri_domain_table) = self.dual_low_degree_extension(
            quotient_domain,
            fri_domain,
            num_trace_randomizers,
            ext_columns,
        );

        (
            Self::new(quotient_domain_table),
            Self::new(fri_domain_table),
        )
    }
}

impl TableLike<BFieldElement> for RamTable {}

impl Extendable for RamTable {
    fn get_padding_rows(&self) -> (Option<usize>, Vec<Vec<BFieldElement>>) {
        panic!("This function should not be called: the Ram Table implements `.pad` directly.")
    }

    fn pad(&mut self, padded_height: usize) {
        let max_clock = self.data().len() as u64 - 1;
        let num_padding_rows = padded_height - self.data().len();

        if self.data().is_empty() {
            let mut all_padding = vec![];
            for i in 0..padded_height {
                let mut padding_row = vec![BFieldElement::zero(); BASE_WIDTH];
                padding_row[usize::from(CLK)] = BFieldElement::new(i as u64);
                all_padding.push(padding_row);
            }
            self.mut_data().append(&mut all_padding);
            return;
        }

        let template_index = self
            .data()
            .iter()
            .enumerate()
            .find(|(_, row)| row[usize::from(CLK)].value() == max_clock)
            .map(|(idx, _)| idx)
            .unwrap();
        let insertion_index = template_index + 1;

        let padding_template = &mut self.mut_data()[template_index];
        let difference_inverse = padding_template[usize::from(InverseOfRampDifference)];
        padding_template[usize::from(InverseOfRampDifference)] = BFieldElement::zero();
        padding_template[usize::from(InverseOfClkDiffMinusOne)] = 0_u64.into();

        let mut padding_rows = vec![];
        while padding_rows.len() < num_padding_rows {
            let mut padding_row = padding_template.clone();
            padding_row[usize::from(CLK)] += (padding_rows.len() as u32 + 1).into();
            padding_rows.push(padding_row)
        }

        if let Some(row) = padding_rows.last_mut() {
            row[usize::from(InverseOfRampDifference)] = difference_inverse;

            if let Some(next_row) = self.data().get(insertion_index) {
                let clk_diff = next_row[usize::from(CLK)] - row[usize::from(CLK)];
                row[usize::from(InverseOfClkDiffMinusOne)] =
                    (clk_diff - BFieldElement::one()).inverse_or_zero();
            }
        }

        let old_tail_length = self.data().len() - insertion_index;
        self.mut_data().append(&mut padding_rows);
        self.mut_data()[insertion_index..].rotate_left(old_tail_length);

        assert_eq!(padded_height, self.data().len());
    }
}

impl TableLike<XFieldElement> for ExtRamTable {}

impl ExtRamTable {
    pub fn ext_initial_constraints_as_circuits(
    ) -> Vec<ConstraintCircuit<RamTableChallenges, SingleRowIndicator<FULL_WIDTH>>> {
        use RamTableChallengeId::*;

        let circuit_builder = ConstraintCircuitBuilder::new(FULL_WIDTH);
        let one = circuit_builder.b_constant(1_u32.into());

        let bezout_challenge = circuit_builder.challenge(BezoutRelationIndeterminate);
        let rppa_challenge = circuit_builder.challenge(ProcessorPermIndeterminate);

        let clk = circuit_builder.input(Row(CLK.into()));
        let ramp = circuit_builder.input(Row(RAMP.into()));
        let ramv = circuit_builder.input(Row(RAMV.into()));
        let bcpc0 = circuit_builder.input(Row(BezoutCoefficientPolynomialCoefficient0.into()));
        let bcpc1 = circuit_builder.input(Row(BezoutCoefficientPolynomialCoefficient1.into()));
        let rp = circuit_builder.input(Row(RunningProductOfRAMP.into()));
        let fd = circuit_builder.input(Row(FormalDerivative.into()));
        let bc0 = circuit_builder.input(Row(BezoutCoefficient0.into()));
        let bc1 = circuit_builder.input(Row(BezoutCoefficient1.into()));
        let rppa = circuit_builder.input(Row(RunningProductPermArg.into()));

        let clk_is_0 = clk;
        let ramp_is_0 = ramp;
        let ramv_is_0 = ramv;
        let bezout_coefficient_polynomial_coefficient_0_is_0 = bcpc0;
        let bezout_coefficient_0_is_0 = bc0;
        let bezout_coefficient_1_is_bezout_coefficient_polynomial_coefficient_1 = bc1 - bcpc1;
        let formal_derivative_is_1 = fd - one;
        // This should be rp - (bezout_challenge - ramp). However, `ramp` is already constrained to
        // be 0, and can thus be omitted.
        let running_product_polynomial_is_initialized_correctly = rp - bezout_challenge;
        let running_product_permutation_argument_is_initialized_correctly = rppa - rppa_challenge;

        [
            clk_is_0,
            ramp_is_0,
            ramv_is_0,
            bezout_coefficient_polynomial_coefficient_0_is_0,
            bezout_coefficient_0_is_0,
            bezout_coefficient_1_is_bezout_coefficient_polynomial_coefficient_1,
            formal_derivative_is_1,
            running_product_polynomial_is_initialized_correctly,
            running_product_permutation_argument_is_initialized_correctly,
        ]
        .map(|circuit| circuit.consume())
        .to_vec()
    }

    pub fn ext_consistency_constraints_as_circuits(
    ) -> Vec<ConstraintCircuit<RamTableChallenges, SingleRowIndicator<FULL_WIDTH>>> {
        // no further constraints
        vec![]
    }

    pub fn ext_transition_constraints_as_circuits(
    ) -> Vec<ConstraintCircuit<RamTableChallenges, DualRowIndicator<FULL_WIDTH>>> {
        let circuit_builder = ConstraintCircuitBuilder::new(2 * FULL_WIDTH);
        let one = circuit_builder.b_constant(1u32.into());

        let bezout_challenge = circuit_builder.challenge(BezoutRelationIndeterminate);
        let cjd_challenge =
            circuit_builder.challenge(AllClockJumpDifferencesMultiPermIndeterminate);
        let rppa_challenge = circuit_builder.challenge(ProcessorPermIndeterminate);
        let clk_weight = circuit_builder.challenge(ClkWeight);
        let ramp_weight = circuit_builder.challenge(RampWeight);
        let ramv_weight = circuit_builder.challenge(RamvWeight);

        let clk = circuit_builder.input(CurrentRow(CLK.into()));
        let ramp = circuit_builder.input(CurrentRow(RAMP.into()));
        let ramv = circuit_builder.input(CurrentRow(RAMV.into()));
        let iord = circuit_builder.input(CurrentRow(InverseOfRampDifference.into()));
        let bcpc0 =
            circuit_builder.input(CurrentRow(BezoutCoefficientPolynomialCoefficient0.into()));
        let bcpc1 =
            circuit_builder.input(CurrentRow(BezoutCoefficientPolynomialCoefficient1.into()));
        let rp = circuit_builder.input(CurrentRow(RunningProductOfRAMP.into()));
        let fd = circuit_builder.input(CurrentRow(FormalDerivative.into()));
        let bc0 = circuit_builder.input(CurrentRow(BezoutCoefficient0.into()));
        let bc1 = circuit_builder.input(CurrentRow(BezoutCoefficient1.into()));
        let clk_di = circuit_builder.input(CurrentRow(InverseOfClkDiffMinusOne.into()));
        let rpcjd = circuit_builder.input(CurrentRow(AllClockJumpDifferencesPermArg.into()));
        let rppa = circuit_builder.input(CurrentRow(RunningProductPermArg.into()));

        let clk_next = circuit_builder.input(NextRow(CLK.into()));
        let ramp_next = circuit_builder.input(NextRow(RAMP.into()));
        let ramv_next = circuit_builder.input(NextRow(RAMV.into()));
        let bcpc0_next =
            circuit_builder.input(NextRow(BezoutCoefficientPolynomialCoefficient0.into()));
        let bcpc1_next =
            circuit_builder.input(NextRow(BezoutCoefficientPolynomialCoefficient1.into()));
        let rp_next = circuit_builder.input(NextRow(RunningProductOfRAMP.into()));
        let fd_next = circuit_builder.input(NextRow(FormalDerivative.into()));
        let bc0_next = circuit_builder.input(NextRow(BezoutCoefficient0.into()));
        let bc1_next = circuit_builder.input(NextRow(BezoutCoefficient1.into()));
        let rpcjd_next = circuit_builder.input(NextRow(AllClockJumpDifferencesPermArg.into()));
        let rppa_next = circuit_builder.input(NextRow(RunningProductPermArg.into()));

        let ramp_diff = ramp_next.clone() - ramp.clone();
        let ramp_changes = ramp_diff.clone() * iord.clone();

        // iord is 0 or iord is the inverse of (ramp' - ramp)
        let iord_is_0_or_iord_is_inverse_of_ramp_diff =
            iord.clone() * (ramp_changes.clone() - one.clone());

        // (ramp' - ramp) is zero or iord is the inverse of (ramp' - ramp)
        let ramp_diff_is_0_or_iord_is_inverse_of_ramp_diff =
            ramp_diff.clone() * (ramp_changes.clone() - one.clone());

        // The ramp does not change or the new ramv is 0
        let ramp_does_not_change_or_ramv_becomes_0 = ramp_diff.clone() * ramv_next.clone();

        // The ramp does change or the ramv does not change or the clk increases by 1
        let ramp_does_not_change_or_ramv_does_not_change_or_clk_increases_by_1 =
            (ramp_changes.clone() - one.clone())
                * (ramv_next.clone() - ramv)
                * (clk_next.clone() - (clk.clone() + one.clone()));

        let bcbp0_only_changes_if_ramp_changes =
            (one.clone() - ramp_changes.clone()) * (bcpc0_next.clone() - bcpc0);

        let bcbp1_only_changes_if_ramp_changes =
            (one.clone() - ramp_changes.clone()) * (bcpc1_next.clone() - bcpc1);

        let running_product_ramp_updates_correctly = ramp_diff.clone()
            * (rp_next.clone() - rp.clone() * (bezout_challenge.clone() - ramp_next.clone()))
            + (one.clone() - ramp_changes.clone()) * (rp_next - rp.clone());

        let formal_derivative_updates_correctly = ramp_diff.clone()
            * (fd_next.clone() - rp - (bezout_challenge.clone() - ramp_next.clone()) * fd.clone())
            + (one.clone() - ramp_changes.clone()) * (fd_next - fd);

        let bezout_coefficient_0_is_constructed_correctly = ramp_diff.clone()
            * (bc0_next.clone() - bezout_challenge.clone() * bc0.clone() - bcpc0_next)
            + (one.clone() - ramp_changes.clone()) * (bc0_next - bc0);

        let bezout_coefficient_1_is_constructed_correctly = ramp_diff
            * (bc1_next.clone() - bezout_challenge * bc1.clone() - bcpc1_next)
            + (one.clone() - ramp_changes) * (bc1_next - bc1);

        let clk_di_is_inverse_of_clkd =
            clk_di.clone() * (clk_next.clone() - clk.clone() - one.clone());
        let clk_di_is_zero_or_inverse_of_clkd = clk_di.clone() * clk_di_is_inverse_of_clkd.clone();
        let clkd_is_zero_or_inverse_of_clk_di =
            (clk_next.clone() - clk.clone() - one.clone()) * clk_di_is_inverse_of_clkd;

        let rpcjd_updates_correctly = (clk_next.clone() - clk.clone() - one.clone())
            * (rpcjd_next.clone() - rpcjd.clone())
            + (one.clone() - (ramp_next.clone() - ramp.clone()) * iord)
                * (rpcjd_next.clone() - rpcjd.clone())
            + (one.clone() - (clk_next.clone() - clk - one) * clk_di)
                * ramp.clone()
                * (rpcjd_next - rpcjd * (cjd_challenge - ramp));

        let compressed_row_for_permutation_argument =
            clk_next * clk_weight + ramp_next * ramp_weight + ramv_next * ramv_weight;
        let rppa_updates_correctly =
            rppa_next - rppa * (rppa_challenge - compressed_row_for_permutation_argument);

        [
            iord_is_0_or_iord_is_inverse_of_ramp_diff,
            ramp_diff_is_0_or_iord_is_inverse_of_ramp_diff,
            ramp_does_not_change_or_ramv_becomes_0,
            ramp_does_not_change_or_ramv_does_not_change_or_clk_increases_by_1,
            bcbp0_only_changes_if_ramp_changes,
            bcbp1_only_changes_if_ramp_changes,
            running_product_ramp_updates_correctly,
            formal_derivative_updates_correctly,
            bezout_coefficient_0_is_constructed_correctly,
            bezout_coefficient_1_is_constructed_correctly,
            clk_di_is_zero_or_inverse_of_clkd,
            clkd_is_zero_or_inverse_of_clk_di,
            rpcjd_updates_correctly,
            rppa_updates_correctly,
        ]
        .map(|circuit| circuit.consume())
        .to_vec()
    }

    pub fn ext_terminal_constraints_as_circuits(
    ) -> Vec<ConstraintCircuit<RamTableChallenges, SingleRowIndicator<FULL_WIDTH>>> {
        let circuit_builder = ConstraintCircuitBuilder::new(FULL_WIDTH);
        let one = circuit_builder.b_constant(1_u32.into());

        let rp = circuit_builder.input(Row(RunningProductOfRAMP.into()));
        let fd = circuit_builder.input(Row(FormalDerivative.into()));
        let bc0 = circuit_builder.input(Row(BezoutCoefficient0.into()));
        let bc1 = circuit_builder.input(Row(BezoutCoefficient1.into()));

        let bezout_relation_holds = bc0 * rp + bc1 * fd - one;

        vec![bezout_relation_holds.consume()]
    }
}

#[derive(Debug, Copy, Clone, Display, EnumCountMacro, EnumIter, PartialEq, Eq, Hash)]
pub enum RamTableChallengeId {
    BezoutRelationIndeterminate,
    ProcessorPermIndeterminate,
    ClkWeight,
    RamvWeight,
    RampWeight,
    AllClockJumpDifferencesMultiPermIndeterminate,
}

impl From<RamTableChallengeId> for usize {
    fn from(val: RamTableChallengeId) -> Self {
        val as usize
    }
}

#[derive(Debug, Clone)]
pub struct RamTableChallenges {
    /// The point in which the Bézout relation establishing contiguous memory regions is queried.
    pub bezout_relation_indeterminate: XFieldElement,

    /// Point of evaluation for the row set equality argument between RAM and Processor Tables
    pub processor_perm_indeterminate: XFieldElement,

    /// Weights for condensing part of a row into a single column. (Related to processor table.)
    pub clk_weight: XFieldElement,
    pub ramv_weight: XFieldElement,
    pub ramp_weight: XFieldElement,

    /// Point of evaluation for accumulating all clock jump differences into a running product
    pub all_clock_jump_differences_multi_perm_indeterminate: XFieldElement,
}

impl TableChallenges for RamTableChallenges {
    type Id = RamTableChallengeId;

    #[inline]
    fn get_challenge(&self, id: Self::Id) -> XFieldElement {
        match id {
            BezoutRelationIndeterminate => self.bezout_relation_indeterminate,
            ProcessorPermIndeterminate => self.processor_perm_indeterminate,
            ClkWeight => self.clk_weight,
            RamvWeight => self.ramv_weight,
            RampWeight => self.ramp_weight,
            AllClockJumpDifferencesMultiPermIndeterminate => {
                self.all_clock_jump_differences_multi_perm_indeterminate
            }
        }
    }
}

impl ExtensionTable for ExtRamTable {}
