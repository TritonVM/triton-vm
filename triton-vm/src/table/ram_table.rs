use itertools::Itertools;
use num_traits::{One, Zero};
use strum::EnumCount;
use strum_macros::{Display, EnumCount as EnumCountMacro, EnumIter};
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::mpolynomial::{Degree, MPolynomial};
use twenty_first::shared_math::traits::Inverse;
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::cross_table_arguments::{CrossTableArg, PermArg};
use crate::fri_domain::FriDomain;
use crate::table::base_table::Extendable;
use crate::table::extension_table::Evaluable;
use crate::table::table_column::RamBaseTableColumn::{self, *};
use crate::table::table_column::RamExtTableColumn::{self, *};

use super::base_table::{InheritsFromTable, Table, TableLike};
use super::challenges::{AllChallenges, TableChallenges};
use super::constraint_circuit::{ConstraintCircuit, ConstraintCircuitBuilder};
use super::extension_table::{ExtensionTable, Quotientable, QuotientableExtensionTable};

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

impl Evaluable for ExtRamTable {}
impl Quotientable for ExtRamTable {}
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
    pub fn new_prover(matrix: Vec<Vec<BFieldElement>>) -> Self {
        let inherited_table = Table::new(BASE_WIDTH, FULL_WIDTH, matrix, "RamTable".to_string());
        Self { inherited_table }
    }

    pub fn to_fri_domain_table(
        &self,
        fri_domain: &FriDomain<BFieldElement>,
        omicron: BFieldElement,
        padded_height: usize,
        num_trace_randomizers: usize,
    ) -> Self {
        let base_columns = 0..self.base_width();
        let fri_domain_codewords = self.low_degree_extension(
            fri_domain,
            omicron,
            padded_height,
            num_trace_randomizers,
            base_columns,
        );
        let inherited_table = self.inherited_table.with_data(fri_domain_codewords);
        Self { inherited_table }
    }

    pub fn extend(
        &self,
        challenges: &RamTableChallenges,
        interpolant_degree: Degree,
    ) -> ExtRamTable {
        let mut extension_matrix: Vec<Vec<XFieldElement>> = Vec::with_capacity(self.data().len());
        let mut running_product_for_perm_arg = PermArg::default_initial();
        let mut all_clock_jump_differences_running_product = PermArg::default_initial();

        // initialize columns establishing Bézout relation
        let mut running_product_of_ramp = challenges.bezout_relation_indeterminate
            - XFieldElement::new_const(self.data().first().unwrap()[usize::from(RAMP)]);
        let mut formal_derivative = XFieldElement::one();
        let mut bezout_coefficient_0 = XFieldElement::zero();
        let mut bezout_coefficient_1 = self.data().first().unwrap()
            [usize::from(BezoutCoefficientPolynomialCoefficient1)]
        .lift();

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
        let padded_height = extension_matrix.len();
        let inherited_table = self.extension(
            extension_matrix,
            interpolant_degree,
            padded_height,
            ExtRamTable::ext_initial_constraints(challenges),
            ExtRamTable::ext_consistency_constraints(challenges),
            ExtRamTable::ext_transition_constraints(challenges),
            ExtRamTable::ext_terminal_constraints(challenges),
        );
        ExtRamTable { inherited_table }
    }

    pub fn for_verifier(
        interpolant_degree: Degree,
        padded_height: usize,
        all_challenges: &AllChallenges,
    ) -> ExtRamTable {
        let inherited_table = Table::new(BASE_WIDTH, FULL_WIDTH, vec![], "ExtRamTable".to_string());
        let base_table = Self { inherited_table };
        let empty_matrix: Vec<Vec<XFieldElement>> = vec![];
        let extension_table = base_table.extension(
            empty_matrix,
            interpolant_degree,
            padded_height,
            ExtRamTable::ext_initial_constraints(&all_challenges.ram_table_challenges),
            ExtRamTable::ext_consistency_constraints(&all_challenges.ram_table_challenges),
            ExtRamTable::ext_transition_constraints(&all_challenges.ram_table_challenges),
            ExtRamTable::ext_terminal_constraints(&all_challenges.ram_table_challenges),
        );

        ExtRamTable {
            inherited_table: extension_table,
        }
    }
}

impl ExtRamTable {
    pub fn to_fri_domain_table(
        &self,
        fri_domain: &FriDomain<XFieldElement>,
        omicron: XFieldElement,
        padded_height: usize,
        num_trace_randomizers: usize,
    ) -> Self {
        let ext_columns = self.base_width()..self.full_width();
        let fri_domain_codewords_ext = self.low_degree_extension(
            fri_domain,
            omicron,
            padded_height,
            num_trace_randomizers,
            ext_columns,
        );

        let inherited_table = self.inherited_table.with_data(fri_domain_codewords_ext);
        ExtRamTable { inherited_table }
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
                    (clk_diff - 1_u64.into()).inverse_or_zero();
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
    fn ext_initial_constraints(challenges: &RamTableChallenges) -> Vec<MPolynomial<XFieldElement>> {
        let constant = |xfe| MPolynomial::from_constant(xfe, FULL_WIDTH);
        let one = constant(XFieldElement::one());
        let bezout_challenge = constant(challenges.bezout_relation_indeterminate);
        let rppa_challenge = constant(challenges.processor_perm_indeterminate);

        let variables: Vec<MPolynomial<XFieldElement>> = MPolynomial::variables(FULL_WIDTH);
        let clk = variables[usize::from(CLK)].clone();
        let ramp = variables[usize::from(RAMP)].clone();
        let ramv = variables[usize::from(RAMV)].clone();
        let bcpc0 = variables[usize::from(BezoutCoefficientPolynomialCoefficient0)].clone();
        let bcpc1 = variables[usize::from(BezoutCoefficientPolynomialCoefficient1)].clone();
        let rp = variables[usize::from(RunningProductOfRAMP)].clone();
        let fd = variables[usize::from(FormalDerivative)].clone();
        let bc0 = variables[usize::from(BezoutCoefficient0)].clone();
        let bc1 = variables[usize::from(BezoutCoefficient1)].clone();
        let rppa = variables[usize::from(RunningProductPermArg)].clone();

        let clk_is_0 = clk;
        let ramp_is_0 = ramp.clone();
        let ramv_is_0 = ramv;
        let bezout_coefficient_polynomial_coefficient_0_is_0 = bcpc0;
        let bezout_coefficient_0_is_0 = bc0;
        let bezout_coefficient_1_is_bezout_coefficient_polynomial_coefficient_1 = bc1 - bcpc1;
        let formal_derivative_is_1 = fd - one;
        let running_product_polynomial_is_initialized_correctly = rp - (bezout_challenge - ramp);
        let running_product_permutation_argument_is_initialized_correctly = rppa - rppa_challenge;

        vec![
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
    }

    fn ext_consistency_constraints(
        _challenges: &RamTableChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        // no further constraints
        vec![]
    }

    pub fn ext_transition_constraints_as_circuits() -> Vec<ConstraintCircuit<RamTableChallenges>> {
        let mut circuit_builder = ConstraintCircuitBuilder::new(2 * FULL_WIDTH);
        let one = circuit_builder.constant(1.into());

        let bezout_challenge =
            circuit_builder.challenge(RamTableChallengesId::BezoutRelationIndeterminate);
        let cjd_challenge = circuit_builder
            .challenge(RamTableChallengesId::AllClockJumpDifferencesMultiPermIndeterminate);
        let rppa_challenge =
            circuit_builder.challenge(RamTableChallengesId::ProcessorPermIndeterminate);
        let clk_weight = circuit_builder.challenge(RamTableChallengesId::ClkWeight);
        let ramp_weight = circuit_builder.challenge(RamTableChallengesId::RampWeight);
        let ramv_weight = circuit_builder.challenge(RamTableChallengesId::RamvWeight);

        let clk = circuit_builder.input(usize::from(CLK));
        let ramp = circuit_builder.input(usize::from(RAMP));
        let ramv = circuit_builder.input(usize::from(RAMV));
        let iord = circuit_builder.input(usize::from(InverseOfRampDifference));
        let bcpc0 = circuit_builder.input(usize::from(BezoutCoefficientPolynomialCoefficient0));
        let bcpc1 = circuit_builder.input(usize::from(BezoutCoefficientPolynomialCoefficient1));
        let rp = circuit_builder.input(usize::from(RunningProductOfRAMP));
        let fd = circuit_builder.input(usize::from(FormalDerivative));
        let bc0 = circuit_builder.input(usize::from(BezoutCoefficient0));
        let bc1 = circuit_builder.input(usize::from(BezoutCoefficient1));
        let clk_di = circuit_builder.input(usize::from(InverseOfClkDiffMinusOne));
        let rpcjd = circuit_builder.input(usize::from(AllClockJumpDifferencesPermArg));
        let rppa = circuit_builder.input(usize::from(RunningProductPermArg));

        let clk_next = circuit_builder.input(FULL_WIDTH + usize::from(CLK));
        let ramp_next = circuit_builder.input(FULL_WIDTH + usize::from(RAMP));
        let ramv_next = circuit_builder.input(FULL_WIDTH + usize::from(RAMV));
        let bcpc0_next = circuit_builder
            .input(FULL_WIDTH + usize::from(BezoutCoefficientPolynomialCoefficient0));
        let bcpc1_next = circuit_builder
            .input(FULL_WIDTH + usize::from(BezoutCoefficientPolynomialCoefficient1));
        let rp_next = circuit_builder.input(FULL_WIDTH + usize::from(RunningProductOfRAMP));
        let fd_next = circuit_builder.input(FULL_WIDTH + usize::from(FormalDerivative));
        let bc0_next = circuit_builder.input(FULL_WIDTH + usize::from(BezoutCoefficient0));
        let bc1_next = circuit_builder.input(FULL_WIDTH + usize::from(BezoutCoefficient1));
        let rpcjd_next =
            circuit_builder.input(FULL_WIDTH + usize::from(AllClockJumpDifferencesPermArg));
        let rppa_next = circuit_builder.input(FULL_WIDTH + usize::from(RunningProductPermArg));

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

        vec![
            iord_is_0_or_iord_is_inverse_of_ramp_diff.consume(),
            ramp_diff_is_0_or_iord_is_inverse_of_ramp_diff.consume(),
            ramp_does_not_change_or_ramv_becomes_0.consume(),
            ramp_does_not_change_or_ramv_does_not_change_or_clk_increases_by_1.consume(),
            bcbp0_only_changes_if_ramp_changes.consume(),
            bcbp1_only_changes_if_ramp_changes.consume(),
            running_product_ramp_updates_correctly.consume(),
            formal_derivative_updates_correctly.consume(),
            bezout_coefficient_0_is_constructed_correctly.consume(),
            bezout_coefficient_1_is_constructed_correctly.consume(),
            clk_di_is_zero_or_inverse_of_clkd.consume(),
            clkd_is_zero_or_inverse_of_clk_di.consume(),
            rpcjd_updates_correctly.consume(),
            rppa_updates_correctly.consume(),
        ]
    }

    fn ext_transition_constraints(
        challenges: &RamTableChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        let circuits = Self::ext_transition_constraints_as_circuits();
        let mut ret: Vec<MPolynomial<XFieldElement>> = vec![];
        for circuit in circuits {
            ret.push(circuit.partial_evaluate(challenges));
        }

        ret
    }

    fn ext_terminal_constraints(
        _challenges: &RamTableChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        let one = MPolynomial::from_constant(1.into(), FULL_WIDTH);
        let variables = MPolynomial::variables(FULL_WIDTH);

        let rp = variables[usize::from(RunningProductOfRAMP)].clone();
        let fd = variables[usize::from(FormalDerivative)].clone();
        let bc0 = variables[usize::from(BezoutCoefficient0)].clone();
        let bc1 = variables[usize::from(BezoutCoefficient1)].clone();

        let bezout_relation_holds = bc0 * rp + bc1 * fd - one;

        vec![bezout_relation_holds]
    }
}

#[derive(Debug, Copy, Clone, Display, EnumCountMacro, EnumIter, PartialEq, Hash)]
pub enum RamTableChallengesId {
    BezoutRelationIndeterminate,
    ProcessorPermIndeterminate,
    ClkWeight,
    RamvWeight,
    RampWeight,
    AllClockJumpDifferencesMultiPermIndeterminate,
}

impl Eq for RamTableChallengesId {}

impl From<RamTableChallengesId> for usize {
    fn from(val: RamTableChallengesId) -> Self {
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
    type Id = RamTableChallengesId;

    fn get_challenge(&self, id: Self::Id) -> XFieldElement {
        match id {
            RamTableChallengesId::BezoutRelationIndeterminate => self.bezout_relation_indeterminate,
            RamTableChallengesId::ProcessorPermIndeterminate => self.processor_perm_indeterminate,
            RamTableChallengesId::ClkWeight => self.clk_weight,
            RamTableChallengesId::RamvWeight => self.ramv_weight,
            RamTableChallengesId::RampWeight => self.ramp_weight,
            RamTableChallengesId::AllClockJumpDifferencesMultiPermIndeterminate => {
                self.all_clock_jump_differences_multi_perm_indeterminate
            }
        }
    }
}

impl ExtensionTable for ExtRamTable {
    fn dynamic_initial_constraints(
        &self,
        challenges: &AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtRamTable::ext_initial_constraints(&challenges.ram_table_challenges)
    }

    fn dynamic_consistency_constraints(
        &self,
        challenges: &AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtRamTable::ext_consistency_constraints(&challenges.ram_table_challenges)
    }

    fn dynamic_transition_constraints(
        &self,
        challenges: &super::challenges::AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtRamTable::ext_transition_constraints(&challenges.ram_table_challenges)
    }

    fn dynamic_terminal_constraints(
        &self,
        challenges: &AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtRamTable::ext_terminal_constraints(&challenges.ram_table_challenges)
    }
}
