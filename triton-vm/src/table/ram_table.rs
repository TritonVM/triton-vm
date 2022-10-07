use itertools::Itertools;
use num_traits::{One, Zero};
use strum::EnumCount;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::mpolynomial::{Degree, MPolynomial};
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::cross_table_arguments::{CrossTableArg, PermArg};
use crate::fri_domain::FriDomain;
use crate::table::base_table::Extendable;
use crate::table::extension_table::Evaluable;
use crate::table::table_column::RamBaseTableColumn::{self, *};
use crate::table::table_column::RamExtTableColumn::{self, *};

use super::base_table::{InheritsFromTable, Table, TableLike};
use super::challenges::AllChallenges;
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
    inherited_table: Table<XFieldElement>,
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
        challenges: &RamTableChallenges,
        interpolant_degree: Degree,
    ) -> ExtRamTable {
        let mut extension_matrix: Vec<Vec<XFieldElement>> = Vec::with_capacity(self.data().len());
        let mut running_product_for_perm_arg = PermArg::default_initial();

        // initialize columns establishing Bézout relation
        let mut running_product_of_ramp = challenges.bezout_relation_sample_point
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
                    let bcpc0 = extension_row[usize::from(BezoutCoefficientPolynomialCoefficient0)];
                    let bcpc1 = extension_row[usize::from(BezoutCoefficientPolynomialCoefficient1)];
                    let bezout_challenge = challenges.bezout_relation_sample_point;

                    formal_derivative =
                        (bezout_challenge - ramp) * formal_derivative + running_product_of_ramp;
                    running_product_of_ramp *= bezout_challenge - ramp;
                    bezout_coefficient_0 = bezout_coefficient_0 * bezout_challenge + bcpc0;
                    bezout_coefficient_1 = bezout_coefficient_1 * bezout_challenge + bcpc1;
                }
            }

            extension_row[usize::from(RunningProductOfRAMP)] = running_product_of_ramp;
            extension_row[usize::from(FormalDerivative)] = formal_derivative;
            extension_row[usize::from(BezoutCoefficient0)] = bezout_coefficient_0;
            extension_row[usize::from(BezoutCoefficient1)] = bezout_coefficient_1;

            // permutation argument to Processor Table
            let clk_w = challenges.clk_weight;
            let ramp_w = challenges.ramp_weight;
            let ramv_w = challenges.ramv_weight;

            // compress multiple values within one row so they become one value
            let compressed_row_for_permutation_argument =
                clk * clk_w + ramp * ramp_w + ramv * ramv_w;

            // compute the running product of the compressed column for permutation argument
            running_product_for_perm_arg *=
                challenges.processor_perm_row_weight - compressed_row_for_permutation_argument;
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
            self.mut_data().append(&mut vec![
                vec![BFieldElement::zero(); BASE_WIDTH];
                padded_height
            ]);
            return;
        }

        let idx = self
            .mut_data()
            .iter()
            .enumerate()
            .find(|(_, row)| row[usize::from(RamBaseTableColumn::CLK)].value() == max_clock)
            .map(|(idx, _)| idx)
            .unwrap();

        let padding_template = &mut self.mut_data()[idx];
        let difference_inverse =
            padding_template[usize::from(RamBaseTableColumn::InverseOfRampDifference)];
        padding_template[usize::from(RamBaseTableColumn::InverseOfRampDifference)] =
            BFieldElement::zero();

        let mut padding_rows = vec![];
        while padding_rows.len() < num_padding_rows {
            let mut padding_row = padding_template.clone();
            padding_row[usize::from(RamBaseTableColumn::CLK)] +=
                (padding_rows.len() as u32 + 1).into();
            padding_rows.push(padding_row)
        }

        if let Some(row) = padding_rows.last_mut() {
            row[usize::from(RamBaseTableColumn::InverseOfRampDifference)] = difference_inverse;
        }

        let insertion_index = idx + 1;
        let old_tail_length = self.data().len() - insertion_index;
        self.mut_data().append(&mut padding_rows);
        self.mut_data()[insertion_index..].rotate_left(old_tail_length);

        assert_eq!(padded_height, self.data().len());
    }
}

impl TableLike<XFieldElement> for ExtRamTable {}

impl ExtRamTable {
    fn ext_initial_constraints(challenges: &RamTableChallenges) -> Vec<MPolynomial<XFieldElement>> {
        use RamBaseTableColumn::*;

        let one = MPolynomial::from_constant(1.into(), FULL_WIDTH);
        let bezout_challenge =
            MPolynomial::from_constant(challenges.bezout_relation_sample_point, FULL_WIDTH);
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

        let clk_is_0 = clk;
        let ramp_is_0 = ramp.clone();
        let ramv_is_0 = ramv;
        let bezout_coefficient_polynomial_coefficient_0_is_0 = bcpc0;
        let bezout_coefficient_0_is_0 = bc0;
        let bezout_coefficient_1_is_bezout_coefficient_polynomial_coefficient_1 = bc1 - bcpc1;
        let formal_derivative_is_1 = fd - one;
        let running_product_is_initialized_correctly = rp - (bezout_challenge - ramp);

        vec![
            clk_is_0,
            ramp_is_0,
            ramv_is_0,
            bezout_coefficient_polynomial_coefficient_0_is_0,
            bezout_coefficient_0_is_0,
            bezout_coefficient_1_is_bezout_coefficient_polynomial_coefficient_1,
            formal_derivative_is_1,
            running_product_is_initialized_correctly,
        ]
    }

    fn ext_consistency_constraints(
        _challenges: &RamTableChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        // no further constraints
        vec![]
    }

    fn ext_transition_constraints(
        challenges: &RamTableChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        use RamBaseTableColumn::*;

        let variables: Vec<MPolynomial<XFieldElement>> = MPolynomial::variables(2 * FULL_WIDTH);
        let one = MPolynomial::from_constant(1.into(), 2 * FULL_WIDTH);

        let bezout_challenge =
            MPolynomial::from_constant(challenges.bezout_relation_sample_point, FULL_WIDTH);

        let clk = variables[usize::from(CLK)].clone();
        let ramp = variables[usize::from(RAMP)].clone();
        let ramv = variables[usize::from(RAMV)].clone();
        let iord = variables[usize::from(InverseOfRampDifference)].clone();
        let bcpc0 = variables[usize::from(BezoutCoefficientPolynomialCoefficient0)].clone();
        let bcpc1 = variables[usize::from(BezoutCoefficientPolynomialCoefficient1)].clone();
        let rp = variables[usize::from(RunningProductOfRAMP)].clone();
        let fd = variables[usize::from(FormalDerivative)].clone();
        let bc0 = variables[usize::from(BezoutCoefficient0)].clone();
        let bc1 = variables[usize::from(BezoutCoefficient1)].clone();

        let clk_next = variables[FULL_WIDTH + usize::from(CLK)].clone();
        let ramp_next = variables[FULL_WIDTH + usize::from(RAMP)].clone();
        let ramv_next = variables[FULL_WIDTH + usize::from(RAMV)].clone();
        let bcpc0_next =
            variables[FULL_WIDTH + usize::from(BezoutCoefficientPolynomialCoefficient0)].clone();
        let bcpc1_next =
            variables[FULL_WIDTH + usize::from(BezoutCoefficientPolynomialCoefficient1)].clone();
        let rp_next = variables[FULL_WIDTH + usize::from(RunningProductOfRAMP)].clone();
        let fd_next = variables[FULL_WIDTH + usize::from(FormalDerivative)].clone();
        let bc0_next = variables[FULL_WIDTH + usize::from(BezoutCoefficient0)].clone();
        let bc1_next = variables[FULL_WIDTH + usize::from(BezoutCoefficient1)].clone();

        let ramp_diff = ramp_next.clone() - ramp;
        let ramp_changes = ramp_diff.clone() * iord.clone();

        // iord is 0 or iord is the inverse of (ramp' - ramp)
        let iord_is_0_or_iord_is_inverse_of_ramp_diff = iord * (ramp_changes.clone() - one.clone());

        // (ramp' - ramp) is zero or iord is the inverse of (ramp' - ramp)
        let ramp_diff_is_0_or_iord_is_inverse_of_ramp_diff =
            ramp_diff.clone() * (ramp_changes.clone() - one.clone());

        // The ramp does not change or the new ramv is 0
        let ramp_does_not_change_or_ramv_becomes_0 = ramp_diff.clone() * ramv_next.clone();

        // The ramp does change or the ramv does not change or the clk increases by 1
        let ramp_does_not_change_or_ramv_does_not_change_or_clk_increases_by_1 =
            (ramp_changes.clone() - one.clone())
                * (ramv_next - ramv)
                * (clk_next - (clk + one.clone()));

        let bcbp0_only_changes_if_ramp_changes =
            (one.clone() - ramp_changes.clone()) * (bcpc0_next.clone() - bcpc0);

        let bcbp1_only_changes_if_ramp_changes =
            (one.clone() - ramp_changes.clone()) * (bcpc1_next.clone() - bcpc1);

        let running_product_ramp_updates_correctly = ramp_diff.clone()
            * (rp_next.clone() - rp.clone() * (bezout_challenge.clone() - ramp_next.clone()))
            + (one.clone() - ramp_changes.clone()) * (rp_next - rp.clone());

        let formal_derivative_updates_correctly = ramp_diff.clone()
            * (fd_next.clone() - rp - (bezout_challenge.clone() - ramp_next) * fd.clone())
            + (one.clone() - ramp_changes.clone()) * (fd_next - fd);

        let bezout_coefficient_0_is_constructed_correctly = ramp_diff.clone()
            * (bc0_next.clone() - bezout_challenge.clone() * bc0.clone() - bcpc0_next)
            + (one.clone() - ramp_changes.clone()) * (bc0_next - bc0);

        let bezout_coefficient_1_is_constructed_correctly = ramp_diff
            * (bc1_next.clone() - bezout_challenge * bc1.clone() - bcpc1_next)
            + (one - ramp_changes) * (bc1_next - bc1);

        vec![
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
        ]
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

#[derive(Debug, Clone)]
pub struct RamTableChallenges {
    /// The point in which the Bézout relation establishing contiguous memory regions is queried.
    pub bezout_relation_sample_point: XFieldElement,

    /// The weight that combines two consecutive rows in the
    /// permutation/evaluation column of the op-stack table.
    pub processor_perm_row_weight: XFieldElement,

    /// Weights for condensing part of a row into a single column. (Related to processor table.)
    pub clk_weight: XFieldElement,
    pub ramv_weight: XFieldElement,
    pub ramp_weight: XFieldElement,
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
