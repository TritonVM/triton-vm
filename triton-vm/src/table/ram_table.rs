use itertools::Itertools;
use num_traits::Zero;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::mpolynomial::MPolynomial;
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::fri_domain::FriDomain;
use crate::stark::StarkHasher;
use crate::table::base_table::Extendable;
use crate::table::extension_table::Evaluable;

use super::base_table::{self, InheritsFromTable, Table, TableLike};
use super::challenges_endpoints::{AllChallenges, AllEndpoints};
use super::extension_table::{ExtensionTable, Quotientable, QuotientableExtensionTable};
use super::table_column::RamTableColumn::{self, *};

pub const RAM_TABLE_PERMUTATION_ARGUMENTS_COUNT: usize = 1;
pub const RAM_TABLE_EVALUATION_ARGUMENT_COUNT: usize = 0;
pub const RAM_TABLE_INITIALS_COUNT: usize =
    RAM_TABLE_PERMUTATION_ARGUMENTS_COUNT + RAM_TABLE_EVALUATION_ARGUMENT_COUNT;

/// This is 3 because it combines: clk, ramv, ramp
pub const RAM_TABLE_EXTENSION_CHALLENGE_COUNT: usize = 3;

pub const BASE_WIDTH: usize = 4;
pub const FULL_WIDTH: usize = 6; // BASE_WIDTH + 2 * INITIALS_COUNT

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
    pub fn new_prover(num_trace_randomizers: usize, matrix: Vec<Vec<BFieldElement>>) -> Self {
        let unpadded_height = matrix.len();
        let padded_height = base_table::padded_height(unpadded_height);

        let omicron = base_table::derive_omicron(padded_height as u64);
        let inherited_table = Table::new(
            BASE_WIDTH,
            FULL_WIDTH,
            padded_height,
            num_trace_randomizers,
            omicron,
            matrix,
            "RamTable".to_string(),
        );

        Self { inherited_table }
    }

    pub fn codeword_table(&self, fri_domain: &FriDomain<BFieldElement>) -> Self {
        let base_columns = 0..self.base_width();
        let codewords = self.low_degree_extension(fri_domain, base_columns);

        let inherited_table = self.inherited_table.with_data(codewords);
        Self { inherited_table }
    }

    pub fn extend(
        &self,
        challenges: &RamTableChallenges,
        initials: &RamTableEndpoints,
    ) -> (ExtRamTable, RamTableEndpoints) {
        let mut extension_matrix: Vec<Vec<XFieldElement>> = Vec::with_capacity(self.data().len());
        let mut running_product = initials.processor_perm_product;

        for row in self.data().iter() {
            let mut extension_row = Vec::with_capacity(FULL_WIDTH);
            extension_row.extend(row.iter().map(|elem| elem.lift()));

            let (clk, ramp, ramv) = (
                extension_row[CLK as usize],
                extension_row[RAMP as usize],
                extension_row[RAMV as usize],
            );

            let (clk_w, ramp_w, ramv_w) = (
                challenges.clk_weight,
                challenges.ramp_weight,
                challenges.ramv_weight,
            );

            // 1. Compress multiple values within one row so they become one value.
            let compressed_row_for_permutation_argument =
                clk * clk_w + ramp * ramp_w + ramv * ramv_w;

            extension_row.push(compressed_row_for_permutation_argument);

            // 2. Compute the running *product* of the compressed column (permutation value)
            extension_row.push(running_product);
            running_product *=
                challenges.processor_perm_row_weight - compressed_row_for_permutation_argument;

            extension_matrix.push(extension_row);
        }

        let terminals = RamTableEndpoints {
            processor_perm_product: running_product,
        };

        let inherited_table = self.extension(
            extension_matrix,
            ExtRamTable::ext_boundary_constraints(),
            ExtRamTable::ext_transition_constraints(challenges),
            ExtRamTable::ext_consistency_constraints(challenges),
            ExtRamTable::ext_terminal_constraints(challenges, &terminals),
        );
        (ExtRamTable { inherited_table }, terminals)
    }

    pub fn for_verifier(
        num_trace_randomizers: usize,
        padded_height: usize,
        all_challenges: &AllChallenges,
        all_terminals: &AllEndpoints<StarkHasher>,
    ) -> ExtRamTable {
        let omicron = base_table::derive_omicron(padded_height as u64);
        let inherited_table = Table::new(
            BASE_WIDTH,
            FULL_WIDTH,
            padded_height,
            num_trace_randomizers,
            omicron,
            vec![],
            "ExtRamTable".to_string(),
        );
        let base_table = Self { inherited_table };
        let empty_matrix: Vec<Vec<XFieldElement>> = vec![];
        let extension_table = base_table.extension(
            empty_matrix,
            ExtRamTable::ext_boundary_constraints(),
            ExtRamTable::ext_transition_constraints(&all_challenges.ram_table_challenges),
            ExtRamTable::ext_consistency_constraints(&all_challenges.ram_table_challenges),
            ExtRamTable::ext_terminal_constraints(
                &all_challenges.ram_table_challenges,
                &all_terminals.ram_table_endpoints,
            ),
        );

        ExtRamTable {
            inherited_table: extension_table,
        }
    }
}

impl ExtRamTable {
    pub fn with_padded_height(num_trace_randomizers: usize, padded_height: usize) -> Self {
        let matrix: Vec<Vec<XFieldElement>> = vec![];

        let omicron = base_table::derive_omicron(padded_height as u64);
        let inherited_table = Table::new(
            BASE_WIDTH,
            FULL_WIDTH,
            padded_height,
            num_trace_randomizers,
            omicron,
            matrix,
            "ExtRamTable".to_string(),
        );

        Self { inherited_table }
    }

    pub fn ext_codeword_table(
        &self,
        fri_domain: &FriDomain<XFieldElement>,
        base_codewords: &[Vec<BFieldElement>],
    ) -> Self {
        let ext_columns = self.base_width()..self.full_width();
        let ext_codewords = self.low_degree_extension(fri_domain, ext_columns);

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

    fn pad(&mut self) {
        let max_clock = self.data().len() as u64 - 1;
        let padded_height = self.padded_height();
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
            .find(|(_, row)| row[RamTableColumn::CLK as usize].value() == max_clock)
            .map(|(idx, _)| idx)
            .unwrap();

        let padding_template = &mut self.mut_data()[idx];
        let difference_inverse = padding_template[RamTableColumn::InverseOfRampDifference as usize];
        padding_template[RamTableColumn::InverseOfRampDifference as usize] = BFieldElement::zero();

        let mut padding_rows = vec![];
        while padding_rows.len() < num_padding_rows {
            let mut padding_row = padding_template.clone();
            padding_row[RamTableColumn::CLK as usize] += (padding_rows.len() as u32 + 1).into();
            padding_rows.push(padding_row)
        }

        if let Some(row) = padding_rows.last_mut() {
            row[RamTableColumn::InverseOfRampDifference as usize] = difference_inverse;
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
    fn ext_boundary_constraints() -> Vec<MPolynomial<XFieldElement>> {
        use RamTableColumn::*;

        let variables: Vec<MPolynomial<XFieldElement>> =
            MPolynomial::variables(FULL_WIDTH, 1.into());
        let clk = variables[usize::from(CLK)].clone();
        let ramp = variables[usize::from(RAMP)].clone();
        let ramv = variables[usize::from(RAMV)].clone();

        // Cycle count clk is 0.
        let clk_is_0 = clk;

        // RAM pointer ramp is 0.
        let ramp_is_0 = ramp;

        // RAM value ramv is 0.
        let ramv_is_0 = ramv;

        vec![clk_is_0, ramp_is_0, ramv_is_0]
    }

    fn ext_consistency_constraints(
        _challenges: &RamTableChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        // no further constraints
        vec![]
    }

    fn ext_transition_constraints(
        _challenges: &RamTableChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        use RamTableColumn::*;

        let variables: Vec<MPolynomial<XFieldElement>> =
            MPolynomial::variables(2 * FULL_WIDTH, 1.into());
        let one = MPolynomial::from_constant(1.into(), 2 * FULL_WIDTH);

        let clk = variables[usize::from(CLK)].clone();
        let ramp = variables[usize::from(RAMP)].clone();
        let ramv = variables[usize::from(RAMV)].clone();
        let hv6 = variables[usize::from(InverseOfRampDifference)].clone();

        let clk_next = variables[FULL_WIDTH + usize::from(CLK)].clone();
        let ramp_next = variables[FULL_WIDTH + usize::from(RAMP)].clone();
        let ramv_next = variables[FULL_WIDTH + usize::from(RAMV)].clone();

        let ramp_diff = ramp_next - ramp;

        // hv6 is 0 or hv6 is the inverse of (ramp' - ramp).
        //
        // $ hv6·(hv6·(ramp' - ramp) - 1) = 0 $
        let hv6_is_0_or_hv6_is_inverse_of_ramp_diff =
            hv6.clone() * (hv6.clone() * ramp_diff.clone() - one.clone());

        // (ramp' - ramp) is zero or hv6 is the inverse of (ramp' - ramp).
        //
        // $ (ramp' - ramp)·(hv6·(ramp' - ramp) - 1) = 0 $
        let ramp_diff_is_0_or_hv6_is_inverse_of_ramp_diff =
            ramp_diff.clone() * (hv6.clone() * ramp_diff.clone() - one.clone());

        // The ramp does not change or the new ramv is 0.
        //
        // (ramp' - ramp)·ramv'
        let ramp_does_not_change_or_ramv_becomes_0 = ramp_diff.clone() * ramv_next.clone();

        // The ramp does change or the ramv does not change or the clk increases by 1.
        //
        // $ (hv6·(ramp' - ramp) - 1)·(ramv' - ramv)·(clk' - (clk + 1)) = 0 $
        let ramp_does_not_change_or_ramv_does_not_change_or_clk_increases_by_1 =
            (hv6 * ramp_diff - one.clone()) * (ramv_next - ramv) * (clk_next - (clk + one));

        vec![
            hv6_is_0_or_hv6_is_inverse_of_ramp_diff,
            ramp_diff_is_0_or_hv6_is_inverse_of_ramp_diff,
            ramp_does_not_change_or_ramv_becomes_0,
            ramp_does_not_change_or_ramv_does_not_change_or_clk_increases_by_1,
        ]
    }

    fn ext_terminal_constraints(
        _challenges: &RamTableChallenges,
        _terminals: &RamTableEndpoints,
    ) -> Vec<MPolynomial<XFieldElement>> {
        // no further constraints
        vec![]
    }
}

#[derive(Debug, Clone)]
pub struct RamTableChallenges {
    /// The weight that combines two consecutive rows in the
    /// permutation/evaluation column of the op-stack table.
    pub processor_perm_row_weight: XFieldElement,

    /// Weights for condensing part of a row into a single column. (Related to processor table.)
    pub clk_weight: XFieldElement,
    pub ramv_weight: XFieldElement,
    pub ramp_weight: XFieldElement,
}

#[derive(Debug, Clone)]
pub struct RamTableEndpoints {
    /// Values randomly generated by the prover for zero-knowledge.
    pub processor_perm_product: XFieldElement,
}

impl ExtensionTable for ExtRamTable {
    fn dynamic_boundary_constraints(&self) -> Vec<MPolynomial<XFieldElement>> {
        ExtRamTable::ext_boundary_constraints()
    }

    fn dynamic_transition_constraints(
        &self,
        challenges: &super::challenges_endpoints::AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtRamTable::ext_transition_constraints(&challenges.ram_table_challenges)
    }

    fn dynamic_consistency_constraints(
        &self,
        challenges: &AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtRamTable::ext_consistency_constraints(&challenges.ram_table_challenges)
    }

    fn dynamic_terminal_constraints(
        &self,
        challenges: &super::challenges_endpoints::AllChallenges,
        terminals: &super::challenges_endpoints::AllEndpoints<StarkHasher>,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtRamTable::ext_terminal_constraints(
            &challenges.ram_table_challenges,
            &terminals.ram_table_endpoints,
        )
    }
}
