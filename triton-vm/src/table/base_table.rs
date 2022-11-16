use super::super::arithmetic_domain::ArithmeticDomain;
use crate::table::table_collection::derive_trace_domain_generator;
use num_traits::Zero;
use rand_distr::{Distribution, Standard};
use rayon::iter::{IntoParallelIterator, IntoParallelRefIterator, ParallelIterator};
use std::ops::{Mul, MulAssign, Range};
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::mpolynomial::{Degree, MPolynomial};
use twenty_first::shared_math::other::{random_elements, roundup_npo2};
use twenty_first::shared_math::polynomial::Polynomial;
use twenty_first::shared_math::traits::{FiniteField, PrimitiveRootOfUnity};
use twenty_first::shared_math::x_field_element::XFieldElement;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Table<FieldElement: FiniteField> {
    /// The width of each `data` row in the base version of the table
    base_width: usize,

    /// The width of each `data` row in the extended version of the table
    full_width: usize,

    /// The table data (trace data). Represents every intermediate
    matrix: Vec<Vec<FieldElement>>,

    /// The name of the table. Mostly for debugging purpose.
    pub name: String,

    /// AIR constraints, to be populated upon extension
    pub(crate) initial_constraints: Option<Vec<MPolynomial<FieldElement>>>,
    pub(crate) consistency_constraints: Option<Vec<MPolynomial<FieldElement>>>,
    pub(crate) transition_constraints: Option<Vec<MPolynomial<FieldElement>>>,
    pub(crate) terminal_constraints: Option<Vec<MPolynomial<FieldElement>>>,

    /// quotient degrees, to be populated upon extension
    pub(crate) initial_quotient_degree_bounds: Option<Vec<i64>>,
    pub(crate) consistency_quotient_degree_bounds: Option<Vec<i64>>,
    pub(crate) transition_quotient_degree_bounds: Option<Vec<i64>>,
    pub(crate) terminal_quotient_degree_bounds: Option<Vec<i64>>,
}

#[allow(clippy::too_many_arguments)]
impl<FF: FiniteField> Table<FF> {
    pub fn new(base_width: usize, full_width: usize, matrix: Vec<Vec<FF>>, name: String) -> Self {
        Table {
            base_width,
            full_width,
            matrix,
            name,
            initial_constraints: None,
            consistency_constraints: None,
            transition_constraints: None,
            terminal_constraints: None,
            initial_quotient_degree_bounds: None,
            consistency_quotient_degree_bounds: None,
            transition_quotient_degree_bounds: None,
            terminal_quotient_degree_bounds: None,
        }
    }

    /// Create a `BaseTable<FF>` with the same parameters, but new `matrix` data.
    pub fn with_data(&self, matrix: Vec<Vec<FF>>) -> Self {
        Table {
            matrix,
            name: format!("{} with data", self.name),
            ..self.to_owned()
        }
    }
}

pub trait InheritsFromTable<FF: FiniteField> {
    fn inherited_table(&self) -> &Table<FF>;
    fn mut_inherited_table(&mut self) -> &mut Table<FF>;

    fn base_width(&self) -> usize {
        self.inherited_table().base_width
    }

    fn full_width(&self) -> usize {
        self.inherited_table().full_width
    }

    fn data(&self) -> &Vec<Vec<FF>> {
        &self.inherited_table().matrix
    }

    fn mut_data(&mut self) -> &mut Vec<Vec<FF>> {
        &mut self.mut_inherited_table().matrix
    }
}

pub trait Extendable: TableLike<BFieldElement> {
    // Abstract functions that individual structs implement

    /// Computes some (or all) padding rows and, if appropriate, the index where they are to be
    /// inserted.
    fn get_padding_rows(&self) -> (Option<usize>, Vec<Vec<BFieldElement>>);

    // Generic functions common to all extendable tables

    fn new_from_lifted_matrix(&self, matrix: Vec<Vec<XFieldElement>>) -> Table<XFieldElement> {
        Table::new(
            self.base_width(),
            self.full_width(),
            matrix,
            format!("{} with lifted matrix", self.name()),
        )
    }
    /// Add padding to a table so that its height becomes the same as other tables. Uses
    /// table-specific padding via `.get_padding_rows()`, which might specify an insertion index for
    /// the padding row(s).
    fn pad(&mut self, padded_height: usize) {
        while self.data().len() != padded_height {
            let (maybe_index, mut rows) = self.get_padding_rows();
            match maybe_index {
                Some(idx) => {
                    let old_tail_length = self.data().len() - idx;
                    self.mut_data().append(&mut rows);
                    self.mut_data()[idx..].rotate_left(old_tail_length);
                }
                None => self.mut_data().append(&mut rows),
            }
        }
        assert_eq!(padded_height, self.data().len());
    }

    /// Computes the degree bounds of the quotients given the AIR constraints and the interpolant
    /// degree. The AIR constraints are defined over a symbolic ring with `full_width`-many
    /// variables.
    fn compute_degree_bounds(
        air_constraints: &[MPolynomial<XFieldElement>],
        interpolant_degree: Degree,
        zerofier_degree: Degree,
        full_width: usize,
    ) -> Vec<Degree> {
        air_constraints
            .iter()
            .map(|mpo| {
                mpo.symbolic_degree_bound(&vec![interpolant_degree; full_width]) - zerofier_degree
            })
            .collect()
    }

    fn get_initial_quotient_degree_bounds(
        &self,
        initial_constraints: &[MPolynomial<XFieldElement>],
        interpolant_degree: Degree,
    ) -> Vec<Degree> {
        let zerofier_degree = 1;
        let full_width = self.full_width();
        Self::compute_degree_bounds(
            initial_constraints,
            interpolant_degree,
            zerofier_degree,
            full_width,
        )
    }

    fn get_consistency_quotient_degree_bounds(
        &self,
        consistency_constraints: &[MPolynomial<XFieldElement>],
        interpolant_degree: Degree,
        padded_height: usize,
    ) -> Vec<Degree> {
        let zerofier_degree = padded_height as Degree;
        let full_width = self.full_width();
        Self::compute_degree_bounds(
            consistency_constraints,
            interpolant_degree,
            zerofier_degree,
            full_width,
        )
    }

    fn get_transition_quotient_degree_bounds(
        &self,
        transition_constraints: &[MPolynomial<XFieldElement>],
        interpolant_degree: Degree,
        padded_height: usize,
    ) -> Vec<Degree> {
        let zerofier_degree = (padded_height - 1) as Degree;
        let full_width = self.full_width();
        Self::compute_degree_bounds(
            transition_constraints,
            interpolant_degree,
            zerofier_degree,
            2 * full_width,
        )
    }

    fn get_terminal_quotient_degree_bounds(
        &self,
        terminal_constraints: &[MPolynomial<XFieldElement>],
        interpolant_degree: Degree,
    ) -> Vec<Degree> {
        let zerofier_degree = 1;
        let full_width = self.full_width();
        Self::compute_degree_bounds(
            terminal_constraints,
            interpolant_degree,
            zerofier_degree,
            full_width,
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn extension(
        &self,
        extended_matrix: Vec<Vec<XFieldElement>>,
        interpolant_degree: Degree,
        padded_height: usize,
        initial_constraints: Vec<MPolynomial<XFieldElement>>,
        consistency_constraints: Vec<MPolynomial<XFieldElement>>,
        transition_constraints: Vec<MPolynomial<XFieldElement>>,
        terminal_constraints: Vec<MPolynomial<XFieldElement>>,
    ) -> Table<XFieldElement> {
        let bqdb =
            self.get_initial_quotient_degree_bounds(&initial_constraints, interpolant_degree);
        let cqdb = self.get_consistency_quotient_degree_bounds(
            &consistency_constraints,
            interpolant_degree,
            padded_height,
        );
        let tqdb = self.get_transition_quotient_degree_bounds(
            &transition_constraints,
            interpolant_degree,
            padded_height,
        );
        let termqdb =
            self.get_terminal_quotient_degree_bounds(&terminal_constraints, interpolant_degree);
        let new_table = self.new_from_lifted_matrix(extended_matrix);
        Table {
            initial_constraints: Some(initial_constraints),
            consistency_constraints: Some(consistency_constraints),
            transition_constraints: Some(transition_constraints),
            terminal_constraints: Some(terminal_constraints),
            initial_quotient_degree_bounds: Some(bqdb),
            consistency_quotient_degree_bounds: Some(cqdb),
            transition_quotient_degree_bounds: Some(tqdb),
            terminal_quotient_degree_bounds: Some(termqdb),
            ..new_table
        }
    }
}

fn disjoint_domain<FF: FiniteField>(domain_length: usize, disjoint_domain: &[FF]) -> Vec<FF> {
    let mut domain = Vec::with_capacity(domain_length);
    let mut elm = FF::one();
    while domain.len() != domain_length {
        if !disjoint_domain.contains(&elm) {
            domain.push(elm);
        }
        elm += FF::one();
    }
    domain
}

pub trait TableLike<FF>: InheritsFromTable<FF>
where
    FF: FiniteField
        + From<BFieldElement>
        + Mul<BFieldElement, Output = FF>
        + MulAssign<BFieldElement>,
    Standard: Distribution<FF>,
{
    // Generic functions common to all tables

    fn name(&self) -> String {
        self.inherited_table().name.clone()
    }

    /// Low-degree extends the trace that is `self` over the indicated columns `columns` and
    /// returns two codewords per column:
    /// - one codeword evaluated on the `quotient_domain`, and
    /// - one codeword evaluated on the `fri_domain`,
    /// in that order.
    fn dual_low_degree_extension(
        &self,
        quotient_domain: &ArithmeticDomain<BFieldElement>,
        fri_domain: &ArithmeticDomain<BFieldElement>,
        num_trace_randomizers: usize,
        columns: Range<usize>,
    ) -> (Table<FF>, Table<FF>) {
        // FIXME: Table<> supports Vec<[FF; WIDTH]>, but Domain does not (yet).
        let interpolated_columns = self.interpolate_columns(num_trace_randomizers, columns);
        let quotient_domain_codewords = interpolated_columns
            .par_iter()
            .map(|polynomial| quotient_domain.evaluate(polynomial))
            .collect();
        let quotient_domain_codeword_table =
            self.inherited_table().with_data(quotient_domain_codewords);
        let fri_domain_codewords = interpolated_columns
            .par_iter()
            .map(|polynomial| fri_domain.evaluate(polynomial))
            .collect();
        let fri_domain_codeword_table = self.inherited_table().with_data(fri_domain_codewords);

        (quotient_domain_codeword_table, fri_domain_codeword_table)
    }

    /// Return the interpolation of columns. The `column_indices` variable
    /// must be called with *all* the column indices for this particular table,
    /// if it is called with a subset, it *will* fail.
    fn interpolate_columns(
        &self,
        num_trace_randomizers: usize,
        columns: Range<usize>,
    ) -> Vec<Polynomial<FF>> {
        let all_trace_columns = self.data();
        let padded_height = all_trace_columns.len();
        if padded_height == 0 {
            return vec![Polynomial::zero(); columns.len()];
        }

        assert!(
            padded_height.is_power_of_two(),
            "{}: Table data must be padded before interpolation",
            self.name()
        );

        let trace_domain_generator = derive_trace_domain_generator(padded_height as u64);
        let trace_domain =
            ArithmeticDomain::new(1_u32.into(), trace_domain_generator, padded_height)
                .domain_values();

        let randomizer_domain = disjoint_domain(num_trace_randomizers, &trace_domain);
        let interpolation_domain = vec![trace_domain, randomizer_domain].concat();

        // Generator (and its order) for a subgroup of the B-field. The subgroup is as small as
        // possible given the constraints that its length is
        // - larger than or equal to `interpolation_domain`, and
        // - a power of two.
        let order_of_root_of_unity = roundup_npo2(interpolation_domain.len() as u64);
        let root_of_unity = BFieldElement::primitive_root_of_unity(order_of_root_of_unity).unwrap();

        columns
            .into_par_iter()
            .map(|col| {
                let trace = all_trace_columns.iter().map(|row| row[col]).collect();
                let randomizers = random_elements(num_trace_randomizers);
                let randomized_trace = vec![trace, randomizers].concat();
                Polynomial::fast_interpolate(
                    &interpolation_domain,
                    &randomized_trace,
                    &root_of_unity,
                    order_of_root_of_unity as usize,
                )
            })
            .collect()
    }
}

#[cfg(test)]
mod test_base_table {
    use crate::table::base_table::disjoint_domain;
    use twenty_first::shared_math::b_field_element::BFieldElement;

    #[test]
    fn disjoint_domain_test() {
        let domain = [
            BFieldElement::new(2),
            BFieldElement::new(5),
            BFieldElement::new(4),
        ];
        let ddomain = disjoint_domain(5, &domain);
        for d in ddomain {
            assert!(!domain.contains(&d));
        }
    }
}
