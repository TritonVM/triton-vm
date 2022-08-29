use super::super::fri_domain::FriDomain;
use itertools::Itertools;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::ops::Range;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::mpolynomial::{Degree, MPolynomial};
use twenty_first::shared_math::other::{is_power_of_two, roundup_npo2};
use twenty_first::shared_math::polynomial::Polynomial;
use twenty_first::shared_math::traits::{GetRandomElements, PrimeField};
use twenty_first::shared_math::x_field_element::XFieldElement;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Table<FieldElement: PrimeField> {
    /// The width of each `data` row in the base version of the table
    base_width: usize,

    /// The width of each `data` row in the extended version of the table
    full_width: usize,

    /// The number of `data` rows after padding
    padded_height: usize,

    /// The number of random rows added for each row in the execution trace.
    num_trace_randomizers: usize,

    /// The generator of the ο-domain, which is used to interpolate the trace data.
    omicron: FieldElement,

    /// The table data (trace data). Represents every intermediate
    matrix: Vec<Vec<FieldElement>>,

    /// The name of the table. Mostly for debugging purpose.
    pub(crate) name: String,

    /// AIR constraints, to be populated upon extension
    pub(crate) boundary_constraints: Option<Vec<MPolynomial<FieldElement>>>,
    pub(crate) transition_constraints: Option<Vec<MPolynomial<FieldElement>>>,
    pub(crate) consistency_constraints: Option<Vec<MPolynomial<FieldElement>>>,
    pub(crate) terminal_constraints: Option<Vec<MPolynomial<FieldElement>>>,

    /// quotient degrees, to be populated upon extension
    pub(crate) boundary_quotient_degree_bounds: Option<Vec<i64>>,
    pub(crate) transition_quotient_degree_bounds: Option<Vec<i64>>,
    pub(crate) consistency_quotient_degree_bounds: Option<Vec<i64>>,
    pub(crate) terminal_quotient_degree_bounds: Option<Vec<i64>>,
}

#[allow(clippy::too_many_arguments)]
impl<DataPF: PrimeField> Table<DataPF> {
    pub fn new(
        base_width: usize,
        full_width: usize,
        padded_height: usize,
        num_trace_randomizers: usize,
        omicron: DataPF,
        matrix: Vec<Vec<DataPF>>,
        name: String,
    ) -> Self {
        Table {
            base_width,
            full_width,
            padded_height,
            num_trace_randomizers,
            omicron,
            matrix,
            name,
            boundary_constraints: None,
            transition_constraints: None,
            consistency_constraints: None,
            terminal_constraints: None,
            boundary_quotient_degree_bounds: None,
            transition_quotient_degree_bounds: None,
            consistency_quotient_degree_bounds: None,
            terminal_quotient_degree_bounds: None,
        }
    }

    fn interpolant_degree(&self) -> Degree {
        (self.padded_height + self.num_trace_randomizers - 1) as Degree
    }

    /// Create a `BaseTable<DataPF>` with the same parameters, but new `matrix` data.
    pub fn with_data(&self, matrix: Vec<Vec<DataPF>>) -> Self {
        Table {
            matrix,
            name: format!("{} with data", self.name),
            ..self.to_owned()
        }
    }
}

pub trait InheritsFromTable<DataPF: PrimeField> {
    fn inherited_table(&self) -> &Table<DataPF>;
    fn mut_inherited_table(&mut self) -> &mut Table<DataPF>;

    fn base_width(&self) -> usize {
        self.inherited_table().base_width
    }

    fn full_width(&self) -> usize {
        self.inherited_table().full_width
    }

    fn padded_height(&self) -> usize {
        self.inherited_table().padded_height
    }

    fn num_trace_randomizers(&self) -> usize {
        self.inherited_table().num_trace_randomizers
    }

    fn interpolant_degree(&self) -> Degree {
        self.inherited_table().interpolant_degree()
    }

    fn omicron(&self) -> DataPF {
        self.inherited_table().omicron
    }

    fn data(&self) -> &Vec<Vec<DataPF>> {
        &self.inherited_table().matrix
    }

    fn mut_data(&mut self) -> &mut Vec<Vec<DataPF>> {
        &mut self.mut_inherited_table().matrix
    }
}

pub trait Extendable: TableLike<BFieldElement> {
    // Abstract functions that individual structs implement

    fn get_padding_row(&self) -> Vec<BFieldElement>;

    // Generic functions common to all extendable tables

    fn new_from_lifted_matrix(&self, matrix: Vec<Vec<XFieldElement>>) -> Table<XFieldElement> {
        Table::new(
            self.base_width(),
            self.full_width(),
            self.padded_height(),
            self.num_trace_randomizers(),
            self.omicron().lift(),
            matrix,
            format!("{} with lifted matrix", self.name()),
        )
    }

    fn pad(&mut self) {
        while self.data().len() != pad_height(self.data().len()) {
            let padding_row = self.get_padding_row();
            self.mut_data().push(padding_row);
        }
    }

    /// Computes the degree bounds of the quotients given the AIR constraints and the interpolant
    /// degree. The AIR constraints are defined over a symbolic ring with `full_width`-many
    /// variables.
    fn compute_degree_bounds(
        air_constraints: &[MPolynomial<XFieldElement>],
        interpolant_degree: Degree,
        full_width: usize,
    ) -> Vec<Degree> {
        air_constraints
            .iter()
            .map(|mpo| mpo.symbolic_degree_bound(&vec![interpolant_degree; full_width]) - 1)
            .collect()
    }

    fn get_boundary_quotient_degree_bounds(
        &self,
        boundary_constraints: &[MPolynomial<XFieldElement>],
    ) -> Vec<Degree> {
        let interpolant_degree = self.interpolant_degree();
        let full_width = self.full_width();
        Self::compute_degree_bounds(boundary_constraints, interpolant_degree, full_width)
    }

    fn get_transition_quotient_degree_bounds(
        &self,
        transition_constraints: &[MPolynomial<XFieldElement>],
    ) -> Vec<Degree> {
        let interpolant_degree = self.interpolant_degree();
        let full_width = self.full_width();
        Self::compute_degree_bounds(transition_constraints, interpolant_degree, 2 * full_width)
    }

    fn get_consistency_quotient_degree_bounds(
        &self,
        consistency_constraints: &[MPolynomial<XFieldElement>],
    ) -> Vec<Degree> {
        let interpolant_degree = self.interpolant_degree();
        let full_width = self.full_width();
        Self::compute_degree_bounds(consistency_constraints, interpolant_degree, full_width)
    }

    fn get_terminal_quotient_degree_bounds(
        &self,
        terminal_constraints: &[MPolynomial<XFieldElement>],
    ) -> Vec<Degree> {
        let interpolant_degree = self.interpolant_degree();
        let full_width = self.full_width();
        Self::compute_degree_bounds(terminal_constraints, interpolant_degree, full_width)
    }

    fn extension(
        &self,
        extended_matrix: Vec<Vec<XFieldElement>>,
        boundary_constraints: Vec<MPolynomial<XFieldElement>>,
        transition_constraints: Vec<MPolynomial<XFieldElement>>,
        consistency_constraints: Vec<MPolynomial<XFieldElement>>,
        terminal_constraints: Vec<MPolynomial<XFieldElement>>,
    ) -> Table<XFieldElement> {
        let bqdb = self.get_boundary_quotient_degree_bounds(&boundary_constraints);
        let tqdb = self.get_transition_quotient_degree_bounds(&transition_constraints);
        let cqdb = self.get_consistency_quotient_degree_bounds(&consistency_constraints);
        let termqdb = self.get_terminal_quotient_degree_bounds(&terminal_constraints);
        let new_table = self.new_from_lifted_matrix(extended_matrix);
        Table {
            boundary_constraints: Some(boundary_constraints),
            transition_constraints: Some(transition_constraints),
            consistency_constraints: Some(consistency_constraints),
            terminal_constraints: Some(terminal_constraints),
            boundary_quotient_degree_bounds: Some(bqdb),
            transition_quotient_degree_bounds: Some(tqdb),
            consistency_quotient_degree_bounds: Some(cqdb),
            terminal_quotient_degree_bounds: Some(termqdb),
            ..new_table
        }
    }
}

pub fn derive_omicron<DataPF: PrimeField>(padded_height: u64) -> DataPF {
    debug_assert!(
        0 == padded_height || is_power_of_two(padded_height),
        "The padded height was: {}",
        padded_height
    );
    DataPF::default()
        .get_primitive_root_of_unity(padded_height)
        .0
        .unwrap()
}

pub fn pad_height(height: usize) -> usize {
    if height == 0 {
        0
    } else {
        roundup_npo2(height as u64) as usize
    }
}

fn disjoint_domain<DataPF: PrimeField>(
    domain_length: usize,
    disjoint_domain: &[DataPF],
    ring_one: DataPF,
) -> Vec<DataPF> {
    // Why do we still have this? 😩
    let zero = ring_one.ring_zero();
    (0..2_usize.pow(32))
        .map(|d| zero.new_from_usize(d))
        .filter(|d| !disjoint_domain.contains(d))
        .take(domain_length)
        .collect_vec()
}

pub trait TableLike<DataPF>: InheritsFromTable<DataPF>
where
    // Self: Sized,
    DataPF: PrimeField + GetRandomElements,
{
    // Generic functions common to all tables

    fn name(&self) -> String {
        self.inherited_table().name.clone()
    }

    /// Returns the relation between the FRI domain and the omicron domain
    fn unit_distance(&self, omega_order: usize) -> usize {
        if self.padded_height() == 0 {
            0
        } else {
            omega_order / self.padded_height()
        }
    }

    fn low_degree_extension(
        &self,
        fri_domain: &FriDomain<DataPF>,
        columns: Range<usize>,
    ) -> Vec<Vec<DataPF>> {
        // FIXME: Table<> supports Vec<[DataPF; WIDTH]>, but FriDomain does not (yet).
        self.interpolate_columns(fri_domain.omega, fri_domain.length, columns)
            .par_iter()
            .map(|polynomial| fri_domain.evaluate(polynomial))
            .collect()
    }

    /// Return the interpolation of columns. The `column_indices` variable
    /// must be called with *all* the column indices for this particular table,
    /// if it is called with a subset, it *will* fail.
    fn interpolate_columns(
        &self,
        omega: DataPF,
        omega_order: usize,
        columns: Range<usize>,
    ) -> Vec<Polynomial<DataPF>> {
        // FIXME: Inject `rng` instead.
        let mut rng = rand::thread_rng();

        // Ensure that `matrix` is set and padded before running this function
        assert_eq!(
            self.padded_height(),
            self.data().len(),
            "{}: Table data must be padded before interpolation",
            self.name()
        );

        if self.padded_height() == 0 {
            return vec![Polynomial::ring_zero(); columns.len()];
        }

        // FIXME: Unfold with multiplication instead of mapping with power.
        let omicron_domain = (0..self.padded_height())
            .map(|i| self.omicron().mod_pow_u32(i as u32))
            .collect_vec();

        let one = omega.ring_one();
        let num_trace_randomizers = self.num_trace_randomizers();
        let randomizer_domain = disjoint_domain(num_trace_randomizers, &omicron_domain, one);

        let interpolation_domain = vec![omicron_domain, randomizer_domain].concat();
        let mut all_randomized_traces = vec![];
        let data = self.data();

        for col in columns {
            let trace = data.iter().map(|row| row[col]).collect();
            let randomizers = DataPF::random_elements(num_trace_randomizers, &mut rng);
            let randomized_trace = vec![trace, randomizers].concat();
            assert_eq!(
                randomized_trace.len(),
                interpolation_domain.len(),
                "Length of x values and y values must match"
            );
            all_randomized_traces.push(randomized_trace);
        }

        all_randomized_traces
            .par_iter()
            .map(|values| {
                Polynomial::fast_interpolate(&interpolation_domain, values, &omega, omega_order)
            })
            .collect()
    }
}

#[cfg(test)]
mod test_base_table {
    use crate::table::base_table::{disjoint_domain, pad_height};
    use twenty_first::shared_math::b_field_element::BFieldElement;
    use twenty_first::shared_math::other;

    #[test]
    fn pad_height_test() {
        assert_eq!(0, pad_height(0));
        for x in 1..=1025 {
            let padded_x = pad_height(x);
            assert_eq!(other::roundup_npo2(x as u64) as usize, padded_x);
            assert_eq!(padded_x, pad_height(padded_x))
        }
    }

    #[test]
    fn disjoint_domain_test() {
        let one = BFieldElement::ring_one();
        let domain = [2.into(), 5.into(), 4.into()];
        let ddomain = disjoint_domain(5, &domain, one);
        assert_eq!(vec![0.into(), one, 3.into(), 6.into(), 7.into()], ddomain);
    }
}
