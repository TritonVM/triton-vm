use super::super::fri_domain::FriDomain;
use super::table_collection::TableId;
use itertools::Itertools;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use std::cmp::max;
use std::ops::Range;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::mpolynomial::{Degree, MPolynomial};
use twenty_first::shared_math::other::{is_power_of_two, roundup_npo2};
use twenty_first::shared_math::polynomial::Polynomial;
use twenty_first::shared_math::traits::{GetRandomElements, PrimeField};
use twenty_first::shared_math::x_field_element::XFieldElement;

type BWord = BFieldElement;
type XWord = XFieldElement;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct BaseTable<FieldElement: PrimeField> {
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

    /// Table id, for dynamic specializations of statically abtract types
    id: TableId,

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
impl<DataPF: PrimeField> BaseTable<DataPF> {
    pub fn new(
        base_width: usize,
        full_width: usize,
        padded_height: usize,
        num_trace_randomizers: usize,
        omicron: DataPF,
        matrix: Vec<Vec<DataPF>>,
        name: String,
        id: TableId,
    ) -> Self {
        BaseTable {
            base_width,
            full_width,
            padded_height,
            num_trace_randomizers,
            omicron,
            matrix,
            name,
            id,
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
    pub fn extension(
        base_table: BaseTable<DataPF>,
        boundary_constraints: Vec<MPolynomial<DataPF>>,
        transition_constraints: Vec<MPolynomial<DataPF>>,
        consistency_constraints: Vec<MPolynomial<DataPF>>,
        terminal_constraints: Vec<MPolynomial<DataPF>>,
    ) -> Self {
        let interpolant_degree = base_table.interpolant_degree();
        let full_width = base_table.full_width;

        let bqdb =
            Self::compute_degree_bounds(&boundary_constraints, interpolant_degree, full_width);
        let tqdb = Self::compute_degree_bounds(
            &transition_constraints,
            interpolant_degree,
            2 * full_width,
        );
        let cqdb =
            Self::compute_degree_bounds(&consistency_constraints, interpolant_degree, full_width);
        let termqdb =
            Self::compute_degree_bounds(&terminal_constraints, interpolant_degree, full_width);
        BaseTable {
            boundary_constraints: Some(boundary_constraints),
            transition_constraints: Some(transition_constraints),
            consistency_constraints: Some(consistency_constraints),
            terminal_constraints: Some(terminal_constraints),
            boundary_quotient_degree_bounds: Some(bqdb),
            transition_quotient_degree_bounds: Some(tqdb),
            consistency_quotient_degree_bounds: Some(cqdb),
            terminal_quotient_degree_bounds: Some(termqdb),
            ..base_table
        }
    }

    fn interpolant_degree(&self) -> Degree {
        (self.padded_height + self.num_trace_randomizers - 1) as Degree
    }

    /// Computes the degree bounds of the quotients given the AIR constraints and the interpolant
    /// degree. The AIR constraints are defined over a symbolic ring with `full_width`-many
    /// variables.
    fn compute_degree_bounds(
        air_constraints: &[MPolynomial<DataPF>],
        interpolant_degree: Degree,
        full_width: usize,
    ) -> Vec<Degree> {
        air_constraints
            .iter()
            .map(|mpo| mpo.symbolic_degree_bound(&vec![interpolant_degree; full_width]) - 1)
            .collect()
    }

    /// Create a `BaseTable<DataPF>` with the same parameters, but new `matrix` data.
    pub fn with_data(&self, matrix: Vec<Vec<DataPF>>) -> Self {
        BaseTable {
            matrix,
            name: format!("{} with data", self.name),
            ..self.to_owned()
        }
    }
}

/// Create a `BaseTable<XWord` from a `BaseTable<BWord>` with the same parameters lifted from the
/// B-Field into the X-Field (where applicable), but new `matrix` data.
impl BaseTable<BWord> {
    pub fn with_lifted_data(&self, matrix: Vec<Vec<XWord>>) -> BaseTable<XWord> {
        BaseTable::new(
            self.base_width,
            self.full_width,
            self.padded_height,
            self.num_trace_randomizers,
            self.omicron.lift(),
            matrix,
            format!("{} with lifted data", self.name),
            self.id,
        )
    }
}

pub trait HasBaseTable<DataPF: PrimeField> {
    fn to_base(&self) -> &BaseTable<DataPF>;
    fn to_mut_base(&mut self) -> &mut BaseTable<DataPF>;

    fn base_width(&self) -> usize {
        self.to_base().base_width
    }

    fn full_width(&self) -> usize {
        self.to_base().full_width
    }

    fn padded_height(&self) -> usize {
        self.to_base().padded_height
    }

    fn num_trace_randomizers(&self) -> usize {
        self.to_base().num_trace_randomizers
    }

    fn interpolant_degree(&self) -> Degree {
        self.to_base().interpolant_degree()
    }

    fn omicron(&self) -> DataPF {
        self.to_base().omicron
    }

    fn data(&self) -> &Vec<Vec<DataPF>> {
        &self.to_base().matrix
    }

    fn mut_data(&mut self) -> &mut Vec<Vec<DataPF>> {
        &mut self.to_mut_base().matrix
    }

    fn id(&self) -> TableId {
        self.to_base().id
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

pub fn pad_height(height: usize, num_trace_randomizers: usize) -> usize {
    if height == 0 {
        0
    } else {
        roundup_npo2(max(height as u64, num_trace_randomizers as u64)) as usize
    }
}

pub trait BaseTableTrait<DataPF>: HasBaseTable<DataPF>
where
    // Self: Sized,
    DataPF: PrimeField + GetRandomElements,
{
    // Abstract functions that individual structs implement

    fn get_padding_row(&self) -> Vec<DataPF>;

    // Generic functions common to all tables

    fn name(&self) -> String {
        self.to_base().name.clone()
    }

    fn pad(&mut self) {
        while self.data().len() != pad_height(self.data().len(), self.num_trace_randomizers()) {
            let padding_row = self.get_padding_row();
            self.mut_data().push(padding_row);
        }
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
        num_trace_randomizers: usize,
        columns: Range<usize>,
    ) -> Vec<Vec<DataPF>> {
        // FIXME: Table<> supports Vec<[DataPF; WIDTH]>, but FriDomain does not (yet).
        self.interpolate_columns(
            fri_domain.omega,
            fri_domain.length,
            num_trace_randomizers,
            columns,
        )
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
        num_trace_randomizers: usize,
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

        assert!(
            self.padded_height() >= num_trace_randomizers,
            "Number of trace randomizers must not exceed padded table height. \
            {} height: {} Num trace randomizers: {}",
            self.name(),
            self.padded_height(),
            num_trace_randomizers
        );

        // FIXME: Unfold with multiplication instead of mapping with power.
        let omicron_domain = (0..self.padded_height())
            .map(|i| self.omicron().mod_pow_u32(i as u32))
            .collect_vec();

        let randomizer_domain = (0..num_trace_randomizers)
            .map(|i| omega * omicron_domain[i])
            .collect_vec();

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
    use crate::table::base_table::pad_height;
    use twenty_first::shared_math::other;

    #[ignore]
    #[test]
    /// padding should be idempotent if number of trace randomizers is <= 1
    fn pad_height_test() {
        let num_trace_randomizers = 1;
        assert_eq!(0, pad_height(0, num_trace_randomizers));
        for x in 1..=1025 {
            let padded_x = pad_height(x, num_trace_randomizers);
            assert_eq!(other::roundup_npo2(x as u64) as usize, padded_x);
            assert_eq!(padded_x, pad_height(padded_x, num_trace_randomizers))
        }
    }
}
