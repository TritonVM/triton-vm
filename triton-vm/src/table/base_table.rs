use super::super::arithmetic_domain::ArithmeticDomain;
use itertools::Itertools;
use rand_distr::{Distribution, Standard};
use rayon::iter::{IntoParallelIterator, ParallelIterator};
use std::ops::{Mul, MulAssign, Range};
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::other::{random_elements, roundup_npo2};
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
}

#[allow(clippy::too_many_arguments)]
impl<FF: FiniteField> Table<FF> {
    pub fn new(base_width: usize, full_width: usize, matrix: Vec<Vec<FF>>, name: String) -> Self {
        Table {
            base_width,
            full_width,
            matrix,
            name,
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

    fn low_degree_extension(
        &self,
        fri_domain: &ArithmeticDomain<BFieldElement>,
        num_trace_randomizers: usize,
        columns: Range<usize>,
    ) -> Table<FF> {
        let all_trace_columns = self.data();
        let padded_height = all_trace_columns.len();

        assert_ne!(
            0,
            padded_height,
            "{}: Low-degree extension must be performed on some codeword, but got nothing.",
            self.name()
        );
        assert!(
            padded_height.is_power_of_two(),
            "{}: Table data must be padded before interpolation",
            self.name()
        );

        let randomized_trace_domain_len =
            roundup_npo2((padded_height + num_trace_randomizers) as u64);
        let randomized_trace_domain_gen =
            BFieldElement::primitive_root_of_unity(randomized_trace_domain_len).unwrap();
        let randomized_trace_domain = ArithmeticDomain::new(
            1_u32.into(),
            randomized_trace_domain_gen,
            randomized_trace_domain_len as usize,
        );

        // how many elements to skip in the randomized trace domain to only refer to elements
        // in the non-randomized trace domain
        let unit_distance = randomized_trace_domain_len as usize / padded_height;

        let fri_domain_codewords = columns
            .into_par_iter()
            .map(|idx| {
                let trace = all_trace_columns.iter().map(|row| row[idx]).collect_vec();
                let mut randomized_trace = random_elements(randomized_trace_domain_len as usize);
                for i in 0..padded_height {
                    randomized_trace[unit_distance * i] = trace[i]
                }
                randomized_trace_domain.low_degree_extension(&randomized_trace, fri_domain)
            })
            .collect();
        self.inherited_table().with_data(fri_domain_codewords)
    }
}
