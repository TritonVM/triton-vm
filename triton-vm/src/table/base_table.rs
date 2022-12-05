use std::ops::Mul;
use std::ops::MulAssign;

use rand_distr::Distribution;
use rand_distr::Standard;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::traits::FiniteField;
use twenty_first::shared_math::x_field_element::XFieldElement;

// todo: replace the complicated trait structure with
//  “TraceTable<FF>” and “Extendable: TraceTable<BFieldElement>”
//  and define struct (eg) ProgramTable with field trace_table: ArrayView2

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
}
