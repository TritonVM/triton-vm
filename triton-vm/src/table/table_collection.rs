use itertools::Itertools;
use triton_profiler::{prof_start, prof_stop};
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::mpolynomial::Degree;
use twenty_first::shared_math::other::{is_power_of_two, roundup_npo2};
use twenty_first::shared_math::traits::FiniteField;
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::fri_domain::FriDomain;
use crate::stark::Stark;
use crate::table::base_table::{Extendable, InheritsFromTable};
use crate::table::extension_table::DegreeWithOrigin;
use triton_profiler::triton_profiler::TritonProfiler;

use super::base_matrix::BaseMatrices;
use super::base_table::TableLike;
use super::challenges::AllChallenges;
use super::extension_table::QuotientableExtensionTable;
use super::hash_table::{ExtHashTable, HashTable};
use super::instruction_table::{ExtInstructionTable, InstructionTable};
use super::jump_stack_table::{ExtJumpStackTable, JumpStackTable};
use super::op_stack_table::{ExtOpStackTable, OpStackTable};
use super::processor_table::{ExtProcessorTable, ProcessorTable};
use super::program_table::{ExtProgramTable, ProgramTable};
use super::ram_table::{ExtRamTable, RamTable};

pub const NUM_TABLES: usize = 7;

#[derive(Debug, Clone)]
pub struct BaseTableCollection {
    /// The number of `data` rows after padding
    pub padded_height: usize,

    pub program_table: ProgramTable,
    pub instruction_table: InstructionTable,
    pub processor_table: ProcessorTable,
    pub op_stack_table: OpStackTable,
    pub ram_table: RamTable,
    pub jump_stack_table: JumpStackTable,
    pub hash_table: HashTable,
}

#[derive(Debug, Clone)]
pub struct ExtTableCollection {
    /// The number of `data` rows after padding
    pub padded_height: usize,

    pub program_table: ExtProgramTable,
    pub instruction_table: ExtInstructionTable,
    pub processor_table: ExtProcessorTable,
    pub op_stack_table: ExtOpStackTable,
    pub ram_table: ExtRamTable,
    pub jump_stack_table: ExtJumpStackTable,
    pub hash_table: ExtHashTable,
}

/// A `TableId` uniquely determines one of Triton VM's tables.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TableId {
    ProgramTable,
    InstructionTable,
    ProcessorTable,
    OpStackTable,
    RamTable,
    JumpStackTable,
    HashTable,
}

/// Convert vector-of-arrays to vector-of-vectors.
fn to_vec_vecs<T: Sized + Clone, const S: usize>(vector_of_arrays: &[[T; S]]) -> Vec<Vec<T>> {
    vector_of_arrays
        .iter()
        .map(|arr| arr.to_vec())
        .collect_vec()
}

pub fn interpolant_degree(padded_height: usize, num_trace_randomizers: usize) -> Degree {
    (padded_height + num_trace_randomizers - 1) as Degree
}

pub fn derive_omicron<FF: FiniteField>(padded_height: u64) -> FF {
    debug_assert!(
        0 == padded_height || is_power_of_two(padded_height),
        "The padded height was: {}",
        padded_height
    );
    FF::primitive_root_of_unity(padded_height).unwrap()
}

impl BaseTableCollection {
    pub fn from_base_matrices(base_matrices: &BaseMatrices) -> Self {
        let padded_height = Self::padded_height(base_matrices);

        let program_table = ProgramTable::new_prover(to_vec_vecs(&base_matrices.program_matrix));
        let processor_table =
            ProcessorTable::new_prover(to_vec_vecs(&base_matrices.processor_matrix));
        let instruction_table =
            InstructionTable::new_prover(to_vec_vecs(&base_matrices.instruction_matrix));
        let op_stack_table = OpStackTable::new_prover(to_vec_vecs(&base_matrices.op_stack_matrix));
        let ram_table = RamTable::new_prover(to_vec_vecs(&base_matrices.ram_matrix));
        let jump_stack_table =
            JumpStackTable::new_prover(to_vec_vecs(&base_matrices.jump_stack_matrix));
        let hash_table = HashTable::new_prover(to_vec_vecs(&base_matrices.hash_matrix));

        BaseTableCollection {
            padded_height,
            program_table,
            instruction_table,
            processor_table,
            op_stack_table,
            ram_table,
            jump_stack_table,
            hash_table,
        }
    }

    pub fn padded_height(base_matrices: &BaseMatrices) -> usize {
        let max_height = [
            1, // minimum max height
            base_matrices.program_matrix.len(),
            base_matrices.processor_matrix.len(),
            base_matrices.instruction_matrix.len(),
            base_matrices.op_stack_matrix.len(),
            base_matrices.ram_matrix.len(),
            base_matrices.jump_stack_matrix.len(),
            base_matrices.hash_matrix.len(),
        ]
        .into_iter()
        .max()
        .unwrap();

        roundup_npo2(max_height as u64) as usize
    }

    pub fn to_fri_domain_tables(
        &self,
        fri_domain: &FriDomain<BFieldElement>,
        num_trace_randomizers: usize,
    ) -> BaseTableCollection {
        let padded_height = self.padded_height;
        let omicron = derive_omicron(padded_height as u64);

        let program_table = self.program_table.to_fri_domain_table(
            fri_domain,
            omicron,
            padded_height,
            num_trace_randomizers,
        );
        let instruction_table = self.instruction_table.to_fri_domain_table(
            fri_domain,
            omicron,
            padded_height,
            num_trace_randomizers,
        );
        let processor_table = self.processor_table.to_fri_domain_table(
            fri_domain,
            omicron,
            padded_height,
            num_trace_randomizers,
        );
        let op_stack_table = self.op_stack_table.to_fri_domain_table(
            fri_domain,
            omicron,
            padded_height,
            num_trace_randomizers,
        );
        let ram_table = self.ram_table.to_fri_domain_table(
            fri_domain,
            omicron,
            padded_height,
            num_trace_randomizers,
        );
        let jump_stack_table = self.jump_stack_table.to_fri_domain_table(
            fri_domain,
            omicron,
            padded_height,
            num_trace_randomizers,
        );
        let hash_table = self.hash_table.to_fri_domain_table(
            fri_domain,
            omicron,
            padded_height,
            num_trace_randomizers,
        );

        BaseTableCollection {
            padded_height,
            program_table,
            instruction_table,
            processor_table,
            op_stack_table,
            ram_table,
            jump_stack_table,
            hash_table,
        }
    }

    pub fn get_all_base_columns(&self) -> Vec<Vec<BFieldElement>> {
        self.into_iter()
            .map(|table| table.data().clone())
            .collect_vec()
            .concat()
    }

    pub fn get_base_degree_bounds(&self, num_trace_randomizers: usize) -> Vec<Degree> {
        let sum_of_base_widths = self.into_iter().map(|table| table.base_width()).sum();
        vec![interpolant_degree(self.padded_height, num_trace_randomizers); sum_of_base_widths]
    }

    pub fn pad(&mut self) {
        let padded_height = self.padded_height;
        self.program_table.pad(padded_height);
        self.instruction_table.pad(padded_height);
        self.processor_table.pad(padded_height);
        self.op_stack_table.pad(padded_height);
        self.ram_table.pad(padded_height);
        self.jump_stack_table.pad(padded_height);
        self.hash_table.pad(padded_height);
    }
}

impl<'a> IntoIterator for &'a BaseTableCollection {
    type Item = &'a dyn TableLike<BFieldElement>;

    type IntoIter = std::array::IntoIter<&'a dyn TableLike<BFieldElement>, NUM_TABLES>;

    fn into_iter(self) -> Self::IntoIter {
        [
            &self.program_table as &'a dyn TableLike<BFieldElement>,
            &self.instruction_table as &'a dyn TableLike<BFieldElement>,
            &self.processor_table as &'a dyn TableLike<BFieldElement>,
            &self.op_stack_table as &'a dyn TableLike<BFieldElement>,
            &self.ram_table as &'a dyn TableLike<BFieldElement>,
            &self.jump_stack_table as &'a dyn TableLike<BFieldElement>,
            &self.hash_table as &'a dyn TableLike<BFieldElement>,
        ]
        .into_iter()
    }
}

impl ExtTableCollection {
    pub fn with_padded_height(padded_height: usize) -> Self {
        ExtTableCollection {
            padded_height,
            program_table: Default::default(),
            instruction_table: Default::default(),
            processor_table: Default::default(),
            op_stack_table: Default::default(),
            ram_table: Default::default(),
            jump_stack_table: Default::default(),
            hash_table: Default::default(),
        }
    }

    pub fn for_verifier(
        num_trace_randomizers: usize,
        padded_height: usize,
        challenges: &AllChallenges,
        maybe_profiler: &mut Option<TritonProfiler>,
    ) -> Self {
        let interpolant_degree = interpolant_degree(padded_height, num_trace_randomizers);

        prof_start!(maybe_profiler, "program table");
        let ext_program_table =
            ProgramTable::for_verifier(interpolant_degree, padded_height, challenges);
        prof_stop!(maybe_profiler, "program table");
        prof_start!(maybe_profiler, "instruction table");
        let ext_instruction_table =
            InstructionTable::for_verifier(interpolant_degree, padded_height, challenges);
        prof_stop!(maybe_profiler, "instruction table");
        prof_start!(maybe_profiler, "processor table");
        let ext_processor_table =
            ProcessorTable::for_verifier(interpolant_degree, padded_height, challenges);
        prof_stop!(maybe_profiler, "processor table");
        prof_start!(maybe_profiler, "op stack table");
        let ext_op_stack_table =
            OpStackTable::for_verifier(interpolant_degree, padded_height, challenges);
        prof_stop!(maybe_profiler, "op stack table");
        prof_start!(maybe_profiler, "ram table");
        let ext_ram_table = RamTable::for_verifier(interpolant_degree, padded_height, challenges);
        prof_stop!(maybe_profiler, "ram table");
        prof_start!(maybe_profiler, "jump stack table");
        let ext_jump_stack_table =
            JumpStackTable::for_verifier(interpolant_degree, padded_height, challenges);
        prof_stop!(maybe_profiler, "jump stack table");
        prof_start!(maybe_profiler, "hash table");
        let ext_hash_table = HashTable::for_verifier(interpolant_degree, padded_height, challenges);
        prof_stop!(maybe_profiler, "hash table");

        ExtTableCollection {
            padded_height,
            program_table: ext_program_table,
            instruction_table: ext_instruction_table,
            processor_table: ext_processor_table,
            op_stack_table: ext_op_stack_table,
            ram_table: ext_ram_table,
            jump_stack_table: ext_jump_stack_table,
            hash_table: ext_hash_table,
        }
    }

    pub fn max_degree_with_origin(&self, num_trace_randomizers: usize) -> DegreeWithOrigin {
        self.into_iter()
            .map(|ext_table| {
                ext_table.all_degrees_with_origin(self.padded_height, num_trace_randomizers)
            })
            .concat()
            .into_iter()
            .max()
            .unwrap_or_default()
    }

    /// Create an ExtTableCollection from a BaseTableCollection by `.extend()`ing each base table.
    /// The `.extend()` for each table is specific to that table, but always
    /// involves adding some number of columns. Each table only needs their
    /// own challenges, but `AllChallenges` are passed everywhere to keep each table's `.extend()`
    /// homogenous.
    pub fn extend_tables(
        base_tables: &BaseTableCollection,
        all_challenges: &AllChallenges,
        num_trace_randomizers: usize,
    ) -> Self {
        let padded_height = base_tables.padded_height;
        let interpolant_degree = interpolant_degree(padded_height, num_trace_randomizers);

        let program_table = base_tables
            .program_table
            .extend(&all_challenges.program_table_challenges, interpolant_degree);

        let instruction_table = base_tables.instruction_table.extend(
            &all_challenges.instruction_table_challenges,
            interpolant_degree,
        );

        let processor_table = base_tables.processor_table.extend(
            &all_challenges.processor_table_challenges,
            interpolant_degree,
        );

        let op_stack_table = base_tables.op_stack_table.extend(
            &all_challenges.op_stack_table_challenges,
            interpolant_degree,
        );

        let ram_table = base_tables
            .ram_table
            .extend(&all_challenges.ram_table_challenges, interpolant_degree);

        let jump_stack_table = base_tables.jump_stack_table.extend(
            &all_challenges.jump_stack_table_challenges,
            interpolant_degree,
        );

        let hash_table = base_tables
            .hash_table
            .extend(&all_challenges.hash_table_challenges, interpolant_degree);

        ExtTableCollection {
            padded_height,
            program_table,
            instruction_table,
            processor_table,
            op_stack_table,
            ram_table,
            jump_stack_table,
            hash_table,
        }
    }

    /// Heads up: only extension columns are being low degree extended. todo: better naming.
    pub fn to_fri_domain_tables(
        &self,
        fri_domain: &FriDomain<XFieldElement>,
        num_trace_randomizers: usize,
    ) -> Self {
        let padded_height = self.padded_height;
        let omicron = derive_omicron(padded_height as u64);

        let program_table = self.program_table.to_fri_domain_table(
            fri_domain,
            omicron,
            padded_height,
            num_trace_randomizers,
        );
        let instruction_table = self.instruction_table.to_fri_domain_table(
            fri_domain,
            omicron,
            padded_height,
            num_trace_randomizers,
        );
        let processor_table = self.processor_table.to_fri_domain_table(
            fri_domain,
            omicron,
            padded_height,
            num_trace_randomizers,
        );
        let op_stack_table = self.op_stack_table.to_fri_domain_table(
            fri_domain,
            omicron,
            padded_height,
            num_trace_randomizers,
        );
        let ram_table = self.ram_table.to_fri_domain_table(
            fri_domain,
            omicron,
            padded_height,
            num_trace_randomizers,
        );
        let jump_stack_table = self.jump_stack_table.to_fri_domain_table(
            fri_domain,
            omicron,
            padded_height,
            num_trace_randomizers,
        );
        let hash_table = self.hash_table.to_fri_domain_table(
            fri_domain,
            omicron,
            padded_height,
            num_trace_randomizers,
        );

        ExtTableCollection {
            padded_height,
            program_table,
            instruction_table,
            processor_table,
            op_stack_table,
            ram_table,
            jump_stack_table,
            hash_table,
        }
    }

    pub fn collect_all_columns(&self) -> Vec<Vec<XFieldElement>> {
        let mut all_ext_cols = vec![];

        for table in self.into_iter() {
            for col in table.data().iter() {
                all_ext_cols.push(col.clone());
            }
        }
        all_ext_cols
    }

    pub fn data(&self, table_id: TableId) -> &Vec<Vec<XFieldElement>> {
        use TableId::*;

        match table_id {
            ProgramTable => self.program_table.data(),
            InstructionTable => self.instruction_table.data(),
            ProcessorTable => self.processor_table.data(),
            OpStackTable => self.op_stack_table.data(),
            RamTable => self.ram_table.data(),
            JumpStackTable => self.jump_stack_table.data(),
            HashTable => self.hash_table.data(),
        }
    }

    pub fn get_all_base_degree_bounds(&self, num_trace_randomizers: usize) -> Vec<Degree> {
        let sum_base_widths = self.into_iter().map(|table| table.base_width()).sum();
        vec![interpolant_degree(self.padded_height, num_trace_randomizers); sum_base_widths]
    }

    pub fn get_extension_degree_bounds(&self, num_trace_randomizers: usize) -> Vec<Degree> {
        let sum_base_widths: usize = self.into_iter().map(|table| table.base_width()).sum();
        let sum_full_widths: usize = self.into_iter().map(|table| table.full_width()).sum();
        let num_extension_columns = sum_full_widths - sum_base_widths;
        vec![interpolant_degree(self.padded_height, num_trace_randomizers); num_extension_columns]
    }

    pub fn get_all_quotients(
        &self,
        fri_domain: &FriDomain<XFieldElement>,
        challenges: &AllChallenges,
        maybe_profiler: &mut Option<TritonProfiler>,
    ) -> Vec<Vec<XFieldElement>> {
        let padded_height = self.padded_height;
        let omicron = derive_omicron(padded_height as u64);

        self.into_iter()
            .map(|ext_codeword_table| {
                if let Some(profiler) = maybe_profiler.as_mut() {
                    profiler.start(&ext_codeword_table.name());
                }
                // TODO: Consider if we can use `transposed_ext_codewords` from caller, Stark::prove().
                // This would require more complicated indexing, but it would save a lot of allocation.
                let transposed_codewords = Stark::transpose(ext_codeword_table.data());
                let res = ext_codeword_table.all_quotients(
                    fri_domain,
                    transposed_codewords,
                    challenges,
                    omicron,
                    padded_height,
                    maybe_profiler,
                );
                if let Some(profiler) = maybe_profiler.as_mut() {
                    profiler.stop(&ext_codeword_table.name());
                }
                res
            })
            .concat()
    }

    pub fn get_all_quotient_degree_bounds(&self, num_trace_randomizers: usize) -> Vec<Degree> {
        self.into_iter() // Can we parallelize this? -> implement into_par_iter for TableCollection
            .map(|ext_table| {
                ext_table.get_all_quotient_degree_bounds(self.padded_height, num_trace_randomizers)
            })
            .concat()
    }

    pub fn join(
        base_codeword_tables: BaseTableCollection,
        ext_codeword_tables: ExtTableCollection,
    ) -> ExtTableCollection {
        let padded_height = base_codeword_tables.padded_height;

        let program_base_matrix = base_codeword_tables.program_table.data();
        let lifted_program_base_matrix = program_base_matrix
            .iter()
            .map(|cdwd| cdwd.iter().map(|bfe| bfe.lift()).collect_vec())
            .collect_vec();
        let program_ext_matrix = ext_codeword_tables.program_table.data();
        let full_program_matrix =
            vec![lifted_program_base_matrix, program_ext_matrix.to_vec()].concat();
        let joined_program_table = ext_codeword_tables
            .program_table
            .inherited_table()
            .with_data(full_program_matrix);
        let program_table = ExtProgramTable {
            inherited_table: joined_program_table,
        };

        let instruction_base_matrix = base_codeword_tables.instruction_table.data();
        let lifted_instruction_base_matrix = instruction_base_matrix
            .iter()
            .map(|cdwd| cdwd.iter().map(|bfe| bfe.lift()).collect_vec())
            .collect_vec();
        let instruction_ext_matrix = ext_codeword_tables.instruction_table.data();
        let full_instruction_matrix = vec![
            lifted_instruction_base_matrix,
            instruction_ext_matrix.to_vec(),
        ]
        .concat();
        let joined_instruction_table = ext_codeword_tables
            .instruction_table
            .inherited_table()
            .with_data(full_instruction_matrix);
        let instruction_table = ExtInstructionTable {
            inherited_table: joined_instruction_table,
        };

        let processor_base_matrix = base_codeword_tables.processor_table.data();
        let lifted_processor_base_matrix = processor_base_matrix
            .iter()
            .map(|cdwd| cdwd.iter().map(|bfe| bfe.lift()).collect_vec())
            .collect_vec();
        let processor_ext_matrix = ext_codeword_tables.processor_table.data();
        let full_processor_matrix =
            vec![lifted_processor_base_matrix, processor_ext_matrix.to_vec()].concat();
        let joined_processor_table = ext_codeword_tables
            .processor_table
            .inherited_table()
            .with_data(full_processor_matrix);
        let processor_table = ExtProcessorTable {
            inherited_table: joined_processor_table,
        };

        let op_stack_base_matrix = base_codeword_tables.op_stack_table.data();
        let lifted_op_stack_base_matrix = op_stack_base_matrix
            .iter()
            .map(|cdwd| cdwd.iter().map(|bfe| bfe.lift()).collect_vec())
            .collect_vec();
        let op_stack_ext_matrix = ext_codeword_tables.op_stack_table.data();
        let full_op_stack_matrix =
            vec![lifted_op_stack_base_matrix, op_stack_ext_matrix.to_vec()].concat();
        let joined_op_stack_table = ext_codeword_tables
            .op_stack_table
            .inherited_table()
            .with_data(full_op_stack_matrix);
        let op_stack_table = ExtOpStackTable {
            inherited_table: joined_op_stack_table,
        };

        let ram_base_matrix = base_codeword_tables.ram_table.data();
        let lifted_ram_base_matrix = ram_base_matrix
            .iter()
            .map(|cdwd| cdwd.iter().map(|bfe| bfe.lift()).collect_vec())
            .collect_vec();
        let ram_ext_matrix = ext_codeword_tables.ram_table.data();
        let full_ram_matrix = vec![lifted_ram_base_matrix, ram_ext_matrix.to_vec()].concat();
        let joined_ram_table = ext_codeword_tables
            .ram_table
            .inherited_table()
            .with_data(full_ram_matrix);
        let ram_table = ExtRamTable {
            inherited_table: joined_ram_table,
        };

        let jump_stack_base_matrix = base_codeword_tables.jump_stack_table.data();
        let lifted_jump_stack_base_matrix = jump_stack_base_matrix
            .iter()
            .map(|cdwd| cdwd.iter().map(|bfe| bfe.lift()).collect_vec())
            .collect_vec();
        let jump_stack_ext_matrix = ext_codeword_tables.jump_stack_table.data();
        let full_jump_stack_matrix = vec![
            lifted_jump_stack_base_matrix,
            jump_stack_ext_matrix.to_vec(),
        ]
        .concat();
        let joined_jump_stack_table = ext_codeword_tables
            .jump_stack_table
            .inherited_table()
            .with_data(full_jump_stack_matrix);
        let jump_stack_table = ExtJumpStackTable {
            inherited_table: joined_jump_stack_table,
        };

        let hash_base_matrix = base_codeword_tables.hash_table.data();
        let lifted_hash_base_matrix = hash_base_matrix
            .iter()
            .map(|cdwd| cdwd.iter().map(|bfe| bfe.lift()).collect_vec())
            .collect_vec();
        let hash_ext_matrix = ext_codeword_tables.hash_table.data();
        let full_hash_matrix = vec![lifted_hash_base_matrix, hash_ext_matrix.to_vec()].concat();
        let joined_hash_table = ext_codeword_tables
            .hash_table
            .inherited_table()
            .with_data(full_hash_matrix);
        let hash_table = ExtHashTable {
            inherited_table: joined_hash_table,
        };

        ExtTableCollection {
            padded_height,
            program_table,
            instruction_table,
            processor_table,
            op_stack_table,
            ram_table,
            jump_stack_table,
            hash_table,
        }
    }
}

impl<'a> IntoIterator for &'a ExtTableCollection {
    type Item = &'a dyn QuotientableExtensionTable;

    type IntoIter = std::array::IntoIter<&'a dyn QuotientableExtensionTable, NUM_TABLES>;

    fn into_iter(self) -> Self::IntoIter {
        [
            &self.program_table as &'a dyn QuotientableExtensionTable,
            &self.instruction_table as &'a dyn QuotientableExtensionTable,
            &self.processor_table as &'a dyn QuotientableExtensionTable,
            &self.op_stack_table as &'a dyn QuotientableExtensionTable,
            &self.ram_table as &'a dyn QuotientableExtensionTable,
            &self.jump_stack_table as &'a dyn QuotientableExtensionTable,
            &self.hash_table as &'a dyn QuotientableExtensionTable,
        ]
        .into_iter()
    }
}

#[cfg(test)]
mod table_collection_tests {
    use crate::table::{
        hash_table, instruction_table, jump_stack_table, op_stack_table, processor_table,
        program_table, ram_table,
    };

    use super::*;

    fn dummy_ext_table_collection() -> ExtTableCollection {
        let max_padded_height = 1;
        ExtTableCollection::with_padded_height(max_padded_height)
    }

    #[test]
    fn base_table_width_is_correct() {
        let base_matrices = BaseMatrices::default();
        let base_tables = BaseTableCollection::from_base_matrices(&base_matrices);

        assert_eq!(
            program_table::BASE_WIDTH,
            base_tables.program_table.base_width()
        );
        assert_eq!(
            instruction_table::BASE_WIDTH,
            base_tables.instruction_table.base_width()
        );
        assert_eq!(
            processor_table::BASE_WIDTH,
            base_tables.processor_table.base_width()
        );
        assert_eq!(
            op_stack_table::BASE_WIDTH,
            base_tables.op_stack_table.base_width()
        );
        assert_eq!(ram_table::BASE_WIDTH, base_tables.ram_table.base_width());
        assert_eq!(
            jump_stack_table::BASE_WIDTH,
            base_tables.jump_stack_table.base_width()
        );
        assert_eq!(hash_table::BASE_WIDTH, base_tables.hash_table.base_width());
    }

    #[test]
    fn ext_table_width_is_correct() {
        let ext_tables = dummy_ext_table_collection();

        assert_eq!(
            program_table::FULL_WIDTH,
            ext_tables.program_table.full_width()
        );
        assert_eq!(
            instruction_table::FULL_WIDTH,
            ext_tables.instruction_table.full_width()
        );
        assert_eq!(
            processor_table::FULL_WIDTH,
            ext_tables.processor_table.full_width()
        );
        assert_eq!(
            op_stack_table::FULL_WIDTH,
            ext_tables.op_stack_table.full_width()
        );
        assert_eq!(ram_table::FULL_WIDTH, ext_tables.ram_table.full_width());
        assert_eq!(
            jump_stack_table::FULL_WIDTH,
            ext_tables.jump_stack_table.full_width()
        );
        assert_eq!(hash_table::FULL_WIDTH, ext_tables.hash_table.full_width());
    }

    /// intended use: `cargo t print_all_table_widths -- --nocapture`
    #[test]
    fn print_all_table_widths() {
        println!("| table name         | #base cols | #ext cols | full width |");
        println!("|:-------------------|-----------:|----------:|-----------:|");
        for table in dummy_ext_table_collection().into_iter() {
            println!(
                "| {:<18} | {:>10} | {:>9} | {:>10} |",
                table.name().split_off(8),
                table.base_width(),
                table.full_width() - table.base_width(),
                table.full_width(),
            );
        }
        let sum_base_columns: usize = dummy_ext_table_collection()
            .into_iter()
            .map(|table| table.base_width())
            .sum();
        let sum_full_widths: usize = dummy_ext_table_collection()
            .into_iter()
            .map(|table| table.full_width())
            .sum();
        println!("|                    |            |           |            |");
        println!(
            "| Sum                | {:>10} | {:>9} | {:>10} |",
            sum_base_columns,
            sum_full_widths - sum_base_columns,
            sum_full_widths
        );
    }
}
