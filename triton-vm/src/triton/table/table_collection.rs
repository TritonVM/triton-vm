use super::base_matrix::BaseMatrices;
use super::base_table::Table;
use super::challenges_endpoints::{AllChallenges, AllEndpoints};
use super::extension_table::ExtensionTable;
use super::hash_table::{ExtHashTable, HashTable};
use super::instruction_table::{ExtInstructionTable, InstructionTable};
use super::jump_stack_table::{ExtJumpStackTable, JumpStackTable};
use super::op_stack_table::{ExtOpStackTable, OpStackTable};
use super::processor_table::{ExtProcessorTable, ProcessorTable};
use super::program_table::{ExtProgramTable, ProgramTable};
use super::ram_table::{ExtRamTable, RamTable};
use super::u32_op_table::{ExtU32OpTable, U32OpTable};
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::mpolynomial::Degree;
use crate::shared_math::stark::triton::fri_domain::FriDomain;
use crate::shared_math::stark::triton::table::base_table::HasBaseTable;
use crate::shared_math::x_field_element::XFieldElement;
use crate::timing_reporter::TimingReporter;
use itertools::Itertools;

type BWord = BFieldElement;
type XWord = XFieldElement;

pub const NUM_TABLES: usize = 8;

#[derive(Debug, Clone)]
pub struct BaseTableCollection {
    pub program_table: ProgramTable,
    pub instruction_table: InstructionTable,
    pub processor_table: ProcessorTable,
    pub op_stack_table: OpStackTable,
    pub ram_table: RamTable,
    pub jump_stack_table: JumpStackTable,
    pub hash_table: HashTable,
    pub u32_op_table: U32OpTable,
}

#[derive(Debug, Clone)]
pub struct ExtTableCollection {
    pub program_table: ExtProgramTable,
    pub instruction_table: ExtInstructionTable,
    pub processor_table: ExtProcessorTable,
    pub op_stack_table: ExtOpStackTable,
    pub ram_table: ExtRamTable,
    pub jump_stack_table: ExtJumpStackTable,
    pub hash_table: ExtHashTable,
    pub u32_op_table: ExtU32OpTable,
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
    U32OpTable,
}

/// Convert vector-of-arrays to vector-of-vectors.
fn to_vec_vecs<T: Sized + Clone, const S: usize>(vector_of_arrays: &[[T; S]]) -> Vec<Vec<T>> {
    vector_of_arrays
        .iter()
        .map(|arr| arr.to_vec())
        .collect_vec()
}

impl BaseTableCollection {
    pub fn from_base_matrices(num_trace_randomizers: usize, base_matrices: &BaseMatrices) -> Self {
        let program_table = ProgramTable::new_prover(
            num_trace_randomizers,
            to_vec_vecs(&base_matrices.program_matrix),
        );

        let processor_table = ProcessorTable::new_prover(
            num_trace_randomizers,
            to_vec_vecs(&base_matrices.processor_matrix),
        );

        let instruction_table = InstructionTable::new_prover(
            num_trace_randomizers,
            to_vec_vecs(&base_matrices.instruction_matrix),
        );

        let op_stack_table = OpStackTable::new_prover(
            num_trace_randomizers,
            to_vec_vecs(&base_matrices.op_stack_matrix),
        );

        let ram_table = RamTable::new_prover(
            num_trace_randomizers,
            to_vec_vecs(&base_matrices.ram_matrix),
        );

        let jump_stack_table = JumpStackTable::new_prover(
            num_trace_randomizers,
            to_vec_vecs(&base_matrices.jump_stack_matrix),
        );

        let hash_table = HashTable::new_prover(
            num_trace_randomizers,
            to_vec_vecs(&base_matrices.hash_matrix),
        );

        let u32_op_table = U32OpTable::new_prover(
            num_trace_randomizers,
            to_vec_vecs(&base_matrices.u32_op_matrix),
        );

        BaseTableCollection {
            program_table,
            instruction_table,
            processor_table,
            op_stack_table,
            ram_table,
            jump_stack_table,
            hash_table,
            u32_op_table,
        }
    }

    pub fn codeword_tables(&self, fri_domain: &FriDomain<BWord>) -> BaseTableCollection {
        BaseTableCollection {
            program_table: self.program_table.codeword_table(fri_domain),
            instruction_table: self.instruction_table.codeword_table(fri_domain),
            processor_table: self.processor_table.codeword_table(fri_domain),
            op_stack_table: self.op_stack_table.codeword_table(fri_domain),
            ram_table: self.ram_table.codeword_table(fri_domain),
            jump_stack_table: self.jump_stack_table.codeword_table(fri_domain),
            hash_table: self.hash_table.codeword_table(fri_domain),
            u32_op_table: self.u32_op_table.codeword_table(fri_domain),
        }
    }

    pub fn get_all_base_columns(&self) -> Vec<Vec<BWord>> {
        self.into_iter()
            .map(|table| table.data().clone())
            .collect_vec()
            .concat()
    }

    pub fn get_base_column_degree_bounds(&self) -> Vec<Degree> {
        self.into_iter()
            .map(|table| vec![table.interpolant_degree(); table.base_width()])
            .concat()
    }

    pub fn pad(&mut self) {
        self.program_table.pad();
        self.instruction_table.pad();
        self.processor_table.pad();
        self.op_stack_table.pad();
        self.ram_table.pad();
        self.jump_stack_table.pad();
        self.hash_table.pad();
        self.u32_op_table.pad();
    }
}

impl<'a> IntoIterator for &'a BaseTableCollection {
    type Item = &'a dyn Table<BWord>;

    type IntoIter = std::array::IntoIter<&'a dyn Table<BWord>, NUM_TABLES>;

    fn into_iter(self) -> Self::IntoIter {
        [
            &self.program_table as &'a dyn Table<BWord>,
            &self.instruction_table as &'a dyn Table<BWord>,
            &self.processor_table as &'a dyn Table<BWord>,
            &self.op_stack_table as &'a dyn Table<BWord>,
            &self.ram_table as &'a dyn Table<BWord>,
            &self.jump_stack_table as &'a dyn Table<BWord>,
            &self.hash_table as &'a dyn Table<BWord>,
            &self.u32_op_table as &'a dyn Table<BWord>,
        ]
        .into_iter()
    }
}

impl ExtTableCollection {
    pub fn with_padded_heights(num_trace_randomizers: usize, padded_heights: &[usize]) -> Self {
        // FIXME there must be a better way to access the padded heights
        let ext_program_table =
            ExtProgramTable::with_padded_height(num_trace_randomizers, padded_heights[0]);

        let ext_instruction_table =
            ExtInstructionTable::with_padded_height(num_trace_randomizers, padded_heights[1]);

        let ext_processor_table =
            ExtProcessorTable::with_padded_height(num_trace_randomizers, padded_heights[2]);

        let ext_op_stack_table =
            ExtOpStackTable::with_padded_height(num_trace_randomizers, padded_heights[3]);

        let ext_ram_table =
            ExtRamTable::with_padded_height(num_trace_randomizers, padded_heights[4]);

        let ext_jump_stack_table =
            ExtJumpStackTable::with_padded_height(num_trace_randomizers, padded_heights[5]);

        let ext_hash_table =
            ExtHashTable::with_padded_height(num_trace_randomizers, padded_heights[6]);

        let ext_u32_op_table =
            ExtU32OpTable::with_padded_height(num_trace_randomizers, padded_heights[7]);

        ExtTableCollection {
            program_table: ext_program_table,
            instruction_table: ext_instruction_table,
            processor_table: ext_processor_table,
            op_stack_table: ext_op_stack_table,
            ram_table: ext_ram_table,
            jump_stack_table: ext_jump_stack_table,
            hash_table: ext_hash_table,
            u32_op_table: ext_u32_op_table,
        }
    }

    pub fn max_degree(&self) -> Degree {
        self.into_iter()
            .map(|ext_table| ext_table.max_degree())
            .max()
            .unwrap_or(1)
    }

    /// Create an ExtTableCollection from a BaseTableCollection by
    /// `.extend()`ing each base table.
    ///
    /// The `.extend()` for each table is specific to that table, but always
    /// involves adding some number of columns. Each table only needs their
    /// own challenges and initials, but `AllChallenges` and `AllInitials`
    /// are passed everywhere to keep each table's `.extend()` homogenous.
    pub fn extend_tables(
        base_tables: &BaseTableCollection,
        all_challenges: &AllChallenges,
        all_initials: &AllEndpoints,
    ) -> (Self, AllEndpoints) {
        let (program_table, program_table_terminals) = base_tables.program_table.extend(
            &all_challenges.program_table_challenges,
            &all_initials.program_table_endpoints,
        );

        let (instruction_table, instruction_table_terminals) =
            base_tables.instruction_table.extend(
                &all_challenges.instruction_table_challenges,
                &all_initials.instruction_table_endpoints,
            );

        let (processor_table, processor_table_terminals) = base_tables.processor_table.extend(
            &all_challenges.processor_table_challenges,
            &all_initials.processor_table_endpoints,
        );

        let (op_stack_table, op_stack_table_terminals) = base_tables.op_stack_table.extend(
            &all_challenges.op_stack_table_challenges,
            &all_initials.op_stack_table_endpoints,
        );

        let (ram_table, ram_table_terminals) = base_tables.ram_table.extend(
            &all_challenges.ram_table_challenges,
            &all_initials.ram_table_endpoints,
        );

        let (jump_stack_table, jump_stack_table_terminals) = base_tables.jump_stack_table.extend(
            &all_challenges.jump_stack_table_challenges,
            &all_initials.jump_stack_table_endpoints,
        );

        let (hash_table, hash_table_terminals) = base_tables.hash_table.extend(
            &all_challenges.hash_table_challenges,
            &all_initials.hash_table_endpoints,
        );

        let (u32_op_table, u32_op_table_terminals) = base_tables.u32_op_table.extend(
            &all_challenges.u32_op_table_challenges,
            &all_initials.u32_op_table_endpoints,
        );

        let ext_tables = ExtTableCollection {
            program_table,
            instruction_table,
            processor_table,
            op_stack_table,
            ram_table,
            jump_stack_table,
            hash_table,
            u32_op_table,
        };

        let terminals = AllEndpoints {
            program_table_endpoints: program_table_terminals,
            instruction_table_endpoints: instruction_table_terminals,
            processor_table_endpoints: processor_table_terminals,
            op_stack_table_endpoints: op_stack_table_terminals,
            ram_table_endpoints: ram_table_terminals,
            jump_stack_table_endpoints: jump_stack_table_terminals,
            hash_table_endpoints: hash_table_terminals,
            u32_op_table_endpoints: u32_op_table_terminals,
        };

        (ext_tables, terminals)
    }

    pub fn codeword_index_to_table_name(&self, idx: usize) -> String {
        self.into_iter()
            .flat_map(|ext_table| vec![ext_table.name(); ext_table.full_width()])
            .collect_vec()[idx]
            .clone()
    }

    pub fn codeword_tables(
        &self,
        fri_domain: &FriDomain<XWord>,
        base_codeword_tables: BaseTableCollection,
    ) -> Self {
        let program_table = self
            .program_table
            .ext_codeword_table(fri_domain, base_codeword_tables.program_table.data());
        let instruction_table = self
            .instruction_table
            .ext_codeword_table(fri_domain, base_codeword_tables.instruction_table.data());
        let processor_table = self
            .processor_table
            .ext_codeword_table(fri_domain, base_codeword_tables.processor_table.data());
        let op_stack_table = self
            .op_stack_table
            .ext_codeword_table(fri_domain, base_codeword_tables.op_stack_table.data());
        let ram_table = self
            .ram_table
            .ext_codeword_table(fri_domain, base_codeword_tables.ram_table.data());
        let jump_stack_table = self
            .jump_stack_table
            .ext_codeword_table(fri_domain, base_codeword_tables.jump_stack_table.data());
        let hash_table = self
            .hash_table
            .ext_codeword_table(fri_domain, base_codeword_tables.hash_table.data());
        let u32_op_table = self
            .u32_op_table
            .ext_codeword_table(fri_domain, base_codeword_tables.u32_op_table.data());

        ExtTableCollection {
            program_table,
            instruction_table,
            processor_table,
            op_stack_table,
            ram_table,
            jump_stack_table,
            hash_table,
            u32_op_table,
        }
    }

    pub fn get_all_extension_columns(&self) -> Vec<Vec<XWord>> {
        let mut all_ext_cols = vec![];

        for table in self.into_iter() {
            for col in table.data().iter().skip(table.base_width()) {
                all_ext_cols.push(col.clone());
            }
        }
        all_ext_cols
    }

    pub fn data(&self, table_id: TableId) -> &Vec<Vec<XWord>> {
        use TableId::*;

        match table_id {
            ProgramTable => self.program_table.data(),
            InstructionTable => self.instruction_table.data(),
            ProcessorTable => self.processor_table.data(),
            OpStackTable => self.op_stack_table.data(),
            RamTable => self.ram_table.data(),
            JumpStackTable => self.jump_stack_table.data(),
            HashTable => self.hash_table.data(),
            U32OpTable => self.u32_op_table.data(),
        }
    }

    pub fn interpolant_degree(&self, table_id: TableId) -> Degree {
        use TableId::*;

        match table_id {
            ProgramTable => self.program_table.interpolant_degree(),
            InstructionTable => self.instruction_table.interpolant_degree(),
            ProcessorTable => self.processor_table.interpolant_degree(),
            OpStackTable => self.op_stack_table.interpolant_degree(),
            RamTable => self.ram_table.interpolant_degree(),
            JumpStackTable => self.jump_stack_table.interpolant_degree(),
            HashTable => self.hash_table.interpolant_degree(),
            U32OpTable => self.u32_op_table.interpolant_degree(),
        }
    }

    pub fn get_all_base_degree_bounds(&self) -> Vec<i64> {
        self.into_iter()
            .map(|ext_table| vec![ext_table.interpolant_degree(); ext_table.base_width()])
            .concat()
    }

    pub fn get_extension_column_degree_bounds(&self) -> Vec<i64> {
        self.into_iter()
            .map(|ext_table| {
                vec![
                    ext_table.interpolant_degree();
                    ext_table.full_width() - ext_table.base_width()
                ]
            })
            .concat()
    }

    pub fn get_all_quotients(
        &self,
        fri_domain: &FriDomain<XWord>,
        all_challenges: &AllChallenges,
        all_terminals: &AllEndpoints,
    ) -> Vec<Vec<XWord>> {
        let mut timer = TimingReporter::start();
        self.into_iter()
            .map(|ext_codeword_table| {
                timer.elapsed(&format!(
                    "Start calculating quotient: {}",
                    ext_codeword_table.name()
                ));
                let res = ext_codeword_table.all_quotients(
                    fri_domain,
                    ext_codeword_table.data(),
                    all_challenges,
                    all_terminals,
                );
                timer.elapsed(&format!(
                    "Ended calculating quotient: {}",
                    ext_codeword_table.name()
                ));
                res
            })
            .concat()
    }

    pub fn get_all_quotient_degree_bounds(
        &self,
        all_challenges: &AllChallenges,
        all_terminals: &AllEndpoints,
    ) -> Vec<Degree> {
        self.into_iter()
            .map(|ext_table| ext_table.all_quotient_degree_bounds(all_challenges, all_terminals))
            .concat()
    }
}

impl<'a> IntoIterator for &'a ExtTableCollection {
    type Item = &'a dyn ExtensionTable;

    type IntoIter = std::array::IntoIter<&'a dyn ExtensionTable, NUM_TABLES>;

    fn into_iter(self) -> Self::IntoIter {
        [
            &self.program_table as &'a dyn ExtensionTable,
            &self.instruction_table as &'a dyn ExtensionTable,
            &self.processor_table as &'a dyn ExtensionTable,
            &self.op_stack_table as &'a dyn ExtensionTable,
            &self.ram_table as &'a dyn ExtensionTable,
            &self.jump_stack_table as &'a dyn ExtensionTable,
            &self.hash_table as &'a dyn ExtensionTable,
            &self.u32_op_table as &'a dyn ExtensionTable,
        ]
        .into_iter()
    }
}

#[cfg(test)]
mod table_collection_tests {
    use super::*;
    use crate::shared_math::stark::triton::stark;
    use crate::shared_math::stark::triton::table::{
        hash_table, instruction_table, jump_stack_table, op_stack_table, processor_table,
        program_table, ram_table, u32_op_table,
    };

    fn dummy_ext_table_collection() -> ExtTableCollection {
        let num_trace_randomizers = 2;
        let max_padded_height = 1;

        ExtTableCollection::with_padded_heights(
            num_trace_randomizers,
            &[max_padded_height; NUM_TABLES],
        )
    }

    #[test]
    fn base_table_width_is_correct() {
        let num_trace_randomizers = 2;
        let base_matrices = BaseMatrices::default();
        let base_tables =
            BaseTableCollection::from_base_matrices(num_trace_randomizers, &base_matrices);

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
        assert_eq!(
            u32_op_table::BASE_WIDTH,
            base_tables.u32_op_table.base_width()
        );
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
        assert_eq!(
            u32_op_table::FULL_WIDTH,
            ext_tables.u32_op_table.full_width()
        );
    }

    #[test]
    fn constraint_polynomials_use_right_variable_count_test() {
        let ext_tables = dummy_ext_table_collection();
        let (all_challenges, all_initials) = stark::triton_stark_tests::dummy_challenges_initials();

        for ext_table in ext_tables.into_iter() {
            let boundary_constraints = ext_table.ext_boundary_constraints(&all_challenges);
            for (i, poly) in boundary_constraints.iter().enumerate() {
                assert_eq!(
                    ext_table.full_width(),
                    poly.variable_count,
                    "{}: The {}'th boundary constraint should have {} variables, has {} variables.",
                    ext_table.name(),
                    i,
                    ext_table.full_width(),
                    poly.variable_count,
                );
            }

            let consistency_constraints = ext_table.ext_consistency_constraints(&all_challenges);
            for (i, poly) in consistency_constraints.iter().enumerate() {
                assert_eq!(
                    ext_table.full_width(),
                    poly.variable_count,
                    "{}: The {}'th consistency constraint should have {} variables, has {} variables.",
                    ext_table.name(),
                    i,
                    ext_table.full_width(),
                    poly.variable_count,
                );
            }

            let transition_constraints = ext_table.ext_transition_constraints(&all_challenges);
            for (i, poly) in transition_constraints.iter().enumerate() {
                assert_eq!(
                    2 * ext_table.full_width(),
                    poly.variable_count,
                    "{}: The {}'th transition constraint should have {} variables, has {} variables.",
                    ext_table.name(),
                    i,
                    ext_table.full_width(),
                    poly.variable_count,
                );
            }

            let terminal_constraints =
                ext_table.ext_terminal_constraints(&all_challenges, &all_initials);
            for (i, poly) in terminal_constraints.iter().enumerate() {
                assert_eq!(
                    ext_table.full_width(),
                    poly.variable_count,
                    "{}: The {}'th terminal constraint should have {} variables, has {} variables.",
                    ext_table.name(),
                    i,
                    ext_table.full_width(),
                    poly.variable_count,
                );
            }
        }
    }
}
