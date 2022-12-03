use std::cmp::max;
use std::ops::MulAssign;

use itertools::Itertools;
use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use ndarray::s;
use ndarray::Array2;
use ndarray::ArrayView2;
use ndarray::ArrayViewMut2;
use num_traits::One;
use rand::distributions::Standard;
use rand::prelude::Distribution;
use rand::random;
use strum::EnumCount;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::mpolynomial::Degree;
use twenty_first::shared_math::other::is_power_of_two;
use twenty_first::shared_math::other::roundup_npo2;
use twenty_first::shared_math::traits::{FiniteField, ModPowU32};
use twenty_first::shared_math::traits::{Inverse, PrimitiveRootOfUnity};
use twenty_first::shared_math::x_field_element::XFieldElement;
use twenty_first::shared_math::x_field_element::EXTENSION_DEGREE;
use twenty_first::util_types::algebraic_hasher::AlgebraicHasher;
use twenty_first::util_types::merkle_tree::CpuParallel;
use twenty_first::util_types::merkle_tree::MerkleTree;
use twenty_first::util_types::merkle_tree_maker::MerkleTreeMaker;

use triton_profiler::prof_start;
use triton_profiler::prof_stop;
use triton_profiler::triton_profiler::TritonProfiler;

use crate::arithmetic_domain::ArithmeticDomain;
use crate::stark::StarkHasher;
use crate::table::base_matrix::AlgebraicExecutionTrace;
use crate::table::base_matrix::BaseMatrices;
use crate::table::base_table::Extendable;
use crate::table::base_table::InheritsFromTable;
use crate::table::base_table::Table;
use crate::table::base_table::TableLike;
use crate::table::challenges::AllChallenges;
use crate::table::extension_table::DegreeWithOrigin;
use crate::table::extension_table::Evaluable;
use crate::table::extension_table::ExtensionTable;
use crate::table::extension_table::Quotientable;
use crate::table::hash_table::ExtHashTable;
use crate::table::hash_table::HashTable;
use crate::table::instruction_table::ExtInstructionTable;
use crate::table::instruction_table::InstructionTable;
use crate::table::jump_stack_table::ExtJumpStackTable;
use crate::table::jump_stack_table::JumpStackTable;
use crate::table::op_stack_table::ExtOpStackTable;
use crate::table::op_stack_table::OpStackTable;
use crate::table::processor_table::ExtProcessorTable;
use crate::table::processor_table::ProcessorTable;
use crate::table::program_table::ExtProgramTable;
use crate::table::program_table::ProgramTable;
use crate::table::ram_table::ExtRamTable;
use crate::table::ram_table::RamTable;
use crate::table::table_column::*;
use crate::table::*;

pub const NUM_TABLES: usize = 7;

pub const NUM_BASE_COLUMNS: usize = program_table::BASE_WIDTH
    + instruction_table::BASE_WIDTH
    + processor_table::BASE_WIDTH
    + op_stack_table::BASE_WIDTH
    + ram_table::BASE_WIDTH
    + jump_stack_table::BASE_WIDTH
    + hash_table::BASE_WIDTH;
pub const NUM_EXT_COLUMNS: usize = program_table::EXT_WIDTH
    + instruction_table::EXT_WIDTH
    + processor_table::EXT_WIDTH
    + op_stack_table::EXT_WIDTH
    + ram_table::EXT_WIDTH
    + jump_stack_table::EXT_WIDTH
    + hash_table::EXT_WIDTH;
pub const NUM_COLUMNS: usize = NUM_BASE_COLUMNS + NUM_EXT_COLUMNS;

pub const PROGRAM_TABLE_START: usize = 0;
pub const PROGRAM_TABLE_END: usize = PROGRAM_TABLE_START + program_table::BASE_WIDTH;
pub const INSTRUCTION_TABLE_START: usize = PROGRAM_TABLE_END;
pub const INSTRUCTION_TABLE_END: usize = INSTRUCTION_TABLE_START + instruction_table::BASE_WIDTH;
pub const PROCESSOR_TABLE_START: usize = INSTRUCTION_TABLE_END;
pub const PROCESSOR_TABLE_END: usize = PROCESSOR_TABLE_START + processor_table::BASE_WIDTH;
pub const OP_STACK_TABLE_START: usize = PROCESSOR_TABLE_END;
pub const OP_STACK_TABLE_END: usize = OP_STACK_TABLE_START + op_stack_table::BASE_WIDTH;
pub const RAM_TABLE_START: usize = OP_STACK_TABLE_END;
pub const RAM_TABLE_END: usize = RAM_TABLE_START + ram_table::BASE_WIDTH;
pub const JUMP_STACK_TABLE_START: usize = RAM_TABLE_END;
pub const JUMP_STACK_TABLE_END: usize = JUMP_STACK_TABLE_START + jump_stack_table::BASE_WIDTH;
pub const HASH_TABLE_START: usize = JUMP_STACK_TABLE_END;
pub const HASH_TABLE_END: usize = HASH_TABLE_START + hash_table::BASE_WIDTH;

pub const EXT_PROGRAM_TABLE_START: usize = 0;
pub const EXT_PROGRAM_TABLE_END: usize = EXT_PROGRAM_TABLE_START + program_table::EXT_WIDTH;
pub const EXT_INSTRUCTION_TABLE_START: usize = EXT_PROGRAM_TABLE_END;
pub const EXT_INSTRUCTION_TABLE_END: usize =
    EXT_INSTRUCTION_TABLE_START + instruction_table::EXT_WIDTH;
pub const EXT_PROCESSOR_TABLE_START: usize = EXT_INSTRUCTION_TABLE_END;
pub const EXT_PROCESSOR_TABLE_END: usize = EXT_PROCESSOR_TABLE_START + processor_table::EXT_WIDTH;
pub const EXT_OP_STACK_TABLE_START: usize = EXT_PROCESSOR_TABLE_END;
pub const EXT_OP_STACK_TABLE_END: usize = EXT_OP_STACK_TABLE_START + op_stack_table::EXT_WIDTH;
pub const EXT_RAM_TABLE_START: usize = EXT_OP_STACK_TABLE_END;
pub const EXT_RAM_TABLE_END: usize = EXT_RAM_TABLE_START + ram_table::EXT_WIDTH;
pub const EXT_JUMP_STACK_TABLE_START: usize = EXT_RAM_TABLE_END;
pub const EXT_JUMP_STACK_TABLE_END: usize =
    EXT_JUMP_STACK_TABLE_START + jump_stack_table::EXT_WIDTH;
pub const EXT_HASH_TABLE_START: usize = EXT_JUMP_STACK_TABLE_END;
pub const EXT_HASH_TABLE_END: usize = EXT_HASH_TABLE_START + hash_table::EXT_WIDTH;

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

pub trait MasterTable<FF>
where
    FF: FiniteField + MulAssign<BFieldElement>,
    Standard: Distribution<FF>,
{
    fn randomized_padded_trace_len(&self) -> usize;
    fn rand_trace_to_padded_trace_unit_distance(&self) -> usize;
    fn master_matrix(&self) -> ArrayView2<FF>;
    fn master_matrix_mut(&mut self) -> ArrayViewMut2<FF>;
    fn fri_domain(&self) -> ArithmeticDomain;

    /// set all rows _not_ needed for the (padded) trace to random values
    fn randomize_trace(&mut self) {
        let randomized_padded_trace_len = self.randomized_padded_trace_len();
        let unit_distance = self.rand_trace_to_padded_trace_unit_distance();
        (1..unit_distance).for_each(|offset| {
            self.master_matrix_mut()
                .slice_mut(s![offset..randomized_padded_trace_len; unit_distance, ..])
                .par_mapv_inplace(|_| random::<FF>())
        });
    }

    /// requires underlying Array2 to be stored in column-major order
    /// resulting Array2 is stored row-major order
    fn low_degree_extend_all_columns(&self) -> Array2<FF>
    where
        Self: Sync,
    {
        // todo assert that underlying Array2 is is actually stored in column-major order
        let randomized_trace_domain_len = self.randomized_padded_trace_len();
        let randomized_trace_domain_gen =
            BFieldElement::primitive_root_of_unity(randomized_trace_domain_len as u64)
                .expect("Length of randomized trace domain must be a power of 2");
        let randomized_trace_domain = ArithmeticDomain::new(
            BFieldElement::one(),
            randomized_trace_domain_gen,
            randomized_trace_domain_len,
        );

        // todo
        //  - create Array2 with dimensions [fri.domain.length, master_base_table.ncols()]
        //  - per column: do LDE, move result in new Array2's column
        //  - change memory layout of Array2 from column-major to row-major (“into_shape_and_order”)
        //  - try: is it faster to create Array2 in row-major, move columns in, no transform?
        let a: Vec<_> = self
            .master_matrix()
            .slice(s![..randomized_trace_domain_len, ..])
            .axis_iter(Axis(1)) // Axis(1) corresponds to getting all columns.
            .into_iter()
            .map(|column| {
                let randomized_trace = column
                    .as_slice()
                    .expect("Column must be contiguous & non-empty.");
                randomized_trace_domain.low_degree_extension(randomized_trace, self.fri_domain())
            })
            .collect::<Vec<_>>()
            .concat();

        let num_rows = self.fri_domain().length;
        let num_columns = self.master_matrix().ncols();
        Array2::from_shape_vec([num_rows, num_columns], a)
            .expect("FRI domain codewords must fit into Array2 of given dimensions.")
    }
}

#[derive(Clone)]
pub struct MasterBaseTable {
    pub padded_height: usize,
    pub num_trace_randomizers: usize,
    pub num_randomizer_polynomials: usize,

    pub program_len: usize,
    pub main_execution_len: usize,
    pub hash_coprocessor_execution_len: usize,

    pub randomized_padded_trace_len: usize,

    /// how many elements to skip in the randomized (padded) trace domain to only refer to
    /// elements in the _non_-randomized (padded) trace domain
    pub rand_trace_to_padded_trace_unit_distance: usize,

    pub fri_domain: ArithmeticDomain,
    pub master_base_matrix: Array2<BFieldElement>,
}

pub struct MasterExtTable {
    pub padded_height: usize,
    pub num_trace_randomizers: usize,

    pub randomized_padded_trace_len: usize,

    /// how many elements to skip in the randomized (padded) trace domain to only refer to
    /// elements in the _non_-randomized (padded) trace domain
    pub rand_trace_to_padded_trace_unit_distance: usize,

    pub fri_domain: ArithmeticDomain,
    pub master_ext_matrix: Array2<XFieldElement>,
}

impl MasterTable<BFieldElement> for MasterBaseTable {
    fn randomized_padded_trace_len(&self) -> usize {
        self.randomized_padded_trace_len
    }

    fn rand_trace_to_padded_trace_unit_distance(&self) -> usize {
        self.rand_trace_to_padded_trace_unit_distance
    }

    fn master_matrix(&self) -> ArrayView2<BFieldElement> {
        self.master_base_matrix.view()
    }

    fn master_matrix_mut(&mut self) -> ArrayViewMut2<BFieldElement> {
        self.master_base_matrix.view_mut()
    }

    fn fri_domain(&self) -> ArithmeticDomain {
        self.fri_domain
    }
}

impl MasterTable<XFieldElement> for MasterExtTable {
    fn randomized_padded_trace_len(&self) -> usize {
        self.randomized_padded_trace_len
    }

    fn rand_trace_to_padded_trace_unit_distance(&self) -> usize {
        self.rand_trace_to_padded_trace_unit_distance
    }

    fn master_matrix(&self) -> ArrayView2<XFieldElement> {
        self.master_ext_matrix.view()
    }

    fn master_matrix_mut(&mut self) -> ArrayViewMut2<XFieldElement> {
        self.master_ext_matrix.view_mut()
    }

    fn fri_domain(&self) -> ArithmeticDomain {
        self.fri_domain
    }
}

impl MasterBaseTable {
    pub fn padded_height(aet: &AlgebraicExecutionTrace, program: &[BFieldElement]) -> usize {
        let instruction_table_len = program.len() + aet.processor_matrix.len();
        let hash_table_len = aet.hash_matrix.len();
        let max_height = max(instruction_table_len, hash_table_len);
        roundup_npo2(max_height as u64) as usize
    }

    pub fn new(
        aet: AlgebraicExecutionTrace,
        program: &[BFieldElement],
        num_trace_randomizers: usize,
        num_randomizer_polynomials: usize,
        fri_domain: ArithmeticDomain,
    ) -> Self {
        let padded_height = Self::padded_height(&aet, program);
        let randomized_padded_trace_len =
            roundup_npo2((padded_height + num_trace_randomizers) as u64) as usize;
        let unit_distance = randomized_padded_trace_len as usize / padded_height;
        let program_len = program.len();
        let main_execution_len = aet.processor_matrix.len();
        let hash_coprocessor_execution_len = aet.hash_matrix.len();

        let num_rows = randomized_padded_trace_len;
        let num_columns = num_randomizer_polynomials + NUM_BASE_COLUMNS;
        let mut master_base_matrix = Array2::zeros([num_rows, num_columns].f());

        // randomizer polynomials
        let num_randomizer_columns = EXTENSION_DEGREE * num_randomizer_polynomials;
        master_base_matrix
            .slice_mut(s![.., ..num_randomizer_columns])
            .par_mapv_inplace(|_| random::<BFieldElement>());

        let mut master_base_table = Self {
            padded_height,
            num_trace_randomizers,
            num_randomizer_polynomials,
            program_len,
            main_execution_len,
            hash_coprocessor_execution_len,
            randomized_padded_trace_len,
            rand_trace_to_padded_trace_unit_distance: unit_distance,
            fri_domain,
            master_base_matrix,
        };

        let program_table = &mut master_base_table.table_mut(TableId::ProgramTable);
        ProgramTable::fill_trace(program_table, program);
        let instruction_table = &mut master_base_table.table_mut(TableId::InstructionTable);
        InstructionTable::fill_trace(instruction_table, &aet, program);

        master_base_table
    }

    pub fn pad(&mut self) {
        let program_len = self.program_len;

        let program_table = &mut self.table_mut(TableId::ProgramTable);
        ProgramTable::pad_trace(program_table, program_len);
    }

    pub fn to_fri_domain_table(&self) -> Self {
        Self {
            master_base_matrix: self.low_degree_extend_all_columns(),
            ..*self
        }
    }

    /// requires underlying Array2 to be stored row-major order
    pub fn merkle_tree(&self) -> MerkleTree<StarkHasher, CpuParallel> {
        let mut hashed_rows = Vec::with_capacity(self.fri_domain.length);
        self.master_base_matrix
            .axis_iter(Axis(0)) // Axis(0) corresponds to getting all rows.
            .into_par_iter()
            .map(|row| {
                let contiguous_row = row.as_slice().expect("Row must be contiguous & non-empty.");
                StarkHasher::hash_slice(contiguous_row)
            })
            .collect_into_vec(&mut hashed_rows);
        CpuParallel::from_digests(&hashed_rows)
    }

    pub fn extend(&self, challenges: &AllChallenges) -> MasterExtTable {
        let master_ext_matrix = Array2::zeros([self.fri_domain.length, NUM_EXT_COLUMNS].f());

        let mut master_ext_table = MasterExtTable {
            padded_height: self.padded_height,
            num_trace_randomizers: self.num_trace_randomizers,
            randomized_padded_trace_len: self.randomized_padded_trace_len,
            rand_trace_to_padded_trace_unit_distance: self.rand_trace_to_padded_trace_unit_distance,
            fri_domain: self.fri_domain,
            master_ext_matrix,
        };

        let base_program_table = self.table(TableId::ProgramTable);
        let mut ext_program_table = master_ext_table.table_mut(TableId::ProgramTable);
        ProgramTable::the_new_extend_method_is_in_place(
            &base_program_table,
            &mut ext_program_table,
            &challenges.program_table_challenges,
        );

        master_ext_table
    }

    // todo: move randomizer codewords to extension table, stay type-native
    pub fn randomizer_polynomials(&self) -> Vec<Vec<XFieldElement>> {
        self.master_base_matrix
            .slice(s![.., ..EXTENSION_DEGREE * self.num_randomizer_polynomials])
            .exact_chunks([self.fri_domain.length, EXTENSION_DEGREE])
            .into_iter()
            .map(|three_entire_cols| {
                three_entire_cols
                    .axis_iter(Axis(0))
                    .into_par_iter()
                    .map(|three_bfes| {
                        XFieldElement::new([three_bfes[0], three_bfes[1], three_bfes[2]])
                    })
                    .collect::<Vec<_>>()
            })
            .collect()
    }

    fn table_slice_info(id: TableId) -> (usize, usize) {
        use TableId::*;
        match id {
            ProgramTable => (PROGRAM_TABLE_START, PROGRAM_TABLE_END),
            InstructionTable => (INSTRUCTION_TABLE_START, INSTRUCTION_TABLE_END),
            ProcessorTable => (PROCESSOR_TABLE_START, PROCESSOR_TABLE_END),
            OpStackTable => (OP_STACK_TABLE_START, OP_STACK_TABLE_END),
            RamTable => (RAM_TABLE_START, RAM_TABLE_END),
            JumpStackTable => (JUMP_STACK_TABLE_START, JUMP_STACK_TABLE_END),
            HashTable => (HASH_TABLE_START, HASH_TABLE_END),
        }
    }

    pub fn table(&self, id: TableId) -> ArrayView2<BFieldElement> {
        let num_randomizer_columns = EXTENSION_DEGREE * self.num_randomizer_polynomials;
        let (table_start, table_end) = Self::table_slice_info(id);
        let starting_col = num_randomizer_columns + table_start;
        let ending_col = num_randomizer_columns + table_end;

        let unit_distance = self.rand_trace_to_padded_trace_unit_distance;
        self.master_base_matrix
            .slice(s![..; unit_distance, starting_col..ending_col])
    }

    pub fn table_mut(&mut self, id: TableId) -> ArrayViewMut2<BFieldElement> {
        let num_randomizer_columns = EXTENSION_DEGREE * self.num_randomizer_polynomials;
        let (table_start, table_end) = Self::table_slice_info(id);
        let starting_col = num_randomizer_columns + table_start;
        let ending_col = num_randomizer_columns + table_end;

        let unit_distance = self.rand_trace_to_padded_trace_unit_distance;
        self.master_base_matrix
            .slice_mut(s![..; unit_distance, starting_col..ending_col])
    }
}

impl MasterExtTable {
    pub fn to_fri_domain_table(&self) -> Self {
        Self {
            master_ext_matrix: self.low_degree_extend_all_columns(),
            ..*self
        }
    }

    /// requires underlying Array2 to be stored row-major order
    pub fn merkle_tree(&self) -> MerkleTree<StarkHasher, CpuParallel> {
        let mut hashed_rows = Vec::with_capacity(self.fri_domain.length);
        self.master_ext_matrix
            .axis_iter(Axis(0)) // Axis(0) corresponds to getting all rows.
            .into_par_iter()
            .map(|row| {
                let contiguous_row = row.as_slice().expect("Row must be contiguous & non-empty.");
                let contiguous_row_bfe = contiguous_row
                    .iter()
                    .map(|elem| elem.coefficients.to_vec())
                    .concat();
                StarkHasher::hash_slice(&contiguous_row_bfe)
            })
            .collect_into_vec(&mut hashed_rows);
        CpuParallel::from_digests(&hashed_rows)
    }

    fn table_slice_info(id: TableId) -> (usize, usize) {
        use TableId::*;
        match id {
            ProgramTable => (EXT_PROGRAM_TABLE_START, EXT_PROGRAM_TABLE_END),
            InstructionTable => (EXT_INSTRUCTION_TABLE_START, EXT_INSTRUCTION_TABLE_END),
            ProcessorTable => (EXT_PROCESSOR_TABLE_START, EXT_PROCESSOR_TABLE_END),
            OpStackTable => (EXT_OP_STACK_TABLE_START, EXT_OP_STACK_TABLE_END),
            RamTable => (EXT_RAM_TABLE_START, EXT_RAM_TABLE_END),
            JumpStackTable => (EXT_JUMP_STACK_TABLE_START, EXT_JUMP_STACK_TABLE_END),
            HashTable => (EXT_HASH_TABLE_START, EXT_HASH_TABLE_END),
        }
    }

    pub fn table(&self, id: TableId) -> ArrayView2<XFieldElement> {
        let unit_distance = self.rand_trace_to_padded_trace_unit_distance;
        let (starting_col, ending_col) = Self::table_slice_info(id);
        self.master_ext_matrix
            .slice(s![..; unit_distance, starting_col..ending_col])
    }

    pub fn table_mut(&mut self, id: TableId) -> ArrayViewMut2<XFieldElement> {
        let unit_distance = self.rand_trace_to_padded_trace_unit_distance;
        let (starting_col, ending_col) = Self::table_slice_info(id);
        self.master_ext_matrix
            .slice_mut(s![..; unit_distance, starting_col..ending_col])
    }
}

pub fn base_degree_bounds(
    padded_height: usize,
    num_trace_randomizers: usize,
) -> [Degree; NUM_BASE_COLUMNS] {
    [interpolant_degree(padded_height, num_trace_randomizers); NUM_BASE_COLUMNS]
}

pub fn extension_degree_bounds(
    padded_height: usize,
    num_trace_randomizers: usize,
) -> [Degree; NUM_EXT_COLUMNS] {
    [interpolant_degree(padded_height, num_trace_randomizers); NUM_EXT_COLUMNS]
}

pub fn all_degrees_with_origin(
    padded_height: usize,
    num_trace_randomizers: usize,
) -> Vec<DegreeWithOrigin> {
    let num_rand = num_trace_randomizers;
    [
        ExtProgramTable::all_degrees_with_origin("program table", padded_height, num_rand),
        ExtInstructionTable::all_degrees_with_origin("instruction table", padded_height, num_rand),
        ExtProcessorTable::all_degrees_with_origin("processor table", padded_height, num_rand),
        ExtOpStackTable::all_degrees_with_origin("op stack table", padded_height, num_rand),
        ExtRamTable::all_degrees_with_origin("ram table", padded_height, num_rand),
        ExtJumpStackTable::all_degrees_with_origin("jump stack table", padded_height, num_rand),
        ExtHashTable::all_degrees_with_origin("hash table", padded_height, num_rand),
    ]
    .concat()
}

pub fn max_degree_with_origin(
    padded_height: usize,
    num_trace_randomizers: usize,
) -> DegreeWithOrigin {
    all_degrees_with_origin(padded_height, num_trace_randomizers)
        .into_iter()
        .max()
        .unwrap_or_default()
}

pub fn all_initial_quotient_degree_bounds(
    padded_height: usize,
    num_trace_randomizers: usize,
) -> Vec<Degree> {
    [
        ExtProgramTable::initial_quotient_degree_bounds(padded_height, num_trace_randomizers),
        ExtInstructionTable::initial_quotient_degree_bounds(padded_height, num_trace_randomizers),
        ExtProcessorTable::initial_quotient_degree_bounds(padded_height, num_trace_randomizers),
        ExtOpStackTable::initial_quotient_degree_bounds(padded_height, num_trace_randomizers),
        ExtRamTable::initial_quotient_degree_bounds(padded_height, num_trace_randomizers),
        ExtJumpStackTable::initial_quotient_degree_bounds(padded_height, num_trace_randomizers),
        ExtHashTable::initial_quotient_degree_bounds(padded_height, num_trace_randomizers),
    ]
    .concat()
}

pub fn all_consistency_quotient_degree_bounds(
    padded_height: usize,
    num_trace_randomizers: usize,
) -> Vec<Degree> {
    let ntr = num_trace_randomizers;
    [
        ExtProgramTable::consistency_quotient_degree_bounds(padded_height, ntr),
        ExtInstructionTable::consistency_quotient_degree_bounds(padded_height, ntr),
        ExtProcessorTable::consistency_quotient_degree_bounds(padded_height, ntr),
        ExtOpStackTable::consistency_quotient_degree_bounds(padded_height, ntr),
        ExtRamTable::consistency_quotient_degree_bounds(padded_height, ntr),
        ExtJumpStackTable::consistency_quotient_degree_bounds(padded_height, ntr),
        ExtHashTable::consistency_quotient_degree_bounds(padded_height, ntr),
    ]
    .concat()
}

pub fn all_transition_quotient_degree_bounds(
    padded_height: usize,
    num_trace_randomizers: usize,
) -> Vec<Degree> {
    let ntr = num_trace_randomizers;
    [
        ExtProgramTable::transition_quotient_degree_bounds(padded_height, ntr),
        ExtInstructionTable::transition_quotient_degree_bounds(padded_height, ntr),
        ExtProcessorTable::transition_quotient_degree_bounds(padded_height, ntr),
        ExtOpStackTable::transition_quotient_degree_bounds(padded_height, ntr),
        ExtRamTable::transition_quotient_degree_bounds(padded_height, ntr),
        ExtJumpStackTable::transition_quotient_degree_bounds(padded_height, ntr),
        ExtHashTable::transition_quotient_degree_bounds(padded_height, ntr),
    ]
    .concat()
}

pub fn all_terminal_quotient_degree_bounds(
    padded_height: usize,
    num_trace_randomizers: usize,
) -> Vec<Degree> {
    [
        ExtProgramTable::terminal_quotient_degree_bounds(padded_height, num_trace_randomizers),
        ExtInstructionTable::terminal_quotient_degree_bounds(padded_height, num_trace_randomizers),
        ExtProcessorTable::terminal_quotient_degree_bounds(padded_height, num_trace_randomizers),
        ExtOpStackTable::terminal_quotient_degree_bounds(padded_height, num_trace_randomizers),
        ExtRamTable::terminal_quotient_degree_bounds(padded_height, num_trace_randomizers),
        ExtJumpStackTable::terminal_quotient_degree_bounds(padded_height, num_trace_randomizers),
        ExtHashTable::terminal_quotient_degree_bounds(padded_height, num_trace_randomizers),
    ]
    .concat()
}

pub fn all_quotient_degree_bounds(
    padded_height: usize,
    num_trace_randomizers: usize,
) -> Vec<Degree> {
    [
        all_initial_quotient_degree_bounds(padded_height, num_trace_randomizers),
        all_consistency_quotient_degree_bounds(padded_height, num_trace_randomizers),
        all_transition_quotient_degree_bounds(padded_height, num_trace_randomizers),
        all_terminal_quotient_degree_bounds(padded_height, num_trace_randomizers),
    ]
    .concat()
}

pub fn initial_quotient_zerofier_inverse(domain: ArithmeticDomain) -> Array1<BFieldElement> {
    let zerofier_codeword = domain
        .domain_values()
        .into_iter()
        .map(|x| x - BFieldElement::one())
        .collect();
    BFieldElement::batch_inversion(zerofier_codeword).into()
}

pub fn consistency_quotient_zerofier_inverse(
    domain: ArithmeticDomain,
    padded_height: usize,
) -> Array1<BFieldElement> {
    let zerofier_codeword = domain
        .domain_values()
        .iter()
        .map(|x| x.mod_pow_u32(padded_height as u32) - BFieldElement::one())
        .collect();
    BFieldElement::batch_inversion(zerofier_codeword).into()
}

pub fn transition_quotient_zerofier_inverse(
    domain: ArithmeticDomain,
    trace_domain_generator: BFieldElement,
    padded_height: usize,
) -> Array1<BFieldElement> {
    let one = BFieldElement::one();
    let trace_domain_generator_inverse = trace_domain_generator.inverse();
    let domain_values = domain.domain_values();

    let subgroup_zerofier: Vec<_> = domain_values
        .par_iter()
        .map(|domain_value| domain_value.mod_pow_u32(padded_height as u32) - one)
        .collect();
    let subgroup_zerofier_inverse = BFieldElement::batch_inversion(subgroup_zerofier);
    let zerofier_inverse: Vec<_> = domain_values
        .into_par_iter()
        .zip_eq(subgroup_zerofier_inverse.into_par_iter())
        .map(|(domain_value, sub_z_inv)| {
            (domain_value - trace_domain_generator_inverse) * sub_z_inv
        })
        .collect();
    zerofier_inverse.into()
}

pub fn terminal_quotient_zerofier_inverse(
    domain: ArithmeticDomain,
    trace_domain_generator: BFieldElement,
) -> Array1<BFieldElement> {
    // The zerofier for the terminal quotient has a root in the last
    // value in the cyclical group generated from the trace domain's generator.
    let zerofier_codeword = domain
        .domain_values()
        .into_iter()
        .map(|x| x - trace_domain_generator.inverse())
        .collect_vec();
    BFieldElement::batch_inversion(zerofier_codeword).into()
}

pub fn fill_all_initial_quotients(
    master_base_table: ArrayView2<BFieldElement>,
    master_ext_table: ArrayView2<XFieldElement>,
    quot_table: &mut ArrayViewMut2<XFieldElement>,
    zerofier_inverse: ArrayView1<BFieldElement>,
    challenges: &AllChallenges,
) {
    let num_program_quotients = ExtProgramTable::initial_quotient_degree_bounds(2, 2).len();
    let num_instruction_quotients = ExtInstructionTable::initial_quotient_degree_bounds(2, 2).len();
    let num_processor_quotients = ExtProcessorTable::initial_quotient_degree_bounds(2, 2).len();
    let num_op_stack_quotients = ExtOpStackTable::initial_quotient_degree_bounds(2, 2).len();
    let num_ram_quotients = ExtRamTable::initial_quotient_degree_bounds(2, 2).len();
    let num_jump_stack_quotients = ExtJumpStackTable::initial_quotient_degree_bounds(2, 2).len();

    // The order of the quotient tables is not actually important. However, it must be consistent
    // between prover and verifier, and the shapes must check out.
    let mut program_quot_table = quot_table.slice_mut(s![.., ..num_program_quotients]);

    ExtProgramTable::fill_initial_quotients(
        master_base_table,
        master_ext_table,
        &mut program_quot_table,
        zerofier_inverse,
        challenges,
    );
    let mut instruction_quot_table =
        quot_table.slice_mut(s![.., num_program_quotients..num_instruction_quotients]);
    ExtInstructionTable::fill_initial_quotients(
        master_base_table,
        master_ext_table,
        &mut instruction_quot_table,
        zerofier_inverse,
        challenges,
    );
    let mut processor_quot_table =
        quot_table.slice_mut(s![.., num_instruction_quotients..num_processor_quotients]);
    ExtProcessorTable::fill_initial_quotients(
        master_base_table,
        master_ext_table,
        &mut processor_quot_table,
        zerofier_inverse,
        challenges,
    );
    let mut op_stack_quot_table =
        quot_table.slice_mut(s![.., num_processor_quotients..num_op_stack_quotients]);
    ExtOpStackTable::fill_initial_quotients(
        master_base_table,
        master_ext_table,
        &mut op_stack_quot_table,
        zerofier_inverse,
        challenges,
    );
    let mut ram_quot_table =
        quot_table.slice_mut(s![.., num_op_stack_quotients..num_ram_quotients]);
    ExtRamTable::fill_initial_quotients(
        master_base_table,
        master_ext_table,
        &mut ram_quot_table,
        zerofier_inverse,
        challenges,
    );
    let mut jump_stack_quot_table =
        quot_table.slice_mut(s![.., num_ram_quotients..num_jump_stack_quotients]);
    ExtJumpStackTable::fill_initial_quotients(
        master_base_table,
        master_ext_table,
        &mut jump_stack_quot_table,
        zerofier_inverse,
        challenges,
    );
    let mut hash_quot_table = quot_table.slice_mut(s![.., num_jump_stack_quotients..]);
    ExtHashTable::fill_initial_quotients(
        master_base_table,
        master_ext_table,
        &mut hash_quot_table,
        zerofier_inverse,
        challenges,
    );
}

pub fn fill_all_consistency_quotients(
    master_base_table: ArrayView2<BFieldElement>,
    master_ext_table: ArrayView2<XFieldElement>,
    quot_table: &mut ArrayViewMut2<XFieldElement>,
    zerofier_inverse: ArrayView1<BFieldElement>,
    challenges: &AllChallenges,
) {
    let num_program_quotients = ExtProgramTable::consistency_quotient_degree_bounds(2, 2).len();
    let num_instruction_quotients =
        ExtInstructionTable::consistency_quotient_degree_bounds(2, 2).len();
    let num_processor_quotients = ExtProcessorTable::consistency_quotient_degree_bounds(2, 2).len();
    let num_op_stack_quotients = ExtOpStackTable::consistency_quotient_degree_bounds(2, 2).len();
    let num_ram_quotients = ExtRamTable::consistency_quotient_degree_bounds(2, 2).len();
    let num_jump_stack_quotients =
        ExtJumpStackTable::consistency_quotient_degree_bounds(2, 2).len();

    // The order of the quotient tables is not actually important. However, it must be consistent
    // between prover and verifier, and the shapes must check out.
    let mut program_quot_table = quot_table.slice_mut(s![.., ..num_program_quotients]);
    ExtProgramTable::fill_consistency_quotients(
        master_base_table,
        master_ext_table,
        &mut program_quot_table,
        zerofier_inverse,
        challenges,
    );
    let mut instruction_quot_table =
        quot_table.slice_mut(s![.., num_program_quotients..num_instruction_quotients]);
    ExtInstructionTable::fill_consistency_quotients(
        master_base_table,
        master_ext_table,
        &mut instruction_quot_table,
        zerofier_inverse,
        challenges,
    );
    let mut processor_quot_table =
        quot_table.slice_mut(s![.., num_instruction_quotients..num_processor_quotients]);
    ExtProcessorTable::fill_consistency_quotients(
        master_base_table,
        master_ext_table,
        &mut processor_quot_table,
        zerofier_inverse,
        challenges,
    );
    let mut op_stack_quot_table =
        quot_table.slice_mut(s![.., num_processor_quotients..num_op_stack_quotients]);
    ExtOpStackTable::fill_consistency_quotients(
        master_base_table,
        master_ext_table,
        &mut op_stack_quot_table,
        zerofier_inverse,
        challenges,
    );
    let mut ram_quot_table =
        quot_table.slice_mut(s![.., num_op_stack_quotients..num_ram_quotients]);
    ExtRamTable::fill_consistency_quotients(
        master_base_table,
        master_ext_table,
        &mut ram_quot_table,
        zerofier_inverse,
        challenges,
    );
    let mut jump_stack_quot_table =
        quot_table.slice_mut(s![.., num_ram_quotients..num_jump_stack_quotients]);
    ExtJumpStackTable::fill_consistency_quotients(
        master_base_table,
        master_ext_table,
        &mut jump_stack_quot_table,
        zerofier_inverse,
        challenges,
    );
    let mut hash_quot_table = quot_table.slice_mut(s![.., num_jump_stack_quotients..]);
    ExtHashTable::fill_consistency_quotients(
        master_base_table,
        master_ext_table,
        &mut hash_quot_table,
        zerofier_inverse,
        challenges,
    );
}

pub fn fill_all_transition_quotients(
    master_base_table: ArrayView2<BFieldElement>,
    master_ext_table: ArrayView2<XFieldElement>,
    quot_table: &mut ArrayViewMut2<XFieldElement>,
    zerofier_inverse: ArrayView1<BFieldElement>,
    challenges: &AllChallenges,
    quotient_domain: ArithmeticDomain,
    padded_height: usize,
) {
    let num_program_quotients = ExtProgramTable::transition_quotient_degree_bounds(2, 2).len();
    let num_instruction_quotients =
        ExtInstructionTable::transition_quotient_degree_bounds(2, 2).len();
    let num_processor_quotients = ExtProcessorTable::transition_quotient_degree_bounds(2, 2).len();
    let num_op_stack_quotients = ExtOpStackTable::transition_quotient_degree_bounds(2, 2).len();
    let num_ram_quotients = ExtRamTable::transition_quotient_degree_bounds(2, 2).len();
    let num_jump_stack_quotients = ExtJumpStackTable::transition_quotient_degree_bounds(2, 2).len();

    // The order of the quotient tables is not actually important. However, it must be consistent
    // between prover and verifier, and the shapes must check out.
    let mut program_quot_table = quot_table.slice_mut(s![.., ..num_program_quotients]);

    ExtProgramTable::fill_transition_quotients(
        master_base_table,
        master_ext_table,
        &mut program_quot_table,
        zerofier_inverse,
        challenges,
        quotient_domain,
        padded_height,
    );
    let mut instruction_quot_table =
        quot_table.slice_mut(s![.., num_program_quotients..num_instruction_quotients]);
    ExtInstructionTable::fill_transition_quotients(
        master_base_table,
        master_ext_table,
        &mut instruction_quot_table,
        zerofier_inverse,
        challenges,
        quotient_domain,
        padded_height,
    );
    let mut processor_quot_table =
        quot_table.slice_mut(s![.., num_instruction_quotients..num_processor_quotients]);
    ExtProcessorTable::fill_transition_quotients(
        master_base_table,
        master_ext_table,
        &mut processor_quot_table,
        zerofier_inverse,
        challenges,
        quotient_domain,
        padded_height,
    );
    let mut op_stack_quot_table =
        quot_table.slice_mut(s![.., num_processor_quotients..num_op_stack_quotients]);
    ExtOpStackTable::fill_transition_quotients(
        master_base_table,
        master_ext_table,
        &mut op_stack_quot_table,
        zerofier_inverse,
        challenges,
        quotient_domain,
        padded_height,
    );
    let mut ram_quot_table =
        quot_table.slice_mut(s![.., num_op_stack_quotients..num_ram_quotients]);
    ExtRamTable::fill_transition_quotients(
        master_base_table,
        master_ext_table,
        &mut ram_quot_table,
        zerofier_inverse,
        challenges,
        quotient_domain,
        padded_height,
    );
    let mut jump_stack_quot_table =
        quot_table.slice_mut(s![.., num_ram_quotients..num_jump_stack_quotients]);
    ExtJumpStackTable::fill_transition_quotients(
        master_base_table,
        master_ext_table,
        &mut jump_stack_quot_table,
        zerofier_inverse,
        challenges,
        quotient_domain,
        padded_height,
    );
    let mut hash_quot_table = quot_table.slice_mut(s![.., num_jump_stack_quotients..]);
    ExtHashTable::fill_transition_quotients(
        master_base_table,
        master_ext_table,
        &mut hash_quot_table,
        zerofier_inverse,
        challenges,
        quotient_domain,
        padded_height,
    );
}

pub fn fill_all_terminal_quotients(
    master_base_table: ArrayView2<BFieldElement>,
    master_ext_table: ArrayView2<XFieldElement>,
    quot_table: &mut ArrayViewMut2<XFieldElement>,
    zerofier_inverse: ArrayView1<BFieldElement>,
    challenges: &AllChallenges,
) {
    let num_program_quotients = ExtProgramTable::terminal_quotient_degree_bounds(2, 2).len();
    let num_instruction_quotients =
        ExtInstructionTable::terminal_quotient_degree_bounds(2, 2).len();
    let num_processor_quotients = ExtProcessorTable::terminal_quotient_degree_bounds(2, 2).len();
    let num_op_stack_quotients = ExtOpStackTable::terminal_quotient_degree_bounds(2, 2).len();
    let num_ram_quotients = ExtRamTable::terminal_quotient_degree_bounds(2, 2).len();
    let num_jump_stack_quotients = ExtJumpStackTable::terminal_quotient_degree_bounds(2, 2).len();

    // The order of the quotient tables is not actually important. However, it must be consistent
    // between prover and verifier, and the shapes must check out.
    let mut program_quot_table = quot_table.slice_mut(s![.., ..num_program_quotients]);
    ExtProgramTable::fill_terminal_quotients(
        master_base_table,
        master_ext_table,
        &mut program_quot_table,
        zerofier_inverse,
        challenges,
    );
    let mut instruction_quot_table =
        quot_table.slice_mut(s![.., num_program_quotients..num_instruction_quotients]);
    ExtInstructionTable::fill_terminal_quotients(
        master_base_table,
        master_ext_table,
        &mut instruction_quot_table,
        zerofier_inverse,
        challenges,
    );
    let mut processor_quot_table =
        quot_table.slice_mut(s![.., num_instruction_quotients..num_processor_quotients]);
    ExtProcessorTable::fill_terminal_quotients(
        master_base_table,
        master_ext_table,
        &mut processor_quot_table,
        zerofier_inverse,
        challenges,
    );
    let mut op_stack_quot_table =
        quot_table.slice_mut(s![.., num_processor_quotients..num_op_stack_quotients]);
    ExtOpStackTable::fill_terminal_quotients(
        master_base_table,
        master_ext_table,
        &mut op_stack_quot_table,
        zerofier_inverse,
        challenges,
    );
    let mut ram_quot_table =
        quot_table.slice_mut(s![.., num_op_stack_quotients..num_ram_quotients]);
    ExtRamTable::fill_terminal_quotients(
        master_base_table,
        master_ext_table,
        &mut ram_quot_table,
        zerofier_inverse,
        challenges,
    );
    let mut jump_stack_quot_table =
        quot_table.slice_mut(s![.., num_ram_quotients..num_jump_stack_quotients]);
    ExtJumpStackTable::fill_terminal_quotients(
        master_base_table,
        master_ext_table,
        &mut jump_stack_quot_table,
        zerofier_inverse,
        challenges,
    );
    let mut hash_quot_table = quot_table.slice_mut(s![.., num_jump_stack_quotients..]);
    ExtHashTable::fill_terminal_quotients(
        master_base_table,
        master_ext_table,
        &mut hash_quot_table,
        zerofier_inverse,
        challenges,
    );
}

/// Computes an array containing all quotients. Each column corresponds to a different quotient.
/// The quotients are ordered by category – initial, consistency, transition, and then terminal.
/// Within each category, the quotients follow the canonical order of the tables. The order of the
/// quotients is not actually important. However, it must be consistent between prover and verifier.
///
/// The right-most column is not filled in, leaving room for the grand cross-table argument.
/// The returned array is in row-major order.
// todo should the array be in column-major order instead?
pub fn all_quotients(
    quotient_domain_master_base_table: ArrayView2<BFieldElement>,
    quotient_domain_master_ext_table: ArrayView2<XFieldElement>,
    padded_height: usize,
    quotient_domain: ArithmeticDomain,
    num_quotients: usize,
    challenges: &AllChallenges,
    maybe_profiler: &mut Option<TritonProfiler>,
) -> Array2<XFieldElement> {
    assert_eq!(
        quotient_domain.length,
        quotient_domain_master_base_table.nrows(),
    );
    assert_eq!(
        quotient_domain.length,
        quotient_domain_master_ext_table.nrows()
    );

    let trace_domain_generator = derive_trace_domain_generator(padded_height as u64);
    let mut all_quotients = Array2::zeros([quotient_domain.length, num_quotients]);

    let num_initial_quotients = all_initial_quotient_degree_bounds(2, 2).len();
    let num_consistency_quotients = all_consistency_quotient_degree_bounds(2, 2).len();
    let num_transition_quotients = all_transition_quotient_degree_bounds(2, 2).len();
    let num_terminal_quotients = all_terminal_quotient_degree_bounds(2, 2).len();

    prof_start!(maybe_profiler, "initial");
    let mut initial_quot_table = all_quotients.slice_mut(s![.., ..num_initial_quotients]);
    let initial_quotient_zerofier_inverse = initial_quotient_zerofier_inverse(quotient_domain);
    fill_all_initial_quotients(
        quotient_domain_master_base_table,
        quotient_domain_master_ext_table,
        &mut initial_quot_table,
        initial_quotient_zerofier_inverse.view(),
        challenges,
    );
    prof_stop!(maybe_profiler, "initial");

    prof_start!(maybe_profiler, "consistency");
    let mut consistency_quotients =
        all_quotients.slice_mut(s![.., num_initial_quotients..num_consistency_quotients]);
    let consistency_quotient_zerofier_inverse =
        consistency_quotient_zerofier_inverse(quotient_domain, padded_height);
    fill_all_consistency_quotients(
        quotient_domain_master_base_table,
        quotient_domain_master_ext_table,
        &mut consistency_quotients,
        consistency_quotient_zerofier_inverse.view(),
        challenges,
    );
    prof_stop!(maybe_profiler, "consistency");

    prof_start!(maybe_profiler, "transition");
    let mut transition_quotients =
        all_quotients.slice_mut(s![.., num_consistency_quotients..num_transition_quotients]);
    let transition_quotient_zerofier_inverse = transition_quotient_zerofier_inverse(
        quotient_domain,
        trace_domain_generator,
        padded_height,
    );
    fill_all_transition_quotients(
        quotient_domain_master_base_table,
        quotient_domain_master_ext_table,
        &mut transition_quotients,
        transition_quotient_zerofier_inverse.view(),
        challenges,
        quotient_domain,
        padded_height,
    );
    prof_stop!(maybe_profiler, "transition");

    prof_start!(maybe_profiler, "terminal");
    let mut terminal_quot_table =
        all_quotients.slice_mut(s![.., num_transition_quotients..num_terminal_quotients]);
    let initial_quotient_zerofier_inverse =
        terminal_quotient_zerofier_inverse(quotient_domain, trace_domain_generator);
    fill_all_terminal_quotients(
        quotient_domain_master_base_table,
        quotient_domain_master_ext_table,
        &mut terminal_quot_table,
        initial_quotient_zerofier_inverse.view(),
        challenges,
    );
    prof_stop!(maybe_profiler, "terminal");

    all_quotients
}

pub fn evaluate_all_initial_constraints(
    base_row: ArrayView1<BFieldElement>,
    ext_row: ArrayView1<XFieldElement>,
    challenges: &AllChallenges,
) -> Vec<XFieldElement> {
    [
        ExtProgramTable::evaluate_initial_constraints(base_row, ext_row, challenges),
        ExtInstructionTable::evaluate_initial_constraints(base_row, ext_row, challenges),
        ExtProcessorTable::evaluate_initial_constraints(base_row, ext_row, challenges),
        ExtOpStackTable::evaluate_initial_constraints(base_row, ext_row, challenges),
        ExtRamTable::evaluate_initial_constraints(base_row, ext_row, challenges),
        ExtJumpStackTable::evaluate_initial_constraints(base_row, ext_row, challenges),
        ExtHashTable::evaluate_initial_constraints(base_row, ext_row, challenges),
    ]
    .concat()
}

pub fn evaluate_all_consistency_constraints(
    base_row: ArrayView1<BFieldElement>,
    ext_row: ArrayView1<XFieldElement>,
    challenges: &AllChallenges,
) -> Vec<XFieldElement> {
    [
        ExtProgramTable::evaluate_consistency_constraints(base_row, ext_row, challenges),
        ExtInstructionTable::evaluate_consistency_constraints(base_row, ext_row, challenges),
        ExtProcessorTable::evaluate_consistency_constraints(base_row, ext_row, challenges),
        ExtOpStackTable::evaluate_consistency_constraints(base_row, ext_row, challenges),
        ExtRamTable::evaluate_consistency_constraints(base_row, ext_row, challenges),
        ExtJumpStackTable::evaluate_consistency_constraints(base_row, ext_row, challenges),
        ExtHashTable::evaluate_consistency_constraints(base_row, ext_row, challenges),
    ]
    .concat()
}

pub fn evaluate_all_transition_constraints(
    current_base_row: ArrayView1<BFieldElement>,
    current_ext_row: ArrayView1<XFieldElement>,
    next_base_row: ArrayView1<BFieldElement>,
    next_ext_row: ArrayView1<XFieldElement>,
    challenges: &AllChallenges,
) -> Vec<XFieldElement> {
    let cbr = current_base_row;
    let cer = current_ext_row;
    let nbr = next_base_row;
    let ner = next_ext_row;
    [
        ExtProgramTable::evaluate_transition_constraints(cbr, cer, nbr, ner, challenges),
        ExtInstructionTable::evaluate_transition_constraints(cbr, cer, nbr, ner, challenges),
        ExtProcessorTable::evaluate_transition_constraints(cbr, cer, nbr, ner, challenges),
        ExtOpStackTable::evaluate_transition_constraints(cbr, cer, nbr, ner, challenges),
        ExtRamTable::evaluate_transition_constraints(cbr, cer, nbr, ner, challenges),
        ExtJumpStackTable::evaluate_transition_constraints(cbr, cer, nbr, ner, challenges),
        ExtHashTable::evaluate_transition_constraints(cbr, cer, nbr, ner, challenges),
    ]
    .concat()
}

pub fn evaluate_all_terminal_constraints(
    base_row: ArrayView1<BFieldElement>,
    ext_row: ArrayView1<XFieldElement>,
    challenges: &AllChallenges,
) -> Vec<XFieldElement> {
    [
        ExtProgramTable::evaluate_terminal_constraints(base_row, ext_row, challenges),
        ExtInstructionTable::evaluate_terminal_constraints(base_row, ext_row, challenges),
        ExtProcessorTable::evaluate_terminal_constraints(base_row, ext_row, challenges),
        ExtOpStackTable::evaluate_terminal_constraints(base_row, ext_row, challenges),
        ExtRamTable::evaluate_terminal_constraints(base_row, ext_row, challenges),
        ExtJumpStackTable::evaluate_terminal_constraints(base_row, ext_row, challenges),
        ExtHashTable::evaluate_terminal_constraints(base_row, ext_row, challenges),
    ]
    .concat()
}

pub fn evaluate_all_constraints(
    current_base_row: ArrayView1<BFieldElement>,
    current_ext_row: ArrayView1<XFieldElement>,
    next_base_row: ArrayView1<BFieldElement>,
    next_ext_row: ArrayView1<XFieldElement>,
    challenges: &AllChallenges,
) -> Vec<XFieldElement> {
    [
        evaluate_all_initial_constraints(current_base_row, current_ext_row, challenges),
        evaluate_all_consistency_constraints(current_base_row, current_ext_row, challenges),
        evaluate_all_transition_constraints(
            current_base_row,
            current_ext_row,
            next_base_row,
            next_ext_row,
            challenges,
        ),
        evaluate_all_terminal_constraints(current_base_row, current_ext_row, challenges),
    ]
    .concat()
}

/// Convert vector-of-arrays to vector-of-vectors.
fn to_vec_vecs<T: Sized + Clone, const S: usize>(vector_of_arrays: &[[T; S]]) -> Vec<Vec<T>> {
    vector_of_arrays
        .iter()
        .map(|arr| arr.to_vec())
        .collect_vec()
}

pub fn interpolant_degree(padded_height: usize, num_trace_randomizers: usize) -> Degree {
    let randomized_trace_length = roundup_npo2((padded_height + num_trace_randomizers) as u64);
    (randomized_trace_length - 1) as Degree
}

pub fn derive_trace_domain_generator(padded_height: u64) -> BFieldElement {
    debug_assert!(
        0 == padded_height || is_power_of_two(padded_height),
        "The padded height was: {}",
        padded_height
    );
    BFieldElement::primitive_root_of_unity(padded_height).unwrap()
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

    pub fn all_base_columns(&self) -> Vec<Vec<BFieldElement>> {
        self.into_iter()
            .map(|table| table.data().clone())
            .collect_vec()
            .concat()
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

    /// todo: Temporary method, to be replaced in issue #139
    pub fn with_data(
        padded_height: usize,
        base_codewords: Vec<Vec<BFieldElement>>,
        extension_codewords: Vec<Vec<XFieldElement>>,
    ) -> Self {
        let (base_program_table_data, base_codewords) =
            base_codewords.split_at(ProgramBaseTableColumn::COUNT);
        let (ext_program_table_data, extension_codewords) =
            extension_codewords.split_at(ProgramExtTableColumn::COUNT);
        let lifted_base_program_table_data: Vec<_> = base_program_table_data
            .into_par_iter()
            .map(|codeword| codeword.par_iter().map(|bfe| bfe.lift()).collect())
            .collect();
        let program_table_full_data = vec![
            lifted_base_program_table_data,
            ext_program_table_data.to_vec(),
        ]
        .concat();
        let program_table = ExtProgramTable::new(Table::new(
            ProgramBaseTableColumn::COUNT,
            ProgramBaseTableColumn::COUNT + ProgramExtTableColumn::COUNT,
            program_table_full_data,
            "ExtProgramTable over quotient domain".to_string(),
        ));

        let (base_instruction_table_data, base_codewords) =
            base_codewords.split_at(InstructionBaseTableColumn::COUNT);
        let (ext_instruction_table_data, extension_codewords) =
            extension_codewords.split_at(InstructionExtTableColumn::COUNT);
        let lifted_base_instruction_table_data: Vec<_> = base_instruction_table_data
            .into_par_iter()
            .map(|codeword| codeword.par_iter().map(|bfe| bfe.lift()).collect())
            .collect();
        let instruction_table_full_data = vec![
            lifted_base_instruction_table_data,
            ext_instruction_table_data.to_vec(),
        ]
        .concat();
        let instruction_table = ExtInstructionTable::new(Table::new(
            InstructionBaseTableColumn::COUNT,
            InstructionBaseTableColumn::COUNT + InstructionExtTableColumn::COUNT,
            instruction_table_full_data,
            "ExtInstructionTable over quotient domain".to_string(),
        ));

        let (base_processor_table_data, base_codewords) =
            base_codewords.split_at(ProcessorBaseTableColumn::COUNT);
        let (ext_processor_table_data, extension_codewords) =
            extension_codewords.split_at(ProcessorExtTableColumn::COUNT);
        let lifted_base_processor_table_data: Vec<_> = base_processor_table_data
            .into_par_iter()
            .map(|codeword| codeword.par_iter().map(|bfe| bfe.lift()).collect())
            .collect();
        let processor_table_full_data = vec![
            lifted_base_processor_table_data,
            ext_processor_table_data.to_vec(),
        ]
        .concat();
        let processor_table = ExtProcessorTable::new(Table::new(
            ProcessorBaseTableColumn::COUNT,
            ProcessorBaseTableColumn::COUNT + ProcessorExtTableColumn::COUNT,
            processor_table_full_data,
            "ExtProcessorTable over quotient domain".to_string(),
        ));

        let (base_op_stack_table_data, base_codewords) =
            base_codewords.split_at(OpStackBaseTableColumn::COUNT);
        let (ext_op_stack_table_data, extension_codewords) =
            extension_codewords.split_at(OpStackExtTableColumn::COUNT);
        let lifted_base_op_stack_table_data: Vec<_> = base_op_stack_table_data
            .into_par_iter()
            .map(|codeword| codeword.par_iter().map(|bfe| bfe.lift()).collect())
            .collect();
        let op_stack_table_full_data = vec![
            lifted_base_op_stack_table_data,
            ext_op_stack_table_data.to_vec(),
        ]
        .concat();
        let op_stack_table = ExtOpStackTable::new(Table::new(
            OpStackBaseTableColumn::COUNT,
            OpStackBaseTableColumn::COUNT + OpStackExtTableColumn::COUNT,
            op_stack_table_full_data,
            "ExtOpStackTable over quotient domain".to_string(),
        ));

        let (base_ram_table_data, base_codewords) =
            base_codewords.split_at(RamBaseTableColumn::COUNT);
        let (ext_ram_table_data, extension_codewords) =
            extension_codewords.split_at(RamExtTableColumn::COUNT);
        let lifted_base_ram_table_data: Vec<_> = base_ram_table_data
            .into_par_iter()
            .map(|codeword| codeword.par_iter().map(|bfe| bfe.lift()).collect())
            .collect();
        let ram_table_full_data =
            vec![lifted_base_ram_table_data, ext_ram_table_data.to_vec()].concat();
        let ram_table = ExtRamTable::new(Table::new(
            RamBaseTableColumn::COUNT,
            RamBaseTableColumn::COUNT + RamExtTableColumn::COUNT,
            ram_table_full_data,
            "ExtRamTable over quotient domain".to_string(),
        ));

        let (base_jump_stack_table_data, base_codewords) =
            base_codewords.split_at(JumpStackBaseTableColumn::COUNT);
        let (ext_jump_stack_table_data, extension_codewords) =
            extension_codewords.split_at(JumpStackExtTableColumn::COUNT);
        let lifted_base_jump_stack_table_data: Vec<_> = base_jump_stack_table_data
            .into_par_iter()
            .map(|codeword| codeword.par_iter().map(|bfe| bfe.lift()).collect())
            .collect();
        let jump_stack_table_full_data = vec![
            lifted_base_jump_stack_table_data,
            ext_jump_stack_table_data.to_vec(),
        ]
        .concat();
        let jump_stack_table = ExtJumpStackTable::new(Table::new(
            JumpStackBaseTableColumn::COUNT,
            JumpStackBaseTableColumn::COUNT + JumpStackExtTableColumn::COUNT,
            jump_stack_table_full_data,
            "ExtJumpStackTable over quotient domain".to_string(),
        ));

        let (base_hash_table_data, base_codewords) =
            base_codewords.split_at(HashBaseTableColumn::COUNT);
        let (ext_hash_table_data, extension_codewords) =
            extension_codewords.split_at(HashExtTableColumn::COUNT);
        let lifted_base_hash_table_data: Vec<_> = base_hash_table_data
            .into_par_iter()
            .map(|codeword| codeword.par_iter().map(|bfe| bfe.lift()).collect())
            .collect();
        let hash_table_full_data =
            vec![lifted_base_hash_table_data, ext_hash_table_data.to_vec()].concat();
        let hash_table = ExtHashTable::new(Table::new(
            HashBaseTableColumn::COUNT,
            HashBaseTableColumn::COUNT + HashExtTableColumn::COUNT,
            hash_table_full_data,
            "ExtHashTable over quotient domain".to_string(),
        ));

        assert!(base_codewords.is_empty());
        assert!(extension_codewords.is_empty());

        Self {
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

    pub fn for_verifier(padded_height: usize, maybe_profiler: &mut Option<TritonProfiler>) -> Self {
        prof_start!(maybe_profiler, "program table");
        let ext_program_table = ProgramTable::for_verifier();
        prof_stop!(maybe_profiler, "program table");
        prof_start!(maybe_profiler, "instruction table");
        let ext_instruction_table = InstructionTable::for_verifier();
        prof_stop!(maybe_profiler, "instruction table");
        prof_start!(maybe_profiler, "processor table");
        let ext_processor_table = ProcessorTable::for_verifier();
        prof_stop!(maybe_profiler, "processor table");
        prof_start!(maybe_profiler, "op stack table");
        let ext_op_stack_table = OpStackTable::for_verifier();
        prof_stop!(maybe_profiler, "op stack table");
        prof_start!(maybe_profiler, "ram table");
        let ext_ram_table = RamTable::for_verifier();
        prof_stop!(maybe_profiler, "ram table");
        prof_start!(maybe_profiler, "jump stack table");
        let ext_jump_stack_table = JumpStackTable::for_verifier();
        prof_stop!(maybe_profiler, "jump stack table");
        prof_start!(maybe_profiler, "hash table");
        let ext_hash_table = HashTable::for_verifier();
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

    /// Create an ExtTableCollection from a BaseTableCollection by `.extend()`ing each base table.
    /// The `.extend()` for each table is specific to that table, but always
    /// involves adding some number of columns. Each table only needs their
    /// own challenges, but `AllChallenges` are passed everywhere to keep each table's `.extend()`
    /// homogenous.
    pub fn extend_tables(
        base_tables: &BaseTableCollection,
        all_challenges: &AllChallenges,
    ) -> Self {
        let padded_height = base_tables.padded_height;
        let program_table = base_tables
            .program_table
            .extend(&all_challenges.program_table_challenges);
        let instruction_table = base_tables
            .instruction_table
            .extend(&all_challenges.instruction_table_challenges);
        let processor_table = base_tables
            .processor_table
            .extend(&all_challenges.processor_table_challenges);
        let op_stack_table = base_tables
            .op_stack_table
            .extend(&all_challenges.op_stack_table_challenges);
        let ram_table = base_tables
            .ram_table
            .extend(&all_challenges.ram_table_challenges);
        let jump_stack_table = base_tables
            .jump_stack_table
            .extend(&all_challenges.jump_stack_table_challenges);
        let hash_table = base_tables
            .hash_table
            .extend(&all_challenges.hash_table_challenges);

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

    /// todo – remove this once the master table is in place
    pub fn dummy_fri_domain_tables(padded_height: usize) -> Self {
        let program_table = ExtProgramTable::default();
        let instruction_table = ExtInstructionTable::default();
        let processor_table = ExtProcessorTable::default();
        let op_stack_table = ExtOpStackTable::default();
        let ram_table = ExtRamTable::default();
        let jump_stack_table = ExtJumpStackTable::default();
        let hash_table = ExtHashTable::default();
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
