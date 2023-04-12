use std::cmp::max;
use std::ops::MulAssign;

use itertools::Itertools;
use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use ndarray::s;
use ndarray::Array2;
use ndarray::ArrayView2;
use ndarray::ArrayViewMut2;
use ndarray::Zip;
use num_traits::One;
use rand::distributions::Standard;
use rand::prelude::Distribution;
use rand::random;
use strum::EnumCount;
use strum_macros::Display;
use strum_macros::EnumCount as EnumCountMacro;
use strum_macros::EnumIter;
use triton_profiler::prof_start;
use triton_profiler::prof_stop;
use triton_profiler::triton_profiler::TritonProfiler;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::mpolynomial::Degree;
use twenty_first::shared_math::other::is_power_of_two;
use twenty_first::shared_math::other::log_2_floor;
use twenty_first::shared_math::other::roundup_npo2;
use twenty_first::shared_math::polynomial::Polynomial;
use twenty_first::shared_math::traits::FiniteField;
use twenty_first::shared_math::traits::Inverse;
use twenty_first::shared_math::traits::ModPowU32;
use twenty_first::shared_math::traits::PrimitiveRootOfUnity;
use twenty_first::shared_math::x_field_element::XFieldElement;
use twenty_first::util_types::algebraic_hasher::AlgebraicHasher;
use twenty_first::util_types::merkle_tree::CpuParallel;
use twenty_first::util_types::merkle_tree::MerkleTree;
use twenty_first::util_types::merkle_tree_maker::MerkleTreeMaker;

use triton_opcodes::instruction::Instruction;

use crate::arithmetic_domain::ArithmeticDomain;
use crate::stark::StarkHasher;
use crate::table::cascade_table::CascadeTable;
use crate::table::cascade_table::ExtCascadeTable;
use crate::table::challenges::Challenges;
use crate::table::cross_table_argument::GrandCrossTableArg;
use crate::table::extension_table::DegreeWithOrigin;
use crate::table::extension_table::Evaluable;
use crate::table::extension_table::Quotientable;
use crate::table::hash_table::ExtHashTable;
use crate::table::hash_table::HashTable;
use crate::table::jump_stack_table::ExtJumpStackTable;
use crate::table::jump_stack_table::JumpStackTable;
use crate::table::lookup_table::ExtLookupTable;
use crate::table::lookup_table::LookupTable;
use crate::table::op_stack_table::ExtOpStackTable;
use crate::table::op_stack_table::OpStackTable;
use crate::table::processor_table::ExtProcessorTable;
use crate::table::processor_table::ProcessorTable;
use crate::table::program_table::ExtProgramTable;
use crate::table::program_table::ProgramTable;
use crate::table::ram_table::ExtRamTable;
use crate::table::ram_table::RamTable;
use crate::table::u32_table::ExtU32Table;
use crate::table::u32_table::U32Table;
use crate::table::*;
use crate::vm::AlgebraicExecutionTrace;

pub const NUM_TABLES: usize = TableId::COUNT;

pub const NUM_BASE_COLUMNS: usize = program_table::BASE_WIDTH
    + processor_table::BASE_WIDTH
    + op_stack_table::BASE_WIDTH
    + ram_table::BASE_WIDTH
    + jump_stack_table::BASE_WIDTH
    + hash_table::BASE_WIDTH
    + cascade_table::BASE_WIDTH
    + lookup_table::BASE_WIDTH
    + u32_table::BASE_WIDTH;
pub const NUM_EXT_COLUMNS: usize = program_table::EXT_WIDTH
    + processor_table::EXT_WIDTH
    + op_stack_table::EXT_WIDTH
    + ram_table::EXT_WIDTH
    + jump_stack_table::EXT_WIDTH
    + hash_table::EXT_WIDTH
    + cascade_table::EXT_WIDTH
    + lookup_table::EXT_WIDTH
    + u32_table::EXT_WIDTH;
pub const NUM_COLUMNS: usize = NUM_BASE_COLUMNS + NUM_EXT_COLUMNS;

pub const PROGRAM_TABLE_START: usize = 0;
pub const PROGRAM_TABLE_END: usize = PROGRAM_TABLE_START + program_table::BASE_WIDTH;
pub const PROCESSOR_TABLE_START: usize = PROGRAM_TABLE_END;
pub const PROCESSOR_TABLE_END: usize = PROCESSOR_TABLE_START + processor_table::BASE_WIDTH;
pub const OP_STACK_TABLE_START: usize = PROCESSOR_TABLE_END;
pub const OP_STACK_TABLE_END: usize = OP_STACK_TABLE_START + op_stack_table::BASE_WIDTH;
pub const RAM_TABLE_START: usize = OP_STACK_TABLE_END;
pub const RAM_TABLE_END: usize = RAM_TABLE_START + ram_table::BASE_WIDTH;
pub const JUMP_STACK_TABLE_START: usize = RAM_TABLE_END;
pub const JUMP_STACK_TABLE_END: usize = JUMP_STACK_TABLE_START + jump_stack_table::BASE_WIDTH;
pub const HASH_TABLE_START: usize = JUMP_STACK_TABLE_END;
pub const HASH_TABLE_END: usize = HASH_TABLE_START + hash_table::BASE_WIDTH;
pub const CASCADE_TABLE_START: usize = HASH_TABLE_END;
pub const CASCADE_TABLE_END: usize = CASCADE_TABLE_START + cascade_table::BASE_WIDTH;
pub const LOOKUP_TABLE_START: usize = CASCADE_TABLE_END;
pub const LOOKUP_TABLE_END: usize = LOOKUP_TABLE_START + lookup_table::BASE_WIDTH;
pub const U32_TABLE_START: usize = LOOKUP_TABLE_END;
pub const U32_TABLE_END: usize = U32_TABLE_START + u32_table::BASE_WIDTH;

pub const EXT_PROGRAM_TABLE_START: usize = 0;
pub const EXT_PROGRAM_TABLE_END: usize = EXT_PROGRAM_TABLE_START + program_table::EXT_WIDTH;
pub const EXT_PROCESSOR_TABLE_START: usize = EXT_PROGRAM_TABLE_END;
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
pub const EXT_CASCADE_TABLE_START: usize = EXT_HASH_TABLE_END;
pub const EXT_CASCADE_TABLE_END: usize = EXT_CASCADE_TABLE_START + cascade_table::EXT_WIDTH;
pub const EXT_LOOKUP_TABLE_START: usize = EXT_CASCADE_TABLE_END;
pub const EXT_LOOKUP_TABLE_END: usize = EXT_LOOKUP_TABLE_START + lookup_table::EXT_WIDTH;
pub const EXT_U32_TABLE_START: usize = EXT_LOOKUP_TABLE_END;
pub const EXT_U32_TABLE_END: usize = EXT_U32_TABLE_START + u32_table::EXT_WIDTH;

/// A `TableId` uniquely determines one of Triton VM's tables.
#[derive(Debug, Copy, Clone, Display, EnumCountMacro, EnumIter, PartialEq, Eq, Hash)]
pub enum TableId {
    ProgramTable,
    ProcessorTable,
    OpStackTable,
    RamTable,
    JumpStackTable,
    HashTable,
    CascadeTable,
    LookupTable,
    U32Table,
}

/// A Master Table is, in some sense, a top-level table of Triton VM. It contains all the data
/// but little logic beyond bookkeeping and presenting the data in a useful way. Conversely, the
/// individual tables contain no data but all of the respective logic. Master Tables are
/// responsible for managing the individual tables and for presenting the right data to the right
/// tables, serving as a clean interface between the VM and the individual tables.
///
/// As a mental model, it is perfectly fine to think of the data for the individual tables as
/// completely separate from each other. Only the cross-table argument links all tables together.
///
/// Conceptually, there are three Master Tables: the Master Base Table, the Master Extension
/// Table, and the Master Quotient Table. The lifecycle of the Master Tables is as follows:
/// 1. The Master Base Table is instantiated and filled using the Algebraic Execution Trace.
///     This is the first time a Master Base Table is instantiated. It is in column-major form.
/// 2. The Master Base Table is padded using logic from the individual tables.
/// 3. The still-empty entries in the Master Base Table are filled with random elements. This
///     step is also known as “trace randomization.”
/// 4. Each column of the Master Base Table is low-degree extended. The result is the Master Base
///     Table over the FRI domain. This is the second and last time a Master Base Table is
///     instantiated. It is in row-major form.
/// 5. The Master Base Table and the Master Base Table over the FRI domain are used to derive the
///     Master Extension Table using logic from the individual tables. This is the first time a
///     Master Extension Table is instantiated. It is in column-major form.
/// 6. The Master Extension Table is trace-randomized.
/// 7. Each column of the Master Extension Table is low-degree extended. The result is the Master
///     Extension Table over the FRI domain. This is the second and last time a Master Extension
///     Table is instantiated. It is in row-major form.
/// 8. Using the Master Base Table over the FRI domain and the Master Extension Table over the
///     FRI domain, the Quotient Master Table is derived using the AIR. Each individual table
///     defines that part of the AIR that is relevant to it.
///
/// The following points are of note:
/// - The Master Extension Table's rightmost columns are the randomizer codewords. These are
///     necessary for zero-knowledge.
/// - The terminal quotient of the cross-table argument, which links the individual tables together,
///     is also stored in the Master Quotient Table. Even though the cross-table argument is not
///     a table, it does define part of the AIR. Hence, the cross-table argument does not contribute
///     to padding or extending the Master Tables, but is incorporated when deriving the Master
///     Qoutient Table.
/// - For better performance, it is possible to derive the Master Quotient Table (step 8) from the
///     Master Base Table and Master Extension Table over a smaller domain than the FRI domain –
///     the “quotient domain.” The quotient domain is a subset of the FRI domain. This
///     performance improvement changes nothing conceptually.
pub trait MasterTable<FF>
where
    FF: FiniteField + MulAssign<BFieldElement>,
    Standard: Distribution<FF>,
{
    fn randomized_padded_trace_len(&self) -> usize;
    fn rand_trace_to_padded_trace_unit_distance(&self) -> usize;

    /// Presents underlying trace data, excluding trace randomizers. Makes little sense over the
    /// FRI domain.
    fn trace_table(&self) -> ArrayView2<FF>;

    /// Presents all underlying data.
    fn master_matrix(&self) -> ArrayView2<FF>;

    /// Presents all underlying data in a mutable manner.
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

    fn randomized_trace_domain(&self) -> ArithmeticDomain {
        let randomized_trace_domain_len = self.randomized_padded_trace_len();
        ArithmeticDomain::new_no_offset(randomized_trace_domain_len)
    }

    /// Low-degree extends all columns.
    /// Returns the thusly extended columns as well as the polynomials interpolating the columns.
    /// The number of rows in the resulting table is equal to the length of the FRI domain.
    /// The returned table is in row-major order.
    /// The interpolation polynomials can be used to compute out-of-domain rows, _e.g._, for DEEP.
    fn low_degree_extend_all_columns(&self) -> (Array2<FF>, Array1<Polynomial<FF>>)
    where
        Self: Sync,
    {
        let randomized_trace_domain = self.randomized_trace_domain();

        let num_rows = self.fri_domain().length;
        let num_columns = self.master_matrix().ncols();
        let mut interpolation_polynomials = Array1::zeros(num_columns);
        let mut extended_columns = Array2::zeros([num_rows, num_columns]);
        Zip::from(extended_columns.axis_iter_mut(Axis(1)))
            .and(self.master_matrix().axis_iter(Axis(1)))
            .and(interpolation_polynomials.axis_iter_mut(Axis(0)))
            .par_for_each(|lde_column, trace_column, poly| {
                let inter_poly = randomized_trace_domain.interpolate(&trace_column.to_vec());
                let fri_codeword = self.fri_domain().evaluate(&inter_poly);
                Array1::from(fri_codeword).move_into(lde_column);
                Array0::from_elem((), inter_poly).move_into(poly);
            });
        (extended_columns, interpolation_polynomials)
    }
}

#[derive(Clone)]
pub struct MasterBaseTable {
    pub padded_height: usize,
    pub num_trace_randomizers: usize,

    pub program_len: usize,
    pub main_execution_len: usize,
    pub hash_coprocessor_execution_len: usize,
    pub cascade_table_len: usize,
    pub u32_coprocesor_execution_len: usize,

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
    pub num_randomizer_polynomials: usize,

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

    fn trace_table(&self) -> ArrayView2<BFieldElement> {
        self.master_base_matrix
            .slice(s![..; self.rand_trace_to_padded_trace_unit_distance, ..])
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

    fn trace_table(&self) -> ArrayView2<XFieldElement> {
        self.master_ext_matrix
            .slice(s![..; self.rand_trace_to_padded_trace_unit_distance, ..])
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
    pub fn padded_height(aet: &AlgebraicExecutionTrace) -> usize {
        let max_height = [
            // The Program Table's side of the instruction lookup argument requires at least one
            // padding row to account for the processor's “next instruction or argument.”
            Self::program_table_length(aet) + 1,
            Self::processor_table_length(aet),
            Self::hash_table_length(aet),
            Self::cascade_table_length(aet),
            Self::lookup_table_length(),
            Self::u32_table_length(aet),
        ]
        .iter()
        .max()
        .unwrap()
        .to_owned();
        roundup_npo2(max_height as u64) as usize
    }

    pub fn program_table_length(aet: &AlgebraicExecutionTrace) -> usize {
        aet.program.len_bwords()
    }

    pub fn processor_table_length(aet: &AlgebraicExecutionTrace) -> usize {
        aet.processor_trace.nrows()
    }

    pub fn hash_table_length(aet: &AlgebraicExecutionTrace) -> usize {
        aet.sponge_trace.nrows() + aet.hash_trace.nrows()
    }

    pub fn cascade_table_length(aet: &AlgebraicExecutionTrace) -> usize {
        aet.cascade_table_lookup_multiplicities.len()
    }

    pub fn lookup_table_length() -> usize {
        1 << 8
    }

    pub fn u32_table_length(aet: &AlgebraicExecutionTrace) -> usize {
        aet.u32_entries
            .keys()
            .map(|(instruction, lhs, rhs)| match instruction {
                // for instruction `pow`, the left-hand side doesn't change between rows
                Instruction::Pow => rhs.value(),
                _ => max(lhs.value(), rhs.value()),
            })
            .map(|relevant_entry| match relevant_entry == 0 {
                true => 1,
                false => 2 + log_2_floor(relevant_entry as u128) as usize,
            })
            .sum()
    }

    pub fn new(
        aet: AlgebraicExecutionTrace,
        num_trace_randomizers: usize,
        fri_domain: ArithmeticDomain,
    ) -> Self {
        let padded_height = Self::padded_height(&aet);
        let randomized_padded_trace_len =
            randomized_padded_trace_len(num_trace_randomizers, padded_height);
        let unit_distance = randomized_padded_trace_len / padded_height;

        let num_rows = randomized_padded_trace_len;
        let num_columns = NUM_BASE_COLUMNS;
        let master_base_matrix = Array2::zeros([num_rows, num_columns].f());

        let mut master_base_table = Self {
            padded_height,
            num_trace_randomizers,
            program_len: Self::program_table_length(&aet),
            main_execution_len: Self::processor_table_length(&aet),
            hash_coprocessor_execution_len: Self::hash_table_length(&aet),
            cascade_table_len: Self::cascade_table_length(&aet),
            u32_coprocesor_execution_len: Self::u32_table_length(&aet),
            randomized_padded_trace_len,
            rand_trace_to_padded_trace_unit_distance: unit_distance,
            fri_domain,
            master_base_matrix,
        };

        let program_table = &mut master_base_table.table_mut(TableId::ProgramTable);
        ProgramTable::fill_trace(program_table, &aet);
        let op_stack_table = &mut master_base_table.table_mut(TableId::OpStackTable);
        let clk_jump_diffs_op_stack = OpStackTable::fill_trace(op_stack_table, &aet);
        let ram_table = &mut master_base_table.table_mut(TableId::RamTable);
        let clk_jump_diffs_ram = RamTable::fill_trace(ram_table, &aet);
        let jump_stack_table = &mut master_base_table.table_mut(TableId::JumpStackTable);
        let clk_jump_diffs_jump_stack = JumpStackTable::fill_trace(jump_stack_table, &aet);
        let hash_table = &mut master_base_table.table_mut(TableId::HashTable);
        HashTable::fill_trace(hash_table, &aet);
        let cascade_table = &mut master_base_table.table_mut(TableId::CascadeTable);
        CascadeTable::fill_trace(cascade_table, &aet);
        let lookup_table = &mut master_base_table.table_mut(TableId::LookupTable);
        LookupTable::fill_trace(lookup_table, &aet);
        let u32_table = &mut master_base_table.table_mut(TableId::U32Table);
        U32Table::fill_trace(u32_table, &aet);

        // memory-like tables must be filled in before clock jump differences are known, hence
        // the break from the usual order
        let processor_table = &mut master_base_table.table_mut(TableId::ProcessorTable);
        ProcessorTable::fill_trace(
            processor_table,
            &aet,
            &clk_jump_diffs_op_stack,
            &clk_jump_diffs_ram,
            &clk_jump_diffs_jump_stack,
        );

        master_base_table
    }

    pub fn pad(&mut self) {
        let program_len = self.program_len;
        let main_execution_len = self.main_execution_len;
        let hash_coprocessor_execution_len = self.hash_coprocessor_execution_len;
        let cascade_table_len = self.cascade_table_len;
        let u32_table_len = self.u32_coprocesor_execution_len;

        let program_table = &mut self.table_mut(TableId::ProgramTable);
        ProgramTable::pad_trace(program_table, program_len);
        let processor_table = &mut self.table_mut(TableId::ProcessorTable);
        ProcessorTable::pad_trace(processor_table, main_execution_len);
        let op_stack_table = &mut self.table_mut(TableId::OpStackTable);
        OpStackTable::pad_trace(op_stack_table, main_execution_len);
        let ram_table = &mut self.table_mut(TableId::RamTable);
        RamTable::pad_trace(ram_table, main_execution_len);
        let jump_stack_table = &mut self.table_mut(TableId::JumpStackTable);
        JumpStackTable::pad_trace(jump_stack_table, main_execution_len);
        let hash_table = &mut self.table_mut(TableId::HashTable);
        HashTable::pad_trace(hash_table, hash_coprocessor_execution_len);
        let cascade_table = &mut self.table_mut(TableId::CascadeTable);
        CascadeTable::pad_trace(cascade_table, cascade_table_len);
        let lookup_table = &mut self.table_mut(TableId::LookupTable);
        LookupTable::pad_trace(lookup_table);
        let u32_table = &mut self.table_mut(TableId::U32Table);
        U32Table::pad_trace(u32_table, u32_table_len);
    }

    /// Returns the low-degree extended columns as well as the columns' interpolation polynomials.
    /// The polynomials are in the same order as the columns.
    /// The interpolation polynomials can be used to compute out-of-domain rows, _e.g._, for DEEP.
    pub fn to_fri_domain_table(&self) -> (Self, Array1<Polynomial<BFieldElement>>) {
        let (master_base_matrix, interpolation_polynomials) = self.low_degree_extend_all_columns();
        let master_base_table = Self {
            master_base_matrix,
            ..*self
        };
        (master_base_table, interpolation_polynomials)
    }

    pub fn merkle_tree(
        &self,
        maybe_profiler: &mut Option<TritonProfiler>,
    ) -> MerkleTree<StarkHasher, CpuParallel> {
        prof_start!(maybe_profiler, "leafs");
        let hashed_rows = self
            .master_base_matrix
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|row| StarkHasher::hash_varlen(&row.to_vec()))
            .collect::<Vec<_>>();
        prof_stop!(maybe_profiler, "leafs");
        prof_start!(maybe_profiler, "Merkle tree");
        let ret = CpuParallel::from_digests(&hashed_rows);
        prof_stop!(maybe_profiler, "Merkle tree");

        ret
    }

    /// Create a `MasterExtTable` from a `MasterBaseTable` by `.extend()`ing each individual base
    /// table. The `.extend()` for each table is specific to that table, but always involves
    /// adding some number of columns.
    pub fn extend(
        &self,
        challenges: &Challenges,
        num_randomizer_polynomials: usize,
    ) -> MasterExtTable {
        // randomizer polynomials
        let num_rows = self.master_base_matrix.nrows();
        let num_columns = NUM_EXT_COLUMNS + num_randomizer_polynomials;
        let mut master_ext_matrix = Array2::zeros([num_rows, num_columns].f());
        master_ext_matrix
            .slice_mut(s![.., NUM_EXT_COLUMNS..])
            .par_mapv_inplace(|_| random::<XFieldElement>());

        let mut master_ext_table = MasterExtTable {
            padded_height: self.padded_height,
            num_trace_randomizers: self.num_trace_randomizers,
            num_randomizer_polynomials,
            randomized_padded_trace_len: self.randomized_padded_trace_len,
            rand_trace_to_padded_trace_unit_distance: self.rand_trace_to_padded_trace_unit_distance,
            fri_domain: self.fri_domain,
            master_ext_matrix,
        };

        ProgramTable::extend(
            self.table(TableId::ProgramTable),
            master_ext_table.table_mut(TableId::ProgramTable),
            challenges,
        );
        ProcessorTable::extend(
            self.table(TableId::ProcessorTable),
            master_ext_table.table_mut(TableId::ProcessorTable),
            challenges,
        );
        OpStackTable::extend(
            self.table(TableId::OpStackTable),
            master_ext_table.table_mut(TableId::OpStackTable),
            challenges,
        );
        RamTable::extend(
            self.table(TableId::RamTable),
            master_ext_table.table_mut(TableId::RamTable),
            challenges,
        );
        JumpStackTable::extend(
            self.table(TableId::JumpStackTable),
            master_ext_table.table_mut(TableId::JumpStackTable),
            challenges,
        );
        HashTable::extend(
            self.table(TableId::HashTable),
            master_ext_table.table_mut(TableId::HashTable),
            challenges,
        );
        CascadeTable::extend(
            self.table(TableId::CascadeTable),
            master_ext_table.table_mut(TableId::CascadeTable),
            challenges,
        );
        LookupTable::extend(
            self.table(TableId::LookupTable),
            master_ext_table.table_mut(TableId::LookupTable),
            challenges,
        );
        U32Table::extend(
            self.table(TableId::U32Table),
            master_ext_table.table_mut(TableId::U32Table),
            challenges,
        );

        master_ext_table
    }

    fn table_slice_info(id: TableId) -> (usize, usize) {
        use TableId::*;
        match id {
            ProgramTable => (PROGRAM_TABLE_START, PROGRAM_TABLE_END),
            ProcessorTable => (PROCESSOR_TABLE_START, PROCESSOR_TABLE_END),
            OpStackTable => (OP_STACK_TABLE_START, OP_STACK_TABLE_END),
            RamTable => (RAM_TABLE_START, RAM_TABLE_END),
            JumpStackTable => (JUMP_STACK_TABLE_START, JUMP_STACK_TABLE_END),
            HashTable => (HASH_TABLE_START, HASH_TABLE_END),
            CascadeTable => (CASCADE_TABLE_START, CASCADE_TABLE_END),
            LookupTable => (LOOKUP_TABLE_START, LOOKUP_TABLE_END),
            U32Table => (U32_TABLE_START, U32_TABLE_END),
        }
    }

    pub fn table(&self, id: TableId) -> ArrayView2<BFieldElement> {
        let (table_start, table_end) = Self::table_slice_info(id);
        let unit_distance = self.rand_trace_to_padded_trace_unit_distance;
        self.master_base_matrix
            .slice(s![..; unit_distance, table_start..table_end])
    }

    pub fn table_mut(&mut self, id: TableId) -> ArrayViewMut2<BFieldElement> {
        let (table_start, table_end) = Self::table_slice_info(id);
        let unit_distance = self.rand_trace_to_padded_trace_unit_distance;
        self.master_base_matrix
            .slice_mut(s![..; unit_distance, table_start..table_end])
    }
}

impl MasterExtTable {
    /// Returns the low-degree extended columns as well as the columns' interpolation polynomials.
    /// The polynomials are in the same order as the columns.
    /// The interpolation polynomials can be used to compute out-of-domain rows, _e.g._, for DEEP.
    pub fn to_fri_domain_table(&self) -> (Self, Array1<Polynomial<XFieldElement>>) {
        let (master_ext_matrix, interpolation_polynomials) = self.low_degree_extend_all_columns();
        let master_ext_table = Self {
            master_ext_matrix,
            ..*self
        };
        (master_ext_table, interpolation_polynomials)
    }

    pub fn randomizer_polynomials(&self) -> Vec<Array1<XFieldElement>> {
        let mut randomizer_polynomials = Vec::with_capacity(self.num_randomizer_polynomials);
        for col_idx in NUM_EXT_COLUMNS..self.master_ext_matrix.ncols() {
            let randomizer_polynomial = self.master_ext_matrix.column(col_idx);
            randomizer_polynomials.push(randomizer_polynomial.to_owned());
        }
        randomizer_polynomials
    }

    pub fn merkle_tree(
        &self,
        maybe_profiler: &mut Option<TritonProfiler>,
    ) -> MerkleTree<StarkHasher, CpuParallel> {
        prof_start!(maybe_profiler, "leafs");
        let hashed_rows = self
            .master_ext_matrix
            .axis_iter(Axis(0))
            .into_par_iter()
            .map(|row| {
                let contiguous_row_bfe = row
                    .to_vec()
                    .iter()
                    .map(|xfe| xfe.coefficients.to_vec())
                    .concat();
                StarkHasher::hash_varlen(&contiguous_row_bfe)
            })
            .collect::<Vec<_>>();
        prof_stop!(maybe_profiler, "leafs");
        prof_start!(maybe_profiler, "Merkle tree");
        let ret = CpuParallel::from_digests(&hashed_rows);
        prof_stop!(maybe_profiler, "Merkle tree");

        ret
    }

    fn table_slice_info(id: TableId) -> (usize, usize) {
        use TableId::*;
        match id {
            ProgramTable => (EXT_PROGRAM_TABLE_START, EXT_PROGRAM_TABLE_END),
            ProcessorTable => (EXT_PROCESSOR_TABLE_START, EXT_PROCESSOR_TABLE_END),
            OpStackTable => (EXT_OP_STACK_TABLE_START, EXT_OP_STACK_TABLE_END),
            RamTable => (EXT_RAM_TABLE_START, EXT_RAM_TABLE_END),
            JumpStackTable => (EXT_JUMP_STACK_TABLE_START, EXT_JUMP_STACK_TABLE_END),
            HashTable => (EXT_HASH_TABLE_START, EXT_HASH_TABLE_END),
            CascadeTable => (EXT_CASCADE_TABLE_START, EXT_CASCADE_TABLE_END),
            LookupTable => (EXT_LOOKUP_TABLE_START, EXT_LOOKUP_TABLE_END),
            U32Table => (EXT_U32_TABLE_START, EXT_U32_TABLE_END),
        }
    }

    pub fn table(&self, id: TableId) -> ArrayView2<XFieldElement> {
        let unit_distance = self.rand_trace_to_padded_trace_unit_distance;
        let (table_start, table_end) = Self::table_slice_info(id);
        self.master_ext_matrix
            .slice(s![..; unit_distance, table_start..table_end])
    }

    pub fn table_mut(&mut self, id: TableId) -> ArrayViewMut2<XFieldElement> {
        let unit_distance = self.rand_trace_to_padded_trace_unit_distance;
        let (table_start, table_end) = Self::table_slice_info(id);
        self.master_ext_matrix
            .slice_mut(s![..; unit_distance, table_start..table_end])
    }
}

pub fn all_degrees_with_origin(
    interpolant_degree: Degree,
    padded_height: usize,
) -> Vec<DegreeWithOrigin> {
    let id = interpolant_degree;
    let ph = padded_height;
    [
        ExtProgramTable::all_degrees_with_origin("program table", id, ph),
        ExtProcessorTable::all_degrees_with_origin("processor table", id, ph),
        ExtOpStackTable::all_degrees_with_origin("op stack table", id, ph),
        ExtRamTable::all_degrees_with_origin("ram table", id, ph),
        ExtJumpStackTable::all_degrees_with_origin("jump stack table", id, ph),
        ExtHashTable::all_degrees_with_origin("hash table", id, ph),
        ExtCascadeTable::all_degrees_with_origin("cascade table", id, ph),
        ExtLookupTable::all_degrees_with_origin("lookup table", id, ph),
        ExtU32Table::all_degrees_with_origin("u32 table", id, ph),
    ]
    .concat()
}

pub fn max_degree_with_origin(
    interpolant_degree: Degree,
    padded_height: usize,
) -> DegreeWithOrigin {
    all_degrees_with_origin(interpolant_degree, padded_height)
        .into_iter()
        .max()
        .unwrap_or_default()
}

pub fn num_all_table_quotients() -> usize {
    num_all_initial_quotients()
        + num_all_consistency_quotients()
        + num_all_transition_quotients()
        + num_all_terminal_quotients()
}

pub fn num_all_initial_quotients() -> usize {
    ExtProgramTable::num_initial_quotients()
        + ExtProcessorTable::num_initial_quotients()
        + ExtOpStackTable::num_initial_quotients()
        + ExtRamTable::num_initial_quotients()
        + ExtJumpStackTable::num_initial_quotients()
        + ExtHashTable::num_initial_quotients()
        + ExtCascadeTable::num_initial_quotients()
        + ExtLookupTable::num_initial_quotients()
        + ExtU32Table::num_initial_quotients()
}

pub fn num_all_consistency_quotients() -> usize {
    ExtProgramTable::num_consistency_quotients()
        + ExtProcessorTable::num_consistency_quotients()
        + ExtOpStackTable::num_consistency_quotients()
        + ExtRamTable::num_consistency_quotients()
        + ExtJumpStackTable::num_consistency_quotients()
        + ExtHashTable::num_consistency_quotients()
        + ExtCascadeTable::num_consistency_quotients()
        + ExtLookupTable::num_consistency_quotients()
        + ExtU32Table::num_consistency_quotients()
}

pub fn num_all_transition_quotients() -> usize {
    ExtProgramTable::num_transition_quotients()
        + ExtProcessorTable::num_transition_quotients()
        + ExtOpStackTable::num_transition_quotients()
        + ExtRamTable::num_transition_quotients()
        + ExtJumpStackTable::num_transition_quotients()
        + ExtHashTable::num_transition_quotients()
        + ExtCascadeTable::num_transition_quotients()
        + ExtLookupTable::num_transition_quotients()
        + ExtU32Table::num_transition_quotients()
}

pub fn num_all_terminal_quotients() -> usize {
    ExtProgramTable::num_terminal_quotients()
        + ExtProcessorTable::num_terminal_quotients()
        + ExtOpStackTable::num_terminal_quotients()
        + ExtRamTable::num_terminal_quotients()
        + ExtJumpStackTable::num_terminal_quotients()
        + ExtHashTable::num_terminal_quotients()
        + ExtCascadeTable::num_terminal_quotients()
        + ExtLookupTable::num_terminal_quotients()
        + ExtU32Table::num_terminal_quotients()
        + GrandCrossTableArg::num_terminal_quotients()
}

pub fn all_initial_quotient_degree_bounds(interpolant_degree: Degree) -> Vec<Degree> {
    [
        ExtProgramTable::initial_quotient_degree_bounds(interpolant_degree),
        ExtProcessorTable::initial_quotient_degree_bounds(interpolant_degree),
        ExtOpStackTable::initial_quotient_degree_bounds(interpolant_degree),
        ExtRamTable::initial_quotient_degree_bounds(interpolant_degree),
        ExtJumpStackTable::initial_quotient_degree_bounds(interpolant_degree),
        ExtHashTable::initial_quotient_degree_bounds(interpolant_degree),
        ExtCascadeTable::initial_quotient_degree_bounds(interpolant_degree),
        ExtLookupTable::initial_quotient_degree_bounds(interpolant_degree),
        ExtU32Table::initial_quotient_degree_bounds(interpolant_degree),
    ]
    .concat()
}

pub fn all_consistency_quotient_degree_bounds(
    interpolant_degree: Degree,
    padded_height: usize,
) -> Vec<Degree> {
    [
        ExtProgramTable::consistency_quotient_degree_bounds(interpolant_degree, padded_height),
        ExtProcessorTable::consistency_quotient_degree_bounds(interpolant_degree, padded_height),
        ExtOpStackTable::consistency_quotient_degree_bounds(interpolant_degree, padded_height),
        ExtRamTable::consistency_quotient_degree_bounds(interpolant_degree, padded_height),
        ExtJumpStackTable::consistency_quotient_degree_bounds(interpolant_degree, padded_height),
        ExtHashTable::consistency_quotient_degree_bounds(interpolant_degree, padded_height),
        ExtCascadeTable::consistency_quotient_degree_bounds(interpolant_degree, padded_height),
        ExtLookupTable::consistency_quotient_degree_bounds(interpolant_degree, padded_height),
        ExtU32Table::consistency_quotient_degree_bounds(interpolant_degree, padded_height),
    ]
    .concat()
}

pub fn all_transition_quotient_degree_bounds(
    interpolant_degree: Degree,
    padded_height: usize,
) -> Vec<Degree> {
    [
        ExtProgramTable::transition_quotient_degree_bounds(interpolant_degree, padded_height),
        ExtProcessorTable::transition_quotient_degree_bounds(interpolant_degree, padded_height),
        ExtOpStackTable::transition_quotient_degree_bounds(interpolant_degree, padded_height),
        ExtRamTable::transition_quotient_degree_bounds(interpolant_degree, padded_height),
        ExtJumpStackTable::transition_quotient_degree_bounds(interpolant_degree, padded_height),
        ExtHashTable::transition_quotient_degree_bounds(interpolant_degree, padded_height),
        ExtCascadeTable::transition_quotient_degree_bounds(interpolant_degree, padded_height),
        ExtLookupTable::transition_quotient_degree_bounds(interpolant_degree, padded_height),
        ExtU32Table::transition_quotient_degree_bounds(interpolant_degree, padded_height),
    ]
    .concat()
}

pub fn all_terminal_quotient_degree_bounds(interpolant_degree: Degree) -> Vec<Degree> {
    [
        ExtProgramTable::terminal_quotient_degree_bounds(interpolant_degree),
        ExtProcessorTable::terminal_quotient_degree_bounds(interpolant_degree),
        ExtOpStackTable::terminal_quotient_degree_bounds(interpolant_degree),
        ExtRamTable::terminal_quotient_degree_bounds(interpolant_degree),
        ExtJumpStackTable::terminal_quotient_degree_bounds(interpolant_degree),
        ExtHashTable::terminal_quotient_degree_bounds(interpolant_degree),
        ExtCascadeTable::terminal_quotient_degree_bounds(interpolant_degree),
        ExtLookupTable::terminal_quotient_degree_bounds(interpolant_degree),
        ExtU32Table::terminal_quotient_degree_bounds(interpolant_degree),
        GrandCrossTableArg::terminal_quotient_degree_bounds(interpolant_degree),
    ]
    .concat()
}

pub fn all_quotient_degree_bounds(interpolant_degree: Degree, padded_height: usize) -> Vec<Degree> {
    [
        all_initial_quotient_degree_bounds(interpolant_degree),
        all_consistency_quotient_degree_bounds(interpolant_degree, padded_height),
        all_transition_quotient_degree_bounds(interpolant_degree, padded_height),
        all_terminal_quotient_degree_bounds(interpolant_degree),
    ]
    .concat()
}

pub fn initial_quotient_zerofier_inverse(
    quotient_domain: ArithmeticDomain,
) -> Array1<BFieldElement> {
    let zerofier_codeword = quotient_domain
        .domain_values()
        .into_iter()
        .map(|x| x - BFieldElement::one())
        .collect();
    BFieldElement::batch_inversion(zerofier_codeword).into()
}

pub fn consistency_quotient_zerofier_inverse(
    trace_domain: ArithmeticDomain,
    quotient_domain: ArithmeticDomain,
) -> Array1<BFieldElement> {
    let zerofier_codeword = quotient_domain
        .domain_values()
        .iter()
        .map(|x| x.mod_pow_u32(trace_domain.length as u32) - BFieldElement::one())
        .collect();
    BFieldElement::batch_inversion(zerofier_codeword).into()
}

pub fn transition_quotient_zerofier_inverse(
    trace_domain: ArithmeticDomain,
    quotient_domain: ArithmeticDomain,
) -> Array1<BFieldElement> {
    let one = BFieldElement::one();
    let trace_domain_generator_inverse = trace_domain.generator.inverse();
    let quotient_domain_values = quotient_domain.domain_values();

    let subgroup_zerofier: Vec<_> = quotient_domain_values
        .par_iter()
        .map(|domain_value| domain_value.mod_pow_u32(trace_domain.length as u32) - one)
        .collect();
    let subgroup_zerofier_inverse = BFieldElement::batch_inversion(subgroup_zerofier);
    let zerofier_inverse: Vec<_> = quotient_domain_values
        .into_par_iter()
        .zip_eq(subgroup_zerofier_inverse.into_par_iter())
        .map(|(domain_value, sub_z_inv)| {
            (domain_value - trace_domain_generator_inverse) * sub_z_inv
        })
        .collect();
    zerofier_inverse.into()
}

pub fn terminal_quotient_zerofier_inverse(
    trace_domain: ArithmeticDomain,
    quotient_domain: ArithmeticDomain,
) -> Array1<BFieldElement> {
    // The zerofier for the terminal quotient has a root in the last
    // value in the cyclical group generated from the trace domain's generator.
    let trace_domain_generator_inverse = trace_domain.generator.inverse();
    let zerofier_codeword = quotient_domain
        .domain_values()
        .into_iter()
        .map(|x| x - trace_domain_generator_inverse)
        .collect_vec();
    BFieldElement::batch_inversion(zerofier_codeword).into()
}

pub fn fill_all_initial_quotients(
    master_base_table: ArrayView2<BFieldElement>,
    master_ext_table: ArrayView2<XFieldElement>,
    quot_table: &mut ArrayViewMut2<XFieldElement>,
    zerofier_inverse: ArrayView1<BFieldElement>,
    challenges: &Challenges,
) {
    // The order of the quotient tables is not actually important. However, it must be consistent
    // between prover and verifier, and the shapes must check out.
    let program_section_start = 0;
    let program_section_end = program_section_start + ExtProgramTable::num_initial_quotients();
    let processor_section_start = program_section_end;
    let processor_section_end =
        processor_section_start + ExtProcessorTable::num_initial_quotients();
    let op_stack_section_start = processor_section_end;
    let op_stack_section_end = op_stack_section_start + ExtOpStackTable::num_initial_quotients();
    let ram_section_start = op_stack_section_end;
    let ram_section_end = ram_section_start + ExtRamTable::num_initial_quotients();
    let jump_stack_section_start = ram_section_end;
    let jump_stack_section_end =
        jump_stack_section_start + ExtJumpStackTable::num_initial_quotients();
    let hash_section_start = jump_stack_section_end;
    let hash_section_end = hash_section_start + ExtHashTable::num_initial_quotients();
    let cascade_section_start = hash_section_end;
    let cascade_section_end = cascade_section_start + ExtCascadeTable::num_initial_quotients();
    let lookup_section_start = cascade_section_end;
    let lookup_section_end = lookup_section_start + ExtLookupTable::num_initial_quotients();
    let u32_section_start = lookup_section_end;
    let u32_section_end = u32_section_start + ExtU32Table::num_initial_quotients();

    let mut program_quot_table =
        quot_table.slice_mut(s![.., program_section_start..program_section_end]);
    ExtProgramTable::fill_initial_quotients(
        master_base_table,
        master_ext_table,
        &mut program_quot_table,
        zerofier_inverse,
        challenges,
    );
    let mut processor_quot_table =
        quot_table.slice_mut(s![.., processor_section_start..processor_section_end]);
    ExtProcessorTable::fill_initial_quotients(
        master_base_table,
        master_ext_table,
        &mut processor_quot_table,
        zerofier_inverse,
        challenges,
    );
    let mut op_stack_quot_table =
        quot_table.slice_mut(s![.., op_stack_section_start..op_stack_section_end]);
    ExtOpStackTable::fill_initial_quotients(
        master_base_table,
        master_ext_table,
        &mut op_stack_quot_table,
        zerofier_inverse,
        challenges,
    );
    let mut ram_quot_table = quot_table.slice_mut(s![.., ram_section_start..ram_section_end]);
    ExtRamTable::fill_initial_quotients(
        master_base_table,
        master_ext_table,
        &mut ram_quot_table,
        zerofier_inverse,
        challenges,
    );
    let mut jump_stack_quot_table =
        quot_table.slice_mut(s![.., jump_stack_section_start..jump_stack_section_end]);
    ExtJumpStackTable::fill_initial_quotients(
        master_base_table,
        master_ext_table,
        &mut jump_stack_quot_table,
        zerofier_inverse,
        challenges,
    );
    let mut hash_quot_table = quot_table.slice_mut(s![.., hash_section_start..hash_section_end]);
    ExtHashTable::fill_initial_quotients(
        master_base_table,
        master_ext_table,
        &mut hash_quot_table,
        zerofier_inverse,
        challenges,
    );
    let mut cascade_quot_table =
        quot_table.slice_mut(s![.., cascade_section_start..cascade_section_end]);
    ExtCascadeTable::fill_initial_quotients(
        master_base_table,
        master_ext_table,
        &mut cascade_quot_table,
        zerofier_inverse,
        challenges,
    );
    let mut lookup_quot_table =
        quot_table.slice_mut(s![.., lookup_section_start..lookup_section_end]);
    ExtLookupTable::fill_initial_quotients(
        master_base_table,
        master_ext_table,
        &mut lookup_quot_table,
        zerofier_inverse,
        challenges,
    );
    let mut u32_quot_table = quot_table.slice_mut(s![.., u32_section_start..u32_section_end]);
    ExtU32Table::fill_initial_quotients(
        master_base_table,
        master_ext_table,
        &mut u32_quot_table,
        zerofier_inverse,
        challenges,
    );
}

pub fn fill_all_consistency_quotients(
    master_base_table: ArrayView2<BFieldElement>,
    master_ext_table: ArrayView2<XFieldElement>,
    quot_table: &mut ArrayViewMut2<XFieldElement>,
    zerofier_inverse: ArrayView1<BFieldElement>,
    challenges: &Challenges,
) {
    // The order of the quotient tables is not actually important. However, it must be consistent
    // between prover and verifier, and the shapes must check out.
    let program_section_start = 0;
    let program_section_end = program_section_start + ExtProgramTable::num_consistency_quotients();
    let processor_section_start = program_section_end;
    let processor_section_end =
        processor_section_start + ExtProcessorTable::num_consistency_quotients();
    let op_stack_section_start = processor_section_end;
    let op_stack_section_end =
        op_stack_section_start + ExtOpStackTable::num_consistency_quotients();
    let ram_section_start = op_stack_section_end;
    let ram_section_end = ram_section_start + ExtRamTable::num_consistency_quotients();
    let jump_stack_section_start = ram_section_end;
    let jump_stack_section_end =
        jump_stack_section_start + ExtJumpStackTable::num_consistency_quotients();
    let hash_section_start = jump_stack_section_end;
    let hash_section_end = hash_section_start + ExtHashTable::num_consistency_quotients();
    let cascade_section_start = hash_section_end;
    let cascade_section_end = cascade_section_start + ExtCascadeTable::num_consistency_quotients();
    let lookup_section_start = cascade_section_end;
    let lookup_section_end = lookup_section_start + ExtLookupTable::num_consistency_quotients();
    let u32_section_start = lookup_section_end;
    let u32_section_end = u32_section_start + ExtU32Table::num_consistency_quotients();

    let mut program_quot_table =
        quot_table.slice_mut(s![.., program_section_start..program_section_end]);
    ExtProgramTable::fill_consistency_quotients(
        master_base_table,
        master_ext_table,
        &mut program_quot_table,
        zerofier_inverse,
        challenges,
    );
    let mut processor_quot_table =
        quot_table.slice_mut(s![.., processor_section_start..processor_section_end]);
    ExtProcessorTable::fill_consistency_quotients(
        master_base_table,
        master_ext_table,
        &mut processor_quot_table,
        zerofier_inverse,
        challenges,
    );
    let mut op_stack_quot_table =
        quot_table.slice_mut(s![.., op_stack_section_start..op_stack_section_end]);
    ExtOpStackTable::fill_consistency_quotients(
        master_base_table,
        master_ext_table,
        &mut op_stack_quot_table,
        zerofier_inverse,
        challenges,
    );
    let mut ram_quot_table = quot_table.slice_mut(s![.., ram_section_start..ram_section_end]);
    ExtRamTable::fill_consistency_quotients(
        master_base_table,
        master_ext_table,
        &mut ram_quot_table,
        zerofier_inverse,
        challenges,
    );
    let mut jump_stack_quot_table =
        quot_table.slice_mut(s![.., jump_stack_section_start..jump_stack_section_end]);
    ExtJumpStackTable::fill_consistency_quotients(
        master_base_table,
        master_ext_table,
        &mut jump_stack_quot_table,
        zerofier_inverse,
        challenges,
    );
    let mut hash_quot_table = quot_table.slice_mut(s![.., hash_section_start..hash_section_end]);
    ExtHashTable::fill_consistency_quotients(
        master_base_table,
        master_ext_table,
        &mut hash_quot_table,
        zerofier_inverse,
        challenges,
    );
    let mut cascade_quot_table =
        quot_table.slice_mut(s![.., cascade_section_start..cascade_section_end]);
    ExtCascadeTable::fill_consistency_quotients(
        master_base_table,
        master_ext_table,
        &mut cascade_quot_table,
        zerofier_inverse,
        challenges,
    );
    let mut lookup_quot_table =
        quot_table.slice_mut(s![.., lookup_section_start..lookup_section_end]);
    ExtLookupTable::fill_consistency_quotients(
        master_base_table,
        master_ext_table,
        &mut lookup_quot_table,
        zerofier_inverse,
        challenges,
    );
    let mut u32_quot_table = quot_table.slice_mut(s![.., u32_section_start..u32_section_end]);
    ExtU32Table::fill_consistency_quotients(
        master_base_table,
        master_ext_table,
        &mut u32_quot_table,
        zerofier_inverse,
        challenges,
    );
}

pub fn fill_all_transition_quotients(
    master_base_table: ArrayView2<BFieldElement>,
    master_ext_table: ArrayView2<XFieldElement>,
    quot_table: &mut ArrayViewMut2<XFieldElement>,
    zerofier_inverse: ArrayView1<BFieldElement>,
    challenges: &Challenges,
    trace_domain: ArithmeticDomain,
    quotient_domain: ArithmeticDomain,
) {
    // The order of the quotient tables is not actually important. However, it must be consistent
    // between prover and verifier, and the shapes must check out.
    let program_section_start = 0;
    let program_section_end = program_section_start + ExtProgramTable::num_transition_quotients();
    let processor_section_start = program_section_end;
    let processor_section_end =
        processor_section_start + ExtProcessorTable::num_transition_quotients();
    let op_stack_section_start = processor_section_end;
    let op_stack_section_end = op_stack_section_start + ExtOpStackTable::num_transition_quotients();
    let ram_section_start = op_stack_section_end;
    let ram_section_end = ram_section_start + ExtRamTable::num_transition_quotients();
    let jump_stack_section_start = ram_section_end;
    let jump_stack_section_end =
        jump_stack_section_start + ExtJumpStackTable::num_transition_quotients();
    let hash_section_start = jump_stack_section_end;
    let hash_section_end = hash_section_start + ExtHashTable::num_transition_quotients();
    let cascade_section_start = hash_section_end;
    let cascade_section_end = cascade_section_start + ExtCascadeTable::num_transition_quotients();
    let lookup_section_start = cascade_section_end;
    let lookup_section_end = lookup_section_start + ExtLookupTable::num_transition_quotients();
    let u32_section_start = lookup_section_end;
    let u32_section_end = u32_section_start + ExtU32Table::num_transition_quotients();

    let mut program_quot_table =
        quot_table.slice_mut(s![.., program_section_start..program_section_end]);
    ExtProgramTable::fill_transition_quotients(
        master_base_table,
        master_ext_table,
        &mut program_quot_table,
        zerofier_inverse,
        challenges,
        trace_domain,
        quotient_domain,
    );
    let mut processor_quot_table =
        quot_table.slice_mut(s![.., processor_section_start..processor_section_end]);
    ExtProcessorTable::fill_transition_quotients(
        master_base_table,
        master_ext_table,
        &mut processor_quot_table,
        zerofier_inverse,
        challenges,
        trace_domain,
        quotient_domain,
    );
    let mut op_stack_quot_table =
        quot_table.slice_mut(s![.., op_stack_section_start..op_stack_section_end]);
    ExtOpStackTable::fill_transition_quotients(
        master_base_table,
        master_ext_table,
        &mut op_stack_quot_table,
        zerofier_inverse,
        challenges,
        trace_domain,
        quotient_domain,
    );
    let mut ram_quot_table = quot_table.slice_mut(s![.., ram_section_start..ram_section_end]);
    ExtRamTable::fill_transition_quotients(
        master_base_table,
        master_ext_table,
        &mut ram_quot_table,
        zerofier_inverse,
        challenges,
        trace_domain,
        quotient_domain,
    );
    let mut jump_stack_quot_table =
        quot_table.slice_mut(s![.., jump_stack_section_start..jump_stack_section_end]);
    ExtJumpStackTable::fill_transition_quotients(
        master_base_table,
        master_ext_table,
        &mut jump_stack_quot_table,
        zerofier_inverse,
        challenges,
        trace_domain,
        quotient_domain,
    );
    let mut hash_quot_table = quot_table.slice_mut(s![.., hash_section_start..hash_section_end]);
    ExtHashTable::fill_transition_quotients(
        master_base_table,
        master_ext_table,
        &mut hash_quot_table,
        zerofier_inverse,
        challenges,
        trace_domain,
        quotient_domain,
    );
    let mut cascade_quot_table =
        quot_table.slice_mut(s![.., cascade_section_start..cascade_section_end]);
    ExtCascadeTable::fill_transition_quotients(
        master_base_table,
        master_ext_table,
        &mut cascade_quot_table,
        zerofier_inverse,
        challenges,
        trace_domain,
        quotient_domain,
    );
    let mut lookup_quot_table =
        quot_table.slice_mut(s![.., lookup_section_start..lookup_section_end]);
    ExtLookupTable::fill_transition_quotients(
        master_base_table,
        master_ext_table,
        &mut lookup_quot_table,
        zerofier_inverse,
        challenges,
        trace_domain,
        quotient_domain,
    );
    let mut u32_quot_table = quot_table.slice_mut(s![.., u32_section_start..u32_section_end]);
    ExtU32Table::fill_transition_quotients(
        master_base_table,
        master_ext_table,
        &mut u32_quot_table,
        zerofier_inverse,
        challenges,
        trace_domain,
        quotient_domain,
    );
}

pub fn fill_all_terminal_quotients(
    master_base_table: ArrayView2<BFieldElement>,
    master_ext_table: ArrayView2<XFieldElement>,
    quot_table: &mut ArrayViewMut2<XFieldElement>,
    zerofier_inverse: ArrayView1<BFieldElement>,
    challenges: &Challenges,
) {
    // The order of the quotient tables is not actually important. However, it must be consistent
    // between prover and verifier, and the shapes must check out.
    let program_section_start = 0;
    let program_section_end = program_section_start + ExtProgramTable::num_terminal_quotients();
    let processor_section_start = program_section_end;
    let processor_section_end =
        processor_section_start + ExtProcessorTable::num_terminal_quotients();
    let op_stack_section_start = processor_section_end;
    let op_stack_section_end = op_stack_section_start + ExtOpStackTable::num_terminal_quotients();
    let ram_section_start = op_stack_section_end;
    let ram_section_end = ram_section_start + ExtRamTable::num_terminal_quotients();
    let jump_stack_section_start = ram_section_end;
    let jump_stack_section_end =
        jump_stack_section_start + ExtJumpStackTable::num_terminal_quotients();
    let hash_section_start = jump_stack_section_end;
    let hash_section_end = hash_section_start + ExtHashTable::num_terminal_quotients();
    let cascade_section_start = hash_section_end;
    let cascade_section_end = cascade_section_start + ExtCascadeTable::num_terminal_quotients();
    let lookup_section_start = cascade_section_end;
    let lookup_section_end = lookup_section_start + ExtLookupTable::num_terminal_quotients();
    let u32_section_start = lookup_section_end;
    let u32_section_end = u32_section_start + ExtU32Table::num_terminal_quotients();
    let cross_table_section_start = u32_section_end;
    let cross_table_section_end =
        cross_table_section_start + GrandCrossTableArg::num_terminal_quotients();

    let mut program_quot_table =
        quot_table.slice_mut(s![.., program_section_start..program_section_end]);
    ExtProgramTable::fill_terminal_quotients(
        master_base_table,
        master_ext_table,
        &mut program_quot_table,
        zerofier_inverse,
        challenges,
    );
    let mut processor_quot_table =
        quot_table.slice_mut(s![.., processor_section_start..processor_section_end]);
    ExtProcessorTable::fill_terminal_quotients(
        master_base_table,
        master_ext_table,
        &mut processor_quot_table,
        zerofier_inverse,
        challenges,
    );
    let mut op_stack_quot_table =
        quot_table.slice_mut(s![.., op_stack_section_start..op_stack_section_end]);
    ExtOpStackTable::fill_terminal_quotients(
        master_base_table,
        master_ext_table,
        &mut op_stack_quot_table,
        zerofier_inverse,
        challenges,
    );
    let mut ram_quot_table = quot_table.slice_mut(s![.., ram_section_start..ram_section_end]);
    ExtRamTable::fill_terminal_quotients(
        master_base_table,
        master_ext_table,
        &mut ram_quot_table,
        zerofier_inverse,
        challenges,
    );
    let mut jump_stack_quot_table =
        quot_table.slice_mut(s![.., jump_stack_section_start..jump_stack_section_end]);
    ExtJumpStackTable::fill_terminal_quotients(
        master_base_table,
        master_ext_table,
        &mut jump_stack_quot_table,
        zerofier_inverse,
        challenges,
    );
    let mut hash_quot_table = quot_table.slice_mut(s![.., hash_section_start..hash_section_end]);
    ExtHashTable::fill_terminal_quotients(
        master_base_table,
        master_ext_table,
        &mut hash_quot_table,
        zerofier_inverse,
        challenges,
    );
    let mut cascade_quot_table =
        quot_table.slice_mut(s![.., cascade_section_start..cascade_section_end]);
    ExtCascadeTable::fill_terminal_quotients(
        master_base_table,
        master_ext_table,
        &mut cascade_quot_table,
        zerofier_inverse,
        challenges,
    );
    let mut lookup_quot_table =
        quot_table.slice_mut(s![.., lookup_section_start..lookup_section_end]);
    ExtLookupTable::fill_terminal_quotients(
        master_base_table,
        master_ext_table,
        &mut lookup_quot_table,
        zerofier_inverse,
        challenges,
    );
    let mut u32_quot_table = quot_table.slice_mut(s![.., u32_section_start..u32_section_end]);
    ExtU32Table::fill_terminal_quotients(
        master_base_table,
        master_ext_table,
        &mut u32_quot_table,
        zerofier_inverse,
        challenges,
    );
    let mut cross_table_argument_quot_table =
        quot_table.slice_mut(s![.., cross_table_section_start..cross_table_section_end]);
    GrandCrossTableArg::fill_terminal_quotients(
        master_base_table,
        master_ext_table,
        &mut cross_table_argument_quot_table,
        zerofier_inverse,
        challenges,
    );
}

/// Computes an array containing all quotients – the Master Quotient Table. Each column corresponds
/// to a different quotient. The quotients are ordered by category – initial, consistency,
/// transition, and then terminal. Within each category, the quotients follow the canonical order
/// of the tables. The last column holds the terminal quotient of the cross-table argument, which
/// is strictly speaking not a table.
/// The order of the quotients is not actually important. However, it must be consistent between
/// prover and verifier.
///
/// The returned array is in row-major order.
pub fn all_quotients(
    quotient_domain_master_base_table: ArrayView2<BFieldElement>,
    quotient_domain_master_ext_table: ArrayView2<XFieldElement>,
    trace_domain: ArithmeticDomain,
    quotient_domain: ArithmeticDomain,
    challenges: &Challenges,
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

    prof_start!(maybe_profiler, "malloc");
    let num_columns = num_all_table_quotients();
    let mut all_quotients = Array2::zeros([quotient_domain.length, num_columns]);
    prof_stop!(maybe_profiler, "malloc");

    let initial_quotient_section_start = 0;
    let initial_quotient_section_end = initial_quotient_section_start + num_all_initial_quotients();
    let consistency_quotient_section_start = initial_quotient_section_end;
    let consistency_quotient_section_end =
        consistency_quotient_section_start + num_all_consistency_quotients();
    let transition_quotient_section_start = consistency_quotient_section_end;
    let transition_quotient_section_end =
        transition_quotient_section_start + num_all_transition_quotients();
    let terminal_quotient_section_start = transition_quotient_section_end;
    let terminal_quotient_section_end =
        terminal_quotient_section_start + num_all_terminal_quotients();

    prof_start!(maybe_profiler, "initial");
    let mut initial_quot_table = all_quotients.slice_mut(s![
        ..,
        initial_quotient_section_start..initial_quotient_section_end
    ]);
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
    let mut consistency_quotients = all_quotients.slice_mut(s![
        ..,
        consistency_quotient_section_start..consistency_quotient_section_end
    ]);
    let consistency_quotient_zerofier_inverse =
        consistency_quotient_zerofier_inverse(trace_domain, quotient_domain);
    fill_all_consistency_quotients(
        quotient_domain_master_base_table,
        quotient_domain_master_ext_table,
        &mut consistency_quotients,
        consistency_quotient_zerofier_inverse.view(),
        challenges,
    );
    prof_stop!(maybe_profiler, "consistency");

    prof_start!(maybe_profiler, "transition");
    let mut transition_quotients = all_quotients.slice_mut(s![
        ..,
        transition_quotient_section_start..transition_quotient_section_end
    ]);
    let transition_quotient_zerofier_inverse =
        transition_quotient_zerofier_inverse(trace_domain, quotient_domain);
    fill_all_transition_quotients(
        quotient_domain_master_base_table,
        quotient_domain_master_ext_table,
        &mut transition_quotients,
        transition_quotient_zerofier_inverse.view(),
        challenges,
        trace_domain,
        quotient_domain,
    );
    prof_stop!(maybe_profiler, "transition");

    prof_start!(maybe_profiler, "terminal");
    let mut terminal_quot_table = all_quotients.slice_mut(s![
        ..,
        terminal_quotient_section_start..terminal_quotient_section_end
    ]);
    let initial_quotient_zerofier_inverse =
        terminal_quotient_zerofier_inverse(trace_domain, quotient_domain);
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
    challenges: &Challenges,
) -> Vec<XFieldElement> {
    [
        ExtProgramTable::evaluate_initial_constraints(base_row, ext_row, challenges),
        ExtProcessorTable::evaluate_initial_constraints(base_row, ext_row, challenges),
        ExtOpStackTable::evaluate_initial_constraints(base_row, ext_row, challenges),
        ExtRamTable::evaluate_initial_constraints(base_row, ext_row, challenges),
        ExtJumpStackTable::evaluate_initial_constraints(base_row, ext_row, challenges),
        ExtHashTable::evaluate_initial_constraints(base_row, ext_row, challenges),
        ExtCascadeTable::evaluate_initial_constraints(base_row, ext_row, challenges),
        ExtLookupTable::evaluate_initial_constraints(base_row, ext_row, challenges),
        ExtU32Table::evaluate_initial_constraints(base_row, ext_row, challenges),
    ]
    .concat()
}

pub fn evaluate_all_consistency_constraints(
    base_row: ArrayView1<BFieldElement>,
    ext_row: ArrayView1<XFieldElement>,
    challenges: &Challenges,
) -> Vec<XFieldElement> {
    [
        ExtProgramTable::evaluate_consistency_constraints(base_row, ext_row, challenges),
        ExtProcessorTable::evaluate_consistency_constraints(base_row, ext_row, challenges),
        ExtOpStackTable::evaluate_consistency_constraints(base_row, ext_row, challenges),
        ExtRamTable::evaluate_consistency_constraints(base_row, ext_row, challenges),
        ExtJumpStackTable::evaluate_consistency_constraints(base_row, ext_row, challenges),
        ExtHashTable::evaluate_consistency_constraints(base_row, ext_row, challenges),
        ExtCascadeTable::evaluate_consistency_constraints(base_row, ext_row, challenges),
        ExtLookupTable::evaluate_consistency_constraints(base_row, ext_row, challenges),
        ExtU32Table::evaluate_consistency_constraints(base_row, ext_row, challenges),
    ]
    .concat()
}

pub fn evaluate_all_transition_constraints(
    current_base_row: ArrayView1<BFieldElement>,
    current_ext_row: ArrayView1<XFieldElement>,
    next_base_row: ArrayView1<BFieldElement>,
    next_ext_row: ArrayView1<XFieldElement>,
    challenges: &Challenges,
) -> Vec<XFieldElement> {
    let cbr = current_base_row;
    let cer = current_ext_row;
    let nbr = next_base_row;
    let ner = next_ext_row;
    [
        ExtProgramTable::evaluate_transition_constraints(cbr, cer, nbr, ner, challenges),
        ExtProcessorTable::evaluate_transition_constraints(cbr, cer, nbr, ner, challenges),
        ExtOpStackTable::evaluate_transition_constraints(cbr, cer, nbr, ner, challenges),
        ExtRamTable::evaluate_transition_constraints(cbr, cer, nbr, ner, challenges),
        ExtJumpStackTable::evaluate_transition_constraints(cbr, cer, nbr, ner, challenges),
        ExtHashTable::evaluate_transition_constraints(cbr, cer, nbr, ner, challenges),
        ExtCascadeTable::evaluate_transition_constraints(cbr, cer, nbr, ner, challenges),
        ExtLookupTable::evaluate_transition_constraints(cbr, cer, nbr, ner, challenges),
        ExtU32Table::evaluate_transition_constraints(cbr, cer, nbr, ner, challenges),
    ]
    .concat()
}

pub fn evaluate_all_terminal_constraints(
    base_row: ArrayView1<BFieldElement>,
    ext_row: ArrayView1<XFieldElement>,
    challenges: &Challenges,
) -> Vec<XFieldElement> {
    [
        ExtProgramTable::evaluate_terminal_constraints(base_row, ext_row, challenges),
        ExtProcessorTable::evaluate_terminal_constraints(base_row, ext_row, challenges),
        ExtOpStackTable::evaluate_terminal_constraints(base_row, ext_row, challenges),
        ExtRamTable::evaluate_terminal_constraints(base_row, ext_row, challenges),
        ExtJumpStackTable::evaluate_terminal_constraints(base_row, ext_row, challenges),
        ExtHashTable::evaluate_terminal_constraints(base_row, ext_row, challenges),
        ExtCascadeTable::evaluate_terminal_constraints(base_row, ext_row, challenges),
        ExtLookupTable::evaluate_terminal_constraints(base_row, ext_row, challenges),
        ExtU32Table::evaluate_terminal_constraints(base_row, ext_row, challenges),
        GrandCrossTableArg::evaluate_terminal_constraints(base_row, ext_row, challenges),
    ]
    .concat()
}

pub fn evaluate_all_constraints(
    current_base_row: ArrayView1<BFieldElement>,
    current_ext_row: ArrayView1<XFieldElement>,
    next_base_row: ArrayView1<BFieldElement>,
    next_ext_row: ArrayView1<XFieldElement>,
    challenges: &Challenges,
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

pub fn randomized_padded_trace_len(num_trace_randomizers: usize, padded_height: usize) -> usize {
    roundup_npo2((padded_height + num_trace_randomizers) as u64) as usize
}

pub fn interpolant_degree(padded_height: usize, num_trace_randomizers: usize) -> Degree {
    (randomized_padded_trace_len(padded_height, num_trace_randomizers) - 1) as Degree
}

pub fn derive_domain_generator(domain_length: u64) -> BFieldElement {
    debug_assert!(
        0 == domain_length || is_power_of_two(domain_length),
        "The domain length must be a power of 2 but was {domain_length}.",
    );
    BFieldElement::primitive_root_of_unity(domain_length).unwrap()
}

/// Primarily for debugging purposes.
/// Given the global index of some constraint, returns
/// 1. the type of constraint (initial, consistency, transition, or terminal),
/// 2. the name of the table that the constraint is in, and
/// 3. the index within that table.
pub fn constraint_type_and_index_and_table_name(
    constraint_idx: usize,
) -> (&'static str, &'static str, usize) {
    let initial_section_start = 0;
    let initial_section_end = initial_section_start + num_all_initial_quotients();
    let consistency_section_start = initial_section_end;
    let consistency_section_end = consistency_section_start + num_all_consistency_quotients();
    let transition_section_start = consistency_section_end;
    let transition_section_end = transition_section_start + num_all_transition_quotients();
    let terminal_section_start = transition_section_end;
    let terminal_section_end = terminal_section_start + num_all_terminal_quotients();
    assert_eq!(num_all_table_quotients(), terminal_section_end);
    match constraint_idx {
        idx if initial_section_start <= idx && idx < initial_section_end => {
            let section_idx = idx - initial_section_start;
            let (table_idx, table_name) = initial_constraint_table_idx_and_name(section_idx);
            ("initial", table_name, table_idx)
        }
        idx if consistency_section_start <= idx && idx < consistency_section_end => {
            let section_idx = idx - consistency_section_start;
            let (table_idx, table_name) = consistency_constraint_table_idx_and_name(section_idx);
            ("consistency", table_name, table_idx)
        }
        idx if transition_section_start <= idx && idx < transition_section_end => {
            let section_idx = idx - transition_section_start;
            let (table_idx, table_name) = transition_constraint_table_idx_and_name(section_idx);
            ("transition", table_name, table_idx)
        }
        idx if terminal_section_start <= idx && idx < terminal_section_end => {
            let section_idx = idx - terminal_section_start;
            let (table_idx, table_name) = terminal_constraint_table_idx_and_name(section_idx);
            ("terminal", table_name, table_idx)
        }
        _ => ("unknown", "unknown", 0),
    }
}

/// Primarily for debugging purposes.
/// Given the section index of some initial constraint, returns
/// 1. the index within the specific table for that constraint, and
/// 2. the name of that table.
pub fn initial_constraint_table_idx_and_name(constraint_idx: usize) -> (usize, &'static str) {
    let program_start = 0;
    let program_end = program_start + ExtProgramTable::num_initial_quotients();
    let processor_start = program_end;
    let processor_end = processor_start + ExtProcessorTable::num_initial_quotients();
    let op_stack_start = processor_end;
    let op_stack_end = op_stack_start + ExtOpStackTable::num_initial_quotients();
    let ram_start = op_stack_end;
    let ram_end = ram_start + ExtRamTable::num_initial_quotients();
    let jump_stack_start = ram_end;
    let jump_stack_end = jump_stack_start + ExtJumpStackTable::num_initial_quotients();
    let hash_start = jump_stack_end;
    let hash_end = hash_start + ExtHashTable::num_initial_quotients();
    let cascade_start = hash_end;
    let cascade_end = cascade_start + ExtCascadeTable::num_initial_quotients();
    let lookup_start = cascade_end;
    let lookup_end = lookup_start + ExtLookupTable::num_initial_quotients();
    let u32_start = lookup_end;
    let u32_end = u32_start + ExtU32Table::num_initial_quotients();
    assert_eq!(num_all_initial_quotients(), u32_end);
    match constraint_idx {
        i if program_start <= i && i < program_end => (i - program_start, "Program"),
        i if processor_start <= i && i < processor_end => (i - processor_start, "Processor"),
        i if op_stack_start <= i && i < op_stack_end => (i - op_stack_start, "OpStack"),
        i if ram_start <= i && i < ram_end => (i - ram_start, "Ram"),
        i if jump_stack_start <= i && i < jump_stack_end => (i - jump_stack_start, "JumpStack"),
        i if hash_start <= i && i < hash_end => (i - hash_start, "Hash"),
        i if cascade_start <= i && i < cascade_end => (i - cascade_start, "Cascade"),
        i if lookup_start <= i && i < lookup_end => (i - lookup_start, "Lookup"),
        i if u32_start <= i && i < u32_end => (i - u32_start, "U32"),
        _ => (0, "Unknown"),
    }
}

/// Primarily for debugging purposes.
/// Given the section index of some consistency constraint, returns
/// 1. the index within the specific table for that constraint, and
/// 2. the name of that table.
pub fn consistency_constraint_table_idx_and_name(constraint_idx: usize) -> (usize, &'static str) {
    let program_start = 0;
    let program_end = program_start + ExtProgramTable::num_consistency_quotients();
    let processor_start = program_end;
    let processor_end = processor_start + ExtProcessorTable::num_consistency_quotients();
    let op_stack_start = processor_end;
    let op_stack_end = op_stack_start + ExtOpStackTable::num_consistency_quotients();
    let ram_start = op_stack_end;
    let ram_end = ram_start + ExtRamTable::num_consistency_quotients();
    let jump_stack_start = ram_end;
    let jump_stack_end = jump_stack_start + ExtJumpStackTable::num_consistency_quotients();
    let hash_start = jump_stack_end;
    let hash_end = hash_start + ExtHashTable::num_consistency_quotients();
    let cascade_start = hash_end;
    let cascade_end = cascade_start + ExtCascadeTable::num_consistency_quotients();
    let lookup_start = cascade_end;
    let lookup_end = lookup_start + ExtLookupTable::num_consistency_quotients();
    let u32_start = lookup_end;
    let u32_end = u32_start + ExtU32Table::num_consistency_quotients();
    assert_eq!(num_all_consistency_quotients(), u32_end);
    match constraint_idx {
        i if program_start <= i && i < program_end => (i - program_start, "Program"),
        i if processor_start <= i && i < processor_end => (i - processor_start, "Processor"),
        i if op_stack_start <= i && i < op_stack_end => (i - op_stack_start, "OpStack"),
        i if ram_start <= i && i < ram_end => (i - ram_start, "Ram"),
        i if jump_stack_start <= i && i < jump_stack_end => (i - jump_stack_start, "JumpStack"),
        i if hash_start <= i && i < hash_end => (i - hash_start, "Hash"),
        i if cascade_start <= i && i < cascade_end => (i - cascade_start, "Cascade"),
        i if lookup_start <= i && i < lookup_end => (i - lookup_start, "Lookup"),
        i if u32_start <= i && i < u32_end => (i - u32_start, "U32"),
        _ => (0, "Unknown"),
    }
}

/// Primarily for debugging purposes.
/// Given the section index of some transition constraint, returns
/// 1. the index within the specific table for that constraint, and
/// 2. the name of that table.
pub fn transition_constraint_table_idx_and_name(constraint_idx: usize) -> (usize, &'static str) {
    let program_start = 0;
    let program_end = program_start + ExtProgramTable::num_transition_quotients();
    let processor_start = program_end;
    let processor_end = processor_start + ExtProcessorTable::num_transition_quotients();
    let op_stack_start = processor_end;
    let op_stack_end = op_stack_start + ExtOpStackTable::num_transition_quotients();
    let ram_start = op_stack_end;
    let ram_end = ram_start + ExtRamTable::num_transition_quotients();
    let jump_stack_start = ram_end;
    let jump_stack_end = jump_stack_start + ExtJumpStackTable::num_transition_quotients();
    let hash_start = jump_stack_end;
    let hash_end = hash_start + ExtHashTable::num_transition_quotients();
    let cascade_start = hash_end;
    let cascade_end = cascade_start + ExtCascadeTable::num_transition_quotients();
    let lookup_start = cascade_end;
    let lookup_end = lookup_start + ExtLookupTable::num_transition_quotients();
    let u32_start = lookup_end;
    let u32_end = u32_start + ExtU32Table::num_transition_quotients();
    assert_eq!(num_all_transition_quotients(), u32_end);
    match constraint_idx {
        i if program_start <= i && i < program_end => (i - program_start, "Program"),
        i if processor_start <= i && i < processor_end => (i - processor_start, "Processor"),
        i if op_stack_start <= i && i < op_stack_end => (i - op_stack_start, "OpStack"),
        i if ram_start <= i && i < ram_end => (i - ram_start, "Ram"),
        i if jump_stack_start <= i && i < jump_stack_end => (i - jump_stack_start, "JumpStack"),
        i if hash_start <= i && i < hash_end => (i - hash_start, "Hash"),
        i if cascade_start <= i && i < cascade_end => (i - cascade_start, "Cascade"),
        i if lookup_start <= i && i < lookup_end => (i - lookup_start, "Lookup"),
        i if u32_start <= i && i < u32_end => (i - u32_start, "U32"),
        _ => (0, "Unknown"),
    }
}

/// Primarily for debugging purposes.
/// Given the section index of some terminal constraint, returns
/// 1. the index within the specific table for that constraint, and
/// 2. the name of that table.
pub fn terminal_constraint_table_idx_and_name(constraint_idx: usize) -> (usize, &'static str) {
    let program_start = 0;
    let program_end = program_start + ExtProgramTable::num_terminal_quotients();
    let processor_start = program_end;
    let processor_end = processor_start + ExtProcessorTable::num_terminal_quotients();
    let op_stack_start = processor_end;
    let op_stack_end = op_stack_start + ExtOpStackTable::num_terminal_quotients();
    let ram_start = op_stack_end;
    let ram_end = ram_start + ExtRamTable::num_terminal_quotients();
    let jump_stack_start = ram_end;
    let jump_stack_end = jump_stack_start + ExtJumpStackTable::num_terminal_quotients();
    let hash_start = jump_stack_end;
    let hash_end = hash_start + ExtHashTable::num_terminal_quotients();
    let cascade_start = hash_end;
    let cascade_end = cascade_start + ExtCascadeTable::num_terminal_quotients();
    let lookup_start = cascade_end;
    let lookup_end = lookup_start + ExtLookupTable::num_terminal_quotients();
    let u32_start = lookup_end;
    let u32_end = u32_start + ExtU32Table::num_terminal_quotients();
    let cross_table_start = u32_end;
    let cross_table_end = cross_table_start + GrandCrossTableArg::num_terminal_quotients();
    assert_eq!(num_all_terminal_quotients(), cross_table_end);
    match constraint_idx {
        i if program_start <= i && i < program_end => (i - program_start, "Program"),
        i if processor_start <= i && i < processor_end => (i - processor_start, "Processor"),
        i if op_stack_start <= i && i < op_stack_end => (i - op_stack_start, "OpStack"),
        i if ram_start <= i && i < ram_end => (i - ram_start, "Ram"),
        i if jump_stack_start <= i && i < jump_stack_end => (i - jump_stack_start, "JumpStack"),
        i if hash_start <= i && i < hash_end => (i - hash_start, "Hash"),
        i if cascade_start <= i && i < cascade_end => (i - cascade_start, "Cascade"),
        i if lookup_start <= i && i < lookup_end => (i - lookup_start, "Lookup"),
        i if u32_start <= i && i < u32_end => (i - u32_start, "U32"),
        i if cross_table_start <= i && i < cross_table_end => {
            (i - cross_table_start, "GrandCrossTableArgument")
        }
        _ => (0, "Unknown"),
    }
}

#[cfg(test)]
mod master_table_tests {
    use ndarray::s;
    use num_traits::Zero;
    use strum::IntoEnumIterator;
    use twenty_first::shared_math::b_field_element::BFieldElement;
    use twenty_first::shared_math::traits::FiniteField;

    use crate::arithmetic_domain::ArithmeticDomain;
    use crate::stark::triton_stark_tests::parse_simulate_pad;
    use crate::stark::triton_stark_tests::parse_simulate_pad_extend;
    use crate::table::cascade_table;
    use crate::table::hash_table;
    use crate::table::jump_stack_table;
    use crate::table::lookup_table;
    use crate::table::master_table::consistency_quotient_zerofier_inverse;
    use crate::table::master_table::initial_quotient_zerofier_inverse;
    use crate::table::master_table::terminal_quotient_zerofier_inverse;
    use crate::table::master_table::transition_quotient_zerofier_inverse;
    use crate::table::master_table::TableId::*;
    use crate::table::master_table::EXT_U32_TABLE_END;
    use crate::table::master_table::NUM_BASE_COLUMNS;
    use crate::table::master_table::NUM_COLUMNS;
    use crate::table::master_table::NUM_EXT_COLUMNS;
    use crate::table::op_stack_table;
    use crate::table::processor_table;
    use crate::table::program_table;
    use crate::table::ram_table;
    use crate::table::table_column::CascadeBaseTableColumn;
    use crate::table::table_column::CascadeExtTableColumn;
    use crate::table::table_column::HashBaseTableColumn;
    use crate::table::table_column::HashExtTableColumn;
    use crate::table::table_column::JumpStackBaseTableColumn;
    use crate::table::table_column::JumpStackExtTableColumn;
    use crate::table::table_column::LookupBaseTableColumn;
    use crate::table::table_column::LookupExtTableColumn;
    use crate::table::table_column::MasterBaseTableColumn;
    use crate::table::table_column::MasterExtTableColumn;
    use crate::table::table_column::OpStackBaseTableColumn;
    use crate::table::table_column::OpStackExtTableColumn;
    use crate::table::table_column::ProcessorBaseTableColumn;
    use crate::table::table_column::ProcessorExtTableColumn;
    use crate::table::table_column::ProgramBaseTableColumn;
    use crate::table::table_column::ProgramExtTableColumn;
    use crate::table::table_column::RamBaseTableColumn;
    use crate::table::table_column::RamExtTableColumn;
    use crate::table::table_column::U32BaseTableColumn;
    use crate::table::table_column::U32ExtTableColumn;
    use crate::table::u32_table;

    #[test]
    fn base_table_width_is_correct() {
        let (_, _, master_base_table) = parse_simulate_pad("halt", vec![], vec![]);

        assert_eq!(
            program_table::BASE_WIDTH,
            master_base_table.table(ProgramTable).ncols()
        );
        assert_eq!(
            processor_table::BASE_WIDTH,
            master_base_table.table(ProcessorTable).ncols()
        );
        assert_eq!(
            op_stack_table::BASE_WIDTH,
            master_base_table.table(OpStackTable).ncols()
        );
        assert_eq!(
            ram_table::BASE_WIDTH,
            master_base_table.table(RamTable).ncols()
        );
        assert_eq!(
            jump_stack_table::BASE_WIDTH,
            master_base_table.table(JumpStackTable).ncols()
        );
        assert_eq!(
            hash_table::BASE_WIDTH,
            master_base_table.table(HashTable).ncols()
        );
        assert_eq!(
            cascade_table::BASE_WIDTH,
            master_base_table.table(CascadeTable).ncols()
        );
        assert_eq!(
            lookup_table::BASE_WIDTH,
            master_base_table.table(LookupTable).ncols()
        );
        assert_eq!(
            u32_table::BASE_WIDTH,
            master_base_table.table(U32Table).ncols()
        );
    }

    #[test]
    fn ext_table_width_is_correct() {
        let (stark, _, _, master_ext_table, _) = parse_simulate_pad_extend("halt", vec![], vec![]);

        assert_eq!(
            program_table::EXT_WIDTH,
            master_ext_table.table(ProgramTable).ncols()
        );
        assert_eq!(
            processor_table::EXT_WIDTH,
            master_ext_table.table(ProcessorTable).ncols()
        );
        assert_eq!(
            op_stack_table::EXT_WIDTH,
            master_ext_table.table(OpStackTable).ncols()
        );
        assert_eq!(
            ram_table::EXT_WIDTH,
            master_ext_table.table(RamTable).ncols()
        );
        assert_eq!(
            jump_stack_table::EXT_WIDTH,
            master_ext_table.table(JumpStackTable).ncols()
        );
        assert_eq!(
            hash_table::EXT_WIDTH,
            master_ext_table.table(HashTable).ncols()
        );
        assert_eq!(
            cascade_table::EXT_WIDTH,
            master_ext_table.table(CascadeTable).ncols()
        );
        assert_eq!(
            lookup_table::EXT_WIDTH,
            master_ext_table.table(LookupTable).ncols()
        );
        assert_eq!(
            u32_table::EXT_WIDTH,
            master_ext_table.table(U32Table).ncols()
        );
        // use some domain-specific knowledge to also check for the randomizer columns
        assert_eq!(
            stark.parameters.num_randomizer_polynomials,
            master_ext_table
                .master_ext_matrix
                .slice(s![.., EXT_U32_TABLE_END..])
                .ncols()
        );
    }

    #[test]
    fn zerofiers_are_correct_test() {
        let big_order = 16;
        let big_offset = BFieldElement::generator();
        let big_domain = ArithmeticDomain::new(big_offset, big_order as usize);

        let small_order = 8;
        let small_domain = ArithmeticDomain::new_no_offset(small_order as usize);

        let initial_zerofier_inv = initial_quotient_zerofier_inverse(big_domain);
        let initial_zerofier = BFieldElement::batch_inversion(initial_zerofier_inv.to_vec());
        let initial_zerofier_poly = big_domain.interpolate(&initial_zerofier);
        assert_eq!(big_order as usize, initial_zerofier_inv.len());
        assert_eq!(1, initial_zerofier_poly.degree());
        assert!(initial_zerofier_poly
            .evaluate(&small_domain.domain_value(0))
            .is_zero());

        let consistency_zerofier_inv =
            consistency_quotient_zerofier_inverse(small_domain, big_domain);
        let consistency_zerofier =
            BFieldElement::batch_inversion(consistency_zerofier_inv.to_vec());
        let consistency_zerofier_poly = big_domain.interpolate(&consistency_zerofier);
        assert_eq!(big_order as usize, consistency_zerofier_inv.len());
        assert_eq!(small_order as isize, consistency_zerofier_poly.degree());
        for val in small_domain.domain_values() {
            assert!(consistency_zerofier_poly.evaluate(&val).is_zero());
        }

        let transition_zerofier_inv =
            transition_quotient_zerofier_inverse(small_domain, big_domain);
        let transition_zerofier = BFieldElement::batch_inversion(transition_zerofier_inv.to_vec());
        let transition_zerofier_poly = big_domain.interpolate(&transition_zerofier);
        assert_eq!(big_order as usize, transition_zerofier_inv.len());
        assert_eq!(small_order as isize - 1, transition_zerofier_poly.degree());
        for val in small_domain
            .domain_values()
            .iter()
            .take(small_order as usize - 1)
        {
            assert!(transition_zerofier_poly.evaluate(val).is_zero());
        }

        let terminal_zerofier_inv = terminal_quotient_zerofier_inverse(small_domain, big_domain);
        let terminal_zerofier = BFieldElement::batch_inversion(terminal_zerofier_inv.to_vec());
        let terminal_zerofier_poly = big_domain.interpolate(&terminal_zerofier);
        assert_eq!(big_order as usize, terminal_zerofier_inv.len());
        assert_eq!(1, terminal_zerofier_poly.degree());
        assert!(terminal_zerofier_poly
            .evaluate(&small_domain.domain_value(small_order as u32 - 1))
            .is_zero());
    }

    /// intended use: `cargo t print_all_table_widths -- --nocapture`
    #[test]
    fn print_all_table_widths() {
        println!();
        println!("| table name         | #base cols | #ext cols | full width |");
        println!("|:-------------------|-----------:|----------:|-----------:|");
        println!(
            "| {:<18} | {:>10} | {:>9} | {:>10} |",
            "ProgramTable",
            program_table::BASE_WIDTH,
            program_table::EXT_WIDTH,
            program_table::FULL_WIDTH
        );
        println!(
            "| {:<18} | {:>10} | {:>9} | {:>10} |",
            "ProcessorTable",
            processor_table::BASE_WIDTH,
            processor_table::EXT_WIDTH,
            processor_table::FULL_WIDTH
        );
        println!(
            "| {:<18} | {:>10} | {:>9} | {:>10} |",
            "OpStackTable",
            op_stack_table::BASE_WIDTH,
            op_stack_table::EXT_WIDTH,
            op_stack_table::FULL_WIDTH
        );
        println!(
            "| {:<18} | {:>10} | {:>9} | {:>10} |",
            "RamTable",
            ram_table::BASE_WIDTH,
            ram_table::EXT_WIDTH,
            ram_table::FULL_WIDTH
        );
        println!(
            "| {:<18} | {:>10} | {:>9} | {:>10} |",
            "JumpStackTable",
            jump_stack_table::BASE_WIDTH,
            jump_stack_table::EXT_WIDTH,
            jump_stack_table::FULL_WIDTH
        );
        println!(
            "| {:<18} | {:>10} | {:>9} | {:>10} |",
            "HashTable",
            hash_table::BASE_WIDTH,
            hash_table::EXT_WIDTH,
            hash_table::FULL_WIDTH
        );
        println!(
            "| {:<18} | {:>10} | {:>9} | {:>10} |",
            "CascadeTable",
            cascade_table::BASE_WIDTH,
            cascade_table::EXT_WIDTH,
            cascade_table::FULL_WIDTH
        );
        println!(
            "| {:<18} | {:>10} | {:>9} | {:>10} |",
            "LookupTable",
            lookup_table::BASE_WIDTH,
            lookup_table::EXT_WIDTH,
            lookup_table::FULL_WIDTH
        );
        println!(
            "| {:<18} | {:>10} | {:>9} | {:>10} |",
            "U32Table",
            u32_table::BASE_WIDTH,
            u32_table::EXT_WIDTH,
            u32_table::FULL_WIDTH
        );
        println!("|                    |            |           |            |");
        println!(
            "| Sum                | {NUM_BASE_COLUMNS:>10} \
             | {NUM_EXT_COLUMNS:>9} | {NUM_COLUMNS:>10} |",
        );
    }

    /// intended use: `cargo t print_all_master_table_indices -- --nocapture`
    #[test]
    fn print_all_master_table_indices() {
        println!();
        println!("idx | table       | base column");
        println!("---:|:------------|:-----------");
        for column in ProgramBaseTableColumn::iter() {
            println!(
                "{:>3} | program     | {column}",
                column.master_base_table_index()
            );
        }
        for column in ProcessorBaseTableColumn::iter() {
            println!(
                "{:>3} | processor   | {column}",
                column.master_base_table_index()
            );
        }
        for column in OpStackBaseTableColumn::iter() {
            println!(
                "{:>3} | op stack    | {column}",
                column.master_base_table_index()
            );
        }
        for column in RamBaseTableColumn::iter() {
            println!(
                "{:>3} | ram         | {column}",
                column.master_base_table_index()
            );
        }
        for column in JumpStackBaseTableColumn::iter() {
            println!(
                "{:>3} | jump stack  | {column}",
                column.master_base_table_index()
            );
        }
        for column in HashBaseTableColumn::iter() {
            println!(
                "{:>3} | hash        | {column}",
                column.master_base_table_index()
            );
        }
        for column in CascadeBaseTableColumn::iter() {
            println!(
                "{:>3} | cascade     | {column}",
                column.master_base_table_index()
            );
        }
        for column in LookupBaseTableColumn::iter() {
            println!(
                "{:>3} | lookup      | {column}",
                column.master_base_table_index()
            );
        }
        for column in U32BaseTableColumn::iter() {
            println!(
                "{:>3} | u32         | {column}",
                column.master_base_table_index()
            );
        }
        println!();
        println!("idx | table       | extension column");
        println!("---:|:------------|:----------------");
        for column in ProgramExtTableColumn::iter() {
            println!(
                "{:>3} | program     | {column}",
                column.master_ext_table_index()
            );
        }
        for column in ProcessorExtTableColumn::iter() {
            println!(
                "{:>3} | processor   | {column}",
                column.master_ext_table_index()
            );
        }
        for column in OpStackExtTableColumn::iter() {
            println!(
                "{:>3} | op stack    | {column}",
                column.master_ext_table_index()
            );
        }
        for column in RamExtTableColumn::iter() {
            println!(
                "{:>3} | ram         | {column}",
                column.master_ext_table_index()
            );
        }
        for column in JumpStackExtTableColumn::iter() {
            println!(
                "{:>3} | jump stack  | {column}",
                column.master_ext_table_index()
            );
        }
        for column in HashExtTableColumn::iter() {
            println!(
                "{:>3} | hash        | {column}",
                column.master_ext_table_index()
            );
        }
        for column in CascadeExtTableColumn::iter() {
            println!(
                "{:>3} | cascade     | {column}",
                column.master_ext_table_index()
            );
        }
        for column in LookupExtTableColumn::iter() {
            println!(
                "{:>3} | lookup      | {column}",
                column.master_ext_table_index()
            );
        }
        for column in U32ExtTableColumn::iter() {
            println!(
                "{:>3} | u32         | {column}",
                column.master_ext_table_index()
            );
        }
    }
}
