use std::borrow::Borrow;
use std::ops::Mul;
use std::ops::MulAssign;
use std::ops::Range;

use itertools::Itertools;
use master_table::extension_table::Evaluable;
use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use ndarray::s;
use ndarray::Array2;
use ndarray::ArrayView2;
use ndarray::ArrayViewMut2;
use ndarray::Zip;
use num_traits::One;
use num_traits::Zero;
use rand::distributions::Standard;
use rand::prelude::Distribution;
use rand::random;
use strum::Display;
use strum::EnumCount;
use strum::EnumIter;
use twenty_first::math::tip5::DIGEST_LENGTH;
use twenty_first::math::tip5::RATE;
use twenty_first::math::traits::FiniteField;
use twenty_first::prelude::*;
use twenty_first::util_types::algebraic_hasher;

use crate::aet::AlgebraicExecutionTrace;
use crate::arithmetic_domain::ArithmeticDomain;
use crate::config::CacheDecision;
use crate::error::ProvingError;
use crate::profiler::profiler;
use crate::stark::NUM_RANDOMIZER_POLYNOMIALS;
use crate::table::cascade_table::CascadeTable;
use crate::table::challenges::Challenges;
use crate::table::degree_lowering_table::DegreeLoweringTable;
use crate::table::extension_table::all_degrees_with_origin;
use crate::table::extension_table::DegreeWithOrigin;
use crate::table::extension_table::Quotientable;
use crate::table::hash_table::HashTable;
use crate::table::jump_stack_table::JumpStackTable;
use crate::table::lookup_table::LookupTable;
use crate::table::op_stack_table::OpStackTable;
use crate::table::processor_table::ProcessorTable;
use crate::table::program_table::ProgramTable;
use crate::table::ram_table::RamTable;
use crate::table::table_column::*;
use crate::table::u32_table::U32Table;
use crate::table::*;

/// The degree of the AIR after the degree lowering step.
///
/// Using substitution and the introduction of new variables, the degree of the AIR as specified
/// in the respective tables
/// (e.g., in [`processor_table::ExtProcessorTable::transition_constraints`])
/// is lowered to this value.
/// For example, with a target degree of 2 and a (fictional) constraint of the form
/// `a = b²·c²·d`,
/// the degree lowering step could (as one among multiple possibilities)
/// - introduce new variables `e`, `f`, and `g`,
/// - introduce new constraints `e = b²`, `f = c²`, and `g = e·f`,
/// - replace the original constraint with `a = g·d`.
///
/// The degree lowering happens in the constraint evaluation generator.
/// It can be executed by running `cargo run --bin constraint-evaluation-generator`.
/// Executing the constraint evaluator is a prerequisite for running both the Stark prover
/// and the Stark verifier.
///
/// The new variables introduced by the degree lowering step are called “derived columns.”
/// They are added to the [`DegreeLoweringTable`], whose sole purpose is to store the values
/// of these derived columns.
pub const AIR_TARGET_DEGREE: isize = 4;

/// The total number of base columns across all tables.
pub const NUM_BASE_COLUMNS: usize = program_table::BASE_WIDTH
    + processor_table::BASE_WIDTH
    + op_stack_table::BASE_WIDTH
    + ram_table::BASE_WIDTH
    + jump_stack_table::BASE_WIDTH
    + hash_table::BASE_WIDTH
    + cascade_table::BASE_WIDTH
    + lookup_table::BASE_WIDTH
    + u32_table::BASE_WIDTH
    + degree_lowering_table::BASE_WIDTH;

const NUM_EXT_COLUMNS_WITHOUT_RANDOMIZER_POLYS: usize = program_table::EXT_WIDTH
    + processor_table::EXT_WIDTH
    + op_stack_table::EXT_WIDTH
    + ram_table::EXT_WIDTH
    + jump_stack_table::EXT_WIDTH
    + hash_table::EXT_WIDTH
    + cascade_table::EXT_WIDTH
    + lookup_table::EXT_WIDTH
    + u32_table::EXT_WIDTH
    + degree_lowering_table::EXT_WIDTH;

/// The total number of extension columns across all tables.
/// Includes the columns required for [randomizer polynomials](NUM_RANDOMIZER_POLYNOMIALS).
pub const NUM_EXT_COLUMNS: usize =
    NUM_EXT_COLUMNS_WITHOUT_RANDOMIZER_POLYS + NUM_RANDOMIZER_POLYNOMIALS;

/// The total number of columns across all tables.
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
pub const DEGREE_LOWERING_TABLE_START: usize = U32_TABLE_END;
pub const DEGREE_LOWERING_TABLE_END: usize =
    DEGREE_LOWERING_TABLE_START + degree_lowering_table::BASE_WIDTH;

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
pub const EXT_DEGREE_LOWERING_TABLE_START: usize = EXT_U32_TABLE_END;
pub const EXT_DEGREE_LOWERING_TABLE_END: usize =
    EXT_DEGREE_LOWERING_TABLE_START + degree_lowering_table::EXT_WIDTH;

const NUM_TABLES_WITHOUT_DEGREE_LOWERING: usize = TableId::COUNT - 1;

/// A `TableId` uniquely determines one of Triton VM's tables.
#[derive(Debug, Display, Copy, Clone, Eq, PartialEq, Hash, EnumCount, EnumIter, Arbitrary)]
pub enum TableId {
    Program,
    Processor,
    OpStack,
    Ram,
    JumpStack,
    Hash,
    Cascade,
    Lookup,
    U32,
    DegreeLowering,
}

/// A Master Table is, in some sense, a top-level table of Triton VM. It contains all the data
/// but little logic beyond bookkeeping and presenting the data in useful ways. Conversely, the
/// individual tables contain no data but all the respective logic. Master Tables are
/// responsible for managing the individual tables and for presenting the right data to the right
/// tables, serving as a clean interface between the VM and the individual tables.
///
/// As a mental model, it is perfectly fine to think of the data for the individual tables as
/// completely separate from each other. Only the [cross-table arguments][cross_arg] link all tables
/// together.
///
/// Conceptually, there are two Master Tables: the [`MasterBaseTable`] ("main"), the Master
/// Extension Table ("auxiliary"). The lifecycle of the Master Tables is
/// as follows:
/// 1. The [`MasterBaseTable`] is instantiated and filled using the Algebraic Execution Trace.
/// 2. The [`MasterBaseTable`] is padded using logic from the individual tables.
/// 3. The still-empty entries in the [`MasterBaseTable`] are filled with random elements. This
///     step is also known as “trace randomization.”
/// 4. If there is enough RAM, then each column of the [`MasterBaseTable`] is low-degree extended.
///    The results are stored on the [`MasterBaseTable`] for quick access later.
///    If there is not enough RAM, then the low-degree extensions of the trace columns will be
///    computed and sometimes recomputed just-in-time, and the memory freed afterward.
///    The caching behavior [can be forced][overwrite_cache].
/// 5. The [`MasterBaseTable`] is used to derive the [`MasterExtensionTable`][master_ext_table]
///     using logic from the individual tables.
/// 6. The [`MasterExtensionTable`][master_ext_table] is trace-randomized.
/// 7. Each column of the [`MasterExtensionTable`][master_ext_table] is [low-degree extended][lde].
///     The effects are the same as for the [`MasterBaseTable`].
/// 8. Using the [`MasterBaseTable`] and the [`MasterExtensionTable`][master_ext_table], the
///     [quotient codeword][master_quot_table] is derived using the AIR. Each individual table
///     defines that part of the AIR that is relevant to it.
///
/// The following points are of note:
/// - The [`MasterExtensionTable`][master_ext_table]'s rightmost columns are the randomizer
///     codewords. These are necessary for zero-knowledge.
/// - The cross-table argument has zero width for the [`MasterBaseTable`] and
///   [`MasterExtensionTable`][master_ext_table] but does induce a nonzero number of constraints
///   and thus terms in the [quotient combination][all_quotients_combined].
///
/// [cross_arg]: cross_table_argument::GrandCrossTableArg
/// [overwrite_cache]: crate::config::overwrite_lde_trace_caching_to
/// [lde]: Self::low_degree_extend_all_columns
/// [quot_table]: Self::quotient_domain_table
/// [master_ext_table]: MasterExtTable
/// [master_quot_table]: all_quotients_combined
pub trait MasterTable<FF>: Sync
where
    FF: FiniteField
        + MulAssign<BFieldElement>
        + From<BFieldElement>
        + BFieldCodec
        + Mul<BFieldElement, Output = FF>,
    Standard: Distribution<FF>,
{
    const NUM_COLUMNS: usize;

    fn trace_domain(&self) -> ArithmeticDomain;
    fn randomized_trace_domain(&self) -> ArithmeticDomain;

    /// The [`ArithmeticDomain`] _just_ large enough to compute
    /// [all quotients](all_quotients_combined).
    fn quotient_domain(&self) -> ArithmeticDomain;

    /// The [`ArithmeticDomain`] large enough for [`FRI`](crate::fri::Fri).
    fn fri_domain(&self) -> ArithmeticDomain;

    /// The [`ArithmeticDomain`] to [low-degree extend](Self::low_degree_extend_all_columns) into.
    /// The larger of the [`quotient_domain`](Self::quotient_domain) and the
    /// [`fri_domain`](Self::fri_domain).
    fn evaluation_domain(&self) -> ArithmeticDomain {
        if self.quotient_domain().length > self.fri_domain().length {
            self.quotient_domain()
        } else {
            self.fri_domain()
        }
    }

    /// Presents underlying trace data, excluding trace randomizers and randomizer polynomials.
    fn trace_table(&self) -> ArrayView2<FF>;

    /// Mutably presents underlying trace data, excluding trace randomizers and randomizer
    /// polynomials.
    fn trace_table_mut(&mut self) -> ArrayViewMut2<FF>;

    fn randomized_trace_table(&self) -> ArrayView2<FF>;

    fn randomized_trace_table_mut(&mut self) -> ArrayViewMut2<FF>;

    /// The quotient-domain view of the cached low-degree-extended table, if
    /// 1. the table has been [low-degree extended][lde], and
    /// 2. the low-degree-extended table [has been cached][cache].
    ///
    /// [lde]: Self::low_degree_extend_all_columns
    /// [cache]: crate::config::overwrite_lde_trace_caching_to
    // This cannot be implemented generically on the trait because it returns a
    // pointer to an array that must live somewhere and cannot live on the stack.
    // From the trait implementation we cannot access the implementing object's
    // fields.
    fn quotient_domain_table(&self) -> Option<ArrayView2<FF>>;

    /// Set all rows _not_ part of the actual (padded) trace to random values.
    fn randomize_trace(&mut self) {
        let unit_distance = self.randomized_trace_domain().length / self.trace_domain().length;
        (1..unit_distance).for_each(|offset| {
            self.randomized_trace_table_mut()
                .slice_mut(s![offset..; unit_distance, ..])
                .par_mapv_inplace(|_| random::<FF>())
        });
    }

    /// Low-degree extend all columns of the randomized trace domain table. The resulting
    /// low-degree extended columns can be accessed using [`quotient_domain_table`][table]
    /// if it is cached; see [`overwrite_lde_trace_caching_to`][cache].
    ///
    /// [table]: Self::quotient_domain_table
    /// [cache]: crate::config::overwrite_lde_trace_caching_to
    fn low_degree_extend_all_columns(&mut self) {
        let evaluation_domain = self.evaluation_domain();
        let randomized_trace_domain = self.randomized_trace_domain();
        let num_rows = evaluation_domain.length;
        let mut interpolation_polynomials = Array1::zeros(Self::NUM_COLUMNS);

        // compute interpolants
        Zip::from(self.randomized_trace_table().axis_iter(Axis(1)))
            .and(interpolation_polynomials.axis_iter_mut(Axis(0)))
            .par_for_each(|trace_column, poly| {
                let trace_column = trace_column.as_slice().unwrap();
                let interpolation_polynomial = randomized_trace_domain.interpolate(trace_column);
                Array0::from_elem((), interpolation_polynomial).move_into(poly);
            });

        let mut extended_trace = Vec::<FF>::new();
        let num_elements = num_rows * Self::NUM_COLUMNS;
        let should_cache = crate::config::cache_lde_trace().map_or_else(
            || extended_trace.try_reserve_exact(num_elements).is_ok(),
            |cd| cd == CacheDecision::Cache,
        );

        if should_cache {
            extended_trace.resize(num_elements, FF::zero());
            let mut extended_columns =
                Array2::from_shape_vec([num_rows, Self::NUM_COLUMNS], extended_trace).unwrap();
            Zip::from(extended_columns.axis_iter_mut(Axis(1)))
                .and(interpolation_polynomials.axis_iter(Axis(0)))
                .par_for_each(|lde_column, interpolant| {
                    let lde_codeword = evaluation_domain.evaluate(&interpolant[()]);
                    Array1::from(lde_codeword).move_into(lde_column);
                });
            self.memoize_low_degree_extended_table(extended_columns);
        }

        self.memoize_interpolation_polynomials(interpolation_polynomials);
    }

    /// Not intended for direct use, but through [`Self::low_degree_extend_all_columns`].
    #[doc(hidden)]
    fn memoize_low_degree_extended_table(&mut self, low_degree_extended_columns: Array2<FF>);

    /// Return the cached low-degree-extended table, if any.
    fn low_degree_extended_table(&self) -> Option<ArrayView2<FF>>;

    /// Return the FRI domain view of the cached low-degree-extended table, if any.
    ///
    /// This method cannot be implemented generically on the trait because it returns a pointer to
    /// an array and that array has to live somewhere; it cannot live on stack and from the trait
    /// implementation we cannot access the implementing object's fields.
    fn fri_domain_table(&self) -> Option<ArrayView2<FF>>;

    /// Memoize the polynomials interpolating the columns.
    /// Not intended for direct use, but through [`Self::low_degree_extend_all_columns`].
    #[doc(hidden)]
    fn memoize_interpolation_polynomials(
        &mut self,
        interpolation_polynomials: Array1<Polynomial<FF>>,
    );

    /// Requires having called
    /// [`low_degree_extend_all_columns`](Self::low_degree_extend_all_columns) first.    
    fn interpolation_polynomials(&self) -> ArrayView1<Polynomial<FF>>;

    /// Get one row of the table at an arbitrary index. Notably, the index does not have to be in
    /// any of the domains. In other words, can be used to compute out-of-domain rows. Requires
    /// having called [`low_degree_extend_all_columns`](Self::low_degree_extend_all_columns) first.
    /// Does not include randomizer polynomials.
    fn out_of_domain_row(&self, indeterminate: XFieldElement) -> Array1<XFieldElement>;

    /// Compute a Merkle tree of the FRI domain table. Every row gives one leaf in the tree.
    /// The function [`hash_row`](Self::hash_one_row) is used to hash each row.
    fn merkle_tree(&self) -> MerkleTree<Tip5> {
        profiler!(start "leafs");
        let hashed_rows = self.hash_all_fri_domain_rows();
        profiler!(stop "leafs");

        profiler!(start "Merkle tree");
        let merkle_tree = CpuParallel::from_digests(&hashed_rows).unwrap();
        profiler!(stop "Merkle tree");

        merkle_tree
    }

    fn hash_all_fri_domain_rows(&self) -> Vec<Digest> {
        if let Some(fri_domain_table) = self.fri_domain_table() {
            let all_rows = fri_domain_table.axis_iter(Axis(0)).into_par_iter();
            all_rows.map(Self::hash_one_row).collect()
        } else {
            self.hash_all_fri_domain_rows_just_in_time()
        }
    }

    fn hash_one_row(row: ArrayView1<FF>) -> Digest {
        Tip5::hash_varlen(&row.iter().flat_map(|e| e.encode()).collect_vec())
    }

    /// Hash all FRI domain rows of the table using just-in-time low-degree-extension, assuming this
    /// low-degree-extended table is not stored in cache.
    ///
    /// Has reduced memory footprint but increased computation time compared to a table with a
    /// cached low-degree extended trace.
    fn hash_all_fri_domain_rows_just_in_time(&self) -> Vec<Digest> {
        // Iterate over the table's columns in batches of `num_threads`. After a batch of columns is
        // low-degree-extended and absorbed into the sponge state, the memory is released and can be
        // reused in the next iteration.

        let num_threads = std::thread::available_parallelism()
            .map(|x| x.get())
            .unwrap_or(1);
        let fri_domain = self.fri_domain();
        let mut sponge_states = vec![SpongeWithPendingAbsorb::new(); fri_domain.length];
        let interpolants = self.interpolation_polynomials();

        let mut codewords = Array2::zeros([fri_domain.length, num_threads]);
        for interpolants_chunk in interpolants.axis_chunks_iter(Axis(0), num_threads) {
            let mut codewords = codewords.slice_mut(s![.., 0..interpolants_chunk.len()]);
            Zip::from(codewords.axis_iter_mut(Axis(1)))
                .and(interpolants_chunk.axis_iter(Axis(0)))
                .par_for_each(|codeword, interpolant| {
                    let lde_codeword = fri_domain.evaluate(&interpolant[()]);
                    Array1::from(lde_codeword).move_into(codeword);
                });
            sponge_states
                .par_iter_mut()
                .zip(codewords.axis_iter(Axis(0)))
                .for_each(|(sponge, row)| sponge.absorb(row.iter().flat_map(|e| e.encode())));
        }

        sponge_states
            .into_par_iter()
            .map(|sponge| sponge.finalize())
            .collect()
    }
}

/// Helper struct and function to absorb however many elements are available; used in
/// the context of hashing rows in a streaming fashion.
#[derive(Clone)]
struct SpongeWithPendingAbsorb {
    sponge: Tip5,

    /// A re-usable buffer of pending input elements.
    /// Only the first [`Self::num_symbols_pending`] elements are valid.
    pending_input: [BFieldElement; RATE],
    num_symbols_pending: usize,
}

impl SpongeWithPendingAbsorb {
    pub fn new() -> Self {
        Self {
            sponge: Tip5::new(algebraic_hasher::Domain::VariableLength),
            pending_input: bfe_array![0; RATE],
            num_symbols_pending: 0,
        }
    }

    /// Similar to [`Tip5::absorb`] but buffers input elements until a full block is available.
    pub fn absorb<I>(&mut self, some_input: I)
    where
        I: IntoIterator,
        I::Item: Borrow<BFieldElement>,
    {
        for symbol in some_input {
            let &symbol = symbol.borrow();
            self.pending_input[self.num_symbols_pending] = symbol;
            self.num_symbols_pending += 1;
            if self.num_symbols_pending == RATE {
                self.num_symbols_pending = 0;
                self.sponge.absorb(self.pending_input);
            }
        }
    }

    pub fn finalize(mut self) -> Digest {
        // apply padding
        self.pending_input[self.num_symbols_pending] = BFieldElement::one();
        for i in self.num_symbols_pending + 1..RATE {
            self.pending_input[i] = BFieldElement::zero();
        }
        self.sponge.absorb(self.pending_input);
        self.num_symbols_pending = 0;

        self.sponge.squeeze()[0..DIGEST_LENGTH]
            .to_vec()
            .try_into()
            .unwrap()
    }
}

/// See [`MasterTable`].
#[derive(Debug, Clone)]
pub struct MasterBaseTable {
    pub num_trace_randomizers: usize,

    program_table_len: usize,
    main_execution_len: usize,
    op_stack_table_len: usize,
    ram_table_len: usize,
    hash_coprocessor_execution_len: usize,
    cascade_table_len: usize,
    u32_coprocesor_execution_len: usize,

    trace_domain: ArithmeticDomain,
    randomized_trace_domain: ArithmeticDomain,
    quotient_domain: ArithmeticDomain,
    fri_domain: ArithmeticDomain,

    randomized_trace_table: Array2<BFieldElement>,
    low_degree_extended_table: Option<Array2<BFieldElement>>,
    interpolation_polynomials: Option<Array1<Polynomial<BFieldElement>>>,
}

/// See [`MasterTable`].
#[derive(Debug, Clone)]
pub struct MasterExtTable {
    pub num_trace_randomizers: usize,

    trace_domain: ArithmeticDomain,
    randomized_trace_domain: ArithmeticDomain,
    quotient_domain: ArithmeticDomain,
    fri_domain: ArithmeticDomain,

    randomized_trace_table: Array2<XFieldElement>,
    low_degree_extended_table: Option<Array2<XFieldElement>>,
    interpolation_polynomials: Option<Array1<Polynomial<XFieldElement>>>,
}

impl MasterTable<BFieldElement> for MasterBaseTable {
    const NUM_COLUMNS: usize = NUM_BASE_COLUMNS;

    fn trace_domain(&self) -> ArithmeticDomain {
        self.trace_domain
    }

    fn randomized_trace_domain(&self) -> ArithmeticDomain {
        self.randomized_trace_domain
    }

    fn quotient_domain(&self) -> ArithmeticDomain {
        self.quotient_domain
    }

    fn fri_domain(&self) -> ArithmeticDomain {
        self.fri_domain
    }

    fn trace_table(&self) -> ArrayView2<BFieldElement> {
        let unit_distance = self.randomized_trace_domain().length / self.trace_domain().length;
        self.randomized_trace_table.slice(s![..; unit_distance, ..])
    }

    fn trace_table_mut(&mut self) -> ArrayViewMut2<BFieldElement> {
        let unit_distance = self.randomized_trace_domain().length / self.trace_domain().length;
        self.randomized_trace_table
            .slice_mut(s![..; unit_distance, ..])
    }

    fn randomized_trace_table(&self) -> ArrayView2<BFieldElement> {
        self.randomized_trace_table.view()
    }

    fn randomized_trace_table_mut(&mut self) -> ArrayViewMut2<BFieldElement> {
        self.randomized_trace_table.view_mut()
    }

    fn quotient_domain_table(&self) -> Option<ArrayView2<BFieldElement>> {
        let table = &self.low_degree_extended_table.as_ref()?;
        let nrows = table.nrows();
        if self.quotient_domain.length < nrows {
            let unit_distance = nrows / self.quotient_domain.length;
            Some(table.slice(s![0..nrows;unit_distance, ..]))
        } else {
            Some(table.view())
        }
    }

    fn memoize_low_degree_extended_table(
        &mut self,
        low_degree_extended_columns: Array2<BFieldElement>,
    ) {
        self.low_degree_extended_table = Some(low_degree_extended_columns);
    }

    fn low_degree_extended_table(&self) -> Option<ArrayView2<BFieldElement>> {
        let low_degree_extended_table = self.low_degree_extended_table.as_ref()?;
        Some(low_degree_extended_table.view())
    }

    fn fri_domain_table(&self) -> Option<ArrayView2<BFieldElement>> {
        let table = self.low_degree_extended_table.as_ref()?;
        let nrows = table.nrows();
        if nrows > self.fri_domain.length {
            let unit_step = nrows / self.fri_domain.length;
            Some(table.slice(s![0..nrows;unit_step, ..]))
        } else {
            Some(table.view())
        }
    }

    fn memoize_interpolation_polynomials(
        &mut self,
        interpolation_polynomials: Array1<Polynomial<BFieldElement>>,
    ) {
        self.interpolation_polynomials = Some(interpolation_polynomials);
    }

    fn interpolation_polynomials(&self) -> ArrayView1<Polynomial<BFieldElement>> {
        let Some(interpolation_polynomials) = &self.interpolation_polynomials else {
            panic!("Interpolation polynomials must be computed first.");
        };
        interpolation_polynomials.view()
    }

    fn out_of_domain_row(&self, indeterminate: XFieldElement) -> Array1<XFieldElement> {
        // Evaluate a base field polynomial in an extension field point. Manual re-implementation
        // to overcome the lack of the corresponding functionality in `twenty-first`.
        let evaluate = |bfp: &Polynomial<_>, x| {
            let mut acc = XFieldElement::zero();
            for &coefficient in bfp.coefficients.iter().rev() {
                acc *= x;
                acc += coefficient;
            }
            acc
        };

        self.interpolation_polynomials()
            .into_par_iter()
            .map(|polynomial| evaluate(polynomial, indeterminate))
            .collect::<Vec<_>>()
            .into()
    }
}

impl MasterTable<XFieldElement> for MasterExtTable {
    const NUM_COLUMNS: usize = NUM_EXT_COLUMNS;

    fn trace_domain(&self) -> ArithmeticDomain {
        self.trace_domain
    }

    fn randomized_trace_domain(&self) -> ArithmeticDomain {
        self.randomized_trace_domain
    }

    fn quotient_domain(&self) -> ArithmeticDomain {
        self.quotient_domain
    }

    fn fri_domain(&self) -> ArithmeticDomain {
        self.fri_domain
    }

    fn trace_table(&self) -> ArrayView2<XFieldElement> {
        let unit_distance = self.randomized_trace_domain().length / self.trace_domain().length;
        self.randomized_trace_table
            .slice(s![..; unit_distance, ..NUM_EXT_COLUMNS])
    }

    fn trace_table_mut(&mut self) -> ArrayViewMut2<XFieldElement> {
        let unit_distance = self.randomized_trace_domain().length / self.trace_domain().length;
        self.randomized_trace_table
            .slice_mut(s![..; unit_distance, ..NUM_EXT_COLUMNS])
    }

    fn randomized_trace_table(&self) -> ArrayView2<XFieldElement> {
        self.randomized_trace_table.view()
    }

    fn randomized_trace_table_mut(&mut self) -> ArrayViewMut2<XFieldElement> {
        self.randomized_trace_table.view_mut()
    }

    fn quotient_domain_table(&self) -> Option<ArrayView2<XFieldElement>> {
        let table = self.low_degree_extended_table.as_ref()?;
        let nrows = table.nrows();
        if nrows > self.quotient_domain.length {
            let unit_distance = nrows / self.quotient_domain.length;
            Some(table.slice(s![0..nrows;unit_distance, ..]))
        } else {
            Some(table.view())
        }
    }

    fn memoize_low_degree_extended_table(
        &mut self,
        low_degree_extended_columns: Array2<XFieldElement>,
    ) {
        self.low_degree_extended_table = Some(low_degree_extended_columns);
    }

    fn low_degree_extended_table(&self) -> Option<ArrayView2<XFieldElement>> {
        let low_degree_extended_table = self.low_degree_extended_table.as_ref()?;
        Some(low_degree_extended_table.view())
    }

    fn fri_domain_table(&self) -> Option<ArrayView2<XFieldElement>> {
        let table = self.low_degree_extended_table.as_ref()?;
        let nrows = table.nrows();
        if nrows > self.fri_domain.length {
            let unit_step = nrows / self.fri_domain.length;
            Some(table.slice(s![0..nrows;unit_step, ..]))
        } else {
            Some(table.view())
        }
    }

    fn memoize_interpolation_polynomials(
        &mut self,
        interpolation_polynomials: Array1<Polynomial<XFieldElement>>,
    ) {
        self.interpolation_polynomials = Some(interpolation_polynomials);
    }

    fn interpolation_polynomials(&self) -> ArrayView1<Polynomial<XFieldElement>> {
        let Some(interpolation_polynomials) = &self.interpolation_polynomials else {
            panic!("Interpolation polynomials must be computed first.");
        };
        interpolation_polynomials.view()
    }

    fn out_of_domain_row(&self, indeterminate: XFieldElement) -> Array1<XFieldElement> {
        self.interpolation_polynomials()
            .slice(s![..NUM_EXT_COLUMNS])
            .into_par_iter()
            .map(|polynomial| polynomial.evaluate(indeterminate))
            .collect::<Vec<_>>()
            .into()
    }
}

type PadFunction = fn(ArrayViewMut2<BFieldElement>, usize);
type ExtendFunction = fn(ArrayView2<BFieldElement>, ArrayViewMut2<XFieldElement>, &Challenges);

impl MasterBaseTable {
    pub fn new(
        aet: &AlgebraicExecutionTrace,
        num_trace_randomizers: usize,
        quotient_domain: ArithmeticDomain,
        fri_domain: ArithmeticDomain,
    ) -> Self {
        let padded_height = aet.padded_height();
        let trace_domain = ArithmeticDomain::of_length(padded_height).unwrap();

        let randomized_padded_trace_len =
            randomized_padded_trace_len(padded_height, num_trace_randomizers);
        let randomized_trace_domain =
            ArithmeticDomain::of_length(randomized_padded_trace_len).unwrap();

        let num_rows = randomized_padded_trace_len;
        let num_columns = NUM_BASE_COLUMNS;
        let randomized_trace_table = Array2::zeros([num_rows, num_columns].f());

        let mut master_base_table = Self {
            num_trace_randomizers,
            program_table_len: aet.height_of_table(TableId::Program),
            main_execution_len: aet.height_of_table(TableId::Processor),
            op_stack_table_len: aet.height_of_table(TableId::OpStack),
            ram_table_len: aet.height_of_table(TableId::Ram),
            hash_coprocessor_execution_len: aet.height_of_table(TableId::Hash),
            cascade_table_len: aet.height_of_table(TableId::Cascade),
            u32_coprocesor_execution_len: aet.height_of_table(TableId::U32),
            trace_domain,
            randomized_trace_domain,
            quotient_domain,
            fri_domain,
            randomized_trace_table,
            low_degree_extended_table: None,
            interpolation_polynomials: None,
        };

        // memory-like tables must be filled in before clock jump differences are known, hence
        // the break from the usual order
        let clk_jump_diffs_op_stack =
            OpStackTable::fill_trace(&mut master_base_table.table_mut(TableId::OpStack), aet);
        let clk_jump_diffs_ram =
            RamTable::fill_trace(&mut master_base_table.table_mut(TableId::Ram), aet);
        let clk_jump_diffs_jump_stack =
            JumpStackTable::fill_trace(&mut master_base_table.table_mut(TableId::JumpStack), aet);

        let processor_table = &mut master_base_table.table_mut(TableId::Processor);
        ProcessorTable::fill_trace(
            processor_table,
            aet,
            &clk_jump_diffs_op_stack,
            &clk_jump_diffs_ram,
            &clk_jump_diffs_jump_stack,
        );

        ProgramTable::fill_trace(&mut master_base_table.table_mut(TableId::Program), aet);
        HashTable::fill_trace(&mut master_base_table.table_mut(TableId::Hash), aet);
        CascadeTable::fill_trace(&mut master_base_table.table_mut(TableId::Cascade), aet);
        LookupTable::fill_trace(&mut master_base_table.table_mut(TableId::Lookup), aet);
        U32Table::fill_trace(&mut master_base_table.table_mut(TableId::U32), aet);

        // Filling the degree-lowering table only makes sense after padding has happened.
        // Hence, this table is omitted here.

        master_base_table
    }

    /// Pad the trace to the next power of two using the various, table-specific padding rules.
    /// All tables must have the same height for reasons of verifier efficiency.
    /// Furthermore, that height must be a power of two for reasons of prover efficiency.
    /// Concretely, the Number Theory Transform (NTT) performed by the prover is particularly
    /// efficient over the used base field when the number of rows is a power of two.
    pub fn pad(&mut self) {
        let table_lengths = self.all_table_lengths();

        // Due to limitations in ndarray, a 10-way multi-slice is not possible. Hence, (1) slicing
        // has to be done in multiple steps, and (2) cannot be put into a method.
        let unit_distance = self.randomized_trace_domain().length / self.trace_domain().length;
        let mut master_table_without_randomizers = self
            .randomized_trace_table
            .slice_mut(s![..; unit_distance, ..]);
        let (program_table, mut rest) = master_table_without_randomizers.multi_slice_mut((
            s![.., ..ProgramBaseTableColumn::COUNT],
            s![.., ProgramBaseTableColumn::COUNT..],
        ));
        let (processor_table, mut rest) = rest.multi_slice_mut((
            s![.., ..ProcessorBaseTableColumn::COUNT],
            s![.., ProcessorBaseTableColumn::COUNT..],
        ));
        let (op_stack_table, mut rest) = rest.multi_slice_mut((
            s![.., ..OpStackBaseTableColumn::COUNT],
            s![.., OpStackBaseTableColumn::COUNT..],
        ));
        let (ram_table, mut rest) = rest.multi_slice_mut((
            s![.., ..RamBaseTableColumn::COUNT],
            s![.., RamBaseTableColumn::COUNT..],
        ));
        let (jump_stack_table, mut rest) = rest.multi_slice_mut((
            s![.., ..JumpStackBaseTableColumn::COUNT],
            s![.., JumpStackBaseTableColumn::COUNT..],
        ));
        let (hash_table, mut rest) = rest.multi_slice_mut((
            s![.., ..HashBaseTableColumn::COUNT],
            s![.., HashBaseTableColumn::COUNT..],
        ));
        let (cascade_table, mut rest) = rest.multi_slice_mut((
            s![.., ..CascadeBaseTableColumn::COUNT],
            s![.., CascadeBaseTableColumn::COUNT..],
        ));
        let (lookup_table, mut rest) = rest.multi_slice_mut((
            s![.., ..LookupBaseTableColumn::COUNT],
            s![.., LookupBaseTableColumn::COUNT..],
        ));
        let (u32_table, _) = rest.multi_slice_mut((
            s![.., ..U32BaseTableColumn::COUNT],
            s![.., U32BaseTableColumn::COUNT..],
        ));

        let base_tables = [
            program_table,
            processor_table,
            op_stack_table,
            ram_table,
            jump_stack_table,
            hash_table,
            cascade_table,
            lookup_table,
            u32_table,
        ];

        Self::all_pad_functions()
            .into_par_iter()
            .zip_eq(base_tables.into_par_iter())
            .zip_eq(table_lengths.into_par_iter())
            .for_each(|((pad, base_table), table_length)| {
                pad(base_table, table_length);
            });

        DegreeLoweringTable::fill_derived_base_columns(self.trace_table_mut());
    }

    fn all_pad_functions() -> [PadFunction; NUM_TABLES_WITHOUT_DEGREE_LOWERING] {
        [
            ProgramTable::pad_trace,
            ProcessorTable::pad_trace,
            OpStackTable::pad_trace,
            RamTable::pad_trace,
            JumpStackTable::pad_trace,
            HashTable::pad_trace,
            CascadeTable::pad_trace,
            LookupTable::pad_trace,
            U32Table::pad_trace,
        ]
    }

    fn all_table_lengths(&self) -> [usize; NUM_TABLES_WITHOUT_DEGREE_LOWERING] {
        let processor_table_len = self.main_execution_len;
        let jump_stack_table_len = self.main_execution_len;

        [
            self.program_table_len,
            processor_table_len,
            self.op_stack_table_len,
            self.ram_table_len,
            jump_stack_table_len,
            self.hash_coprocessor_execution_len,
            self.cascade_table_len,
            AlgebraicExecutionTrace::LOOKUP_TABLE_HEIGHT,
            self.u32_coprocesor_execution_len,
        ]
    }

    /// Create a `MasterExtTable` from a `MasterBaseTable` by `.extend()`ing each individual base
    /// table. The `.extend()` for each table is specific to that table, but always involves
    /// adding some number of columns.
    pub fn extend(&self, challenges: &Challenges) -> MasterExtTable {
        // randomizer polynomials
        let num_rows = self.randomized_trace_table().nrows();
        let mut randomized_trace_extension_table = Array2::zeros([num_rows, NUM_EXT_COLUMNS].f());
        randomized_trace_extension_table
            .slice_mut(s![.., NUM_EXT_COLUMNS_WITHOUT_RANDOMIZER_POLYS..])
            .par_mapv_inplace(|_| random::<XFieldElement>());

        let mut master_ext_table = MasterExtTable {
            num_trace_randomizers: self.num_trace_randomizers,
            trace_domain: self.trace_domain(),
            randomized_trace_domain: self.randomized_trace_domain(),
            quotient_domain: self.quotient_domain(),
            fri_domain: self.fri_domain(),
            randomized_trace_table: randomized_trace_extension_table,
            low_degree_extended_table: None,
            interpolation_polynomials: None,
        };

        // Due to limitations in ndarray, a 10-way multi-slice is not possible. Hence, (1) slicing
        // has to be done in multiple steps, and (2) cannot be put into a method.
        let unit_distance = self.randomized_trace_domain().length / self.trace_domain().length;
        let mut master_ext_table_without_randomizers = master_ext_table
            .randomized_trace_table
            .slice_mut(s![..; unit_distance, ..NUM_EXT_COLUMNS]);
        let (program_table, mut rest) = master_ext_table_without_randomizers.multi_slice_mut((
            s![.., ..ProgramExtTableColumn::COUNT],
            s![.., ProgramExtTableColumn::COUNT..],
        ));
        let (processor_table, mut rest) = rest.multi_slice_mut((
            s![.., ..ProcessorExtTableColumn::COUNT],
            s![.., ProcessorExtTableColumn::COUNT..],
        ));
        let (op_stack_table, mut rest) = rest.multi_slice_mut((
            s![.., ..OpStackExtTableColumn::COUNT],
            s![.., OpStackExtTableColumn::COUNT..],
        ));
        let (ram_table, mut rest) = rest.multi_slice_mut((
            s![.., ..RamExtTableColumn::COUNT],
            s![.., RamExtTableColumn::COUNT..],
        ));
        let (jump_stack_table, mut rest) = rest.multi_slice_mut((
            s![.., ..JumpStackExtTableColumn::COUNT],
            s![.., JumpStackExtTableColumn::COUNT..],
        ));
        let (hash_table, mut rest) = rest.multi_slice_mut((
            s![.., ..HashExtTableColumn::COUNT],
            s![.., HashExtTableColumn::COUNT..],
        ));
        let (cascade_table, mut rest) = rest.multi_slice_mut((
            s![.., ..CascadeExtTableColumn::COUNT],
            s![.., CascadeExtTableColumn::COUNT..],
        ));
        let (lookup_table, mut rest) = rest.multi_slice_mut((
            s![.., ..LookupExtTableColumn::COUNT],
            s![.., LookupExtTableColumn::COUNT..],
        ));
        let u32_table = rest.slice_mut(s![.., ..U32ExtTableColumn::COUNT]);

        let extension_tables = [
            program_table,
            processor_table,
            op_stack_table,
            ram_table,
            jump_stack_table,
            hash_table,
            cascade_table,
            lookup_table,
            u32_table,
        ];

        Self::all_extend_functions()
            .into_par_iter()
            .zip_eq(self.base_tables_for_extending().into_par_iter())
            .zip_eq(extension_tables.into_par_iter())
            .for_each(|((extend, base_table), ext_table)| {
                extend(base_table, ext_table, challenges)
            });

        DegreeLoweringTable::fill_derived_ext_columns(
            self.trace_table(),
            master_ext_table.trace_table_mut(),
            challenges,
        );

        master_ext_table
    }

    fn all_extend_functions() -> [ExtendFunction; NUM_TABLES_WITHOUT_DEGREE_LOWERING] {
        [
            ProgramTable::extend,
            ProcessorTable::extend,
            OpStackTable::extend,
            RamTable::extend,
            JumpStackTable::extend,
            HashTable::extend,
            CascadeTable::extend,
            LookupTable::extend,
            U32Table::extend,
        ]
    }

    fn base_tables_for_extending(
        &self,
    ) -> [ArrayView2<BFieldElement>; NUM_TABLES_WITHOUT_DEGREE_LOWERING] {
        [
            self.table(TableId::Program),
            self.table(TableId::Processor),
            self.table(TableId::OpStack),
            self.table(TableId::Ram),
            self.table(TableId::JumpStack),
            self.table(TableId::Hash),
            self.table(TableId::Cascade),
            self.table(TableId::Lookup),
            self.table(TableId::U32),
        ]
    }

    fn column_indices_for_table(id: TableId) -> Range<usize> {
        use TableId::*;
        match id {
            Program => PROGRAM_TABLE_START..PROGRAM_TABLE_END,
            Processor => PROCESSOR_TABLE_START..PROCESSOR_TABLE_END,
            OpStack => OP_STACK_TABLE_START..OP_STACK_TABLE_END,
            Ram => RAM_TABLE_START..RAM_TABLE_END,
            JumpStack => JUMP_STACK_TABLE_START..JUMP_STACK_TABLE_END,
            Hash => HASH_TABLE_START..HASH_TABLE_END,
            Cascade => CASCADE_TABLE_START..CASCADE_TABLE_END,
            Lookup => LOOKUP_TABLE_START..LOOKUP_TABLE_END,
            U32 => U32_TABLE_START..U32_TABLE_END,
            DegreeLowering => DEGREE_LOWERING_TABLE_START..DEGREE_LOWERING_TABLE_END,
        }
    }

    /// A view of the specified table, without any randomizers.
    pub fn table(&self, table_id: TableId) -> ArrayView2<BFieldElement> {
        let column_indices = Self::column_indices_for_table(table_id);
        let unit_distance = self.randomized_trace_domain().length / self.trace_domain().length;
        self.randomized_trace_table
            .slice(s![..; unit_distance, column_indices])
    }

    /// A mutable view of the specified table, without any randomizers.
    pub fn table_mut(&mut self, table_id: TableId) -> ArrayViewMut2<BFieldElement> {
        let column_indices = Self::column_indices_for_table(table_id);
        let unit_distance = self.randomized_trace_domain().length / self.trace_domain().length;
        self.randomized_trace_table
            .slice_mut(s![..; unit_distance, column_indices])
    }

    pub(crate) fn try_to_base_row<T: FiniteField>(
        row: Array1<T>,
    ) -> Result<BaseRow<T>, ProvingError> {
        let err = || ProvingError::TableRowConversionError {
            expected_len: NUM_BASE_COLUMNS,
            actual_len: row.len(),
        };
        row.to_vec().try_into().map_err(|_| err())
    }
}

impl MasterExtTable {
    fn column_indices_for_table(id: TableId) -> Range<usize> {
        use TableId::*;
        match id {
            Program => EXT_PROGRAM_TABLE_START..EXT_PROGRAM_TABLE_END,
            Processor => EXT_PROCESSOR_TABLE_START..EXT_PROCESSOR_TABLE_END,
            OpStack => EXT_OP_STACK_TABLE_START..EXT_OP_STACK_TABLE_END,
            Ram => EXT_RAM_TABLE_START..EXT_RAM_TABLE_END,
            JumpStack => EXT_JUMP_STACK_TABLE_START..EXT_JUMP_STACK_TABLE_END,
            Hash => EXT_HASH_TABLE_START..EXT_HASH_TABLE_END,
            Cascade => EXT_CASCADE_TABLE_START..EXT_CASCADE_TABLE_END,
            Lookup => EXT_LOOKUP_TABLE_START..EXT_LOOKUP_TABLE_END,
            U32 => EXT_U32_TABLE_START..EXT_U32_TABLE_END,
            DegreeLowering => EXT_DEGREE_LOWERING_TABLE_START..EXT_DEGREE_LOWERING_TABLE_END,
        }
    }

    /// A view of the specified table, without any randomizers.
    pub fn table(&self, table_id: TableId) -> ArrayView2<XFieldElement> {
        let column_indices = Self::column_indices_for_table(table_id);
        let unit_distance = self.randomized_trace_domain().length / self.trace_domain().length;
        self.randomized_trace_table
            .slice(s![..; unit_distance, column_indices])
    }

    /// A mutable view of the specified table, without any randomizers.
    pub fn table_mut(&mut self, table_id: TableId) -> ArrayViewMut2<XFieldElement> {
        let column_indices = Self::column_indices_for_table(table_id);
        let unit_distance = self.randomized_trace_domain().length / self.trace_domain().length;
        self.randomized_trace_table
            .slice_mut(s![..; unit_distance, column_indices])
    }

    pub(crate) fn try_to_ext_row(row: Array1<XFieldElement>) -> Result<ExtensionRow, ProvingError> {
        let err = || ProvingError::TableRowConversionError {
            expected_len: NUM_EXT_COLUMNS,
            actual_len: row.len(),
        };
        row.to_vec().try_into().map_err(|_| err())
    }
}

pub(crate) fn max_degree_with_origin(
    interpolant_degree: isize,
    padded_height: usize,
) -> DegreeWithOrigin {
    all_degrees_with_origin(interpolant_degree, padded_height)
        .into_iter()
        .max()
        .unwrap()
}

pub fn initial_quotient_zerofier_inverse(
    quotient_domain: ArithmeticDomain,
) -> Array1<BFieldElement> {
    let zerofier_codeword = quotient_domain
        .domain_values()
        .into_iter()
        .map(|x| x - bfe!(1))
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
        .map(|x| x.mod_pow_u32(trace_domain.length as u32) - bfe!(1))
        .collect();
    BFieldElement::batch_inversion(zerofier_codeword).into()
}

pub fn transition_quotient_zerofier_inverse(
    trace_domain: ArithmeticDomain,
    quotient_domain: ArithmeticDomain,
) -> Array1<BFieldElement> {
    let trace_domain_generator_inverse = trace_domain.generator.inverse();
    let quotient_domain_values = quotient_domain.domain_values();

    let subgroup_zerofier: Vec<_> = quotient_domain_values
        .par_iter()
        .map(|domain_value| domain_value.mod_pow_u32(trace_domain.length as u32) - bfe!(1))
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

/// Computes the quotient codeword, which is the randomized linear combination of all individual
/// quotients.
///
/// About assigning weights to quotients: the quotients are ordered by category – initial,
/// consistency, transition, and then terminal. Within each category, the quotients follow the
/// canonical order of the tables. The last column holds the terminal quotient of the cross-table
/// argument, which is strictly speaking not a table.
/// The order of the quotients is not actually important. However, it must be consistent between
/// [prover] and [verifier].
///
/// [prover]: crate::stark::Stark::prove
/// [verifier]: crate::stark::Stark::verify
pub fn all_quotients_combined(
    quotient_domain_master_base_table: ArrayView2<BFieldElement>,
    quotient_domain_master_ext_table: ArrayView2<XFieldElement>,
    trace_domain: ArithmeticDomain,
    quotient_domain: ArithmeticDomain,
    challenges: &Challenges,
    quotient_weights: &[XFieldElement],
) -> Vec<XFieldElement> {
    assert_eq!(
        quotient_domain.length,
        quotient_domain_master_base_table.nrows(),
    );
    assert_eq!(
        quotient_domain.length,
        quotient_domain_master_ext_table.nrows()
    );
    assert_eq!(MasterExtTable::NUM_CONSTRAINTS, quotient_weights.len());

    let init_section_end = MasterExtTable::NUM_INITIAL_CONSTRAINTS;
    let cons_section_end = init_section_end + MasterExtTable::NUM_CONSISTENCY_CONSTRAINTS;
    let tran_section_end = cons_section_end + MasterExtTable::NUM_TRANSITION_CONSTRAINTS;

    profiler!(start "zerofier inverse");
    let initial_zerofier_inverse = initial_quotient_zerofier_inverse(quotient_domain);
    let consistency_zerofier_inverse =
        consistency_quotient_zerofier_inverse(trace_domain, quotient_domain);
    let transition_zerofier_inverse =
        transition_quotient_zerofier_inverse(trace_domain, quotient_domain);
    let terminal_zerofier_inverse =
        terminal_quotient_zerofier_inverse(trace_domain, quotient_domain);
    profiler!(stop "zerofier inverse");

    profiler!(start "evaluate AIR, compute quotient codeword");
    let dot_product = |partial_row: Vec<_>, weights: &[_]| -> XFieldElement {
        let pairs = partial_row.into_iter().zip_eq(weights.iter());
        pairs.map(|(v, &w)| v * w).sum()
    };

    let quotient_codeword = (0..quotient_domain.length)
        .into_par_iter()
        .map(|row_index| {
            let unit_distance = quotient_domain.length / trace_domain.length;
            let next_row_index = (row_index + unit_distance) % quotient_domain.length;
            let current_row_main = quotient_domain_master_base_table.row(row_index);
            let current_row_aux = quotient_domain_master_ext_table.row(row_index);
            let next_row_main = quotient_domain_master_base_table.row(next_row_index);
            let next_row_aux = quotient_domain_master_ext_table.row(next_row_index);

            let initial_constraint_values = MasterExtTable::evaluate_initial_constraints(
                current_row_main,
                current_row_aux,
                challenges,
            );
            let initial_inner_product = dot_product(
                initial_constraint_values,
                &quotient_weights[..init_section_end],
            );
            let mut quotient_value = initial_inner_product * initial_zerofier_inverse[row_index];

            let consistency_constraint_values = MasterExtTable::evaluate_consistency_constraints(
                current_row_main,
                current_row_aux,
                challenges,
            );
            let consistency_inner_product = dot_product(
                consistency_constraint_values,
                &quotient_weights[init_section_end..cons_section_end],
            );
            quotient_value += consistency_inner_product * consistency_zerofier_inverse[row_index];

            let transition_constraint_values = MasterExtTable::evaluate_transition_constraints(
                current_row_main,
                current_row_aux,
                next_row_main,
                next_row_aux,
                challenges,
            );
            let transition_inner_product = dot_product(
                transition_constraint_values,
                &quotient_weights[cons_section_end..tran_section_end],
            );
            quotient_value += transition_inner_product * transition_zerofier_inverse[row_index];

            let terminal_constraint_values = MasterExtTable::evaluate_terminal_constraints(
                current_row_main,
                current_row_aux,
                challenges,
            );
            let terminal_inner_product = dot_product(
                terminal_constraint_values,
                &quotient_weights[tran_section_end..],
            );
            quotient_value += terminal_inner_product * terminal_zerofier_inverse[row_index];
            quotient_value
        })
        .collect();
    profiler!(stop "evaluate AIR, compute quotient codeword");

    quotient_codeword
}

/// Guaranteed to be a power of two.
pub fn randomized_padded_trace_len(padded_height: usize, num_trace_randomizers: usize) -> usize {
    let total_table_length = padded_height + num_trace_randomizers;
    total_table_length.next_power_of_two()
}

pub fn interpolant_degree(padded_height: usize, num_trace_randomizers: usize) -> isize {
    (randomized_padded_trace_len(padded_height, num_trace_randomizers) - 1) as isize
}

#[cfg(test)]
mod tests {
    use fs_err as fs;
    use std::path::Path;

    use master_table::cross_table_argument::GrandCrossTableArg;
    use ndarray::s;
    use ndarray::Array2;
    use num_traits::Zero;
    use proptest::prelude::*;
    use proptest_arbitrary_interop::arb;
    use strum::EnumCount;
    use strum::IntoEnumIterator;
    use test_strategy::proptest;
    use twenty_first::math::b_field_element::BFieldElement;
    use twenty_first::math::traits::FiniteField;
    use twenty_first::prelude::x_field_element::EXTENSION_DEGREE;

    use crate::arithmetic_domain::ArithmeticDomain;
    use crate::shared_tests::ProgramAndInput;
    use crate::stark::tests::*;
    use crate::table::degree_lowering_table::DegreeLoweringBaseTableColumn;
    use crate::table::degree_lowering_table::DegreeLoweringExtTableColumn;
    use crate::table::table_column::*;
    use crate::table::*;
    use crate::triton_program;

    use self::cascade_table::ExtCascadeTable;
    use self::constraint_circuit::ConstraintCircuitBuilder;
    use self::constraint_circuit::ConstraintCircuitMonad;
    use self::constraint_circuit::DualRowIndicator;
    use self::constraint_circuit::SingleRowIndicator;
    use self::hash_table::ExtHashTable;
    use self::jump_stack_table::ExtJumpStackTable;
    use self::lookup_table::ExtLookupTable;
    use self::op_stack_table::ExtOpStackTable;
    use self::processor_table::ExtProcessorTable;
    use self::program_table::ExtProgramTable;
    use self::ram_table::ExtRamTable;
    use self::u32_table::ExtU32Table;

    use super::*;

    #[test]
    fn base_table_width_is_correct() {
        let program = ProgramAndInput::new(triton_program!(halt));
        let (_, _, master_base_table) = master_base_table_for_low_security_level(program);

        assert_eq!(
            program_table::BASE_WIDTH,
            master_base_table.table(TableId::Program).ncols()
        );
        assert_eq!(
            processor_table::BASE_WIDTH,
            master_base_table.table(TableId::Processor).ncols()
        );
        assert_eq!(
            op_stack_table::BASE_WIDTH,
            master_base_table.table(TableId::OpStack).ncols()
        );
        assert_eq!(
            ram_table::BASE_WIDTH,
            master_base_table.table(TableId::Ram).ncols()
        );
        assert_eq!(
            jump_stack_table::BASE_WIDTH,
            master_base_table.table(TableId::JumpStack).ncols()
        );
        assert_eq!(
            hash_table::BASE_WIDTH,
            master_base_table.table(TableId::Hash).ncols()
        );
        assert_eq!(
            cascade_table::BASE_WIDTH,
            master_base_table.table(TableId::Cascade).ncols()
        );
        assert_eq!(
            lookup_table::BASE_WIDTH,
            master_base_table.table(TableId::Lookup).ncols()
        );
        assert_eq!(
            u32_table::BASE_WIDTH,
            master_base_table.table(TableId::U32).ncols()
        );
        assert_eq!(
            degree_lowering_table::BASE_WIDTH,
            master_base_table.table(TableId::DegreeLowering).ncols()
        );
    }

    #[test]
    fn ext_table_width_is_correct() {
        let program = ProgramAndInput::new(triton_program!(halt));
        let (_, _, _, master_ext_table, _) = master_tables_for_low_security_level(program);

        assert_eq!(
            program_table::EXT_WIDTH,
            master_ext_table.table(TableId::Program).ncols()
        );
        assert_eq!(
            processor_table::EXT_WIDTH,
            master_ext_table.table(TableId::Processor).ncols()
        );
        assert_eq!(
            op_stack_table::EXT_WIDTH,
            master_ext_table.table(TableId::OpStack).ncols()
        );
        assert_eq!(
            ram_table::EXT_WIDTH,
            master_ext_table.table(TableId::Ram).ncols()
        );
        assert_eq!(
            jump_stack_table::EXT_WIDTH,
            master_ext_table.table(TableId::JumpStack).ncols()
        );
        assert_eq!(
            hash_table::EXT_WIDTH,
            master_ext_table.table(TableId::Hash).ncols()
        );
        assert_eq!(
            cascade_table::EXT_WIDTH,
            master_ext_table.table(TableId::Cascade).ncols()
        );
        assert_eq!(
            lookup_table::EXT_WIDTH,
            master_ext_table.table(TableId::Lookup).ncols()
        );
        assert_eq!(
            u32_table::EXT_WIDTH,
            master_ext_table.table(TableId::U32).ncols()
        );
        assert_eq!(
            degree_lowering_table::EXT_WIDTH,
            master_ext_table.table(TableId::DegreeLowering).ncols()
        );
        // use some domain-specific knowledge to also check for the randomizer columns
        assert_eq!(
            NUM_RANDOMIZER_POLYNOMIALS,
            master_ext_table
                .randomized_trace_table()
                .slice(s![.., EXT_DEGREE_LOWERING_TABLE_END..])
                .ncols()
        );
    }

    #[test]
    fn zerofiers_are_correct() {
        let big_order = 16;
        let big_offset = BFieldElement::generator();
        let big_domain = ArithmeticDomain::of_length(big_order as usize)
            .unwrap()
            .with_offset(big_offset);

        let small_order = 8;
        let small_domain = ArithmeticDomain::of_length(small_order as usize).unwrap();

        let initial_zerofier_inv = initial_quotient_zerofier_inverse(big_domain);
        let initial_zerofier = BFieldElement::batch_inversion(initial_zerofier_inv.to_vec());
        let initial_zerofier_poly = big_domain.interpolate(&initial_zerofier);
        assert_eq!(big_order as usize, initial_zerofier_inv.len());
        assert_eq!(1, initial_zerofier_poly.degree());
        assert!(initial_zerofier_poly
            .evaluate(small_domain.domain_value(0))
            .is_zero());

        let consistency_zerofier_inv =
            consistency_quotient_zerofier_inverse(small_domain, big_domain);
        let consistency_zerofier =
            BFieldElement::batch_inversion(consistency_zerofier_inv.to_vec());
        let consistency_zerofier_poly = big_domain.interpolate(&consistency_zerofier);
        assert_eq!(big_order as usize, consistency_zerofier_inv.len());
        assert_eq!(small_order as isize, consistency_zerofier_poly.degree());
        for val in small_domain.domain_values() {
            assert!(consistency_zerofier_poly.evaluate(val).is_zero());
        }

        let transition_zerofier_inv =
            transition_quotient_zerofier_inverse(small_domain, big_domain);
        let transition_zerofier = BFieldElement::batch_inversion(transition_zerofier_inv.to_vec());
        let transition_zerofier_poly = big_domain.interpolate(&transition_zerofier);
        assert_eq!(big_order as usize, transition_zerofier_inv.len());
        assert_eq!(small_order as isize - 1, transition_zerofier_poly.degree());
        for &val in small_domain
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
            .evaluate(small_domain.domain_value(small_order as u32 - 1))
            .is_zero());
    }

    fn assert_spec_has(file_path: &Path, snippet: String) {
        let contents = fs::read_to_string(file_path).unwrap().replace("\r\n", "\n");
        assert!(
            contents.contains(&snippet),
            "Could not find correct snippet in file \"{}\".",
            file_path.display(),
        );
    }

    #[test]
    fn spec_has_correct_table_overview() {
        let tables = [
            (
                "[ProgramTable](program-table.md)",
                program_table::BASE_WIDTH,
                program_table::EXT_WIDTH,
            ),
            (
                "[ProcessorTable](processor-table.md)",
                processor_table::BASE_WIDTH,
                processor_table::EXT_WIDTH,
            ),
            (
                "[OpStackTable](operational-stack-table.md)",
                op_stack_table::BASE_WIDTH,
                op_stack_table::EXT_WIDTH,
            ),
            (
                "[RamTable](random-access-memory-table.md)",
                ram_table::BASE_WIDTH,
                ram_table::EXT_WIDTH,
            ),
            (
                "[JumpStackTable](jump-stack-table.md)",
                jump_stack_table::BASE_WIDTH,
                jump_stack_table::EXT_WIDTH,
            ),
            (
                "[HashTable](hash-table.md)",
                hash_table::BASE_WIDTH,
                hash_table::EXT_WIDTH,
            ),
            (
                "[CascadeTable](cascade-table.md)",
                cascade_table::BASE_WIDTH,
                cascade_table::EXT_WIDTH,
            ),
            (
                "[LookupTable](lookup-table.md)",
                lookup_table::BASE_WIDTH,
                lookup_table::EXT_WIDTH,
            ),
            (
                "[U32Table](u32-table.md)",
                u32_table::BASE_WIDTH,
                u32_table::EXT_WIDTH,
            ),
            (
                "DegreeLowering",
                degree_lowering_table::BASE_WIDTH,
                degree_lowering_table::EXT_WIDTH,
            ),
            ("Randomizers", 0, NUM_RANDOMIZER_POLYNOMIALS),
        ];

        // produce table code
        let mut ft = format!("| {:<42} ", "table name");
        ft = format!("{ft}| {:<10} ", "#main cols");
        ft = format!("{ft}| {:<9} ", "#aux cols");
        ft = format!("{ft}| {:<11} |\n", "total width");

        ft = format!("{ft}|:{:-<42}-", "-");
        ft = format!("{ft}|-{:-<10}:", "-");
        ft = format!("{ft}|-{:-<9}:", "-");
        ft = format!("{ft}|-{:-<11}:|\n", "-");

        let mut total_main = 0;
        let mut total_aux = 0;
        for (name, num_main, num_aux) in tables {
            let num_total = num_main + EXTENSION_DEGREE * num_aux;
            ft = format!("{ft}| {name:<42} | {num_main:>10} | {num_aux:9} | {num_total:>11} |\n");
            total_main += num_main;
            total_aux += num_aux;
        }
        ft = format!(
            "{ft}| {:<42} | {:>10} | {:>9} | {:>11} |\n",
            "**TOTAL**",
            format!("**{total_main}**"),
            format!("**{total_aux}**"),
            format!("**{}**", total_main + EXTENSION_DEGREE * total_aux)
        );

        // print embeddable code
        let specification_generator_name = "spec_has_correct_table_overview";
        let comment_marker_start =
            format!("<!-- auto-gen info {} -->", specification_generator_name);
        let how_reproduce = format!(
            "<!-- To reproduce this code, please run `cargo run {}`. -->",
            specification_generator_name
        );
        let comment_marker_stop = "<!-- auto-gen info stop -->".to_string();
        let generated_code = format!(
            "{}\n{}\n{}{}",
            comment_marker_start, how_reproduce, ft, comment_marker_stop
        );

        // current directory is triton-vm/triton-vm/
        let file_path = Path::new("../specification/src/arithmetization-overview.md");
        println!(
            "Please include this code snippet in file \"{}\".",
            file_path.display()
        );
        println!("```");
        println!("{}", generated_code);
        println!("```");

        assert_spec_has(file_path, generated_code);
    }

    struct ConstraintsOverviewRow {
        pub name: String,
        pub initial_constraints: Vec<ConstraintCircuitMonad<SingleRowIndicator>>,
        pub consistency_constraints: Vec<ConstraintCircuitMonad<SingleRowIndicator>>,
        pub transition_constraints: Vec<ConstraintCircuitMonad<DualRowIndicator>>,
        pub terminal_constraints: Vec<ConstraintCircuitMonad<SingleRowIndicator>>,
        pub last_base_column_index: usize,
        pub last_ext_column_index: usize,
    }

    macro_rules! constraint_overview_rows {
        ($($table:ident ends at $base_end:ident and $ext_end: ident.
        Spec: [$spec_name:literal]($spec_file:literal)),* $(,)?) => {{
            let single_row_builder = || ConstraintCircuitBuilder::new();
            let dual_row_builder = || ConstraintCircuitBuilder::new();
            let mut rows = Vec::new();
            $(
            let name = format!("[{}]({})", $spec_name, $spec_file);
            let row = ConstraintsOverviewRow {
                name,
                initial_constraints: $table::initial_constraints(&single_row_builder()),
                consistency_constraints: $table::consistency_constraints(&single_row_builder()),
                transition_constraints: $table::transition_constraints(&dual_row_builder()),
                terminal_constraints: $table::terminal_constraints(&single_row_builder()),
                last_base_column_index: $base_end,
                last_ext_column_index: $ext_end,
            };
            rows.push(row);
            )*
            rows
        }};
    }

    #[test]
    fn spec_has_correct_constraints_overview() {
        // Declarative macro workaround (because I'm bad at them):
        // an `expr` cannot be followed up with `and`. Instead, declare this `const` to
        // have an `ident`, which _can_ be followed up with `and`.
        const ZERO: usize = 0;
        let mut tables = constraint_overview_rows!(
            ExtProgramTable ends at PROGRAM_TABLE_END and EXT_PROGRAM_TABLE_END.
                Spec: ["ProgramTable"]("program-table.md"),
            ExtProcessorTable ends at PROCESSOR_TABLE_END and EXT_PROCESSOR_TABLE_END.
                Spec: ["ProcessorTable"]("processor-table.md"),
            ExtOpStackTable ends at OP_STACK_TABLE_END and EXT_OP_STACK_TABLE_END.
                Spec: ["OpStackTable"]("operational-stack-table.md"),
            ExtRamTable ends at RAM_TABLE_END and EXT_RAM_TABLE_END.
                Spec: ["RamTable"]("random-access-memory-table.md"),
            ExtJumpStackTable ends at JUMP_STACK_TABLE_END and EXT_JUMP_STACK_TABLE_END.
                Spec: ["JumpStackTable"]("jump-stack-table.md"),
            ExtHashTable ends at HASH_TABLE_END and EXT_HASH_TABLE_END.
                Spec: ["HashTable"]("hash-table.md"),
            ExtCascadeTable ends at CASCADE_TABLE_END and EXT_CASCADE_TABLE_END.
                Spec: ["CascadeTable"]("cascade-table.md"),
            ExtLookupTable ends at LOOKUP_TABLE_END and EXT_LOOKUP_TABLE_END.
                Spec: ["LookupTable"]("lookup-table.md"),
            ExtU32Table ends at U32_TABLE_END and EXT_U32_TABLE_END.
                Spec: ["U32Table"]("u32-table.md"),
            GrandCrossTableArg ends at ZERO and ZERO.
                Spec: ["Grand Cross-Table Argument"]("table-linking.md"),
        );

        let mut ft = String::new();
        ft = format!("{ft}\nBefore automatic degree lowering:\n\n");
        ft = format!("{ft}| {:<46} ", "table name");
        ft = format!("{ft}| #initial ");
        ft = format!("{ft}| #consistency ");
        ft = format!("{ft}| #transition ");
        ft = format!("{ft}| #terminal ");
        ft = format!("{ft}| max degree |\n");

        ft = format!("{ft}|:{:-<46}-", "-");
        ft = format!("{ft}|-{:-<8}:", "-");
        ft = format!("{ft}|-{:-<12}:", "-");
        ft = format!("{ft}|-{:-<11}:", "-");
        ft = format!("{ft}|-{:-<9}:", "-");
        ft = format!("{ft}|-{:-<10}:|\n", "-");

        let mut total_initial = 0;
        let mut total_consistency = 0;
        let mut total_transition = 0;
        let mut total_terminal = 0;
        let mut total_max_degree = 0;
        for table in &tables {
            let table_max_degree = [
                ConstraintCircuitMonad::multicircuit_degree(&table.initial_constraints),
                ConstraintCircuitMonad::multicircuit_degree(&table.consistency_constraints),
                ConstraintCircuitMonad::multicircuit_degree(&table.transition_constraints),
                ConstraintCircuitMonad::multicircuit_degree(&table.terminal_constraints),
            ]
            .into_iter()
            .max()
            .unwrap_or(-1);

            let num_init = table.initial_constraints.len();
            let num_cons = table.consistency_constraints.len();
            let num_tran = table.transition_constraints.len();
            let num_term = table.terminal_constraints.len();
            ft = format!(
                "{ft}| {:<46} | {:>8} | {:12} | {:>11} | {:>9} | {:>10} |\n",
                table.name, num_init, num_cons, num_tran, num_term, table_max_degree,
            );
            total_initial += num_init;
            total_consistency += num_cons;
            total_transition += num_tran;
            total_terminal += num_term;
            total_max_degree = total_max_degree.max(table_max_degree);
        }
        ft = format!(
            "{ft}| {:<46} | {:>8} | {:>12} | {:>11} | {:>9} | {:>10} |\n",
            "**TOTAL**",
            format!("**{total_initial}**"),
            format!("**{total_consistency}**"),
            format!("**{total_transition}**"),
            format!("**{total_terminal}**"),
            format!("**{}**", total_max_degree)
        );
        ft = format!("{ft}\nAfter automatically lowering degree to {AIR_TARGET_DEGREE}:\n\n");
        ft = format!("{ft}| {:<46} ", "table name");
        ft = format!("{ft}| #initial ");
        ft = format!("{ft}| #consistency ");
        ft = format!("{ft}| #transition ");
        ft = format!("{ft}| #terminal |\n");

        ft = format!("{ft}|:{:-<46}-", "-");
        ft = format!("{ft}|-{:-<8}:", "-");
        ft = format!("{ft}|-{:-<12}:", "-");
        ft = format!("{ft}|-{:-<11}:", "-");
        ft = format!("{ft}|-{:-<9}:|\n", "-");

        for table in &mut tables {
            let (new_base_initial, new_ext_initial) = ConstraintCircuitMonad::lower_to_degree(
                &mut table.initial_constraints,
                AIR_TARGET_DEGREE,
                table.last_base_column_index,
                table.last_ext_column_index,
            );
            let (new_base_consistency, new_ext_consistency) =
                ConstraintCircuitMonad::lower_to_degree(
                    &mut table.consistency_constraints,
                    AIR_TARGET_DEGREE,
                    table.last_base_column_index,
                    table.last_ext_column_index,
                );
            let (new_base_transition, new_ext_transition) = ConstraintCircuitMonad::lower_to_degree(
                &mut table.transition_constraints,
                AIR_TARGET_DEGREE,
                table.last_base_column_index,
                table.last_ext_column_index,
            );
            let (new_base_terminal, new_ext_terminal) = ConstraintCircuitMonad::lower_to_degree(
                &mut table.terminal_constraints,
                AIR_TARGET_DEGREE,
                table.last_base_column_index,
                table.last_ext_column_index,
            );
            ft = format!(
                "{ft}| {:<46} | {:>8} | {:12} | {:>11} | {:>9} |\n",
                table.name,
                table.initial_constraints.len() + new_base_initial.len() + new_ext_initial.len(),
                table.consistency_constraints.len()
                    + new_base_consistency.len()
                    + new_ext_consistency.len(),
                table.transition_constraints.len()
                    + new_base_transition.len()
                    + new_ext_transition.len(),
                table.terminal_constraints.len() + new_base_terminal.len() + new_ext_terminal.len(),
            );
            total_initial += table.initial_constraints.len();
            total_consistency += table.consistency_constraints.len();
            total_transition += table.transition_constraints.len();
            total_terminal += table.terminal_constraints.len();
        }
        ft = format!(
            "{ft}| {:<46} | {:>8} | {:>12} | {:>11} | {:>9} |\n",
            "**TOTAL**",
            format!("**{total_initial}**"),
            format!("**{total_consistency}**"),
            format!("**{total_transition}**"),
            format!("**{total_terminal}**"),
        );

        // print embeddable code
        let specification_generator_name = "spec_has_correct_constraints_overview";
        let comment_marker_start =
            format!("<!-- auto-gen info {} -->", specification_generator_name);
        let how_reproduce = format!(
            "<!-- To reproduce this code, please run `cargo run {}`. -->",
            specification_generator_name
        );
        let comment_marker_stop = "<!-- auto-gen info stop -->".to_string();
        let generated_code = format!(
            "{}\n{}\n{}{}",
            comment_marker_start, how_reproduce, ft, comment_marker_stop
        );

        // current directory is triton-vm/triton-vm/
        let file_path = Path::new("../specification/src/arithmetization-overview.md");
        println!(
            "Please include this code snippet in file \"{}\".",
            file_path.display()
        );
        println!("```");
        println!("{}", generated_code);
        println!("```");

        assert_spec_has(file_path, generated_code);
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
        println!(
            "| {:<18} | {:>10} | {:>9} | {:>10} |",
            "DegreeLowering",
            degree_lowering_table::BASE_WIDTH,
            degree_lowering_table::EXT_WIDTH,
            degree_lowering_table::FULL_WIDTH,
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
        for column in DegreeLoweringBaseTableColumn::iter() {
            println!(
                "{:>3} | degree low. | {column}",
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
        for column in DegreeLoweringExtTableColumn::iter() {
            println!(
                "{:>3} | degree low. | {column}",
                column.master_ext_table_index()
            );
        }
    }

    #[test]
    fn master_ext_table_mut() {
        let trace_domain = ArithmeticDomain::of_length(1 << 8).unwrap();
        let randomized_trace_domain = ArithmeticDomain::of_length(1 << 9).unwrap();
        let quotient_domain = ArithmeticDomain::of_length(1 << 10).unwrap();
        let fri_domain = ArithmeticDomain::of_length(1 << 11).unwrap();

        let randomized_trace_table =
            Array2::zeros((randomized_trace_domain.length, NUM_EXT_COLUMNS));

        let mut master_table = MasterExtTable {
            num_trace_randomizers: 16,
            trace_domain,
            randomized_trace_domain,
            quotient_domain,
            fri_domain,
            randomized_trace_table,
            low_degree_extended_table: None,
            interpolation_polynomials: None,
        };

        let num_rows = trace_domain.length;
        Array2::from_elem((num_rows, ProgramExtTableColumn::COUNT), 1.into())
            .move_into(&mut master_table.table_mut(TableId::Program));
        Array2::from_elem((num_rows, ProcessorExtTableColumn::COUNT), 2.into())
            .move_into(&mut master_table.table_mut(TableId::Processor));
        Array2::from_elem((num_rows, OpStackExtTableColumn::COUNT), 3.into())
            .move_into(&mut master_table.table_mut(TableId::OpStack));
        Array2::from_elem((num_rows, RamExtTableColumn::COUNT), 4.into())
            .move_into(&mut master_table.table_mut(TableId::Ram));
        Array2::from_elem((num_rows, JumpStackExtTableColumn::COUNT), 5.into())
            .move_into(&mut master_table.table_mut(TableId::JumpStack));
        Array2::from_elem((num_rows, HashExtTableColumn::COUNT), 6.into())
            .move_into(&mut master_table.table_mut(TableId::Hash));
        Array2::from_elem((num_rows, CascadeExtTableColumn::COUNT), 7.into())
            .move_into(&mut master_table.table_mut(TableId::Cascade));
        Array2::from_elem((num_rows, LookupExtTableColumn::COUNT), 8.into())
            .move_into(&mut master_table.table_mut(TableId::Lookup));
        Array2::from_elem((num_rows, U32ExtTableColumn::COUNT), 9.into())
            .move_into(&mut master_table.table_mut(TableId::U32));

        let trace_domain_element = |column| {
            let maybe_element = master_table.randomized_trace_table.get((0, column));
            let xfe = maybe_element.unwrap().to_owned();
            xfe.unlift().unwrap().value()
        };
        let not_trace_domain_element = |column| {
            let maybe_element = master_table.randomized_trace_table.get((1, column));
            let xfe = maybe_element.unwrap().to_owned();
            xfe.unlift().unwrap().value()
        };

        assert_eq!(1, trace_domain_element(EXT_PROGRAM_TABLE_START));
        assert_eq!(2, trace_domain_element(EXT_PROCESSOR_TABLE_START));
        assert_eq!(3, trace_domain_element(EXT_OP_STACK_TABLE_START));
        assert_eq!(4, trace_domain_element(EXT_RAM_TABLE_START));
        assert_eq!(5, trace_domain_element(EXT_JUMP_STACK_TABLE_START));
        assert_eq!(6, trace_domain_element(EXT_HASH_TABLE_START));
        assert_eq!(7, trace_domain_element(EXT_CASCADE_TABLE_START));
        assert_eq!(8, trace_domain_element(EXT_LOOKUP_TABLE_START));
        assert_eq!(9, trace_domain_element(EXT_U32_TABLE_START));

        assert_eq!(0, not_trace_domain_element(EXT_PROGRAM_TABLE_START));
        assert_eq!(0, not_trace_domain_element(EXT_PROCESSOR_TABLE_START));
        assert_eq!(0, not_trace_domain_element(EXT_OP_STACK_TABLE_START));
        assert_eq!(0, not_trace_domain_element(EXT_RAM_TABLE_START));
        assert_eq!(0, not_trace_domain_element(EXT_JUMP_STACK_TABLE_START));
        assert_eq!(0, not_trace_domain_element(EXT_HASH_TABLE_START));
        assert_eq!(0, not_trace_domain_element(EXT_CASCADE_TABLE_START));
        assert_eq!(0, not_trace_domain_element(EXT_LOOKUP_TABLE_START));
        assert_eq!(0, not_trace_domain_element(EXT_U32_TABLE_START));
    }

    #[proptest]
    fn test_sponge_with_pending_absorb(
        #[strategy(arb())] elements: Vec<BFieldElement>,
        #[strategy(0_usize..=#elements.len())] substring_index: usize,
    ) {
        let (substring_0, substring_1) = elements.split_at(substring_index);
        let mut sponge = SpongeWithPendingAbsorb::new();
        sponge.absorb(substring_0);
        sponge.absorb(substring_1);
        let pending_absorb_digest = sponge.finalize();

        let expected_digest = Tip5::hash_varlen(&elements);
        prop_assert_eq!(expected_digest, pending_absorb_digest);
    }
}
