use std::ops::Add;
use std::ops::Mul;

use anyhow::bail;
use anyhow::Result;
use itertools::izip;
use itertools::Itertools;
use ndarray::s;
use ndarray::Array1;
use ndarray::ArrayBase;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use ndarray::Axis;
use num_traits::One;
use rayon::prelude::*;
use serde::Deserialize;
use serde::Serialize;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::mpolynomial::Degree;
use twenty_first::shared_math::other::roundup_npo2;
use twenty_first::shared_math::tip5::Tip5;
use twenty_first::shared_math::traits::FiniteField;
use twenty_first::shared_math::traits::Inverse;
use twenty_first::shared_math::traits::ModPowU32;
use twenty_first::shared_math::x_field_element::XFieldElement;
use twenty_first::util_types::algebraic_hasher::AlgebraicHasher;
use twenty_first::util_types::merkle_tree::CpuParallel;
use twenty_first::util_types::merkle_tree::MerkleTree;
use twenty_first::util_types::merkle_tree_maker::MerkleTreeMaker;

use triton_profiler::prof_itr0;
use triton_profiler::prof_start;
use triton_profiler::prof_stop;
use triton_profiler::triton_profiler::TritonProfiler;
use twenty_first::shared_math::polynomial::Polynomial;
use twenty_first::shared_math::x_field_element;

use crate::arithmetic_domain::ArithmeticDomain;
use crate::fri::Fri;
use crate::proof::Claim;
use crate::proof::Proof;
use crate::proof_item::ProofItem;
use crate::proof_stream::ProofStream;
use crate::table::challenges::Challenges;
use crate::table::master_table::*;
use crate::vm::AlgebraicExecutionTrace;

pub type StarkHasher = Tip5;
pub type MTMaker = CpuParallel;
pub type StarkProofStream = ProofStream<StarkHasher>;

/// All the security-related parameters for the zk-STARK.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Hash)]
pub struct StarkParameters {
    /// The conjectured security level in bits. Concretely, the system
    /// - is perfectly complete, and
    /// - has soundness error 2^(-security_level).
    pub security_level: usize,

    /// The ratio between the lengths of the randomized trace domain and the FRI domain.
    /// Must be a power of 2 for efficiency reasons.
    pub fri_expansion_factor: usize,

    /// The number of randomizers for the execution trace. The trace randomizers are integral for
    /// achieving zero-knowledge. In particular, they achieve ZK for the (DEEP) ALI part of the
    /// zk-STARK.
    pub num_trace_randomizers: usize,

    /// The number of randomizer polynomials. A single randomizer polynomial should be sufficient
    /// in all cases. It is integral for achieving zero-knowledge for the FRI part of the zk-STARK.
    pub num_randomizer_polynomials: usize,

    /// The number of colinearity checks to perform in FRI.
    pub num_colinearity_checks: usize,

    /// The number of combination codeword checks. These checks link the (DEEP) ALI part and the
    /// FRI part of the zk-STARK. The number of combination codeword checks directly depends on the
    /// number of colinearity checks and the FRI folding factor.
    pub num_combination_codeword_checks: usize,
}

impl StarkParameters {
    pub fn new(security_level: usize, log2_of_fri_expansion_factor: usize) -> Self {
        assert_ne!(
            0, log2_of_fri_expansion_factor,
            "FRI expansion factor must be greater than one."
        );

        let num_randomizer_polynomials = 1; // over the XField
        let fri_expansion_factor = 1 << log2_of_fri_expansion_factor;
        let num_colinearity_checks = security_level / log2_of_fri_expansion_factor;

        // For now, the FRI folding factor is hardcoded in our zk-STARK.
        let fri_folding_factor = 2;
        let num_combination_codeword_checks = num_colinearity_checks * fri_folding_factor;

        let num_out_of_domain_rows = 2;
        let num_trace_randomizers = num_combination_codeword_checks
            + num_out_of_domain_rows * x_field_element::EXTENSION_DEGREE;

        StarkParameters {
            security_level,
            fri_expansion_factor,
            num_trace_randomizers,
            num_randomizer_polynomials,
            num_colinearity_checks,
            num_combination_codeword_checks,
        }
    }
}

impl Default for StarkParameters {
    fn default() -> Self {
        let log_2_of_fri_expansion_factor = 2;
        let security_level = 160;

        Self::new(security_level, log_2_of_fri_expansion_factor)
    }
}

pub struct Stark {}

impl Stark {
    pub fn prove(
        parameters: &StarkParameters,
        claim: &Claim,
        aet: &AlgebraicExecutionTrace,
        maybe_profiler: &mut Option<TritonProfiler>,
    ) -> Proof {
        assert_eq!(
            claim.padded_height(),
            MasterBaseTable::padded_height(aet),
            "The claimed padded height must match the actual padded height of the trace."
        );

        prof_start!(maybe_profiler, "derive additional parameters");
        let max_degree =
            Self::derive_max_degree(claim.padded_height(), parameters.num_trace_randomizers);
        let fri = Self::derive_fri(parameters, max_degree);
        prof_stop!(maybe_profiler, "derive additional parameters");

        prof_start!(maybe_profiler, "base tables");
        prof_start!(maybe_profiler, "create");
        let mut master_base_table =
            MasterBaseTable::new(aet, parameters.num_trace_randomizers, fri.domain);
        prof_stop!(maybe_profiler, "create");

        prof_start!(maybe_profiler, "pad");
        master_base_table.pad();
        prof_stop!(maybe_profiler, "pad");

        prof_start!(maybe_profiler, "LDE", "LDE");
        master_base_table.randomize_trace();
        let (fri_domain_master_base_table, base_interpolation_polys) =
            master_base_table.to_fri_domain_table();
        prof_stop!(maybe_profiler, "LDE");

        prof_start!(maybe_profiler, "Merkle tree", "hash");
        let base_merkle_tree = fri_domain_master_base_table.merkle_tree(maybe_profiler);
        let base_merkle_tree_root = base_merkle_tree.get_root();
        prof_stop!(maybe_profiler, "Merkle tree");

        prof_start!(maybe_profiler, "Fiat-Shamir", "hash");
        let padded_height = BFieldElement::new(master_base_table.padded_height as u64);
        let mut proof_stream = StarkProofStream::new();
        proof_stream.enqueue(&ProofItem::MerkleRoot(base_merkle_tree_root));
        let extension_weights = proof_stream.sample_scalars(Challenges::num_challenges_to_sample());
        let extension_challenges = Challenges::new(extension_weights, &claim.input, &claim.output);
        prof_stop!(maybe_profiler, "Fiat-Shamir");

        prof_start!(maybe_profiler, "extend");
        let mut master_ext_table =
            master_base_table.extend(&extension_challenges, parameters.num_randomizer_polynomials);
        prof_stop!(maybe_profiler, "extend");
        prof_stop!(maybe_profiler, "base tables");

        prof_start!(maybe_profiler, "ext tables");
        prof_start!(maybe_profiler, "LDE", "LDE");
        master_ext_table.randomize_trace();
        let (fri_domain_ext_master_table, ext_interpolation_polys) =
            master_ext_table.to_fri_domain_table();
        prof_stop!(maybe_profiler, "LDE");

        prof_start!(maybe_profiler, "Merkle tree", "hash");
        let ext_merkle_tree = fri_domain_ext_master_table.merkle_tree(maybe_profiler);
        let ext_merkle_tree_root = ext_merkle_tree.get_root();
        proof_stream.enqueue(&ProofItem::MerkleRoot(ext_merkle_tree_root));
        prof_stop!(maybe_profiler, "Merkle tree");
        prof_stop!(maybe_profiler, "ext tables");

        prof_start!(maybe_profiler, "quotient-domain codewords");
        let trace_domain = ArithmeticDomain::new_no_offset(master_base_table.padded_height);
        // When debugging, it is useful to check the degree of some intermediate polynomials.
        // The quotient domain is chosen to be _just_ large enough to perform all the necessary
        // computations on polynomials. Concretely, the maximal degree of a polynomial over the
        // quotient domain is at most only slightly larger than the maximal degree allowed in the
        // STARK proof, and could be equal. This makes computation for the prover much faster.
        // However, it can also make it impossible to check if some operation (e.g., dividing out
        // the zerofier) has (erroneously) increased the polynomial's degree beyond the allowed
        // maximum.
        let quotient_domain = if cfg!(debug_assertions) {
            fri.domain
        } else {
            let offset = fri.domain.offset;
            let length = roundup_npo2(max_degree as u64);
            ArithmeticDomain::new(offset, length as usize)
        };
        let unit_distance = fri.domain.length / quotient_domain.length;
        let base_quotient_domain_codewords = fri_domain_master_base_table
            .master_base_matrix
            .slice(s![..; unit_distance, ..]);
        let extension_quotient_domain_codewords = fri_domain_ext_master_table
            .master_ext_matrix
            .slice(s![..; unit_distance, ..]);
        prof_stop!(maybe_profiler, "quotient-domain codewords");

        prof_start!(maybe_profiler, "quotient codewords", "AIR");
        let master_quotient_table = all_quotients(
            base_quotient_domain_codewords,
            extension_quotient_domain_codewords,
            trace_domain,
            quotient_domain,
            &extension_challenges,
            maybe_profiler,
        );
        prof_stop!(maybe_profiler, "quotient codewords");

        prof_start!(maybe_profiler, "linearly combine quotient codewords", "CC");
        // Create quotient codeword. This is a part of the combination codeword. To reduce the
        // amount of hashing necessary, the quotient codeword is linearly summed instead of
        // hashed prior to committing to it.
        let quotient_combination_weights =
            Array1::from(proof_stream.sample_scalars(num_all_table_quotients()));
        assert_eq!(
            quotient_combination_weights.len(),
            master_quotient_table.ncols()
        );

        // Note: `*` is the element-wise (Hadamard) product
        let weighted_codewords = master_quotient_table * quotient_combination_weights;
        let quotient_codeword = weighted_codewords.sum_axis(Axis(1));

        assert_eq!(quotient_domain.length, quotient_codeword.len());
        #[cfg(debug_assertions)]
        Self::debug_check_degree(&quotient_codeword.to_vec(), quotient_domain, max_degree);
        prof_stop!(maybe_profiler, "linearly combine quotient codewords");

        prof_start!(maybe_profiler, "commit to quotient codeword");
        prof_start!(maybe_profiler, "LDE", "LDE");
        let quotient_interpolation_poly = quotient_domain.interpolate(&quotient_codeword.to_vec());
        let fri_quotient_codeword = Array1::from(fri.domain.evaluate(&quotient_interpolation_poly));
        prof_stop!(maybe_profiler, "LDE");
        prof_start!(maybe_profiler, "interpret XFEs as Digests");
        let fri_quotient_codeword_digests = fri_quotient_codeword
            .iter()
            .map(|&x| x.into())
            .collect_vec();
        prof_stop!(maybe_profiler, "interpret XFEs as Digests");
        prof_start!(maybe_profiler, "Merkle tree", "hash");
        let quot_merkle_tree: MerkleTree<StarkHasher> =
            MTMaker::from_digests(&fri_quotient_codeword_digests);
        let quot_merkle_tree_root = quot_merkle_tree.get_root();
        proof_stream.enqueue(&ProofItem::MerkleRoot(quot_merkle_tree_root));
        prof_stop!(maybe_profiler, "Merkle tree");
        prof_stop!(maybe_profiler, "commit to quotient codeword");
        debug_assert_eq!(fri.domain.length, quot_merkle_tree.get_leaf_count());

        prof_start!(maybe_profiler, "out-of-domain rows");
        let trace_domain_generator = derive_domain_generator(padded_height.value());
        let out_of_domain_point_curr_row = proof_stream.sample_scalars(1)[0];
        let out_of_domain_point_next_row = trace_domain_generator * out_of_domain_point_curr_row;

        prof_start!(maybe_profiler, "lift base polys");
        let base_interpolation_polys = base_interpolation_polys
            .map(|poly| Polynomial::new(poly.coefficients.iter().map(|b| b.lift()).collect_vec()));
        // ignore randomizer codewords / polynomials
        let ext_interpolation_polys = ext_interpolation_polys.slice(s![..NUM_EXT_COLUMNS]);
        prof_stop!(maybe_profiler, "lift base polys");
        let out_of_domain_curr_base_row =
            base_interpolation_polys.map(|poly| poly.evaluate(&out_of_domain_point_curr_row));
        let out_of_domain_next_base_row =
            base_interpolation_polys.map(|poly| poly.evaluate(&out_of_domain_point_next_row));
        let out_of_domain_curr_ext_row =
            ext_interpolation_polys.map(|poly| poly.evaluate(&out_of_domain_point_curr_row));
        let out_of_domain_next_ext_row =
            ext_interpolation_polys.map(|poly| poly.evaluate(&out_of_domain_point_next_row));
        proof_stream.enqueue(&ProofItem::OutOfDomainBaseRow(
            out_of_domain_curr_base_row.to_vec(),
        ));
        proof_stream.enqueue(&ProofItem::OutOfDomainExtRow(
            out_of_domain_curr_ext_row.to_vec(),
        ));
        proof_stream.enqueue(&ProofItem::OutOfDomainBaseRow(
            out_of_domain_next_base_row.to_vec(),
        ));
        proof_stream.enqueue(&ProofItem::OutOfDomainExtRow(
            out_of_domain_next_ext_row.to_vec(),
        ));
        prof_stop!(maybe_profiler, "out-of-domain rows");

        // Get weights for remainder of the combination codeword.
        prof_start!(maybe_profiler, "Fiat-Shamir", "hash");
        let num_base_and_ext_codeword_weights = NUM_BASE_COLUMNS + NUM_EXT_COLUMNS;
        let base_and_ext_codeword_weights =
            proof_stream.sample_scalars(num_base_and_ext_codeword_weights);
        prof_stop!(maybe_profiler, "Fiat-Shamir");

        prof_start!(maybe_profiler, "base&ext: linear combination", "CC");
        let extension_codewords =
            extension_quotient_domain_codewords.slice(s![.., ..NUM_EXT_COLUMNS]);
        let (base_weights, ext_weights) = base_and_ext_codeword_weights.split_at(NUM_BASE_COLUMNS);
        let base_weights = Array1::from(base_weights.to_vec());
        let ext_weights = Array1::from(ext_weights.to_vec());

        assert_eq!(base_weights.len(), base_quotient_domain_codewords.ncols());
        assert_eq!(ext_weights.len(), extension_codewords.ncols());

        // Note: `*` is the element-wise (Hadamard) product
        let weighted_base_codewords = &base_quotient_domain_codewords * base_weights;
        let weighted_ext_codewords = &extension_codewords * &ext_weights;

        let base_and_ext_codeword =
            weighted_base_codewords.sum_axis(Axis(1)) + weighted_ext_codewords.sum_axis(Axis(1));

        assert_eq!(quotient_domain.length, base_and_ext_codeword.len());
        #[cfg(debug_assertions)]
        Self::debug_check_degree(&base_and_ext_codeword.to_vec(), quotient_domain, max_degree);
        prof_stop!(maybe_profiler, "base&ext: linear combination");

        prof_start!(maybe_profiler, "DEEP");
        prof_start!(maybe_profiler, "base&ext");
        let base_and_ext_interpolation_poly =
            quotient_domain.interpolate(&base_and_ext_codeword.to_vec());
        let out_of_domain_next_row_base_and_ext_value =
            base_and_ext_interpolation_poly.evaluate(&out_of_domain_point_next_row);
        let base_and_ext_next_row_deep_codeword = Self::deep_codeword(
            &base_and_ext_codeword.to_vec(),
            quotient_domain,
            out_of_domain_point_next_row,
            out_of_domain_next_row_base_and_ext_value,
        );
        prof_stop!(maybe_profiler, "base&ext");

        prof_start!(maybe_profiler, "base&ext + quot");
        let base_and_ext_and_quot_codeword = (base_and_ext_codeword + quotient_codeword).to_vec();
        let base_and_ext_and_quot_interpolation_poly =
            quotient_domain.interpolate(&base_and_ext_and_quot_codeword);
        let out_of_domain_curr_row_base_and_ext_and_quot_value =
            base_and_ext_and_quot_interpolation_poly.evaluate(&out_of_domain_point_curr_row);
        let base_and_ext_and_quot_curr_row_deep_codeword = Self::deep_codeword(
            &base_and_ext_and_quot_codeword,
            quotient_domain,
            out_of_domain_point_curr_row,
            out_of_domain_curr_row_base_and_ext_and_quot_value,
        );
        prof_stop!(maybe_profiler, "base&ext + quot");
        prof_stop!(maybe_profiler, "DEEP");

        #[cfg(debug_assertions)]
        {
            let out_of_domain_quotient_value =
                quotient_interpolation_poly.evaluate(&out_of_domain_point_curr_row);
            let base_and_ext_weights = Array1::from(base_and_ext_codeword_weights);
            let out_of_domain_curr_row_base_and_ext_value = Self::linearly_sum_base_and_ext_row(
                out_of_domain_curr_base_row.view(),
                out_of_domain_curr_ext_row.view(),
                base_and_ext_weights.view(),
                &mut None,
            );
            assert_eq!(
                out_of_domain_curr_row_base_and_ext_and_quot_value,
                out_of_domain_curr_row_base_and_ext_value + out_of_domain_quotient_value,
            );
            let out_of_domain_next_row_base_and_ext_value_2 = Self::linearly_sum_base_and_ext_row(
                out_of_domain_next_base_row.view(),
                out_of_domain_next_ext_row.view(),
                base_and_ext_weights.view(),
                &mut None,
            );
            assert_eq!(
                out_of_domain_next_row_base_and_ext_value,
                out_of_domain_next_row_base_and_ext_value_2,
            );
        }

        prof_start!(maybe_profiler, "combined DEEP polynomial");
        prof_start!(maybe_profiler, "sum", "CC");
        let base_and_ext_and_quot_curr_row_deep_codeword =
            Array1::from(base_and_ext_and_quot_curr_row_deep_codeword);
        let base_and_ext_next_row_deep_codeword = Array1::from(base_and_ext_next_row_deep_codeword);
        let deep_codeword =
            &base_and_ext_and_quot_curr_row_deep_codeword + &base_and_ext_next_row_deep_codeword;
        prof_stop!(maybe_profiler, "sum");
        prof_start!(maybe_profiler, "LDE", "LDE");
        let fri_deep_codeword =
            Array1::from(quotient_domain.low_degree_extension(&deep_codeword.to_vec(), fri.domain));
        prof_stop!(maybe_profiler, "LDE");
        assert_eq!(fri.domain.length, fri_deep_codeword.len());
        prof_start!(maybe_profiler, "add randomizer codeword", "CC");
        let fri_combination_codeword = fri_domain_ext_master_table
            .randomizer_polynomials()
            .into_iter()
            .fold(fri_deep_codeword, ArrayBase::add)
            .to_vec();
        prof_stop!(maybe_profiler, "add randomizer codeword");
        assert_eq!(fri.domain.length, fri_combination_codeword.len());
        prof_stop!(maybe_profiler, "combined DEEP polynomial");

        prof_start!(maybe_profiler, "FRI");
        let (revealed_current_row_indices, _) =
            fri.prove(&fri_combination_codeword, &mut proof_stream);
        assert_eq!(
            parameters.num_combination_codeword_checks,
            revealed_current_row_indices.len()
        );
        prof_stop!(maybe_profiler, "FRI");

        prof_start!(maybe_profiler, "open trace leafs");
        // Open leafs of zipped codewords at indicated positions
        let revealed_base_elems = Self::get_revealed_elements(
            fri_domain_master_base_table.master_base_matrix.view(),
            &revealed_current_row_indices,
        );
        let auth_paths_base =
            base_merkle_tree.get_authentication_structure(&revealed_current_row_indices);
        proof_stream.enqueue(&ProofItem::MasterBaseTableRows(revealed_base_elems));
        proof_stream.enqueue(&ProofItem::CompressedAuthenticationPaths(auth_paths_base));

        let revealed_ext_elems = Self::get_revealed_elements(
            fri_domain_ext_master_table.master_ext_matrix.view(),
            &revealed_current_row_indices,
        );
        let auth_paths_ext =
            ext_merkle_tree.get_authentication_structure(&revealed_current_row_indices);
        proof_stream.enqueue(&ProofItem::MasterExtTableRows(revealed_ext_elems));
        proof_stream.enqueue(&ProofItem::CompressedAuthenticationPaths(auth_paths_ext));

        // Open quotient & combination codewords at the same positions as base & ext codewords.
        let revealed_quotient_elements = revealed_current_row_indices
            .iter()
            .map(|&i| fri_quotient_codeword[i])
            .collect_vec();
        let revealed_quotient_auth_paths =
            quot_merkle_tree.get_authentication_structure(&revealed_current_row_indices);
        proof_stream.enqueue(&ProofItem::RevealedCombinationElements(
            revealed_quotient_elements,
        ));
        proof_stream.enqueue(&ProofItem::CompressedAuthenticationPaths(
            revealed_quotient_auth_paths,
        ));
        prof_stop!(maybe_profiler, "open trace leafs");

        #[cfg(debug_assertions)]
        {
            let transcript_length = proof_stream.transcript_length();
            let kib = (transcript_length * 8 / 1024) + 1;
            println!("Created proof containing {transcript_length} B-field elements ({kib} kiB).");
        }

        proof_stream.into()
    }

    /// Compute the upper bound to use for the maximum degree the quotients given the length of the
    /// trace and the number of trace randomizers.
    /// The degree of the quotients depends on the constraints, _i.e._, the AIR.
    /// The upper bound is computed as follows:
    /// 1. Compute the degree of the trace interpolation polynomials.
    /// 1. Compute the maximum degree of the quotients.
    /// 1. Round up to the next power of 2.
    pub fn derive_max_degree(padded_height: usize, num_trace_randomizers: usize) -> Degree {
        let interpolant_degree = interpolant_degree(padded_height, num_trace_randomizers);
        let max_degree_with_origin = max_degree_with_origin(interpolant_degree, padded_height);
        (roundup_npo2(max_degree_with_origin.degree as u64) - 1) as Degree
    }

    /// Compute the parameters for FRI. The size of the FRI domain depends on the
    /// quotients' maximum degree, see [`derive_max_degree`](Self::derive_max_degree).
    /// It also depends on the FRI expansion factor, which is a parameter of the STARK.
    pub fn derive_fri(parameters: &StarkParameters, max_degree: Degree) -> Fri<StarkHasher> {
        let fri_domain_length = parameters.fri_expansion_factor * (max_degree as usize + 1);
        let fri_coset_offset = BFieldElement::generator();
        Fri::new(
            fri_coset_offset,
            fri_domain_length,
            parameters.fri_expansion_factor,
            parameters.num_colinearity_checks,
        )
    }

    fn get_revealed_elements<FF: FiniteField>(
        master_matrix: ArrayView2<FF>,
        revealed_indices: &[usize],
    ) -> Vec<Vec<FF>> {
        revealed_indices
            .iter()
            .map(|&idx| master_matrix.row(idx).to_vec())
            .collect_vec()
    }

    /// Apply the [DEEP update](Self::deep_update) to a polynomial in value form, _i.e._, to a
    /// codeword.
    fn deep_codeword(
        codeword: &[XFieldElement],
        domain: ArithmeticDomain,
        out_of_domain_point: XFieldElement,
        out_of_domain_value: XFieldElement,
    ) -> Vec<XFieldElement> {
        domain
            .domain_values()
            .par_iter()
            .zip_eq(codeword.par_iter())
            .map(|(&in_domain_value, &in_domain_evaluation)| {
                Self::deep_update(
                    in_domain_value,
                    in_domain_evaluation,
                    out_of_domain_point,
                    out_of_domain_value,
                )
            })
            .collect()
    }

    /// Given `f(x)` (the in-domain evaluation of polynomial `f` in `x`), the domain point `x` at
    /// which polynomial `f` was evaluated, the out-of-domain evaluation `f(α)`, and the
    /// out-of-domain domain point `α`, apply the DEEP update: `(f(x) - f(α)) / (x - α)`.
    #[inline]
    fn deep_update(
        in_domain_point: BFieldElement,
        in_domain_value: XFieldElement,
        out_of_domain_point: XFieldElement,
        out_of_domain_value: XFieldElement,
    ) -> XFieldElement {
        (in_domain_value - out_of_domain_value) / (in_domain_point - out_of_domain_point)
    }

    #[cfg(debug_assertions)]
    fn debug_check_degree(
        combination_codeword: &[XFieldElement],
        quotient_domain: ArithmeticDomain,
        max_degree: Degree,
    ) {
        let max_degree = max_degree as isize;
        let degree = quotient_domain.interpolate(combination_codeword).degree();
        let maybe_excl_mark = match degree > max_degree {
            true => "!",
            false => " ",
        };
        println!(
            "{maybe_excl_mark} Combination codeword has degree {degree}. \
            Must be of maximal degree {max_degree}."
        );
    }

    pub fn verify(
        parameters: &StarkParameters,
        claim: &Claim,
        proof: &Proof,
        maybe_profiler: &mut Option<TritonProfiler>,
    ) -> Result<bool> {
        prof_start!(maybe_profiler, "derive additional parameters");
        assert_eq!(claim.padded_height(), proof.padded_height(parameters));
        let max_degree =
            Self::derive_max_degree(claim.padded_height(), parameters.num_trace_randomizers);
        let fri = Self::derive_fri(parameters, max_degree);
        let merkle_tree_height = fri.domain.length.ilog2() as usize;
        prof_stop!(maybe_profiler, "derive additional parameters");

        prof_start!(maybe_profiler, "deserialize");
        let mut proof_stream = StarkProofStream::try_from(proof)?;
        prof_stop!(maybe_profiler, "deserialize");

        prof_start!(maybe_profiler, "Fiat-Shamir 1", "hash");
        let base_merkle_tree_root = proof_stream.dequeue()?.as_merkle_root()?;
        let extension_challenge_weights =
            proof_stream.sample_scalars(Challenges::num_challenges_to_sample());
        let challenges = Challenges::new(extension_challenge_weights, &claim.input, &claim.output);
        let extension_tree_merkle_root = proof_stream.dequeue()?.as_merkle_root()?;
        // Sample weights for quotient codeword, which is a part of the combination codeword.
        // See corresponding part in the prover for a more detailed explanation.
        let quot_codeword_weights =
            Array1::from(proof_stream.sample_scalars(num_all_table_quotients()));
        let quotient_codeword_merkle_root = proof_stream.dequeue()?.as_merkle_root()?;
        prof_stop!(maybe_profiler, "Fiat-Shamir 1");

        prof_start!(maybe_profiler, "dequeue ood point and rows", "hash");
        let trace_domain_generator = derive_domain_generator(claim.padded_height() as u64);
        let out_of_domain_point_curr_row = proof_stream.sample_scalars(1)[0];
        let out_of_domain_point_next_row = trace_domain_generator * out_of_domain_point_curr_row;

        let out_of_domain_curr_base_row = proof_stream.dequeue()?.as_out_of_domain_base_row()?;
        let out_of_domain_curr_ext_row = proof_stream.dequeue()?.as_out_of_domain_ext_row()?;
        let out_of_domain_next_base_row = proof_stream.dequeue()?.as_out_of_domain_base_row()?;
        let out_of_domain_next_ext_row = proof_stream.dequeue()?.as_out_of_domain_ext_row()?;

        let out_of_domain_curr_base_row = Array1::from(out_of_domain_curr_base_row);
        let out_of_domain_curr_ext_row = Array1::from(out_of_domain_curr_ext_row);
        let out_of_domain_next_base_row = Array1::from(out_of_domain_next_base_row);
        let out_of_domain_next_ext_row = Array1::from(out_of_domain_next_ext_row);
        prof_stop!(maybe_profiler, "dequeue ood point and rows");

        prof_start!(maybe_profiler, "out-of-domain quotient element");
        prof_start!(maybe_profiler, "zerofiers");
        let one = BFieldElement::one();
        let initial_zerofier_inv = (out_of_domain_point_curr_row - one).inverse();
        let consistency_zerofier_inv =
            (out_of_domain_point_curr_row.mod_pow_u32(claim.padded_height() as u32) - one)
                .inverse();
        let except_last_row = out_of_domain_point_curr_row - trace_domain_generator.inverse();
        let transition_zerofier_inv = except_last_row * consistency_zerofier_inv;
        let terminal_zerofier_inv = except_last_row.inverse(); // i.e., only last row
        prof_stop!(maybe_profiler, "zerofiers");

        prof_start!(maybe_profiler, "evaluate AIR", "AIR");
        let evaluated_initial_constraints = evaluate_all_initial_constraints(
            out_of_domain_curr_base_row.view(),
            out_of_domain_curr_ext_row.view(),
            &challenges,
        );
        let evaluated_consistency_constraints = evaluate_all_consistency_constraints(
            out_of_domain_curr_base_row.view(),
            out_of_domain_curr_ext_row.view(),
            &challenges,
        );
        let evaluated_transition_constraints = evaluate_all_transition_constraints(
            out_of_domain_curr_base_row.view(),
            out_of_domain_curr_ext_row.view(),
            out_of_domain_next_base_row.view(),
            out_of_domain_next_ext_row.view(),
            &challenges,
        );
        let evaluated_terminal_constraints = evaluate_all_terminal_constraints(
            out_of_domain_curr_base_row.view(),
            out_of_domain_curr_ext_row.view(),
            &challenges,
        );
        prof_stop!(maybe_profiler, "evaluate AIR");

        prof_start!(maybe_profiler, "divide");
        let mut quotient_summands = Vec::with_capacity(num_all_table_quotients());
        for (evaluated_constraints_category, zerofier_inverse) in [
            (evaluated_initial_constraints, initial_zerofier_inv),
            (evaluated_consistency_constraints, consistency_zerofier_inv),
            (evaluated_transition_constraints, transition_zerofier_inv),
            (evaluated_terminal_constraints, terminal_zerofier_inv),
        ] {
            let category_quotients = evaluated_constraints_category
                .into_iter()
                .map(|evaluated_constraint| evaluated_constraint * zerofier_inverse)
                .collect_vec();
            quotient_summands.extend(category_quotients);
        }
        prof_stop!(maybe_profiler, "divide");

        prof_start!(maybe_profiler, "inner product", "CC");
        let out_of_domain_quotient_value =
            (&quot_codeword_weights * &Array1::from(quotient_summands)).sum();
        prof_stop!(maybe_profiler, "inner product");
        prof_stop!(maybe_profiler, "out-of-domain quotient element");

        prof_start!(maybe_profiler, "Fiat-Shamir 2", "hash");
        let num_base_and_ext_codeword_weights = NUM_BASE_COLUMNS + NUM_EXT_COLUMNS;
        let base_and_ext_codeword_weights =
            Array1::from(proof_stream.sample_scalars(num_base_and_ext_codeword_weights));
        prof_stop!(maybe_profiler, "Fiat-Shamir 2");

        prof_start!(maybe_profiler, "sum out-of-domain values", "CC");
        let out_of_domain_curr_row_base_and_ext_value = Self::linearly_sum_base_and_ext_row(
            out_of_domain_curr_base_row.view(),
            out_of_domain_curr_ext_row.view(),
            base_and_ext_codeword_weights.view(),
            maybe_profiler,
        );
        let out_of_domain_curr_row_base_and_ext_and_quot_value =
            out_of_domain_curr_row_base_and_ext_value + out_of_domain_quotient_value;
        let out_of_domain_next_row_base_and_ext_value = Self::linearly_sum_base_and_ext_row(
            out_of_domain_next_base_row.view(),
            out_of_domain_next_ext_row.view(),
            base_and_ext_codeword_weights.view(),
            maybe_profiler,
        );
        prof_stop!(maybe_profiler, "sum out-of-domain values");

        // verify low degree of combination polynomial with FRI
        prof_start!(maybe_profiler, "FRI");
        let revealed_fri_indices_and_elements = fri.verify(&mut proof_stream, maybe_profiler)?;
        let (revealed_current_row_indices, revealed_fri_values): (Vec<_>, Vec<_>) =
            revealed_fri_indices_and_elements.into_iter().unzip();
        prof_stop!(maybe_profiler, "FRI");

        prof_start!(maybe_profiler, "check leafs");
        prof_start!(maybe_profiler, "dequeue base elements");
        let base_table_rows = proof_stream.dequeue()?.as_master_base_table_rows()?;
        let base_auth_paths = proof_stream
            .dequeue()?
            .as_compressed_authentication_paths()?;
        let leaf_digests_base: Vec<_> = base_table_rows
            .par_iter()
            .map(|revealed_base_elem| StarkHasher::hash_varlen(revealed_base_elem))
            .collect();
        prof_stop!(maybe_profiler, "dequeue base elements");

        prof_start!(maybe_profiler, "Merkle verify (base tree)", "hash");
        if !MerkleTree::<StarkHasher>::verify_authentication_structure_from_leaves(
            base_merkle_tree_root,
            merkle_tree_height,
            &revealed_current_row_indices,
            &leaf_digests_base,
            &base_auth_paths,
        ) {
            bail!("Failed to verify authentication path for base codeword");
        }
        prof_stop!(maybe_profiler, "Merkle verify (base tree)");

        prof_start!(maybe_profiler, "dequeue extension elements");
        let ext_table_rows = proof_stream.dequeue()?.as_master_ext_table_rows()?;
        let auth_paths_ext = proof_stream
            .dequeue()?
            .as_compressed_authentication_paths()?;
        let leaf_digests_ext = ext_table_rows
            .par_iter()
            .map(|xvalues| {
                let bvalues = xvalues
                    .iter()
                    .flat_map(|xfe| xfe.coefficients.to_vec())
                    .collect_vec();
                StarkHasher::hash_varlen(&bvalues)
            })
            .collect::<Vec<_>>();
        prof_stop!(maybe_profiler, "dequeue extension elements");

        prof_start!(maybe_profiler, "Merkle verify (extension tree)", "hash");
        if !MerkleTree::<StarkHasher>::verify_authentication_structure_from_leaves(
            extension_tree_merkle_root,
            merkle_tree_height,
            &revealed_current_row_indices,
            &leaf_digests_ext,
            &auth_paths_ext,
        ) {
            bail!("Failed to verify authentication path for extension codeword");
        }
        prof_stop!(maybe_profiler, "Merkle verify (extension tree)");

        prof_start!(maybe_profiler, "dequeue quotient elements");
        let revealed_quotient_values =
            proof_stream.dequeue()?.as_revealed_combination_elements()?;
        // Interpret the leaves, which are XFieldElements, as Digests, without hashing.
        let revealed_quotient_digests = revealed_quotient_values
            .par_iter()
            .map(|&x| x.into())
            .collect::<Vec<_>>();
        let revealed_quotient_auth_paths = proof_stream
            .dequeue()?
            .as_compressed_authentication_paths()?;
        prof_stop!(maybe_profiler, "dequeue quotient elements");

        prof_start!(maybe_profiler, "Merkle verify (combined quotient)", "hash");
        if !MerkleTree::<StarkHasher>::verify_authentication_structure_from_leaves(
            quotient_codeword_merkle_root,
            merkle_tree_height,
            &revealed_current_row_indices,
            &revealed_quotient_digests,
            &revealed_quotient_auth_paths,
        ) {
            bail!("Failed to verify authentication path for combined quotient codeword");
        }
        prof_stop!(maybe_profiler, "Merkle verify (combined quotient)");
        prof_stop!(maybe_profiler, "check leafs");

        prof_start!(maybe_profiler, "linear combination");
        let num_checks = parameters.num_combination_codeword_checks;
        let num_revealed_row_indices = revealed_current_row_indices.len();
        let num_base_table_rows = base_table_rows.len();
        let num_ext_table_rows = ext_table_rows.len();
        let num_revealed_quotient_values = revealed_quotient_values.len();
        let num_revealed_fri_values = revealed_fri_values.len();
        if num_revealed_row_indices != num_checks
            || num_base_table_rows != num_checks
            || num_ext_table_rows != num_checks
            || num_revealed_quotient_values != num_checks
            || num_revealed_fri_values != num_checks
        {
            bail!(
                "Expected {num_checks} revealed indices and values, but got \
                    {num_revealed_row_indices} revealed row indices, \
                    {num_base_table_rows} base table rows, \
                    {num_ext_table_rows} extension table rows, \
                    {num_revealed_quotient_values} quotient values, and \
                    {num_revealed_fri_values} FRI values."
            );
        }
        prof_start!(maybe_profiler, "main loop");
        for (row_idx, base_row, ext_row, quotient_value, fri_value) in izip!(
            revealed_current_row_indices,
            base_table_rows,
            ext_table_rows,
            revealed_quotient_values,
            revealed_fri_values,
        ) {
            prof_itr0!(maybe_profiler, "main loop");
            let (current_ext_row, randomizer_row) = ext_row.split_at(NUM_EXT_COLUMNS);
            let base_row = Array1::from(base_row);
            let ext_row = Array1::from(current_ext_row.to_vec());
            let randomizer_row = Array1::from(randomizer_row.to_vec());
            let current_fri_domain_value = fri.domain.domain_value(row_idx as u32);

            prof_start!(maybe_profiler, "base & ext elements", "CC");
            let base_and_ext_curr_row_element = Self::linearly_sum_base_and_ext_row(
                base_row.view(),
                ext_row.view(),
                base_and_ext_codeword_weights.view(),
                maybe_profiler,
            );
            prof_stop!(maybe_profiler, "base & ext elements");

            prof_start!(maybe_profiler, "DEEP update");
            let base_and_ext_and_quot_curr_row_deep_value = Self::deep_update(
                current_fri_domain_value,
                base_and_ext_curr_row_element + quotient_value,
                out_of_domain_point_curr_row,
                out_of_domain_curr_row_base_and_ext_and_quot_value,
            );
            let base_and_ext_next_row_deep_value = Self::deep_update(
                current_fri_domain_value,
                base_and_ext_curr_row_element,
                out_of_domain_point_next_row,
                out_of_domain_next_row_base_and_ext_value,
            );
            prof_stop!(maybe_profiler, "DEEP update");

            prof_start!(maybe_profiler, "combination codeword equality");
            let randomizer_codewords_contribution = randomizer_row.sum();
            if fri_value
                != base_and_ext_and_quot_curr_row_deep_value
                    + base_and_ext_next_row_deep_value
                    + randomizer_codewords_contribution
            {
                bail!("Revealed and computed leaf of the combination codeword must equal.");
            }
            prof_stop!(maybe_profiler, "combination codeword equality");
        }
        prof_stop!(maybe_profiler, "main loop");
        prof_stop!(maybe_profiler, "linear combination");
        Ok(true)
    }

    fn linearly_sum_base_and_ext_row<FF>(
        base_row: ArrayView1<FF>,
        ext_row: ArrayView1<XFieldElement>,
        weights: ArrayView1<XFieldElement>,
        maybe_profiler: &mut Option<TritonProfiler>,
    ) -> XFieldElement
    where
        FF: FiniteField + Into<XFieldElement>,
        XFieldElement: Mul<FF, Output = XFieldElement>,
    {
        prof_start!(maybe_profiler, "collect");
        let mut row = base_row.map(|&element| element.into());
        row.append(Axis(0), ext_row).unwrap();
        prof_stop!(maybe_profiler, "collect");
        prof_start!(maybe_profiler, "inner product");
        // todo: Try to get rid of this clone. The alternative line
        //   `let base_and_ext_element = (&weights * &summands).sum();`
        //   without cloning the weights does not compile for a seemingly nonsensical reason.
        let weights = weights.to_owned();
        let base_and_ext_element = (weights * row).sum();
        prof_stop!(maybe_profiler, "inner product");
        base_and_ext_element
    }
}

#[cfg(test)]
pub(crate) mod triton_stark_tests {
    use itertools::izip;
    use ndarray::Array1;
    use num_traits::Zero;
    use rand::prelude::ThreadRng;
    use rand::thread_rng;
    use rand::Rng;
    use rand_core::RngCore;
    use triton_opcodes::instruction::AnInstruction;
    use triton_opcodes::program::Program;
    use twenty_first::shared_math::other::random_elements;

    use crate::shared_tests::*;
    use crate::table::cascade_table::ExtCascadeTable;
    use crate::table::challenges::ChallengeId::StandardInputIndeterminate;
    use crate::table::challenges::ChallengeId::StandardInputTerminal;
    use crate::table::challenges::ChallengeId::StandardOutputIndeterminate;
    use crate::table::challenges::ChallengeId::StandardOutputTerminal;
    use crate::table::cross_table_argument::CrossTableArg;
    use crate::table::cross_table_argument::EvalArg;
    use crate::table::cross_table_argument::GrandCrossTableArg;
    use crate::table::extension_table::Evaluable;
    use crate::table::extension_table::Quotientable;
    use crate::table::hash_table::ExtHashTable;
    use crate::table::jump_stack_table::ExtJumpStackTable;
    use crate::table::lookup_table::ExtLookupTable;
    use crate::table::master_table::all_degrees_with_origin;
    use crate::table::master_table::MasterExtTable;
    use crate::table::master_table::TableId::ProcessorTable;
    use crate::table::op_stack_table::ExtOpStackTable;
    use crate::table::processor_table::ExtProcessorTable;
    use crate::table::program_table::ExtProgramTable;
    use crate::table::ram_table::ExtRamTable;
    use crate::table::table_column::MasterBaseTableColumn;
    use crate::table::table_column::MasterExtTableColumn;
    use crate::table::table_column::ProcessorBaseTableColumn;
    use crate::table::table_column::ProcessorExtTableColumn::InputTableEvalArg;
    use crate::table::table_column::ProcessorExtTableColumn::OutputTableEvalArg;
    use crate::table::table_column::RamBaseTableColumn;
    use crate::table::u32_table::ExtU32Table;
    use crate::vm::simulate;
    use crate::vm::triton_vm_tests::property_based_test_programs;
    use crate::vm::triton_vm_tests::small_tasm_test_programs;
    use crate::vm::triton_vm_tests::test_hash_nop_nop_lt;
    use crate::vm::AlgebraicExecutionTrace;

    use super::*;

    pub fn parse_setup_simulate(
        code: &str,
        public_input: Vec<BFieldElement>,
        secret_input: Vec<BFieldElement>,
    ) -> (AlgebraicExecutionTrace, Vec<BFieldElement>) {
        let program = Program::from_code(code);

        assert!(program.is_ok(), "program parses correctly");
        let program = program.unwrap();

        let (aet, stdout, err) = simulate(&program, public_input, secret_input);
        if let Some(error) = err {
            panic!("The VM encountered the following problem: {error}");
        }

        (aet, stdout)
    }

    pub fn parse_simulate_pad(
        code: &str,
        stdin: Vec<BFieldElement>,
        secret_in: Vec<BFieldElement>,
    ) -> (StarkParameters, Claim, MasterBaseTable, MasterBaseTable) {
        let (aet, stdout) = parse_setup_simulate(code, stdin.clone(), secret_in);

        let padded_height = MasterBaseTable::padded_height(&aet);
        let padded_height = BFieldElement::new(padded_height as u64);
        let claim = Claim {
            input: stdin,
            program_digest: Tip5::hash(&aet.program),
            output: stdout,
            padded_height,
        };
        let log_expansion_factor = 2;
        let security_level = 32;
        let parameters = StarkParameters::new(security_level, log_expansion_factor);
        let max_degree =
            Stark::derive_max_degree(claim.padded_height(), parameters.num_trace_randomizers);
        let fri = Stark::derive_fri(&parameters, max_degree);

        let mut master_base_table =
            MasterBaseTable::new(&aet, parameters.num_trace_randomizers, fri.domain);

        let unpadded_master_base_table = master_base_table.clone();
        master_base_table.pad();

        (
            parameters,
            claim,
            unpadded_master_base_table,
            master_base_table,
        )
    }

    pub fn parse_simulate_pad_extend(
        code: &str,
        stdin: Vec<BFieldElement>,
        secret_in: Vec<BFieldElement>,
    ) -> (
        StarkParameters,
        Claim,
        MasterBaseTable,
        MasterBaseTable,
        MasterExtTable,
        Challenges,
    ) {
        let (parameters, claim, unpadded_master_base_table, master_base_table) =
            parse_simulate_pad(code, stdin, secret_in);

        let dummy_challenges = Challenges::placeholder(&claim.input, &claim.output);
        let master_ext_table =
            master_base_table.extend(&dummy_challenges, parameters.num_randomizer_polynomials);

        (
            parameters,
            claim,
            unpadded_master_base_table,
            master_base_table,
            master_ext_table,
            dummy_challenges,
        )
    }

    #[test]
    pub fn print_ram_table_example_for_specification() {
        let program = "
        push  5 push  6 write_mem pop
        push 15 push 16 write_mem pop
        push  5 read_mem pop pop
        push 15 read_mem pop pop
        push  5 push  7 write_mem pop
        push 15 read_mem
        push  5 read_mem
        halt
        ";
        let (_, _, master_base_table, _) = parse_simulate_pad(program, vec![], vec![]);

        println!();
        println!("Processor Table:");
        println!(
            "| clk        | pi         | ci         | nia        | st0        \
             | st1        | st2        | st3        | ramp       | ramv       |"
        );
        println!(
            "|-----------:|:-----------|:-----------|:-----------|-----------:\
             |-----------:|-----------:|-----------:|-----------:|-----------:|"
        );
        for row in master_base_table.table(ProcessorTable).rows() {
            let clk = row[ProcessorBaseTableColumn::CLK.base_table_index()].to_string();
            let st0 = row[ProcessorBaseTableColumn::ST0.base_table_index()].to_string();
            let st1 = row[ProcessorBaseTableColumn::ST1.base_table_index()].to_string();
            let st2 = row[ProcessorBaseTableColumn::ST2.base_table_index()].to_string();
            let st3 = row[ProcessorBaseTableColumn::ST3.base_table_index()].to_string();
            let ramp = row[ProcessorBaseTableColumn::RAMP.base_table_index()].to_string();
            let ramv = row[ProcessorBaseTableColumn::RAMV.base_table_index()].to_string();

            let prev_instruction =
                row[ProcessorBaseTableColumn::PreviousInstruction.base_table_index()].value();
            let curr_instruction = row[ProcessorBaseTableColumn::CI.base_table_index()].value();
            let next_instruction_or_arg =
                row[ProcessorBaseTableColumn::NIA.base_table_index()].value();

            // sorry about this mess – this is just a test.
            let pi = match AnInstruction::<BFieldElement>::try_from(prev_instruction) {
                Ok(AnInstruction::Halt) | Err(_) => "-".to_string(),
                Ok(instr) => instr.to_string().split('0').collect_vec()[0].to_owned(),
            };
            let ci = AnInstruction::<BFieldElement>::try_from(curr_instruction).unwrap();
            let nia = if ci.size() == 2 {
                next_instruction_or_arg.to_string()
            } else {
                AnInstruction::<BFieldElement>::try_from(next_instruction_or_arg)
                    .unwrap()
                    .to_string()
                    .split('0')
                    .collect_vec()[0]
                    .to_owned()
            };
            let ci_string = if ci.size() == 1 {
                ci.to_string()
            } else {
                ci.to_string().split('0').collect_vec()[0].to_owned()
            };
            let interesting_cols = [clk, pi, ci_string, nia, st0, st1, st2, st3, ramp, ramv];
            println!(
                "{}",
                interesting_cols
                    .iter()
                    .map(|ff| format!("{:>10}", format!("{ff}")))
                    .collect_vec()
                    .join(" | ")
            );
        }
        println!();
        println!("RAM Table:");
        println!("| clk        | pi         | ramp       | ramv       | iord |");
        println!("|-----------:|:-----------|-----------:|-----------:|-----:|");
        for row in master_base_table.table(TableId::RamTable).rows() {
            let clk = row[RamBaseTableColumn::CLK.base_table_index()].to_string();
            let ramp = row[RamBaseTableColumn::RAMP.base_table_index()].to_string();
            let ramv = row[RamBaseTableColumn::RAMV.base_table_index()].to_string();
            let iord =
                row[RamBaseTableColumn::InverseOfRampDifference.base_table_index()].to_string();

            let prev_instruction =
                row[RamBaseTableColumn::PreviousInstruction.base_table_index()].value();
            let pi = match AnInstruction::<BFieldElement>::try_from(prev_instruction) {
                Ok(AnInstruction::Halt) | Err(_) => "-".to_string(),
                Ok(instr) => instr.to_string().split('0').collect_vec()[0].to_owned(),
            };
            let interersting_cols = [clk, pi, ramp, ramv, iord];
            println!(
                "{}",
                interersting_cols
                    .iter()
                    .map(|ff| format!("{:>10}", format!("{ff}")))
                    .collect_vec()
                    .join(" | ")
            );
        }
    }

    /// To be used with `-- --nocapture`. Has mainly informative purpose.
    #[test]
    pub fn print_all_constraint_degrees() {
        let padded_height = 2;
        let num_trace_randomizers = 2;
        let interpolant_degree = interpolant_degree(padded_height, num_trace_randomizers);
        for deg in all_degrees_with_origin(interpolant_degree, padded_height) {
            println!("{deg}");
        }
    }

    #[test]
    pub fn check_io_terminals() {
        let read_nop_code = "read_io read_io read_io nop nop write_io push 17 write_io halt";
        let public_input = [3, 5, 7].map(BFieldElement::new).to_vec();
        let (_, claim, _, _, master_ext_table, all_challenges) =
            parse_simulate_pad_extend(read_nop_code, public_input, vec![]);

        let processor_table = master_ext_table.table(ProcessorTable);
        let processor_table_last_row = processor_table.slice(s![-1, ..]);
        let ptie = processor_table_last_row[InputTableEvalArg.ext_table_index()];
        let ine = EvalArg::compute_terminal(
            &claim.input,
            EvalArg::default_initial(),
            all_challenges.get_challenge(StandardInputIndeterminate),
        );
        assert_eq!(ptie, ine, "The input evaluation arguments do not match.");

        let ptoe = processor_table_last_row[OutputTableEvalArg.ext_table_index()];
        let oute = EvalArg::compute_terminal(
            &claim.output,
            EvalArg::default_initial(),
            all_challenges.get_challenge(StandardOutputIndeterminate),
        );
        assert_eq!(ptoe, oute, "The output evaluation arguments do not match.");
    }

    #[test]
    pub fn check_grand_cross_table_argument() {
        let mut code_collection = small_tasm_test_programs();
        code_collection.append(&mut property_based_test_programs());

        for (code_idx, code_with_input) in code_collection.into_iter().enumerate() {
            println!("Checking Grand Cross-Table Argument for TASM snippet {code_idx}.");
            let (_, _, _, master_base_table, master_ext_table, all_challenges) =
                parse_simulate_pad_extend(
                    &code_with_input.source_code,
                    code_with_input.public_input(),
                    code_with_input.secret_input(),
                );

            let processor_table = master_ext_table.table(ProcessorTable);
            let processor_table_last_row = processor_table.slice(s![-1, ..]);
            assert_eq!(
                all_challenges.get_challenge(StandardInputTerminal),
                processor_table_last_row[InputTableEvalArg.ext_table_index()],
                "The input terminal must match for TASM snippet #{code_idx}."
            );
            assert_eq!(
                all_challenges.get_challenge(StandardOutputTerminal),
                processor_table_last_row[OutputTableEvalArg.ext_table_index()],
                "The output terminal must match for TASM snippet #{code_idx}."
            );

            let master_base_trace_table = master_base_table.trace_table();
            let master_ext_trace_table = master_ext_table.trace_table();
            let last_master_base_row = master_base_trace_table.slice(s![-1, ..]).map(|e| e.lift());
            let last_master_base_row = last_master_base_row.view();
            let last_master_ext_row = master_ext_trace_table.slice(s![-1, ..]);
            let evaluated_terminal_constraints = GrandCrossTableArg::evaluate_terminal_constraints(
                last_master_base_row,
                last_master_ext_row,
                &all_challenges,
            );
            assert_eq!(
                1,
                evaluated_terminal_constraints.len(),
                "The number of terminal constraints must be 1 – has the design changed?"
            );
            assert!(
                evaluated_terminal_constraints[0].is_zero(),
                "The terminal constraint must evaluate to 0 for TASM snippet #{code_idx}."
            );
        }
    }

    #[test]
    fn constraint_polynomials_use_right_variable_count_test() {
        let challenges = Challenges::placeholder(&[], &[]);
        let base_row = Array1::<BFieldElement>::zeros(NUM_BASE_COLUMNS);
        let ext_row = Array1::zeros(NUM_EXT_COLUMNS);

        let br = base_row.view();
        let er = ext_row.view();

        ExtProgramTable::evaluate_initial_constraints(br, er, &challenges);
        ExtProgramTable::evaluate_consistency_constraints(br, er, &challenges);
        ExtProgramTable::evaluate_transition_constraints(br, er, br, er, &challenges);
        ExtProgramTable::evaluate_terminal_constraints(br, er, &challenges);

        ExtProcessorTable::evaluate_initial_constraints(br, er, &challenges);
        ExtProcessorTable::evaluate_consistency_constraints(br, er, &challenges);
        ExtProcessorTable::evaluate_transition_constraints(br, er, br, er, &challenges);
        ExtProcessorTable::evaluate_terminal_constraints(br, er, &challenges);

        ExtOpStackTable::evaluate_initial_constraints(br, er, &challenges);
        ExtOpStackTable::evaluate_consistency_constraints(br, er, &challenges);
        ExtOpStackTable::evaluate_transition_constraints(br, er, br, er, &challenges);
        ExtOpStackTable::evaluate_terminal_constraints(br, er, &challenges);

        ExtRamTable::evaluate_initial_constraints(br, er, &challenges);
        ExtRamTable::evaluate_consistency_constraints(br, er, &challenges);
        ExtRamTable::evaluate_transition_constraints(br, er, br, er, &challenges);
        ExtRamTable::evaluate_terminal_constraints(br, er, &challenges);

        ExtJumpStackTable::evaluate_initial_constraints(br, er, &challenges);
        ExtJumpStackTable::evaluate_consistency_constraints(br, er, &challenges);
        ExtJumpStackTable::evaluate_transition_constraints(br, er, br, er, &challenges);
        ExtJumpStackTable::evaluate_terminal_constraints(br, er, &challenges);

        ExtHashTable::evaluate_initial_constraints(br, er, &challenges);
        ExtHashTable::evaluate_consistency_constraints(br, er, &challenges);
        ExtHashTable::evaluate_transition_constraints(br, er, br, er, &challenges);
        ExtHashTable::evaluate_terminal_constraints(br, er, &challenges);

        ExtU32Table::evaluate_initial_constraints(br, er, &challenges);
        ExtU32Table::evaluate_consistency_constraints(br, er, &challenges);
        ExtU32Table::evaluate_transition_constraints(br, er, br, er, &challenges);
        ExtU32Table::evaluate_terminal_constraints(br, er, &challenges);
    }

    #[test]
    fn print_number_of_all_constraints_per_table() {
        let table_names = [
            "program table",
            "processor table",
            "op stack table",
            "ram table",
            "jump stack table",
            "hash table",
            "cascade table",
            "lookup table",
            "u32 table",
            "cross-table arg",
        ];
        let all_init = [
            ExtProgramTable::num_initial_quotients(),
            ExtProcessorTable::num_initial_quotients(),
            ExtOpStackTable::num_initial_quotients(),
            ExtRamTable::num_initial_quotients(),
            ExtJumpStackTable::num_initial_quotients(),
            ExtHashTable::num_initial_quotients(),
            ExtCascadeTable::num_initial_quotients(),
            ExtLookupTable::num_initial_quotients(),
            ExtU32Table::num_initial_quotients(),
            GrandCrossTableArg::num_initial_quotients(),
        ];
        let all_cons = [
            ExtProgramTable::num_consistency_quotients(),
            ExtProcessorTable::num_consistency_quotients(),
            ExtOpStackTable::num_consistency_quotients(),
            ExtRamTable::num_consistency_quotients(),
            ExtJumpStackTable::num_consistency_quotients(),
            ExtHashTable::num_consistency_quotients(),
            ExtCascadeTable::num_consistency_quotients(),
            ExtLookupTable::num_consistency_quotients(),
            ExtU32Table::num_consistency_quotients(),
            GrandCrossTableArg::num_consistency_quotients(),
        ];
        let all_trans = [
            ExtProgramTable::num_transition_quotients(),
            ExtProcessorTable::num_transition_quotients(),
            ExtOpStackTable::num_transition_quotients(),
            ExtRamTable::num_transition_quotients(),
            ExtJumpStackTable::num_transition_quotients(),
            ExtHashTable::num_transition_quotients(),
            ExtCascadeTable::num_transition_quotients(),
            ExtLookupTable::num_transition_quotients(),
            ExtU32Table::num_transition_quotients(),
            GrandCrossTableArg::num_transition_quotients(),
        ];
        let all_term = [
            ExtProgramTable::num_terminal_quotients(),
            ExtProcessorTable::num_terminal_quotients(),
            ExtOpStackTable::num_terminal_quotients(),
            ExtRamTable::num_terminal_quotients(),
            ExtJumpStackTable::num_terminal_quotients(),
            ExtHashTable::num_terminal_quotients(),
            ExtCascadeTable::num_terminal_quotients(),
            ExtLookupTable::num_terminal_quotients(),
            ExtU32Table::num_terminal_quotients(),
            GrandCrossTableArg::num_terminal_quotients(),
        ];

        let num_total_init: usize = all_init.iter().sum();
        let num_total_cons: usize = all_cons.iter().sum();
        let num_total_trans: usize = all_trans.iter().sum();
        let num_total_term: usize = all_term.iter().sum();
        let num_total = num_total_init + num_total_cons + num_total_trans + num_total_term;

        println!();
        println!("| Table                |  Init |  Cons | Trans |  Term |   Sum |");
        println!("|:---------------------|------:|------:|------:|------:|------:|");
        for (name, num_init, num_cons, num_trans, num_term) in
            izip!(table_names, all_init, all_cons, all_trans, all_term)
        {
            let num_total = num_init + num_cons + num_trans + num_term;
            println!(
                "| {name:<20} | {num_init:>5} | {num_cons:>5} \
                 | {num_trans:>5} | {num_term:>5} | {num_total:>5} |",
            );
        }
        println!(
            "| {:<20} | {num_total_init:>5} | {num_total_cons:>5} \
             | {num_total_trans:>5} | {num_total_term:>5} | {num_total:>5} |",
            "Sum",
        );
    }

    #[test]
    fn number_of_quotient_degree_bounds_match_number_of_constraints_test() {
        let base_row = Array1::<BFieldElement>::zeros(NUM_BASE_COLUMNS);
        let ext_row = Array1::zeros(NUM_EXT_COLUMNS);
        let challenges = Challenges::placeholder(&[], &[]);
        let padded_height = 2;
        let num_trace_randomizers = 2;
        let interpolant_degree = interpolant_degree(padded_height, num_trace_randomizers);

        // Shorten some names for better formatting. This is just a test.
        let ph = padded_height;
        let id = interpolant_degree;
        let br = base_row.view();
        let er = ext_row.view();

        assert_eq!(
            ExtProgramTable::num_initial_quotients(),
            ExtProgramTable::evaluate_initial_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtProgramTable::num_initial_quotients(),
            ExtProgramTable::initial_quotient_degree_bounds(id).len()
        );
        assert_eq!(
            ExtProcessorTable::num_initial_quotients(),
            ExtProcessorTable::evaluate_initial_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtProcessorTable::num_initial_quotients(),
            ExtProcessorTable::initial_quotient_degree_bounds(id).len()
        );
        assert_eq!(
            ExtOpStackTable::num_initial_quotients(),
            ExtOpStackTable::evaluate_initial_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtOpStackTable::num_initial_quotients(),
            ExtOpStackTable::initial_quotient_degree_bounds(id).len()
        );
        assert_eq!(
            ExtRamTable::num_initial_quotients(),
            ExtRamTable::evaluate_initial_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtRamTable::num_initial_quotients(),
            ExtRamTable::initial_quotient_degree_bounds(id).len()
        );
        assert_eq!(
            ExtJumpStackTable::num_initial_quotients(),
            ExtJumpStackTable::evaluate_initial_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtJumpStackTable::num_initial_quotients(),
            ExtJumpStackTable::initial_quotient_degree_bounds(id).len()
        );
        assert_eq!(
            ExtHashTable::num_initial_quotients(),
            ExtHashTable::evaluate_initial_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtHashTable::num_initial_quotients(),
            ExtHashTable::initial_quotient_degree_bounds(id).len()
        );
        assert_eq!(
            ExtCascadeTable::num_initial_quotients(),
            ExtCascadeTable::evaluate_initial_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtCascadeTable::num_initial_quotients(),
            ExtCascadeTable::initial_quotient_degree_bounds(id).len()
        );
        assert_eq!(
            ExtLookupTable::num_initial_quotients(),
            ExtLookupTable::evaluate_initial_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtLookupTable::num_initial_quotients(),
            ExtLookupTable::initial_quotient_degree_bounds(id).len()
        );
        assert_eq!(
            ExtU32Table::num_initial_quotients(),
            ExtU32Table::evaluate_initial_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtU32Table::num_initial_quotients(),
            ExtU32Table::initial_quotient_degree_bounds(id).len()
        );
        assert_eq!(
            GrandCrossTableArg::num_initial_quotients(),
            GrandCrossTableArg::evaluate_initial_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            GrandCrossTableArg::num_initial_quotients(),
            GrandCrossTableArg::initial_quotient_degree_bounds(id).len()
        );

        assert_eq!(
            ExtProgramTable::num_consistency_quotients(),
            ExtProgramTable::evaluate_consistency_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtProgramTable::num_consistency_quotients(),
            ExtProgramTable::consistency_quotient_degree_bounds(id, ph).len()
        );
        assert_eq!(
            ExtProcessorTable::num_consistency_quotients(),
            ExtProcessorTable::evaluate_consistency_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtProcessorTable::num_consistency_quotients(),
            ExtProcessorTable::consistency_quotient_degree_bounds(id, ph).len()
        );
        assert_eq!(
            ExtOpStackTable::num_consistency_quotients(),
            ExtOpStackTable::evaluate_consistency_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtOpStackTable::num_consistency_quotients(),
            ExtOpStackTable::consistency_quotient_degree_bounds(id, ph).len()
        );
        assert_eq!(
            ExtRamTable::num_consistency_quotients(),
            ExtRamTable::evaluate_consistency_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtRamTable::num_consistency_quotients(),
            ExtRamTable::consistency_quotient_degree_bounds(id, ph).len()
        );
        assert_eq!(
            ExtJumpStackTable::num_consistency_quotients(),
            ExtJumpStackTable::evaluate_consistency_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtJumpStackTable::num_consistency_quotients(),
            ExtJumpStackTable::consistency_quotient_degree_bounds(id, ph).len()
        );
        assert_eq!(
            ExtHashTable::num_consistency_quotients(),
            ExtHashTable::evaluate_consistency_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtHashTable::num_consistency_quotients(),
            ExtHashTable::consistency_quotient_degree_bounds(id, ph).len()
        );
        assert_eq!(
            ExtCascadeTable::num_consistency_quotients(),
            ExtCascadeTable::evaluate_consistency_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtCascadeTable::num_consistency_quotients(),
            ExtCascadeTable::consistency_quotient_degree_bounds(id, ph).len()
        );
        assert_eq!(
            ExtLookupTable::num_consistency_quotients(),
            ExtLookupTable::evaluate_consistency_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtLookupTable::num_consistency_quotients(),
            ExtLookupTable::consistency_quotient_degree_bounds(id, ph).len()
        );
        assert_eq!(
            ExtU32Table::num_consistency_quotients(),
            ExtU32Table::evaluate_consistency_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtU32Table::num_consistency_quotients(),
            ExtU32Table::consistency_quotient_degree_bounds(id, ph).len()
        );
        assert_eq!(
            GrandCrossTableArg::num_consistency_quotients(),
            GrandCrossTableArg::evaluate_consistency_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            GrandCrossTableArg::num_consistency_quotients(),
            GrandCrossTableArg::consistency_quotient_degree_bounds(id, ph).len()
        );

        assert_eq!(
            ExtProgramTable::num_transition_quotients(),
            ExtProgramTable::evaluate_transition_constraints(br, er, br, er, &challenges).len(),
        );
        assert_eq!(
            ExtProgramTable::num_transition_quotients(),
            ExtProgramTable::transition_quotient_degree_bounds(id, ph).len()
        );
        assert_eq!(
            ExtProcessorTable::num_transition_quotients(),
            ExtProcessorTable::evaluate_transition_constraints(br, er, br, er, &challenges).len(),
        );
        assert_eq!(
            ExtProcessorTable::num_transition_quotients(),
            ExtProcessorTable::transition_quotient_degree_bounds(id, ph).len()
        );
        assert_eq!(
            ExtOpStackTable::num_transition_quotients(),
            ExtOpStackTable::evaluate_transition_constraints(br, er, br, er, &challenges).len(),
        );
        assert_eq!(
            ExtOpStackTable::num_transition_quotients(),
            ExtOpStackTable::transition_quotient_degree_bounds(id, ph).len()
        );
        assert_eq!(
            ExtRamTable::num_transition_quotients(),
            ExtRamTable::evaluate_transition_constraints(br, er, br, er, &challenges).len(),
        );
        assert_eq!(
            ExtRamTable::num_transition_quotients(),
            ExtRamTable::transition_quotient_degree_bounds(id, ph).len()
        );
        assert_eq!(
            ExtJumpStackTable::num_transition_quotients(),
            ExtJumpStackTable::evaluate_transition_constraints(br, er, br, er, &challenges).len(),
        );
        assert_eq!(
            ExtJumpStackTable::num_transition_quotients(),
            ExtJumpStackTable::transition_quotient_degree_bounds(id, ph).len()
        );
        assert_eq!(
            ExtHashTable::num_transition_quotients(),
            ExtHashTable::evaluate_transition_constraints(br, er, br, er, &challenges).len(),
        );
        assert_eq!(
            ExtHashTable::num_transition_quotients(),
            ExtHashTable::transition_quotient_degree_bounds(id, ph).len()
        );
        assert_eq!(
            ExtCascadeTable::num_transition_quotients(),
            ExtCascadeTable::evaluate_transition_constraints(br, er, br, er, &challenges).len(),
        );
        assert_eq!(
            ExtCascadeTable::num_transition_quotients(),
            ExtCascadeTable::transition_quotient_degree_bounds(id, ph).len()
        );
        assert_eq!(
            ExtLookupTable::num_transition_quotients(),
            ExtLookupTable::evaluate_transition_constraints(br, er, br, er, &challenges).len(),
        );
        assert_eq!(
            ExtLookupTable::num_transition_quotients(),
            ExtLookupTable::transition_quotient_degree_bounds(id, ph).len()
        );
        assert_eq!(
            ExtU32Table::num_transition_quotients(),
            ExtU32Table::evaluate_transition_constraints(br, er, br, er, &challenges).len(),
        );
        assert_eq!(
            ExtU32Table::num_transition_quotients(),
            ExtU32Table::transition_quotient_degree_bounds(id, ph).len()
        );
        assert_eq!(
            GrandCrossTableArg::num_transition_quotients(),
            GrandCrossTableArg::evaluate_transition_constraints(br, er, br, er, &challenges).len(),
        );
        assert_eq!(
            GrandCrossTableArg::num_transition_quotients(),
            GrandCrossTableArg::transition_quotient_degree_bounds(id, ph).len()
        );

        assert_eq!(
            ExtProgramTable::num_terminal_quotients(),
            ExtProgramTable::evaluate_terminal_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtProgramTable::num_terminal_quotients(),
            ExtProgramTable::terminal_quotient_degree_bounds(id).len()
        );
        assert_eq!(
            ExtProcessorTable::num_terminal_quotients(),
            ExtProcessorTable::evaluate_terminal_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtProcessorTable::num_terminal_quotients(),
            ExtProcessorTable::terminal_quotient_degree_bounds(id).len()
        );
        assert_eq!(
            ExtOpStackTable::num_terminal_quotients(),
            ExtOpStackTable::evaluate_terminal_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtOpStackTable::num_terminal_quotients(),
            ExtOpStackTable::terminal_quotient_degree_bounds(id).len()
        );
        assert_eq!(
            ExtRamTable::num_terminal_quotients(),
            ExtRamTable::evaluate_terminal_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtRamTable::num_terminal_quotients(),
            ExtRamTable::terminal_quotient_degree_bounds(id).len()
        );
        assert_eq!(
            ExtJumpStackTable::num_terminal_quotients(),
            ExtJumpStackTable::evaluate_terminal_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtJumpStackTable::num_terminal_quotients(),
            ExtJumpStackTable::terminal_quotient_degree_bounds(id).len()
        );
        assert_eq!(
            ExtHashTable::num_terminal_quotients(),
            ExtHashTable::evaluate_terminal_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtHashTable::num_terminal_quotients(),
            ExtHashTable::terminal_quotient_degree_bounds(id).len()
        );
        assert_eq!(
            ExtCascadeTable::num_terminal_quotients(),
            ExtCascadeTable::evaluate_terminal_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtCascadeTable::num_terminal_quotients(),
            ExtCascadeTable::terminal_quotient_degree_bounds(id).len()
        );
        assert_eq!(
            ExtLookupTable::num_terminal_quotients(),
            ExtLookupTable::evaluate_terminal_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtLookupTable::num_terminal_quotients(),
            ExtLookupTable::terminal_quotient_degree_bounds(id).len()
        );
        assert_eq!(
            ExtU32Table::num_terminal_quotients(),
            ExtU32Table::evaluate_terminal_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtU32Table::num_terminal_quotients(),
            ExtU32Table::terminal_quotient_degree_bounds(id).len()
        );
        assert_eq!(
            GrandCrossTableArg::num_terminal_quotients(),
            GrandCrossTableArg::evaluate_terminal_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            GrandCrossTableArg::num_terminal_quotients(),
            GrandCrossTableArg::terminal_quotient_degree_bounds(id).len()
        );
    }

    #[test]
    fn triton_table_constraints_evaluate_to_zero_on_halt_test() {
        triton_table_constraints_evaluate_to_zero(test_halt());
    }

    #[test]
    fn triton_table_constraints_evaluate_to_zero_on_fibonacci_test() {
        let source_code_and_input = SourceCodeAndInput {
            source_code: FIBONACCI_SEQUENCE.to_string(),
            input: vec![100],
            secret_input: vec![],
        };
        triton_table_constraints_evaluate_to_zero(source_code_and_input);
    }

    #[test]
    fn triton_table_constraints_evaluate_to_zero_on_small_programs_test() {
        for (program_idx, program) in small_tasm_test_programs().into_iter().enumerate() {
            println!("Testing program with index {program_idx}.");
            triton_table_constraints_evaluate_to_zero(program);
        }
    }

    #[test]
    fn triton_table_constraints_evaluate_to_zero_on_property_based_programs_test() {
        for (program_idx, program) in property_based_test_programs().into_iter().enumerate() {
            println!("Testing program with index {program_idx}.");
            triton_table_constraints_evaluate_to_zero(program);
        }
    }

    pub fn triton_table_constraints_evaluate_to_zero(source_code_and_input: SourceCodeAndInput) {
        let zero = XFieldElement::zero();
        let (_, _, _, master_base_table, master_ext_table, challenges) = parse_simulate_pad_extend(
            &source_code_and_input.source_code,
            source_code_and_input.public_input(),
            source_code_and_input.secret_input(),
        );

        assert_eq!(
            master_base_table.master_base_matrix.nrows(),
            master_ext_table.master_ext_matrix.nrows()
        );
        let master_base_trace_table = master_base_table.trace_table();
        let master_ext_trace_table = master_ext_table.trace_table();
        assert_eq!(
            master_base_trace_table.nrows(),
            master_ext_trace_table.nrows()
        );

        let first_base_row_lifted = master_base_trace_table.row(0).map(|e| e.lift());
        let evaluated_initial_constraints = evaluate_all_initial_constraints(
            first_base_row_lifted.view(),
            master_ext_trace_table.row(0),
            &challenges,
        );
        let num_initial_constraints = evaluated_initial_constraints.len();
        assert_eq!(num_all_initial_quotients(), num_initial_constraints);
        for (constraint_idx, ebc) in evaluated_initial_constraints.into_iter().enumerate() {
            let (table_idx, table_name) = initial_constraint_table_idx_and_name(constraint_idx);
            assert_eq!(
                zero, ebc,
                "Failed initial constraint with global index {constraint_idx}. \
                Total number of initial constraints: {num_initial_constraints}. \
                Table: {table_name}. Index within table: {table_idx}",
            );
        }

        let num_rows = master_base_trace_table.nrows();
        for row_idx in 0..num_rows {
            let base_row = master_base_trace_table.row(row_idx);
            let ext_row = master_ext_trace_table.row(row_idx);
            let base_row_lifted = base_row.map(|e| e.lift());
            let evaluated_consistency_constraints =
                evaluate_all_consistency_constraints(base_row_lifted.view(), ext_row, &challenges);
            let num_consistency_constraints = evaluated_consistency_constraints.len();
            assert_eq!(num_all_consistency_quotients(), num_consistency_constraints);
            for (constraint_idx, ecc) in evaluated_consistency_constraints.into_iter().enumerate() {
                let (table_idx, table_name) =
                    consistency_constraint_table_idx_and_name(constraint_idx);
                assert_eq!(
                    zero, ecc,
                    "Failed consistency constraint with global index {constraint_idx}. \
                    Total number of consistency constraints: {num_consistency_constraints}. \
                    Table: {table_name}. Index within table: {table_idx} \
                    Row index: {row_idx}. \
                    Total rows: {num_rows}",
                );
            }
        }

        for row_idx in 0..num_rows - 1 {
            let base_row = master_base_trace_table.row(row_idx);
            let ext_row = master_ext_trace_table.row(row_idx);
            let next_base_row = master_base_trace_table.row(row_idx + 1);
            let next_ext_row = master_ext_trace_table.row(row_idx + 1);
            let base_row_lifted = base_row.map(|e| e.lift());
            let next_base_row_lifted = next_base_row.map(|e| e.lift());
            let evaluated_transition_constraints = evaluate_all_transition_constraints(
                base_row_lifted.view(),
                ext_row,
                next_base_row_lifted.view(),
                next_ext_row,
                &challenges,
            );
            let num_transition_constraints = evaluated_transition_constraints.len();
            assert_eq!(num_all_transition_quotients(), num_transition_constraints);
            for (constraint_idx, etc) in evaluated_transition_constraints.into_iter().enumerate() {
                if zero != etc {
                    let pi_idx =
                        ProcessorBaseTableColumn::PreviousInstruction.master_base_table_index();
                    let ci_idx = ProcessorBaseTableColumn::CI.master_base_table_index();
                    let nia_idx = ProcessorBaseTableColumn::NIA.master_base_table_index();
                    let pi = base_row[pi_idx].value();
                    let ci = base_row[ci_idx].value();
                    let nia = base_row[nia_idx].value();
                    let previous_instruction =
                        AnInstruction::<BFieldElement>::try_from(pi).unwrap();
                    let current_instruction = AnInstruction::<BFieldElement>::try_from(ci).unwrap();
                    let next_instruction_str = match AnInstruction::<BFieldElement>::try_from(nia) {
                        Ok(instr) => format!("{instr:?}"),
                        Err(_) => "not an instruction".to_string(),
                    };
                    let (table_idx, table_name) =
                        transition_constraint_table_idx_and_name(constraint_idx);
                    panic!(
                        "Failed transition constraint with global index {constraint_idx}. \
                        Total number of transition constraints: {num_transition_constraints}. \
                        Table: {table_name}. Index within table: {table_idx} \
                        Row index: {row_idx}. \
                        Total rows: {num_rows}\n\
                        Previous Instruction: {previous_instruction:?} – opcode: {pi}\n\
                        Current Instruction:  {current_instruction:?} – opcode: {ci}\n\
                        Next Instruction:     {next_instruction_str} – opcode: {nia}\n"
                    );
                }
            }
        }

        let last_row_lifted = master_base_trace_table.row(num_rows - 1).map(|e| e.lift());
        let evaluated_terminal_constraints = evaluate_all_terminal_constraints(
            last_row_lifted.view(),
            master_ext_trace_table.row(num_rows - 1),
            &challenges,
        );
        let num_terminal_constraints = evaluated_terminal_constraints.len();
        assert_eq!(num_all_terminal_quotients(), num_terminal_constraints);
        for (constraint_idx, etermc) in evaluated_terminal_constraints.into_iter().enumerate() {
            let (table_idx, table_name) = terminal_constraint_table_idx_and_name(constraint_idx);
            assert_eq!(
                zero, etermc,
                "Failed terminal constraint with global index {constraint_idx}. \
                Total number of terminal constraints: {num_terminal_constraints}. \
                Table: {table_name}. Index within table: {table_idx}",
            );
        }
    }

    #[test]
    fn triton_prove_verify_simple_program_test() {
        let code_with_input = test_hash_nop_nop_lt();
        let (parameters, claim, proof) = parse_simulate_prove(
            &code_with_input.source_code,
            code_with_input.public_input(),
            code_with_input.secret_input(),
            &mut None,
        );

        println!("between prove and verify");

        let result = Stark::verify(&parameters, &claim, &proof, &mut None);
        if let Err(e) = result {
            panic!("The Verifier is unhappy! {e}");
        }
        assert!(result.unwrap());
    }

    #[test]
    fn triton_prove_verify_halt_test() {
        let code_with_input = test_halt();
        let mut profiler = Some(TritonProfiler::new("Prove Halt"));
        let (parameters, claim, proof) = parse_simulate_prove(
            &code_with_input.source_code,
            code_with_input.public_input(),
            code_with_input.secret_input(),
            &mut profiler,
        );
        let mut profiler = profiler.unwrap();
        profiler.finish();

        let result = Stark::verify(&parameters, &claim, &proof, &mut None);
        if let Err(e) = result {
            panic!("The Verifier is unhappy! {e}");
        }
        assert!(result.unwrap());

        let max_degree =
            Stark::derive_max_degree(claim.padded_height(), parameters.num_trace_randomizers);
        let fri = Stark::derive_fri(&parameters, max_degree);
        let report = profiler.report(None, Some(claim.padded_height()), Some(fri.domain.length));
        println!("{report}");
    }

    #[test]
    #[ignore = "used for tracking&debugging deserialization errors"]
    fn triton_prove_halt_save_error_test() {
        let code_with_input = test_halt();

        for _ in 0..100 {
            let (parameters, claim, proof) = parse_simulate_prove(
                &code_with_input.source_code,
                code_with_input.public_input(),
                code_with_input.secret_input(),
                &mut None,
            );

            let filename = "halt_error.tsp";
            let result = Stark::verify(&parameters, &claim, &proof, &mut None);
            if let Err(e) = result {
                if let Err(e) = save_proof(filename, proof) {
                    panic!("Unsyntactical proof and can't save! {e}");
                }
                panic!("Saved proof to {filename} because verifier is unhappy! {e}");
            }
            assert!(result.unwrap());
        }
    }

    #[test]
    #[ignore = "used for tracking&debugging deserialization errors"]
    fn triton_load_verify_halt_test() {
        let code_with_input = test_halt();
        let (parameters, claim, _) = parse_simulate_prove(
            &code_with_input.source_code,
            code_with_input.public_input(),
            code_with_input.secret_input(),
            &mut None,
        );

        let filename = "halt_error.tsp";
        let proof = match load_proof(filename) {
            Ok(p) => p,
            Err(e) => panic!("Could not load proof from disk at {filename}: {e}"),
        };

        let result = Stark::verify(&parameters, &claim, &proof, &mut None);
        if let Err(e) = result {
            panic!("Verifier is unhappy! {e}");
        }
        assert!(result.unwrap());
    }

    #[test]
    fn prove_verify_fibonacci_100_test() {
        let source_code = FIBONACCI_SEQUENCE;
        let stdin = [100].map(BFieldElement::new).to_vec();
        let secret_in = vec![];

        let mut profiler = Some(TritonProfiler::new("Prove Fib 100"));
        let (parameters, claim, proof) =
            parse_simulate_prove(source_code, stdin, secret_in, &mut profiler);
        let mut profiler = profiler.unwrap();
        profiler.finish();

        println!("between prove and verify");

        let result = Stark::verify(&parameters, &claim, &proof, &mut None);
        if let Err(e) = result {
            panic!("The Verifier is unhappy! {e}");
        }
        assert!(result.unwrap());

        let max_degree =
            Stark::derive_max_degree(claim.padded_height(), parameters.num_trace_randomizers);
        let fri = Stark::derive_fri(&parameters, max_degree);
        let report = profiler.report(None, Some(claim.padded_height()), Some(fri.domain.length));
        println!("{report}");
    }

    #[test]
    fn prove_verify_fib_shootout_test() {
        let code = FIBONACCI_SEQUENCE;

        for (fib_seq_idx, fib_seq_val) in [(0, 1), (7, 21), (11, 144)] {
            let stdin = [fib_seq_idx].map(BFieldElement::new).to_vec();
            let secret_in = vec![];
            let (parameters, claim, proof) =
                parse_simulate_prove(code, stdin, secret_in, &mut None);
            match Stark::verify(&parameters, &claim, &proof, &mut None) {
                Ok(result) => assert!(result, "The Verifier disagrees!"),
                Err(err) => panic!("The Verifier is unhappy! {err}"),
            }

            assert_eq!(vec![fib_seq_val], claim.public_output());
        }
    }

    #[test]
    fn constraints_evaluate_to_zero_on_many_u32_operations_test() {
        let many_u32_instructions = SourceCodeAndInput::without_input(MANY_U32_INSTRUCTIONS);
        triton_table_constraints_evaluate_to_zero(many_u32_instructions);
    }

    #[test]
    fn triton_prove_verify_many_u32_operations_test() {
        let mut profiler = Some(TritonProfiler::new("Prove Many U32 Ops"));
        let (parameters, claim, proof) =
            parse_simulate_prove(MANY_U32_INSTRUCTIONS, vec![], vec![], &mut profiler);
        let mut profiler = profiler.unwrap();
        profiler.finish();

        let result = Stark::verify(&parameters, &claim, &proof, &mut None);
        if let Err(e) = result {
            panic!("The Verifier is unhappy! {e}");
        }
        assert!(result.unwrap());

        let max_degree =
            Stark::derive_max_degree(claim.padded_height(), parameters.num_trace_randomizers);
        let fri = Stark::derive_fri(&parameters, max_degree);
        let report = profiler.report(None, Some(claim.padded_height()), Some(fri.domain.length));
        println!("{report}");
    }

    #[test]
    #[ignore = "stress test"]
    fn prove_fib_successively_larger() {
        let source_code = FIBONACCI_SEQUENCE;

        for fibonacci_number in [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200] {
            let stdin = [fibonacci_number].map(BFieldElement::new).to_vec();
            let fib_test_name = format!("element #{fibonacci_number:>4} from Fibonacci sequence");
            let mut profiler = Some(TritonProfiler::new(&fib_test_name));
            let (parameters, claim, _) =
                parse_simulate_prove(source_code, stdin, vec![], &mut profiler);
            let mut profiler = profiler.unwrap();
            profiler.finish();

            let max_degree =
                Stark::derive_max_degree(claim.padded_height(), parameters.num_trace_randomizers);
            let fri = Stark::derive_fri(&parameters, max_degree);
            let report =
                profiler.report(None, Some(claim.padded_height()), Some(fri.domain.length));
            println!("{report}");
        }
    }

    #[test]
    #[should_panic(expected = "Failed to convert BFieldElement")]
    pub fn negative_log_2_floor_test() {
        let mut rng = ThreadRng::default();
        let st0 = (rng.next_u32() as u64) << 32;

        let source_code = format!("push {st0} log_2_floor halt");
        let (parameters, claim, proof) =
            parse_simulate_prove(&source_code, vec![], vec![], &mut None);
        let result = Stark::verify(&parameters, &claim, &proof, &mut None);
        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    #[should_panic(expected = "The logarithm of 0 does not exist")]
    pub fn negative_log_2_floor_of_0_test() {
        let source_code = "push 0 log_2_floor halt";
        let (parameters, claim, proof) =
            parse_simulate_prove(source_code, vec![], vec![], &mut None);
        let result = Stark::verify(&parameters, &claim, &proof, &mut None);
        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    pub fn deep_update_test() {
        let domain_length = 1 << 10;
        let domain = ArithmeticDomain::new_no_offset(domain_length);

        let poly_degree = thread_rng().gen_range(2..20);
        let low_deg_poly_coeffs: Vec<XFieldElement> = random_elements(poly_degree);
        let low_deg_poly = Polynomial::new(low_deg_poly_coeffs.to_vec());
        let low_deg_codeword = domain.evaluate(&low_deg_poly);

        let out_of_domain_point: XFieldElement = thread_rng().gen();
        let out_of_domain_value = low_deg_poly.evaluate(&out_of_domain_point);

        let deep_poly = Stark::deep_codeword(
            &low_deg_codeword,
            domain,
            out_of_domain_point,
            out_of_domain_value,
        );
        let poly_of_maybe_low_degree = domain.interpolate(&deep_poly);
        assert_eq!(poly_degree as isize - 2, poly_of_maybe_low_degree.degree());

        let bogus_out_of_domain_value = thread_rng().gen();
        let bogus_deep_poly = Stark::deep_codeword(
            &low_deg_codeword,
            domain,
            out_of_domain_point,
            bogus_out_of_domain_value,
        );
        let poly_of_hopefully_high_degree = domain.interpolate(&bogus_deep_poly);
        assert_eq!(
            domain_length as isize - 1,
            poly_of_hopefully_high_degree.degree()
        );
    }
}
