use std::ops::Add;
use std::ops::Mul;

use anyhow::bail;
use anyhow::Result;
use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use itertools::izip;
use itertools::Itertools;
use ndarray::prelude::*;
use ndarray::Zip;
use num_traits::One;
use rayon::prelude::*;
use serde::Deserialize;
use serde::Serialize;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::digest::Digest;
use twenty_first::shared_math::mpolynomial::Degree;
use twenty_first::shared_math::polynomial::Polynomial;
use twenty_first::shared_math::tip5::Tip5;
use twenty_first::shared_math::traits::FiniteField;
use twenty_first::shared_math::traits::Inverse;
use twenty_first::shared_math::traits::ModPowU32;
use twenty_first::shared_math::x_field_element;
use twenty_first::shared_math::x_field_element::XFieldElement;
use twenty_first::util_types::algebraic_hasher::AlgebraicHasher;
use twenty_first::util_types::merkle_tree::CpuParallel;
use twenty_first::util_types::merkle_tree::MerkleTree;
use twenty_first::util_types::merkle_tree_maker::MerkleTreeMaker;

use crate::aet::AlgebraicExecutionTrace;
use crate::arithmetic_domain::ArithmeticDomain;
use crate::ensure_eq;
use crate::fri::Fri;
use crate::profiler::prof_itr0;
use crate::profiler::prof_start;
use crate::profiler::prof_stop;
use crate::profiler::TritonProfiler;
use crate::proof::Claim;
use crate::proof::Proof;
use crate::proof_item::ProofItem;
use crate::proof_stream::ProofStream;
use crate::table::challenges::Challenges;
use crate::table::extension_table::Evaluable;
use crate::table::master_table::*;

pub type StarkHasher = Tip5;
pub type StarkProofStream = ProofStream<StarkHasher>;

/// The Merkle tree maker in use. Keeping this as a type alias should make it easier to switch
/// between different Merkle tree makers.
pub type MTMaker = CpuParallel;

/// The number of segments the quotient polynomial is split into.
/// Helps keeping the FRI domain small.
pub(crate) const NUM_QUOTIENT_SEGMENTS: usize = AIR_TARGET_DEGREE as usize;

const NUM_DEEP_CODEWORD_COMPONENTS: usize = 3;

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

impl<'a> Arbitrary<'a> for StarkParameters {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let security_level = u.int_in_range(1..=640)?;
        let log_2_of_fri_expansion_factor = u.int_in_range(1..=8)?;
        Ok(Self::new(security_level, log_2_of_fri_expansion_factor))
    }
}

pub struct Stark {}

impl Stark {
    pub fn prove(
        parameters: StarkParameters,
        claim: &Claim,
        aet: &AlgebraicExecutionTrace,
        maybe_profiler: &mut Option<TritonProfiler>,
    ) -> Proof {
        prof_start!(maybe_profiler, "Fiat-Shamir: claim", "hash");
        let mut proof_stream = StarkProofStream::new();
        proof_stream.alter_fiat_shamir_state_with(claim);
        prof_stop!(maybe_profiler, "Fiat-Shamir: claim");

        prof_start!(maybe_profiler, "derive additional parameters");
        let padded_height = aet.padded_height();
        let max_degree = Self::derive_max_degree(padded_height, parameters.num_trace_randomizers);
        let fri = Self::derive_fri(parameters, padded_height);
        let quotient_domain = Self::quotient_domain(fri.domain, max_degree);
        proof_stream.enqueue(ProofItem::Log2PaddedHeight(padded_height.ilog2()));
        prof_stop!(maybe_profiler, "derive additional parameters");

        prof_start!(maybe_profiler, "base tables");
        prof_start!(maybe_profiler, "create", "gen");
        let mut master_base_table = MasterBaseTable::new(
            aet,
            parameters.num_trace_randomizers,
            quotient_domain,
            fri.domain,
        );
        prof_stop!(maybe_profiler, "create");

        prof_start!(maybe_profiler, "pad", "gen");
        master_base_table.pad();
        prof_stop!(maybe_profiler, "pad");

        prof_start!(maybe_profiler, "randomize trace", "gen");
        master_base_table.randomize_trace();
        prof_stop!(maybe_profiler, "randomize trace");

        prof_start!(maybe_profiler, "LDE", "LDE");
        master_base_table.low_degree_extend_all_columns();
        prof_stop!(maybe_profiler, "LDE");

        prof_start!(maybe_profiler, "Merkle tree", "hash");
        let base_merkle_tree = master_base_table.merkle_tree(maybe_profiler);
        prof_stop!(maybe_profiler, "Merkle tree");

        prof_start!(maybe_profiler, "Fiat-Shamir", "hash");
        proof_stream.enqueue(ProofItem::MerkleRoot(base_merkle_tree.get_root()));
        let challenges = proof_stream.sample_scalars(Challenges::num_challenges_to_sample());
        let challenges = Challenges::new(challenges, claim);
        prof_stop!(maybe_profiler, "Fiat-Shamir");

        prof_start!(maybe_profiler, "extend", "gen");
        let mut master_ext_table =
            master_base_table.extend(&challenges, parameters.num_randomizer_polynomials);
        prof_stop!(maybe_profiler, "extend");

        prof_start!(maybe_profiler, "randomize trace", "gen");
        master_ext_table.randomize_trace();
        prof_stop!(maybe_profiler, "randomize trace");
        prof_stop!(maybe_profiler, "base tables");

        prof_start!(maybe_profiler, "ext tables");
        prof_start!(maybe_profiler, "LDE", "LDE");
        master_ext_table.low_degree_extend_all_columns();
        prof_stop!(maybe_profiler, "LDE");

        prof_start!(maybe_profiler, "Merkle tree", "hash");
        let ext_merkle_tree = master_ext_table.merkle_tree(maybe_profiler);
        prof_stop!(maybe_profiler, "Merkle tree");

        prof_start!(maybe_profiler, "Fiat-Shamir", "hash");
        proof_stream.enqueue(ProofItem::MerkleRoot(ext_merkle_tree.get_root()));
        prof_stop!(maybe_profiler, "Fiat-Shamir");
        prof_stop!(maybe_profiler, "ext tables");

        prof_start!(maybe_profiler, "quotient-domain codewords");
        let base_quotient_domain_codewords = master_base_table.quotient_domain_table();
        let ext_quotient_domain_codewords = master_ext_table.quotient_domain_table();
        prof_stop!(maybe_profiler, "quotient-domain codewords");

        prof_start!(maybe_profiler, "quotient codewords");
        let master_quotient_table = all_quotients(
            base_quotient_domain_codewords,
            ext_quotient_domain_codewords,
            master_base_table.trace_domain(),
            quotient_domain,
            &challenges,
            maybe_profiler,
        );
        prof_stop!(maybe_profiler, "quotient codewords");

        #[cfg(debug_assertions)]
        {
            prof_start!(maybe_profiler, "debug degree check", "debug");
            println!(" -- checking degree of base columns --");
            Self::debug_check_degree(base_quotient_domain_codewords, quotient_domain, max_degree);
            println!(" -- checking degree of extension columns --");
            Self::debug_check_degree(ext_quotient_domain_codewords, quotient_domain, max_degree);
            println!(" -- checking degree of quotient columns --");
            Self::debug_check_degree(master_quotient_table.view(), quotient_domain, max_degree);
            prof_stop!(maybe_profiler, "debug degree check");
        }

        prof_start!(maybe_profiler, "linearly combine quotient codewords", "CC");
        // Create quotient codeword. This is a part of the combination codeword. To reduce the
        // amount of hashing necessary, the quotient codeword is linearly summed instead of
        // hashed prior to committing to it.
        let quotient_combination_weights = proof_stream.sample_scalars(num_quotients());
        let quotient_combination_weights = Array1::from(quotient_combination_weights);
        assert_eq!(
            quotient_combination_weights.len(),
            master_quotient_table.ncols()
        );

        let quotient_codeword =
            Self::random_linear_sum(master_quotient_table.view(), quotient_combination_weights);
        assert_eq!(quotient_domain.length, quotient_codeword.len());
        prof_stop!(maybe_profiler, "linearly combine quotient codewords");

        prof_start!(maybe_profiler, "commit to quotient codeword segments");
        prof_start!(maybe_profiler, "LDE", "LDE");
        let quotient_segment_polynomials =
            Self::interpolate_quotient_segments(quotient_codeword, quotient_domain);
        let fri_domain_quotient_segment_codewords =
            Self::fri_domain_segment_polynomials(quotient_segment_polynomials.view(), fri.domain);
        prof_stop!(maybe_profiler, "LDE");
        prof_start!(maybe_profiler, "hash rows of quotient segments", "hash");
        let interpret_xfe_as_bfes = |xfe: &XFieldElement| xfe.coefficients.to_vec();
        let hash_row = |row: ArrayView1<_>| {
            let row_as_bfes = row.iter().map(interpret_xfe_as_bfes).concat();
            StarkHasher::hash_varlen(&row_as_bfes)
        };
        let quotient_segments_rows = fri_domain_quotient_segment_codewords
            .axis_iter(Axis(0))
            .into_par_iter();
        let fri_domain_quotient_segment_codewords_digests =
            quotient_segments_rows.map(hash_row).collect::<Vec<_>>();
        prof_stop!(maybe_profiler, "hash rows of quotient segments");
        prof_start!(maybe_profiler, "Merkle tree", "hash");
        let quot_merkle_tree: MerkleTree<StarkHasher> =
            MTMaker::from_digests(&fri_domain_quotient_segment_codewords_digests);
        let quot_merkle_tree_root = quot_merkle_tree.get_root();
        proof_stream.enqueue(ProofItem::MerkleRoot(quot_merkle_tree_root));
        prof_stop!(maybe_profiler, "Merkle tree");
        prof_stop!(maybe_profiler, "commit to quotient codeword segments");
        debug_assert_eq!(fri.domain.length, quot_merkle_tree.get_leaf_count());

        prof_start!(maybe_profiler, "out-of-domain rows");
        let trace_domain_generator = master_base_table.trace_domain().generator;
        let out_of_domain_point_curr_row = proof_stream.sample_scalars(1)[0];
        let out_of_domain_point_next_row = trace_domain_generator * out_of_domain_point_curr_row;

        proof_stream.enqueue(ProofItem::OutOfDomainBaseRow(
            master_base_table.row(out_of_domain_point_curr_row).to_vec(),
        ));
        proof_stream.enqueue(ProofItem::OutOfDomainExtRow(
            master_ext_table.row(out_of_domain_point_curr_row).to_vec(),
        ));
        proof_stream.enqueue(ProofItem::OutOfDomainBaseRow(
            master_base_table.row(out_of_domain_point_next_row).to_vec(),
        ));
        proof_stream.enqueue(ProofItem::OutOfDomainExtRow(
            master_ext_table.row(out_of_domain_point_next_row).to_vec(),
        ));

        let out_of_domain_point_curr_row_pow_num_segments =
            out_of_domain_point_curr_row.mod_pow_u32(NUM_QUOTIENT_SEGMENTS as u32);
        let out_of_domain_curr_row_quot_segments = quotient_segment_polynomials
            .map(|poly| poly.evaluate(&out_of_domain_point_curr_row_pow_num_segments))
            .to_vec()
            .try_into()
            .unwrap();
        proof_stream.enqueue(ProofItem::OutOfDomainQuotientSegments(
            out_of_domain_curr_row_quot_segments,
        ));
        prof_stop!(maybe_profiler, "out-of-domain rows");

        // Get weights for remainder of the combination codeword.
        prof_start!(maybe_profiler, "Fiat-Shamir", "hash");
        let (base_weights, ext_weights, quotient_segment_weights) =
            Self::sample_linear_combination_weights(&mut proof_stream);
        assert_eq!(NUM_BASE_COLUMNS, base_weights.len());
        assert_eq!(NUM_EXT_COLUMNS, ext_weights.len());
        assert_eq!(NUM_QUOTIENT_SEGMENTS, quotient_segment_weights.len());
        prof_stop!(maybe_profiler, "Fiat-Shamir");

        let fri_domain_is_short_domain = fri.domain.length <= quotient_domain.length;
        let short_domain = match fri_domain_is_short_domain {
            true => fri.domain,
            false => quotient_domain,
        };
        let short_domain_base_codewords = match fri_domain_is_short_domain {
            true => master_base_table.fri_domain_table(),
            false => master_base_table.quotient_domain_table(),
        };
        let short_domain_ext_codewords = match fri_domain_is_short_domain {
            true => master_ext_table.fri_domain_table(),
            false => master_ext_table.quotient_domain_table(),
        };
        let short_domain_ext_codewords =
            short_domain_ext_codewords.slice(s![.., ..NUM_EXT_COLUMNS]);

        let fri_to_quotient_domain_unit_distance = match fri_domain_is_short_domain {
            true => 1,
            false => fri.domain.length / quotient_domain.length,
        };
        let short_domain_quot_segment_codewords = fri_domain_quotient_segment_codewords
            .slice(s![..; fri_to_quotient_domain_unit_distance, ..]);

        prof_start!(maybe_profiler, "linear combination");
        prof_start!(maybe_profiler, "base", "CC");
        let base_codeword =
            Self::random_linear_sum_base_field(short_domain_base_codewords, base_weights);
        prof_stop!(maybe_profiler, "base");
        prof_start!(maybe_profiler, "ext", "CC");
        let ext_codeword = Self::random_linear_sum(short_domain_ext_codewords, ext_weights);
        prof_stop!(maybe_profiler, "ext");
        let base_and_ext_codeword = base_codeword + ext_codeword;

        prof_start!(maybe_profiler, "quotient", "CC");
        let quotient_segments_codeword = Self::random_linear_sum(
            short_domain_quot_segment_codewords.view(),
            quotient_segment_weights,
        );
        prof_stop!(maybe_profiler, "quotient");

        assert_eq!(short_domain.length, base_and_ext_codeword.len());
        assert_eq!(short_domain.length, quotient_segments_codeword.len());
        prof_stop!(maybe_profiler, "linear combination");

        prof_start!(maybe_profiler, "DEEP");
        prof_start!(maybe_profiler, "interpolate");
        let base_and_ext_interpolation_poly =
            short_domain.interpolate(&base_and_ext_codeword.to_vec());
        let quotient_segments_interpolation_poly =
            short_domain.interpolate(&quotient_segments_codeword.to_vec());
        prof_stop!(maybe_profiler, "interpolate");
        prof_start!(maybe_profiler, "base&ext curr row");
        let out_of_domain_curr_row_base_and_ext_value =
            base_and_ext_interpolation_poly.evaluate(&out_of_domain_point_curr_row);
        let base_and_ext_curr_row_deep_codeword = Self::deep_codeword(
            &base_and_ext_codeword.to_vec(),
            short_domain,
            out_of_domain_point_curr_row,
            out_of_domain_curr_row_base_and_ext_value,
        );
        prof_stop!(maybe_profiler, "base&ext curr row");

        prof_start!(maybe_profiler, "base&ext next row");
        let out_of_domain_next_row_base_and_ext_value =
            base_and_ext_interpolation_poly.evaluate(&out_of_domain_point_next_row);
        let base_and_ext_next_row_deep_codeword = Self::deep_codeword(
            &base_and_ext_codeword.to_vec(),
            short_domain,
            out_of_domain_point_next_row,
            out_of_domain_next_row_base_and_ext_value,
        );
        prof_stop!(maybe_profiler, "base&ext next row");

        prof_start!(maybe_profiler, "segmented quotient");
        let out_of_domain_curr_row_quot_segments_value = quotient_segments_interpolation_poly
            .evaluate(&out_of_domain_point_curr_row_pow_num_segments);
        let quotient_segments_curr_row_deep_codeword = Self::deep_codeword(
            &quotient_segments_codeword.to_vec(),
            short_domain,
            out_of_domain_point_curr_row_pow_num_segments,
            out_of_domain_curr_row_quot_segments_value,
        );
        prof_stop!(maybe_profiler, "segmented quotient");
        prof_stop!(maybe_profiler, "DEEP");

        prof_start!(maybe_profiler, "combined DEEP polynomial");
        prof_start!(maybe_profiler, "Fiat-Shamir", "hash");
        let deep_codeword_weights =
            Array1::from(proof_stream.sample_scalars(NUM_DEEP_CODEWORD_COMPONENTS));
        prof_stop!(maybe_profiler, "Fiat-Shamir");
        prof_start!(maybe_profiler, "sum", "CC");
        let deep_codeword_components = [
            base_and_ext_curr_row_deep_codeword,
            base_and_ext_next_row_deep_codeword,
            quotient_segments_curr_row_deep_codeword,
        ];
        let deep_codeword_components = Array2::from_shape_vec(
            [short_domain.length, NUM_DEEP_CODEWORD_COMPONENTS].f(),
            deep_codeword_components.concat(),
        )
        .unwrap();
        let weighted_deep_codeword_components = &deep_codeword_components * &deep_codeword_weights;
        let deep_codeword = weighted_deep_codeword_components.sum_axis(Axis(1));
        prof_stop!(maybe_profiler, "sum");
        let fri_deep_codeword = match fri_domain_is_short_domain {
            true => deep_codeword,
            false => {
                prof_start!(maybe_profiler, "LDE", "LDE");
                let deep_codeword =
                    quotient_domain.low_degree_extension(&deep_codeword.to_vec(), fri.domain);
                prof_stop!(maybe_profiler, "LDE");
                Array1::from(deep_codeword)
            }
        };
        assert_eq!(fri.domain.length, fri_deep_codeword.len());
        prof_start!(maybe_profiler, "add randomizer codeword", "CC");
        let fri_combination_codeword = master_ext_table
            .fri_domain_randomizer_polynomials()
            .into_iter()
            .fold(fri_deep_codeword, ArrayBase::add)
            .to_vec();
        prof_stop!(maybe_profiler, "add randomizer codeword");
        assert_eq!(fri.domain.length, fri_combination_codeword.len());
        prof_stop!(maybe_profiler, "combined DEEP polynomial");

        prof_start!(maybe_profiler, "FRI");
        let revealed_current_row_indices = fri.prove(&fri_combination_codeword, &mut proof_stream);
        assert_eq!(
            parameters.num_combination_codeword_checks,
            revealed_current_row_indices.len()
        );
        prof_stop!(maybe_profiler, "FRI");

        prof_start!(maybe_profiler, "open trace leafs");
        // Open leafs of zipped codewords at indicated positions
        let revealed_base_elems = Self::get_revealed_elements(
            master_base_table.fri_domain_table(),
            &revealed_current_row_indices,
        );
        let base_authentication_structure =
            base_merkle_tree.get_authentication_structure(&revealed_current_row_indices);
        proof_stream.enqueue(ProofItem::MasterBaseTableRows(revealed_base_elems));
        proof_stream.enqueue(ProofItem::AuthenticationStructure(
            base_authentication_structure,
        ));

        let revealed_ext_elems = Self::get_revealed_elements(
            master_ext_table.fri_domain_table(),
            &revealed_current_row_indices,
        );
        let ext_authentication_structure =
            ext_merkle_tree.get_authentication_structure(&revealed_current_row_indices);
        proof_stream.enqueue(ProofItem::MasterExtTableRows(revealed_ext_elems));
        proof_stream.enqueue(ProofItem::AuthenticationStructure(
            ext_authentication_structure,
        ));

        // Open quotient & combination codewords at the same positions as base & ext codewords.
        let into_fixed_width_row =
            |row: ArrayView1<_>| -> [_; NUM_QUOTIENT_SEGMENTS] { row.to_vec().try_into().unwrap() };
        let revealed_quotient_segments_rows = revealed_current_row_indices
            .iter()
            .map(|&i| fri_domain_quotient_segment_codewords.row(i))
            .map(into_fixed_width_row)
            .collect_vec();
        let revealed_quotient_authentication_structure =
            quot_merkle_tree.get_authentication_structure(&revealed_current_row_indices);
        proof_stream.enqueue(ProofItem::QuotientSegmentsElements(
            revealed_quotient_segments_rows,
        ));
        proof_stream.enqueue(ProofItem::AuthenticationStructure(
            revealed_quotient_authentication_structure,
        ));
        prof_stop!(maybe_profiler, "open trace leafs");

        #[cfg(debug_assertions)]
        Self::debug_print_proof_size(&proof_stream);

        proof_stream.into()
    }

    fn random_linear_sum_base_field(
        codewords: ArrayView2<BFieldElement>,
        weights: Array1<XFieldElement>,
    ) -> Array1<XFieldElement> {
        assert_eq!(codewords.ncols(), weights.len());
        let weight_coefficients_0: Array1<_> = weights.iter().map(|&w| w.coefficients[0]).collect();
        let weight_coefficients_1: Array1<_> = weights.iter().map(|&w| w.coefficients[1]).collect();
        let weight_coefficients_2: Array1<_> = weights.iter().map(|&w| w.coefficients[2]).collect();

        let mut random_linear_sum = Array1::zeros(codewords.nrows());
        Zip::from(codewords.axis_iter(Axis(0)))
            .and(random_linear_sum.axis_iter_mut(Axis(0)))
            .par_for_each(|codeword, target_element| {
                let random_linear_element_coefficients = [
                    codeword.dot(&weight_coefficients_0),
                    codeword.dot(&weight_coefficients_1),
                    codeword.dot(&weight_coefficients_2),
                ];
                let random_linear_element = XFieldElement::new(random_linear_element_coefficients);
                Array0::from_elem((), random_linear_element).move_into(target_element);
            });
        random_linear_sum
    }

    fn random_linear_sum(
        codewords: ArrayView2<XFieldElement>,
        weights: Array1<XFieldElement>,
    ) -> Array1<XFieldElement> {
        assert_eq!(codewords.ncols(), weights.len());
        let mut random_linear_sum = Array1::zeros(codewords.nrows());
        Zip::from(codewords.axis_iter(Axis(0)))
            .and(random_linear_sum.axis_iter_mut(Axis(0)))
            .par_for_each(|codeword, target_element| {
                let random_linear_element = codeword.dot(&weights);
                Array0::from_elem((), random_linear_element).move_into(target_element);
            });
        random_linear_sum
    }

    fn sample_linear_combination_weights(
        proof_stream: &mut ProofStream<StarkHasher>,
    ) -> (
        Array1<XFieldElement>,
        Array1<XFieldElement>,
        Array1<XFieldElement>,
    ) {
        let num_weights = NUM_BASE_COLUMNS + NUM_EXT_COLUMNS + NUM_QUOTIENT_SEGMENTS;
        let all_weights = proof_stream.sample_scalars(num_weights);

        let base_weights = all_weights[..NUM_BASE_COLUMNS].iter().copied().collect();
        let ext_weights = all_weights[NUM_BASE_COLUMNS..NUM_BASE_COLUMNS + NUM_EXT_COLUMNS]
            .iter()
            .copied()
            .collect();
        let quotient_segment_weights = all_weights[NUM_BASE_COLUMNS + NUM_EXT_COLUMNS..]
            .iter()
            .copied()
            .collect();

        (base_weights, ext_weights, quotient_segment_weights)
    }

    fn fri_domain_segment_polynomials(
        quotient_segment_polynomials: ArrayView1<Polynomial<XFieldElement>>,
        fri_domain: ArithmeticDomain,
    ) -> Array2<XFieldElement> {
        let fri_domain_codewords = quotient_segment_polynomials
            .iter()
            .map(|segment| fri_domain.evaluate(segment));
        Array2::from_shape_vec(
            [fri_domain.length, NUM_QUOTIENT_SEGMENTS].f(),
            fri_domain_codewords.concat(),
        )
        .unwrap()
    }

    fn interpolate_quotient_segments(
        quotient_codeword: Array1<XFieldElement>,
        quotient_domain: ArithmeticDomain,
    ) -> Array1<Polynomial<XFieldElement>> {
        let quotient_interpolation_poly = quotient_domain.interpolate(&quotient_codeword.to_vec());
        let quotient_segments: [_; NUM_QUOTIENT_SEGMENTS] =
            Self::split_polynomial_into_segments(&quotient_interpolation_poly);
        Array1::from(quotient_segments.to_vec())
    }

    /// An [`ArithmeticDomain`] _just_ large enough to perform all the necessary computations on
    /// polynomials. Concretely, the maximal degree of a polynomial over the quotient domain is at
    /// most only slightly larger than the maximal degree allowed in the STARK proof, and could be
    /// equal. This makes computation for the prover much faster.
    ///
    /// When debugging, it is useful to check the degree of some intermediate polynomials.
    /// However, the quotient domain's minimal length can make it impossible to check if some
    /// operation (e.g., dividing out the zerofier) has (erroneously) increased the polynomial's
    /// degree beyond the allowed maximum. Hence, a larger quotient domain is chosen when debugging
    /// and testing.
    pub(crate) fn quotient_domain(
        fri_domain: ArithmeticDomain,
        max_degree: Degree,
    ) -> ArithmeticDomain {
        let maybe_blowup_factor = match cfg!(debug_assertions) {
            true => 2,
            false => 1,
        };
        let domain_length = (max_degree as u64).next_power_of_two() as usize;
        let domain_length = maybe_blowup_factor * domain_length;
        ArithmeticDomain::of_length(domain_length).with_offset(fri_domain.offset)
    }

    /// Compute the upper bound to use for the maximum degree the quotients given the length of the
    /// trace and the number of trace randomizers.
    /// The degree of the quotients depends on the constraints, _i.e._, the AIR.
    pub fn derive_max_degree(padded_height: usize, num_trace_randomizers: usize) -> Degree {
        let interpolant_degree = interpolant_degree(padded_height, num_trace_randomizers);
        let max_constraint_degree_with_origin =
            max_degree_with_origin(interpolant_degree, padded_height);
        let max_constraint_degree = max_constraint_degree_with_origin.degree as u64;
        let min_arithmetic_domain_length_supporting_max_constraint_degree =
            max_constraint_degree.next_power_of_two();
        let max_degree_supported_by_that_smallest_arithmetic_domain =
            min_arithmetic_domain_length_supporting_max_constraint_degree - 1;

        max_degree_supported_by_that_smallest_arithmetic_domain as Degree
    }

    /// Compute the parameters for FRI. The length of the FRI domain, _i.e._, the number of
    /// elements in the FRI domain, has a major influence on proving time. It is influenced by the
    /// length of the execution trace and the FRI expansion factor, a security parameter.
    ///
    /// In principle, the FRI domain is also influenced by the AIR's degree
    /// (see [`AIR_TARGET_DEGREE`]). However, by segmenting the quotient polynomial into
    /// [`AIR_TARGET_DEGREE`]-many parts, that influence is mitigated.
    pub fn derive_fri(parameters: StarkParameters, padded_height: usize) -> Fri<StarkHasher> {
        let interpolant_degree =
            interpolant_degree(padded_height, parameters.num_trace_randomizers);
        let interpolant_codeword_length = interpolant_degree as usize + 1;
        let fri_domain_length = parameters.fri_expansion_factor * interpolant_codeword_length;
        let coset_offset = BFieldElement::generator();
        let domain = ArithmeticDomain::of_length(fri_domain_length).with_offset(coset_offset);
        Fri::new(
            domain,
            parameters.fri_expansion_factor,
            parameters.num_colinearity_checks,
        )
    }

    fn get_revealed_elements<FF: FiniteField>(
        fri_domain_table: ArrayView2<FF>,
        revealed_indices: &[usize],
    ) -> Vec<Vec<FF>> {
        revealed_indices
            .iter()
            .map(|&idx| fri_domain_table.row(idx).to_vec())
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

    /// Losslessly split the given polynomial `f` into `N` segments of (roughly) equal degree.
    /// The degree of each segment is at most `f.degree() / N`.
    /// It holds that `f(x) = Σ_{i=0}^{N-1} x^i·f_i(x^N)`, where the `f_i` are the segments.
    ///
    /// For example, let
    /// - `N = 3`, and
    /// - `f(x) = 7·x^7 + 6·x^6 + 5·x^5 + 4·x^4 + 3·x^3 + 2·x^2 + 1·x + 0`.
    ///
    /// Then, the function returns the array:
    ///
    /// ```text
    /// [f_0(x) = 6·x^2 + 3·x + 0,
    ///  f_1(x) = 7·x^2 + 4·x + 1,
    ///  f_2(x) =         5·x + 2]
    /// ```
    ///
    /// The following equality holds: `f(x) == f_0(x^3) + x·f_1(x^3) + x^2·f_2(x^3)`.
    fn split_polynomial_into_segments<const N: usize, FF: FiniteField>(
        polynomial: &Polynomial<FF>,
    ) -> [Polynomial<FF>; N] {
        let mut segments = vec![];
        for segment_index in 0..N {
            let coefficient_iter_at_start = polynomial.coefficients.iter().skip(segment_index);
            let segment_coefficients = coefficient_iter_at_start.step_by(N).copied().collect();
            let segment = Polynomial::new(segment_coefficients);
            segments.push(segment);
        }
        segments.try_into().unwrap()
    }

    #[cfg(debug_assertions)]
    #[allow(clippy::absolute_paths)]
    fn debug_check_degree<FF>(
        table: ArrayView2<FF>,
        quotient_domain: ArithmeticDomain,
        max_degree: Degree,
    ) where
        FF: FiniteField + std::ops::MulAssign<BFieldElement>,
    {
        let max_degree = max_degree as isize;
        for (col_idx, codeword) in table.columns().into_iter().enumerate() {
            let degree = quotient_domain.interpolate(&codeword.to_vec()).degree();
            let maybe_excl_mark = match degree > max_degree {
                true => "!",
                false => " ",
            };
            println!(
                "{maybe_excl_mark} Codeword {col_idx:>3} has degree {degree:>5}. \
                Must be of maximal degree {max_degree:>5}."
            );
        }
    }

    #[cfg(debug_assertions)]
    fn debug_print_proof_size(proof_stream: &ProofStream<StarkHasher>) {
        let transcript_length = proof_stream.transcript_length();
        let kib = (transcript_length * 8 / 1024) + 1;
        println!("Created proof containing {transcript_length} B-field elements ({kib} kiB).");
    }

    pub fn verify(
        parameters: StarkParameters,
        claim: &Claim,
        proof: &Proof,
        maybe_profiler: &mut Option<TritonProfiler>,
    ) -> Result<bool> {
        prof_start!(maybe_profiler, "deserialize");
        let mut proof_stream = StarkProofStream::try_from(proof)?;
        prof_stop!(maybe_profiler, "deserialize");

        prof_start!(maybe_profiler, "Fiat-Shamir: Claim", "hash");
        proof_stream.alter_fiat_shamir_state_with(claim);
        prof_stop!(maybe_profiler, "Fiat-Shamir: Claim");

        prof_start!(maybe_profiler, "derive additional parameters");
        let log_2_padded_height = proof_stream.dequeue()?.as_log2_padded_height()?;
        let padded_height = 1 << log_2_padded_height;
        let fri = Self::derive_fri(parameters, padded_height);
        let merkle_tree_height = fri.domain.length.ilog2() as usize;
        prof_stop!(maybe_profiler, "derive additional parameters");

        prof_start!(maybe_profiler, "Fiat-Shamir 1", "hash");
        let base_merkle_tree_root = proof_stream.dequeue()?.as_merkle_root()?;
        let extension_challenge_weights =
            proof_stream.sample_scalars(Challenges::num_challenges_to_sample());
        let challenges = Challenges::new(extension_challenge_weights, claim);
        let extension_tree_merkle_root = proof_stream.dequeue()?.as_merkle_root()?;
        // Sample weights for quotient codeword, which is a part of the combination codeword.
        // See corresponding part in the prover for a more detailed explanation.
        let quot_codeword_weights = proof_stream.sample_scalars(num_quotients());
        let quot_codeword_weights = Array1::from(quot_codeword_weights);
        let quotient_codeword_merkle_root = proof_stream.dequeue()?.as_merkle_root()?;
        prof_stop!(maybe_profiler, "Fiat-Shamir 1");

        prof_start!(maybe_profiler, "dequeue ood point and rows", "hash");
        let trace_domain_generator = ArithmeticDomain::generator_for_length(padded_height as u64);
        let out_of_domain_point_curr_row = proof_stream.sample_scalars(1)[0];
        let out_of_domain_point_next_row = trace_domain_generator * out_of_domain_point_curr_row;
        let out_of_domain_point_curr_row_pow_num_segments =
            out_of_domain_point_curr_row.mod_pow_u32(NUM_QUOTIENT_SEGMENTS as u32);

        let out_of_domain_curr_base_row = proof_stream.dequeue()?.as_out_of_domain_base_row()?;
        let out_of_domain_curr_ext_row = proof_stream.dequeue()?.as_out_of_domain_ext_row()?;
        let out_of_domain_next_base_row = proof_stream.dequeue()?.as_out_of_domain_base_row()?;
        let out_of_domain_next_ext_row = proof_stream.dequeue()?.as_out_of_domain_ext_row()?;
        let out_of_domain_curr_row_quot_segments = proof_stream
            .dequeue()?
            .as_out_of_domain_quotient_segments()?;

        let out_of_domain_curr_base_row = Array1::from(out_of_domain_curr_base_row);
        let out_of_domain_curr_ext_row = Array1::from(out_of_domain_curr_ext_row);
        let out_of_domain_next_base_row = Array1::from(out_of_domain_next_base_row);
        let out_of_domain_next_ext_row = Array1::from(out_of_domain_next_ext_row);
        let out_of_domain_curr_row_quot_segments =
            Array1::from(out_of_domain_curr_row_quot_segments.to_vec());
        prof_stop!(maybe_profiler, "dequeue ood point and rows");

        prof_start!(maybe_profiler, "out-of-domain quotient element");
        prof_start!(maybe_profiler, "zerofiers");
        let one = BFieldElement::one();
        let initial_zerofier_inv = (out_of_domain_point_curr_row - one).inverse();
        let consistency_zerofier_inv =
            (out_of_domain_point_curr_row.mod_pow_u32(padded_height as u32) - one).inverse();
        let except_last_row = out_of_domain_point_curr_row - trace_domain_generator.inverse();
        let transition_zerofier_inv = except_last_row * consistency_zerofier_inv;
        let terminal_zerofier_inv = except_last_row.inverse(); // i.e., only last row
        prof_stop!(maybe_profiler, "zerofiers");

        prof_start!(maybe_profiler, "evaluate AIR", "AIR");
        let evaluated_initial_constraints = MasterExtTable::evaluate_initial_constraints(
            out_of_domain_curr_base_row.view(),
            out_of_domain_curr_ext_row.view(),
            &challenges,
        );
        let evaluated_consistency_constraints = MasterExtTable::evaluate_consistency_constraints(
            out_of_domain_curr_base_row.view(),
            out_of_domain_curr_ext_row.view(),
            &challenges,
        );
        let evaluated_transition_constraints = MasterExtTable::evaluate_transition_constraints(
            out_of_domain_curr_base_row.view(),
            out_of_domain_curr_ext_row.view(),
            out_of_domain_next_base_row.view(),
            out_of_domain_next_ext_row.view(),
            &challenges,
        );
        let evaluated_terminal_constraints = MasterExtTable::evaluate_terminal_constraints(
            out_of_domain_curr_base_row.view(),
            out_of_domain_curr_ext_row.view(),
            &challenges,
        );
        prof_stop!(maybe_profiler, "evaluate AIR");

        prof_start!(maybe_profiler, "divide");
        let mut quotient_summands = Vec::with_capacity(num_quotients());
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
            quot_codeword_weights.dot(&Array1::from(quotient_summands));
        prof_stop!(maybe_profiler, "inner product");
        prof_stop!(maybe_profiler, "out-of-domain quotient element");

        prof_start!(maybe_profiler, "verify quotient's segments");
        let powers_of_out_of_domain_point_curr_row = (0..NUM_QUOTIENT_SEGMENTS as u32)
            .map(|exponent| out_of_domain_point_curr_row.mod_pow_u32(exponent))
            .collect::<Array1<_>>();
        let sum_of_evaluated_out_of_domain_quotient_segments =
            powers_of_out_of_domain_point_curr_row.dot(&out_of_domain_curr_row_quot_segments);
        ensure_eq!(
            out_of_domain_quotient_value,
            sum_of_evaluated_out_of_domain_quotient_segments
        );
        prof_stop!(maybe_profiler, "verify quotient's segments");

        prof_start!(maybe_profiler, "Fiat-Shamir 2", "hash");
        let num_base_and_ext_and_quotient_segment_codeword_weights =
            NUM_BASE_COLUMNS + NUM_EXT_COLUMNS + NUM_QUOTIENT_SEGMENTS;
        let base_and_ext_and_quotient_segment_codeword_weights =
            proof_stream.sample_scalars(num_base_and_ext_and_quotient_segment_codeword_weights);
        let (base_and_ext_codeword_weights, quotient_segment_codeword_weights) =
            base_and_ext_and_quotient_segment_codeword_weights
                .split_at(NUM_BASE_COLUMNS + NUM_EXT_COLUMNS);
        let base_and_ext_codeword_weights = Array1::from(base_and_ext_codeword_weights.to_vec());
        let quotient_segment_codeword_weights =
            Array1::from(quotient_segment_codeword_weights.to_vec());
        prof_stop!(maybe_profiler, "Fiat-Shamir 2");

        prof_start!(maybe_profiler, "sum out-of-domain values", "CC");
        let out_of_domain_curr_row_base_and_ext_value = Self::linearly_sum_base_and_ext_row(
            out_of_domain_curr_base_row.view(),
            out_of_domain_curr_ext_row.view(),
            base_and_ext_codeword_weights.view(),
            maybe_profiler,
        );
        let out_of_domain_next_row_base_and_ext_value = Self::linearly_sum_base_and_ext_row(
            out_of_domain_next_base_row.view(),
            out_of_domain_next_ext_row.view(),
            base_and_ext_codeword_weights.view(),
            maybe_profiler,
        );
        let out_of_domain_curr_row_quotient_segment_value =
            quotient_segment_codeword_weights.dot(&out_of_domain_curr_row_quot_segments);
        prof_stop!(maybe_profiler, "sum out-of-domain values");

        prof_start!(maybe_profiler, "Fiat-Shamir", "hash");
        let deep_codeword_weights =
            Array1::from(proof_stream.sample_scalars(NUM_DEEP_CODEWORD_COMPONENTS));
        prof_stop!(maybe_profiler, "Fiat-Shamir");

        // verify low degree of combination polynomial with FRI
        prof_start!(maybe_profiler, "FRI");
        let revealed_fri_indices_and_elements = fri.verify(&mut proof_stream, maybe_profiler)?;
        let (revealed_current_row_indices, revealed_fri_values): (Vec<_>, Vec<_>) =
            revealed_fri_indices_and_elements.into_iter().unzip();
        prof_stop!(maybe_profiler, "FRI");

        prof_start!(maybe_profiler, "check leafs");
        prof_start!(maybe_profiler, "dequeue base elements");
        let base_table_rows = proof_stream.dequeue()?.as_master_base_table_rows()?;
        let base_authentication_structure =
            proof_stream.dequeue()?.as_authentication_structure()?;
        let leaf_digests_base: Vec<_> = base_table_rows
            .par_iter()
            .map(|revealed_base_elem| StarkHasher::hash_varlen(revealed_base_elem))
            .collect();
        prof_stop!(maybe_profiler, "dequeue base elements");

        prof_start!(maybe_profiler, "Merkle verify (base tree)", "hash");
        if !MerkleTree::<StarkHasher>::verify_authentication_structure(
            base_merkle_tree_root,
            merkle_tree_height,
            &revealed_current_row_indices,
            &leaf_digests_base,
            &base_authentication_structure,
        ) {
            bail!("Failed to verify authentication path for base codeword");
        }
        prof_stop!(maybe_profiler, "Merkle verify (base tree)");

        prof_start!(maybe_profiler, "dequeue extension elements");
        let ext_table_rows = proof_stream.dequeue()?.as_master_ext_table_rows()?;
        let ext_authentication_structure = proof_stream.dequeue()?.as_authentication_structure()?;
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
        if !MerkleTree::<StarkHasher>::verify_authentication_structure(
            extension_tree_merkle_root,
            merkle_tree_height,
            &revealed_current_row_indices,
            &leaf_digests_ext,
            &ext_authentication_structure,
        ) {
            bail!("Failed to verify authentication path for extension codeword");
        }
        prof_stop!(maybe_profiler, "Merkle verify (extension tree)");

        prof_start!(maybe_profiler, "dequeue quotient segments' elements");
        let revealed_quotient_segments_elements =
            proof_stream.dequeue()?.as_quotient_segments_elements()?;
        let revealed_quotient_segments_digests =
            Self::hash_quotient_segment_elements(&revealed_quotient_segments_elements);
        let revealed_quotient_authentication_structure =
            proof_stream.dequeue()?.as_authentication_structure()?;
        prof_stop!(maybe_profiler, "dequeue quotient segments' elements");

        prof_start!(maybe_profiler, "Merkle verify (combined quotient)", "hash");
        if !MerkleTree::<StarkHasher>::verify_authentication_structure(
            quotient_codeword_merkle_root,
            merkle_tree_height,
            &revealed_current_row_indices,
            &revealed_quotient_segments_digests,
            &revealed_quotient_authentication_structure,
        ) {
            bail!("Failed to verify authentication path for combined quotient codeword");
        }
        prof_stop!(maybe_profiler, "Merkle verify (combined quotient)");
        prof_stop!(maybe_profiler, "check leafs");

        prof_start!(maybe_profiler, "linear combination");
        let num_checks = parameters.num_combination_codeword_checks;
        ensure_eq!(num_checks, revealed_current_row_indices.len());
        ensure_eq!(num_checks, revealed_fri_values.len());
        ensure_eq!(num_checks, revealed_quotient_segments_elements.len());
        ensure_eq!(num_checks, base_table_rows.len());
        ensure_eq!(num_checks, ext_table_rows.len());
        prof_start!(maybe_profiler, "main loop");
        for (row_idx, base_row, ext_row, quotient_segments_elements, fri_value) in izip!(
            revealed_current_row_indices,
            base_table_rows,
            ext_table_rows,
            revealed_quotient_segments_elements,
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
            let quotient_segments_curr_row_element = quotient_segment_codeword_weights
                .dot(&Array1::from(quotient_segments_elements.to_vec()));
            prof_stop!(maybe_profiler, "base & ext elements");

            prof_start!(maybe_profiler, "DEEP update");
            let base_and_ext_curr_row_deep_value = Self::deep_update(
                current_fri_domain_value,
                base_and_ext_curr_row_element,
                out_of_domain_point_curr_row,
                out_of_domain_curr_row_base_and_ext_value,
            );
            let base_and_ext_next_row_deep_value = Self::deep_update(
                current_fri_domain_value,
                base_and_ext_curr_row_element,
                out_of_domain_point_next_row,
                out_of_domain_next_row_base_and_ext_value,
            );
            let quot_curr_row_deep_value = Self::deep_update(
                current_fri_domain_value,
                quotient_segments_curr_row_element,
                out_of_domain_point_curr_row_pow_num_segments,
                out_of_domain_curr_row_quotient_segment_value,
            );
            prof_stop!(maybe_profiler, "DEEP update");

            prof_start!(maybe_profiler, "combination codeword equality");
            let deep_value_components = Array1::from(vec![
                base_and_ext_curr_row_deep_value,
                base_and_ext_next_row_deep_value,
                quot_curr_row_deep_value,
            ]);
            let deep_value = deep_codeword_weights.dot(&deep_value_components);
            let randomizer_codewords_contribution = randomizer_row.sum();
            ensure_eq!(fri_value, deep_value + randomizer_codewords_contribution);
            prof_stop!(maybe_profiler, "combination codeword equality");
        }
        prof_stop!(maybe_profiler, "main loop");
        prof_stop!(maybe_profiler, "linear combination");
        Ok(true)
    }

    fn hash_quotient_segment_elements(
        quotient_segment_rows: &[[XFieldElement; NUM_QUOTIENT_SEGMENTS]],
    ) -> Vec<Digest> {
        let interpret_xfe_as_bfes = |xfe: XFieldElement| xfe.coefficients.to_vec();
        let collect_row_as_bfes =
            |row: &[_; NUM_QUOTIENT_SEGMENTS]| row.map(interpret_xfe_as_bfes).concat();
        quotient_segment_rows
            .par_iter()
            .map(collect_row_as_bfes)
            .map(|row| StarkHasher::hash_varlen(&row))
            .collect()
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
pub(crate) mod tests {
    use itertools::izip;
    use num_traits::Zero;
    use proptest_arbitrary_interop::arb;
    use rand::prelude::ThreadRng;
    use rand::thread_rng;
    use rand::Rng;
    use rand_core::RngCore;
    use strum::EnumCount;
    use test_strategy::proptest;
    use twenty_first::shared_math::other::random_elements;

    use crate::example_programs::*;
    use crate::instruction::AnInstruction;
    use crate::instruction::Instruction;
    use crate::op_stack::OpStackElement;
    use crate::program::Program;
    use crate::shared_tests::*;
    use crate::table::cascade_table;
    use crate::table::cascade_table::ExtCascadeTable;
    use crate::table::challenges::ChallengeId::LookupTablePublicTerminal;
    use crate::table::challenges::ChallengeId::StandardInputIndeterminate;
    use crate::table::challenges::ChallengeId::StandardInputTerminal;
    use crate::table::challenges::ChallengeId::StandardOutputIndeterminate;
    use crate::table::challenges::ChallengeId::StandardOutputTerminal;
    use crate::table::constraint_circuit::ConstraintCircuitBuilder;
    use crate::table::cross_table_argument::CrossTableArg;
    use crate::table::cross_table_argument::EvalArg;
    use crate::table::cross_table_argument::GrandCrossTableArg;
    use crate::table::extension_table::Evaluable;
    use crate::table::extension_table::Quotientable;
    use crate::table::hash_table;
    use crate::table::hash_table::ExtHashTable;
    use crate::table::jump_stack_table;
    use crate::table::jump_stack_table::ExtJumpStackTable;
    use crate::table::lookup_table;
    use crate::table::lookup_table::ExtLookupTable;
    use crate::table::master_table::all_degrees_with_origin;
    use crate::table::master_table::MasterExtTable;
    use crate::table::master_table::TableId::LookupTable;
    use crate::table::master_table::TableId::ProcessorTable;
    use crate::table::op_stack_table;
    use crate::table::op_stack_table::ExtOpStackTable;
    use crate::table::processor_table;
    use crate::table::processor_table::ExtProcessorTable;
    use crate::table::program_table;
    use crate::table::program_table::ExtProgramTable;
    use crate::table::ram_table;
    use crate::table::ram_table::ExtRamTable;
    use crate::table::table_column::LookupExtTableColumn::PublicEvaluationArgument;
    use crate::table::table_column::MasterBaseTableColumn;
    use crate::table::table_column::MasterExtTableColumn;
    use crate::table::table_column::OpStackBaseTableColumn;
    use crate::table::table_column::ProcessorBaseTableColumn;
    use crate::table::table_column::ProcessorExtTableColumn::InputTableEvalArg;
    use crate::table::table_column::ProcessorExtTableColumn::OutputTableEvalArg;
    use crate::table::table_column::RamBaseTableColumn;
    use crate::table::u32_table;
    use crate::table::u32_table::ExtU32Table;
    use crate::triton_program;
    use crate::vm::tests::*;
    use crate::NonDeterminism;
    use crate::PublicInput;

    use super::*;

    pub(crate) fn master_base_table_for_low_security_level(
        program: &Program,
        public_input: PublicInput,
        non_determinism: NonDeterminism<BFieldElement>,
    ) -> (StarkParameters, Claim, MasterBaseTable) {
        let (aet, stdout) = program
            .trace_execution(public_input.clone(), non_determinism)
            .unwrap();
        let parameters = stark_parameters_with_low_security_level();
        let claim = construct_claim(&aet, public_input.individual_tokens, stdout);
        let master_base_table = construct_master_base_table(parameters, &aet);

        (parameters, claim, master_base_table)
    }

    pub(crate) fn master_tables_for_low_security_level(
        program: &Program,
        public_input: PublicInput,
        non_determinism: NonDeterminism<BFieldElement>,
    ) -> (
        StarkParameters,
        Claim,
        MasterBaseTable,
        MasterExtTable,
        Challenges,
    ) {
        let (parameters, claim, mut master_base_table) =
            master_base_table_for_low_security_level(program, public_input, non_determinism);

        let dummy_challenges = Challenges::placeholder(Some(&claim));
        master_base_table.pad();
        let master_ext_table =
            master_base_table.extend(&dummy_challenges, parameters.num_randomizer_polynomials);

        (
            parameters,
            claim,
            master_base_table,
            master_ext_table,
            dummy_challenges,
        )
    }

    #[test]
    fn print_ram_table_example_for_specification() {
        let program = triton_program!(
            push  5 push  6 write_mem pop 1
            push 15 push 16 write_mem pop 1
            push  5 read_mem pop 2
            push 15 read_mem pop 2
            push  5 push  7 write_mem pop 1
            push 15 read_mem
            push  5 read_mem
            halt
        );
        let (_, _, master_base_table) =
            master_base_table_for_low_security_level(&program, [].into(), [].into());

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
            let pi = match Instruction::try_from(prev_instruction) {
                Ok(AnInstruction::Halt) | Err(_) => "-".to_string(),
                Ok(instr) => instr.name().to_string(),
            };
            let (ci, nia) = ci_and_nia_from_master_table_row(row);

            let interesting_cols = [clk, pi, ci, nia, st0, st1, st2, st3, ramp, ramv];
            let interesting_cols = interesting_cols
                .iter()
                .map(|ff| format!("{:>10}", format!("{ff}")))
                .collect_vec()
                .join(" | ");
            println!("{interesting_cols}");
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
            let pi = match Instruction::try_from(prev_instruction) {
                Ok(AnInstruction::Halt) | Err(_) => "-".to_string(),
                Ok(instr) => instr.name().to_string(),
            };

            let interesting_cols = [clk, pi, ramp, ramv, iord];
            let interesting_cols = interesting_cols
                .iter()
                .map(|ff| format!("{:>10}", format!("{ff}")))
                .collect_vec()
                .join(" | ");
            println!("{interesting_cols}");
        }
    }

    #[test]
    fn print_op_stack_table_example_for_specification() {
        let num_interesting_rows = 30;
        let fake_op_stack_size = 4;

        let program = triton_program! {
            push 42 push 43 push 44 push 45 push 46 push 47 push 48
            nop pop 1 pop 2 pop 1
            push 77 swap 3 push 78 swap 3 push 79
            pop 1 pop 2 pop 3
            halt
        };
        let (_, _, master_base_table) =
            master_base_table_for_low_security_level(&program, [].into(), [].into());

        println!();
        println!("Processor Table:");
        println!(
            "| clk        | ci         | nia        | st0        | st1        \
             | st2        | st3        | underflow  | pointer    |"
        );
        println!(
            "|-----------:|:-----------|-----------:|-----------:|-----------:\
             |-----------:|-----------:|:-----------|-----------:|"
        );
        for row in master_base_table
            .table(ProcessorTable)
            .rows()
            .into_iter()
            .take(num_interesting_rows)
        {
            let clk = row[ProcessorBaseTableColumn::CLK.base_table_index()].to_string();
            let st0 = row[ProcessorBaseTableColumn::ST0.base_table_index()].to_string();
            let st1 = row[ProcessorBaseTableColumn::ST1.base_table_index()].to_string();
            let st2 = row[ProcessorBaseTableColumn::ST2.base_table_index()].to_string();
            let st3 = row[ProcessorBaseTableColumn::ST3.base_table_index()].to_string();
            let st4 = row[ProcessorBaseTableColumn::ST4.base_table_index()].to_string();
            let st5 = row[ProcessorBaseTableColumn::ST5.base_table_index()].to_string();
            let st6 = row[ProcessorBaseTableColumn::ST6.base_table_index()].to_string();
            let st7 = row[ProcessorBaseTableColumn::ST7.base_table_index()].to_string();
            let st8 = row[ProcessorBaseTableColumn::ST8.base_table_index()].to_string();
            let st9 = row[ProcessorBaseTableColumn::ST9.base_table_index()].to_string();

            let osp = row[ProcessorBaseTableColumn::OpStackPointer.base_table_index()];
            let osp =
                (osp.value() + fake_op_stack_size).saturating_sub(OpStackElement::COUNT as u64);

            let underflow_size = osp.saturating_sub(fake_op_stack_size);
            let underflow_candidates = [st4, st5, st6, st7, st8, st9];
            let underflow = underflow_candidates
                .into_iter()
                .take(underflow_size as usize);
            let underflow = underflow.map(|ff| format!("{:>2}", format!("{ff}")));
            let underflow = format!("[{}]", underflow.collect_vec().join(", "));

            let osp = osp.to_string();
            let (ci, nia) = ci_and_nia_from_master_table_row(row);

            let interesting_cols = [clk, ci, nia, st0, st1, st2, st3, underflow, osp];
            let interesting_cols = interesting_cols
                .map(|ff| format!("{:>10}", format!("{ff}")))
                .join(" | ");
            println!("{interesting_cols}");
        }

        println!();
        println!("Op Stack Table:");
        println!("|        clk |        ib1 |    pointer |      value |");
        println!("|-----------:|-----------:|-----------:|-----------:|");
        for row in master_base_table
            .table(TableId::OpStackTable)
            .rows()
            .into_iter()
            .take(num_interesting_rows)
        {
            let clk = row[OpStackBaseTableColumn::CLK.base_table_index()].to_string();
            let ib1 = row[OpStackBaseTableColumn::IB1ShrinkStack.base_table_index()].to_string();

            let osp = row[OpStackBaseTableColumn::StackPointer.base_table_index()];
            let osp =
                (osp.value() + fake_op_stack_size).saturating_sub(OpStackElement::COUNT as u64);
            let osp = osp.to_string();

            let value =
                row[OpStackBaseTableColumn::FirstUnderflowElement.base_table_index()].to_string();

            let interesting_cols = [clk, ib1, osp, value];
            let interesting_cols = interesting_cols
                .map(|ff| format!("{:>10}", format!("{ff}")))
                .join(" | ");
            println!("{interesting_cols}");
        }
    }

    fn ci_and_nia_from_master_table_row(row: ArrayView1<BFieldElement>) -> (String, String) {
        let curr_instruction = row[ProcessorBaseTableColumn::CI.base_table_index()].value();
        let next_instruction_or_arg = row[ProcessorBaseTableColumn::NIA.base_table_index()].value();

        let curr_instruction = Instruction::try_from(curr_instruction).unwrap();
        let nia = if curr_instruction.has_arg() {
            next_instruction_or_arg.to_string()
        } else {
            "".to_string()
        };
        (curr_instruction.name().to_string(), nia)
    }

    /// To be used with `-- --nocapture`. Has mainly informative purpose.
    #[test]
    fn print_all_constraint_degrees() {
        let padded_height = 2;
        let num_trace_randomizers = 2;
        let interpolant_degree = interpolant_degree(padded_height, num_trace_randomizers);
        for deg in all_degrees_with_origin(interpolant_degree, padded_height) {
            println!("{deg}");
        }
    }

    #[test]
    fn check_io_terminals() {
        let read_nop_program = triton_program!(
            read_io read_io read_io nop nop write_io push 17 write_io halt
        );
        let public_input = vec![3, 5, 7].into();
        let (_, claim, _, master_ext_table, all_challenges) =
            master_tables_for_low_security_level(&read_nop_program, public_input, [].into());

        let processor_table = master_ext_table.table(ProcessorTable);
        let processor_table_last_row = processor_table.slice(s![-1, ..]);
        let ptie = processor_table_last_row[InputTableEvalArg.ext_table_index()];
        let ine = EvalArg::compute_terminal(
            &claim.input,
            EvalArg::default_initial(),
            all_challenges[StandardInputIndeterminate],
        );
        assert_eq!(ptie, ine, "The input evaluation arguments do not match.");

        let ptoe = processor_table_last_row[OutputTableEvalArg.ext_table_index()];
        let oute = EvalArg::compute_terminal(
            &claim.output,
            EvalArg::default_initial(),
            all_challenges[StandardOutputIndeterminate],
        );
        assert_eq!(ptoe, oute, "The output evaluation arguments do not match.");
    }

    #[test]
    fn check_grand_cross_table_argument() {
        let mut code_collection = small_tasm_test_programs();
        code_collection.append(&mut property_based_test_programs());

        let zero = XFieldElement::zero();
        let circuit_builder = ConstraintCircuitBuilder::new();
        let terminal_constraints = GrandCrossTableArg::terminal_constraints(&circuit_builder);
        let terminal_constraints = terminal_constraints
            .into_iter()
            .map(|c| c.consume())
            .collect_vec();

        for (code_idx, code_with_input) in code_collection.into_iter().enumerate() {
            println!("Checking Grand Cross-Table Argument for TASM snippet {code_idx}.");
            let (_, _, master_base_table, master_ext_table, challenges) =
                master_tables_for_low_security_level(
                    &code_with_input.program,
                    code_with_input.public_input(),
                    code_with_input.non_determinism(),
                );

            let processor_table = master_ext_table.table(ProcessorTable);
            let processor_table_last_row = processor_table.slice(s![-1, ..]);
            assert_eq!(
                challenges[StandardInputTerminal],
                processor_table_last_row[InputTableEvalArg.ext_table_index()],
                "The input terminal must match for TASM snippet #{code_idx}."
            );
            assert_eq!(
                challenges[StandardOutputTerminal],
                processor_table_last_row[OutputTableEvalArg.ext_table_index()],
                "The output terminal must match for TASM snippet #{code_idx}."
            );

            let lookup_table = master_ext_table.table(LookupTable);
            let lookup_table_last_row = lookup_table.slice(s![-1, ..]);
            assert_eq!(
                challenges[LookupTablePublicTerminal],
                lookup_table_last_row[PublicEvaluationArgument.ext_table_index()],
                "The lookup's terminal must match for TASM snippet #{code_idx}."
            );

            let master_base_trace_table = master_base_table.trace_table();
            let master_ext_trace_table = master_ext_table.trace_table();
            let last_master_base_row = master_base_trace_table.slice(s![-1.., ..]);
            let last_master_ext_row = master_ext_trace_table.slice(s![-1.., ..]);

            for (i, constraint) in terminal_constraints.iter().enumerate() {
                assert_eq!(
                    zero,
                    constraint.evaluate(last_master_base_row, last_master_ext_row, &challenges),
                    "Terminal constraint {i} must evaluate to 0 for snippet #{code_idx}."
                );
            }
        }
    }

    #[test]
    fn constraint_polynomials_use_right_variable_count() {
        let challenges = Challenges::placeholder(None);
        let base_row = Array1::<BFieldElement>::zeros(NUM_BASE_COLUMNS);
        let ext_row = Array1::zeros(NUM_EXT_COLUMNS);

        let br = base_row.view();
        let er = ext_row.view();

        MasterExtTable::evaluate_initial_constraints(br, er, &challenges);
        MasterExtTable::evaluate_consistency_constraints(br, er, &challenges);
        MasterExtTable::evaluate_transition_constraints(br, er, br, er, &challenges);
        MasterExtTable::evaluate_terminal_constraints(br, er, &challenges);
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
        let circuit_builder = ConstraintCircuitBuilder::new();
        let all_init = [
            ExtProgramTable::initial_constraints(&circuit_builder),
            ExtProcessorTable::initial_constraints(&circuit_builder),
            ExtOpStackTable::initial_constraints(&circuit_builder),
            ExtRamTable::initial_constraints(&circuit_builder),
            ExtJumpStackTable::initial_constraints(&circuit_builder),
            ExtHashTable::initial_constraints(&circuit_builder),
            ExtCascadeTable::initial_constraints(&circuit_builder),
            ExtLookupTable::initial_constraints(&circuit_builder),
            ExtU32Table::initial_constraints(&circuit_builder),
            GrandCrossTableArg::initial_constraints(&circuit_builder),
        ]
        .map(|vec| vec.len());
        let circuit_builder = ConstraintCircuitBuilder::new();
        let all_cons = [
            ExtProgramTable::consistency_constraints(&circuit_builder),
            ExtProcessorTable::consistency_constraints(&circuit_builder),
            ExtOpStackTable::consistency_constraints(&circuit_builder),
            ExtRamTable::consistency_constraints(&circuit_builder),
            ExtJumpStackTable::consistency_constraints(&circuit_builder),
            ExtHashTable::consistency_constraints(&circuit_builder),
            ExtCascadeTable::consistency_constraints(&circuit_builder),
            ExtLookupTable::consistency_constraints(&circuit_builder),
            ExtU32Table::consistency_constraints(&circuit_builder),
            GrandCrossTableArg::consistency_constraints(&circuit_builder),
        ]
        .map(|vec| vec.len());
        let circuit_builder = ConstraintCircuitBuilder::new();
        let all_trans = [
            ExtProgramTable::transition_constraints(&circuit_builder),
            ExtProcessorTable::transition_constraints(&circuit_builder),
            ExtOpStackTable::transition_constraints(&circuit_builder),
            ExtRamTable::transition_constraints(&circuit_builder),
            ExtJumpStackTable::transition_constraints(&circuit_builder),
            ExtHashTable::transition_constraints(&circuit_builder),
            ExtCascadeTable::transition_constraints(&circuit_builder),
            ExtLookupTable::transition_constraints(&circuit_builder),
            ExtU32Table::transition_constraints(&circuit_builder),
            GrandCrossTableArg::transition_constraints(&circuit_builder),
        ]
        .map(|vec| vec.len());
        let circuit_builder = ConstraintCircuitBuilder::new();
        let all_term = [
            ExtProgramTable::terminal_constraints(&circuit_builder),
            ExtProcessorTable::terminal_constraints(&circuit_builder),
            ExtOpStackTable::terminal_constraints(&circuit_builder),
            ExtRamTable::terminal_constraints(&circuit_builder),
            ExtJumpStackTable::terminal_constraints(&circuit_builder),
            ExtHashTable::terminal_constraints(&circuit_builder),
            ExtCascadeTable::terminal_constraints(&circuit_builder),
            ExtLookupTable::terminal_constraints(&circuit_builder),
            ExtU32Table::terminal_constraints(&circuit_builder),
            GrandCrossTableArg::terminal_constraints(&circuit_builder),
        ]
        .map(|vec| vec.len());

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
    fn number_of_quotient_degree_bounds_match_number_of_constraints() {
        let base_row = Array1::<BFieldElement>::zeros(NUM_BASE_COLUMNS);
        let ext_row = Array1::zeros(NUM_EXT_COLUMNS);
        let challenges = Challenges::placeholder(None);
        let padded_height = 2;
        let num_trace_randomizers = 2;
        let interpolant_degree = interpolant_degree(padded_height, num_trace_randomizers);

        // Shorten some names for better formatting. This is just a test.
        let ph = padded_height;
        let id = interpolant_degree;
        let br = base_row.view();
        let er = ext_row.view();

        assert_eq!(
            MasterExtTable::num_initial_quotients(),
            MasterExtTable::evaluate_initial_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            MasterExtTable::num_initial_quotients(),
            MasterExtTable::initial_quotient_degree_bounds(id).len()
        );
        assert_eq!(
            MasterExtTable::num_consistency_quotients(),
            MasterExtTable::evaluate_consistency_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            MasterExtTable::num_consistency_quotients(),
            MasterExtTable::consistency_quotient_degree_bounds(id, ph).len()
        );
        assert_eq!(
            MasterExtTable::num_transition_quotients(),
            MasterExtTable::evaluate_transition_constraints(br, er, br, er, &challenges).len(),
        );
        assert_eq!(
            MasterExtTable::num_transition_quotients(),
            MasterExtTable::transition_quotient_degree_bounds(id, ph).len()
        );
        assert_eq!(
            MasterExtTable::num_terminal_quotients(),
            MasterExtTable::evaluate_terminal_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            MasterExtTable::num_terminal_quotients(),
            MasterExtTable::terminal_quotient_degree_bounds(id).len()
        );
    }

    #[test]
    fn triton_table_constraints_evaluate_to_zero_on_halt() {
        triton_table_constraints_evaluate_to_zero(test_program_for_halt());
    }

    #[test]
    fn triton_table_constraints_evaluate_to_zero_on_fibonacci() {
        let source_code_and_input = ProgramAndInput {
            program: FIBONACCI_SEQUENCE.clone(),
            public_input: vec![100],
            non_determinism: [].into(),
        };
        triton_table_constraints_evaluate_to_zero(source_code_and_input);
    }

    #[test]
    fn triton_table_constraints_evaluate_to_zero_on_big_mmr_snippet() {
        let source_code_and_input = ProgramAndInput::without_input(
            CALCULATE_NEW_MMR_PEAKS_FROM_APPEND_WITH_SAFE_LISTS.clone(),
        );
        triton_table_constraints_evaluate_to_zero(source_code_and_input);
    }

    #[test]
    fn triton_table_constraints_evaluate_to_zero_on_small_programs() {
        for (program_idx, program) in small_tasm_test_programs().into_iter().enumerate() {
            println!("Testing program with index {program_idx}.");
            triton_table_constraints_evaluate_to_zero(program);
        }
    }

    #[test]
    fn triton_table_constraints_evaluate_to_zero_on_property_based_programs() {
        for (program_idx, program) in property_based_test_programs().into_iter().enumerate() {
            println!("Testing program with index {program_idx}.");
            triton_table_constraints_evaluate_to_zero(program);
        }
    }

    fn triton_table_constraints_evaluate_to_zero(source_code_and_input: ProgramAndInput) {
        let (_, _, master_base_table, master_ext_table, challenges) =
            master_tables_for_low_security_level(
                &source_code_and_input.program,
                source_code_and_input.public_input(),
                source_code_and_input.non_determinism(),
            );

        assert_eq!(
            master_base_table.randomized_trace_table().nrows(),
            master_ext_table.randomized_trace_table().nrows()
        );
        let master_base_trace_table = master_base_table.trace_table();
        let master_ext_trace_table = master_ext_table.trace_table();
        assert_eq!(
            master_base_trace_table.nrows(),
            master_ext_trace_table.nrows()
        );

        assert!(program_table::tests::constraints_evaluate_to_zero(
            master_base_trace_table,
            master_ext_trace_table,
            &challenges,
        ));
        assert!(processor_table::tests::constraints_evaluate_to_zero(
            master_base_trace_table,
            master_ext_trace_table,
            &challenges,
        ));
        assert!(op_stack_table::tests::constraints_evaluate_to_zero(
            master_base_trace_table,
            master_ext_trace_table,
            &challenges,
        ));
        assert!(ram_table::tests::constraints_evaluate_to_zero(
            master_base_trace_table,
            master_ext_trace_table,
            &challenges,
        ));
        assert!(jump_stack_table::tests::constraints_evaluate_to_zero(
            master_base_trace_table,
            master_ext_trace_table,
            &challenges,
        ));
        assert!(hash_table::tests::constraints_evaluate_to_zero(
            master_base_trace_table,
            master_ext_trace_table,
            &challenges,
        ));
        assert!(cascade_table::tests::constraints_evaluate_to_zero(
            master_base_trace_table,
            master_ext_trace_table,
            &challenges,
        ));
        assert!(lookup_table::tests::constraints_evaluate_to_zero(
            master_base_trace_table,
            master_ext_trace_table,
            &challenges,
        ));
        assert!(u32_table::tests::constraints_evaluate_to_zero(
            master_base_trace_table,
            master_ext_trace_table,
            &challenges,
        ));
    }

    #[test]
    fn derived_constraints_evaluate_to_zero_on_halt() {
        derived_constraints_evaluate_to_zero(test_program_for_halt());
    }

    fn derived_constraints_evaluate_to_zero(source_code_and_input: ProgramAndInput) {
        let (_, _, master_base_table, master_ext_table, challenges) =
            master_tables_for_low_security_level(
                &source_code_and_input.program,
                source_code_and_input.public_input(),
                source_code_and_input.non_determinism(),
            );

        let zero = XFieldElement::zero();
        let master_base_trace_table = master_base_table.trace_table();
        let master_ext_trace_table = master_ext_table.trace_table();

        let evaluated_initial_constraints = MasterExtTable::evaluate_initial_constraints(
            master_base_trace_table.row(0),
            master_ext_trace_table.row(0),
            &challenges,
        );
        for (constraint_idx, evaluated_constraint) in
            evaluated_initial_constraints.into_iter().enumerate()
        {
            assert_eq!(
                zero, evaluated_constraint,
                "Initial constraint {constraint_idx} failed.",
            );
        }

        for row_idx in 0..master_base_trace_table.nrows() {
            let evaluated_consistency_constraints =
                MasterExtTable::evaluate_consistency_constraints(
                    master_base_trace_table.row(row_idx),
                    master_ext_trace_table.row(row_idx),
                    &challenges,
                );
            for (constraint_idx, evaluated_constraint) in
                evaluated_consistency_constraints.into_iter().enumerate()
            {
                assert_eq!(
                    zero, evaluated_constraint,
                    "Consistency constraint {constraint_idx} failed in row {row_idx}.",
                );
            }
        }

        for curr_row_idx in 0..master_base_trace_table.nrows() - 1 {
            let next_row_idx = curr_row_idx + 1;
            let evaluated_transition_constraints = MasterExtTable::evaluate_transition_constraints(
                master_base_trace_table.row(curr_row_idx),
                master_ext_trace_table.row(curr_row_idx),
                master_base_trace_table.row(next_row_idx),
                master_ext_trace_table.row(next_row_idx),
                &challenges,
            );
            for (constraint_idx, evaluated_constraint) in
                evaluated_transition_constraints.into_iter().enumerate()
            {
                assert_eq!(
                    zero, evaluated_constraint,
                    "Transition constraint {constraint_idx} failed in row {curr_row_idx}.",
                );
            }
        }

        let evaluated_terminal_constraints = MasterExtTable::evaluate_terminal_constraints(
            master_base_trace_table.row(master_base_trace_table.nrows() - 1),
            master_ext_trace_table.row(master_ext_trace_table.nrows() - 1),
            &challenges,
        );
        for (constraint_idx, evaluated_constraint) in
            evaluated_terminal_constraints.into_iter().enumerate()
        {
            assert_eq!(
                zero, evaluated_constraint,
                "Terminal constraint {constraint_idx} failed.",
            );
        }
    }

    #[test]
    fn triton_prove_verify_simple_program() {
        let program_with_input = test_program_hash_nop_nop_lt();
        let (parameters, claim, proof) = prove_with_low_security_level(
            &program_with_input.program,
            program_with_input.public_input(),
            program_with_input.non_determinism(),
            &mut None,
        );

        let verdict = Stark::verify(parameters, &claim, &proof, &mut None).unwrap();
        assert!(verdict);
    }

    #[test]
    fn triton_prove_verify_halt() {
        let code_with_input = test_program_for_halt();
        let mut profiler = Some(TritonProfiler::new("Prove Halt"));
        let (parameters, claim, proof) = prove_with_low_security_level(
            &code_with_input.program,
            code_with_input.public_input(),
            code_with_input.non_determinism(),
            &mut profiler,
        );
        let mut profiler = profiler.unwrap();
        profiler.finish();

        let result = Stark::verify(parameters, &claim, &proof, &mut None);
        if let Err(e) = result {
            panic!("The Verifier is unhappy! {e}");
        }
        assert!(result.unwrap());

        let padded_height = proof.padded_height().unwrap();
        let fri = Stark::derive_fri(parameters, padded_height);
        let report = profiler
            .report()
            .with_padded_height(padded_height)
            .with_fri_domain_len(fri.domain.length);
        println!("{report}");
    }

    #[test]
    #[ignore = "used for tracking&debugging deserialization errors"]
    fn triton_prove_halt_save_error() {
        let code_with_input = test_program_for_halt();

        for _ in 0..100 {
            let (parameters, claim, proof) = prove_with_low_security_level(
                &code_with_input.program,
                code_with_input.public_input(),
                code_with_input.non_determinism(),
                &mut None,
            );

            let verdict = Stark::verify(parameters, &claim, &proof, &mut None);
            if verdict.is_err() {
                let filename = "halt_error.tsp";
                save_proof(filename, proof).unwrap();
                eprintln!("Saved proof to {filename}.");
            };
            assert!(verdict.unwrap());
        }
    }

    #[test]
    #[ignore = "used for tracking&debugging deserialization errors"]
    fn triton_load_verify_halt() {
        let code_with_input = test_program_for_halt();
        let (parameters, claim, _) = prove_with_low_security_level(
            &code_with_input.program,
            code_with_input.public_input(),
            code_with_input.non_determinism(),
            &mut None,
        );

        let filename = "halt_error.tsp";
        let proof = load_proof(filename).unwrap();
        let verdict = Stark::verify(parameters, &claim, &proof, &mut None).unwrap();
        assert!(verdict);
    }

    #[test]
    fn prove_verify_fibonacci_100() {
        let stdin = vec![100].into();
        let secret_in = [].into();

        let mut profiler = Some(TritonProfiler::new("Prove Fib 100"));
        let (parameters, claim, proof) =
            prove_with_low_security_level(&FIBONACCI_SEQUENCE, stdin, secret_in, &mut profiler);
        let mut profiler = profiler.unwrap();
        profiler.finish();

        println!("between prove and verify");

        let result = Stark::verify(parameters, &claim, &proof, &mut None);
        if let Err(e) = result {
            panic!("The Verifier is unhappy! {e}");
        }
        assert!(result.unwrap());

        let padded_height = proof.padded_height().unwrap();
        let fri = Stark::derive_fri(parameters, padded_height);
        let report = profiler
            .report()
            .with_padded_height(padded_height)
            .with_fri_domain_len(fri.domain.length);
        println!("{report}");
    }

    #[test]
    fn prove_verify_fib_shootout() {
        for (fib_seq_idx, fib_seq_val) in [(0, 1), (7, 21), (11, 144)] {
            let stdin = vec![fib_seq_idx].into();
            let secret_in = [].into();
            let (parameters, claim, proof) =
                prove_with_low_security_level(&FIBONACCI_SEQUENCE, stdin, secret_in, &mut None);
            match Stark::verify(parameters, &claim, &proof, &mut None) {
                Ok(result) => assert!(result, "The Verifier disagrees!"),
                Err(err) => panic!("The Verifier is unhappy! {err}"),
            }

            assert_eq!(vec![fib_seq_val], claim.public_output());
        }
    }

    #[test]
    fn constraints_evaluate_to_zero_on_many_u32_operations() {
        let many_u32_instructions =
            ProgramAndInput::without_input(PROGRAM_WITH_MANY_U32_INSTRUCTIONS.clone());
        triton_table_constraints_evaluate_to_zero(many_u32_instructions);
    }

    #[test]
    fn triton_prove_verify_many_u32_operations() {
        let mut profiler = Some(TritonProfiler::new("Prove Many U32 Ops"));
        let (parameters, claim, proof) = prove_with_low_security_level(
            &PROGRAM_WITH_MANY_U32_INSTRUCTIONS,
            [].into(),
            [].into(),
            &mut profiler,
        );
        let mut profiler = profiler.unwrap();
        profiler.finish();

        let result = Stark::verify(parameters, &claim, &proof, &mut None);
        if let Err(e) = result {
            panic!("The Verifier is unhappy! {e}");
        }
        assert!(result.unwrap());

        let padded_height = proof.padded_height().unwrap();
        let fri = Stark::derive_fri(parameters, padded_height);
        let report = profiler
            .report()
            .with_padded_height(padded_height)
            .with_fri_domain_len(fri.domain.length);
        println!("{report}");
    }

    #[proptest]
    fn verifying_arbitrary_proof_does_not_panic(
        #[strategy(arb())] parameters: StarkParameters,
        #[strategy(arb())] claim: Claim,
        #[strategy(arb())] proof: Proof,
    ) {
        let _ = Stark::verify(parameters, &claim, &proof, &mut None);
    }

    #[test]
    #[ignore = "stress test"]
    fn prove_fib_successively_larger() {
        for fibonacci_number in [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200] {
            let stdin = vec![fibonacci_number].into();
            let fib_test_name = format!("element #{fibonacci_number:>4} from Fibonacci sequence");
            let mut profiler = Some(TritonProfiler::new(&fib_test_name));
            let (parameters, _, proof) =
                prove_with_low_security_level(&FIBONACCI_SEQUENCE, stdin, [].into(), &mut profiler);
            let mut profiler = profiler.unwrap();
            profiler.finish();

            let padded_height = proof.padded_height().unwrap();
            let fri = Stark::derive_fri(parameters, padded_height);
            let report = profiler
                .report()
                .with_padded_height(padded_height)
                .with_fri_domain_len(fri.domain.length);
            println!("{report}");
        }
    }

    #[test]
    #[should_panic(expected = "Failed to convert BFieldElement")]
    pub fn negative_log_2_floor() {
        let mut rng = ThreadRng::default();
        let st0 = (rng.next_u32() as u64) << 32;

        let program = triton_program!(push {st0} log_2_floor halt);
        let (parameters, claim, proof) =
            prove_with_low_security_level(&program, [].into(), [].into(), &mut None);
        let result = Stark::verify(parameters, &claim, &proof, &mut None);
        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    #[should_panic(expected = "The logarithm of 0 does not exist")]
    pub fn negative_log_2_floor_of_0() {
        let program = triton_program!(push 0 log_2_floor halt);
        let (parameters, claim, proof) =
            prove_with_low_security_level(&program, [].into(), [].into(), &mut None);
        let result = Stark::verify(parameters, &claim, &proof, &mut None);
        assert!(result.is_ok());
        assert!(result.unwrap());
    }

    #[test]
    fn deep_update() {
        let domain_length = 1 << 10;
        let domain = ArithmeticDomain::of_length(domain_length);

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

    /// Re-compose the segments of a polynomial and assert that the result is equal to the
    /// polynomial itself. Uses the Schwartz-Zippel lemma to test polynomial equality.
    fn assert_polynomial_equals_recomposed_segments<const N: usize, FF: FiniteField>(
        f: &Polynomial<FF>,
        segments: &[Polynomial<FF>; N],
        x: FF,
    ) {
        let x_pow_n = x.mod_pow_u32(N as u32);
        let evaluate_segment = |(segment_idx, segment): (_, &Polynomial<_>)| {
            segment.evaluate(&x_pow_n) * x.mod_pow_u32(segment_idx as u32)
        };
        let evaluated_segments = segments.iter().enumerate().map(evaluate_segment);
        let sum_of_evaluated_segments = evaluated_segments.fold(FF::zero(), |acc, x| acc + x);
        assert_eq!(f.evaluate(&x), sum_of_evaluated_segments);
    }

    fn assert_segments_degrees_are_small_enough<const N: usize, FF: FiniteField>(
        f: &Polynomial<FF>,
        segments: &[Polynomial<FF>; N],
    ) {
        let max_allowed_degree = f.degree() / (N as isize);
        let all_degrees_are_small_enough =
            segments.iter().all(|s| s.degree() <= max_allowed_degree);
        assert!(all_degrees_are_small_enough);
    }

    #[test]
    fn split_polynomial_into_segments_of_unequal_size() {
        let coefficients: [XFieldElement; 211] = thread_rng().gen();
        let f = Polynomial::new(coefficients.to_vec());

        let segments_2 = Stark::split_polynomial_into_segments::<2, _>(&f);
        let segments_3 = Stark::split_polynomial_into_segments::<3, _>(&f);
        let segments_4 = Stark::split_polynomial_into_segments::<4, _>(&f);
        let segments_7 = Stark::split_polynomial_into_segments::<7, _>(&f);

        assert_segments_degrees_are_small_enough(&f, &segments_2);
        assert_segments_degrees_are_small_enough(&f, &segments_3);
        assert_segments_degrees_are_small_enough(&f, &segments_4);
        assert_segments_degrees_are_small_enough(&f, &segments_7);

        let x = thread_rng().gen();
        assert_polynomial_equals_recomposed_segments(&f, &segments_2, x);
        assert_polynomial_equals_recomposed_segments(&f, &segments_3, x);
        assert_polynomial_equals_recomposed_segments(&f, &segments_4, x);
        assert_polynomial_equals_recomposed_segments(&f, &segments_7, x);
    }

    #[test]
    fn split_polynomial_into_segments_of_equal_size() {
        let coefficients: [BFieldElement; 2 * 3 * 4 * 7] = thread_rng().gen();
        let f = Polynomial::new(coefficients.to_vec());

        let segments_2 = Stark::split_polynomial_into_segments::<2, _>(&f);
        let segments_3 = Stark::split_polynomial_into_segments::<3, _>(&f);
        let segments_4 = Stark::split_polynomial_into_segments::<4, _>(&f);
        let segments_7 = Stark::split_polynomial_into_segments::<7, _>(&f);

        assert_segments_degrees_are_small_enough(&f, &segments_2);
        assert_segments_degrees_are_small_enough(&f, &segments_3);
        assert_segments_degrees_are_small_enough(&f, &segments_4);
        assert_segments_degrees_are_small_enough(&f, &segments_7);

        let x = thread_rng().gen();
        assert_polynomial_equals_recomposed_segments(&f, &segments_2, x);
        assert_polynomial_equals_recomposed_segments(&f, &segments_3, x);
        assert_polynomial_equals_recomposed_segments(&f, &segments_4, x);
        assert_polynomial_equals_recomposed_segments(&f, &segments_7, x);
    }
}
