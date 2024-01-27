use std::ops::Add;
use std::ops::Mul;

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
use twenty_first::prelude::*;
use twenty_first::shared_math::mpolynomial::Degree;
use twenty_first::shared_math::traits::FiniteField;

use crate::aet::AlgebraicExecutionTrace;
use crate::arithmetic_domain::ArithmeticDomain;
use crate::error::ProvingError;
use crate::error::VerificationError;
use crate::error::VerificationError::*;
use crate::fri;
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

    /// The number of collinearity checks to perform in FRI.
    pub num_collinearity_checks: usize,

    /// The number of combination codeword checks. These checks link the (DEEP) ALI part and the
    /// FRI part of the zk-STARK. The number of combination codeword checks directly depends on the
    /// number of collinearity checks and the FRI folding factor.
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
        let num_collinearity_checks = security_level / log2_of_fri_expansion_factor;

        // For now, the FRI folding factor is hardcoded in our zk-STARK.
        let fri_folding_factor = 2;
        let num_combination_codeword_checks = num_collinearity_checks * fri_folding_factor;

        let num_out_of_domain_rows = 2;
        let num_trace_randomizers = num_combination_codeword_checks
            + num_out_of_domain_rows * x_field_element::EXTENSION_DEGREE;

        StarkParameters {
            security_level,
            fri_expansion_factor,
            num_trace_randomizers,
            num_randomizer_polynomials,
            num_collinearity_checks,
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
    ) -> Result<Proof, ProvingError> {
        prof_start!(maybe_profiler, "Fiat-Shamir: claim", "hash");
        let mut proof_stream = StarkProofStream::new();
        proof_stream.alter_fiat_shamir_state_with(claim);
        prof_stop!(maybe_profiler, "Fiat-Shamir: claim");

        prof_start!(maybe_profiler, "derive additional parameters");
        let padded_height = aet.padded_height();
        let max_degree = Self::derive_max_degree(padded_height, parameters.num_trace_randomizers);
        let fri = Self::derive_fri(parameters, padded_height)?;
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
        proof_stream.enqueue(ProofItem::MerkleRoot(base_merkle_tree.root()));
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
        proof_stream.enqueue(ProofItem::MerkleRoot(ext_merkle_tree.root()));
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
            MTMaker::from_digests(&fri_domain_quotient_segment_codewords_digests)?;
        let quot_merkle_tree_root = quot_merkle_tree.root();
        proof_stream.enqueue(ProofItem::MerkleRoot(quot_merkle_tree_root));
        prof_stop!(maybe_profiler, "Merkle tree");
        prof_stop!(maybe_profiler, "commit to quotient codeword segments");
        debug_assert_eq!(fri.domain.length, quot_merkle_tree.num_leafs());

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
        let revealed_current_row_indices =
            fri.prove(&fri_combination_codeword, &mut proof_stream)?;
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
            base_merkle_tree.authentication_structure(&revealed_current_row_indices)?;
        proof_stream.enqueue(ProofItem::MasterBaseTableRows(revealed_base_elems));
        proof_stream.enqueue(ProofItem::AuthenticationStructure(
            base_authentication_structure,
        ));

        let revealed_ext_elems = Self::get_revealed_elements(
            master_ext_table.fri_domain_table(),
            &revealed_current_row_indices,
        );
        let ext_authentication_structure =
            ext_merkle_tree.authentication_structure(&revealed_current_row_indices)?;
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
            quot_merkle_tree.authentication_structure(&revealed_current_row_indices)?;
        proof_stream.enqueue(ProofItem::QuotientSegmentsElements(
            revealed_quotient_segments_rows,
        ));
        proof_stream.enqueue(ProofItem::AuthenticationStructure(
            revealed_quotient_authentication_structure,
        ));
        prof_stop!(maybe_profiler, "open trace leafs");

        Ok(proof_stream.into())
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
    pub fn derive_fri(
        parameters: StarkParameters,
        padded_height: usize,
    ) -> fri::SetupResult<Fri<StarkHasher>> {
        let interpolant_degree =
            interpolant_degree(padded_height, parameters.num_trace_randomizers);
        let interpolant_codeword_length = interpolant_degree as usize + 1;
        let fri_domain_length = parameters.fri_expansion_factor * interpolant_codeword_length;
        let coset_offset = BFieldElement::generator();
        let domain = ArithmeticDomain::of_length(fri_domain_length).with_offset(coset_offset);

        Fri::new(
            domain,
            parameters.fri_expansion_factor,
            parameters.num_collinearity_checks,
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

    pub fn verify(
        parameters: StarkParameters,
        claim: &Claim,
        proof: &Proof,
        maybe_profiler: &mut Option<TritonProfiler>,
    ) -> Result<(), VerificationError> {
        prof_start!(maybe_profiler, "deserialize");
        let mut proof_stream = StarkProofStream::try_from(proof)?;
        prof_stop!(maybe_profiler, "deserialize");

        prof_start!(maybe_profiler, "Fiat-Shamir: Claim", "hash");
        proof_stream.alter_fiat_shamir_state_with(claim);
        prof_stop!(maybe_profiler, "Fiat-Shamir: Claim");

        prof_start!(maybe_profiler, "derive additional parameters");
        let log_2_padded_height = proof_stream.dequeue()?.try_into_log2_padded_height()?;
        let padded_height = 1 << log_2_padded_height;
        let fri = Self::derive_fri(parameters, padded_height)?;
        let merkle_tree_height = fri.domain.length.ilog2() as usize;
        prof_stop!(maybe_profiler, "derive additional parameters");

        prof_start!(maybe_profiler, "Fiat-Shamir 1", "hash");
        let base_merkle_tree_root = proof_stream.dequeue()?.try_into_merkle_root()?;
        let extension_challenge_weights =
            proof_stream.sample_scalars(Challenges::num_challenges_to_sample());
        let challenges = Challenges::new(extension_challenge_weights, claim);
        let extension_tree_merkle_root = proof_stream.dequeue()?.try_into_merkle_root()?;
        // Sample weights for quotient codeword, which is a part of the combination codeword.
        // See corresponding part in the prover for a more detailed explanation.
        let quot_codeword_weights = proof_stream.sample_scalars(num_quotients());
        let quot_codeword_weights = Array1::from(quot_codeword_weights);
        let quotient_codeword_merkle_root = proof_stream.dequeue()?.try_into_merkle_root()?;
        prof_stop!(maybe_profiler, "Fiat-Shamir 1");

        prof_start!(maybe_profiler, "dequeue ood point and rows", "hash");
        let trace_domain_generator = ArithmeticDomain::generator_for_length(padded_height as u64);
        let out_of_domain_point_curr_row = proof_stream.sample_scalars(1)[0];
        let out_of_domain_point_next_row = trace_domain_generator * out_of_domain_point_curr_row;
        let out_of_domain_point_curr_row_pow_num_segments =
            out_of_domain_point_curr_row.mod_pow_u32(NUM_QUOTIENT_SEGMENTS as u32);

        let out_of_domain_curr_base_row =
            proof_stream.dequeue()?.try_into_out_of_domain_base_row()?;
        let out_of_domain_curr_ext_row =
            proof_stream.dequeue()?.try_into_out_of_domain_ext_row()?;
        let out_of_domain_next_base_row =
            proof_stream.dequeue()?.try_into_out_of_domain_base_row()?;
        let out_of_domain_next_ext_row =
            proof_stream.dequeue()?.try_into_out_of_domain_ext_row()?;
        let out_of_domain_curr_row_quot_segments = proof_stream
            .dequeue()?
            .try_into_out_of_domain_quot_segments()?;

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
        if out_of_domain_quotient_value != sum_of_evaluated_out_of_domain_quotient_segments {
            return Err(OutOfDomainQuotientValueMismatch);
        };
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
        let base_table_rows = proof_stream.dequeue()?.try_into_master_base_table_rows()?;
        let base_authentication_structure = proof_stream
            .dequeue()?
            .try_into_authentication_structure()?;
        let leaf_digests_base: Vec<_> = base_table_rows
            .par_iter()
            .map(|revealed_base_elem| StarkHasher::hash_varlen(revealed_base_elem))
            .collect();
        prof_stop!(maybe_profiler, "dequeue base elements");

        let index_leaves = |leaves| {
            let index_iter = revealed_current_row_indices.iter().copied();
            index_iter.zip_eq(leaves).collect()
        };
        prof_start!(maybe_profiler, "Merkle verify (base tree)", "hash");
        let base_merkle_tree_inclusion_proof = MerkleTreeInclusionProof::<Tip5> {
            tree_height: merkle_tree_height,
            indexed_leaves: index_leaves(leaf_digests_base),
            authentication_structure: base_authentication_structure,
            ..Default::default()
        };
        if !base_merkle_tree_inclusion_proof.verify(base_merkle_tree_root) {
            return Err(BaseCodewordAuthenticationFailure);
        }
        prof_stop!(maybe_profiler, "Merkle verify (base tree)");

        prof_start!(maybe_profiler, "dequeue extension elements");
        let ext_table_rows = proof_stream.dequeue()?.try_into_master_ext_table_rows()?;
        let ext_authentication_structure = proof_stream
            .dequeue()?
            .try_into_authentication_structure()?;
        let leaf_digests_ext = ext_table_rows
            .par_iter()
            .map(|xvalues| {
                let b_values = xvalues.iter().flat_map(|xfe| xfe.coefficients.to_vec());
                StarkHasher::hash_varlen(&b_values.collect_vec())
            })
            .collect::<Vec<_>>();
        prof_stop!(maybe_profiler, "dequeue extension elements");

        prof_start!(maybe_profiler, "Merkle verify (extension tree)", "hash");
        let ext_merkle_tree_inclusion_proof = MerkleTreeInclusionProof::<Tip5> {
            tree_height: merkle_tree_height,
            indexed_leaves: index_leaves(leaf_digests_ext),
            authentication_structure: ext_authentication_structure,
            ..Default::default()
        };
        if !ext_merkle_tree_inclusion_proof.verify(extension_tree_merkle_root) {
            return Err(ExtensionCodewordAuthenticationFailure);
        }
        prof_stop!(maybe_profiler, "Merkle verify (extension tree)");

        prof_start!(maybe_profiler, "dequeue quotient segments' elements");
        let revealed_quotient_segments_elements =
            proof_stream.dequeue()?.try_into_quot_segments_elements()?;
        let revealed_quotient_segments_digests =
            Self::hash_quotient_segment_elements(&revealed_quotient_segments_elements);
        let revealed_quotient_authentication_structure = proof_stream
            .dequeue()?
            .try_into_authentication_structure()?;
        prof_stop!(maybe_profiler, "dequeue quotient segments' elements");

        prof_start!(maybe_profiler, "Merkle verify (combined quotient)", "hash");
        let quot_merkle_tree_inclusion_proof = MerkleTreeInclusionProof::<Tip5> {
            tree_height: merkle_tree_height,
            indexed_leaves: index_leaves(revealed_quotient_segments_digests),
            authentication_structure: revealed_quotient_authentication_structure,
            ..Default::default()
        };
        if !quot_merkle_tree_inclusion_proof.verify(quotient_codeword_merkle_root) {
            return Err(QuotientCodewordAuthenticationFailure);
        }
        prof_stop!(maybe_profiler, "Merkle verify (combined quotient)");
        prof_stop!(maybe_profiler, "check leafs");

        prof_start!(maybe_profiler, "linear combination");
        if parameters.num_combination_codeword_checks != revealed_current_row_indices.len() {
            return Err(IncorrectNumberOfRowIndices);
        };
        if parameters.num_combination_codeword_checks != revealed_fri_values.len() {
            return Err(IncorrectNumberOfFRIValues);
        };
        if parameters.num_combination_codeword_checks != revealed_quotient_segments_elements.len() {
            return Err(IncorrectNumberOfQuotientSegmentElements);
        };
        if parameters.num_combination_codeword_checks != base_table_rows.len() {
            return Err(IncorrectNumberOfBaseTableRows);
        };
        if parameters.num_combination_codeword_checks != ext_table_rows.len() {
            return Err(IncorrectNumberOfExtTableRows);
        };

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
            if fri_value != deep_value + randomizer_codewords_contribution {
                return Err(CombinationCodewordMismatch);
            };
            prof_stop!(maybe_profiler, "combination codeword equality");
        }
        prof_stop!(maybe_profiler, "main loop");
        prof_stop!(maybe_profiler, "linear combination");
        Ok(())
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
    use crate::error::InstructionError;
    use assert2::assert;
    use assert2::check;
    use assert2::let_assert;
    use itertools::izip;
    use num_traits::Zero;
    use proptest_arbitrary_interop::arb;
    use rand::thread_rng;
    use rand::Rng;
    use strum::EnumCount;
    use test_strategy::proptest;
    use twenty_first::shared_math::other::random_elements;

    use crate::example_programs::*;
    use crate::instruction::Instruction;
    use crate::op_stack::OpStackElement;
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
    use crate::PublicInput;

    use super::*;

    pub(crate) fn master_base_table_for_low_security_level(
        program_and_input: ProgramAndInput,
    ) -> (StarkParameters, Claim, MasterBaseTable) {
        let ProgramAndInput {
            program,
            public_input,
            non_determinism,
        } = program_and_input;
        let public_input: PublicInput = public_input.into();
        let non_determinism = (&non_determinism).into();

        let (aet, stdout) = program
            .trace_execution(public_input.clone(), non_determinism)
            .unwrap();
        let parameters = stark_parameters_with_low_security_level();
        let claim = construct_claim(&aet, public_input.individual_tokens, stdout);
        let master_base_table = construct_master_base_table(parameters, &aet);

        (parameters, claim, master_base_table)
    }

    pub(crate) fn master_tables_for_low_security_level(
        program_and_input: ProgramAndInput,
    ) -> (
        StarkParameters,
        Claim,
        MasterBaseTable,
        MasterExtTable,
        Challenges,
    ) {
        let (parameters, claim, mut master_base_table) =
            master_base_table_for_low_security_level(program_and_input);

        let challenges = Challenges::deterministic_placeholder(Some(&claim));
        master_base_table.pad();
        let master_ext_table =
            master_base_table.extend(&challenges, parameters.num_randomizer_polynomials);

        (
            parameters,
            claim,
            master_base_table,
            master_ext_table,
            challenges,
        )
    }

    #[test]
    fn print_ram_table_example_for_specification() {
        let program = triton_program!(
            push 20 push 100 write_mem 1 pop 1  // write 20 to address 100
            push 5 push 6 push 7 push 8 push 9
            push 42 write_mem 5 pop 1           // write 5..=9 to addresses 42..=46
            push 42 read_mem 1 pop 2            // read from address 42
            push 45 read_mem 3 pop 4            // read from address 42..=44
            push 17 push 18 push 19
            push 43 write_mem 3 pop 1           // write 17..=19 to addresses 43..=45
            push 46 read_mem 5 pop 1 pop 5      // read from addresses 42..=46
            push 42 read_mem 1 pop 2            // read from address 42
            push 100 read_mem 1 pop 2           // read from address 100
            halt
        );
        let (_, _, master_base_table, _, _) =
            master_tables_for_low_security_level(ProgramAndInput::without_input(program));

        println!();
        println!("Processor Table:\n");
        println!("| clk | ci  | nia | st0 | st1 | st2 | st3 | st4 | st5 |");
        println!("|----:|:----|:----|----:|----:|----:|----:|----:|----:|");
        for row in master_base_table
            .table(ProcessorTable)
            .rows()
            .into_iter()
            .take(40)
        {
            let clk = row[ProcessorBaseTableColumn::CLK.base_table_index()].to_string();
            let st0 = row[ProcessorBaseTableColumn::ST0.base_table_index()].to_string();
            let st1 = row[ProcessorBaseTableColumn::ST1.base_table_index()].to_string();
            let st2 = row[ProcessorBaseTableColumn::ST2.base_table_index()].to_string();
            let st3 = row[ProcessorBaseTableColumn::ST3.base_table_index()].to_string();
            let st4 = row[ProcessorBaseTableColumn::ST4.base_table_index()].to_string();
            let st5 = row[ProcessorBaseTableColumn::ST5.base_table_index()].to_string();

            let (ci, nia) = ci_and_nia_from_master_table_row(row);

            let interesting_cols = [clk, ci, nia, st0, st1, st2, st3, st4, st5];
            let interesting_cols = interesting_cols
                .iter()
                .map(|ff| format!("{:>10}", format!("{ff}")))
                .join(" | ");
            println!("| {interesting_cols} |");
        }
        println!();
        println!("RAM Table:\n");
        println!("| clk | type | pointer | value | iord |");
        println!("|----:|:-----|--------:|------:|-----:|");
        for row in master_base_table
            .table(TableId::RamTable)
            .rows()
            .into_iter()
            .take(25)
        {
            let clk = row[RamBaseTableColumn::CLK.base_table_index()].to_string();
            let ramp = row[RamBaseTableColumn::RamPointer.base_table_index()].to_string();
            let ramv = row[RamBaseTableColumn::RamValue.base_table_index()].to_string();
            let iord =
                row[RamBaseTableColumn::InverseOfRampDifference.base_table_index()].to_string();

            let instruction_type =
                match row[RamBaseTableColumn::InstructionType.base_table_index()] {
                    ram_table::INSTRUCTION_TYPE_READ => "read",
                    ram_table::INSTRUCTION_TYPE_WRITE => "write",
                    ram_table::PADDING_INDICATOR => "pad",
                    _ => "-",
                }
                .to_string();

            let interesting_cols = [clk, instruction_type, ramp, ramv, iord];
            let interesting_cols = interesting_cols
                .iter()
                .map(|ff| format!("{:>10}", format!("{ff}")))
                .join(" | ");
            println!("| {interesting_cols} |");
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
            master_base_table_for_low_security_level(ProgramAndInput::without_input(program));

        println!();
        println!("Processor Table:");
        println!("| clk | ci  | nia | st0 | st1 | st2 | st3 | underflow | pointer |");
        println!("|----:|:----|----:|----:|----:|----:|----:|:----------|--------:|");
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
            println!("| {interesting_cols} |");
        }

        println!();
        println!("Op Stack Table:");
        println!("| clk | ib1 | pointer | value |");
        println!("|----:|----:|--------:|------:|");
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
            println!("| {interesting_cols} |");
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
            read_io 3 nop nop write_io 2 push 17 write_io 1 halt
        );
        let mut program_and_input = ProgramAndInput::without_input(read_nop_program);
        program_and_input.public_input = vec![3, 5, 7];
        let (_, claim, _, master_ext_table, all_challenges) =
            master_tables_for_low_security_level(program_and_input);

        let processor_table = master_ext_table.table(ProcessorTable);
        let processor_table_last_row = processor_table.slice(s![-1, ..]);
        let ptie = processor_table_last_row[InputTableEvalArg.ext_table_index()];
        let ine = EvalArg::compute_terminal(
            &claim.input,
            EvalArg::default_initial(),
            all_challenges[StandardInputIndeterminate],
        );
        check!(ptie == ine);

        let ptoe = processor_table_last_row[OutputTableEvalArg.ext_table_index()];
        let oute = EvalArg::compute_terminal(
            &claim.output,
            EvalArg::default_initial(),
            all_challenges[StandardOutputIndeterminate],
        );
        check!(ptoe == oute);
    }

    #[test]
    fn check_grand_cross_table_argument_for_test_program_for_halt() {
        check_grand_cross_table_argument(test_program_for_halt())
    }

    #[test]
    fn check_grand_cross_table_argument_for_test_program_hash_nop_nop_lt() {
        check_grand_cross_table_argument(test_program_hash_nop_nop_lt())
    }

    #[test]
    fn check_grand_cross_table_argument_for_test_program_for_push_pop_dup_swap_nop() {
        check_grand_cross_table_argument(test_program_for_push_pop_dup_swap_nop())
    }

    #[test]
    fn check_grand_cross_table_argument_for_test_program_for_divine() {
        check_grand_cross_table_argument(test_program_for_divine())
    }

    #[test]
    fn check_grand_cross_table_argument_for_test_program_for_skiz() {
        check_grand_cross_table_argument(test_program_for_skiz())
    }

    #[test]
    fn check_grand_cross_table_argument_for_test_program_for_call_recurse_return() {
        check_grand_cross_table_argument(test_program_for_call_recurse_return())
    }

    #[test]
    fn check_grand_cross_table_argument_for_test_program_for_write_mem_read_mem() {
        check_grand_cross_table_argument(test_program_for_write_mem_read_mem())
    }

    #[test]
    fn check_grand_cross_table_argument_for_test_program_for_hash() {
        check_grand_cross_table_argument(test_program_for_hash())
    }

    #[test]
    fn check_grand_cross_table_argument_for_test_program_for_divine_sibling_no_switch() {
        check_grand_cross_table_argument(test_program_for_divine_sibling_no_switch())
    }

    #[test]
    fn check_grand_cross_table_argument_for_test_program_for_divine_sibling_switch() {
        check_grand_cross_table_argument(test_program_for_divine_sibling_switch())
    }

    #[test]
    fn check_grand_cross_table_argument_for_test_program_for_assert_vector() {
        check_grand_cross_table_argument(test_program_for_assert_vector())
    }

    #[test]
    fn check_grand_cross_table_argument_for_test_program_for_sponge_instructions() {
        check_grand_cross_table_argument(test_program_for_sponge_instructions())
    }

    #[test]
    fn check_grand_cross_table_argument_for_test_program_for_sponge_instructions_2() {
        check_grand_cross_table_argument(test_program_for_sponge_instructions_2())
    }

    #[test]
    fn check_grand_cross_table_argument_for_test_program_for_many_sponge_instructions() {
        check_grand_cross_table_argument(test_program_for_many_sponge_instructions())
    }

    #[test]
    fn check_grand_cross_table_argument_for_test_program_for_add_mul_invert() {
        check_grand_cross_table_argument(test_program_for_add_mul_invert())
    }

    #[test]
    fn check_grand_cross_table_argument_for_test_program_for_eq() {
        check_grand_cross_table_argument(test_program_for_eq())
    }

    #[test]
    fn check_grand_cross_table_argument_for_test_program_for_lsb() {
        check_grand_cross_table_argument(test_program_for_lsb())
    }

    #[test]
    fn check_grand_cross_table_argument_for_test_program_for_split() {
        check_grand_cross_table_argument(test_program_for_split())
    }

    #[test]
    fn check_grand_cross_table_argument_for_test_program_0_lt_0() {
        check_grand_cross_table_argument(test_program_0_lt_0())
    }

    #[test]
    fn check_grand_cross_table_argument_for_test_program_for_lt() {
        check_grand_cross_table_argument(test_program_for_lt())
    }

    #[test]
    fn check_grand_cross_table_argument_for_test_program_for_and() {
        check_grand_cross_table_argument(test_program_for_and())
    }

    #[test]
    fn check_grand_cross_table_argument_for_test_program_for_xor() {
        check_grand_cross_table_argument(test_program_for_xor())
    }

    #[test]
    fn check_grand_cross_table_argument_for_test_program_for_log2floor() {
        check_grand_cross_table_argument(test_program_for_log2floor())
    }

    #[test]
    fn check_grand_cross_table_argument_for_test_program_for_pow() {
        check_grand_cross_table_argument(test_program_for_pow())
    }

    #[test]
    fn check_grand_cross_table_argument_for_test_program_for_div_mod() {
        check_grand_cross_table_argument(test_program_for_div_mod())
    }

    #[test]
    fn check_grand_cross_table_argument_for_test_program_for_starting_with_pop_count() {
        check_grand_cross_table_argument(test_program_for_starting_with_pop_count())
    }

    #[test]
    fn check_grand_cross_table_argument_for_test_program_for_pop_count() {
        check_grand_cross_table_argument(test_program_for_pop_count())
    }

    #[test]
    fn check_grand_cross_table_argument_for_test_program_for_xxadd() {
        check_grand_cross_table_argument(test_program_for_xxadd())
    }

    #[test]
    fn check_grand_cross_table_argument_for_test_program_for_xxmul() {
        check_grand_cross_table_argument(test_program_for_xxmul())
    }

    #[test]
    fn check_grand_cross_table_argument_for_test_program_for_xinvert() {
        check_grand_cross_table_argument(test_program_for_xinvert())
    }

    #[test]
    fn check_grand_cross_table_argument_for_test_program_for_xbmul() {
        check_grand_cross_table_argument(test_program_for_xbmul())
    }

    #[test]
    fn check_grand_cross_table_argument_for_test_program_for_read_io_write_io() {
        check_grand_cross_table_argument(test_program_for_read_io_write_io())
    }

    #[test]
    fn check_grand_cross_table_argument_for_property_based_test_program_for_assert_vector() {
        check_grand_cross_table_argument(property_based_test_program_for_assert_vector())
    }

    #[test]
    fn check_grand_cross_table_argument_for_property_based_test_program_for_sponge_instructions() {
        check_grand_cross_table_argument(property_based_test_program_for_sponge_instructions())
    }

    #[test]
    fn check_grand_cross_table_argument_for_property_based_test_program_for_split() {
        check_grand_cross_table_argument(property_based_test_program_for_split())
    }

    #[test]
    fn check_grand_cross_table_argument_for_property_based_test_program_for_eq() {
        check_grand_cross_table_argument(property_based_test_program_for_eq())
    }

    #[test]
    fn check_grand_cross_table_argument_for_property_based_test_program_for_lsb() {
        check_grand_cross_table_argument(property_based_test_program_for_lsb())
    }

    #[test]
    fn check_grand_cross_table_argument_for_property_based_test_program_for_lt() {
        check_grand_cross_table_argument(property_based_test_program_for_lt())
    }

    #[test]
    fn check_grand_cross_table_argument_for_property_based_test_program_for_and() {
        check_grand_cross_table_argument(property_based_test_program_for_and())
    }

    #[test]
    fn check_grand_cross_table_argument_for_property_based_test_program_for_xor() {
        check_grand_cross_table_argument(property_based_test_program_for_xor())
    }

    #[test]
    fn check_grand_cross_table_argument_for_property_based_test_program_for_log2floor() {
        check_grand_cross_table_argument(property_based_test_program_for_log2floor())
    }

    #[test]
    fn check_grand_cross_table_argument_for_property_based_test_program_for_pow() {
        check_grand_cross_table_argument(property_based_test_program_for_pow())
    }

    #[test]
    fn check_grand_cross_table_argument_for_property_based_test_program_for_div_mod() {
        check_grand_cross_table_argument(property_based_test_program_for_div_mod())
    }

    #[test]
    fn check_grand_cross_table_argument_for_property_based_test_program_for_pop_count() {
        check_grand_cross_table_argument(property_based_test_program_for_pop_count())
    }

    #[test]
    fn check_grand_cross_table_argument_for_property_based_test_program_for_is_u32() {
        check_grand_cross_table_argument(property_based_test_program_for_is_u32())
    }

    #[test]
    fn check_grand_cross_table_argument_for_property_based_test_program_for_random_ram_access() {
        check_grand_cross_table_argument(property_based_test_program_for_random_ram_access())
    }

    fn check_grand_cross_table_argument(program_and_input: ProgramAndInput) {
        let zero = XFieldElement::zero();
        let circuit_builder = ConstraintCircuitBuilder::new();
        let terminal_constraints = GrandCrossTableArg::terminal_constraints(&circuit_builder);
        let terminal_constraints = terminal_constraints
            .into_iter()
            .map(|c| c.consume())
            .collect_vec();

        let (_, _, master_base_table, master_ext_table, challenges) =
            master_tables_for_low_security_level(program_and_input);

        let processor_table = master_ext_table.table(ProcessorTable);
        let processor_table_last_row = processor_table.slice(s![-1, ..]);
        check!(
            challenges[StandardInputTerminal]
                == processor_table_last_row[InputTableEvalArg.ext_table_index()],
        );
        check!(
            challenges[StandardOutputTerminal]
                == processor_table_last_row[OutputTableEvalArg.ext_table_index()],
        );

        let lookup_table = master_ext_table.table(LookupTable);
        let lookup_table_last_row = lookup_table.slice(s![-1, ..]);
        check!(
            challenges[LookupTablePublicTerminal]
                == lookup_table_last_row[PublicEvaluationArgument.ext_table_index()],
        );

        let master_base_trace_table = master_base_table.trace_table();
        let master_ext_trace_table = master_ext_table.trace_table();
        let last_master_base_row = master_base_trace_table.slice(s![-1.., ..]);
        let last_master_ext_row = master_ext_trace_table.slice(s![-1.., ..]);

        for (i, constraint) in terminal_constraints.iter().enumerate() {
            check!(
                zero == constraint.evaluate(last_master_base_row, last_master_ext_row, &challenges),
                "Terminal constraint {i} must evaluate to 0."
            );
        }
    }

    #[test]
    fn constraint_polynomials_use_right_number_of_variables() {
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
        let ch = Challenges::placeholder(None);
        let padded_height = 2;
        let num_trace_randomizers = 2;
        let interpolant_degree = interpolant_degree(padded_height, num_trace_randomizers);

        // Shorten some names for better formatting. This is just a test.
        let ph = padded_height;
        let id = interpolant_degree;
        let br = base_row.view();
        let er = ext_row.view();

        let num_init_quots = MasterExtTable::num_initial_quotients();
        let num_cons_quots = MasterExtTable::num_consistency_quotients();
        let num_tran_quots = MasterExtTable::num_transition_quotients();
        let num_term_quots = MasterExtTable::num_terminal_quotients();

        let eval_init_consts = MasterExtTable::evaluate_initial_constraints(br, er, &ch);
        let eval_cons_consts = MasterExtTable::evaluate_consistency_constraints(br, er, &ch);
        let eval_tran_consts = MasterExtTable::evaluate_transition_constraints(br, er, br, er, &ch);
        let eval_term_consts = MasterExtTable::evaluate_terminal_constraints(br, er, &ch);

        assert!(num_init_quots == eval_init_consts.len());
        assert!(num_cons_quots == eval_cons_consts.len());
        assert!(num_tran_quots == eval_tran_consts.len());
        assert!(num_term_quots == eval_term_consts.len());

        assert!(num_init_quots == MasterExtTable::initial_quotient_degree_bounds(id).len());
        assert!(num_cons_quots == MasterExtTable::consistency_quotient_degree_bounds(id, ph).len());
        assert!(num_tran_quots == MasterExtTable::transition_quotient_degree_bounds(id, ph).len());
        assert!(num_term_quots == MasterExtTable::terminal_quotient_degree_bounds(id).len());
    }

    #[test]
    fn triton_table_constraints_evaluate_to_zero_on_halt() {
        triton_constraints_evaluate_to_zero(test_program_for_halt());
    }

    #[test]
    fn triton_table_constraints_evaluate_to_zero_on_fibonacci() {
        let source_code_and_input = ProgramAndInput {
            program: FIBONACCI_SEQUENCE.clone(),
            public_input: vec![100],
            non_determinism: [].into(),
        };
        triton_constraints_evaluate_to_zero(source_code_and_input);
    }

    #[test]
    fn triton_table_constraints_evaluate_to_zero_on_big_mmr_snippet() {
        let source_code_and_input = ProgramAndInput::without_input(
            CALCULATE_NEW_MMR_PEAKS_FROM_APPEND_WITH_SAFE_LISTS.clone(),
        );
        triton_constraints_evaluate_to_zero(source_code_and_input);
    }

    #[test]
    fn constraints_evaluate_to_zero_on_program_for_halt() {
        triton_constraints_evaluate_to_zero(test_program_for_halt())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_program_hash_nop_nop_lt() {
        triton_constraints_evaluate_to_zero(test_program_hash_nop_nop_lt())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_program_for_push_pop_dup_swap_nop() {
        triton_constraints_evaluate_to_zero(test_program_for_push_pop_dup_swap_nop())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_program_for_divine() {
        triton_constraints_evaluate_to_zero(test_program_for_divine())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_program_for_skiz() {
        triton_constraints_evaluate_to_zero(test_program_for_skiz())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_program_for_call_recurse_return() {
        triton_constraints_evaluate_to_zero(test_program_for_call_recurse_return())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_program_for_write_mem_read_mem() {
        triton_constraints_evaluate_to_zero(test_program_for_write_mem_read_mem())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_program_for_hash() {
        triton_constraints_evaluate_to_zero(test_program_for_hash())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_program_for_divine_sibling_no_switch() {
        triton_constraints_evaluate_to_zero(test_program_for_divine_sibling_no_switch())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_program_for_divine_sibling_switch() {
        triton_constraints_evaluate_to_zero(test_program_for_divine_sibling_switch())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_program_for_assert_vector() {
        triton_constraints_evaluate_to_zero(test_program_for_assert_vector())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_program_for_sponge_instructions() {
        triton_constraints_evaluate_to_zero(test_program_for_sponge_instructions())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_program_for_sponge_instructions_2() {
        triton_constraints_evaluate_to_zero(test_program_for_sponge_instructions_2())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_program_for_many_sponge_instructions() {
        triton_constraints_evaluate_to_zero(test_program_for_many_sponge_instructions())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_program_for_add_mul_invert() {
        triton_constraints_evaluate_to_zero(test_program_for_add_mul_invert())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_program_for_eq() {
        triton_constraints_evaluate_to_zero(test_program_for_eq())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_program_for_lsb() {
        triton_constraints_evaluate_to_zero(test_program_for_lsb())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_program_for_split() {
        triton_constraints_evaluate_to_zero(test_program_for_split())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_program_0_lt_0() {
        triton_constraints_evaluate_to_zero(test_program_0_lt_0())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_program_for_lt() {
        triton_constraints_evaluate_to_zero(test_program_for_lt())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_program_for_and() {
        triton_constraints_evaluate_to_zero(test_program_for_and())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_program_for_xor() {
        triton_constraints_evaluate_to_zero(test_program_for_xor())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_program_for_log2floor() {
        triton_constraints_evaluate_to_zero(test_program_for_log2floor())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_program_for_pow() {
        triton_constraints_evaluate_to_zero(test_program_for_pow())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_program_for_div_mod() {
        triton_constraints_evaluate_to_zero(test_program_for_div_mod())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_program_for_starting_with_pop_count() {
        triton_constraints_evaluate_to_zero(test_program_for_starting_with_pop_count())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_program_for_pop_count() {
        triton_constraints_evaluate_to_zero(test_program_for_pop_count())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_program_for_xxadd() {
        triton_constraints_evaluate_to_zero(test_program_for_xxadd())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_program_for_xxmul() {
        triton_constraints_evaluate_to_zero(test_program_for_xxmul())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_program_for_xinvert() {
        triton_constraints_evaluate_to_zero(test_program_for_xinvert())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_program_for_xbmul() {
        triton_constraints_evaluate_to_zero(test_program_for_xbmul())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_program_for_read_io_write_io() {
        triton_constraints_evaluate_to_zero(test_program_for_read_io_write_io())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_property_based_test_program_for_assert_vector() {
        triton_constraints_evaluate_to_zero(property_based_test_program_for_assert_vector())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_property_based_test_program_for_sponge_instructions() {
        triton_constraints_evaluate_to_zero(property_based_test_program_for_sponge_instructions())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_property_based_test_program_for_split() {
        triton_constraints_evaluate_to_zero(property_based_test_program_for_split())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_property_based_test_program_for_eq() {
        triton_constraints_evaluate_to_zero(property_based_test_program_for_eq())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_property_based_test_program_for_lsb() {
        triton_constraints_evaluate_to_zero(property_based_test_program_for_lsb())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_property_based_test_program_for_lt() {
        triton_constraints_evaluate_to_zero(property_based_test_program_for_lt())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_property_based_test_program_for_and() {
        triton_constraints_evaluate_to_zero(property_based_test_program_for_and())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_property_based_test_program_for_xor() {
        triton_constraints_evaluate_to_zero(property_based_test_program_for_xor())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_property_based_test_program_for_log2floor() {
        triton_constraints_evaluate_to_zero(property_based_test_program_for_log2floor())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_property_based_test_program_for_pow() {
        triton_constraints_evaluate_to_zero(property_based_test_program_for_pow())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_property_based_test_program_for_div_mod() {
        triton_constraints_evaluate_to_zero(property_based_test_program_for_div_mod())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_property_based_test_program_for_pop_count() {
        triton_constraints_evaluate_to_zero(property_based_test_program_for_pop_count())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_property_based_test_program_for_is_u32() {
        triton_constraints_evaluate_to_zero(property_based_test_program_for_is_u32())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_property_based_test_program_for_random_ram_access() {
        triton_constraints_evaluate_to_zero(property_based_test_program_for_random_ram_access())
    }

    #[test]
    fn claim_in_ram_corresponds_to_currently_running_program() {
        triton_constraints_evaluate_to_zero(
            test_program_claim_in_ram_corresponds_to_currently_running_program(),
        );
    }

    fn triton_constraints_evaluate_to_zero(program_and_input: ProgramAndInput) {
        let (_, _, master_base_table, master_ext_table, challenges) =
            master_tables_for_low_security_level(program_and_input);

        let num_base_rows = master_base_table.randomized_trace_table().nrows();
        let num_ext_rows = master_ext_table.randomized_trace_table().nrows();
        assert!(num_base_rows == num_ext_rows);

        let mbt = master_base_table.trace_table();
        let met = master_ext_table.trace_table();
        assert!(mbt.nrows() == met.nrows());

        program_table::tests::check_constraints(mbt, met, &challenges);
        processor_table::tests::check_constraints(mbt, met, &challenges);
        op_stack_table::tests::check_constraints(mbt, met, &challenges);
        ram_table::tests::check_constraints(mbt, met, &challenges);
        jump_stack_table::tests::check_constraints(mbt, met, &challenges);
        hash_table::tests::check_constraints(mbt, met, &challenges);
        cascade_table::tests::check_constraints(mbt, met, &challenges);
        lookup_table::tests::check_constraints(mbt, met, &challenges);
        u32_table::tests::check_constraints(mbt, met, &challenges);
    }

    #[test]
    fn derived_constraints_evaluate_to_zero_on_halt() {
        derived_constraints_evaluate_to_zero(test_program_for_halt());
    }

    fn derived_constraints_evaluate_to_zero(program_and_input: ProgramAndInput) {
        let (_, _, master_base_table, master_ext_table, challenges) =
            master_tables_for_low_security_level(program_and_input);

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
            assert!(
                zero == evaluated_constraint,
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
                assert!(
                    zero == evaluated_constraint,
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
                assert!(
                    zero == evaluated_constraint,
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
            assert!(
                zero == evaluated_constraint,
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

        assert!(let Ok(()) = Stark::verify(parameters, &claim, &proof, &mut None));
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

        assert!(let Ok(()) = Stark::verify(parameters, &claim, &proof, &mut None));

        let_assert!(Ok(padded_height) = proof.padded_height());
        let fri = Stark::derive_fri(parameters, padded_height).unwrap();
        let report = profiler
            .report()
            .with_padded_height(padded_height)
            .with_fri_domain_len(fri.domain.length);
        println!("{report}");
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

        assert!(let Ok(()) = Stark::verify(parameters, &claim, &proof, &mut None));

        let_assert!(Ok(padded_height) = proof.padded_height());
        let fri = Stark::derive_fri(parameters, padded_height).unwrap();
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
            assert!(let Ok(()) = Stark::verify(parameters, &claim, &proof, &mut None));

            assert!(vec![fib_seq_val] == claim.public_output());
        }
    }

    #[test]
    fn constraints_evaluate_to_zero_on_many_u32_operations() {
        let many_u32_instructions =
            ProgramAndInput::without_input(PROGRAM_WITH_MANY_U32_INSTRUCTIONS.clone());
        triton_constraints_evaluate_to_zero(many_u32_instructions);
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
        assert!(let Ok(()) = Stark::verify(parameters, &claim, &proof, &mut None));

        let_assert!(Ok(padded_height) = proof.padded_height());
        let fri = Stark::derive_fri(parameters, padded_height).unwrap();
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

    #[proptest]
    fn negative_log_2_floor(
        #[strategy(arb())]
        #[filter(#st0.value() > u32::MAX as u64)]
        st0: BFieldElement,
    ) {
        let program = triton_program!(push {st0} log_2_floor halt);
        let_assert!(Err(err) = program.run([].into(), [].into()));
        let_assert!(InstructionError::FailedU32Conversion(element) = err.source);
        assert!(st0 == element);
    }

    #[test]
    fn negative_log_2_floor_of_0() {
        let program = triton_program!(push 0 log_2_floor halt);
        let_assert!(Err(err) = program.run([].into(), [].into()));
        let_assert!(InstructionError::LogarithmOfZero = err.source);
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
        assert!(poly_degree as isize - 2 == poly_of_maybe_low_degree.degree());

        let bogus_out_of_domain_value = thread_rng().gen();
        let bogus_deep_poly = Stark::deep_codeword(
            &low_deg_codeword,
            domain,
            out_of_domain_point,
            bogus_out_of_domain_value,
        );
        let poly_of_hopefully_high_degree = domain.interpolate(&bogus_deep_poly);
        assert!(domain_length as isize - 1 == poly_of_hopefully_high_degree.degree());
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
        assert!(f.evaluate(&x) == sum_of_evaluated_segments);
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
