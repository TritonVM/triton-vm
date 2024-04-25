use std::ops::Mul;

use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use itertools::izip;
use itertools::Itertools;
use ndarray::prelude::*;
use ndarray::Zip;
use rayon::prelude::*;
use serde::Deserialize;
use serde::Serialize;
use twenty_first::math::traits::FiniteField;
use twenty_first::prelude::*;

use crate::aet::AlgebraicExecutionTrace;
use crate::arithmetic_domain::ArithmeticDomain;
use crate::error::ProvingError;
use crate::error::VerificationError;
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
use crate::table::extension_table::Quotientable;
use crate::table::master_table::all_quotients_combined;
use crate::table::master_table::interpolant_degree;
use crate::table::master_table::max_degree_with_origin;
use crate::table::master_table::MasterBaseTable;
use crate::table::master_table::MasterExtTable;
use crate::table::master_table::MasterTable;
use crate::table::master_table::AIR_TARGET_DEGREE;
use crate::table::QuotientSegments;
use crate::table::NUM_BASE_COLUMNS;
use crate::table::NUM_EXT_COLUMNS;

/// The number of segments the quotient polynomial is split into.
/// Helps keeping the FRI domain small.
pub const NUM_QUOTIENT_SEGMENTS: usize = AIR_TARGET_DEGREE as usize;

/// The number of randomizer polynomials over the [extension field](XFieldElement) used in the
/// [`STARK`](Stark). Integral for achieving zero-knowledge in [FRI](Fri).
pub const NUM_RANDOMIZER_POLYNOMIALS: usize = 1;

const NUM_DEEP_CODEWORD_COMPONENTS: usize = 3;

/// The Zero-Knowledge [Scalable Transparent ARgument of Knowledge (STARK)][stark] for Triton VM.
///
/// [stark]: https://www.iacr.org/archive/crypto2019/116940201/116940201.pdf
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Serialize, Deserialize)]
pub struct Stark {
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

    /// The number of collinearity checks to perform in FRI.
    pub num_collinearity_checks: usize,

    /// The number of combination codeword checks. These checks link the (DEEP) ALI part and the
    /// FRI part of the zk-STARK. The number of combination codeword checks directly depends on the
    /// number of collinearity checks and the FRI folding factor.
    pub num_combination_codeword_checks: usize,
}

impl Stark {
    pub fn new(security_level: usize, log2_of_fri_expansion_factor: usize) -> Self {
        assert_ne!(
            0, log2_of_fri_expansion_factor,
            "FRI expansion factor must be greater than one."
        );

        let fri_expansion_factor = 1 << log2_of_fri_expansion_factor;
        let num_collinearity_checks = security_level / log2_of_fri_expansion_factor;

        // For now, the FRI folding factor is hardcoded in our zk-STARK.
        let fri_folding_factor = 2;
        let num_combination_codeword_checks = num_collinearity_checks * fri_folding_factor;

        let num_out_of_domain_rows = 2;
        let num_trace_randomizers = num_combination_codeword_checks
            + num_out_of_domain_rows * x_field_element::EXTENSION_DEGREE;

        Stark {
            security_level,
            fri_expansion_factor,
            num_trace_randomizers,
            num_collinearity_checks,
            num_combination_codeword_checks,
        }
    }

    pub fn prove(
        &self,
        claim: &Claim,
        aet: &AlgebraicExecutionTrace,
        maybe_profiler: &mut Option<TritonProfiler>,
    ) -> Result<Proof, ProvingError> {
        prof_start!(maybe_profiler, "Fiat-Shamir: claim", "hash");
        let mut proof_stream = ProofStream::new();
        proof_stream.alter_fiat_shamir_state_with(claim);
        prof_stop!(maybe_profiler, "Fiat-Shamir: claim");

        prof_start!(maybe_profiler, "derive additional parameters");
        let padded_height = aet.padded_height();
        let max_degree = self.derive_max_degree(padded_height);
        let fri = self.derive_fri(padded_height)?;
        let quotient_domain = Self::quotient_domain(fri.domain, max_degree)?;
        proof_stream.enqueue(ProofItem::Log2PaddedHeight(padded_height.ilog2()));
        prof_stop!(maybe_profiler, "derive additional parameters");

        prof_start!(maybe_profiler, "base tables");
        prof_start!(maybe_profiler, "create", "gen");
        let mut master_base_table =
            MasterBaseTable::new(aet, self.num_trace_randomizers, quotient_domain, fri.domain);
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
        let challenges = proof_stream.sample_scalars(Challenges::SAMPLE_COUNT);
        let challenges = Challenges::new(challenges, claim);
        prof_stop!(maybe_profiler, "Fiat-Shamir");

        prof_start!(maybe_profiler, "extend", "gen");
        let mut master_ext_table = master_base_table.extend(&challenges);
        prof_stop!(maybe_profiler, "extend");
        prof_stop!(maybe_profiler, "base tables");

        prof_start!(maybe_profiler, "ext tables");
        prof_start!(maybe_profiler, "randomize trace", "gen");
        master_ext_table.randomize_trace();
        prof_stop!(maybe_profiler, "randomize trace");

        prof_start!(maybe_profiler, "LDE", "LDE");
        master_ext_table.low_degree_extend_all_columns();
        prof_stop!(maybe_profiler, "LDE");

        prof_start!(maybe_profiler, "Merkle tree", "hash");
        let ext_merkle_tree = master_ext_table.merkle_tree(maybe_profiler);
        prof_stop!(maybe_profiler, "Merkle tree");

        prof_start!(maybe_profiler, "Fiat-Shamir", "hash");
        proof_stream.enqueue(ProofItem::MerkleRoot(ext_merkle_tree.root()));

        // Get the weights with which to compress the many quotients into one.
        let quotient_combination_weights =
            proof_stream.sample_scalars(MasterExtTable::NUM_CONSTRAINTS);
        prof_stop!(maybe_profiler, "Fiat-Shamir");
        prof_stop!(maybe_profiler, "ext tables");

        // low-degree extend the trace codewords so that all the quotient codewords
        // can be obtained by element-wise evaluation of the AIR constraints
        prof_start!(maybe_profiler, "quotient-domain codewords");
        let base_quotient_domain_codewords = master_base_table.quotient_domain_table();
        let ext_quotient_domain_codewords = master_ext_table.quotient_domain_table();
        prof_stop!(maybe_profiler, "quotient-domain codewords");

        prof_start!(
            maybe_profiler,
            "compute and combine quotient codewords",
            "CC"
        );
        let quotient_codeword = all_quotients_combined(
            base_quotient_domain_codewords,
            ext_quotient_domain_codewords,
            master_base_table.trace_domain(),
            quotient_domain,
            &challenges,
            &quotient_combination_weights,
            maybe_profiler,
        );
        let quotient_codeword = Array1::from(quotient_codeword);
        assert_eq!(quotient_domain.length, quotient_codeword.len());
        prof_stop!(maybe_profiler, "compute and combine quotient codewords");

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
            Tip5::hash_varlen(&row_as_bfes)
        };
        let quotient_segments_rows = fri_domain_quotient_segment_codewords
            .axis_iter(Axis(0))
            .into_par_iter();
        let fri_domain_quotient_segment_codewords_digests =
            quotient_segments_rows.map(hash_row).collect::<Vec<_>>();
        prof_stop!(maybe_profiler, "hash rows of quotient segments");
        prof_start!(maybe_profiler, "Merkle tree", "hash");
        let quot_merkle_tree: MerkleTree<Tip5> =
            CpuParallel::from_digests(&fri_domain_quotient_segment_codewords_digests)?;
        let quot_merkle_tree_root = quot_merkle_tree.root();
        proof_stream.enqueue(ProofItem::MerkleRoot(quot_merkle_tree_root));
        prof_stop!(maybe_profiler, "Merkle tree");
        prof_stop!(maybe_profiler, "commit to quotient codeword segments");
        debug_assert_eq!(fri.domain.length, quot_merkle_tree.num_leafs());

        prof_start!(maybe_profiler, "out-of-domain rows");
        let trace_domain_generator = master_base_table.trace_domain().generator;
        let out_of_domain_point_curr_row = proof_stream.sample_scalars(1)[0];
        let out_of_domain_point_next_row = trace_domain_generator * out_of_domain_point_curr_row;

        let ood_base_row = master_base_table.row(out_of_domain_point_curr_row);
        let ood_base_row = MasterBaseTable::try_to_base_row(ood_base_row)?;
        proof_stream.enqueue(ProofItem::OutOfDomainBaseRow(Box::new(ood_base_row)));

        let ood_ext_row = master_ext_table.row(out_of_domain_point_curr_row);
        let ood_ext_row = MasterExtTable::try_to_ext_row(ood_ext_row)?;
        proof_stream.enqueue(ProofItem::OutOfDomainExtRow(Box::new(ood_ext_row)));

        let ood_next_base_row = master_base_table.row(out_of_domain_point_next_row);
        let ood_next_base_row = MasterBaseTable::try_to_base_row(ood_next_base_row)?;
        proof_stream.enqueue(ProofItem::OutOfDomainBaseRow(Box::new(ood_next_base_row)));

        let ood_next_ext_row = master_ext_table.row(out_of_domain_point_next_row);
        let ood_next_ext_row = MasterExtTable::try_to_ext_row(ood_next_ext_row)?;
        proof_stream.enqueue(ProofItem::OutOfDomainExtRow(Box::new(ood_next_ext_row)));

        let out_of_domain_point_curr_row_pow_num_segments =
            out_of_domain_point_curr_row.mod_pow_u32(NUM_QUOTIENT_SEGMENTS as u32);
        let out_of_domain_curr_row_quot_segments = quotient_segment_polynomials
            .map(|poly| poly.evaluate(out_of_domain_point_curr_row_pow_num_segments))
            .to_vec()
            .try_into()
            .unwrap();
        proof_stream.enqueue(ProofItem::OutOfDomainQuotientSegments(
            out_of_domain_curr_row_quot_segments,
        ));
        prof_stop!(maybe_profiler, "out-of-domain rows");

        prof_start!(maybe_profiler, "Fiat-Shamir", "hash");
        let weights = LinearCombinationWeights::sample(&mut proof_stream);
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
            Self::random_linear_sum_base_field(short_domain_base_codewords, weights.base);
        prof_stop!(maybe_profiler, "base");
        prof_start!(maybe_profiler, "ext", "CC");
        let ext_codeword = Self::random_linear_sum(short_domain_ext_codewords, weights.ext);
        prof_stop!(maybe_profiler, "ext");
        let base_and_ext_codeword = base_codeword + ext_codeword;

        prof_start!(maybe_profiler, "quotient", "CC");
        let quotient_segments_codeword = Self::random_linear_sum(
            short_domain_quot_segment_codewords.view(),
            weights.quot_segments,
        );
        prof_stop!(maybe_profiler, "quotient");

        assert_eq!(short_domain.length, base_and_ext_codeword.len());
        assert_eq!(short_domain.length, quotient_segments_codeword.len());
        prof_stop!(maybe_profiler, "linear combination");

        prof_start!(maybe_profiler, "DEEP");
        // There are (at least) two possible ways to perform the DEEP update.
        // 1. The one used here, where base & ext codewords are DEEP'd twice: once with the out-of-
        //    domain point for the current row (i.e., α) and once using the out-of-domain point for
        //    the next row (i.e., ω·α). The DEEP update's denominator is a degree-1 polynomial in
        //    both cases, namely (ω^i - α) and (ω^i - ω·α) respectively.
        // 2. One where the base & ext codewords are DEEP'd only once, using the degree-2 polynomial
        //    (ω^i - α)·(ω^i - ω·α) as the denominator. This requires a linear interpolation in the
        //    numerator: b(ω^i) - i((b(α), α) + (b(ω·α), ω·α))(w^i).
        //
        // In either case, the DEEP'd quotient polynomial is an additional summand for the
        // combination codeword: (q(ω^i) - q(α)) / (ω^i - α).
        // All (three or two) summands are weighted and summed to form the combination codeword.
        // The weights are sampled through the Fiat-Shamir heuristic.
        //
        // Both approaches are sound. The first approach is more efficient, as it requires fewer
        // operations.
        prof_start!(maybe_profiler, "interpolate");
        let base_and_ext_interpolation_poly =
            short_domain.interpolate(&base_and_ext_codeword.to_vec());
        let quotient_segments_interpolation_poly =
            short_domain.interpolate(&quotient_segments_codeword.to_vec());
        prof_stop!(maybe_profiler, "interpolate");
        prof_start!(maybe_profiler, "base&ext curr row");
        let out_of_domain_curr_row_base_and_ext_value =
            base_and_ext_interpolation_poly.evaluate(out_of_domain_point_curr_row);
        let base_and_ext_curr_row_deep_codeword = Self::deep_codeword(
            &base_and_ext_codeword.to_vec(),
            short_domain,
            out_of_domain_point_curr_row,
            out_of_domain_curr_row_base_and_ext_value,
        );
        prof_stop!(maybe_profiler, "base&ext curr row");

        prof_start!(maybe_profiler, "base&ext next row");
        let out_of_domain_next_row_base_and_ext_value =
            base_and_ext_interpolation_poly.evaluate(out_of_domain_point_next_row);
        let base_and_ext_next_row_deep_codeword = Self::deep_codeword(
            &base_and_ext_codeword.to_vec(),
            short_domain,
            out_of_domain_point_next_row,
            out_of_domain_next_row_base_and_ext_value,
        );
        prof_stop!(maybe_profiler, "base&ext next row");

        prof_start!(maybe_profiler, "segmented quotient");
        let out_of_domain_curr_row_quot_segments_value = quotient_segments_interpolation_poly
            .evaluate(out_of_domain_point_curr_row_pow_num_segments);
        let quotient_segments_curr_row_deep_codeword = Self::deep_codeword(
            &quotient_segments_codeword.to_vec(),
            short_domain,
            out_of_domain_point_curr_row_pow_num_segments,
            out_of_domain_curr_row_quot_segments_value,
        );
        prof_stop!(maybe_profiler, "segmented quotient");
        prof_stop!(maybe_profiler, "DEEP");

        prof_start!(maybe_profiler, "combined DEEP polynomial");
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
        let weighted_deep_codeword_components = &deep_codeword_components * &weights.deep;
        let deep_codeword = weighted_deep_codeword_components.sum_axis(Axis(1));
        prof_stop!(maybe_profiler, "sum");
        let fri_combination_codeword = if fri_domain_is_short_domain {
            deep_codeword.to_vec()
        } else {
            prof_start!(maybe_profiler, "LDE", "LDE");
            let deep_codeword =
                quotient_domain.low_degree_extension(&deep_codeword.to_vec(), fri.domain);
            prof_stop!(maybe_profiler, "LDE");
            deep_codeword
        };
        assert_eq!(fri.domain.length, fri_combination_codeword.len());
        prof_stop!(maybe_profiler, "combined DEEP polynomial");

        prof_start!(maybe_profiler, "FRI");
        let revealed_current_row_indices =
            fri.prove(&fri_combination_codeword, &mut proof_stream)?;
        assert_eq!(
            self.num_combination_codeword_checks,
            revealed_current_row_indices.len()
        );
        prof_stop!(maybe_profiler, "FRI");

        prof_start!(maybe_profiler, "open trace leafs");
        // Open leafs of zipped codewords at indicated positions
        let revealed_base_elems = Self::get_revealed_elements(
            master_base_table.fri_domain_table(),
            &revealed_current_row_indices,
        )?;
        let base_authentication_structure =
            base_merkle_tree.authentication_structure(&revealed_current_row_indices)?;
        proof_stream.enqueue(ProofItem::MasterBaseTableRows(revealed_base_elems));
        proof_stream.enqueue(ProofItem::AuthenticationStructure(
            base_authentication_structure,
        ));

        let revealed_ext_elems = Self::get_revealed_elements(
            master_ext_table.fri_domain_table(),
            &revealed_current_row_indices,
        )?;
        let ext_authentication_structure =
            ext_merkle_tree.authentication_structure(&revealed_current_row_indices)?;
        proof_stream.enqueue(ProofItem::MasterExtTableRows(revealed_ext_elems));
        proof_stream.enqueue(ProofItem::AuthenticationStructure(
            ext_authentication_structure,
        ));

        // Open quotient & combination codewords at the same positions as base & ext codewords.
        let into_fixed_width_row =
            |row: ArrayView1<_>| -> QuotientSegments { row.to_vec().try_into().unwrap() };
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
                let random_linear_element = xfe!(random_linear_element_coefficients);
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
    pub(crate) fn quotient_domain(
        fri_domain: ArithmeticDomain,
        max_degree: isize,
    ) -> Result<ArithmeticDomain, ProvingError> {
        let max_degree = usize::try_from(max_degree).expect("AIR should constrain the VM");
        let domain_length = max_degree.next_power_of_two();
        Ok(ArithmeticDomain::of_length(domain_length)?.with_offset(fri_domain.offset))
    }

    /// Compute the upper bound to use for the maximum degree the quotients given the length of the
    /// trace and the number of trace randomizers.
    /// The degree of the quotients depends on the constraints, _i.e._, the AIR.
    pub fn derive_max_degree(&self, padded_height: usize) -> isize {
        let interpolant_degree = interpolant_degree(padded_height, self.num_trace_randomizers);
        let max_constraint_degree_with_origin =
            max_degree_with_origin(interpolant_degree, padded_height);
        let max_constraint_degree = max_constraint_degree_with_origin.degree as u64;
        let min_arithmetic_domain_length_supporting_max_constraint_degree =
            max_constraint_degree.next_power_of_two();
        let max_degree_supported_by_that_smallest_arithmetic_domain =
            min_arithmetic_domain_length_supporting_max_constraint_degree - 1;

        max_degree_supported_by_that_smallest_arithmetic_domain as isize
    }

    /// Compute the parameters for FRI. The length of the FRI domain, _i.e._, the number of
    /// elements in the FRI domain, has a major influence on proving time. It is influenced by the
    /// length of the execution trace and the FRI expansion factor, a security parameter.
    ///
    /// In principle, the FRI domain is also influenced by the AIR's degree
    /// (see [`AIR_TARGET_DEGREE`]). However, by segmenting the quotient polynomial into
    /// [`AIR_TARGET_DEGREE`]-many parts, that influence is mitigated.
    pub fn derive_fri(&self, padded_height: usize) -> fri::SetupResult<Fri<Tip5>> {
        let interpolant_degree = interpolant_degree(padded_height, self.num_trace_randomizers);
        let interpolant_codeword_length = interpolant_degree as usize + 1;
        let fri_domain_length = self.fri_expansion_factor * interpolant_codeword_length;
        let coset_offset = BFieldElement::generator();
        let domain = ArithmeticDomain::of_length(fri_domain_length)?.with_offset(coset_offset);

        Fri::new(
            domain,
            self.fri_expansion_factor,
            self.num_collinearity_checks,
        )
    }

    fn get_revealed_elements<const N: usize, FF: FiniteField>(
        fri_domain_table: ArrayView2<FF>,
        revealed_indices: &[usize],
    ) -> Result<Vec<[FF; N]>, ProvingError> {
        let err = || ProvingError::TableRowConversionError {
            expected_len: N,
            actual_len: fri_domain_table.ncols(),
        };
        let row = |&row_idx| {
            let row = fri_domain_table.row(row_idx).to_vec();
            row.try_into().map_err(|_| err())
        };

        revealed_indices.iter().map(row).collect()
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
        &self,
        claim: &Claim,
        proof: &Proof,
        maybe_profiler: &mut Option<TritonProfiler>,
    ) -> Result<(), VerificationError> {
        prof_start!(maybe_profiler, "deserialize");
        let mut proof_stream = ProofStream::try_from(proof)?;
        prof_stop!(maybe_profiler, "deserialize");

        prof_start!(maybe_profiler, "Fiat-Shamir: Claim", "hash");
        proof_stream.alter_fiat_shamir_state_with(claim);
        prof_stop!(maybe_profiler, "Fiat-Shamir: Claim");

        prof_start!(maybe_profiler, "derive additional parameters");
        let log_2_padded_height = proof_stream.dequeue()?.try_into_log2_padded_height()?;
        let padded_height = 1 << log_2_padded_height;
        let fri = self.derive_fri(padded_height)?;
        let merkle_tree_height = fri.domain.length.ilog2() as usize;
        prof_stop!(maybe_profiler, "derive additional parameters");

        prof_start!(maybe_profiler, "Fiat-Shamir 1", "hash");
        let base_merkle_tree_root = proof_stream.dequeue()?.try_into_merkle_root()?;
        let extension_challenge_weights = proof_stream.sample_scalars(Challenges::SAMPLE_COUNT);
        let challenges = Challenges::new(extension_challenge_weights, claim);
        let extension_tree_merkle_root = proof_stream.dequeue()?.try_into_merkle_root()?;
        // Sample weights for quotient codeword, which is a part of the combination codeword.
        // See corresponding part in the prover for a more detailed explanation.
        let quot_codeword_weights = proof_stream.sample_scalars(MasterExtTable::NUM_CONSTRAINTS);
        let quot_codeword_weights = Array1::from(quot_codeword_weights);
        let quotient_codeword_merkle_root = proof_stream.dequeue()?.try_into_merkle_root()?;
        prof_stop!(maybe_profiler, "Fiat-Shamir 1");

        prof_start!(maybe_profiler, "dequeue ood point and rows", "hash");
        let trace_domain_generator = ArithmeticDomain::generator_for_length(padded_height as u64)?;
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

        let out_of_domain_curr_base_row = Array1::from(out_of_domain_curr_base_row.to_vec());
        let out_of_domain_curr_ext_row = Array1::from(out_of_domain_curr_ext_row.to_vec());
        let out_of_domain_next_base_row = Array1::from(out_of_domain_next_base_row.to_vec());
        let out_of_domain_next_ext_row = Array1::from(out_of_domain_next_ext_row.to_vec());
        let out_of_domain_curr_row_quot_segments =
            Array1::from(out_of_domain_curr_row_quot_segments.to_vec());
        prof_stop!(maybe_profiler, "dequeue ood point and rows");

        prof_start!(maybe_profiler, "out-of-domain quotient element");
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

        prof_start!(maybe_profiler, "zerofiers");
        let initial_zerofier_inv = (out_of_domain_point_curr_row - bfe!(1)).inverse();
        let consistency_zerofier_inv =
            (out_of_domain_point_curr_row.mod_pow_u32(padded_height as u32) - bfe!(1)).inverse();
        let except_last_row = out_of_domain_point_curr_row - trace_domain_generator.inverse();
        let transition_zerofier_inv = except_last_row * consistency_zerofier_inv;
        let terminal_zerofier_inv = except_last_row.inverse(); // i.e., only last row
        prof_stop!(maybe_profiler, "zerofiers");

        prof_start!(maybe_profiler, "divide");
        let divide = |constraints: Vec<_>, z_inv| constraints.into_iter().map(move |c| c * z_inv);
        let initial_quotients = divide(evaluated_initial_constraints, initial_zerofier_inv);
        let consistency_quotients =
            divide(evaluated_consistency_constraints, consistency_zerofier_inv);
        let transition_quotients =
            divide(evaluated_transition_constraints, transition_zerofier_inv);
        let terminal_quotients = divide(evaluated_terminal_constraints, terminal_zerofier_inv);

        let quotient_summands = initial_quotients
            .chain(consistency_quotients)
            .chain(transition_quotients)
            .chain(terminal_quotients)
            .collect_vec();
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
            return Err(VerificationError::OutOfDomainQuotientValueMismatch);
        };
        prof_stop!(maybe_profiler, "verify quotient's segments");

        prof_start!(maybe_profiler, "Fiat-Shamir 2", "hash");
        let weights = LinearCombinationWeights::sample(&mut proof_stream);
        let base_and_ext_codeword_weights = weights.base_and_ext();
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
        let out_of_domain_curr_row_quotient_segment_value = weights
            .quot_segments
            .dot(&out_of_domain_curr_row_quot_segments);
        prof_stop!(maybe_profiler, "sum out-of-domain values");

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
            .map(|revealed_base_elem| Tip5::hash_varlen(revealed_base_elem))
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
            ..MerkleTreeInclusionProof::default()
        };
        if !base_merkle_tree_inclusion_proof.verify(base_merkle_tree_root) {
            return Err(VerificationError::BaseCodewordAuthenticationFailure);
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
                Tip5::hash_varlen(&b_values.collect_vec())
            })
            .collect::<Vec<_>>();
        prof_stop!(maybe_profiler, "dequeue extension elements");

        prof_start!(maybe_profiler, "Merkle verify (extension tree)", "hash");
        let ext_merkle_tree_inclusion_proof = MerkleTreeInclusionProof::<Tip5> {
            tree_height: merkle_tree_height,
            indexed_leaves: index_leaves(leaf_digests_ext),
            authentication_structure: ext_authentication_structure,
            ..MerkleTreeInclusionProof::default()
        };
        if !ext_merkle_tree_inclusion_proof.verify(extension_tree_merkle_root) {
            return Err(VerificationError::ExtensionCodewordAuthenticationFailure);
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
            ..MerkleTreeInclusionProof::default()
        };
        if !quot_merkle_tree_inclusion_proof.verify(quotient_codeword_merkle_root) {
            return Err(VerificationError::QuotientCodewordAuthenticationFailure);
        }
        prof_stop!(maybe_profiler, "Merkle verify (combined quotient)");
        prof_stop!(maybe_profiler, "check leafs");

        prof_start!(maybe_profiler, "linear combination");
        if self.num_combination_codeword_checks != revealed_current_row_indices.len() {
            return Err(VerificationError::IncorrectNumberOfRowIndices);
        };
        if self.num_combination_codeword_checks != revealed_fri_values.len() {
            return Err(VerificationError::IncorrectNumberOfFRIValues);
        };
        if self.num_combination_codeword_checks != revealed_quotient_segments_elements.len() {
            return Err(VerificationError::IncorrectNumberOfQuotientSegmentElements);
        };
        if self.num_combination_codeword_checks != base_table_rows.len() {
            return Err(VerificationError::IncorrectNumberOfBaseTableRows);
        };
        if self.num_combination_codeword_checks != ext_table_rows.len() {
            return Err(VerificationError::IncorrectNumberOfExtTableRows);
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
            let base_row = Array1::from(base_row.to_vec());
            let ext_row = Array1::from(ext_row.to_vec());
            let current_fri_domain_value = fri.domain.domain_value(row_idx as u32);

            prof_start!(maybe_profiler, "base & ext elements", "CC");
            let base_and_ext_curr_row_element = Self::linearly_sum_base_and_ext_row(
                base_row.view(),
                ext_row.view(),
                base_and_ext_codeword_weights.view(),
                maybe_profiler,
            );
            let quotient_segments_curr_row_element = weights
                .quot_segments
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
            if fri_value != weights.deep.dot(&deep_value_components) {
                return Err(VerificationError::CombinationCodewordMismatch);
            };
            prof_stop!(maybe_profiler, "combination codeword equality");
        }
        prof_stop!(maybe_profiler, "main loop");
        prof_stop!(maybe_profiler, "linear combination");
        Ok(())
    }

    fn hash_quotient_segment_elements(quotient_segment_rows: &[QuotientSegments]) -> Vec<Digest> {
        let interpret_xfe_as_bfes = |xfe: XFieldElement| xfe.coefficients.to_vec();
        let collect_row_as_bfes = |row: &QuotientSegments| row.map(interpret_xfe_as_bfes).concat();
        quotient_segment_rows
            .par_iter()
            .map(collect_row_as_bfes)
            .map(|row| Tip5::hash_varlen(&row))
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

impl Default for Stark {
    fn default() -> Self {
        let log_2_of_fri_expansion_factor = 2;
        let security_level = 160;

        Self::new(security_level, log_2_of_fri_expansion_factor)
    }
}

impl<'a> Arbitrary<'a> for Stark {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let security_level = u.int_in_range(1..=640)?;
        let log_2_of_fri_expansion_factor = u.int_in_range(1..=8)?;
        Ok(Self::new(security_level, log_2_of_fri_expansion_factor))
    }
}

/// Fiat-Shamir-sampled challenges to compress a row into a single
/// [extension field element][XFieldElement].
struct LinearCombinationWeights {
    /// of length [`NUM_BASE_COLUMNS`]
    base: Array1<XFieldElement>,

    /// of length [`NUM_EXT_COLUMNS`]
    ext: Array1<XFieldElement>,

    /// of length [`NUM_QUOTIENT_SEGMENTS`]
    quot_segments: Array1<XFieldElement>,

    /// of length [`NUM_DEEP_CODEWORD_COMPONENTS`]
    deep: Array1<XFieldElement>,
}

impl LinearCombinationWeights {
    const NUM: usize =
        NUM_BASE_COLUMNS + NUM_EXT_COLUMNS + NUM_QUOTIENT_SEGMENTS + NUM_DEEP_CODEWORD_COMPONENTS;

    fn sample(proof_stream: &mut ProofStream) -> Self {
        const BASE_END: usize = NUM_BASE_COLUMNS;
        const EXT_END: usize = BASE_END + NUM_EXT_COLUMNS;
        const QUOT_END: usize = EXT_END + NUM_QUOTIENT_SEGMENTS;

        let weights = proof_stream.sample_scalars(Self::NUM);

        Self {
            base: weights[..BASE_END].to_vec().into(),
            ext: weights[BASE_END..EXT_END].to_vec().into(),
            quot_segments: weights[EXT_END..QUOT_END].to_vec().into(),
            deep: weights[QUOT_END..].to_vec().into(),
        }
    }

    fn base_and_ext(&self) -> Array1<XFieldElement> {
        let base = self.base.clone().into_iter();
        base.chain(self.ext.clone()).collect()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::HashMap;

    use assert2::assert;
    use assert2::check;
    use assert2::let_assert;
    use itertools::izip;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use proptest_arbitrary_interop::arb;
    use rand::thread_rng;
    use rand::Rng;
    use strum::EnumCount;
    use test_strategy::proptest;
    use twenty_first::math::other::random_elements;
    use twenty_first::prelude::x_field_element::EXTENSION_DEGREE;

    use crate::error::InstructionError;
    use crate::example_programs::*;
    use crate::instruction::AnInstruction;
    use crate::instruction::Instruction;
    use crate::instruction::LabelledInstruction;
    use crate::op_stack::OpStackElement;
    use crate::prelude::Program;
    use crate::program::NonDeterminism;
    use crate::shared_tests::*;
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
    use crate::table::extension_table;
    use crate::table::extension_table::Evaluable;
    use crate::table::extension_table::Quotientable;
    use crate::table::hash_table::ExtHashTable;
    use crate::table::jump_stack_table::ExtJumpStackTable;
    use crate::table::lookup_table::ExtLookupTable;
    use crate::table::master_table::MasterExtTable;
    use crate::table::master_table::TableId;
    use crate::table::op_stack_table::ExtOpStackTable;
    use crate::table::processor_table::ExtProcessorTable;
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
    use crate::table::tasm_air_constraints::air_constraint_evaluation_tasm;
    use crate::table::u32_table::ExtU32Table;
    use crate::table::MemoryRegion;
    use crate::table::TasmConstraintEvaluationMemoryLayout;
    use crate::triton_instr;
    use crate::triton_program;
    use crate::vm::tests::*;
    use crate::vm::VMState;
    use crate::PublicInput;

    use super::*;

    pub(crate) fn master_base_table_for_low_security_level(
        program_and_input: ProgramAndInput,
    ) -> (Stark, Claim, MasterBaseTable) {
        let ProgramAndInput {
            program,
            public_input,
            non_determinism,
        } = program_and_input;

        let (aet, stdout) = program
            .trace_execution(public_input.clone(), non_determinism)
            .unwrap();
        let stark = low_security_stark();
        let claim = Claim::about_program(&aet.program)
            .with_input(public_input.individual_tokens)
            .with_output(stdout);
        let master_base_table = construct_master_base_table(stark, &aet);

        (stark, claim, master_base_table)
    }

    pub(crate) fn master_tables_for_low_security_level(
        program_and_input: ProgramAndInput,
    ) -> (Stark, Claim, MasterBaseTable, MasterExtTable, Challenges) {
        let (stark, claim, mut master_base_table) =
            master_base_table_for_low_security_level(program_and_input);

        let challenges = Challenges::placeholder(&claim);
        master_base_table.pad();
        let master_ext_table = master_base_table.extend(&challenges);

        (
            stark,
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
            master_tables_for_low_security_level(ProgramAndInput::new(program));

        println!();
        println!("Processor Table:\n");
        println!("| clk | ci  | nia | st0 | st1 | st2 | st3 | st4 | st5 |");
        println!("|----:|:----|:----|----:|----:|----:|----:|----:|----:|");
        for row in master_base_table
            .table(TableId::Processor)
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
            .table(TableId::Ram)
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
            master_base_table_for_low_security_level(ProgramAndInput::new(program));

        println!();
        println!("Processor Table:");
        println!("| clk | ci  | nia | st0 | st1 | st2 | st3 | underflow | pointer |");
        println!("|----:|:----|----:|----:|----:|----:|----:|:----------|--------:|");
        for row in master_base_table
            .table(TableId::Processor)
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
            .table(TableId::OpStack)
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
        let nia = curr_instruction
            .arg()
            .map(|_| next_instruction_or_arg.to_string())
            .unwrap_or_default();
        (curr_instruction.name().to_string(), nia)
    }

    /// To be used with `-- --nocapture`. Has mainly informative purpose.
    #[test]
    fn print_all_constraint_degrees() {
        let padded_height = 2;
        let num_trace_randomizers = 2;
        let interpolant_degree = interpolant_degree(padded_height, num_trace_randomizers);
        for deg in extension_table::all_degrees_with_origin(interpolant_degree, padded_height) {
            println!("{deg}");
        }
    }

    #[test]
    fn check_io_terminals() {
        let read_nop_program = triton_program!(
            read_io 3 nop nop write_io 2 push 17 write_io 1 halt
        );
        let mut program_and_input = ProgramAndInput::new(read_nop_program);
        program_and_input.public_input = PublicInput::from([3, 5, 7].map(|b| bfe!(b)));
        let (_, claim, _, master_ext_table, all_challenges) =
            master_tables_for_low_security_level(program_and_input);

        let processor_table = master_ext_table.table(TableId::Processor);
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
        let circuit_builder = ConstraintCircuitBuilder::new();
        let terminal_constraints = GrandCrossTableArg::terminal_constraints(&circuit_builder);
        let terminal_constraints = terminal_constraints
            .into_iter()
            .map(|c| c.consume())
            .collect_vec();

        let (_, _, master_base_table, master_ext_table, challenges) =
            master_tables_for_low_security_level(program_and_input);

        let processor_table = master_ext_table.table(TableId::Processor);
        let processor_table_last_row = processor_table.slice(s![-1, ..]);
        check!(
            challenges[StandardInputTerminal]
                == processor_table_last_row[InputTableEvalArg.ext_table_index()],
        );
        check!(
            challenges[StandardOutputTerminal]
                == processor_table_last_row[OutputTableEvalArg.ext_table_index()],
        );

        let lookup_table = master_ext_table.table(TableId::Lookup);
        let lookup_table_last_row = lookup_table.slice(s![-1, ..]);
        check!(
            challenges[LookupTablePublicTerminal]
                == lookup_table_last_row[PublicEvaluationArgument.ext_table_index()],
        );

        let master_base_trace_table = master_base_table.trace_table();
        let master_ext_trace_table = master_ext_table.trace_table();
        let last_master_base_row = master_base_trace_table.slice(s![-1.., ..]);
        let last_master_ext_row = master_ext_trace_table.slice(s![-1.., ..]);
        let challenges = challenges.challenges;

        for (i, constraint) in terminal_constraints.iter().enumerate() {
            let evaluation =
                constraint.evaluate(last_master_base_row, last_master_ext_row, &challenges);
            check!(
                xfe!(0) == evaluation,
                "Terminal constraint {i} must evaluate to 0."
            );
        }
    }

    #[test]
    fn constraint_polynomials_use_right_number_of_variables() {
        let challenges = Challenges::default();
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
        let ch = Challenges::default();
        let padded_height = 2;
        let num_trace_randomizers = 2;
        let interpolant_degree = interpolant_degree(padded_height, num_trace_randomizers);

        // Shorten some names for better formatting. This is just a test.
        let ph = padded_height;
        let id = interpolant_degree;
        let br = base_row.view();
        let er = ext_row.view();

        let num_init_quots = MasterExtTable::NUM_INITIAL_CONSTRAINTS;
        let num_cons_quots = MasterExtTable::NUM_CONSISTENCY_CONSTRAINTS;
        let num_tran_quots = MasterExtTable::NUM_TRANSITION_CONSTRAINTS;
        let num_term_quots = MasterExtTable::NUM_TERMINAL_CONSTRAINTS;

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
    fn constraints_evaluate_to_zero_on_halt() {
        triton_constraints_evaluate_to_zero(test_program_for_halt());
    }

    #[test]
    fn constraints_evaluate_to_zero_on_fibonacci() {
        let source_code_and_input =
            ProgramAndInput::new(FIBONACCI_SEQUENCE.clone()).with_input(bfe_array![100]);
        triton_constraints_evaluate_to_zero(source_code_and_input);
    }

    #[test]
    fn constraints_evaluate_to_zero_on_big_mmr_snippet() {
        let source_code_and_input =
            ProgramAndInput::new(CALCULATE_NEW_MMR_PEAKS_FROM_APPEND_WITH_SAFE_LISTS.clone());
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
    fn constraints_evaluate_to_zero_on_property_based_test_program_for_xxdotstep() {
        let program_and_input = property_based_test_program_for_xxdotstep();
        let mut vm_state = VMState::new(
            &program_and_input.program,
            Default::default(),
            Default::default(),
        );
        // invoke master_tables_for_low_security_level(..)
        let (stark, claim, master_base_table, master_ext_table, mut challenges) = {
            let (stark, claim, mut master_base_table) =
                master_base_table_for_low_security_level(program_and_input);

            let challenges = Challenges::placeholder(&claim);
            master_base_table.pad();
            let master_ext_table = master_base_table.extend(&challenges);

            (
                stark,
                claim,
                master_base_table,
                master_ext_table,
                challenges,
            )
        };
        challenges.challenges.iter_mut().for_each(|ch| {
            *ch = thread_rng().gen::<XFieldElement>();
        });

        let mbt = master_base_table.trace_table();
        let met = master_ext_table.trace_table();

        let clk_star = mbt.slice(s![1, 7]);
        let cjddiff_star = mbt.slice(s![1, 45]);
        let logder = met.slice(s![0, 13]);
        let logder_star = met.slice(s![1, 13]);
        let chall = challenges[11];
        let constraint = (logder_star[()] - logder[()]) * (chall - clk_star[()]) - cjddiff_star[()];
        println!(
            "mbt -- \nclk*: {clk_star}\ncjddiff*: {cjddiff_star}\nlog: {logder}\nlog*: {logder_star}\nchallenge: {chall}\n",
        );
        println!("constraint: {:?}", constraint);

        let builder = ConstraintCircuitBuilder::new();
        for (constraint_idx, constraint) in ExtProcessorTable::transition_constraints(&builder)
            .into_iter()
            .map(|constraint_monad| constraint_monad.consume())
            .enumerate()
        {
            for row_idx in 0..mbt.nrows() - 1 {
                let evaluated_constraint = constraint.evaluate(
                    mbt.slice(s![row_idx..=row_idx + 1, ..]),
                    met.slice(s![row_idx..=row_idx + 1, ..]),
                    &challenges.challenges,
                );
                assert_eq!(
                    evaluated_constraint,
                    xfe!(0),
                    "transition constraint {constraint_idx} failed on row {row_idx}"
                );
            }
        }
        // check_processor_table_constraints(mbt, met, &challenges);
        // triton_constraints_evaluate_to_zero(property_based_test_program_for_xxdotstep());
    }

    #[test]
    fn constraints_evaluate_to_zero_on_property_based_test_program_for_xbdotstep() {
        triton_constraints_evaluate_to_zero(property_based_test_program_for_xbdotstep());
    }

    #[test]
    fn claim_in_ram_corresponds_to_currently_running_program() {
        triton_constraints_evaluate_to_zero(
            test_program_claim_in_ram_corresponds_to_currently_running_program(),
        );
    }

    macro_rules! check_constraints_fn {
        (fn $fn_name:ident for $table:ident) => {
            fn $fn_name(
                master_base_trace_table: ArrayView2<BFieldElement>,
                master_ext_trace_table: ArrayView2<XFieldElement>,
                challenges: &Challenges,
            ) {
                assert!(master_base_trace_table.nrows() == master_ext_trace_table.nrows());
                let challenges = &challenges.challenges;

                let builder = ConstraintCircuitBuilder::new();
                for (constraint_idx, constraint) in $table::initial_constraints(&builder)
                    .into_iter()
                    .map(|constraint_monad| constraint_monad.consume())
                    .enumerate()
                {
                    let evaluated_constraint = constraint.evaluate(
                        master_base_trace_table.slice(s![..1, ..]),
                        master_ext_trace_table.slice(s![..1, ..]),
                        challenges,
                    );
                    check!(
                        xfe!(0) == evaluated_constraint,
                        "{}: Initial constraint {constraint_idx} failed.",
                        stringify!($table),
                    );
                }

                let builder = ConstraintCircuitBuilder::new();
                for (constraint_idx, constraint) in $table::consistency_constraints(&builder)
                    .into_iter()
                    .map(|constraint_monad| constraint_monad.consume())
                    .enumerate()
                {
                    for row_idx in 0..master_base_trace_table.nrows() {
                        let evaluated_constraint = constraint.evaluate(
                            master_base_trace_table.slice(s![row_idx..=row_idx, ..]),
                            master_ext_trace_table.slice(s![row_idx..=row_idx, ..]),
                            challenges,
                        );
                        check!(
                            xfe!(0) == evaluated_constraint,
                            "{}: Consistency constraint {constraint_idx} failed on row {row_idx}.",
                            stringify!($table),
                        );
                    }
                }

                let builder = ConstraintCircuitBuilder::new();
                for (constraint_idx, constraint) in $table::transition_constraints(&builder)
                    .into_iter()
                    .map(|constraint_monad| constraint_monad.consume())
                    .enumerate()
                {
                    for row_idx in 0..master_base_trace_table.nrows() - 1 {
                        let evaluated_constraint = constraint.evaluate(
                            master_base_trace_table.slice(s![row_idx..=row_idx + 1, ..]),
                            master_ext_trace_table.slice(s![row_idx..=row_idx + 1, ..]),
                            challenges,
                        );
                        check!(
                            xfe!(0) == evaluated_constraint,
                            "{}: Transition constraint {constraint_idx} failed on row {row_idx}.",
                            stringify!($table),
                        );
                    }
                }

                let builder = ConstraintCircuitBuilder::new();
                for (constraint_idx, constraint) in $table::terminal_constraints(&builder)
                    .into_iter()
                    .map(|constraint_monad| constraint_monad.consume())
                    .enumerate()
                {
                    let evaluated_constraint = constraint.evaluate(
                        master_base_trace_table.slice(s![-1.., ..]),
                        master_ext_trace_table.slice(s![-1.., ..]),
                        challenges,
                    );
                    check!(
                        xfe!(0) == evaluated_constraint,
                        "{}: Terminal constraint {constraint_idx} failed.",
                        stringify!($table),
                    );
                }
            }
        };
    }

    check_constraints_fn!(fn check_program_table_constraints for ExtProgramTable);
    check_constraints_fn!(fn check_processor_table_constraints for ExtProcessorTable);
    check_constraints_fn!(fn check_op_stack_table_constraints for ExtOpStackTable);
    check_constraints_fn!(fn check_ram_table_constraints for ExtRamTable);
    check_constraints_fn!(fn check_jump_stack_table_constraints for ExtJumpStackTable);
    check_constraints_fn!(fn check_hash_table_constraints for ExtHashTable);
    check_constraints_fn!(fn check_cascade_table_constraints for ExtCascadeTable);
    check_constraints_fn!(fn check_lookup_table_constraints for ExtLookupTable);
    check_constraints_fn!(fn check_u32_table_constraints for ExtU32Table);

    fn triton_constraints_evaluate_to_zero(program_and_input: ProgramAndInput) {
        let (_, _, master_base_table, master_ext_table, challenges) =
            master_tables_for_low_security_level(program_and_input);

        let num_base_rows = master_base_table.randomized_trace_table().nrows();
        let num_ext_rows = master_ext_table.randomized_trace_table().nrows();
        assert!(num_base_rows == num_ext_rows);

        let mbt = master_base_table.trace_table();
        let met = master_ext_table.trace_table();
        assert!(mbt.nrows() == met.nrows());

        check_program_table_constraints(mbt, met, &challenges);
        check_processor_table_constraints(mbt, met, &challenges);
        check_op_stack_table_constraints(mbt, met, &challenges);
        check_ram_table_constraints(mbt, met, &challenges);
        check_jump_stack_table_constraints(mbt, met, &challenges);
        check_hash_table_constraints(mbt, met, &challenges);
        check_cascade_table_constraints(mbt, met, &challenges);
        check_lookup_table_constraints(mbt, met, &challenges);
        check_u32_table_constraints(mbt, met, &challenges);
    }

    #[test]
    fn derived_constraints_evaluate_to_zero_on_halt() {
        derived_constraints_evaluate_to_zero(test_program_for_halt());
    }

    fn derived_constraints_evaluate_to_zero(program_and_input: ProgramAndInput) {
        let (_, _, master_base_table, master_ext_table, challenges) =
            master_tables_for_low_security_level(program_and_input);

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
                xfe!(0) == evaluated_constraint,
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
                    xfe!(0) == evaluated_constraint,
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
                    xfe!(0) == evaluated_constraint,
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
                xfe!(0) == evaluated_constraint,
                "Terminal constraint {constraint_idx} failed.",
            );
        }
    }

    #[test]
    fn prove_verify_simple_program() {
        let program_with_input = test_program_hash_nop_nop_lt();
        let (stark, claim, proof) = prove_with_low_security_level(
            &program_with_input.program,
            program_with_input.public_input(),
            program_with_input.non_determinism(),
            &mut None,
        );

        assert!(let Ok(()) = stark.verify(&claim, &proof, &mut None));
    }

    #[test]
    fn prove_verify_halt() {
        let code_with_input = test_program_for_halt();
        let mut profiler = Some(TritonProfiler::new("Prove Halt"));
        let (stark, claim, proof) = prove_with_low_security_level(
            &code_with_input.program,
            code_with_input.public_input(),
            code_with_input.non_determinism(),
            &mut profiler,
        );
        let mut profiler = profiler.unwrap();
        profiler.finish();

        assert!(let Ok(()) = stark.verify(&claim, &proof, &mut None));

        let_assert!(Ok(padded_height) = proof.padded_height());
        let fri = stark.derive_fri(padded_height).unwrap();
        let report = profiler
            .report()
            .with_padded_height(padded_height)
            .with_fri_domain_len(fri.domain.length);
        println!("{report}");
    }

    #[test]
    fn prove_verify_fibonacci_100() {
        let stdin = PublicInput::from(bfe_array![100]);
        let secret_in = NonDeterminism::default();

        let mut profiler = Some(TritonProfiler::new("Prove Fib 100"));
        let (stark, claim, proof) =
            prove_with_low_security_level(&FIBONACCI_SEQUENCE, stdin, secret_in, &mut profiler);
        let mut profiler = profiler.unwrap();
        profiler.finish();

        println!("between prove and verify");

        assert!(let Ok(()) = stark.verify(&claim, &proof, &mut None));

        let_assert!(Ok(padded_height) = proof.padded_height());
        let fri = stark.derive_fri(padded_height).unwrap();
        let report = profiler
            .report()
            .with_padded_height(padded_height)
            .with_fri_domain_len(fri.domain.length);
        println!("{report}");
    }

    #[test]
    fn prove_verify_fib_shootout() {
        for (fib_seq_idx, fib_seq_val) in [(0, 1), (7, 21), (11, 144)] {
            let stdin = PublicInput::from(bfe_array![fib_seq_idx]);
            let secret_in = NonDeterminism::default();
            let (stark, claim, proof) =
                prove_with_low_security_level(&FIBONACCI_SEQUENCE, stdin, secret_in, &mut None);
            assert!(let Ok(()) = stark.verify(&claim, &proof, &mut None));

            assert!(bfe_vec![fib_seq_val] == claim.output);
        }
    }

    #[test]
    fn constraints_evaluate_to_zero_on_many_u32_operations() {
        let many_u32_instructions =
            ProgramAndInput::new(PROGRAM_WITH_MANY_U32_INSTRUCTIONS.clone());
        triton_constraints_evaluate_to_zero(many_u32_instructions);
    }

    #[test]
    fn prove_verify_many_u32_operations() {
        let mut profiler = Some(TritonProfiler::new("Prove Many U32 Ops"));
        let (stark, claim, proof) = prove_with_low_security_level(
            &PROGRAM_WITH_MANY_U32_INSTRUCTIONS,
            [].into(),
            [].into(),
            &mut profiler,
        );
        let mut profiler = profiler.unwrap();
        profiler.finish();
        assert!(let Ok(()) = stark.verify(&claim, &proof, &mut None));

        let_assert!(Ok(padded_height) = proof.padded_height());
        let fri = stark.derive_fri(padded_height).unwrap();
        let report = profiler
            .report()
            .with_padded_height(padded_height)
            .with_fri_domain_len(fri.domain.length);
        println!("{report}");
    }

    #[proptest]
    fn verifying_arbitrary_proof_does_not_panic(
        #[strategy(arb())] stark: Stark,
        #[strategy(arb())] claim: Claim,
        #[strategy(arb())] proof: Proof,
    ) {
        let _ = stark.verify(&claim, &proof, &mut None);
    }

    #[proptest]
    fn negative_log_2_floor(
        #[strategy(arb())]
        #[filter(#st0.value() > u64::from(u32::MAX))]
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
        let domain = ArithmeticDomain::of_length(domain_length).unwrap();

        let poly_degree = thread_rng().gen_range(2..20);
        let low_deg_poly_coeffs: Vec<XFieldElement> = random_elements(poly_degree);
        let low_deg_poly = Polynomial::new(low_deg_poly_coeffs.clone());
        let low_deg_codeword = domain.evaluate(&low_deg_poly);

        let out_of_domain_point: XFieldElement = thread_rng().gen();
        let out_of_domain_value = low_deg_poly.evaluate(out_of_domain_point);

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
            segment.evaluate(x_pow_n) * x.mod_pow_u32(segment_idx as u32)
        };
        let evaluated_segments = segments.iter().enumerate().map(evaluate_segment);
        let sum_of_evaluated_segments = evaluated_segments.fold(FF::zero(), |acc, x| acc + x);
        assert!(f.evaluate(x) == sum_of_evaluated_segments);
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

    #[derive(Debug, Clone, test_strategy::Arbitrary)]
    struct ConstraintEvaluationPoint {
        #[strategy(vec(arb(), NUM_BASE_COLUMNS))]
        #[map(Array1::from)]
        curr_base_row: Array1<XFieldElement>,

        #[strategy(vec(arb(), NUM_EXT_COLUMNS))]
        #[map(Array1::from)]
        curr_ext_row: Array1<XFieldElement>,

        #[strategy(vec(arb(), NUM_BASE_COLUMNS))]
        #[map(Array1::from)]
        next_base_row: Array1<XFieldElement>,

        #[strategy(vec(arb(), NUM_EXT_COLUMNS))]
        #[map(Array1::from)]
        next_ext_row: Array1<XFieldElement>,

        #[strategy(arb())]
        challenges: Challenges,

        #[strategy(arb())]
        #[filter(#memory_layout.is_integral())]
        memory_layout: TasmConstraintEvaluationMemoryLayout,
    }

    impl ConstraintEvaluationPoint {
        fn evaluate_all_constraints_rust(&self) -> Vec<XFieldElement> {
            let init = MasterExtTable::evaluate_initial_constraints(
                self.curr_base_row.view(),
                self.curr_ext_row.view(),
                &self.challenges,
            );
            let cons = MasterExtTable::evaluate_consistency_constraints(
                self.curr_base_row.view(),
                self.curr_ext_row.view(),
                &self.challenges,
            );
            let tran = MasterExtTable::evaluate_transition_constraints(
                self.curr_base_row.view(),
                self.curr_ext_row.view(),
                self.next_base_row.view(),
                self.next_ext_row.view(),
                &self.challenges,
            );
            let term = MasterExtTable::evaluate_terminal_constraints(
                self.curr_base_row.view(),
                self.curr_ext_row.view(),
                &self.challenges,
            );

            [init, cons, tran, term].concat()
        }

        fn evaluate_all_constraints_tasm(&self) -> Vec<XFieldElement> {
            let program = self.tasm_constraint_evaluation_code();
            let mut vm_state = self.set_up_triton_vm_to_evaluate_constraints_in_tasm(&program);

            vm_state.run().unwrap();

            let output_list_ptr = vm_state.op_stack.pop().unwrap().value();
            let num_quotients = MasterExtTable::NUM_CONSTRAINTS;
            Self::read_xfe_list_at_address(vm_state.ram, output_list_ptr, num_quotients)
        }

        fn tasm_constraint_evaluation_code(&self) -> Program {
            let mut source_code = air_constraint_evaluation_tasm(self.memory_layout);
            source_code.push(triton_instr!(halt));
            Program::new(&source_code)
        }

        fn set_up_triton_vm_to_evaluate_constraints_in_tasm(&self, program: &Program) -> VMState {
            let curr_base_row_ptr = self.memory_layout.curr_base_row_ptr;
            let curr_ext_row_ptr = self.memory_layout.curr_ext_row_ptr;
            let next_base_row_ptr = self.memory_layout.next_base_row_ptr;
            let next_ext_row_ptr = self.memory_layout.next_ext_row_ptr;
            let challenges_ptr = self.memory_layout.challenges_ptr;

            let mut ram = HashMap::default();
            Self::extend_ram_at_address(&mut ram, self.curr_base_row.to_vec(), curr_base_row_ptr);
            Self::extend_ram_at_address(&mut ram, self.curr_ext_row.to_vec(), curr_ext_row_ptr);
            Self::extend_ram_at_address(&mut ram, self.next_base_row.to_vec(), next_base_row_ptr);
            Self::extend_ram_at_address(&mut ram, self.next_ext_row.to_vec(), next_ext_row_ptr);
            Self::extend_ram_at_address(&mut ram, self.challenges.challenges, challenges_ptr);
            let non_determinism = NonDeterminism::default().with_ram(ram);

            VMState::new(program, PublicInput::default(), non_determinism)
        }

        fn extend_ram_at_address(
            ram: &mut HashMap<BFieldElement, BFieldElement>,
            list: impl IntoIterator<Item = impl Into<XFieldElement>>,
            address: BFieldElement,
        ) {
            let list = list.into_iter().flat_map(|xfe| xfe.into().coefficients);
            let indexed_list = list.enumerate();
            let offset_address = |i| bfe!(i as u64) + address;
            let ram_extension = indexed_list.map(|(i, bfe)| (offset_address(i), bfe));
            ram.extend(ram_extension);
        }

        fn read_xfe_list_at_address(
            ram: HashMap<BFieldElement, BFieldElement>,
            address: u64,
            len: usize,
        ) -> Vec<XFieldElement> {
            let mem_region_end = address + (len * EXTENSION_DEGREE) as u64;
            (address..mem_region_end)
                .map(BFieldElement::new)
                .map(|i| ram[&i])
                .chunks(EXTENSION_DEGREE)
                .into_iter()
                .map(|c| XFieldElement::try_from(c.collect_vec()).unwrap())
                .collect()
        }
    }

    #[proptest]
    fn triton_constraints_and_assembly_constraints_agree(point: ConstraintEvaluationPoint) {
        let all_constraints_rust = point.evaluate_all_constraints_rust();
        let all_constraints_tasm = point.evaluate_all_constraints_tasm();
        prop_assert_eq!(all_constraints_rust, all_constraints_tasm);
    }

    #[proptest]
    fn triton_assembly_constraint_evaluator_does_not_write_outside_of_dedicated_memory_region(
        point: ConstraintEvaluationPoint,
    ) {
        let program = point.tasm_constraint_evaluation_code();
        let mut initial_state = point.set_up_triton_vm_to_evaluate_constraints_in_tasm(&program);
        let mut terminal_state = initial_state.clone();
        terminal_state.run().unwrap();

        let free_mem_page_ptr = point.memory_layout.free_mem_page_ptr;
        let mem_page_size = TasmConstraintEvaluationMemoryLayout::MEM_PAGE_SIZE;
        let mem_page = MemoryRegion::new(free_mem_page_ptr, mem_page_size);
        let not_in_mem_page = |addr: &_| !mem_page.contains_address(addr);

        initial_state.ram.retain(|k, _| not_in_mem_page(k));
        terminal_state.ram.retain(|k, _| not_in_mem_page(k));
        prop_assert_eq!(initial_state.ram, terminal_state.ram);
    }

    #[proptest]
    fn triton_assembly_constraint_evaluator_declares_no_labels(
        #[strategy(arb())] memory_layout: TasmConstraintEvaluationMemoryLayout,
    ) {
        for instruction in air_constraint_evaluation_tasm(memory_layout) {
            if let LabelledInstruction::Label(label) = instruction {
                return Err(TestCaseError::Fail(format!("Found label: {label}").into()));
            }
        }
    }

    #[proptest]
    fn triton_assembly_constraint_evaluator_is_straight_line_and_does_not_halt(
        #[strategy(arb())] memory_layout: TasmConstraintEvaluationMemoryLayout,
    ) {
        type I = AnInstruction<String>;
        let is_legal = |i| !matches!(i, I::Call(_) | I::Return | I::Recurse | I::Skiz | I::Halt);

        for instruction in air_constraint_evaluation_tasm(memory_layout) {
            if let LabelledInstruction::Instruction(instruction) = instruction {
                prop_assert!(is_legal(instruction));
            }
        }
    }

    #[proptest]
    fn linear_combination_weights_samples_correct_number_of_elements(
        #[strategy(arb())] mut proof_stream: ProofStream,
    ) {
        let weights = LinearCombinationWeights::sample(&mut proof_stream);

        prop_assert_eq!(NUM_BASE_COLUMNS, weights.base.len());
        prop_assert_eq!(NUM_EXT_COLUMNS, weights.ext.len());
        prop_assert_eq!(NUM_QUOTIENT_SEGMENTS, weights.quot_segments.len());
        prop_assert_eq!(NUM_DEEP_CODEWORD_COMPONENTS, weights.deep.len());
        prop_assert_eq!(
            NUM_BASE_COLUMNS + NUM_EXT_COLUMNS,
            weights.base_and_ext().len()
        );
    }
}
