use std::ops::Mul;

use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use itertools::izip;
use itertools::Itertools;
use ndarray::prelude::*;
use ndarray::Zip;
use num_traits::ConstZero;
use num_traits::Zero;
use rayon::prelude::*;
use serde::Deserialize;
use serde::Serialize;
use twenty_first::math::ntt::intt;
use twenty_first::math::traits::FiniteField;
use twenty_first::math::traits::PrimitiveRootOfUnity;
use twenty_first::prelude::*;

use crate::aet::AlgebraicExecutionTrace;
use crate::arithmetic_domain::ArithmeticDomain;
use crate::challenges::Challenges;
use crate::error::ProvingError;
use crate::error::VerificationError;
use crate::fri;
use crate::fri::Fri;
use crate::ndarray_helper::fast_zeros_column_major;
use crate::profiler::profiler;
use crate::proof::Claim;
use crate::proof::Proof;
use crate::proof_item::ProofItem;
use crate::proof_stream::ProofStream;
use crate::table::auxiliary_table::Evaluable;
use crate::table::master_table::all_quotients_combined;
use crate::table::master_table::interpolant_degree;
use crate::table::master_table::max_degree_with_origin;
use crate::table::master_table::randomized_trace_len;
use crate::table::master_table::MasterAuxTable;
use crate::table::master_table::MasterMainTable;
use crate::table::master_table::MasterTable;
use crate::table::QuotientSegments;

/// The number of segments the quotient polynomial is split into.
/// Helps keeping the FRI domain small.
pub const NUM_QUOTIENT_SEGMENTS: usize = air::TARGET_DEGREE as usize;

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
}

impl Stark {
    /// # Panics
    ///
    /// Panics if `log2_of_fri_expansion_factor` is zero.
    pub fn new(security_level: usize, log2_of_fri_expansion_factor: usize) -> Self {
        assert_ne!(
            0, log2_of_fri_expansion_factor,
            "FRI expansion factor must be greater than one."
        );

        let fri_expansion_factor = 1 << log2_of_fri_expansion_factor;
        let num_collinearity_checks = security_level / log2_of_fri_expansion_factor;

        let num_out_of_domain_rows = 2;
        let num_trace_randomizers = num_collinearity_checks
            + num_out_of_domain_rows * x_field_element::EXTENSION_DEGREE
            + NUM_QUOTIENT_SEGMENTS * x_field_element::EXTENSION_DEGREE;

        Stark {
            security_level,
            fri_expansion_factor,
            num_trace_randomizers,
            num_collinearity_checks,
        }
    }

    pub fn prove(
        &self,
        claim: &Claim,
        aet: &AlgebraicExecutionTrace,
    ) -> Result<Proof, ProvingError> {
        profiler!(start "Fiat-Shamir: claim" ("hash"));
        let mut proof_stream = ProofStream::new();
        proof_stream.alter_fiat_shamir_state_with(claim);
        profiler!(stop "Fiat-Shamir: claim");

        profiler!(start "derive additional parameters");
        let padded_height = aet.padded_height();
        let max_degree = self.derive_max_degree(padded_height);
        let fri = self.derive_fri(padded_height)?;
        let quotient_domain = Self::quotient_domain(fri.domain, max_degree)?;
        proof_stream.enqueue(ProofItem::Log2PaddedHeight(padded_height.ilog2()));
        profiler!(stop "derive additional parameters");

        profiler!(start "main tables");
        profiler!(start "create" ("gen"));
        let mut master_main_table =
            MasterMainTable::new(aet, self.num_trace_randomizers, quotient_domain, fri.domain);
        profiler!(stop "create");

        profiler!(start "pad" ("gen"));
        master_main_table.pad();
        profiler!(stop "pad");

        master_main_table.maybe_low_degree_extend_all_columns();

        profiler!(start "Merkle tree");
        let main_merkle_tree = master_main_table.merkle_tree();
        profiler!(stop "Merkle tree");

        profiler!(start "Fiat-Shamir" ("hash"));
        proof_stream.enqueue(ProofItem::MerkleRoot(main_merkle_tree.root()));
        let challenges = proof_stream.sample_scalars(Challenges::SAMPLE_COUNT);
        let challenges = Challenges::new(challenges, claim);
        profiler!(stop "Fiat-Shamir");

        profiler!(start "extend" ("gen"));
        let mut master_aux_table = master_main_table.extend(&challenges);
        profiler!(stop "extend");
        profiler!(stop "main tables");

        profiler!(start "aux tables");
        master_aux_table.maybe_low_degree_extend_all_columns();

        profiler!(start "Merkle tree");
        let aux_merkle_tree = master_aux_table.merkle_tree();
        profiler!(stop "Merkle tree");

        profiler!(start "Fiat-Shamir" ("hash"));
        proof_stream.enqueue(ProofItem::MerkleRoot(aux_merkle_tree.root()));

        // Get the weights with which to compress the many quotients into one.
        let quotient_combination_weights =
            proof_stream.sample_scalars(MasterAuxTable::NUM_CONSTRAINTS);
        profiler!(stop "Fiat-Shamir");
        profiler!(stop "aux tables");

        let (fri_domain_quotient_segment_codewords, quotient_segment_polynomials) =
            Self::compute_quotient_segments(
                &master_main_table,
                &master_aux_table,
                fri.domain,
                quotient_domain,
                &challenges,
                &quotient_combination_weights,
            );

        profiler!(start "hash rows of quotient segments" ("hash"));
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
        profiler!(stop "hash rows of quotient segments");
        profiler!(start "Merkle tree" ("hash"));
        let quot_merkle_tree =
            MerkleTree::new::<CpuParallel>(&fri_domain_quotient_segment_codewords_digests)?;
        let quot_merkle_tree_root = quot_merkle_tree.root();
        proof_stream.enqueue(ProofItem::MerkleRoot(quot_merkle_tree_root));
        profiler!(stop "Merkle tree");

        debug_assert_eq!(fri.domain.length, quot_merkle_tree.num_leafs());

        profiler!(start "out-of-domain rows");
        let trace_domain_generator = master_main_table.trace_domain().generator;
        let out_of_domain_point_curr_row = proof_stream.sample_scalars(1)[0];
        let out_of_domain_point_next_row = trace_domain_generator * out_of_domain_point_curr_row;

        let ood_main_row = master_main_table.out_of_domain_row(out_of_domain_point_curr_row);
        let ood_main_row = MasterMainTable::try_to_main_row(ood_main_row)?;
        proof_stream.enqueue(ProofItem::OutOfDomainMainRow(Box::new(ood_main_row)));

        let ood_aux_row = master_aux_table.out_of_domain_row(out_of_domain_point_curr_row);
        let ood_aux_row = MasterAuxTable::try_to_aux_row(ood_aux_row)?;
        proof_stream.enqueue(ProofItem::OutOfDomainAuxRow(Box::new(ood_aux_row)));

        let ood_next_main_row = master_main_table.out_of_domain_row(out_of_domain_point_next_row);
        let ood_next_main_row = MasterMainTable::try_to_main_row(ood_next_main_row)?;
        proof_stream.enqueue(ProofItem::OutOfDomainMainRow(Box::new(ood_next_main_row)));

        let ood_next_aux_row = master_aux_table.out_of_domain_row(out_of_domain_point_next_row);
        let ood_next_aux_row = MasterAuxTable::try_to_aux_row(ood_next_aux_row)?;
        proof_stream.enqueue(ProofItem::OutOfDomainAuxRow(Box::new(ood_next_aux_row)));

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
        profiler!(stop "out-of-domain rows");

        profiler!(start "Fiat-Shamir" ("hash"));
        let weights = LinearCombinationWeights::sample(&mut proof_stream);
        profiler!(stop "Fiat-Shamir");

        let fri_domain_is_short_domain = fri.domain.length <= quotient_domain.length;
        let short_domain = if fri_domain_is_short_domain {
            fri.domain
        } else {
            quotient_domain
        };

        profiler!(start "linear combination");
        profiler!(start "main" ("CC"));
        let main_combination_poly = master_main_table.weighted_sum_of_columns(weights.main);
        profiler!(stop "main");

        profiler!(start "aux" ("CC"));
        let aux_combination_poly = master_aux_table.weighted_sum_of_columns(weights.aux);
        profiler!(stop "aux");
        let main_and_aux_combination_polynomial = main_combination_poly + aux_combination_poly;
        let main_and_aux_codeword = short_domain.evaluate(&main_and_aux_combination_polynomial);

        profiler!(start "quotient" ("CC"));
        let quotient_segments_combination_polynomial = quotient_segment_polynomials
            .into_iter()
            .zip_eq(weights.quot_segments)
            .fold(Polynomial::zero(), |acc, (poly, w)| acc + poly * w);
        let quotient_segments_combination_codeword =
            short_domain.evaluate(&quotient_segments_combination_polynomial);
        profiler!(stop "quotient");

        profiler!(stop "linear combination");

        profiler!(start "DEEP");
        // There are (at least) two possible ways to perform the DEEP update.
        // 1. The one used here, where main & aux codewords are DEEP'd twice: once with the out-of-
        //    domain point for the current row (i.e., α) and once using the out-of-domain point for
        //    the next row (i.e., ω·α). The DEEP update's denominator is a degree-1 polynomial in
        //    both cases, namely (ω^i - α) and (ω^i - ω·α) respectively.
        // 2. One where the main & aux codewords are DEEP'd only once, using the degree-2 polynomial
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
        profiler!(start "main&aux curr row");
        let out_of_domain_curr_row_main_and_aux_value =
            main_and_aux_combination_polynomial.evaluate(out_of_domain_point_curr_row);
        let main_and_aux_curr_row_deep_codeword = Self::deep_codeword(
            &main_and_aux_codeword,
            short_domain,
            out_of_domain_point_curr_row,
            out_of_domain_curr_row_main_and_aux_value,
        );
        profiler!(stop "main&aux curr row");

        profiler!(start "main&aux next row");
        let out_of_domain_next_row_main_and_aux_value =
            main_and_aux_combination_polynomial.evaluate(out_of_domain_point_next_row);
        let main_and_aux_next_row_deep_codeword = Self::deep_codeword(
            &main_and_aux_codeword,
            short_domain,
            out_of_domain_point_next_row,
            out_of_domain_next_row_main_and_aux_value,
        );
        profiler!(stop "main&aux next row");

        profiler!(start "segmented quotient");
        let out_of_domain_curr_row_quot_segments_value = quotient_segments_combination_polynomial
            .evaluate(out_of_domain_point_curr_row_pow_num_segments);
        let quotient_segments_curr_row_deep_codeword = Self::deep_codeword(
            &quotient_segments_combination_codeword,
            short_domain,
            out_of_domain_point_curr_row_pow_num_segments,
            out_of_domain_curr_row_quot_segments_value,
        );
        profiler!(stop "segmented quotient");
        profiler!(stop "DEEP");

        profiler!(start "combined DEEP polynomial");
        profiler!(start "sum" ("CC"));
        let deep_codeword = [
            main_and_aux_curr_row_deep_codeword,
            main_and_aux_next_row_deep_codeword,
            quotient_segments_curr_row_deep_codeword,
        ]
        .into_par_iter()
        .zip_eq(weights.deep.as_slice().unwrap())
        .map(|(codeword, &weight)| codeword.into_par_iter().map(|c| c * weight).collect())
        .reduce(
            || vec![XFieldElement::ZERO; short_domain.length],
            |left, right| left.into_iter().zip(right).map(|(l, r)| l + r).collect(),
        );

        profiler!(stop "sum");
        let fri_combination_codeword = if fri_domain_is_short_domain {
            deep_codeword
        } else {
            profiler!(start "LDE" ("LDE"));
            let deep_codeword = quotient_domain.low_degree_extension(&deep_codeword, fri.domain);
            profiler!(stop "LDE");
            deep_codeword
        };
        assert_eq!(fri.domain.length, fri_combination_codeword.len());
        profiler!(stop "combined DEEP polynomial");

        profiler!(start "FRI");
        let revealed_current_row_indices =
            fri.prove(&fri_combination_codeword, &mut proof_stream)?;
        assert_eq!(
            self.num_collinearity_checks,
            revealed_current_row_indices.len()
        );
        profiler!(stop "FRI");

        profiler!(start "open trace leafs");
        // Open leafs of zipped codewords at indicated positions
        let main_row_err = |row: Vec<_>| ProvingError::TableRowConversionError {
            expected_len: MasterMainTable::NUM_COLUMNS,
            actual_len: row.len(),
        };
        let revealed_main_elems = master_main_table
            .reveal_rows(&revealed_current_row_indices)
            .into_iter()
            .map(|row| row.try_into().map_err(main_row_err))
            .try_collect()?;
        let base_authentication_structure =
            main_merkle_tree.authentication_structure(&revealed_current_row_indices)?;
        proof_stream.enqueue(ProofItem::MasterMainTableRows(revealed_main_elems));
        proof_stream.enqueue(ProofItem::AuthenticationStructure(
            base_authentication_structure,
        ));

        let aux_row_err = |row: Vec<_>| ProvingError::TableRowConversionError {
            expected_len: MasterAuxTable::NUM_COLUMNS,
            actual_len: row.len(),
        };
        let revealed_aux_elems = master_aux_table
            .reveal_rows(&revealed_current_row_indices)
            .into_iter()
            .map(|row| row.try_into().map_err(aux_row_err))
            .try_collect()?;
        let aux_authentication_structure =
            aux_merkle_tree.authentication_structure(&revealed_current_row_indices)?;
        proof_stream.enqueue(ProofItem::MasterAuxTableRows(revealed_aux_elems));
        proof_stream.enqueue(ProofItem::AuthenticationStructure(
            aux_authentication_structure,
        ));

        // Open quotient & combination codewords at the same positions as main & aux codewords.
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
        profiler!(stop "open trace leafs");

        Ok(proof_stream.into())
    }

    fn compute_quotient_segments(
        master_main_table: &MasterMainTable,
        master_aux_table: &MasterAuxTable,
        fri_domain: ArithmeticDomain,
        quotient_domain: ArithmeticDomain,
        challenges: &Challenges,
        quotient_combination_weights: &[XFieldElement],
    ) -> (Array2<XFieldElement>, Array1<Polynomial<XFieldElement>>) {
        let (Some(main_quotient_domain_codewords), Some(aux_quotient_domain_codewords)) = (
            master_main_table.quotient_domain_table(),
            master_aux_table.quotient_domain_table(),
        ) else {
            profiler!(start "quotient calculation (just-in-time)");
            profiler!(start "interpolate" ("LDE"));
            let randomized_main_columns = (0..MasterMainTable::NUM_COLUMNS)
                .into_par_iter()
                .map(|i| master_main_table.randomized_column_interpolant(i))
                .collect::<Vec<_>>()
                .into();
            let randomized_aux_columns = (0..MasterAuxTable::NUM_COLUMNS)
                .into_par_iter()
                .map(|i| master_aux_table.randomized_column_interpolant(i))
                .collect::<Vec<_>>()
                .into();
            profiler!(stop "interpolate");

            let (fri_domain_quotient_segment_codewords, quotient_segment_polynomials) =
                Self::compute_quotient_segments_with_jit_lde(
                    randomized_main_columns,
                    randomized_aux_columns,
                    master_main_table.trace_domain(),
                    master_main_table.randomized_trace_domain(),
                    master_main_table.fri_domain(),
                    challenges,
                    quotient_combination_weights,
                );
            profiler!(stop "quotient calculation (just-in-time)");

            return (
                fri_domain_quotient_segment_codewords,
                quotient_segment_polynomials,
            );
        };

        profiler!(start "quotient calculation (cached)" ("CC"));
        let quotient_codeword = all_quotients_combined(
            main_quotient_domain_codewords,
            aux_quotient_domain_codewords,
            master_main_table.trace_domain(),
            quotient_domain,
            challenges,
            quotient_combination_weights,
        );
        let quotient_codeword = Array1::from(quotient_codeword);
        assert_eq!(quotient_domain.length, quotient_codeword.len());
        profiler!(stop "quotient calculation (cached)");

        profiler!(start "quotient LDE" ("LDE"));
        let quotient_segment_polynomials =
            Self::interpolate_quotient_segments(quotient_codeword, quotient_domain);
        let fri_domain_quotient_segment_codewords =
            Self::fri_domain_segment_polynomials(quotient_segment_polynomials.view(), fri_domain);
        profiler!(stop "quotient LDE");

        (
            fri_domain_quotient_segment_codewords,
            quotient_segment_polynomials,
        )
    }

    fn fri_domain_segment_polynomials(
        quotient_segment_polynomials: ArrayView1<Polynomial<XFieldElement>>,
        fri_domain: ArithmeticDomain,
    ) -> Array2<XFieldElement> {
        let fri_domain_codewords: Vec<_> = quotient_segment_polynomials
            .into_par_iter()
            .flat_map(|segment| fri_domain.evaluate(segment))
            .collect();
        Array2::from_shape_vec(
            [fri_domain.length, NUM_QUOTIENT_SEGMENTS].f(),
            fri_domain_codewords,
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
    /// (see [`air::TARGET_DEGREE`]). However, by segmenting the quotient polynomial into
    /// `TARGET_DEGREE`-many parts, that influence is mitigated.
    pub fn derive_fri(&self, padded_height: usize) -> fri::SetupResult<Fri> {
        let fri_domain_length = self.fri_expansion_factor
            * randomized_trace_len(padded_height, self.num_trace_randomizers);
        let coset_offset = BFieldElement::generator();
        let domain = ArithmeticDomain::of_length(fri_domain_length)?.with_offset(coset_offset);

        Fri::new(
            domain,
            self.fri_expansion_factor,
            self.num_collinearity_checks,
        )
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
            .zip_eq(codeword)
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

    pub fn verify(&self, claim: &Claim, proof: &Proof) -> Result<(), VerificationError> {
        profiler!(start "deserialize");
        let mut proof_stream = ProofStream::try_from(proof)?;
        profiler!(stop "deserialize");

        profiler!(start "Fiat-Shamir: Claim" ("hash"));
        proof_stream.alter_fiat_shamir_state_with(claim);
        profiler!(stop "Fiat-Shamir: Claim");

        profiler!(start "derive additional parameters");
        let log_2_padded_height = proof_stream.dequeue()?.try_into_log2_padded_height()?;
        let padded_height = 1 << log_2_padded_height;
        let fri = self.derive_fri(padded_height)?;
        let merkle_tree_height = fri.domain.length.ilog2() as usize;
        profiler!(stop "derive additional parameters");

        profiler!(start "Fiat-Shamir 1" ("hash"));
        let main_merkle_tree_root = proof_stream.dequeue()?.try_into_merkle_root()?;
        let extension_challenge_weights = proof_stream.sample_scalars(Challenges::SAMPLE_COUNT);
        let challenges = Challenges::new(extension_challenge_weights, claim);
        let auxiliary_tree_merkle_root = proof_stream.dequeue()?.try_into_merkle_root()?;
        // Sample weights for quotient codeword, which is a part of the combination codeword.
        // See corresponding part in the prover for a more detailed explanation.
        let quot_codeword_weights = proof_stream.sample_scalars(MasterAuxTable::NUM_CONSTRAINTS);
        let quot_codeword_weights = Array1::from(quot_codeword_weights);
        let quotient_codeword_merkle_root = proof_stream.dequeue()?.try_into_merkle_root()?;
        profiler!(stop "Fiat-Shamir 1");

        profiler!(start "dequeue ood point and rows" ("hash"));
        let trace_domain_generator = ArithmeticDomain::generator_for_length(padded_height as u64)?;
        let out_of_domain_point_curr_row = proof_stream.sample_scalars(1)[0];
        let out_of_domain_point_next_row = trace_domain_generator * out_of_domain_point_curr_row;
        let out_of_domain_point_curr_row_pow_num_segments =
            out_of_domain_point_curr_row.mod_pow_u32(NUM_QUOTIENT_SEGMENTS as u32);

        let out_of_domain_curr_main_row =
            proof_stream.dequeue()?.try_into_out_of_domain_main_row()?;
        let out_of_domain_curr_aux_row =
            proof_stream.dequeue()?.try_into_out_of_domain_aux_row()?;
        let out_of_domain_next_main_row =
            proof_stream.dequeue()?.try_into_out_of_domain_main_row()?;
        let out_of_domain_next_aux_row =
            proof_stream.dequeue()?.try_into_out_of_domain_aux_row()?;
        let out_of_domain_curr_row_quot_segments = proof_stream
            .dequeue()?
            .try_into_out_of_domain_quot_segments()?;

        let out_of_domain_curr_main_row = Array1::from(out_of_domain_curr_main_row.to_vec());
        let out_of_domain_curr_aux_row = Array1::from(out_of_domain_curr_aux_row.to_vec());
        let out_of_domain_next_main_row = Array1::from(out_of_domain_next_main_row.to_vec());
        let out_of_domain_next_aux_row = Array1::from(out_of_domain_next_aux_row.to_vec());
        let out_of_domain_curr_row_quot_segments =
            Array1::from(out_of_domain_curr_row_quot_segments.to_vec());
        profiler!(stop "dequeue ood point and rows");

        profiler!(start "out-of-domain quotient element");
        profiler!(start "evaluate AIR" ("AIR"));
        let evaluated_initial_constraints = MasterAuxTable::evaluate_initial_constraints(
            out_of_domain_curr_main_row.view(),
            out_of_domain_curr_aux_row.view(),
            &challenges,
        );
        let evaluated_consistency_constraints = MasterAuxTable::evaluate_consistency_constraints(
            out_of_domain_curr_main_row.view(),
            out_of_domain_curr_aux_row.view(),
            &challenges,
        );
        let evaluated_transition_constraints = MasterAuxTable::evaluate_transition_constraints(
            out_of_domain_curr_main_row.view(),
            out_of_domain_curr_aux_row.view(),
            out_of_domain_next_main_row.view(),
            out_of_domain_next_aux_row.view(),
            &challenges,
        );
        let evaluated_terminal_constraints = MasterAuxTable::evaluate_terminal_constraints(
            out_of_domain_curr_main_row.view(),
            out_of_domain_curr_aux_row.view(),
            &challenges,
        );
        profiler!(stop "evaluate AIR");

        profiler!(start "zerofiers");
        let initial_zerofier_inv = (out_of_domain_point_curr_row - bfe!(1)).inverse();
        let consistency_zerofier_inv =
            (out_of_domain_point_curr_row.mod_pow_u32(padded_height as u32) - bfe!(1)).inverse();
        let except_last_row = out_of_domain_point_curr_row - trace_domain_generator.inverse();
        let transition_zerofier_inv = except_last_row * consistency_zerofier_inv;
        let terminal_zerofier_inv = except_last_row.inverse(); // i.e., only last row
        profiler!(stop "zerofiers");

        profiler!(start "divide");
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
        profiler!(stop "divide");

        profiler!(start "inner product" ("CC"));
        let out_of_domain_quotient_value =
            quot_codeword_weights.dot(&Array1::from(quotient_summands));
        profiler!(stop "inner product");
        profiler!(stop "out-of-domain quotient element");

        profiler!(start "verify quotient's segments");
        let powers_of_out_of_domain_point_curr_row = (0..NUM_QUOTIENT_SEGMENTS as u32)
            .map(|exponent| out_of_domain_point_curr_row.mod_pow_u32(exponent))
            .collect::<Array1<_>>();
        let sum_of_evaluated_out_of_domain_quotient_segments =
            powers_of_out_of_domain_point_curr_row.dot(&out_of_domain_curr_row_quot_segments);
        if out_of_domain_quotient_value != sum_of_evaluated_out_of_domain_quotient_segments {
            return Err(VerificationError::OutOfDomainQuotientValueMismatch);
        };
        profiler!(stop "verify quotient's segments");

        profiler!(start "Fiat-Shamir 2" ("hash"));
        let weights = LinearCombinationWeights::sample(&mut proof_stream);
        let main_and_aux_codeword_weights = weights.main_and_aux();
        profiler!(stop "Fiat-Shamir 2");

        profiler!(start "sum out-of-domain values" ("CC"));
        let out_of_domain_curr_row_main_and_aux_value = Self::linearly_sum_main_and_aux_row(
            out_of_domain_curr_main_row.view(),
            out_of_domain_curr_aux_row.view(),
            main_and_aux_codeword_weights.view(),
        );
        let out_of_domain_next_row_main_and_aux_value = Self::linearly_sum_main_and_aux_row(
            out_of_domain_next_main_row.view(),
            out_of_domain_next_aux_row.view(),
            main_and_aux_codeword_weights.view(),
        );
        let out_of_domain_curr_row_quotient_segment_value = weights
            .quot_segments
            .dot(&out_of_domain_curr_row_quot_segments);
        profiler!(stop "sum out-of-domain values");

        // verify low degree of combination polynomial with FRI
        profiler!(start "FRI");
        let revealed_fri_indices_and_elements = fri.verify(&mut proof_stream)?;
        let (revealed_current_row_indices, revealed_fri_values): (Vec<_>, Vec<_>) =
            revealed_fri_indices_and_elements.into_iter().unzip();
        profiler!(stop "FRI");

        profiler!(start "check leafs");
        profiler!(start "dequeue main elements");
        let main_table_rows = proof_stream.dequeue()?.try_into_master_main_table_rows()?;
        let main_authentication_structure = proof_stream
            .dequeue()?
            .try_into_authentication_structure()?;
        let leaf_digests_main: Vec<_> = main_table_rows
            .par_iter()
            .map(|revealed_main_elem| Tip5::hash_varlen(revealed_main_elem))
            .collect();
        profiler!(stop "dequeue main elements");

        let index_leaves = |leaves| {
            let index_iter = revealed_current_row_indices.iter().copied();
            index_iter.zip_eq(leaves).collect()
        };
        profiler!(start "Merkle verify (main tree)" ("hash"));
        let base_merkle_tree_inclusion_proof = MerkleTreeInclusionProof {
            tree_height: merkle_tree_height,
            indexed_leafs: index_leaves(leaf_digests_main),
            authentication_structure: main_authentication_structure,
        };
        if !base_merkle_tree_inclusion_proof.verify(main_merkle_tree_root) {
            return Err(VerificationError::MainCodewordAuthenticationFailure);
        }
        profiler!(stop "Merkle verify (main tree)");

        profiler!(start "dequeue auxiliary elements");
        let aux_table_rows = proof_stream.dequeue()?.try_into_master_aux_table_rows()?;
        let aux_authentication_structure = proof_stream
            .dequeue()?
            .try_into_authentication_structure()?;
        let leaf_digests_aux = aux_table_rows
            .par_iter()
            .map(|xvalues| {
                let b_values = xvalues.iter().flat_map(|xfe| xfe.coefficients.to_vec());
                Tip5::hash_varlen(&b_values.collect_vec())
            })
            .collect::<Vec<_>>();
        profiler!(stop "dequeue auxiliary elements");

        profiler!(start "Merkle verify (auxiliary tree)" ("hash"));
        let aux_merkle_tree_inclusion_proof = MerkleTreeInclusionProof {
            tree_height: merkle_tree_height,
            indexed_leafs: index_leaves(leaf_digests_aux),
            authentication_structure: aux_authentication_structure,
        };
        if !aux_merkle_tree_inclusion_proof.verify(auxiliary_tree_merkle_root) {
            return Err(VerificationError::AuxiliaryCodewordAuthenticationFailure);
        }
        profiler!(stop "Merkle verify (auxiliary tree)");

        profiler!(start "dequeue quotient segments' elements");
        let revealed_quotient_segments_elements =
            proof_stream.dequeue()?.try_into_quot_segments_elements()?;
        let revealed_quotient_segments_digests =
            Self::hash_quotient_segment_elements(&revealed_quotient_segments_elements);
        let revealed_quotient_authentication_structure = proof_stream
            .dequeue()?
            .try_into_authentication_structure()?;
        profiler!(stop "dequeue quotient segments' elements");

        profiler!(start "Merkle verify (combined quotient)" ("hash"));
        let quot_merkle_tree_inclusion_proof = MerkleTreeInclusionProof {
            tree_height: merkle_tree_height,
            indexed_leafs: index_leaves(revealed_quotient_segments_digests),
            authentication_structure: revealed_quotient_authentication_structure,
        };
        if !quot_merkle_tree_inclusion_proof.verify(quotient_codeword_merkle_root) {
            return Err(VerificationError::QuotientCodewordAuthenticationFailure);
        }
        profiler!(stop "Merkle verify (combined quotient)");
        profiler!(stop "check leafs");

        profiler!(start "linear combination");
        if self.num_collinearity_checks != revealed_current_row_indices.len() {
            return Err(VerificationError::IncorrectNumberOfRowIndices);
        };
        if self.num_collinearity_checks != revealed_fri_values.len() {
            return Err(VerificationError::IncorrectNumberOfFRIValues);
        };
        if self.num_collinearity_checks != revealed_quotient_segments_elements.len() {
            return Err(VerificationError::IncorrectNumberOfQuotientSegmentElements);
        };
        if self.num_collinearity_checks != main_table_rows.len() {
            return Err(VerificationError::IncorrectNumberOfMainTableRows);
        };
        if self.num_collinearity_checks != aux_table_rows.len() {
            return Err(VerificationError::IncorrectNumberOfAuxTableRows);
        };

        for (row_idx, main_row, aux_row, quotient_segments_elements, fri_value) in izip!(
            revealed_current_row_indices,
            main_table_rows,
            aux_table_rows,
            revealed_quotient_segments_elements,
            revealed_fri_values,
        ) {
            let main_row = Array1::from(main_row.to_vec());
            let aux_row = Array1::from(aux_row.to_vec());
            let current_fri_domain_value = fri.domain.domain_value(row_idx as u32);

            profiler!(start "main & aux elements" ("CC"));
            let main_and_aux_curr_row_element = Self::linearly_sum_main_and_aux_row(
                main_row.view(),
                aux_row.view(),
                main_and_aux_codeword_weights.view(),
            );
            let quotient_segments_curr_row_element = weights
                .quot_segments
                .dot(&Array1::from(quotient_segments_elements.to_vec()));
            profiler!(stop "main & aux elements");

            profiler!(start "DEEP update");
            let main_and_aux_curr_row_deep_value = Self::deep_update(
                current_fri_domain_value,
                main_and_aux_curr_row_element,
                out_of_domain_point_curr_row,
                out_of_domain_curr_row_main_and_aux_value,
            );
            let main_and_aux_next_row_deep_value = Self::deep_update(
                current_fri_domain_value,
                main_and_aux_curr_row_element,
                out_of_domain_point_next_row,
                out_of_domain_next_row_main_and_aux_value,
            );
            let quot_curr_row_deep_value = Self::deep_update(
                current_fri_domain_value,
                quotient_segments_curr_row_element,
                out_of_domain_point_curr_row_pow_num_segments,
                out_of_domain_curr_row_quotient_segment_value,
            );
            profiler!(stop "DEEP update");

            profiler!(start "combination codeword equality");
            let deep_value_components = Array1::from(vec![
                main_and_aux_curr_row_deep_value,
                main_and_aux_next_row_deep_value,
                quot_curr_row_deep_value,
            ]);
            if fri_value != weights.deep.dot(&deep_value_components) {
                return Err(VerificationError::CombinationCodewordMismatch);
            };
            profiler!(stop "combination codeword equality");
        }
        profiler!(stop "linear combination");
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

    fn linearly_sum_main_and_aux_row<FF>(
        main_row: ArrayView1<FF>,
        aux_row: ArrayView1<XFieldElement>,
        weights: ArrayView1<XFieldElement>,
    ) -> XFieldElement
    where
        FF: FiniteField + Into<XFieldElement>,
        XFieldElement: Mul<FF, Output = XFieldElement>,
    {
        profiler!(start "collect");
        let mut row = main_row.map(|&element| element.into());
        row.append(Axis(0), aux_row).unwrap();
        profiler!(stop "collect");
        profiler!(start "inner product");
        // todo: Try to get rid of this clone. The alternative line
        //   `let main_and_aux_element = (&weights * &summands).sum();`
        //   without cloning the weights does not compile for a seemingly nonsensical reason.
        let weights = weights.to_owned();
        let main_and_aux_element = (weights * row).sum();
        profiler!(stop "inner product");
        main_and_aux_element
    }

    /// Computes the quotient segments in a memory-friendly way, i.e., without ever
    /// representing the entire low-degree extended trace. Instead, the trace is
    /// extrapolated over cosets of the trace domain, and the quotients are computed
    /// there. The resulting coset-quotients are linearly recombined to produce the
    /// quotient segment codewords.
    fn compute_quotient_segments_with_jit_lde(
        randomized_main_columns: Array1<Polynomial<BFieldElement>>,
        randomized_aux_columns: Array1<Polynomial<XFieldElement>>,
        trace_domain: ArithmeticDomain,
        randomized_trace_domain: ArithmeticDomain,
        fri_domain: ArithmeticDomain,
        challenges: &Challenges,
        quotient_combination_weights: &[XFieldElement],
    ) -> (Array2<XFieldElement>, Array1<Polynomial<XFieldElement>>) {
        let num_rows = randomized_trace_domain.length;
        let root_order = (num_rows * NUM_QUOTIENT_SEGMENTS).try_into().unwrap();

        // the powers of ι define `num_quotient_segments`-many cosets of the randomized trace domain
        let iota = BFieldElement::primitive_root_of_unity(root_order)
            .expect("Cannot find ι, a primitive nth root of unity of the right order n.");
        let domain = ArithmeticDomain::of_length(num_rows).unwrap();

        // for every coset, evaluate constraints
        profiler!(start "zero-initialization");
        let mut quotient_multicoset_evaluations =
            fast_zeros_column_major(num_rows, NUM_QUOTIENT_SEGMENTS);
        let mut main_columns = fast_zeros_column_major(num_rows, MasterMainTable::NUM_COLUMNS);
        let mut aux_columns = fast_zeros_column_major(num_rows, MasterAuxTable::NUM_COLUMNS);
        profiler!(stop "zero-initialization");

        profiler!(start "calculate quotients");
        for (coset_index, quotient_column) in (0..u64::try_from(NUM_QUOTIENT_SEGMENTS).unwrap())
            .zip(quotient_multicoset_evaluations.columns_mut())
        {
            // always also offset by fri domain offset to avoid division-by-zero errors
            let domain = domain.with_offset(iota.mod_pow(coset_index) * fri_domain.offset);
            profiler!(start "evaluate" ("LDE"));
            Zip::from(randomized_main_columns.axis_iter(Axis(0)))
                .and(main_columns.axis_iter_mut(Axis(1)))
                .into_par_iter()
                .for_each(|(trace_poly, target_column)| {
                    Array1::from(domain.evaluate(&trace_poly[[]])).move_into(target_column);
                });
            Zip::from(randomized_aux_columns.axis_iter(Axis(0)))
                .and(aux_columns.axis_iter_mut(Axis(1)))
                .into_par_iter()
                .for_each(|(trace_poly, target_column)| {
                    Array1::from(domain.evaluate(&trace_poly[[]])).move_into(target_column);
                });
            profiler!(stop "evaluate");
            profiler!(start "AIR evaluation" ("AIR"));
            let all_quotients = all_quotients_combined(
                main_columns.view(),
                aux_columns.view(),
                trace_domain,
                domain,
                challenges,
                quotient_combination_weights,
            );
            Array1::from(all_quotients).move_into(quotient_column);
            profiler!(stop "AIR evaluation");
        }
        profiler!(stop "calculate quotients");

        profiler!(start "segmentify");
        let segmentification = Self::segmentify(
            quotient_multicoset_evaluations,
            fri_domain.offset,
            iota,
            randomized_trace_domain,
            fri_domain,
        );
        profiler!(stop "segmentify");

        segmentification
    }

    /// Map a matrix whose columns represent the evaluation of a high-degree
    /// polynomial on all cosets of the trace domain, to
    /// 1. a matrix of segment codewords (on the FRI domain), and
    /// 2. an array of matching segment polynomials,
    ///
    /// such that the segment polynomials correspond to the interleaving split of
    /// the original high-degree polynomial.
    ///
    /// For example, let f(X) have degree 2N where N is the trace domain length.
    /// Then the input is an Nx2 matrix representing the values of f(X) on the trace
    /// domain and its coset. The segment polynomials are f_E(X) and f_O(X) such
    /// that f(X) = f_E(X^2) + X*f_O(X^2) and the segment codewords are their
    /// evaluations on the FRI domain.
    //
    // This method is factored out from `compute_quotient_segments` for the purpose
    // of testing. Conceptually, it belongs there.
    fn segmentify(
        quotient_multicoset_evaluations: Array2<XFieldElement>,
        psi: BFieldElement,
        iota: BFieldElement,
        randomized_trace_domain: ArithmeticDomain,
        fri_domain: ArithmeticDomain,
    ) -> (Array2<XFieldElement>, Array1<Polynomial<XFieldElement>>) {
        let num_rows = randomized_trace_domain.length;
        let num_segments = quotient_multicoset_evaluations.ncols();
        assert!(
            num_rows > num_segments,
            "trace domain length: {num_rows} versus num segments: {num_segments}",
        );

        // Matrix `quotients` contains q(Ψ · ι^j · ω^i) in location (i,j) where ω is the
        // trace domain generator, and where iota is an Fth root of ω such that ι^F = ω,
        // where F is `num_quotient_segments`. So `quotients` contains q(Ψ · ι^(j+i·F)).

        // We need F-tuples from this matrix of elements separated by N/F rows.
        let step_size = num_rows / num_segments;
        let quotient_segments = (0..num_rows)
            .into_par_iter()
            .flat_map(|jif| {
                let col_idx = jif % num_segments;
                let start_row = (jif - col_idx) / num_segments;
                quotient_multicoset_evaluations
                    .slice(s![start_row..; step_size, col_idx])
                    .to_vec()
            })
            .collect();
        let mut quotient_segments =
            Array2::from_shape_vec((num_rows, num_segments), quotient_segments).unwrap();

        // Matrix `quotient_segments` now contains q(Ψ · ι^(j+i·F+l·N/F)) in cell
        // (j+i·F, l). So *row* j+i·F contains {q(Ψ · ι^(j+i·F+l·N/F)) for l in [0..F-1]}.

        // apply inverse of Vandermonde matrix for ω^(N/F) matrix to every row
        let n_over_f = (num_rows / num_segments).try_into().unwrap();
        let xi = randomized_trace_domain.generator.mod_pow_u32(n_over_f);
        assert_eq!(bfe!(1), xi.mod_pow(num_segments.try_into().unwrap()));
        let logn = num_segments.ilog2();
        quotient_segments
            .axis_iter_mut(Axis(0))
            .into_par_iter()
            .for_each(|mut row| {
                // `.unwrap()` is safe because `quotient_segments` is in row-major order
                let row = row.as_slice_mut().unwrap();
                intt(row, xi, logn);
            });

        // scale every row by Ψ^-k · ι^(-k(j+i·F))
        let num_threads = std::thread::available_parallelism()
            .map(|t| t.get())
            .unwrap_or(1);
        let chunk_size = (num_rows / num_threads).max(1);
        let iota_inverse = iota.inverse();
        let psi_inverse = psi.inverse();
        quotient_segments
            .axis_chunks_iter_mut(Axis(0), chunk_size)
            .into_par_iter()
            .enumerate()
            .for_each(|(thread, mut chunk)| {
                let chunk_start = thread * chunk_size;
                let mut psi_iotajif_inv =
                    psi_inverse * iota_inverse.mod_pow(chunk_start.try_into().unwrap());
                for mut row in chunk.rows_mut() {
                    let mut psi_iotajif_invk = xfe!(1);
                    for cell in &mut row {
                        *cell *= psi_iotajif_invk;
                        psi_iotajif_invk *= psi_iotajif_inv;
                    }
                    psi_iotajif_inv *= iota_inverse;
                }
            });

        // Matrix `quotients_codewords` contains q_k(Ψ^F · ω^(j+i·F)) in cell (j+i·F, k).
        // To see this, observe that
        //
        //     ⎛      …       ⎞   ⎛ ⎛    …                ⎞   ⎛     …                ⎞ ⎞
        //     ⎜ …  ξ^(l·k) … ⎟ · ⎜ ⎜ ψ^k · ι^(j·k+i·k·F) ⎟ ∘ ⎜ q_k(ψ^F · ω^(j+i·F)) ⎟ ⎟
        //     ⎝      …       ⎠   ⎝ ⎝    …                ⎠   ⎝     …                ⎠ ⎠
        //  =
        //     ⎛      …                            ⎞   ⎛    …                 ⎞
        //     ⎜ …  ψ^k · ι^(j·k+i·k·F+l·k·N/F)  … ⎟ · ⎜ q_k(ψ^F · ω^(j+i·F)) ⎟
        //     ⎝      …                            ⎠   ⎝    …                 ⎠
        //  =
        //     ⎛      …                       ⎞
        //     ⎜ q(ψ · ι^j · ω^(i + l · N/F)) ⎟
        //     ⎝      …                       ⎠

        // low-degree extend columns from trace to FRI domain
        let mut quotient_codewords = Array2::zeros([fri_domain.length, num_segments]);
        let mut quotient_polynomials = Array1::zeros([num_segments]);
        Zip::from(quotient_segments.axis_iter(Axis(1)))
            .and(quotient_codewords.axis_iter_mut(Axis(1)))
            .and(quotient_polynomials.axis_iter_mut(Axis(0)))
            .par_for_each(|segment, codeword, polynomial| {
                let psi_exponent = num_segments.try_into().unwrap();
                let segment_domain_offset = psi.mod_pow(psi_exponent);
                let segment_domain = randomized_trace_domain.with_offset(segment_domain_offset);

                // `.to_vec()` is necessary because `segment` is a column of `quotient_segments`,
                // which is in row-major order
                let interpolant = segment_domain.interpolate(&segment.to_vec());
                let lde_codeword = fri_domain.evaluate(&interpolant);
                Array1::from(lde_codeword).move_into(codeword);
                Array0::from_elem((), interpolant).move_into(polynomial);
            });

        (quotient_codewords, quotient_polynomials)
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
    /// of length [`MasterMainTable::NUM_COLUMNS`]
    main: Array1<XFieldElement>,

    /// of length [`MasterAuxTable::NUM_COLUMNS`]
    aux: Array1<XFieldElement>,

    /// of length [`NUM_QUOTIENT_SEGMENTS`]
    quot_segments: Array1<XFieldElement>,

    /// of length [`NUM_DEEP_CODEWORD_COMPONENTS`]
    deep: Array1<XFieldElement>,
}

impl LinearCombinationWeights {
    const NUM: usize = MasterMainTable::NUM_COLUMNS
        + MasterAuxTable::NUM_COLUMNS
        + NUM_QUOTIENT_SEGMENTS
        + NUM_DEEP_CODEWORD_COMPONENTS;

    fn sample(proof_stream: &mut ProofStream) -> Self {
        const MAIN_END: usize = MasterMainTable::NUM_COLUMNS;
        const AUX_END: usize = MAIN_END + MasterAuxTable::NUM_COLUMNS;
        const QUOT_END: usize = AUX_END + NUM_QUOTIENT_SEGMENTS;

        let weights = proof_stream.sample_scalars(Self::NUM);

        Self {
            main: weights[..MAIN_END].to_vec().into(),
            aux: weights[MAIN_END..AUX_END].to_vec().into(),
            quot_segments: weights[AUX_END..QUOT_END].to_vec().into(),
            deep: weights[QUOT_END..].to_vec().into(),
        }
    }

    fn main_and_aux(&self) -> Array1<XFieldElement> {
        let main = self.main.clone().into_iter();
        main.chain(self.aux.clone()).collect()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::HashMap;
    use std::collections::HashSet;

    use air::challenge_id::ChallengeId::StandardInputIndeterminate;
    use air::challenge_id::ChallengeId::StandardOutputIndeterminate;
    use air::cross_table_argument::CrossTableArg;
    use air::cross_table_argument::EvalArg;
    use air::cross_table_argument::GrandCrossTableArg;
    use air::table::cascade::CascadeTable;
    use air::table::hash::HashTable;
    use air::table::jump_stack::JumpStackTable;
    use air::table::lookup::LookupTable;
    use air::table::op_stack::OpStackTable;
    use air::table::processor::ProcessorTable;
    use air::table::program::ProgramTable;
    use air::table::ram;
    use air::table::ram::RamTable;
    use air::table::u32::U32Table;
    use air::table::TableId;
    use air::table_column::MasterAuxColumn;
    use air::table_column::MasterMainColumn;
    use air::table_column::OpStackMainColumn;
    use air::table_column::ProcessorAuxColumn::InputTableEvalArg;
    use air::table_column::ProcessorAuxColumn::OutputTableEvalArg;
    use air::table_column::ProcessorMainColumn;
    use air::table_column::RamMainColumn;
    use air::AIR;
    use assert2::assert;
    use assert2::check;
    use assert2::let_assert;
    use constraint_circuit::ConstraintCircuitBuilder;
    use isa::error::OpStackError;
    use isa::instruction::Instruction;
    use isa::op_stack::OpStackElement;
    use itertools::izip;
    use num_traits::Zero;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use proptest_arbitrary_interop::arb;
    use rand::prelude::StdRng;
    use rand::thread_rng;
    use rand::Rng;
    use rand_core::SeedableRng;
    use strum::EnumCount;
    use strum::IntoEnumIterator;
    use test_strategy::proptest;
    use twenty_first::math::other::random_elements;

    use super::*;
    use crate::config::CacheDecision;
    use crate::error::InstructionError;
    use crate::shared_tests::construct_master_main_table;
    use crate::shared_tests::low_security_stark;
    use crate::shared_tests::prove_and_verify;
    use crate::shared_tests::ProgramAndInput;
    use crate::shared_tests::DEFAULT_LOG2_FRI_EXPANSION_FACTOR_FOR_TESTS;
    use crate::table::auxiliary_table;
    use crate::table::auxiliary_table::Evaluable;
    use crate::table::master_table::MasterAuxTable;
    use crate::triton_program;
    use crate::vm::tests::property_based_test_program_for_and;
    use crate::vm::tests::property_based_test_program_for_assert_vector;
    use crate::vm::tests::property_based_test_program_for_div_mod;
    use crate::vm::tests::property_based_test_program_for_eq;
    use crate::vm::tests::property_based_test_program_for_is_u32;
    use crate::vm::tests::property_based_test_program_for_log2floor;
    use crate::vm::tests::property_based_test_program_for_lsb;
    use crate::vm::tests::property_based_test_program_for_lt;
    use crate::vm::tests::property_based_test_program_for_pop_count;
    use crate::vm::tests::property_based_test_program_for_pow;
    use crate::vm::tests::property_based_test_program_for_random_ram_access;
    use crate::vm::tests::property_based_test_program_for_split;
    use crate::vm::tests::property_based_test_program_for_xb_dot_step;
    use crate::vm::tests::property_based_test_program_for_xor;
    use crate::vm::tests::property_based_test_program_for_xx_dot_step;
    use crate::vm::tests::test_program_0_lt_0;
    use crate::vm::tests::test_program_claim_in_ram_corresponds_to_currently_running_program;
    use crate::vm::tests::test_program_for_add_mul_invert;
    use crate::vm::tests::test_program_for_and;
    use crate::vm::tests::test_program_for_assert_vector;
    use crate::vm::tests::test_program_for_call_recurse_return;
    use crate::vm::tests::test_program_for_div_mod;
    use crate::vm::tests::test_program_for_divine;
    use crate::vm::tests::test_program_for_eq;
    use crate::vm::tests::test_program_for_halt;
    use crate::vm::tests::test_program_for_hash;
    use crate::vm::tests::test_program_for_log2floor;
    use crate::vm::tests::test_program_for_lsb;
    use crate::vm::tests::test_program_for_lt;
    use crate::vm::tests::test_program_for_many_sponge_instructions;
    use crate::vm::tests::test_program_for_merkle_step_left_sibling;
    use crate::vm::tests::test_program_for_merkle_step_mem_left_sibling;
    use crate::vm::tests::test_program_for_merkle_step_mem_right_sibling;
    use crate::vm::tests::test_program_for_merkle_step_right_sibling;
    use crate::vm::tests::test_program_for_pop_count;
    use crate::vm::tests::test_program_for_pow;
    use crate::vm::tests::test_program_for_push_pop_dup_swap_nop;
    use crate::vm::tests::test_program_for_read_io_write_io;
    use crate::vm::tests::test_program_for_recurse_or_return;
    use crate::vm::tests::test_program_for_skiz;
    use crate::vm::tests::test_program_for_split;
    use crate::vm::tests::test_program_for_sponge_instructions;
    use crate::vm::tests::test_program_for_sponge_instructions_2;
    use crate::vm::tests::test_program_for_starting_with_pop_count;
    use crate::vm::tests::test_program_for_write_mem_read_mem;
    use crate::vm::tests::test_program_for_x_invert;
    use crate::vm::tests::test_program_for_xb_mul;
    use crate::vm::tests::test_program_for_xor;
    use crate::vm::tests::test_program_for_xx_add;
    use crate::vm::tests::test_program_for_xx_mul;
    use crate::vm::tests::test_program_hash_nop_nop_lt;
    use crate::vm::tests::ProgramForMerkleTreeUpdate;
    use crate::vm::tests::ProgramForRecurseOrReturn;
    use crate::vm::tests::ProgramForSpongeAndHashInstructions;
    use crate::vm::NonDeterminism;
    use crate::vm::VM;
    use crate::PublicInput;

    pub(crate) fn master_main_table_for_low_security_level(
        program_and_input: ProgramAndInput,
    ) -> (Stark, Claim, MasterMainTable) {
        let ProgramAndInput {
            program,
            public_input,
            non_determinism,
        } = program_and_input;

        let (aet, stdout) =
            VM::trace_execution(&program, public_input.clone(), non_determinism).unwrap();
        let stark = low_security_stark(DEFAULT_LOG2_FRI_EXPANSION_FACTOR_FOR_TESTS);
        let claim = Claim::about_program(&aet.program)
            .with_input(public_input.individual_tokens)
            .with_output(stdout);
        let master_main_table = construct_master_main_table(stark, &aet);

        (stark, claim, master_main_table)
    }

    pub(crate) fn master_tables_for_low_security_level(
        program_and_input: ProgramAndInput,
    ) -> (Stark, Claim, MasterMainTable, MasterAuxTable, Challenges) {
        let (stark, claim, mut master_main_table) =
            master_main_table_for_low_security_level(program_and_input);

        let challenges = Challenges::placeholder(&claim);
        master_main_table.pad();
        let master_aux_table = master_main_table.extend(&challenges);

        (
            stark,
            claim,
            master_main_table,
            master_aux_table,
            challenges,
        )
    }

    #[test]
    fn quotient_segments_are_independent_of_fri_table_caching() {
        // ensure caching _can_ happen by overwriting environment variables
        crate::config::overwrite_lde_trace_caching_to(CacheDecision::Cache);

        let mut rng = StdRng::seed_from_u64(1632525295622789151);
        let weights = rng.gen::<[XFieldElement; MasterAuxTable::NUM_CONSTRAINTS]>();

        let program = ProgramAndInput::new(triton_program!(halt));
        let (stark, _, mut main, mut aux, ch) = master_tables_for_low_security_level(program);
        let padded_height = main.trace_table().nrows();
        let fri_dom = stark.derive_fri(padded_height).unwrap().domain;
        let max_degree = stark.derive_max_degree(padded_height);
        let quot_dom = Stark::quotient_domain(fri_dom, max_degree).unwrap();

        debug_assert!(main.fri_domain_table().is_none());
        debug_assert!(aux.fri_domain_table().is_none());
        let jit_segments =
            Stark::compute_quotient_segments(&main, &aux, fri_dom, quot_dom, &ch, &weights);

        debug_assert!(main.fri_domain_table().is_none());
        main.maybe_low_degree_extend_all_columns();
        debug_assert!(main.fri_domain_table().is_some());

        debug_assert!(aux.fri_domain_table().is_none());
        aux.maybe_low_degree_extend_all_columns();
        debug_assert!(aux.fri_domain_table().is_some());

        let cache_segments =
            Stark::compute_quotient_segments(&main, &aux, fri_dom, quot_dom, &ch, &weights);

        assert_eq!(jit_segments, cache_segments);
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
        let (_, _, master_main_table, _, _) =
            master_tables_for_low_security_level(ProgramAndInput::new(program));

        println!();
        println!("Processor Table:\n");
        println!("| clk | ci  | nia | st0 | st1 | st2 | st3 | st4 | st5 |");
        println!("|----:|:----|:----|----:|----:|----:|----:|----:|----:|");
        for row in master_main_table
            .table(TableId::Processor)
            .rows()
            .into_iter()
            .take(40)
        {
            let clk = row[ProcessorMainColumn::CLK.main_index()].to_string();
            let st0 = row[ProcessorMainColumn::ST0.main_index()].to_string();
            let st1 = row[ProcessorMainColumn::ST1.main_index()].to_string();
            let st2 = row[ProcessorMainColumn::ST2.main_index()].to_string();
            let st3 = row[ProcessorMainColumn::ST3.main_index()].to_string();
            let st4 = row[ProcessorMainColumn::ST4.main_index()].to_string();
            let st5 = row[ProcessorMainColumn::ST5.main_index()].to_string();

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
        for row in master_main_table
            .table(TableId::Ram)
            .rows()
            .into_iter()
            .take(25)
        {
            let clk = row[RamMainColumn::CLK.main_index()].to_string();
            let ramp = row[RamMainColumn::RamPointer.main_index()].to_string();
            let ramv = row[RamMainColumn::RamValue.main_index()].to_string();
            let iord = row[RamMainColumn::InverseOfRampDifference.main_index()].to_string();

            let instruction_type = match row[RamMainColumn::InstructionType.main_index()] {
                ram::INSTRUCTION_TYPE_READ => "read",
                ram::INSTRUCTION_TYPE_WRITE => "write",
                ram::PADDING_INDICATOR => "pad",
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
        let (_, _, master_main_table) =
            master_main_table_for_low_security_level(ProgramAndInput::new(program));

        println!();
        println!("Processor Table:");
        println!("| clk | ci  | nia | st0 | st1 | st2 | st3 | underflow | pointer |");
        println!("|----:|:----|----:|----:|----:|----:|----:|:----------|--------:|");
        for row in master_main_table
            .table(TableId::Processor)
            .rows()
            .into_iter()
            .take(num_interesting_rows)
        {
            let clk = row[ProcessorMainColumn::CLK.main_index()].to_string();
            let st0 = row[ProcessorMainColumn::ST0.main_index()].to_string();
            let st1 = row[ProcessorMainColumn::ST1.main_index()].to_string();
            let st2 = row[ProcessorMainColumn::ST2.main_index()].to_string();
            let st3 = row[ProcessorMainColumn::ST3.main_index()].to_string();
            let st4 = row[ProcessorMainColumn::ST4.main_index()].to_string();
            let st5 = row[ProcessorMainColumn::ST5.main_index()].to_string();
            let st6 = row[ProcessorMainColumn::ST6.main_index()].to_string();
            let st7 = row[ProcessorMainColumn::ST7.main_index()].to_string();
            let st8 = row[ProcessorMainColumn::ST8.main_index()].to_string();
            let st9 = row[ProcessorMainColumn::ST9.main_index()].to_string();

            let osp = row[ProcessorMainColumn::OpStackPointer.main_index()];
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
        for row in master_main_table
            .table(TableId::OpStack)
            .rows()
            .into_iter()
            .take(num_interesting_rows)
        {
            let clk = row[OpStackMainColumn::CLK.main_index()].to_string();
            let ib1 = row[OpStackMainColumn::IB1ShrinkStack.main_index()].to_string();

            let osp = row[OpStackMainColumn::StackPointer.main_index()];
            let osp =
                (osp.value() + fake_op_stack_size).saturating_sub(OpStackElement::COUNT as u64);
            let osp = osp.to_string();

            let value = row[OpStackMainColumn::FirstUnderflowElement.main_index()].to_string();

            let interesting_cols = [clk, ib1, osp, value];
            let interesting_cols = interesting_cols
                .map(|ff| format!("{:>10}", format!("{ff}")))
                .join(" | ");
            println!("| {interesting_cols} |");
        }
    }

    fn ci_and_nia_from_master_table_row(row: ArrayView1<BFieldElement>) -> (String, String) {
        let curr_instruction = row[ProcessorMainColumn::CI.main_index()].value();
        let next_instruction_or_arg = row[ProcessorMainColumn::NIA.main_index()].value();

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
        for deg in auxiliary_table::all_degrees_with_origin(interpolant_degree, padded_height) {
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
        let (_, claim, _, master_aux_table, all_challenges) =
            master_tables_for_low_security_level(program_and_input);

        let processor_table = master_aux_table.table(TableId::Processor);
        let processor_table_last_row = processor_table.slice(s![-1, ..]);
        let ptie = processor_table_last_row[InputTableEvalArg.aux_index()];
        let ine = EvalArg::compute_terminal(
            &claim.input,
            EvalArg::default_initial(),
            all_challenges[StandardInputIndeterminate],
        );
        check!(ptie == ine);

        let ptoe = processor_table_last_row[OutputTableEvalArg.aux_index()];
        let oute = EvalArg::compute_terminal(
            &claim.output,
            EvalArg::default_initial(),
            all_challenges[StandardOutputIndeterminate],
        );
        check!(ptoe == oute);
    }

    #[test]
    fn constraint_polynomials_use_right_number_of_variables() {
        let challenges = Challenges::default();
        let main_row = Array1::<BFieldElement>::zeros(MasterMainTable::NUM_COLUMNS);
        let aux_row = Array1::zeros(MasterAuxTable::NUM_COLUMNS);

        let br = main_row.view();
        let er = aux_row.view();

        MasterAuxTable::evaluate_initial_constraints(br, er, &challenges);
        MasterAuxTable::evaluate_consistency_constraints(br, er, &challenges);
        MasterAuxTable::evaluate_transition_constraints(br, er, br, er, &challenges);
        MasterAuxTable::evaluate_terminal_constraints(br, er, &challenges);
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
            ProgramTable::initial_constraints(&circuit_builder),
            ProcessorTable::initial_constraints(&circuit_builder),
            OpStackTable::initial_constraints(&circuit_builder),
            RamTable::initial_constraints(&circuit_builder),
            JumpStackTable::initial_constraints(&circuit_builder),
            HashTable::initial_constraints(&circuit_builder),
            CascadeTable::initial_constraints(&circuit_builder),
            LookupTable::initial_constraints(&circuit_builder),
            U32Table::initial_constraints(&circuit_builder),
            GrandCrossTableArg::initial_constraints(&circuit_builder),
        ]
        .map(|vec| vec.len());
        let circuit_builder = ConstraintCircuitBuilder::new();
        let all_cons = [
            ProgramTable::consistency_constraints(&circuit_builder),
            ProcessorTable::consistency_constraints(&circuit_builder),
            OpStackTable::consistency_constraints(&circuit_builder),
            RamTable::consistency_constraints(&circuit_builder),
            JumpStackTable::consistency_constraints(&circuit_builder),
            HashTable::consistency_constraints(&circuit_builder),
            CascadeTable::consistency_constraints(&circuit_builder),
            LookupTable::consistency_constraints(&circuit_builder),
            U32Table::consistency_constraints(&circuit_builder),
            GrandCrossTableArg::consistency_constraints(&circuit_builder),
        ]
        .map(|vec| vec.len());
        let circuit_builder = ConstraintCircuitBuilder::new();
        let all_trans = [
            ProgramTable::transition_constraints(&circuit_builder),
            ProcessorTable::transition_constraints(&circuit_builder),
            OpStackTable::transition_constraints(&circuit_builder),
            RamTable::transition_constraints(&circuit_builder),
            JumpStackTable::transition_constraints(&circuit_builder),
            HashTable::transition_constraints(&circuit_builder),
            CascadeTable::transition_constraints(&circuit_builder),
            LookupTable::transition_constraints(&circuit_builder),
            U32Table::transition_constraints(&circuit_builder),
            GrandCrossTableArg::transition_constraints(&circuit_builder),
        ]
        .map(|vec| vec.len());
        let circuit_builder = ConstraintCircuitBuilder::new();
        let all_term = [
            ProgramTable::terminal_constraints(&circuit_builder),
            ProcessorTable::terminal_constraints(&circuit_builder),
            OpStackTable::terminal_constraints(&circuit_builder),
            RamTable::terminal_constraints(&circuit_builder),
            JumpStackTable::terminal_constraints(&circuit_builder),
            HashTable::terminal_constraints(&circuit_builder),
            CascadeTable::terminal_constraints(&circuit_builder),
            LookupTable::terminal_constraints(&circuit_builder),
            U32Table::terminal_constraints(&circuit_builder),
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
        let main_row = Array1::<BFieldElement>::zeros(MasterMainTable::NUM_COLUMNS);
        let aux_row = Array1::zeros(MasterAuxTable::NUM_COLUMNS);
        let ch = Challenges::default();
        let padded_height = 2;
        let num_trace_randomizers = 2;
        let interpolant_degree = interpolant_degree(padded_height, num_trace_randomizers);

        // Shorten some names for better formatting. This is just a test.
        let ph = padded_height;
        let id = interpolant_degree;
        let br = main_row.view();
        let er = aux_row.view();

        let num_init_quots = MasterAuxTable::NUM_INITIAL_CONSTRAINTS;
        let num_cons_quots = MasterAuxTable::NUM_CONSISTENCY_CONSTRAINTS;
        let num_tran_quots = MasterAuxTable::NUM_TRANSITION_CONSTRAINTS;
        let num_term_quots = MasterAuxTable::NUM_TERMINAL_CONSTRAINTS;

        let eval_init_consts = MasterAuxTable::evaluate_initial_constraints(br, er, &ch);
        let eval_cons_consts = MasterAuxTable::evaluate_consistency_constraints(br, er, &ch);
        let eval_tran_consts = MasterAuxTable::evaluate_transition_constraints(br, er, br, er, &ch);
        let eval_term_consts = MasterAuxTable::evaluate_terminal_constraints(br, er, &ch);

        assert!(num_init_quots == eval_init_consts.len());
        assert!(num_cons_quots == eval_cons_consts.len());
        assert!(num_tran_quots == eval_tran_consts.len());
        assert!(num_term_quots == eval_term_consts.len());

        assert!(num_init_quots == MasterAuxTable::initial_quotient_degree_bounds(id).len());
        assert!(num_cons_quots == MasterAuxTable::consistency_quotient_degree_bounds(id, ph).len());
        assert!(num_tran_quots == MasterAuxTable::transition_quotient_degree_bounds(id, ph).len());
        assert!(num_term_quots == MasterAuxTable::terminal_quotient_degree_bounds(id).len());
    }

    #[test]
    fn constraints_evaluate_to_zero_on_fibonacci() {
        let source_code_and_input =
            ProgramAndInput::new(crate::example_programs::FIBONACCI_SEQUENCE.clone())
                .with_input(bfe_array![100]);
        triton_constraints_evaluate_to_zero(source_code_and_input);
    }

    #[test]
    fn constraints_evaluate_to_zero_on_big_mmr_snippet() {
        let source_code_and_input = ProgramAndInput::new(
            crate::example_programs::CALCULATE_NEW_MMR_PEAKS_FROM_APPEND_WITH_SAFE_LISTS.clone(),
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
    fn constraints_evaluate_to_zero_on_program_for_recurse_or_return() {
        triton_constraints_evaluate_to_zero(test_program_for_recurse_or_return())
    }

    #[proptest(cases = 20)]
    fn constraints_evaluate_to_zero_on_property_based_test_program_for_recurse_or_return(
        program: ProgramForRecurseOrReturn,
    ) {
        triton_constraints_evaluate_to_zero(program.assemble())
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
    fn constraints_evaluate_to_zero_on_program_for_merkle_step_right_sibling() {
        triton_constraints_evaluate_to_zero(test_program_for_merkle_step_right_sibling())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_program_for_merkle_step_left_sibling() {
        triton_constraints_evaluate_to_zero(test_program_for_merkle_step_left_sibling())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_program_for_merkle_step_mem_right_sibling() {
        triton_constraints_evaluate_to_zero(test_program_for_merkle_step_mem_right_sibling())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_program_for_merkle_step_mem_left_sibling() {
        triton_constraints_evaluate_to_zero(test_program_for_merkle_step_mem_left_sibling())
    }

    #[proptest(cases = 20)]
    fn constraints_evaluate_to_zero_on_property_based_test_program_for_merkle_tree_update(
        program: ProgramForMerkleTreeUpdate,
    ) {
        triton_constraints_evaluate_to_zero(program.assemble())
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
    fn constraints_evaluate_to_zero_on_program_for_xx_add() {
        triton_constraints_evaluate_to_zero(test_program_for_xx_add())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_program_for_xx_mul() {
        triton_constraints_evaluate_to_zero(test_program_for_xx_mul())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_program_for_x_invert() {
        triton_constraints_evaluate_to_zero(test_program_for_x_invert())
    }

    #[test]
    fn constraints_evaluate_to_zero_on_program_for_xb_mul() {
        triton_constraints_evaluate_to_zero(test_program_for_xb_mul())
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
    fn constraints_evaluate_to_zero_on_single_sponge_absorb_mem_instructions() {
        let program = triton_program!(sponge_init sponge_absorb_mem halt);
        let program = ProgramAndInput::new(program);
        triton_constraints_evaluate_to_zero(program);
    }

    #[proptest(cases = 3)]
    fn constraints_evaluate_to_zero_on_property_based_test_program_for_sponge_instructions(
        program: ProgramForSpongeAndHashInstructions,
    ) {
        triton_constraints_evaluate_to_zero(program.assemble());
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
    fn constraints_evaluate_to_zero_on_property_based_test_program_for_xx_dot_step() {
        triton_constraints_evaluate_to_zero(property_based_test_program_for_xx_dot_step());
    }

    #[test]
    fn constraints_evaluate_to_zero_on_property_based_test_program_for_xb_dot_step() {
        triton_constraints_evaluate_to_zero(property_based_test_program_for_xb_dot_step());
    }

    #[test]
    fn can_read_twice_from_same_ram_address_within_one_cycle() {
        for i in 0..=2 {
            // This program reads from the same address twice, even if the stack
            // is not well-initialized.
            let program = triton_program! {
                dup 0
                push {i} add
                xx_dot_step
                halt
            };
            let result = VM::run(&program, PublicInput::default(), NonDeterminism::default());
            assert!(result.is_ok());
            let program_and_input = ProgramAndInput::new(program);
            triton_constraints_evaluate_to_zero(program_and_input);
        }
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
                master_main_trace_table: ArrayView2<BFieldElement>,
                master_aux_trace_table: ArrayView2<XFieldElement>,
                challenges: &Challenges,
            ) {
                assert!(master_main_trace_table.nrows() == master_aux_trace_table.nrows());
                let challenges = &challenges.challenges;

                let builder = ConstraintCircuitBuilder::new();
                for (constraint_idx, constraint) in $table::initial_constraints(&builder)
                    .into_iter()
                    .map(|constraint_monad| constraint_monad.consume())
                    .enumerate()
                {
                    let evaluated_constraint = constraint.evaluate(
                        master_main_trace_table.slice(s![..1, ..]),
                        master_aux_trace_table.slice(s![..1, ..]),
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
                    for row_idx in 0..master_main_trace_table.nrows() {
                        let evaluated_constraint = constraint.evaluate(
                            master_main_trace_table.slice(s![row_idx..=row_idx, ..]),
                            master_aux_trace_table.slice(s![row_idx..=row_idx, ..]),
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
                    for row_idx in 0..master_main_trace_table.nrows() - 1 {
                        let evaluated_constraint = constraint.evaluate(
                            master_main_trace_table.slice(s![row_idx..=row_idx + 1, ..]),
                            master_aux_trace_table.slice(s![row_idx..=row_idx + 1, ..]),
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
                        master_main_trace_table.slice(s![-1.., ..]),
                        master_aux_trace_table.slice(s![-1.., ..]),
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

    check_constraints_fn!(fn check_program_table_constraints for ProgramTable);
    check_constraints_fn!(fn check_processor_table_constraints for ProcessorTable);
    check_constraints_fn!(fn check_op_stack_table_constraints for OpStackTable);
    check_constraints_fn!(fn check_ram_table_constraints for RamTable);
    check_constraints_fn!(fn check_jump_stack_table_constraints for JumpStackTable);
    check_constraints_fn!(fn check_hash_table_constraints for HashTable);
    check_constraints_fn!(fn check_cascade_table_constraints for CascadeTable);
    check_constraints_fn!(fn check_lookup_table_constraints for LookupTable);
    check_constraints_fn!(fn check_u32_table_constraints for U32Table);
    check_constraints_fn!(fn check_cross_table_constraints for GrandCrossTableArg);

    fn triton_constraints_evaluate_to_zero(program_and_input: ProgramAndInput) {
        let (_, _, master_main_table, master_aux_table, challenges) =
            master_tables_for_low_security_level(program_and_input);

        let mbt = master_main_table.trace_table();
        let met = master_aux_table.trace_table();
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
        check_cross_table_constraints(mbt, met, &challenges);
    }

    #[test]
    fn derived_constraints_evaluate_to_zero_on_halt() {
        derived_constraints_evaluate_to_zero(test_program_for_halt());
    }

    fn derived_constraints_evaluate_to_zero(program_and_input: ProgramAndInput) {
        let (_, _, master_main_table, master_aux_table, challenges) =
            master_tables_for_low_security_level(program_and_input);

        let master_main_trace_table = master_main_table.trace_table();
        let master_aux_trace_table = master_aux_table.trace_table();

        let evaluated_initial_constraints = MasterAuxTable::evaluate_initial_constraints(
            master_main_trace_table.row(0),
            master_aux_trace_table.row(0),
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

        for row_idx in 0..master_main_trace_table.nrows() {
            let evaluated_consistency_constraints =
                MasterAuxTable::evaluate_consistency_constraints(
                    master_main_trace_table.row(row_idx),
                    master_aux_trace_table.row(row_idx),
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

        for curr_row_idx in 0..master_main_trace_table.nrows() - 1 {
            let next_row_idx = curr_row_idx + 1;
            let evaluated_transition_constraints = MasterAuxTable::evaluate_transition_constraints(
                master_main_trace_table.row(curr_row_idx),
                master_aux_trace_table.row(curr_row_idx),
                master_main_trace_table.row(next_row_idx),
                master_aux_trace_table.row(next_row_idx),
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

        let evaluated_terminal_constraints = MasterAuxTable::evaluate_terminal_constraints(
            master_main_trace_table.row(master_main_trace_table.nrows() - 1),
            master_aux_trace_table.row(master_aux_trace_table.nrows() - 1),
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
    fn prove_and_verify_simple_program() {
        prove_and_verify(
            test_program_hash_nop_nop_lt(),
            DEFAULT_LOG2_FRI_EXPANSION_FACTOR_FOR_TESTS,
        );
    }

    #[test]
    fn prove_and_verify_halt_with_different_fri_expansion_factors() {
        for log_2_fri_expansion_factor in 1..5 {
            println!("Testing with log2_fri_expansion_factor = {log_2_fri_expansion_factor}");
            prove_and_verify(test_program_for_halt(), log_2_fri_expansion_factor);
        }
    }

    #[test]
    fn prove_and_verify_fibonacci_100() {
        let program_and_input =
            ProgramAndInput::new(crate::example_programs::FIBONACCI_SEQUENCE.clone())
                .with_input(PublicInput::from(bfe_array![100]));
        prove_and_verify(
            program_and_input,
            DEFAULT_LOG2_FRI_EXPANSION_FACTOR_FOR_TESTS,
        );
    }

    #[test]
    fn prove_verify_program_using_pick_and_place() {
        let input = bfe_vec![6, 3, 7, 5, 1, 2, 4, 4, 7, 3, 6, 1, 5, 2];
        let program = triton_program! {       // i: 13 12 11 10  9  8  7  6  5  4  3  2  1  0
            read_io 5 read_io 5 read_io 4     //  _  6  3  7  5 ›1‹ 2  4  4  7  3  6 ›1‹ 5  2
            pick 2 pick 9 place 13 place 13   //  _  1  1  6  3  7  5 ›2‹ 4  4  7  3  6  5 ›2‹
            pick 0 pick 7 place 13 place 13   //  _  2  2  1  1  6 ›3‹ 7  5  4  4  7 ›3‹ 6  5
            pick 2 pick 8 place 13 place 13   //  _  3  3  2  2  1  1  6  7  5 ›4‹›4‹ 7  6  5
            pick 3 pick 4 place 13 place 13   //  _  4  4  3  3  2  2  1  1  6  7 ›5‹ 7  6 ›5‹
            pick 0 pick 3 place 13 place 13   //  _  5  5  4  4  3  3  2  2  1  1 ›6‹ 7  7 ›6‹
            pick 0 pick 3 place 13 place 13   //  _  6  6  5  5  4  4  3  3  2  2  1  1 ›7‹›7‹
            pick 1 pick 1 place 13 place 13   //  _  7  7  6  6  5  5  4  4  3  3  2  2  1  1
            write_io 5 write_io 5 write_io 4  //  _
            halt
        };

        let program_and_input = ProgramAndInput::new(program).with_input(input);
        let output = program_and_input.run().unwrap();
        let expected_output = bfe_vec![1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7];
        assert!(expected_output == output);

        prove_and_verify(
            program_and_input,
            DEFAULT_LOG2_FRI_EXPANSION_FACTOR_FOR_TESTS,
        );
    }

    #[test]
    fn constraints_evaluate_to_zero_on_many_u32_operations() {
        let many_u32_instructions = ProgramAndInput::new(
            crate::example_programs::PROGRAM_WITH_MANY_U32_INSTRUCTIONS.clone(),
        );
        triton_constraints_evaluate_to_zero(many_u32_instructions);
    }

    #[test]
    fn prove_verify_many_u32_operations() {
        prove_and_verify(
            ProgramAndInput::new(
                crate::example_programs::PROGRAM_WITH_MANY_U32_INSTRUCTIONS.clone(),
            ),
            DEFAULT_LOG2_FRI_EXPANSION_FACTOR_FOR_TESTS,
        );
    }

    #[proptest]
    fn verifying_arbitrary_proof_does_not_panic(
        #[strategy(arb())] stark: Stark,
        #[strategy(arb())] claim: Claim,
        #[strategy(arb())] proof: Proof,
    ) {
        let _verdict = stark.verify(&claim, &proof);
    }

    #[proptest]
    fn negative_log_2_floor(
        #[strategy(arb())]
        #[filter(#st0.value() > u64::from(u32::MAX))]
        st0: BFieldElement,
    ) {
        let program = triton_program!(push {st0} log_2_floor halt);
        let_assert!(Err(err) = VM::run(&program, [].into(), [].into()));
        let_assert!(InstructionError::OpStackError(err) = err.source);
        let_assert!(OpStackError::FailedU32Conversion(element) = err);
        assert!(st0 == element);
    }

    #[test]
    fn negative_log_2_floor_of_0() {
        let program = triton_program!(push 0 log_2_floor halt);
        let_assert!(Err(err) = VM::run(&program, [].into(), [].into()));
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
            segment.evaluate::<_, FF>(x_pow_n) * x.mod_pow_u32(segment_idx as u32)
        };
        let evaluated_segments = segments.iter().enumerate().map(evaluate_segment);
        let sum_of_evaluated_segments = evaluated_segments.fold(FF::zero(), |acc, x| acc + x);
        assert!(f.evaluate::<_, FF>(x) == sum_of_evaluated_segments);
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

    #[proptest]
    fn quotient_segments_of_old_and_new_methods_are_identical(
        #[strategy(2_usize..8)] _log_trace_length: usize,
        #[strategy(Just(1 << #_log_trace_length))] trace_length: usize,
        #[strategy(Just(2 * #trace_length))] randomized_trace_length: usize,
        #[strategy(arb())]
        #[filter(!#offset.is_zero())]
        offset: BFieldElement,
        #[strategy(arb())] main_polynomials: [Polynomial<BFieldElement>;
            MasterMainTable::NUM_COLUMNS],
        #[strategy(arb())] aux_polynomials: [Polynomial<XFieldElement>;
            MasterAuxTable::NUM_COLUMNS],
        #[strategy(arb())] challenges: Challenges,
        #[strategy(arb())] quotient_weights: [XFieldElement; MasterAuxTable::NUM_CONSTRAINTS],
    ) {
        // truncate polynomials to randomized trace domain length many coefficients
        fn truncate_coefficients<FF: FiniteField>(
            mut polynomials: Vec<Polynomial<FF>>,
            num_coefficients: usize,
        ) -> Array1<Polynomial<FF>> {
            polynomials
                .par_iter_mut()
                .for_each(|p| p.coefficients.truncate(num_coefficients));
            Array1::from_vec(polynomials)
        }

        let main_polynomials =
            truncate_coefficients(main_polynomials.to_vec(), randomized_trace_length);
        let aux_polynomials =
            truncate_coefficients(aux_polynomials.to_vec(), randomized_trace_length);

        let trace_domain = ArithmeticDomain::of_length(trace_length)?;
        let randomized_trace_domain = ArithmeticDomain::of_length(randomized_trace_length)?;
        let fri_domain =
            ArithmeticDomain::of_length(4 * randomized_trace_length)?.with_offset(offset);
        let quotient_domain =
            ArithmeticDomain::of_length(4 * randomized_trace_length)?.with_offset(offset);

        let (quotient_segment_codewords_old, quotient_segment_polynomials_old) =
            compute_quotient_segments_old(
                main_polynomials.view(),
                aux_polynomials.view(),
                trace_domain,
                quotient_domain,
                fri_domain,
                &challenges,
                &quotient_weights,
            );

        let (quotient_segment_codewords_new, quotient_segment_polynomials_new) =
            Stark::compute_quotient_segments_with_jit_lde(
                main_polynomials,
                aux_polynomials,
                trace_domain,
                randomized_trace_domain,
                fri_domain,
                &challenges,
                &quotient_weights,
            );

        prop_assert_eq!(
            quotient_segment_codewords_old,
            quotient_segment_codewords_new
        );
        prop_assert_eq!(
            quotient_segment_polynomials_old,
            quotient_segment_polynomials_new
        );
    }

    fn compute_quotient_segments_old(
        main_polynomials: ArrayView1<Polynomial<BFieldElement>>,
        aux_polynomials: ArrayView1<Polynomial<XFieldElement>>,
        trace_domain: ArithmeticDomain,
        quotient_domain: ArithmeticDomain,
        fri_domain: ArithmeticDomain,
        challenges: &Challenges,
        quotient_weights: &[XFieldElement],
    ) -> (Array2<XFieldElement>, Array1<Polynomial<XFieldElement>>) {
        let mut main_quotient_domain_codewords =
            Array2::<BFieldElement>::zeros([quotient_domain.length, MasterMainTable::NUM_COLUMNS]);
        Zip::from(main_quotient_domain_codewords.axis_iter_mut(Axis(1)))
            .and(main_polynomials.axis_iter(Axis(0)))
            .for_each(|codeword, polynomial| {
                Array1::from_vec(quotient_domain.evaluate(&polynomial[()])).move_into(codeword);
            });
        let mut aux_quotient_domain_codewords =
            Array2::<XFieldElement>::zeros([quotient_domain.length, MasterAuxTable::NUM_COLUMNS]);
        Zip::from(aux_quotient_domain_codewords.axis_iter_mut(Axis(1)))
            .and(aux_polynomials.axis_iter(Axis(0)))
            .for_each(|codeword, polynomial| {
                Array1::from_vec(quotient_domain.evaluate(&polynomial[()])).move_into(codeword);
            });

        let quotient_codeword = all_quotients_combined(
            main_quotient_domain_codewords.view(),
            aux_quotient_domain_codewords.view(),
            trace_domain,
            quotient_domain,
            challenges,
            quotient_weights,
        );
        let quotient_codeword = Array1::from(quotient_codeword);
        let quotient_segment_polynomials =
            Stark::interpolate_quotient_segments(quotient_codeword, fri_domain);
        let quotient_segment_codewords =
            Stark::fri_domain_segment_polynomials(quotient_segment_polynomials.view(), fri_domain);

        (quotient_segment_codewords, quotient_segment_polynomials)
    }

    #[proptest]
    fn polynomial_segments_cohere_with_originating_polynomial(
        #[strategy(2_usize..8)] log_trace_length: usize,
        #[strategy(1_usize..#log_trace_length.min(3))] log_num_segments: usize,
        #[strategy(1_usize..6)] log_expansion_factor: usize,
        #[strategy(vec(arb(), (1 << #log_num_segments) * (1 << #log_trace_length)))]
        coefficients: Vec<XFieldElement>,
        #[strategy(arb())] random_point: XFieldElement,
    ) {
        let polynomial = Polynomial::new(coefficients);

        let num_segments = 1 << log_num_segments;
        let trace_length = 1 << log_trace_length;
        let expansion_factor = 1 << log_expansion_factor;

        let iota =
            BFieldElement::primitive_root_of_unity((trace_length * num_segments) as u64).unwrap();
        let psi = bfe!(7);
        let trace_domain = ArithmeticDomain::of_length(trace_length).unwrap();
        let fri_domain = ArithmeticDomain::of_length(trace_length * expansion_factor)
            .unwrap()
            .with_offset(psi);

        let multi_coset_values = (0..u32::try_from(num_segments).unwrap())
            .flat_map(|i| {
                let coset = trace_domain.with_offset(iota.mod_pow_u32(i) * psi);
                coset.evaluate(&polynomial)
            })
            .collect_vec();
        let multi_coset_values =
            Array2::from_shape_vec((trace_length, num_segments).f(), multi_coset_values).unwrap();

        let (actual_segment_codewords, segment_polynomials) =
            Stark::segmentify(multi_coset_values, psi, iota, trace_domain, fri_domain);

        let segments_evaluated = (0..)
            .zip(&segment_polynomials)
            .map(|(segment_index, segment_polynomial)| -> XFieldElement {
                let point_to_the_seg_idx = random_point.mod_pow_u32(segment_index);
                let point_to_the_num_seg = random_point.mod_pow_u32(num_segments as u32);
                let segment_in_point_to_the_num_seg =
                    segment_polynomial.evaluate_in_same_field(point_to_the_num_seg);
                point_to_the_seg_idx * segment_in_point_to_the_num_seg
            })
            .sum::<XFieldElement>();
        let evaluation_in_random_point = polynomial.evaluate_in_same_field(random_point);
        prop_assert_eq!(segments_evaluated, evaluation_in_random_point);

        let segments_codewords = segment_polynomials
            .iter()
            .flat_map(|polynomial| Array1::from(fri_domain.evaluate(polynomial)))
            .collect_vec();
        let segments_codewords =
            Array2::from_shape_vec((fri_domain.length, num_segments).f(), segments_codewords)
                .unwrap();
        prop_assert_eq!(segments_codewords, actual_segment_codewords);
    }

    #[proptest]
    fn linear_combination_weights_samples_correct_number_of_elements(
        #[strategy(arb())] mut proof_stream: ProofStream,
    ) {
        let weights = LinearCombinationWeights::sample(&mut proof_stream);

        prop_assert_eq!(MasterMainTable::NUM_COLUMNS, weights.main.len());
        prop_assert_eq!(MasterAuxTable::NUM_COLUMNS, weights.aux.len());
        prop_assert_eq!(NUM_QUOTIENT_SEGMENTS, weights.quot_segments.len());
        prop_assert_eq!(NUM_DEEP_CODEWORD_COMPONENTS, weights.deep.len());
        prop_assert_eq!(
            MasterMainTable::NUM_COLUMNS + MasterAuxTable::NUM_COLUMNS,
            weights.main_and_aux().len()
        );
    }

    /// A program that executes every instruction in the instruction set.
    fn program_executing_every_instruction() -> ProgramAndInput {
        let m_step_mem_addr = 100_000;

        let program = triton_program! {
            // merkle_step using the following fake tree:
            //     ─── 1 ───
            //    ╱         ╲
            //   2           3
            //  ╱  ╲
            // 4    5
            push {m_step_mem_addr}  // _ addr (address for `merkle_step_mem`)
            push 0                  // _ addr 0 (spacer)
            push 5                  // _ addr 0 5 (node index for `merkle_step`s)
            read_io 5               // _ addr 0 5 [digest; 5]
            merkle_step             // _ addr 0 2 [digest; 5]
            merkle_step_mem         // _ addr 0 1 [digest; 5]
            divine 5                // _ addr 0 1 [digest; 5] [digest; 5]
            assert_vector           // _ addr 0 1 [digest; 5]
            pop 5                   // _ addr 0 1
            assert                  // _ addr 0
            pop 2                   // _

            // stack manipulation
            push 1 push 2 push 3    // _  1  2  3
            place 2                 // _  3  1  2
            pick 1                  // _  3  2  1
            swap 2                  // _  1  2  3
            dup 2 assert            // _  1  2  3
            addi -2 assert          // _  1  2
            addi -1 assert          // _  1
            assert                  // _

            // dot_step
            push 0 push 0 push 0    // _ [accumulator; 3]
            push 500                // _ [accumulator; 3] addr_0
            push 800                // _ [accumulator; 3] addr_0 addr_1
            xb_dot_step             // _ [accumulator; 3] addr_0 addr_1
            xx_dot_step             // _ [accumulator; 3] addr_0 addr_1
            write_io 5              // _

            // extension field arithmetic
            push 1 push 2 push 3    // _ [xfe_0; 3]
            push 7 push 8 push 9    // _ [xfe_0; 3] [xfe_1; 3]
            dup 3 dup 3 dup 3       // _ [xfe_0; 3] [xfe_1; 3] [xfe_2; 3]
            xx_add                  // _ [xfe_0; 3] [xfe_1; 3]
            dup 4 dup 4 dup 4       // _ [xfe_0; 3] [xfe_1; 3] [xfe_2; 3]
            xx_mul                  // _ [xfe_0; 3] [xfe_1; 3]
            x_invert                // _ [xfe_0; 3] [xfe_1; 3]
            push 42                 // _ [xfe_0; 3] [xfe_1; 3] 42
            xb_mul                  // _ [xfe_0; 3] [xfe_1; 3]

            // base field arithmetic
            add mul                 // _ bfe_0 bfe_1 bfe_2 bfe_3
            addi 0                  // _ bfe_0 bfe_1 bfe_2 bfe_3
            invert                  // _ bfe_0 bfe_1 bfe_2 bfe_3
            mul add                 // _ bfe_0 bfe_1
            eq                      // _ bfe_0
            pop 1                   // _

            // bit-wise arithmetic
            push 38                 // _ 38
            push 2                  // _ 38 2
            pow                     // _ big_num
            push 1337               // _ big_num 1337
            add                     // _ big_num
            split                   // _ u32_0 u32_1
            dup 1 dup 1 lt pop 1    // _ u32_0 u32_1
            dup 1 and               // _ u32_0 u32_1
            dup 1 xor               // _ u32_0 u32_1
            push 9                  // _ u32_0 u32_1 9
            log_2_floor pop 1       // _ u32_0 u32_1
            div_mod                 // _ u32_0 u32_1
            pop_count               // _ u32_0 u32_1
            pop 2                   // _

            // Sponge
            sponge_init             // _
            divine 5 divine 5       // _ [stuff; 10]
            sponge_absorb           // _
            push 42                 // _ 42
            sponge_absorb_mem       // _ 52
            pop 1                   // _
            sponge_squeeze          // _ [stuff; 10]
            hash                    // _ [stuff; 5]
            pop 5                   // _

            // RAM
            push 300                // _ address
            read_mem 5              // _ [stuff; 5] address
            swap 6                  // _ [stuff; 5] address
            write_mem 5             // _ address
            pop 1                   // _

            // control flow
            push 0 skiz nop         // _
            push 1 skiz nop         // _
            push 0 push 2           // _ 0 2
            push 0 push 0 push 0    // _ 0 2 [0; 3]
            push 0 push 0           // _ 0 2 [0; 5]
            call rec_or_ret         // _ 0 0 [0; 5]
            pop 5 pop 2             // _
            push 2                  // _ 2
            call rec                // _ 0
            pop 1
            halt

            // BEFORE: _ n
            // AFTER:  _ 0
            rec:
                dup 0 push 0 eq     // _ n n==0
                skiz return         // _ n
                push -1 add         // _ n-1
                recurse             // _ n-1

            // BEFORE: _ m n [_; 5]
            // AFTER:  _ m m [_; 5]
            rec_or_ret:
                swap 5              // _ m [_; 5] n
                push -1 add         // _ m [_; 5] n-1
                swap 5              // _ m n-1 [_; 5]
                recurse_or_return   // _ m n-1 [_; 5]
        };

        let tree_node_5 = Digest::new(bfe_array![5; 5]);
        let tree_node_4 = Digest::new(bfe_array![4; 5]);
        let tree_node_3 = Digest::new(bfe_array![3; 5]);
        let tree_node_2 = Tip5::hash_pair(tree_node_4, tree_node_5);
        let tree_node_1 = Tip5::hash_pair(tree_node_2, tree_node_3);

        let public_input = tree_node_5.values().to_vec();

        let secret_input = [tree_node_1.reversed().values().to_vec(), bfe_vec![1337; 10]].concat();
        let mut ram = (0..)
            .zip(42..)
            .take(1_000)
            .map(|(l, r)| (bfe!(l), bfe!(r)))
            .collect::<HashMap<_, _>>();
        for (address, digest_element) in (m_step_mem_addr..).zip(tree_node_3.values()) {
            ram.insert(bfe!(address), digest_element);
        }
        let non_determinism = NonDeterminism::new(secret_input)
            .with_digests([tree_node_4])
            .with_ram(ram);

        ProgramAndInput::new(program)
            .with_input(public_input)
            .with_non_determinism(non_determinism)
    }

    #[test]
    fn program_executing_every_instruction_actually_executes_every_instruction() {
        let ProgramAndInput {
            program,
            public_input,
            non_determinism,
        } = program_executing_every_instruction();
        let (aet, _) = VM::trace_execution(&program, public_input, non_determinism).unwrap();
        let opcodes_of_all_executed_instructions = aet
            .processor_trace
            .column(ProcessorMainColumn::CI.main_index())
            .iter()
            .copied()
            .collect::<HashSet<_>>();

        let all_opcodes = Instruction::iter()
            .map(|instruction| instruction.opcode_b())
            .collect::<HashSet<_>>();

        all_opcodes
            .difference(&opcodes_of_all_executed_instructions)
            .for_each(|&opcode| {
                let instruction = Instruction::try_from(opcode).unwrap();
                eprintln!("Instruction {instruction} was not executed.");
            });
        assert_eq!(all_opcodes, opcodes_of_all_executed_instructions);
    }

    #[test]
    fn constraints_evaluate_to_zero_on_program_executing_every_instruction() {
        triton_constraints_evaluate_to_zero(program_executing_every_instruction());
    }
}
