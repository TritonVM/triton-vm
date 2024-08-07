use std::ops::Mul;
use std::ops::MulAssign;

use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use itertools::izip;
use itertools::Itertools;
use ndarray::prelude::*;
use ndarray::Zip;
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
use crate::error::ProvingError;
use crate::error::VerificationError;
use crate::fri;
use crate::fri::Fri;
use crate::profiler::profiler;
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

        profiler!(start "base tables");
        profiler!(start "create" ("gen"));
        let mut master_base_table =
            MasterBaseTable::new(aet, self.num_trace_randomizers, quotient_domain, fri.domain);
        profiler!(stop "create");

        profiler!(start "pad" ("gen"));
        master_base_table.pad();
        profiler!(stop "pad");

        profiler!(start "randomize trace" ("gen"));
        master_base_table.randomize_trace();
        profiler!(stop "randomize trace");

        profiler!(start "LDE" ("LDE"));
        master_base_table.low_degree_extend_all_columns();
        profiler!(stop "LDE");

        profiler!(start "Merkle tree" ("hash"));
        let base_merkle_tree = master_base_table.merkle_tree();
        profiler!(stop "Merkle tree");

        profiler!(start "Fiat-Shamir" ("hash"));
        proof_stream.enqueue(ProofItem::MerkleRoot(base_merkle_tree.root()));
        let challenges = proof_stream.sample_scalars(Challenges::SAMPLE_COUNT);
        let challenges = Challenges::new(challenges, claim);
        profiler!(stop "Fiat-Shamir");

        profiler!(start "extend" ("gen"));
        let mut master_ext_table = master_base_table.extend(&challenges);
        profiler!(stop "extend");
        profiler!(stop "base tables");

        profiler!(start "ext tables");
        profiler!(start "randomize trace" ("gen"));
        master_ext_table.randomize_trace();
        profiler!(stop "randomize trace");

        profiler!(start "LDE" ("LDE"));
        master_ext_table.low_degree_extend_all_columns();
        profiler!(stop "LDE");

        profiler!(start "Merkle tree" ("hash"));
        let ext_merkle_tree = master_ext_table.merkle_tree();
        profiler!(stop "Merkle tree");

        profiler!(start "Fiat-Shamir" ("hash"));
        proof_stream.enqueue(ProofItem::MerkleRoot(ext_merkle_tree.root()));

        // Get the weights with which to compress the many quotients into one.
        let quotient_combination_weights =
            proof_stream.sample_scalars(MasterExtTable::NUM_CONSTRAINTS);
        profiler!(stop "Fiat-Shamir");
        profiler!(stop "ext tables");

        let (fri_domain_quotient_segment_codewords, quotient_segment_polynomials) =
            Self::compute_quotient_segments(
                &master_base_table,
                &master_ext_table,
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
        let quot_merkle_tree: MerkleTree<Tip5> =
            CpuParallel::from_digests(&fri_domain_quotient_segment_codewords_digests)?;
        let quot_merkle_tree_root = quot_merkle_tree.root();
        proof_stream.enqueue(ProofItem::MerkleRoot(quot_merkle_tree_root));
        profiler!(stop "Merkle tree");

        debug_assert_eq!(fri.domain.length, quot_merkle_tree.num_leafs());

        profiler!(start "out-of-domain rows");
        let trace_domain_generator = master_base_table.trace_domain().generator;
        let out_of_domain_point_curr_row = proof_stream.sample_scalars(1)[0];
        let out_of_domain_point_next_row = trace_domain_generator * out_of_domain_point_curr_row;

        let ood_base_row = master_base_table.out_of_domain_row(out_of_domain_point_curr_row);
        let ood_base_row = MasterBaseTable::try_to_base_row(ood_base_row)?;
        proof_stream.enqueue(ProofItem::OutOfDomainBaseRow(Box::new(ood_base_row)));

        let ood_ext_row = master_ext_table.out_of_domain_row(out_of_domain_point_curr_row);
        let ood_ext_row = MasterExtTable::try_to_ext_row(ood_ext_row)?;
        proof_stream.enqueue(ProofItem::OutOfDomainExtRow(Box::new(ood_ext_row)));

        let ood_next_base_row = master_base_table.out_of_domain_row(out_of_domain_point_next_row);
        let ood_next_base_row = MasterBaseTable::try_to_base_row(ood_next_base_row)?;
        proof_stream.enqueue(ProofItem::OutOfDomainBaseRow(Box::new(ood_next_base_row)));

        let ood_next_ext_row = master_ext_table.out_of_domain_row(out_of_domain_point_next_row);
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
        profiler!(start "base" ("CC"));
        let base_combination_polynomial =
            Self::random_linear_sum(master_base_table.interpolation_polynomials(), weights.base);

        profiler!(stop "base");
        profiler!(start "ext" ("CC"));
        let ext_combination_polynomial =
            Self::random_linear_sum(master_ext_table.interpolation_polynomials(), weights.ext);
        profiler!(stop "ext");
        let base_and_ext_combination_polynomial =
            base_combination_polynomial + ext_combination_polynomial;
        let base_and_ext_codeword = short_domain.evaluate(&base_and_ext_combination_polynomial);

        profiler!(start "quotient" ("CC"));
        let quotient_segments_combination_polynomial =
            Self::random_linear_sum(quotient_segment_polynomials.view(), weights.quot_segments);
        let quotient_segments_combination_codeword =
            short_domain.evaluate(&quotient_segments_combination_polynomial);
        profiler!(stop "quotient");

        profiler!(stop "linear combination");

        profiler!(start "DEEP");
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
        profiler!(start "base&ext curr row");
        let out_of_domain_curr_row_base_and_ext_value =
            base_and_ext_combination_polynomial.evaluate(out_of_domain_point_curr_row);
        let base_and_ext_curr_row_deep_codeword = Self::deep_codeword(
            &base_and_ext_codeword.clone(),
            short_domain,
            out_of_domain_point_curr_row,
            out_of_domain_curr_row_base_and_ext_value,
        );
        profiler!(stop "base&ext curr row");

        profiler!(start "base&ext next row");
        let out_of_domain_next_row_base_and_ext_value =
            base_and_ext_combination_polynomial.evaluate(out_of_domain_point_next_row);
        let base_and_ext_next_row_deep_codeword = Self::deep_codeword(
            &base_and_ext_codeword.clone(),
            short_domain,
            out_of_domain_point_next_row,
            out_of_domain_next_row_base_and_ext_value,
        );
        profiler!(stop "base&ext next row");

        profiler!(start "segmented quotient");
        let out_of_domain_curr_row_quot_segments_value = quotient_segments_combination_polynomial
            .evaluate(out_of_domain_point_curr_row_pow_num_segments);
        let quotient_segments_curr_row_deep_codeword = Self::deep_codeword(
            &quotient_segments_combination_codeword.clone(),
            short_domain,
            out_of_domain_point_curr_row_pow_num_segments,
            out_of_domain_curr_row_quot_segments_value,
        );
        profiler!(stop "segmented quotient");
        profiler!(stop "DEEP");

        profiler!(start "combined DEEP polynomial");
        profiler!(start "sum" ("CC"));
        let deep_codeword = [
            base_and_ext_curr_row_deep_codeword,
            base_and_ext_next_row_deep_codeword,
            quotient_segments_curr_row_deep_codeword,
        ]
        .into_par_iter()
        .zip_eq(weights.deep.to_vec())
        .map(|(codeword, weight)| codeword.into_par_iter().map(|c| c * weight).collect())
        .reduce(
            || vec![XFieldElement::zero(); short_domain.length],
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
        let revealed_base_elems =
            if let Some(fri_domain_table) = master_base_table.fri_domain_table() {
                Self::read_revealed_rows(fri_domain_table, &revealed_current_row_indices)?
            } else {
                Self::recompute_revealed_rows::<NUM_BASE_COLUMNS, BFieldElement>(
                    &master_base_table.interpolation_polynomials(),
                    &revealed_current_row_indices,
                    fri.domain,
                )
            };
        let base_authentication_structure =
            base_merkle_tree.authentication_structure(&revealed_current_row_indices)?;
        proof_stream.enqueue(ProofItem::MasterBaseTableRows(revealed_base_elems));
        proof_stream.enqueue(ProofItem::AuthenticationStructure(
            base_authentication_structure,
        ));

        let revealed_ext_elems = if let Some(fri_domain_table) = master_ext_table.fri_domain_table()
        {
            Self::read_revealed_rows(fri_domain_table, &revealed_current_row_indices)?
        } else {
            Self::recompute_revealed_rows(
                &master_ext_table.interpolation_polynomials(),
                &revealed_current_row_indices,
                fri.domain,
            )
        };
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
        profiler!(stop "open trace leafs");

        Ok(proof_stream.into())
    }

    fn compute_quotient_segments(
        master_base_table: &MasterBaseTable,
        master_ext_table: &MasterExtTable,
        fri_domain: ArithmeticDomain,
        quotient_domain: ArithmeticDomain,
        challenges: &Challenges,
        quotient_combination_weights: &[XFieldElement],
    ) -> (Array2<XFieldElement>, Array1<Polynomial<XFieldElement>>) {
        let calculate_quotients_with_just_in_time_low_degree_extension = || {
            profiler!(start "quotient calculation (just-in-time)");
            let (fri_domain_quotient_segment_codewords, quotient_segment_polynomials) =
                Self::compute_quotient_segments_with_jit_lde(
                    master_base_table.interpolation_polynomials(),
                    master_ext_table.interpolation_polynomials(),
                    master_base_table.trace_domain(),
                    master_base_table.randomized_trace_domain(),
                    master_base_table.fri_domain(),
                    challenges,
                    quotient_combination_weights,
                );
            profiler!(stop "quotient calculation (just-in-time)");
            (
                fri_domain_quotient_segment_codewords,
                quotient_segment_polynomials,
            )
        };

        let Some(base_quotient_domain_codewords) = master_base_table.quotient_domain_table() else {
            return calculate_quotients_with_just_in_time_low_degree_extension();
        };
        let Some(ext_quotient_domain_codewords) = master_ext_table.quotient_domain_table() else {
            return calculate_quotients_with_just_in_time_low_degree_extension();
        };

        profiler!(start "quotient calculation (cached)" ("CC"));
        let quotient_codeword = all_quotients_combined(
            base_quotient_domain_codewords,
            ext_quotient_domain_codewords,
            master_base_table.trace_domain(),
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

    /// # Panics
    ///
    /// Panics if the number of polynomials and weights are not equal.
    fn random_linear_sum<FF>(
        polynomials: ArrayView1<Polynomial<FF>>,
        weights: Array1<XFieldElement>,
    ) -> Polynomial<XFieldElement>
    where
        FF: FiniteField + Mul<XFieldElement, Output = XFieldElement>,
    {
        assert_eq!(polynomials.len(), weights.len());

        let random_linear_sum = (0..polynomials[0].coefficients.len())
            .into_par_iter()
            .map(|i| {
                polynomials
                    .axis_iter(Axis(0))
                    .zip(&weights)
                    .map(|(poly, &w)| poly[()].coefficients[i] * w)
                    .sum()
            })
            .collect();
        Polynomial::new(random_linear_sum)

        // todo: replace by
        //  ```
        //  Zip::from(polynomials)
        //      .and(&weights)
        //      .fold(Polynomial::zero(), |acc, poly, &w| acc + poly.scalar_mul(w))
        //  ```
        //  (and maybe alter trait bounds) once `twenty-first` v0.42.0 is released.
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

    /// Read the indicated rows from the cached table. The indices come from FRI.
    fn read_revealed_rows<const N: usize, FF: FiniteField>(
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

    /// Recompute and return the rows indicated by FRI.
    ///
    /// This function performs fast multi-point evaluation in parallel over the
    /// columns.
    ///
    /// Parameters:
    ///  - `W : const usize` -- the width of the table, in number of field elements
    ///  - `FF : {BFieldElement, XFieldElement}`
    ///  - `table_as_interpolation_polynomials : &[Polynomial<FF>]` -- the table as
    ///    a slice of X- or B-FieldElements polynomials, one for each column
    ///  - `revealed_indices: &[usize]` -- the indices coming from FRI
    ///  - `trace_domain : ArithmeticDomain` -- the domain over which the trace is
    ///    interpolated; the trace domain informs this function how the coefficients
    ///    in the table are to be interpreted
    ///  - `fri_domain : ArithmeticDomain` -- the domain over which FRI is done; the
    ///    FRI domain is used to determine which indeterminates the given indices
    ///    correspond to.
    ///
    /// Returns:
    ///  - `rows : Vec<[FF; W]>` -- a `Vec` of arrays of `W` field elements each;
    ///    one array per queried index.
    fn recompute_revealed_rows<
        const W: usize,
        FF: FiniteField + From<BFieldElement> + MulAssign<BFieldElement>,
    >(
        table_as_interpolation_polynomials: &ArrayView1<Polynomial<FF>>,
        revealed_indices: &[usize],
        fri_domain: ArithmeticDomain,
    ) -> Vec<[FF; W]> {
        // obtain the evaluation points from the FRI domain
        let indeterminates = revealed_indices
            .iter()
            .map(|i| fri_domain.domain_value(*i as u32))
            .map(FF::from)
            .collect_vec();

        // for every column (in parallel), fast multi-point evaluate
        let columns = table_as_interpolation_polynomials
            .into_par_iter()
            .flat_map(|poly| poly.batch_evaluate(&indeterminates))
            .collect::<Vec<_>>();

        // transpose the resulting matrix out-of-place
        let n = revealed_indices.len();
        let mut rows = vec![FF::zero(); W * n];
        for i in 0..W {
            for j in 0..n {
                rows[j * W + i] = columns[i * n + j];
            }
        }

        rows.chunks(W)
            .map(|ch| ch.try_into().unwrap())
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
        let base_merkle_tree_root = proof_stream.dequeue()?.try_into_merkle_root()?;
        let extension_challenge_weights = proof_stream.sample_scalars(Challenges::SAMPLE_COUNT);
        let challenges = Challenges::new(extension_challenge_weights, claim);
        let extension_tree_merkle_root = proof_stream.dequeue()?.try_into_merkle_root()?;
        // Sample weights for quotient codeword, which is a part of the combination codeword.
        // See corresponding part in the prover for a more detailed explanation.
        let quot_codeword_weights = proof_stream.sample_scalars(MasterExtTable::NUM_CONSTRAINTS);
        let quot_codeword_weights = Array1::from(quot_codeword_weights);
        let quotient_codeword_merkle_root = proof_stream.dequeue()?.try_into_merkle_root()?;
        profiler!(stop "Fiat-Shamir 1");

        profiler!(start "dequeue ood point and rows" ("hash"));
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
        profiler!(stop "dequeue ood point and rows");

        profiler!(start "out-of-domain quotient element");
        profiler!(start "evaluate AIR" ("AIR"));
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
        let base_and_ext_codeword_weights = weights.base_and_ext();
        profiler!(stop "Fiat-Shamir 2");

        profiler!(start "sum out-of-domain values" ("CC"));
        let out_of_domain_curr_row_base_and_ext_value = Self::linearly_sum_base_and_ext_row(
            out_of_domain_curr_base_row.view(),
            out_of_domain_curr_ext_row.view(),
            base_and_ext_codeword_weights.view(),
        );
        let out_of_domain_next_row_base_and_ext_value = Self::linearly_sum_base_and_ext_row(
            out_of_domain_next_base_row.view(),
            out_of_domain_next_ext_row.view(),
            base_and_ext_codeword_weights.view(),
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
        profiler!(start "dequeue base elements");
        let base_table_rows = proof_stream.dequeue()?.try_into_master_base_table_rows()?;
        let base_authentication_structure = proof_stream
            .dequeue()?
            .try_into_authentication_structure()?;
        let leaf_digests_base: Vec<_> = base_table_rows
            .par_iter()
            .map(|revealed_base_elem| Tip5::hash_varlen(revealed_base_elem))
            .collect();
        profiler!(stop "dequeue base elements");

        let index_leaves = |leaves| {
            let index_iter = revealed_current_row_indices.iter().copied();
            index_iter.zip_eq(leaves).collect()
        };
        profiler!(start "Merkle verify (base tree)" ("hash"));
        let base_merkle_tree_inclusion_proof = MerkleTreeInclusionProof::<Tip5> {
            tree_height: merkle_tree_height,
            indexed_leafs: index_leaves(leaf_digests_base),
            authentication_structure: base_authentication_structure,
            ..MerkleTreeInclusionProof::default()
        };
        if !base_merkle_tree_inclusion_proof.verify(base_merkle_tree_root) {
            return Err(VerificationError::BaseCodewordAuthenticationFailure);
        }
        profiler!(stop "Merkle verify (base tree)");

        profiler!(start "dequeue extension elements");
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
        profiler!(stop "dequeue extension elements");

        profiler!(start "Merkle verify (extension tree)" ("hash"));
        let ext_merkle_tree_inclusion_proof = MerkleTreeInclusionProof::<Tip5> {
            tree_height: merkle_tree_height,
            indexed_leafs: index_leaves(leaf_digests_ext),
            authentication_structure: ext_authentication_structure,
            ..MerkleTreeInclusionProof::default()
        };
        if !ext_merkle_tree_inclusion_proof.verify(extension_tree_merkle_root) {
            return Err(VerificationError::ExtensionCodewordAuthenticationFailure);
        }
        profiler!(stop "Merkle verify (extension tree)");

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
        let quot_merkle_tree_inclusion_proof = MerkleTreeInclusionProof::<Tip5> {
            tree_height: merkle_tree_height,
            indexed_leafs: index_leaves(revealed_quotient_segments_digests),
            authentication_structure: revealed_quotient_authentication_structure,
            ..MerkleTreeInclusionProof::default()
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
        if self.num_collinearity_checks != base_table_rows.len() {
            return Err(VerificationError::IncorrectNumberOfBaseTableRows);
        };
        if self.num_collinearity_checks != ext_table_rows.len() {
            return Err(VerificationError::IncorrectNumberOfExtTableRows);
        };

        for (row_idx, base_row, ext_row, quotient_segments_elements, fri_value) in izip!(
            revealed_current_row_indices,
            base_table_rows,
            ext_table_rows,
            revealed_quotient_segments_elements,
            revealed_fri_values,
        ) {
            let base_row = Array1::from(base_row.to_vec());
            let ext_row = Array1::from(ext_row.to_vec());
            let current_fri_domain_value = fri.domain.domain_value(row_idx as u32);

            profiler!(start "base & ext elements" ("CC"));
            let base_and_ext_curr_row_element = Self::linearly_sum_base_and_ext_row(
                base_row.view(),
                ext_row.view(),
                base_and_ext_codeword_weights.view(),
            );
            let quotient_segments_curr_row_element = weights
                .quot_segments
                .dot(&Array1::from(quotient_segments_elements.to_vec()));
            profiler!(stop "base & ext elements");

            profiler!(start "DEEP update");
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
            profiler!(stop "DEEP update");

            profiler!(start "combination codeword equality");
            let deep_value_components = Array1::from(vec![
                base_and_ext_curr_row_deep_value,
                base_and_ext_next_row_deep_value,
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

    fn linearly_sum_base_and_ext_row<FF>(
        base_row: ArrayView1<FF>,
        ext_row: ArrayView1<XFieldElement>,
        weights: ArrayView1<XFieldElement>,
    ) -> XFieldElement
    where
        FF: FiniteField + Into<XFieldElement>,
        XFieldElement: Mul<FF, Output = XFieldElement>,
    {
        profiler!(start "collect");
        let mut row = base_row.map(|&element| element.into());
        row.append(Axis(0), ext_row).unwrap();
        profiler!(stop "collect");
        profiler!(start "inner product");
        // todo: Try to get rid of this clone. The alternative line
        //   `let base_and_ext_element = (&weights * &summands).sum();`
        //   without cloning the weights does not compile for a seemingly nonsensical reason.
        let weights = weights.to_owned();
        let base_and_ext_element = (weights * row).sum();
        profiler!(stop "inner product");
        base_and_ext_element
    }

    /// Computes the quotient segments in a memory-friendly way, i.e., without ever
    /// representing the entire low-degree extended trace. Instead, the trace is
    /// extrapolated over cosets of the trace domain, and the quotients are computed
    /// there. The resulting coset-quotients are linearly recombined to produce the
    /// quotient segment codewords.
    fn compute_quotient_segments_with_jit_lde(
        main_polynomials: ArrayView1<Polynomial<BFieldElement>>,
        aux_polynomials: ArrayView1<Polynomial<XFieldElement>>,
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
        let mut quotient_multicoset_evaluations = Array2::zeros([num_rows, NUM_QUOTIENT_SEGMENTS]);
        let mut main_columns = Array2::zeros([num_rows, main_polynomials.len()]);
        let mut aux_columns = Array2::zeros([num_rows, aux_polynomials.len()]);
        for (coset_index, quotient_column) in (0..u64::try_from(NUM_QUOTIENT_SEGMENTS).unwrap())
            .zip(quotient_multicoset_evaluations.columns_mut())
        {
            // always also offset by fri domain offset to avoid division-by-zero errors
            let domain = domain.with_offset(iota.mod_pow(coset_index) * fri_domain.offset);
            Zip::from(main_polynomials)
                .and(main_columns.axis_iter_mut(Axis(1)))
                .into_par_iter()
                .for_each(|(polynomial, column)| {
                    Array1::from(domain.evaluate(polynomial)).move_into(column)
                });
            Zip::from(aux_polynomials)
                .and(aux_columns.axis_iter_mut(Axis(1)))
                .into_par_iter()
                .for_each(|(polynomial, column)| {
                    Array1::from(domain.evaluate(polynomial)).move_into(column)
                });
            Array1::from(all_quotients_combined(
                main_columns.view(),
                aux_columns.view(),
                trace_domain,
                domain,
                challenges,
                quotient_combination_weights,
            ))
            .move_into(quotient_column);
        }

        Self::segmentify(
            quotient_multicoset_evaluations,
            fri_domain.offset,
            iota,
            randomized_trace_domain,
            fri_domain,
        )
    }

    /// Map a matrix whose columns represent the evaluation of a high-degree
    /// polynomial on all cosets of the trace domain, to
    /// 1. a matrix of segment codewords (on the FRI domain), and
    /// 2. an array of matching segment polynomials,
    ///
    /// such that the segment polynomials correspond to the interleaving split of
    /// the original high-degree polynomial.
    ///
    /// For examnple, let f(X) have degree 2N where N is the trace domain length.
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
        //     (      ·       )   ( (    ·                )   (     ·                ) )
        //     ( ·  ξ^(l·k) · ) · ( ( ψ^k · ι^(j·k+i·k·F) ) o ( q_k(ψ^F · ω^(j+i·F)) ) )
        //     (      ·       )   ( (    ·                )   (     ·                ) )
        //  =
        //     (      ·                            )   (    ·                 )
        //     ( ·  ψ^k · ι^(j·k+i·k·F+l·k·N/F)  · ) · ( q_k(ψ^F · ω^(j+i·F)) )
        //     (      ·                            )   (    ·                 )
        //  =
        //     (      ·                       )
        //     ( q(ψ · ι^j · ω^(i + l · N/F)) )
        //     (      ·                       )

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
    use std::collections::HashSet;

    use assert2::assert;
    use assert2::check;
    use assert2::let_assert;
    use itertools::izip;
    use num_traits::Zero;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use proptest_arbitrary_interop::arb;
    use rand::thread_rng;
    use rand::Rng;
    use strum::EnumCount;
    use strum::IntoEnumIterator;
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
    use crate::table::challenges::ChallengeId::StandardInputIndeterminate;
    use crate::table::challenges::ChallengeId::StandardOutputIndeterminate;
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
        let stark = low_security_stark(DEFAULT_LOG2_FRI_EXPANSION_FACTOR_FOR_TESTS);
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
            let result = program.run(PublicInput::default(), NonDeterminism::default());
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
    check_constraints_fn!(fn check_cross_table_constraints for GrandCrossTableArg);

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
        check_cross_table_constraints(mbt, met, &challenges);
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
        let program_and_input = ProgramAndInput::new(FIBONACCI_SEQUENCE.clone())
            .with_input(PublicInput::from(bfe_array![100]));
        prove_and_verify(
            program_and_input,
            DEFAULT_LOG2_FRI_EXPANSION_FACTOR_FOR_TESTS,
        );
    }

    #[test]
    fn constraints_evaluate_to_zero_on_many_u32_operations() {
        let many_u32_instructions =
            ProgramAndInput::new(PROGRAM_WITH_MANY_U32_INSTRUCTIONS.clone());
        triton_constraints_evaluate_to_zero(many_u32_instructions);
    }

    #[test]
    fn prove_verify_many_u32_operations() {
        prove_and_verify(
            ProgramAndInput::new(PROGRAM_WITH_MANY_U32_INSTRUCTIONS.clone()),
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

    #[proptest]
    fn quotient_segments_of_old_and_new_methods_are_identical(
        #[strategy(2_usize..8)] _log_trace_length: usize,
        #[strategy(Just(1 << #_log_trace_length))] trace_length: usize,
        #[strategy(Just(2 * #trace_length))] randomized_trace_length: usize,
        #[strategy(arb())]
        #[filter(!#offset.is_zero())]
        offset: BFieldElement,
        #[strategy(arb())] main_polynomials: [Polynomial<BFieldElement>; NUM_BASE_COLUMNS],
        #[strategy(arb())] aux_polynomials: [Polynomial<XFieldElement>; NUM_EXT_COLUMNS],
        #[strategy(arb())] challenges: Challenges,
        #[strategy(arb())] quotient_weights: [XFieldElement; MasterExtTable::NUM_CONSTRAINTS],
    ) {
        // set up
        let main_polynomials = Array1::from_vec(main_polynomials.to_vec());
        let aux_polynomials = Array1::from_vec(aux_polynomials.to_vec());

        let trace_domain = ArithmeticDomain::of_length(trace_length).unwrap();
        let randomized_trace_domain = ArithmeticDomain::of_length(randomized_trace_length).unwrap();
        let fri_domain = ArithmeticDomain::of_length(4 * randomized_trace_length).unwrap();
        let fri_domain = fri_domain.with_offset(offset);
        let quotient_domain = ArithmeticDomain::of_length(4 * randomized_trace_length).unwrap();
        let quotient_domain = quotient_domain.with_offset(offset);

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
                main_polynomials.view(),
                aux_polynomials.view(),
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
        let mut base_quotient_domain_codewords =
            Array2::<BFieldElement>::zeros([quotient_domain.length, NUM_BASE_COLUMNS]);
        Zip::from(base_quotient_domain_codewords.axis_iter_mut(Axis(1)))
            .and(main_polynomials.axis_iter(Axis(0)))
            .for_each(|codeword, polynomial| {
                Array1::from_vec(quotient_domain.evaluate(&polynomial[()])).move_into(codeword);
            });
        let mut ext_quotient_domain_codewords =
            Array2::<XFieldElement>::zeros([quotient_domain.length, NUM_EXT_COLUMNS]);
        Zip::from(ext_quotient_domain_codewords.axis_iter_mut(Axis(1)))
            .and(aux_polynomials.axis_iter(Axis(0)))
            .for_each(|codeword, polynomial| {
                Array1::from_vec(quotient_domain.evaluate(&polynomial[()])).move_into(codeword);
            });

        let quotient_codeword = all_quotients_combined(
            base_quotient_domain_codewords.view(),
            ext_quotient_domain_codewords.view(),
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
    fn polynomial_segments_are_coherent_with_the_polynomial_they_originate_from(
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
            .map(|(segment_index, segment_polynomial)| {
                let point_to_the_seg_idx = random_point.mod_pow_u32(segment_index);
                let point_to_the_num_seg = random_point.mod_pow_u32(num_segments as u32);
                point_to_the_seg_idx * segment_polynomial.evaluate(point_to_the_num_seg)
            })
            .sum::<XFieldElement>();
        prop_assert_eq!(segments_evaluated, polynomial.evaluate(random_point));

        let segments_codewords = segment_polynomials
            .iter()
            .flat_map(|polynomial| Array1::from(fri_domain.evaluate(polynomial)))
            .collect_vec();
        let segments_codewords =
            Array2::from_shape_vec((fri_domain.length, num_segments).f(), segments_codewords)
                .unwrap();
        prop_assert_eq!(segments_codewords, actual_segment_codewords);
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

    /// A program that executes every instruction in the instruction set.
    fn program_executing_every_instruction() -> ProgramAndInput {
        let program = triton_program! {
            // merkle_step using the following fake tree:
            //     ─── 1 ───
            //    ╱         ╲
            //   2           3
            //  ╱  ╲
            // 4    5
            push 5                  // _ 5 (node index for `merkle_step`)
            read_io 5               // _ 5 [digest; 5]
            merkle_step             // _ 2 [digest; 5]
            merkle_step             // _ 1 [digest; 5]
            divine 5                // _ 1 [digest; 5] [digest; 5]
            assert_vector           // _ 1 [digest; 5]
            pop 5                   // _ 1
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
        let dummy_ram = (0..)
            .zip(42..)
            .take(1_000)
            .map(|(l, r)| (bfe!(l), bfe!(r)))
            .collect::<HashMap<_, _>>();
        let non_determinism = NonDeterminism::new(secret_input)
            .with_digests([tree_node_4, tree_node_3])
            .with_ram(dummy_ram);

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
        let (aet, _) = program
            .trace_execution(public_input, non_determinism)
            .unwrap();
        let opcodes_of_all_executed_instructions = aet
            .processor_trace
            .column(ProcessorBaseTableColumn::CI.base_table_index())
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
