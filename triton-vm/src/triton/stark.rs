use std::collections::HashMap;
use std::error::Error;

use super::super::triton;
use super::table::base_matrix::BaseMatrices;
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::mpolynomial::Degree;
use crate::shared_math::other;
use crate::shared_math::polynomial::Polynomial;
use crate::shared_math::rescue_prime_xlix::{
    self, RescuePrimeXlix, RP_DEFAULT_OUTPUT_SIZE, RP_DEFAULT_WIDTH,
};
use crate::shared_math::stark::stark_verify_error::StarkVerifyError;
use crate::shared_math::stark::triton::arguments::evaluation_argument::verify_evaluation_argument;
use crate::shared_math::stark::triton::arguments::permutation_argument::PermArg;
use crate::shared_math::stark::triton::proof_item::{Item, StarkProofStream};
use crate::shared_math::stark::triton::state::DIGEST_LEN;
use crate::shared_math::stark::triton::table::challenges_endpoints::{AllChallenges, AllEndpoints};
use crate::shared_math::stark::triton::table::table_collection::{
    BaseTableCollection, ExtTableCollection, NUM_TABLES,
};
use crate::shared_math::stark::triton::triton_xfri;
use crate::shared_math::traits::{
    GetPrimitiveRootOfUnity, GetRandomElements, Inverse, ModPowU32, PrimeField,
};
use crate::shared_math::x_field_element::XFieldElement;
use crate::timing_reporter::TimingReporter;
use crate::util_types::merkle_tree::MerkleTree;
use crate::util_types::simple_hasher::{Hasher, ToDigest};
use itertools::Itertools;
use rand::{thread_rng, Rng};
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};

type BWord = BFieldElement;
type XWord = XFieldElement;
type StarkHasher = RescuePrimeXlix<RP_DEFAULT_WIDTH>;

// We use a type-parameterised FriDomain to avoid duplicate `b_*()` and `x_*()` methods.
pub struct Stark {
    num_trace_randomizers: usize,
    num_randomizer_polynomials: usize,
    security_level: usize,
    max_degree: Degree,
    bfri_domain: triton::fri_domain::FriDomain<BWord>,
    xfri: triton_xfri::Fri<StarkHasher>,
    input_symbols: Vec<BWord>,
    output_symbols: Vec<BWord>,
}

impl Stark {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        max_height: usize,
        num_trace_randomizers: usize,
        num_randomizer_polynomials: usize,
        log_expansion_factor: usize,
        security_level: usize,
        co_set_fri_offset: BWord,
        input_symbols: &[BWord],
        output_symbols: &[BWord],
    ) -> Self {
        assert_eq!(
            0,
            security_level % log_expansion_factor,
            "security_level/log_expansion_factor must be a positive integer"
        );

        let expansion_factor = 1 << log_expansion_factor;
        let colinearity_checks = security_level / log_expansion_factor;

        assert!(
            colinearity_checks > 0,
            "At least one colinearity check is required"
        );

        assert!(
            expansion_factor >= 4,
            "expansion factor must be at least 4."
        );

        let empty_table_collection = ExtTableCollection::with_padded_heights(
            num_trace_randomizers,
            &[max_height; NUM_TABLES],
        );

        let max_degree =
            (other::roundup_npo2(empty_table_collection.max_degree() as u64) - 1) as i64;
        let fri_domain_length = ((max_degree as u64 + 1) * expansion_factor) as usize;

        let omega = BWord::ring_zero()
            .get_primitive_root_of_unity(fri_domain_length as u64)
            .0
            .unwrap();

        let bfri_domain = triton::fri_domain::FriDomain {
            offset: co_set_fri_offset,
            omega,
            length: fri_domain_length as usize,
        };

        let xfri = triton_xfri::Fri::new(
            co_set_fri_offset.lift(),
            omega.lift(),
            fri_domain_length,
            expansion_factor as usize,
            colinearity_checks,
        );

        Stark {
            num_trace_randomizers,
            num_randomizer_polynomials,
            security_level,
            max_degree,
            bfri_domain,
            xfri,
            input_symbols: input_symbols.to_vec(),
            output_symbols: output_symbols.to_vec(),
        }
    }

    pub fn prove(&self, base_matrices: BaseMatrices) -> StarkProofStream {
        let mut timer = TimingReporter::start();

        let base_tables = self.get_padded_base_tables(&base_matrices);
        timer.elapsed("pad");

        let (x_rand_codeword, b_rand_codewords) = self.get_randomizer_codewords();
        timer.elapsed("randomizer_codewords");

        let base_codeword_tables = base_tables.codeword_tables(&self.bfri_domain);
        let base_codewords = base_codeword_tables.get_all_base_columns();
        let all_base_codewords = vec![b_rand_codewords, base_codewords.clone()].concat();
        timer.elapsed("get_all_base_codewords");

        let transposed_base_codewords = Self::transpose_codewords(&all_base_codewords);
        timer.elapsed("transposed_base_codewords");

        let hasher = rescue_prime_xlix::neptune_params();
        let base_tree = Self::get_merkle_tree(&hasher, &transposed_base_codewords);
        let base_merkle_tree_root = base_tree.get_root();
        timer.elapsed("base_merkle_tree");

        // Commit to base codewords
        let mut proof_stream = StarkProofStream::default();
        proof_stream.enqueue(&Item::MerkleRoot(base_merkle_tree_root));
        timer.elapsed("proof_stream.enqueue");

        let extension_challenge_seed = proof_stream.prover_fiat_shamir();
        timer.elapsed("prover_fiat_shamir");

        let extension_challenge_weights =
            hasher.sample_n_weights(&extension_challenge_seed, AllChallenges::TOTAL_CHALLENGES);
        let extension_challenges = AllChallenges::create_challenges(&extension_challenge_weights);
        timer.elapsed("challenges");

        let initials_seed = Self::get_initials_seed();
        let initial_weights =
            hasher.sample_n_weights(&initials_seed, AllEndpoints::TOTAL_ENDPOINTS);
        let all_initials = AllEndpoints::create_initials(&initial_weights);

        timer.elapsed("initials");

        let (ext_tables, all_terminals) =
            ExtTableCollection::extend_tables(&base_tables, &extension_challenges, &all_initials);
        timer.elapsed("extend + get_terminals");

        let padded_heights = (&ext_tables)
            .into_iter()
            .map(|ext_table| BWord::new(ext_table.padded_height() as u64))
            .collect_vec();
        proof_stream.enqueue(&Item::PaddedHeights(padded_heights));
        timer.elapsed("Sent all padded heights");

        let ext_codeword_tables =
            ext_tables.codeword_tables(&self.xfri.domain, base_codeword_tables);
        let extension_codewords = ext_codeword_tables.get_all_extension_columns();
        timer.elapsed("Calculated extension codewords");

        let transposed_ext_codewords = Self::transpose_codewords(&extension_codewords);

        let extension_tree = Self::get_extension_merkle_tree(&hasher, &transposed_ext_codewords);

        proof_stream.enqueue(&Item::MerkleRoot(extension_tree.get_root()));
        proof_stream.enqueue(&Item::Terminals(all_terminals.clone()));
        timer.elapsed("extension_tree");

        let base_degree_bounds = base_tables.get_base_column_degree_bounds();
        timer.elapsed("Calculated base degree bounds");

        let extension_degree_bounds = ext_tables.get_extension_column_degree_bounds();
        timer.elapsed("Calculated extension degree bounds");

        let mut quotient_codewords = ext_codeword_tables.get_all_quotients(
            &self.xfri.domain,
            &extension_challenges,
            &all_terminals,
        );
        timer.elapsed("Calculated quotient codewords");

        let mut quotient_degree_bounds = ext_codeword_tables
            .get_all_quotient_degree_bounds(&extension_challenges, &all_terminals);
        timer.elapsed("Calculated quotient degree bounds");

        // Prove equal initial values for the permutation-extension column pairs
        for pa in PermArg::all_permutation_arguments().iter() {
            quotient_codewords.push(pa.quotient(&ext_codeword_tables, &self.xfri.domain));
            quotient_degree_bounds.push(pa.quotient_degree_bound(&ext_codeword_tables));
        }

        let non_lin_combi_weights_seed = proof_stream.prover_fiat_shamir();
        timer.elapsed("Fiat-Shamir to get seed for sampling non-linear-combination weights");

        let non_lin_combi_weights_count = self.num_randomizer_polynomials
            + 2 * base_codewords.len()
            + 2 * extension_codewords.len()
            + 2 * quotient_degree_bounds.len()
            + 2 * PermArg::all_permutation_arguments().len();
        let non_lin_combi_weights =
            hasher.sample_n_weights(&non_lin_combi_weights_seed, non_lin_combi_weights_count);
        timer.elapsed("Sampled weights for non-linear combination");

        let combination_codeword = self.create_combination_codeword(
            &mut timer,
            vec![x_rand_codeword],
            base_codewords,
            extension_codewords,
            quotient_codewords,
            non_lin_combi_weights,
            base_degree_bounds,
            extension_degree_bounds,
            quotient_degree_bounds,
            &ext_tables,
        );
        timer.elapsed("non-linear sum");

        // TODO use Self::get_extension_merkle_tree (or similar) here?
        let mut combination_codeword_digests: Vec<Vec<BFieldElement>> =
            Vec::with_capacity(combination_codeword.len());
        combination_codeword
            .clone()
            .into_par_iter()
            .map(|xfe| {
                let digest: Vec<BFieldElement> = xfe.to_digest();
                hasher.hash(&digest, RP_DEFAULT_OUTPUT_SIZE)
            })
            .collect_into_vec(&mut combination_codeword_digests);
        let combination_tree =
            MerkleTree::<StarkHasher>::from_digests(&combination_codeword_digests);
        let combination_root: Vec<BFieldElement> = combination_tree.get_root();

        proof_stream.enqueue(&Item::MerkleRoot(combination_root.clone()));

        timer.elapsed("combination_tree");

        // Get indices of slices that go across codewords to prove nonlinear combination
        let indices_seed = proof_stream.prover_fiat_shamir();
        let cross_codeword_slice_indices =
            hasher.sample_indices(self.security_level, &indices_seed, self.xfri.domain.length);

        timer.elapsed("sample_indices");

        match self.xfri.prove(&combination_codeword, &mut proof_stream) {
            Ok((_, fri_first_round_merkle_root)) => assert_eq!(
                combination_root, fri_first_round_merkle_root,
                "Combination root from STARK and from FRI must agree."
            ),
            Err(e) => panic!("The FRI prover failed because of: {}", e),
        }
        timer.elapsed("fri.prove");

        let unit_distances = self.get_unit_distances(&ext_tables);
        timer.elapsed("unit_distances");

        // Open leafs of zipped codewords at indicated positions
        let revealed_indices =
            self.get_revealed_indices(&unit_distances, &cross_codeword_slice_indices);

        let revealed_base_elems =
            Self::get_revealed_elements(&transposed_base_codewords, &revealed_indices);
        let auth_paths_base = base_tree.get_multi_proof(&revealed_indices);
        proof_stream.enqueue(&Item::TransposedBaseElementVectors(revealed_base_elems));
        proof_stream.enqueue(&Item::CompressedAuthenticationPaths(auth_paths_base));

        let revealed_ext_elems =
            Self::get_revealed_elements(&transposed_ext_codewords, &revealed_indices);
        let auth_paths_ext = extension_tree.get_multi_proof(&revealed_indices);
        proof_stream.enqueue(&Item::TransposedExtensionElementVectors(revealed_ext_elems));
        proof_stream.enqueue(&Item::CompressedAuthenticationPaths(auth_paths_ext));
        timer.elapsed("open leafs of zipped codewords");

        // open combination codeword at the same positions
        // Notice that we need to loop over `indices` here, not `revealed_indices`
        // as the latter includes adjacent table rows relative to the values in `indices`
        let revealed_combination_elements: Vec<XWord> = cross_codeword_slice_indices
            .iter()
            .map(|i| combination_codeword[*i])
            .collect();
        let revealed_combination_auth_paths =
            combination_tree.get_multi_proof(&cross_codeword_slice_indices);
        proof_stream.enqueue(&Item::RevealedCombinationElements(
            revealed_combination_elements,
        ));
        proof_stream.enqueue(&Item::CompressedAuthenticationPaths(
            revealed_combination_auth_paths,
        ));

        timer.elapsed("open combination codeword at same positions");

        let report = timer.finish();
        println!("{}", report);
        println!(
            "Created proof containing {} B-field elements",
            proof_stream.transcript_length()
        );

        proof_stream
    }

    fn get_revealed_indices(
        &self,
        unit_distances: &[usize],
        cross_codeword_slice_indices: &[usize],
    ) -> Vec<usize> {
        let mut revealed_indices: Vec<usize> = vec![];
        for index in cross_codeword_slice_indices.iter() {
            for unit_distance in unit_distances.iter() {
                revealed_indices.push((index + unit_distance) % self.xfri.domain.length);
            }
        }
        revealed_indices.sort_unstable();
        revealed_indices.dedup();
        revealed_indices
    }

    // FIXME: `padded_heights` could be `&[usize; NUM_TABLES]` (but that const doesn't exist yet)
    fn get_unit_distances(&self, ext_tables: &ExtTableCollection) -> Vec<usize> {
        let mut unit_distances = ext_tables
            .into_iter()
            .map(|table| table.unit_distance(self.xfri.domain.length))
            .collect_vec();
        unit_distances.push(0);
        unit_distances.sort_unstable();
        unit_distances.dedup();
        unit_distances
    }

    fn get_revealed_elements<PF: PrimeField>(
        transposed_base_codewords: &[Vec<PF>],
        revealed_indices: &[usize],
    ) -> Vec<Vec<PF>> {
        let revealed_base_elements = revealed_indices
            .iter()
            .map(|idx| transposed_base_codewords[*idx].clone())
            .collect_vec();
        revealed_base_elements
    }

    fn get_initials_seed() -> Vec<BFieldElement> {
        let mut rng = thread_rng();
        let initials_seed_u64: [u64; RP_DEFAULT_OUTPUT_SIZE] = rng.gen();
        let initials_seed: Vec<BFieldElement> = initials_seed_u64
            .into_iter()
            .map(BFieldElement::new)
            .collect();
        initials_seed
    }

    // TODO try to reduce the number of arguments
    #[allow(clippy::too_many_arguments)]
    fn create_combination_codeword(
        &self,
        timer: &mut TimingReporter,
        randomizer_codewords: Vec<Vec<XFieldElement>>,
        base_codewords: Vec<Vec<BFieldElement>>,
        extension_codewords: Vec<Vec<XFieldElement>>,
        quotient_codewords: Vec<Vec<XFieldElement>>,
        weights: Vec<XFieldElement>,
        base_degree_bounds: Vec<i64>,
        extension_degree_bounds: Vec<i64>,
        quotient_degree_bounds: Vec<i64>,
        ext_tables: &ExtTableCollection, // TODO this is just for debugging – try to remove again
    ) -> Vec<XFieldElement> {
        assert_eq!(self.num_randomizer_polynomials, randomizer_codewords.len());

        let base_codewords_lifted = base_codewords
            .into_iter()
            .map(|base_codeword| {
                base_codeword
                    .into_iter()
                    .map(|bfe| bfe.lift())
                    .collect_vec()
            })
            .collect_vec();
        let mut weights_iterator = weights.into_iter();
        let mut combination_codeword: Vec<XFieldElement> = vec![0.into(); self.xfri.domain.length];

        // TODO don't keep the entire domain's values in memory, create them lazily when needed
        let fri_x_values = self.xfri.domain.domain_values();
        timer.elapsed("x_domain_values");

        for randomizer_codeword in randomizer_codewords {
            combination_codeword = Self::non_linearly_add_to_codeword(
                &combination_codeword,
                &randomizer_codeword,
                &weights_iterator.next().unwrap(),
                &randomizer_codeword,
                &0.into(),
            );
        }

        for (codewords, bounds, identifier) in [
            (base_codewords_lifted, base_degree_bounds, "base"),
            (extension_codewords, extension_degree_bounds, "ext"),
            (quotient_codewords, quotient_degree_bounds, "quot"),
        ] {
            // TODO with the DEBUG CODE and enumerate removed, the iterators can be `into_par_iter`
            for (idx, (codeword, degree_bound)) in
                codewords.into_iter().zip_eq(bounds.iter()).enumerate()
            {
                let shift = (self.max_degree as Degree - degree_bound) as u32;
                let codeword_shifted = Self::shift_codeword(&fri_x_values, &codeword, shift);

                combination_codeword = Self::non_linearly_add_to_codeword(
                    &combination_codeword,
                    &codeword,
                    &weights_iterator.next().unwrap(),
                    &codeword_shifted,
                    &weights_iterator.next().unwrap(),
                );
                self.debug_check_degrees(
                    &idx,
                    degree_bound,
                    &shift,
                    &codeword,
                    &codeword_shifted,
                    ext_tables,
                    identifier,
                );
            }
            timer.elapsed(&format!("...shift and collect {} codewords", identifier));
        }

        if std::env::var("DEBUG").is_ok() {
            println!(
                "The combination codeword corresponds to a polynomial of degree {}",
                self.xfri.domain.interpolate(&combination_codeword).degree()
            );
        }

        combination_codeword
    }

    #[allow(clippy::too_many_arguments)]
    fn debug_check_degrees(
        &self,
        idx: &usize,
        degree_bound: &Degree,
        shift: &u32,
        extension_codeword: &[XFieldElement],
        extension_codeword_shifted: &[XFieldElement],
        ext_tables: &ExtTableCollection,
        poly_type: &str,
    ) {
        if std::env::var("DEBUG").is_err() {
            return;
        }
        let interpolated = self.xfri.domain.interpolate(extension_codeword);
        let interpolated_shifted = self.xfri.domain.interpolate(extension_codeword_shifted);
        let int_shift_deg = interpolated_shifted.degree();
        let maybe_excl_mark = if int_shift_deg > self.max_degree as isize {
            "!!!"
        } else if int_shift_deg != -1 && int_shift_deg != self.max_degree as isize {
            " ! "
        } else {
            "   "
        };
        println!(
            "{maybe_excl_mark} The shifted {poly_type} codeword for {} with index {idx:>2} \
            must be of maximal degree {}. Got {}. Predicted degree of unshifted codeword: \
            {degree_bound}. Actual degree of unshifted codeword: {}. Shift = {shift}.",
            ext_tables.codeword_index_to_table_name(*idx),
            self.max_degree,
            int_shift_deg,
            interpolated.degree(),
        );
    }

    fn non_linearly_add_to_codeword(
        combination_codeword: &Vec<XFieldElement>,
        summand: &Vec<XFieldElement>,
        weight: &XFieldElement,
        summand_shifted: &Vec<XFieldElement>,
        weight_shifted: &XFieldElement,
    ) -> Vec<XFieldElement> {
        combination_codeword
            .par_iter()
            .zip_eq(summand.par_iter())
            .map(|(cc_elem, &summand_elem)| *cc_elem + *weight * summand_elem)
            .zip_eq(summand_shifted.par_iter())
            .map(|(cc_elem, &summand_shifted_elem)| {
                cc_elem + *weight_shifted * summand_shifted_elem
            })
            .collect()
    }

    fn shift_codeword(
        fri_x_values: &Vec<XFieldElement>,
        codeword: &Vec<XFieldElement>,
        shift: u32,
    ) -> Vec<XFieldElement> {
        fri_x_values
            .par_iter()
            .zip_eq(codeword.par_iter())
            .map(|(x, &codeword_element)| (codeword_element * x.mod_pow_u32(shift)))
            .collect()
    }

    fn get_extension_merkle_tree(
        hasher: &RescuePrimeXlix<16>,
        transposed_extension_codewords: &Vec<Vec<XWord>>,
    ) -> MerkleTree<StarkHasher> {
        let mut extension_codeword_digests_by_index =
            Vec::with_capacity(transposed_extension_codewords.len());

        transposed_extension_codewords
            .into_par_iter()
            .map(|transposed_ext_codeword| {
                let transposed_ext_codeword_coeffs: Vec<BFieldElement> = transposed_ext_codeword
                    .iter()
                    .map(|x| x.coefficients.clone().to_vec())
                    .concat();

                // DEBUG CODE BELOW
                // let sum_of_full_widths: usize = base_tables
                //     .into_iter()
                //     .map(|table| table.full_width())
                //     .sum();
                //
                // assert_eq!(
                //     3 * sum_of_full_widths,
                //     transposed_ext_codeword_coeffs.len(),
                //     "There are just as many coefficients as there are BFieldElements in the full width of all extension tables."
                // );

                hasher.hash(&transposed_ext_codeword_coeffs, RP_DEFAULT_OUTPUT_SIZE)
            })
            .collect_into_vec(&mut extension_codeword_digests_by_index);

        MerkleTree::<StarkHasher>::from_digests(&extension_codeword_digests_by_index)
    }

    fn get_merkle_tree(
        hasher: &RescuePrimeXlix<RP_DEFAULT_WIDTH>,
        codewords: &Vec<Vec<BWord>>,
    ) -> MerkleTree<StarkHasher> {
        let mut codeword_digests_by_index = Vec::with_capacity(codewords.len());
        codewords
            .par_iter()
            .map(|values| hasher.hash(values, DIGEST_LEN))
            .collect_into_vec(&mut codeword_digests_by_index);
        MerkleTree::<StarkHasher>::from_digests(&codeword_digests_by_index)
    }

    /// Essentially a matrix transpose. Given
    ///
    /// ```norun
    /// [a b c]
    /// [d e f]
    /// ```
    ///
    /// returns
    ///
    /// ```norun
    /// [a d]
    /// [b e]
    /// [c f]
    /// ```
    /// Assumes that input is of rectangular shape.
    fn transpose_codewords<P: Copy>(codewords: &[Vec<P>]) -> Vec<Vec<P>> {
        (0..codewords[0].len())
            .map(|col_idx| codewords.iter().map(|row| row[col_idx]).collect())
            .collect()
    }

    fn get_padded_base_tables(&self, base_matrices: &BaseMatrices) -> BaseTableCollection {
        let mut base_tables =
            BaseTableCollection::from_base_matrices(self.num_trace_randomizers, base_matrices);
        base_tables.pad();
        base_tables
    }

    fn get_randomizer_codewords(&self) -> (Vec<XWord>, Vec<Vec<BWord>>) {
        let mut rng = rand::thread_rng();
        let randomizer_coefficients =
            XWord::random_elements(self.max_degree as usize + 1, &mut rng);
        let randomizer_polynomial = Polynomial::new(randomizer_coefficients);

        let x_randomizer_codeword = self.xfri.domain.evaluate(&randomizer_polynomial);
        let mut b_randomizer_codewords = vec![vec![], vec![], vec![]];
        for x_elem in x_randomizer_codeword.iter() {
            b_randomizer_codewords[0].push(x_elem.coefficients[0]);
            b_randomizer_codewords[1].push(x_elem.coefficients[1]);
            b_randomizer_codewords[2].push(x_elem.coefficients[2]);
        }
        (x_randomizer_codeword, b_randomizer_codewords)
    }

    pub fn verify(&self, proof_stream: &mut StarkProofStream) -> Result<bool, Box<dyn Error>> {
        let mut timer = TimingReporter::start();
        let hasher = StarkHasher::new();

        let base_merkle_tree_root = proof_stream.dequeue()?.as_merkle_root()?;
        let extension_challenge_seed = proof_stream.verifier_fiat_shamir();
        timer.elapsed("Fiat-Shamir seed for extension challenges");

        let extension_challenge_weights: [XFieldElement; AllChallenges::TOTAL_CHALLENGES] = hasher
            .sample_n_weights(&extension_challenge_seed, AllChallenges::TOTAL_CHALLENGES)
            .try_into()
            .unwrap();
        let extension_challenges = AllChallenges::create_challenges(&extension_challenge_weights);
        timer.elapsed("Create extension challenges");

        let padded_heights = proof_stream
            .dequeue()?
            .as_padded_heights()?
            .iter()
            .map(|bfe| bfe.value() as usize)
            .collect_vec();
        timer.elapsed("Got all padded heights");

        let extension_tree_merkle_root = proof_stream.dequeue()?.as_merkle_root()?;
        let all_terminals = proof_stream.dequeue()?.as_terminals()?;
        timer.elapsed("Get extension tree's root & terminals from proof stream");

        let ext_table_collection =
            ExtTableCollection::with_padded_heights(self.num_trace_randomizers, &padded_heights);

        let base_degree_bounds: Vec<Degree> = ext_table_collection.get_all_base_degree_bounds();
        timer.elapsed("Calculated base degree bounds");

        let extension_degree_bounds = ext_table_collection.get_extension_column_degree_bounds();
        timer.elapsed("Calculated extension degree bounds");

        let quotient_degree_bounds = ext_table_collection
            .get_all_quotient_degree_bounds(&extension_challenges, &all_terminals);
        timer.elapsed("Calculated quotient degree bounds");

        let non_lin_combi_weights_seed = proof_stream.verifier_fiat_shamir();
        timer.elapsed("Fiat-Shamir to get seed for sampling non-linear-combination weights");

        // get weights for nonlinear combination
        //  - 1 randomizer
        //  - 2 for every other polynomial (base, extension, quotients)
        let num_base_polynomials = base_degree_bounds.len();
        let num_extension_polynomials = extension_degree_bounds.len();
        let num_quotient_polynomials = quotient_degree_bounds.len();
        let num_difference_quotients = PermArg::all_permutation_arguments().len();
        let non_lin_combi_weights_count = self.num_randomizer_polynomials
            + 2 * num_base_polynomials
            + 2 * num_extension_polynomials
            + 2 * num_quotient_polynomials
            + 2 * num_difference_quotients;
        let non_lin_combi_weights =
            hasher.sample_n_weights(&non_lin_combi_weights_seed, non_lin_combi_weights_count);
        timer.elapsed("Sampled weights for non-linear combination");

        let combination_root = proof_stream.dequeue()?.as_merkle_root()?;

        let indices_seed = proof_stream.verifier_fiat_shamir();
        let cross_codeword_slice_indices =
            hasher.sample_indices(self.security_level, &indices_seed, self.xfri.domain.length);
        let num_idxs = cross_codeword_slice_indices.len();
        timer.elapsed("Got indices");

        // Verify low degree of combination polynomial
        self.xfri.verify(proof_stream, &combination_root)?;
        timer.elapsed("Verified FRI proof");

        let unit_distances = self.get_unit_distances(&ext_table_collection);
        timer.elapsed("Got unit distances");

        // Open leafs of zipped codewords at indicated positions
        let revealed_indices =
            self.get_revealed_indices(&unit_distances, &cross_codeword_slice_indices);
        timer.elapsed("Calculated revealed indices");

        // TODO: in the following ~80 lines, we (conceptually) do the same thing three times. DRY.
        let revealed_base_elems = proof_stream
            .dequeue()?
            .as_transposed_base_element_vectors()?;
        let auth_paths_base = proof_stream
            .dequeue()?
            .as_compressed_authentication_paths()?;
        timer.elapsed("Read base elements and auth paths from proof stream");
        let leaf_digests_base: Vec<_> = revealed_base_elems
            .par_iter()
            .map(|revealed_base_elem| hasher.hash(revealed_base_elem, RP_DEFAULT_OUTPUT_SIZE))
            .collect();
        timer.elapsed(&format!("Got {num_idxs} leaf digests for base elements"));

        if !MerkleTree::<StarkHasher>::verify_multi_proof_from_leaves(
            base_merkle_tree_root,
            &revealed_indices,
            &leaf_digests_base,
            &auth_paths_base,
        ) {
            // TODO: Replace this by a specific error type, or just return `Ok(false)`
            panic!("Failed to verify authentication path for base codeword");
        }
        timer.elapsed(&format!("Verified auth paths for {num_idxs} base elements"));

        let revealed_ext_elems = proof_stream
            .dequeue()?
            .as_transposed_extension_element_vectors()?;
        let auth_paths_ext = proof_stream
            .dequeue()?
            .as_compressed_authentication_paths()?;
        timer.elapsed("Read extension elements and auth paths from proof stream");
        let leaf_digests_ext: Vec<_> = revealed_ext_elems
            .par_iter()
            .map(|xvalues| {
                let bvalues: Vec<BFieldElement> = xvalues
                    .iter()
                    .map(|x| x.coefficients.clone().to_vec())
                    .concat();
                // FIXME this is a bad assertion. Come up with something better.
                debug_assert_eq!(
                    3 * 50, // 50 is the number of extension columns
                    bvalues.len(),
                    "9 X-field elements must become 27 B-field elements"
                );
                hasher.hash(&bvalues, RP_DEFAULT_OUTPUT_SIZE)
            })
            .collect();
        timer.elapsed(&format!("Got {num_idxs} leaf digests for ext elements"));

        if !MerkleTree::<StarkHasher>::verify_multi_proof_from_leaves(
            extension_tree_merkle_root,
            &revealed_indices,
            &leaf_digests_ext,
            &auth_paths_ext,
        ) {
            // TODO: Replace this by a specific error type, or just return `Ok(false)`
            panic!("Failed to verify authentication path for extension codeword");
        }
        timer.elapsed(&format!("Verified auth paths for {num_idxs} ext elements"));

        // Verify Merkle authentication path for combination elements
        let revealed_combination_elements =
            proof_stream.dequeue()?.as_revealed_combination_elements()?;
        let revealed_combination_digests: Vec<_> = revealed_combination_elements
            .par_iter()
            .map(|xfe| {
                let b_elements: Vec<BFieldElement> = xfe.to_digest();
                hasher.hash(&b_elements, RP_DEFAULT_OUTPUT_SIZE)
            })
            .collect();
        let revealed_combination_auth_paths = proof_stream
            .dequeue()?
            .as_compressed_authentication_paths()?;
        if !MerkleTree::<StarkHasher>::verify_multi_proof_from_leaves(
            combination_root.clone(),
            &cross_codeword_slice_indices,
            &revealed_combination_digests,
            &revealed_combination_auth_paths,
        ) {
            // TODO: Replace this by a specific error type, or just return `Ok(false)`
            panic!("Failed to verify authentication path for combination codeword");
        }
        timer.elapsed(&format!("Verified auth paths for {num_idxs} combi elems"));

        // TODO: we can store the elements mushed into "index_map_of_revealed_elems" separately,
        //  like in "cross_slice_by_table" below, to avoid unmushing later
        let index_map_of_revealed_elems = Self::get_index_map_of_revealed_elems(
            self.num_randomizer_polynomials,
            revealed_indices,
            revealed_base_elems,
            revealed_ext_elems,
        );
        timer.elapsed(&format!("Collected {num_idxs} values into a hash map"));

        // =======================================
        // ==== verify non-linear combination ====
        // =======================================
        let base_offset = self.num_randomizer_polynomials;
        let ext_offset = base_offset + num_base_polynomials;
        let final_offset = ext_offset + num_extension_polynomials;
        for (cross_slice_index, revealed_combination_element) in cross_codeword_slice_indices
            .into_iter()
            .zip_eq(revealed_combination_elements)
        {
            let curr_fri_domain_value = self.xfri.domain.domain_value(cross_slice_index as u32);
            let cross_slice = &index_map_of_revealed_elems[&cross_slice_index];
            let mut summands = cross_slice[0..base_offset].to_vec();

            for (index_range, degree_bounds) in [
                (base_offset..ext_offset, base_degree_bounds.iter()),
                (ext_offset..final_offset, extension_degree_bounds.iter()),
            ] {
                for (codeword_index, degree_bound) in index_range.zip_eq(degree_bounds) {
                    let shift = self.max_degree - degree_bound;
                    let curr_codeword_elem = cross_slice[codeword_index];
                    let curr_codeword_elem_shifted =
                        curr_codeword_elem * curr_fri_domain_value.mod_pow_u32(shift as u32);
                    summands.push(curr_codeword_elem);
                    summands.push(curr_codeword_elem_shifted);
                }
            }

            // unmush cross-codeword slice: pick (base, ext) columns per table
            let mut curr_base_idx = base_offset;
            let mut curr_ext_idx = ext_offset;
            let mut cross_slice_by_table: Vec<Vec<XFieldElement>> = vec![];
            let mut next_cross_slice_by_table: Vec<Vec<XFieldElement>> = vec![];

            for table in ext_table_collection.into_iter() {
                let num_base_cols = table.base_width();
                let num_ext_cols = table.full_width() - table.base_width();

                let base_col_slice =
                    cross_slice[curr_base_idx..curr_base_idx + num_base_cols].to_vec();
                let ext_col_slice = cross_slice[curr_ext_idx..curr_ext_idx + num_ext_cols].to_vec();
                let table_slice = [base_col_slice, ext_col_slice].concat();
                cross_slice_by_table.push(table_slice);

                let curr_unit_distance = table.unit_distance(self.xfri.domain.length);
                let next_cross_slice_index =
                    (cross_slice_index + curr_unit_distance) % self.xfri.domain.length;
                let next_cross_slice = &index_map_of_revealed_elems[&next_cross_slice_index];

                let next_base_col_slice =
                    next_cross_slice[curr_base_idx..curr_base_idx + num_base_cols].to_vec();
                let next_ext_col_slice =
                    next_cross_slice[curr_ext_idx..curr_ext_idx + num_ext_cols].to_vec();
                let next_table_slice = [next_base_col_slice, next_ext_col_slice].concat();
                next_cross_slice_by_table.push(next_table_slice);

                curr_base_idx += num_base_cols;
                curr_ext_idx += num_ext_cols;
            }
            assert_eq!(ext_offset, curr_base_idx);
            assert_eq!(final_offset, curr_ext_idx);

            // use AIR (actually RAP) to get relevant parts of quotient codewords
            for ((table_row, next_table_row), table) in cross_slice_by_table
                .iter()
                .zip_eq(next_cross_slice_by_table.iter())
                .zip_eq(ext_table_collection.into_iter())
            {
                let table_height = table.padded_height() as u32;

                let boundary_constraints = table.ext_boundary_constraints(&extension_challenges);
                let bndry_deg_bnds = table.boundary_quotient_degree_bounds(&extension_challenges);
                for (boundary_constraint, degree_bound) in
                    boundary_constraints.iter().zip_eq(bndry_deg_bnds.iter())
                {
                    let shift = self.max_degree - degree_bound;
                    let quotient = boundary_constraint.evaluate(table_row)
                        / (curr_fri_domain_value - 1.into());
                    let quotient_shifted =
                        quotient * curr_fri_domain_value.mod_pow_u32(shift as u32);
                    summands.push(quotient);
                    summands.push(quotient_shifted);
                }

                let transition_constraints =
                    table.ext_transition_constraints(&extension_challenges);
                let trnstn_deg_bnds =
                    table.transition_quotient_degree_bounds(&extension_challenges);
                for (transition_constraint, degree_bound) in
                    transition_constraints.iter().zip_eq(trnstn_deg_bnds.iter())
                {
                    let shift = self.max_degree - degree_bound;
                    let quotient = if table_height == 0 {
                        // transition has no meaning on empty table
                        0.into()
                    } else {
                        let evaluated_transition_constraint = transition_constraint
                            .evaluate(&vec![table_row.to_owned(), next_table_row.clone()].concat());
                        let numerator = curr_fri_domain_value - table.omicron().inverse();
                        let denominator =
                            curr_fri_domain_value.mod_pow_u32(table_height) - 1.into();
                        evaluated_transition_constraint * numerator / denominator
                    };
                    let quotient_shifted =
                        quotient * curr_fri_domain_value.mod_pow_u32(shift as u32);
                    summands.push(quotient);
                    summands.push(quotient_shifted);
                }

                let terminal_constraints =
                    table.ext_terminal_constraints(&extension_challenges, &all_terminals);
                let trmnl_deg_bnds =
                    table.terminal_quotient_degree_bounds(&extension_challenges, &all_terminals);
                for (terminal_constraint, degree_bound) in
                    terminal_constraints.iter().zip_eq(trmnl_deg_bnds.iter())
                {
                    let shift = self.max_degree - degree_bound;
                    let quotient = terminal_constraint.evaluate(table_row)
                        / (curr_fri_domain_value - table.omicron().inverse());
                    let quotient_shifted =
                        quotient * curr_fri_domain_value.mod_pow_u32(shift as u32);
                    summands.push(quotient);
                    summands.push(quotient_shifted);
                }
            }

            for arg in PermArg::all_permutation_arguments().iter() {
                let perm_arg_deg_bound = arg.quotient_degree_bound(&ext_table_collection);
                let shift = self.max_degree - perm_arg_deg_bound;
                let quotient = arg.evaluate_difference(&cross_slice_by_table)
                    / (curr_fri_domain_value - 1.into());
                let quotient_shifted = quotient * curr_fri_domain_value.mod_pow_u32(shift as u32);
                summands.push(quotient);
                summands.push(quotient_shifted);
            }

            let inner_product = non_lin_combi_weights
                .par_iter()
                .zip_eq(summands.par_iter())
                .map(|(weight, summand)| *weight * *summand)
                .sum();

            // FIXME: This assert looks like it's for development, but it's the actual integrity
            //  check. Change to `if (…) { return Ok(false) }` or whatever is suitable.
            assert_eq!(
                revealed_combination_element, inner_product,
                "The combination leaf must equal the inner product"
            );
        }
        timer.elapsed(&format!("Verified {num_idxs} non-linear combinations"));

        // TODO: check cross-table difference boundary constraints for PermArgs
        // TODO: check cross-table difference boundary constraints for EvalArgs

        // Verify external terminals
        if !verify_evaluation_argument(
            &self.input_symbols,
            extension_challenges
                .processor_table_challenges
                .input_table_eval_row_weight,
            all_terminals.processor_table_endpoints.input_table_eval_sum,
        ) {
            return Err(Box::new(StarkVerifyError::EvaluationArgument(0)));
        }

        if !verify_evaluation_argument(
            &self.output_symbols,
            extension_challenges
                .processor_table_challenges
                .output_table_eval_row_weight,
            all_terminals
                .processor_table_endpoints
                .output_table_eval_sum,
        ) {
            return Err(Box::new(StarkVerifyError::EvaluationArgument(1)));
        }
        timer.elapsed("Verified terminals");

        println!("{}", timer.finish());
        Ok(true)
    }

    fn get_index_map_of_revealed_elems(
        num_randomizer_polynomials: usize,
        revealed_indices: Vec<usize>,
        revealed_base_elems: Vec<Vec<BFieldElement>>,
        revealed_ext_elems: Vec<Vec<XFieldElement>>,
    ) -> HashMap<usize, Vec<XFieldElement>> {
        let mut index_map: HashMap<usize, Vec<XFieldElement>> = HashMap::new();
        for (i, &idx) in revealed_indices.iter().enumerate() {
            let mut rand_elems = vec![];
            for (coeff_0, coeff_1, coeff_2) in revealed_base_elems[i]
                .iter()
                .take(3 * num_randomizer_polynomials)
                .tuples()
            {
                rand_elems.push(XFieldElement::new([*coeff_0, *coeff_1, *coeff_2]));
            }

            let base_elems = revealed_base_elems[i]
                .iter()
                .skip(3 * num_randomizer_polynomials)
                .map(|bfe| bfe.lift())
                .collect_vec();

            let cross_slice = [rand_elems, base_elems, revealed_ext_elems[i].clone()].concat();
            index_map.insert(idx, cross_slice);
        }
        index_map
    }
}

#[cfg(test)]
pub(crate) mod triton_stark_tests {
    use super::*;
    use crate::shared_math::mpolynomial::MPolynomial;
    use crate::shared_math::ntt::ntt;
    use crate::shared_math::other::log_2_floor;
    use crate::shared_math::stark::triton::arguments::evaluation_argument;
    use crate::shared_math::stark::triton::instruction::sample_programs;
    use crate::shared_math::stark::triton::stdio::VecStream;
    use crate::shared_math::stark::triton::table::base_table;
    use crate::shared_math::stark::triton::vm::Program;
    use crate::shared_math::traits::PrimeField;
    use crate::util_types::proof_stream_typed::ProofStream;

    pub(crate) fn dummy_challenges_initials() -> (AllChallenges, AllEndpoints) {
        let hasher = StarkHasher::new();
        let mock_seed = hasher.hash(&[], DIGEST_LEN);

        let challenge_weights =
            hasher.sample_n_weights(&mock_seed, AllChallenges::TOTAL_CHALLENGES);
        let dummy_challenges: AllChallenges = AllChallenges::create_challenges(&challenge_weights);

        let initial_weights = hasher.sample_n_weights(&mock_seed, AllEndpoints::TOTAL_ENDPOINTS);
        let dummy_initials: AllEndpoints = AllEndpoints::create_initials(&initial_weights);

        (dummy_challenges, dummy_initials)
    }

    fn parse_simulate_prove(
        code: &str,
        co_set_fri_offset: BWord,
        input_symbols: &[BWord],
        output_symbols: &[BWord],
    ) -> (Stark, ProofStream<Item, RescuePrimeXlix<RP_DEFAULT_WIDTH>>) {
        let (base_matrices, _) = parse_simulate(code, input_symbols, &[], output_symbols);

        let num_trace_randomizers = 2;
        let num_randomizer_polynomials = 1;
        let log_expansion_factor = 2;
        let security_level = 32;

        let unpadded_height = [
            base_matrices.program_matrix.len(),
            base_matrices.processor_matrix.len(),
            base_matrices.instruction_matrix.len(),
            base_matrices.op_stack_matrix.len(),
            base_matrices.ram_matrix.len(),
            base_matrices.jump_stack_matrix.len(),
            base_matrices.hash_matrix.len(),
            base_matrices.u32_op_matrix.len(),
        ]
        .into_iter()
        .max()
        .unwrap_or(0);
        let padded_height = base_table::pad_height(unpadded_height, num_trace_randomizers);

        let stark = Stark::new(
            padded_height,
            num_trace_randomizers,
            num_randomizer_polynomials,
            log_expansion_factor,
            security_level,
            co_set_fri_offset,
            input_symbols,
            output_symbols,
        );
        let proof_stream = stark.prove(base_matrices);

        (stark, proof_stream)
    }

    fn parse_simulate(
        code: &str,
        input_symbols: &[BWord],
        secret_input_symbols: &[BWord],
        output_symbols: &[BWord],
    ) -> (BaseMatrices, VecStream) {
        let program = Program::from_code(code);

        assert!(program.is_ok(), "program parses correctly");
        let program = program.unwrap();

        let mut stdin = VecStream::new_bwords(input_symbols);
        let mut secret_in = VecStream::new_bwords(secret_input_symbols);
        let mut stdout = VecStream::new_bwords(output_symbols);
        let rescue_prime = rescue_prime_xlix::neptune_params();

        let (base_matrices, err) =
            program.simulate(&mut stdin, &mut secret_in, &mut stdout, &rescue_prime);
        if let Some(error) = err {
            panic!("The VM encountered the following problem: {}", error);
        }
        (base_matrices, stdout)
    }

    fn parse_simulate_pad_extend(
        code: &str,
        stdin: &[BWord],
        secret_in: &[BWord],
        stdout: &[BWord],
    ) -> (
        VecStream,
        BaseTableCollection,
        BaseTableCollection,
        ExtTableCollection,
        AllChallenges,
        AllEndpoints,
        AllEndpoints,
    ) {
        let (base_matrices, stdout) = parse_simulate(code, stdin, secret_in, stdout);
        let num_trace_randomizers = 2;

        let mut base_tables =
            BaseTableCollection::from_base_matrices(num_trace_randomizers, &base_matrices);

        let unpadded_base_tables = base_tables.clone();

        base_tables.pad();

        let (all_challenges, all_initials) = dummy_challenges_initials();
        let (ext_tables, all_terminals) =
            ExtTableCollection::extend_tables(&base_tables, &all_challenges, &all_initials);

        (
            stdout,
            unpadded_base_tables,
            base_tables,
            ext_tables,
            all_challenges,
            all_initials,
            all_terminals,
        )
    }

    fn assert_transition_constraints_on_table<PF: PrimeField>(
        table_data: &[Vec<PF>],
        air_constraints: &[MPolynomial<PF>],
        message: &str,
    ) {
        for (row_idx, (curr_row, next_row)) in table_data.iter().tuple_windows().enumerate() {
            let air_point = [curr_row.to_vec(), next_row.to_vec()].concat();

            for (constraint_idx, air_constraint) in air_constraints.iter().enumerate() {
                assert!(
                    air_constraint.evaluate(&air_point).is_zero(),
                    "{}. Constraint index: {}. Row index: {}\n{}",
                    message,
                    constraint_idx,
                    row_idx,
                    air_constraint,
                );
            }
        }
    }

    fn assert_consistency_boundary_constraints_on_table<PF: PrimeField>(
        table_data: &[Vec<PF>],
        air_constraints: &[MPolynomial<PF>],
        message: &str,
    ) {
        for (row_idx, curr_row) in table_data.iter().enumerate() {
            for (constraint_idx, air_constraint) in air_constraints.iter().enumerate() {
                assert!(
                    air_constraint.evaluate(&curr_row).is_zero(),
                    "{}. Constraint index: {}. Row index: {}",
                    message,
                    constraint_idx,
                    row_idx,
                );
            }
        }
    }

    #[test]
    pub fn shift_codeword_test() {
        let stark = Stark::new(2, 2, 1, 2, 32, 1.into(), &[], &[]);
        let fri_x_values = stark.xfri.domain.domain_values();

        let mut test_codeword: Vec<XFieldElement> = vec![0.into(); stark.xfri.domain.length];
        let poly_degree = 4;
        test_codeword[0..=poly_degree].copy_from_slice(&[
            2.into(),
            42.into(),
            1.into(),
            3.into(),
            17.into(),
        ]);

        ntt(
            &mut test_codeword,
            stark.xfri.domain.omega,
            log_2_floor(stark.xfri.domain.length as u128) as u32,
        );
        for shift in [0, 1, 5, 17, 63, 121, 128] {
            let shifted_codeword = Stark::shift_codeword(&fri_x_values, &test_codeword, shift);
            let interpolated_shifted_codeword = stark.xfri.domain.interpolate(&shifted_codeword);
            assert_eq!(
                (poly_degree + shift as usize) as isize,
                interpolated_shifted_codeword.degree()
            );
        }
    }

    // 1. simulate(), pad(), extend(), test terminals
    #[test]
    pub fn check_terminals() {
        let read_nop_code = "
            read_io read_io read_io
            nop nop
        ";
        let input_symbols = vec![3.into(), 5.into(), 7.into()];
        let (
            stdout,
            _unpadded_base_tables,
            _base_tables,
            _ext_tables,
            all_challenges,
            _all_initials,
            all_terminals,
        ) = parse_simulate_pad_extend(read_nop_code, &input_symbols, &[], &[]);

        let ptie = all_terminals.processor_table_endpoints.input_table_eval_sum;
        let ine = evaluation_argument::compute_terminal(
            &input_symbols,
            XFieldElement::ring_zero(),
            all_challenges.input_challenges.processor_eval_row_weight,
        );
        assert_eq!(ptie, ine, "The input evaluation arguments do not match.");

        let ptoe = all_terminals
            .processor_table_endpoints
            .output_table_eval_sum;

        let oute = evaluation_argument::compute_terminal(
            &stdout.to_bword_vec(),
            XFieldElement::ring_zero(),
            all_challenges.output_challenges.processor_eval_row_weight,
        );
        assert_eq!(ptoe, oute, "The output evaluation arguments do not match.");
    }

    // 2. simulate(), test constraints
    #[test]
    fn triton_table_constraints_evaluate_to_zero_test() {
        let (
            _stdout,
            _unpadded_base_tables,
            _padded_base_tables,
            ext_tables,
            all_challenges,
            _all_initials,
            all_terminals,
        ) = parse_simulate_pad_extend(sample_programs::FIBONACCI_LT, &[], &[], &[]);

        for ext_table in (&ext_tables).into_iter() {
            let ext_transition_constraints = ext_table.ext_transition_constraints(&all_challenges);
            let message_1 = format!("ext_transition_constraints on {}", &ext_table.name());
            assert_transition_constraints_on_table(
                ext_table.data(),
                &ext_transition_constraints,
                &message_1,
            );

            let ext_consistency_constraints =
                ext_table.ext_consistency_constraints(&all_challenges);
            let message_2 = format!("ext_consistency_constraints on {}", &ext_table.name());
            assert_consistency_boundary_constraints_on_table(
                ext_table.data(),
                &ext_consistency_constraints,
                &message_2,
            );

            if ext_table.data().len() > 0 {
                let ext_boundary_constraints = ext_table.ext_boundary_constraints(&all_challenges);
                let message_3 = format!("ext_boundary_constraints on {}", &ext_table.name());
                assert_consistency_boundary_constraints_on_table(
                    &[ext_table.data()[0].clone()],
                    &ext_boundary_constraints,
                    &message_3,
                );

                let ext_terminal_constraints =
                    ext_table.ext_terminal_constraints(&all_challenges, &all_terminals);
                let message_4 = format!("ext_terminal_constraints on {}", &ext_table.name());
                assert_consistency_boundary_constraints_on_table(
                    &[ext_table.data().last().unwrap().to_vec()],
                    &ext_terminal_constraints,
                    &message_4,
                );
            }
        }
    }

    #[test]
    fn triton_prove_verify_test() {
        let co_set_fri_offset = BWord::generator();
        let (stark, mut proof_stream) = parse_simulate_prove(
            "hash nop hash nop nop hash halt",
            co_set_fri_offset,
            &[],
            &[],
        );

        println!("between prove and verify");

        let result = stark.verify(&mut proof_stream);
        if let Err(e) = result {
            panic!("The Verifier is unhappy! {}", e);
        }
        assert!(result.unwrap());
    }
}
