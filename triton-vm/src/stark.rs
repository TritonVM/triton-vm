use std::collections::HashMap;
use std::error::Error;

use itertools::Itertools;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::mpolynomial::Degree;
use twenty_first::shared_math::other;
use twenty_first::shared_math::polynomial::Polynomial;
use twenty_first::shared_math::rescue_prime_regular::RescuePrimeRegular;
use twenty_first::shared_math::traits::{
    FiniteField, GetRandomElements, Inverse, ModPowU32, PrimitiveRootOfUnity,
};
use twenty_first::shared_math::x_field_element::XFieldElement;
use twenty_first::timing_reporter::TimingReporter;
use twenty_first::util_types::merkle_tree::MerkleTree;
use twenty_first::util_types::proof_stream_typed::ProofStream;
use twenty_first::util_types::simple_hasher::{Hashable, Hasher, SamplableFrom};

use crate::cross_table_arguments::{
    CrossTableArg, EvalArg, GrandCrossTableArg, NUM_CROSS_TABLE_ARGS, NUM_PUBLIC_EVAL_ARGS,
};
use crate::fri_domain::FriDomain;
use crate::proof_item::ProofItem;
use crate::table::challenges::AllChallenges;
use crate::table::table_collection::{derive_omicron, BaseTableCollection, ExtTableCollection};
use crate::triton_xfri::{self, Fri};

use super::table::base_matrix::BaseMatrices;

pub type StarkHasher = RescuePrimeRegular;
pub type StarkProofStream = ProofStream<ProofItem<StarkHasher>, StarkHasher>;

pub struct Stark {
    num_trace_randomizers: usize,
    num_randomizer_polynomials: usize,
    security_level: usize,
    max_degree: Degree,
    bfri_domain: FriDomain<BFieldElement>,
    xfri: Fri<StarkHasher>,
    input_symbols: Vec<BFieldElement>,
    output_symbols: Vec<BFieldElement>,
}

impl Stark {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        max_height: usize,
        num_randomizer_polynomials: usize,
        log_expansion_factor: usize,
        security_level: usize,
        co_set_fri_offset: BFieldElement,
        input_symbols: &[BFieldElement],
        output_symbols: &[BFieldElement],
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

        let num_trace_randomizers = Self::num_trace_randomizers(security_level);
        let empty_table_collection = ExtTableCollection::with_padded_height(max_height);

        let max_degree_with_origin =
            empty_table_collection.max_degree_with_origin(num_trace_randomizers);
        let max_degree = (other::roundup_npo2(max_degree_with_origin.degree as u64) - 1) as i64;
        let fri_domain_length = ((max_degree as u64 + 1) * expansion_factor) as usize;
        println!("Max Degree: {}", max_degree_with_origin);
        println!("FRI domain length: {fri_domain_length}, expansion factor: {expansion_factor}");

        let omega = BFieldElement::primitive_root_of_unity(fri_domain_length as u64).unwrap();

        let bfri_domain = FriDomain {
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

        let base_codeword_tables =
            base_tables.codeword_tables(&self.bfri_domain, self.num_trace_randomizers);
        let base_codewords = base_codeword_tables.get_all_base_columns();
        let all_base_codewords = vec![b_rand_codewords, base_codewords.clone()].concat();
        timer.elapsed("get_all_base_codewords");

        let transposed_base_codewords = Self::transpose_codewords(&all_base_codewords);
        timer.elapsed("transposed_base_codewords");

        let hasher = RescuePrimeRegular::new();
        let base_tree = Self::get_merkle_tree(&hasher, &transposed_base_codewords);
        let base_merkle_tree_root = base_tree.get_root();
        timer.elapsed("base_merkle_tree");

        // Commit to base codewords
        let mut proof_stream = StarkProofStream::default();
        proof_stream.enqueue(&ProofItem::MerkleRoot(base_merkle_tree_root));
        timer.elapsed("proof_stream.enqueue");

        let extension_challenge_seed = proof_stream.prover_fiat_shamir();
        timer.elapsed("prover_fiat_shamir");

        let extension_challenge_weights = Self::sample_weights(
            extension_challenge_seed,
            &hasher,
            AllChallenges::TOTAL_CHALLENGES,
        );
        let extension_challenges = AllChallenges::create_challenges(extension_challenge_weights);
        timer.elapsed("challenges");

        let ext_tables = ExtTableCollection::extend_tables(
            &base_tables,
            &extension_challenges,
            self.num_trace_randomizers,
        );
        timer.elapsed("extend + get_terminals");

        proof_stream.enqueue(&ProofItem::PaddedHeight(BFieldElement::new(
            base_tables.padded_height as u64,
        )));
        timer.elapsed("Sent all padded heights");

        let ext_codeword_tables = ext_tables.codeword_tables(
            &self.xfri.domain,
            base_codeword_tables,
            self.num_trace_randomizers,
        );
        let extension_codewords = ext_codeword_tables.get_all_extension_columns();
        timer.elapsed("Calculated extension codewords");

        let transposed_ext_codewords = Self::transpose_codewords(&extension_codewords);

        let extension_tree = Self::get_extension_merkle_tree(&hasher, &transposed_ext_codewords);

        proof_stream.enqueue(&ProofItem::MerkleRoot(extension_tree.get_root()));
        timer.elapsed("extension_tree");

        let base_degree_bounds = base_tables.get_base_degree_bounds(self.num_trace_randomizers);
        timer.elapsed("Calculated base degree bounds");

        let extension_degree_bounds =
            ext_tables.get_extension_degree_bounds(self.num_trace_randomizers);
        timer.elapsed("Calculated extension degree bounds");

        let mut quotient_codewords = ext_codeword_tables.get_all_quotients(&self.xfri.domain);
        timer.elapsed("Calculated quotient codewords");

        let mut quotient_degree_bounds =
            ext_codeword_tables.get_all_quotient_degree_bounds(self.num_trace_randomizers);
        timer.elapsed("Calculated quotient degree bounds");

        let num_grand_cross_table_args = 1;
        let num_non_lin_combi_weights = self.num_randomizer_polynomials
            + 2 * base_codewords.len()
            + 2 * extension_codewords.len()
            + 2 * quotient_degree_bounds.len()
            + 2 * num_grand_cross_table_args;
        let num_grand_cross_table_arg_weights = NUM_CROSS_TABLE_ARGS + NUM_PUBLIC_EVAL_ARGS;

        let grand_cross_table_arg_and_non_lin_combi_weights_seed =
            proof_stream.prover_fiat_shamir();
        let grand_cross_table_arg_and_non_lin_combi_weights = Self::sample_weights(
            grand_cross_table_arg_and_non_lin_combi_weights_seed,
            &hasher,
            num_grand_cross_table_arg_weights + num_non_lin_combi_weights,
        );
        let (grand_cross_table_argument_weights, non_lin_combi_weights) =
            grand_cross_table_arg_and_non_lin_combi_weights
                .split_at(num_grand_cross_table_arg_weights);
        timer.elapsed("Sample weights for grand cross-table argument and non-linear combination");

        // Prove equal terminal values for the column pairs pertaining to cross table arguments
        let input_terminal = EvalArg::compute_terminal(
            &self.input_symbols,
            EvalArg::default_initial(),
            extension_challenges
                .processor_table_challenges
                .input_table_eval_row_weight,
        );
        let output_terminal = EvalArg::compute_terminal(
            &self.output_symbols,
            EvalArg::default_initial(),
            extension_challenges
                .processor_table_challenges
                .output_table_eval_row_weight,
        );
        let grand_cross_table_arg = GrandCrossTableArg::new(
            grand_cross_table_argument_weights.try_into().unwrap(),
            input_terminal,
            output_terminal,
        );
        let grand_cross_table_arg_quotient_codeword = grand_cross_table_arg
            .terminal_quotient_codeword(
                &ext_codeword_tables,
                &self.xfri.domain,
                derive_omicron(ext_codeword_tables.padded_height as u64),
            );
        quotient_codewords.push(grand_cross_table_arg_quotient_codeword);

        let grand_cross_table_arg_quotient_degree_bound = grand_cross_table_arg
            .quotient_degree_bound(&ext_codeword_tables, self.num_trace_randomizers);
        quotient_degree_bounds.push(grand_cross_table_arg_quotient_degree_bound);
        timer.elapsed("Grand Cross Table Argument");

        let combination_codeword = self.create_combination_codeword(
            &mut timer,
            vec![x_rand_codeword],
            base_codewords,
            extension_codewords,
            quotient_codewords,
            non_lin_combi_weights.to_vec(),
            base_degree_bounds,
            extension_degree_bounds,
            quotient_degree_bounds,
        );
        timer.elapsed("non-linear sum");

        // TODO use Self::get_extension_merkle_tree (or similar) here?
        let mut combination_codeword_digests: Vec<<StarkHasher as Hasher>::Digest> =
            Vec::with_capacity(combination_codeword.len());
        combination_codeword
            .clone()
            .into_par_iter()
            .map(|xfe| hasher.hash_sequence(&xfe.to_sequence()))
            .collect_into_vec(&mut combination_codeword_digests);
        let combination_tree =
            MerkleTree::<StarkHasher>::from_digests(&combination_codeword_digests);
        let combination_root: <StarkHasher as Hasher>::Digest = combination_tree.get_root();

        proof_stream.enqueue(&ProofItem::MerkleRoot(combination_root));

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

        // the relation between the FRI domain and the omicron domain
        let unit_distance = self.xfri.domain.length / base_tables.padded_height;
        // Open leafs of zipped codewords at indicated positions
        let revealed_indices =
            self.get_revealed_indices(unit_distance, &cross_codeword_slice_indices);

        let revealed_base_elems =
            Self::get_revealed_elements(&transposed_base_codewords, &revealed_indices);
        let auth_paths_base = base_tree.get_authentication_structure(&revealed_indices);
        proof_stream.enqueue(&ProofItem::TransposedBaseElementVectors(
            revealed_base_elems,
        ));
        proof_stream.enqueue(&ProofItem::CompressedAuthenticationPaths(auth_paths_base));

        let revealed_ext_elems =
            Self::get_revealed_elements(&transposed_ext_codewords, &revealed_indices);
        let auth_paths_ext = extension_tree.get_authentication_structure(&revealed_indices);
        proof_stream.enqueue(&ProofItem::TransposedExtensionElementVectors(
            revealed_ext_elems,
        ));
        proof_stream.enqueue(&ProofItem::CompressedAuthenticationPaths(auth_paths_ext));
        timer.elapsed("open leafs of zipped codewords");

        // open combination codeword at the same positions
        // Notice that we need to loop over `indices` here, not `revealed_indices`
        // as the latter includes adjacent table rows relative to the values in `indices`
        let revealed_combination_elements: Vec<XFieldElement> = cross_codeword_slice_indices
            .iter()
            .map(|i| combination_codeword[*i])
            .collect();
        let revealed_combination_auth_paths =
            combination_tree.get_authentication_structure(&cross_codeword_slice_indices);
        proof_stream.enqueue(&ProofItem::RevealedCombinationElements(
            revealed_combination_elements,
        ));
        proof_stream.enqueue(&ProofItem::CompressedAuthenticationPaths(
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
        unit_distance: usize,
        cross_codeword_slice_indices: &[usize],
    ) -> Vec<usize> {
        let mut revealed_indices: Vec<usize> = vec![];
        for &index in cross_codeword_slice_indices.iter() {
            revealed_indices.push(index);
            revealed_indices.push((index + unit_distance) % self.xfri.domain.length);
        }
        revealed_indices.sort_unstable();
        revealed_indices.dedup();
        revealed_indices
    }

    fn get_revealed_elements<PF: FiniteField>(
        transposed_base_codewords: &[Vec<PF>],
        revealed_indices: &[usize],
    ) -> Vec<Vec<PF>> {
        let revealed_base_elements = revealed_indices
            .iter()
            .map(|idx| transposed_base_codewords[*idx].clone())
            .collect_vec();
        revealed_base_elements
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
            "{maybe_excl_mark} The shifted {poly_type} codeword with index {idx:>2} \
            must be of maximal degree {}. Got {}. Predicted degree of unshifted codeword: \
            {degree_bound}. Actual degree of unshifted codeword: {}. Shift = {shift}.",
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
        hasher: &RescuePrimeRegular,
        transposed_extension_codewords: &Vec<Vec<XFieldElement>>,
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

                hasher.hash_sequence(&transposed_ext_codeword_coeffs)
            })
            .collect_into_vec(&mut extension_codeword_digests_by_index);

        MerkleTree::<StarkHasher>::from_digests(&extension_codeword_digests_by_index)
    }

    fn get_merkle_tree(
        hasher: &StarkHasher,
        codewords: &Vec<Vec<BFieldElement>>,
    ) -> MerkleTree<StarkHasher> {
        let mut codeword_digests_by_index = Vec::with_capacity(codewords.len());
        codewords
            .par_iter()
            .map(|values| hasher.hash_sequence(values))
            .collect_into_vec(&mut codeword_digests_by_index);
        MerkleTree::<StarkHasher>::from_digests(&codeword_digests_by_index)
    }

    /// Essentially a matrix transpose. Given
    ///
    /// ```py
    /// [a b c]
    /// [d e f]
    /// ```
    ///
    /// returns
    ///
    /// ```py
    /// [a d]
    /// [b e]
    /// [c f]
    /// ```
    /// Assumes that input is of rectangular shape.
    pub fn transpose_codewords<P: Copy>(codewords: &[Vec<P>]) -> Vec<Vec<P>> {
        (0..codewords[0].len())
            .map(|col_idx| codewords.iter().map(|row| row[col_idx]).collect())
            .collect()
    }

    fn get_padded_base_tables(&self, base_matrices: &BaseMatrices) -> BaseTableCollection {
        let mut base_tables = BaseTableCollection::from_base_matrices(base_matrices);
        base_tables.pad();
        base_tables
    }

    fn get_randomizer_codewords(&self) -> (Vec<XFieldElement>, Vec<Vec<BFieldElement>>) {
        let mut rng = rand::thread_rng();
        let randomizer_coefficients =
            XFieldElement::random_elements(self.max_degree as usize + 1, &mut rng);
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

    fn sample_weights(
        seed: <StarkHasher as Hasher>::Digest,
        hasher: &StarkHasher,
        num: usize,
    ) -> Vec<XFieldElement> {
        (0..num as u64)
            .map(BFieldElement::new)
            .map(|bfe| [seed.to_sequence(), bfe.to_sequence()].concat())
            .map(|s| hasher.hash_sequence(&s))
            .map(|digest| XFieldElement::sample(&digest))
            .collect()
    }

    pub fn verify(&self, proof_stream: &mut StarkProofStream) -> Result<bool, Box<dyn Error>> {
        let mut timer = TimingReporter::start();
        let hasher = StarkHasher::new();

        let base_merkle_tree_root = proof_stream.dequeue()?.as_merkle_root()?;
        let extension_challenge_seed = proof_stream.verifier_fiat_shamir();
        timer.elapsed("Fiat-Shamir seed for extension challenges");

        let extension_challenge_weights = Self::sample_weights(
            extension_challenge_seed,
            &hasher,
            AllChallenges::TOTAL_CHALLENGES,
        );
        let extension_challenges = AllChallenges::create_challenges(extension_challenge_weights);
        timer.elapsed("Create extension challenges");

        let padded_height = proof_stream.dequeue()?.as_padded_heights()?.value() as usize;
        timer.elapsed("Got padded height");

        let extension_tree_merkle_root = proof_stream.dequeue()?.as_merkle_root()?;
        timer.elapsed("Get extension tree's root & terminals from proof stream");

        let ext_table_collection = ExtTableCollection::for_verifier(
            self.num_trace_randomizers,
            padded_height,
            &extension_challenges,
        );

        let base_degree_bounds =
            ext_table_collection.get_all_base_degree_bounds(self.num_trace_randomizers);
        timer.elapsed("Calculated base degree bounds");

        let extension_degree_bounds =
            ext_table_collection.get_extension_degree_bounds(self.num_trace_randomizers);
        timer.elapsed("Calculated extension degree bounds");

        let quotient_degree_bounds =
            ext_table_collection.get_all_quotient_degree_bounds(self.num_trace_randomizers);
        timer.elapsed("Calculated quotient degree bounds");

        // get weights for nonlinear combination:
        //  - 1 for randomizer polynomials,
        //  - 2 for {base, extension} polynomials and quotients.
        // The latter require 2 weights because transition constraints check 2 rows.
        let num_base_polynomials = base_degree_bounds.len();
        let num_extension_polynomials = extension_degree_bounds.len();
        let num_grand_cross_table_args = 1;
        let num_non_lin_combi_weights = self.num_randomizer_polynomials
            + 2 * num_base_polynomials
            + 2 * num_extension_polynomials
            + 2 * quotient_degree_bounds.len()
            + 2 * num_grand_cross_table_args;
        let num_grand_cross_table_arg_weights = NUM_CROSS_TABLE_ARGS + NUM_PUBLIC_EVAL_ARGS;

        let grand_cross_table_arg_and_non_lin_combi_weights_seed =
            proof_stream.verifier_fiat_shamir();
        let grand_cross_table_arg_and_non_lin_combi_weights = Self::sample_weights(
            grand_cross_table_arg_and_non_lin_combi_weights_seed,
            &hasher,
            num_grand_cross_table_arg_weights + num_non_lin_combi_weights,
        );
        let (grand_cross_table_argument_weights, non_lin_combi_weights) =
            grand_cross_table_arg_and_non_lin_combi_weights
                .split_at(num_grand_cross_table_arg_weights);
        timer.elapsed("Sample weights for grand cross-table argument and non-linear combination");

        let input_terminal = EvalArg::compute_terminal(
            &self.input_symbols,
            EvalArg::default_initial(),
            extension_challenges
                .processor_table_challenges
                .input_table_eval_row_weight,
        );
        let output_terminal = EvalArg::compute_terminal(
            &self.output_symbols,
            EvalArg::default_initial(),
            extension_challenges
                .processor_table_challenges
                .output_table_eval_row_weight,
        );
        let grand_cross_table_arg = GrandCrossTableArg::new(
            grand_cross_table_argument_weights.try_into().unwrap(),
            input_terminal,
            output_terminal,
        );
        timer.elapsed("Setup for grand cross-table argument");

        let combination_root = proof_stream.dequeue()?.as_merkle_root()?;

        let indices_seed = proof_stream.verifier_fiat_shamir();
        let combination_check_indices =
            hasher.sample_indices(self.security_level, &indices_seed, self.xfri.domain.length);
        let num_idxs = combination_check_indices.len();
        timer.elapsed("Got indices");

        // Verify low degree of combination polynomial
        self.xfri.verify(proof_stream, &combination_root)?;
        timer.elapsed("Verified FRI proof");

        // the relation between the FRI domain and the omicron domain
        let unit_distance = self.xfri.domain.length / ext_table_collection.padded_height;
        // Open leafs of zipped codewords at indicated positions
        let revealed_indices = self.get_revealed_indices(unit_distance, &combination_check_indices);
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
            .map(|revealed_base_elem| hasher.hash_sequence(revealed_base_elem))
            .collect();
        timer.elapsed(&format!("Got {num_idxs} leaf digests for base elements"));

        if !MerkleTree::<StarkHasher>::verify_authentication_structure_from_leaves(
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
                hasher.hash_sequence(&bvalues)
            })
            .collect();
        timer.elapsed(&format!("Got {num_idxs} leaf digests for ext elements"));

        if !MerkleTree::<StarkHasher>::verify_authentication_structure_from_leaves(
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
        let revealed_combination_leafs =
            proof_stream.dequeue()?.as_revealed_combination_elements()?;
        let revealed_combination_digests: Vec<_> = revealed_combination_leafs
            .par_iter()
            .map(|xfe| hasher.hash_sequence(&xfe.to_sequence()))
            .collect();
        let revealed_combination_auth_paths = proof_stream
            .dequeue()?
            .as_compressed_authentication_paths()?;
        if !MerkleTree::<StarkHasher>::verify_authentication_structure_from_leaves(
            combination_root,
            &combination_check_indices,
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
        let omicron: XFieldElement = derive_omicron(padded_height as u64);
        let omicron_inverse = omicron.inverse();
        for (combination_check_index, revealed_combination_leaf) in combination_check_indices
            .into_iter()
            .zip_eq(revealed_combination_leafs)
        {
            let current_fri_domain_value = self
                .xfri
                .domain
                .domain_value(combination_check_index as u32);
            let cross_slice = &index_map_of_revealed_elems[&combination_check_index];

            // populate summands with a cross-slice from the randomizer codewords
            let mut summands = cross_slice[0..base_offset].to_vec();

            // populate summands with a cross-slice of (base,ext) codewords and their shifts
            for (index_range, degree_bounds) in [
                (base_offset..ext_offset, base_degree_bounds.iter()),
                (ext_offset..final_offset, extension_degree_bounds.iter()),
            ] {
                for (codeword_index, degree_bound) in index_range.zip_eq(degree_bounds) {
                    let shift = self.max_degree - degree_bound;
                    let curr_codeword_elem = cross_slice[codeword_index];
                    let curr_codeword_elem_shifted =
                        curr_codeword_elem * current_fri_domain_value.mod_pow_u32(shift as u32);
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

                let next_cross_slice_index =
                    (combination_check_index + unit_distance) % self.xfri.domain.length;
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
                let initial_quotient_degree_bounds = table
                    .get_initial_quotient_degree_bounds(padded_height, self.num_trace_randomizers);
                let consistency_quotient_degree_bounds = table
                    .get_consistency_quotient_degree_bounds(
                        padded_height,
                        self.num_trace_randomizers,
                    );
                let transition_quotient_degree_bounds = table
                    .get_transition_quotient_degree_bounds(
                        padded_height,
                        self.num_trace_randomizers,
                    );
                let terminal_quotient_degree_bounds = table
                    .get_terminal_quotient_degree_bounds(padded_height, self.num_trace_randomizers);

                for (evaluated_bc, degree_bound) in table
                    .evaluate_initial_constraints(table_row)
                    .into_iter()
                    .zip_eq(initial_quotient_degree_bounds.iter())
                {
                    let shift = self.max_degree - degree_bound;
                    let quotient = evaluated_bc / (current_fri_domain_value - 1.into());
                    let quotient_shifted =
                        quotient * current_fri_domain_value.mod_pow_u32(shift as u32);
                    summands.push(quotient);
                    summands.push(quotient_shifted);
                }

                for (evaluated_cc, degree_bound) in table
                    .evaluate_consistency_constraints(table_row)
                    .into_iter()
                    .zip_eq(consistency_quotient_degree_bounds.iter())
                {
                    let shift = self.max_degree - degree_bound;
                    let quotient = evaluated_cc
                        / (current_fri_domain_value.mod_pow_u32(padded_height as u32) - 1.into());
                    let quotient_shifted =
                        quotient * current_fri_domain_value.mod_pow_u32(shift as u32);
                    summands.push(quotient);
                    summands.push(quotient_shifted);
                }

                let tc_evaluation_point = [table_row.clone(), next_table_row.clone()].concat();
                for (evaluated_tc, degree_bound) in table
                    .evaluate_transition_constraints(&tc_evaluation_point)
                    .into_iter()
                    .zip_eq(transition_quotient_degree_bounds.iter())
                {
                    let shift = self.max_degree - degree_bound;
                    let quotient = {
                        let numerator = current_fri_domain_value - omicron_inverse;
                        let denominator =
                            current_fri_domain_value.mod_pow_u32(padded_height as u32) - 1.into();
                        evaluated_tc * numerator / denominator
                    };
                    let quotient_shifted =
                        quotient * current_fri_domain_value.mod_pow_u32(shift as u32);
                    summands.push(quotient);
                    summands.push(quotient_shifted);
                }

                for (evaluated_termc, degree_bound) in table
                    .evaluate_terminal_constraints(table_row)
                    .into_iter()
                    .zip_eq(terminal_quotient_degree_bounds.iter())
                {
                    let shift = self.max_degree - degree_bound;
                    let quotient = evaluated_termc / (current_fri_domain_value - omicron_inverse);
                    let quotient_shifted =
                        quotient * current_fri_domain_value.mod_pow_u32(shift as u32);
                    summands.push(quotient);
                    summands.push(quotient_shifted);
                }
            }

            let grand_cross_table_arg_degree_bound = grand_cross_table_arg
                .quotient_degree_bound(&ext_table_collection, self.num_trace_randomizers);
            let shift = self.max_degree - grand_cross_table_arg_degree_bound;
            let grand_cross_table_arg_evaluated =
                grand_cross_table_arg.evaluate_non_linear_sum_of_differences(&cross_slice_by_table);
            let grand_cross_table_arg_quotient =
                grand_cross_table_arg_evaluated / (current_fri_domain_value - omicron_inverse);
            let grand_cross_table_arg_quotient_shifted =
                grand_cross_table_arg_quotient * current_fri_domain_value.mod_pow_u32(shift as u32);
            summands.push(grand_cross_table_arg_quotient);
            summands.push(grand_cross_table_arg_quotient_shifted);

            let inner_product = non_lin_combi_weights
                .par_iter()
                .zip_eq(summands.par_iter())
                .map(|(&weight, &summand)| weight * summand)
                .sum();

            // FIXME: This assert looks like it's for development, but it's the actual integrity
            //  check. Change to `if (…) { return Ok(false) }` or whatever is suitable.
            assert_eq!(
                revealed_combination_leaf, inner_product,
                "The combination leaf must equal the inner product"
            );
        }
        timer.elapsed(&format!("Verified {num_idxs} non-linear combinations"));
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

    fn num_trace_randomizers(security_level: usize) -> usize {
        2 * security_level
    }
}

#[cfg(test)]
pub(crate) mod triton_stark_tests {
    use num_traits::{One, Zero};
    use twenty_first::shared_math::ntt::ntt;
    use twenty_first::shared_math::other::log_2_floor;
    use twenty_first::util_types::proof_stream_typed::ProofStream;

    use crate::cross_table_arguments::EvalArg;
    use crate::instruction::sample_programs;
    use crate::stdio::VecStream;
    use crate::table::base_matrix::AlgebraicExecutionTrace;
    use crate::table::base_table::InheritsFromTable;
    use crate::table::table_collection::TableId::ProcessorTable;
    use crate::table::table_column::ProcessorExtTableColumn::{
        InputTableEvalArg, OutputTableEvalArg,
    };
    use crate::vm::triton_vm_tests::{all_tasm_test_programs, test_hash_nop_nop_lt};
    use crate::vm::Program;

    use super::*;

    #[test]
    fn all_tables_pad_to_same_height_test() {
        let code = "read_io read_io push -1 mul add split push 0 eq swap1 pop "; // simulates LTE
        let input_symbols = [5_u64.into(), 7_u64.into()];
        let (aet, _, program) = parse_setup_simulate(code, &input_symbols, &[]);
        let base_matrices = BaseMatrices::new(aet, &program);
        let mut base_tables = BaseTableCollection::from_base_matrices(&base_matrices);
        base_tables.pad();
        let padded_height = base_tables.padded_height;
        assert_eq!(padded_height, base_tables.program_table.data().len());
        assert_eq!(padded_height, base_tables.instruction_table.data().len());
        assert_eq!(padded_height, base_tables.processor_table.data().len());
        assert_eq!(padded_height, base_tables.op_stack_table.data().len());
        assert_eq!(padded_height, base_tables.ram_table.data().len());
        assert_eq!(padded_height, base_tables.jump_stack_table.data().len());
        assert_eq!(padded_height, base_tables.hash_table.data().len());
        assert_eq!(padded_height, base_tables.u32_op_table.data().len());
    }

    pub fn parse_simulate_prove(
        code: &str,
        co_set_fri_offset: BFieldElement,
        input_symbols: &[BFieldElement],
        output_symbols: &[BFieldElement],
    ) -> (
        Stark,
        ProofStream<ProofItem<RescuePrimeRegular>, RescuePrimeRegular>,
    ) {
        let (aet, _, program) = parse_setup_simulate(code, input_symbols, &[]);
        let base_matrices = BaseMatrices::new(aet, &program);

        let num_randomizer_polynomials = 1;
        let log_expansion_factor = 2;
        let security_level = 32;
        let padded_height = BaseTableCollection::padded_height(&base_matrices);

        let stark = Stark::new(
            padded_height,
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

    pub fn parse_setup_simulate(
        code: &str,
        input_symbols: &[BFieldElement],
        secret_input_symbols: &[BFieldElement],
    ) -> (AlgebraicExecutionTrace, VecStream, Program) {
        let program = Program::from_code(code);

        assert!(program.is_ok(), "program parses correctly");
        let program = program.unwrap();

        let mut stdin = VecStream::new(input_symbols);
        let mut secret_in = VecStream::new(secret_input_symbols);
        let mut stdout = VecStream::new(&[]);

        let (aet, err) = program.simulate(&mut stdin, &mut secret_in, &mut stdout);
        if let Some(error) = err {
            panic!("The VM encountered the following problem: {}", error);
        }
        (aet, stdout, program)
    }

    pub fn parse_simulate_pad_extend(
        code: &str,
        stdin: &[BFieldElement],
        secret_in: &[BFieldElement],
    ) -> (
        VecStream,
        BaseTableCollection,
        BaseTableCollection,
        ExtTableCollection,
        AllChallenges,
        usize,
    ) {
        let (aet, stdout, program) = parse_setup_simulate(code, stdin, secret_in);
        let base_matrices = BaseMatrices::new(aet, &program);

        let num_trace_randomizers = 2;
        let mut base_tables = BaseTableCollection::from_base_matrices(&base_matrices);

        let unpadded_base_tables = base_tables.clone();

        base_tables.pad();

        let dummy_challenges = AllChallenges::dummy();
        let ext_tables = ExtTableCollection::extend_tables(
            &base_tables,
            &dummy_challenges,
            num_trace_randomizers,
        );

        (
            stdout,
            unpadded_base_tables,
            base_tables,
            ext_tables,
            dummy_challenges,
            num_trace_randomizers,
        )
    }

    /// To be used with `-- --nocapture`. Has mainly informative purpose.
    #[test]
    pub fn print_all_constraint_degrees() {
        let (_, _, _, ext_tables, _, num_trace_randomizers) =
            parse_simulate_pad_extend("halt", &[], &[]);
        let padded_height = ext_tables.padded_height;
        let all_degrees = ext_tables
            .into_iter()
            .map(|ext_table| {
                ext_table.all_degrees_with_origin(padded_height, num_trace_randomizers)
            })
            .concat();
        for deg in all_degrees {
            println!("{}", deg);
        }
    }

    #[test]
    pub fn shift_codeword_test() {
        let stark = Stark::new(2, 1, 2, 32, BFieldElement::one(), &[], &[]);
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
    pub fn check_io_terminals() {
        let read_nop_code = "
            read_io read_io read_io
            nop nop
            write_io push 17 write_io
        ";
        let input_symbols = vec![
            BFieldElement::new(3),
            BFieldElement::new(5),
            BFieldElement::new(7),
        ];
        let (stdout, _, _, ext_table_collection, all_challenges, _) =
            parse_simulate_pad_extend(read_nop_code, &input_symbols, &[]);

        let ptie = ext_table_collection.data(ProcessorTable).last().unwrap()
            [usize::from(InputTableEvalArg)];
        let ine = EvalArg::compute_terminal(
            &input_symbols,
            EvalArg::default_initial(),
            all_challenges.input_challenges.processor_eval_row_weight,
        );
        assert_eq!(ptie, ine, "The input evaluation arguments do not match.");

        let ptoe = ext_table_collection.data(ProcessorTable).last().unwrap()
            [usize::from(OutputTableEvalArg)];

        let oute = EvalArg::compute_terminal(
            &stdout.to_bword_vec(),
            EvalArg::default_initial(),
            all_challenges.output_challenges.processor_eval_row_weight,
        );
        assert_eq!(ptoe, oute, "The output evaluation arguments do not match.");
    }

    #[test]
    pub fn check_all_cross_table_terminals() {
        let code_collection = all_tasm_test_programs();
        for (code_idx, code_with_input) in code_collection.into_iter().enumerate() {
            let code = code_with_input.source_code;
            let input = code_with_input.input;
            let secret_input = code_with_input.secret_input;
            let (output, _, _, ext_table_collection, all_challenges, _) =
                parse_simulate_pad_extend(&code, &input, &secret_input);

            let input_terminal = EvalArg::compute_terminal(
                &input,
                EvalArg::default_initial(),
                all_challenges
                    .processor_table_challenges
                    .input_table_eval_row_weight,
            );

            let output_terminal = EvalArg::compute_terminal(
                &output.to_bword_vec(),
                EvalArg::default_initial(),
                all_challenges
                    .processor_table_challenges
                    .output_table_eval_row_weight,
            );

            let grand_cross_table_arg = GrandCrossTableArg::new(
                &[XFieldElement::one(); NUM_CROSS_TABLE_ARGS + NUM_PUBLIC_EVAL_ARGS],
                input_terminal,
                output_terminal,
            );

            for (idx, (arg, _)) in grand_cross_table_arg.into_iter().enumerate() {
                let (from_table, from_column) = arg.from();
                let (to_table, to_column) = arg.to();
                assert_eq!(
                    ext_table_collection.data(from_table).last().unwrap()[from_column],
                    ext_table_collection.data(to_table).last().unwrap()[to_column],
                    "Cross-table argument #{idx} must match for TASM snipped #{code_idx}."
                );
            }

            let ptie = ext_table_collection.data(ProcessorTable).last().unwrap()
                [usize::from(InputTableEvalArg)];
            assert_eq!(
                ptie, input_terminal,
                "The input terminal must match for TASM snipped #{code_idx}."
            );

            let ptoe = ext_table_collection.data(ProcessorTable).last().unwrap()
                [usize::from(OutputTableEvalArg)];
            assert_eq!(
                ptoe, output_terminal,
                "The output terminal must match for TASM snipped #{code_idx}."
            );
        }
    }

    #[test]
    fn constraint_polynomials_use_right_variable_count_test() {
        let (_, _, _, ext_tables, _, _) = parse_simulate_pad_extend("halt", &[], &[]);

        for table in ext_tables.into_iter() {
            let dummy_row = vec![0.into(); table.full_width()];
            let double_length_dummy_row = vec![0.into(); 2 * table.full_width()];

            // will panic if the number of variables is wrong
            table.evaluate_initial_constraints(&dummy_row);
            table.evaluate_consistency_constraints(&dummy_row);
            table.evaluate_transition_constraints(&double_length_dummy_row);
            table.evaluate_terminal_constraints(&dummy_row);
        }
    }

    #[test]
    fn triton_table_constraints_evaluate_to_zero_test() {
        let zero = XFieldElement::zero();
        let (_, _, _, ext_tables, _, _) =
            parse_simulate_pad_extend(sample_programs::FIBONACCI_LT, &[], &[]);

        for table in (&ext_tables).into_iter() {
            if let Some(row) = table.data().get(0) {
                let evaluated_bcs = table.evaluate_initial_constraints(&row);
                for (constraint_idx, ebc) in evaluated_bcs.into_iter().enumerate() {
                    assert_eq!(
                        zero,
                        ebc,
                        "Failed initial constraint on {}. Constraint index: {}. Row index: {}",
                        table.name(),
                        constraint_idx,
                        0,
                    );
                }
            }

            for (row_idx, curr_row) in table.data().iter().enumerate() {
                let evaluated_ccs = table.evaluate_consistency_constraints(&curr_row);
                for (constraint_idx, ecc) in evaluated_ccs.into_iter().enumerate() {
                    assert_eq!(
                        zero,
                        ecc,
                        "Failed consistency constraint {}. Constraint index: {}. Row index: {}",
                        table.name(),
                        constraint_idx,
                        row_idx,
                    );
                }
            }

            for (row_idx, (curr_row, next_row)) in table.data().iter().tuple_windows().enumerate() {
                let evaluation_point = [curr_row.to_vec(), next_row.to_vec()].concat();
                let evaluated_tcs = table.evaluate_transition_constraints(&evaluation_point);
                for (constraint_idx, evaluated_tc) in evaluated_tcs.into_iter().enumerate() {
                    assert_eq!(
                        zero,
                        evaluated_tc,
                        "Failed transition constraint on {}. Constraint index: {}. Row index: {}",
                        table.name(),
                        constraint_idx,
                        row_idx,
                    );
                }
            }

            if let Some(row) = table.data().last() {
                let evaluated_termcs = table.evaluate_terminal_constraints(&row);
                for (constraint_idx, etermc) in evaluated_termcs.into_iter().enumerate() {
                    assert_eq!(
                        zero,
                        etermc,
                        "Failed terminal constraint on {}. Constraint index: {}. Row index: {}",
                        table.name(),
                        constraint_idx,
                        table.data().len() - 1,
                    );
                }
            }
        }
    }

    #[test]
    fn triton_prove_verify_test() {
        let co_set_fri_offset = BFieldElement::generator();
        let code_with_input = test_hash_nop_nop_lt();
        let (stark, mut proof_stream) = parse_simulate_prove(
            &code_with_input.source_code,
            co_set_fri_offset,
            &code_with_input.input,
            &code_with_input.secret_input,
        );

        println!("between prove and verify");

        let result = stark.verify(&mut proof_stream);
        if let Err(e) = result {
            panic!("The Verifier is unhappy! {}", e);
        }
        assert!(result.unwrap());
    }
}
