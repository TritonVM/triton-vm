use std::collections::HashMap;
use std::error::Error;
use std::fmt;

use itertools::Itertools;
use num_traits::One;
use rayon::iter::{
    IndexedParallelIterator, IntoParallelIterator, IntoParallelRefIterator, ParallelIterator,
};
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::mpolynomial::Degree;
use twenty_first::shared_math::other::{self, is_power_of_two, random_elements};
use twenty_first::shared_math::polynomial::Polynomial;
use twenty_first::shared_math::rescue_prime_digest::Digest;
use twenty_first::shared_math::rescue_prime_regular::RescuePrimeRegular;
use twenty_first::shared_math::traits::{FiniteField, Inverse, ModPowU32, PrimitiveRootOfUnity};
use twenty_first::shared_math::x_field_element::XFieldElement;
use twenty_first::util_types::algebraic_hasher::AlgebraicHasher;
use twenty_first::util_types::merkle_tree::MerkleTree;

use triton_profiler::triton_profiler::TritonProfiler;
use triton_profiler::{prof_itr0, prof_start, prof_stop};

use crate::cross_table_arguments::{
    CrossTableArg, EvalArg, GrandCrossTableArg, NUM_CROSS_TABLE_ARGS, NUM_PUBLIC_EVAL_ARGS,
};
use crate::fri::{Fri, FriValidationError};
use crate::fri_domain::FriDomain;
use crate::proof::{Claim, Proof};
use crate::proof_item::ProofItem;
use crate::proof_stream::ProofStream;
use crate::table::base_matrix::AlgebraicExecutionTrace;
use crate::table::challenges::AllChallenges;
use crate::table::table_collection::{derive_omicron, BaseTableCollection, ExtTableCollection};

use super::table::base_matrix::BaseMatrices;

pub type StarkHasher = RescuePrimeRegular;
pub type StarkProofStream = ProofStream<ProofItem, StarkHasher>;

pub struct StarkParameters {
    security_level: usize,
    fri_expansion_factor: usize,
    num_trace_randomizers: usize,
    num_randomizer_polynomials: usize,
    num_colinearity_checks: usize,
}

impl StarkParameters {
    pub fn new(security_level: usize, fri_expansion_factor: usize) -> Self {
        let num_randomizer_polynomials = 1; // over the XField

        assert!(
            is_power_of_two(fri_expansion_factor),
            "FRI expansion factor must be a power of two, but got {}.",
            fri_expansion_factor
        );
        assert!(
            fri_expansion_factor > 1,
            "FRI expansion factor must be greater than one, but got {}.",
            fri_expansion_factor
        );

        let mut log2_of_fri_expansion_factor = 0;
        while (1 << log2_of_fri_expansion_factor) < fri_expansion_factor {
            log2_of_fri_expansion_factor += 1;
        }
        // post-condition: 2^(log2_of_fri_expansion_factor) == fri_expansion_factor

        let num_colinearity_checks = security_level / log2_of_fri_expansion_factor;
        let num_trace_randomizers = num_colinearity_checks * 2;

        StarkParameters {
            security_level,
            fri_expansion_factor,
            num_trace_randomizers,
            num_randomizer_polynomials,
            num_colinearity_checks,
        }
    }
}

impl Default for StarkParameters {
    fn default() -> Self {
        let fri_expansion_factor = 4;
        let security_level = 160;

        Self::new(security_level, fri_expansion_factor)
    }
}

#[derive(PartialEq, Eq, Debug)]
pub enum StarkValidationError {
    CombinationLeafInequality,
    PaddedHeightInequality,
    FriValidationError(FriValidationError),
}

impl Error for StarkValidationError {}

impl fmt::Display for StarkValidationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "STARK error: {:?}", self)
    }
}

pub struct Stark {
    parameters: StarkParameters,
    claim: Claim,
    max_degree: Degree,
    fri_domain: FriDomain<BFieldElement>,
    fri: Fri<StarkHasher>,
}

impl Stark {
    pub fn new(claim: Claim, parameters: StarkParameters) -> Self {
        let empty_table_collection = ExtTableCollection::with_padded_height(claim.padded_height);
        let max_degree_with_origin =
            empty_table_collection.max_degree_with_origin(parameters.num_trace_randomizers);
        let max_degree = (other::roundup_npo2(max_degree_with_origin.degree as u64) - 1) as i64;
        let fri_domain_length = parameters.fri_expansion_factor * (max_degree as usize + 1);
        let omega =
            BFieldElement::primitive_root_of_unity(fri_domain_length.try_into().unwrap()).unwrap();
        let coset_offset = BFieldElement::generator();
        let fri_domain: FriDomain<BFieldElement> =
            FriDomain::new(coset_offset, omega, fri_domain_length);
        let fri = Fri::new(
            coset_offset,
            omega,
            fri_domain_length,
            parameters.fri_expansion_factor,
            parameters.num_colinearity_checks,
        );
        Self {
            parameters,
            claim,
            max_degree,
            fri_domain,
            fri,
        }
    }

    pub fn prove(
        &self,
        aet: AlgebraicExecutionTrace,
        maybe_profiler: &mut Option<TritonProfiler>,
    ) -> Proof {
        prof_start!(maybe_profiler, "pad");
        let base_matrices = BaseMatrices::new(aet, &self.claim.program);
        let base_trace_tables = self.padded(&base_matrices);
        prof_stop!(maybe_profiler, "pad");

        let (x_rand_codeword, b_rand_codewords) = self.get_randomizer_codewords();

        prof_start!(maybe_profiler, "LDE 1");
        let base_fri_domain_tables = base_trace_tables
            .to_fri_domain_tables(&self.fri_domain, self.parameters.num_trace_randomizers);
        let base_fri_domain_codewords = base_fri_domain_tables.get_all_base_columns();
        let randomizer_and_base_fri_domain_codewords =
            vec![b_rand_codewords, base_fri_domain_codewords.clone()].concat();
        prof_stop!(maybe_profiler, "LDE 1");

        prof_start!(maybe_profiler, "Merkle tree 1");
        let transposed_base_codewords = Self::transpose(&randomizer_and_base_fri_domain_codewords);
        let base_tree = Self::get_merkle_tree(&transposed_base_codewords);
        let base_merkle_tree_root = base_tree.get_root();
        prof_stop!(maybe_profiler, "Merkle tree 1");

        // send first message
        prof_start!(maybe_profiler, "Fiat-Shamir 1");
        let mut proof_stream = StarkProofStream::new();
        proof_stream.enqueue(&ProofItem::PaddedHeight(BFieldElement::new(
            base_trace_tables.padded_height as u64,
        )));
        proof_stream.enqueue(&ProofItem::MerkleRoot(base_merkle_tree_root));
        let extension_challenge_seed = proof_stream.prover_fiat_shamir();
        let extension_challenge_weights =
            Self::sample_weights(extension_challenge_seed, AllChallenges::TOTAL_CHALLENGES);
        let extension_challenges = AllChallenges::create_challenges(extension_challenge_weights);
        prof_stop!(maybe_profiler, "Fiat-Shamir 1");

        prof_start!(maybe_profiler, "extend");
        let ext_trace_tables = ExtTableCollection::extend_tables(
            &base_trace_tables,
            &extension_challenges,
            self.parameters.num_trace_randomizers,
        );
        prof_stop!(maybe_profiler, "extend");

        prof_start!(maybe_profiler, "LDE 2");
        let ext_fri_domain_tables = ext_trace_tables
            .to_fri_domain_tables(&self.fri.domain, self.parameters.num_trace_randomizers);
        let extension_fri_domain_codewords = ext_fri_domain_tables.collect_all_columns();

        prof_stop!(maybe_profiler, "LDE 2");

        prof_start!(maybe_profiler, "Merkle tree 2");
        let transposed_ext_codewords = Self::transpose(&extension_fri_domain_codewords);
        let extension_tree = Self::get_extension_merkle_tree(&transposed_ext_codewords);
        prof_stop!(maybe_profiler, "Merkle tree 2");

        // send root for extension codewords
        proof_stream.enqueue(&ProofItem::MerkleRoot(extension_tree.get_root()));

        prof_start!(maybe_profiler, "degree bounds");
        prof_start!(maybe_profiler, "base");
        let base_degree_bounds =
            base_fri_domain_tables.get_base_degree_bounds(self.parameters.num_trace_randomizers);
        prof_stop!(maybe_profiler, "base");

        prof_start!(maybe_profiler, "extension");
        let extension_degree_bounds = ext_fri_domain_tables
            .get_extension_degree_bounds(self.parameters.num_trace_randomizers);
        prof_stop!(maybe_profiler, "extension");

        prof_start!(maybe_profiler, "quotient");
        let full_fri_domain_tables =
            ExtTableCollection::join(base_fri_domain_tables, ext_fri_domain_tables);
        let mut quotient_degree_bounds = full_fri_domain_tables
            .get_all_quotient_degree_bounds(self.parameters.num_trace_randomizers);
        prof_stop!(maybe_profiler, "quotient");
        prof_stop!(maybe_profiler, "degree bounds");

        prof_start!(maybe_profiler, "quotient codewords");
        let mut quotient_codewords = full_fri_domain_tables.get_all_quotients(
            &self.fri.domain,
            &extension_challenges,
            maybe_profiler,
        );
        prof_stop!(maybe_profiler, "quotient codewords");

        prof_start!(maybe_profiler, "grand cross table");
        let num_grand_cross_table_args = 1;
        let num_non_lin_combi_weights = self.parameters.num_randomizer_polynomials
            + 2 * base_fri_domain_codewords.len()
            + 2 * extension_fri_domain_codewords.len()
            + 2 * quotient_degree_bounds.len()
            + 2 * num_grand_cross_table_args;
        let num_grand_cross_table_arg_weights = NUM_CROSS_TABLE_ARGS + NUM_PUBLIC_EVAL_ARGS;

        let grand_cross_table_arg_and_non_lin_combi_weights_seed =
            proof_stream.prover_fiat_shamir();
        let grand_cross_table_arg_and_non_lin_combi_weights = Self::sample_weights(
            grand_cross_table_arg_and_non_lin_combi_weights_seed,
            num_grand_cross_table_arg_weights + num_non_lin_combi_weights,
        );
        let (grand_cross_table_argument_weights, non_lin_combi_weights) =
            grand_cross_table_arg_and_non_lin_combi_weights
                .split_at(num_grand_cross_table_arg_weights);

        // prove equal terminal values for the column tuples pertaining to cross table arguments
        let input_terminal = EvalArg::compute_terminal(
            &self.claim.input,
            EvalArg::default_initial(),
            extension_challenges
                .processor_table_challenges
                .standard_input_eval_indeterminate,
        );
        let output_terminal = EvalArg::compute_terminal(
            &self.claim.output,
            EvalArg::default_initial(),
            extension_challenges
                .processor_table_challenges
                .standard_output_eval_indeterminate,
        );
        let grand_cross_table_arg = GrandCrossTableArg::new(
            grand_cross_table_argument_weights.try_into().unwrap(),
            input_terminal,
            output_terminal,
        );
        let grand_cross_table_arg_quotient_codeword = grand_cross_table_arg
            .terminal_quotient_codeword(
                &full_fri_domain_tables,
                &self.fri.domain,
                derive_omicron(full_fri_domain_tables.padded_height as u64),
            );
        quotient_codewords.push(grand_cross_table_arg_quotient_codeword);

        let grand_cross_table_arg_quotient_degree_bound = grand_cross_table_arg
            .quotient_degree_bound(
                &full_fri_domain_tables,
                self.parameters.num_trace_randomizers,
            );
        quotient_degree_bounds.push(grand_cross_table_arg_quotient_degree_bound);
        prof_stop!(maybe_profiler, "grand cross table");

        prof_start!(maybe_profiler, "nonlinear combination");
        let combination_codeword = self.create_combination_codeword(
            vec![x_rand_codeword],
            base_fri_domain_codewords,
            extension_fri_domain_codewords,
            quotient_codewords,
            non_lin_combi_weights.to_vec(),
            base_degree_bounds,
            extension_degree_bounds,
            quotient_degree_bounds,
            maybe_profiler,
        );
        prof_stop!(maybe_profiler, "nonlinear combination");

        prof_start!(maybe_profiler, "Merkle tree 3");
        let mut combination_codeword_digests: Vec<Digest> =
            Vec::with_capacity(combination_codeword.len());
        combination_codeword
            .clone()
            .into_par_iter()
            .map(|elem| StarkHasher::hash(&elem))
            .collect_into_vec(&mut combination_codeword_digests);
        let combination_tree =
            MerkleTree::<StarkHasher>::from_digests(&combination_codeword_digests);
        let combination_root: Digest = combination_tree.get_root();

        proof_stream.enqueue(&ProofItem::MerkleRoot(combination_root));

        prof_stop!(maybe_profiler, "Merkle tree 3");

        // Get indices of slices that go across codewords to prove nonlinear combination
        if let Some(profiler) = maybe_profiler.as_mut() {
            profiler.start("Fiat-Shamir 3");
        }
        let indices_seed = proof_stream.prover_fiat_shamir();
        let cross_codeword_slice_indices = StarkHasher::sample_indices(
            self.parameters.security_level,
            &indices_seed,
            self.fri.domain.length,
        );
        if let Some(profiler) = maybe_profiler.as_mut() {
            profiler.stop("Fiat-Shamir 3");
        }

        prof_start!(maybe_profiler, "FRI");
        match self.fri.prove(&combination_codeword, &mut proof_stream) {
            Ok((_, fri_first_round_merkle_root)) => assert_eq!(
                combination_root, fri_first_round_merkle_root,
                "Combination root from STARK and from FRI must agree."
            ),
            Err(e) => panic!("The FRI prover failed because of: {}", e),
        }
        prof_stop!(maybe_profiler, "FRI");

        prof_start!(maybe_profiler, "open trace leafs");
        // the relation between the FRI domain and the omicron domain
        let unit_distance = self.fri.domain.length / base_trace_tables.padded_height;
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

        prof_stop!(maybe_profiler, "open trace leafs");

        println!(
            "Created proof containing {} B-field elements",
            proof_stream.transcript_length()
        );

        proof_stream.to_proof()
    }

    fn get_revealed_indices(
        &self,
        unit_distance: usize,
        cross_codeword_slice_indices: &[usize],
    ) -> Vec<usize> {
        let mut revealed_indices: Vec<usize> = vec![];
        for &index in cross_codeword_slice_indices.iter() {
            revealed_indices.push(index);
            revealed_indices.push((index + unit_distance) % self.fri.domain.length);
        }
        revealed_indices.sort_unstable();
        revealed_indices.dedup();
        revealed_indices
    }

    fn get_revealed_elements<FF: FiniteField>(
        transposed_base_codewords: &[Vec<FF>],
        revealed_indices: &[usize],
    ) -> Vec<Vec<FF>> {
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
        randomizer_codewords: Vec<Vec<XFieldElement>>,
        base_codewords: Vec<Vec<BFieldElement>>,
        extension_codewords: Vec<Vec<XFieldElement>>,
        quotient_codewords: Vec<Vec<XFieldElement>>,
        weights: Vec<XFieldElement>,
        base_degree_bounds: Vec<i64>,
        extension_degree_bounds: Vec<i64>,
        quotient_degree_bounds: Vec<i64>,
        maybe_profiler: &mut Option<TritonProfiler>,
    ) -> Vec<XFieldElement> {
        assert_eq!(
            self.parameters.num_randomizer_polynomials,
            randomizer_codewords.len()
        );

        prof_start!(maybe_profiler, "create combination codeword");

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
        let mut combination_codeword: Vec<XFieldElement> = vec![0.into(); self.fri.domain.length];

        // TODO don't keep the entire domain's values in memory, create them lazily when needed
        let fri_x_values = self.fri.domain.domain_values();

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
            println!("{}", identifier);
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
        }

        if std::env::var("DEBUG").is_ok() {
            println!(
                "The combination codeword corresponds to a polynomial of degree {}",
                self.fri.domain.interpolate(&combination_codeword).degree()
            );
        }

        prof_stop!(maybe_profiler, "create combination codeword");

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
        let interpolated = self.fri.domain.interpolate(extension_codeword);
        let interpolated_shifted = self.fri.domain.interpolate(extension_codeword_shifted);
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
        transposed_extension_codewords: &Vec<Vec<XFieldElement>>,
    ) -> MerkleTree<StarkHasher> {
        let mut extension_codeword_digests_by_index =
            Vec::with_capacity(transposed_extension_codewords.len());

        transposed_extension_codewords
            .into_par_iter()
            .map(|transposed_ext_codeword| {
                let transposed_ext_codeword_coeffs: Vec<BFieldElement> = transposed_ext_codeword
                    .iter()
                    .map(|elem| elem.coefficients.to_vec())
                    .concat();

                StarkHasher::hash_slice(&transposed_ext_codeword_coeffs)
            })
            .collect_into_vec(&mut extension_codeword_digests_by_index);

        MerkleTree::<StarkHasher>::from_digests(&extension_codeword_digests_by_index)
    }

    fn get_merkle_tree(codewords: &Vec<Vec<BFieldElement>>) -> MerkleTree<StarkHasher> {
        let mut codeword_digests_by_index = Vec::with_capacity(codewords.len());
        codewords
            .par_iter()
            .map(|values| StarkHasher::hash_slice(values))
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
    pub fn transpose<P: Copy>(codewords: &[Vec<P>]) -> Vec<Vec<P>> {
        (0..codewords[0].len())
            .map(|col_idx| codewords.iter().map(|row| row[col_idx]).collect())
            .collect()
    }

    fn padded(&self, base_matrices: &BaseMatrices) -> BaseTableCollection {
        let mut base_tables = BaseTableCollection::from_base_matrices(base_matrices);
        base_tables.pad();
        base_tables
    }

    fn get_randomizer_codewords(&self) -> (Vec<XFieldElement>, Vec<Vec<BFieldElement>>) {
        let randomizer_coefficients = random_elements(self.max_degree as usize + 1);
        let randomizer_polynomial = Polynomial::new(randomizer_coefficients);

        let x_randomizer_codeword = self.fri.domain.evaluate(&randomizer_polynomial);
        let mut b_randomizer_codewords = vec![vec![], vec![], vec![]];
        for x_elem in x_randomizer_codeword.iter() {
            b_randomizer_codewords[0].push(x_elem.coefficients[0]);
            b_randomizer_codewords[1].push(x_elem.coefficients[1]);
            b_randomizer_codewords[2].push(x_elem.coefficients[2]);
        }
        (x_randomizer_codeword, b_randomizer_codewords)
    }

    fn sample_weights(seed: Digest, num_weights: usize) -> Vec<XFieldElement> {
        StarkHasher::get_n_hash_rounds(&seed, num_weights)
            .iter()
            .map(XFieldElement::sample)
            .collect()
    }

    pub fn verify(
        &self,
        proof: Proof,
        maybe_profiler: &mut Option<TritonProfiler>,
    ) -> Result<bool, Box<dyn Error>> {
        prof_start!(maybe_profiler, "deserialize");
        let mut proof_stream = StarkProofStream::from_proof(&proof)?;
        prof_stop!(maybe_profiler, "deserialize");

        prof_start!(maybe_profiler, "Fiat-Shamir 1");
        let padded_height = proof_stream.dequeue()?.as_padded_heights()?.value() as usize;
        let base_merkle_tree_root = proof_stream.dequeue()?.as_merkle_root()?;

        let extension_challenge_seed = proof_stream.verifier_fiat_shamir();

        let extension_challenge_weights =
            Self::sample_weights(extension_challenge_seed, AllChallenges::TOTAL_CHALLENGES);
        let extension_challenges = AllChallenges::create_challenges(extension_challenge_weights);
        if self.claim.padded_height != padded_height && self.claim.padded_height != 0 {
            return Err(Box::new(StarkValidationError::PaddedHeightInequality));
        }
        prof_stop!(maybe_profiler, "Fiat-Shamir 1");

        prof_start!(maybe_profiler, "dequeue");

        let extension_tree_merkle_root = proof_stream.dequeue()?.as_merkle_root()?;
        prof_stop!(maybe_profiler, "dequeue");

        prof_start!(maybe_profiler, "degree bounds");
        prof_start!(maybe_profiler, "generate tables");
        let ext_table_collection = ExtTableCollection::for_verifier(
            self.parameters.num_trace_randomizers,
            padded_height,
            &extension_challenges,
            maybe_profiler,
        );
        prof_stop!(maybe_profiler, "generate tables");

        prof_start!(maybe_profiler, "base");
        let base_degree_bounds =
            ext_table_collection.get_all_base_degree_bounds(self.parameters.num_trace_randomizers);
        prof_stop!(maybe_profiler, "base");

        prof_start!(maybe_profiler, "extension");
        let extension_degree_bounds =
            ext_table_collection.get_extension_degree_bounds(self.parameters.num_trace_randomizers);
        prof_stop!(maybe_profiler, "extension");

        prof_start!(maybe_profiler, "quotient");
        let quotient_degree_bounds = ext_table_collection
            .get_all_quotient_degree_bounds(self.parameters.num_trace_randomizers);
        prof_stop!(maybe_profiler, "quotient");
        prof_stop!(maybe_profiler, "degree bounds");

        // get weights for nonlinear combination:
        //  - 1 for randomizer polynomials,
        //  - 2 for {base, extension} polynomials and quotients.
        // The latter require 2 weights because transition constraints check 2 rows.
        prof_start!(maybe_profiler, "Fiat-Shamir 2");
        let num_base_polynomials = base_degree_bounds.len();
        let num_extension_polynomials = extension_degree_bounds.len();
        let num_grand_cross_table_args = 1;
        let num_non_lin_combi_weights = self.parameters.num_randomizer_polynomials
            + 2 * num_base_polynomials
            + 2 * num_extension_polynomials
            + 2 * quotient_degree_bounds.len()
            + 2 * num_grand_cross_table_args;
        let num_grand_cross_table_arg_weights = NUM_CROSS_TABLE_ARGS + NUM_PUBLIC_EVAL_ARGS;

        let grand_cross_table_arg_and_non_lin_combi_weights_seed =
            proof_stream.verifier_fiat_shamir();
        let grand_cross_table_arg_and_non_lin_combi_weights = Self::sample_weights(
            grand_cross_table_arg_and_non_lin_combi_weights_seed,
            num_grand_cross_table_arg_weights + num_non_lin_combi_weights,
        );
        let (grand_cross_table_argument_weights, non_lin_combi_weights) =
            grand_cross_table_arg_and_non_lin_combi_weights
                .split_at(num_grand_cross_table_arg_weights);

        let input_terminal = EvalArg::compute_terminal(
            &self.claim.input,
            EvalArg::default_initial(),
            extension_challenges
                .processor_table_challenges
                .standard_input_eval_indeterminate,
        );
        let output_terminal = EvalArg::compute_terminal(
            &self.claim.output,
            EvalArg::default_initial(),
            extension_challenges
                .processor_table_challenges
                .standard_output_eval_indeterminate,
        );
        let grand_cross_table_arg = GrandCrossTableArg::new(
            grand_cross_table_argument_weights.try_into().unwrap(),
            input_terminal,
            output_terminal,
        );
        prof_stop!(maybe_profiler, "Fiat-Shamir 2");

        prof_start!(maybe_profiler, "Fiat-Shamir 3");
        let combination_root = proof_stream.dequeue()?.as_merkle_root()?;

        let indices_seed = proof_stream.verifier_fiat_shamir();
        let combination_check_indices = StarkHasher::sample_indices(
            self.parameters.security_level,
            &indices_seed,
            self.fri.domain.length,
        );
        prof_stop!(maybe_profiler, "Fiat-Shamir 3");

        // verify low degree of combination polynomial with FRI
        prof_start!(maybe_profiler, "FRI");
        self.fri
            .verify(&mut proof_stream, &combination_root, maybe_profiler)?;
        prof_stop!(maybe_profiler, "FRI");

        prof_start!(maybe_profiler, "check leafs");
        prof_start!(maybe_profiler, "get indices");
        // the relation between the FRI domain and the omicron domain
        let unit_distance = self.fri.domain.length / ext_table_collection.padded_height;
        // Open leafs of zipped codewords at indicated positions
        let revealed_indices = self.get_revealed_indices(unit_distance, &combination_check_indices);
        prof_stop!(maybe_profiler, "get indices");

        // TODO: in the following ~80 lines, we (conceptually) do the same thing three times. DRY.
        prof_start!(maybe_profiler, "dequeue");
        let revealed_base_elems = proof_stream
            .dequeue()?
            .as_transposed_base_element_vectors()?;
        let auth_paths_base = proof_stream
            .dequeue()?
            .as_compressed_authentication_paths()?;
        let leaf_digests_base: Vec<_> = revealed_base_elems
            .par_iter()
            .map(|revealed_base_elem| StarkHasher::hash_slice(revealed_base_elem))
            .collect();
        prof_stop!(maybe_profiler, "dequeue");

        prof_start!(maybe_profiler, "Merkle verify");
        if !MerkleTree::<StarkHasher>::verify_authentication_structure_from_leaves(
            base_merkle_tree_root,
            &revealed_indices,
            &leaf_digests_base,
            &auth_paths_base,
        ) {
            // TODO: Replace this by a specific error type, or just return `Ok(false)`
            panic!("Failed to verify authentication path for base codeword");
        }
        prof_stop!(maybe_profiler, "Merkle verify");

        prof_start!(maybe_profiler, "dequeue");
        let revealed_ext_elems = proof_stream
            .dequeue()?
            .as_transposed_extension_element_vectors()?;
        let auth_paths_ext = proof_stream
            .dequeue()?
            .as_compressed_authentication_paths()?;
        let leaf_digests_ext: Vec<_> = revealed_ext_elems
            .par_iter()
            .map(|xvalues| {
                let bvalues: Vec<BFieldElement> = xvalues
                    .iter()
                    .flat_map(|xfe| xfe.coefficients.to_vec())
                    .collect();
                StarkHasher::hash_slice(&bvalues)
            })
            .collect();
        prof_stop!(maybe_profiler, "dequeue");

        prof_start!(maybe_profiler, "Merkle verify (auth struct)");
        if !MerkleTree::<StarkHasher>::verify_authentication_structure_from_leaves(
            extension_tree_merkle_root,
            &revealed_indices,
            &leaf_digests_ext,
            &auth_paths_ext,
        ) {
            // TODO: Replace this by a specific error type, or just return `Ok(false)`
            panic!("Failed to verify authentication path for extension codeword");
        }
        prof_stop!(maybe_profiler, "Merkle verify (auth struct)");

        // Verify Merkle authentication path for combination elements
        prof_start!(maybe_profiler, "Merkle verify");
        let revealed_combination_leafs =
            proof_stream.dequeue()?.as_revealed_combination_elements()?;
        let revealed_combination_digests: Vec<_> = revealed_combination_leafs
            .par_iter()
            .map(StarkHasher::hash)
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
        prof_stop!(maybe_profiler, "Merkle verify");
        prof_stop!(maybe_profiler, "check leafs");

        // TODO: we can store the elements mushed into "index_map_of_revealed_elems" separately,
        //  like in "cross_slice_by_table" below, to avoid unmushing later
        prof_start!(maybe_profiler, "nonlinear combination");
        prof_start!(maybe_profiler, "restructure");
        let index_map_of_revealed_elems = Self::get_index_map_of_revealed_elems(
            self.parameters.num_randomizer_polynomials,
            revealed_indices,
            revealed_base_elems,
            revealed_ext_elems,
        );
        prof_stop!(maybe_profiler, "restructure");

        // =======================================
        // ==== verify non-linear combination ====
        // =======================================
        prof_start!(maybe_profiler, "main loop");
        let base_offset = self.parameters.num_randomizer_polynomials;
        let ext_offset = base_offset + num_base_polynomials;
        let final_offset = ext_offset + num_extension_polynomials;
        let omicron: XFieldElement = derive_omicron(padded_height as u64);
        let omicron_inverse = omicron.inverse();
        for (combination_check_index, revealed_combination_leaf) in combination_check_indices
            .into_iter()
            .zip_eq(revealed_combination_leafs)
        {
            prof_itr0!(maybe_profiler, "main loop");
            prof_start!(maybe_profiler, "populate");
            let current_fri_domain_value =
                self.fri.domain.domain_value(combination_check_index as u32);
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
            prof_stop!(maybe_profiler, "populate");

            prof_start!(maybe_profiler, "unmush");
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
                    (combination_check_index + unit_distance) % self.fri.domain.length;
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
            prof_stop!(maybe_profiler, "unmush");

            prof_start!(maybe_profiler, "inner loop");
            // use AIR (actually RAP) to get relevant parts of quotient codewords
            for ((table_row, next_table_row), table) in cross_slice_by_table
                .iter()
                .zip_eq(next_cross_slice_by_table.iter())
                .zip_eq(ext_table_collection.into_iter())
            {
                prof_itr0!(maybe_profiler, "inner loop");
                prof_start!(maybe_profiler, "degree bounds");
                let initial_quotient_degree_bounds = table.get_initial_quotient_degree_bounds(
                    padded_height,
                    self.parameters.num_trace_randomizers,
                );
                let consistency_quotient_degree_bounds = table
                    .get_consistency_quotient_degree_bounds(
                        padded_height,
                        self.parameters.num_trace_randomizers,
                    );
                let transition_quotient_degree_bounds = table
                    .get_transition_quotient_degree_bounds(
                        padded_height,
                        self.parameters.num_trace_randomizers,
                    );
                let terminal_quotient_degree_bounds = table.get_terminal_quotient_degree_bounds(
                    padded_height,
                    self.parameters.num_trace_randomizers,
                );
                prof_stop!(maybe_profiler, "degree bounds");

                prof_start!(maybe_profiler, "initial constraint quotient points");
                for (evaluated_bc, degree_bound) in table
                    .evaluate_initial_constraints(table_row, &extension_challenges)
                    .into_iter()
                    .zip_eq(initial_quotient_degree_bounds.iter())
                {
                    let shift = self.max_degree - degree_bound;
                    let quotient = evaluated_bc / (current_fri_domain_value - XFieldElement::one());
                    let quotient_shifted =
                        quotient * current_fri_domain_value.mod_pow_u32(shift as u32);
                    summands.push(quotient);
                    summands.push(quotient_shifted);
                }
                prof_stop!(maybe_profiler, "initial constraint quotient points");

                prof_start!(maybe_profiler, "consistency constraint quotient points");
                for (evaluated_cc, degree_bound) in table
                    .evaluate_consistency_constraints(table_row, &extension_challenges)
                    .into_iter()
                    .zip_eq(consistency_quotient_degree_bounds.iter())
                {
                    let shift = self.max_degree - degree_bound;
                    let quotient = evaluated_cc
                        / (current_fri_domain_value.mod_pow_u32(padded_height as u32)
                            - XFieldElement::one());
                    let quotient_shifted =
                        quotient * current_fri_domain_value.mod_pow_u32(shift as u32);
                    summands.push(quotient);
                    summands.push(quotient_shifted);
                }
                prof_stop!(maybe_profiler, "consistency constraint quotient points");

                prof_start!(maybe_profiler, "transition constraint quotient points");
                for (evaluated_tc, degree_bound) in table
                    .evaluate_transition_constraints(
                        table_row,
                        next_table_row,
                        &extension_challenges,
                    )
                    .into_iter()
                    .zip_eq(transition_quotient_degree_bounds.iter())
                {
                    let shift = self.max_degree - degree_bound;
                    let quotient = {
                        let numerator = current_fri_domain_value - omicron_inverse;
                        let denominator = current_fri_domain_value
                            .mod_pow_u32(padded_height as u32)
                            - XFieldElement::one();
                        evaluated_tc * numerator / denominator
                    };
                    let quotient_shifted =
                        quotient * current_fri_domain_value.mod_pow_u32(shift as u32);
                    summands.push(quotient);
                    summands.push(quotient_shifted);
                }
                prof_stop!(maybe_profiler, "transition constraint quotient points");

                prof_start!(maybe_profiler, "terminal constraint quotient points");
                for (evaluated_termc, degree_bound) in table
                    .evaluate_terminal_constraints(table_row, &extension_challenges)
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

                prof_stop!(maybe_profiler, "terminal constraint quotient points");
            }
            prof_stop!(maybe_profiler, "inner loop");

            prof_start!(maybe_profiler, "grand cross-table argument");

            let grand_cross_table_arg_degree_bound = grand_cross_table_arg.quotient_degree_bound(
                &ext_table_collection,
                self.parameters.num_trace_randomizers,
            );
            let shift = self.max_degree - grand_cross_table_arg_degree_bound;
            let grand_cross_table_arg_evaluated =
                grand_cross_table_arg.evaluate_non_linear_sum_of_differences(&cross_slice_by_table);
            let grand_cross_table_arg_quotient =
                grand_cross_table_arg_evaluated / (current_fri_domain_value - omicron_inverse);
            let grand_cross_table_arg_quotient_shifted =
                grand_cross_table_arg_quotient * current_fri_domain_value.mod_pow_u32(shift as u32);
            summands.push(grand_cross_table_arg_quotient);
            summands.push(grand_cross_table_arg_quotient_shifted);
            prof_stop!(maybe_profiler, "grand cross-table argument");

            prof_start!(maybe_profiler, "compute inner product");
            let inner_product = non_lin_combi_weights
                .par_iter()
                .zip_eq(summands.par_iter())
                .map(|(&weight, &summand)| weight * summand)
                .sum();

            if revealed_combination_leaf != inner_product {
                return Err(Box::new(StarkValidationError::CombinationLeafInequality));
            }
            prof_stop!(maybe_profiler, "compute inner product");
        }
        prof_stop!(maybe_profiler, "main loop");
        prof_stop!(maybe_profiler, "nonlinear combination");
        Ok(true)
    }

    fn get_index_map_of_revealed_elems(
        num_randomizer_polynomials: usize,
        revealed_indices: Vec<usize>,
        revealed_base_elems: Vec<Vec<BFieldElement>>,
        revealed_ext_elems: Vec<Vec<XFieldElement>>,
    ) -> HashMap<usize, Vec<XFieldElement>> {
        let extension_degree = 3;
        let mut index_map: HashMap<usize, Vec<XFieldElement>> = HashMap::new();
        for (i, &idx) in revealed_indices.iter().enumerate() {
            let mut rand_elems = vec![];
            for (coeff_0, coeff_1, coeff_2) in revealed_base_elems[i]
                .iter()
                .take(extension_degree * num_randomizer_polynomials)
                .tuples()
            {
                rand_elems.push(XFieldElement::new([*coeff_0, *coeff_1, *coeff_2]));
            }

            let base_elems = revealed_base_elems[i]
                .iter()
                .skip(extension_degree * num_randomizer_polynomials)
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
    use std::ops::Mul;

    use num_traits::{One, Zero};
    use twenty_first::shared_math::ntt::ntt;
    use twenty_first::shared_math::other::log_2_floor;

    use crate::cross_table_arguments::EvalArg;
    use crate::instruction::sample_programs;
    use crate::shared_tests::*;
    use crate::table::base_matrix::AlgebraicExecutionTrace;
    use crate::table::base_table::InheritsFromTable;
    use crate::table::table_collection::TableId::ProcessorTable;
    use crate::table::table_column::ProcessorExtTableColumn::{
        InputTableEvalArg, OutputTableEvalArg,
    };
    use crate::vm::triton_vm_tests::{
        bigger_tasm_test_programs, property_based_test_programs, small_tasm_test_programs,
        test_hash_nop_nop_lt,
    };
    use crate::vm::Program;

    use super::*;

    #[test]
    fn all_tables_pad_to_same_height_test() {
        let code = "read_io read_io push -1 mul add split push 0 eq swap1 pop "; // simulates LTE
        let input_symbols = vec![5_u64.into(), 7_u64.into()];
        let (aet, _, program) = parse_setup_simulate(code, input_symbols, vec![]);
        let base_matrices = BaseMatrices::new(aet, &program.to_bwords());
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
    }

    pub fn parse_setup_simulate(
        code: &str,
        input_symbols: Vec<BFieldElement>,
        secret_input_symbols: Vec<BFieldElement>,
    ) -> (AlgebraicExecutionTrace, Vec<BFieldElement>, Program) {
        let program = Program::from_code(code);

        assert!(program.is_ok(), "program parses correctly");
        let program = program.unwrap();

        let (aet, stdout, err) = program.simulate(input_symbols, secret_input_symbols);
        if let Some(error) = err {
            panic!("The VM encountered the following problem: {}", error);
        }
        (aet, stdout, program)
    }

    pub fn parse_simulate_pad(
        code: &str,
        stdin: Vec<BFieldElement>,
        secret_in: Vec<BFieldElement>,
    ) -> (
        BaseTableCollection,
        BaseTableCollection,
        usize,
        Vec<BFieldElement>,
    ) {
        let (aet, stdout, program) = parse_setup_simulate(code, stdin, secret_in);
        let base_matrices = BaseMatrices::new(aet, &program.to_bwords());

        let num_trace_randomizers = 2;
        let mut base_tables = BaseTableCollection::from_base_matrices(&base_matrices);

        let unpadded_base_tables = base_tables.clone();

        base_tables.pad();

        (
            base_tables,
            unpadded_base_tables,
            num_trace_randomizers,
            stdout,
        )
    }

    pub fn parse_simulate_pad_extend(
        code: &str,
        stdin: Vec<BFieldElement>,
        secret_in: Vec<BFieldElement>,
    ) -> (
        Vec<BFieldElement>,
        BaseTableCollection,
        BaseTableCollection,
        ExtTableCollection,
        AllChallenges,
        usize,
    ) {
        let (base_tables, unpadded_base_tables, num_trace_randomizers, stdout) =
            parse_simulate_pad(code, stdin, secret_in);

        let dummy_challenges = AllChallenges::placeholder();
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
            parse_simulate_pad_extend("halt", vec![], vec![]);
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
    pub fn print_all_coefficient_counts() {
        let ext_tables = ExtTableCollection::with_padded_height(2);
        for table in ext_tables
            .into_iter()
            .filter(|&table| table.name() != "EmptyExtHashTable")
        {
            for (idx, constraint) in table
                .dynamic_initial_constraints(&AllChallenges::placeholder())
                .into_iter()
                .enumerate()
            {
                println!(
                    "{} initial constraint with index {idx} has {} coefficients.",
                    table.name(),
                    constraint.coefficients.len()
                );
            }
            for (idx, constraint) in table
                .dynamic_consistency_constraints(&AllChallenges::placeholder())
                .into_iter()
                .enumerate()
            {
                println!(
                    "{} consistency constraint with index {idx} has {} coefficients.",
                    table.name(),
                    constraint.coefficients.len()
                );
            }
            for (idx, constraint) in table
                .dynamic_transition_constraints(&AllChallenges::placeholder())
                .into_iter()
                .enumerate()
            {
                println!(
                    "{} transition constraint with index {idx} has {} coefficients.",
                    table.name(),
                    constraint.coefficients.len()
                );
            }
            for (idx, constraint) in table
                .dynamic_terminal_constraints(&AllChallenges::placeholder())
                .into_iter()
                .enumerate()
            {
                println!(
                    "{} terminal constraint with index {idx} has {} coefficients.",
                    table.name(),
                    constraint.coefficients.len()
                );
            }
        }
        println!("HashTable AIR's coefficients cannot be counted because they are hardcoded.");
    }

    #[test]
    pub fn shift_codeword_test() {
        let claim = Claim {
            input: vec![],
            program: vec![],
            output: vec![],
            padded_height: 32,
        };
        let parameters = StarkParameters::default();
        let stark = Stark::new(claim, parameters);
        let fri_x_values = stark.fri.domain.domain_values();

        let mut test_codeword: Vec<XFieldElement> = vec![0.into(); stark.fri.domain.length];
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
            stark.fri.domain.omega,
            log_2_floor(stark.fri.domain.length as u128) as u32,
        );
        for shift in [0, 1, 5, 17, 63, 121, 128] {
            let shifted_codeword = Stark::shift_codeword(&fri_x_values, &test_codeword, shift);
            let interpolated_shifted_codeword = stark.fri.domain.interpolate(&shifted_codeword);
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
            parse_simulate_pad_extend(read_nop_code, input_symbols.clone(), vec![]);

        let ptie = ext_table_collection.data(ProcessorTable).last().unwrap()
            [usize::from(InputTableEvalArg)];
        let ine = EvalArg::compute_terminal(
            &input_symbols,
            EvalArg::default_initial(),
            all_challenges.input_challenges.processor_eval_indeterminate,
        );
        assert_eq!(ptie, ine, "The input evaluation arguments do not match.");

        let ptoe = ext_table_collection.data(ProcessorTable).last().unwrap()
            [usize::from(OutputTableEvalArg)];

        let oute = EvalArg::compute_terminal(
            &stdout,
            EvalArg::default_initial(),
            all_challenges
                .output_challenges
                .processor_eval_indeterminate,
        );
        assert_eq!(ptoe, oute, "The output evaluation arguments do not match.");
    }

    #[test]
    pub fn check_all_cross_table_terminals() {
        let mut code_collection = small_tasm_test_programs();
        code_collection.append(&mut bigger_tasm_test_programs());
        code_collection.append(&mut property_based_test_programs());

        for (code_idx, code_with_input) in code_collection.into_iter().enumerate() {
            let code = code_with_input.source_code;
            let input = code_with_input.input;
            let secret_input = code_with_input.secret_input.clone();
            let (output, _, _, ext_table_collection, all_challenges, _) =
                parse_simulate_pad_extend(&code, input.clone(), secret_input);

            let input_terminal = EvalArg::compute_terminal(
                &input,
                EvalArg::default_initial(),
                all_challenges
                    .processor_table_challenges
                    .standard_input_eval_indeterminate,
            );

            let output_terminal = EvalArg::compute_terminal(
                &output,
                EvalArg::default_initial(),
                all_challenges
                    .processor_table_challenges
                    .standard_output_eval_indeterminate,
            );

            let grand_cross_table_arg = GrandCrossTableArg::new(
                &[XFieldElement::one(); NUM_CROSS_TABLE_ARGS + NUM_PUBLIC_EVAL_ARGS],
                input_terminal,
                output_terminal,
            );

            for (idx, (arg, _)) in grand_cross_table_arg.into_iter().enumerate() {
                let from = arg
                    .from()
                    .iter()
                    .map(|&(from_table, from_column)| {
                        ext_table_collection.data(from_table).last().unwrap()[from_column]
                    })
                    .fold(XFieldElement::one(), XFieldElement::mul);
                let to = arg
                    .to()
                    .iter()
                    .map(|&(to_table, to_column)| {
                        ext_table_collection.data(to_table).last().unwrap()[to_column]
                    })
                    .fold(XFieldElement::one(), XFieldElement::mul);
                assert_eq!(
                    from, to,
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
        let (_, _, _, ext_tables, challenges, _) =
            parse_simulate_pad_extend("halt", vec![], vec![]);

        for table in ext_tables.into_iter() {
            let dummy_row = vec![0.into(); table.full_width()];

            // will panic if the number of variables is wrong
            table.evaluate_initial_constraints(&dummy_row, &challenges);
            table.evaluate_consistency_constraints(&dummy_row, &challenges);
            table.evaluate_transition_constraints(&dummy_row, &dummy_row, &challenges);
            table.evaluate_terminal_constraints(&dummy_row, &challenges);
        }
    }

    #[test]
    fn extend_does_not_change_base_table() {
        let (base_tables, _, num_trace_randomizers, _) =
            parse_simulate_pad(sample_programs::FIBONACCI_LT, vec![], vec![]);

        let dummy_challenges = AllChallenges::placeholder();
        let ext_tables = ExtTableCollection::extend_tables(
            &base_tables,
            &dummy_challenges,
            num_trace_randomizers,
        );

        for (base_table, extension_table) in base_tables.into_iter().zip(ext_tables.into_iter()) {
            for column in 0..base_table.base_width() {
                for row in 0..base_tables.padded_height {
                    assert_eq!(
                        base_table.data()[row][column].lift(),
                        extension_table.data()[row][column]
                    );
                }
            }
        }
    }

    #[test]
    fn print_number_of_all_constraints_per_table() {
        let (_, _, _, ext_tables, challenges, _) =
            parse_simulate_pad_extend(sample_programs::COUNTDOWN_FROM_10, vec![], vec![]);

        println!("| Table                |  Init |  Cons | Trans |  Term |   Sum |");
        println!("|:---------------------|------:|------:|------:|------:|------:|");

        let mut num_total_initial_constraints = 0;
        let mut num_total_consistency_constraints = 0;
        let mut num_total_transition_constraints = 0;
        let mut num_total_terminal_constraints = 0;
        for table in ext_tables.into_iter() {
            let num_initial_constraints = table
                .evaluate_initial_constraints(&table.data()[0], &challenges)
                .len();
            let num_consistency_constraints = table
                .evaluate_consistency_constraints(&table.data()[0], &challenges)
                .len();
            let num_transition_constraints = table
                .evaluate_transition_constraints(&table.data()[0], &table.data()[1], &challenges)
                .len();
            let num_terminal_constraints = table
                .evaluate_terminal_constraints(table.data().last().unwrap(), &challenges)
                .len();

            let num_total_constraints = num_initial_constraints
                + num_consistency_constraints
                + num_transition_constraints
                + num_terminal_constraints;
            num_total_initial_constraints += num_initial_constraints;
            num_total_consistency_constraints += num_consistency_constraints;
            num_total_transition_constraints += num_transition_constraints;
            num_total_terminal_constraints += num_terminal_constraints;
            println!(
                "| {:<20} | {:>5} | {:>5} | {:>5} | {:>5} | {:>5} |",
                table.name().split_whitespace().next().unwrap(),
                num_initial_constraints,
                num_consistency_constraints,
                num_transition_constraints,
                num_terminal_constraints,
                num_total_constraints,
            );
        }

        let num_total_constraints = num_total_initial_constraints
            + num_total_consistency_constraints
            + num_total_transition_constraints
            + num_total_terminal_constraints;
        println!(
            "| {:<20} | {:>5} | {:>5} | {:>5} | {:>5} | {:>5} |",
            "Sum",
            num_total_initial_constraints,
            num_total_consistency_constraints,
            num_total_transition_constraints,
            num_total_terminal_constraints,
            num_total_constraints
        );
    }

    #[test]
    fn number_of_quotient_degree_bound_matches_number_of_constraints_test() {
        let (_, _, _, ext_tables, challenges, num_trace_randomizers) =
            parse_simulate_pad_extend(sample_programs::FIBONACCI_LT, vec![], vec![]);
        let padded_height = ext_tables.padded_height;

        for table in ext_tables.into_iter() {
            let num_initial_constraints = table
                .evaluate_initial_constraints(&table.data()[0], &challenges)
                .len();
            let num_initial_quotient_degree_bounds = table
                .get_initial_quotient_degree_bounds(padded_height, num_trace_randomizers)
                .len();
            assert_eq!(
                num_initial_constraints,
                num_initial_quotient_degree_bounds,
                "{} has mismatching number of initial constraints and quotient degree bounds.",
                table.name()
            );

            let num_consistency_constraints = table
                .evaluate_consistency_constraints(&table.data()[0], &challenges)
                .len();
            let num_consistency_quotient_degree_bounds = table
                .get_consistency_quotient_degree_bounds(padded_height, num_trace_randomizers)
                .len();
            assert_eq!(
                num_consistency_constraints,
                num_consistency_quotient_degree_bounds,
                "{} has mismatching number of consistency constraints and quotient degree bounds.",
                table.name()
            );

            let num_transition_constraints = table
                .evaluate_transition_constraints(&table.data()[0], &table.data()[1], &challenges)
                .len();
            let num_transition_quotient_degree_bounds = table
                .get_transition_quotient_degree_bounds(padded_height, num_trace_randomizers)
                .len();
            assert_eq!(
                num_transition_constraints,
                num_transition_quotient_degree_bounds,
                "{} has mismatching number of transition constraints and quotient degree bounds.",
                table.name()
            );

            let num_terminal_constraints = table
                .evaluate_terminal_constraints(table.data().last().unwrap(), &challenges)
                .len();
            let num_terminal_quotient_degree_bounds = table
                .get_terminal_quotient_degree_bounds(padded_height, num_trace_randomizers)
                .len();
            assert_eq!(
                num_terminal_constraints,
                num_terminal_quotient_degree_bounds,
                "{} has mismatching number of terminal constraints and quotient degree bounds.",
                table.name()
            );
        }
    }

    #[test]
    fn triton_table_constraints_evaluate_to_zero_test_on_halt() {
        let zero = XFieldElement::zero();
        let source_code_and_input = test_halt();
        let (_, _, _, ext_tables, challenges, _) =
            parse_simulate_pad_extend(&source_code_and_input.source_code, vec![], vec![]);

        for table in (&ext_tables).into_iter() {
            if let Some(row) = table.data().get(0) {
                let evaluated_bcs = table.evaluate_initial_constraints(row, &challenges);
                let num_initial_constraints = evaluated_bcs.len();
                for (constraint_idx, ebc) in evaluated_bcs.into_iter().enumerate() {
                    assert_eq!(
                        zero,
                        ebc,
                        "Failed initial constraint on {}. Constraint index: {}/{}. Row index: {}/{}",
                        table.name(),
                        num_initial_constraints,
                        constraint_idx,
                        0,
                        table.data().len()
                    );
                }
            }

            for (row_idx, curr_row) in table.data().iter().enumerate() {
                let evaluated_ccs = table.evaluate_consistency_constraints(curr_row, &challenges);
                for (constraint_idx, ecc) in evaluated_ccs.into_iter().enumerate() {
                    assert_eq!(
                        zero,
                        ecc,
                        "Failed consistency constraint {}. Constraint index: {}. Row index: {}/{}",
                        table.name(),
                        constraint_idx,
                        row_idx,
                        table.data().len()
                    );
                }
            }

            for (row_idx, (curr_row, next_row)) in table.data().iter().tuple_windows().enumerate() {
                let evaluated_tcs =
                    table.evaluate_transition_constraints(curr_row, next_row, &challenges);
                for (constraint_idx, evaluated_tc) in evaluated_tcs.into_iter().enumerate() {
                    assert_eq!(
                        zero,
                        evaluated_tc,
                        "Failed transition constraint on {}. Constraint index: {}. Row index: {}/{}",
                        table.name(),
                        constraint_idx,
                        row_idx,
                        table.data().len()
                    );
                }
            }

            if let Some(row) = table.data().last() {
                let evaluated_termcs = table.evaluate_terminal_constraints(row, &challenges);
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
    fn triton_table_constraints_evaluate_to_zero_test_on_simple_program() {
        let zero = XFieldElement::zero();
        let (_, _, _, ext_tables, _, _) =
            parse_simulate_pad_extend(sample_programs::FIBONACCI_LT, vec![], vec![]);

        let challenges = AllChallenges::placeholder();
        for table in (&ext_tables).into_iter() {
            if let Some(row) = table.data().get(0) {
                let evaluated_bcs = table.evaluate_initial_constraints(row, &challenges);
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
                let evaluated_ccs = table.evaluate_consistency_constraints(curr_row, &challenges);
                for (constraint_idx, ecc) in evaluated_ccs.into_iter().enumerate() {
                    assert_eq!(
                        zero,
                        ecc,
                        "Failed consistency constraint {}. Constraint index: {}. Row index: {}/{}",
                        table.name(),
                        constraint_idx,
                        row_idx,
                        table.data().len()
                    );
                }
            }

            for (row_idx, (current_row, next_row)) in
                table.data().iter().tuple_windows().enumerate()
            {
                let evaluated_tcs =
                    table.evaluate_transition_constraints(&current_row, next_row, &challenges);
                for (constraint_idx, evaluated_tc) in evaluated_tcs.into_iter().enumerate() {
                    assert_eq!(
                        zero,
                        evaluated_tc,
                        "Failed transition constraint on {}. Constraint index: {}. Row index: {}/{}",
                        table.name(),
                        constraint_idx,
                        row_idx,
                        table.data().len()
                    );
                }
            }

            if let Some(row) = table.data().last() {
                let evaluated_termcs = table.evaluate_terminal_constraints(row, &challenges);
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
    fn triton_prove_verify_simple_program_test() {
        let code_with_input = test_hash_nop_nop_lt();
        let (stark, proof) = parse_simulate_prove(
            &code_with_input.source_code,
            code_with_input.input.clone(),
            code_with_input.secret_input.clone(),
            vec![],
            &mut None,
        );

        println!("between prove and verify");

        let result = stark.verify(proof, &mut None);
        if let Err(e) = result {
            panic!("The Verifier is unhappy! {}", e);
        }
        assert!(result.unwrap());
    }

    #[test]
    fn triton_prove_verify_halt_test() {
        let code_with_input = test_halt();
        let (stark, proof) = parse_simulate_prove(
            &code_with_input.source_code,
            code_with_input.input.clone(),
            code_with_input.secret_input.clone(),
            vec![],
            &mut None,
        );

        println!("between prove and verify");

        let result = stark.verify(proof, &mut None);
        if let Err(e) = result {
            panic!("The Verifier is unhappy! {}", e);
        }
        assert!(result.unwrap());
    }

    #[test]
    #[ignore = "too slow"]
    fn prove_verify_fibonacci_100_test() {
        let source_code = sample_programs::FIBONACCI_VIT;
        let program = Program::from_code(source_code).unwrap();

        let stdin = vec![100_u64.into()];
        let secret_in = vec![];

        let (_, stdout, _) = program.run(stdin.clone(), secret_in.clone());
        let (stark, proof) = parse_simulate_prove(source_code, stdin, secret_in, stdout, &mut None);

        println!("between prove and verify");

        let result = stark.verify(proof, &mut None);
        if let Err(e) = result {
            panic!("The Verifier is unhappy! {}", e);
        }
        assert!(result.unwrap());
    }
}
