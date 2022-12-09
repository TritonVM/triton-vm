use std::collections::HashMap;
use std::error::Error;
use std::fmt;
use std::ops::Add;

use anyhow::anyhow;
use anyhow::bail;
use anyhow::Result;
use itertools::Itertools;
use ndarray::par_azip;
use ndarray::s;
use ndarray::Array1;
use ndarray::ArrayBase;
use ndarray::ArrayView2;
use num_traits::One;
use num_traits::Zero;
use rayon::prelude::*;
use triton_profiler::prof_itr0;
use triton_profiler::prof_start;
use triton_profiler::prof_stop;
use triton_profiler::triton_profiler::TritonProfiler;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::mpolynomial::Degree;
use twenty_first::shared_math::other::is_power_of_two;
use twenty_first::shared_math::other::roundup_npo2;
use twenty_first::shared_math::rescue_prime_digest::Digest;
use twenty_first::shared_math::rescue_prime_regular::RescuePrimeRegular;
use twenty_first::shared_math::traits::FiniteField;
use twenty_first::shared_math::traits::Inverse;
use twenty_first::shared_math::traits::ModPowU32;
use twenty_first::shared_math::traits::PrimitiveRootOfUnity;
use twenty_first::shared_math::x_field_element::XFieldElement;
use twenty_first::util_types::algebraic_hasher::AlgebraicHasher;
use twenty_first::util_types::merkle_tree::CpuParallel;
use twenty_first::util_types::merkle_tree::MerkleTree;
use twenty_first::util_types::merkle_tree_maker::MerkleTreeMaker;

use crate::arithmetic_domain::ArithmeticDomain;
use crate::cross_table_arguments::CrossTableArg;
use crate::cross_table_arguments::EvalArg;
use crate::cross_table_arguments::GrandCrossTableArg;
use crate::cross_table_arguments::NUM_CROSS_TABLE_ARGS;
use crate::cross_table_arguments::NUM_PUBLIC_EVAL_ARGS;
use crate::fri::Fri;
use crate::fri::FriValidationError;
use crate::proof::Claim;
use crate::proof::Proof;
use crate::proof_item::ProofItem;
use crate::proof_stream::ProofStream;
use crate::table::challenges::AllChallenges;
use crate::table::table_collection::*;
use crate::vm::AlgebraicExecutionTrace;

pub type StarkHasher = RescuePrimeRegular;
pub type Maker = CpuParallel;
pub type StarkProofStream = ProofStream<ProofItem, StarkHasher>;

pub struct StarkParameters {
    pub security_level: usize,
    pub fri_expansion_factor: usize,
    pub num_trace_randomizers: usize,
    pub num_randomizer_polynomials: usize,
    pub num_colinearity_checks: usize,
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
    pub parameters: StarkParameters,
    pub claim: Claim,
    pub max_degree: Degree,
    pub fri: Fri<StarkHasher>,
}

impl Stark {
    pub fn new(claim: Claim, parameters: StarkParameters) -> Self {
        let max_degree_with_origin =
            max_degree_with_origin(claim.padded_height, parameters.num_trace_randomizers);
        let max_degree = (roundup_npo2(max_degree_with_origin.degree as u64) - 1) as Degree;
        let fri_domain_length = parameters.fri_expansion_factor * (max_degree as usize + 1);
        let fri_domain_generator =
            BFieldElement::primitive_root_of_unity(fri_domain_length.try_into().unwrap()).unwrap();
        let coset_offset = BFieldElement::generator();
        let fri = Fri::new(
            coset_offset,
            fri_domain_generator,
            fri_domain_length,
            parameters.fri_expansion_factor,
            parameters.num_colinearity_checks,
        );
        Self {
            parameters,
            claim,
            max_degree,
            fri,
        }
    }

    pub fn prove(
        &self,
        aet: AlgebraicExecutionTrace,
        maybe_profiler: &mut Option<TritonProfiler>,
    ) -> Proof {
        prof_start!(maybe_profiler, "base tables");
        prof_start!(maybe_profiler, "create");
        let mut master_base_table = MasterBaseTable::new(
            aet,
            &self.claim.program,
            self.parameters.num_trace_randomizers,
            self.fri.domain,
        );
        prof_stop!(maybe_profiler, "create");

        prof_start!(maybe_profiler, "pad");
        master_base_table.pad();
        prof_stop!(maybe_profiler, "pad");

        prof_start!(maybe_profiler, "LDE");
        master_base_table.randomize_trace();
        let fri_domain_base_master_table = master_base_table.to_fri_domain_table();
        prof_stop!(maybe_profiler, "LDE");

        prof_start!(maybe_profiler, "Merkle tree");
        let base_merkle_tree = fri_domain_base_master_table.merkle_tree();
        let base_merkle_tree_root = base_merkle_tree.get_root();
        prof_stop!(maybe_profiler, "Merkle tree");

        prof_start!(maybe_profiler, "Fiat-Shamir");
        let padded_height = BFieldElement::new(master_base_table.padded_height as u64);
        let mut proof_stream = StarkProofStream::new();
        proof_stream.enqueue(&ProofItem::PaddedHeight(padded_height));
        proof_stream.enqueue(&ProofItem::MerkleRoot(base_merkle_tree_root));
        let extension_weights = Self::sample_weights(
            proof_stream.prover_fiat_shamir(),
            AllChallenges::TOTAL_CHALLENGES,
        );
        let extension_challenges = AllChallenges::create_challenges(extension_weights);
        prof_stop!(maybe_profiler, "Fiat-Shamir");

        prof_start!(maybe_profiler, "extend");
        let mut master_ext_table = master_base_table.extend(
            &extension_challenges,
            self.parameters.num_randomizer_polynomials,
        );
        prof_stop!(maybe_profiler, "extend");

        prof_start!(maybe_profiler, "ext tables");
        prof_start!(maybe_profiler, "LDE");
        master_ext_table.randomize_trace();
        let fri_domain_ext_master_table = master_ext_table.to_fri_domain_table();
        prof_stop!(maybe_profiler, "LDE");

        prof_start!(maybe_profiler, "Merkle tree");
        let ext_merkle_tree = fri_domain_ext_master_table.merkle_tree();
        let ext_merkle_tree_root = ext_merkle_tree.get_root();
        proof_stream.enqueue(&ProofItem::MerkleRoot(ext_merkle_tree_root));
        prof_stop!(maybe_profiler, "Merkle tree");
        prof_stop!(maybe_profiler, "ext tables");

        prof_start!(maybe_profiler, "quotient degree bounds");
        let mut quotient_degree_bounds = all_quotient_degree_bounds(
            master_base_table.padded_height,
            self.parameters.num_trace_randomizers,
        );
        prof_stop!(maybe_profiler, "quotient degree bounds");

        prof_start!(maybe_profiler, "quotient-domain codewords");
        let quotient_domain = self.quotient_domain();
        let unit_distance = self.fri.domain.length / quotient_domain.length;
        let base_quotient_domain_codewords = fri_domain_base_master_table
            .master_base_matrix
            .slice(s![..; unit_distance, ..]);
        let extension_quotient_domain_codewords = fri_domain_ext_master_table
            .master_ext_matrix
            .slice(s![..; unit_distance, ..]);
        prof_stop!(maybe_profiler, "quotient-domain codewords");

        prof_start!(maybe_profiler, "quotient codewords");
        let num_grand_cross_table_args = 1;
        let num_quotients = num_all_table_quotients() + num_grand_cross_table_args;
        let mut quotient_codewords = all_quotients(
            base_quotient_domain_codewords,
            extension_quotient_domain_codewords,
            master_base_table.padded_height,
            quotient_domain,
            num_quotients,
            &extension_challenges,
            maybe_profiler,
        );
        prof_stop!(maybe_profiler, "quotient codewords");

        prof_start!(maybe_profiler, "grand cross table");
        let num_non_lin_combi_weights = 2 * (NUM_BASE_COLUMNS + NUM_EXT_COLUMNS + num_quotients);
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
                extension_quotient_domain_codewords,
                quotient_domain,
                derive_trace_domain_generator(master_base_table.padded_height as u64),
            );
        // Add the grand cross-table argument's quotient to the quotient codewords. The memory for
        // this was allocated in the call to `all_quotients`.
        grand_cross_table_arg_quotient_codeword
            .move_into(&mut quotient_codewords.slice_mut(s![.., quotient_codewords.ncols() - 1]));

        let grand_cross_table_arg_quotient_degree_bound = grand_cross_table_arg
            .quotient_degree_bound(
                master_base_table.padded_height,
                self.parameters.num_trace_randomizers,
            );
        quotient_degree_bounds.push(grand_cross_table_arg_quotient_degree_bound);
        prof_stop!(maybe_profiler, "grand cross table");

        prof_start!(maybe_profiler, "nonlinear combination");
        prof_start!(maybe_profiler, "create combination codeword");
        let combination_codeword = self.create_combination_codeword(
            quotient_domain,
            base_quotient_domain_codewords,
            extension_quotient_domain_codewords,
            quotient_codewords.view(),
            non_lin_combi_weights,
            quotient_degree_bounds,
            master_base_table.padded_height,
            master_base_table.num_trace_randomizers,
        );
        prof_stop!(maybe_profiler, "create combination codeword");

        prof_start!(maybe_profiler, "LDE 3");
        let fri_combination_codeword_without_randomizer = Array1::from(
            quotient_domain.low_degree_extension(&combination_codeword, self.fri.domain),
        );
        prof_stop!(maybe_profiler, "LDE 3");

        let fri_combination_codeword = fri_domain_ext_master_table
            .randomizer_polynomials()
            .into_iter()
            .fold(fri_combination_codeword_without_randomizer, ArrayBase::add)
            .to_vec();
        prof_stop!(maybe_profiler, "nonlinear combination");

        prof_start!(maybe_profiler, "Merkle tree 3");
        let combination_codeword_digests = fri_combination_codeword
            .par_iter()
            .map(StarkHasher::hash)
            .collect::<Vec<_>>();
        let combination_tree: MerkleTree<StarkHasher, _> =
            Maker::from_digests(&combination_codeword_digests);
        let combination_root = combination_tree.get_root();
        proof_stream.enqueue(&ProofItem::MerkleRoot(combination_root));
        prof_stop!(maybe_profiler, "Merkle tree 3");

        // Get indices of slices that go across codewords to prove nonlinear combination
        prof_start!(maybe_profiler, "Fiat-Shamir 3");
        let indices_seed = proof_stream.prover_fiat_shamir();
        let cross_codeword_slice_indices = StarkHasher::sample_indices(
            self.parameters.security_level,
            &indices_seed,
            self.fri.domain.length,
        );
        prof_stop!(maybe_profiler, "Fiat-Shamir 3");

        prof_start!(maybe_profiler, "FRI");
        match self.fri.prove(&fri_combination_codeword, &mut proof_stream) {
            Ok((_, fri_first_round_merkle_root)) => assert_eq!(
                combination_root, fri_first_round_merkle_root,
                "Combination root from STARK and from FRI must agree."
            ),
            Err(e) => panic!("The FRI prover failed because of: {}", e),
        }
        prof_stop!(maybe_profiler, "FRI");

        prof_start!(maybe_profiler, "open trace leafs");
        // the relation between the FRI domain and the trace domain
        let unit_distance = self.fri.domain.length / master_base_table.padded_height;
        // Open leafs of zipped codewords at indicated positions
        let revealed_indices =
            self.get_revealed_indices(unit_distance, &cross_codeword_slice_indices);

        let revealed_base_elems = Self::get_revealed_elements(
            fri_domain_base_master_table.master_base_matrix.view(),
            &revealed_indices,
        );
        let auth_paths_base = base_merkle_tree.get_authentication_structure(&revealed_indices);
        proof_stream.enqueue(&ProofItem::TransposedBaseElementVectors(
            revealed_base_elems,
        ));
        proof_stream.enqueue(&ProofItem::CompressedAuthenticationPaths(auth_paths_base));

        let revealed_ext_elems = Self::get_revealed_elements(
            fri_domain_ext_master_table.master_ext_matrix.view(),
            &revealed_indices,
        );
        let auth_paths_ext = ext_merkle_tree.get_authentication_structure(&revealed_indices);
        proof_stream.enqueue(&ProofItem::TransposedExtensionElementVectors(
            revealed_ext_elems,
        ));
        proof_stream.enqueue(&ProofItem::CompressedAuthenticationPaths(auth_paths_ext));

        // open combination codeword at the same positions
        // Notice that we need to loop over `indices` here, not `revealed_indices`
        // as the latter includes adjacent table rows relative to the values in `indices`
        let revealed_combination_elements: Vec<XFieldElement> = cross_codeword_slice_indices
            .iter()
            .map(|i| fri_combination_codeword[*i])
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

        if std::env::var("DEBUG").is_ok() {
            println!(
                "Created proof containing {} B-field elements",
                proof_stream.transcript_length()
            );
        }

        proof_stream.to_proof()
    }

    fn quotient_domain(&self) -> ArithmeticDomain {
        let offset = self.fri.domain.offset;
        let length = roundup_npo2(self.max_degree as u64);
        let generator = BFieldElement::primitive_root_of_unity(length).unwrap();
        ArithmeticDomain::new(offset, generator, length as usize)
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

    // todo: can we get rid of the cloning? I.e., the call of to_vec(). Irrelevant number of times?
    fn get_revealed_elements<FF: FiniteField>(
        master_matrix: ArrayView2<FF>,
        revealed_indices: &[usize],
    ) -> Vec<Vec<FF>> {
        revealed_indices
            .iter()
            .map(|&idx| master_matrix.slice(s![idx, ..]).to_vec())
            .collect_vec()
    }

    // TODO try to reduce the number of arguments
    #[allow(clippy::too_many_arguments)]
    fn create_combination_codeword(
        &self,
        quotient_domain: ArithmeticDomain,
        base_codewords: ArrayView2<BFieldElement>,
        extension_codewords: ArrayView2<XFieldElement>,
        quotient_codewords: ArrayView2<XFieldElement>,
        weights: &[XFieldElement],
        quotient_degree_bounds: Vec<Degree>,
        padded_height: usize,
        num_trace_randomizers: usize,
    ) -> Vec<XFieldElement> {
        let (base_weights, remaining_weights) = weights.split_at(2 * NUM_BASE_COLUMNS);
        let (ext_weights, quot_weights) = remaining_weights.split_at(2 * NUM_EXT_COLUMNS);

        assert_eq!(base_weights.len(), 2 * base_codewords.ncols());
        assert_eq!(ext_weights.len(), 2 * extension_codewords.ncols());
        assert_eq!(quot_weights.len(), 2 * quotient_codewords.ncols());

        // todo Propagate this domain-specific knowledge to the rest of the code base:
        //  `base_degree_bounds` and `ext_degree_bounds` is always just `interpolant_degree`
        let interpolant_degree = interpolant_degree(padded_height, num_trace_randomizers);
        let base_and_ext_col_shift = self.max_degree - interpolant_degree;
        let quotient_domain_values = quotient_domain.domain_values();
        let shifted_domain_values: Array1<BFieldElement> = quotient_domain_values
            .clone()
            .into_par_iter()
            .map(|domain_value| domain_value.mod_pow_u32(base_and_ext_col_shift as u32))
            .collect::<Vec<_>>()
            .into();

        let mut combination_codeword = vec![XFieldElement::zero(); quotient_domain.length];

        for (idx, (codeword, weights)) in base_codewords
            .columns()
            .into_iter()
            .zip_eq(base_weights.chunks_exact(2))
            .enumerate()
        {
            par_azip!((acc in &mut combination_codeword,
                &bfe in codeword)
                    *acc += weights[0] * bfe);
            par_azip!((acc in &mut combination_codeword,
                &bfe in codeword,
                &shift in shifted_domain_values.view())
                    *acc += weights[1] * bfe * shift);
            self.debug_check_degree(idx, &combination_codeword);
        }
        if std::env::var("DEBUG").is_ok() {
            println!(" --- next up: extension codewords");
        }
        for (idx, (codeword, weights)) in extension_codewords
            .columns()
            .into_iter()
            .zip_eq(ext_weights.chunks_exact(2))
            .enumerate()
        {
            par_azip!((acc in &mut combination_codeword,
                &xfe in codeword)
                    *acc += weights[0] * xfe);
            par_azip!((acc in &mut combination_codeword,
                &xfe in codeword,
                &shift in shifted_domain_values.view())
                    *acc += weights[1] * xfe * shift);
            self.debug_check_degree(idx, &combination_codeword);
        }
        if std::env::var("DEBUG").is_ok() {
            println!(" --- next up: quotient codewords");
        }
        for (idx, ((codeword, weights), degree_bound)) in quotient_codewords
            .columns()
            .into_iter()
            .zip_eq(quot_weights.chunks_exact(2))
            .zip_eq(quotient_degree_bounds)
            .enumerate()
        {
            par_azip!((acc in &mut combination_codeword,
                &xfe in codeword)
                    *acc += weights[0] * xfe);
            let shift = self.max_degree - degree_bound;
            let shifted_domain_values: Array1<BFieldElement> = quotient_domain_values
                .clone()
                .into_par_iter()
                .map(|domain_value| domain_value.mod_pow_u32(shift as u32))
                .collect::<Vec<_>>()
                .into();
            par_azip!((acc in &mut combination_codeword,
                &xfe in codeword,
                &shift in shifted_domain_values.view())
                    *acc += weights[1] * xfe * shift);
            self.debug_check_degree(idx, &combination_codeword);
        }

        combination_codeword
    }

    fn debug_check_degree(&self, index: usize, combination_codeword: &[XFieldElement]) {
        if std::env::var("DEBUG").is_err() {
            return;
        }
        let max_degree = self.max_degree;
        let degree = self.fri.domain.interpolate(combination_codeword).degree();
        let maybe_excl_mark = if degree > max_degree as isize {
            "!!!"
        } else if degree != -1 && degree != max_degree as isize {
            "!"
        } else {
            ""
        };
        println!(
            "{maybe_excl_mark:^3} combination codeword has degree {degree} after absorbing \
            shifted codeword with index {index:>2}. Must be of maximal degree {max_degree}."
        );
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
    ) -> Result<bool> {
        prof_start!(maybe_profiler, "deserialize");
        let mut proof_stream = StarkProofStream::from_proof(&proof)?;
        prof_stop!(maybe_profiler, "deserialize");

        prof_start!(maybe_profiler, "Fiat-Shamir 1");
        let padded_height = proof_stream.dequeue()?.as_padded_heights()?.value() as usize;
        let base_merkle_tree_root = proof_stream.dequeue()?.as_merkle_root()?;

        let extension_challenge_seed = proof_stream.verifier_fiat_shamir();

        let extension_challenge_weights =
            Self::sample_weights(extension_challenge_seed, AllChallenges::TOTAL_CHALLENGES);
        let challenges = AllChallenges::create_challenges(extension_challenge_weights);
        if self.claim.padded_height != padded_height && self.claim.padded_height != 0 {
            return Err(anyhow!(StarkValidationError::PaddedHeightInequality));
        }
        prof_stop!(maybe_profiler, "Fiat-Shamir 1");

        prof_start!(maybe_profiler, "dequeue");

        let extension_tree_merkle_root = proof_stream.dequeue()?.as_merkle_root()?;
        prof_stop!(maybe_profiler, "dequeue");

        // Get weights for nonlinear combination. Concretely, sample 2 weights for each base, and
        // extension polynomial and each quotients, because transition constraints check 2 rows.
        prof_start!(maybe_profiler, "Fiat-Shamir 2");
        let num_grand_cross_table_args = 1;
        let num_non_lin_combi_weights = 2 * NUM_BASE_COLUMNS
            + 2 * NUM_EXT_COLUMNS
            + 2 * num_all_table_quotients()
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
            challenges
                .processor_table_challenges
                .standard_input_eval_indeterminate,
        );
        let output_terminal = EvalArg::compute_terminal(
            &self.claim.output,
            EvalArg::default_initial(),
            challenges
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
        // the relation between the FRI domain and the trace domain
        let unit_distance = self.fri.domain.length / padded_height;
        // Open leafs of zipped codewords at indicated positions
        let revealed_indices = self.get_revealed_indices(unit_distance, &combination_check_indices);
        prof_stop!(maybe_profiler, "get indices");

        // todo rename the operation of the proof stream
        prof_start!(maybe_profiler, "dequeue base elements");
        let base_table_rows = proof_stream
            .dequeue()?
            .as_transposed_base_element_vectors()?;
        let base_auth_paths = proof_stream
            .dequeue()?
            .as_compressed_authentication_paths()?;
        let leaf_digests_base: Vec<_> = base_table_rows
            .par_iter()
            .map(|revealed_base_elem| StarkHasher::hash_slice(revealed_base_elem))
            .collect();
        prof_stop!(maybe_profiler, "dequeue base elements");

        prof_start!(maybe_profiler, "Merkle verify (base tree)");
        if !MerkleTree::<StarkHasher, Maker>::verify_authentication_structure_from_leaves(
            base_merkle_tree_root,
            &revealed_indices,
            &leaf_digests_base,
            &base_auth_paths,
        ) {
            bail!("Failed to verify authentication path for base codeword");
        }
        prof_stop!(maybe_profiler, "Merkle verify (base tree)");

        // todo rename the operation of the proof stream
        prof_start!(maybe_profiler, "dequeue extension elements");
        let ext_table_rows = proof_stream
            .dequeue()?
            .as_transposed_extension_element_vectors()?;
        let auth_paths_ext = proof_stream
            .dequeue()?
            .as_compressed_authentication_paths()?;
        let leaf_digests_ext: Vec<_> = ext_table_rows
            .par_iter()
            .map(|xvalues| {
                let bvalues: Vec<BFieldElement> = xvalues
                    .iter()
                    .flat_map(|xfe| xfe.coefficients.to_vec())
                    .collect();
                StarkHasher::hash_slice(&bvalues)
            })
            .collect();
        prof_stop!(maybe_profiler, "dequeue extension elements");

        prof_start!(maybe_profiler, "Merkle verify (extension tree)");
        if !MerkleTree::<StarkHasher, Maker>::verify_authentication_structure_from_leaves(
            extension_tree_merkle_root,
            &revealed_indices,
            &leaf_digests_ext,
            &auth_paths_ext,
        ) {
            bail!("Failed to verify authentication path for extension codeword");
        }
        prof_stop!(maybe_profiler, "Merkle verify (extension tree)");

        // Verify Merkle authentication path for combination elements
        prof_start!(maybe_profiler, "Merkle verify (combination tree)");
        let revealed_combination_leafs =
            proof_stream.dequeue()?.as_revealed_combination_elements()?;
        let revealed_combination_digests: Vec<_> = revealed_combination_leafs
            .par_iter()
            .map(StarkHasher::hash)
            .collect();
        let revealed_combination_auth_paths = proof_stream
            .dequeue()?
            .as_compressed_authentication_paths()?;
        if !MerkleTree::<StarkHasher, Maker>::verify_authentication_structure_from_leaves(
            combination_root,
            &combination_check_indices,
            &revealed_combination_digests,
            &revealed_combination_auth_paths,
        ) {
            bail!("Failed to verify authentication path for combination codeword");
        }
        prof_stop!(maybe_profiler, "Merkle verify (combination tree)");
        prof_stop!(maybe_profiler, "check leafs");

        prof_start!(maybe_profiler, "nonlinear combination");
        prof_start!(maybe_profiler, "index");
        let (indexed_base_table_rows, indexed_ext_table_rows, indexed_randomizer_rows) =
            Self::index_revealed_rows(revealed_indices, base_table_rows, ext_table_rows);
        prof_stop!(maybe_profiler, "index");

        // =======================================
        // ==== verify non-linear combination ====
        // =======================================
        let base_degree_bounds =
            base_degree_bounds(padded_height, self.parameters.num_trace_randomizers);
        let ext_degree_bounds =
            extension_degree_bounds(padded_height, self.parameters.num_trace_randomizers);

        prof_start!(maybe_profiler, "main loop");
        let trace_domain_generator = derive_trace_domain_generator(padded_height as u64);
        let trace_domain_generator_inverse = trace_domain_generator.inverse();
        for (current_row_idx, revealed_combination_leaf) in combination_check_indices
            .into_iter()
            .zip_eq(revealed_combination_leafs)
        {
            prof_itr0!(maybe_profiler, "main loop");
            let current_fri_domain_value = self.fri.domain.domain_value(current_row_idx as u32);
            let base_row = indexed_base_table_rows[&current_row_idx].view();
            let ext_row = indexed_ext_table_rows[&current_row_idx].view();

            let next_row_idx = (current_row_idx + unit_distance) % self.fri.domain.length;
            let next_base_row = indexed_base_table_rows[&next_row_idx].view();
            let next_ext_row = indexed_ext_table_rows[&next_row_idx].view();

            // todo pre-compute for all revealed fri domain values, use batch invert?
            let one = BFieldElement::one();
            let initial_zerofier_inverse = (current_fri_domain_value - one).lift();
            let consistency_zerofier_inverse =
                (current_fri_domain_value.mod_pow_u32(padded_height as u32) - one)
                    .inverse()
                    .lift();
            let except_last_row = current_fri_domain_value - trace_domain_generator_inverse;
            let transition_zerofier_inverse = except_last_row
                * (current_fri_domain_value.mod_pow_u32(padded_height as u32) - one)
                    .inverse()
                    .lift();
            let terminal_zerofier_inverse = except_last_row.lift();

            let mut summands = vec![];

            // populate summands with a cross-slice of (base,ext) codewords and their shifts
            prof_start!(maybe_profiler, "populate");
            for (&base_row_element, degree_bound) in
                base_row.iter().zip_eq(base_degree_bounds.iter())
            {
                // todo for base & ext tables, this shift is always max_degree - interpolant_degree.
                //  Pre-compute the corresponding shifted current_fri_domain_value.
                let shift = self.max_degree - degree_bound;
                let base_row_element_shifted =
                    base_row_element * current_fri_domain_value.mod_pow_u32(shift as u32);
                // todo non-lin combine as far as possible staying in the BField, lift at very end
                summands.push(base_row_element.lift());
                summands.push(base_row_element_shifted.lift());
            }

            for (&ext_row_element, degree_bound) in ext_row.iter().zip_eq(ext_degree_bounds.iter())
            {
                let shift = self.max_degree - degree_bound;
                let base_row_element_shifted =
                    ext_row_element * current_fri_domain_value.mod_pow_u32(shift as u32);
                summands.push(ext_row_element);
                summands.push(base_row_element_shifted);
            }
            prof_stop!(maybe_profiler, "populate");

            prof_start!(maybe_profiler, "degree bounds");
            let initial_quotient_degree_bounds = all_initial_quotient_degree_bounds(
                padded_height,
                self.parameters.num_trace_randomizers,
            );
            let consistency_quotient_degree_bounds = all_consistency_quotient_degree_bounds(
                padded_height,
                self.parameters.num_trace_randomizers,
            );
            let transition_quotient_degree_bounds = all_transition_quotient_degree_bounds(
                padded_height,
                self.parameters.num_trace_randomizers,
            );
            let terminal_quotient_degree_bounds = all_terminal_quotient_degree_bounds(
                padded_height,
                self.parameters.num_trace_randomizers,
            );
            prof_stop!(maybe_profiler, "degree bounds");

            prof_start!(maybe_profiler, "evaluate AIR");
            let evaluated_initial_constraints =
                evaluate_all_initial_constraints(base_row, ext_row, &challenges);
            let evaluated_consistency_constraints =
                evaluate_all_consistency_constraints(base_row, ext_row, &challenges);
            let evaluated_transition_constraints = evaluate_all_transition_constraints(
                base_row,
                ext_row,
                next_base_row,
                next_ext_row,
                &challenges,
            );
            let evaluated_terminal_constraints =
                evaluate_all_terminal_constraints(base_row, ext_row, &challenges);
            prof_stop!(maybe_profiler, "evaluate AIR");

            prof_start!(maybe_profiler, "shift");
            for (degree_bound_category, evaluated_constraints_category, zerofier_inverse) in [
                (
                    initial_quotient_degree_bounds,
                    evaluated_initial_constraints,
                    initial_zerofier_inverse,
                ),
                (
                    consistency_quotient_degree_bounds,
                    evaluated_consistency_constraints,
                    consistency_zerofier_inverse,
                ),
                (
                    transition_quotient_degree_bounds,
                    evaluated_transition_constraints,
                    transition_zerofier_inverse,
                ),
                (
                    terminal_quotient_degree_bounds,
                    evaluated_terminal_constraints,
                    terminal_zerofier_inverse,
                ),
            ] {
                for (degree_bound, evaluated_constraint) in degree_bound_category
                    .into_iter()
                    .zip_eq(evaluated_constraints_category.into_iter())
                {
                    let shift = self.max_degree - degree_bound;
                    let quotient = evaluated_constraint * zerofier_inverse;
                    let quotient_shifted =
                        quotient * current_fri_domain_value.mod_pow_u32(shift as u32);
                    summands.push(quotient);
                    summands.push(quotient_shifted);
                }
            }
            prof_stop!(maybe_profiler, "shift");

            prof_start!(maybe_profiler, "grand cross-table argument");

            let grand_cross_table_arg_degree_bound = grand_cross_table_arg
                .quotient_degree_bound(padded_height, self.parameters.num_trace_randomizers);
            let shift = self.max_degree - grand_cross_table_arg_degree_bound;
            let grand_cross_table_arg_evaluated =
                grand_cross_table_arg.evaluate_non_linear_sum_of_differences(ext_row);
            let grand_cross_table_arg_quotient = grand_cross_table_arg_evaluated
                / (current_fri_domain_value - trace_domain_generator_inverse).lift();
            let grand_cross_table_arg_quotient_shifted =
                grand_cross_table_arg_quotient * current_fri_domain_value.mod_pow_u32(shift as u32);
            summands.push(grand_cross_table_arg_quotient);
            summands.push(grand_cross_table_arg_quotient_shifted);
            prof_stop!(maybe_profiler, "grand cross-table argument");

            prof_start!(maybe_profiler, "compute inner product");
            let inner_product: XFieldElement = non_lin_combi_weights
                .par_iter()
                .zip_eq(summands.par_iter())
                .map(|(&weight, &summand)| weight * summand)
                .sum();
            let randomizer_codewords_contribution = indexed_randomizer_rows[&current_row_idx].sum();
            if revealed_combination_leaf != inner_product + randomizer_codewords_contribution {
                return Err(anyhow!(StarkValidationError::CombinationLeafInequality));
            }
            prof_stop!(maybe_profiler, "compute inner product");
        }
        prof_stop!(maybe_profiler, "main loop");
        prof_stop!(maybe_profiler, "nonlinear combination");
        Ok(true)
    }

    // todo these hash-maps allow doing
    //      indexed_revealed_base_rows[i]
    //  instead of
    //      revealed_base_rows[revealed_indices[i]]
    //  but is that really worth all this compute & malloc?
    //  Also, the proof stream should probably just return Array1<FF> or even Array2<FF> instead.
    #[allow(clippy::type_complexity)]
    fn index_revealed_rows(
        revealed_indices: Vec<usize>,
        revealed_base_rows: Vec<Vec<BFieldElement>>,
        revealed_ext_rows: Vec<Vec<XFieldElement>>,
    ) -> (
        HashMap<usize, Array1<BFieldElement>>,
        HashMap<usize, Array1<XFieldElement>>,
        HashMap<usize, Array1<XFieldElement>>,
    ) {
        assert_eq!(revealed_indices.len(), revealed_base_rows.len());
        assert_eq!(revealed_indices.len(), revealed_ext_rows.len());

        let mut indexed_revealed_base_rows: HashMap<usize, Array1<BFieldElement>> = HashMap::new();
        let mut indexed_revealed_ext_rows: HashMap<usize, Array1<XFieldElement>> = HashMap::new();
        let mut indexed_revealed_rand_rows: HashMap<usize, Array1<XFieldElement>> = HashMap::new();

        for (i, &idx) in revealed_indices.iter().enumerate() {
            let base_row = Array1::from(revealed_base_rows[i].to_vec());
            let ext_row = Array1::from(revealed_ext_rows[i][..NUM_EXT_COLUMNS].to_vec());
            let rand_row = Array1::from(revealed_ext_rows[i][NUM_EXT_COLUMNS..].to_vec());

            indexed_revealed_base_rows.insert(idx, base_row);
            indexed_revealed_ext_rows.insert(idx, ext_row);
            indexed_revealed_rand_rows.insert(idx, rand_row);
        }

        (
            indexed_revealed_base_rows,
            indexed_revealed_ext_rows,
            indexed_revealed_rand_rows,
        )
    }
}

#[cfg(test)]
pub(crate) mod triton_stark_tests {
    use std::ops::Mul;

    use itertools::izip;
    use ndarray::Array1;
    use num_traits::One;
    use num_traits::Zero;
    use twenty_first::shared_math::other::log_2_floor;

    use crate::cross_table_arguments::EvalArg;
    use crate::instruction::sample_programs;
    use crate::shared_tests::*;
    use crate::table::extension_table::Evaluable;
    use crate::table::extension_table::Quotientable;
    use crate::table::hash_table::ExtHashTable;
    use crate::table::instruction_table::ExtInstructionTable;
    use crate::table::jump_stack_table::ExtJumpStackTable;
    use crate::table::op_stack_table::ExtOpStackTable;
    use crate::table::processor_table::ExtProcessorTable;
    use crate::table::program_table::ExtProgramTable;
    use crate::table::ram_table::ExtRamTable;
    use crate::table::table_collection::all_degrees_with_origin;
    use crate::table::table_collection::MasterExtTable;
    use crate::table::table_collection::TableId::ProcessorTable;
    use crate::table::table_column::ExtTableColumn;
    use crate::table::table_column::ProcessorExtTableColumn::InputTableEvalArg;
    use crate::table::table_column::ProcessorExtTableColumn::OutputTableEvalArg;
    use crate::vm::triton_vm_tests::bigger_tasm_test_programs;
    use crate::vm::triton_vm_tests::property_based_test_programs;
    use crate::vm::triton_vm_tests::small_tasm_test_programs;
    use crate::vm::triton_vm_tests::test_hash_nop_nop_lt;
    use crate::vm::AlgebraicExecutionTrace;
    use crate::vm::Program;

    use super::*;

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
    ) -> (Stark, MasterBaseTable, MasterBaseTable) {
        let (aet, stdout, program) = parse_setup_simulate(code, stdin.clone(), secret_in);

        let claim = Claim {
            input: stdin,
            program: program.to_bwords(),
            output: stdout,
            padded_height: 0,
        };
        let log_expansion_factor = 2;
        let security_level = 32;
        let parameters = StarkParameters::new(security_level, 1 << log_expansion_factor);
        let stark = Stark::new(claim, parameters);

        let mut master_base_table = MasterBaseTable::new(
            aet,
            &program.to_bwords(),
            stark.parameters.num_trace_randomizers,
            stark.fri.domain,
        );

        let unpadded_master_base_table = master_base_table.clone();
        master_base_table.pad();

        (stark, unpadded_master_base_table, master_base_table)
    }

    pub fn parse_simulate_pad_extend(
        code: &str,
        stdin: Vec<BFieldElement>,
        secret_in: Vec<BFieldElement>,
    ) -> (
        Stark,
        MasterBaseTable,
        MasterBaseTable,
        MasterExtTable,
        AllChallenges,
    ) {
        let (stark, unpadded_master_base_table, master_base_table) =
            parse_simulate_pad(code, stdin, secret_in);

        let dummy_challenges = AllChallenges::placeholder();
        let master_ext_table = master_base_table.extend(
            &dummy_challenges,
            stark.parameters.num_randomizer_polynomials,
        );

        (
            stark,
            unpadded_master_base_table,
            master_base_table,
            master_ext_table,
            dummy_challenges,
        )
    }

    /// To be used with `-- --nocapture`. Has mainly informative purpose.
    #[test]
    pub fn print_all_constraint_degrees() {
        for deg in all_degrees_with_origin(2, 2) {
            println!("{}", deg);
        }
    }

    #[test]
    pub fn check_io_terminals() {
        let read_nop_code = "read_io read_io read_io nop nop write_io push 17 write_io halt";
        let input_symbols = [3, 5, 7].map(BFieldElement::new).to_vec();
        let (stark, _, _, master_ext_table, all_challenges) =
            parse_simulate_pad_extend(read_nop_code, input_symbols, vec![]);

        let processor_table = master_ext_table.table(ProcessorTable);
        let processor_table_last_row = processor_table.slice(s![-1, ..]);
        let ptie = processor_table_last_row[InputTableEvalArg.table_index()];
        let ine = EvalArg::compute_terminal(
            &stark.claim.input,
            EvalArg::default_initial(),
            all_challenges.input_challenges.processor_eval_indeterminate,
        );
        assert_eq!(ptie, ine, "The input evaluation arguments do not match.");

        let ptoe = processor_table_last_row[OutputTableEvalArg.table_index()];
        let oute = EvalArg::compute_terminal(
            &stark.claim.output,
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
            let (stark, _, master_base_table, master_ext_table, all_challenges) =
                parse_simulate_pad_extend(&code, input, secret_input);

            let input_terminal = EvalArg::compute_terminal(
                &stark.claim.input,
                EvalArg::default_initial(),
                all_challenges
                    .processor_table_challenges
                    .standard_input_eval_indeterminate,
            );

            let output_terminal = EvalArg::compute_terminal(
                &stark.claim.output,
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

            let padded_height = master_base_table.padded_height;
            for (idx, (arg, _)) in grand_cross_table_arg.into_iter().enumerate() {
                let from = arg
                    .from()
                    .iter()
                    .map(|&from_column| {
                        master_ext_table.master_ext_matrix[(padded_height - 1, from_column)]
                    })
                    .fold(XFieldElement::one(), XFieldElement::mul);
                let to = arg
                    .to()
                    .iter()
                    .map(|&to_column| {
                        master_ext_table.master_ext_matrix[(padded_height - 1, to_column)]
                    })
                    .fold(XFieldElement::one(), XFieldElement::mul);
                assert_eq!(
                    from, to,
                    "Cross-table argument #{idx} must match for TASM snipped #{code_idx}."
                );
            }

            let processor_table = master_ext_table.table(ProcessorTable);
            let processor_table_last_row = processor_table.slice(s![-1, ..]);
            let ptie = processor_table_last_row[InputTableEvalArg.table_index()];
            assert_eq!(
                ptie, input_terminal,
                "The input terminal must match for TASM snipped #{code_idx}."
            );

            let ptoe = processor_table_last_row[OutputTableEvalArg.table_index()];
            assert_eq!(
                ptoe, output_terminal,
                "The output terminal must match for TASM snipped #{code_idx}."
            );
        }
    }

    #[test]
    fn constraint_polynomials_use_right_variable_count_test() {
        let challenges = AllChallenges::placeholder();
        let base_row = Array1::zeros(NUM_BASE_COLUMNS);
        let ext_row = Array1::zeros(NUM_EXT_COLUMNS);

        let br = base_row.view();
        let er = ext_row.view();

        ExtProgramTable::evaluate_initial_constraints(br, er, &challenges);
        ExtProgramTable::evaluate_consistency_constraints(br, er, &challenges);
        ExtProgramTable::evaluate_transition_constraints(br, er, br, er, &challenges);
        ExtProgramTable::evaluate_terminal_constraints(br, er, &challenges);

        ExtInstructionTable::evaluate_initial_constraints(br, er, &challenges);
        ExtInstructionTable::evaluate_consistency_constraints(br, er, &challenges);
        ExtInstructionTable::evaluate_transition_constraints(br, er, br, er, &challenges);
        ExtInstructionTable::evaluate_terminal_constraints(br, er, &challenges);

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
    }

    #[test]
    fn print_number_of_all_constraints_per_table() {
        let table_names = [
            "program table",
            "instruction table",
            "processor table",
            "op stack table",
            "ram table",
            "jump stack table",
            "hash table",
        ];
        let all_init = [
            ExtProgramTable::num_initial_quotients(),
            ExtInstructionTable::num_initial_quotients(),
            ExtProcessorTable::num_initial_quotients(),
            ExtOpStackTable::num_initial_quotients(),
            ExtRamTable::num_initial_quotients(),
            ExtJumpStackTable::num_initial_quotients(),
            ExtHashTable::num_initial_quotients(),
        ];
        let all_cons = [
            ExtProgramTable::num_consistency_quotients(),
            ExtInstructionTable::num_consistency_quotients(),
            ExtProcessorTable::num_consistency_quotients(),
            ExtOpStackTable::num_consistency_quotients(),
            ExtRamTable::num_consistency_quotients(),
            ExtJumpStackTable::num_consistency_quotients(),
            ExtHashTable::num_consistency_quotients(),
        ];
        let all_trans = [
            ExtProgramTable::num_transition_quotients(),
            ExtInstructionTable::num_transition_quotients(),
            ExtProcessorTable::num_transition_quotients(),
            ExtOpStackTable::num_transition_quotients(),
            ExtRamTable::num_transition_quotients(),
            ExtJumpStackTable::num_transition_quotients(),
            ExtHashTable::num_transition_quotients(),
        ];
        let all_term = [
            ExtProgramTable::num_terminal_quotients(),
            ExtInstructionTable::num_terminal_quotients(),
            ExtProcessorTable::num_terminal_quotients(),
            ExtOpStackTable::num_terminal_quotients(),
            ExtRamTable::num_terminal_quotients(),
            ExtJumpStackTable::num_terminal_quotients(),
            ExtHashTable::num_terminal_quotients(),
        ];

        let num_total_init: usize = all_init.iter().sum();
        let num_total_cons: usize = all_cons.iter().sum();
        let num_total_trans: usize = all_trans.iter().sum();
        let num_total_term: usize = all_term.iter().sum();
        let num_total = num_total_init + num_total_cons + num_total_trans + num_total_term;

        println!("| Table                |  Init |  Cons | Trans |  Term |   Sum |");
        println!("|:---------------------|------:|------:|------:|------:|------:|");
        for (name, num_init, num_cons, num_trans, num_term) in
            izip!(table_names, all_init, all_cons, all_trans, all_term)
        {
            let num_total = num_init + num_cons + num_trans + num_term;
            println!(
                "| {:<20} | {:>5} | {:>5} | {:>5} | {:>5} | {:>5} |",
                name, num_init, num_cons, num_trans, num_term, num_total,
            );
        }
        println!(
            "| {:<20} | {:>5} | {:>5} | {:>5} | {:>5} | {:>5} |",
            "Sum", num_total_init, num_total_cons, num_total_trans, num_total_term, num_total
        );
    }

    #[test]
    fn number_of_quotient_degree_bounds_match_number_of_constraints_test() {
        let base_row = Array1::zeros(NUM_BASE_COLUMNS);
        let ext_row = Array1::zeros(NUM_EXT_COLUMNS);
        let challenges = AllChallenges::placeholder();

        let br = base_row.view();
        let er = ext_row.view();

        assert_eq!(
            ExtProgramTable::num_initial_quotients(),
            ExtProgramTable::evaluate_initial_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtProgramTable::num_initial_quotients(),
            ExtProgramTable::initial_quotient_degree_bounds(2, 2).len()
        );
        assert_eq!(
            ExtInstructionTable::num_initial_quotients(),
            ExtInstructionTable::evaluate_initial_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtInstructionTable::num_initial_quotients(),
            ExtInstructionTable::initial_quotient_degree_bounds(2, 2).len()
        );
        assert_eq!(
            ExtProcessorTable::num_initial_quotients(),
            ExtProcessorTable::evaluate_initial_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtProcessorTable::num_initial_quotients(),
            ExtProcessorTable::initial_quotient_degree_bounds(2, 2).len()
        );
        assert_eq!(
            ExtOpStackTable::num_initial_quotients(),
            ExtOpStackTable::evaluate_initial_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtOpStackTable::num_initial_quotients(),
            ExtOpStackTable::initial_quotient_degree_bounds(2, 2).len()
        );
        assert_eq!(
            ExtRamTable::num_initial_quotients(),
            ExtRamTable::evaluate_initial_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtRamTable::num_initial_quotients(),
            ExtRamTable::initial_quotient_degree_bounds(2, 2).len()
        );
        assert_eq!(
            ExtJumpStackTable::num_initial_quotients(),
            ExtJumpStackTable::evaluate_initial_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtJumpStackTable::num_initial_quotients(),
            ExtJumpStackTable::initial_quotient_degree_bounds(2, 2).len()
        );
        assert_eq!(
            ExtHashTable::num_initial_quotients(),
            ExtHashTable::evaluate_initial_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtHashTable::num_initial_quotients(),
            ExtHashTable::initial_quotient_degree_bounds(2, 2).len()
        );

        assert_eq!(
            ExtProgramTable::num_consistency_quotients(),
            ExtProgramTable::evaluate_consistency_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtProgramTable::num_consistency_quotients(),
            ExtProgramTable::consistency_quotient_degree_bounds(2, 2).len()
        );
        assert_eq!(
            ExtInstructionTable::num_consistency_quotients(),
            ExtInstructionTable::evaluate_consistency_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtInstructionTable::num_consistency_quotients(),
            ExtInstructionTable::consistency_quotient_degree_bounds(2, 2).len()
        );
        assert_eq!(
            ExtProcessorTable::num_consistency_quotients(),
            ExtProcessorTable::evaluate_consistency_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtProcessorTable::num_consistency_quotients(),
            ExtProcessorTable::consistency_quotient_degree_bounds(2, 2).len()
        );
        assert_eq!(
            ExtOpStackTable::num_consistency_quotients(),
            ExtOpStackTable::evaluate_consistency_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtOpStackTable::num_consistency_quotients(),
            ExtOpStackTable::consistency_quotient_degree_bounds(2, 2).len()
        );
        assert_eq!(
            ExtRamTable::num_consistency_quotients(),
            ExtRamTable::evaluate_consistency_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtRamTable::num_consistency_quotients(),
            ExtRamTable::consistency_quotient_degree_bounds(2, 2).len()
        );
        assert_eq!(
            ExtJumpStackTable::num_consistency_quotients(),
            ExtJumpStackTable::evaluate_consistency_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtJumpStackTable::num_consistency_quotients(),
            ExtJumpStackTable::consistency_quotient_degree_bounds(2, 2).len()
        );
        assert_eq!(
            ExtHashTable::num_consistency_quotients(),
            ExtHashTable::evaluate_consistency_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtHashTable::num_consistency_quotients(),
            ExtHashTable::consistency_quotient_degree_bounds(2, 2).len()
        );

        assert_eq!(
            ExtProgramTable::num_transition_quotients(),
            ExtProgramTable::evaluate_transition_constraints(br, er, br, er, &challenges).len(),
        );
        assert_eq!(
            ExtProgramTable::num_transition_quotients(),
            ExtProgramTable::transition_quotient_degree_bounds(2, 2).len()
        );
        assert_eq!(
            ExtInstructionTable::num_transition_quotients(),
            ExtInstructionTable::evaluate_transition_constraints(br, er, br, er, &challenges).len(),
        );
        assert_eq!(
            ExtInstructionTable::num_transition_quotients(),
            ExtInstructionTable::transition_quotient_degree_bounds(2, 2).len()
        );
        assert_eq!(
            ExtProcessorTable::num_transition_quotients(),
            ExtProcessorTable::evaluate_transition_constraints(br, er, br, er, &challenges).len(),
        );
        assert_eq!(
            ExtProcessorTable::num_transition_quotients(),
            ExtProcessorTable::transition_quotient_degree_bounds(2, 2).len()
        );
        assert_eq!(
            ExtOpStackTable::num_transition_quotients(),
            ExtOpStackTable::evaluate_transition_constraints(br, er, br, er, &challenges).len(),
        );
        assert_eq!(
            ExtOpStackTable::num_transition_quotients(),
            ExtOpStackTable::transition_quotient_degree_bounds(2, 2).len()
        );
        assert_eq!(
            ExtRamTable::num_transition_quotients(),
            ExtRamTable::evaluate_transition_constraints(br, er, br, er, &challenges).len(),
        );
        assert_eq!(
            ExtRamTable::num_transition_quotients(),
            ExtRamTable::transition_quotient_degree_bounds(2, 2).len()
        );
        assert_eq!(
            ExtJumpStackTable::num_transition_quotients(),
            ExtJumpStackTable::evaluate_transition_constraints(br, er, br, er, &challenges).len(),
        );
        assert_eq!(
            ExtJumpStackTable::num_transition_quotients(),
            ExtJumpStackTable::transition_quotient_degree_bounds(2, 2).len()
        );
        assert_eq!(
            ExtHashTable::num_transition_quotients(),
            ExtHashTable::evaluate_transition_constraints(br, er, br, er, &challenges).len(),
        );
        assert_eq!(
            ExtHashTable::num_transition_quotients(),
            ExtHashTable::transition_quotient_degree_bounds(2, 2).len()
        );

        assert_eq!(
            ExtProgramTable::num_terminal_quotients(),
            ExtProgramTable::evaluate_terminal_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtProgramTable::num_terminal_quotients(),
            ExtProgramTable::terminal_quotient_degree_bounds(2, 2).len()
        );
        assert_eq!(
            ExtInstructionTable::num_terminal_quotients(),
            ExtInstructionTable::evaluate_terminal_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtInstructionTable::num_terminal_quotients(),
            ExtInstructionTable::terminal_quotient_degree_bounds(2, 2).len()
        );
        assert_eq!(
            ExtProcessorTable::num_terminal_quotients(),
            ExtProcessorTable::evaluate_terminal_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtProcessorTable::num_terminal_quotients(),
            ExtProcessorTable::terminal_quotient_degree_bounds(2, 2).len()
        );
        assert_eq!(
            ExtOpStackTable::num_terminal_quotients(),
            ExtOpStackTable::evaluate_terminal_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtOpStackTable::num_terminal_quotients(),
            ExtOpStackTable::terminal_quotient_degree_bounds(2, 2).len()
        );
        assert_eq!(
            ExtRamTable::num_terminal_quotients(),
            ExtRamTable::evaluate_terminal_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtRamTable::num_terminal_quotients(),
            ExtRamTable::terminal_quotient_degree_bounds(2, 2).len()
        );
        assert_eq!(
            ExtJumpStackTable::num_terminal_quotients(),
            ExtJumpStackTable::evaluate_terminal_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtJumpStackTable::num_terminal_quotients(),
            ExtJumpStackTable::terminal_quotient_degree_bounds(2, 2).len()
        );
        assert_eq!(
            ExtHashTable::num_terminal_quotients(),
            ExtHashTable::evaluate_terminal_constraints(br, er, &challenges).len(),
        );
        assert_eq!(
            ExtHashTable::num_terminal_quotients(),
            ExtHashTable::terminal_quotient_degree_bounds(2, 2).len()
        );
    }

    #[test]
    fn triton_table_constraints_evaluate_to_zero_on_halt_test() {
        let zero = XFieldElement::zero();
        let source_code_and_input = test_halt();
        let (_, _, master_base_table, master_ext_table, challenges) =
            parse_simulate_pad_extend(&source_code_and_input.source_code, vec![], vec![]);

        let evaluated_initial_constraints = evaluate_all_initial_constraints(
            master_base_table.master_base_matrix.row(0),
            master_ext_table.master_ext_matrix.row(0),
            &challenges,
        );
        let num_initial_constraints = evaluated_initial_constraints.len();
        for (constraint_idx, ebc) in evaluated_initial_constraints.into_iter().enumerate() {
            assert_eq!(
                zero, ebc,
                "Failed initial constraint {constraint_idx}/{num_initial_constraints}.",
            );
        }

        let num_rows = master_base_table.master_base_matrix.nrows();
        for row_idx in 0..num_rows {
            let base_row = master_base_table.master_base_matrix.row(row_idx);
            let ext_row = master_ext_table.master_ext_matrix.row(row_idx);
            let evaluated_consistency_constraints =
                evaluate_all_consistency_constraints(base_row, ext_row, &challenges);
            let num_consistency_constraints = evaluated_consistency_constraints.len();
            for (constraint_idx, ecc) in evaluated_consistency_constraints.into_iter().enumerate() {
                assert_eq!(
                    zero, ecc,
                    "Failed consistency constraint {}/{}. Row index: {}/{}",
                    constraint_idx, num_consistency_constraints, row_idx, num_rows,
                );
            }
        }

        for row_idx in 0..num_rows - 1 {
            let base_row = master_base_table.master_base_matrix.row(row_idx);
            let ext_row = master_ext_table.master_ext_matrix.row(row_idx);
            let next_base_row = master_base_table.master_base_matrix.row(row_idx + 1);
            let next_ext_row = master_ext_table.master_ext_matrix.row(row_idx + 1);
            let evaluated_transition_constraints = evaluate_all_transition_constraints(
                base_row,
                ext_row,
                next_base_row,
                next_ext_row,
                &challenges,
            );
            let num_transition_constraints = evaluated_transition_constraints.len();
            for (constraint_idx, etc) in evaluated_transition_constraints.into_iter().enumerate() {
                assert_eq!(
                    zero, etc,
                    "Failed transition constraint {}/{}. Row index: {}/{}",
                    constraint_idx, num_transition_constraints, row_idx, num_rows,
                );
            }
        }

        let evaluated_terminal_constraints = evaluate_all_terminal_constraints(
            master_base_table.master_base_matrix.row(num_rows - 1),
            master_ext_table.master_ext_matrix.row(num_rows - 1),
            &challenges,
        );
        let num_terminal_constraints = evaluated_terminal_constraints.len();
        for (constraint_idx, etermc) in evaluated_terminal_constraints.into_iter().enumerate() {
            assert_eq!(
                zero, etermc,
                "Failed terminal constraint {constraint_idx}/{num_terminal_constraints}.",
            );
        }
    }

    #[test]
    fn triton_table_constraints_evaluate_to_zero_test_on_simple_program() {
        let zero = XFieldElement::zero();
        let (_, _, mbt, met, challenges) =
            parse_simulate_pad_extend(sample_programs::FIBONACCI_LT, vec![], vec![]);

        let fbr = mbt.master_base_matrix.row(0);
        let fer = met.master_ext_matrix.row(0);
        let evaluated_initial_constraints = [
            ExtProgramTable::evaluate_initial_constraints(fbr, fer, &challenges),
            ExtInstructionTable::evaluate_initial_constraints(fbr, fer, &challenges),
            ExtProcessorTable::evaluate_initial_constraints(fbr, fer, &challenges),
            ExtOpStackTable::evaluate_initial_constraints(fbr, fer, &challenges),
            ExtRamTable::evaluate_initial_constraints(fbr, fer, &challenges),
            ExtJumpStackTable::evaluate_initial_constraints(fbr, fer, &challenges),
            ExtHashTable::evaluate_initial_constraints(fbr, fer, &challenges),
        ]
        .concat();
        for (idx, xfe) in evaluated_initial_constraints.into_iter().enumerate() {
            assert_eq!(
                zero, xfe,
                "Failed initial constraint with global index {idx}.",
            )
        }

        for (row_idx, (br, er)) in mbt
            .master_base_matrix
            .rows()
            .into_iter()
            .zip_eq(met.master_ext_matrix.rows().into_iter())
            .enumerate()
        {
            let evaluated_consistency_constraints = [
                ExtProgramTable::evaluate_consistency_constraints(br, er, &challenges),
                ExtInstructionTable::evaluate_consistency_constraints(br, er, &challenges),
                ExtProcessorTable::evaluate_consistency_constraints(br, er, &challenges),
                ExtOpStackTable::evaluate_consistency_constraints(br, er, &challenges),
                ExtRamTable::evaluate_consistency_constraints(br, er, &challenges),
                ExtJumpStackTable::evaluate_consistency_constraints(br, er, &challenges),
                ExtHashTable::evaluate_consistency_constraints(br, er, &challenges),
            ]
            .concat();
            for (idx, xfe) in evaluated_consistency_constraints.into_iter().enumerate() {
                assert_eq!(
                    zero, xfe,
                    "Failed consistency constraint with global index {idx} on row {row_idx}.",
                )
            }
        }

        for row_idx in 0..mbt.master_base_matrix.nrows() - 1 {
            let curr_base_row = mbt.master_base_matrix.row(row_idx);
            let curr_ext_row = met.master_ext_matrix.row(row_idx);
            let next_base_row = mbt.master_base_matrix.row(row_idx + 1);
            let next_ext_row = met.master_ext_matrix.row(row_idx + 1);
            let evaluated_transition_constraints = [
                ExtProgramTable::evaluate_transition_constraints(
                    curr_base_row,
                    curr_ext_row,
                    next_base_row,
                    next_ext_row,
                    &challenges,
                ),
                ExtInstructionTable::evaluate_transition_constraints(
                    curr_base_row,
                    curr_ext_row,
                    next_base_row,
                    next_ext_row,
                    &challenges,
                ),
                ExtProcessorTable::evaluate_transition_constraints(
                    curr_base_row,
                    curr_ext_row,
                    next_base_row,
                    next_ext_row,
                    &challenges,
                ),
                ExtOpStackTable::evaluate_transition_constraints(
                    curr_base_row,
                    curr_ext_row,
                    next_base_row,
                    next_ext_row,
                    &challenges,
                ),
                ExtRamTable::evaluate_transition_constraints(
                    curr_base_row,
                    curr_ext_row,
                    next_base_row,
                    next_ext_row,
                    &challenges,
                ),
                ExtJumpStackTable::evaluate_transition_constraints(
                    curr_base_row,
                    curr_ext_row,
                    next_base_row,
                    next_ext_row,
                    &challenges,
                ),
                ExtHashTable::evaluate_transition_constraints(
                    curr_base_row,
                    curr_ext_row,
                    next_base_row,
                    next_ext_row,
                    &challenges,
                ),
            ]
            .concat();
            for (idx, xfe) in evaluated_transition_constraints.into_iter().enumerate() {
                assert_eq!(
                    zero, xfe,
                    "Failed transition constraint with global index {idx} on row {row_idx}.",
                )
            }
        }

        let lbr = mbt
            .master_base_matrix
            .row(mbt.master_base_matrix.nrows() - 1);
        let ler = met.master_ext_matrix.row(met.master_ext_matrix.nrows() - 1);
        let evaluated_terminal_constraints = [
            ExtProgramTable::evaluate_terminal_constraints(lbr, ler, &challenges),
            ExtInstructionTable::evaluate_terminal_constraints(lbr, ler, &challenges),
            ExtProcessorTable::evaluate_terminal_constraints(lbr, ler, &challenges),
            ExtOpStackTable::evaluate_terminal_constraints(lbr, ler, &challenges),
            ExtRamTable::evaluate_terminal_constraints(lbr, ler, &challenges),
            ExtJumpStackTable::evaluate_terminal_constraints(lbr, ler, &challenges),
            ExtHashTable::evaluate_terminal_constraints(lbr, ler, &challenges),
        ]
        .concat();
        for (idx, xfe) in evaluated_terminal_constraints.into_iter().enumerate() {
            assert_eq!(
                zero, xfe,
                "Failed terminal constraint with global index {idx}",
            )
        }
    }

    #[test]
    fn triton_prove_verify_simple_program_test() {
        let code_with_input = test_hash_nop_nop_lt();
        let (stark, proof) = parse_simulate_prove(
            &code_with_input.source_code,
            code_with_input.input.clone(),
            code_with_input.secret_input.clone(),
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
            &mut None,
        );

        let result = stark.verify(proof, &mut None);
        if let Err(e) = result {
            panic!("The Verifier is unhappy! {}", e);
        }
        assert!(result.unwrap());
    }

    #[test]
    #[ignore = "used for tracking&debugging deserialization errors"]
    fn triton_prove_halt_save_error_test() {
        let code_with_input = test_halt();

        for _ in 0..100 {
            let (stark, proof) = parse_simulate_prove(
                &code_with_input.source_code,
                code_with_input.input.clone(),
                code_with_input.secret_input.clone(),
                &mut None,
            );

            let filename = "halt_error.tsp";
            let result = stark.verify(proof.clone(), &mut None);
            if let Err(e) = result {
                if let Err(e) = save_proof(filename, proof) {
                    panic!("Unsyntactical proof and can't save! {}", e);
                }
                panic!(
                    "Saved proof to {} because verifier is unhappy! {}",
                    filename, e
                );
            }
            assert!(result.unwrap());
        }
    }

    #[test]
    #[ignore = "used for tracking&debugging deserialization errors"]
    fn triton_load_verify_halt_test() {
        let code_with_input = test_halt();
        let (stark, _) = parse_simulate_prove(
            &code_with_input.source_code,
            code_with_input.input.clone(),
            code_with_input.secret_input.clone(),
            &mut None,
        );

        let filename = "halt_error.tsp";
        let proof = match load_proof(filename) {
            Ok(p) => p,
            Err(e) => panic!("Could not load proof from disk at {}: {}", filename, e),
        };

        let result = stark.verify(proof, &mut None);
        if let Err(e) = result {
            panic!("Verifier is unhappy! {}", e);
        }
        assert!(result.unwrap());
    }

    #[test]
    fn prove_verify_fibonacci_100_test() {
        let source_code = sample_programs::FIBONACCI_VIT;
        let stdin = vec![100_u64.into()];
        let secret_in = vec![];

        let profiler = TritonProfiler::new("prove_verify_fibonacci_100_test");
        let (stark, proof) =
            parse_simulate_prove(source_code, stdin, secret_in, &mut Some(profiler));

        println!("between prove and verify");

        let result = stark.verify(proof, &mut None);
        if let Err(e) = result {
            panic!("The Verifier is unhappy! {}", e);
        }
        assert!(result.unwrap());
    }

    #[test]
    #[ignore = "stress test"]
    fn prove_fib_successively_larger() {
        let source_code = sample_programs::FIBONACCI_VIT;

        for fibonacci_number in [100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600, 51200] {
            let mut profiler = Some(TritonProfiler::new(&format!(
                "element #{fibonacci_number:>4} from Fibonacci sequence"
            )));
            let stdin = vec![BFieldElement::new(fibonacci_number)];
            let (stark, _) = parse_simulate_prove(source_code, stdin, vec![], &mut profiler);
            let log_fri_dom_len = log_2_floor(stark.fri.domain.length as u128);
            let fri_dom_len_str = format!("log_2 of FRI domain length: {}", log_fri_dom_len);
            prof_start!(profiler, &fri_dom_len_str);
            prof_stop!(profiler, &fri_dom_len_str);
            if let Some(mut p) = profiler {
                p.finish();
                println!("{}", p.report())
            }
        }
    }
}
