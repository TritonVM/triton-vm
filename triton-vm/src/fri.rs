use std::error::Error;
use std::fmt;
use std::marker::PhantomData;

use anyhow::bail;
use anyhow::Result;
use itertools::Itertools;
use num_traits::One;
use rayon::iter::*;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::ntt::intt;
use twenty_first::shared_math::other::log_2_ceil;
use twenty_first::shared_math::polynomial::Polynomial;
use twenty_first::shared_math::tip5::Digest;
use twenty_first::shared_math::traits::CyclicGroupGenerator;
use twenty_first::shared_math::traits::FiniteField;
use twenty_first::shared_math::traits::ModPowU32;
use twenty_first::shared_math::x_field_element::XFieldElement;
use twenty_first::util_types::algebraic_hasher::AlgebraicHasher;
use twenty_first::util_types::merkle_tree::MerkleTree;
use twenty_first::util_types::merkle_tree_maker::MerkleTreeMaker;

use crate::arithmetic_domain::ArithmeticDomain;
use crate::profiler::prof_start;
use crate::profiler::prof_stop;
use crate::profiler::TritonProfiler;
use crate::proof_item::FriResponse;
use crate::proof_item::ProofItem;
use crate::proof_stream::ProofStream;
use crate::stark::MTMaker;

impl Error for FriValidationError {}

impl fmt::Display for FriValidationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Deserialization error for LowDegreeProof: {self:?}")
    }
}

#[derive(PartialEq, Eq, Debug)]
pub enum FriValidationError {
    BadMerkleAuthenticationPath,
    MismatchingLastCodeword,
    LastIterationTooHighDegree,
    BadMerkleRootForLastCodeword,
}

#[derive(Debug, Clone)]
pub struct Fri<H> {
    // In STARK, the expansion factor <FRI domain length> / max_degree, where
    // `max_degree` is the max degree of any interpolation rounded up to the
    // nearest power of 2.
    pub expansion_factor: usize,
    pub num_colinearity_checks: usize,
    pub domain: ArithmeticDomain,
    _hasher: PhantomData<H>,
}

impl<H: AlgebraicHasher> Fri<H> {
    pub fn new(
        offset: BFieldElement,
        domain_length: usize,
        expansion_factor: usize,
        colinearity_checks_count: usize,
    ) -> Self {
        let domain = ArithmeticDomain::of_length(domain_length).with_offset(offset);
        let _hasher = PhantomData;
        Self {
            domain,
            expansion_factor,
            num_colinearity_checks: colinearity_checks_count,
            _hasher,
        }
    }

    /// Build the Merkle authentication structure for the codeword at the given indices
    /// and enqueue the corresponding values and the authentication structure on the proof stream.
    fn enqueue_auth_pairs(
        indices: &[usize],
        codeword: &[XFieldElement],
        merkle_tree: &MerkleTree<H>,
        proof_stream: &mut ProofStream<H>,
    ) {
        let auth_structure = merkle_tree.get_authentication_structure(indices);
        let revealed_leaves = indices.iter().map(|&i| codeword[i]).collect_vec();
        let fri_response = FriResponse {
            auth_structure,
            revealed_leaves,
        };
        let fri_response = ProofItem::FriResponse(fri_response);
        proof_stream.enqueue(&fri_response)
    }

    /// Given a merkle `root`, the `tree_height`, the `indices` of the values to dequeue from the
    /// proof stream, and the (correctly set) `proof_stream`, verify whether the values at the
    /// `indices` are members of the set committed to by the merkle `root` and return these values
    /// if they are. Fails otherwise.
    fn dequeue_and_authenticate(
        root: Digest,
        tree_height: usize,
        indices: &[usize],
        proof_stream: &mut ProofStream<H>,
    ) -> Result<Vec<XFieldElement>> {
        let fri_response = proof_stream.dequeue()?.as_fri_response()?;
        let FriResponse {
            auth_structure,
            revealed_leaves,
        } = fri_response;
        debug_assert_eq!(indices.len(), revealed_leaves.len());
        let leaf_digests = revealed_leaves.iter().map(|&xfe| xfe.into()).collect_vec();

        match MerkleTree::<H>::verify_authentication_structure(
            root,
            tree_height,
            indices,
            &leaf_digests,
            &auth_structure,
        ) {
            true => Ok(revealed_leaves),
            false => bail!(FriValidationError::BadMerkleAuthenticationPath),
        }
    }

    /// Create a FRI proof and return chosen indices of round 0 and Merkle root of round 0 codeword
    pub fn prove(
        &self,
        codeword: &[XFieldElement],
        proof_stream: &mut ProofStream<H>,
    ) -> (Vec<usize>, Digest) {
        debug_assert_eq!(
            self.domain.length,
            codeword.len(),
            "Initial codeword length must match FRI domain length."
        );

        let (codewords, merkle_trees): (Vec<_>, Vec<_>) =
            self.commit(codeword, proof_stream).into_iter().unzip();

        // Fiat-Shamir to get indices at which to reveal the codeword
        let initial_a_indices =
            proof_stream.sample_indices(self.domain.length, self.num_colinearity_checks);
        let initial_b_indices = initial_a_indices
            .iter()
            .map(|&idx| (idx + self.domain.length / 2) % self.domain.length)
            .collect_vec();
        let top_level_indices = initial_a_indices
            .iter()
            .copied()
            .chain(initial_b_indices)
            .collect_vec();

        // query phase
        // query step 0: enqueue authentication structure for all points `A` into proof stream
        Self::enqueue_auth_pairs(&initial_a_indices, codeword, &merkle_trees[0], proof_stream);
        // query step 1: loop over FRI rounds, enqueue authentication structure for all points `B`
        let mut current_domain_len = self.domain.length;
        let mut b_indices = initial_a_indices;
        // the last codeword is transmitted to the verifier in the clear. Thus, no co-linearity
        // check is needed for the last codeword and we only have to look at the interval given here
        for r in 0..merkle_trees.len() - 1 {
            assert_eq!(codewords[r].len(), current_domain_len);
            b_indices = b_indices
                .iter()
                .map(|x| (x + current_domain_len / 2) % current_domain_len)
                .collect();
            Self::enqueue_auth_pairs(&b_indices, &codewords[r], &merkle_trees[r], proof_stream);
            current_domain_len /= 2;
        }

        let merkle_root_of_1st_round = merkle_trees[0].get_root();
        (top_level_indices, merkle_root_of_1st_round)
    }

    fn commit(
        &self,
        codeword: &[XFieldElement],
        proof_stream: &mut ProofStream<H>,
    ) -> Vec<(Vec<XFieldElement>, MerkleTree<H>)> {
        let one = XFieldElement::one();
        let two_inv = one / (one + one);
        let num_rounds = self.num_rounds();

        let mut subgroup_generator = self.domain.generator;
        let mut offset = self.domain.offset;
        let mut codeword = codeword.to_vec();
        let mut codewords_and_merkle_trees = Vec::with_capacity(num_rounds);

        // Compute and send Merkle root
        let mut digests = Vec::with_capacity(codeword.len());
        codeword
            .par_iter()
            .map(|&xfe| xfe.into())
            .collect_into_vec(&mut digests);

        let mt = MTMaker::from_digests(&digests);
        proof_stream.enqueue(&ProofItem::MerkleRoot(mt.get_root()));
        codewords_and_merkle_trees.push((codeword.clone(), mt));

        for _round in 0..num_rounds {
            // Get challenge for folding
            let alpha = proof_stream.sample_scalars(1)[0];

            let x_offset = subgroup_generator
                .get_cyclic_group_elements(None)
                .into_par_iter()
                .map(|x| x * offset)
                .collect();
            let x_offset_inverses = BFieldElement::batch_inversion(x_offset);

            let n = codeword.len();
            codeword = (0..n / 2)
                .into_par_iter()
                .map(|i| {
                    let scaled_offset_inv = alpha * x_offset_inverses[i];
                    let left_summand = (one + scaled_offset_inv) * codeword[i];
                    let right_summand = (one - scaled_offset_inv) * codeword[n / 2 + i];
                    (left_summand + right_summand) * two_inv
                })
                .collect();

            // Compute and send Merkle root. We have to do that within this loop, since the next
            // round's alpha must be calculated from the previous round's Merkle root.
            codeword
                .par_iter()
                .map(|&xfe| xfe.into())
                .collect_into_vec(&mut digests);

            let mt = MTMaker::from_digests(&digests);
            proof_stream.enqueue(&ProofItem::MerkleRoot(mt.get_root()));
            codewords_and_merkle_trees.push((codeword.clone(), mt));

            // Update subgroup generator and offset
            subgroup_generator = subgroup_generator * subgroup_generator;
            offset = offset * offset;
        }

        // Send the last codeword in the clear
        proof_stream.enqueue(&ProofItem::FriCodeword(codeword));

        codewords_and_merkle_trees
    }

    /// Verify low-degreeness of the polynomial on the proof stream.
    /// Returns the indices and revealed elements of the codeword at the top level of the FRI proof.
    pub fn verify(
        &self,
        proof_stream: &mut ProofStream<H>,
        maybe_profiler: &mut Option<TritonProfiler>,
    ) -> Result<Vec<(usize, XFieldElement)>> {
        prof_start!(maybe_profiler, "init");
        let num_rounds = self.num_rounds();

        // Extract all roots and calculate alpha based on Fiat-Shamir challenge
        let mut roots = Vec::with_capacity(num_rounds);
        let mut alphas = Vec::with_capacity(num_rounds);

        let first_root = proof_stream.dequeue()?.as_merkle_root()?;
        roots.push(first_root);
        prof_stop!(maybe_profiler, "init");

        prof_start!(maybe_profiler, "roots and alpha");
        for _round in 0..num_rounds {
            // Get a challenge from the proof stream
            let alpha = proof_stream.sample_scalars(1)[0];
            alphas.push(alpha);

            let root = proof_stream.dequeue()?.as_merkle_root()?;
            roots.push(root);
        }
        prof_stop!(maybe_profiler, "roots and alpha");

        prof_start!(maybe_profiler, "last codeword matches root", "hash");
        // Extract last codeword
        let last_codeword = proof_stream.dequeue()?.as_fri_codeword()?;

        // Check if last codeword matches the given root
        let codeword_digests = last_codeword.iter().map(|&xfe| xfe.into()).collect_vec();
        let last_codeword_mt: MerkleTree<H> = MTMaker::from_digests(&codeword_digests);
        let last_root = roots.last().unwrap();
        if *last_root != last_codeword_mt.get_root() {
            bail!(FriValidationError::BadMerkleRootForLastCodeword);
        }
        prof_stop!(maybe_profiler, "last codeword matches root");

        // Verify that last codeword is of sufficiently low degree

        prof_start!(maybe_profiler, "last codeword has low degree");
        // Compute interpolant to get the degree of the last codeword.
        // Note that we don't have to scale the polynomial back to the trace subgroup since we
        // only check its degree and don't use it further.
        let log_2_of_n = last_codeword.len().ilog2();
        let mut last_polynomial = last_codeword.clone();

        let last_fri_domain_generator = self
            .domain
            .generator
            .mod_pow_u32(2u32.pow(num_rounds as u32));
        intt::<XFieldElement>(&mut last_polynomial, last_fri_domain_generator, log_2_of_n);
        let last_poly_degree = Polynomial::new(last_polynomial).degree();

        let last_round_max_degree = self.last_round_max_degree();
        if last_poly_degree > last_round_max_degree as isize {
            eprintln!(
                "last_poly_degree is {last_poly_degree}, \
                 last_round_max_degree is {last_round_max_degree}",
            );
            bail!(FriValidationError::LastIterationTooHighDegree);
        }
        prof_stop!(maybe_profiler, "last codeword has low degree");

        // Query phase
        prof_start!(maybe_profiler, "query phase");
        // query step 0: get "A" indices and verify set membership of corresponding values.
        prof_start!(maybe_profiler, "sample indices");
        let mut a_indices =
            proof_stream.sample_indices(self.domain.length, self.num_colinearity_checks);
        prof_stop!(maybe_profiler, "sample indices");

        prof_start!(maybe_profiler, "dequeue and authenticate", "hash");
        let tree_height = self.domain.length.ilog2() as usize;
        let mut a_values =
            Self::dequeue_and_authenticate(roots[0], tree_height, &a_indices, proof_stream)?;
        prof_stop!(maybe_profiler, "dequeue and authenticate");

        // save indices and revealed leafs of first round's codeword for returning
        let revealed_indices_and_elements_first_half = a_indices
            .iter()
            .copied()
            .zip_eq(a_values.iter().copied())
            .collect_vec();
        // these indices and values will be computed in the first iteration of the main loop below
        let mut revealed_indices_and_elements_second_half = vec![];

        // set up "B" for offsetting inside loop.  Note that "B" and "A" indices can be calcuated
        // from each other.
        let mut b_indices = a_indices.clone();
        let mut current_domain_len = self.domain.length;
        let mut current_tree_height = tree_height;

        // query step 1:  loop over FRI rounds, verify "B"s, compute values for "C"s
        prof_start!(maybe_profiler, "loop");
        for round in 0..num_rounds {
            // get "B" indices and verify set membership of corresponding values
            b_indices = b_indices
                .iter()
                .map(|x| (x + current_domain_len / 2) % current_domain_len)
                .collect();
            let b_values = Self::dequeue_and_authenticate(
                roots[round],
                current_tree_height,
                &b_indices,
                proof_stream,
            )?;
            debug_assert_eq!(self.num_colinearity_checks, a_indices.len());
            debug_assert_eq!(self.num_colinearity_checks, b_indices.len());
            debug_assert_eq!(self.num_colinearity_checks, a_values.len());
            debug_assert_eq!(self.num_colinearity_checks, b_values.len());

            if round == 0 {
                // save other half of indices and revealed leafs of first round for returning
                revealed_indices_and_elements_second_half = b_indices
                    .iter()
                    .copied()
                    .zip_eq(b_values.iter().copied())
                    .collect_vec();
            }

            // compute "C" indices and values for next round from "A" and "B" of current round
            current_domain_len /= 2;
            current_tree_height -= 1;
            let c_indices = a_indices.iter().map(|x| x % current_domain_len).collect();
            let c_values = (0..self.num_colinearity_checks)
                .into_par_iter()
                .map(|i| {
                    let point_a_x = self.domain_value(a_indices[i], round);
                    let point_b_x = self.domain_value(b_indices[i], round);
                    let point_a = (point_a_x, a_values[i]);
                    let point_b = (point_b_x, b_values[i]);
                    Polynomial::<XFieldElement>::get_colinear_y(point_a, point_b, alphas[round])
                })
                .collect();

            // next rounds "A"s correspond to current rounds "C"s
            a_indices = c_indices;
            a_values = c_values;
        }
        prof_stop!(maybe_profiler, "loop");
        prof_stop!(maybe_profiler, "query phase");

        // Finally compare "C" values (which are named "A" values in this enclosing scope) with
        // last codeword from the proofstream.
        prof_start!(maybe_profiler, "compare last codeword");
        a_indices = a_indices.iter().map(|x| x % current_domain_len).collect();
        if (0..self.num_colinearity_checks).any(|i| last_codeword[a_indices[i]] != a_values[i]) {
            bail!(FriValidationError::MismatchingLastCodeword);
        }
        prof_stop!(maybe_profiler, "compare last codeword");

        let revealed_indices_and_elements = revealed_indices_and_elements_first_half
            .into_iter()
            .chain(revealed_indices_and_elements_second_half)
            .collect_vec();
        Ok(revealed_indices_and_elements)
    }

    /// Given index `i` of the FRI codeword in round `round`, compute the corresponding value in the
    /// FRI (co-)domain. This corresponds to `ω^i` in `f(ω^i)` from
    /// [STARK-Anatomy](https://neptune.cash/learn/stark-anatomy/fri/#split-and-fold).
    pub fn domain_value(&self, idx: usize, round: usize) -> XFieldElement {
        let domain_value = self.domain.domain_value(idx as u32);
        let round_adjusting_exponent = 1 << round;
        let round_adjusted_domain_value = domain_value.mod_pow(round_adjusting_exponent);
        round_adjusted_domain_value.lift()
    }

    pub fn num_rounds(&self) -> usize {
        let first_round_code_dimension = self.first_round_max_degree() + 1;
        let max_num_rounds = log_2_ceil(first_round_code_dimension as u128);

        // Skip rounds for which Merkle tree verification cost exceeds arithmetic cost,
        // because more than half the codeword's locations are queried.
        let num_rounds_checking_all_locations = self.num_colinearity_checks.ilog2() as u64;
        let num_rounds_checking_most_locations = num_rounds_checking_all_locations + 1;

        max_num_rounds.saturating_sub(num_rounds_checking_most_locations) as usize
    }

    pub fn last_round_max_degree(&self) -> usize {
        self.first_round_max_degree() >> self.num_rounds()
    }

    pub fn first_round_max_degree(&self) -> usize {
        assert!(self.domain.length >= self.expansion_factor);
        (self.domain.length / self.expansion_factor) - 1
    }
}

#[cfg(test)]
mod triton_xfri_tests {
    use itertools::Itertools;
    use num_traits::Zero;
    use rand::thread_rng;
    use rand::Rng;
    use twenty_first::shared_math::b_field_element::BFieldElement;
    use twenty_first::shared_math::other::random_elements;
    use twenty_first::shared_math::tip5::Tip5;
    use twenty_first::shared_math::tip5::RATE;
    use twenty_first::shared_math::traits::CyclicGroupGenerator;
    use twenty_first::shared_math::traits::ModPowU32;
    use twenty_first::shared_math::x_field_element::XFieldElement;
    use twenty_first::util_types::algebraic_hasher::SpongeHasher;

    use super::*;

    #[test]
    fn sample_indices_test() {
        type H = Tip5;

        let subgroup_order = 1 << 20;
        let expansion_factor = 4;
        let colinearity_checks = 16;
        // todo: Figure out by how much to oversample for the given parameters.
        let oversampling_summand = 10;

        let random_bfield_elements: [BFieldElement; RATE] =
            random_elements(RATE).try_into().unwrap();
        let mut sponge_state = H::init();
        H::absorb(&mut sponge_state, &random_bfield_elements);

        let num_indices_to_sample = colinearity_checks + oversampling_summand;
        let fri = get_x_field_fri_test_object::<H>(
            subgroup_order,
            expansion_factor,
            num_indices_to_sample,
        );
        let mut indices = H::sample_indices(
            &mut sponge_state,
            fri.domain.length as u32,
            fri.num_colinearity_checks,
        );
        indices.sort_unstable();
        indices.dedup();
        let num_unique_indices = indices.len();

        assert!(
            num_unique_indices >= colinearity_checks,
            "Too few unique indices: only got {num_unique_indices} uniques \
            (expecting at least {colinearity_checks} uniques) \
            after sampling {num_indices_to_sample} indices \
            from a domain of length {}.",
            fri.domain.length,
        );
    }

    #[test]
    fn num_rounds_are_reasonable() {
        let mut rng = thread_rng();
        for _ in 0..1 << 11 {
            let log_2_expansion_factor = rng.gen_range(0..=8);
            let expansion_factor = 1 << log_2_expansion_factor;

            let log_2_domain_length = rng.gen_range(log_2_expansion_factor..=20);
            let domain_length = 1 << log_2_domain_length;

            let num_colinearity_checks = rng.gen_range(1..=320);

            let fri = get_x_field_fri_test_object::<Tip5>(
                domain_length,
                expansion_factor,
                num_colinearity_checks,
            );
            let expected_last_round_max_degree = fri.first_round_max_degree() >> fri.num_rounds();

            println!("fri: {:?}", fri);
            println!("expected_last_round_max_degree = {expected_last_round_max_degree}");

            assert_eq!(expected_last_round_max_degree, fri.last_round_max_degree());

            if fri.num_rounds() > 0 {
                assert!(fri.num_colinearity_checks <= expected_last_round_max_degree);
                assert!(expected_last_round_max_degree < 2 * fri.num_colinearity_checks);
            }
        }
    }

    #[test]
    fn fri_on_x_field_test() {
        type H = Tip5;

        let subgroup_order = 1024;
        let expansion_factor = 4;
        let colinearity_check_count = 6;
        let fri: Fri<H> =
            get_x_field_fri_test_object(subgroup_order, expansion_factor, colinearity_check_count);
        let mut proof_stream = ProofStream::new();

        // corresponds to the linear polynomial `x`
        let subgroup = fri.domain.generator.lift().get_cyclic_group_elements(None);

        fri.prove(&subgroup, &mut proof_stream);

        // reset sponge state to start verification
        proof_stream.sponge_state = H::init();
        let verdict = fri.verify(&mut proof_stream, &mut None);
        verdict.unwrap();
    }

    #[test]
    fn prove_and_verify_low_degree_of_twice_cubing_plus_one() {
        type H = Tip5;

        let subgroup_order = 1024;
        let expansion_factor = 4;
        let colinearity_check_count = 6;
        let fri: Fri<H> =
            get_x_field_fri_test_object(subgroup_order, expansion_factor, colinearity_check_count);
        let mut proof_stream = ProofStream::new();

        let zero = XFieldElement::zero();
        let one = XFieldElement::one();
        let two = one + one;
        let poly = Polynomial::<XFieldElement>::new(vec![one, zero, zero, two]);
        let codeword = fri.domain.evaluate(&poly);

        fri.prove(&codeword, &mut proof_stream);

        // reset sponge state to start verification
        proof_stream.sponge_state = H::init();
        let verdict = fri.verify(&mut proof_stream, &mut None);
        verdict.unwrap();
    }

    #[test]
    fn fri_x_field_limit_test() {
        type H = Tip5;

        let subgroup_order = 128;
        let expansion_factor = 4;
        let colinearity_check_count = 6;
        let fri: Fri<H> =
            get_x_field_fri_test_object(subgroup_order, expansion_factor, colinearity_check_count);
        let subgroup = fri.domain.generator.lift().get_cyclic_group_elements(None);

        let mut points: Vec<XFieldElement>;
        for n in [1, 5, 20, 30, 31] {
            points = subgroup.clone().iter().map(|p| p.mod_pow_u32(n)).collect();

            let mut proof_stream = ProofStream::new();
            fri.prove(&points, &mut proof_stream);

            // reset sponge state to start verification
            proof_stream.sponge_state = H::init();
            let verify_result = fri.verify(&mut proof_stream, &mut None);
            if verify_result.is_err() {
                panic!(
                    "There are {} points, |<128>^{n}| = {}, and verify_result = {verify_result:?}",
                    points.len(),
                    points.iter().unique().count(),
                );
            }

            // Reset proof stream's read items
            proof_stream.sponge_state = H::init();
            proof_stream.items_index = 0;

            // Manipulate Merkle root of 0 and verify failure with expected error message.
            // This uses the domain-specific knowledge that the first element on the proof stream
            // is the first Merkle root.
            let random_digest = thread_rng().gen();
            proof_stream.items[0] = ProofItem::MerkleRoot(random_digest);

            let bad_verify_result = fri.verify(&mut proof_stream, &mut None);
            assert!(bad_verify_result.is_err());
            println!("bad_verify_result = {bad_verify_result:?}");

            // TODO: Add negative test with bad Merkle authentication path
            // This probably requires manipulating the proof stream somehow.
        }

        // Negative test with too high degree
        let too_high = subgroup_order as u32 / expansion_factor as u32;
        points = subgroup.iter().map(|p| p.mod_pow_u32(too_high)).collect();
        let mut proof_stream = ProofStream::new();
        fri.prove(&points, &mut proof_stream);
        let verify_result = fri.verify(&mut proof_stream, &mut None);
        assert!(verify_result.is_err());
    }

    fn get_x_field_fri_test_object<H: AlgebraicHasher>(
        subgroup_order: u64,
        expansion_factor: usize,
        colinearity_checks: usize,
    ) -> Fri<H> {
        let offset = BFieldElement::generator();
        let fri: Fri<H> = Fri::new(
            offset,
            subgroup_order as usize,
            expansion_factor,
            colinearity_checks,
        );
        fri
    }

    #[test]
    fn test_fri_deserialization() {
        type H = Tip5;

        let subgroup_order = 64;
        let expansion_factor = 4;
        let colinearity_check_count = 2;
        let fri =
            get_x_field_fri_test_object(subgroup_order, expansion_factor, colinearity_check_count);

        let mut prover_proof_stream = ProofStream::new();

        let zero = XFieldElement::zero();
        let one = XFieldElement::one();
        let two = one + one;
        let poly = Polynomial::<XFieldElement>::new(vec![one, zero, zero, two]);
        let codeword = fri.domain.evaluate(&poly);

        fri.prove(&codeword, &mut prover_proof_stream);

        let proof = (&prover_proof_stream).into();
        let mut verifier_proof_stream: ProofStream<H> = ProofStream::try_from(&proof).unwrap();

        assert_eq!(prover_proof_stream.len(), verifier_proof_stream.len());
        for (prover_item, verifier_item) in prover_proof_stream
            .items
            .iter()
            .zip_eq(verifier_proof_stream.items.iter())
        {
            use ProofItem::*;
            match prover_item {
                MerkleRoot(prover_root) => {
                    assert_eq!(*prover_root, verifier_item.as_merkle_root().unwrap())
                }
                FriResponse(prover_response) => {
                    assert_eq!(*prover_response, verifier_item.as_fri_response().unwrap())
                }
                FriCodeword(prover_codeword) => {
                    assert_eq!(*prover_codeword, verifier_item.as_fri_codeword().unwrap())
                }
                _ => {
                    panic!(
                        "Did not recognize FRI proof items.\n\
                         Prover item:   {prover_item:?}\n\
                         Verifier item: {verifier_item:?}"
                    )
                }
            }
        }

        let verdict = fri.verify(&mut verifier_proof_stream, &mut None);
        verdict.unwrap();
    }
}
