use std::error::Error;
use std::fmt;
use std::marker::PhantomData;

use anyhow::bail;
use anyhow::Result;
use itertools::Itertools;
use num_traits::One;
use rayon::iter::*;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::other::log_2_ceil;
use twenty_first::shared_math::polynomial::Polynomial;
use twenty_first::shared_math::tip5::Digest;
use twenty_first::shared_math::traits::FiniteField;
use twenty_first::shared_math::traits::Inverse;
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
    LastRoundPolynomialHasTooHighDegree,
    BadMerkleRootForLastCodeword,
}

#[derive(Debug, Clone)]
pub struct Fri<H> {
    pub expansion_factor: usize,
    pub num_colinearity_checks: usize,
    pub domain: ArithmeticDomain,
    _hasher: PhantomData<H>,
}

impl<H: AlgebraicHasher> Fri<H> {
    pub fn new(
        domain: ArithmeticDomain,
        expansion_factor: usize,
        num_colinearity_checks: usize,
    ) -> Self {
        assert!(expansion_factor > 1);
        assert!(expansion_factor.is_power_of_two());
        assert!(domain.length >= expansion_factor);

        let _hasher = PhantomData;
        Self {
            domain,
            expansion_factor,
            num_colinearity_checks,
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
        debug_assert_eq!(self.domain.length, codeword.len());

        let (codewords, merkle_trees) = self.commit(codeword, proof_stream);
        debug_assert_eq!(codewords.len(), merkle_trees.len());

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
    ) -> (Vec<Vec<XFieldElement>>, Vec<MerkleTree<H>>) {
        let one = XFieldElement::one();
        let two_inverse = XFieldElement::from(2).inverse();

        let mut current_codeword = codeword.to_vec();
        let mut all_codewords = vec![];
        let mut all_merkle_trees = vec![];

        let digests = Self::codeword_as_digests(&current_codeword);
        let merkle_tree = MTMaker::from_digests(&digests);
        let merkle_root = ProofItem::MerkleRoot(merkle_tree.get_root());
        proof_stream.enqueue(&merkle_root);

        all_codewords.push(current_codeword.clone());
        all_merkle_trees.push(merkle_tree);

        let mut current_domain = self.domain;
        for _ in 0..self.num_rounds() {
            let folding_challenge = proof_stream.sample_scalars(1)[0];

            let domain_points = current_domain.domain_values();
            let domain_point_inverses = BFieldElement::batch_inversion(domain_points);

            let n = current_codeword.len();
            current_codeword = (0..n / 2)
                .into_par_iter()
                .map(|i| {
                    let scaled_offset_inv = folding_challenge * domain_point_inverses[i];
                    let left_summand = (one + scaled_offset_inv) * current_codeword[i];
                    let right_summand = (one - scaled_offset_inv) * current_codeword[n / 2 + i];
                    (left_summand + right_summand) * two_inverse
                })
                .collect();

            let digests = Self::codeword_as_digests(&current_codeword);
            let merkle_tree = MTMaker::from_digests(&digests);
            let merkle_root = ProofItem::MerkleRoot(merkle_tree.get_root());
            proof_stream.enqueue(&merkle_root);

            all_codewords.push(current_codeword.clone());
            all_merkle_trees.push(merkle_tree);

            current_domain.generator = current_domain.generator.square();
            current_domain.offset = current_domain.offset.square();
        }

        let last_codeword = ProofItem::FriCodeword(current_codeword);
        proof_stream.enqueue(&last_codeword);

        (all_codewords, all_merkle_trees)
    }

    fn codeword_as_digests(codeword: &[XFieldElement]) -> Vec<Digest> {
        codeword.par_iter().map(|&xfe| xfe.into()).collect()
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
        self.assert_last_round_polynomial_is_of_low_degree(&last_codeword)?;
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

    fn assert_last_round_polynomial_is_of_low_degree(
        &self,
        last_codeword: &[XFieldElement],
    ) -> Result<()> {
        // The domain's offset is irrelevant for the polynomial's degree.
        let last_round_domain = ArithmeticDomain::of_length(last_codeword.len());
        let last_round_polynomial = last_round_domain.interpolate(last_codeword);
        if last_round_polynomial.degree() > self.last_round_max_degree() as isize {
            bail!(FriValidationError::LastRoundPolynomialHasTooHighDegree);
        }
        Ok(())
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
mod tests {
    use std::cmp::max;
    use std::cmp::min;

    use itertools::Itertools;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use twenty_first::shared_math::b_field_element::BFieldElement;
    use twenty_first::shared_math::other::roundup_npo2;
    use twenty_first::shared_math::polynomial::Polynomial;
    use twenty_first::shared_math::tip5::Tip5;
    use twenty_first::shared_math::tip5::RATE;
    use twenty_first::util_types::algebraic_hasher::SpongeHasher;

    use ProofItem::*;

    use crate::shared_tests::*;

    use super::*;

    prop_compose! {
        fn arbitrary_element_to_absorb()(
            absorb_array in vec(arbitrary_bfield_element(), RATE)
        ) -> [BFieldElement; RATE] {
            absorb_array.try_into().unwrap()
        }
    }

    prop_compose! {
        fn arbitrary_fri()(
            fri in arbitrary_fri_supporting_degree(-1)
        ) -> Fri<Tip5> {
            fri
        }
    }

    prop_compose! {
        fn arbitrary_fri_supporting_degree(min_supported_degree: i64)(
            log_2_expansion_factor in 1_usize..=8
        )(
            log_2_expansion_factor in Just(log_2_expansion_factor),
            log_2_domain_length in log_2_expansion_factor..=18,
            num_colinearity_checks in 1_usize..=320,
            offset in arbitrary_bfield_element(),
        ) -> Fri<Tip5> {
            let expansion_factor = (1 << log_2_expansion_factor) as usize;
            let sampled_domain_length = (1 << log_2_domain_length) as usize;

            let min_domain_length = match min_supported_degree {
                d if d <= -1 => 0,
                _ => roundup_npo2(min_supported_degree as u64 + 1) as usize,
            };
            let min_expanded_domain_length = min_domain_length * expansion_factor;
            let domain_length = max(sampled_domain_length, min_expanded_domain_length);

            let fri_domain = ArithmeticDomain::of_length(domain_length).with_offset(offset);
            Fri::new(fri_domain, expansion_factor, num_colinearity_checks)
        }
    }

    prop_compose! {
        fn arbitrary_supported_polynomial(fri: &Fri::<Tip5>)(
            degree in -1_i64..=fri.first_round_max_degree() as i64,
        )(
            polynomial in arbitrary_polynomial_of_degree(degree),
        ) -> Polynomial<XFieldElement> {
            polynomial
        }
    }

    prop_compose! {
        fn arbitrary_unsupported_polynomial(fri: &Fri::<Tip5>)(
            degree in 1 + fri.first_round_max_degree()..2 * (1 + fri.first_round_max_degree()),
        )(
            polynomial in arbitrary_polynomial_of_degree(degree as i64),
        ) -> Polynomial<XFieldElement> {
            polynomial
        }
    }

    prop_compose! {
        fn arbitrary_matching_fri_and_polynomial_pair()(
            fri in arbitrary_fri(),
        )(
            polynomial in arbitrary_supported_polynomial(&fri),
            fri in Just(fri),
        ) -> (Fri::<Tip5>, Polynomial<XFieldElement>) {
            (fri, polynomial)
        }
    }

    prop_compose! {
        fn arbitrary_non_matching_fri_and_polynomial_pair()(
            fri in arbitrary_fri(),
        )(
            polynomial in arbitrary_unsupported_polynomial(&fri),
            fri in Just(fri),
        ) -> (Fri::<Tip5>, Polynomial<XFieldElement>) {
            (fri, polynomial)
        }
    }

    proptest! {
        #[test]
        fn sample_indices_test(
            fri in arbitrary_fri(),
            initial_absorb in arbitrary_element_to_absorb(),
        ) {
            let mut sponge_state = Tip5::init();
            Tip5::absorb(&mut sponge_state, &initial_absorb);

            // todo: Figure out by how much to oversample for the given parameters.
            let oversampling_summand = 1 << 13;
            let num_indices_to_sample = fri.num_colinearity_checks + oversampling_summand;
            let indices = Tip5::sample_indices(
                &mut sponge_state,
                fri.domain.length as u32,
                num_indices_to_sample,
            );
            let num_unique_indices = indices.iter().unique().count();

            let required_unique_indices = min(fri.domain.length, fri.num_colinearity_checks);
            prop_assert!(num_unique_indices >= required_unique_indices);
        }
    }

    proptest! {
        #[test]
        fn num_rounds_are_reasonable(fri in arbitrary_fri()) {
            let expected_last_round_max_degree = fri.first_round_max_degree() >> fri.num_rounds();
            prop_assert_eq!(expected_last_round_max_degree, fri.last_round_max_degree());
            if fri.num_rounds() > 0 {
                prop_assert!(fri.num_colinearity_checks <= expected_last_round_max_degree);
                prop_assert!(expected_last_round_max_degree < 2 * fri.num_colinearity_checks);
            }
        }
    }

    proptest! {
        #[test]
        fn prove_and_verify_low_degree_of_twice_cubing_plus_one(
            fri in arbitrary_fri_supporting_degree(3)
        ) {
            let coefficients = [1, 0, 0, 2].map(|c| c.into()).to_vec();
            let polynomial = Polynomial::new(coefficients);
            let codeword = fri.domain.evaluate(&polynomial);

            let mut proof_stream = ProofStream::new();
            fri.prove(&codeword, &mut proof_stream);

            // reset sponge state to start verification
            proof_stream.sponge_state = Tip5::init();
            let verdict = fri.verify(&mut proof_stream, &mut None);
            prop_assert!(verdict.is_ok());
        }
    }

    proptest! {
        #[test]
        fn prove_and_verify_low_degree_polynomial(
            (fri, polynomial) in arbitrary_matching_fri_and_polynomial_pair(),
        ) {
            debug_assert!(polynomial.degree() <= fri.first_round_max_degree() as isize);
            let codeword = fri.domain.evaluate(&polynomial);
            let mut proof_stream = ProofStream::new();
            fri.prove(&codeword, &mut proof_stream);

            // reset sponge state to start verification
            proof_stream.sponge_state = Tip5::init();
            let verdict = fri.verify(&mut proof_stream, &mut None);
            prop_assert!(verdict.is_ok());
        }
    }

    proptest! {
        #[test]
        fn prove_and_fail_to_verify_high_degree_polynomial(
            (fri, polynomial) in arbitrary_non_matching_fri_and_polynomial_pair(),
        ) {
            debug_assert!(polynomial.degree() > fri.first_round_max_degree() as isize);
            let codeword = fri.domain.evaluate(&polynomial);
            let mut proof_stream = ProofStream::new();
            fri.prove(&codeword, &mut proof_stream);

            // reset sponge state to start verification
            proof_stream.sponge_state = Tip5::init();
            let verdict = fri.verify(&mut proof_stream, &mut None);
            prop_assert!(verdict.is_err());
        }
    }

    #[test]
    fn smallest_possible_fri_has_no_rounds() {
        assert_eq!(0, smallest_fri().num_rounds());
    }

    #[test]
    fn smallest_possible_fri_can_only_verify_constant_polynomials() {
        assert_eq!(0, smallest_fri().first_round_max_degree());
    }

    fn smallest_fri() -> Fri<Tip5> {
        let domain = ArithmeticDomain::of_length(2);
        let expansion_factor = 2;
        let num_colinearity_checks = 1;
        Fri::new(domain, expansion_factor, num_colinearity_checks)
    }

    #[test]
    #[should_panic]
    fn too_small_expansion_factor_is_rejected() {
        let domain = ArithmeticDomain::of_length(2);
        let expansion_factor = 1;
        let num_colinearity_checks = 1;
        Fri::<Tip5>::new(domain, expansion_factor, num_colinearity_checks);
    }

    proptest! {
        #[test]
        #[should_panic]
        fn expansion_factor_not_a_power_of_two_is_rejected(
            expansion_factor in 2..=usize::MAX,
            offset in arbitrary_bfield_element(),
        ) {
            if expansion_factor.is_power_of_two() {
                return Ok(());
            }
            let domain = ArithmeticDomain::of_length(2 * expansion_factor).with_offset(offset);
            let num_colinearity_checks = 1;
            Fri::<Tip5>::new(
                domain,
                expansion_factor,
                num_colinearity_checks,
            );
        }
    }

    proptest! {
        #[test]
        #[should_panic]
        fn domain_size_smaller_than_expansion_factor_is_rejected(
            log_2_expansion_factor in 1..=8,
            offset in arbitrary_bfield_element(),
        ) {
            let expansion_factor = (1 << log_2_expansion_factor) as usize;
            let domain = ArithmeticDomain::of_length(expansion_factor - 1).with_offset(offset);
            let num_colinearity_checks = 1;
            Fri::<Tip5>::new(
                domain,
                expansion_factor,
                num_colinearity_checks,
            );
        }
    }

    // todo: add test fuzzing proof_stream

    proptest! {
        #[test]
        fn serialization(fri in arbitrary_fri_supporting_degree(3)) {
            let coefficients = [1, 0, 0, 2].map(|c| c.into()).to_vec();
            let polynomial = Polynomial::new(coefficients);
            let codeword = fri.domain.evaluate(&polynomial);

            let mut prover_proof_stream = ProofStream::new();
            fri.prove(&codeword, &mut prover_proof_stream);
            let proof = (&prover_proof_stream).into();
            let verifier_proof_stream = ProofStream::<Tip5>::try_from(&proof).unwrap();

            let prover_items = prover_proof_stream.items.iter();
            let verifier_items = verifier_proof_stream.items.iter();
            for (prover_item, verifier_item) in prover_items.zip_eq(verifier_items) {
                match (prover_item, verifier_item) {
                    (MerkleRoot(p), MerkleRoot(v)) => prop_assert_eq!(p, v),
                    (FriResponse(p), FriResponse(v)) => prop_assert_eq!(p, v),
                    (FriCodeword(p), FriCodeword(v)) => prop_assert_eq!(p, v),
                    _ => panic!(
                        "Unknown items.\nProver: {prover_item:?}\nVerifier: {verifier_item:?}"
                    ),
                }
            }
        }
    }
}
