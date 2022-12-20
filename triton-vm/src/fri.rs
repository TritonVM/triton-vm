use std::error::Error;
use std::fmt;
use std::marker::PhantomData;

use anyhow::Result;
use itertools::Itertools;
use num_traits::One;
use rayon::iter::*;
use triton_profiler::prof_start;
use triton_profiler::prof_stop;
use triton_profiler::triton_profiler::TritonProfiler;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::ntt::intt;
use twenty_first::shared_math::other::log_2_ceil;
use twenty_first::shared_math::other::log_2_floor;
use twenty_first::shared_math::polynomial::Polynomial;
use twenty_first::shared_math::rescue_prime_digest::Digest;
use twenty_first::shared_math::traits::CyclicGroupGenerator;
use twenty_first::shared_math::traits::FiniteField;
use twenty_first::shared_math::traits::ModPowU32;
use twenty_first::shared_math::x_field_element::XFieldElement;
use twenty_first::util_types::algebraic_hasher::AlgebraicHasher;
use twenty_first::util_types::algebraic_hasher::Hashable;
use twenty_first::util_types::merkle_tree::MerkleTree;
use twenty_first::util_types::merkle_tree::PartialAuthenticationPath;
use twenty_first::util_types::merkle_tree_maker::MerkleTreeMaker;

use crate::arithmetic_domain::ArithmeticDomain;
use crate::proof_item::FriResponse;
use crate::proof_item::ProofItem;
use crate::proof_stream::ProofStream;
use crate::stark::Maker;

impl Error for FriValidationError {}

impl fmt::Display for FriValidationError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "Deserialization error for LowDegreeProof: {:?}", self)
    }
}

#[derive(PartialEq, Eq, Debug)]
pub enum FriValidationError {
    BadMerkleAuthenticationPath,
    BadSizedProof,
    NonPostiveRoundCount,
    MismatchingLastCodeword,
    LastIterationTooHighDegree,
    BadMerkleRootForFirstCodeword,
    BadMerkleRootForLastCodeword,
}

#[derive(Debug, Clone)]
pub struct Fri<H> {
    // In STARK, the expansion factor <FRI domain length> / max_degree, where
    // `max_degree` is the max degree of any interpolation rounded up to the
    // nearest power of 2.
    pub expansion_factor: usize,
    pub colinearity_checks_count: usize,
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
        let domain = ArithmeticDomain::new(offset, domain_length);
        let _hasher = PhantomData;
        Self {
            domain,
            expansion_factor,
            colinearity_checks_count,
            _hasher,
        }
    }

    /// Build the (deduplicated) Merkle authentication paths for the codeword at the given indices
    /// and enqueue the corresponding values and (partial) authentication paths on the proof stream.
    fn enqueue_auth_pairs(
        indices: &[usize],
        codeword: &[XFieldElement],
        merkle_tree: &MerkleTree<H, Maker>,
        proof_stream: &mut ProofStream<ProofItem, H>,
    ) {
        let value_ap_pairs: Vec<(PartialAuthenticationPath<Digest>, XFieldElement)> = merkle_tree
            .get_authentication_structure(indices)
            .into_iter()
            .zip(indices.iter())
            .map(|(ap, i)| (ap, codeword[*i]))
            .collect_vec();
        proof_stream.enqueue(&ProofItem::FriResponse(FriResponse(value_ap_pairs)))
    }

    /// Given a set of `indices`, a merkle `root`, and the (correctly set) `proof_stream`, verify
    /// whether the values at the `indices` are members of the set committed to by the merkle `root`
    /// and return these values if they are. Fails otherwise.
    fn dequeue_and_authenticate(
        indices: &[usize],
        root: Digest,
        proof_stream: &mut ProofStream<ProofItem, H>,
    ) -> Result<Vec<XFieldElement>> {
        let fri_response = proof_stream.dequeue()?.as_fri_response()?;
        let dequeued_paths_and_leafs = fri_response.0;
        let paths = dequeued_paths_and_leafs.clone().into_iter().map(|(p, _)| p);
        let values: Vec<XFieldElement> = dequeued_paths_and_leafs
            .into_iter()
            .map(|(_, v)| v)
            .collect();
        let digests: Vec<Digest> = values.par_iter().map(H::hash).collect();
        let path_digest_pairs = paths.into_iter().zip(digests).collect_vec();
        if MerkleTree::<H, Maker>::verify_authentication_structure(
            root,
            indices,
            &path_digest_pairs,
        ) {
            Ok(values)
        } else {
            Err(anyhow::Error::new(
                FriValidationError::BadMerkleAuthenticationPath,
            ))
        }
    }

    /// Create a FRI proof and return chosen indices of round 0 and Merkle root of round 0 codeword
    pub fn prove(
        &self,
        codeword: &[XFieldElement],
        proof_stream: &mut ProofStream<ProofItem, H>,
    ) -> Result<(Vec<usize>, Digest)> {
        debug_assert_eq!(
            self.domain.length,
            codeword.len(),
            "Initial codeword length must match that set in FRI object"
        );

        // commit phase
        let (codewords, merkle_trees): (Vec<Vec<XFieldElement>>, Vec<MerkleTree<H, Maker>>) =
            self.commit(codeword, proof_stream)?.into_iter().unzip();

        // Fiat-Shamir to get indices
        let top_level_indices: Vec<usize> = self.sample_indices(&proof_stream.prover_fiat_shamir());

        // query phase
        // query step 0: enqueue authentication paths for all points `A` into proof stream
        let initial_a_indices: Vec<usize> = top_level_indices.clone();
        Self::enqueue_auth_pairs(&initial_a_indices, codeword, &merkle_trees[0], proof_stream);
        // query step 1: loop over FRI rounds, enqueue authentication paths for all points `B`
        let mut current_domain_len = self.domain.length;
        let mut b_indices: Vec<usize> = initial_a_indices;
        // the last codeword is transmitted to the verifier in the clear. Thus, no co-linearity
        // check is needed for the last codeword and we only have to look at the interval given here
        for r in 0..merkle_trees.len() - 1 {
            debug_assert_eq!(
                codewords[r].len(),
                current_domain_len,
                "The current domain length needs to be the same as the length of the \
                current codeword"
            );
            b_indices = b_indices
                .iter()
                .map(|x| (x + current_domain_len / 2) % current_domain_len)
                .collect();
            Self::enqueue_auth_pairs(&b_indices, &codewords[r], &merkle_trees[r], proof_stream);
            current_domain_len /= 2;
        }

        let merkle_root_of_1st_round: Digest = merkle_trees[0].get_root();
        Ok((top_level_indices, merkle_root_of_1st_round))
    }

    #[allow(clippy::type_complexity)]
    fn commit(
        &self,
        codeword: &[XFieldElement],
        proof_stream: &mut ProofStream<ProofItem, H>,
    ) -> Result<Vec<(Vec<XFieldElement>, MerkleTree<H, Maker>)>> {
        let mut subgroup_generator = self.domain.generator;
        let mut offset = self.domain.offset;
        let mut codeword_local = codeword.to_vec();

        let one: XFieldElement = XFieldElement::one();
        let two: XFieldElement = one + one;
        let two_inv = one / two;

        // Compute and send Merkle root
        let mut digests: Vec<Digest> = Vec::with_capacity(codeword_local.len());
        codeword_local
            .clone()
            .into_par_iter()
            .map(|xfe| H::hash(&xfe))
            .collect_into_vec(&mut digests);
        let mut mt: MerkleTree<H, Maker> = Maker::from_digests(&digests);
        let mut mt_root: Digest = mt.get_root();

        proof_stream.enqueue(&ProofItem::MerkleRoot(mt_root));
        let mut values_and_merkle_trees = vec![(codeword_local.clone(), mt)];

        let (num_rounds, _) = self.num_rounds();
        for _round in 0..num_rounds {
            let n = codeword_local.len();

            // Get challenge
            let challenge_digest = proof_stream.prover_fiat_shamir();
            let alpha: XFieldElement = XFieldElement::sample(&challenge_digest);

            let x_offset: Vec<XFieldElement> = subgroup_generator
                .get_cyclic_group_elements(None)
                .into_par_iter()
                .map(|x| (x * offset).lift())
                .collect();

            let x_offset_inverses = XFieldElement::batch_inversion(x_offset);
            codeword_local = (0..n / 2)
                .into_par_iter()
                .map(|i| {
                    two_inv
                        * ((one + alpha * x_offset_inverses[i]) * codeword_local[i]
                            + (one - alpha * x_offset_inverses[i]) * codeword_local[n / 2 + i])
                })
                .collect();

            // Compute and send Merkle root. We have to do that within this loops, since
            // the next round's alpha must be calculated from the previous round's Merkle root.
            codeword_local
                .clone()
                .into_par_iter()
                .map(|xfe| H::hash(&xfe))
                .collect_into_vec(&mut digests);

            mt = Maker::from_digests(&digests);
            mt_root = mt.get_root();
            proof_stream.enqueue(&ProofItem::MerkleRoot(mt_root));
            values_and_merkle_trees.push((codeword_local.clone(), mt));

            // Update subgroup generator and offset
            subgroup_generator = subgroup_generator * subgroup_generator;
            offset = offset * offset;
        }

        // Send the last codeword
        let last_codeword: Vec<XFieldElement> = codeword_local;
        proof_stream.enqueue(&ProofItem::FriCodeword(last_codeword));

        Ok(values_and_merkle_trees)
    }

    // Return the c-indices for the 1st round of FRI
    fn sample_indices(&self, seed: &Digest) -> Vec<usize> {
        // This algorithm starts with the inner-most indices to pick up
        // to `last_codeword_length` indices from the codeword in the last round.
        // It then calculates the indices in the subsequent rounds by choosing
        // between the two possible next indices in the next round until we get
        // the c-indices for the first round.
        let num_rounds = self.num_rounds().0;
        let last_codeword_length = self.domain.length >> num_rounds;
        assert!(
            self.colinearity_checks_count <= last_codeword_length,
            "Requested number of indices must not exceed length of last codeword"
        );

        let mut last_indices: Vec<usize> = vec![];
        let mut remaining_last_round_exponents: Vec<usize> = (0..last_codeword_length).collect();
        let mut counter = 0usize;
        for _ in 0..self.colinearity_checks_count {
            let mut seed_local = seed.to_sequence();
            seed_local.append(&mut counter.to_sequence());
            let digest: Digest = H::hash_slice(&seed_local);
            let index: usize =
                H::sample_index_not_power_of_two(&digest, remaining_last_round_exponents.len());
            last_indices.push(remaining_last_round_exponents.remove(index));
            counter += 1;
        }

        // Use last indices to derive first c-indices
        let mut indices = last_indices;
        for i in 1..num_rounds {
            let codeword_length = last_codeword_length << i;

            indices = indices
                .into_par_iter()
                .zip((counter..counter + self.colinearity_checks_count).into_par_iter())
                .map(|(index, _count)| {
                    let mut seed_local = seed.to_sequence();
                    seed_local.append(&mut counter.to_sequence());
                    let digest: Digest = H::hash_slice(&seed_local);
                    let reduce_modulo: bool = H::sample_index(&digest, 2) == 0;
                    if reduce_modulo {
                        index + codeword_length / 2
                    } else {
                        index
                    }
                })
                .collect();
        }

        indices
    }

    pub fn verify(
        &self,
        proof_stream: &mut ProofStream<ProofItem, H>,
        first_codeword_mt_root: &Digest,
        maybe_profiler: &mut Option<TritonProfiler>,
    ) -> Result<()> {
        prof_start!(maybe_profiler, "init");
        let (num_rounds, degree_of_last_round) = self.num_rounds();
        let num_rounds = num_rounds as usize;

        // Extract all roots and calculate alpha, the challenges
        let mut roots: Vec<Digest> = vec![];
        let mut alphas: Vec<XFieldElement> = vec![];

        let first_root: Digest = proof_stream.dequeue()?.as_merkle_root()?;
        if first_root != *first_codeword_mt_root {
            return Err(anyhow::Error::new(
                FriValidationError::BadMerkleRootForFirstCodeword,
            ));
        }

        roots.push(first_root);
        prof_stop!(maybe_profiler, "init");

        prof_start!(maybe_profiler, "roots and alpha");
        for _round in 0..num_rounds {
            // Get a challenge from the proof stream
            let challenge = proof_stream.verifier_fiat_shamir();
            let alpha: XFieldElement = XFieldElement::sample(&challenge);
            alphas.push(alpha);

            let root: Digest = proof_stream.dequeue()?.as_merkle_root()?;
            roots.push(root);
        }
        prof_stop!(maybe_profiler, "roots and alpha");

        prof_start!(maybe_profiler, "last codeword matches root");
        // Extract last codeword
        let last_codeword: Vec<XFieldElement> = proof_stream.dequeue()?.as_fri_codeword()?;

        // Check if last codeword matches the given root
        let codeword_digests = last_codeword.iter().map(H::hash).collect_vec();
        let last_codeword_mt: MerkleTree<H, Maker> = Maker::from_digests(&codeword_digests);
        let last_root = roots.last().unwrap();
        if *last_root != last_codeword_mt.get_root() {
            return Err(anyhow::Error::new(
                FriValidationError::BadMerkleRootForLastCodeword,
            ));
        }
        prof_stop!(maybe_profiler, "last codeword matches root");

        // Verify that last codeword is of sufficiently low degree

        prof_start!(maybe_profiler, "last codeword has low degree");
        // Compute interpolant to get the degree of the last codeword
        // Note that we don't have to scale the polynomial back to the
        // trace subgroup since we only check its degree and don't use
        // it further.
        let log_2_of_n = log_2_floor(last_codeword.len() as u128) as u32;
        let mut last_polynomial = last_codeword.clone();

        let last_fri_domain_generator = self
            .domain
            .generator
            .mod_pow_u32(2u32.pow(num_rounds as u32));
        intt::<XFieldElement>(&mut last_polynomial, last_fri_domain_generator, log_2_of_n);

        let last_poly_degree: isize = (Polynomial::<XFieldElement> {
            coefficients: last_polynomial,
        })
        .degree();

        if last_poly_degree > degree_of_last_round as isize {
            println!(
                "last_poly_degree is {}, degree_of_last_round is {}",
                last_poly_degree, degree_of_last_round
            );
            return Err(anyhow::Error::new(
                FriValidationError::LastIterationTooHighDegree,
            ));
        }
        prof_stop!(maybe_profiler, "last codeword has low degree");

        // Query phase
        prof_start!(maybe_profiler, "query phase");
        // query step 0: get "A" indices and verify set membership of corresponding values.
        prof_start!(maybe_profiler, "sample indices");
        let mut a_indices: Vec<usize> = self.sample_indices(&proof_stream.verifier_fiat_shamir());
        prof_stop!(maybe_profiler, "sample indices");
        prof_start!(maybe_profiler, "dequeue and authenticate");
        let mut a_values = Self::dequeue_and_authenticate(&a_indices, roots[0], proof_stream)?;
        prof_stop!(maybe_profiler, "dequeue and authenticate");

        // set up "B" for offsetting inside loop.  Note that "B" and "A" indices
        // can be calcuated from each other.
        let mut b_indices = a_indices.clone();
        let mut current_domain_len = self.domain.length;

        // query step 1:  loop over FRI rounds, verify "B"s, compute values for "C"s
        prof_start!(maybe_profiler, "loop");
        for r in 0..num_rounds {
            // get "B" indices and verify set membership of corresponding values
            b_indices = b_indices
                .iter()
                .map(|x| (x + current_domain_len / 2) % current_domain_len)
                .collect();
            let b_values = Self::dequeue_and_authenticate(&b_indices, roots[r], proof_stream)?;
            debug_assert_eq!(
                self.colinearity_checks_count,
                a_indices.len(),
                "There must be equally many 'a indices' as there are colinearity checks."
            );
            debug_assert_eq!(
                self.colinearity_checks_count,
                b_indices.len(),
                "There must be equally many 'b indices' as there are colinearity checks."
            );
            debug_assert_eq!(
                self.colinearity_checks_count,
                a_values.len(),
                "There must be equally many 'a values' as there are colinearity checks."
            );
            debug_assert_eq!(
                self.colinearity_checks_count,
                b_values.len(),
                "There must be equally many 'b values' as there are colinearity checks."
            );

            // compute "C" indices and values for next round from "A" and "B`"" of current round
            current_domain_len /= 2;
            let c_indices = a_indices.iter().map(|x| x % current_domain_len).collect();
            let c_values = (0..self.colinearity_checks_count)
                .into_par_iter()
                .map(|i| {
                    Polynomial::<XFieldElement>::get_colinear_y(
                        (self.get_evaluation_argument(a_indices[i], r), a_values[i]),
                        (self.get_evaluation_argument(b_indices[i], r), b_values[i]),
                        alphas[r],
                    )
                })
                .collect();

            // Notice that next rounds "A"s correspond to current rounds "C":
            a_indices = c_indices;
            a_values = c_values;
        }
        prof_stop!(maybe_profiler, "loop");
        prof_stop!(maybe_profiler, "query phase");

        // Finally compare "C" values (which are named "A" values in this
        // enclosing scope) with last codeword from the proofstream.
        prof_start!(maybe_profiler, "compare last codeword");
        a_indices = a_indices.iter().map(|x| x % current_domain_len).collect();
        if (0..self.colinearity_checks_count).any(|i| last_codeword[a_indices[i]] != a_values[i]) {
            return Err(anyhow::Error::new(
                FriValidationError::MismatchingLastCodeword,
            ));
        }
        prof_stop!(maybe_profiler, "compare last codeword");
        Ok(())
    }

    /// Given index `i` of the FRI codeword in round `round`, compute the corresponding value in the
    /// FRI (co-)domain. This corresponds to `ω^i` in `f(ω^i)` from
    /// [STARK-Anatomy](https://neptune.cash/learn/stark-anatomy/fri/#split-and-fold).
    fn get_evaluation_argument(&self, idx: usize, round: usize) -> XFieldElement {
        let domain_value = self.domain.offset * self.domain.generator.mod_pow_u32(idx as u32);
        let round_exponent = 2u32.pow(round as u32);
        let evaluation_argument = domain_value.mod_pow_u32(round_exponent);

        evaluation_argument.lift()
    }

    fn num_rounds(&self) -> (u8, u32) {
        let max_degree = (self.domain.length / self.expansion_factor) - 1;
        let mut rounds_count = log_2_ceil(max_degree as u128 + 1) as u8;
        let mut max_degree_of_last_round = 0u32;
        if self.expansion_factor < self.colinearity_checks_count {
            let num_missed_rounds = log_2_ceil(
                (self.colinearity_checks_count as f64 / self.expansion_factor as f64).ceil()
                    as u128,
            ) as u8;
            rounds_count -= num_missed_rounds;
            max_degree_of_last_round = 2u32.pow(num_missed_rounds as u32) - 1;
        }

        (rounds_count, max_degree_of_last_round)
    }
}

#[cfg(test)]
mod triton_xfri_tests {
    use itertools::Itertools;
    use num_traits::Zero;
    use rand::thread_rng;
    use rand::RngCore;
    use twenty_first::shared_math::b_field_element::BFieldElement;
    use twenty_first::shared_math::rescue_prime_regular::RescuePrimeRegular;
    use twenty_first::shared_math::traits::CyclicGroupGenerator;
    use twenty_first::shared_math::traits::ModPowU32;
    use twenty_first::shared_math::x_field_element::XFieldElement;
    use twenty_first::test_shared::corrupt_digest;
    use twenty_first::utils::has_unique_elements;

    use super::*;

    #[test]
    fn sample_indices_test() {
        type H = RescuePrimeRegular;
        let mut rng = thread_rng();

        let subgroup_order = 16;
        let expansion_factor = 4;
        let colinearity_checks = 16;
        let fri: Fri<H> =
            get_x_field_fri_test_object::<H>(subgroup_order, expansion_factor, colinearity_checks);
        let indices = fri.sample_indices(&H::hash(&BFieldElement::new(rng.next_u64())));
        assert!(
            has_unique_elements(indices.iter()),
            "Picked indices must be unique"
        );
    }

    #[test]
    fn get_rounds_count_test() {
        type Hasher = RescuePrimeRegular;

        let subgroup_order = 512;
        let expansion_factor = 4;
        let mut fri: Fri<Hasher> =
            get_x_field_fri_test_object::<Hasher>(subgroup_order, expansion_factor, 2);

        assert_eq!((7, 0), fri.num_rounds());
        fri.colinearity_checks_count = 8;
        assert_eq!((6, 1), fri.num_rounds());
        fri.colinearity_checks_count = 10;
        assert_eq!((5, 3), fri.num_rounds());
        fri.colinearity_checks_count = 16;
        assert_eq!((5, 3), fri.num_rounds());
        fri.colinearity_checks_count = 17;
        assert_eq!((4, 7), fri.num_rounds());
        fri.colinearity_checks_count = 18;
        assert_eq!((4, 7), fri.num_rounds());
        fri.colinearity_checks_count = 31;
        assert_eq!((4, 7), fri.num_rounds());
        fri.colinearity_checks_count = 32;
        assert_eq!((4, 7), fri.num_rounds());
        fri.colinearity_checks_count = 33;
        assert_eq!((3, 15), fri.num_rounds());

        fri.domain.length = 256;
        assert_eq!((2, 15), fri.num_rounds());
        fri.colinearity_checks_count = 32;
        assert_eq!((3, 7), fri.num_rounds());

        fri.colinearity_checks_count = 32;
        fri.domain.length = 1048576;
        fri.expansion_factor = 8;
        assert_eq!((15, 3), fri.num_rounds());

        fri.colinearity_checks_count = 33;
        fri.domain.length = 1048576;
        fri.expansion_factor = 8;
        assert_eq!((14, 7), fri.num_rounds());

        fri.colinearity_checks_count = 63;
        fri.domain.length = 1048576;
        fri.expansion_factor = 8;
        assert_eq!((14, 7), fri.num_rounds());

        fri.colinearity_checks_count = 64;
        fri.domain.length = 1048576;
        fri.expansion_factor = 8;
        assert_eq!((14, 7), fri.num_rounds());

        fri.colinearity_checks_count = 65;
        fri.domain.length = 1048576;
        fri.expansion_factor = 8;
        assert_eq!((13, 15), fri.num_rounds());

        fri.domain.length = 256;
        fri.expansion_factor = 4;
        fri.colinearity_checks_count = 17;
        assert_eq!((3, 7), fri.num_rounds());
    }

    #[test]
    fn fri_on_x_field_test() {
        type Hasher = RescuePrimeRegular;

        let subgroup_order = 1024;
        let expansion_factor = 4;
        let colinearity_check_count = 6;
        let fri: Fri<Hasher> =
            get_x_field_fri_test_object(subgroup_order, expansion_factor, colinearity_check_count);
        let mut proof_stream: ProofStream<ProofItem, Hasher> = ProofStream::new();
        let subgroup = fri.domain.generator.lift().get_cyclic_group_elements(None);

        let (_, merkle_root_of_round_0) = fri.prove(&subgroup, &mut proof_stream).unwrap();
        let verdict = fri.verify(&mut proof_stream, &merkle_root_of_round_0, &mut None);
        if let Err(e) = verdict {
            panic!("Found error: {}", e);
        }
    }

    #[test]
    fn prove_and_verify_low_degree_of_twice_cubing_plus_one() {
        type Hasher = RescuePrimeRegular;

        let subgroup_order = 1024;
        let expansion_factor = 4;
        let colinearity_check_count = 6;
        let fri: Fri<Hasher> =
            get_x_field_fri_test_object(subgroup_order, expansion_factor, colinearity_check_count);
        let mut proof_stream: ProofStream<ProofItem, Hasher> = ProofStream::new();

        let zero = XFieldElement::zero();
        let one = XFieldElement::one();
        let two = one + one;
        let poly = Polynomial::<XFieldElement>::new(vec![one, zero, zero, two]);
        let codeword = fri.domain.evaluate(&poly);

        let (_, merkle_root_of_round_0) = fri.prove(&codeword, &mut proof_stream).unwrap();
        let verdict = fri.verify(&mut proof_stream, &merkle_root_of_round_0, &mut None);
        if let Err(e) = verdict {
            panic!("Found error: {}", e);
        }
    }

    #[test]
    fn fri_x_field_limit_test() {
        type Hasher = RescuePrimeRegular;

        let subgroup_order = 128;
        let expansion_factor = 4;
        let colinearity_check_count = 6;
        let fri: Fri<Hasher> =
            get_x_field_fri_test_object(subgroup_order, expansion_factor, colinearity_check_count);
        let subgroup = fri.domain.generator.lift().get_cyclic_group_elements(None);

        let mut points: Vec<XFieldElement>;
        for n in [1, 5, 20, 30, 31] {
            points = subgroup.clone().iter().map(|p| p.mod_pow_u32(n)).collect();

            // TODO: Test elsewhere that proof_stream can be re-used for multiple .prove().
            let mut proof_stream: ProofStream<ProofItem, Hasher> = ProofStream::new();
            let (_, merkle_root_of_round_0) = fri.prove(&points, &mut proof_stream).unwrap();

            let verify_result = fri.verify(&mut proof_stream, &merkle_root_of_round_0, &mut None);
            if verify_result.is_err() {
                println!(
                    "There are {} points, |<128>^{}| = {}, and verify_result = {:?}",
                    points.len(),
                    n,
                    points.iter().unique().count(),
                    verify_result
                );
            }

            assert!(verify_result.is_ok());

            // Manipulate Merkle root of 0 and verify failure with expected error message
            proof_stream.reset_for_verifier();
            let bad_root_digest = corrupt_digest(&merkle_root_of_round_0);
            let bad_verify_result = fri.verify(&mut proof_stream, &bad_root_digest, &mut None);
            assert!(bad_verify_result.is_err());
            println!("bad_verify_result = {:?}", bad_verify_result);

            // TODO: Add negative test with bad Merkle authentication path
            // This probably requires manipulating the proof stream somehow.
        }

        // Negative test with too high degree
        let too_high = subgroup_order as u32 / expansion_factor as u32;
        points = subgroup.iter().map(|p| p.mod_pow_u32(too_high)).collect();
        let mut proof_stream: ProofStream<ProofItem, Hasher> = ProofStream::new();
        let (_, merkle_root_of_round_0) = fri.prove(&points, &mut proof_stream).unwrap();
        let verify_result = fri.verify(&mut proof_stream, &merkle_root_of_round_0, &mut None);
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
        type Hasher = RescuePrimeRegular;

        let subgroup_order = 64;
        let expansion_factor = 4;
        let colinearity_check_count = 2;
        let fri: Fri<Hasher> =
            get_x_field_fri_test_object(subgroup_order, expansion_factor, colinearity_check_count);
        let mut prover_proof_stream: ProofStream<ProofItem, Hasher> = ProofStream::new();

        let zero = XFieldElement::zero();
        let one = XFieldElement::one();
        let two = one + one;
        let poly = Polynomial::<XFieldElement>::new(vec![one, zero, zero, two]);
        let codeword = fri.domain.evaluate(&poly);

        let (_, merkle_root_of_round_0) = fri.prove(&codeword, &mut prover_proof_stream).unwrap();

        let proof = prover_proof_stream.to_proof();

        let mut verifier_proof_stream: ProofStream<ProofItem, Hasher> =
            ProofStream::from_proof(&proof).unwrap();

        for (left, right) in prover_proof_stream
            .items
            .iter()
            .zip_eq(verifier_proof_stream.items.iter())
        {
            if let ProofItem::MerkleRoot(left_root) = left {
                assert_eq!(*left_root, right.as_merkle_root().unwrap());
            } else if let ProofItem::FriResponse(left_response) = left {
                assert_eq!(*left_response, right.as_fri_response().unwrap());
            } else if let ProofItem::FriCodeword(left_codeword) = left {
                assert_eq!(*left_codeword, right.as_fri_codeword().unwrap());
            } else {
                panic!("did not recognize FRI proof item");
            }
        }

        let verdict = fri.verify(
            &mut verifier_proof_stream,
            &merkle_root_of_round_0,
            &mut None,
        );
        if let Err(e) = verdict {
            panic!("Found error: {}", e);
        }
    }
}
