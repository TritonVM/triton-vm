use std::marker::PhantomData;

use itertools::Itertools;
use num_traits::One;
use rayon::iter::*;
use twenty_first::prelude::*;
use twenty_first::shared_math::other::log_2_ceil;
use twenty_first::shared_math::traits::FiniteField;

use crate::arithmetic_domain::ArithmeticDomain;
use crate::error::FriProvingError;
use crate::error::FriSetupError;
use crate::error::FriValidationError;
use crate::error::FriValidationError::*;
use crate::profiler::prof_start;
use crate::profiler::prof_stop;
use crate::profiler::TritonProfiler;
use crate::proof_item::FriResponse;
use crate::proof_item::ProofItem;
use crate::proof_stream::ProofStream;
use crate::stark::MTMaker;

pub(crate) type SetupResult<T> = Result<T, FriSetupError>;
pub(crate) type ProverResult<T> = Result<T, FriProvingError>;
pub(crate) type VerifierResult<T> = Result<T, FriValidationError>;

pub type AuthenticationStructure = Vec<Digest>;

#[derive(Debug, Copy, Clone)]
pub struct Fri<H: AlgebraicHasher> {
    pub expansion_factor: usize,
    pub num_collinearity_checks: usize,
    pub domain: ArithmeticDomain,
    _hasher: PhantomData<H>,
}

struct FriProver<'stream, H: AlgebraicHasher> {
    proof_stream: &'stream mut ProofStream<H>,
    rounds: Vec<ProverRound<H>>,
    first_round_domain: ArithmeticDomain,
    num_rounds: usize,
    num_collinearity_checks: usize,
    first_round_collinearity_check_indices: Vec<usize>,
}

struct ProverRound<H: AlgebraicHasher> {
    domain: ArithmeticDomain,
    codeword: Vec<XFieldElement>,
    merkle_tree: MerkleTree<H>,
}

impl<'stream, H: AlgebraicHasher> FriProver<'stream, H> {
    fn commit(&mut self, codeword: &[XFieldElement]) -> ProverResult<()> {
        self.commit_to_first_round(codeword)?;
        for _ in 0..self.num_rounds {
            self.commit_to_next_round()?;
        }
        self.send_last_codeword();
        Ok(())
    }

    fn commit_to_first_round(&mut self, codeword: &[XFieldElement]) -> ProverResult<()> {
        let first_round = ProverRound::new(self.first_round_domain, codeword)?;
        self.commit_to_round(&first_round);
        self.store_round(first_round);
        Ok(())
    }

    fn commit_to_next_round(&mut self) -> ProverResult<()> {
        let next_round = self.construct_next_round()?;
        self.commit_to_round(&next_round);
        self.store_round(next_round);
        Ok(())
    }

    fn commit_to_round(&mut self, round: &ProverRound<H>) {
        let merkle_root = round.merkle_tree.root();
        let proof_item = ProofItem::MerkleRoot(merkle_root);
        self.proof_stream.enqueue(proof_item);
    }

    fn store_round(&mut self, round: ProverRound<H>) {
        self.rounds.push(round);
    }

    fn construct_next_round(&mut self) -> ProverResult<ProverRound<H>> {
        let previous_round = self.rounds.last().unwrap();
        let folding_challenge = self.proof_stream.sample_scalars(1)[0];
        let codeword = previous_round.split_and_fold(folding_challenge);
        let domain = previous_round.domain.halve();
        ProverRound::new(domain, &codeword)
    }

    fn send_last_codeword(&mut self) {
        let last_codeword = self.rounds.last().unwrap().codeword.clone();
        let proof_item = ProofItem::FriCodeword(last_codeword);
        self.proof_stream.enqueue(proof_item);
    }

    fn query(&mut self) -> ProverResult<()> {
        self.sample_first_round_collinearity_check_indices();

        let initial_a_indices = self.first_round_collinearity_check_indices.clone();
        self.authentically_reveal_codeword_of_round_at_indices(0, &initial_a_indices)?;

        let num_rounds_that_have_a_next_round = self.rounds.len() - 1;
        for round_number in 0..num_rounds_that_have_a_next_round {
            let b_indices = self.collinearity_check_b_indices_for_round(round_number);
            self.authentically_reveal_codeword_of_round_at_indices(round_number, &b_indices)?;
        }

        Ok(())
    }

    fn sample_first_round_collinearity_check_indices(&mut self) {
        let indices_upper_bound = self.first_round_domain.length;
        self.first_round_collinearity_check_indices = self
            .proof_stream
            .sample_indices(indices_upper_bound, self.num_collinearity_checks);
    }

    fn all_top_level_collinearity_check_indices(&self) -> Vec<usize> {
        let a_indices = self.first_round_collinearity_check_indices.clone();
        let b_indices = self.collinearity_check_b_indices_for_round(0);
        a_indices.into_iter().chain(b_indices).collect()
    }

    fn collinearity_check_b_indices_for_round(&self, round_number: usize) -> Vec<usize> {
        let domain_length = self.rounds[round_number].domain.length;
        self.first_round_collinearity_check_indices
            .iter()
            .map(|&a_index| (a_index + domain_length / 2) % domain_length)
            .collect()
    }

    fn authentically_reveal_codeword_of_round_at_indices(
        &mut self,
        round_number: usize,
        indices: &[usize],
    ) -> ProverResult<()> {
        let codeword = &self.rounds[round_number].codeword;
        let revealed_leaves = indices.iter().map(|&i| codeword[i]).collect_vec();

        let merkle_tree = &self.rounds[round_number].merkle_tree;
        let auth_structure = merkle_tree.authentication_structure(indices)?;

        let fri_response = FriResponse {
            auth_structure,
            revealed_leaves,
        };
        let proof_item = ProofItem::FriResponse(fri_response);
        self.proof_stream.enqueue(proof_item);
        Ok(())
    }
}

impl<H: AlgebraicHasher> ProverRound<H> {
    fn new(domain: ArithmeticDomain, codeword: &[XFieldElement]) -> ProverResult<Self> {
        debug_assert_eq!(domain.length, codeword.len());
        let merkle_tree = Self::merkle_tree_from_codeword(codeword)?;
        let round = Self {
            domain,
            codeword: codeword.to_vec(),
            merkle_tree,
        };
        Ok(round)
    }

    fn merkle_tree_from_codeword(codeword: &[XFieldElement]) -> ProverResult<MerkleTree<H>> {
        let digests = codeword_as_digests(codeword);
        MTMaker::from_digests(&digests).map_err(FriProvingError::MerkleTreeError)
    }

    fn split_and_fold(&self, folding_challenge: XFieldElement) -> Vec<XFieldElement> {
        let one = XFieldElement::one();
        let two_inverse = XFieldElement::from(2).inverse();

        let domain_points = self.domain.domain_values();
        let domain_point_inverses = BFieldElement::batch_inversion(domain_points);

        let n = self.codeword.len();
        (0..n / 2)
            .into_par_iter()
            .map(|i| {
                let scaled_offset_inv = folding_challenge * domain_point_inverses[i];
                let left_summand = (one + scaled_offset_inv) * self.codeword[i];
                let right_summand = (one - scaled_offset_inv) * self.codeword[n / 2 + i];
                (left_summand + right_summand) * two_inverse
            })
            .collect()
    }
}

struct FriVerifier<'stream, H: AlgebraicHasher> {
    proof_stream: &'stream mut ProofStream<H>,
    rounds: Vec<VerifierRound>,
    first_round_domain: ArithmeticDomain,
    last_round_codeword: Vec<XFieldElement>,
    last_round_max_degree: usize,
    num_rounds: usize,
    num_collinearity_checks: usize,
    first_round_collinearity_check_indices: Vec<usize>,
}

struct VerifierRound {
    domain: ArithmeticDomain,
    partial_codeword_a: Vec<XFieldElement>,
    partial_codeword_b: Vec<XFieldElement>,
    merkle_root: Digest,
    folding_challenge: Option<XFieldElement>,
}

impl<'stream, H: AlgebraicHasher> FriVerifier<'stream, H> {
    fn initialize(&mut self) -> VerifierResult<()> {
        self.initialize_verification_rounds()?;
        self.receive_last_round_codeword()
    }

    fn initialize_verification_rounds(&mut self) -> VerifierResult<()> {
        self.initialize_first_round()?;
        for _ in 0..self.num_rounds {
            self.initialize_next_round()?;
        }
        Ok(())
    }

    fn initialize_first_round(&mut self) -> VerifierResult<()> {
        let first_round = self.construct_first_round()?;
        self.store_round(first_round);
        Ok(())
    }

    fn initialize_next_round(&mut self) -> VerifierResult<()> {
        let next_round = self.construct_next_round()?;
        self.store_round(next_round);
        Ok(())
    }

    fn construct_first_round(&mut self) -> VerifierResult<VerifierRound> {
        let domain = self.first_round_domain;
        self.construct_round_with_domain(domain)
    }

    fn construct_next_round(&mut self) -> VerifierResult<VerifierRound> {
        let previous_round = self.rounds.last().unwrap();
        let domain = previous_round.domain.halve();
        self.construct_round_with_domain(domain)
    }

    fn construct_round_with_domain(
        &mut self,
        domain: ArithmeticDomain,
    ) -> VerifierResult<VerifierRound> {
        let merkle_root = self.proof_stream.dequeue()?.try_into_merkle_root()?;
        let folding_challenge = self.maybe_sample_folding_challenge();

        let verifier_round = VerifierRound {
            domain,
            partial_codeword_a: vec![],
            partial_codeword_b: vec![],
            merkle_root,
            folding_challenge,
        };
        Ok(verifier_round)
    }

    fn maybe_sample_folding_challenge(&mut self) -> Option<XFieldElement> {
        match self.need_more_folding_challenges() {
            true => Some(self.proof_stream.sample_scalars(1)[0]),
            false => None,
        }
    }

    fn need_more_folding_challenges(&self) -> bool {
        if self.num_rounds == 0 {
            return false;
        }

        let num_initialized_rounds = self.rounds.len();
        let num_rounds_that_have_a_next_round = self.num_rounds - 1;
        num_initialized_rounds <= num_rounds_that_have_a_next_round
    }

    fn store_round(&mut self, round: VerifierRound) {
        self.rounds.push(round);
    }

    fn receive_last_round_codeword(&mut self) -> VerifierResult<()> {
        self.last_round_codeword = self.proof_stream.dequeue()?.try_into_fri_codeword()?;
        Ok(())
    }

    fn compute_last_round_folded_partial_codeword(&mut self) -> VerifierResult<()> {
        self.sample_first_round_collinearity_check_indices();
        self.receive_authentic_partially_revealed_codewords()?;
        self.successively_fold_partial_codeword_of_each_round();
        Ok(())
    }

    fn sample_first_round_collinearity_check_indices(&mut self) {
        let upper_bound = self.first_round_domain.length;
        self.first_round_collinearity_check_indices = self
            .proof_stream
            .sample_indices(upper_bound, self.num_collinearity_checks);
    }

    fn receive_authentic_partially_revealed_codewords(&mut self) -> VerifierResult<()> {
        let auth_structure = self.receive_partial_codeword_a_for_first_round()?;
        self.authenticate_partial_codeword_a_for_first_round(auth_structure)?;

        let num_rounds_that_have_a_next_round = self.rounds.len() - 1;
        for round_number in 0..num_rounds_that_have_a_next_round {
            let auth_structure = self.receive_partial_codeword_b_for_round(round_number)?;
            self.authenticate_partial_codeword_b_for_round(round_number, auth_structure)?;
        }
        Ok(())
    }

    fn receive_partial_codeword_a_for_first_round(
        &mut self,
    ) -> VerifierResult<AuthenticationStructure> {
        let fri_response = self.proof_stream.dequeue()?.try_into_fri_response()?;
        let FriResponse {
            auth_structure,
            revealed_leaves,
        } = fri_response;

        self.assert_enough_leaves_were_received(&revealed_leaves)?;
        self.rounds[0].partial_codeword_a = revealed_leaves;
        Ok(auth_structure)
    }

    fn receive_partial_codeword_b_for_round(
        &mut self,
        round_number: usize,
    ) -> VerifierResult<AuthenticationStructure> {
        let fri_response = self.proof_stream.dequeue()?.try_into_fri_response()?;
        let FriResponse {
            auth_structure,
            revealed_leaves,
        } = fri_response;

        self.assert_enough_leaves_were_received(&revealed_leaves)?;
        self.rounds[round_number].partial_codeword_b = revealed_leaves;
        Ok(auth_structure)
    }

    fn assert_enough_leaves_were_received(&self, leaves: &[XFieldElement]) -> VerifierResult<()> {
        match self.num_collinearity_checks == leaves.len() {
            true => Ok(()),
            false => Err(IncorrectNumberOfRevealedLeaves),
        }
    }

    fn authenticate_partial_codeword_a_for_first_round(
        &self,
        authentication_structure: AuthenticationStructure,
    ) -> VerifierResult<()> {
        let round = &self.rounds[0];
        let revealed_leaves = &round.partial_codeword_a;
        let revealed_digests = codeword_as_digests(revealed_leaves);

        let leaf_indices = self.collinearity_check_a_indices_for_round(0);
        let indexed_leaves = leaf_indices.into_iter().zip_eq(revealed_digests).collect();

        let inclusion_proof = MerkleTreeInclusionProof::<H> {
            tree_height: round.merkle_tree_height(),
            indexed_leaves,
            authentication_structure,
            ..Default::default()
        };
        match inclusion_proof.verify(round.merkle_root) {
            true => Ok(()),
            false => Err(BadMerkleAuthenticationPath),
        }
    }

    fn authenticate_partial_codeword_b_for_round(
        &self,
        round_number: usize,
        authentication_structure: AuthenticationStructure,
    ) -> VerifierResult<()> {
        let round = &self.rounds[round_number];
        let revealed_leaves = &round.partial_codeword_b;
        let revealed_digests = codeword_as_digests(revealed_leaves);

        let leaf_indices = self.collinearity_check_b_indices_for_round(round_number);
        let indexed_leaves = leaf_indices.into_iter().zip_eq(revealed_digests).collect();

        let inclusion_proof = MerkleTreeInclusionProof::<H> {
            tree_height: round.merkle_tree_height(),
            indexed_leaves,
            authentication_structure,
            ..Default::default()
        };
        match inclusion_proof.verify(round.merkle_root) {
            true => Ok(()),
            false => Err(BadMerkleAuthenticationPath),
        }
    }

    fn successively_fold_partial_codeword_of_each_round(&mut self) {
        let num_rounds_that_have_a_next_round = self.rounds.len() - 1;
        for round_number in 0..num_rounds_that_have_a_next_round {
            let folded_partial_codeword = self.fold_partial_codeword_of_round(round_number);
            let next_round = &mut self.rounds[round_number + 1];
            next_round.partial_codeword_a = folded_partial_codeword;
        }
    }

    fn fold_partial_codeword_of_round(&self, round_number: usize) -> Vec<XFieldElement> {
        let round = &self.rounds[round_number];
        let a_indices = self.collinearity_check_a_indices_for_round(round_number);
        let b_indices = self.collinearity_check_b_indices_for_round(round_number);
        let partial_codeword_a = &round.partial_codeword_a;
        let partial_codeword_b = &round.partial_codeword_b;
        let domain = round.domain;
        let folding_challenge = round.folding_challenge.unwrap();

        (0..self.num_collinearity_checks)
            .into_par_iter()
            .map(|i| {
                let point_a_x = domain.domain_value(a_indices[i] as u32).lift();
                let point_b_x = domain.domain_value(b_indices[i] as u32).lift();
                let point_a = (point_a_x, partial_codeword_a[i]);
                let point_b = (point_b_x, partial_codeword_b[i]);
                Polynomial::get_colinear_y(point_a, point_b, folding_challenge)
            })
            .collect()
    }

    fn collinearity_check_a_indices_for_round(&self, round_number: usize) -> Vec<usize> {
        let domain_length = self.rounds[round_number].domain.length;
        let a_offset = 0;
        self.collinearity_check_indices_with_offset_and_modulus(a_offset, domain_length)
    }

    fn collinearity_check_b_indices_for_round(&self, round_number: usize) -> Vec<usize> {
        let domain_length = self.rounds[round_number].domain.length;
        let b_offset = domain_length / 2;
        self.collinearity_check_indices_with_offset_and_modulus(b_offset, domain_length)
    }

    fn collinearity_check_indices_with_offset_and_modulus(
        &self,
        offset: usize,
        modulus: usize,
    ) -> Vec<usize> {
        self.first_round_collinearity_check_indices
            .iter()
            .map(|&i| (i + offset) % modulus)
            .collect()
    }

    fn authenticate_last_round_codeword(&self) -> VerifierResult<()> {
        self.assert_last_round_codeword_matches_last_round_commitment()?;
        self.assert_last_round_codeword_agrees_with_last_round_folded_codeword()?;
        self.assert_last_round_codeword_corresponds_to_low_degree_polynomial()
    }

    fn assert_last_round_codeword_matches_last_round_commitment(&self) -> VerifierResult<()> {
        match self.last_round_merkle_root() == self.last_round_codeword_merkle_root()? {
            true => Ok(()),
            false => Err(BadMerkleRootForLastCodeword),
        }
    }

    fn last_round_codeword_merkle_root(&self) -> VerifierResult<Digest> {
        let codeword_digests = codeword_as_digests(&self.last_round_codeword);
        let merkle_tree: MerkleTree<H> = MTMaker::from_digests(&codeword_digests)
            .map_err(FriValidationError::MerkleTreeError)?;

        Ok(merkle_tree.root())
    }

    fn last_round_merkle_root(&self) -> Digest {
        self.rounds.last().unwrap().merkle_root
    }

    fn assert_last_round_codeword_agrees_with_last_round_folded_codeword(
        &self,
    ) -> VerifierResult<()> {
        let partial_folded_codeword = self.folded_last_round_codeword_at_indices_a();
        let partial_received_codeword = self.received_last_round_codeword_at_indices_a();
        match partial_received_codeword == partial_folded_codeword {
            true => Ok(()),
            false => Err(LastCodewordMismatch),
        }
    }

    fn folded_last_round_codeword_at_indices_a(&self) -> &[XFieldElement] {
        &self.rounds.last().unwrap().partial_codeword_a
    }

    fn received_last_round_codeword_at_indices_a(&self) -> Vec<XFieldElement> {
        let last_round_number = self.rounds.len() - 1;
        let last_round_indices_a = self.collinearity_check_a_indices_for_round(last_round_number);
        last_round_indices_a
            .iter()
            .map(|&last_round_index_a| self.last_round_codeword[last_round_index_a])
            .collect()
    }

    fn assert_last_round_codeword_corresponds_to_low_degree_polynomial(
        &self,
    ) -> VerifierResult<()> {
        if self.last_round_polynomial().degree() > self.last_round_max_degree as isize {
            return Err(LastRoundPolynomialHasTooHighDegree);
        }
        Ok(())
    }

    fn last_round_polynomial(&self) -> Polynomial<XFieldElement> {
        let domain = self.rounds.last().unwrap().domain;
        domain.interpolate(&self.last_round_codeword)
    }

    fn first_round_partially_revealed_codeword(&self) -> Vec<(usize, XFieldElement)> {
        let partial_codeword_a = self.rounds[0].partial_codeword_a.clone();
        let partial_codeword_b = self.rounds[0].partial_codeword_b.clone();

        let indices_a = self.collinearity_check_a_indices_for_round(0).into_iter();

        let first_round_codeword_b_has_been_revealed = self.num_rounds > 0;
        let indices_b = match first_round_codeword_b_has_been_revealed {
            true => self.collinearity_check_b_indices_for_round(0).into_iter(),
            false => vec![].into_iter(),
        };

        let codeword_a = indices_a.zip_eq(partial_codeword_a);
        let codeword_b = indices_b.zip_eq(partial_codeword_b);

        codeword_a.chain(codeword_b).collect()
    }
}

impl VerifierRound {
    fn merkle_tree_height(&self) -> usize {
        self.domain.length.ilog2() as usize
    }
}

impl<H: AlgebraicHasher> Fri<H> {
    pub fn new(
        domain: ArithmeticDomain,
        expansion_factor: usize,
        num_collinearity_checks: usize,
    ) -> SetupResult<Self> {
        match expansion_factor {
            ef if ef <= 1 => return Err(FriSetupError::ExpansionFactorTooSmall),
            ef if !ef.is_power_of_two() => return Err(FriSetupError::ExpansionFactorUnsupported),
            ef if ef > domain.length => return Err(FriSetupError::ExpansionFactorMismatch),
            _ => (),
        };

        let fri = Self {
            domain,
            expansion_factor,
            num_collinearity_checks,
            _hasher: PhantomData,
        };
        Ok(fri)
    }

    /// Create a FRI proof and return indices of revealed elements of round 0.
    pub fn prove(
        &self,
        codeword: &[XFieldElement],
        proof_stream: &mut ProofStream<H>,
    ) -> ProverResult<Vec<usize>> {
        let mut prover = self.prover(proof_stream);

        prover.commit(codeword)?;
        prover.query()?;

        let indices = prover.all_top_level_collinearity_check_indices();
        Ok(indices)
    }

    fn prover<'stream>(&'stream self, proof_stream: &'stream mut ProofStream<H>) -> FriProver<H> {
        FriProver {
            proof_stream,
            rounds: vec![],
            first_round_domain: self.domain,
            num_rounds: self.num_rounds(),
            num_collinearity_checks: self.num_collinearity_checks,
            first_round_collinearity_check_indices: vec![],
        }
    }

    /// Verify low-degreeness of the polynomial on the proof stream.
    /// Returns the indices and revealed elements of the codeword at the top level of the FRI proof.
    pub fn verify(
        &self,
        proof_stream: &mut ProofStream<H>,
        maybe_profiler: &mut Option<TritonProfiler>,
    ) -> VerifierResult<Vec<(usize, XFieldElement)>> {
        prof_start!(maybe_profiler, "init");
        let mut verifier = self.verifier(proof_stream);
        verifier.initialize()?;
        prof_stop!(maybe_profiler, "init");

        prof_start!(maybe_profiler, "fold all rounds");
        verifier.compute_last_round_folded_partial_codeword()?;
        prof_stop!(maybe_profiler, "fold all rounds");

        prof_start!(maybe_profiler, "authenticate last round codeword");
        verifier.authenticate_last_round_codeword()?;
        prof_stop!(maybe_profiler, "authenticate last round codeword");

        Ok(verifier.first_round_partially_revealed_codeword())
    }

    fn verifier<'stream>(
        &'stream self,
        proof_stream: &'stream mut ProofStream<H>,
    ) -> FriVerifier<H> {
        FriVerifier {
            proof_stream,
            rounds: vec![],
            first_round_domain: self.domain,
            last_round_codeword: vec![],
            last_round_max_degree: self.last_round_max_degree(),
            num_rounds: self.num_rounds(),
            num_collinearity_checks: self.num_collinearity_checks,
            first_round_collinearity_check_indices: vec![],
        }
    }

    pub fn num_rounds(&self) -> usize {
        let first_round_code_dimension = self.first_round_max_degree() + 1;
        let max_num_rounds = log_2_ceil(first_round_code_dimension as u128);

        // Skip rounds for which Merkle tree verification cost exceeds arithmetic cost,
        // because more than half the codeword's locations are queried.
        let num_rounds_checking_all_locations = self.num_collinearity_checks.ilog2() as u64;
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

fn codeword_as_digests(codeword: &[XFieldElement]) -> Vec<Digest> {
    codeword.par_iter().map(|&xfe| xfe.into()).collect()
}

#[cfg(test)]
mod tests {
    use std::cmp::max;
    use std::cmp::min;

    use assert2::assert;
    use assert2::let_assert;
    use itertools::Itertools;
    use proptest::prelude::*;
    use proptest_arbitrary_interop::arb;
    use rand::prelude::*;
    use rand_core::SeedableRng;
    use test_strategy::proptest;

    use ProofItem::*;

    use crate::error::FriValidationError;
    use crate::shared_tests::*;

    use super::*;

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
            num_collinearity_checks in 1_usize..=320,
            offset in arb(),
        ) -> Fri<Tip5> {
            let expansion_factor = (1 << log_2_expansion_factor) as usize;
            let sampled_domain_length = (1 << log_2_domain_length) as usize;

            let min_domain_length = match min_supported_degree {
                d if d <= -1 => 0,
                _ => (min_supported_degree as u64 + 1).next_power_of_two() as usize,
            };
            let min_expanded_domain_length = min_domain_length * expansion_factor;
            let domain_length = max(sampled_domain_length, min_expanded_domain_length);

            let fri_domain = ArithmeticDomain::of_length(domain_length).with_offset(offset);
            Fri::new(fri_domain, expansion_factor, num_collinearity_checks).unwrap()
        }
    }

    #[proptest]
    fn sample_indices(
        #[strategy(arbitrary_fri())] fri: Fri<Tip5>,
        #[strategy(arb())] initial_absorb: [BFieldElement; tip5::RATE],
    ) {
        let mut sponge_state = Tip5::init();
        Tip5::absorb(&mut sponge_state, initial_absorb);

        // todo: Figure out by how much to oversample for the given parameters.
        let oversampling_summand = 1 << 13;
        let num_indices_to_sample = fri.num_collinearity_checks + oversampling_summand;
        let indices = Tip5::sample_indices(
            &mut sponge_state,
            fri.domain.length as u32,
            num_indices_to_sample,
        );
        let num_unique_indices = indices.iter().unique().count();

        let required_unique_indices = min(fri.domain.length, fri.num_collinearity_checks);
        prop_assert!(num_unique_indices >= required_unique_indices);
    }

    #[proptest]
    fn num_rounds_are_reasonable(#[strategy(arbitrary_fri())] fri: Fri<Tip5>) {
        let expected_last_round_max_degree = fri.first_round_max_degree() >> fri.num_rounds();
        prop_assert_eq!(expected_last_round_max_degree, fri.last_round_max_degree());
        if fri.num_rounds() > 0 {
            prop_assert!(fri.num_collinearity_checks <= expected_last_round_max_degree);
            prop_assert!(expected_last_round_max_degree < 2 * fri.num_collinearity_checks);
        }
    }

    #[proptest(cases = 20)]
    fn prove_and_verify_low_degree_of_twice_cubing_plus_one(
        #[strategy(arbitrary_fri_supporting_degree(3))] fri: Fri<Tip5>,
    ) {
        let coefficients = [1, 0, 0, 2].map(|c| c.into()).to_vec();
        let polynomial = Polynomial::new(coefficients);
        let codeword = fri.domain.evaluate(&polynomial);

        let mut proof_stream = ProofStream::new();
        fri.prove(&codeword, &mut proof_stream).unwrap();

        let mut proof_stream = prepare_proof_stream_for_verification(proof_stream);
        let verdict = fri.verify(&mut proof_stream, &mut None);
        prop_assert!(verdict.is_ok());
    }

    #[proptest(cases = 50)]
    fn prove_and_verify_low_degree_polynomial(
        #[strategy(arbitrary_fri())] fri: Fri<Tip5>,
        #[strategy(-1_i64..=#fri.first_round_max_degree() as i64)] _degree: i64,
        #[strategy(arbitrary_polynomial_of_degree(#_degree))] polynomial: Polynomial<XFieldElement>,
    ) {
        debug_assert!(polynomial.degree() <= fri.first_round_max_degree() as isize);
        let codeword = fri.domain.evaluate(&polynomial);
        let mut proof_stream = ProofStream::new();
        fri.prove(&codeword, &mut proof_stream).unwrap();

        let mut proof_stream = prepare_proof_stream_for_verification(proof_stream);
        let verdict = fri.verify(&mut proof_stream, &mut None);
        prop_assert!(verdict.is_ok());
    }

    #[proptest(cases = 50)]
    fn prove_and_fail_to_verify_high_degree_polynomial(
        #[strategy(arbitrary_fri())] fri: Fri<Tip5>,
        #[strategy(Just((1 + #fri.first_round_max_degree()) as i64))] _too_high_degree: i64,
        #[strategy(#_too_high_degree..2 * #_too_high_degree)] _degree: i64,
        #[strategy(arbitrary_polynomial_of_degree(#_degree))] polynomial: Polynomial<XFieldElement>,
    ) {
        debug_assert!(polynomial.degree() > fri.first_round_max_degree() as isize);
        let codeword = fri.domain.evaluate(&polynomial);
        let mut proof_stream = ProofStream::new();
        fri.prove(&codeword, &mut proof_stream).unwrap();

        let mut proof_stream = prepare_proof_stream_for_verification(proof_stream);
        let verdict = fri.verify(&mut proof_stream, &mut None);
        prop_assert!(verdict.is_err());
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
        let num_collinearity_checks = 1;
        Fri::new(domain, expansion_factor, num_collinearity_checks).unwrap()
    }

    #[test]
    fn too_small_expansion_factor_is_rejected() {
        let domain = ArithmeticDomain::of_length(2);
        let expansion_factor = 1;
        let num_collinearity_checks = 1;
        let err = Fri::<Tip5>::new(domain, expansion_factor, num_collinearity_checks).unwrap_err();
        assert_eq!(FriSetupError::ExpansionFactorTooSmall, err);
    }

    #[proptest]
    fn expansion_factor_not_a_power_of_two_is_rejected(
        #[strategy(2_usize..(1 << 32))]
        #[filter(!#expansion_factor.is_power_of_two())]
        expansion_factor: usize,
    ) {
        let largest_supported_domain_size = 1 << 32;
        let domain = ArithmeticDomain::of_length(largest_supported_domain_size);
        let num_collinearity_checks = 1;
        let err = Fri::<Tip5>::new(domain, expansion_factor, num_collinearity_checks).unwrap_err();
        prop_assert_eq!(FriSetupError::ExpansionFactorUnsupported, err);
    }

    #[proptest]
    fn domain_size_smaller_than_expansion_factor_is_rejected(
        #[strategy(1_usize..32)] log_2_expansion_factor: usize,
        #[strategy(..#log_2_expansion_factor)] log_2_domain_length: usize,
    ) {
        let expansion_factor = 1 << log_2_expansion_factor;
        let domain_length = 1 << log_2_domain_length;
        let domain = ArithmeticDomain::of_length(domain_length);
        let num_collinearity_checks = 1;
        let err = Fri::<Tip5>::new(domain, expansion_factor, num_collinearity_checks).unwrap_err();
        prop_assert_eq!(FriSetupError::ExpansionFactorMismatch, err);
    }

    // todo: add test fuzzing proof_stream

    #[proptest(cases = 50)]
    fn serialization(
        #[strategy(arbitrary_fri())] fri: Fri<Tip5>,
        #[strategy(-1_i64..=#fri.first_round_max_degree() as i64)] _degree: i64,
        #[strategy(arbitrary_polynomial_of_degree(#_degree))] polynomial: Polynomial<XFieldElement>,
    ) {
        let codeword = fri.domain.evaluate(&polynomial);
        let mut prover_proof_stream = ProofStream::new();
        fri.prove(&codeword, &mut prover_proof_stream).unwrap();

        let proof = (&prover_proof_stream).into();
        let verifier_proof_stream = ProofStream::<Tip5>::try_from(&proof).unwrap();

        let prover_items = prover_proof_stream.items.iter();
        let verifier_items = verifier_proof_stream.items.iter();
        for (prover_item, verifier_item) in prover_items.zip_eq(verifier_items) {
            match (prover_item, verifier_item) {
                (MerkleRoot(p), MerkleRoot(v)) => prop_assert_eq!(p, v),
                (FriResponse(p), FriResponse(v)) => prop_assert_eq!(p, v),
                (FriCodeword(p), FriCodeword(v)) => prop_assert_eq!(p, v),
                _ => panic!("Unknown items.\nProver: {prover_item:?}\nVerifier: {verifier_item:?}"),
            }
        }
    }

    #[proptest(cases = 50)]
    fn last_round_codeword_unequal_to_last_round_commitment_results_in_validation_failure(
        #[strategy(arbitrary_fri())] fri: Fri<Tip5>,
        #[strategy(arbitrary_polynomial())] polynomial: Polynomial<XFieldElement>,
        rng_seed: u64,
    ) {
        let codeword = fri.domain.evaluate(&polynomial);
        let mut proof_stream = ProofStream::new();
        fri.prove(&codeword, &mut proof_stream).unwrap();

        let proof_stream = prepare_proof_stream_for_verification(proof_stream);
        let mut proof_stream =
            modify_last_round_codeword_in_proof_stream_using_seed(proof_stream, rng_seed);

        let verdict = fri.verify(&mut proof_stream, &mut None);
        let err = verdict.unwrap_err();
        let FriValidationError::BadMerkleRootForLastCodeword = err else {
            return Err(TestCaseError::Fail("validation must fail".into()));
        };
    }

    #[must_use]
    fn prepare_proof_stream_for_verification<H: AlgebraicHasher>(
        mut proof_stream: ProofStream<H>,
    ) -> ProofStream<H> {
        proof_stream.items_index = 0;
        proof_stream.sponge_state = H::init();
        proof_stream
    }

    #[must_use]
    fn modify_last_round_codeword_in_proof_stream_using_seed<H: AlgebraicHasher>(
        mut proof_stream: ProofStream<H>,
        seed: u64,
    ) -> ProofStream<H> {
        let mut proof_items = proof_stream.items.iter_mut();
        let last_round_codeword = proof_items.find_map(fri_codeword_filter()).unwrap();

        let mut rng = StdRng::seed_from_u64(seed);
        let modification_index = rng.gen_range(0..last_round_codeword.len());
        let replacement_element = rng.gen();

        last_round_codeword[modification_index] = replacement_element;
        proof_stream
    }

    fn fri_codeword_filter() -> fn(&mut ProofItem) -> Option<&mut Vec<XFieldElement>> {
        |proof_item| match proof_item {
            FriCodeword(codeword) => Some(codeword),
            _ => None,
        }
    }

    #[proptest(cases = 50)]
    fn revealing_wrong_number_of_leaves_results_in_validation_failure(
        #[strategy(arbitrary_fri())] fri: Fri<Tip5>,
        #[strategy(arbitrary_polynomial())] polynomial: Polynomial<XFieldElement>,
        rng_seed: u64,
    ) {
        let codeword = fri.domain.evaluate(&polynomial);
        let mut proof_stream = ProofStream::new();
        fri.prove(&codeword, &mut proof_stream).unwrap();

        let proof_stream = prepare_proof_stream_for_verification(proof_stream);
        let mut proof_stream =
            change_size_of_some_fri_response_in_proof_stream_using_seed(proof_stream, rng_seed);

        let verdict = fri.verify(&mut proof_stream, &mut None);
        let err = verdict.unwrap_err();
        let FriValidationError::IncorrectNumberOfRevealedLeaves = err else {
            return Err(TestCaseError::Fail("validation must fail".into()));
        };
    }

    #[must_use]
    fn change_size_of_some_fri_response_in_proof_stream_using_seed<H: AlgebraicHasher>(
        mut proof_stream: ProofStream<H>,
        seed: u64,
    ) -> ProofStream<H> {
        let proof_items = proof_stream.items.iter_mut();
        let fri_responses = proof_items.filter_map(fri_response_filter());

        let mut rng = StdRng::seed_from_u64(seed);
        let fri_response = fri_responses.choose(&mut rng).unwrap();
        let revealed_leaves = &mut fri_response.revealed_leaves;
        let modification_index = rng.gen_range(0..revealed_leaves.len());
        match rng.gen() {
            true => _ = revealed_leaves.remove(modification_index),
            false => revealed_leaves.insert(modification_index, rng.gen()),
        };

        proof_stream
    }

    fn fri_response_filter() -> fn(&mut ProofItem) -> Option<&mut super::FriResponse> {
        |proof_item| match proof_item {
            FriResponse(fri_response) => Some(fri_response),
            _ => None,
        }
    }

    #[proptest(cases = 50)]
    fn incorrect_authentication_structure_results_in_validation_failure(
        #[strategy(arbitrary_fri())] fri: Fri<Tip5>,
        #[strategy(arbitrary_polynomial())] polynomial: Polynomial<XFieldElement>,
        rng_seed: u64,
    ) {
        let all_authentication_structures_are_trivial =
            fri.num_collinearity_checks >= fri.domain.length;
        if all_authentication_structures_are_trivial {
            return Ok(());
        }

        let codeword = fri.domain.evaluate(&polynomial);
        let mut proof_stream = ProofStream::new();
        fri.prove(&codeword, &mut proof_stream).unwrap();

        let proof_stream = prepare_proof_stream_for_verification(proof_stream);
        let mut proof_stream =
            modify_some_auth_structure_in_proof_stream_using_seed(proof_stream, rng_seed);

        let verdict = fri.verify(&mut proof_stream, &mut None);
        let_assert!(Err(err) = verdict);
        assert!(let BadMerkleAuthenticationPath = err);
    }

    #[must_use]
    fn modify_some_auth_structure_in_proof_stream_using_seed<H: AlgebraicHasher>(
        mut proof_stream: ProofStream<H>,
        seed: u64,
    ) -> ProofStream<H> {
        let proof_items = proof_stream.items.iter_mut();
        let auth_structures = proof_items.filter_map(non_trivial_auth_structure_filter());

        let mut rng = StdRng::seed_from_u64(seed);
        let auth_structure = auth_structures.choose(&mut rng).unwrap();
        let modification_index = rng.gen_range(0..auth_structure.len());
        match rng.gen_range(0..3) {
            0 => _ = auth_structure.remove(modification_index),
            1 => auth_structure.insert(modification_index, rng.gen()),
            2 => auth_structure[modification_index] = rng.gen(),
            _ => unreachable!(),
        };

        proof_stream
    }

    fn non_trivial_auth_structure_filter(
    ) -> fn(&mut ProofItem) -> Option<&mut super::AuthenticationStructure> {
        |proof_item| match proof_item {
            FriResponse(fri_response) if fri_response.auth_structure.is_empty() => None,
            FriResponse(fri_response) => Some(&mut fri_response.auth_structure),
            _ => None,
        }
    }

    #[proptest]
    fn verifying_arbitrary_proof_does_not_panic(
        #[strategy(arbitrary_fri())] fri: Fri<Tip5>,
        #[strategy(arb())] mut proof_stream: ProofStream<Tip5>,
    ) {
        let _ = fri.verify(&mut proof_stream, &mut None);
    }
}
