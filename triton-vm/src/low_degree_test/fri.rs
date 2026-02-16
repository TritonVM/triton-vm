use arbitrary::Arbitrary;
use itertools::Itertools;
use num_traits::Zero;
use rayon::prelude::*;
use twenty_first::math::polynomial::barycentric_evaluate;
use twenty_first::math::traits::FiniteField;
use twenty_first::prelude::*;

use crate::arithmetic_domain::ArithmeticDomain;
use crate::error::LdtParameterError;
use crate::error::LdtProvingError;
use crate::error::LdtVerificationError;
use crate::error::USIZE_TO_U64_ERR;
use crate::low_degree_test::LowDegreeTest;
use crate::low_degree_test::ProverResult;
use crate::low_degree_test::ProximityRegime;
use crate::low_degree_test::ReedSolomonCode;
use crate::low_degree_test::SetupResult;
use crate::low_degree_test::VerifierPostscript;
use crate::low_degree_test::VerifierResult;
use crate::profiler::profiler;
use crate::proof_item::AuthenticationStructure;
use crate::proof_item::ProofItem;
use crate::proof_stream::ProofStream;

/// The initial parameters from which to derive a [FRI](Fri) instance.
///
/// This struct captures the defining protocol parameters. It can be used to
/// [create an instance of FRI](Fri::new). The documentation on this struct's
/// fields informs about the legal parameter space.
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
#[cfg_attr(test, derive(test_strategy::Arbitrary))]
pub struct FriParameters {
    /// The desired security level in bits.
    ///
    /// See also: [`Stark::security_level`](crate::Stark::security_level).
    #[cfg_attr(test, strategy(16_usize..=192))]
    pub security_level: usize,

    /// The soundness-influencing assumption (or lack thereof) you are willing
    /// to make.
    pub soundness: ProximityRegime,

    /// The amount of “redundancy” in the [prover](Fri::prove)'s input.
    ///
    /// In particular, the Reed-Solomon code's rate is the reciprocal of 2
    /// raised to this value. In other words, the initial rate equals
    /// `1 / 2^initial_log2_expansion_factor`.
    ///
    /// Must be greater than 0.
    #[cfg_attr(test, strategy(1_usize..=6))]
    pub log2_initial_expansion_factor: usize,

    /// The low-degree test's degree bound.
    ///
    /// In particular, the (log₂ of the) polynomial degree that is considered
    /// “high” (_i.e._, “not low”) for this FRI instance.
    ///
    /// In other words, the low-degreeness of polynomials with degree
    /// `2^log2_high_degree_bound` (and higher) cannot be [proven](Fri::prove)
    /// (in a way that the [verifier](Fri::verify) accepts with high
    /// probability). On the other hand, the low-degreeness of polynomials with
    /// degree `2^log2_high_degree_bound - 1` (and lower) _can_ be proven.
    #[cfg_attr(test, strategy(0_usize..=15))]
    pub log2_high_degree_bound: usize,
}

/// The “Fast Reed-Solomon Interactive Oracle Proof of Proximity” (“[FRI][fri]”)
/// low-degree test (“[LDT](LowDegreeTest)”).
///
/// To construct a new instance of FRI, see [`FriParameters`].
///
/// [fri]: https://doi.org/10.4230/LIPIcs.ICALP.2018.14
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct Fri {
    expansion_factor: usize,
    num_collinearity_checks: usize,
    domain: ArithmeticDomain,
}

#[derive(Debug, Eq, PartialEq)]
struct FriProver<'stream> {
    proof_stream: &'stream mut ProofStream,
    rounds: Vec<ProverRound>,
    first_round_domain: ArithmeticDomain,
    num_rounds: usize,
    num_collinearity_checks: usize,
    first_round_collinearity_check_indices: Vec<usize>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct ProverRound {
    domain: ArithmeticDomain,
    codeword: Vec<XFieldElement>,
    merkle_tree: MerkleTree,
}

/// A [FRI](Fri) round's revealed values together with an authentication
/// structure.
#[derive(Debug, Clone, Eq, PartialEq, Hash, BFieldCodec, Arbitrary)]
pub struct FriResponse {
    /// The values of the queried leaves of the Merkle tree.
    pub queried_leaves: Vec<XFieldElement>,

    /// The authentication structure of the Merkle tree.
    pub auth_structure: AuthenticationStructure,
}

/// A postscript of a [FRI verification](Fri::verify).
///
/// For additional details, see [`VerifierPostscript`].
//
// Marked `#[non_exhaustive]` because
// 1. additional fields might be added in the future and I don't want that to be
//    a breaking change, and
// 2. this type is not intended to be constructed anywhere but in this module.
//
// Also applies to the other “Postscript” structs.
#[non_exhaustive]
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct Postscript {
    /// The postscript of the initial round of FRI.
    pub initial_round: InitialRoundPostscript,

    /// One round-postscript per full round of FRI.
    pub rounds: Vec<RoundPostscript>,

    /// The final, fully folded codeword.
    pub last_round_codeword: Vec<XFieldElement>,

    /// The final polynomial, corresponding to the
    /// [final codeword](Self::last_round_codeword) (in the honest case).
    ///
    /// Note: the equality between the final codeword and the final polynomial
    /// is established with respect to the final domain but with its
    /// [`offset`](ArithmeticDomain::with_offset) set to 1.
    pub last_round_polynomial: Polynomial<'static, XFieldElement>,
}

/// A postscript of the initial round of a [FRI verification](Fri::verify).
///
/// This postscript pertains to “A-indices” only.
///
/// See [`Postscript`] for a full explanation.
#[non_exhaustive]
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct InitialRoundPostscript {
    /// The (unfolded) domain used in this round.
    pub domain: ArithmeticDomain,

    /// The observed commitment to the
    /// [partial codeword](Self::partial_codeword).
    pub merkle_root: Digest,

    /// The partially revealed initial codeword.
    pub partial_codeword: Vec<XFieldElement>,

    /// The indices at which the [partial codeword](Self::partial_codeword)
    /// is revealed.
    ///
    /// In internal lingo, these are “A-indices”.
    pub indices: Vec<usize>,

    /// The authentication structure for the
    /// [partial codeword](Self::partial_codeword).
    pub auth_structure: AuthenticationStructure,
}

/// A postscript of a single full round of a [FRI verification](Fri::verify).
///
/// This postscript pertains to “B-indices” only.
///
/// See [`Postscript`] for a full explanation.
#[non_exhaustive]
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct RoundPostscript {
    /// The (unfolded) domain used in this round.
    pub domain: ArithmeticDomain,

    /// The randomness used for folding this round's polynomial.
    ///
    /// Is `None` in the last round (and in the last round only), as there is no
    /// next round for which a folded polynomial is needed.
    pub folding_challenge: Option<XFieldElement>,

    /// The observed commitment to the
    /// [partial codeword](Self::partial_codeword).
    pub merkle_root: Digest,

    /// The partially revealed B-codeword of this round.
    pub partial_codeword: Vec<XFieldElement>,

    /// The indices of the domain points that were queried.
    ///
    /// In internal lingo, these are “B-indices”.
    pub indices: Vec<usize>,

    /// The authentication structure proving the inclusion of the polynomial's
    /// evaluations at the [queried indices](Self::indices).
    pub auth_structure: AuthenticationStructure,
}

impl TryFrom<FriParameters> for Fri {
    type Error = LdtParameterError;

    fn try_from(params: FriParameters) -> SetupResult<Self> {
        params.try_into_fri()
    }
}

impl FriProver<'_> {
    fn commit(&mut self, codeword: &[XFieldElement]) -> ProverResult<()> {
        self.commit_to_first_round(codeword)?;
        for _ in 0..self.num_rounds {
            self.commit_to_next_round()?;
        }
        self.send_last_codeword();
        self.send_last_polynomial();
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

    fn commit_to_round(&mut self, round: &ProverRound) {
        let merkle_root = round.merkle_tree.root();
        let proof_item = ProofItem::MerkleRoot(merkle_root);
        self.proof_stream.enqueue(proof_item);
    }

    fn store_round(&mut self, round: ProverRound) {
        self.rounds.push(round);
    }

    fn construct_next_round(&mut self) -> ProverResult<ProverRound> {
        let previous_round = self.rounds.last().unwrap();
        let folding_challenge = self.proof_stream.sample_scalars(1)[0];
        let codeword = previous_round.split_and_fold(folding_challenge);
        let domain = previous_round.domain.pow(2).unwrap();

        ProverRound::new(domain, &codeword)
    }

    fn send_last_codeword(&mut self) {
        let last_codeword = self.rounds.last().unwrap().codeword.clone();
        let proof_item = ProofItem::FriCodeword(last_codeword);
        self.proof_stream.enqueue(proof_item);
    }

    fn send_last_polynomial(&mut self) {
        let last_codeword = &self.rounds.last().unwrap().codeword;
        let last_polynomial = ArithmeticDomain::of_length(last_codeword.len())
            .unwrap()
            .interpolate(last_codeword);
        let proof_item = ProofItem::Polynomial(last_polynomial);
        self.proof_stream.enqueue(proof_item);
    }

    fn query(&mut self) {
        self.sample_first_round_collinearity_check_indices();

        let initial_a_indices = self.first_round_collinearity_check_indices.clone();
        self.authentically_reveal_codeword_of_round_at_indices(0, &initial_a_indices);

        let num_rounds_that_have_a_next_round = self.rounds.len() - 1;
        for round_number in 0..num_rounds_that_have_a_next_round {
            let b_indices = self.collinearity_check_b_indices_for_round(round_number);
            self.authentically_reveal_codeword_of_round_at_indices(round_number, &b_indices);
        }
    }

    fn sample_first_round_collinearity_check_indices(&mut self) {
        let indices_upper_bound = self.first_round_domain.len();
        self.first_round_collinearity_check_indices = self
            .proof_stream
            .sample_indices(indices_upper_bound, self.num_collinearity_checks);
    }

    fn collinearity_check_b_indices_for_round(&self, round_number: usize) -> Vec<usize> {
        let domain_length = self.rounds[round_number].domain.len();
        self.first_round_collinearity_check_indices
            .iter()
            .map(|&a_index| (a_index + domain_length / 2) % domain_length)
            .collect()
    }

    /// # Panics
    ///
    /// Panics if any of the indices is bigger than (or equal to) the number of
    /// leafs in the respective round's Merkle tree.
    fn authentically_reveal_codeword_of_round_at_indices(
        &mut self,
        round_number: usize,
        indices: &[usize],
    ) {
        let codeword = &self.rounds[round_number].codeword;
        let queried_leaves = indices.iter().map(|&i| codeword[i]).collect_vec();

        let merkle_tree = &self.rounds[round_number].merkle_tree;
        let auth_structure = merkle_tree.authentication_structure(indices).unwrap();

        let fri_response = FriResponse {
            queried_leaves,
            auth_structure,
        };
        let proof_item = ProofItem::FriResponse(fri_response);
        self.proof_stream.enqueue(proof_item);
    }
}

impl ProverRound {
    fn new(domain: ArithmeticDomain, codeword: &[XFieldElement]) -> ProverResult<Self> {
        if domain.len() != codeword.len() {
            return Err(LdtProvingError::InitialCodewordMismatch {
                domain_len: domain.len(),
                codeword_len: codeword.len(),
            });
        }

        let merkle_tree = Self::merkle_tree_from_codeword(codeword);
        let round = Self {
            domain,
            codeword: codeword.to_vec(),
            merkle_tree,
        };
        Ok(round)
    }

    /// # Panics
    ///
    /// Panics if the codeword's length is not a power of two.
    fn merkle_tree_from_codeword(codeword: &[XFieldElement]) -> MerkleTree {
        let digests: Vec<_> = codeword.par_iter().map(|&xfe| xfe.into()).collect();

        MerkleTree::par_new(&digests).unwrap()
    }

    fn split_and_fold(&self, folding_challenge: XFieldElement) -> Vec<XFieldElement> {
        let one = xfe!(1);
        let two_inverse = xfe!(2).inverse();

        let domain_points = self.domain.values();
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

#[derive(Debug, Eq, PartialEq)]
struct FriVerifier<'stream> {
    proof_stream: &'stream mut ProofStream,
    rounds: Vec<VerifierRound>,
    first_round_domain: ArithmeticDomain,
    last_round_codeword: Vec<XFieldElement>,
    last_round_polynomial: Polynomial<'static, XFieldElement>,
    last_round_max_degree: usize,
    num_rounds: usize,
    num_collinearity_checks: usize,
    first_round_collinearity_check_indices: Vec<usize>,
}

#[derive(Debug, Clone, Eq, PartialEq)]
struct VerifierRound {
    domain: ArithmeticDomain,
    merkle_root: Digest,
    partial_codeword_a: Vec<XFieldElement>,
    auth_structure_a: Vec<Digest>,
    partial_codeword_b: Vec<XFieldElement>,
    auth_structure_b: Vec<Digest>,
    folding_challenge: Option<XFieldElement>,
}

impl FriVerifier<'_> {
    fn initialize(&mut self) -> VerifierResult<()> {
        let domain = self.first_round_domain;
        let first_round = self.construct_round_with_domain(domain)?;
        self.rounds.push(first_round);

        for _ in 0..self.num_rounds {
            let previous_round = self.rounds.last().unwrap();
            let domain = previous_round.domain.pow(2).unwrap();
            let next_round = self.construct_round_with_domain(domain)?;
            self.rounds.push(next_round);
        }

        self.last_round_codeword = self.proof_stream.dequeue()?.try_into_fri_codeword()?;
        self.last_round_polynomial = self.proof_stream.dequeue()?.try_into_polynomial()?;

        Ok(())
    }

    fn construct_round_with_domain(
        &mut self,
        domain: ArithmeticDomain,
    ) -> VerifierResult<VerifierRound> {
        let merkle_root = self.proof_stream.dequeue()?.try_into_merkle_root()?;
        let folding_challenge = self
            .need_more_folding_challenges()
            .then(|| self.proof_stream.sample_scalars(1)[0]);

        let verifier_round = VerifierRound {
            domain,
            merkle_root,
            partial_codeword_a: Vec::new(),
            auth_structure_a: Vec::new(),
            partial_codeword_b: Vec::new(),
            auth_structure_b: Vec::new(),
            folding_challenge,
        };

        Ok(verifier_round)
    }

    fn need_more_folding_challenges(&self) -> bool {
        if self.num_rounds == 0 {
            return false;
        }

        let num_initialized_rounds = self.rounds.len();
        let num_rounds_that_have_a_next_round = self.num_rounds - 1;

        num_initialized_rounds <= num_rounds_that_have_a_next_round
    }

    fn compute_last_round_folded_partial_codeword(&mut self) -> VerifierResult<()> {
        self.sample_first_round_collinearity_check_indices();
        self.receive_authentic_partially_revealed_codewords()?;
        self.successively_fold_partial_codeword_of_each_round();

        Ok(())
    }

    fn sample_first_round_collinearity_check_indices(&mut self) {
        let upper_bound = self.first_round_domain.len();
        self.first_round_collinearity_check_indices = self
            .proof_stream
            .sample_indices(upper_bound, self.num_collinearity_checks);
    }

    fn receive_authentic_partially_revealed_codewords(&mut self) -> VerifierResult<()> {
        self.receive_partial_codeword_a_for_first_round()?;
        self.authenticate_partial_codeword_a_for_first_round()?;

        let num_rounds_that_have_a_next_round = self.rounds.len() - 1;
        for round_number in 0..num_rounds_that_have_a_next_round {
            self.receive_partial_codeword_b_for_round(round_number)?;
            self.authenticate_partial_codeword_b_for_round(round_number)?;
        }

        Ok(())
    }

    fn receive_partial_codeword_a_for_first_round(&mut self) -> VerifierResult<()> {
        let fri_response = self.proof_stream.dequeue()?.try_into_fri_response()?;
        let FriResponse {
            queried_leaves,
            auth_structure,
        } = fri_response;

        self.assert_enough_leaves_were_received(&queried_leaves)?;
        self.rounds[0].partial_codeword_a = queried_leaves;
        self.rounds[0].auth_structure_a = auth_structure;

        Ok(())
    }

    fn receive_partial_codeword_b_for_round(&mut self, round_number: usize) -> VerifierResult<()> {
        let fri_response = self.proof_stream.dequeue()?.try_into_fri_response()?;
        let FriResponse {
            queried_leaves,
            auth_structure,
        } = fri_response;

        self.assert_enough_leaves_were_received(&queried_leaves)?;
        self.rounds[round_number].partial_codeword_b = queried_leaves;
        self.rounds[round_number].auth_structure_b = auth_structure;

        Ok(())
    }

    fn assert_enough_leaves_were_received(&self, leaves: &[XFieldElement]) -> VerifierResult<()> {
        if self.num_collinearity_checks == leaves.len() {
            Ok(())
        } else {
            Err(LdtVerificationError::IncorrectNumberOfRevealedLeaves)
        }
    }

    fn authenticate_partial_codeword_a_for_first_round(&self) -> VerifierResult<()> {
        let round = &self.rounds[0];
        let revealed_leaves = &round.partial_codeword_a;
        let revealed_digests = codeword_as_digests(revealed_leaves);

        let leaf_indices = self.collinearity_check_a_indices_for_round(0);
        let indexed_leafs = leaf_indices.into_iter().zip_eq(revealed_digests).collect();

        let inclusion_proof = MerkleTreeInclusionProof {
            tree_height: round.merkle_tree_height(),
            indexed_leafs,
            authentication_structure: round.auth_structure_a.clone(),
        };

        if inclusion_proof.verify(round.merkle_root) {
            Ok(())
        } else {
            Err(LdtVerificationError::BadMerkleAuthenticationPath)
        }
    }

    fn authenticate_partial_codeword_b_for_round(&self, round_number: usize) -> VerifierResult<()> {
        let round = &self.rounds[round_number];
        let revealed_leaves = &round.partial_codeword_b;
        let revealed_digests = codeword_as_digests(revealed_leaves);

        let leaf_indices = self.collinearity_check_b_indices_for_round(round_number);
        let indexed_leafs = leaf_indices.into_iter().zip_eq(revealed_digests).collect();

        let inclusion_proof = MerkleTreeInclusionProof {
            tree_height: round.merkle_tree_height(),
            indexed_leafs,
            authentication_structure: round.auth_structure_b.clone(),
        };

        if inclusion_proof.verify(round.merkle_root) {
            Ok(())
        } else {
            Err(LdtVerificationError::BadMerkleAuthenticationPath)
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
            .map(|i| {
                let point_a_x = domain.value(a_indices[i] as u32).lift();
                let point_b_x = domain.value(b_indices[i] as u32).lift();
                let point_a = (point_a_x, partial_codeword_a[i]);
                let point_b = (point_b_x, partial_codeword_b[i]);
                Polynomial::get_colinear_y(point_a, point_b, folding_challenge)
            })
            .collect()
    }

    fn collinearity_check_a_indices_for_round(&self, round_number: usize) -> Vec<usize> {
        let domain_length = self.rounds[round_number].domain.len();
        let a_offset = 0;

        self.collinearity_check_indices_with_offset_and_modulus(a_offset, domain_length)
    }

    fn collinearity_check_b_indices_for_round(&self, round_number: usize) -> Vec<usize> {
        let domain_length = self.rounds[round_number].domain.len();
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

    fn authenticate_last_round_codeword(&mut self) -> VerifierResult<()> {
        self.assert_last_round_codeword_matches_last_round_commitment()?;
        self.assert_last_round_codeword_agrees_with_last_round_folded_codeword()?;
        self.assert_last_round_codeword_corresponds_to_low_degree_polynomial()?;

        Ok(())
    }

    fn assert_last_round_codeword_matches_last_round_commitment(&self) -> VerifierResult<()> {
        if self.last_round_merkle_root() == self.last_round_codeword_merkle_root()? {
            Ok(())
        } else {
            Err(LdtVerificationError::BadMerkleRootForLastCodeword)
        }
    }

    fn last_round_codeword_merkle_root(&self) -> VerifierResult<Digest> {
        let codeword_digests = codeword_as_digests(&self.last_round_codeword);
        let merkle_tree = MerkleTree::sequential_new(&codeword_digests)
            .map_err(LdtVerificationError::MerkleTreeError)?;

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

        if partial_received_codeword == partial_folded_codeword {
            Ok(())
        } else {
            Err(LdtVerificationError::LastCodewordMismatch)
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
        &mut self,
    ) -> VerifierResult<()> {
        if self.last_round_polynomial.degree() > self.last_round_max_degree.try_into().unwrap() {
            return Err(LdtVerificationError::LastRoundPolynomialHasTooHighDegree);
        }

        let indeterminate = self.proof_stream.sample_scalars(1)[0];
        let horner_evaluation = self
            .last_round_polynomial
            .evaluate_in_same_field(indeterminate);
        let barycentric_evaluation = barycentric_evaluate(&self.last_round_codeword, indeterminate);
        if horner_evaluation != barycentric_evaluation {
            return Err(LdtVerificationError::LastRoundPolynomialEvaluationMismatch);
        }

        Ok(())
    }

    fn postscript(self) -> Postscript {
        // to de-structure `self`, pre-compute things that depend on it
        let mut all_b_indices = Vec::with_capacity(self.num_rounds);
        for round_no in 0..self.num_rounds {
            all_b_indices.push(self.collinearity_check_b_indices_for_round(round_no));
        }

        // the last round doesn't include a partial codeword
        all_b_indices.push(Vec::new());
        let all_b_indices = all_b_indices;

        let Self {
            proof_stream: _,
            rounds,
            first_round_domain,
            last_round_codeword,
            last_round_polynomial,
            last_round_max_degree: _,
            num_rounds,
            num_collinearity_checks: _,
            first_round_collinearity_check_indices,
        } = self;

        // running FRI results in a minimum of one round
        let initial_round = InitialRoundPostscript {
            domain: first_round_domain,
            merkle_root: rounds[0].merkle_root,
            partial_codeword: rounds[0].partial_codeword_a.clone(),
            indices: first_round_collinearity_check_indices,
            auth_structure: rounds[0].auth_structure_a.clone(),
        };

        let mut round_postscripts = Vec::with_capacity(num_rounds);
        for (round, b_indices) in rounds.into_iter().zip_eq(all_b_indices) {
            let round_postscript = RoundPostscript {
                domain: round.domain,
                folding_challenge: round.folding_challenge,
                merkle_root: round.merkle_root,
                partial_codeword: round.partial_codeword_b,
                indices: b_indices,
                auth_structure: round.auth_structure_b,
            };
            round_postscripts.push(round_postscript);
        }

        Postscript {
            initial_round,
            rounds: round_postscripts,
            last_round_codeword,
            last_round_polynomial,
        }
    }
}

impl VerifierRound {
    fn merkle_tree_height(&self) -> u32 {
        self.domain.len().ilog2()
    }
}

impl super::private::Seal for Fri {}

impl LowDegreeTest for Fri {
    fn initial_domain(&self) -> ArithmeticDomain {
        self.domain
    }

    fn num_first_round_queries(&self) -> usize {
        self.num_collinearity_checks
    }

    fn prove(
        &self,
        codeword: &[XFieldElement],
        proof_stream: &mut ProofStream,
    ) -> ProverResult<Vec<usize>> {
        let mut prover = self.prover(proof_stream);

        prover.commit(codeword)?;
        prover.query();

        // Sample one XFieldElement from Fiat-Shamir and then throw it away.
        // This scalar is the indeterminate for the low degree test using the
        // barycentric evaluation formula. This indeterminate is used only by
        // the verifier, but it is important to modify the sponge state the same
        // way.
        prover.proof_stream.sample_scalars(1);

        Ok(prover.first_round_collinearity_check_indices)
    }

    fn verify(&self, proof_stream: &mut ProofStream) -> VerifierResult<VerifierPostscript> {
        profiler!(start "init");
        let mut verifier = self.verifier(proof_stream);
        verifier.initialize()?;
        profiler!(stop "init");

        profiler!(start "fold all rounds");
        verifier.compute_last_round_folded_partial_codeword()?;
        profiler!(stop "fold all rounds");

        profiler!(start "authenticate last round codeword");
        verifier.authenticate_last_round_codeword()?;
        profiler!(stop "authenticate last round codeword");

        Ok(VerifierPostscript::Fri(verifier.postscript()))
    }

    #[cfg(test)]
    fn as_any(&self) -> &dyn std::any::Any {
        self
    }
}

impl FriParameters {
    /// Create a new FRI instance from the given parameters.
    fn try_into_fri(self) -> SetupResult<Fri> {
        if self.log2_initial_expansion_factor == 0 {
            return Err(LdtParameterError::TooSmallInitialExpansionFactor);
        }

        let Ok(log2_expansion_factor) = u32::try_from(self.log2_initial_expansion_factor) else {
            return Err(LdtParameterError::TooBigInitialExpansionFactor);
        };
        let Some(expansion_factor) = 1_usize.checked_shl(log2_expansion_factor) else {
            return Err(LdtParameterError::TooBigInitialExpansionFactor);
        };

        let as_u64 = |int| u64::try_from(int).expect(USIZE_TO_U64_ERR);
        let log2_high_degree_bound = as_u64(self.log2_high_degree_bound);
        let log2_expansion_factor = as_u64(self.log2_initial_expansion_factor);

        let domain_too_big_err = |x| Err(LdtParameterError::InitialDomainTooBig(x));
        let Some(log2_domain_len) = log2_high_degree_bound.checked_add(log2_expansion_factor)
        else {
            return domain_too_big_err(u64::MAX);
        };

        let Ok(log2_domain_len) = u32::try_from(log2_domain_len) else {
            return domain_too_big_err(log2_domain_len);
        };
        let Some(domain_len) = 1_usize.checked_shl(log2_domain_len) else {
            return domain_too_big_err(log2_domain_len.into());
        };
        let domain = ArithmeticDomain::of_length(domain_len)
            .map_err(|_| LdtParameterError::InitialDomainTooBig(log2_domain_len.into()))?
            .with_offset(BFieldElement::generator());

        // See also: https://eprint.iacr.org/2020/654.pdf Theorem 1.2
        let rs_code =
            ReedSolomonCode::new(self.log2_initial_expansion_factor).with_soundness(self.soundness);
        let proximity_parameter = rs_code.proximity_parameter()?;
        let num_collinearity_checks =
            (-(self.security_level as f64) / (1.0 - proximity_parameter).log2()).ceil() as usize;

        let fri = Fri {
            expansion_factor,
            num_collinearity_checks,
            domain,
        };

        Ok(fri)
    }

    pub(crate) fn expansion_factor(&self) -> usize {
        let err = "internal error: log₂(expansion factor) exceeds expected maximum";
        let log2_expansion_factor = u32::try_from(self.log2_initial_expansion_factor).expect(err);

        1_usize.checked_shl(log2_expansion_factor).expect(err)
    }
}

impl Fri {
    /// (Try to) construct a new FRI instance for the given initial parameters.
    ///
    /// It is equivalent to use the provided trait implementation of
    /// [`TryFrom<FriParameters> for Fri`](TryFrom).
    ///
    /// # Errors
    ///
    /// Errors if the
    /// [initial expansion factor](FriParameters::log2_initial_expansion_factor)
    /// is 0.
    pub fn new(parameters: FriParameters) -> SetupResult<Self> {
        parameters.try_into_fri()
    }

    /// The highest polynomial degree for which low-degreeness can be proven
    /// with this FRI instance.
    pub fn max_degree(&self) -> usize {
        (self.domain.len() / self.expansion_factor) - 1
    }

    fn prover<'stream>(
        &'stream self,
        proof_stream: &'stream mut ProofStream,
    ) -> FriProver<'stream> {
        FriProver {
            proof_stream,
            rounds: vec![],
            first_round_domain: self.domain,
            num_rounds: self.num_rounds(),
            num_collinearity_checks: self.num_collinearity_checks,
            first_round_collinearity_check_indices: vec![],
        }
    }

    fn verifier<'stream>(
        &'stream self,
        proof_stream: &'stream mut ProofStream,
    ) -> FriVerifier<'stream> {
        FriVerifier {
            proof_stream,
            rounds: vec![],
            first_round_domain: self.domain,
            last_round_codeword: vec![],
            last_round_polynomial: Polynomial::zero(),
            last_round_max_degree: self.last_round_max_degree(),
            num_rounds: self.num_rounds(),
            num_collinearity_checks: self.num_collinearity_checks,
            first_round_collinearity_check_indices: vec![],
        }
    }

    fn num_rounds(&self) -> usize {
        let first_round_code_dimension = self.max_degree() + 1;
        let max_num_rounds = first_round_code_dimension.next_power_of_two().ilog2();

        // Skip rounds for which Merkle tree verification cost exceeds
        // arithmetic cost, because more than half the codeword's locations are
        // queried.
        let num_rounds_checking_all_locations =
            self.num_collinearity_checks.checked_ilog2().unwrap_or(0);
        let num_rounds_checking_most_locations = num_rounds_checking_all_locations + 1;

        let num_rounds = max_num_rounds.saturating_sub(num_rounds_checking_most_locations);
        num_rounds.try_into().unwrap()
    }

    fn last_round_max_degree(&self) -> usize {
        self.max_degree() >> self.num_rounds()
    }
}

fn codeword_as_digests(codeword: &[XFieldElement]) -> Vec<Digest> {
    codeword.iter().map(|&xfe| xfe.into()).collect()
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use assert2::assert;
    use itertools::Itertools;
    use proptest::prelude::*;
    use proptest::test_runner::TestCaseResult;
    use proptest_arbitrary_adapter::arb;
    use rand::prelude::*;

    use super::*;
    use crate::error::U32_TO_USIZE_ERR;
    use crate::low_degree_test::tests::LdtStats;
    use crate::shared_tests::arbitrary_polynomial;
    use crate::shared_tests::arbitrary_polynomial_of_degree;
    use crate::tests::proptest;
    use crate::tests::test;

    /// A type alias exclusive to this test module.
    type XfePoly = Polynomial<'static, XFieldElement>;

    impl proptest::prelude::Arbitrary for Fri {
        type Parameters = ();

        fn arbitrary_with(_: Self::Parameters) -> Self::Strategy {
            FriParameters::arbitrary()
                .prop_map(|args| args.try_into_fri().unwrap())
                .boxed()
        }

        type Strategy = BoxedStrategy<Self>;
    }

    impl LdtStats for Fri {
        fn num_rounds(&self) -> usize {
            self.num_rounds()
        }

        fn num_total_queries(&self) -> usize {
            self.num_collinearity_checks * self.num_rounds()
        }

        fn log2_initial_domain_len(&self) -> usize {
            self.initial_domain()
                .len()
                .ilog2()
                .try_into()
                .expect(U32_TO_USIZE_ERR)
        }

        fn log2_final_degree_plus_1(&self) -> usize {
            (self.last_round_max_degree() + 1)
                .ilog2()
                .try_into()
                .expect(U32_TO_USIZE_ERR)
        }
    }

    #[macro_rules_attr::apply(proptest)]
    fn num_rounds_are_reasonable(fri: Fri) {
        let expected_last_round_max_degree = fri.max_degree() >> fri.num_rounds();
        prop_assert_eq!(expected_last_round_max_degree, fri.last_round_max_degree());
        if fri.num_rounds() > 0 {
            prop_assert!(fri.num_collinearity_checks <= expected_last_round_max_degree);
            prop_assert!(expected_last_round_max_degree < 2 * fri.num_collinearity_checks);
        }
    }

    #[macro_rules_attr::apply(proptest(cases = 20))]
    fn prove_and_verify_low_degree_of_twice_cubing_plus_one(
        #[filter(#fri.max_degree() >= 3)] fri: Fri,
    ) {
        let polynomial = Polynomial::new(xfe_vec![1, 0, 0, 2]);
        let codeword = fri.domain.evaluate(&polynomial);
        let mut proof_stream = ProofStream::new();
        fri.prove(&codeword, &mut proof_stream).unwrap();

        proof_stream.reset_sponge();
        let verdict = fri.verify(&mut proof_stream);
        prop_assert!(verdict.is_ok());
    }

    #[macro_rules_attr::apply(proptest(cases = 50))]
    fn prove_and_verify_low_degree_polynomial(
        fri: Fri,
        #[strategy(-1_i64..=#fri.max_degree() as i64)] _degree: i64,
        #[strategy(arbitrary_polynomial_of_degree(#_degree))] polynomial: XfePoly,
    ) {
        debug_assert!(polynomial.degree() <= fri.max_degree() as isize);
        let codeword = fri.domain.evaluate(&polynomial);
        let mut proof_stream = ProofStream::new();
        fri.prove(&codeword, &mut proof_stream).unwrap();

        proof_stream.reset_sponge();
        let verdict = fri.verify(&mut proof_stream);
        prop_assert!(verdict.is_ok());
    }

    #[macro_rules_attr::apply(proptest)]
    fn fail_to_prove_codeword_of_incorrect_length(
        fri: Fri,
        #[strategy(arb())]
        #[filter(#codeword.len() != #fri.domain.len())]
        codeword: Vec<XFieldElement>,
    ) {
        let mut proof_stream = ProofStream::new();
        assert!(let Err(err) = fri.prove(&codeword, &mut proof_stream));
        assert!(let
            LdtProvingError::InitialCodewordMismatch {
                domain_len,
                codeword_len
            } = err
        );
        prop_assert_eq!(codeword.len(), codeword_len);
        prop_assert_eq!(fri.domain.len(), domain_len);
    }

    #[macro_rules_attr::apply(proptest(cases = 50))]
    fn prove_and_fail_to_verify_high_degree_polynomial(
        fri: Fri,
        #[strategy(Just((1 + #fri.max_degree()) as i64))] _too_high_degree: i64,
        #[strategy(#_too_high_degree..2 * #_too_high_degree)] _degree: i64,
        #[strategy(arbitrary_polynomial_of_degree(#_degree))] polynomial: XfePoly,
    ) {
        debug_assert!(polynomial.degree() > fri.max_degree() as isize);
        let codeword = fri.domain.evaluate(&polynomial);
        let mut proof_stream = ProofStream::new();
        fri.prove(&codeword, &mut proof_stream).unwrap();

        proof_stream.reset_sponge();
        let verdict = fri.verify(&mut proof_stream);
        prop_assert!(verdict.is_err());
    }

    #[macro_rules_attr::apply(test)]
    fn smallest_possible_fri_has_no_rounds() {
        assert_eq!(0, smallest_fri().num_rounds());
    }

    #[macro_rules_attr::apply(test)]
    fn smallest_possible_fri_can_only_verify_constant_polynomials() {
        assert_eq!(0, smallest_fri().max_degree());
    }

    fn smallest_fri() -> Fri {
        FriParameters {
            security_level: 1,
            soundness: ProximityRegime::Conjectured,
            log2_initial_expansion_factor: 1,
            log2_high_degree_bound: 0,
        }
        .try_into_fri()
        .unwrap()
    }

    #[macro_rules_attr::apply(test)]
    fn too_small_expansion_factor_is_rejected() {
        let parameters = FriParameters {
            security_level: 1,
            soundness: ProximityRegime::default(),
            log2_initial_expansion_factor: 0,
            log2_high_degree_bound: 0,
        };
        let err = parameters.try_into_fri().unwrap_err();
        assert_eq!(LdtParameterError::TooSmallInitialExpansionFactor, err);
    }

    #[macro_rules_attr::apply(proptest)]
    fn too_big_expansion_factor_is_rejected(
        #[strategy(32_usize..)] log2_initial_expansion_factor: usize,
    ) {
        let parameters = FriParameters {
            security_level: 1,
            soundness: ProximityRegime::default(),
            log2_initial_expansion_factor,
            log2_high_degree_bound: 0,
        };
        let err = parameters.try_into_fri().unwrap_err();
        prop_assert_eq!(LdtParameterError::TooBigInitialExpansionFactor, err);
    }

    #[macro_rules_attr::apply(proptest(cases = 50))]
    fn serialization(
        fri: Fri,
        #[strategy(-1_i64..=#fri.max_degree() as i64)] _degree: i64,
        #[strategy(arbitrary_polynomial_of_degree(#_degree))] polynomial: XfePoly,
    ) {
        let codeword = fri.domain.evaluate(&polynomial);
        let mut prover_proof_stream = ProofStream::new();
        fri.prove(&codeword, &mut prover_proof_stream).unwrap();

        let proof = (&prover_proof_stream).into();
        let verifier_proof_stream = ProofStream::try_from(&proof).unwrap();

        let prover_items = prover_proof_stream.items.iter();
        let verifier_items = verifier_proof_stream.items.iter();
        for (prover_item, verifier_item) in prover_items.zip_eq(verifier_items) {
            use ProofItem as PI;
            match (prover_item, verifier_item) {
                (PI::MerkleRoot(p), PI::MerkleRoot(v)) => prop_assert_eq!(p, v),
                (PI::FriResponse(p), PI::FriResponse(v)) => prop_assert_eq!(p, v),
                (PI::FriCodeword(p), PI::FriCodeword(v)) => prop_assert_eq!(p, v),
                (PI::Polynomial(p), PI::Polynomial(v)) => prop_assert_eq!(p, v),
                _ => panic!("Unknown items.\nProver: {prover_item:?}\nVerifier: {verifier_item:?}"),
            }
        }
    }

    #[macro_rules_attr::apply(proptest(cases = 50))]
    fn last_round_codeword_unequal_to_last_round_commitment_results_in_validation_failure(
        fri: Fri,
        #[strategy(arbitrary_polynomial())] polynomial: XfePoly,
        rng_seed: u64,
    ) {
        let codeword = fri.domain.evaluate(&polynomial);
        let mut proof_stream = ProofStream::new();
        fri.prove(&codeword, &mut proof_stream).unwrap();

        proof_stream.reset_sponge();
        let mut proof_stream =
            modify_last_round_codeword_in_proof_stream_using_seed(proof_stream, rng_seed);

        let verdict = fri.verify(&mut proof_stream);
        let err = verdict.unwrap_err();
        let LdtVerificationError::BadMerkleRootForLastCodeword = err else {
            return Err(TestCaseError::Fail("validation must fail".into()));
        };
    }

    #[must_use]
    fn modify_last_round_codeword_in_proof_stream_using_seed(
        mut proof_stream: ProofStream,
        seed: u64,
    ) -> ProofStream {
        let mut proof_items = proof_stream.items.iter_mut();
        let last_round_codeword = proof_items.find_map(fri_codeword_filter()).unwrap();

        let mut rng = StdRng::seed_from_u64(seed);
        let modification_index = rng.random_range(0..last_round_codeword.len());
        let replacement_element = rng.random();

        last_round_codeword[modification_index] = replacement_element;
        proof_stream
    }

    fn fri_codeword_filter() -> fn(&mut ProofItem) -> Option<&mut Vec<XFieldElement>> {
        |proof_item| match proof_item {
            ProofItem::FriCodeword(codeword) => Some(codeword),
            _ => None,
        }
    }

    #[macro_rules_attr::apply(proptest(cases = 50))]
    fn revealing_wrong_number_of_leaves_results_in_validation_failure(
        fri: Fri,
        #[strategy(arbitrary_polynomial())] polynomial: XfePoly,
        rng_seed: u64,
    ) {
        let codeword = fri.domain.evaluate(&polynomial);
        let mut proof_stream = ProofStream::new();
        fri.prove(&codeword, &mut proof_stream).unwrap();

        proof_stream.reset_sponge();
        let mut proof_stream =
            change_size_of_some_fri_response_in_proof_stream_using_seed(proof_stream, rng_seed);

        let verdict = fri.verify(&mut proof_stream);
        let err = verdict.unwrap_err();
        let LdtVerificationError::IncorrectNumberOfRevealedLeaves = err else {
            return Err(TestCaseError::Fail("validation must fail".into()));
        };
    }

    #[must_use]
    fn change_size_of_some_fri_response_in_proof_stream_using_seed(
        mut proof_stream: ProofStream,
        seed: u64,
    ) -> ProofStream {
        let proof_items = proof_stream.items.iter_mut();
        let fri_responses = proof_items.filter_map(fri_response_filter());

        let mut rng = StdRng::seed_from_u64(seed);
        let fri_response = fri_responses.choose(&mut rng).unwrap();
        let queried_leaves = &mut fri_response.queried_leaves;
        let modification_index = rng.random_range(0..queried_leaves.len());
        if rng.random() {
            queried_leaves.remove(modification_index);
        } else {
            queried_leaves.insert(modification_index, rng.random());
        };

        proof_stream
    }

    fn fri_response_filter() -> fn(&mut ProofItem) -> Option<&mut FriResponse> {
        |proof_item| match proof_item {
            ProofItem::FriResponse(fri_response) => Some(fri_response),
            _ => None,
        }
    }

    #[macro_rules_attr::apply(proptest(cases = 50))]
    fn incorrect_authentication_structure_results_in_validation_failure(
        fri: Fri,
        #[strategy(arbitrary_polynomial())] polynomial: XfePoly,
        rng_seed: u64,
    ) {
        let all_authentication_structures_are_trivial =
            fri.num_collinearity_checks >= fri.domain.len();
        if all_authentication_structures_are_trivial {
            return Ok(());
        }

        let codeword = fri.domain.evaluate(&polynomial);
        let mut proof_stream = ProofStream::new();
        fri.prove(&codeword, &mut proof_stream).unwrap();

        proof_stream.reset_sponge();
        let mut proof_stream =
            modify_some_auth_structure_in_proof_stream_using_seed(proof_stream, rng_seed);

        let verdict = fri.verify(&mut proof_stream);
        assert!(let Err(err) = verdict);
        assert!(let LdtVerificationError::BadMerkleAuthenticationPath = err);
    }

    #[must_use]
    fn modify_some_auth_structure_in_proof_stream_using_seed(
        mut proof_stream: ProofStream,
        seed: u64,
    ) -> ProofStream {
        let proof_items = proof_stream.items.iter_mut();
        let auth_structures = proof_items.filter_map(non_trivial_auth_structure_filter());

        let mut rng = StdRng::seed_from_u64(seed);
        let auth_structure = auth_structures.choose(&mut rng).unwrap();
        let modification_index = rng.random_range(0..auth_structure.len());
        match rng.random_range(0..3) {
            0 => _ = auth_structure.remove(modification_index),
            1 => auth_structure.insert(modification_index, rng.random()),
            2 => auth_structure[modification_index] = rng.random(),
            _ => unreachable!(),
        };

        proof_stream
    }

    fn non_trivial_auth_structure_filter()
    -> fn(&mut ProofItem) -> Option<&mut AuthenticationStructure> {
        |proof_item| match proof_item {
            ProofItem::FriResponse(fri_response) if fri_response.auth_structure.is_empty() => None,
            ProofItem::FriResponse(fri_response) => Some(&mut fri_response.auth_structure),
            _ => None,
        }
    }

    #[macro_rules_attr::apply(proptest)]
    fn incorrect_last_round_polynomial_results_in_verification_failure(
        fri: Fri,
        #[strategy(arbitrary_polynomial().no_shrink())] fri_polynomial: XfePoly,
        #[strategy(arbitrary_polynomial_of_degree(#fri.last_round_max_degree() as i64))]
        incorrect_polynomial: XfePoly,
    ) {
        let codeword = fri.domain.evaluate(&fri_polynomial);
        let mut proof_stream = ProofStream::new();
        fri.prove(&codeword, &mut proof_stream).unwrap();

        proof_stream.reset_sponge();
        for item in &mut proof_stream.items {
            if let ProofItem::Polynomial(polynomial) = item {
                *polynomial = incorrect_polynomial.clone();
            }
        }

        assert!(let Err(err) = fri.verify(&mut proof_stream));

        // In some cases, the same authentication path is valid even for
        // differing indices. The simplest such case is if the `fri_polynomial`
        // is constant and there is only one index to check:
        // since all internal nodes in a given layer of the Merkle tree are
        // identical, all authentication paths are identical, too. Ergo, the
        // same authentication structure is valid for any index. The concept
        // generalizes and is therefore not easily filtered out when generating
        // the input to the proptest.
        assert!(matches!(
            err,
            LdtVerificationError::LastRoundPolynomialEvaluationMismatch
                | LdtVerificationError::BadMerkleAuthenticationPath
        ));
    }

    #[macro_rules_attr::apply(proptest)]
    fn codeword_corresponding_to_high_degree_polynomial_results_in_verification_failure(
        fri: Fri,
        #[strategy(Just(#fri.max_degree() as i64 + 1))] _min_fail_deg: i64,
        #[strategy(#_min_fail_deg..2 * #_min_fail_deg)] _degree: i64,
        #[strategy(arbitrary_polynomial_of_degree(#_degree))] poly: XfePoly,
    ) {
        let codeword = fri.domain.evaluate(&poly);
        let mut proof_stream = ProofStream::new();
        fri.prove(&codeword, &mut proof_stream).unwrap();

        proof_stream.reset_sponge();
        assert!(let Err(err) = fri.verify(&mut proof_stream));
        assert!(let LdtVerificationError::LastRoundPolynomialHasTooHighDegree = err);
    }

    #[macro_rules_attr::apply(proptest)]
    fn verifying_arbitrary_proof_does_not_panic(
        fri: Fri,
        #[strategy(arb())] mut proof_stream: ProofStream,
    ) {
        let _verdict = fri.verify(&mut proof_stream);
    }

    #[macro_rules_attr::apply(proptest)]
    fn postscript_is_integral(
        fri: Fri,
        #[strategy(arbitrary_polynomial_of_degree(#fri.max_degree() as i64).no_shrink())]
        poly: XfePoly,
    ) {
        let codeword = fri.domain.evaluate(&poly);
        let mut proof_stream = ProofStream::new();
        fri.prove(&codeword, &mut proof_stream)?;
        proof_stream.reset_sponge();
        assert!(let VerifierPostscript::Fri(postscript) = fri.verify(&mut proof_stream)?);

        // folding challenge is only `None` in last round
        for round in postscript.rounds.iter().dropping_back(1) {
            prop_assert!(round.folding_challenge.is_some());
        }
        let last_round = &postscript.rounds.last().unwrap();
        prop_assert!(last_round.folding_challenge.is_none());

        // initial A and B domain are identical
        let initial_domain = postscript.initial_round.domain;
        prop_assert_eq!(initial_domain, postscript.rounds[0].domain);
        if let Some(round_1) = postscript.rounds.get(1) {
            prop_assert_ne!(initial_domain, round_1.domain);
        }

        // domain of final round and final codeword's length correspond
        let final_domain = postscript.rounds.last().unwrap().domain;
        prop_assert_eq!(final_domain.len(), postscript.last_round_codeword.len());

        // last polynomial & codeword correspond
        let reconstructed_polynomial = final_domain
            .with_offset(bfe!(1))
            .interpolate(&postscript.last_round_codeword);
        prop_assert_eq!(postscript.last_round_polynomial, reconstructed_polynomial);

        // Merkle tree verification of all rounds works out
        assert_merkle_tree_integrity(
            postscript.initial_round.domain,
            postscript.initial_round.indices,
            postscript.initial_round.partial_codeword,
            postscript.initial_round.auth_structure,
            postscript.initial_round.merkle_root,
        )?;
        for round in postscript.rounds {
            assert_merkle_tree_integrity(
                round.domain,
                round.indices,
                round.partial_codeword,
                round.auth_structure,
                round.merkle_root,
            )?;
        }
    }

    fn assert_merkle_tree_integrity(
        domain: ArithmeticDomain,
        indices: Vec<usize>,
        partial_codeword: Vec<XFieldElement>,
        auth_structure: AuthenticationStructure,
        merkle_root: Digest,
    ) -> TestCaseResult {
        let tree_height = domain.len().ilog2();
        let leafs = partial_codeword.into_iter().map(Digest::from);
        let indexed_leafs = indices.into_iter().zip_eq(leafs).collect();
        let inclusion_proof = MerkleTreeInclusionProof {
            tree_height,
            indexed_leafs,
            authentication_structure: auth_structure.clone(),
        };
        prop_assert!(inclusion_proof.verify(merkle_root));

        Ok(())
    }
}
