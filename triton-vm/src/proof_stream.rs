use arbitrary::Arbitrary;
use twenty_first::prelude::*;

use crate::error::ProofStreamError;
use crate::proof::Proof;
use crate::proof_item::ProofItem;

#[derive(Debug, Default, Clone, Eq, PartialEq, BFieldCodec, Arbitrary)]
pub struct ProofStream {
    pub items: Vec<ProofItem>,

    #[bfield_codec(ignore)]
    pub items_index: usize,

    #[bfield_codec(ignore)]
    pub sponge: Tip5,
}

impl ProofStream {
    pub fn new() -> Self {
        ProofStream {
            items: vec![],
            items_index: 0,
            sponge: Tip5::init(),
        }
    }

    /// The number of field elements required to encode the proof.
    pub fn transcript_length(&self) -> usize {
        let Proof(b_field_elements) = self.into();
        b_field_elements.len()
    }

    /// Alters the Fiat-Shamir's sponge state with the encoding of the given
    /// item. Does _not_ record the given item in the proof stream.
    /// This is useful for items that are not sent to the verifier, _e.g._, the
    /// [`Claim`](crate::proof::Claim).
    ///
    /// See also [`Self::enqueue()`] and [`Self::dequeue()`].
    pub fn alter_fiat_shamir_state_with(&mut self, item: &impl BFieldCodec) {
        self.sponge.pad_and_absorb_all(&item.encode())
    }

    /// Send a proof item as prover to verifier.
    /// Some items do not need to be included in the Fiat-Shamir heuristic,
    /// _i.e._, they do not need to modify the sponge state. For those
    /// items, namely those that evaluate to `false` according to
    /// [`ProofItem::include_in_fiat_shamir_heuristic`], the sponge state is not
    /// modified.
    /// For example:
    /// - Merkle authentication structure do not need to be hashed if the root
    ///   of the tree in question was hashed previously.
    /// - If the proof stream is not used to sample any more randomness, _i.e._,
    ///   after the last round of interaction, no further items need to be
    ///   hashed.
    pub fn enqueue(&mut self, item: ProofItem) {
        if item.include_in_fiat_shamir_heuristic() {
            self.alter_fiat_shamir_state_with(&item);
        }
        self.items.push(item);
    }

    /// Receive a proof item from prover as verifier.
    /// See [`ProofStream::enqueue`] for more details.
    pub fn dequeue(&mut self) -> Result<ProofItem, ProofStreamError> {
        let Some(item) = self.items.get(self.items_index) else {
            return Err(ProofStreamError::EmptyQueue);
        };
        let item = item.to_owned();
        if item.include_in_fiat_shamir_heuristic() {
            self.alter_fiat_shamir_state_with(&item);
        }
        self.items_index += 1;
        Ok(item)
    }

    /// Given an `upper_bound` that is a power of 2, produce `num_indices`
    /// uniform random numbers in the interval `[0; upper_bound)`.
    ///
    /// - `upper_bound`: The (non-inclusive) upper bound. Must be a power of
    ///   two.
    /// - `num_indices`: The number of indices to sample
    pub fn sample_indices(&mut self, upper_bound: usize, num_indices: usize) -> Vec<usize> {
        assert!(upper_bound.is_power_of_two());
        assert!(upper_bound <= BFieldElement::MAX as usize);
        self.sponge
            .sample_indices(upper_bound as u32, num_indices)
            .into_iter()
            .map(|i| i as usize)
            .collect()
    }

    /// A thin wrapper around [`Tip5::sample_scalars`].
    pub fn sample_scalars(&mut self, num_scalars: usize) -> Vec<XFieldElement> {
        self.sponge.sample_scalars(num_scalars)
    }
}

impl TryFrom<&Proof> for ProofStream {
    type Error = ProofStreamError;

    fn try_from(proof: &Proof) -> Result<Self, ProofStreamError> {
        let proof_stream = *ProofStream::decode(&proof.0)?;
        Ok(proof_stream)
    }
}

impl From<&ProofStream> for Proof {
    fn from(proof_stream: &ProofStream) -> Self {
        Proof(proof_stream.encode())
    }
}

impl From<ProofStream> for Proof {
    fn from(proof_stream: ProofStream) -> Self {
        (&proof_stream).into()
    }
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use std::collections::VecDeque;

    use assert2::assert;
    use assert2::let_assert;
    use itertools::Itertools;
    use proptest::collection::vec;
    use proptest_arbitrary_interop::arb;
    use test_strategy::proptest;
    use twenty_first::math::other::random_elements;

    use crate::proof_item::FriResponse;
    use crate::proof_item::ProofItem;
    use crate::shared_tests::LeavedMerkleTreeTestData;
    use crate::table::AuxiliaryRow;
    use crate::table::MainRow;
    use crate::table::QuotientSegments;

    use super::*;

    #[proptest]
    fn serialize_proof_with_fiat_shamir(
        #[strategy(vec(arb(), 2..100))] main_rows: Vec<MainRow<BFieldElement>>,
        #[strategy(vec(arb(), 2..100))] aux_rows: Vec<AuxiliaryRow>,
        #[strategy(arb())] ood_main_row: Box<MainRow<XFieldElement>>,
        #[strategy(arb())] ood_aux_row: Box<AuxiliaryRow>,
        #[strategy(arb())] quot_elements: Vec<QuotientSegments>,
        leaved_merkle_tree: LeavedMerkleTreeTestData,
    ) {
        let auth_structure = leaved_merkle_tree.auth_structure.clone();
        let root = leaved_merkle_tree.root();
        let fri_codeword = leaved_merkle_tree.leaves().to_owned();
        let fri_response = leaved_merkle_tree.into_fri_response();

        let mut sponge_states = VecDeque::new();
        let mut proof_stream = ProofStream::new();

        sponge_states.push_back(proof_stream.sponge.state);
        proof_stream.enqueue(ProofItem::AuthenticationStructure(auth_structure.clone()));
        sponge_states.push_back(proof_stream.sponge.state);
        proof_stream.enqueue(ProofItem::MasterMainTableRows(main_rows.clone()));
        sponge_states.push_back(proof_stream.sponge.state);
        proof_stream.enqueue(ProofItem::MasterAuxTableRows(aux_rows.clone()));
        sponge_states.push_back(proof_stream.sponge.state);
        proof_stream.enqueue(ProofItem::OutOfDomainMainRow(ood_main_row.clone()));
        sponge_states.push_back(proof_stream.sponge.state);
        proof_stream.enqueue(ProofItem::OutOfDomainAuxRow(ood_aux_row.clone()));
        sponge_states.push_back(proof_stream.sponge.state);
        proof_stream.enqueue(ProofItem::MerkleRoot(root));
        sponge_states.push_back(proof_stream.sponge.state);
        proof_stream.enqueue(ProofItem::QuotientSegmentsElements(quot_elements.clone()));
        sponge_states.push_back(proof_stream.sponge.state);
        proof_stream.enqueue(ProofItem::FriCodeword(fri_codeword.clone()));
        sponge_states.push_back(proof_stream.sponge.state);
        proof_stream.enqueue(ProofItem::FriResponse(fri_response.clone()));
        sponge_states.push_back(proof_stream.sponge.state);

        let proof = proof_stream.into();
        let mut proof_stream = ProofStream::try_from(&proof).unwrap();

        assert!(sponge_states.pop_front() == Some(proof_stream.sponge.state));
        let_assert!(Ok(proof_item) = proof_stream.dequeue());
        let_assert!(ProofItem::AuthenticationStructure(auth_structure_) = proof_item);
        assert!(auth_structure == auth_structure_);

        assert!(sponge_states.pop_front() == Some(proof_stream.sponge.state));
        let_assert!(Ok(ProofItem::MasterMainTableRows(main_rows_)) = proof_stream.dequeue());
        assert!(main_rows == main_rows_);

        assert!(sponge_states.pop_front() == Some(proof_stream.sponge.state));
        let_assert!(Ok(ProofItem::MasterAuxTableRows(aux_rows_)) = proof_stream.dequeue());
        assert!(aux_rows == aux_rows_);

        assert!(sponge_states.pop_front() == Some(proof_stream.sponge.state));
        let_assert!(Ok(ProofItem::OutOfDomainMainRow(ood_main_row_)) = proof_stream.dequeue());
        assert!(ood_main_row == ood_main_row_);

        assert!(sponge_states.pop_front() == Some(proof_stream.sponge.state));
        let_assert!(Ok(ProofItem::OutOfDomainAuxRow(ood_aux_row_)) = proof_stream.dequeue());
        assert!(ood_aux_row == ood_aux_row_);

        assert!(sponge_states.pop_front() == Some(proof_stream.sponge.state));
        let_assert!(Ok(ProofItem::MerkleRoot(root_)) = proof_stream.dequeue());
        assert!(root == root_);

        assert!(sponge_states.pop_front() == Some(proof_stream.sponge.state));
        let_assert!(Ok(proof_item) = proof_stream.dequeue());
        let_assert!(ProofItem::QuotientSegmentsElements(quot_elements_) = proof_item);
        assert!(quot_elements == quot_elements_);

        assert!(sponge_states.pop_front() == Some(proof_stream.sponge.state));
        let_assert!(Ok(ProofItem::FriCodeword(fri_codeword_)) = proof_stream.dequeue());
        assert!(fri_codeword == fri_codeword_);

        assert!(sponge_states.pop_front() == Some(proof_stream.sponge.state));
        let_assert!(Ok(ProofItem::FriResponse(fri_response_)) = proof_stream.dequeue());
        assert!(fri_response == fri_response_);

        assert!(sponge_states.pop_front() == Some(proof_stream.sponge.state));
        assert!(0 == sponge_states.len());
    }

    #[test]
    fn enqueue_dequeue_verify_partial_authentication_structure() {
        let tree_height = 8;
        let num_leaves = 1 << tree_height;
        let leaf_values: Vec<XFieldElement> = random_elements(num_leaves);
        let leaf_digests = leaf_values.iter().map(|&xfe| xfe.into()).collect_vec();
        let merkle_tree = MerkleTree::par_new(&leaf_digests).unwrap();
        let indices_to_check = vec![5, 173, 175, 167, 228, 140, 252, 149, 232, 182, 5, 5, 182];
        let auth_structure = merkle_tree
            .authentication_structure(&indices_to_check)
            .unwrap();
        let revealed_leaves = indices_to_check
            .iter()
            .map(|&idx| leaf_values[idx])
            .collect_vec();
        let fri_response = FriResponse {
            auth_structure,
            revealed_leaves,
        };

        let mut proof_stream = ProofStream::new();
        proof_stream.enqueue(ProofItem::FriResponse(fri_response));

        // TODO: Also check that deserializing from Proof works here.

        let proof_item = proof_stream.dequeue().unwrap();
        let maybe_same_fri_response = proof_item.try_into_fri_response().unwrap();
        let FriResponse {
            auth_structure,
            revealed_leaves,
        } = maybe_same_fri_response;
        let maybe_same_leaf_digests = revealed_leaves.iter().map(|&xfe| xfe.into()).collect_vec();
        let indexed_leafs = indices_to_check
            .into_iter()
            .zip_eq(maybe_same_leaf_digests)
            .collect();

        let inclusion_proof = MerkleTreeInclusionProof {
            tree_height,
            indexed_leafs,
            authentication_structure: auth_structure,
        };
        assert!(inclusion_proof.verify(merkle_tree.root()));
    }

    #[test]
    fn dequeuing_from_empty_stream_fails() {
        let mut proof_stream = ProofStream::new();
        let_assert!(Err(ProofStreamError::EmptyQueue) = proof_stream.dequeue());
    }

    #[test]
    fn dequeuing_more_items_than_have_been_enqueued_fails() {
        let mut proof_stream = ProofStream::new();
        proof_stream.enqueue(ProofItem::FriCodeword(vec![]));
        proof_stream.enqueue(ProofItem::Log2PaddedHeight(7));

        let_assert!(Ok(_) = proof_stream.dequeue());
        let_assert!(Ok(_) = proof_stream.dequeue());
        let_assert!(Err(ProofStreamError::EmptyQueue) = proof_stream.dequeue());
    }

    #[test]
    fn encoded_length_of_prove_stream_is_not_known_at_compile_time() {
        assert!(ProofStream::static_length().is_none());
    }
}
