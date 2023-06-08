use std::error::Error;
use std::fmt::Display;

use anyhow::Result;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::b_field_element::BFIELD_ONE;
use twenty_first::shared_math::b_field_element::BFIELD_ZERO;
use twenty_first::shared_math::bfield_codec::BFieldCodec;
use twenty_first::shared_math::other::is_power_of_two;
use twenty_first::shared_math::x_field_element::XFieldElement;
use twenty_first::util_types::algebraic_hasher::AlgebraicHasher;

use crate::proof::Proof;
use crate::proof_item::ProofItem;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProofStream<H>
where
    H: AlgebraicHasher,
{
    pub items: Vec<ProofItem>,
    pub items_index: usize,
    pub sponge_state: H::SpongeState,
}

#[derive(Debug, Clone)]
pub struct ProofStreamError {
    pub message: String,
}

impl ProofStreamError {
    #[allow(clippy::new_ret_no_self)]
    pub fn new(message: &str) -> anyhow::Error {
        anyhow::Error::new(Self {
            message: message.to_string(),
        })
    }
}

impl Display for ProofStreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.message)
    }
}

impl Error for ProofStreamError {}

impl<H> ProofStream<H>
where
    H: AlgebraicHasher,
{
    pub fn new() -> Self {
        ProofStream {
            items: vec![],
            items_index: 0,
            sponge_state: H::init(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    /// The number of items in the proof stream.
    pub fn len(&self) -> usize {
        self.items.len()
    }

    /// The number of field elements required to encode the proof.
    pub fn transcript_length(&self) -> usize {
        let Proof(b_field_elements) = self.into();
        b_field_elements.len()
    }

    fn encode_and_pad_item(item: &ProofItem) -> Vec<BFieldElement> {
        let encoding = item.encode();
        let last_chunk_len = (encoding.len() + 1) % H::RATE;
        let num_padding_zeros = match last_chunk_len {
            0 => 0,
            _ => H::RATE - last_chunk_len,
        };
        [
            encoding,
            vec![BFIELD_ONE],
            vec![BFIELD_ZERO; num_padding_zeros],
        ]
        .concat()
    }

    /// Send a proof item as prover to verifier.
    /// Some items do not need to be included in the Fiat-Shamir heuristic, _i.e._, they do not
    /// need to modify the sponge state. For those items, `include_in_fs_heuristic` should be set
    /// to `false`. For example:
    /// - Merkle authentication paths do not need to be included (hashed) if the root of the tree
    ///     in question was included (hashed) previously.
    /// - If the proof stream is not used to sample any more randomness, _i.e._, after the last
    ///     round of interaction, no further items need to be included.
    pub fn enqueue(&mut self, item: &ProofItem, include_in_fs_heuristic: bool) {
        if include_in_fs_heuristic {
            H::absorb_repeatedly(
                &mut self.sponge_state,
                Self::encode_and_pad_item(item).iter(),
            )
        }
        self.items.push(item.clone());
    }

    /// Receive a proof item from prover as verifier.
    /// See [`ProofStream::enqueue`] for more details.
    pub fn dequeue(&mut self, include_in_fs_heuristic: bool) -> Result<ProofItem> {
        let item = self
            .items
            .get(self.items_index)
            .ok_or_else(|| ProofStreamError::new("Could not dequeue, queue empty"))?;
        if include_in_fs_heuristic {
            H::absorb_repeatedly(
                &mut self.sponge_state,
                Self::encode_and_pad_item(item).iter(),
            )
        }
        self.items_index += 1;
        Ok(item.clone())
    }

    /// Given an `upper_bound` that is a power of 2, produce `num_indices` uniform random numbers
    /// in the interval `[0; upper_bound)`.
    ///
    /// - `upper_bound`: The (non-inclusive) upper bound. Must be a power of two.
    /// - `num_indices`: The number of indices to sample
    pub fn sample_indices(&mut self, upper_bound: usize, num_indices: usize) -> Vec<usize> {
        assert!(is_power_of_two(upper_bound));
        assert!(upper_bound <= BFieldElement::MAX as usize);
        H::sample_indices(&mut self.sponge_state, upper_bound as u32, num_indices)
            .into_iter()
            .map(|i| i as usize)
            .collect()
    }

    /// A thin wrapper around [`H::sample_scalars`].
    pub fn sample_scalars(&mut self, num_scalars: usize) -> Vec<XFieldElement> {
        H::sample_scalars(&mut self.sponge_state, num_scalars)
    }
}

impl<H> Default for ProofStream<H>
where
    H: AlgebraicHasher,
{
    fn default() -> Self {
        Self::new()
    }
}

impl<H> BFieldCodec for ProofStream<H>
where
    H: AlgebraicHasher,
{
    fn decode(sequence: &[BFieldElement]) -> Result<Box<Self>> {
        let items = *Vec::<ProofItem>::decode(sequence)?;
        let proof_stream = ProofStream {
            items,
            items_index: 0,
            sponge_state: H::init(),
        };
        Ok(Box::new(proof_stream))
    }

    fn encode(&self) -> Vec<BFieldElement> {
        self.items.encode()
    }

    fn static_length() -> Option<usize> {
        None
    }
}

impl<H> TryFrom<&Proof> for ProofStream<H>
where
    H: AlgebraicHasher,
{
    type Error = anyhow::Error;

    fn try_from(proof: &Proof) -> Result<Self> {
        let proof_stream = *ProofStream::decode(&proof.0)?;
        Ok(proof_stream)
    }
}

impl<H> From<&ProofStream<H>> for Proof
where
    H: AlgebraicHasher,
{
    fn from(proof_stream: &ProofStream<H>) -> Self {
        Proof(proof_stream.encode())
    }
}

impl<H> From<ProofStream<H>> for Proof
where
    H: AlgebraicHasher,
{
    fn from(proof_stream: ProofStream<H>) -> Self {
        (&proof_stream).into()
    }
}

#[cfg(test)]
mod proof_stream_typed_tests {
    use itertools::Itertools;
    use rand::distributions::Standard;
    use rand::prelude::Distribution;
    use rand::prelude::SeedableRng;
    use rand::prelude::StdRng;
    use rand::random;
    use rand::Rng;
    use rand_core::RngCore;
    use std::collections::VecDeque;
    use twenty_first::shared_math::other::random_elements;
    use twenty_first::shared_math::tip5::Tip5;
    use twenty_first::shared_math::x_field_element::XFieldElement;
    use twenty_first::util_types::merkle_tree::CpuParallel;
    use twenty_first::util_types::merkle_tree::MerkleTree;
    use twenty_first::util_types::merkle_tree_maker::MerkleTreeMaker;

    use crate::proof_item::FriResponse;
    use crate::proof_item::ProofItem;
    use crate::table::master_table::NUM_BASE_COLUMNS;
    use crate::table::master_table::NUM_EXT_COLUMNS;

    use super::*;

    #[test]
    fn test_serialize_proof_with_fiat_shamir() {
        type H = Tip5;

        fn random_elements<T>(seed: u64, n: usize) -> Vec<T>
        where
            Standard: Distribution<T>,
        {
            let rng = StdRng::seed_from_u64(seed);
            rng.sample_iter(Standard).take(n).collect()
        }

        let seed = random();
        let mut rng = StdRng::seed_from_u64(seed);
        println!("seed: {seed}");

        let base_rows = vec![
            random_elements(rng.next_u64(), NUM_BASE_COLUMNS),
            random_elements(rng.next_u64(), NUM_BASE_COLUMNS),
        ];
        let ext_rows = vec![
            random_elements(rng.next_u64(), NUM_EXT_COLUMNS),
            random_elements(rng.next_u64(), NUM_EXT_COLUMNS),
        ];

        let codeword_len = 32;
        let fri_codeword: Vec<XFieldElement> = random_elements(rng.next_u64(), codeword_len);
        let fri_codeword_digests = fri_codeword.iter().map(|&x| x.into()).collect_vec();
        let merkle_tree: MerkleTree<H> = CpuParallel::from_digests(&fri_codeword_digests);
        let root = merkle_tree.get_root();

        let revealed_index = rng.gen_range(0..codeword_len);
        let auth_path = merkle_tree.get_authentication_path(revealed_index);

        let num_revealed_indices = rng.gen_range(1..=codeword_len);
        let revealed_indices = random_elements(rng.next_u64(), num_revealed_indices)
            .into_iter()
            .map(|idx: usize| idx % codeword_len)
            .collect_vec();
        let auth_structure = merkle_tree.get_authentication_structure(&revealed_indices);

        let ood_base_row = random_elements(rng.next_u64(), NUM_BASE_COLUMNS);
        let ood_ext_row = random_elements(rng.next_u64(), NUM_EXT_COLUMNS);
        let combination_elements = random_elements(rng.next_u64(), 5);

        let revealed_elements = revealed_indices
            .iter()
            .map(|&idx| fri_codeword[idx])
            .collect_vec();
        let fri_response = auth_structure
            .clone()
            .into_iter()
            .zip(revealed_elements)
            .collect_vec();
        let fri_response = FriResponse(fri_response);

        let mut sponge_states = VecDeque::new();
        let mut proof_stream = ProofStream::<H>::new();

        sponge_states.push_back(proof_stream.sponge_state.state);
        proof_stream.enqueue(
            &ProofItem::CompressedAuthenticationPaths(auth_structure.clone()),
            false,
        );
        sponge_states.push_back(proof_stream.sponge_state.state);
        proof_stream.enqueue(&ProofItem::MasterBaseTableRows(base_rows.clone()), false);
        sponge_states.push_back(proof_stream.sponge_state.state);
        proof_stream.enqueue(&ProofItem::MasterExtTableRows(ext_rows.clone()), true);
        sponge_states.push_back(proof_stream.sponge_state.state);
        proof_stream.enqueue(&ProofItem::OutOfDomainBaseRow(ood_base_row.clone()), true);
        sponge_states.push_back(proof_stream.sponge_state.state);
        proof_stream.enqueue(&ProofItem::OutOfDomainExtRow(ood_ext_row.clone()), true);
        sponge_states.push_back(proof_stream.sponge_state.state);
        proof_stream.enqueue(&ProofItem::MerkleRoot(root), true);
        sponge_states.push_back(proof_stream.sponge_state.state);
        proof_stream.enqueue(&ProofItem::AuthenticationPath(auth_path.clone()), true);
        sponge_states.push_back(proof_stream.sponge_state.state);
        proof_stream.enqueue(
            &ProofItem::RevealedCombinationElements(combination_elements.clone()),
            true,
        );
        sponge_states.push_back(proof_stream.sponge_state.state);
        proof_stream.enqueue(&ProofItem::FriCodeword(fri_codeword.clone()), true);
        sponge_states.push_back(proof_stream.sponge_state.state);
        proof_stream.enqueue(&ProofItem::FriResponse(fri_response.clone()), true);
        sponge_states.push_back(proof_stream.sponge_state.state);

        let proof = proof_stream.into();
        let mut proof_stream: ProofStream<H> =
            ProofStream::try_from(&proof).expect("invalid parsing of proof");

        assert_eq!(
            sponge_states.pop_front(),
            Some(proof_stream.sponge_state.state)
        );
        match proof_stream.dequeue(false).unwrap() {
            ProofItem::CompressedAuthenticationPaths(auth_structure_) => {
                assert_eq!(auth_structure, auth_structure_)
            }
            _ => panic!(),
        };

        assert_eq!(
            sponge_states.pop_front(),
            Some(proof_stream.sponge_state.state)
        );
        match proof_stream.dequeue(false).unwrap() {
            ProofItem::MasterBaseTableRows(base_rows_) => assert_eq!(base_rows, base_rows_),
            _ => panic!(),
        };

        assert_eq!(
            sponge_states.pop_front(),
            Some(proof_stream.sponge_state.state)
        );
        match proof_stream.dequeue(true).unwrap() {
            ProofItem::MasterExtTableRows(ext_rows_) => assert_eq!(ext_rows, ext_rows_),
            _ => panic!(),
        };

        assert_eq!(
            sponge_states.pop_front(),
            Some(proof_stream.sponge_state.state)
        );
        match proof_stream.dequeue(true).unwrap() {
            ProofItem::OutOfDomainBaseRow(ood_base_row_) => assert_eq!(ood_base_row, ood_base_row_),
            _ => panic!(),
        };

        assert_eq!(
            sponge_states.pop_front(),
            Some(proof_stream.sponge_state.state)
        );
        match proof_stream.dequeue(true).unwrap() {
            ProofItem::OutOfDomainExtRow(ood_ext_row_) => assert_eq!(ood_ext_row, ood_ext_row_),
            _ => panic!(),
        };

        assert_eq!(
            sponge_states.pop_front(),
            Some(proof_stream.sponge_state.state)
        );
        match proof_stream.dequeue(true).unwrap() {
            ProofItem::MerkleRoot(root_) => assert_eq!(root, root_),
            _ => panic!(),
        };

        assert_eq!(
            sponge_states.pop_front(),
            Some(proof_stream.sponge_state.state)
        );
        match proof_stream.dequeue(true).unwrap() {
            ProofItem::AuthenticationPath(auth_path_) => assert_eq!(auth_path, auth_path_),
            _ => panic!(),
        };

        assert_eq!(
            sponge_states.pop_front(),
            Some(proof_stream.sponge_state.state)
        );
        match proof_stream.dequeue(true).unwrap() {
            ProofItem::RevealedCombinationElements(combination_elements_) => {
                assert_eq!(combination_elements, combination_elements_)
            }
            _ => panic!(),
        };

        assert_eq!(
            sponge_states.pop_front(),
            Some(proof_stream.sponge_state.state)
        );
        match proof_stream.dequeue(true).unwrap() {
            ProofItem::FriCodeword(fri_codeword_) => assert_eq!(fri_codeword, fri_codeword_),
            _ => panic!(),
        };

        assert_eq!(
            sponge_states.pop_front(),
            Some(proof_stream.sponge_state.state)
        );
        match proof_stream.dequeue(true).unwrap() {
            ProofItem::FriResponse(fri_response_) => assert_eq!(fri_response, fri_response_),
            _ => panic!(),
        };

        assert_eq!(
            sponge_states.pop_front(),
            Some(proof_stream.sponge_state.state)
        );
        assert_eq!(sponge_states.len(), 0);
    }

    #[test]
    fn enqueue_dequeue_verify_partial_authentication_structure() {
        type H = Tip5;

        let tree_height = 8;
        let num_leaves = 1 << tree_height;
        let leaf_values: Vec<XFieldElement> = random_elements(num_leaves);
        let leaf_digests = leaf_values.iter().map(|&xfe| xfe.into()).collect_vec();
        let merkle_tree: MerkleTree<H> = CpuParallel::from_digests(&leaf_digests);
        let indices_to_check = vec![5, 173, 175, 167, 228, 140, 252, 149, 232, 182, 5, 5, 182];
        let authentication_structure = merkle_tree.get_authentication_structure(&indices_to_check);
        let fri_response_content = authentication_structure
            .into_iter()
            .zip_eq(indices_to_check.iter())
            .map(|(path, &idx)| (path, leaf_values[idx]))
            .collect_vec();
        let fri_response = FriResponse(fri_response_content);

        let mut proof_stream = ProofStream::<H>::new();
        proof_stream.enqueue(&ProofItem::FriResponse(fri_response), false);

        // TODO: Also check that deserializing from Proof works here.

        let maybe_same_fri_response = proof_stream
            .dequeue(false)
            .unwrap()
            .as_fri_response()
            .unwrap();
        let FriResponse(dequeued_paths_and_leafs) = maybe_same_fri_response;
        let (paths, leaf_values): (Vec<_>, Vec<_>) = dequeued_paths_and_leafs.into_iter().unzip();
        let maybe_same_leaf_digests = leaf_values.iter().map(|&xfe| xfe.into()).collect_vec();
        let verdict = MerkleTree::<H>::verify_authentication_structure_from_leaves(
            merkle_tree.get_root(),
            tree_height,
            &indices_to_check,
            &maybe_same_leaf_digests,
            &paths,
        );
        assert!(verdict);
    }
}
