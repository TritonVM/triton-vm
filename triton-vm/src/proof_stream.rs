use std::error::Error;
use std::fmt::Display;

use anyhow::bail;
use anyhow::Result;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::b_field_element::BFIELD_ONE;
use twenty_first::shared_math::b_field_element::BFIELD_ZERO;
use twenty_first::shared_math::other::is_power_of_two;
use twenty_first::shared_math::x_field_element::XFieldElement;
use twenty_first::util_types::algebraic_hasher::AlgebraicHasher;

use crate::bfield_codec::BFieldCodec;
use crate::proof::Proof;
use crate::proof_item::MayBeUncast;

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ProofStream<Item, H>
where
    Item: Clone + BFieldCodec + MayBeUncast,
    H: AlgebraicHasher,
{
    pub items: Vec<Item>,
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

impl<Item, H> ProofStream<Item, H>
where
    Item: Clone + BFieldCodec + MayBeUncast,
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

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn transcript_length(&self) -> usize {
        let Proof(b_field_elements) = self.to_proof();
        b_field_elements.len()
    }

    /// Convert the proof stream, _i.e._, the transcript, into a Proof.
    pub fn to_proof(&self) -> Proof {
        let mut bfes = vec![];
        for item in self.items.iter() {
            bfes.append(&mut item.encode());
        }
        Proof(bfes)
    }

    /// Convert the proof into a proof stream for the verifier.
    pub fn from_proof(proof: &Proof) -> Result<Self> {
        let proof_len = proof.0.len();
        let mut index = 0;
        let mut items = vec![];
        while index < proof.0.len() {
            let next_item_len = proof.0[index].value() as usize;
            let next_index = index + 1 + next_item_len;
            if proof_len < next_index {
                bail!(ProofStreamError::new(&format!(
                    "failed to decode proof; wrong length: \
                    have {proof_len} but expected {next_index}"
                )));
            }
            let str = &proof.0[index..(next_index)];
            let item = Item::decode(str)?;
            items.push(*item);
            index = next_index;
        }
        Ok(ProofStream {
            items,
            items_index: 0,
            sponge_state: H::init(),
        })
    }

    fn encode_and_pad_item(item: &Item) -> Vec<BFieldElement> {
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
    pub fn enqueue(&mut self, item: &Item, include_in_fs_heuristic: bool) {
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
    pub fn dequeue(&mut self, include_in_fs_heuristic: bool) -> Result<Item> {
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

impl<Item, H> Default for ProofStream<Item, H>
where
    Item: Clone + BFieldCodec + MayBeUncast,
    H: AlgebraicHasher,
{
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod proof_stream_typed_tests {
    use itertools::Itertools;
    use twenty_first::shared_math::b_field_element::BFieldElement;
    use twenty_first::shared_math::other::random_elements;
    use twenty_first::shared_math::tip5::Tip5;
    use twenty_first::shared_math::x_field_element::XFieldElement;
    use twenty_first::util_types::merkle_tree::CpuParallel;
    use twenty_first::util_types::merkle_tree::MerkleTree;
    use twenty_first::util_types::merkle_tree_maker::MerkleTreeMaker;

    use crate::proof_item::FriResponse;
    use crate::proof_item::ProofItem;

    use super::*;

    #[derive(Clone, Debug, PartialEq)]
    enum TestItem {
        ManyB(Vec<BFieldElement>),
        ManyX(Vec<XFieldElement>),
        Uncast(Vec<BFieldElement>),
    }

    impl MayBeUncast for TestItem {
        fn uncast(&self) -> Vec<BFieldElement> {
            if let Self::Uncast(vector) = self {
                let mut str = vec![];
                str.push(BFieldElement::new(vector.len().try_into().unwrap()));
                str.append(&mut vector.clone());
                str
            } else {
                self.encode()
            }
        }
    }

    impl TestItem {
        pub fn as_bs(&self) -> Self {
            match self {
                Self::Uncast(bs) => Self::ManyB(bs.to_vec()),
                _ => panic!("can only cast from Uncast"),
            }
        }

        pub fn as_xs(&self) -> Self {
            match self {
                Self::Uncast(bs) => Self::ManyX(
                    bs.chunks(3)
                        .collect_vec()
                        .into_iter()
                        .map(|bbb| {
                            XFieldElement::new(
                                bbb.try_into()
                                    .expect("cannot unwrap chunk of 3 (?) BFieldElements"),
                            )
                        })
                        .collect_vec(),
                ),
                _ => panic!("can only cast from Uncast"),
            }
        }
    }

    impl BFieldCodec for TestItem {
        fn decode(str: &[BFieldElement]) -> Result<Box<Self>> {
            let maybe_element_zero = str.get(0);
            match maybe_element_zero {
                None => Err(ProofStreamError::new(
                    "trying to decode empty string into test item",
                )),
                Some(bfe) => {
                    if str.len() != 1 + (bfe.value() as usize) {
                        Err(ProofStreamError::new("length mismatch"))
                    } else {
                        Ok(Box::new(Self::Uncast(str[1..].to_vec())))
                    }
                }
            }
        }

        fn encode(&self) -> Vec<BFieldElement> {
            let mut vect = vec![];
            match self {
                Self::ManyB(bs) => {
                    for b in bs {
                        vect.append(&mut b.encode());
                    }
                }
                Self::ManyX(xs) => {
                    for x in xs {
                        vect.append(&mut x.encode());
                    }
                }
                Self::Uncast(bs) => {
                    for b in bs {
                        vect.append(&mut b.encode());
                    }
                }
            }
            vect.insert(0, BFieldElement::new(vect.len().try_into().unwrap()));

            vect
        }
    }

    #[test]
    fn test_serialize_proof_with_fiat_shamir() {
        type H = Tip5;
        let mut proof_stream: ProofStream<_, H> = ProofStream::new();
        let manyb1: Vec<BFieldElement> = random_elements(10);
        let manyx: Vec<XFieldElement> = random_elements(13);
        let manyb2: Vec<BFieldElement> = random_elements(11);

        let fs1 = proof_stream.sponge_state.state;
        proof_stream.enqueue(&TestItem::ManyB(manyb1.clone()), false);
        let fs2 = proof_stream.sponge_state.state;
        proof_stream.enqueue(&TestItem::ManyX(manyx.clone()), true);
        let fs3 = proof_stream.sponge_state.state;
        proof_stream.enqueue(&TestItem::ManyB(manyb2.clone()), true);
        let fs4 = proof_stream.sponge_state.state;

        let proof = proof_stream.to_proof();

        let mut proof_stream: ProofStream<TestItem, H> =
            ProofStream::from_proof(&proof).expect("invalid parsing of proof");

        let fs1_ = proof_stream.sponge_state.state;
        match proof_stream
            .dequeue(false)
            .expect("can't dequeue item")
            .as_bs()
        {
            TestItem::ManyB(manyb1_) => assert_eq!(manyb1, manyb1_),
            TestItem::ManyX(_) => panic!(),
            TestItem::Uncast(_) => panic!(),
        };
        let fs2_ = proof_stream.sponge_state.state;
        match proof_stream
            .dequeue(true)
            .expect("can't dequeue item")
            .as_xs()
        {
            TestItem::ManyB(_) => panic!(),
            TestItem::ManyX(manyx_) => assert_eq!(manyx, manyx_),
            TestItem::Uncast(_) => panic!(),
        };
        let fs3_ = proof_stream.sponge_state.state;
        match proof_stream
            .dequeue(true)
            .expect("can't dequeue item")
            .as_bs()
        {
            TestItem::ManyB(manyb2_) => assert_eq!(manyb2, manyb2_),
            TestItem::ManyX(_) => panic!(),
            TestItem::Uncast(_) => panic!(),
        };
        let fs4_ = proof_stream.sponge_state.state;

        assert_eq!(fs1, fs1_);
        assert_eq!(fs2, fs2_);
        assert_eq!(fs3, fs3_);
        assert_eq!(fs4, fs4_);
    }

    #[test]
    fn enqueue_dequeue_verify_partial_authentication_structure() {
        type H = Tip5;

        let leaf_values: Vec<XFieldElement> = random_elements(256);
        let leaf_digests = leaf_values.iter().map(H::hash).collect_vec();
        let merkle_tree: MerkleTree<H, _> = CpuParallel::from_digests(&leaf_digests);
        let indices_to_check = vec![5, 173, 175, 167, 228, 140, 252, 149, 232, 182, 5, 5, 182];
        let authentication_structure = merkle_tree.get_authentication_structure(&indices_to_check);
        let fri_response_content = authentication_structure
            .iter()
            .zip_eq(indices_to_check.iter())
            .map(|(path, &idx)| (path.to_owned(), leaf_values[idx]))
            .collect_vec();
        let fri_response = FriResponse(fri_response_content);

        let mut proof_stream = ProofStream::<ProofItem, H>::new();
        proof_stream.enqueue(&ProofItem::FriResponse(fri_response), true);

        // TODO: Also check that deserializing from Proof works here.

        let maybe_same_fri_response = proof_stream
            .dequeue(true)
            .unwrap()
            .as_fri_response()
            .unwrap();
        let FriResponse(dequeued_paths_and_leafs) = maybe_same_fri_response;
        let (paths, leaf_values): (Vec<_>, Vec<_>) = dequeued_paths_and_leafs.into_iter().unzip();
        let maybe_same_leaf_digests = leaf_values.iter().map(H::hash).collect_vec();
        let path_digest_pairs = paths
            .into_iter()
            .zip_eq(maybe_same_leaf_digests)
            .collect_vec();
        assert_eq!(indices_to_check.len(), path_digest_pairs.len());
        MerkleTree::<H, CpuParallel>::verify_authentication_structure(
            merkle_tree.get_root(),
            &indices_to_check,
            &path_digest_pairs,
        );
    }
}
