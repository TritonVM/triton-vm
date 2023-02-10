use std::error::Error;
use std::fmt::Display;

use anyhow::Result;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::other::{is_power_of_two, roundup_nearest_multiple};
use twenty_first::shared_math::x_field_element::{XFieldElement, EXTENSION_DEGREE};
use twenty_first::util_types::algebraic_hasher::{AlgebraicHasher, RATE};

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
    pub fn new(sponge_state: H::SpongeState) -> Self {
        ProofStream {
            items: vec![],
            items_index: 0,
            sponge_state,
        }
    }

    /// Reset the counter counting how many items were read. For testing purposes, so
    /// we don't have to re-run tests needlessly.
    pub fn reset_for_verifier(&mut self) {
        self.items_index = 0;
    }

    pub fn is_empty(&self) -> bool {
        self.items.is_empty()
    }

    pub fn len(&self) -> usize {
        self.items.len()
    }

    pub fn transcript_length(&self) -> usize {
        self.to_proof().0.len()
    }

    /// Convert the proof stream (or its transcript really) into a
    /// Proof.
    pub fn to_proof(&self) -> Proof {
        let mut bfes = vec![];
        for item in self.items.iter() {
            bfes.append(&mut item.encode());
        }
        Proof(bfes)
    }

    /// Convert the proof into a proof stream for the verifier.
    pub fn from_proof(proof: &Proof, sponge_state: H::SpongeState) -> Result<Self> {
        let mut index = 0;
        let mut items = vec![];
        while index < proof.0.len() {
            let len = proof.0[index].value() as usize;
            if proof.0.len() < index + 1 + len {
                return Err(ProofStreamError::new(&format!(
                    "failed to decode proof; wrong length: have {} but expected {}",
                    proof.0.len(),
                    index + 1 + len
                )));
            }
            let str = &proof.0[index..(index + 1 + len)];
            let maybe_item = Item::decode(str);
            match maybe_item {
                Ok(item) => {
                    items.push(*item);
                }
                Err(e) => {
                    return Err(e);
                }
            }
            index += 1 + len;
        }
        Ok(ProofStream {
            items,
            items_index: 0,
            sponge_state,
        })
    }

    /// Send a proof item as prover to verifier.
    pub fn enqueue(&mut self, item: &Item) {
        // TODO: Absorb item into sponge.
        self.items.push(item.clone());
    }

    /// Receive a proof item from prover as verifier.
    pub fn dequeue(&mut self) -> Result<Item> {
        // TODO: Absorb item into sponge.
        let item = self
            .items
            .get(self.items_index)
            .ok_or_else(|| ProofStreamError::new("Could not dequeue, queue empty"))?;

        self.items_index += 1;
        Ok(item.clone())
    }

    // // TODO: Provide challenge by squeezing instead.
    // pub fn prover_fiat_shamir(&self) -> Digest {
    //     let mut transcript = vec![];
    //     for item in self.items.iter() {
    //         transcript.append(&mut item.encode());
    //     }
    //     H::hash_varlen(&transcript)
    // }

    // // TODO: Provide challenge by squeezing instead.
    // pub fn verifier_fiat_shamir(&self) -> Digest {
    //     let mut transcript = vec![];
    //     for item in self.items[0..self.items_index].iter() {
    //         transcript.append(&mut item.uncast());
    //     }
    //     H::hash_varlen(&transcript)
    // }

    /// Given a sponge state and an `upper_bound` that is a power of two,
    /// produce `num_indices` uniform random numbers (sample indices) in
    /// the interval `[0; upper_bound)`.
    ///
    /// - `state`: A `Self::SpongeState`
    /// - `upper_bound`: The (non-inclusive) upper bound (a power of two)
    /// - `num_indices`: The number of sample indices
    pub fn sample_indices(
        state: &mut H::SpongeState,
        upper_bound: usize,
        num_indices: usize,
    ) -> Vec<usize> {
        assert!(is_power_of_two(upper_bound));
        assert!(upper_bound <= BFieldElement::MAX as usize);
        let num_squeezes = roundup_nearest_multiple(num_indices, RATE) / RATE;
        (0..num_squeezes)
            .flat_map(|_| H::squeeze(state))
            .take(num_indices)
            .map(|elem| elem.value() as usize % upper_bound)
            .collect()
    }

    pub fn sample_weights(state: &mut H::SpongeState, num_weights: usize) -> Vec<XFieldElement> {
        let num_squeezes = roundup_nearest_multiple(num_weights * EXTENSION_DEGREE, RATE) / RATE;
        (0..num_squeezes)
            .map(|_| H::squeeze(state))
            .flat_map(|elems| {
                vec![
                    XFieldElement::new([elems[0], elems[1], elems[2]]),
                    XFieldElement::new([elems[3], elems[4], elems[5]]),
                    XFieldElement::new([elems[6], elems[7], elems[8]]),
                    // spill 1 element, elems[9], per squeeze
                ]
            })
            .collect()
    }
}

#[cfg(test)]
mod proof_stream_typed_tests {
    use itertools::Itertools;
    use twenty_first::shared_math::b_field_element::BFieldElement;
    use twenty_first::shared_math::other::random_elements;
    use twenty_first::shared_math::rescue_prime_regular::RescuePrimeRegular;
    use twenty_first::shared_math::x_field_element::XFieldElement;
    use twenty_first::util_types::algebraic_hasher::SpongeHasher;
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
        type H = RescuePrimeRegular;
        let sponge_state = H::init();
        let mut proof_stream = ProofStream::<TestItem, H>::new(sponge_state);
        let manyb1: Vec<BFieldElement> = random_elements(10);
        let manyx: Vec<XFieldElement> = random_elements(13);
        let manyb2: Vec<BFieldElement> = random_elements(11);

        let fs1 = proof_stream.sponge_state.state.clone();
        proof_stream.enqueue(&TestItem::ManyB(manyb1.clone()));
        let fs2 = proof_stream.sponge_state.state.clone();
        proof_stream.enqueue(&TestItem::ManyX(manyx.clone()));
        let fs3 = proof_stream.sponge_state.state.clone();
        proof_stream.enqueue(&TestItem::ManyB(manyb2.clone()));
        let fs4 = proof_stream.sponge_state.state.clone();

        let proof = proof_stream.to_proof();

        let another_sponge_state = H::init();
        let mut proof_stream = ProofStream::<TestItem, H>::from_proof(&proof, another_sponge_state)
            .expect("invalid parsing of proof");

        let fs1_ = proof_stream.sponge_state.state.clone();
        match proof_stream.dequeue().expect("can't dequeue item").as_bs() {
            TestItem::ManyB(manyb1_) => assert_eq!(manyb1, manyb1_),
            TestItem::ManyX(_) => panic!(),
            TestItem::Uncast(_) => panic!(),
        };
        let fs2_ = proof_stream.sponge_state.state.clone();
        match proof_stream.dequeue().expect("can't dequeue item").as_xs() {
            TestItem::ManyB(_) => panic!(),
            TestItem::ManyX(manyx_) => assert_eq!(manyx, manyx_),
            TestItem::Uncast(_) => panic!(),
        };
        let fs3_ = proof_stream.sponge_state.state.clone();
        match proof_stream.dequeue().expect("can't dequeue item").as_bs() {
            TestItem::ManyB(manyb2_) => assert_eq!(manyb2, manyb2_),
            TestItem::ManyX(_) => panic!(),
            TestItem::Uncast(_) => panic!(),
        };
        let fs4_ = proof_stream.sponge_state.state.clone();

        assert_eq!(fs1, fs1_);
        assert_eq!(fs2, fs2_);
        assert_eq!(fs3, fs3_);
        assert_eq!(fs4, fs4_);
    }

    #[test]
    fn enqueue_dequeue_verify_partial_authentication_structure() {
        type H = RescuePrimeRegular;

        let leaf_values: Vec<XFieldElement> = random_elements(256);
        let leaf_digests = leaf_values.iter().map(H::hash).collect_vec();
        let merkle_tree: MerkleTree<H, _> = CpuParallel::from_digests(&leaf_digests);
        let indices_to_check = vec![5, 173, 175, 167, 228, 140, 252, 149, 232, 182, 5, 5, 182];
        let authentication_structure = merkle_tree.get_authentication_structure(&indices_to_check);
        let fri_response_content = authentication_structure
            .iter()
            .zip_eq(indices_to_check.iter())
            .map(|(path, &idx)| (path.to_owned(), leaf_values[idx].clone()))
            .collect_vec();
        let fri_response = FriResponse(fri_response_content);

        let sponge_state = H::init();
        let mut proof_stream = ProofStream::<ProofItem, H>::new(sponge_state);
        proof_stream.enqueue(&ProofItem::FriResponse(fri_response.clone()));

        // TODO: Also check that deserializing from Proof works here.

        let maybe_same_fri_response = proof_stream.dequeue().unwrap().as_fri_response().unwrap();
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
