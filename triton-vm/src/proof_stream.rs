use std::error::Error;
use std::fmt::Display;
use std::marker::PhantomData;

use twenty_first::shared_math::rescue_prime_digest::Digest;
use twenty_first::util_types::algebraic_hasher::AlgebraicHasher;

use crate::bfield_codec::BFieldCodec;
use crate::proof::Proof;
use crate::proof_item::MayBeUncast;

#[derive(Debug, PartialEq, Eq)]
pub struct ProofStream<Item: Clone + BFieldCodec + MayBeUncast, H: AlgebraicHasher> {
    pub items: Vec<Item>,
    items_index: usize,
    _hasher: PhantomData<H>,
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
    #[allow(clippy::new_without_default)]
    pub fn new() -> Self {
        ProofStream {
            items: vec![],
            items_index: 0,
            _hasher: PhantomData,
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
    pub fn from_proof(proof: &Proof) -> anyhow::Result<Self> {
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
            _hasher: PhantomData,
        })
    }

    /// Send a proof item as prover to verifier.
    pub fn enqueue(&mut self, item: &Item) {
        self.items.push(item.clone());
    }

    /// Receive a proof item from prover as verifier.
    pub fn dequeue(&mut self) -> anyhow::Result<Item> {
        let item = self
            .items
            .get(self.items_index)
            .ok_or_else(|| ProofStreamError::new("Could not dequeue, queue empty"))?;

        self.items_index += 1;
        Ok(item.clone())
    }

    pub fn prover_fiat_shamir(&self) -> Digest {
        let mut transcript = vec![];
        for item in self.items.iter() {
            transcript.append(&mut item.encode());
        }
        H::hash_slice(&transcript)
    }

    pub fn verifier_fiat_shamir(&self) -> Digest {
        let mut transcript = vec![];
        for item in self.items[0..self.items_index].iter() {
            transcript.append(&mut item.uncast());
        }
        H::hash_slice(&transcript)
    }
}

#[cfg(test)]
mod proof_stream_typed_tests {
    use itertools::Itertools;
    use num_traits::One;

    use super::*;
    use twenty_first::shared_math::{
        b_field_element::BFieldElement, other::random_elements,
        rescue_prime_regular::RescuePrimeRegular, x_field_element::XFieldElement,
    };

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
        fn decode(str: &[BFieldElement]) -> anyhow::Result<Box<Self>> {
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

    // Property: prover_fiat_shamir() is equivalent to verifier_fiat_shamir() when the entire stream has been read.
    #[test]
    fn prover_verifier_fiat_shamir_test() {
        type H = RescuePrimeRegular;
        let mut proof_stream = ProofStream::<TestItem, H>::new();
        let ps: &mut ProofStream<TestItem, H> = &mut proof_stream;

        let digest_1 = H::hash(&BFieldElement::one());
        ps.enqueue(&TestItem::ManyB(digest_1.values().to_vec()));
        let _ = ps.dequeue();

        assert_eq!(
            ps.prover_fiat_shamir(),
            ps.verifier_fiat_shamir(),
            "prover_fiat_shamir() and verifier_fiat_shamir() are equivalent when the entire stream is read"
        );

        let digest_2 = H::hash(&BFieldElement::one());
        ps.enqueue(&TestItem::ManyB(digest_2.values().to_vec()));

        assert_ne!(
            ps.prover_fiat_shamir(),
            ps.verifier_fiat_shamir(),
            "prover_fiat_shamir() and verifier_fiat_shamir() are different when the stream isn't fully read"
        );

        let _ = ps.dequeue();

        assert_eq!(
            ps.prover_fiat_shamir(),
            ps.verifier_fiat_shamir(),
            "prover_fiat_shamir() and verifier_fiat_shamir() are equivalent when the entire stream is read again",
        );
    }

    #[test]
    fn test_serialize_proof_with_fiat_shamir() {
        type H = RescuePrimeRegular;
        let mut proof_stream = ProofStream::<TestItem, H>::new();
        let manyb1: Vec<BFieldElement> = random_elements(10);
        let manyx: Vec<XFieldElement> = random_elements(13);
        let manyb2: Vec<BFieldElement> = random_elements(11);

        let fs1 = proof_stream.prover_fiat_shamir();
        proof_stream.enqueue(&TestItem::ManyB(manyb1.clone()));
        let fs2 = proof_stream.prover_fiat_shamir();
        proof_stream.enqueue(&TestItem::ManyX(manyx.clone()));
        let fs3 = proof_stream.prover_fiat_shamir();
        proof_stream.enqueue(&TestItem::ManyB(manyb2.clone()));
        let fs4 = proof_stream.prover_fiat_shamir();

        let proof = proof_stream.to_proof();

        let mut proof_stream =
            ProofStream::<TestItem, H>::from_proof(&proof).expect("invalid parsing of proof");

        let fs1_ = proof_stream.verifier_fiat_shamir();
        match proof_stream.dequeue().expect("can't dequeue item").as_bs() {
            TestItem::ManyB(manyb1_) => assert_eq!(manyb1, manyb1_),
            TestItem::ManyX(_) => panic!(),
            TestItem::Uncast(_) => panic!(),
        };
        let fs2_ = proof_stream.verifier_fiat_shamir();
        match proof_stream.dequeue().expect("can't dequeue item").as_xs() {
            TestItem::ManyB(_) => panic!(),
            TestItem::ManyX(manyx_) => assert_eq!(manyx, manyx_),
            TestItem::Uncast(_) => panic!(),
        };
        let fs3_ = proof_stream.verifier_fiat_shamir();
        match proof_stream.dequeue().expect("can't dequeue item").as_bs() {
            TestItem::ManyB(manyb2_) => assert_eq!(manyb2, manyb2_),
            TestItem::ManyX(_) => panic!(),
            TestItem::Uncast(_) => panic!(),
        };
        let fs4_ = proof_stream.verifier_fiat_shamir();

        assert_eq!(fs1, fs1_);
        assert_eq!(fs2, fs2_);
        assert_eq!(fs3, fs3_);
        assert_eq!(fs4, fs4_);
    }
}
