use std::error::Error;
use std::fmt::Display;
use std::marker::PhantomData;

use twenty_first::shared_math::{b_field_element::BFieldElement, rescue_prime_digest::Digest};
use twenty_first::util_types::algebraic_hasher::AlgebraicHasher;

use crate::proof_item::BFieldCodec;

#[derive(Debug, PartialEq, Eq)]
pub struct ProofStream<Item, H: AlgebraicHasher> {
    items: Vec<Item>,
    items_index: usize,
    _hasher: PhantomData<H>,
}

impl<Item, H: AlgebraicHasher> Default for ProofStream<Item, H> {
    fn default() -> Self {
        Self {
            items: vec![],
            items_index: 0,
            _hasher: PhantomData,
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProofStreamError {
    pub message: String,
}

impl ProofStreamError {
    pub fn new(message: &str) -> Self {
        Self {
            message: message.to_string(),
        }
    }

    pub fn boxed(message: &str) -> Box<dyn Error> {
        Box::new(Self::new(message))
    }
}

impl Display for ProofStreamError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.message)
    }
}

impl Error for ProofStreamError {}

pub struct Proof(Vec<BFieldElement>);

impl<Item, H> ProofStream<Item, H>
where
    Item: IntoIterator<Item = BFieldElement> + Clone + BFieldCodec,
    H: AlgebraicHasher,
{
    /// Candidate for removal. This function does not make any sense.
    pub fn default() -> Self {
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
    pub fn from_proof(proof: &Proof) -> Result<Self, Box<dyn Error>> {
        let mut index = 0;
        let mut items = vec![];
        while index < proof.0.len() {
            let len = proof.0[index].value() as usize;
            if proof.0.len() < index + 1 + len {
                return Err(ProofStreamError::boxed(
                    "failed to decode proof; wrong length",
                ));
            }
            let str = &proof.0[(index + 1)..(index + 1 + len)];
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
    pub fn dequeue(&mut self) -> Result<Item, Box<dyn Error>> {
        let item = self
            .items
            .get(self.items_index)
            .ok_or_else(|| ProofStreamError::boxed("Could not dequeue, queue empty"))?;

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
            transcript.append(&mut item.encode());
        }
        H::hash_slice(&transcript)
    }
}

#[cfg(test)]
mod proof_stream_typed_tests {
    use itertools::Itertools;
    use num_traits::{One, Zero};

    use super::*;
    use twenty_first::shared_math::{
        b_field_element::BFieldElement, rescue_prime_regular::RescuePrimeRegular,
        x_field_element::XFieldElement,
    };

    #[derive(Clone, Debug, PartialEq)]
    enum TestItem {
        ManyB(Vec<BFieldElement>),
        ManyX(Vec<XFieldElement>),
    }

    impl TestItem {
        pub fn as_bs(&self) -> Option<Vec<BFieldElement>> {
            match self {
                Self::ManyB(bs) => Some(bs.clone()),
                _ => None,
            }
        }

        pub fn as_xs(&self) -> Option<Vec<XFieldElement>> {
            match self {
                Self::ManyX(xs) => Some(xs.clone()),
                _ => None,
            }
        }
    }

    impl Default for TestItem {
        fn default() -> Self {
            TestItem::ManyB(vec![BFieldElement::zero()])
        }
    }

    impl IntoIterator for TestItem {
        type Item = BFieldElement;

        type IntoIter = std::vec::IntoIter<BFieldElement>;

        fn into_iter(self) -> Self::IntoIter {
            match self {
                TestItem::ManyB(bs) => bs.into_iter(),
                TestItem::ManyX(xs) => xs
                    .into_iter()
                    .map(|x| x.coefficients.to_vec())
                    .concat()
                    .into_iter(),
            }
        }
    }

    impl BFieldCodec for TestItem {
        fn decode(str: &[BFieldElement]) -> Result<Box<Self>, Box<dyn Error>> {
            let maybe_element_zero = str.get(0);
            match maybe_element_zero {
                None => Err(ProofStreamError::boxed(
                    "trying to decode empty string into test item",
                )),
                Some(bfe) => {
                    if bfe.is_zero() {
                        Ok(Box::new(Self::ManyB(str[1..].to_vec())))
                    } else {
                        let mut vect = vec![];
                        for mut chunk in &str[1..].iter().chunks(3) {
                            let mut xfe = XFieldElement::zero();
                            xfe.coefficients[0] = *chunk.next().expect(
                                "cannot cast chunk of BFieldElements into XFieldElements (1)",
                            );
                            xfe.coefficients[1] = *chunk.next().expect(
                                "cannot cast chunk of BFieldElements into XFieldElements (2)",
                            );
                            xfe.coefficients[2] = *chunk.next().expect(
                                "cannot cast chunk of BFieldElements into XFieldElements (3)",
                            );
                            vect.push(xfe);
                        }

                        Ok(Box::new(Self::ManyX(vect)))
                    }
                }
            }
        }

        fn encode(&self) -> Vec<BFieldElement> {
            let mut vect = vec![];
            match self {
                Self::ManyB(bs) => {
                    vect.push(BFieldElement::zero());
                    for b in bs {
                        vect.append(&mut b.encode());
                    }
                }
                Self::ManyX(xs) => {
                    vect.push(BFieldElement::one());
                    for x in xs {
                        vect.append(&mut x.encode());
                    }
                }
            }
            vect
        }
    }

    #[test]
    fn enqueue_dequeue_test() {
        let mut proof_stream = ProofStream::<TestItem, RescuePrimeRegular>::default();
        let ps: &mut ProofStream<TestItem, RescuePrimeRegular> = &mut proof_stream;

        // Empty

        assert!(ps.dequeue().is_err(), "cannot dequeue empty");

        // B

        let b_one = BFieldElement::one();
        let bs_expected = vec![b_one; 3];
        let item_1 = TestItem::ManyB(bs_expected.clone());
        ps.enqueue(&item_1);

        let item_1_option = ps.dequeue();
        assert!(item_1_option.is_ok(), "item 1 exists in queue");

        let item_1_actual: TestItem = item_1_option.unwrap();
        assert!(item_1_actual.as_xs().is_none(), "wrong type of item 1");
        let bs_option: Option<Vec<BFieldElement>> = item_1_actual.as_bs();
        assert!(bs_option.is_some(), "item 1 decodes to the right type");

        let bs_actual: Vec<BFieldElement> = bs_option.unwrap();
        assert_eq!(bs_expected, bs_actual, "enqueue/dequeue item 2");

        // Empty

        assert!(ps.dequeue().is_err(), "queue has become empty");

        // X

        let x_one = XFieldElement::one();

        let xs_expected = vec![x_one; 3];
        let item_2 = TestItem::ManyX(xs_expected.clone());
        ps.enqueue(&item_2);

        let item_2_option = ps.dequeue();
        assert!(item_2_option.is_ok(), "item 2 exists in queue");

        let item_2_actual: TestItem = item_2_option.unwrap();
        assert!(item_2_actual.as_bs().is_none(), "wrong type of item 2");
        let xs_option: Option<Vec<XFieldElement>> = item_2_actual.as_xs();
        assert!(xs_option.is_some(), "item 2 decodes to the right type");

        let xs_actual: Vec<XFieldElement> = xs_option.unwrap();
        assert_eq!(xs_expected, xs_actual, "enqueue/dequeue item 2");
    }

    // Property: prover_fiat_shamir() is equivalent to verifier_fiat_shamir() when the entire stream has been read.
    #[test]
    fn prover_verifier_fiat_shamir_test() {
        type H = RescuePrimeRegular;
        let mut proof_stream = ProofStream::<TestItem, H>::default();
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
}
