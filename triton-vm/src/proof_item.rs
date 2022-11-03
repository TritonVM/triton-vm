use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::rescue_prime_digest::Digest;
use twenty_first::shared_math::x_field_element::XFieldElement;
use twenty_first::util_types::merkle_tree::PartialAuthenticationPath;
use twenty_first::util_types::proof_stream_typed::ProofStreamError;

use crate::bfield_codec::BFieldCodec;

type FriProof = Vec<(PartialAuthenticationPath<Digest>, XFieldElement)>;
type AuthenticationStructure<Digest> = Vec<PartialAuthenticationPath<Digest>>;

#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum ProofItem {
    CompressedAuthenticationPaths(AuthenticationStructure<Digest>),
    TransposedBaseElementVectors(Vec<Vec<BFieldElement>>),
    TransposedExtensionElementVectors(Vec<Vec<XFieldElement>>),
    MerkleRoot(Digest),
    TransposedBaseElements(Vec<BFieldElement>),
    TransposedExtensionElements(Vec<XFieldElement>),
    AuthenticationPath(Vec<Digest>),
    // FIXME: Redundancy.
    RevealedCombinationElement(XFieldElement),
    RevealedCombinationElements(Vec<XFieldElement>),
    FriCodeword(Vec<XFieldElement>),
    FriProof(FriProof),
    PaddedHeight(BFieldElement),
    Uncast(Vec<BFieldElement>),
}

impl ProofItem
where
    AuthenticationStructure<Digest>: BFieldCodec,
    Vec<Vec<BFieldElement>>: BFieldCodec,
    Vec<Vec<XFieldElement>>: BFieldCodec,
    Digest: BFieldCodec,
    Vec<BFieldElement>: BFieldCodec,
    Vec<XFieldElement>: BFieldCodec,
    Vec<Digest>: BFieldCodec,
    BFieldElement: BFieldCodec,
    XFieldElement: BFieldCodec,
    FriProof: BFieldCodec,
{
    pub fn as_compressed_authentication_paths(
        &self,
    ) -> Result<AuthenticationStructure<Digest>, Box<dyn std::error::Error>> {
        match self {
            Self::CompressedAuthenticationPaths(caps) => Ok(caps.to_owned()),
            Self::Uncast(str) => match AuthenticationStructure::<Digest>::decode(str) {
                Ok(boxed_auth_struct) => Ok(*boxed_auth_struct),
                Err(_) => Err(ProofStreamError::boxed(
                    "cast to authentication structure failed",
                )),
            },
            _ => Err(ProofStreamError::boxed(
                "expected compressed authentication paths, but got something else",
            )),
        }
    }

    pub fn as_transposed_base_element_vectors(
        &self,
    ) -> Result<Vec<Vec<BFieldElement>>, Box<dyn std::error::Error>> {
        match self {
            Self::TransposedBaseElementVectors(bss) => Ok(bss.to_owned()),
            Self::Uncast(str) => match Vec::<Vec<BFieldElement>>::decode(str) {
                Ok(base_element_vectors) => Ok(*base_element_vectors),
                Err(_) => Err(ProofStreamError::boxed(
                    "cast to base element vectors failed",
                )),
            },
            _ => Err(ProofStreamError::boxed(
                "expected transposed base element vectors, but got something else",
            )),
        }
    }

    pub fn as_transposed_extension_element_vectors(
        &self,
    ) -> Result<Vec<Vec<XFieldElement>>, Box<dyn std::error::Error>> {
        match self {
            Self::TransposedExtensionElementVectors(xss) => Ok(xss.to_owned()),
            Self::Uncast(str) => match Vec::<Vec<XFieldElement>>::decode(str) {
                Ok(ext_element_vectors) => Ok(*ext_element_vectors),
                Err(_) => Err(ProofStreamError::boxed(
                    "cast to extension field element vectors failed",
                )),
            },
            _ => Err(ProofStreamError::boxed(
                "expected transposed extension element vectors, but got something else",
            )),
        }
    }

    pub fn as_merkle_root(&self) -> Result<Digest, Box<dyn std::error::Error>> {
        match self {
            Self::MerkleRoot(bs) => Ok(*bs),
            Self::Uncast(str) => match Digest::decode(str) {
                Ok(merkle_root) => Ok(*merkle_root),
                Err(_) => Err(ProofStreamError::boxed("cast to Merkle root failed")),
            },
            _ => Err(ProofStreamError::boxed(
                "expected merkle root, but got something else",
            )),
        }
    }

    pub fn as_transposed_base_elements(
        &self,
    ) -> Result<Vec<BFieldElement>, Box<dyn std::error::Error>> {
        match self {
            Self::TransposedBaseElements(bs) => Ok(bs.to_owned()),
            Self::Uncast(str) => match Vec::<BFieldElement>::decode(str) {
                Ok(transposed_base_elements) => Ok(*transposed_base_elements),
                Err(_) => Err(ProofStreamError::boxed(
                    "cast to transposed base field elements failed",
                )),
            },
            _ => Err(ProofStreamError::boxed(
                "expected tranposed base elements, but got something else",
            )),
        }
    }

    pub fn as_transposed_extension_elements(
        &self,
    ) -> Result<Vec<XFieldElement>, Box<dyn std::error::Error>> {
        match self {
            Self::TransposedExtensionElements(xs) => Ok(xs.to_owned()),
            Self::Uncast(str) => match Vec::<XFieldElement>::decode(str) {
                Ok(transposed_ext_elements) => Ok(*transposed_ext_elements),
                Err(_) => Err(ProofStreamError::boxed(
                    "cast to transposed extension field elements failed",
                )),
            },
            _ => Err(ProofStreamError::boxed(
                "expected tranposed extension elements, but got something else",
            )),
        }
    }

    pub fn as_authentication_path(&self) -> Result<Vec<Digest>, Box<dyn std::error::Error>> {
        match self {
            Self::AuthenticationPath(bss) => Ok(bss.to_owned()),
            Self::Uncast(str) => match Vec::<Digest>::decode(str) {
                Ok(authentication_path) => Ok(*authentication_path),
                Err(_) => Err(ProofStreamError::boxed(
                    "cast to authentication path failed",
                )),
            },
            _ => Err(ProofStreamError::boxed(
                "expected authentication path, but got something else",
            )),
        }
    }

    pub fn as_revealed_combination_element(
        &self,
    ) -> Result<XFieldElement, Box<dyn std::error::Error>> {
        match self {
            Self::RevealedCombinationElement(x) => Ok(x.to_owned()),
            Self::Uncast(str) => match XFieldElement::decode(str) {
                Ok(revealed_combination_element) => Ok(*revealed_combination_element),
                Err(_) => Err(ProofStreamError::boxed(
                    "cast to revealed combination element failed",
                )),
            },
            _ => Err(ProofStreamError::boxed(
                "expected revealed combination element, but got something else",
            )),
        }
    }

    pub fn as_revealed_combination_elements(
        &self,
    ) -> Result<Vec<XFieldElement>, Box<dyn std::error::Error>> {
        match self {
            Self::RevealedCombinationElements(xs) => Ok(xs.to_owned()),
            Self::Uncast(str) => match Vec::<XFieldElement>::decode(str) {
                Ok(revealed_combination_elements) => Ok(*revealed_combination_elements),
                Err(_) => Err(ProofStreamError::boxed(
                    "cast to revealed combination elements failed",
                )),
            },
            _ => Err(ProofStreamError::boxed(
                "expected revealed combination elements, but got something else",
            )),
        }
    }

    pub fn as_fri_codeword(&self) -> Result<Vec<XFieldElement>, Box<dyn std::error::Error>> {
        match self {
            Self::FriCodeword(xs) => Ok(xs.to_owned()),
            Self::Uncast(str) => match Vec::<XFieldElement>::decode(str) {
                Ok(fri_codeword) => Ok(*fri_codeword),
                Err(_) => Err(ProofStreamError::boxed("cast to FRI codeword failed")),
            },
            _ => Err(ProofStreamError::boxed(
                "expected FRI codeword, but got something else",
            )),
        }
    }

    pub fn as_fri_proof(&self) -> Result<FriProof, Box<dyn std::error::Error>> {
        match self {
            Self::FriProof(fri_proof) => Ok(fri_proof.to_owned()),
            Self::Uncast(str) => match FriProof::decode(str) {
                Ok(fri_proof) => Ok(*fri_proof),
                Err(_) => Err(ProofStreamError::boxed("cast to FRI proof failed")),
            },
            _ => Err(ProofStreamError::boxed(
                "expected FRI proof, but got something else",
            )),
        }
    }

    pub fn as_padded_heights(&self) -> Result<BFieldElement, Box<dyn std::error::Error>> {
        match self {
            Self::PaddedHeight(padded_height) => Ok(padded_height.to_owned()),
            Self::Uncast(str) => match BFieldElement::decode(str) {
                Ok(padded_height) => Ok(*padded_height),
                Err(_) => Err(ProofStreamError::boxed("cast to padded heights failed")),
            },
            _ => Err(ProofStreamError::boxed(
                "expected padded table height, but got something else",
            )),
        }
    }
}

impl BFieldCodec for ProofItem {
    /// Turn the given string of BFieldElements into a ProofItem.
    /// The first element denotes the length of the encoding; make
    /// sure it is correct!
    fn decode(str: &[BFieldElement]) -> Result<Box<Self>, Box<dyn std::error::Error>> {
        if let Some(len) = str.get(0) {
            if len.value() as usize + 1 != str.len() {
                return Err(ProofStreamError::boxed("length mismatch"));
            } else {
                Ok(Box::new(Self::Uncast(str[1..].to_vec())))
            }
        } else {
            return Err(ProofStreamError::boxed("empty buffer"));
        }
    }

    /// Encode the ProofItem as a string of BFieldElements, with the
    /// first element denoting the length of the rest.
    fn encode(&self) -> Vec<BFieldElement> {
        let mut tail = match self {
            ProofItem::CompressedAuthenticationPaths(something) => something.encode(),
            ProofItem::TransposedBaseElementVectors(something) => something.encode(),
            ProofItem::TransposedExtensionElementVectors(something) => something.encode(),
            ProofItem::MerkleRoot(something) => something.encode(),
            ProofItem::TransposedBaseElements(something) => something.encode(),
            ProofItem::TransposedExtensionElements(something) => something.encode(),
            ProofItem::AuthenticationPath(something) => something.encode(),
            ProofItem::RevealedCombinationElement(something) => something.encode(),
            ProofItem::RevealedCombinationElements(something) => something.encode(),
            ProofItem::FriCodeword(something) => something.encode(),
            ProofItem::FriProof(something) => something.encode(),
            ProofItem::PaddedHeight(something) => something.encode(),
            ProofItem::Uncast(something) => something.encode(),
        };
        let head = BFieldElement::new(tail.len().try_into().unwrap());
        tail.insert(0, head);
        tail
    }
}

#[cfg(test)]
mod proof_item_typed_tests {
    use itertools::Itertools;
    use rand::{thread_rng, RngCore};

    use crate::proof_stream::ProofStream;

    use super::*;
    use twenty_first::shared_math::{
        b_field_element::BFieldElement, rescue_prime_regular::RescuePrimeRegular,
        x_field_element::XFieldElement,
    };

    fn random_bfieldelement() -> BFieldElement {
        let mut rng = thread_rng();
        BFieldElement::new(rng.next_u64())
    }

    fn random_xfieldelement() -> XFieldElement {
        XFieldElement {
            coefficients: [
                random_bfieldelement(),
                random_bfieldelement(),
                random_bfieldelement(),
            ],
        }
    }

    #[test]
    fn serialize_stark_proof_test() {
        type H = RescuePrimeRegular;
        let mut proof_stream = ProofStream::<ProofItem, H>::new();
        let manyb1 = (0..10)
            .into_iter()
            .map(|_| random_bfieldelement())
            .collect_vec();
        let manyx = (0..13)
            .into_iter()
            .map(|_| random_xfieldelement())
            .collect_vec();
        let manyb2 = (0..11)
            .into_iter()
            .map(|_| random_bfieldelement())
            .collect_vec();

        proof_stream.enqueue(&ProofItem::TransposedBaseElements(manyb1.clone()));
        proof_stream.enqueue(&ProofItem::TransposedExtensionElements(manyx.clone()));
        proof_stream.enqueue(&ProofItem::TransposedBaseElements(manyb2.clone()));

        let proof = proof_stream.to_proof();

        let mut proof_stream =
            ProofStream::<ProofItem, H>::from_proof(&proof).expect("invalid parsing of proof");

        let manyb1_ = proof_stream
            .dequeue()
            .expect("can't dequeue item")
            .as_transposed_base_elements()
            .expect("cannot parse dequeued item");
        assert_eq!(manyb1, manyb1_);

        let manyx_ = proof_stream
            .dequeue()
            .expect("can't dequeue item")
            .as_transposed_extension_elements()
            .expect("cannot parse dequeued item");
        assert_eq!(manyx, manyx_);

        let manyb2_ = proof_stream
            .dequeue()
            .expect("can't dequeue item")
            .as_transposed_base_elements()
            .expect("cannot parse dequeued item");
        assert_eq!(manyb2, manyb2_);
    }
}
