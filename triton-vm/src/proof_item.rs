use itertools::Itertools;

use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::rescue_prime_digest::Digest;
use twenty_first::shared_math::x_field_element::XFieldElement;
use twenty_first::util_types::algebraic_hasher::Hashable;
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

impl IntoIterator for ProofItem {
    type Item = BFieldElement;
    type IntoIter = std::vec::IntoIter<BFieldElement>;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            ProofItem::MerkleRoot(bs) => bs.to_sequence().into_iter(),
            ProofItem::TransposedBaseElements(bs) => bs.into_iter(),
            ProofItem::TransposedExtensionElements(xs) => xs_to_bs(&xs).into_iter(),
            ProofItem::AuthenticationPath(bss) => {
                bss.iter().map(|ap| ap.to_sequence()).concat().into_iter()
            }

            ProofItem::RevealedCombinationElement(x) => xs_to_bs(&[x]).into_iter(),
            ProofItem::FriCodeword(xs) => xs_to_bs(&xs).into_iter(),
            ProofItem::RevealedCombinationElements(xs) => xs_to_bs(&xs).into_iter(),
            ProofItem::FriProof(fri_proof) => {
                let mut bs = vec![];

                for (partial_auth_path, xfe) in fri_proof.iter() {
                    let mut elems: Vec<BFieldElement> = partial_auth_path
                        .0
                        .iter()
                        .flatten()
                        .flat_map(|digest| digest.values())
                        .collect();
                    bs.append(&mut elems);
                    bs.append(&mut xfe.to_sequence());
                }

                bs.into_iter()
            }
            ProofItem::CompressedAuthenticationPaths(partial_auth_paths) => {
                let mut bs: Vec<BFieldElement> = vec![];

                for partial_auth_path in partial_auth_paths.iter() {
                    for bs_in_partial_auth_path in partial_auth_path.0.iter().flatten() {
                        bs.append(&mut bs_in_partial_auth_path.to_sequence());
                    }
                }

                bs.into_iter()
            }
            ProofItem::TransposedBaseElementVectors(bss) => bss.concat().into_iter(),
            ProofItem::TransposedExtensionElementVectors(xss) => {
                xss.into_iter().map(|xs| xs_to_bs(&xs)).concat().into_iter()
            }
            ProofItem::PaddedHeight(padded_height) => vec![padded_height].into_iter(),
            ProofItem::Uncast(str) => str.into_iter(),
        }
    }
}

fn xs_to_bs(xs: &[XFieldElement]) -> Vec<BFieldElement> {
    xs.iter().map(|x| x.coefficients.to_vec()).concat()
}
