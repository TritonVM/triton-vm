use super::table::challenges_endpoints::AllEndpoints;
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::rescue_prime_xlix::{RescuePrimeXlix, RP_DEFAULT_WIDTH};
use crate::shared_math::x_field_element::XFieldElement;
use crate::util_types::merkle_tree::PartialAuthenticationPath;
use crate::util_types::proof_stream_typed::{ProofStream, ProofStreamError};
use itertools::Itertools;

pub type StarkProofStream = ProofStream<Item, RescuePrimeXlix<RP_DEFAULT_WIDTH>>;

type FriProof = Vec<(PartialAuthenticationPath<Vec<BFieldElement>>, XFieldElement)>;
type CompressedAuthenticationPaths = Vec<PartialAuthenticationPath<Vec<BFieldElement>>>;

#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum Item {
    CompressedAuthenticationPaths(CompressedAuthenticationPaths),
    TransposedBaseElementVectors(Vec<Vec<BFieldElement>>),
    TransposedExtensionElementVectors(Vec<Vec<XFieldElement>>),
    MerkleRoot(Vec<BFieldElement>),
    Terminals(AllEndpoints),
    TransposedBaseElements(Vec<BFieldElement>),
    TransposedExtensionElements(Vec<XFieldElement>),
    AuthenticationPath(Vec<Vec<BFieldElement>>),
    // FIXME: Redundancy.
    RevealedCombinationElement(XFieldElement),
    RevealedCombinationElements(Vec<XFieldElement>),
    FriCodeword(Vec<XFieldElement>),
    FriProof(FriProof),
    PaddedHeights(Vec<BFieldElement>),
}

impl Item {
    pub fn as_compressed_authentication_paths(
        &self,
    ) -> Result<CompressedAuthenticationPaths, Box<dyn std::error::Error>> {
        match self {
            Self::CompressedAuthenticationPaths(caps) => Ok(caps.to_owned()),
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
            _ => Err(ProofStreamError::boxed(
                "expected transposed extension element vectors, but got something else",
            )),
        }
    }

    pub fn as_merkle_root(&self) -> Result<Vec<BFieldElement>, Box<dyn std::error::Error>> {
        match self {
            Self::MerkleRoot(bs) => Ok(bs.clone()),
            _ => Err(ProofStreamError::boxed(
                "expected merkle root, but got something else",
            )),
        }
    }

    pub fn as_terminals(&self) -> Result<AllEndpoints, Box<dyn std::error::Error>> {
        match self {
            Self::Terminals(all_endpoints) => Ok(all_endpoints.to_owned()),
            _ => Err(ProofStreamError::boxed(
                "expected all terminals, but got something else",
            )),
        }
    }

    pub fn as_transposed_base_elements(
        &self,
    ) -> Result<Vec<BFieldElement>, Box<dyn std::error::Error>> {
        match self {
            Self::TransposedBaseElements(bs) => Ok(bs.to_owned()),
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
            _ => Err(ProofStreamError::boxed(
                "expected tranposed extension elements, but got something else",
            )),
        }
    }

    pub fn as_authentication_path(
        &self,
    ) -> Result<Vec<Vec<BFieldElement>>, Box<dyn std::error::Error>> {
        match self {
            Self::AuthenticationPath(bss) => Ok(bss.to_owned()),
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
            _ => Err(ProofStreamError::boxed(
                "expected revealed combination elements, but got something else",
            )),
        }
    }

    pub fn as_fri_codeword(&self) -> Result<Vec<XFieldElement>, Box<dyn std::error::Error>> {
        match self {
            Self::FriCodeword(xs) => Ok(xs.to_owned()),
            _ => Err(ProofStreamError::boxed(
                "expected FRI codeword, but got something else",
            )),
        }
    }

    pub fn as_fri_proof(&self) -> Result<FriProof, Box<dyn std::error::Error>> {
        match self {
            Self::FriProof(fri_proof) => Ok(fri_proof.to_owned()),
            _ => Err(ProofStreamError::boxed(
                "expected FRI proof, but got something else",
            )),
        }
    }

    pub fn as_padded_heights(&self) -> Result<Vec<BFieldElement>, Box<dyn std::error::Error>> {
        match self {
            Self::PaddedHeights(padded_heights) => Ok(padded_heights.to_owned()),
            _ => Err(ProofStreamError::boxed(
                "expected padded table heights, but got something else",
            )),
        }
    }
}

impl IntoIterator for Item {
    type Item = BFieldElement;

    type IntoIter = std::vec::IntoIter<BFieldElement>;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            Item::MerkleRoot(bs) => bs.into_iter(),
            Item::Terminals(all_endpoints) => all_endpoints.into_iter(),
            Item::TransposedBaseElements(bs) => bs.into_iter(),
            Item::TransposedExtensionElements(xs) => xs_to_bs(&xs),
            Item::AuthenticationPath(bss) => bss.concat().into_iter(),
            Item::RevealedCombinationElement(x) => xs_to_bs(&[x]),
            Item::FriCodeword(xs) => xs_to_bs(&xs),
            Item::RevealedCombinationElements(xs) => xs_to_bs(&xs),
            Item::FriProof(fri_proof) => {
                let mut bs: Vec<BFieldElement> = vec![];

                for (partial_auth_path, x) in fri_proof.iter() {
                    for bs_in_partial_auth_path in partial_auth_path.0.iter().flatten() {
                        bs.append(&mut bs_in_partial_auth_path.clone());
                    }
                    bs.append(&mut xs_to_bs(&[x.to_owned()]).collect());
                }

                bs.into_iter()
            }
            Item::CompressedAuthenticationPaths(partial_auth_paths) => {
                let mut bs: Vec<BFieldElement> = vec![];

                for partial_auth_path in partial_auth_paths.iter() {
                    for bs_in_partial_auth_path in partial_auth_path.0.iter().flatten() {
                        bs.append(&mut bs_in_partial_auth_path.clone());
                    }
                }

                bs.into_iter()
            }
            Item::TransposedBaseElementVectors(bss) => bss.concat().into_iter(),
            Item::TransposedExtensionElementVectors(xss) => xss
                .into_iter()
                .map(|xs| xs_to_bs(&xs).collect::<Vec<_>>())
                .concat()
                .into_iter(),
            Item::PaddedHeights(padded_heights) => padded_heights.into_iter(),
        }
    }
}

fn xs_to_bs(xs: &[XFieldElement]) -> std::vec::IntoIter<BFieldElement> {
    xs.iter()
        .map(|x| x.coefficients.to_vec())
        .concat()
        .into_iter()
}
