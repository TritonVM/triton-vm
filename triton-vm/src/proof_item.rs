use itertools::Itertools;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::x_field_element::XFieldElement;
use twenty_first::util_types::merkle_tree::PartialAuthenticationPath;
use twenty_first::util_types::proof_stream_typed::ProofStreamError;
use twenty_first::util_types::simple_hasher::{Hashable, Hasher};

use super::table::challenges_terminals::AllTerminals;

type FriProof<Digest> = Vec<(PartialAuthenticationPath<Digest>, XFieldElement)>;
type AuthenticationStructure<Digest> = Vec<PartialAuthenticationPath<Digest>>;

#[derive(Debug, Clone)]
#[allow(clippy::large_enum_variant)]
pub enum ProofItem<H: Hasher>
where
    BFieldElement: Hashable<H::T>,
{
    CompressedAuthenticationPaths(AuthenticationStructure<H::Digest>),
    TransposedBaseElementVectors(Vec<Vec<BFieldElement>>),
    TransposedExtensionElementVectors(Vec<Vec<XFieldElement>>),
    MerkleRoot(H::Digest),
    Terminals(AllTerminals<H>),
    TransposedBaseElements(Vec<BFieldElement>),
    TransposedExtensionElements(Vec<XFieldElement>),
    AuthenticationPath(Vec<H::Digest>),
    // FIXME: Redundancy.
    RevealedCombinationElement(XFieldElement),
    RevealedCombinationElements(Vec<XFieldElement>),
    FriCodeword(Vec<XFieldElement>),
    FriProof(FriProof<H::Digest>),
    PaddedHeight(BFieldElement),
}

impl<H: Hasher> ProofItem<H>
where
    BFieldElement: Hashable<H::T>,
{
    pub fn as_compressed_authentication_paths(
        &self,
    ) -> Result<AuthenticationStructure<H::Digest>, Box<dyn std::error::Error>> {
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

    pub fn as_merkle_root(&self) -> Result<H::Digest, Box<dyn std::error::Error>> {
        match self {
            Self::MerkleRoot(bs) => Ok(bs.clone()),
            _ => Err(ProofStreamError::boxed(
                "expected merkle root, but got something else",
            )),
        }
    }

    pub fn as_terminals(&self) -> Result<AllTerminals<H>, Box<dyn std::error::Error>> {
        match self {
            Self::Terminals(all_terminals) => Ok(all_terminals.to_owned()),
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

    pub fn as_authentication_path(&self) -> Result<Vec<H::Digest>, Box<dyn std::error::Error>> {
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

    pub fn as_fri_proof(&self) -> Result<FriProof<H::Digest>, Box<dyn std::error::Error>> {
        match self {
            Self::FriProof(fri_proof) => Ok(fri_proof.to_owned()),
            _ => Err(ProofStreamError::boxed(
                "expected FRI proof, but got something else",
            )),
        }
    }

    pub fn as_padded_heights(&self) -> Result<BFieldElement, Box<dyn std::error::Error>> {
        match self {
            Self::PaddedHeight(padded_height) => Ok(padded_height.to_owned()),
            _ => Err(ProofStreamError::boxed(
                "expected padded table height, but got something else",
            )),
        }
    }
}

impl<H: Hasher> IntoIterator for ProofItem<H>
where
    BFieldElement: Hashable<H::T>,
    XFieldElement: Hashable<H::T>,
{
    type Item = H::T;

    type IntoIter = std::vec::IntoIter<H::T>;

    fn into_iter(self) -> Self::IntoIter {
        match self {
            ProofItem::MerkleRoot(bs) => bs.to_sequence().into_iter(),
            ProofItem::Terminals(all_terminals) => all_terminals.into_iter(),
            ProofItem::TransposedBaseElements(bs) => bs_to_ts::<H>(&bs).into_iter(),
            ProofItem::TransposedExtensionElements(xs) => bs_to_ts::<H>(&xs_to_bs(&xs)).into_iter(),
            ProofItem::AuthenticationPath(bss) => {
                bss.iter().map(|ap| ap.to_sequence()).concat().into_iter()
            }

            ProofItem::RevealedCombinationElement(x) => bs_to_ts::<H>(&xs_to_bs(&[x])).into_iter(),
            ProofItem::FriCodeword(xs) => bs_to_ts::<H>(&xs_to_bs(&xs)).into_iter(),
            ProofItem::RevealedCombinationElements(xs) => bs_to_ts::<H>(&xs_to_bs(&xs)).into_iter(),
            ProofItem::FriProof(fri_proof) => {
                let mut ts: Vec<H::T> = vec![];

                for (partial_auth_path, x) in fri_proof.iter() {
                    for ts_in_partial_auth_path in partial_auth_path.0.iter().flatten() {
                        ts.append(&mut ts_in_partial_auth_path.to_sequence());
                    }
                    ts.append(&mut x.clone().to_sequence());
                }

                ts.into_iter()
            }
            ProofItem::CompressedAuthenticationPaths(partial_auth_paths) => {
                let mut bs: Vec<H::T> = vec![];

                for partial_auth_path in partial_auth_paths.iter() {
                    for bs_in_partial_auth_path in partial_auth_path.0.iter().flatten() {
                        bs.append(&mut bs_in_partial_auth_path.to_sequence());
                    }
                }

                bs.into_iter()
            }
            ProofItem::TransposedBaseElementVectors(bss) => {
                bs_to_ts::<H>(&bss.concat()).into_iter()
            }
            ProofItem::TransposedExtensionElementVectors(xss) => {
                bs_to_ts::<H>(&xss.into_iter().map(|xs| xs_to_bs(&xs)).concat()).into_iter()
            }
            ProofItem::PaddedHeight(padded_height) => bs_to_ts::<H>(&[padded_height]).into_iter(),
        }
    }
}

impl<H: Hasher> Default for ProofItem<H>
where
    BFieldElement: Hashable<H::T>,
{
    fn default() -> Self {
        panic!("Should not have to implement default for ProofItem<H>")
    }
}

fn xs_to_bs(xs: &[XFieldElement]) -> Vec<BFieldElement> {
    xs.iter().map(|x| x.coefficients.to_vec()).concat()
}

fn bs_to_ts<H: Hasher>(bs: &[BFieldElement]) -> Vec<H::T>
where
    BFieldElement: Hashable<H::T>,
{
    bs.iter()
        .flat_map(|b| b.to_sequence())
        .collect::<Vec<H::T>>()
}
