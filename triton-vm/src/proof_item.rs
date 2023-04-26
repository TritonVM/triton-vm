use anyhow::bail;
use anyhow::Result;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::tip5::Digest;
use twenty_first::shared_math::tip5::DIGEST_LENGTH;
use twenty_first::shared_math::x_field_element::XFieldElement;
use twenty_first::shared_math::x_field_element::EXTENSION_DEGREE;
use twenty_first::util_types::merkle_tree::PartialAuthenticationPath;
use twenty_first::util_types::proof_stream_typed::ProofStreamError;

use crate::bfield_codec::BFieldCodec;
use crate::table::master_table::NUM_BASE_COLUMNS;
use crate::table::master_table::NUM_EXT_COLUMNS;

type AuthenticationStructure<Digest> = Vec<PartialAuthenticationPath<Digest>>;

/// A `FriResponse` is a vector of partial authentication paths and `XFieldElements`. The
/// `XFieldElements` are the values of the leaves of the Merkle tree. They correspond to the
/// queried index of the FRI codeword (of that round). The corresponding partial authentication
/// paths are the paths from the queried leaf to the root of the Merkle tree.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FriResponse(pub Vec<(PartialAuthenticationPath<Digest>, XFieldElement)>);

impl BFieldCodec for FriResponse {
    fn decode(str: &[BFieldElement]) -> Result<Box<Self>> {
        let mut index = 0usize;
        let mut vect: Vec<(PartialAuthenticationPath<Digest>, XFieldElement)> = vec![];
        while index < str.len() {
            // length
            let len = match str.get(index) {
                Some(bfe) => bfe.value() as usize,
                None => {
                    bail!(ProofStreamError::new(
                        "invalid index counting in decode for FriResponse",
                    ));
                }
            };
            index += 1;

            // mask
            let mask = match str.get(index) {
                Some(bfe) => bfe.value() as u32,
                None => {
                    bail!(ProofStreamError::new(
                        "invalid mask decoding in decode for FriResponse",
                    ));
                }
            };
            index += 1;

            // partial authentication path
            let mut pap: Vec<Option<Digest>> = vec![];
            for i in (0..len).rev() {
                if mask & (1 << i) == 0 {
                    pap.push(None);
                } else if let Some(digest) = str.get(index..(index + DIGEST_LENGTH)) {
                    pap.push(Some(*Digest::decode(digest)?));
                    index += DIGEST_LENGTH;
                } else {
                    bail!(ProofStreamError::new(
                        "length mismatch in decoding FRI response",
                    ));
                }
            }

            // x field element
            let xfe = match str.get(index..(index + EXTENSION_DEGREE)) {
                Some(substr) => *XFieldElement::decode(substr)?,
                None => {
                    bail!(ProofStreamError::new(
                        "could not decode XFieldElement in decode for FriResponse",
                    ));
                }
            };
            index += EXTENSION_DEGREE;

            // push to vector
            vect.push((PartialAuthenticationPath(pap), xfe));
        }
        Ok(Box::new(FriResponse(vect)))
    }

    fn encode(&self) -> Vec<BFieldElement> {
        let mut str = vec![];
        for (partial_authentication_path, xfe) in self.0.iter() {
            str.push(BFieldElement::new(
                partial_authentication_path.0.len().try_into().unwrap(),
            ));
            let mut mask = 0u32;
            for maybe_digest in partial_authentication_path.0.iter() {
                mask <<= 1;
                if maybe_digest.is_some() {
                    mask |= 1;
                }
            }
            str.push(BFieldElement::new(mask as u64));
            for digest in partial_authentication_path.0.iter().flatten() {
                str.append(&mut digest.encode())
            }
            str.append(&mut xfe.encode());
        }
        str
    }
}

pub trait MayBeUncast {
    fn uncast(&self) -> Vec<BFieldElement>;
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProofItem {
    CompressedAuthenticationPaths(AuthenticationStructure<Digest>),
    MasterBaseTableRows(Vec<Vec<BFieldElement>>),
    MasterExtTableRows(Vec<Vec<XFieldElement>>),
    OutOfDomainBaseRow(Vec<XFieldElement>),
    OutOfDomainExtRow(Vec<XFieldElement>),
    MerkleRoot(Digest),
    AuthenticationPath(Vec<Digest>),
    RevealedCombinationElements(Vec<XFieldElement>),
    FriCodeword(Vec<XFieldElement>),
    FriResponse(FriResponse),
    PaddedHeight(BFieldElement),
    Uncast(Vec<BFieldElement>),
}

impl MayBeUncast for ProofItem {
    fn uncast(&self) -> Vec<BFieldElement> {
        if let Self::Uncast(vector) = self {
            let vector_len = BFieldElement::new(vector.len() as u64);
            vec![vec![vector_len], vector.to_owned()].concat()
        } else {
            self.encode()
        }
    }
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
    FriResponse: BFieldCodec,
{
    pub fn as_compressed_authentication_paths(&self) -> Result<AuthenticationStructure<Digest>> {
        match self {
            Self::CompressedAuthenticationPaths(caps) => Ok(caps.to_owned()),
            Self::Uncast(str) => match AuthenticationStructure::<Digest>::decode(str) {
                Ok(boxed_auth_struct) => Ok(*boxed_auth_struct),
                Err(e) => bail!(ProofStreamError::new(&format!(
                    "cast to authentication structure failed: {e}"
                ))),
            },
            other => bail!(ProofStreamError::new(&format!(
                "expected compressed authentication paths, but got something else: {other:?}",
            ))),
        }
    }

    pub fn as_master_base_table_rows(&self) -> Result<Vec<Vec<BFieldElement>>> {
        match self {
            Self::MasterBaseTableRows(bss) => Ok(bss.to_owned()),
            Self::Uncast(str) => match Vec::<Vec<BFieldElement>>::decode(str) {
                Ok(base_element_vectors) => Ok(*base_element_vectors),
                Err(_) => bail!(ProofStreamError::new("cast to base element vectors failed",)),
            },
            _ => bail!(ProofStreamError::new(
                "expected master base table rows, but got something else",
            )),
        }
    }

    pub fn as_master_ext_table_rows(&self) -> Result<Vec<Vec<XFieldElement>>> {
        match self {
            Self::MasterExtTableRows(xss) => Ok(xss.to_owned()),
            Self::Uncast(str) => match Vec::<Vec<XFieldElement>>::decode(str) {
                Ok(ext_element_vectors) => Ok(*ext_element_vectors),
                Err(_) => bail!(ProofStreamError::new(
                    "cast to extension field element vectors failed",
                )),
            },
            _ => bail!(ProofStreamError::new(
                "expected master extension table rows, but got something else",
            )),
        }
    }

    pub fn as_out_of_domain_base_row(&self) -> Result<Vec<XFieldElement>> {
        match self {
            Self::OutOfDomainBaseRow(xs) => Ok(xs.to_owned()),
            Self::Uncast(str) => match Vec::<XFieldElement>::decode(str) {
                Ok(xs) => {
                    if xs.len() != NUM_BASE_COLUMNS {
                        bail!(ProofStreamError::new(
                            "cast to out of domain base row failed"
                        ));
                    }
                    Ok(*xs)
                }
                Err(_) => bail!(ProofStreamError::new(
                    "cast to out of domain base row failed"
                )),
            },
            _ => bail!(ProofStreamError::new(
                "expected out of domain base row, but got something else",
            )),
        }
    }

    pub fn as_out_of_domain_ext_row(&self) -> Result<Vec<XFieldElement>> {
        match self {
            Self::OutOfDomainExtRow(xs) => Ok(xs.to_owned()),
            Self::Uncast(str) => match Vec::<XFieldElement>::decode(str) {
                Ok(xs) => {
                    if xs.len() != NUM_EXT_COLUMNS {
                        bail!(ProofStreamError::new(
                            "cast to out of domain extension row failed"
                        ));
                    }
                    Ok(*xs)
                }
                Err(_) => bail!(ProofStreamError::new(
                    "cast to out of domain extension row failed"
                )),
            },
            _ => bail!(ProofStreamError::new(
                "expected out of domain extension row, but got something else",
            )),
        }
    }

    pub fn as_merkle_root(&self) -> Result<Digest> {
        match self {
            Self::MerkleRoot(bs) => Ok(*bs),
            Self::Uncast(str) => match Digest::decode(str) {
                Ok(merkle_root) => Ok(*merkle_root),
                Err(_) => bail!(ProofStreamError::new("cast to Merkle root failed",)),
            },
            _ => bail!(ProofStreamError::new(
                "expected merkle root, but got something else",
            )),
        }
    }

    pub fn as_authentication_path(&self) -> Result<Vec<Digest>> {
        match self {
            Self::AuthenticationPath(bss) => Ok(bss.to_owned()),
            Self::Uncast(str) => match Vec::<Digest>::decode(str) {
                Ok(authentication_path) => Ok(*authentication_path),
                Err(_) => bail!(ProofStreamError::new("cast to authentication path failed",)),
            },
            _ => bail!(ProofStreamError::new(
                "expected authentication path, but got something else",
            )),
        }
    }

    pub fn as_revealed_combination_elements(&self) -> Result<Vec<XFieldElement>> {
        match self {
            Self::RevealedCombinationElements(xs) => Ok(xs.to_owned()),
            Self::Uncast(str) => match Vec::<XFieldElement>::decode(str) {
                Ok(revealed_combination_elements) => Ok(*revealed_combination_elements),
                Err(_) => bail!(ProofStreamError::new(
                    "cast to revealed combination elements failed",
                )),
            },
            _ => bail!(ProofStreamError::new(
                "expected revealed combination elements, but got something else",
            )),
        }
    }

    pub fn as_fri_codeword(&self) -> Result<Vec<XFieldElement>> {
        match self {
            Self::FriCodeword(xs) => Ok(xs.to_owned()),
            Self::Uncast(str) => match Vec::<XFieldElement>::decode(str) {
                Ok(fri_codeword) => Ok(*fri_codeword),
                Err(_) => bail!(ProofStreamError::new("cast to FRI codeword failed",)),
            },
            _ => bail!(ProofStreamError::new(
                "expected FRI codeword, but got something else",
            )),
        }
    }

    pub fn as_fri_response(&self) -> Result<FriResponse> {
        match self {
            Self::FriResponse(fri_proof) => Ok(fri_proof.to_owned()),
            Self::Uncast(str) => match FriResponse::decode(str) {
                Ok(fri_proof) => Ok(*fri_proof),
                Err(_) => bail!(ProofStreamError::new("cast to FRI proof failed",)),
            },
            _ => bail!(ProofStreamError::new(
                "expected FRI proof, but got something else",
            )),
        }
    }

    pub fn as_padded_heights(&self) -> Result<BFieldElement> {
        match self {
            Self::PaddedHeight(padded_height) => Ok(padded_height.to_owned()),
            Self::Uncast(str) => match BFieldElement::decode(str) {
                Ok(padded_height) => Ok(*padded_height),
                Err(_) => bail!(ProofStreamError::new("cast to padded heights failed",)),
            },
            _ => bail!(ProofStreamError::new(
                "expected padded table height, but got something else",
            )),
        }
    }
}

impl BFieldCodec for ProofItem {
    /// Turn the given string of BFieldElements into a ProofItem. The first element denotes the
    /// length of the encoding; make sure it is correct!
    fn decode(str: &[BFieldElement]) -> Result<Box<Self>> {
        let Some(len) = str.get(0) else {
            bail!(ProofStreamError::new("empty buffer"))
        };
        if len.value() + 1 != str.len() as u64 {
            bail!(ProofStreamError::new("length mismatch"))
        }

        let raw_item = Self::Uncast(str[1..].to_vec());
        Ok(Box::new(raw_item))
    }

    /// Encode the ProofItem as a string of BFieldElements, with the first element denoting the
    /// length of the rest.
    fn encode(&self) -> Vec<BFieldElement> {
        let mut tail = match self {
            ProofItem::CompressedAuthenticationPaths(something) => something.encode(),
            ProofItem::MasterBaseTableRows(something) => something.encode(),
            ProofItem::MasterExtTableRows(something) => something.encode(),
            ProofItem::OutOfDomainBaseRow(row) => {
                debug_assert_eq!(NUM_BASE_COLUMNS, row.len());
                row.encode()
            }
            ProofItem::OutOfDomainExtRow(row) => {
                debug_assert_eq!(NUM_EXT_COLUMNS, row.len());
                row.encode()
            }
            ProofItem::MerkleRoot(something) => something.encode(),
            ProofItem::AuthenticationPath(something) => something.encode(),
            ProofItem::RevealedCombinationElements(something) => something.encode(),
            ProofItem::FriCodeword(something) => something.encode(),
            ProofItem::FriResponse(something) => something.encode(),
            ProofItem::PaddedHeight(something) => something.encode(),
            ProofItem::Uncast(something) => something.encode(),
        };
        let head = BFieldElement::new(tail.len() as u64);
        tail.insert(0, head);
        tail
    }
}

#[cfg(test)]
mod proof_item_typed_tests {
    use itertools::Itertools;
    use rand::thread_rng;
    use rand::Rng;
    use twenty_first::shared_math::tip5::Tip5;
    use twenty_first::shared_math::x_field_element::XFieldElement;

    use crate::proof_stream::ProofStream;

    use super::*;

    fn random_bool() -> bool {
        thread_rng().gen()
    }

    fn random_x_field_element() -> XFieldElement {
        thread_rng().gen()
    }

    fn random_digest() -> Digest {
        thread_rng().gen()
    }

    fn random_fri_response() -> FriResponse {
        FriResponse(
            (0..18)
                .map(|r| {
                    (
                        PartialAuthenticationPath(
                            (0..(20 - r))
                                .map(|_| {
                                    if random_bool() {
                                        Some(random_digest())
                                    } else {
                                        None
                                    }
                                })
                                .collect_vec(),
                        ),
                        random_x_field_element(),
                    )
                })
                .collect_vec(),
        )
    }

    #[test]
    fn serialize_fri_response_test() {
        let fri_response = random_fri_response();
        let str = fri_response.encode();
        let fri_response_ = *FriResponse::decode(&str).unwrap();
        assert_eq!(fri_response, fri_response_);
    }

    #[test]
    fn test_serialize_stark_proof_with_fiat_shamir() {
        type H = Tip5;
        let mut proof_stream: ProofStream<_, H> = ProofStream::new();
        let map = (0..7).map(|_| random_digest()).collect_vec();
        let auth_struct = (0..8)
            .map(|_| {
                PartialAuthenticationPath(
                    (0..11)
                        .map(|_| {
                            if random_bool() {
                                Some(random_digest())
                            } else {
                                None
                            }
                        })
                        .collect_vec(),
                )
            })
            .collect_vec();
        let root = random_digest();
        let fri_response = random_fri_response();

        let mut fs = vec![];
        fs.push(proof_stream.sponge_state.state);
        proof_stream.enqueue(&ProofItem::AuthenticationPath(map.clone()), false);
        fs.push(proof_stream.sponge_state.state);
        proof_stream.enqueue(
            &ProofItem::CompressedAuthenticationPaths(auth_struct.clone()),
            false,
        );
        fs.push(proof_stream.sponge_state.state);
        proof_stream.enqueue(&ProofItem::MerkleRoot(root), true);
        fs.push(proof_stream.sponge_state.state);
        proof_stream.enqueue(&ProofItem::FriResponse(fri_response.clone()), false);
        fs.push(proof_stream.sponge_state.state);

        let proof = proof_stream.to_proof();

        let mut proof_stream_ =
            ProofStream::<ProofItem, H>::from_proof(&proof).expect("invalid parsing of proof");

        let mut fs_ = vec![];
        fs_.push(proof_stream_.sponge_state.state);

        let map_ = proof_stream_
            .dequeue(false)
            .expect("can't dequeue item")
            .as_authentication_path()
            .expect("cannot parse dequeued item");
        assert_eq!(map, map_);
        fs_.push(proof_stream_.sponge_state.state);

        let auth_struct_ = proof_stream_
            .dequeue(false)
            .expect("can't dequeue item")
            .as_compressed_authentication_paths()
            .expect("cannot parse dequeued item");
        assert_eq!(auth_struct, auth_struct_);
        fs_.push(proof_stream_.sponge_state.state);

        let root_ = proof_stream_
            .dequeue(true)
            .expect("can't dequeue item")
            .as_merkle_root()
            .expect("cannot parse dequeued item");
        assert_eq!(root, root_);
        fs_.push(proof_stream_.sponge_state.state);

        let fri_response_ = proof_stream_
            .dequeue(false)
            .expect("can't dequeue item")
            .as_fri_response()
            .expect("cannot parse dequeued item");
        assert_eq!(fri_response, fri_response_);
        fs_.push(proof_stream_.sponge_state.state);

        assert_eq!(fs, fs_);
    }
}
