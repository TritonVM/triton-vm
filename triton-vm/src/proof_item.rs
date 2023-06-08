use anyhow::bail;
use anyhow::Result;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::bfield_codec::BFieldCodec;
use twenty_first::shared_math::tip5::Digest;
use twenty_first::shared_math::x_field_element::XFieldElement;
use twenty_first::util_types::merkle_tree::PartialAuthenticationPath;
use twenty_first::util_types::proof_stream_typed::ProofStreamError;

use crate::table::master_table::NUM_BASE_COLUMNS;
use crate::table::master_table::NUM_EXT_COLUMNS;

type AuthenticationStructure<Digest> = Vec<PartialAuthenticationPath<Digest>>;

/// A `FriResponse` is a vector of partial authentication paths and `XFieldElements`. The
/// `XFieldElements` are the values of the leaves of the Merkle tree. They correspond to the
/// queried index of the FRI codeword (of that round). The corresponding partial authentication
/// paths are the paths from the queried leaf to the root of the Merkle tree.
#[derive(Debug, Clone, PartialEq, Eq, BFieldCodec)]
pub struct FriResponse(pub Vec<(PartialAuthenticationPath<Digest>, XFieldElement)>);

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
}

impl ProofItem {
    /// The unique identifier for this item type.
    pub const fn discriminant(&self) -> BFieldElement {
        use ProofItem::*;
        let discriminant: u64 = match self {
            CompressedAuthenticationPaths(_) => 0,
            MasterBaseTableRows(_) => 1,
            MasterExtTableRows(_) => 2,
            OutOfDomainBaseRow(_) => 3,
            OutOfDomainExtRow(_) => 4,
            MerkleRoot(_) => 5,
            AuthenticationPath(_) => 6,
            RevealedCombinationElements(_) => 7,
            FriCodeword(_) => 8,
            FriResponse(_) => 9,
        };
        BFieldElement::new(discriminant)
    }

    pub fn as_compressed_authentication_paths(&self) -> Result<AuthenticationStructure<Digest>> {
        match self {
            Self::CompressedAuthenticationPaths(caps) => Ok(caps.to_owned()),
            other => bail!(ProofStreamError::new(&format!(
                "expected compressed authentication paths, but got {other:?}",
            ))),
        }
    }

    pub fn as_master_base_table_rows(&self) -> Result<Vec<Vec<BFieldElement>>> {
        match self {
            Self::MasterBaseTableRows(bss) => Ok(bss.to_owned()),
            other => bail!(ProofStreamError::new(&format!(
                "expected master base table rows, but got something {other:?}",
            ))),
        }
    }

    pub fn as_master_ext_table_rows(&self) -> Result<Vec<Vec<XFieldElement>>> {
        match self {
            Self::MasterExtTableRows(xss) => Ok(xss.to_owned()),
            other => bail!(ProofStreamError::new(&format!(
                "expected master extension table rows, but got {other:?}",
            ))),
        }
    }

    pub fn as_out_of_domain_base_row(&self) -> Result<Vec<XFieldElement>> {
        match self {
            Self::OutOfDomainBaseRow(xs) => Ok(xs.to_owned()),
            other => bail!(ProofStreamError::new(&format!(
                "expected out of domain base row, but got {other:?}",
            ))),
        }
    }

    pub fn as_out_of_domain_ext_row(&self) -> Result<Vec<XFieldElement>> {
        match self {
            Self::OutOfDomainExtRow(xs) => Ok(xs.to_owned()),
            other => bail!(ProofStreamError::new(&format!(
                "expected out of domain extension row, but got {other:?}",
            ))),
        }
    }

    pub fn as_merkle_root(&self) -> Result<Digest> {
        match self {
            Self::MerkleRoot(bs) => Ok(*bs),
            other => bail!(ProofStreamError::new(&format!(
                "expected merkle root, but got {other:?}",
            ))),
        }
    }

    pub fn as_authentication_path(&self) -> Result<Vec<Digest>> {
        match self {
            Self::AuthenticationPath(bss) => Ok(bss.to_owned()),
            other => bail!(ProofStreamError::new(&format!(
                "expected authentication path, but got {other:?}",
            ))),
        }
    }

    pub fn as_revealed_combination_elements(&self) -> Result<Vec<XFieldElement>> {
        match self {
            Self::RevealedCombinationElements(xs) => Ok(xs.to_owned()),
            other => bail!(ProofStreamError::new(&format!(
                "expected revealed combination elements, but got {other:?}",
            ))),
        }
    }

    pub fn as_fri_codeword(&self) -> Result<Vec<XFieldElement>> {
        match self {
            Self::FriCodeword(xs) => Ok(xs.to_owned()),
            other => bail!(ProofStreamError::new(&format!(
                "expected FRI codeword, but got {other:?}",
            ))),
        }
    }

    pub fn as_fri_response(&self) -> Result<FriResponse> {
        match self {
            Self::FriResponse(fri_proof) => Ok(fri_proof.to_owned()),
            other => bail!(ProofStreamError::new(&format!(
                "expected FRI proof, but got {other:?}"
            ),)),
        }
    }
}

impl BFieldCodec for ProofItem {
    /// Turn the given string of BFieldElements into a ProofItem. The first element indicates the
    /// field type, and the rest of the elements are the data for the item.
    fn decode(str: &[BFieldElement]) -> Result<Box<Self>> {
        if str.is_empty() {
            bail!(ProofStreamError::new("empty buffer"));
        }

        let discriminant = str[0].value();
        let str = &str[1..];
        let item = match discriminant {
            0 => Self::CompressedAuthenticationPaths(*AuthenticationStructure::decode(str)?),
            1 => Self::MasterBaseTableRows(*Vec::<Vec<BFieldElement>>::decode(str)?),
            2 => Self::MasterExtTableRows(*Vec::<Vec<XFieldElement>>::decode(str)?),
            3 => Self::OutOfDomainBaseRow(*Vec::<XFieldElement>::decode(str)?),
            4 => Self::OutOfDomainExtRow(*Vec::<XFieldElement>::decode(str)?),
            5 => Self::MerkleRoot(*Digest::decode(str)?),
            6 => Self::AuthenticationPath(*Vec::<Digest>::decode(str)?),
            7 => Self::RevealedCombinationElements(*Vec::<XFieldElement>::decode(str)?),
            8 => Self::FriCodeword(*Vec::<XFieldElement>::decode(str)?),
            9 => Self::FriResponse(*FriResponse::decode(str)?),
            i => bail!(ProofStreamError::new(&format!(
                "Unknown discriminant {i} for ProofItem."
            ))),
        };
        Ok(Box::new(item))
    }

    /// Encode the ProofItem as a string of BFieldElements, with the first element denoting the
    /// length of the rest.
    fn encode(&self) -> Vec<BFieldElement> {
        use ProofItem::*;

        #[cfg(debug_assertions)]
        match self {
            OutOfDomainBaseRow(row) => assert_eq!(NUM_BASE_COLUMNS, row.len()),
            OutOfDomainExtRow(row) => assert_eq!(NUM_EXT_COLUMNS, row.len()),
            _ => (),
        }

        let discriminant = vec![self.discriminant()];
        let encoding = match self {
            CompressedAuthenticationPaths(something) => something.encode(),
            MasterBaseTableRows(something) => something.encode(),
            MasterExtTableRows(something) => something.encode(),
            OutOfDomainBaseRow(row) => row.encode(),
            OutOfDomainExtRow(row) => row.encode(),
            MerkleRoot(something) => something.encode(),
            AuthenticationPath(something) => something.encode(),
            RevealedCombinationElements(something) => something.encode(),
            FriCodeword(something) => something.encode(),
            FriResponse(something) => something.encode(),
        };
        [discriminant, encoding].concat()
    }

    fn static_length() -> Option<usize> {
        None
    }
}

#[cfg(test)]
mod proof_item_typed_tests {
    use itertools::Itertools;
    use rand::distributions::Standard;
    use rand::prelude::StdRng;
    use rand::random;
    use rand::Rng;
    use rand_core::RngCore;
    use rand_core::SeedableRng;
    use twenty_first::shared_math::tip5::Tip5;
    use twenty_first::shared_math::x_field_element::XFieldElement;
    use twenty_first::util_types::merkle_tree::CpuParallel;
    use twenty_first::util_types::merkle_tree::MerkleTree;
    use twenty_first::util_types::merkle_tree_maker::MerkleTreeMaker;

    use crate::proof::Proof;
    use crate::proof_stream::ProofStream;

    use super::*;

    fn random_merkle_tree(seed: u64, num_leaves: usize) -> (MerkleTree<Tip5>, Vec<XFieldElement>) {
        let rng = StdRng::seed_from_u64(seed);
        let leaves: Vec<XFieldElement> = rng.sample_iter(Standard).take(num_leaves).collect();
        let leaves_as_digests: Vec<Digest> = leaves.iter().map(|&x| x.into()).collect();
        (CpuParallel::from_digests(&leaves_as_digests), leaves)
    }

    fn fri_response(
        merkle_tree: &MerkleTree<Tip5>,
        leaves: &[XFieldElement],
        revealed_indices: &[usize],
    ) -> FriResponse {
        let revealed_elements = revealed_indices.iter().map(|&i| leaves[i]).collect_vec();
        let auth_structure = merkle_tree.get_authentication_structure(revealed_indices);
        let fri_response = auth_structure
            .into_iter()
            .zip(revealed_elements)
            .collect_vec();
        FriResponse(fri_response)
    }

    #[test]
    fn serialize_fri_response_test() {
        type H = Tip5;

        let seed = random();
        let mut rng = StdRng::seed_from_u64(seed);
        println!("seed: {seed}");

        let codeword_len = 64;
        let (merkle_tree, leaves) = random_merkle_tree(rng.next_u64(), codeword_len);
        let num_indices = rng.gen_range(1..=codeword_len);
        let revealed_indices = (0..num_indices)
            .map(|_| rng.gen_range(0..codeword_len))
            .collect_vec();
        let fri_response = fri_response(&merkle_tree, &leaves, &revealed_indices);

        // test encoding and decoding in isolation
        let encoding = fri_response.encode();
        let fri_response_ = *FriResponse::decode(&encoding).unwrap();
        assert_eq!(fri_response, fri_response_);

        // test encoding and decoding in a stream
        let mut proof_stream = ProofStream::<H>::new();
        proof_stream.enqueue(&ProofItem::FriResponse(fri_response.clone()), false);
        let proof: Proof = proof_stream.into();
        let mut proof_stream = ProofStream::<H>::try_from(&proof).unwrap();
        let fri_response_ = proof_stream.dequeue(false).unwrap();
        let fri_response_ = fri_response_.as_fri_response().unwrap();
        assert_eq!(fri_response, fri_response_);
    }
}
