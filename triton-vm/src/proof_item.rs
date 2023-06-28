use anyhow::bail;
use anyhow::Result;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::bfield_codec::BFieldCodec;
use twenty_first::shared_math::tip5::Digest;
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::Claim;

type AuthenticationStructure = Vec<Digest>;

/// A `FriResponse` is an [`AuthenticationStructure`] together with the values of the
/// revealed leaves of the Merkle tree. Together, they correspond to the
/// queried indices of the FRI codeword (of that round).
#[derive(Debug, Clone, PartialEq, Eq, BFieldCodec)]
pub struct FriResponse {
    /// The authentication structure of the Merkle tree.
    pub auth_structure: AuthenticationStructure,
    /// The values of the opened leaves of the Merkle tree.
    pub revealed_leaves: Vec<XFieldElement>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProofItem {
    AuthenticationStructure(AuthenticationStructure),
    MasterBaseTableRows(Vec<Vec<BFieldElement>>),
    MasterExtTableRows(Vec<Vec<XFieldElement>>),
    OutOfDomainBaseRow(Vec<XFieldElement>),
    OutOfDomainExtRow(Vec<XFieldElement>),
    MerkleRoot(Digest),
    Log2PaddedHeight(u32),
    RevealedCombinationElements(Vec<XFieldElement>),
    FriCodeword(Vec<XFieldElement>),
    FriResponse(FriResponse),
    Claim(Claim),
}

impl ProofItem {
    /// The unique identifier for this item type.
    pub const fn discriminant(&self) -> BFieldElement {
        use ProofItem::*;
        let discriminant: u64 = match self {
            AuthenticationStructure(_) => 0,
            MasterBaseTableRows(_) => 1,
            MasterExtTableRows(_) => 2,
            OutOfDomainBaseRow(_) => 3,
            OutOfDomainExtRow(_) => 4,
            MerkleRoot(_) => 5,
            Log2PaddedHeight(_) => 6,
            RevealedCombinationElements(_) => 7,
            FriCodeword(_) => 8,
            FriResponse(_) => 9,
            Claim(_) => 10,
        };
        BFieldElement::new(discriminant)
    }

    /// Whether a given proof item should be considered in the Fiat-Shamir heuristic.
    /// The Fiat-Shamir heuristic is sound only if all elements in the (current) transcript are
    /// considered. However, certain elements indirectly appear more than once. For example, a
    /// Merkle root is a commitment to any number of elements. If the Merkle root is part of the
    /// transcript, has been considered in the Fiat-Shamir heuristic, and assuming collision
    /// resistance of the hash function in use, none of the committed-to elements have to be
    /// considered in the Fiat-Shamir heuristic again.
    /// This also extends to the authentication structure of these elements, et cetera.
    pub const fn include_in_fiat_shamir_heuristic(&self) -> bool {
        use ProofItem::*;
        match self {
            MerkleRoot(_) => true,
            OutOfDomainBaseRow(_) => true,
            OutOfDomainExtRow(_) => true,
            Claim(_) => true,
            // all of the following are implied by a corresponding Merkle root
            AuthenticationStructure(_) => false,
            MasterBaseTableRows(_) => false,
            MasterExtTableRows(_) => false,
            Log2PaddedHeight(_) => false,
            RevealedCombinationElements(_) => false,
            FriCodeword(_) => false,
            FriResponse(_) => false,
        }
    }

    pub fn as_authentication_structure(&self) -> Result<AuthenticationStructure> {
        match self {
            Self::AuthenticationStructure(caps) => Ok(caps.to_owned()),
            other => bail!("expected authentication structure, but got {other:?}",),
        }
    }

    pub fn as_master_base_table_rows(&self) -> Result<Vec<Vec<BFieldElement>>> {
        match self {
            Self::MasterBaseTableRows(bss) => Ok(bss.to_owned()),
            other => bail!("expected master base table rows, but got something {other:?}",),
        }
    }

    pub fn as_master_ext_table_rows(&self) -> Result<Vec<Vec<XFieldElement>>> {
        match self {
            Self::MasterExtTableRows(xss) => Ok(xss.to_owned()),
            other => bail!("expected master extension table rows, but got {other:?}",),
        }
    }

    pub fn as_out_of_domain_base_row(&self) -> Result<Vec<XFieldElement>> {
        match self {
            Self::OutOfDomainBaseRow(xs) => Ok(xs.to_owned()),
            other => bail!("expected out of domain base row, but got {other:?}",),
        }
    }

    pub fn as_out_of_domain_ext_row(&self) -> Result<Vec<XFieldElement>> {
        match self {
            Self::OutOfDomainExtRow(xs) => Ok(xs.to_owned()),
            other => bail!("expected out of domain extension row, but got {other:?}",),
        }
    }

    pub fn as_merkle_root(&self) -> Result<Digest> {
        match self {
            Self::MerkleRoot(bs) => Ok(*bs),
            other => bail!("expected merkle root, but got {other:?}",),
        }
    }

    pub fn as_log2_padded_height(&self) -> Result<u32> {
        match self {
            Self::Log2PaddedHeight(log2_padded_height) => Ok(*log2_padded_height),
            other => bail!("expected log2 padded height, but got {other:?}",),
        }
    }

    pub fn as_revealed_combination_elements(&self) -> Result<Vec<XFieldElement>> {
        match self {
            Self::RevealedCombinationElements(xs) => Ok(xs.to_owned()),
            other => bail!("expected revealed combination elements, but got {other:?}",),
        }
    }

    pub fn as_fri_codeword(&self) -> Result<Vec<XFieldElement>> {
        match self {
            Self::FriCodeword(xs) => Ok(xs.to_owned()),
            other => bail!("expected FRI codeword, but got {other:?}",),
        }
    }

    pub fn as_fri_response(&self) -> Result<FriResponse> {
        match self {
            Self::FriResponse(fri_proof) => Ok(fri_proof.to_owned()),
            other => bail!("expected FRI proof, but got {other:?}"),
        }
    }

    pub fn as_claim(&self) -> Result<Claim> {
        match self {
            Self::Claim(claim) => Ok(claim.to_owned()),
            other => bail!("expected claim, but got {other:?}"),
        }
    }
}

impl BFieldCodec for ProofItem {
    /// Turn the given string of BFieldElements into a ProofItem. The first element indicates the
    /// field type, and the rest of the elements are the data for the item.
    fn decode(str: &[BFieldElement]) -> Result<Box<Self>> {
        if str.is_empty() {
            bail!("empty buffer");
        }

        let discriminant = str[0].value();
        let str = &str[1..];
        let item = match discriminant {
            0 => Self::AuthenticationStructure(*AuthenticationStructure::decode(str)?),
            1 => Self::MasterBaseTableRows(*Vec::<Vec<BFieldElement>>::decode(str)?),
            2 => Self::MasterExtTableRows(*Vec::<Vec<XFieldElement>>::decode(str)?),
            3 => Self::OutOfDomainBaseRow(*Vec::<XFieldElement>::decode(str)?),
            4 => Self::OutOfDomainExtRow(*Vec::<XFieldElement>::decode(str)?),
            5 => Self::MerkleRoot(*Digest::decode(str)?),
            6 => Self::Log2PaddedHeight(*u32::decode(str)?),
            7 => Self::RevealedCombinationElements(*Vec::<XFieldElement>::decode(str)?),
            8 => Self::FriCodeword(*Vec::<XFieldElement>::decode(str)?),
            9 => Self::FriResponse(*FriResponse::decode(str)?),
            10 => Self::Claim(*Claim::decode(str)?),
            i => bail!("Unknown discriminant {i} for ProofItem."),
        };
        Ok(Box::new(item))
    }

    /// Encode the ProofItem as a string of BFieldElements, with the first element denoting the
    /// length of the rest.
    fn encode(&self) -> Vec<BFieldElement> {
        use ProofItem::*;

        #[cfg(debug_assertions)]
        {
            use crate::table::master_table::NUM_BASE_COLUMNS;
            use crate::table::master_table::NUM_EXT_COLUMNS;
            match self {
                OutOfDomainBaseRow(row) => assert_eq!(NUM_BASE_COLUMNS, row.len()),
                OutOfDomainExtRow(row) => assert_eq!(NUM_EXT_COLUMNS, row.len()),
                _ => (),
            }
        }

        let discriminant = vec![self.discriminant()];
        let encoding = match self {
            AuthenticationStructure(something) => something.encode(),
            MasterBaseTableRows(something) => something.encode(),
            MasterExtTableRows(something) => something.encode(),
            OutOfDomainBaseRow(row) => row.encode(),
            OutOfDomainExtRow(row) => row.encode(),
            MerkleRoot(something) => something.encode(),
            Log2PaddedHeight(height) => height.encode(),
            RevealedCombinationElements(something) => something.encode(),
            FriCodeword(something) => something.encode(),
            FriResponse(something) => something.encode(),
            Claim(something) => something.encode(),
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

    /// Pseudo-randomly generates an indicated number of leaves,
    /// the corresponding Merkle tree, and returns both.
    fn random_merkle_tree(seed: u64, num_leaves: usize) -> (MerkleTree<Tip5>, Vec<XFieldElement>) {
        let rng = StdRng::seed_from_u64(seed);
        let leaves: Vec<XFieldElement> = rng.sample_iter(Standard).take(num_leaves).collect();
        let leaves_as_digests: Vec<Digest> = leaves.iter().map(|&x| x.into()).collect();
        (CpuParallel::from_digests(&leaves_as_digests), leaves)
    }

    /// Given a Merkle tree and a set of leaves,
    /// return a FRI response for the given revealed indices.
    fn fri_response(
        merkle_tree: &MerkleTree<Tip5>,
        leaves: &[XFieldElement],
        revealed_indices: &[usize],
    ) -> FriResponse {
        let revealed_leaves = revealed_indices.iter().map(|&i| leaves[i]).collect_vec();
        let auth_structure = merkle_tree.get_authentication_structure(revealed_indices);
        FriResponse {
            auth_structure,
            revealed_leaves,
        }
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
        proof_stream.enqueue(&ProofItem::FriResponse(fri_response.clone()));
        let proof: Proof = proof_stream.into();
        let mut proof_stream = ProofStream::<H>::try_from(&proof).unwrap();
        let fri_response_ = proof_stream.dequeue().unwrap();
        let fri_response_ = fri_response_.as_fri_response().unwrap();
        assert_eq!(fri_response, fri_response_);
    }

    #[test]
    fn serialize_authentication_structure_test() {
        type H = Tip5;

        let seed = random();
        let mut rng = StdRng::seed_from_u64(seed);
        println!("seed: {seed}");

        let codeword_len = 64;
        let (merkle_tree, _) = random_merkle_tree(rng.next_u64(), codeword_len);
        let num_indices = rng.gen_range(1..=codeword_len);
        let revealed_indices = (0..num_indices)
            .map(|_| rng.gen_range(0..codeword_len))
            .collect_vec();
        let auth_structure = merkle_tree.get_authentication_structure(&revealed_indices);

        // test encoding and decoding in isolation
        let encoding = auth_structure.encode();
        let auth_structure_ = *AuthenticationStructure::decode(&encoding).unwrap();
        assert_eq!(auth_structure, auth_structure_);

        // test encoding and decoding in a stream
        let mut proof_stream = ProofStream::<H>::new();
        proof_stream.enqueue(&ProofItem::AuthenticationStructure(auth_structure.clone()));
        let proof: Proof = proof_stream.into();
        let mut proof_stream = ProofStream::<H>::try_from(&proof).unwrap();
        let auth_structure_ = proof_stream.dequeue().unwrap();
        let auth_structure_ = auth_structure_.as_authentication_structure().unwrap();
        assert_eq!(auth_structure, auth_structure_);
    }
}
