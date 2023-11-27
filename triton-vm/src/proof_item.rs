use arbitrary::Arbitrary;
use strum::Display;
use strum::EnumCount;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::bfield_codec::BFieldCodec;
use twenty_first::shared_math::tip5::Digest;
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::fri::AuthenticationStructure;
use crate::stark::NUM_QUOTIENT_SEGMENTS;

/// A `FriResponse` is an `AuthenticationStructure` together with the values of the
/// revealed leaves of the Merkle tree. Together, they correspond to the
/// queried indices of the FRI codeword (of that round).
#[derive(Debug, Clone, PartialEq, Eq, Hash, BFieldCodec, Arbitrary)]
pub struct FriResponse {
    /// The authentication structure of the Merkle tree.
    pub auth_structure: AuthenticationStructure,
    /// The values of the opened leaves of the Merkle tree.
    pub revealed_leaves: Vec<XFieldElement>,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash, Display, EnumCount, BFieldCodec, Arbitrary)]
pub enum ProofItem {
    AuthenticationStructure(AuthenticationStructure),
    MasterBaseTableRows(Vec<Vec<BFieldElement>>),
    MasterExtTableRows(Vec<Vec<XFieldElement>>),
    OutOfDomainBaseRow(Vec<XFieldElement>),
    OutOfDomainExtRow(Vec<XFieldElement>),
    OutOfDomainQuotientSegments([XFieldElement; NUM_QUOTIENT_SEGMENTS]),
    MerkleRoot(Digest),
    Log2PaddedHeight(u32),
    QuotientSegmentsElements(Vec<[XFieldElement; NUM_QUOTIENT_SEGMENTS]>),
    FriCodeword(Vec<XFieldElement>),
    FriResponse(FriResponse),
}

impl ProofItem {
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
            OutOfDomainQuotientSegments(_) => true,
            // all of the following are implied by a corresponding Merkle root
            AuthenticationStructure(_) => false,
            MasterBaseTableRows(_) => false,
            MasterExtTableRows(_) => false,
            Log2PaddedHeight(_) => false,
            QuotientSegmentsElements(_) => false,
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

    pub fn as_out_of_domain_quotient_segments(
        &self,
    ) -> Result<[XFieldElement; NUM_QUOTIENT_SEGMENTS]> {
        match self {
            Self::OutOfDomainQuotientSegments(xs) => Ok(*xs),
            other => bail!("expected out of domain quotient segments, but got {other:?}",),
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

    pub fn as_quotient_segments_elements(
        &self,
    ) -> Result<Vec<[XFieldElement; NUM_QUOTIENT_SEGMENTS]>> {
        match self {
            Self::QuotientSegmentsElements(xs) => Ok(xs.to_owned()),
            other => bail!("expected quotient segments' elements, but got {other:?}",),
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
}

#[cfg(test)]
mod tests {
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
    fn serialize_fri_response() {
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
        proof_stream.enqueue(ProofItem::FriResponse(fri_response.clone()));
        let proof: Proof = proof_stream.into();
        let mut proof_stream = ProofStream::<H>::try_from(&proof).unwrap();
        let fri_response_ = proof_stream.dequeue().unwrap();
        let fri_response_ = fri_response_.as_fri_response().unwrap();
        assert_eq!(fri_response, fri_response_);
    }

    #[test]
    fn serialize_authentication_structure() {
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
        proof_stream.enqueue(ProofItem::AuthenticationStructure(auth_structure.clone()));
        let proof: Proof = proof_stream.into();
        let mut proof_stream = ProofStream::<H>::try_from(&proof).unwrap();
        let auth_structure_ = proof_stream.dequeue().unwrap();
        let auth_structure_ = auth_structure_.as_authentication_structure().unwrap();
        assert_eq!(auth_structure, auth_structure_);
    }
}
