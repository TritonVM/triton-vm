use arbitrary::Arbitrary;
use strum::Display;
use strum::EnumCount;
use strum::EnumDiscriminants;
use strum::EnumIter;
use twenty_first::prelude::*;

use crate::error::ProofStreamError;
use crate::error::ProofStreamError::UnexpectedItem;
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

macro_rules! proof_items {
    ($($variant:ident($payload:ty) => $in_fiat_shamir_heuristic:literal, $try_into_fn:ident,)+) => {
        #[derive(
            Debug,
            Clone,
            PartialEq,
            Eq,
            Hash,
            Display,
            EnumCount,
            EnumDiscriminants,
            BFieldCodec,
            Arbitrary
        )]
        #[strum_discriminants(name(ProofItemVariant))]
        #[strum_discriminants(derive(Display, Arbitrary, BFieldCodec, EnumIter))]
        pub enum ProofItem {
            $( $variant($payload), )+
        }

        impl ProofItem {
            /// Whether a given proof item should be considered in the Fiat-Shamir heuristic.
            /// The Fiat-Shamir heuristic is sound only if all elements in the (current) transcript
            /// are considered. However, certain elements indirectly appear more than once. For
            /// example, a Merkle root is a commitment to any number of elements. If the Merkle root
            /// is part of the transcript, has been considered in the Fiat-Shamir heuristic, and
            /// assuming collision resistance of the hash function in use, none of the committed-to
            /// elements have to be considered in the Fiat-Shamir heuristic again.
            /// This also extends to the authentication structure of these elements, et cetera.
            pub const fn include_in_fiat_shamir_heuristic(&self) -> bool {
                match self {
                    $( Self::$variant(_) => $in_fiat_shamir_heuristic, )+
                }
            }

            $(
            pub fn $try_into_fn(self) -> Result<$payload, ProofStreamError> {
                match self {
                    Self::$variant(payload) => Ok(payload),
                    _ => Err(UnexpectedItem {
                        expected: ProofItemVariant::$variant,
                        got: self,
                    }),
                }
            }
            )+
        }

        impl ProofItemVariant {
            pub fn payload_static_length(self) -> Option<usize> {
                match self {
                    $( Self::$variant => <$payload>::static_length(), )+
                }
            }

            /// See [`ProofItem::include_in_fiat_shamir_heuristic`].
            pub const fn include_in_fiat_shamir_heuristic(&self) -> bool {
                match self {
                    $( Self::$variant => $in_fiat_shamir_heuristic, )+
                }
            }
        }
    };
}

proof_items!(
    MerkleRoot(Digest) => true, try_into_merkle_root,
    OutOfDomainBaseRow(Vec<XFieldElement>) => true, try_into_out_of_domain_base_row,
    OutOfDomainExtRow(Vec<XFieldElement>) => true, try_into_out_of_domain_ext_row,
    OutOfDomainQuotientSegments([XFieldElement; NUM_QUOTIENT_SEGMENTS]) => true,
        try_into_out_of_domain_quot_segments,

    // the following are implied by some Merkle root, thus not included in the Fiat-Shamir heuristic
    AuthenticationStructure(AuthenticationStructure) => false, try_into_authentication_structure,
    MasterBaseTableRows(Vec<Vec<BFieldElement>>) => false, try_into_master_base_table_rows,
    MasterExtTableRows(Vec<Vec<XFieldElement>>) => false, try_into_master_ext_table_rows,
    Log2PaddedHeight(u32) => false, try_into_log2_padded_height,
    QuotientSegmentsElements(Vec<[XFieldElement; NUM_QUOTIENT_SEGMENTS]>) => false,
        try_into_quot_segments_elements,
    FriCodeword(Vec<XFieldElement>) => false, try_into_fri_codeword,
    FriResponse(FriResponse) => false, try_into_fri_response,
);

#[cfg(test)]
pub(crate) mod tests {
    use std::collections::HashSet;

    use assert2::assert;
    use assert2::let_assert;
    use proptest::prelude::*;
    use strum::IntoEnumIterator;
    use test_strategy::proptest;
    use twenty_first::prelude::Tip5;

    use crate::proof::Proof;
    use crate::proof_stream::ProofStream;
    use crate::shared_tests::LeavedMerkleTreeTestData;

    use super::*;

    #[proptest]
    fn serialize_fri_response_in_isolation(leaved_merkle_tree: LeavedMerkleTreeTestData) {
        let fri_response = leaved_merkle_tree.into_fri_response();
        let encoding = fri_response.encode();
        let_assert!(Ok(decoding) = FriResponse::decode(&encoding));
        prop_assert_eq!(fri_response, *decoding);
    }

    #[proptest]
    fn serialize_fri_response_in_proof_stream(leaved_merkle_tree: LeavedMerkleTreeTestData) {
        let fri_response = leaved_merkle_tree.into_fri_response();
        let mut proof_stream = ProofStream::<Tip5>::new();
        proof_stream.enqueue(ProofItem::FriResponse(fri_response.clone()));
        let proof: Proof = proof_stream.into();

        let_assert!(Ok(mut proof_stream) = ProofStream::<Tip5>::try_from(&proof));
        let_assert!(Ok(proof_item) = proof_stream.dequeue());
        let_assert!(Ok(fri_response_) = proof_item.try_into_fri_response());
        prop_assert_eq!(fri_response, fri_response_);
    }

    #[proptest]
    fn serialize_authentication_structure_in_isolation(
        leaved_merkle_tree: LeavedMerkleTreeTestData,
    ) {
        let auth_structure = leaved_merkle_tree.auth_structure;
        let encoding = auth_structure.encode();
        let_assert!(Ok(decoding) = AuthenticationStructure::decode(&encoding));
        prop_assert_eq!(auth_structure, *decoding);
    }

    #[proptest]
    fn serialize_authentication_structure_in_proof_stream(
        leaved_merkle_tree: LeavedMerkleTreeTestData,
    ) {
        let auth_structure = leaved_merkle_tree.auth_structure;
        let mut proof_stream = ProofStream::<Tip5>::new();
        proof_stream.enqueue(ProofItem::AuthenticationStructure(auth_structure.clone()));
        let proof: Proof = proof_stream.into();

        let_assert!(Ok(mut proof_stream) = ProofStream::<Tip5>::try_from(&proof));
        let_assert!(Ok(proof_item) = proof_stream.dequeue());
        let_assert!(Ok(auth_structure_) = proof_item.try_into_authentication_structure());
        prop_assert_eq!(auth_structure, auth_structure_);
    }

    #[test]
    fn interpreting_a_merkle_root_as_anything_else_gives_appropriate_error() {
        let fake_root = Digest::default();
        let item = ProofItem::MerkleRoot(fake_root);
        assert!(let Err(UnexpectedItem{..}) = item.clone().try_into_authentication_structure());
        assert!(let Err(UnexpectedItem{..}) = item.clone().try_into_fri_response());
        assert!(let Err(UnexpectedItem{..}) = item.clone().try_into_master_base_table_rows());
        assert!(let Err(UnexpectedItem{..}) = item.clone().try_into_master_ext_table_rows());
        assert!(let Err(UnexpectedItem{..}) = item.clone().try_into_out_of_domain_base_row());
        assert!(let Err(UnexpectedItem{..}) = item.clone().try_into_out_of_domain_ext_row());
        assert!(let Err(UnexpectedItem{..}) = item.clone().try_into_out_of_domain_quot_segments());
        assert!(let Err(UnexpectedItem{..}) = item.clone().try_into_log2_padded_height());
        assert!(let Err(UnexpectedItem{..}) = item.clone().try_into_quot_segments_elements());
        assert!(let Err(UnexpectedItem{..}) = item.clone().try_into_fri_codeword());
        assert!(let Err(UnexpectedItem{..}) = item.try_into_fri_response());
    }

    #[test]
    fn proof_item_payload_static_length_is_as_expected() {
        assert!(let Some(_) =  ProofItemVariant::MerkleRoot.payload_static_length());
        assert!(let Some(_) =  ProofItemVariant::Log2PaddedHeight.payload_static_length());
        assert_eq!(None, ProofItemVariant::FriCodeword.payload_static_length());
        assert_eq!(None, ProofItemVariant::FriResponse.payload_static_length());
    }

    #[test]
    fn can_loop_over_proof_item_variants() {
        let mut all_discriminants = HashSet::new();
        for variant in ProofItemVariant::iter() {
            all_discriminants.insert(variant.bfield_codec_discriminant());
        }

        assert_eq!(ProofItem::COUNT, all_discriminants.len());
    }

    #[test]
    fn proof_item_and_its_variant_have_same_bfield_codec_discriminant() {
        assert_eq!(
            ProofItem::MerkleRoot(Digest::default()).bfield_codec_discriminant(),
            ProofItemVariant::MerkleRoot.bfield_codec_discriminant()
        );
        assert_eq!(
            ProofItem::Log2PaddedHeight(0).bfield_codec_discriminant(),
            ProofItemVariant::Log2PaddedHeight.bfield_codec_discriminant()
        );
        assert_eq!(
            ProofItem::FriCodeword(vec![]).bfield_codec_discriminant(),
            ProofItemVariant::FriCodeword.bfield_codec_discriminant()
        );
    }
}
