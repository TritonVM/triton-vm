use std::fmt::Display;
use std::fmt::Formatter;

use arbitrary::Arbitrary;
use get_size::GetSize;
use itertools::Itertools;
use serde::Deserialize;
use serde::Serialize;
use strum::Display;
use strum::EnumCount;
use strum::EnumDiscriminants;
use strum::EnumIter;
use twenty_first::prelude::*;

use crate::error::ProofStreamError;
use crate::fri::AuthenticationStructure;
use crate::program::Program;
use crate::table::BaseRow;
use crate::table::ExtensionRow;
use crate::table::QuotientSegments;

pub(crate) const CURRENT_PROOF_VERSION: Version = Version(42_000);

/// The version of the [`Proof`] format and, transitively, the [`Stream`].
#[derive(
    Debug, Copy, Clone, Eq, PartialEq, Serialize, Deserialize, GetSize, BFieldCodec, Arbitrary,
)]
pub struct Version(u32);

impl Display for Version {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        let precision = 1000;
        let patch = self.0 % precision;
        let minor = (self.0 / precision) % precision;
        let major = self.0 / (precision * precision);

        write!(f, "{major:03}_{minor:03}_{patch:03}")
    }
}

/// Contains the necessary cryptographic information to verify a computation.
/// Should be used together with a [`Claim`].
///
/// See also [`Stream`].
#[derive(Debug, Clone, Eq, PartialEq, Serialize, Deserialize, GetSize, BFieldCodec, Arbitrary)]
pub struct Proof {
    version: Version,

    // Kept Separate for future-proofing.
    encoded_proof_stream: Vec<BFieldElement>,
}

impl Proof {
    /// Get the height of the trace used during proof generation.
    /// This is an upper bound on the length of the computation this proof is for.
    /// It is one of the main contributing factors to the length of the FRI domain.
    pub fn padded_height(&self) -> Result<usize, ProofStreamError> {
        let proof_stream = Stream::try_from(self)?;
        let proof_items = proof_stream.items.into_iter();
        let log_2_padded_heights = proof_items
            .filter_map(|item| item.try_into_log2_padded_height().ok())
            .collect_vec();

        if log_2_padded_heights.is_empty() {
            return Err(ProofStreamError::NoLog2PaddedHeight);
        }
        let [log_2_height] = log_2_padded_heights[..] else {
            return Err(ProofStreamError::TooManyLog2PaddedHeights);
        };
        Ok(1 << log_2_height)
    }
}

/// Contains the public information of a verifiably correct computation.
/// A corresponding [`Proof`] is needed to verify the computation.
/// One additional piece of public information not explicitly listed in the [`Claim`] is the
/// `padded_height`, an upper bound on the length of the computation.
/// It is derivable from a [`Proof`] by calling [`Proof::padded_height()`].
#[derive(
    Debug, Clone, Eq, PartialEq, Hash, Serialize, Deserialize, GetSize, BFieldCodec, Arbitrary,
)]
pub struct Claim {
    /// The hash digest of the program that was executed. The hash function in use is Tip5.
    pub program_digest: Digest,

    /// The public input to the computation.
    pub input: Vec<BFieldElement>,

    /// The public output of the computation.
    pub output: Vec<BFieldElement>,
}

impl Claim {
    pub fn new(program_digest: Digest) -> Self {
        Self {
            program_digest,
            input: vec![],
            output: vec![],
        }
    }

    #[must_use]
    pub fn about_program(program: &Program) -> Self {
        Self::new(program.hash())
    }

    #[must_use]
    pub fn with_input(mut self, input: Vec<BFieldElement>) -> Self {
        self.input = input;
        self
    }

    #[must_use]
    pub fn with_output(mut self, output: Vec<BFieldElement>) -> Self {
        self.output = output;
        self
    }
}

/// A stream of proof [`Item`]s.
#[derive(Debug, Default, Clone, Eq, PartialEq, BFieldCodec, Arbitrary)]
pub struct Stream {
    pub items: Vec<Item>,

    #[bfield_codec(ignore)]
    pub items_index: usize,

    #[bfield_codec(ignore)]
    pub sponge: Tip5,
}

impl Stream {
    pub fn new() -> Self {
        Stream {
            items: vec![],
            items_index: 0,
            sponge: Tip5::init(),
        }
    }

    /// Alters the Fiat-Shamir's sponge state with the encoding of the given item.
    /// Does _not_ record the given item in the proof stream.
    /// This is useful for items that are not sent to the verifier, _e.g._, the
    /// [`Claim`].
    ///
    /// See also [`Self::enqueue()`] and [`Self::dequeue()`].
    pub fn alter_fiat_shamir_state_with(&mut self, item: &impl BFieldCodec) {
        self.sponge.pad_and_absorb_all(&item.encode())
    }

    /// Send a proof item as prover to verifier.
    /// Some items do not need to be included in the Fiat-Shamir heuristic, _i.e._, they do not
    /// need to modify the sponge state. For those items, namely those that evaluate to `false`
    /// according to [`Item::include_in_fiat_shamir_heuristic`], the sponge state is not
    /// modified.
    /// For example:
    /// - Merkle authentication structure do not need to be hashed if the root of the tree
    ///     in question was hashed previously.
    /// - If the proof stream is not used to sample any more randomness, _i.e._, after the last
    ///     round of interaction, no further items need to be hashed.
    pub fn enqueue(&mut self, item: Item) {
        if item.include_in_fiat_shamir_heuristic() {
            self.alter_fiat_shamir_state_with(&item);
        }
        self.items.push(item);
    }

    /// Receive a proof item from prover as verifier.
    /// See [`Stream::enqueue`] for more details.
    pub fn dequeue(&mut self) -> Result<Item, ProofStreamError> {
        let Some(item) = self.items.get(self.items_index) else {
            return Err(ProofStreamError::EmptyQueue);
        };
        let item = item.to_owned();
        if item.include_in_fiat_shamir_heuristic() {
            self.alter_fiat_shamir_state_with(&item);
        }
        self.items_index += 1;
        Ok(item)
    }

    /// Given an `upper_bound` that is a power of 2, produce `num_indices` uniform random numbers
    /// in the interval `[0; upper_bound)`.
    ///
    /// - `upper_bound`: The (non-inclusive) upper bound. Must be a power of two.
    /// - `num_indices`: The number of indices to sample
    pub fn sample_indices(&mut self, upper_bound: usize, num_indices: usize) -> Vec<usize> {
        assert!(upper_bound.is_power_of_two());
        assert!(upper_bound <= BFieldElement::MAX as usize);
        self.sponge
            .sample_indices(upper_bound as u32, num_indices)
            .into_iter()
            .map(|i| i as usize)
            .collect()
    }

    /// A thin wrapper around [`H::sample_scalars`](AlgebraicHasher::sample_scalars).
    pub fn sample_scalars(&mut self, num_scalars: usize) -> Vec<XFieldElement> {
        self.sponge.sample_scalars(num_scalars)
    }
}

impl TryFrom<&Proof> for Stream {
    type Error = ProofStreamError;

    fn try_from(proof: &Proof) -> Result<Self, ProofStreamError> {
        if proof.version != CURRENT_PROOF_VERSION {
            return Err(ProofStreamError::UnknownProofVersion(proof.version));
        }

        let proof_stream = *Stream::decode(&proof.encoded_proof_stream)?;
        Ok(proof_stream)
    }
}

impl From<&Stream> for Proof {
    fn from(proof_stream: &Stream) -> Self {
        Proof {
            version: CURRENT_PROOF_VERSION,
            encoded_proof_stream: proof_stream.encode(),
        }
    }
}

impl From<Stream> for Proof {
    fn from(proof_stream: Stream) -> Self {
        (&proof_stream).into()
    }
}

/// A `FriResponse` is an `AuthenticationStructure` together with the values of the
/// revealed leaves of the Merkle tree. Together, they correspond to the
/// queried indices of the FRI codeword (of that round).
#[derive(Debug, Clone, Eq, PartialEq, Hash, BFieldCodec, Arbitrary)]
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
            Display,
            Clone,
            Eq,
            PartialEq,
            Hash,
            EnumCount,
            EnumDiscriminants,
            BFieldCodec,
            Arbitrary,
        )]
        #[strum_discriminants(name(ItemVariant))]
        // discriminants' default derives: Debug, Copy, Clone, Eq, PartialEq
        #[strum_discriminants(derive(Display, EnumIter, BFieldCodec, Arbitrary))]
        pub enum Item {
            $( $variant($payload), )+
        }

        impl Item {
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
                if let Self::$variant(payload) = self {
                    Ok(payload)
                } else {
                    Err(ProofStreamError::UnexpectedItem {
                        expected: ItemVariant::$variant,
                        got: self,
                    })
                }
            }
            )+
        }

        impl ItemVariant {
            pub fn payload_static_length(self) -> Option<usize> {
                match self {
                    $( Self::$variant => <$payload>::static_length(), )+
                }
            }

            /// See [`Item::include_in_fiat_shamir_heuristic`].
            pub const fn include_in_fiat_shamir_heuristic(self) -> bool {
                match self {
                    $( Self::$variant => $in_fiat_shamir_heuristic, )+
                }
            }

            /// Can be used as “reflection”, for example through `syn`.
            pub const fn payload_type(self) -> &'static str {
                match self {
                    $( Self::$variant => stringify!($payload), )+
                }
            }
        }
    };
}

proof_items!(
    MerkleRoot(Digest) => true, try_into_merkle_root,
    OutOfDomainBaseRow(Box<BaseRow<XFieldElement>>) => true, try_into_out_of_domain_base_row,
    OutOfDomainExtRow(Box<ExtensionRow>) => true, try_into_out_of_domain_ext_row,
    OutOfDomainQuotientSegments(QuotientSegments) => true, try_into_out_of_domain_quot_segments,

    // the following are implied by some Merkle root, thus not included in the Fiat-Shamir heuristic
    AuthenticationStructure(AuthenticationStructure) => false, try_into_authentication_structure,
    MasterBaseTableRows(Vec<BaseRow<BFieldElement>>) => false, try_into_master_base_table_rows,
    MasterExtTableRows(Vec<ExtensionRow>) => false, try_into_master_ext_table_rows,
    Log2PaddedHeight(u32) => false, try_into_log2_padded_height,
    QuotientSegmentsElements(Vec<QuotientSegments>) => false, try_into_quot_segments_elements,
    FriCodeword(Vec<XFieldElement>) => false, try_into_fri_codeword,
    FriPolynomial(Polynomial<XFieldElement>) => false, try_into_fri_polynomial,
    FriResponse(FriResponse) => false, try_into_fri_response,
);

#[cfg(test)]
mod tests {
    use std::collections::HashSet;
    use std::collections::VecDeque;

    use assert2::assert;
    use assert2::let_assert;
    use itertools::Itertools;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use proptest_arbitrary_interop::arb;
    use strum::IntoEnumIterator;
    use test_strategy::proptest;
    use twenty_first::math::other::random_elements;

    use super::*;
    use crate::error::ProofStreamError::UnexpectedItem;
    use crate::shared_tests::LeavedMerkleTreeTestData;

    impl Default for Claim {
        /// For testing purposes only.
        fn default() -> Self {
            Self::new(Digest::default())
        }
    }

    /// To allow backwards compatibility, the encoding of the [`Proof`]'s
    /// `version` field must not change.
    #[test]
    fn version_has_static_length_of_one() {
        assert!(Some(1) == Version::static_length());
    }

    #[test]
    fn version_is_formatted_correctly() {
        assert!("012_345_678" == format!("{}", Version(12_345_678)));
        assert!("123_004_005" == format!("{}", Version(123_004_005)));
    }

    #[proptest]
    fn decode_proof(#[strategy(arb())] proof: Proof) {
        let encoded = proof.encode();
        let decoded = *Proof::decode(&encoded).unwrap();
        prop_assert_eq!(proof, decoded);
    }

    #[proptest]
    fn decode_claim(#[strategy(arb())] claim: Claim) {
        let encoded = claim.encode();
        let decoded = *Claim::decode(&encoded).unwrap();
        prop_assert_eq!(claim, decoded);
    }

    #[proptest(cases = 10)]
    fn proof_with_no_padded_height_gives_err(#[strategy(arb())] root: Digest) {
        let mut proof_stream = Stream::new();
        proof_stream.enqueue(Item::MerkleRoot(root));
        let proof: Proof = proof_stream.into();
        let maybe_padded_height = proof.padded_height();
        assert!(maybe_padded_height.is_err());
    }

    #[proptest(cases = 10)]
    fn proof_with_multiple_padded_height_gives_err(#[strategy(arb())] root: Digest) {
        let mut proof_stream = Stream::new();
        proof_stream.enqueue(Item::Log2PaddedHeight(8));
        proof_stream.enqueue(Item::MerkleRoot(root));
        proof_stream.enqueue(Item::Log2PaddedHeight(7));
        let proof: Proof = proof_stream.into();
        let maybe_padded_height = proof.padded_height();
        assert!(maybe_padded_height.is_err());
    }

    #[proptest]
    fn decoding_arbitrary_proof_data_does_not_panic(
        #[strategy(vec(arb(), 0..1_000))] proof_data: Vec<BFieldElement>,
    ) {
        let _proof = Proof::decode(&proof_data);
    }

    #[proptest]
    fn serialize_proof_with_fiat_shamir(
        #[strategy(vec(arb(), 2..100))] base_rows: Vec<BaseRow<BFieldElement>>,
        #[strategy(vec(arb(), 2..100))] ext_rows: Vec<ExtensionRow>,
        #[strategy(arb())] ood_base_row: Box<BaseRow<XFieldElement>>,
        #[strategy(arb())] ood_ext_row: Box<ExtensionRow>,
        #[strategy(arb())] quot_elements: Vec<QuotientSegments>,
        leaved_merkle_tree: LeavedMerkleTreeTestData,
    ) {
        let auth_structure = leaved_merkle_tree.auth_structure.clone();
        let root = leaved_merkle_tree.root();
        let fri_codeword = leaved_merkle_tree.leaves().to_owned();
        let fri_response = leaved_merkle_tree.into_fri_response();

        let mut sponge_states = VecDeque::new();
        let mut proof_stream = Stream::new();

        sponge_states.push_back(proof_stream.sponge.state);
        proof_stream.enqueue(Item::AuthenticationStructure(auth_structure.clone()));
        sponge_states.push_back(proof_stream.sponge.state);
        proof_stream.enqueue(Item::MasterBaseTableRows(base_rows.clone()));
        sponge_states.push_back(proof_stream.sponge.state);
        proof_stream.enqueue(Item::MasterExtTableRows(ext_rows.clone()));
        sponge_states.push_back(proof_stream.sponge.state);
        proof_stream.enqueue(Item::OutOfDomainBaseRow(ood_base_row.clone()));
        sponge_states.push_back(proof_stream.sponge.state);
        proof_stream.enqueue(Item::OutOfDomainExtRow(ood_ext_row.clone()));
        sponge_states.push_back(proof_stream.sponge.state);
        proof_stream.enqueue(Item::MerkleRoot(root));
        sponge_states.push_back(proof_stream.sponge.state);
        proof_stream.enqueue(Item::QuotientSegmentsElements(quot_elements.clone()));
        sponge_states.push_back(proof_stream.sponge.state);
        proof_stream.enqueue(Item::FriCodeword(fri_codeword.clone()));
        sponge_states.push_back(proof_stream.sponge.state);
        proof_stream.enqueue(Item::FriResponse(fri_response.clone()));
        sponge_states.push_back(proof_stream.sponge.state);

        let proof = proof_stream.into();
        let mut proof_stream = Stream::try_from(&proof).unwrap();

        assert!(sponge_states.pop_front() == Some(proof_stream.sponge.state));
        let_assert!(Ok(proof_item) = proof_stream.dequeue());
        let_assert!(Item::AuthenticationStructure(auth_structure_) = proof_item);
        assert!(auth_structure == auth_structure_);

        assert!(sponge_states.pop_front() == Some(proof_stream.sponge.state));
        let_assert!(Ok(Item::MasterBaseTableRows(base_rows_)) = proof_stream.dequeue());
        assert!(base_rows == base_rows_);

        assert!(sponge_states.pop_front() == Some(proof_stream.sponge.state));
        let_assert!(Ok(Item::MasterExtTableRows(ext_rows_)) = proof_stream.dequeue());
        assert!(ext_rows == ext_rows_);

        assert!(sponge_states.pop_front() == Some(proof_stream.sponge.state));
        let_assert!(Ok(Item::OutOfDomainBaseRow(ood_base_row_)) = proof_stream.dequeue());
        assert!(ood_base_row == ood_base_row_);

        assert!(sponge_states.pop_front() == Some(proof_stream.sponge.state));
        let_assert!(Ok(Item::OutOfDomainExtRow(ood_ext_row_)) = proof_stream.dequeue());
        assert!(ood_ext_row == ood_ext_row_);

        assert!(sponge_states.pop_front() == Some(proof_stream.sponge.state));
        let_assert!(Ok(Item::MerkleRoot(root_)) = proof_stream.dequeue());
        assert!(root == root_);

        assert!(sponge_states.pop_front() == Some(proof_stream.sponge.state));
        let_assert!(Ok(proof_item) = proof_stream.dequeue());
        let_assert!(Item::QuotientSegmentsElements(quot_elements_) = proof_item);
        assert!(quot_elements == quot_elements_);

        assert!(sponge_states.pop_front() == Some(proof_stream.sponge.state));
        let_assert!(Ok(Item::FriCodeword(fri_codeword_)) = proof_stream.dequeue());
        assert!(fri_codeword == fri_codeword_);

        assert!(sponge_states.pop_front() == Some(proof_stream.sponge.state));
        let_assert!(Ok(Item::FriResponse(fri_response_)) = proof_stream.dequeue());
        assert!(fri_response == fri_response_);

        assert!(sponge_states.pop_front() == Some(proof_stream.sponge.state));
        assert!(0 == sponge_states.len());
    }

    #[test]
    fn enqueue_dequeue_verify_partial_authentication_structure() {
        let tree_height = 8;
        let num_leaves = 1 << tree_height;
        let leaf_values: Vec<XFieldElement> = random_elements(num_leaves);
        let leaf_digests = leaf_values.iter().map(|&xfe| xfe.into()).collect_vec();
        let merkle_tree = MerkleTree::new::<CpuParallel>(&leaf_digests).unwrap();
        let indices_to_check = vec![5, 173, 175, 167, 228, 140, 252, 149, 232, 182, 5, 5, 182];
        let auth_structure = merkle_tree
            .authentication_structure(&indices_to_check)
            .unwrap();
        let revealed_leaves = indices_to_check
            .iter()
            .map(|&idx| leaf_values[idx])
            .collect_vec();
        let fri_response = FriResponse {
            auth_structure,
            revealed_leaves,
        };

        let mut proof_stream = Stream::new();
        proof_stream.enqueue(Item::FriResponse(fri_response));

        // TODO: Also check that deserializing from Proof works here.

        let proof_item = proof_stream.dequeue().unwrap();
        let maybe_same_fri_response = proof_item.try_into_fri_response().unwrap();
        let FriResponse {
            auth_structure,
            revealed_leaves,
        } = maybe_same_fri_response;
        let maybe_same_leaf_digests = revealed_leaves.iter().map(|&xfe| xfe.into()).collect_vec();
        let indexed_leafs = indices_to_check
            .into_iter()
            .zip_eq(maybe_same_leaf_digests)
            .collect();

        let inclusion_proof = MerkleTreeInclusionProof {
            tree_height,
            indexed_leafs,
            authentication_structure: auth_structure,
        };
        assert!(inclusion_proof.verify(merkle_tree.root()));
    }

    #[test]
    fn dequeuing_from_empty_stream_fails() {
        let mut proof_stream = Stream::new();
        let_assert!(Err(ProofStreamError::EmptyQueue) = proof_stream.dequeue());
    }

    #[test]
    fn dequeuing_more_items_than_have_been_enqueued_fails() {
        let mut proof_stream = Stream::new();
        proof_stream.enqueue(Item::FriCodeword(vec![]));
        proof_stream.enqueue(Item::Log2PaddedHeight(7));

        let_assert!(Ok(_) = proof_stream.dequeue());
        let_assert!(Ok(_) = proof_stream.dequeue());
        let_assert!(Err(ProofStreamError::EmptyQueue) = proof_stream.dequeue());
    }

    #[test]
    fn encoded_length_of_prove_stream_is_not_known_at_compile_time() {
        assert!(Stream::static_length().is_none());
    }

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
        let mut proof_stream = Stream::new();
        proof_stream.enqueue(Item::FriResponse(fri_response.clone()));
        let proof: Proof = proof_stream.into();

        let_assert!(Ok(mut proof_stream) = Stream::try_from(&proof));
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
        let mut proof_stream = Stream::new();
        proof_stream.enqueue(Item::AuthenticationStructure(auth_structure.clone()));
        let proof: Proof = proof_stream.into();

        let_assert!(Ok(mut proof_stream) = Stream::try_from(&proof));
        let_assert!(Ok(proof_item) = proof_stream.dequeue());
        let_assert!(Ok(auth_structure_) = proof_item.try_into_authentication_structure());
        prop_assert_eq!(auth_structure, auth_structure_);
    }

    #[test]
    fn interpreting_a_merkle_root_as_anything_else_gives_appropriate_error() {
        let fake_root = Digest::default();
        let item = Item::MerkleRoot(fake_root);
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
        assert!(let Err(UnexpectedItem{..}) = item.clone().try_into_fri_polynomial());
        assert!(let Err(UnexpectedItem{..}) = item.try_into_fri_response());
    }

    #[test]
    fn proof_item_payload_static_length_is_as_expected() {
        assert!(let Some(_) =  ItemVariant::MerkleRoot.payload_static_length());
        assert!(let Some(_) =  ItemVariant::Log2PaddedHeight.payload_static_length());
        assert_eq!(None, ItemVariant::FriCodeword.payload_static_length());
        assert_eq!(None, ItemVariant::FriResponse.payload_static_length());
    }

    #[test]
    fn can_loop_over_proof_item_variants() {
        let all_discriminants: HashSet<_> = ItemVariant::iter()
            .map(|variant| variant.bfield_codec_discriminant())
            .collect();
        assert_eq!(Item::COUNT, all_discriminants.len());
    }

    #[test]
    fn proof_item_and_its_variant_have_same_bfield_codec_discriminant() {
        assert_eq!(
            Item::MerkleRoot(Digest::default()).bfield_codec_discriminant(),
            ItemVariant::MerkleRoot.bfield_codec_discriminant()
        );
        assert_eq!(
            Item::Log2PaddedHeight(0).bfield_codec_discriminant(),
            ItemVariant::Log2PaddedHeight.bfield_codec_discriminant()
        );
        assert_eq!(
            Item::FriCodeword(vec![]).bfield_codec_discriminant(),
            ItemVariant::FriCodeword.bfield_codec_discriminant()
        );
    }

    #[test]
    fn proof_item_variants_payload_type_has_expected_format() {
        assert_eq!("Digest", ItemVariant::MerkleRoot.payload_type());
    }
}
