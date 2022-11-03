use std::error::Error;

use itertools::Itertools;

use num_traits::{One, Zero};

use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::rescue_prime_digest::Digest;
use twenty_first::shared_math::rescue_prime_regular::DIGEST_LENGTH;
use twenty_first::shared_math::x_field_element::XFieldElement;
use twenty_first::util_types::algebraic_hasher::Hashable;
use twenty_first::util_types::merkle_tree::PartialAuthenticationPath;
use twenty_first::util_types::proof_stream_typed::ProofStreamError;

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

/// BFieldCodec
///
/// This trait provides functions for encoding to and decoding from a
/// Vec of BFieldElements. This encoding records the length of
/// variable-size structures, whether implicitly or explicitly via
/// length-prepending. It does not record type informatin; this is
/// the responsibility of the decoder.
pub trait BFieldCodec {
    fn decode(str: &[BFieldElement]) -> Result<Box<Self>, Box<dyn Error>>;
    fn encode(&self) -> Vec<BFieldElement>;
}

impl BFieldCodec for ProofItem {
    /// Turn the given string of BFieldElements into a ProofItem.
    /// Ignore the first element, because it denotes the length of
    /// the encoding.
    fn decode(str: &[BFieldElement]) -> Result<Box<Self>, Box<dyn std::error::Error>> {
        Ok(Box::new(Self::Uncast(str[1..].to_vec())))
    }

    /// Encode the ProofItem as a string of BFieldElements, with the
    /// first element denoting the length of the rest.
    fn encode(&self) -> Vec<BFieldElement> {
        let mut tail = self.clone().into_iter().collect_vec();
        let head = BFieldElement::new(tail.len().try_into().unwrap());
        tail.insert(0, head);
        tail
    }
}

impl BFieldCodec for BFieldElement {
    fn decode(str: &[BFieldElement]) -> Result<Box<Self>, Box<dyn Error>> {
        let maybe_element_zero = str.get(0);
        match maybe_element_zero {
            Some(element) => Ok(Box::new(*element)),
            None => Err(ProofStreamError::boxed(
                "trying to decode empty slice into BFieldElement",
            )),
        }
    }

    fn encode(&self) -> Vec<BFieldElement> {
        [*self].to_vec()
    }
}

impl BFieldCodec for XFieldElement {
    fn decode(str: &[BFieldElement]) -> Result<Box<Self>, Box<dyn Error>> {
        if str.len() != 3 {
            Err(ProofStreamError::boxed(
                "trying to decode slice of not 3 BFieldElements into XFieldElement",
            ))
        } else {
            Ok(Box::new(XFieldElement {
                coefficients: str.try_into().unwrap(),
            }))
        }
    }

    fn encode(&self) -> Vec<BFieldElement> {
        self.coefficients.to_vec()
    }
}

impl BFieldCodec for Digest {
    fn decode(str: &[BFieldElement]) -> Result<Box<Self>, Box<dyn Error>> {
        if str.len() != DIGEST_LENGTH {
            Err(ProofStreamError::boxed(
                "trying to decode slice of not DIGEST_LENGTH BFieldElements into Digest",
            ))
        } else {
            Ok(Box::new(Digest::new(str.try_into().unwrap())))
        }
    }

    fn encode(&self) -> Vec<BFieldElement> {
        self.to_sequence()
    }
}

impl<T: BFieldCodec> BFieldCodec for Vec<T> {
    fn decode(str: &[BFieldElement]) -> Result<Box<Self>, Box<dyn Error>> {
        let mut vect: Vec<T> = vec![];
        let mut index = 0;
        while index < str.len() {
            // we don't know ahead of time how wide each T-element is
            // so every element in the list is length-prepended
            let len = str[index].value();
            let substr = &str[(index + 1)..(index + 1 + len as usize)];
            let decoded = T::decode(substr);
            if let Ok(t) = decoded {
                vect.push(*t);
            } else {
                return Err(ProofStreamError::boxed("cannot decode T element in vec"));
            }

            index += 1 + len as usize;
        }
        Ok(Box::new(vect))
    }

    fn encode(&self) -> Vec<BFieldElement> {
        let mut str = vec![];
        for elem in self.iter() {
            let mut substr = elem.encode();
            str.push(BFieldElement::new(substr.len().try_into().expect("Generic parameter encoding length (as BFieldElements) does not fit into BFieldElement")));
            str.append(&mut substr);
        }
        str
    }
}

impl<T: BFieldCodec, S: BFieldCodec> BFieldCodec for (T, S) {
    fn decode(str: &[BFieldElement]) -> Result<Box<Self>, Box<dyn Error>> {
        // decode T
        let maybe_element_zero = str.get(0);
        if matches!(maybe_element_zero, None) {
            return Err(ProofStreamError::boxed(
                "trying to decode empty slice as tuple",
            ));
        }
        let len_t = maybe_element_zero.unwrap().value() as usize;
        if str.len() < 1 + len_t {
            return Err(ProofStreamError::boxed(
                "prepended length of tuple element does not match with remaining string length",
            ));
        }
        let maybe_t = T::decode(&str[1..(1 + len_t)]);

        // decode S
        let maybe_next_element = str.get(1 + len_t);
        if matches!(maybe_next_element, None) {
            return Err(ProofStreamError::boxed(
                "trying to decode singleton as tuple",
            ));
        }
        let len_s = maybe_next_element.unwrap().value() as usize;
        if str.len() != 1 + len_t + 1 + len_s {
            return Err(ProofStreamError::boxed(
                "prepended length of second tuple element does not match with remaining string length",
            ));
        }
        let maybe_s = S::decode(&str[(1 + len_t + 1)..]);

        if let Ok(t) = maybe_t {
            if let Ok(s) = maybe_s {
                Ok(Box::new((*t, *s)))
            } else {
                Err(ProofStreamError::boxed("could not decode s"))
            }
        } else {
            Err(ProofStreamError::boxed("could not decode t"))
        }
    }

    fn encode(&self) -> Vec<BFieldElement> {
        let mut str = vec![];
        let mut encoding_of_t = self.0.encode();
        let mut encoding_of_s = self.1.encode();
        str.push(BFieldElement::new(encoding_of_t.len().try_into().expect(
            "encoding of t has length that does not fit in BFieldElement",
        )));
        str.append(&mut encoding_of_t);
        str.push(BFieldElement::new(encoding_of_s.len().try_into().expect(
            "encoding of s has length that does not fit in BFieldElement",
        )));
        str.append(&mut encoding_of_s);
        str
    }
}

impl BFieldCodec for PartialAuthenticationPath<Digest> {
    fn decode(str: &[BFieldElement]) -> Result<Box<Self>, Box<dyn Error>> {
        let mut vect: Vec<Option<Digest>> = vec![];
        let mut index = 0;
        while index < str.len() {
            // we don't know ahead of time how wide each T-element is
            // so every element in the list is length-prepended
            let len = str[index].value();
            let substr = &str[(index + 1)..(index + 1 + len as usize)];
            let decoded = Option::<Digest>::decode(substr);
            if let Ok(optional_digest) = decoded {
                vect.push(*optional_digest);
            } else {
                return Err(ProofStreamError::boxed(
                    "cannot decode optional digest in vec",
                ));
            }

            index += 1 + len as usize;
        }
        Ok(Box::new(PartialAuthenticationPath::<Digest>(vect)))
    }

    fn encode(&self) -> Vec<BFieldElement> {
        let mut vect = vec![];
        for optional_authpath in self.0.iter() {
            let mut encoded = optional_authpath.encode();
            vect.push(BFieldElement::new(encoded.len().try_into().expect(
                "encoded optional authpath has length greater than what fits into BFieldElement",
            )));
            vect.append(&mut encoded);
        }
        vect
    }
}
// impl BFieldCodec for AuthenticationStructure<Digest> {}

impl<T: BFieldCodec> BFieldCodec for Option<T> {
    fn decode(str: &[BFieldElement]) -> Result<Box<Self>, Box<dyn Error>> {
        let maybe_element_zero = str.get(0);
        if matches!(maybe_element_zero, None) {
            return Err(ProofStreamError::boxed(
                "trying to decode empty slice into option of elements",
            ));
        }
        if maybe_element_zero.unwrap().is_zero() {
            Ok(Box::new(None))
        } else {
            let maybe_t = T::decode(&str[1..]);
            match maybe_t {
                Ok(t) => Ok(Box::new(Some(*t))),
                Err(e) => Err(e),
            }
        }
    }

    fn encode(&self) -> Vec<BFieldElement> {
        let mut str = vec![];
        match self {
            None => {
                str.push(BFieldElement::zero());
            }
            Some(t) => {
                str.push(BFieldElement::one());
                str.append(&mut t.encode());
            }
        }
        str
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
