use std::{error::Error, fmt::Display};

use itertools::Itertools;
use num_traits::{One, Zero};
use twenty_first::{
    shared_math::{
        b_field_element::BFieldElement, rescue_prime_digest::Digest,
        rescue_prime_regular::DIGEST_LENGTH, x_field_element::XFieldElement,
    },
    util_types::{algebraic_hasher::Hashable, merkle_tree::PartialAuthenticationPath},
};

#[derive(Debug, Clone)]
pub struct BFieldCodecError {
    pub message: String,
}

impl BFieldCodecError {
    pub fn new(message: &str) -> Self {
        Self {
            message: message.to_string(),
        }
    }

    pub fn boxed(message: &str) -> Box<dyn Error> {
        Box::new(Self::new(message))
    }
}

impl Display for BFieldCodecError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        writeln!(f, "{}", self.message)
    }
}

impl Error for BFieldCodecError {}

/// BFieldCodec
///
/// This trait provides functions for encoding to and decoding from a
/// Vec of BFieldElements. This encoding records the length of
/// variable-size structures, whether implicitly or explicitly via
/// length-prepending. It does not record type information; this is
/// the responsibility of the decoder.
pub trait BFieldCodec {
    fn decode(string: &[BFieldElement]) -> Result<Box<Self>, Box<dyn Error>>;
    fn encode(&self) -> Vec<BFieldElement>;
}

impl BFieldCodec for BFieldElement {
    fn decode(str: &[BFieldElement]) -> Result<Box<Self>, Box<dyn Error>> {
        if str.len() != 1 {
            return Err(BFieldCodecError::boxed(
                "trying to decode more or less than one BFieldElements as one BFieldElement",
            ));
        }
        let element_zero = str[0];
        Ok(Box::new(element_zero))
    }

    fn encode(&self) -> Vec<BFieldElement> {
        [*self].to_vec()
    }
}

impl BFieldCodec for XFieldElement {
    fn decode(str: &[BFieldElement]) -> Result<Box<Self>, Box<dyn Error>> {
        if str.len() != 3 {
            Err(BFieldCodecError::boxed(
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
            Err(BFieldCodecError::boxed(
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

impl<T: BFieldCodec, S: BFieldCodec> BFieldCodec for (T, S) {
    fn decode(str: &[BFieldElement]) -> Result<Box<Self>, Box<dyn Error>> {
        // decode T
        let maybe_element_zero = str.get(0);
        if matches!(maybe_element_zero, None) {
            return Err(BFieldCodecError::boxed(
                "trying to decode empty slice as tuple",
            ));
        }
        let len_t = maybe_element_zero.unwrap().value() as usize;
        if str.len() < 1 + len_t {
            return Err(BFieldCodecError::boxed(
                "prepended length of tuple element does not match with remaining string length",
            ));
        }
        let maybe_t = T::decode(&str[1..(1 + len_t)]);

        // decode S
        let maybe_next_element = str.get(1 + len_t);
        if matches!(maybe_next_element, None) {
            return Err(BFieldCodecError::boxed(
                "trying to decode singleton as tuple",
            ));
        }
        let len_s = maybe_next_element.unwrap().value() as usize;
        if str.len() != 1 + len_t + 1 + len_s {
            return Err(BFieldCodecError::boxed(
                "prepended length of second tuple element does not match with remaining string length",
            ));
        }
        let maybe_s = S::decode(&str[(1 + len_t + 1)..]);

        if let Ok(t) = maybe_t {
            if let Ok(s) = maybe_s {
                Ok(Box::new((*t, *s)))
            } else {
                Err(BFieldCodecError::boxed("could not decode s"))
            }
        } else {
            Err(BFieldCodecError::boxed("could not decode t"))
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
        if str.is_empty() {
            return Err(BFieldCodecError::boxed(
                "cannot decode empty string into PartialAuthenticationPath",
            ));
        }
        let mut vect: Vec<Option<Digest>> = vec![];
        let mut index = 0;
        while index < str.len() {
            let len = str[index].value();
            if str.len() < index + 1 + len as usize {
                return Err(BFieldCodecError::boxed(
                    "cannot decode vec of optional digests because of improper length prepending",
                ));
            }
            let substr = &str[(index + 1)..(index + 1 + len as usize)];
            let decoded = Option::<Digest>::decode(substr);
            if let Ok(optional_digest) = decoded {
                vect.push(*optional_digest);
            } else {
                return Err(BFieldCodecError::boxed(
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

impl<T: BFieldCodec> BFieldCodec for Option<T> {
    fn decode(str: &[BFieldElement]) -> Result<Box<Self>, Box<dyn Error>> {
        let maybe_element_zero = str.get(0);
        if matches!(maybe_element_zero, None) {
            return Err(BFieldCodecError::boxed(
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

impl BFieldCodec for Vec<BFieldElement> {
    fn decode(str: &[BFieldElement]) -> Result<Box<Self>, Box<dyn Error>> {
        Ok(Box::new(str.to_vec()))
    }

    fn encode(&self) -> Vec<BFieldElement> {
        self.to_vec()
    }
}

impl BFieldCodec for Vec<XFieldElement> {
    fn decode(str: &[BFieldElement]) -> Result<Box<Self>, Box<dyn Error>> {
        if str.len() % 3 != 0 {
            return Err(BFieldCodecError::boxed("cannot decode string of BFieldElements into XFieldElements when string length is not a multiple of 3"));
        }
        let mut vector = vec![];
        for chunk in str.chunks(3) {
            let coefficients: [BFieldElement; 3] = chunk.try_into().unwrap();
            vector.push(XFieldElement::new(coefficients));
        }
        Ok(Box::new(vector))
    }

    fn encode(&self) -> Vec<BFieldElement> {
        self.iter().map(|xfe| xfe.coefficients.to_vec()).concat()
    }
}

impl BFieldCodec for Vec<Digest> {
    fn decode(str: &[BFieldElement]) -> Result<Box<Self>, Box<dyn Error>> {
        if str.len() % DIGEST_LENGTH != 0 {
            return Err(BFieldCodecError::boxed("cannot decode string of BFieldElements into Digests when string length is not a multiple of DIGEST_LENGTH"));
        }
        let mut vector: Vec<Digest> = vec![];
        for chunk in str.chunks(DIGEST_LENGTH) {
            let digest: [BFieldElement; DIGEST_LENGTH] = chunk.try_into().unwrap();
            vector.push(Digest::new(digest));
        }
        Ok(Box::new(vector))
    }

    fn encode(&self) -> Vec<BFieldElement> {
        self.iter().map(|d| d.encode()).concat()
    }
}

impl<T> BFieldCodec for Vec<Vec<T>>
where
    Vec<T>: BFieldCodec,
{
    fn decode(str: &[BFieldElement]) -> Result<Box<Self>, Box<dyn Error>> {
        let mut index = 0;
        let mut outer_vec: Vec<Vec<T>> = vec![];
        while index < str.len() {
            let len = str[index].value() as usize;
            index += 1;

            if let Some(inner_vec) = str.get(index..(index + len)) {
                outer_vec.push(*Vec::<T>::decode(inner_vec)?);
            } else {
                return Err(BFieldCodecError::boxed(
                    "cannot decode string BFieldElements into Vec<Vec<T>>; length mismatch",
                ));
            }
            index += len;
        }
        Ok(Box::new(outer_vec))
    }

    fn encode(&self) -> Vec<BFieldElement> {
        let mut str = vec![];
        for inner_vec in self {
            let mut encoding = inner_vec.encode();
            str.push(BFieldElement::new(encoding.len().try_into().unwrap()));
            str.append(&mut encoding);
        }
        str
    }
}

impl BFieldCodec for Vec<PartialAuthenticationPath<Digest>> {
    fn decode(str: &[BFieldElement]) -> Result<Box<Self>, Box<dyn Error>> {
        let mut index = 0;
        let mut vector = vec![];
        while index < str.len() {
            let len_remaining = str[index].value() as usize;
            index += 1;

            if len_remaining < 2 || index + len_remaining > str.len() {
                return Err(BFieldCodecError::boxed("cannot decode string of BFieldElements as Vec of PartialAuthenticationPaths due to length mismatch (1)"));
            }

            let vec_len = str[index].value() as usize;
            let mask = str[index + 1].value() as u32;
            index += 2;

            if (vec_len != 0 || mask != 0) && index == str.len() {
                return Err(BFieldCodecError::boxed("cannot decode string of BFieldElements as Vec of PartialAuthenticationPaths due to length mismatch (2)"));
            }

            if (len_remaining - 2) % DIGEST_LENGTH != 0 {
                return Err(BFieldCodecError::boxed("cannot decode string of BFieldElements as Vec of PartialAuthenticationPaths due to length mismatch (3)"));
            }

            let mut pap = vec![];

            for i in (0..vec_len).rev() {
                if mask & (1 << i) == 0 {
                    pap.push(None);
                } else if let Some(chunk) = str.get(index..(index + DIGEST_LENGTH)) {
                    pap.push(Some(*Digest::decode(chunk)?));
                    index += DIGEST_LENGTH;
                } else {
                    return Err(BFieldCodecError::boxed("cannot decode string of BFieldElements as Vec of PartialAuthenticationPaths due to length mismatch (3)"));
                }
            }

            vector.push(PartialAuthenticationPath(pap));
        }

        Ok(Box::new(vector))
    }

    fn encode(&self) -> Vec<BFieldElement> {
        let mut str = vec![];
        for pap in self.iter() {
            let len = pap.0.len();
            let mut mask = 0u32;
            for maybe_digest in pap.0.iter() {
                mask <<= 1;
                if maybe_digest.is_some() {
                    mask |= 1;
                }
            }
            let mut vector = pap.0.iter().flatten().map(|d| d.to_sequence()).concat();

            str.push(BFieldElement::new(
                2u64 + std::convert::TryInto::<u64>::try_into(vector.len()).unwrap(),
            ));
            str.push(BFieldElement::new(len.try_into().unwrap()));
            str.push(BFieldElement::new(mask.try_into().unwrap()));
            str.append(&mut vector);
        }
        str
    }
}

#[cfg(test)]
mod bfield_codec_tests {
    use itertools::Itertools;
    use rand::{thread_rng, RngCore};
    use twenty_first::shared_math::b_field_element::BFieldElement;

    use super::*;

    fn random_bool() -> bool {
        let mut rng = thread_rng();
        rng.next_u32() % 2 == 0
    }

    fn random_length(max: usize) -> usize {
        let mut rng = thread_rng();
        rng.next_u32() as usize % max
    }

    fn random_bfieldelement() -> BFieldElement {
        let mut rng = thread_rng();
        BFieldElement::new(rng.next_u64())
    }

    fn random_xfieldelement() -> XFieldElement {
        XFieldElement {
            coefficients: [
                random_bfieldelement(),
                random_bfieldelement(),
                random_bfieldelement(),
            ],
        }
    }

    fn random_digest() -> Digest {
        Digest::new([
            random_bfieldelement(),
            random_bfieldelement(),
            random_bfieldelement(),
            random_bfieldelement(),
            random_bfieldelement(),
        ])
    }

    fn random_partial_authentication_path(len: usize) -> PartialAuthenticationPath<Digest> {
        PartialAuthenticationPath(
            (0..len)
                .into_iter()
                .map(|_| {
                    if random_bool() {
                        Some(random_digest())
                    } else {
                        None
                    }
                })
                .collect_vec(),
        )
    }

    #[test]
    fn test_encode_decode_random_bfieldelement() {
        for _ in 1..=10 {
            let bfe = random_bfieldelement();
            let str = bfe.encode();
            let bfe_ = *BFieldElement::decode(&str).unwrap();
            assert_eq!(bfe, bfe_);
        }
    }

    #[test]
    fn test_encode_decode_random_xfieldelement() {
        for _ in 1..=10 {
            let xfe = random_xfieldelement();
            let str = xfe.encode();
            let xfe_ = *XFieldElement::decode(&str).unwrap();
            assert_eq!(xfe, xfe_);
        }
    }

    #[test]
    fn test_encode_decode_random_digest() {
        for _ in 1..=10 {
            let dig = random_digest();
            let str = dig.encode();
            let dig_ = *Digest::decode(&str).unwrap();
            assert_eq!(dig, dig_);
        }
    }

    #[test]
    fn test_encode_decode_random_vec_of_bfieldelement() {
        for _ in 1..=10 {
            let len = random_length(100);
            let bfe_vec = (0..len)
                .into_iter()
                .map(|_| random_bfieldelement())
                .collect_vec();
            let str = bfe_vec.encode();
            let bfe_vec_ = *Vec::<BFieldElement>::decode(&str).unwrap();
            assert_eq!(bfe_vec, bfe_vec_);
        }
    }

    #[test]
    fn test_encode_decode_random_vec_of_xfieldelement() {
        for _ in 1..=10 {
            let len = random_length(100);
            let xfe_vec = (0..len)
                .into_iter()
                .map(|_| random_xfieldelement())
                .collect_vec();
            let str = xfe_vec.encode();
            let xfe_vec_ = *Vec::<XFieldElement>::decode(&str).unwrap();
            assert_eq!(xfe_vec, xfe_vec_);
        }
    }

    #[test]
    fn test_encode_decode_random_vec_of_digest() {
        for _ in 1..=10 {
            let len = random_length(100);
            let digest_vec = (0..len).into_iter().map(|_| random_digest()).collect_vec();
            let str = digest_vec.encode();
            let digest_vec_ = *Vec::<Digest>::decode(&str).unwrap();
            assert_eq!(digest_vec, digest_vec_);
        }
    }

    #[test]
    fn test_encode_decode_random_vec_of_vec_of_bfieldelement() {
        for _ in 1..=10 {
            let len = random_length(10);
            let bfe_vec_vec = (0..len)
                .into_iter()
                .map(|_| {
                    let inner_len = random_length(20);
                    (0..inner_len)
                        .into_iter()
                        .map(|_| random_bfieldelement())
                        .collect_vec()
                })
                .collect_vec();
            let str = bfe_vec_vec.encode();
            let bfe_vec_vec_ = *Vec::<Vec<BFieldElement>>::decode(&str).unwrap();
            assert_eq!(bfe_vec_vec, bfe_vec_vec_);
        }
    }

    #[test]
    fn test_encode_decode_random_vec_of_vec_of_xfieldelement() {
        for _ in 1..=10 {
            let len = random_length(10);
            let xfe_vec_vec = (0..len)
                .into_iter()
                .map(|_| {
                    let inner_len = random_length(20);
                    (0..inner_len)
                        .into_iter()
                        .map(|_| random_xfieldelement())
                        .collect_vec()
                })
                .collect_vec();
            let str = xfe_vec_vec.encode();
            let xfe_vec_vec_ = *Vec::<Vec<XFieldElement>>::decode(&str).unwrap();
            assert_eq!(xfe_vec_vec, xfe_vec_vec_);
        }
    }

    #[test]
    fn test_encode_decode_random_partial_authentication_path() {
        for _ in 1..=10 {
            let len = 1 + random_length(10);
            let pap = random_partial_authentication_path(len);
            let str = pap.encode();
            let pap_ = *PartialAuthenticationPath::decode(&str).unwrap();
            assert_eq!(pap, pap_);
        }
    }

    #[test]
    fn test_decode_random_negative() {
        for _ in 1..=10000 {
            let len = random_length(100);
            let str = (0..len)
                .into_iter()
                .map(|_| random_bfieldelement())
                .collect_vec();

            // Some of the following cases can be triggered by false
            // positives. This should occur with probability roughly
            // 2^-60.

            if let Ok(_sth) = BFieldElement::decode(&str) {
                if str.len() != 1 {
                    panic!();
                }
            }

            if let Ok(_sth) = XFieldElement::decode(&str) {
                if str.len() != 3 {
                    panic!();
                }
            }

            if let Ok(_sth) = Digest::decode(&str) {
                if str.len() != DIGEST_LENGTH {
                    panic!();
                }
            }

            // if let Ok(sth) = Vec::<BFieldElement>::decode(&str) {
            //     (should work)
            // }

            if str.len() % 3 != 0 {
                if let Ok(sth) = Vec::<XFieldElement>::decode(&str) {
                    panic!("{:?}", sth);
                }
            }

            if str.len() % DIGEST_LENGTH != 0 {
                if let Ok(sth) = Vec::<Digest>::decode(&str) {
                    panic!("{:?}", sth);
                }
            }

            if let Ok(sth) = Vec::<Vec<BFieldElement>>::decode(&str) {
                if !sth.is_empty() {
                    panic!("{:?}", sth);
                }
            }

            if let Ok(sth) = Vec::<Vec<XFieldElement>>::decode(&str) {
                if !sth.is_empty() {
                    panic!("{:?}", sth);
                }
            }

            if let Ok(_sth) = PartialAuthenticationPath::decode(&str) {
                panic!();
            }
        }
    }
}
