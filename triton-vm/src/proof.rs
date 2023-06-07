use get_size::GetSize;
use serde::Deserialize;
use serde::Serialize;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::bfield_codec::BFieldCodec;
use twenty_first::shared_math::tip5::Digest;

/// Contains the necessary cryptographic information to verify a computation.
/// Should be used together with a [`Claim`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, BFieldCodec)]
pub struct Proof(pub Vec<BFieldElement>);

impl GetSize for Proof {
    fn get_stack_size() -> usize {
        std::mem::size_of::<Self>()
    }

    fn get_heap_size(&self) -> usize {
        self.0.len() * std::mem::size_of::<BFieldElement>()
    }

    fn get_size(&self) -> usize {
        Self::get_stack_size() + GetSize::get_heap_size(self)
    }
}

/// Contains all the public information of a verifiably correct computation.
/// A corresponding [`Proof`] is needed to verify the computation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, GetSize, BFieldCodec)]
pub struct Claim {
    /// The hash digest of the program that was executed. The hash function in use is Tip5.
    pub program_digest: Digest,

    /// The public input to the computation.
    pub input: Vec<BFieldElement>,

    /// The public output of the computation.
    pub output: Vec<BFieldElement>,

    /// An upper bound on the length of the computation.
    pub padded_height: BFieldElement,
}

impl Claim {
    /// The public input as `u64`s.
    /// If `BFieldElement`s are needed, use field `input`.
    pub fn public_input(&self) -> Vec<u64> {
        self.input.iter().map(|x| x.value()).collect()
    }

    /// The public output as `u64`.
    /// If `BFieldElements`s are needed, use field `output`.
    pub fn public_output(&self) -> Vec<u64> {
        self.output.iter().map(|x| x.value()).collect()
    }

    /// The padded height as `u64`.
    /// If a `BFieldElement` is needed, use field `padded_height`.
    pub fn padded_height(&self) -> usize {
        self.padded_height.value() as usize
    }
}

#[cfg(test)]
pub mod test_claim_proof {
    use rand::random;
    use twenty_first::shared_math::b_field_element::BFieldElement;
    use twenty_first::shared_math::bfield_codec::BFieldCodec;
    use twenty_first::shared_math::other::random_elements;
    use twenty_first::shared_math::tip5::Digest;

    use super::*;

    #[test]
    fn test_decode_proof() {
        let data: Vec<BFieldElement> = random_elements(348);
        let proof = Proof(data);

        let encoded = proof.encode();
        let decoded = *Proof::decode(&encoded).unwrap();

        assert_eq!(proof, decoded);
    }

    #[test]
    fn test_decode_claim() {
        let program_digest: Digest = random();
        let input: Vec<BFieldElement> = random_elements(346);
        let output: Vec<BFieldElement> = random_elements(125);
        let padded_height = 11_u64.into();

        let claim = Claim {
            program_digest,
            input,
            output,
            padded_height,
        };

        let encoded = claim.encode();
        let decoded = *Claim::decode(&encoded).unwrap();

        assert_eq!(claim.program_digest, decoded.program_digest);
        assert_eq!(claim.input, decoded.input);
        assert_eq!(claim.output, decoded.output);
        // padded height is ignored
    }
}
