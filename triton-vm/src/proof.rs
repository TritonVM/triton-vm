use anyhow::bail;
use get_size::GetSize;
use itertools::Itertools;
use serde::Deserialize;
use serde::Serialize;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::bfield_codec::BFieldCodec;
use twenty_first::shared_math::tip5::Digest;
use twenty_first::shared_math::tip5::DIGEST_LENGTH;

/// Contains the necessary cryptographic information to verify a computation.
/// Should be used together with a [`Claim`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
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

impl BFieldCodec for Proof {
    fn decode(sequence: &[BFieldElement]) -> anyhow::Result<Box<Self>> {
        Ok(Box::new(Proof(sequence.to_vec())))
    }

    fn encode(&self) -> Vec<BFieldElement> {
        self.0.clone()
    }
}

/// Contains all the public information of a verifiably correct computation.
/// A corresponding [`Proof`] is needed to verify the computation.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq, GetSize)]
pub struct Claim {
    /// The hash digest of the program that was executed. The hash function in use is Tip5.
    pub program_digest: Digest,

    /// The public input to the computation.
    pub input: Vec<u64>,

    /// The public output of the computation.
    pub output: Vec<u64>,

    /// An upper bound on the length of the computation.
    pub padded_height: usize,
}

impl Claim {
    /// The public input as `BFieldElements`.
    /// If u64s are needed, use field `input`.
    pub fn public_input(&self) -> Vec<BFieldElement> {
        self.input.iter().map(|&x| BFieldElement::new(x)).collect()
    }

    /// The public output as `BFieldElements`.
    /// If u64s are needed, use field `output`.
    pub fn public_output(&self) -> Vec<BFieldElement> {
        self.output.iter().map(|&x| BFieldElement::new(x)).collect()
    }
}

impl BFieldCodec for Claim {
    fn decode(sequence: &[BFieldElement]) -> anyhow::Result<Box<Self>> {
        if sequence.len() < 7 {
            bail!(
                "Cannot decode Vec of {} < 7 BFieldElements as Claim",
                sequence.len()
            );
        }
        let program_digest = *Digest::decode(&sequence[0..DIGEST_LENGTH])?;
        let mut read_index = DIGEST_LENGTH;

        let input_length = sequence[read_index].value() as usize;
        if sequence.len() < read_index + input_length + 1 {
            bail!("Cannot decode Vec of BFieldElements as Claim: improper input length");
        }
        read_index += 1;
        let input = sequence[read_index..read_index + input_length]
            .to_vec()
            .iter()
            .map(|b| b.value())
            .collect_vec();
        read_index += input_length;

        let output_length = sequence[read_index].value() as usize;
        if sequence.len() < read_index + output_length {
            bail!("Cannot decode Vec of BFieldElements as Claim: improper output length");
        }
        read_index += 1;
        let output = sequence[read_index..read_index + output_length]
            .to_vec()
            .iter()
            .map(|b| b.value())
            .collect_vec();

        Ok(Box::new(Claim {
            program_digest,
            input,
            output,
            padded_height: 0,
        }))
    }

    fn encode(&self) -> Vec<BFieldElement> {
        let mut string = self.program_digest.encode();
        string.push(BFieldElement::new(self.input.len() as u64));
        string.append(&mut self.public_input());
        string.push(BFieldElement::new(self.output.len() as u64));
        string.append(&mut self.public_output());
        string
    }
}

#[cfg(test)]
pub mod test_claim_proof {
    use itertools::Itertools;
    use rand::random;
    use twenty_first::shared_math::{
        b_field_element::BFieldElement, bfield_codec::BFieldCodec, other::random_elements,
        tip5::Digest,
    };

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
        let input: Vec<u64> = random_elements(346)
            .iter()
            .map(|i: &u64| i % BFieldElement::P)
            .collect_vec();
        let output: Vec<u64> = random_elements(125)
            .iter()
            .map(|i: &u64| *i % BFieldElement::P)
            .collect_vec();
        let padded_height = 11;

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
