use serde::Deserialize;
use serde::Serialize;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::tip5::Digest;

/// Contains the necessary cryptographic information to verify a computation.
/// Should be used together with a [`Claim`].
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Proof(pub Vec<BFieldElement>);

/// Contains all the public information of a verifiably correct computation.
/// A corresponding [`Proof`] is needed to verify the computation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claim {
    /// The public input to the computation.
    pub input: Vec<u64>,

    /// The hash digest of the program that was executed. The hash function in use is Tip5.
    pub program_digest: Digest,

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
