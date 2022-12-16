use serde::Deserialize;
use serde::Serialize;
use twenty_first::shared_math::b_field_element::BFieldElement;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Proof(pub Vec<BFieldElement>);

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claim {
    pub input: Vec<BFieldElement>,
    pub program: Vec<BFieldElement>,
    pub output: Vec<BFieldElement>,
    pub padded_height: usize,
}
