use serde::Deserialize;
use serde::Serialize;
use twenty_first::shared_math::b_field_element::BFieldElement;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Proof(pub Vec<BFieldElement>);

impl Proof {
    pub fn padded_height(&self) -> usize {
        // FIXME: This is very brittle.
        self.0[1].value() as usize
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Claim {
    pub input: Vec<BFieldElement>,
    pub program: Vec<BFieldElement>,
    pub output: Vec<BFieldElement>,
    pub padded_height: usize,
}
