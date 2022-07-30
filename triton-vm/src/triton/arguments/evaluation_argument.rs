use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::x_field_element::XFieldElement;

pub fn verify_evaluation_argument(
    symbols: &[BFieldElement],
    challenge: XFieldElement,
    expected_terminal: XFieldElement,
) -> bool {
    compute_terminal(symbols, XFieldElement::ring_zero(), challenge) == expected_terminal
}

/// Compute the running sum for an evaluation argument as specified by `initial`,
/// This amounts to evaluating polynomial `f(x) = initial·x^n + Σ_i symbols[n-i]·x^i` at position
/// challenge, i.e., returns `f(challenge)`.
pub fn compute_terminal(
    symbols: &[BFieldElement],
    initial: XFieldElement,
    challenge: XFieldElement,
) -> XFieldElement {
    let mut acc = initial;
    for s in symbols.iter() {
        acc = challenge * acc + s.lift();
    }
    acc
}
