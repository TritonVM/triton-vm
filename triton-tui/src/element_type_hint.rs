use std::cmp::Ordering;

use arbitrary::Arbitrary;
use itertools::Itertools;

/// A hint about the type of a single stack element. Helps debugging programs written for Triton VM.
/// **Does not enforce types.**
#[derive(Debug, Clone, PartialEq, Eq, Hash, Arbitrary)]
pub(crate) struct ElementTypeHint {
    /// The name of the type. See [`TypeHint`][type_hint] for details.
    ///
    /// [type_hint]: TypeHint
    pub type_name: Option<String>,

    /// The name of the variable. See [`TypeHint`][type_hint] for details.
    ///
    /// [type_hint]: TypeHint
    pub variable_name: String,

    /// The index of the element within the type. For example, if the type is `Digest`, then this
    /// could be `0` for the first element, `1` for the second element, and so on.
    ///
    /// Does not apply to types that are not composed of multiple [`BFieldElement`][bfe]s, like `u32` or
    /// [`BFieldElement`][bfe] itself.
    ///
    /// [bfe]: triton_vm::BFieldElement
    pub index: Option<usize>,
}

impl ElementTypeHint {
    pub fn is_continuous_sequence(sequence: &[&Option<Self>]) -> bool {
        Self::is_continuous_sequence_for_ordering(sequence, Ordering::Greater)
            || Self::is_continuous_sequence_for_ordering(sequence, Ordering::Less)
    }

    fn is_continuous_sequence_for_ordering(sequence: &[&Option<Self>], ordering: Ordering) -> bool {
        for (left, right) in sequence.iter().tuple_windows() {
            let (Some(left), Some(right)) = (left, right) else {
                return false;
            };
            if left.partial_cmp(right) != Some(ordering) {
                return false;
            }
        }
        true
    }
}

impl PartialOrd for ElementTypeHint {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        if self.variable_name != other.variable_name {
            return None;
        }
        if self.type_name != other.type_name {
            return None;
        }
        match (self.index, other.index) {
            (Some(self_index), Some(other_index)) => self_index.partial_cmp(&other_index),
            (None, None) => Some(Ordering::Equal),
            _ => None,
        }
    }
}
