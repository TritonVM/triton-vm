use std::cmp::Ordering;

use arbitrary::Arbitrary;
use itertools::Itertools;
use ratatui::prelude::*;

/// A hint about the type of a single stack element. Helps debugging programs written for Triton VM.
/// **Does not enforce types.**
#[derive(Debug, Clone, Eq, PartialEq, Hash, Arbitrary)]
pub(crate) struct ElementTypeHint {
    /// The name of the type. See [`TypeHint`][type_hint] for details.
    ///
    /// [type_hint]: triton_vm::instruction::TypeHint
    pub type_name: Option<String>,

    /// The name of the variable. See [`TypeHint`][type_hint] for details.
    ///
    /// [type_hint]: triton_vm::instruction::TypeHint
    pub variable_name: String,

    /// The index of the element within the type. For example, if the type is `Digest`, then this
    /// could be `0` for the first element, `1` for the second element, and so on.
    ///
    /// Does not apply to types that are not composed of multiple [`BFieldElement`][bfe]s, like `u32` or
    /// [`BFieldElement`][bfe] itself.
    ///
    /// [bfe]: triton_vm::prelude::BFieldElement
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

    pub fn render(maybe_self: &Option<Self>) -> Vec<Span> {
        let Some(element_type_hint) = maybe_self else {
            return vec![];
        };

        let mut line = vec![];
        line.push(element_type_hint.variable_name.clone().into());
        if let Some(ref type_name) = element_type_hint.type_name {
            line.push(": ".dim());
            line.push(type_name.into());
        }
        if let Some(index) = element_type_hint.index {
            line.push(format!(" ({index})").dim());
        }
        line
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

#[cfg(test)]
mod tests {
    use proptest::prelude::*;
    use proptest_arbitrary_interop::arb;
    use test_strategy::proptest;

    use super::*;

    #[proptest]
    fn comparison_with_different_variable_names_is_impossible(
        #[strategy(arb())] type_hint_0: ElementTypeHint,
        #[strategy(arb())]
        #[filter(#type_hint_0.variable_name != #type_hint_1.variable_name)]
        type_hint_1: ElementTypeHint,
    ) {
        prop_assert_eq!(type_hint_0.partial_cmp(&type_hint_1), None);
    }

    #[proptest]
    fn comparison_with_different_type_names_is_impossible(
        #[strategy(arb())] type_hint_0: ElementTypeHint,
        #[strategy(arb())]
        #[filter(#type_hint_0.type_name != #type_hint_1.type_name)]
        type_hint_1: ElementTypeHint,
    ) {
        prop_assert_eq!(type_hint_0.partial_cmp(&type_hint_1), None);
    }

    #[test]
    fn continuous_increasing_sequence() {
        let template = ElementTypeHint {
            type_name: None,
            variable_name: "x".to_string(),
            index: None,
        };
        let mut hint_0 = template.clone();
        let mut hint_1 = template.clone();
        let mut hint_2 = template.clone();

        hint_0.index = Some(0);
        hint_1.index = Some(1);
        hint_2.index = Some(2);

        let sequence = [&Some(hint_0), &Some(hint_1), &Some(hint_2)];
        assert!(ElementTypeHint::is_continuous_sequence(&sequence));
    }

    #[test]
    fn continuous_decreasing_sequence() {
        let template = ElementTypeHint {
            type_name: None,
            variable_name: "x".to_string(),
            index: None,
        };
        let mut hint_0 = template.clone();
        let mut hint_1 = template.clone();
        let mut hint_2 = template.clone();

        hint_0.index = Some(2);
        hint_1.index = Some(1);
        hint_2.index = Some(0);

        let sequence = [&Some(hint_0), &Some(hint_1), &Some(hint_2)];
        assert!(ElementTypeHint::is_continuous_sequence(&sequence));
    }

    #[test]
    fn non_continuous_sequence() {
        let template = ElementTypeHint {
            type_name: None,
            variable_name: "x".to_string(),
            index: None,
        };
        let mut hint_0 = template.clone();
        let mut hint_1 = template.clone();
        let mut hint_2 = template.clone();

        hint_0.index = Some(0);
        hint_1.index = Some(2);
        hint_2.index = Some(1);

        let sequence = [&Some(hint_0), &Some(hint_1), &Some(hint_2)];
        assert!(!ElementTypeHint::is_continuous_sequence(&sequence));
    }

    #[test]
    fn interrupted_sequence() {
        let template = ElementTypeHint {
            type_name: None,
            variable_name: "x".to_string(),
            index: None,
        };

        let mut hint_0 = template.clone();
        let mut hint_2 = template.clone();

        hint_0.index = Some(0);
        hint_2.index = Some(2);

        let sequence = [&Some(hint_0), &None, &Some(hint_2)];
        assert!(!ElementTypeHint::is_continuous_sequence(&sequence));
    }
}
