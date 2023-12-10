use std::ops::Index;
use std::ops::IndexMut;

use color_eyre::eyre::bail;
use color_eyre::eyre::Result;

use triton_vm::instruction::TypeHint;
use triton_vm::op_stack::NUM_OP_STACK_REGISTERS;

/// A helper “shadow stack” mimicking the behavior of the actual stack. Helps debugging programs
/// written for Triton VM by (manually set) type hints next to stack elements.
#[derive(Debug, Clone)]
pub(crate) struct TypeHintStack {
    pub type_hints: Vec<Option<ElementTypeHint>>,
}

/// A hint about the type of a single stack element. Helps debugging programs written for Triton VM.
/// **Does not enforce types.**
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ElementTypeHint {
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

impl TypeHintStack {
    pub fn new() -> Self {
        let type_hints = vec![None; NUM_OP_STACK_REGISTERS];
        let mut stack = Self { type_hints };

        let program_hash_type_hint = TypeHint {
            type_name: Some("Digest".to_string()),
            variable_name: "program_digest".to_string(),
            starting_index: 11,
            length: triton_vm::Digest::default().0.len(),
        };
        stack.apply_type_hint(&program_hash_type_hint).unwrap();
        stack
    }

    pub fn len(&self) -> usize {
        self.type_hints.len()
    }

    pub fn apply_type_hint(&mut self, type_hint: &TypeHint) -> Result<()> {
        let type_hint_range_end = type_hint.starting_index + type_hint.length;
        if type_hint_range_end > self.type_hints.len() {
            bail!("the op stack is not large enough to apply the given type hint");
        }

        let element_type_hint_template = ElementTypeHint {
            type_name: type_hint.type_name.clone(),
            variable_name: type_hint.variable_name.clone(),
            index: None,
        };

        if type_hint.length <= 1 {
            let insertion_index = self.len() - type_hint.starting_index - 1;
            self[insertion_index] = Some(element_type_hint_template);
            return Ok(());
        }

        let stack_indices = type_hint.starting_index..type_hint_range_end;
        for (index_in_variable, stack_index) in stack_indices.enumerate() {
            let mut element_type_hint = element_type_hint_template.clone();
            element_type_hint.index = Some(index_in_variable);
            let insertion_index = self.len() - stack_index - 1;
            self[insertion_index] = Some(element_type_hint);
        }
        Ok(())
    }
}

impl Default for TypeHintStack {
    fn default() -> Self {
        Self::new()
    }
}

impl Index<usize> for TypeHintStack {
    type Output = Option<ElementTypeHint>;

    fn index(&self, index: usize) -> &Self::Output {
        &self.type_hints[index]
    }
}

impl IndexMut<usize> for TypeHintStack {
    fn index_mut(&mut self, index: usize) -> &mut Self::Output {
        &mut self.type_hints[index]
    }
}
