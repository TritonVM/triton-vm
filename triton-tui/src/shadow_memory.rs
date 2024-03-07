use std::collections::HashMap;
use std::iter::once;

use color_eyre::eyre::bail;
use color_eyre::eyre::Result;
use itertools::Itertools;
use triton_vm::instruction::*;
use triton_vm::op_stack::NumberOfWords::*;
use triton_vm::op_stack::*;
use triton_vm::prelude::*;

use crate::action::ExecutedInstruction;
use crate::element_type_hint::ElementTypeHint;

pub(crate) type TopOfStack = [BFieldElement; NUM_OP_STACK_REGISTERS];

/// Mimics the behavior of the actual memory. Helps debugging programs written for Triton VM by
/// tracking (manually set) type hints next to stack or RAM elements.
#[derive(Debug, Clone, Eq, PartialEq)]
pub(crate) struct ShadowMemory {
    /// Shadow stack mimicking the actual stack.
    pub stack: Vec<Option<ElementTypeHint>>,

    /// Shadow RAM mimicking the actual RAM.
    pub ram: HashMap<BFieldElement, Option<ElementTypeHint>>,
}

impl ShadowMemory {
    pub fn new_for_default_initial_state() -> Self {
        let stack = vec![None; NUM_OP_STACK_REGISTERS];
        let ram = HashMap::new();
        let initial_hint = Self::initial_program_digest_type_hint();

        let mut hints = Self { stack, ram };
        hints.apply_type_hint(initial_hint).unwrap();
        hints
    }

    pub fn new_for_initial_state(initial_state: &VMState) -> Self {
        let stack = vec![None; initial_state.op_stack.len()];
        let ram = HashMap::new();
        Self { stack, ram }
    }

    fn initial_program_digest_type_hint() -> TypeHint {
        let digest_length = tip5::DIGEST_LENGTH;
        TypeHint {
            type_name: Some("Digest".to_string()),
            variable_name: "program_digest".to_string(),
            starting_index: NUM_OP_STACK_REGISTERS - digest_length,
            length: digest_length,
        }
    }

    pub fn apply_type_hint(&mut self, type_hint: TypeHint) -> Result<()> {
        let type_hint_range_end = type_hint.starting_index + type_hint.length;
        if type_hint_range_end > self.stack.len() {
            bail!("stack is not large enough to apply type hint \"{type_hint}\"");
        }

        let element_type_hint_template = ElementTypeHint {
            type_name: type_hint.type_name,
            variable_name: type_hint.variable_name,
            index: None,
        };

        if type_hint.length <= 1 {
            let insertion_index = self.stack.len() - type_hint.starting_index - 1;
            self.stack[insertion_index] = Some(element_type_hint_template);
            return Ok(());
        }

        let stack_indices = type_hint.starting_index..type_hint_range_end;
        for (index_in_variable, stack_index) in stack_indices.enumerate() {
            let mut element_type_hint = element_type_hint_template.clone();
            element_type_hint.index = Some(index_in_variable);
            let insertion_index = self.stack.len() - stack_index - 1;
            self.stack[insertion_index] = Some(element_type_hint);
        }
        Ok(())
    }

    pub fn mimic_instruction(&mut self, executed_instruction: ExecutedInstruction) {
        let old_top_of_stack = executed_instruction.old_top_of_stack;
        match executed_instruction.instruction {
            Instruction::Pop(n) => _ = self.pop_n(n),
            Instruction::Push(_) => self.push(None),
            Instruction::Divine(n) => self.extend_by(n),
            Instruction::Dup(st) => self.dup(st),
            Instruction::Swap(st) => self.swap_top_with(st),
            Instruction::Halt => (),
            Instruction::Nop => (),
            Instruction::Skiz => _ = self.pop(),
            Instruction::Call(_) => (),
            Instruction::Return => (),
            Instruction::Recurse => (),
            Instruction::Assert => _ = self.pop(),
            Instruction::ReadMem(n) => self.read_mem(n, old_top_of_stack),
            Instruction::WriteMem(n) => self.write_mem(n, old_top_of_stack),
            Instruction::Hash => self.hash(),
            Instruction::DivineSibling => self.divine_sibling(),
            Instruction::AssertVector => _ = self.pop_n(N5),
            Instruction::SpongeInit => (),
            Instruction::SpongeAbsorb => self.sponge_absorb(),
            Instruction::SpongeSqueeze => self.sponge_squeeze(),
            Instruction::Add => self.binop_maybe_keep_hint(),
            Instruction::Mul => self.binop_maybe_keep_hint(),
            Instruction::Invert => self.unop(),
            Instruction::Eq => self.eq(),
            Instruction::Split => self.split(),
            Instruction::Lt => self.lt(),
            Instruction::And => self.binop(),
            Instruction::Xor => self.binop(),
            Instruction::Log2Floor => self.unop(),
            Instruction::Pow => self.binop(),
            Instruction::DivMod => self.div_mod(),
            Instruction::PopCount => self.unop(),
            Instruction::XxAdd => _ = self.pop_n(N3),
            Instruction::XxMul => _ = self.pop_n(N3),
            Instruction::XInvert => self.x_invert(),
            Instruction::XbMul => self.xb_mul(),
            Instruction::ReadIo(n) => self.extend_by(n),
            Instruction::WriteIo(n) => _ = self.pop_n(n),
        }
    }

    fn push(&mut self, element_type_hint: Option<ElementTypeHint>) {
        self.stack.push(element_type_hint);
    }

    fn extend_by(&mut self, n: NumberOfWords) {
        self.stack.extend(vec![None; n.into()]);
    }

    fn swap_top_with(&mut self, index: OpStackElement) {
        let top_index = self.stack.len() - 1;
        let other_index = self.stack.len() - usize::from(index) - 1;
        self.stack.swap(top_index, other_index);
    }

    fn pop(&mut self) -> Option<ElementTypeHint> {
        self.stack.pop().flatten()
    }

    fn pop_n(&mut self, n: NumberOfWords) -> Vec<Option<ElementTypeHint>> {
        let start_index = self.stack.len() - usize::from(n);
        self.stack.drain(start_index..).rev().collect()
    }

    fn dup(&mut self, st: OpStackElement) {
        let dup_index = self.stack.len() - usize::from(st) - 1;
        self.push(self.stack[dup_index].clone());
    }

    fn read_mem(&mut self, n: NumberOfWords, old_top_of_stack: TopOfStack) {
        let ram_pointer_hint = self.pop();
        let mut ram_pointer = old_top_of_stack[0];
        for _ in 0..n.num_words() {
            let hint = self.ram.get(&ram_pointer).cloned().flatten();
            self.push(hint);
            ram_pointer.decrement();
        }
        self.push(ram_pointer_hint);
    }

    fn write_mem(&mut self, n: NumberOfWords, old_top_of_stack: TopOfStack) {
        let ram_pointer_hint = self.pop();
        let mut ram_pointer = old_top_of_stack[0];
        for _ in 0..n.num_words() {
            let hint = self.pop();
            self.ram.insert(ram_pointer, hint);
            ram_pointer.increment();
        }
        self.push(ram_pointer_hint);
    }

    fn hash(&mut self) {
        let mut popped = self.pop_n(N5);
        popped.extend(self.pop_n(N5));
        self.extend_by(N5);

        let all_hashed_elements = popped.iter().collect_vec();

        let index_of_first_non_hashed_element = self.stack.len() - N5.num_words() - 1;
        let first_non_hashed_element = &self.stack[index_of_first_non_hashed_element];
        let all_hashed_and_first_non_hashed_elements = popped
            .iter()
            .chain(once(first_non_hashed_element))
            .collect_vec();

        let hashed_a_sequence = ElementTypeHint::is_continuous_sequence(&all_hashed_elements);
        let did_not_interrupt_sequence =
            !ElementTypeHint::is_continuous_sequence(&all_hashed_and_first_non_hashed_elements);
        let hashed_exactly_one_object = hashed_a_sequence && did_not_interrupt_sequence;

        if hashed_exactly_one_object {
            let Some(ref hash_type_hint) = popped[0] else {
                return;
            };
            let type_hint = TypeHint {
                type_name: Some("Digest".to_string()),
                variable_name: format!("{}_hash", hash_type_hint.variable_name),
                starting_index: 0,
                length: Digest::default().0.len(),
            };
            self.apply_type_hint(type_hint).unwrap();
        }
    }

    fn divine_sibling(&mut self) {
        self.pop_n(N5);
        self.extend_by(N5);
        self.extend_by(N5);
    }

    fn sponge_absorb(&mut self) {
        self.pop_n(N5);
        self.pop_n(N5);
    }

    fn sponge_squeeze(&mut self) {
        self.extend_by(N5);
        self.extend_by(N5);
    }

    fn binop_maybe_keep_hint(&mut self) {
        let lhs = self.pop();
        let rhs = self.pop();
        self.push(lhs.xor(rhs));
    }

    fn unop(&mut self) {
        self.pop();
        self.push(None);
    }

    fn binop(&mut self) {
        self.pop_n(N2);
        self.push(None);
    }

    fn eq(&mut self) {
        let lhs = self.pop();
        let rhs = self.pop();
        let (Some(lhs), Some(rhs)) = (lhs, rhs) else {
            self.push(None);
            return;
        };

        let type_hint = ElementTypeHint {
            type_name: Some("bool".to_string()),
            variable_name: format!("{} == {}", lhs.variable_name, rhs.variable_name),
            index: None,
        };
        self.push(Some(type_hint));
    }

    fn split(&mut self) {
        let maybe_type_hint = self.pop();
        self.extend_by(N2);

        let Some(type_hint) = maybe_type_hint else {
            return;
        };
        let lo_index = self.stack.len() - 1;
        let hi_index = self.stack.len() - 2;

        let mut lo = type_hint.clone();
        lo.variable_name.push_str("_lo");
        self.stack[lo_index] = Some(lo);

        let mut hi = type_hint;
        hi.variable_name.push_str("_hi");
        self.stack[hi_index] = Some(hi);
    }

    fn lt(&mut self) {
        let smaller = self.pop();
        let bigger = self.pop();
        let (Some(smaller), Some(bigger)) = (smaller, bigger) else {
            self.push(None);
            return;
        };

        let type_hint = ElementTypeHint {
            type_name: Some("bool".to_string()),
            variable_name: format!("{} < {}", smaller.variable_name, bigger.variable_name),
            index: None,
        };
        self.push(Some(type_hint));
    }

    fn div_mod(&mut self) {
        self.pop_n(N2);
        self.extend_by(N2);
    }

    fn x_invert(&mut self) {
        self.pop_n(N3);
        self.extend_by(N3);
    }

    fn xb_mul(&mut self) {
        self.pop_n(N4);
        self.extend_by(N3);
    }
}

impl Default for ShadowMemory {
    fn default() -> Self {
        Self::new_for_default_initial_state()
    }
}

#[cfg(test)]
mod tests {
    use assert2::assert;
    use assert2::let_assert;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use proptest_arbitrary_interop::arb;
    use test_strategy::proptest;

    use super::*;

    impl Arbitrary for ShadowMemory {
        type Parameters = ();
        fn arbitrary_with(_args: Self::Parameters) -> Self::Strategy {
            let stack_strategy = vec(arb(), NUM_OP_STACK_REGISTERS..=100);
            let ram_strategy = arb();
            (stack_strategy, ram_strategy)
                .prop_map(|(stack, ram)| Self { stack, ram })
                .boxed()
        }

        type Strategy = BoxedStrategy<Self>;
    }

    #[test]
    fn default_type_hint_stack_is_as_long_as_default_actual_stack() {
        let actual_stack_length = ShadowMemory::default().stack.len();
        let expected_stack_length = OpStack::new(Digest::default()).stack.len();
        assert!(expected_stack_length == actual_stack_length);
    }

    #[proptest]
    fn type_hint_stack_grows_and_shrinks_like_actual_stack(
        mut type_hints: ShadowMemory,
        #[strategy(arb())] executed_instruction: ExecutedInstruction,
    ) {
        let initial_length = type_hints.stack.len();
        type_hints.mimic_instruction(executed_instruction);
        let actual_stack_delta = type_hints.stack.len() as i32 - initial_length as i32;
        let expected_stack_delta = executed_instruction.instruction.op_stack_size_influence();
        assert!(expected_stack_delta == actual_stack_delta);
    }

    #[proptest]
    fn write_mem_then_read_mem_preserves_type_hints_on_stack(
        mut type_hints: ShadowMemory,
        #[strategy(arb())] num_words: NumberOfWords,
        #[strategy(arb())] ram_pointer: BFieldElement,
    ) {
        let mut top_of_stack_before_write = [bfe!(0); NUM_OP_STACK_REGISTERS];
        top_of_stack_before_write[0] = ram_pointer;

        let offset_of_last_written_element = bfe!(num_words) - bfe!(1);
        let mut top_of_stack_before_read = [bfe!(0); NUM_OP_STACK_REGISTERS];
        top_of_stack_before_read[0] = ram_pointer + offset_of_last_written_element;

        let initial_type_hints = type_hints.clone();
        type_hints.mimic_instruction(ExecutedInstruction::new(
            Instruction::WriteMem(num_words),
            top_of_stack_before_write,
            TopOfStack::default(),
        ));
        prop_assert_ne!(&initial_type_hints.stack, &type_hints.stack);
        type_hints.mimic_instruction(ExecutedInstruction::new(
            Instruction::ReadMem(num_words),
            top_of_stack_before_read,
            TopOfStack::default(),
        ));
        prop_assert_eq!(initial_type_hints.stack, type_hints.stack);
    }

    #[test]
    fn apply_type_hint_of_length_one() {
        let type_name = Some("u32".to_string());
        let variable_name = "foo".to_string();
        let type_hint_to_apply = TypeHint {
            type_name: type_name.clone(),
            variable_name: variable_name.clone(),
            starting_index: 0,
            length: 1,
        };
        let expected_hint = ElementTypeHint {
            type_name,
            variable_name,
            index: None,
        };

        let mut type_hints = ShadowMemory::default();
        let_assert!(Ok(()) = type_hints.apply_type_hint(type_hint_to_apply));
        let_assert!(Some(maybe_hint) = type_hints.stack.last());
        let_assert!(Some(hint) = maybe_hint.clone());
        assert!(expected_hint == hint);
    }

    #[test]
    fn applying_type_hint_at_illegal_index_gives_error() {
        let type_hint_to_apply = TypeHint {
            type_name: Some("u32".to_string()),
            variable_name: "foo".to_string(),
            starting_index: 100,
            length: 1,
        };

        let mut type_hints = ShadowMemory::default();
        let_assert!(Err(_) = type_hints.apply_type_hint(type_hint_to_apply));
    }

    #[test]
    fn hashing_one_complete_object_gives_type_hint_for_digest() {
        let type_hint_to_apply = TypeHint {
            type_name: Some("array".to_string()),
            variable_name: "foo".to_string(),
            starting_index: 0,
            length: 10,
        };
        let executed_instruction = ExecutedInstruction::new(
            Instruction::Hash,
            [bfe!(0); NUM_OP_STACK_REGISTERS],
            TopOfStack::default(),
        );

        let mut type_hints = ShadowMemory::default();
        let_assert!(Ok(()) = type_hints.apply_type_hint(type_hint_to_apply));
        type_hints.mimic_instruction(executed_instruction);

        let_assert!(Some(maybe_hint) = type_hints.stack.last());
        let_assert!(Some(hint) = maybe_hint.clone());
        assert!(hint.type_name == Some("Digest".to_string()));
        assert!(hint.variable_name == "foo_hash".to_string());
    }
}
