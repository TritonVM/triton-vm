# Processor's Instruction Constraints

Due to their complexity, the transition constraints for the [Processor Table](processor-table.md) are listed and explained in this section.
To keep the degrees of the AIR polynomials low, instructions are grouped based on their effect.
An instruction's effect not captured by the groups it is part of needs to be arithmetized separately.

To exemplify the interplay of an instruction's group and individual constraints, take instruction `add`.
Executing `add` increases the instruction pointer `ip` by 1, a property shared by many instructions.
Consequently, `add` belongs to instruction group `step_1`.
Instruction `add` manipulates the stack's top two elements, combining them into one, which is captured by instruction group `binop`.
Lastly, the specific behavior of `add` is to, well, add these stack elements, which is described in `add`'s instruction specific constraint.
