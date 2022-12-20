# Registers

This section covers all columns in the Protocol Table.
Only a subset of these registers relate to the instruction set;
the remaining registers exist only to enable an efficient arithmetization and are marked with an asterisk (\*).

| Register               | Name                         | Purpose                                                                                                            |
|:-----------------------|:-----------------------------|:-------------------------------------------------------------------------------------------------------------------|
| *`clk`                 | cycle counter                | counts the number of cycles the program has been running for                                                       |
| *`IsPadding`           | padding indicator            | indicates whether current state is only recorded to improve on STARK's computational runtime                       |
| *`PreviousInstruction` | previous instruction         | holds the opcode of the instruction executed in the previous clock cycle (or 0 if such cycle exists)               |
| `ip`                   | instruction pointer          | contains the memory address (in Program Memory) of the instruction                                                 |
| `ci`                   | current instruction register | contains the current instruction                                                                                   |
| `nia`                  | next instruction register    | contains either the instruction at the next address in Program Memory, or the argument for the current instruction |
| *`ib0` through `ib?`   | instruction bucket           | decomposition of the instruction's opcode used to keep the AIR degree low                                          |
| `jsp`                  | jump stack pointer           | contains the memory address (in jump stack memory) of the top of the jump stack                                    |
| `jso`                  | jump stack origin            | contains the value of the instruction pointer of the last `call`                                                   |
| `jsd`                  | jump stack destination       | contains the argument of the last `call`                                                                           |
| `st0` through `st15`   | operational stack registers  | contain explicit operational stack values                                                                          |
| *`osp`                 | operational stack pointer    | contains the OpStack address of the top of the operational stack                                                   |
| *`osv`                 | operational stack value      | contains the (stack) memory value at the given address                                                             |
| *`hv0` through `hv3`   | helper variable registers    | helper variables for some arithmetic operations                                                                    |
| *`ramp`                | RAM pointer                  | contains an address pointing into the RAM                                                                          |
| *`ramv`                | RAM value                    | contains the value of the RAM element at the address currently held in `ramp`                                      |

## Instruction

Register `ip`, the *instruction pointer*, contains the address of the current instruction in Program Memory.
The instruction is contained in the register *current instruction*, or `ci`.
Register *next instruction (or argument)*, or `nia`, either contains the next instruction or the argument for the current instruction in `ci`.
For reasons of arithmetization, `ci` is decomposed, giving rise to the *instruction bucket registers*, labeled `ib0` through `ib?`.

## Stack

The stack is represented by 16 registers called *stack registers* (`st0` – `st15`) plus the OpStack Underflow Memory.
The top 16 elements of the OpStack are directly accessible, the remainder of the OpStack, i.e, the part held in OpStack Underflow Memory, is not.
In order to access elements of the OpStack held in OpStack Underflow Memory, the stack has to shrink by discarding elements from the top – potentially after writing them to RAM – thus moving lower elements into the stack registers.

The stack grows upwards, in line with the metaphor that justifies the name "stack".

For reasons of arithmetization, the stack always contains a minimum of 16 elements.
All these elements are initially 0.
Trying to run an instruction which would result in a stack of smaller total length than 16 crashes the VM.

The registers `osp` and `osv` are not directly accessible by the program running in TritonVM.
They exist only to allow efficient arithmetization.

## RAM

TritonVM has dedicated Random-Access Memory.
Programs can read from and write to RAM using instructions `read_mem` and `write_mem`.
The address to read from – respectively, to write to – is the stack's second-to-top-most OpStack element, i.e, `st1`.

The registers `ramp` and `ramv` are not directly accessible by the program running in TritonVM.
They exist only to allow efficient arithmetization.

## Helper Variables

Some instructions require helper variables in order to generate an efficient arithmetization.
To this end, there are 4 helper variable registers, labeled `hv0` through `hv3`.
These registers are part of the arithmetization of the architecture, but not needed to define the instruction set.

Because they are only needed for some instructions, the helper variables are not generally defined.
For instruction group [`stack_shrinks_and_top_3_unconstrained`](instruction-groups.md#group-stack_shrinks_and_top_3_unconstrained) and its derivatives, as well as for instructions
[`dup`](instruction-specific-transition-constraints.md#helper-variable-definitions-for-dup--i),
[`swap`](instruction-specific-transition-constraints.md#helper-variable-definitions-for-swap--i),
[`skiz`](instruction-specific-transition-constraints.md#helper-variable-definitions-for-skiz),
[`divine_sibling`](instruction-specific-transition-constraints.md#helper-variable-definitions-for-divine_sibling),
[`split`](instruction-specific-transition-constraints.md#helper-variable-definitions-for-split), and
[`eq`](instruction-specific-transition-constraints.md#helper-variable-definitions-for-eq),
the behavior is defined in the respective sections.
