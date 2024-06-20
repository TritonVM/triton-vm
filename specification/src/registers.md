# Registers

This section covers all columns in the [Processor Table](processor-table.md).
Only a subset of these registers relate to the instruction set;
the remaining registers exist only to enable an efficient arithmetization and are marked with an asterisk (\*).

| Register             | Name                                      | Purpose                                                                                                                                                                                                                   |
|:---------------------|:------------------------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| *`clk`               | cycle counter                             | counts the number of cycles the program has been running for                                                                                                                                                              |
| *`IsPadding`         | padding indicator                         | indicates whether current state is only recorded to improve on STARK's computational runtime                                                                                                                              |
| `ip`                 | instruction pointer                       | contains the memory address (in Program Memory) of the instruction                                                                                                                                                        |
| `ci`                 | current instruction register              | contains the current instruction                                                                                                                                                                                          |
| `nia`                | next instruction register                 | contains either the instruction at the next address in Program Memory, or the argument for the current instruction                                                                                                        |
| *`ib0` through `ib6` | instruction bit                           | decomposition of the instruction's opcode used to keep the AIR degree low                                                                                                                                                 |
| `jsp`                | jump stack pointer                        | contains the memory address (in jump stack memory) of the top of the jump stack                                                                                                                                           |
| `jso`                | jump stack origin                         | contains the value of the instruction pointer of the last `call`                                                                                                                                                          |
| `jsd`                | jump stack destination                    | contains the argument of the last `call`                                                                                                                                                                                  |
| `st0` through `st15` | operational stack registers               | contain explicit operational stack values                                                                                                                                                                                 |
| *`op_stack_pointer`  | operational stack pointer                 | the current size of the operational stack                                                                                                                                                                                 |
| *`hv0` through `hv5` | helper variable registers                 | helper variables for some arithmetic operations                                                                                                                                                                           |
| *`cjd_mul`           | clock jump difference lookup multiplicity | multiplicity with which the current `clk` is [looked up](lookup-argument.md) by the [Op Stack Table](operational-stack-table.md), [RAM Table](random-access-memory-table.md), and [Jump Stack Table](jump-stack-table.md) |

## Instruction

Register `ip`, the *instruction pointer*, contains the address of the current instruction in Program Memory.
The instruction is contained in the register *current instruction*, or `ci`.
Register *next instruction (or argument)*, or `nia`, either contains the next instruction or the argument for the current instruction in `ci`.
For reasons of arithmetization, `ci` is decomposed, giving rise to the *instruction bit registers*, labeled `ib0` through `ib6`.

## Stack

The stack is represented by 16 registers called *stack registers* (`st0` – `st15`) plus the op stack underflow memory.
The top 16 elements of the op stack are directly accessible, the remainder of the op stack, i.e, the part held in op stack underflow memory, is not.
In order to access elements of the op stack held in op stack underflow memory, the stack has to shrink by discarding elements from the top – potentially after writing them to RAM – thus moving lower elements into the stack registers.

The stack grows upwards, in line with the metaphor that justifies the name "stack".

For reasons of arithmetization, the stack always contains a minimum of 16 elements.
Trying to run an instruction which would result in a stack of smaller total length than 16 crashes the VM.

Stack elements `st0` through `st10` are initially 0.
Stack elements `st11` through `st15`, _i.e._, the very bottom of the stack, are initialized with the hash digest of the program that is being executed.
See [the mechanics of program attestation](program-attestation.md#mechanics) for further explanations on stack initialization.

The register `op_stack_pointer` is not directly accessible by the program running in TritonVM.
It exists only to allow efficient arithmetization.

## Helper Variables

Some instructions require helper variables in order to generate an efficient arithmetization.
To this end, there are 6 helper variable registers, labeled `hv0` through `hv5`.
These registers are part of the arithmetization of the architecture, but not needed to define the instruction set.

Because they are only needed for some instructions, the helper variables are not generally defined.
For instruction group [`decompose_arg`](instruction-groups.md#group-decompose_arg) and instructions
[`skiz`](instruction-specific-transition-constraints.md#helper-variable-definitions-for-skiz),
[`split`](instruction-specific-transition-constraints.md#helper-variable-definitions-for-split),
[`eq`](instruction-specific-transition-constraints.md#helper-variable-definitions-for-eq),
[`merkle_step`](instruction-specific-transition-constraints.md#helper-variable-definitions-for-merkle_step),
[`merkle_step_mem`](instruction-specific-transition-constraints.md#helper-variable-definitions-for-merkle_step_mem),
[`xx_dot_step`](instruction-specific-transition-constraints.md#instruction-xx_dot_step), and
[`xb_dot_step`](instruction-specific-transition-constraints.md#instruction-xb_dot_step),
the behavior is defined in the respective sections.
