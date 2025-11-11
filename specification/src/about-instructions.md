# About Instructions

This section describes some general properties and behaviors of Triton VM’s instructions.
The [next section](instructions.md) lists and describes all the instructions that Triton VM can
execute.

### Instruction Sizes

Most instructions are contained within a single, parameterless machine word.
They are considered single-word instructions.
An example of a single-word instruction is `halt`.

Some instructions take one machine word as argument and are so considered double-word instructions.
They are recognized by the form “`instr` + `arg`”.
An example of a double-word instruction is `push` + `a`, which takes immediate argument `a`.

### Automatic Instruction Pointer Increment

Unless a different behavior is indicated, instructions increment the
[instruction pointer](registers.md#instruction) `ip` by their [size](#instruction-sizes).
For example, instruction `halt` increments the `ip` by 1.
Instruction `push` increments the `ip` by 2.
Many [control flow instructions](instructions.md#control-flow) manipulate the instruction pointer in
a manner deviating from this general scheme.

### Non-Deterministic Instructions

Instructions [`divine`](instructions.md#divine--n) and [`merkle_step`](instructions.md#merkle_step)
make Triton a virtual machine that can execute non-deterministic programs.
As programs go, this concept is somewhat unusual and benefits from additional explanation.

From the perspective of the program, the instruction `divine` makes some `n` elements magically
appear on the stack. It is not at all specified what those elements are, but generally speaking,
they have to be exactly correct, else execution fails. Hence, from the perspective of the program,
it just non-deterministically guesses the correct values in a moment of divine clarity.

Looking at the entire system, consisting of the VM, the program, and all inputs – both public and
secret – execution _is_ deterministic: the divined values were supplied as secret input and are read
from there.

### Hashing and Sponge Instructions

The instructions [`sponge_init`](instructions.md#sponge_init),
[`sponge_absorb`](instructions.md#sponge_absorb),
[`sponge_absorb_mem`](instructions.md#sponge_absorb_mem), and
[`sponge_squeeze`](instructions.md#sponge_squeeze) are the interface
for using the [Tip5](https://eprint.iacr.org/2023/107.pdf) permutation in a
[Sponge](https://keccak.team/sponge_duplex.html) construction.
The capacity is never accessible to the program that's being executed by Triton VM.

At any given time, at most one Sponge state exists.
In particular, Triton VM does not start with an initialized Sponge state.
If a program uses the Sponge state, the first Sponge instruction must be `sponge_init`;
otherwise, Triton VM will crash.

Only instruction `sponge_init` resets the state of the Sponge, and only the three Sponge
instructions influence the Sponge's state.
Notably, executing instruction [`hash`](instructions.md#hash) does not modify the Sponge's state.

When using the Sponge instructions, it is the programmer's responsibility to take care of proper
input padding:
Triton VM cannot know the number of elements that will be absorbed.
For more information, see section 2.5 of the [Tip5 paper](https://eprint.iacr.org/2023/107.pdf).

### Regarding Opcodes

An instruction's _[operation code](https://en.wikipedia.org/wiki/Opcode)_, or _opcode_, is the
machine word uniquely identifying the instruction.
For reasons of efficient [arithmetization](arithmetization.md), certain properties of the
instruction are encoded in the opcode.
Concretely, interpreting the field element in standard representation:
- for all double-word instructions, the least significant bit is 1.
- for all instructions shrinking the operational stack, the second-to-least significant bit is 1.
- for all [u32 instructions](instructions.md#bitwise-arithmetic), the third-to-least significant bit
  is 1.

The first property is used by instruction [skiz](instructions.md#skiz) (see also its
[arithmetization](instruction-specific-transition-constraints.md#instruction-skiz)).
The second property helps with proving consistency of the
[Op Stack](data-structures.md#operational-stack).
The third property allows efficient arithmetization of the running product for the
Permutation Argument between [Processor Table](processor-table.md) and [U32 Table](u32-table.md).
