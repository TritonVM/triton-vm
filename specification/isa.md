# Triton VM Instruction Set Architecture

Triton VM is a stack machine with RAM.
It is a [Harvard architecture](https://en.wikipedia.org/wiki/Harvard_architecture) with read-only memory for the program.
The arithmetization of the VM is defined over the *B-field* $\mathbb{F}_p$ where $p=2^{64}-2^{32}+1$.
This means the registers and memory elements take values from $\mathbb{F}_p$, and the transition function gives rise to low-degree transition verification polynomials from the ring of multivariate polynomials over $\mathbb{F}_p$.

Instructions have variable width:
they either consist of one word, i.e., one B-field element, or of two words, i.e., two B-field elements.
An example for a single-word instruction is `pop`, removing the top of the stack.
An example for a double-word instruction is `push` + `arg`, pushing `arg` to the stack.

Triton VM has two interfaces for data input, one for public and one for secret data, and one interface for data output, whose data is always public.
The public interfaces differ from the private one, especially regarding their arithmetization.

## Data Structures

### Memory
The term *memory* refers to a data structure that gives read access (and possibly write access, too) to elements indicated by an *address*.
Regardless of the data structure, the address lives in the B-field.
There are four separate notions of memory:
1. *RAM*, to which the VM can read and write field elements.
2. *Program Memory*, from which the VM reads instructions.
3. *OpStack Memory*, which stores the operational stack.
4. *Jump Stack Memory*, which stores the entire jump stack.

### Operational Stack
The stack is a last-in;first-out data structure that allows the program to store intermediate variables, pass arguments, and keep pointers to objects held in RAM.
In this document, the operational stack is either referred to as just “stack” or, if more clarity is desired, “OpStack.”

From the Virtual Machine's point of view, the stack is a single, continuous object.
The first 16 elements of the stack can be accessed very conveniently.
Elements deeper in the stack require removing some of the higher elements, possibly after storing them in RAM.

For reasons of arithmetization, the stack is actually split into two distinct parts:
1. the _operational stack registers_ `st0` through `st15`, and
1. the _OpStack Underflow Memory_.

The motivation and the interplay between the two parts is described and exemplified in [arithmetization of the OpStack table](arithmetization.md#operational-stack-table).

### Jump Stack
Another last-in;first-out data structure that keeps track of return and destination addresses.
This stack changes only when control follows a `call` or `return` instruction.

## Registers

This section covers all columns in the Protocol Table.
Only a subset of these registers relate to the instruction set;
the remaining registers exist only to enable an efficient arithmetization and are marked with an asterisk.

| Register             | Name                         | Purpose                                                                                                            |
|:---------------------|:-----------------------------|:-------------------------------------------------------------------------------------------------------------------|
| *`clk`               | cycle counter                | counts the number of cycles the program has been running for                                                       |
| `ip`                 | instruction pointer          | contains the memory address (in Program Memory) of the instruction                                                 |
| `ci`                 | current instruction register | contains the current instruction                                                                                   |
| `nia`                | next instruction register    | contains either the instruction at the next address in Program Memory, or the argument for the current instruction |
| *`ib0` through `ib?` | instruction bucket           | decomposition of the instruction's opcode used to keep the AIR degree low                                          |
| `jsp`                | jump stack pointer           | contains the memory address (in jump stack memory) of the top of the jump stack                                    |
| `jso`                | jump stack origin            | contains the value of the instruction pointer of the last `call`                                                   |
| `jsd`                | jump stack destination       | contains the argument of the last `call`                                                                           |
| `st0` through `st15` | operational stack registers  | contain explicit operational stack values                                                                          |
| *`osp`               | operational stack pointer    | contains the OpStack address of the top of the operational stack                                                   |
| *`osv`               | operational stack value      | contains the (stack) memory value at the given address                                                             |
| *`hv0` through `hv3` | helper variable registers    | helper variables for some arithmetic operations                                                                    |
| *`ramv`              | RAM value                    | contains the value of the RAM element at the address currently held in `st1`                                       |

### Instruction

Register `ip`, the *instruction pointer*, contains the address of the current instruction in Program Memory.
The instruction is contained in the register *current instruction*, or `ci`.
Register *next instruction (or argument)*, or `nia`, either contains the next instruction or the argument for the current instruction in `ci`.
For reasons of arithmetization, `ci` is decomposed, giving rise to the *instruction bucket registers*, labeled `ib0` through `ib?`.

### Stack

The stack is represented by 16 registers called *stack registers* (`st0` – `st15`) plus the OpStack Underflow Memory.
The top 16 elements of the OpStack are directly accessible, the remainder of the OpStack, i.e, the part held in OpStack Underflow Memory, is not.
In order to access elements of the OpStack held in OpStack Underflow Memory, the stack has to shrink by discarding elements from the top – potentially after writing them to RAM – thus moving lower elements into the stack registers.

The stack grows upwards, in line with the metaphor that justifies the name "stack".

For reasons of arithmetization, the stack always contains a minimum of 16 elements.
All these elements are initially 0.
Trying to run an instruction which would result in a stack of smaller total length than 16 crashes the VM.

The registers `osp` and `osv` are not directly accessible by the program running in TritonVM.
They exist only to allow efficient arithmetization.

### RAM

TritonVM has dedicated Random-Access Memory.
Programs can read from and write to RAM using instructions `read_mem` and `write_mem`.
The address to read from – respectively, to write to – is the stack's second-to-top-most OpStack element, i.e, `st1`.

The register `ramv` is not directly accessible by the program running in TritonVM.
It exists only to allow efficient arithmetization.

### Helper Variables

Some instructions require helper variables in order to generate an efficient arithmetization.
To this end, there are 4 helper registers, labeled `hv0` through `hv3`.
These registers are part of the arithmetization of the architecture, but not needed to define the instruction set.

## Instructions

Most instructions are contained within a single, parameterless machine word.

Some instructions take a machine word as argument and are so considered double-word instructions.
They are recognized by the form "`instr` + `arg`".

### OpStack Manipulation

| Instruction  | Value | old OpStack         | new OpStack           | Description                                                                      |
|:-------------|:------|:--------------------|:----------------------|:---------------------------------------------------------------------------------|
| `pop`        | ?     | `_ a`               | `_`                   | Pops top element from stack.                                                     |
| `push` + `a` | ?     | `_`                 | `_ a`                 | Pushes `a` onto the stack.                                                       |
| `divine`     | ?     | `_`                 | `_ a`                 | Pushes a non-deterministic element `a` to the stack. Interface for secret input. |
| `dup`  + `i` | ?     | e.g., `_ e d c b a` | e.g., `_ e d c b a d` | Duplicates the element `i` positions away from the top, assuming `0 <= i < 16`.  |
| `swap` + `i` | ?     | e.g., `_ e d c b a` | e.g., `_ e a c b d`   | Swaps the `i`th stack element with the top of the stack, assuming `0 < i < 16`.  |

Instruction `divine` (together with [`divine_sibling`](#hashing)) make TritonVM a virtual machine that can execute non-deterministic programs.
As programs go, this concept is somewhat unusual and benefits from additional explanation.
The name of the instruction is the verb (not the adjective) meaning “to discover by intuition or insight.”

From the perspective of the program, the instruction `divine` makes some element `a` magically appear on the stack.
It is not at all specified what `a` is, but generally speaking, `a` has to be exactly correct, else execution fails.
Hence, from the perspective of the program, it just non-deterministically guesses the correct value of `a` in a moment of divine clarity.

Looking at the entire system, consisting of the VM, the program, and all inputs – both public and secret – execution _is_ deterministic:
the value `a` was supplied as a secret input.

### Control Flow

| Instruction  | Value | old OpStack | new OpStack | old `ip` | new `ip`     | old JumpStack | new JumpStack | Description                                                                                                                 |
|:-------------|:------|:------------|:------------|:---------|:-------------|:--------------|:--------------|:----------------------------------------------------------------------------------------------------------------------------|
| `nop`        | ?     | `_`         | `_`         | `_`      | `_ + 1`      | `_`           | `_`           | Do nothing                                                                                                                  |
| `skiz`       | ?     | `_ a`       | `_`         | `_`      | `_ + s`      | `_`           | `_`           | Skip next instruction if `a` is zero. `s` ∈ {1, 2, 3} depends on `a` and whether or not next instruction takes an argument. |
| `call` + `d` | ?     | `_`         | `_`         | `o`      | `d`          | `_`           | `_ (o+2, d)`  | Push `(o+2,d)` to the jump stack, and jump to absolute address `d`                                                          |
| `return`     | ?     | `_`         | `_`         | `_`      | `o`          | `_ (o, d)`    | `_`           | Pop one pair off the jump stack and jump to that pair's return address (which is the first element).                        |
| `recurse`    | ?     | `_`         | `_`         | `_`      | `d`          | `_ (o, d)`    | `_ (o, d)`    | Peek at the top pair of the jump stack and jump to that pair's destination address (which is the second element).           |
| `assert`     | ?     | `_ a`       | `_`         | `_`      | `_ + 1` or 💥 | `_`           | `_`           | Pops `a` if `a == 1`, else crashes the virtual machine.                                                                     |
| `halt`       | 0     | `_`         | `_`         | `_`      | `_ + 1`      | `_`           | `_`           | Solves the halting problem (if the instruction is reached). Indicates graceful shutdown of the VM.                          |

### Memory Access

| Instruction | Value | old OpStack | new OpStack | old `ramv` | new `ramv` | Description                                                                             |
|:------------|:------|:------------|:------------|:-----------|:-----------|:----------------------------------------------------------------------------------------|
| `read_mem`  | ?     | `_ p a`     | `_ p v`     | `v`        | `v`        | Reads value `v` from RAM at address `p` and overwrites the top of the OpStack with `v`. |
| `write_mem` | ?     | `_ p v`     | `_ p v`     | `_`        | `v`        | Writes OpStack's top-most value `v` to RAM at the address `p`.                          |

### Hashing

| Instruction      | Value | old OpStack       | new OpStack                     | Description                                                                                             |
|:-----------------|:------|:------------------|:--------------------------------|:--------------------------------------------------------------------------------------------------------|
| `hash`           | ?     | `_jihgfedcba`   | `_yxwvu00000`                 | Overwrites the stack's 10 top-most elements with their hash digest (length 6) and 6 zeros.              |
| `divine_sibling` | ?     | `_ i*****edcba` | e.g., `_ (i div 2)edcbazyxwv` | Helps traversing a Merkle tree during authentication path verification. See extended description below. |
| `assert_vector`  | ?     | `_`               | `_`                             | Assert equality of `st(i)` to `st(i+5)` for `0 <= i < 4`. Crashes the VM if any pair is unequal.        |

The instruction `hash` works as follows.
The stack's 10 top-most elements (`jihgfedcba`) are reversed and concatenated with six zeros, resulting in `abcdefghij000000`.
The permutation `xlix` is applied to `abcdefghij000000`, resulting in `αβγδεζηθικuvwxyz`.
The first five elements of this result, i.e., `αβγδε`, are reversed and written to the stack, overwriting `st5` through `st9`.
The top elements of the stack `st5` through `st9` are set to zero.
For example, the old stack was `_ jihgfedcba` and the new stack is `_ εδγβα 00000`.

The instruction `divine_sibling` works as follows.
The 11th element of the stack `i` is taken as the node index for a Merkle tree that is claimed to include data whose digest is the content of stack registers `st0` through `st4`, i.e., `edcba`.
The sibling digest of `edcba` is `zyxwv` and is read from the input interface of secret data.
The least-significant bit of `i` indicates whether `edcba` is the digest of a left leaf or a right leaf of the Merkle tree's base level.
Depending on this least significant bit of `i`, `divine_sibling` either
1. does not change registers `st0` through `st4` and moves `zyxwv` into registers `st5` through `st9`, or
2. moves `edcba` into registers `st5` through `st9` and moves `zyxwv` into registers `st0` through `st4`.
The 11th element of the operational stack is modified by shifting `i` by 1 bit to the right, i.e., dropping the least-significant bit.
In conjunction with instruction `hash` and `assert_vector`, the instruction `divine_sibling` allows to efficiently verify a Merkle authentication path.

### Arithmetic on Stack

| Instruction | Value | old OpStack     | new OpStack     | Description                                                                                                                                                           |
|:------------|:------|:----------------|:----------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `add`       | ?     | `_ b a`         | `_ c`           | Computes the sum (`c`) of the top two elements of the stack (`b` and `a`) over the field.                                                                             |
| `mul`       | ?     | `_ b a`         | `_ c`           | Computes the product (`c`) of the top two elements of the stack (`b` and `a`) over the field.                                                                         |
| `invert`    | ?     | `_ a`           | `_ b`           | Computes the multiplicative inverse (over the field) of the top of the stack. Crashes the VM if the top of the stack is 0.                                            |
| `split`     | ?     | `_ a`           | `_ lo hi`       | Decomposes the top of the stack into the lower 32 bits and the upper 32 bits.                                                                                         |
| `eq`        | ?     | `_ b a`         | `_ (a == b)`    | Tests the top two stack elements for equality.                                                                                                                        |
| `lt`        | ?     | `_ b a`         | `_ (a < b)`     | Tests if the top element on the stack is less than the one-from top element, assuming both are 32-bit integers.                                                       |
| `and`       | ?     | `_ b a`         | `_ (a and b)`   | Computes the bitwise-and of the top two stack elements, assuming both are 32-bit integers.                                                                            |
| `xor`       | ?     | `_ b a`         | `_ (a xor b)`   | Computes the bitwise-xor of the top two stack elements, assuming both are 32-bit integers.                                                                            |
| `reverse`   | ?     | `_ a`           | `_ b`           | Reverses the bit expansion of the top stack element, assuming it is a 32-bit integer.                                                                                 |
| `div`       | ?     | `_ d n`         | `_ q r`         | Computes division with remainder of the top two stack elements, assuming both arguments are unsigned 32-bit integers. The result satisfies `n == d·q + r`and `r < d`. |
| `xxadd`     | ?     | `_ z y x b c a` | `_ z y x w v u` | Adds the two extension field elements encoded by field elements `z y x` and `b c a`, overwriting the top-most extension field element with the result.                |
| `xxmul`     | ?     | `_ z y x b c a` | `_ z y x w v u` | Multiplies the two extension field elements encoded by field elements `z y x` and `b c a`, overwriting the top-most extension field element with the result.          |
| `xinvert`   | ?     | `_ z y x`       | `_ w v u`       | Inverts the extension field element encoded by field elements `z y x` in-place. Crashes the VM if the extension field element is 0.                                   |
| `xbmul`     | ?     | `_ z y x a`     | `_ w v u`       | Scalar multiplication of the extension field element encoded by field elements `z y x` with field element `a`. Overwrites `z y x` with the result.                    |

### Input/Output

| Instruction | Value | old OpStack | new OpStack | Description                                                             |
|:------------|:------|:------------|:------------|:------------------------------------------------------------------------|
| `read_io`   | ?     | `_`         | `_ a`       | Reads a B-Field element from standard input and pushes it to the stack. |
| `write_io`  | ?     | `_ a`       | `_`         | Pops `a` from the stack and writes it to standard output.               |
