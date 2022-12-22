# Instructions

Most instructions are contained within a single, parameterless machine word.

Some instructions take a machine word as argument and are so considered double-word instructions.
They are recognized by the form “`instr` + `arg`”.

## Regarding Opcodes

An instruction's _[operation code](https://en.wikipedia.org/wiki/Opcode)_, or _opcode_, is the machine word uniquely identifying the instruction.
For reasons of efficient [arithmetization](arithmetization.md), certain properties of the instruction are encoded in the opcode.
Concretely:
- all double-word instructions have an odd opcode, _i.e._, the least significant bit is 1.
- all instructions shrinking the operational stack are 2 mod 4, _i.e._, the second-to-least significant bit is 1.

The former property is used by instruction [skiz](instruction-specific-transition-constraints.md#instruction-skiz).
The latter property helps guarantee that operational stack underflow cannot happen.
It is used by several instructions through instruction group [`stack_shrinks_and_top_3_unconstrained`](instruction-groups.md#group-stack_shrinks_and_top_3_unconstrained).

## OpStack Manipulation

| Instruction  | Opcode | old OpStack         | new OpStack           | Description                                                                      |
|:-------------|-------:|:--------------------|:----------------------|:---------------------------------------------------------------------------------|
| `pop`        |      2 | `_ a`               | `_`                   | Pops top element from stack.                                                     |
| `push` + `a` |      1 | `_`                 | `_ a`                 | Pushes `a` onto the stack.                                                       |
| `divine`     |      4 | `_`                 | `_ a`                 | Pushes a non-deterministic element `a` to the stack. Interface for secret input. |
| `dup`  + `i` |      5 | e.g., `_ e d c b a` | e.g., `_ e d c b a d` | Duplicates the element `i` positions away from the top, assuming `0 <= i < 16`.  |
| `swap` + `i` |      9 | e.g., `_ e d c b a` | e.g., `_ e a c b d`   | Swaps the `i`th stack element with the top of the stack, assuming `0 < i < 16`.  |

Instruction `divine` (together with [`divine_sibling`](#hashing)) make TritonVM a virtual machine that can execute non-deterministic programs.
As programs go, this concept is somewhat unusual and benefits from additional explanation.
The name of the instruction is the verb (not the adjective) meaning “to discover by intuition or insight.”

From the perspective of the program, the instruction `divine` makes some element `a` magically appear on the stack.
It is not at all specified what `a` is, but generally speaking, `a` has to be exactly correct, else execution fails.
Hence, from the perspective of the program, it just non-deterministically guesses the correct value of `a` in a moment of divine clarity.

Looking at the entire system, consisting of the VM, the program, and all inputs – both public and secret – execution _is_ deterministic:
the value `a` was supplied as a secret input.

## Control Flow

| Instruction  | Opcode | old OpStack | new OpStack | old `ip` | new `ip`     | old JumpStack | new JumpStack | Description                                                                                                                 |
|:-------------|-------:|:------------|:------------|:---------|:-------------|:--------------|:--------------|:----------------------------------------------------------------------------------------------------------------------------|
| `nop`        |      8 | `_`         | `_`         | `_`      | `_ + 1`      | `_`           | `_`           | Do nothing                                                                                                                  |
| `skiz`       |      6 | `_ a`       | `_`         | `_`      | `_ + s`      | `_`           | `_`           | Skip next instruction if `a` is zero. `s` ∈ {1, 2, 3} depends on `a` and whether or not next instruction takes an argument. |
| `call` + `d` |     13 | `_`         | `_`         | `o`      | `d`          | `_`           | `_ (o+2, d)`  | Push `(o+2,d)` to the jump stack, and jump to absolute address `d`                                                          |
| `return`     |     12 | `_`         | `_`         | `_`      | `o`          | `_ (o, d)`    | `_`           | Pop one pair off the jump stack and jump to that pair's return address (which is the first element).                        |
| `recurse`    |     16 | `_`         | `_`         | `_`      | `d`          | `_ (o, d)`    | `_ (o, d)`    | Peek at the top pair of the jump stack and jump to that pair's destination address (which is the second element).           |
| `assert`     |     10 | `_ a`       | `_`         | `_`      | `_ + 1` or 💥 | `_`           | `_`           | Pops `a` if `a == 1`, else crashes the virtual machine.                                                                     |
| `halt`       |      0 | `_`         | `_`         | `_`      | `_ + 1`      | `_`           | `_`           | Solves the halting problem (if the instruction is reached). Indicates graceful shutdown of the VM.                          |

## Memory Access

| Instruction | Opcode | old OpStack | new OpStack | old `ramv` | new `ramv` | Description                                                                             |
|:------------|-------:|:------------|:------------|:-----------|:-----------|:----------------------------------------------------------------------------------------|
| `read_mem`  |     20 | `_ p a`     | `_ p v`     | `v`        | `v`        | Reads value `v` from RAM at address `p` and overwrites the top of the OpStack with `v`. |
| `write_mem` |     24 | `_ p v`     | `_ p v`     | `_`        | `v`        | Writes OpStack's top-most value `v` to RAM at the address `p`.                          |

## Hashing

| Instruction      | Opcode | old OpStack     | new OpStack                   | Description                                                                                             |
|:-----------------|-------:|:----------------|:------------------------------|:--------------------------------------------------------------------------------------------------------|
| `hash`           |     28 | `_jihgfedcba`   | `_yxwvu00000`                 | Overwrites the stack's 10 top-most elements with their hash digest (length 5) and 5 zeros.              |
| `divine_sibling` |     32 | `_ i*****edcba` | e.g., `_ (i div 2)edcbazyxwv` | Helps traversing a Merkle tree during authentication path verification. See extended description below. |
| `assert_vector`  |     36 | `_`             | `_`                           | Assert equality of `st(i)` to `st(i+5)` for `0 <= i < 4`. Crashes the VM if any pair is unequal.        |

The instruction `hash` works as follows.
The stack's 10 top-most elements (`jihgfedcba`) are reversed and concatenated with six zeros, resulting in `abcdefghij000000`.
The permutation `xlix` is applied to `abcdefghij000000`, resulting in `αβγδεζηθικuvwxyz`.
The first five elements of this result, i.e., `αβγδε`, are reversed and written to the stack, overwriting `st5` through `st9`.
The top elements of the stack `st0` through `st4` are set to zero.
For example, the old stack was `_ jihgfedcba` and the new stack is `_ εδγβα 00000`.

The instruction `divine_sibling` works as follows.
The 11th element of the stack `i` is taken as the node index for a Merkle tree that is claimed to include data whose digest is the content of stack registers `st0` through `st4`, i.e., `edcba`.
The sibling digest of `edcba` is `zyxwv` and is read from the input interface of secret data.
The least-significant bit of `i` indicates whether `edcba` is the digest of a left leaf or a right leaf of the Merkle tree's base level.
Depending on this least significant bit of `i`, `divine_sibling` either
1. (`i` = 0 mod 2, _i.e._, current node is left sibling) does not change registers `st0` through `st4` and moves `zyxwv` into registers `st5` through `st9`, or
2. (`i` = 1 mod 2, _i.e._, current node is right sibling) moves `edcba` into registers `st5` through `st9` and moves `zyxwv` into registers `st0` through `st4`.

The 11th element of the operational stack `i` is shifted by 1 bit to the right, _i.e._, the least-significant bit is dropped.
In conjunction with instruction `hash` and `assert_vector`, the instruction `divine_sibling` allows to efficiently verify a Merkle authentication path.

## Arithmetic on Stack

| Instruction | Opcode | old OpStack     | new OpStack          | Description                                                                                                                                                                     |
|:------------|-------:|:----------------|:---------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `add`       |     14 | `_ b a`         | `_ c`                | Computes the sum (`c`) of the top two elements of the stack (`b` and `a`) over the field.                                                                                       |
| `mul`       |     18 | `_ b a`         | `_ c`                | Computes the product (`c`) of the top two elements of the stack (`b` and `a`) over the field.                                                                                   |
| `invert`    |     40 | `_ a`           | `_ b`                | Computes the multiplicative inverse (over the field) of the top of the stack. Crashes the VM if the top of the stack is 0.                                                      |
| `split`     |     44 | `_ a`           | `_ lo hi`            | Decomposes the top of the stack into the lower 32 bits and the upper 32 bits. Use with care, preferably through [pseudo instructions](pseudo-instructions.md).                  |
| `eq`        |     22 | `_ b a`         | `_ (a == b)`         | Tests the top two stack elements for equality.                                                                                                                                  |
| `lsb`       |     48 | `_ a`           | `_ (a >> 1) (a % 2)` | Bit-shifts `a` to the right by 1 bit and pushes the least significant bit of `a` to the stack. Use with care, preferably through [pseudo instructions](pseudo-instructions.md). |
| `xxadd`     |     52 | `_ z y x b c a` | `_ z y x w v u`      | Adds the two extension field elements encoded by field elements `z y x` and `b c a`, overwriting the top-most extension field element with the result.                          |
| `xxmul`     |     56 | `_ z y x b c a` | `_ z y x w v u`      | Multiplies the two extension field elements encoded by field elements `z y x` and `b c a`, overwriting the top-most extension field element with the result.                    |
| `xinvert`   |     60 | `_ z y x`       | `_ w v u`            | Inverts the extension field element encoded by field elements `z y x` in-place. Crashes the VM if the extension field element is 0.                                             |
| `xbmul`     |     26 | `_ z y x a`     | `_ w v u`            | Scalar multiplication of the extension field element encoded by field elements `z y x` with field element `a`. Overwrites `z y x` with the result.                              |

## Input/Output

| Instruction | Opcode | old OpStack | new OpStack | Description                                                             |
|:------------|-------:|:------------|:------------|:------------------------------------------------------------------------|
| `read_io`   |     64 | `_`         | `_ a`       | Reads a B-Field element from standard input and pushes it to the stack. |
| `write_io`  |     30 | `_ a`       | `_`         | Pops `a` from the stack and writes it to standard output.               |
