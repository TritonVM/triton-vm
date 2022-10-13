# Instructions

Most instructions are contained within a single, parameterless machine word.

Some instructions take a machine word as argument and are so considered double-word instructions.
They are recognized by the form "`instr` + `arg`".

## OpStack Manipulation

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

## Control Flow

| Instruction          | Value | old OpStack | new OpStack | old `ip` | new `ip`     | old JumpStack | new JumpStack       | Description                                                                                                                 |
|:---------------------|:------|:------------|:------------|:---------|:-------------|:--------------|:--------------------|:----------------------------------------------------------------------------------------------------------------------------|
| `nop`                | ?     | `_`         | `_`         | `ip`     | `ip+1`       | `_`           | `_`                 | Do nothing                                                                                                                  |
| `skiz`               | ?     | `_ a`       | `_`         | `ip`     | `ip+s`       | `_`           | `_`                 | Skip next instruction if `a` is zero. `s` ∈ {1, 2, 3} depends on `a` and whether or not next instruction takes an argument. |
| `if_then_call` + `d` | ?     | `_ a`       | `_ a`       | `o`      | `o+2` or `d` | `_`           | `_` or `_ (o+4, d)` | If `a` ≠ 0, push `(o+4,d)` to the jump stack, and jump to absolute address `d`. Else, do nothing.                           |
| `call` + `d`         | ?     | `_`         | `_`         | `o`      | `d`          | `_`           | `_ (o+2, d)`        | Push `(o+2,d)` to the jump stack, and jump to absolute address `d`                                                          |
| `return`             | ?     | `_`         | `_`         | `_`      | `o`          | `_ (o, d)`    | `_`                 | Pop one pair off the jump stack and jump to that pair's return address (which is the first element).                        |
| `recurse`            | ?     | `_`         | `_`         | `_`      | `d`          | `_ (o, d)`    | `_ (o, d)`          | Peek at the top pair of the jump stack and jump to that pair's destination address (which is the second element).           |
| `assert`             | ?     | `_ a`       | `_`         | `ip`     | `ip+1` or 💥  | `_`           | `_`                 | Pops `a` if `a == 1`, else crashes the virtual machine.                                                                     |
| `halt`               | 0     | `_`         | `_`         | `ip`     | `ip+1`       | `_`           | `_`                 | Solves the halting problem (if the instruction is reached). Indicates graceful shutdown of the VM.                          |

For instruction `if_then_call`, note the difference between the potential offsets added to instruction pointer `ip` and jump origin `o`.
Specifically, if the top of the stack `a` is 0, the instruction pointer `ip` is incremented by 2, skipping the instruction's argument `d`.
However, if the top of the stack `a` is not 0, the return address put on the jump stack is `ip+4`.
This skips not only the instruction's argument `d`, but also the next two instructions when `return`ing.
Together, this mechanic allows following instruction `if_then_call` + `d` with instruction `call` + `e` to serve as the “else” branch.
If no “else” branch is desired, instruction `if_then_else` + `d` can be followed with two instructions `nop`.
Of course, instruction `if_then_call` can be followed by any instruction.
The appropriate care should be taken to `return` only to where it makes sense.

## Memory Access

| Instruction | Value | old OpStack | new OpStack | old `ramv` | new `ramv` | Description                                                                             |
|:------------|:------|:------------|:------------|:-----------|:-----------|:----------------------------------------------------------------------------------------|
| `read_mem`  | ?     | `_ p a`     | `_ p v`     | `v`        | `v`        | Reads value `v` from RAM at address `p` and overwrites the top of the OpStack with `v`. |
| `write_mem` | ?     | `_ p v`     | `_ p v`     | `_`        | `v`        | Writes OpStack's top-most value `v` to RAM at the address `p`.                          |

## Hashing

| Instruction      | Value | old OpStack     | new OpStack                   | Description                                                                                             |
|:-----------------|:------|:----------------|:------------------------------|:--------------------------------------------------------------------------------------------------------|
| `hash`           | ?     | `_jihgfedcba`   | `_yxwvu00000`                 | Overwrites the stack's 10 top-most elements with their hash digest (length 6) and 6 zeros.              |
| `divine_sibling` | ?     | `_ i*****edcba` | e.g., `_ (i div 2)edcbazyxwv` | Helps traversing a Merkle tree during authentication path verification. See extended description below. |
| `assert_vector`  | ?     | `_`             | `_`                           | Assert equality of `st(i)` to `st(i+5)` for `0 <= i < 4`. Crashes the VM if any pair is unequal.        |

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
1. does not change registers `st0` through `st4` and moves `zyxwv` into registers `st5` through `st9`, or
2. moves `edcba` into registers `st5` through `st9` and moves `zyxwv` into registers `st0` through `st4`.
The 11th element of the operational stack is modified by shifting `i` by 1 bit to the right, i.e., dropping the least-significant bit.
In conjunction with instruction `hash` and `assert_vector`, the instruction `divine_sibling` allows to efficiently verify a Merkle authentication path.

## Arithmetic on Stack

| Instruction | Value | old OpStack     | new OpStack          | Description                                                                                                                                                                     |
|:------------|:------|:----------------|:---------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `add`       | ?     | `_ b a`         | `_ c`                | Computes the sum (`c`) of the top two elements of the stack (`b` and `a`) over the field.                                                                                       |
| `mul`       | ?     | `_ b a`         | `_ c`                | Computes the product (`c`) of the top two elements of the stack (`b` and `a`) over the field.                                                                                   |
| `invert`    | ?     | `_ a`           | `_ b`                | Computes the multiplicative inverse (over the field) of the top of the stack. Crashes the VM if the top of the stack is 0.                                                      |
| `split`     | ?     | `_ a`           | `_ lo hi`            | Decomposes the top of the stack into the lower 32 bits and the upper 32 bits. Use with care, preferably through [pseudo instructions](pseudo-instructions.md).                  |
| `eq`        | ?     | `_ b a`         | `_ (a == b)`         | Tests the top two stack elements for equality.                                                                                                                                  |
| `lsb`       | ?     | `_ a`           | `_ (a >> 1) (a % 2)` | Bit-shifts `a` to the right by 1 bit and pushes the least significant bit of `a` to the stack. Use with care, preferably through [pseudo instructions](pseudo-instructions.md). |
| `xxadd`     | ?     | `_ z y x b c a` | `_ z y x w v u`      | Adds the two extension field elements encoded by field elements `z y x` and `b c a`, overwriting the top-most extension field element with the result.                          |
| `xxmul`     | ?     | `_ z y x b c a` | `_ z y x w v u`      | Multiplies the two extension field elements encoded by field elements `z y x` and `b c a`, overwriting the top-most extension field element with the result.                    |
| `xinvert`   | ?     | `_ z y x`       | `_ w v u`            | Inverts the extension field element encoded by field elements `z y x` in-place. Crashes the VM if the extension field element is 0.                                             |
| `xbmul`     | ?     | `_ z y x a`     | `_ w v u`            | Scalar multiplication of the extension field element encoded by field elements `z y x` with field element `a`. Overwrites `z y x` with the result.                              |

## Input/Output

| Instruction | Value | old OpStack | new OpStack | Description                                                             |
|:------------|:------|:------------|:------------|:------------------------------------------------------------------------|
| `read_io`   | ?     | `_`         | `_ a`       | Reads a B-Field element from standard input and pushes it to the stack. |
| `write_io`  | ?     | `_ a`       | `_`         | Pops `a` from the stack and writes it to standard output.               |
