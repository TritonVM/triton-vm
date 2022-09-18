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

| Instruction  | Value | old OpStack | new OpStack | old `ip` | new `ip`     | old JumpStack | new JumpStack | Description                                                                                                                 |
|:-------------|:------|:------------|:------------|:---------|:-------------|:--------------|:--------------|:----------------------------------------------------------------------------------------------------------------------------|
| `nop`        | ?     | `_`         | `_`         | `_`      | `_ + 1`      | `_`           | `_`           | Do nothing                                                                                                                  |
| `skiz`       | ?     | `_ a`       | `_`         | `_`      | `_ + s`      | `_`           | `_`           | Skip next instruction if `a` is zero. `s` ∈ {1, 2, 3} depends on `a` and whether or not next instruction takes an argument. |
| `call` + `d` | ?     | `_`         | `_`         | `o`      | `d`          | `_`           | `_ (o+2, d)`  | Push `(o+2,d)` to the jump stack, and jump to absolute address `d`                                                          |
| `return`     | ?     | `_`         | `_`         | `_`      | `o`          | `_ (o, d)`    | `_`           | Pop one pair off the jump stack and jump to that pair's return address (which is the first element).                        |
| `recurse`    | ?     | `_`         | `_`         | `_`      | `d`          | `_ (o, d)`    | `_ (o, d)`    | Peek at the top pair of the jump stack and jump to that pair's destination address (which is the second element).           |
| `assert`     | ?     | `_ a`       | `_`         | `_`      | `_ + 1` or 💥 | `_`           | `_`           | Pops `a` if `a == 1`, else crashes the virtual machine.                                                                     |
| `halt`       | 0     | `_`         | `_`         | `_`      | `_ + 1`      | `_`           | `_`           | Solves the halting problem (if the instruction is reached). Indicates graceful shutdown of the VM.                          |

## Memory Access

| Instruction | Value | old OpStack | new OpStack | old `ramv` | new `ramv` | Description                                                                             |
|:------------|:------|:------------|:------------|:-----------|:-----------|:----------------------------------------------------------------------------------------|
| `read_mem`  | ?     | `_ p a`     | `_ p v`     | `v`        | `v`        | Reads value `v` from RAM at address `p` and overwrites the top of the OpStack with `v`. |
| `write_mem` | ?     | `_ p v`     | `_ p v`     | `_`        | `v`        | Writes OpStack's top-most value `v` to RAM at the address `p`.                          |

## Hashing

| Instruction      | Value | old OpStack       | new OpStack                     | Description                                                                                             |
|:-----------------|:------|:------------------|:--------------------------------|:--------------------------------------------------------------------------------------------------------|
| `hash`           | ?     | `_lkjihgfedcba`   | `_000000zyxwvu`                 | Overwrites the stack's 12 top-most elements with their hash digest (length 6) and 6 zeros.              |
| `divine_sibling` | ?     | `_ i******fedcba` | e.g., `_ (i div 2)fedcbazyxwvu` | Helps traversing a Merkle tree during authentication path verification. See extended description below. |
| `assert_vector`  | ?     | `_`               | `_`                             | Assert equality of `st(i)` to `st(i+6)` for `0 <= i < 6`. Crashes the VM if any pair is unequal.        |

The instruction `hash` works as follows.
The stack's 12 top-most elements (`lkjihgfedcba`) are concatenated to four zeros, resulting in `0000lkjihgfedcba`.
The permutation `xlix` is applied to `0000lkjihgfedcba`, resulting in `zyxwvuκιθηζεδγβα`.
The first six elements of this result, i.e., `zyxwvu`, are written to the stack, overwriting `st0` through `st5`, i.e., `fedcba`.
The next six elements of the stack `st6` through `st11`, i.e,`lkjihg`, are set to zero.

The instruction `divine_sibling` works as follows.
The 13th element of the stack `i` is taken as the node index for a Merkle tree that is claimed to include data whose digest is the content of stack registers `st0` through `st5`, i.e., `fedcba`.
The sibling digest of `fedcba` is `zyxwvu` and is read from the input interface of secret data.
The least-significant bit of `i` indicates whether `fedcba` is the digest of a left leaf or a right leaf of the Merkle tree's base level.
Depending on this least-significant bit of `i`, `divine_sibling` either
1. does not change registers `st0` through `st5` and moves `zyxwvu` into registers `st6` through `st11`, or
2. moves `fedcba` into registers `st6` through `st11` and moves `zyxwvu` into registers `st0` through `st5`.
The 13th element of the operational stack is modified by shifting `i` by 1 bit to the right, i.e., dropping the least-significant bit.
In conjunction with instruction `hash` and `assert_vector`, the instruction `divine_sibling` allows to efficiently verify a Merkle authentication path.

## Arithmetic on Stack

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

## Input/Output

| Instruction | Value | old OpStack | new OpStack | Description                                                             |
|:------------|:------|:------------|:------------|:------------------------------------------------------------------------|
| `read_io`   | ?     | `_`         | `_ a`       | Reads a B-Field element from standard input and pushes it to the stack. |
| `write_io`  | ?     | `_ a`       | `_`         | Pops `a` from the stack and writes it to standard output.               |
