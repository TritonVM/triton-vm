# Instructions

Most instructions are contained within a single, parameterless machine word.

Some instructions take a machine word as argument and are so considered double-word instructions.
They are recognized by the form “`instr` + `arg`”.

## Regarding Opcodes

An instruction's _[operation code](https://en.wikipedia.org/wiki/Opcode)_, or _opcode_, is the machine word uniquely identifying the instruction.
For reasons of efficient [arithmetization](arithmetization.md), certain properties of the instruction are encoded in the opcode.
Concretely, interpreting the field element in standard representation:
- for all double-word instructions, the least significant bit is 1.
- for all instructions shrinking the operational stack, the second-to-least significant bit is 1.
- for all [u32 instructions ](u32-table.md), the third-to-least significant bit is 1.

The first property is used by instruction [skiz](instruction-specific-transition-constraints.md#instruction-skiz).
The second property helps with proving consistency of the [Op Stack](data-structures.md#operational-stack).
The third property allows efficient arithmetization of the running product for the Permutation Argument between [Processor Table](processor-table.md) and [U32 Table](u32-table.md).

## Op Stack Manipulation

| Instruction     | Opcode | old op stack        | new op stack          | Description                                                                                     |
|:----------------|-------:|:--------------------|:----------------------|:------------------------------------------------------------------------------------------------|
| `pop` + `n`     |      3 | e.g., `_ c b a`     | e.g., `_`             | Pops the `n` top elements from the stack. 1 ⩽ `n` ⩽ 5                                           |
| `push` + `a`    |      1 | `_`                 | `_ a`                 | Pushes `a` onto the stack.                                                                      |
| `divine`  + `n` |      9 | e.g., `_`           | e.g., `_ b a`         | Pushes `n` non-deterministic elements `a` to the stack. Interface for secret input. 1 ⩽ `n` ⩽ 5 |
| `dup`  + `i`    |     17 | e.g., `_ e d c b a` | e.g., `_ e d c b a d` | Duplicates the element `i` positions away from the top. 0 ⩽ `i` < 16                            |
| `swap` + `i`    |     25 | e.g., `_ e d c b a` | e.g., `_ e a c b d`   | Swaps the `i`th stack element with the top of the stack. 0 ⩽ `i` < 16                           |

Instruction `divine n` (together with [`merkle_step`](#many-in-one)) make Triton a virtual machine that can execute non-deterministic programs.
As programs go, this concept is somewhat unusual and benefits from additional explanation.
The name of the instruction is the verb (not the adjective) meaning “to discover by intuition or insight.”

From the perspective of the program, the instruction `divine n` makes some `n` elements magically appear on the stack.
It is not at all specified what those elements are, but generally speaking, they have to be exactly correct, else execution fails.
Hence, from the perspective of the program, it just non-deterministically guesses the correct values in a moment of divine clarity.

Looking at the entire system, consisting of the VM, the program, and all inputs – both public and secret – execution _is_ deterministic:
the divined values were supplied as and are read from secret input.

## Control Flow

| Instruction         | Opcode | old op stack  | new op stack  | old `ip` | new `ip`   | old jump stack | new jump stack    | Description                                                                                                              |
|:--------------------|-------:|:--------------|:--------------|:---------|:-----------|:---------------|:------------------|:-------------------------------------------------------------------------------------------------------------------------|
| `halt`              |      0 | `_`           | `_`           | `ip`     | `ip+1`     | `_`            | `_`               | Solves the halting problem (if the instruction is reached). Indicates graceful shutdown of the VM.                       |
| `nop`               |      8 | `_`           | `_`           | `ip`     | `ip+1`     | `_`            | `_`               | Do nothing                                                                                                               |
| `skiz`              |      2 | `_ a`         | `_`           | `ip`     | `ip+s`     | `_`            | `_`               | Skip next instruction if `a` is zero. `s` ∈ {1, 2, 3} depends on `a` and whether the next instruction takes an argument. |
| `call` + `d`        |     33 | `_`           | `_`           | `ip`     | `d`        | `_`            | `_ (ip+2, d)`     | Push `(ip+2,d)` to the jump stack, and jump to absolute address `d`                                                      |
| `return`            |     16 | `_`           | `_`           | `ip`     | `o`        | `_ (o, d)`     | `_`               | Pop one pair off the jump stack and jump to that pair's return address (which is the first element).                     |
| `recurse`           |     24 | `_`           | `_`           | `ip`     | `d`        | `_ (o, d)`     | `_ (o, d)`        | Peek at the top pair of the jump stack and jump to that pair's destination address (which is the second element).        |
| `recurse_or_return` |     32 | `_ b a .....` | `_ b a .....` | `ip`     | `d` or `o` | `_ (o, d)`     | `_ (o, d)` or `_` | Like `recurse` if `st5 = a != b = st6`, like `return` if `a == b`. See also extended description below.                  |
| `assert`            |     10 | `_ a`         | `_`           | `ip`     | `ip+1`     | `_`            | `_`               | Pops `a` if `a == 1`, else crashes the virtual machine.                                                                  |

The instructions `return`, `recurse`, and `recurse_or_return` require a non-empty jump stack.
Should the jump stack be empty, executing any of these instruction causes Triton VM to crash.

Instruction `recurse_or_return` behaves – surprise! – either like instruction `recurse` or like instruction `return`.
The (deterministic) decision which behavior to exhibit is made at runtime and depends on stack elements `st5` and `st6`.
If `st5 != st6`, then `recurse_or_return` acts like instruction `recurse`, else like `return`.
The instruction is designed to facilitate loops using pointer equality as termination condition and to play nicely with instruction `merkle_step`.

## Memory Access

| Instruction       | Opcode | old op stack         | new op stack           | old RAM             | new RAM             | Description                                                                                                                                  |
|:------------------|-------:|:---------------------|:-----------------------|:--------------------|:--------------------|:---------------------------------------------------------------------------------------------------------------------------------------------|
| `read_mem` + `n`  |     41 | e.g., `_ p+2`        | e.g., `_ v2 v1 v0 p-1` | [p: v0, p+1, v1, …] | [p: v0, p+1, v1, …] | Reads consecutive values `vi` from RAM at address `p` and puts them onto the op stack. Decrements RAM pointer (`st0`) by `n`. 1 ⩽ `n` ⩽ 5    |
| `write_mem` + `n` |     11 | e.g., `_ v2 v1 v0 p` | e.g., `_ p+3`          | []                  | [p: v0, p+1, v1, …] | Writes op stack's `n` top-most values `vi` to RAM at the address `p+i`, popping the `vi`. Increments RAM pointer (`st0`) by `n`. 1 ⩽ `n` ⩽ 5 |

For the benefit of clarity, the effect of every possible argument is given below.

| instruction   | old op stack    | new op stack      | old RAM                                | new RAM                                |
|:--------------|:----------------|:------------------|:---------------------------------------|:---------------------------------------|
| `read_mem 1`  | `_ p`           | `_ a p-1`         | [p: a]                                 | [p: a]                                 |
| `read_mem 2`  | `_ p+1`         | `_ b a p-1`       | [p: a, p+1: b]                         | [p: a, p+1: b]                         |
| `read_mem 3`  | `_ p+2`         | `_ c b a p-1`     | [p: a, p+1: b, p+2: c]                 | [p: a, p+1: b, p+2: c]                 |
| `read_mem 4`  | `_ p+3`         | `_ d c b a p-1`   | [p: a, p+1: b, p+2: c, p+3: d]         | [p: a, p+1: b, p+2: c, p+3: d]         |
| `read_mem 5`  | `_ p+4`         | `_ e d c b a p-1` | [p: a, p+1: b, p+2: c, p+3: d, p+4: e] | [p: a, p+1: b, p+2: c, p+3: d, p+4: e] |
| `write_mem 1` | `_ a p`         | `_ p+1`           | []                                     | [p: a]                                 |
| `write_mem 2` | `_ b a p`       | `_ p+2`           | []                                     | [p: a, p+1: b]                         |
| `write_mem 3` | `_ c b a p`     | `_ p+3`           | []                                     | [p: a, p+1: b, p+2: c]                 |
| `write_mem 4` | `_ d c b a p`   | `_ p+4`           | []                                     | [p: a, p+1: b, p+2: c, p+3: d]         |
| `write_mem 5` | `_ e d c b a p` | `_ p+5`           | []                                     | [p: a, p+1: b, p+2: c, p+3: d, p+4: e] |

## Hashing

| Instruction         | Opcode | old op stack    | new op stack    | Description                                                                                                                                                         |
|:--------------------|-------:|:----------------|:----------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `hash`              |     18 | `_ jihgfedcba`  | `_ yxwvu`       | Hashes the stack's 10 top-most elements and puts their digest onto the stack, shrinking the stack by 5.                                                             |
| `assert_vector`     |     26 | `_ edcba edcba` | `_ edcba`       | Assert equality of `st(i)` to `st(i+5)` for `0 <= i < 4`. Crashes the VM if any pair is unequal. Pops the 5 top-most elements.                                      |
| `sponge_init`       |     40 | `_`             | `_`             | Initializes (resets) the Sponge's state. Must be the first Sponge instruction executed.                                                                             |
| `sponge_absorb`     |     34 | `_ jihgfedcba`  | `_`             | Absorbs the stack's ten top-most elements into the Sponge state.                                                                                                    |
| `sponge_absorb_mem` |     48 | `_ dcba p`      | `_ hgfe (p+10)` | Absorbs the ten RAM elements at addresses `p`, `p+1`, … into the Sponge state. Overwrites stack elements `st1` through `st4` with the first four absorbed elements. |
| `sponge_squeeze`    |     56 | `_`             | `_ zyxwvutsrq`  | Squeezes the Sponge and pushes the 10 squeezed elements onto the stack.                                                                                             |

The instruction `hash` works as follows.
The stack's 10 top-most elements (`jihgfedcba`) are popped from the stack, reversed, and concatenated with six zeros, resulting in `abcdefghij000000`.
The Tip5 permutation is applied to `abcdefghij000000`, resulting in `αβγδεζηθικuvwxyz`.
The first five elements of this result, i.e., `αβγδε`, are reversed and pushed to the stack.
For example, the old stack was `_ jihgfedcba` and the new stack is `_ εδγβα`.

The instructions `sponge_init`, `sponge_absorb`, `sponge_absorb_mem`, and `sponge_squeeze` are the interface for using the Tip5 permutation in a [Sponge](https://keccak.team/sponge_duplex.html) construction.
The capacity is never accessible to the program that's being executed by Triton VM.
At any given time, at most one Sponge state exists.
Only instruction `sponge_init` resets the state of the Sponge, and only the three Sponge instructions influence the Sponge's state.
Notably, executing instruction `hash` does not modify the Sponge's state.
When using the Sponge instructions, it is the programmer's responsibility to take care of proper input padding:
Triton VM cannot know the number of elements that will be absorbed.

## Base Field Arithmetic on Stack

| Instruction  | Opcode | old op stack | new op stack | Description                                                                                                                |
|:-------------|-------:|:-------------|:-------------|:---------------------------------------------------------------------------------------------------------------------------|
| `add`        |     42 | `_ b a`      | `_ c`        | Computes the sum (`c`) of the top two elements of the stack (`b` and `a`) over the field.                                  |
| `addi` + `a` |     49 | `_ b`        | `_ c`        | Computes the sum (`c`) of the top element of the stack (`b`) and the immediate argument (`a`).                             |
| `mul`        |     50 | `_ b a`      | `_ c`        | Computes the product (`c`) of the top two elements of the stack (`b` and `a`) over the field.                              |
| `invert`     |     64 | `_ a`        | `_ b`        | Computes the multiplicative inverse (over the field) of the top of the stack. Crashes the VM if the top of the stack is 0. |
| `eq`         |     58 | `_ b a`      | `_ (a == b)` | Tests the top two stack elements for equality.                                                                             |

## Bitwise Arithmetic on Stack

| Instruction   | Opcode | old op stack | new op stack  | Description                                                                                                                                                                |
|:--------------|-------:|:-------------|:--------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `split`       |      4 | `_ a`        | `_ hi lo`     | Decomposes the top of the stack into the lower 32 bits and the upper 32 bits.                                                                                              |
| `lt`          |      6 | `_ b a`      | `_ a<b`       | “Less than” of the stack's two top-most elements. Crashes the VM if `a` or `b` is not u32.                                                                                 |
| `and`         |     14 | `_ b a`      | `_ a&b`       | Bitwise and of the stack's two top-most elements. Crashes the VM if `a` or `b` is not u32.                                                                                 |
| `xor`         |     22 | `_ b a`      | `_ a^b`       | Bitwise exclusive or of the stack's two top-most elements. Crashes the VM if `a` or `b` is not u32.                                                                        |
| `log_2_floor` |     12 | `_ a`        | `_ ⌊log₂(a)⌋` | The number of bits in `a` minus 1, _i.e._, $\lfloor\log_2\texttt{a}\rfloor$. Crashes the VM if `a` is 0 or not u32.                                                        |
| `pow`         |     30 | `_ e b`      | `_ b**e`      | The top of the stack to the power of the stack's runner up. Crashes the VM if exponent `e` is not u32.                                                                     |
| `div_mod`     |     20 | `_ d n`      | `_ q r`       | Division with remainder of numerator `n` by denominator `d`. Guarantees the properties `n == q·d + r` and `r < d`. Crashes the VM if `n` or `d` is not u32 or if `d` is 0. |
| `pop_count`   |     28 | `_ a`        | `_ w`         | Computes the [hamming weight](https://en.wikipedia.org/wiki/Hamming_weight) or “population count” of `a`. Crashes the VM if `a` is not u32.                                |

## Extension Field Arithmetic on Stack

| Instruction | Opcode | old op stack    | new op stack | Description                                                                                                                                        |
|:------------|-------:|:----------------|:-------------|:---------------------------------------------------------------------------------------------------------------------------------------------------|
| `xx_add`    |     66 | `_ z y x b c a` | `_ w v u`    | Adds the two extension field elements encoded by field elements `z y x` and `b c a`.                                                               |
| `xx_mul`    |     74 | `_ z y x b c a` | `_ w v u`    | Multiplies the two extension field elements encoded by field elements `z y x` and `b c a`.                                                         |
| `x_invert`  |     72 | `_ z y x`       | `_ w v u`    | Inverts the extension field element encoded by field elements `z y x` in-place. Crashes the VM if the extension field element is 0.                |
| `xb_mul`    |     82 | `_ z y x a`     | `_ w v u`    | Scalar multiplication of the extension field element encoded by field elements `z y x` with field element `a`. Overwrites `z y x` with the result. |

## Input/Output

| Instruction      | Opcode | old op stack    | new op stack    | Description                                                                              |
|:-----------------|-------:|:----------------|:----------------|:-----------------------------------------------------------------------------------------|
| `read_io` + `n`  |     49 | e.g., `_`       | e.g., `_ c b a` | Reads `n` B-Field elements from standard input and pushes them to the stack. 1 ⩽ `n` ⩽ 5 |
| `write_io` + `n` |     19 | e.g., `_ c b a` | e.g., `_`       | Pops `n` elements from the stack and writes them to standard output. 1 ⩽ `n` ⩽ 5         |

## Many-In-One

| Instruction   | Opcode | old op stack    | new op stack                 | Description                                                                                                                                                                                                                                                                                                                                                                                       |
|:--------------|:-------|:----------------|:-----------------------------|:--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `merkle_step` | 36     | `_ i edcba`     | `_ (i div 2) zyxwv`          | Helps traversing a Merkle tree during authentication path verification. Crashes the VM if `i` is not u32. See extended description below.                                                                                                                                                                                                                                                         |
| `xx_dot_step` | 80     | `_ z y x *b *a` | `_ z+p2 y+p1 x+p0 *b+3 *a+3` | Reads two extension field elements from RAM located at the addresses corresponding to the two top stack elements, multiplies the extension field elements, and adds the product `(p0, p1, p2)` to an accumulator located on stack immediately below the two pointers. Also, increase the pointers by the number of words read.                                                                    |
| `xb_dot_step` | 88     | `_ z y x *b *a` | `_ z+p2 y+p1 x+p0 *b+3 *a+1` | Reads one base field element from RAM located at the addresses corresponding to the top of the stack, one extension field element from RAM located at the address of the second stack element, multiplies the field elements, and adds the product `(p0, p1, p2)` to an accumulator located on stack immediately below the two pointers. Also, increase the pointers by the number of words read. |

The instruction `merkle_step` works as follows.
The 6th element of the stack `i` is taken as the node index for a Merkle tree that is claimed to include data whose digest is the content of stack registers `st4` through `st0`, i.e., `edcba`.
The sibling digest of `edcba` is `εδγβα` and is read from the input interface of secret data.
The least-significant bit of `i` indicates whether `edcba` is the digest of a left leaf or a right leaf of the Merkle tree's current level.
Depending on this least significant bit of `i`, `merkle_step` either
1. (`i` = 0 mod 2) interprets `edcba` as the left digest, `εδγβα` as the right digest, or
2. (`i` = 1 mod 2) interprets `εδγβα` as the left digest, `edcba` as the right digest.

In either case,
1. the left and right digests are hashed, and the resulting digest `zyxwv` replaces the top of the stack, and
1. 6th register `i` is shifted by 1 bit to the right, _i.e._, the least-significant bit is dropped.

In conjunction with instruction `assert_vector`, the instruction `merkle_step` allows to efficiently verify a Merkle authentication path.
