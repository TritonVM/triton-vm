# Instructions

Triton VM’s instructions are (loosely and informally) grouped into the following categories:
- [Stack Manipulation](#stack-manipulation)
- [Control Flow](#control-flow)
- [Memory Access](#memory-access)
- [Hashing](#hashing)
- [Base Field Arithmetic](#base-field-arithmetic)
- [Bitwise Arithmetic](#bitwise-arithmetic)
- [Extension Field Arithmetic](#extension-field-arithmetic)
- [Input/Output](#inputoutput)
- [Many-In-One](#many-in-one)


The following table is a summary of all instructions.
For more details, read on below.

| Instruction                               | Description                                          |
|:------------------------------------------|:-----------------------------------------------------|
| [`push` + `a`](#push--a)                  | Push `a` onto the stack.                             |
| [`pop` + `n`](#pop--n)                    | Pop the `n` top elements from the stack.             |
| [`divine` + `n`](#divine--n)              | Push `n` non-deterministic elements to the stack.    |
| [`pick` + `i`](#pick--i)                  | Move stack element `i` to the top of the stack.      |
| [`place` + `i`](#place--i)                | Move the top of the stack to the position `i`.       |
| [`dup` + `i`](#dup--i)                    | Duplicate stack element `i` onto the stack.          |
| [`swap` + `i`](#swap--i)                  | Swap stack element `i` with the top of the stack.    |
| [`halt`](#halt)                           | Indicate graceful shutdown of the VM.                |
| [`nop`](#nop)                             | Do nothing.                                          |
| [`skiz`](#skiz)                           | Conditionally skip the next instruction.             |
| [`call` + `d`](#call--d)                  | Continue execution at address `d`.                   |
| [`return`](#return)                       | Return to the last `call`-site.                      |
| [`recurse`](#recurse)                     | Continue execution at the location last `call`ed.    |
| [`recurse_or_return`](#recurse_or_return) | Either `recurse` or `return`.                        |
| [`assert`](#assert)                       | Assert that the top of the stack is 1.               |
| [`read_mem + n`](#read_mem--n)            | Read `n` elements from RAM.                          |
| [`write_mem + n`](#write_mem--n)          | Write `n` elements to RAM.                           |
| [`hash`](#hash)                           | Hash the top of the stack.                           |
| [`assert_vector`](#assert_vector)         | Assert equivalence of the two top quintuples.        |
| [`sponge_init`](#sponge_init)             | Initialize the Sponge state.                         |
| [`sponge_absorb`](#sponge_absorb)         | Absorb the top of the stack into the Sponge state.   |
| [`sponge_absorb_mem`](#sponge_absorb_mem) | Absorb from RAM into the Sponge state.               |
| [`sponge_squeeze`](#sponge_squeeze)       | Squeeze the Sponge state onto the stack.             |
| [`add`](#add)                             | Add two base field elements.                         |
| [`addi` + `a`](#addi--a)                  | Add `a` to the top of the stack.                     |
| [`mul`](#mul)                             | Multiply two base field elements.                    |
| [`invert`](#invert)                       | Base-field reciprocal of the top of the stack.       |
| [`eq`](#eq)                               | Compare the top two stack elements for equality.     |
| [`split`](#split)                         | Split the top of the stack into 32-bit words.        |
| [`lt`](#lt)                               | Compare two elements for “less than”.                |
| [`and`](#and)                             | Bitwise “and”.                                       |
| [`xor`](#xor)                             | Bitwise “xor”.                                       |
| [`log_2_floor`](#log_2_floor)             | The log₂ of the top of the stack, rounded down.      |
| [`pow`](#pow)                             | The top of the stack to the power of its runner-up.  |
| [`div_mod`](#div_mod)                     | Division with remainder.                             |
| [`pop_count`](#pop_count)                 | The hamming weight of the top of the stack.          |
| [`xx_add`](#xx_add)                       | Add two extension field elements.                    |
| [`xx_mul`](#xx_mul)                       | Multiply two extension field elements.               |
| [`x_invert`](#x_invert)                   | Extension-field reciprocal of the top of the stack.  |
| [`xb_mul`](#xb_mul)                       | Multiply elements from the extension and base field. |
| [`read_io` + `n`](#read_io--n)            | Read `n` elements from standard input.               |
| [`write_io` + `n`](#write_io--n)          | Write `n` elements to standard output.               |
| [`merkle_step`](#merkle_step)             | Helps traversing a Merkle tree using secret input.   |
| [`merkle_step_mem`](#merkle_step_mem)     | Helps traversing a Merkle tree using RAM.            |
| [`xx_dot_step`](#xx_dot_step)             | Helps computing an extension field dot product.      |
| [`xb_dot_step`](#xx_dot_step)             | Helps computing a mixed-field dot product.           |

## Stack Manipulation

### push + a

**Opcode**: 1

Pushes `a` onto the stack.

| old stack | new stack |
|:----------|:----------|
| `_`       | `_ a`     |

### pop + n

**Opcode**: 3

Pops the `n` top elements from the stack.
1 ⩽ `n` ⩽ 5.

| `n` | old stack     | new stack |
|:----|:--------------|:----------|
| 1   | `_ a`         | `_`       |
| 2   | `_ b a`       | `_`       |
| 3   | `_ c b a`     | `_`       |
| 4   | `_ d c b a`   | `_`       |
| 5   | `_ e d c b a` | `_`       |

### divine + n

**Opcode**: 9

Pushes `n` non-deterministic elements `a` to the stack.
1 ⩽ `n` ⩽ 5.

This is part of the interface for Triton VM’s secret input;
see also the section regarding
[non-determinism](about-instructions.md#non-deterministic-instructions).

The name of the instruction is the verb (not the adjective) meaning “to discover by intuition or
insight.”

| `n` | old stack | new stack     |
|:----|:----------|:--------------|
| 1   | `_`       | `_ a`         |
| 2   | `_`       | `_ b a`       |
| 3   | `_`       | `_ c b a`     |
| 4   | `_`       | `_ d c b a`   |
| 5   | `_`       | `_ e d c b a` |

### pick + i

**Opcode**: 17

Moves the element indicated by `i` to the top of the stack.
0 ⩽ `i` < 16.

| `i` | old stack     | new stack     |
|:----|:--------------|:--------------|
| 0   | `_ d c b a x` | `_ d c b a x` |
| 1   | `_ d c b x a` | `_ d c b a x` |
| 2   | `_ d c x b a` | `_ d c b a x` |
| 3   | `_ d x c b a` | `_ d c b a x` |
| 4   | `_ x d c b a` | `_ d c b a x` |
| …   | …             | …             |

### place + i

**Opcode**: 25

Moves the top of the stack to the indicated position `i`.
0 ⩽ `i` < 16.

| `i` | old stack     | new stack     |
|:----|:--------------|:--------------|
| 0   | `_ d c b a x` | `_ d c b a x` |
| 1   | `_ d c b a x` | `_ d c b x a` |
| 2   | `_ d c b a x` | `_ d c x b a` |
| 3   | `_ d c b a x` | `_ d x c b a` |
| 4   | `_ d c b a x` | `_ x d c b a` |
| …   | …             | …             |

### dup + i

**Opcode**: 33

Duplicates the element `i`th stack element and pushes it onto the stack.
0 ⩽ `i` < 16.

| `i` | old stack     | new stack       |
|:----|:--------------|:----------------|
| 0   | `_ e d c b a` | `_ e d c b a a` |
| 1   | `_ e d c b a` | `_ e d c b a b` |
| 2   | `_ e d c b a` | `_ e d c b a c` |
| 3   | `_ e d c b a` | `_ e d c b a d` |
| 4   | `_ e d c b a` | `_ e d c b a e` |
| …   | …             | …               |


### swap + i

**Opcode**: 41

Swaps the `i`th stack element with the top of the stack.
0 ⩽ `i` < 16.

| `i` | old stack     | new stack     |
|:----|:--------------|:--------------|
| 0   | `_ e d c b a` | `_ e a c b a` |
| 1   | `_ e d c b a` | `_ e d c a b` |
| 2   | `_ e d c b a` | `_ e d a b c` |
| 3   | `_ e d c b a` | `_ e a c b d` |
| 4   | `_ e d c b a` | `_ a d c b e` |
| …   | …             | …             |


## Control Flow

### halt

**Opcode**: 0

Solves the halting problem (if the instruction is reached).
Indicates graceful shutdown of the VM.

The only legal instruction following instruction `halt` is `halt`.
It is only possible to prove correct execution of a program if the last executed instruction is
`halt`.

### nop

**Opcode**: 8

Does nothing.

### skiz

**Opcode**: 2

Pop the top of the stack and skip the next instruction if the popped element is zero.

“`skiz`” stands for “**sk**ip **i**f **z**ero”.
An alternative perspective for this instruction is “execute if non-zero”, or “execute if”, or even
just “if”.

The amount by which `skiz` increases the instruction pointer `ip` depends on both the top of the
stack and the [size](about-instructions.md#instruction-sizes) of the next instruction
[in the program](data-structures.md#program-memory) (not the next instruction that gets actually
executed).

| next instruction’s size | old stack          | new stack | old `ip` | new `ip` |
|:------------------------|:-------------------|:----------|:---------|:---------|
| any                     | `_ a` with `a` ≠ 0 | `_`       | `ip`     | `ip+1`   |
| single word             | `_ 0`              | `_`       | `ip`     | `ip+2`   |
| double word             | `_ 0`              | `_`       | `ip`     | `ip+3`   |

### call + d

**Opcode**: 49

Push address pair `(ip+2, d)` to the jump stack, and jump to absolute address `d`.

| old `ip` | new `ip` | old jump stack | new jump stack |
|:---------|:---------|:---------------|:---------------|
| `ip`     | `d`      | `_`            | `_ (ip+2, d)`  |

### return

**Opcode**: 16

Pop one address pair off the jump stack and jump to that pair's return address (which is the pair’s
first element).

Executing this instruction with an empty jump stack crashes Triton VM.

| old `ip` | new `ip` | old jump stack | new jump stack |
|:---------|:---------|:---------------|:---------------|
| `ip`     | `o`      | `_ (o, d)`     | `_`            |

### recurse

**Opcode**: 24

Peek at the top address pair of the jump stack and jump to that pair's destination address (which is
the pair’s second element).

Executing this instruction with an empty jump stack crashes Triton VM.

| old `ip` | new `ip` | old jump stack | new jump stack |
|:---------|:---------|:---------------|:---------------|
| `ip`     | `d`      | `_ (o, d)`     | `_ (o, d)`     |

### recurse_or_return

**Opcode**: 32

Like `recurse` if `st5` ≠ `st6`, like `return` if `st5` = `st6`.

Instruction `recurse_or_return` behaves – surprise! – either like instruction [`recurse`](#recurse)
or like instruction [`return`](#return).
The (deterministic) decision which behavior to exhibit is made at runtime and depends on stack
elements `st5` and `st6`, the stack elements at indices 5 and 6.
If `st5` ≠ `st6`, then `recurse_or_return` acts like instruction [`recurse`](#recurse), else like
[`return`](#return).

The instruction is designed to facilitate loops using pointer equality as termination condition and
to play nicely with instructions [`merkle_step`](#merkle_step) and
[`merkle_step_mem`](#merkle_step_mem).

Executing this instruction with an empty jump stack crashes Triton VM.

|               | old `ip` | new `ip` | old jump stack | new jump stack |
|:--------------|:---------|:---------|:---------------|:---------------|
| `st5` ≠ `st6` | `ip`     | `d`      | `_ (o, d)`     | `_ (o, d)`     |
| `st5` = `st6` | `ip`     | `o`      | `_ (o, d)`     | `_`            |

### assert

**Opcode**: 10

Pops `a` if `a` = 1, else crashes the virtual machine.

| old stack | new stack |
|:----------|:----------|
| `_ a`     | `_`       |


## Memory Access

### read_mem + n

**Opcode**: 57

Interprets the top of the stack, _i.e._, `st0` as a pointer `p` into
[RAM](data-structures.md#random-access-memory).
Reads consecutive values $\mathsf{v}_\mathsf{i}$ from RAM at address `p` and puts them onto the op
stack below `st0`.
Decrements the RAM pointer, i.e., the top of the stack, by `n`.
1 ⩽ `n` ⩽ 5.

Let the RAM at address `p` contain `a`, at address `p+1` contain `b`, and so on.
Then, this instruction behaves as follows:

| `i` | old stack | new stack         |
|:----|:----------|:------------------|
| `1` | `_ p`     | `_ a p-1`         |
| `2` | `_ p+1`   | `_ b a p-1`       |
| `3` | `_ p+2`   | `_ c b a p-1`     |
| `4` | `_ p+3`   | `_ d c b a p-1`   |
| `5` | `_ p+4`   | `_ e d c b a p-1` |

### write_mem + n

**Opcode**: 11

Interprets the top of the stack, _i.e._, `st0` as a pointer `p` into
[RAM](data-structures.md#random-access-memory).
Writes the stack's `n` top-most values below `st0`, called $\mathsf{v}_\mathsf{i}$, to RAM at the
address `p+i`, popping them.
Increments the RAM pointer, i.e., the top of the stack, by `n`.
1 ⩽ `n` ⩽ 5.

| `i` | old stack       | new stack | old RAM | new RAM                                |
|:----|:----------------|:----------|:--------|:---------------------------------------|
| `1` | `_ a p`         | `_ p+1`   | []      | [p: a]                                 |
| `2` | `_ b a p`       | `_ p+2`   | []      | [p: a, p+1: b]                         |
| `3` | `_ c b a p`     | `_ p+3`   | []      | [p: a, p+1: b, p+2: c]                 |
| `4` | `_ d c b a p`   | `_ p+4`   | []      | [p: a, p+1: b, p+2: c, p+3: d]         |
| `5` | `_ e d c b a p` | `_ p+5`   | []      | [p: a, p+1: b, p+2: c, p+3: d, p+4: e] |


## Hashing

### hash

**Opcode**: 18

Hashes the stack's ten top-most elements and puts the resulting digest onto the stack.

In more detail:
Pops the stack’s ten top-most elements, `jihgfedcba`.
The elements are reversed and one-padded to a length of 16, giving `abcdefghij111111`.
(This padding corresponds to the fixed-length input padding of Tip5;
see also section 2.5 of the [Tip5 paper](https://eprint.iacr.org/2023/107.pdf).)
The Tip5 permutation is applied to `abcdefghij111111`, resulting in `αβγδεζηθικuvwxyz`.
The first five elements of this result, i.e., `αβγδε`, are reversed and pushed onto the stack.
The new top of the stack is then `εδγβα`.

| old stack      | new stack |
|:---------------|:----------|
| `_ jihgfedcba` | `_ εδγβα` |

### assert_vector

**Opcode**: 26

Asserts equality of $\mathsf{st}_i$ to $\mathsf{st}_{i+5}$ for 0 ⩽ `i` ⩽ 4.
Crashes the VM if any pair is unequal.
Pops the 5 top-most stack elements.

| old stack       | new stack |
|:----------------|:----------|
| `_ edcba edcba` | `_ edcba` |

### sponge_init

**Opcode**: 40

Initializes (resets) the Sponge's state. Must be the first Sponge instruction executed.

### sponge_absorb

**Opcode**: 34

Absorbs the stack's ten top-most elements into the Sponge state.

Crashes Triton VM if the Sponge state is not [initialized](#sponge_init).

| old stack      | new stack |
|:---------------|:----------|
| `_ jihgfedcba` | `_`       |

### sponge_absorb_mem

**Opcode**: 48

Absorbs the ten RAM elements at addresses `p`, `p+1`, … into the Sponge state.
Overwrites stack elements `st1` through `st4` with the first four absorbed elements.

Crashes Triton VM if the Sponge state is not [initialized](#sponge_init).

| old stack  | new stack       |
|:-----------|:----------------|
| `_ dcba p` | `_ hgfe (p+10)` |

### sponge_squeeze

**Opcode**: 56

Squeezes the Sponge and pushes the ten squeezed elements onto the stack.

Crashes Triton VM if the Sponge state is not [initialized](#sponge_init).

| old stack | new stack      |
|:----------|:---------------|
| `_`       | `_ zyxwvutsrq` |


## Base Field Arithmetic

### add

**Opcode**: 42

Replaces the stack’s top two elements with their sum (as field elements).

| old stack | new stack   |
|:----------|:------------|
| `_ b a`   | `_ (a + b)` |

### addi + a

**Opcode**: 65

Adds the instruction’s argument `a` to the top of the stack.

| old stack | new stack   |
|:----------|:------------|
| `_ b`     | `_ (a + b)` |

### mul

**Opcode**: 50

Replaces the stack’s top two elements with their product (as field elements).

| old stack | new stack |
|:----------|:----------|
| `_ b a`   | `_ (a·b)` |

### invert

**Opcode**: 64

Computes the multiplicative inverse (over the field) of the top of the stack.
Crashes the VM if the top of the stack is 0.

| old stack | new stack |
|:----------|:----------|
| `_ a`     | `_ (1/a)` |

### eq

**Opcode**: 58

Replaces the stack’s top two elements with `1` if they are equal, with `0` otherwise.

|           | old stack | new stack |
|:----------|:----------|:----------|
| `a` = `b` | `_ b a`   | `_ 1`     |
| `a` ≠ `b` | `_ b a`   | `_ 0`     |


## Bitwise Arithmetic

### split

**Opcode**: 4

Decomposes the top of the stack into its lower 32 bits and its upper 32 bits.
The lower 32 bits are the new top of the stack.

| old stack | new stack |
|:----------|:----------|
| `_ a`     | `_ hi lo` |

### lt

**Opcode**: 6

“Less than” of the stack’s two top-most elements.

Crashes the VM if `a` or `b` is not u32.

| old stack | new stack |
|:----------|:----------|
| `_ b a`   | `_ a<b`   |

### and

**Opcode**: 14

Bitwise “and” of the stack’s two top-most elements.

Crashes the VM if `a` or `b` is not u32.

| old stack | new stack |
|:----------|:----------|
| `_ b a`   | `_ a&b`   |

### xor

**Opcode**: 22

Bitwise “exclusive or” of the stack's two top-most elements.

Crashes the VM if `a` or `b` is not u32.

| old stack | new stack |
|:----------|:----------|
| `_ b a`   | `_ a^b`   |

### log_2_floor

**Opcode**: 12

The number of bits in `a` minus 1, _i.e._, $\lfloor\log_2\mathsf{a}\rfloor$.

Crashes the VM if `a` is 0 or not u32.

| old stack | new stack     |
|:----------|:--------------|
| `_ a`     | `_ ⌊log₂(a)⌋` |

### pow

**Opcode**: 30

The top of the stack to the power of the stack's runner up.

Crashes the VM if exponent `e` is not u32.

| old stack | new stack |
|:----------|:----------|
| `_ e b`   | `_ b**e`  |

### div_mod

**Opcode**: 20

Division with remainder of numerator `n` by denominator `d`.
Guarantees the properties `n == q·d + r` and `r < d`.

Crashes the VM if `n` or `d` is not u32 or if `d` is 0.

| old stack | new stack |
|:----------|:----------|
| `_ d n`   | `_ q r`   |

### pop_count

**Opcode**: 28

Computes the [hamming weight](https://en.wikipedia.org/wiki/Hamming_weight) or “population count”
of the top of the stack, `a`.

Crashes the VM if `a` is not u32.

| old stack | new stack |
|:----------|:----------|
| `_ a`     | `_ w`     |


## Extension Field Arithmetic

### xx_add

**Opcode**: 66

Replaces the top six elements of the stack with the sum of the two extension field elements
encoded by the first and the second triple of stack elements.

| old stack       | new stack |
|:----------------|:----------|
| `_ z y x b c a` | `_ w v u` |

### xx_mul

**Opcode**: 74

Replaces the top six elements of the stack with the product of the two extension field elements
encoded by the first and the second triple of stack elements.

| old stack       | new stack |
|:----------------|:----------|
| `_ z y x b c a` | `_ w v u` |

### x_invert

**Opcode**: 72

Inverts the extension field element encoded by field elements `z y x` in-place.
Crashes the VM if the extension field element is 0.

| old stack | new stack |
|:----------|:----------|
| `_ z y x` | `_ w v u` |

### xb_mul

**Opcode**: 82

Replaces the stack’s top four elements with the scalar product of the top of the stack with the
extension field element encoded by stack elements `st1` through `st3`.

| old stack   | new stack |
|:------------|:----------|
| `_ z y x a` | `_ w v u` |


## Input/Output

### read_io + n

**Opcode**: 73

Reads `n` elements from standard input and pushes them onto the stack.
1 ⩽ `n` ⩽ 5.

| `n` | old stack | new stack     |
|:----|:----------|:--------------|
| 1   | `_`       | `_ a`         |
| 2   | `_`       | `_ b a`       |
| 3   | `_`       | `_ c b a`     |
| 4   | `_`       | `_ d c b a`   |
| 5   | `_`       | `_ e d c b a` |

### write_io + n

**Opcode**: 19

Pops the top `n` elements from the stack and writes them to standard output.
1 ⩽ `n` ⩽ 5.

| `n` | old stack     | new stack |
|:----|:--------------|:----------|
| 1   | `_ a`         | `_`       |
| 2   | `_ b a`       | `_`       |
| 3   | `_ c b a`     | `_`       |
| 4   | `_ d c b a`   | `_`       |
| 5   | `_ e d c b a` | `_`       |


## Many-In-One

### merkle_step

**Opcode**: 36

Helps traversing a Merkle tree during authentication path verification.

The mechanics are as follows.
The 6th element of the stack `i` (also referred to as `st5`) is taken as the node index for a Merkle
tree that is claimed to include data whose digest is the content of stack registers `st4` through
`st0`, i.e., `edcba`.
The sibling digest of `edcba` is `εδγβα` and is read from the
[input interface of secret data](about-instructions.md#non-deterministic-instructions).
The least-significant bit of `i` indicates whether `edcba` is the digest of a left leaf or a right
leaf of the Merkle tree's current level.
Depending on this least significant bit of `i`, `merkle_step` either
1. (`i` = 0 mod 2) interprets `edcba` as the left digest, `εδγβα` as the right digest, or
1. (`i` = 1 mod 2) interprets `εδγβα` as the left digest, `edcba` as the right digest.

In either case,
1. the left and right digests are hashed, and the resulting digest `zyxwv` replaces the top of the
   stack, and
1. 6th register `i` is shifted by 1 bit to the right, _i.e._, the least-significant bit is dropped.

In conjunction with instructions [`recurse_or_return`](#recurse_or_return) and
[`assert_vector`](#assert_vector), instruction `merkle_step` and instruction
[`merkle_step_mem`](#merkle_step_mem) allow efficient verification of a Merkle authentication path.

Crashes the VM if `i` is not u32.

| old stack   | new stack           |
|:------------|:--------------------|
| `_ i edcba` | `_ (i div 2) zyxwv` |

### merkle_step_mem

**Opcode**: 44

Helps traversing a Merkle tree during authentication path verification with the authentication path
being supplied in [RAM](data-structures.md#random-access-memory).

This instruction works very similarly to instruction [`merkle_step`](#merkle_step).
The main difference, as the name suggests, is the source of the sibling digest:
Instead of reading it from the input interface of secret data, it is supplied via RAM.
Stack element `st7` is taken as the RAM pointer, holding the memory address at which the next
sibling digest is located in RAM.
Executing instruction `merkle_step_mem` increments the memory pointer by the length of one digest,
anticipating an authentication path that is laid out continuously.
Stack element `st6` does not change when executing instruction `merkle_step_mem` in order to
facilitate instruction [`recurse_or_return`](#recurse_or_return).

This instruction allows verifiable _re_-use of an authentication path.
This is necessary, for example, when verifiably updating a Merkle tree:
first, the authentication path is used to confirm inclusion of some old leaf,
and then to compute the tree's new root from the new leaf.

Crashes the VM if `i` is not u32.

| old stack       | new stack                 |
|:----------------|:--------------------------|
| `_ p f i edcba` | `_ p+5 f (i div 2) zyxwv` |

### xx_dot_step

**Opcode**: 80

Reads two extension field elements from RAM located at the addresses corresponding to the two top
stack elements, multiplies the extension field elements, and adds the product `(p0, p1, p2)` to an
accumulator located on stack immediately below the two pointers.
Also, increases the pointers by the number of words read.

This instruction facilitates efficient computation of the
[dot product](https://en.wikipedia.org/wiki/Dot_product) of two vectors containing extension field
elements.

| old stack       | new stack                    |
|:----------------|:-----------------------------|
| `_ z y x *b *a` | `_ z+p2 y+p1 x+p0 *b+3 *a+3` |

### xb_dot_step

**Opcode**: 88

Reads one base field element from RAM located at the addresses corresponding to the top of the
stack, one extension field element from RAM located at the address of the second stack element,
multiplies the field elements, and adds the product `(p0, p1, p2)` to an accumulator located on
stack immediately below the two pointers.
Also, increase the pointers by the number of words read.

This instruction facilitates efficient computation of the
[dot product](https://en.wikipedia.org/wiki/Dot_product) of a vector containing base field elements
with a vector containing extension field elements.

| old stack       | new stack                    |
|:----------------|:-----------------------------|
| `_ z y x *b *a` | `_ z+p2 y+p1 x+p0 *b+3 *a+1` |
