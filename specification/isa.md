# Triton VM Instruction Set Architecture

Triton VM is a stack machine with RAM, with a second data structure for evaluating a sponge permutation.
It is a [Harvard architecture](https://en.wikipedia.org/wiki/Harvard_architecture) with read-only memory for the program.
The arithmetization of the VM is defined over the *B-field* $\mathbb{F}_p$ where $p=2^{64}-2^{32}+1$.
This means the registers and memory elements take values from $\mathbb{F}_p$, and the transition function gives rise to low-degree transition verification polynomials from the ring of multivariate polynomials over $\mathbb{F}_p$.

Instructions have variable width:
they either consist of one word, i.e., one B-Field element, or of two words, i.e., two B-Field elements.
An example for a single-word instruction is `pop`, removing the top of the stack.
An example for a double-word instruction is `push` + `arg`, pushing `arg` to the stack.

## Data Structures

### Memory
The term *memory* refers to a data structure that gives read access (and possibly write access, too) to elements indicated by an *address*.
The address lives in the same field.
There are four separate notions of memory:
1. *RAM*, to which the VM can read and write field elements.
2. *Program Memory*, from which the VM reads instructions.
3. *OpStack Memory*, which stores the part of the operational stack that is not represented explicitly by the operational stack registers.
4. *Jump Stack Memory*, which stores the entire jump stack.

### OpStack
The stack is a last-in;first-out data structure that allows the program to store intermediate variables, pass arguments, and keep pointers to objects held in RAM.

### Jump Stack
Another last-in;first-out data structure that keeps track of return and destination addresses.
This stack changes only when control follows a `call` or `return` instruction.

## Registers

This section covers all columns in the Protocol Table.
Only a subset of these registers relate to the instruction set;
the remaining registers exist only to enable an efficient arithmetization and are marked with an asterisk.

| Register               | Name                         | Purpose                                                                                   |
|:-----------------------|:-----------------------------|:------------------------------------------------------------------------------------------|
| *`clk`                 | cycle counter                | counts the number of cycles the program has been running for                              |
| `ip`                   | instruction pointer          | contains the memory address (in Program Memory) of the instruction                        |
| `ci`                   | current instruction register | contains the full instruction                                                             |
| `ni`                   | next instruction register    | contains the full instruction of the next cycle; to be used for immediate instructions    |
| *`ib0` through `ib5`   | instruction bucket           | contains the ith bit of the instruction                                                   |
| *`if0` through `if9`   | instruction flags            | used as intermediate values to keep the AIR degree low                                    |
| `jsp`                  | jump stack pointer           | contains the memory address (in jump stack memory) of the top of the jump stack           |
| `jso`                  | jump stack origin            | contains the value of the instruction pointer of the last `call`                          |
| `jsd`                  | jump stack destination       | contains the immediate argument of the last `call`                                        |
| `st0` through `st?`    | operational stack registers  | contain explicit operational stack values                                                 |
| *`inv`                 | zero indicator               | assumes the inverse of the the top of the stack when it is nonzero, and zero otherwise    |
| *`osp`                 | operational stack pointer    | contains the memory address (in stack memory) of the top of the operational stack minus ? |
| *`osv`                 | operational stack value      | contains the (stack) memory value at the given address                                    |
| *`hv0` through `hv4`   | helper variable registers    | helper variables for some arithmetic operations                                           |
| *`ramp`                | RAM pointer                  | contains an address (in RAM) for reading or writing                                       |
| *`ramv`                | RAM value                    | contains the value of the RAM element at the given address                                |
| `aux0` through `aux15` | auxiliary registers          | data structure dedicated to hashing instructions                                          |

### Instruction

The instruction is represented by one register called the *current instruction register* `ci`.
This value then decomposed into its 6 constituent bits, giving rise to 6 *instruction bit registers*, labeled `ib0` through `ib5`.
Additionally, there is a register called the *instruction pointer* (`ip`), which contains the address of the current instruction in instruction memory.
Also, there is the *next instruction register* `ni` that contains the next instruction.
For immediate instructions, `ni` takes the value of the next instruction.
Otherwise, `ni` holds the argument for current instruction `ci`.

### Stack

The stack is represented by ?-many registers called *stack registers* (`st0` – `st?`) plus the OpStack Memory.
The top ?-many elements of the OpStack are directly accessible, the remainder of the OpStack, i.e, the part held in OpStack Memory, is not.
In order to access elements of the OpStack held in OpStack Memory, the stack has to shrink by discarding elements from the top – potentially after writing them to RAM – thus moving lower elements into the stack registers.

The stack grows upwards, in line with the metaphor that justifies the name "stack".

The registers `osp` and `osv` are not directly accessible by the program running in TritonVM.
They primarily exist to allow efficient [arithmetization](arithmetization.md).
The register _operational stack pointer_ `osp` stores the length of the operational stack plus constant offset ?.
For example, if `osp` is ?, the OpStack is empty.
If `osp` is (?+2), the OpStack contains 2 elements.
The register `osv` holds the top-most value of the OpStack Memory, or zero if no such value exists.

### RAM

TritonVM has dedicated Random-Access Memory.
Programs can read from and write to RAM using instructions `read_mem` and `write_mem`.
The address to read from – respectively, to write to – is one of the two top-most OpStack elements, depending on the instruction.

The registers `ramp` and `ramv` are not directly accessible by the program running in TritonVM.
They primarily exist to allow efficient [arithmetization](arithmetization.md).
For any of the two RAM instructions, register `ramp` is set to the RAM address pointed to, i.e., either `st0` or `st1`, depending on the instruction.
Likewise, `ramv` is set to the value being read or written.

### Helper Variables

Some instructions require helper variables in order to generate an efficient arithmetization.
To this end, there are 5 helper registers, labeled `hv0` through `hv4`.
These registers are part of the arithmetization of the architecture, but not needed to define the instruction set.

## Instructions

Most instructions are contained within a single, parameterless machine word.

Some instructions take a machine word as argument and are so considered double-word instructions. They are recognised by the form "`instr` + `arg`".

### OpStack Manipulation

In this section *stack* is short for *operational stack*.

| Instruction  | Value | old OpStack         | new OpStack           | Description                                                                    |
|:-------------|:------|:--------------------|:----------------------|:-------------------------------------------------------------------------------|
| `pop`        | ?     | `_ a`               | `_`                   | Pops top element from stack.                                                   |
| `push` + `a` | ?     | `_`                 | `_ a`                 | Pushes `a` onto the stack.                                                     |
| `guess`      | ?     | `_`                 | `_ a`                 | Pushes a nondeterministic element `a` to the stack.                            |
| `dup`  + `i` | ?     | e.g., `_ e d c b a` | e.g., `_ e d c b a d` | Duplicates the element `i` positions away from the top, assuming `0 <= i < ?`. |
| `swap` + `i` | ?     | e.g., `_ e d c b a` | e.g., `_ e a c b d`   | Swaps the `i`th stack element with the top of the stack, assuming `0 < i < ?`. |

### Control Flow

| Instruction  | Value | old OpStack | new OpStack | old `ip` | new `ip`     | old JumpStack | new JumpStack | Description                                                                                                                         |
|:-------------|:------|:------------|:------------|:---------|:-------------|:--------------|:--------------|:------------------------------------------------------------------------------------------------------------------------------------|
| `skiz`       | ?     | `_ a`       | `_`         | `_`      | `_ + s`      | `_`           | `_`           | Skip next instruction if `a` is zero. `s` ∈ {1, 2, 3} depends on `a` and whether or not next instruction has an immediate argument. |
| `call` + `d` | ?     | `_`         | `_`         | `o`      | `d`          | `_`           | `_ (o+2, d)`  | Push `(o+2,d)` to the jump stack, and jump to absolute immediate address `d`                                                        |
| `return`     | ?     | `_`         | `_`         | `_`      | `o`          | `_ (o, d)`    | `_`           | Pop one pair off the jump stack and jump to that pair's return address (which is the first element).                                |
| `recurse`    | ?     | `_`         | `_`         | `_`      | `d`          | `_ (o, d)`    | `_ (o, d)`    | Peek at the top pair of the jump stack and jump to that pair's destination address (which is the second element).                   |
| `assert`     | ?     | `_ a`       | `_`         | `_`      | `_ + 1` or 💥 | `_`           | `_`           | Pops `a` if `a == 1`, else crashes the virtual machine.                                                                             |
| `halt`       | 0     | `_`         | `_`         | `_`      | `_ + 1`      | `_`           | `_`           | Solves the halting problem (if the instruction is reached).                                                                         |

### Memory Access

| Instruction | Value | old OpStack | new OpStack | old `ramv` | new `ramv` | Description                                                                          |
|:------------|:------|:------------|:------------|:-----------|:-----------|:-------------------------------------------------------------------------------------|
| `read_mem`  | ?     | `_ p`       | `_ p v`     | `v`        | `v`        | Reads value `v` from RAM at location `p` and pushes the read element to the opstack. |
| `write_mem` | ?     | `_ p v`     | `_ p`       | `_`        | `v`        | Writes value `v` to RAM at the location `p` and pops the top of the opstack.         |

### Auxiliary Register Instructions

| Instruction      | Value | old OpStack | new OpStack   | old `aux`   | new `aux`                | Description                                                                                                                      |
|:-----------------|:------|:------------|:--------------|:------------|:-------------------------|:---------------------------------------------------------------------------------------------------------------------------------|
| `xlix`           | ?     | `_`         | `_`           | `_`         | `xlix(_)`                | Applies the Rescue-XLIX permutation to the auxiliary registers.                                                                  |
| `clearall`       | ?     | `_`         | `_`           | `_`         | `0000000000000000`       | Sets all auxiliary registers to zero.                                                                                            |
| `squeeze` + `i`  | ?     | `_`         | `_ v`         | `…v…`       | `…v…`                    | Pushes to the stack the `i`th auxiliary register. Assumes `0 <= i < 16`.                                                         |
| `absorb`  + `i`  | ?     | `_ a`       | `_`           | `…v…`       | `…(v+a)…`                | Pops the top off the stack and adds it into the `i`th auxiliary register. Assumes `0 <= i < 16`.                                 |
| `guess_sibling`  | ?     | `_ i`       | `_ (i div 2)` | `fedcba__…` | e.g., `zyxwvufedcba0000` | Helps traversing a Merkle tree during authentication path verification. See extended description below.                          |
| `compare_digest` | ?     | `_`         | `_ a`         | `fedcba__…` | `fedcba__…`              | Compare `aux0` through `aux5` (i.e., `fedcba`) to `st0` through `st5` and put the comparison's result `a ∈ {0, 1}` on the stack. |

The instruction `guess_sibling` works as follows.
The value at the top of the stack `i` is taken as the leaf index for a Merkle tree that is claimed to include data whose digest is the content of auxiliary registers `aux0` through `aux5`, i.e., `fedcba`.
The sibling digest of `fedcba` is `zyxwvu` and is guessed non-deterministically, i.e., filled in by the VM's execution environment.
The least-significant bit of `i` indicates whether `fedcba` is the digest of a left leaf or a right leaf of the Merkle tree's base level.
Depending on this least-significant bit of `i`, `guess_sibling` either
1. does not change registers `aux0` through `aux5` and moves `zyxwvu` into registers `aux6` through `aux11`, or
2. moves `fedcba` into registers `aux6` through `aux11` and moves `zyxwvu` into registers `aux0` through `aux5`.
In both cases, auxiliary registers `aux12` through `aux15` are set to 0.
The top of the operational stack is modified by shifting `i` by 1 bit to the right, i.e., dropping the least-significant bit.
In conjunction with instruction `xlix` and `compare_digest`, the instruction `guess_sibling` allows to efficiently verify a Merkle authentication path.

### Arithmetic on Stack

| Instruction | Value | old OpStack     | new OpStack     | Description                                                                                                                                                                      |
|:------------|:------|:----------------|:----------------|:---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `add`       | ?     | `_ b a`         | `_ c`           | Computes the sum (`c`) of the top two elements of the stack (`b` and `a`) over the field.                                                                                        |
| `mul`       | ?     | `_ b a`         | `_ c`           | Computes the product (`c`) of the top two elements of the stack (`b` and `a`) over the field.                                                                                    |
| `invert`    | ?     | `_ a`           | `_ b`           | Computes the multiplicative inverse (over the field) of the top of the stack. Crashes the VM if the top of the stack is 0.                                                       |
| `split`     | ?     | `_ a`           | `_ lo hi`       | Decomposes the top of the stack into the lower 32 bits and the upper 32 bits.                                                                                                    |
| `eq`        | ?     | `_ b a`         | `_ (b == a)`    | Tests the top two stack elements for equality.                                                                                                                                   |
| `lt`        | ?     | `_ b a`         | `_ (b < a)`     | Tests if the one-from top element is less than or equal the top element on the stack, assuming both are 32-bit integers.                                                         |
| `and`       | ?     | `_ b a`         | `_ (b and a)`   | Computes the bitwise-and of the top two stack elements, assuming both are 32-bit integers.                                                                                       |
| `xor`       | ?     | `_ b a`         | `_ (b xor a)`   | Computes the bitwise-xor of the top two stack elements, assuming both are 32-bit integers.                                                                                       |
| `reverse`   | ?     | `_ a`           | `_ b`           | Reverses the bit expansion of the top stack element, assuming it is a 32-bit integer.                                                                                            |
| `div`       | ?     | `_ n d`         | `_ q r`         | Computes division with remainder of the top two stack elements, assuming the arguments are both 32-bit integers. The result satisfies `n == q * d + r` and `r < d` and `q <= n`. |
| `xxadd`     | ?     | `_ z y x b c a` | `_ z y x w v u` | Adds the two extension field elements encoded by field elements `z y x` and `b c a`, overwriting the top-most extension field element with the result.                           |
| `xxmul`     | ?     | `_ z y x b c a` | `_ z y x w v u` | Multiplies the two extension field elements encoded by field elements `z y x` and `b c a`, overwriting the top-most extension field element with the result.                     |
| `xinv`      | ?     | `_ z y x`       | `_ w v u`       | Inverts the extension field element encoded by field elements `z y x` in-place. Crashes the VM if the extension field element is 0.                                              |
| `xbmul`     | ?     | `_ z y x a`     | `_ w v u`       | Scalar multiplication of the extension field element encoded by field elements `z y x` with field element `a`. Overwrites `z y x` with the result.                               |

### Input/Output

| Instruction | Value | old OpStack | new OpStack | Description                                                       |
|:------------|:------|:------------|:------------|:------------------------------------------------------------------|
| `read_io`   | ?     | `_`         | `_ a`       | Reads a character from standard input and pushes it to the stack. |
| `write_io`  | ?     | `_ a`       | `_`         | Pops `a` from the stack and writes it to standard output.         |
