# Triton VM Instruction Set Architecture

Triton VM is a stack machine with RAM, with a second data structure for evaluating a sponge permutation.
The arithmetization of the VM is defined over the *B-field* $\mathbb{F}_p$ where $p=2^{64}-2^{32}+1$.
This means the registers and memory elements take values from $\mathbb{F}_p$, and the transition function gives rise to low-degree transition verification polynomials from the ring of multivariate polynomials over $\mathbb{F}_p$.

## Data Structures

**Memory**
The term *memory* refers to a data structure that gives read access (and possibly write access, too) to elements indicated by an *address*.
The address lives in the same field.
There are four separate notions of memory:
1. *RAM*, to which the VM can read and write field elements.
2. *Program Memory*, from which the VM reads instructions.
3. *OpStack Memory*, which stores the part of the operational stack that is not represented explicitly by the operational stack registers.
4. *Jump Stack Memory*, which stores the entire jump stack.

**OpStack**
The stack is a last-in;first-out data structure that allows the program to store intermediate variables, pass arguments, and keep pointers to objects held in RAM.

**Jump Stack**
Another last-in;first-out data structure that keeps track of return and destination addresses.
This stack changes only when control follows a `call` or `return` instruction.

## Registers

This section covers all columns in the Protocol Table.
Only a subset of these registers relate to the instruction set;
the remaining registers exist only to enable an efficient arithmetization and are marked with an asterisk.

| Register               | Name                         | Purpose                                                                                   |
| ---------              | -----                        | -------                                                                                   |
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

**Instruction**
The instruction is represented by one register called the *current instruction register* `ci`.
This value then decomposed into its 6 constituent bits, giving rise to 6 *instruction bit registers*, labeled `ib0` through `ib5`.
Additionally, there is a register called the *instruction pointer* (`ip`), which contains the address of the current instruction in instruction memory.
Also, there is the *next instruction register* `ni` that contains the next instruction.
For immediate instructions, `ni` takes the value of the next instruction.
Otherwise, `ni` holds the argument for current instruction `ci`.

**Stack**
The stack is represented by ?-many registers called *stack registers* (`st0` – `st?`), which take the values of the top ?-many stack elements.
In addition to these registers, there is the *operational stack pointer* register `osp` whose value is the length of the operational stack minus ?.
The stack grows upwards, in line with the metaphor that justifies the name "stack".

**RAM**

**Helper Variables**
Some instructions require helper variables in order to generate an efficient arithmetization.
To this end, there are 5 helper registers, labeled `hv0` through `hv4`.
These registers are part of the arithmetization of the architecture, but not needed to define the instruction set.

## Instructions

### OpStack Manipulation

In this section *stack* is short for *operational stack*.

| Instruction    | Value | Effect on OpStack                           | Description                                                                        |
| ---            | ---   | ---                                         | ---                                                                                |
| `pop`          | ?     | `stack a  -->  stack`                       | Pops top element from stack.                                                       |
| `push` + `arg` | ?     | `stack  -->  stack arg`                     | Pushes `arg` onto the stack.                                                       |
| `pad`          | ?     | `stack --> stack a`                         | Pushes a nondeterministic element `a` to the stack.                                |
| `dup`  + `arg` | ?     | e.g., `stack a b c d  -->  stack a b c d a` | Duplicates the element `arg` positions away from the top, assuming `0 <= arg < ?`. |
| `swap` + `arg` | ?     | `stack p o … b a  -->  stack p a … b o`     | Swaps the `arg`th stack element with the top of the stack, assuming `0 < arg < ?`. |

### Control Flow

| Instruction     | Value | Effect on OpStack     | Effect on `ip` | Effect on JumpStack             | Description                                                                                                       |
| ---             | ---   | ---                   | ---            | ---                             | ---                                                                                                               |
| `skiz`          | ?     | `stack a  -->  stack` | `+(2-a·inv)`   | identity                        | Skip next instruction if `top` is zero.                                                                           |
| `call` + `addr` | ?     | identity              | `= addr`       | `stack --> stack (ip+2, addr)`  | Push `(ip+2,addr)` to the jump stack, and jump to absolute immediate address `addr`                               |
| `return`        | ?     | identity              | `= o`          | `stack (o, d) --> stack`        | Pop one pair off the jump stack and jump to that pair's return address (which is the first element).              |
| `recurse`       | ?     | identity              | `= d`          | `stack (o, d) --> stack (o, d)` | Peek at the top pair of the jump stack and jump to that pair's destination address (which is the second element). |
| `assert`        | ?     | `stack a  -->  stack` | `+1 + 💥(a-1)` | identity                        | Pops `a` if `a == 1`, else crashes the virtual machine.                                                           |
| `halt`          | ?     | identity              | `+1`           | identity                        | Solves the halting problem (if the instruction is reached).                                                       |

### Memory Access

| Instruction | Value | Effect on OpStack     | Effect on `ramv` | Description                                                                                                 |
| ---         | ---   | ---                   | ---              | ---                                                                                                         |
| `read`      | ?     | `stack  -->  stack v` | identity         | Reads a value `v` from RAM at the location pointed to by `st0`, and pushes the read element to the opstack. |
| `write`     | ?     | `stack v  -->  stack` | `= v`            | Writes value `v` to RAM at the location pointed to by `st0`, and pops the top of the opstack.               |

### Auxiliary Register Instructions

| Instruction       | Value | Effect on OpStack   | Effect on `aux`   | Description                                                                                                                                  |
| ---               | ---   | ---                 | ---               | ---                                                                                                                                          |
| `xlix`            | ?     | identity            | `xlix(_)`         | Applies the Rescue-XLIX permutation to the auxiliary registers.                                                                              |
| `clearall`        | ?     | identity            | `0…0`             | Sets all auxiliary registers to zero.                                                                                                        |
| `squeeze` + `arg` | ?     | `stack --> stack a` | identity          | Pushes to the stack the `arg`th auxiliary register.                                                                                          |
| `absorb`  + `arg` | ?     | `stack a --> stack` | `…(v+a)…`         | Pops the top off the opstack and adds it into the `arg`th auxiliary regiser.                                                                 |
| `merkle_left`     | ?     | identity            | `xlix(l…lm…m0…0)` | Helps traversing a Merkle tree. Non-deterministically guesses the corresponding right digest `m…m`, sets capacity to 0, and computes `xlix`. |
| `merkle_right`    | ?     | identity            | `xlix(m…mr…r0…0)` | Helps traversing a Merkle tree. Non-deterministically guesses the corresponding left digest `m…m`, sets capacity to 0, and computes `xlix`.  |
| `compare_digest`  | ?     | `stack --> stack a` | identity          | Compare `aux0` through `aux5` to `st0` through `st5` and put the comparison's result `a ∈ {0, 1}` on the stack.                              |

### Arithmetic on Stack

| Instruction | Value | Effect on OpStack                         | Description                                                                                                                                                                      |
| ---         | ---   | ---                                       | ---                                                                                                                                                                              |
| `add`       | ?     | `stack a b  -->  stack c`                 | Computes the sum (`c`) of the top two elements of the stack (`b` and `a`) over the field.                                                                                        |
| `mul`       | ?     | `stack a b  -->  stack c`                 | Computes the product (`c`) of the top two elements of the stack (`b` and `a`) over the field.                                                                                    |
| `inv`       | ?     | `stack a  -->  stack b`                   | Computes the multiplicative inverse (over the field) of the top of the stack. Crashes the VM if the top of the stack is 0.                                                       |
| `split`     | ?     | `stack a  -->  stack lo hi`               | Decomposes the top of the stack into the lower 32 bits and the upper 32 bits.                                                                                                    |
| `eq`        | ?     | `stack a b  -->  stack (a == b)`          | Tests the top two stack elements for equality.                                                                                                                                   |
| `lt`        | ?     | `stack a b  -->  stack (a < b)`           | Tests if the one-from top element is less than or equal the top element on the stack, assuming both are 32-bit integers.                                                         |
| `and`       | ?     | `stack a b  -->  stack (a and b)`         | Computes the bitwise-and of the top two stack elements, assuming both are 32-bit integers.                                                                                       |
| `xor`       | ?     | `stack a b  -->  stack (a xor b)`         | Computes the bitwise-xor of the top two stack elements, assuming both are 32-bit integers.                                                                                       |
| `reverse`   | ?     | `stack a  -->  stack b`                   | Flips the bit expansion of the top stack element, assuming it is a 32-bit integer.                                                                                               |
| `div`       | ?     | `stack a b  -->  stack c d`               | Computes division with remainder of the top two stack elements, assuming the arguments are both 32-bit integers. The result satisfies `a == c * b + d` and `d < b` and `c <= a`. |
| `xxadd`     | ?     | `stack z y x b c a --> stack z y x w v u` | Adds the two extension field elements encoded by field elements `z y x` and `b c a`, overwriting the top-most extension field element with the result.                           |
| `xxmul`     | ?     | `stack z y x b c a --> stack z y x w v u` | Multiplies the two extension field elements encoded by field elements `z y x` and `b c a`, overwriting the top-most extension field element with the result.                     |
| `xinv`      | ?     | `stack z y x --> stack w v u`             | Inverts the extension field element encoded by field elements `z y x` in-place.                                                                                                  |
| `xbmul`     | ?     | `stack z y x a --> stack w v u`           | Scalar multiplication of the extension field element encoded by field elements `z y x` with field element `a`. Overwrites `z y x` with the result.                               |

### Input/Output

| Instruction | Value | Effect on OpStack     | Description                                                       |
| ---         | ---   | ---                   | ---                                                               |
| `print`     | ?     | `stack a  -->  stack` | Writes character `a` to standard output.                          |
| `scan`      | ?     | `stack  -->  stack a` | Reads a character from standard input and pushes it to the stack. |
