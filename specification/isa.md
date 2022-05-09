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

| Instruction    | Value | old OpStack         | new OpStack           | Description                                                                    |
| ---            | ---   | ---                 | ---                   | ---                                                                            |
| `pop`          | ?     | `_ a`               | `_`                   | Pops top element from stack.                                                   |
| `push` + `a`   | ?     | `_`                 | `_ a`                 | Pushes `a` onto the stack.                                                     |
| `pad`          | ?     | `_`                 | `_ a`                 | Pushes a nondeterministic element `a` to the stack.                            |
| `dup`  + `i`   | ?     | e.g., `_ e d c b a` | e.g., `_ e d c b a d` | Duplicates the element `i` positions away from the top, assuming `0 <= i < ?`. |
| `swap` + `i`   | ?     | e.g., `_ e d c b a` | e.g., `_ e a c b d`   | Swaps the `i`th stack element with the top of the stack, assuming `0 < i < ?`. |

### Control Flow

| Instruction  | Value | old OpStack | new OpStack | old `ip` | new `ip`          | old JumpStack | new JumpStack  | Description                                                                                                       |
| ---          | ---   | ---         | ---         | ---      | ---               | ---           | ---            | ---                                                                                                               |
| `skiz`       | ?     | `_ a`       | `_`         | `_`      | `_ + 2 - a·inv`   | `_`           | `_`            | Skip next instruction if `a` is zero.                                                                             |
| `call` + `d` | ?     | `_`         | `_`         | `o`      | `d`               | `_`           | `_ (o+2, d)`   | Push `(o+2,d)` to the jump stack, and jump to absolute immediate address `d`                                      |
| `return`     | ?     | `_`         | `_`         | `_`      | `o`               | `_ (o, d)`    | `_`            | Pop one pair off the jump stack and jump to that pair's return address (which is the first element).              |
| `recurse`    | ?     | `_`         | `_`         | `_`      | `d`               | `_ (o, d)`    | `_ (o, d)`     | Peek at the top pair of the jump stack and jump to that pair's destination address (which is the second element). |
| `assert`     | ?     | `_ a`       | `_`         | `_`      | `_ + 1 + 💥(a-1)` | `_`           | `_`            | Pops `a` if `a == 1`, else crashes the virtual machine.                                                           |
| `halt`       | ?     | `_`         | `_`         | `_`      | `_ + 1`           | `_`           | `_`            | Solves the halting problem (if the instruction is reached).                                                       |

### Memory Access

| Instruction | Value | old OpStack | new OpStack | old `ramv` | new `ramv` | Description                                                                          |
| ---         | ---   | ---         | ---         | ---        | ---        | ---                                                                                  |
| `read`      | ?     | `_ p`       | `_ p v`     | `v`        | `v`        | Reads value `v` from RAM at location `p` and pushes the read element to the opstack. |
| `write`     | ?     | `_ p v`     | `_ p`       | `_`        | `v`        | Writes value `v` to RAM at the location `p` and pops the top of the opstack.         |

### Auxiliary Register Instructions

| Instruction      | Value | old OpStack | new OpStack | old `aux`   | new `aux`                | Description                                                                                                                                     |
| ---              | ---   | ---         | ---         | ---         | ---                      | ---                                                                                                                                             |
| `xlix`           | ?     | `_`         | `_`         | `_`         | `xlix(_)`                | Applies the Rescue-XLIX permutation to the auxiliary registers.                                                                                 |
| `clearall`       | ?     | `_`         | `_`         | `_`         | `0000000000000000`       | Sets all auxiliary registers to zero.                                                                                                           |
| `squeeze` + `i`  | ?     | `_`         | `_ v`       | `…v…`       | `…v…`                    | Pushes to the stack the `i`th auxiliary register. Assumes `0 <= i < 16`.                                                                        |
| `absorb`  + `i`  | ?     | `_ a`       | `_`         | `…v…`       | `…(v+a)…`                | Pops the top off the stack and adds it into the `i`th auxiliary register. Assumes `0 <= i < 16`.                                                |
| `merkle_left`    | ?     | `_`         | `_`         | `fedcba__…` | `xlix(fedcbazyxwvu0000)` | Helps traversing a Merkle tree. Non-deterministically guesses the corresponding right digest `zyxwvu`, sets capacity to 0, and computes `xlix`. |
| `merkle_right`   | ?     | `_`         | `_`         | `fedcba__…` | `xlix(zyxwvufedcba0000)` | Helps traversing a Merkle tree. Non-deterministically guesses the corresponding left digest `zyxwvu`, sets capacity to 0, and computes `xlix`.  |
| `compare_digest` | ?     | `_`         | `_ a`       | `fedcba__…` | `fedcba__…`              | Compare `aux0` through `aux5` (i.e., `fedcba`) to `st0` through `st5` and put the comparison's result `a ∈ {0, 1}` on the stack.                |

### Arithmetic on Stack

| Instruction | Value | old OpStack     | new OpStack     | Description                                                                                                                                                                      |
| ---         | ---   | ---             | ---             | ---                                                                                                                                                                              |
| `add`       | ?     | `_ b a`         | `_ c`           | Computes the sum (`c`) of the top two elements of the stack (`b` and `a`) over the field.                                                                                        |
| `mul`       | ?     | `_ b a`         | `_ c`           | Computes the product (`c`) of the top two elements of the stack (`b` and `a`) over the field.                                                                                    |
| `inv`       | ?     | `_ a`           | `_ b`           | Computes the multiplicative inverse (over the field) of the top of the stack. Crashes the VM if the top of the stack is 0.                                                       |
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
| ---         | ---   | ---         | ---         | ---                                                               |
| `print`     | ?     | `_ a`       | `_`         | Writes character `a` to standard output.                          |
| `scan`      | ?     | `_`         | `_ a`       | Reads a character from standard input and pushes it to the stack. |
