# Triton VM Instruction Set Architecture

Triton VM is a stack machine with RAM, with a second data structure for evaluating a sponge permutation. The arithmetization of the VM is defined over the *B-field* $\mathbb{F}_p$ where $p=2^{64}-2^{32}+1$, meaning that the registers and memory elements take values from this field, and that the transition function gives rise to low-degree transition verification polynomials from the ring of multivariate polynomials over this field.

## Data Structures

**Memory** The term *memory* refers to a data structure that gives read access (and possibly write access, too) to elements indicated by an *address*. The address lives in the same field. There are four separate notions of memory:
 1. *RAM*, to which the VM can read and write field elements.
 2. *Program Memory*, from which the VM reads instructions.
 3. *OpStack Memory*, which stores the part of the operational stack that is not represented explicitly by the operational stack registers.
 4. *Jump Stack Memory*, which stores the entire jump stack.

**OpStack** The stack is a first-in;first-out data structure that allows the program to store intermediate variables, pass arguments, and keep pointers to objects held in RAM.

**Jump Stack** Another first-in;first-out data structure that keeps track of return and destination addresses. This stack changes only when control follows a `call` or `return` instruction.

## Registers

This section covers all columns in the Protocol Table. Only a subset of these registers relate to the instruction set; the remaining registers exist only to enable an efficient arithmetization and are marked with an asterisk.

| Register | Name | Purpose |
|----------|------|---------|
| *`clk` | cycle counter | counts the number of cycles the program has been running for |
| `ip` | instruction pointer | contains the memory address (in Program Memory) of the instruction |
| `ci` | current instruction register | contains the full instruction |
| `ni` | next instruction register | contains the full instruction of the next cycle; to be used for immediate instructions |
| *`ib0` through `ib5` | instruction bit | contains the ith bit of the instruction |
| *`if0` through `if9` | instruction flags | used as intermediate values to keep the AIR degree low |
| `jsp` | jump stack pointer | contains the memory address (in jump stack memory) of the top of the jump stack |
| `jsv` | return address | contains the value of the top of the jump stack |
| `st0` through `st3` | operational stack registers | contain explicit operational stack values |
| *`inv` | zero indicator | assumes the inverse of the the top of the stack when it is nonzero, and zero otherwise |
| *`osp` | operational stack pointer | contains the memory address (in stack memory) of the top of the operational stack minus 4 |
| *`osv` | operational stack value | contains the (stack) memory value at the given address |
| *`hv0` through `hv4` | helper variable registers | helper variables for some arithmetic operations |
| *`ramp` | RAM pointer | contains an address (in RAM memory) for reading or writing |
| *`ramv` | RAM value | contains the value of the RAM memory element at the given adress |
| `aux0` through `aux15` | auxiliary registers | data structure dedicated to hashing instructions |
| | | |
| 52 in total | | |

**Instruction** The instruction is represented by one register called the *current instruction register* `ci`. This value then decomposed into its 6 constituent bits, giving rise to 6 *instruction bit registers*, labeled `ib0` through `ib5`. Additionally, there is a register called the *instruction pointer* (`ip`), which contains the address of the current instruction in instruction memory. Also, there is the *next instruction register* that contains the next instruction. For immediate instructions, this register takes the value of the next instruction.

**Stack** The stack is represented by four registers called *stack registers* (`st0` -- `st3`), which take the values of the top four stack elements. In addition to these registers, there a *stack pointer* register (`sp`) whose value is the memory address of the top of the stack minus four. The stack grows upwards, in line with the metaphor that justifies the name "stack".

**RAM**

**Helper Variables** Some instructions require helper variables in order to generate an efficient arithmetization. To this end, there are 5 helper registers, labeled `hv0` through `hv4`. These registers are part of the arithmetization of the architecture, but not needed to define the instruction set.

## Instructions

**OpStack Manipulation.**

In this section *stack* is short for *operational stack*.

| Instruction | Value | Effect on OpStack | Description |
|-|-|-|-|
| `pop` | ? | `stack t  -->  stack` | Pops top element from stack. |
| `push` + `arg` | ? | `stack  -->  stack arg` | Pushes `arg` onto the stack. |
| `pad` | ? | `stack --> stack a` | Pushes a nondeterministic element `a` to the stack. |
| `dup` + `arg` | ? | e.g., `stack a b c d  -->  stack a b c d a` | Duplicates the element `arg` positions away from the top. |
| `pull` + `arg` | ? | e.g., `stack a b c d --> stack a c d b` | Moves the element `arg` positions away from the top, to the top. |

**Control Flow**

| Instruction | Value | Effect on OpStack | Description |
|-|-|-|-|
| `nop` | 0 | identity | Do nothing, just continue to next instruction. |
| `skiz` | ? | `stack top  -->  stack` | Skip next instruction if `top` is zero. |
| `call` + `addr` | ? | identity | Push `(ci+2,addr)` to the jump stack, and jump to absolute immediate address `addr` |
| `return` | ? | identity | Pop one pair off the jump stack and jump to that pair's return address (which is the first element). |
| `recurse` | ? | identity | Peek at the top pair of the jump stack and jump to that pair's destination address (which is the second element). |
| `assert` | ? | `stack a  -->  stack` | Halts and fails if not `a == 1`. |
| `halt` | ? | identity | Solves the halting problem (if the instruction is reached). |

**Memory Access**

| Instruction | Value | Effect on OpStack | Description |
|-|-|-|-|
| `load` | ? | `stack mem  -->  stack mem val` | Reads a value `val` from RAM at the location pointed to by the register `ramp`, and pushes the read element to the opstack. |
| `loadinc` | ? | `stack mem --> stack mem val` | Same as above but additionally increases register `ramp` by one. |
| `loaddec` | ? | `stack mem --> stack mem val` | Same as above bug decrementing instead of incrementing. |
| `save` | ? | `stack mem val  -->  stack mem` | Writes value `val` to RAM at the location pointed to `ramp`, and pops the top of the opstack. |
| `saveinc` | ? | `stack mem val  -->  stack mem` | Same as above but additionally increases register `ramp` by one. |
| `savedec` | ? | `stack mem val  -->  stack mem` | Same as above but decrementing instead of incrementing. |
| `setramp` | ? | `stack mem --> stack` | Pops the top of the opstack and sets the `ramp` register to this value. |
| `getramp` | ? | `stack --> stack mem` | Pushes the value of the `ramp` register to the stack. |

**Auxiliary Register Instructions**

| Instruction | Value | Effect on OpStack | Description |
|-|-|-|-|
| `xlix` | ? | identity | Applies the Rescue-XLIX permutation to the auxiliary registers. |
| `clearall` | ? | identity | Sets all auxiliary registers to zero. |
| `squeeze` + `arg` | ? | `stack --> stack a` | Pushes to the stack the `arg`th auxiliary register. |
| `absorb` + `arg` | ? | `stack a --> stack` | Pops the top off the opstack and adds it into the `arg`th auxiliary regiser. |
| `clear` + `arg` | ? | identity | Sets the `arg`th auxiliary register to zero. |
| `rotate` + `arg` | ? | identity | Rotate the auxiliary registers by `arg` positions. |

**Arithmetic on Stack**

| Instruction | Value | Effect on OpStack | Description |
|-|-|-|-|
| `add` | ? | `stack a b  -->  stack c` | Computes the sum (`c`) of the top two elements of the stack (`b` and `a`) over the field. | 
| `neg` | ? | `stack a  -->  stack b` | Computes the negation (over the field) of the top element of the stack. |
| `mul` | ? | `stack a b  -->  stack c` | Computes the product (`c`) of the top two elements of the stack (`b` and `a`) over the field. |
| `inv` | ? | `stack a  -->  stack b` | Computes the multiplicative inverse (over the field) of the top of the stack. If the argument `a` is zero, the result `b` will be zero as well. |
| `lnot` | ? | `stack a  -->  stack  (1-a)` | Computes the logical negation of the top stack element, assuming it is 0 or 1 |
| `split` | ? | `stack a  -->  stack lo hi` | Decomposes the top of the stack into the lower 32 bits and the upper 32 bits, without making any assumptions about the top stack element. |
| `eq` | ? | `stack a b  -->  stack (a == b)` | Tests the top two stack elements for equality. |
| `lt` | ? | `stack a b  -->  stack (a < b)` | Tests if the one-from top element is less than or equal the top element on the stack, assuming both are 32-bit integers. |
| `and` | ? | `stack a b  -->  stack (a and b)` | Computes the bitwise-and of the top two stack elements, assuming both are 32-bit integers. |
| `or` | ? | `stack a b  -->  stack (a or b)` | Computes the bitwise-or of the top two stack elements, assuming both are 32-bit integers. |
| `xor` | ? | `stack a b  -->  stack (a xor b)` | Computes the bitwise-xor of the top two stack elements, assuming both are 32-bit integers. |
| `reverse` | ? | `stack a  -->  stack b` | Flips the bit expansion of the top stack element, assuming it is a 32-bit integer. |
| `div` | ? | `stack a b  -->  stack c d` | Computes division with remainder of the top two stack elements, assuming the arguments are positive 32-bit integers. The result satisfies `a == c * b + d` and `d < b` and `c <= a`. |

**Input/Output**

| Instruction | Value | Effect on OpStack | Description |
| - | - | - | - |
| `print` | ? | `stack a  -->  stack` | Writes character `a` to standard output. |
| `scan` | ? | `stack  -->  stack a` | Reads a character from standard input and pushes it to the stack. |
