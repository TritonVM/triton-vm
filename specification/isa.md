# Triton VM Instruction Set Architecture

Triton VM is a stack machine with RAM, with a second data structure for evaluating a sponge permutation. The arithmetization of the VM is defined over the *B-field* $\mathbb{F}_p$ where $p=2^{64}-2^{32}+1$, meaning that the registers and memory elements take values from this field, and that the transition function gives rise to low-degree transition verification polynomials from the ring of multivariate polynomials over this field.

## Data Structures

**Memory** The term *memory* refers to a data structure that gives read access (and possibly write access, too) to elements indicated by an *address*. The address lives in the same field. There are four separate notions of memory:
 1. *RAM*, to which the VM can read and write field elements.
 2. *Instruction Memory*, from which the VM reads instructions.
 3. *Operational Stack Memory*, which stores the part of the operational stack that is not represented by the operational stack registers.
 4. *Return Address Stack Memory*, which stores the entire return address stack.

**Operational Stack** The stack is a first-in;first-out data structure that allows the program to store intermediate variables, pass arguments, and keep pointers to objects held in RAM.

**Return Address Stack** Another first-in;first-out data structure that keeps track of return addresses only. This stack changes only when control follows a `call` or `return` instruction.

**SIMD Cache** The SIMD cache is a list of 16 variables designated for performing vectorized instructions. It supports the following operations and access mechanisms.
 - Vectorized addition and multiplication over the B-field.
 - Reading and writing individual registers to and from stack.
 - Reading words (=4 elements) from RAM and writing them back.
 - Shifting elements up or down (and padding with zeros on the other side).
 - Splitting the top 4 elements into 16 chunks in total of 16 bits each.
 - Applying the Rescue-XLIX permutation.
 - Applying uniform-to-Gaussian resampling of all 16 variables.

 **Non-Determinism** The virtual machine has access to a tape of non-deterministic field elements. 

## Registers

| Register | Name | Purpose |
|----------|------|---------|
| `cir` | current instruction register | contains the full instruction |
| `pir` | previous instruction register | contains the full instruction of the last cycle; to be used for immediate instructions |
| `clk` | cycle counter | counts the number of cycles the program has been running for |
| `ib0` through `ib5` | instruction bit | contains the ith bit of the instruction |
| `ip` | instruction pointer | contains the memory address (in instruction memory) of the instruction |
| `rp` | RAM pointer | contains an address (in RAM memory) for reading or writing |
| `rasp` | return address stack pointer | contains the memory address (in return address stack memory) of the top of the return address stack |
| `ra` | return address | contains the value of the top of the return address stack |
| `osp` | operational stack pointer | contains the memory address (in stack memory) of the top of the operational stack minus 4 |
| `os0` through `os3` | operational stack registers | contain explicit operational stack values |
| `hv0` through `hv4` | helper variable registers | helper variables for some arithmetic operations |
| `sc0` through `sc15` | SIMD cache registers | data structure dedicated to vector instructions |
| | | |
| 39 in total | | |

**Instruction** The instruction is represented by one register called the *current instruction register* `cir`. This value then decomposed into its 6 constituent bits, giving rise to 6 *instruction bit registers*, labeled `ib0` through `ib5`. Additionally, there is a register called the *instruction pointer* (`ip`), which contains the address of the current instruction in instruction memory. Also, there is the *previous instruction register* that contains the previous instruction. For immediate instructions, this register takes the value of the previous instruction; for non-immedate instructions, this register takes the value 0.

**Stack** The stack is represented by four registers called *stack registers* (`st0` -- `st3`), which take the values of the top four stack elements. In addition to these registers, there a *stack pointer* register (`sp`) whose value is the memory address of the top of the stack minus four. The stack grows upwards, in line with the metaphor that justifies the name "stack".

**RAM** The virtual machine can write to and read from any RAM address, so long as this address is a field element. RAM can be read and written in chunks of 4 elements ("words"), or in individual elements.

**Helper Variables** Some instructions require helper variables in order to generate an efficient arithmetization. To this end, there are 5 helper registers, labeled `hv0` through `hv4`. These registers are part of the arithmetization of the architecture, but not needed to define the instruction set.

**SIMD Cache** The SIMD cache is represented explicitly by 16 registers, labeled `sc0` through `sc16`.

## Instructions

**Operational Stack Manipulation.**

In this section *stack* is short for *operational stack*.

| Instruction | Value | Effect on Stack | Description |
|-|-|-|-|
| `pop` | ? | `stack t  -->  stack` | Pops top element from stack. |
| `push` + `arg` | ? | `stack  -->  stack arg` | Pushes a `0` onto the stack, and in the next cycle the indicated element from instruction memory is added to the top element. |
| `pad` | ? | `stack --> stack a` | Pushes a nondeterministic element `a` to the stack. |
| `dup0` | ? | `stack a  -->  stack a a` | Duplicates the top stack element. |
| `dup1` | ? | `stack a b  -->  stack a b a` | Duplicates the element 1 position away from the top. |
| `dup2` | ? | `stack a b c  -->  stack a b c a` | Duplicates the element 2 positions away from the top. |
| `dup3` | ? | `stack a b c d  -->  stack a b c d a` | Duplicates the element 3 positions away from the top. |
| `swap` | ? | `stack a b  -->  stack b a` | Swaps the top two stack elements. |
| `pull2` | ? | `stack a b c  -->  stack b c a` | Moves the stack element located at 2 positions from the top, to the top. |
| `pull3` | ? | `stack a b c d  -->  stack b c d a` | Moves the stack element located at 2 positions from the top, to the top. |

**Control Flow**

| Instruction | Value | Effect on Stack | Description |
|-|-|-|-|
| `nop` | 0 | identity | Do nothing, just continue to next instruction. |
| `skiz` | ? | `stack top  -->  stack` | Skip next instruction if `top` is zero. |
| `jumpa` + `mem` | ? | `stack  -->  stack mem` | Set the instruction pointer `ip` to the immediate argument `mem` for jumping to an absolute instruction address. |
| `jumpr` + `mem` | ? | `stack  -->  stack mem` | Set the instruction pointer `ip` to `ip + mem` for jumping to a relative instruction address. |
| `call` + `addr` | ? | identity | Push current instruction pointer plus one to the return address stack, and jump to absolute immediate address `addr` |
| `return` | ? | identity | Pop one element off the return address stack and jump there. |
| `assert` | ? | `stack a  -->  stack` | Halts and fails if not `a == 1`. |
| `halt` | ? | identity | Solves the halting problem (if the instruction is reached). |

**Memory Access**

| Instruction | Value | Effect on Stack | Description |
|-|-|-|-|
| `read` | ? | `stack mem  -->  stack mem val` | Reads a value `val` from RAM at the location pointed to by the top of the stack `mem`, and pushes the read element to the stack. |
| `write` | ? | `stack mem val  -->  stack mem` | Writes value `val` to RAM at the location pointed to `mem`, and pops the top. |

**SIMD Instructions**

| Instruction | Value | Effect on Stack | Description |
|-|-|-|-|
| `xlix` | ? | identity | Applies the Rescue-XLIX permutation to the SIMD cache. |
| `ntt` | ? | identity | Applies the NTT to the SIMD cache. |
| `intt` | ? | identity | Applies the inverse NTT to the SIMD cache. |
| `shift` + `shamt` | ? | identity | Shifts the values of the SIMD registers up or down by immediate argument `shamt` |
| `zero` + `arg` | ? | identity | Sets the lowest `arg`-many SIMD registers to zero. |
| `vpad` + `arg` | ? | identity | Sets the top `arg`-many SIMD registers to nondeterministic values. |
| `vswap` + `arg` | ? | identity | Switches the top `arg`-many SIMD registers with the second `arg`-many SIMD registers. |
| `vsplit` | ? | identity | Splits the top 4 registers into 16 values of 16 bits, using all registers. |
| `vadd` | ? | identity | Sets the top 8 registers to the sum of the top 8 registers plus the bottom 8 registers. |
| `vmul` | ? | identity | Sets the top 8 registers to the element-wise product of the top 8 registers plus the bottom 8 registers. |
| `sum` | ? | `stack  -->  stack sum` | Pushes to the stack the sum of all SIMD register values. |
| `vread` + `idx` | ? | `stack mem  -->  stack mem` | Reads a word from RAM whose first element has address `(mem >> 2) << 2`, and puts it into the top 4 elements (if `idx == 0`), second-from-top 4 elements (if `idx == 1`), third-from-top 4 elements (if `idx == 2`), or bottom-most 4 elements (if `idx == 3`). |
| `vwrite` + `idx` | ? | `stack mem  -->  stack mem` | Wites a word, defined by the top 4 elements (if `idx == 0`), second-from-top 4 elements (if `idx == 1`), third-from-top 4 elements (if `idx == 2`), or bottom-most 4 elements (if `idx == 3`), to RAM at address `(mem >> 2) << 2`. |
| `get` + `idx` | ? | `stack  -->  stack val` | Pushes the value of SIMD register `idx` to the stack. |
| `set` + `idx` | ? | `stack val  -->  stack` | Pops the top of the stack and sets the SIMD register `idx` to this value. |
| `addinto` + `idx` | ? | `stack val  -->  stack val` | Reads the top of the stack and adds it into the SIMD register `idx`. |
| `vgauss` | ? | identity | Computes a uniform-to-gaussian resampling of SIMD registers, assuming they are uniform 16-bit integers. |

**Arithmetic on Stack**

| Instruction | Value | Effect on Stack | Description |
|-|-|-|-|
| `add` | ? | `stack a b  -->  stack c` | Computes the sum (`c`) of the top two elements of the stack (`b` and `a`) over the field. | 
| `neg` | ? | `stack a  -->  stack b` | Computes the negation (over the field) of the top element of the stack. |
| `mul` | ? | `stack a b  -->  stack c` | Computes the product (`c`) of the top two elements of the stack (`b` and `a`) over the field. |
| `inv` | ? | `stack a  -->  stack b` | Computes the multiplicative inverse (over the field) of the top of the stack. If the argument `a` is zero, the result `b` will be zero as well. |
| `split` | ? | `stack a  -->  stack lo hi` | Decomposes the top of the stack into the lower 32 bits and the upper 32 bits, without making any assumptions about the top stack element. |
| `eq` | ? | `stack a b  -->  stack (a == b)` | Tests the top two stack elements for equality. |
| `lt` | ? | `stack a b  -->  stack (a < b)` | Tests if the one-from top element is less than or equal the top element on the stack, assuming both are 32-bit integers. |
| `and` | ? | `stack a b  -->  stack (a and b)` | Computes the bitwise-and of the top two stack elements, assuming both are 32-bit integers. |
| `or` | ? | `stack a b  -->  stack (a or b)` | Computes the bitwise-or of the top two stack elements, assuming both are 32-bit integers. |
| `xor` | ? | `stack a b  -->  stack (a xor b)` | Computes the bitwise-xor of the top two stack elements, assuming both are 32-bit integers. |
| `reverse` | ? | `stack a  -->  stack b` | Flips the bit expansion of the top stack element, assuming it is a 32-bit integer. |
| `div` | ? | `stack a b  -->  stack c d` | Computes division with remainder of the top two stack elements, assuming the arguments are positive 32-bit integers. The result satisfies `a == c * b + d` and `d < b` and `c <= a`. |
| `gauss` | ? | `stack a  -->  stack b` | Computes a uniform-to-gaussian resampling of the top stack element, assuming it is a uniform 16-bit integer. |
