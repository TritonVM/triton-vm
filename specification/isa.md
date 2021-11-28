# Triton VM Instruction Set Architecture

Triton VM is a stack machine with RAM, with a second data structure for evaluating a sponge permutation. The arithmetization of the VM is defined over the *Bobbin Field* $\mathbb{F}_p$ where $p=2^{64}-2^{32}+1$, meaning that the registers and memory elements take values from this field, and that the transition function gives rise to low-degree transition verification polynomials from the ring of multivariate polynomials over this field.

## Data Structures

**Memory** The term *memory* refers to a data structure that gives read access (and possibly write access, too) to elements indicated by an *address*. The address lives in the same field. There are three separate notions of memory:
 1. *RAM*, to which the VM can read and write field elements.
 2. *Instruction Memory*, from which the VM reads instructions.
 3. *Stack Memory*, which stores the part of the stack that is not represented by the stack registers.

**Stack** The stack is a first-in;first-out data structure that allows the program to store intermediate variables and keep track of the location in the concrete syntax tree.

**Hash Cache** The hash cache is a list of 12 variables designated for evaluating the Rescue-XLIX permutation. It has the following access structure:
 - *Absorb* moves all elements down and takes the top element from the stack and puts it on the top of the hash cache. The bottom-most element will be deleted. Note that there is no addition or xor happening.
 - *Squeeze* moves all elements up and padds a zero from below. The top element is moved from the hash cache to the stack.
 - *Switch* switches the first 4 elements with the second 4 elements.
 - *XLIX* applies the Rescue-XLIX permutation.

 **Non-Determinism** The virtual machine has access to a tape of non-deterministic field elements. 

## Registers

| Register | Name | Purpose |
|----------|------|---------|
| `ir` | instruction register | contains the full instruction |
| `ib0` through `ib5` | instruction bit | contains the ith bit of the instruction |
| `ip` | instruction pointer | contains the memory address (in instruction memory) of the instruction |
| `rp` | RAM pointer | contains an address (in RAM memory) for reading or writing |
| `sp` | stack pointer | contains the memory address (in stack memory) of the top of the stack minus 4 |
| `st0` through `st3` | stack registers | contain explicit stack values |
| `hv0` through `hv4` | helper variable registers | helper variables for some arithmetic operations |
| `hc0` through `hc11` | hash cache registers | data structure dedicated to hash evaluations |
| | | |
| 30 in total | | |

**Instruction** The instruction is represented by one register called the *instruction register* `ir`. This value then decomposed into its 6 constituent bits, giving rise to 6 *instruction bit registers*, labeled `ib0` through `ib5`. Additionally, there is a register called the *instruction pointer* (`ip`), which contains the address of the instruction in instruction memory.

**Stack** The stack is represented by four registers called *stack registers* (`st0` -- `st3`), which take the values of the top four stack elements. In addition to these registers, there a *stack pointer* register (`sp`) whose value is the memory address of the top of the stack minus four. The stack grows upwards, in line with the metaphor that justifies the name "stack".

**RAM** The virtual machine can write to and read from any RAM address, so long as this address is a field element. There is a dedicated register called the *RAM pointer* (`rp`) that contains this address value.

**Helper Variables** Some instructions require helper variables. To this end, there are 5 helper registers, labeled `hv0` through `hv4`.

**Hash Cache** The hash cache is represented explicitly by 12 registers, labeled `hc0` through `hc11`.

## Instructions

| Instruction | Value | Effect on Stack | Description |
|-|-|-|-|
| `pop` | ? | `stack || t  -->  stack` | Pops top element from stack. |
| `push` + `arg` | ? | `stack  -->  stack || arg` | Pushes a `0` onto the stack, and in the next cycle the indicated element from instruction memory is added to the top element. |
| `skiz` | ? | `stack || top  -->  stack || top` | Skip next instruction if `top` is zero. |
| `jump` | ? | `stack || mem  -->  stack || mem` | Set the instruction pointer `ip` to `mem`. |
| `getip` | ? | `stack  -->  stack || ip` | Pushes to the stack the value of the instruction pointer. |
| `read` | ? | `stack  -->  stack || val` | Reads a value `val` from RAM at the location pointed to by the RAM pointer `rp`, and pushes this value to the stack. |
| `write` | ? | `stack || val  -->  stack` | Writes value `val` to RAM at the location pointed to by the RAM pointer `rp`. |
| `setrp` | ? | `stack || mem  -->  stack` | Sets the RAM pointer `rp` to the top of the stack, and pops this element. |
| `getrp` | ? | `stack  -->  stack || mem` | Pushes to the stack the value of the RAM pointer `rp`. |
| `absorb` | ? | `stack || top  -->  stack` | Moves the top of the stack to the hash cache. |
| `squeeze` | ? | `stack  -->  stack || val` | Moves the top element of the hash cache to the stack. |
| `switch` | ? | identity | Switches first 4 elements of the hash cache with the second 4 elements. |
| `xlix` | ? | identity | Applies the Rescue-XLIX permutation to the hash cache. |
| `dup0` | ? | `stack || a  -->  stack || a || a` | Duplicates the top stack element. |
| `dup1` | ? | `stack || a || b  -->  stack || a || b || a` | Duplicates the element 1 position away from the top. |
| `dup2` | ? | `stack || a || b || c  -->  stack || a || b || c || a` | Duplicates the element 2 positions away from the top. |
| `dup3` | ? | `stack || a || b || c || d  -->  stack || a || b || c || d || a` | Duplicates the element 3 positions away from the top. |
| `swap` | ? | `stack || a || b  -->  stack || b || a` | Swaps the top two stack elements. |
| `pull2` | ? | `stack || a || b || c  -->  stack || b || c || a` | Moves the stack element located at 2 positions from the top, to the top. |
| `pull3` | ? | `stack || a || b || c || d  -->  stack || b || c || d || a` | Moves the stack element located at 2 positions from the top, to the top. |
| `add` | ? | `stack || a || b  -->  stack || c` | Computes the sum (`c`) of the top two elements of the stack (`b` and `a`) over the field. | 
| `mul` | ? | `stack || a || b  -->  stack || c` | Computes the product (`c`) of the top two elements of the stack (`b` and `a`) over the field. |
| `inv` | ? | `stack || a  -->  stack || b` | Computes the multiplicative inverse (over the field) of the top of the stack. If the argument `a` is zero, the result `b` will be zero as well. ||
| `split` | ? | `stack || a  -->  stack || lo || hi` | Decomposes the top of the stack into the lower 32 bits and the upper 32 bits, without making any assumptions about the top stack element. |
| `eq` | ? | `stack || a || b  -->  stack || (a == b)` | Tests the top two stack elements for equality. |
| `gt` | ? | `stack || a || b  -->  stack || (a > b)` | Tests if the one-from top element is greater than the top element on the stack, assuming both are 32-bit integers. |
| `assert` | ? | identity | Fails if top of stack is zero. |
| `and` | ? | `stack || a || b  -->  stack || (a & b)` | Computes the bitwise-and of the top two stack elements, assuming both are 32-bit integers. |
| `or` | ? | `stack || a || b  -->  stack || (a | b)` | Computes the bitwise-or of the top two stack elements, assuming both are 32-bit integers. |
| `xor` | ? | `stack || a || b  -->  stack || (a ^ b)` | Computes the bitwise-xor of the top two stack elements, assuming both are 32-bit integers. |
| `reverse` | ? | `stack || a  -->  stack || b` | Flips the bit expansion of the top stack element, assuming it is a 32-bit integer. |
| `div` | ? | `stack || a || b  -->  stack || c || d` | Computes division with remainder of the top two stack elements, assuming the arguments are positive 32-bit integers. The result satisfies `a == c * b + d` and `d < b`. |
| `gauss` | ? | `stack || a  -->  stack || b` | Computes a uniform-to-gaussian resampling of the top stack element, assuming it is a uniform 16-bit integer. |
| `nondet` | ? | `stack  -->  stack || a` | Samples a field element non-deterministically and pushes it to the stack. |
