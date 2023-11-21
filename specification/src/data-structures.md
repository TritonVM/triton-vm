# Data Structures

## Memory

The term *memory* refers to a data structure that gives read access (and possibly write access, too) to elements indicated by an *address*.
Regardless of the data structure, the address lives in the B-field.
There are four separate notions of memory:

1. *Program Memory*, from which the VM reads instructions.
1. *Op Stack Memory*, which stores the operational stack.
1. *RAM*, to which the VM can read and write field elements.
1. *Jump Stack Memory*, which stores the entire jump stack.

## Program Memory

Program memory holds the instructions of the program currently executed by Triton VM.
It is immutable.

## Operational Stack

The stack is a last-in;first-out data structure that allows the program to store intermediate variables, pass arguments, and keep pointers to objects held in RAM.
In this document, the operational stack is either referred to as just “stack” or, if more clarity is desired, “op stack.”

From the Virtual Machine's point of view, the stack is a single, continuous object.
The first 16 elements of the stack can be accessed very conveniently.
Elements deeper in the stack require removing some of the higher elements, possibly after storing them in RAM.

For reasons of arithmetization, the stack is actually split into two distinct parts:
1. the _operational stack registers_ `st0` through `st15`, and
1. the _Op Stack Underflow Memory_.

The motivation and the interplay between the two parts is described and exemplified in [arithmetization of the Op Stack table](operational-stack-table.md).

## Random Access Memory

Triton VM has dedicated Random Access Memory.
It can hold up to $p$ many base field elements, where $p$ is the Oxfoi prime[^1].
Programs can read from and write to RAM using [instructions](instructions.md#memory-access) `read_mem` and `write_mem`.

The initial RAM is determined by the entity running Triton VM.
Populating RAM this way can be beneficial for a program's execution and proving time, especially if substantial amounts of data from the input streams needs to be persisted in RAM.
This initialization is one form of secret input, and one of two mechanisms that make Triton VM a non-deterministic virtual machine.
The other mechanism is [dedicated instructions](instructions.md#opstack-manipulation).

## Jump Stack

Another last-in;first-out data structure that keeps track of return and destination addresses.
This stack changes only when control follows a `call` or `return` instruction.
Furthermore, executing instruction `recurse` requires a non-empty jump stack.

---

[^1]:
Of course, the machine running Triton VM might have stricter limitations:
storing or accessing $(2^{64} - 2^{32} + 1)\cdot 63.99$ bits $\approx 148$ exabytes of data is a non-trivial engineering feat.
