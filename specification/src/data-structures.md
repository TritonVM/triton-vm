# Data Structures

## Memory
The term *memory* refers to a data structure that gives read access (and possibly write access, too) to elements indicated by an *address*.
Regardless of the data structure, the address lives in the B-field.
There are four separate notions of memory:
1. *RAM*, to which the VM can read and write field elements.
2. *Program Memory*, from which the VM reads instructions.
3. *OpStack Memory*, which stores the operational stack.
4. *Jump Stack Memory*, which stores the entire jump stack.

## Operational Stack
The stack is a last-in;first-out data structure that allows the program to store intermediate variables, pass arguments, and keep pointers to objects held in RAM.
In this document, the operational stack is either referred to as just “stack” or, if more clarity is desired, “OpStack.”

From the Virtual Machine's point of view, the stack is a single, continuous object.
The first 16 elements of the stack can be accessed very conveniently.
Elements deeper in the stack require removing some of the higher elements, possibly after storing them in RAM.

For reasons of arithmetization, the stack is actually split into two distinct parts:
1. the _operational stack registers_ `st0` through `st15`, and
1. the _OpStack Underflow Memory_.

The motivation and the interplay between the two parts is described and exemplified in [arithmetization of the OpStack table](operational-stack-table.md).

## Jump Stack
Another last-in;first-out data structure that keeps track of return and destination addresses.
This stack changes only when control follows a `if_then_call`, `call`, or `return` instruction.
