# Memory Consistency

Triton has three memory-like units: the RAM, the jump stack, and the op stack. Each such unit has a corresponding table. *Memory-consistency* is the property that whenever the processor reads a data element from these units, the value that it receives corresponds to the value sent the last time when that cell was written. Memory consistency follows from two intermediate properties.

 1. Contiguity of regions of constant memory pointer. After selecting from each table all the rows with any given memory pointer, the resulting sublist is contiguous, which is to say, there are no gaps and two such sublists can never interleave.
 2. Inner-sorting within contiguous regions. Within each such contiguous region, the rows are sorted in ascending order by clock cycle.

The contiguity of regions of constant memory pointer is established differently for the RAM Table than for the Op Stack or Jump Stack Tables. The Op Stack and Jump Stack Tables enjoy a particular property whereby the memory pointer can only ever increase or decrease by one (or stay the same). As a result, simple AIR constraints can enforce the correct sorting by memory pointer. In contrast, the memory pointer for the RAM Table can jump arbitrarily.
As explained in the [next section](contiguity-of-memory-pointer-regions.md), a Contiguity Argument establishes contiguity.

The correct inner sorting is established the same way for all three memory-like tables.
Their respective clock jump differences – differences of clock cycles within regions of constant memory pointer – are shown to be contained in the set of all clock cycles through [Lookup Arguments](lookup-argument.md).
Under reasonable assumptions about the running time, this fact implies that all clock jumps are directed forwards, as opposed to backwards, which in turn implies that the rows are sorted for clock cycle.

The next sections elaborate on these constructions.
A [dedicated section](proof-of-memory-consistency.md) shows that these two properties do indeed suffice to prove memory consistency.
