# Memory Consistency

Triton has three memory-like units: the RAM, the JumpStack, and the OpStack. Each such unit has a corresponding table. *Memory-consistency* is the property that whenever the processor reads a data element from these units, the value that it receives corresponds to the value sent the last time when that cell was written. Memory consistency follows from two intermediate properties.

 1. Contiguity of regions of constant memory pointer. After selecting from each table all the rows with any given memory pointer, the resulting sublist is contiguous, which is to say, there are no gaps and two such sublists can never interleave.
 2. Inner-sorting within contiguous regions. Within each such contiguous region, the rows are sorted in ascending order by clock cycle.

The contiguity of regions of constant memory pointer is established differently for the RAM Table than for the OpStack or JumpStack Tables. The OpStack and JumpStack Tables enjoy a particular property whereby the memory pointer can only ever increase or decrease by one (or stay the same). As a result, simple AIR constraints can enforce the correct sorting by memory pointer. In contrast, the memory pointer for the RAM Table can jump arbitrarily. As explained below, an argument involving formal derivatives and Bézout's relation establishes contiguity.

The correct inner sorting is establish the same way for all three memory-like tables. The set of all clock jump differences – differences greater than 1 of clock cycles within regions of constant memory pointer – is shown to be contained in the set of all clock cycles. Under reasonable assumptions about the running time, this fact implies that all clock jumps are directed forwards, as opposed to backwards, which in turn implies that the rows are sorted for clock cycle.

The next sections elaborate on these constructions. Another section shows that these two properties do indeed suffice to prove memory consistency.

*Historical note.* The constructions documented here correspond to Triton Improvement Proposals (TIPs) [0001](https://github.com/TritonVM/triton-vm/blob/master/tips/tip-0001/tip-0001.md) and [0003](https://github.com/TritonVM/triton-vm/blob/master/tips/tip-0003/tip-0003.md). This note attempts to provide a self-contained summary.
