# Arithmetization

An arithmetization defines two things:
1. algebraic execution tables (AETs), and
1. arithmetic intermediate representation (AIR) constraints.
The nature of Triton VM is that the execution trace is spread out over multiple tables, but linked through permutation and evaluation arguments.

Elsewhere, the acronym AET stands for algebraic execution *trace*.
In the nomenclature of this note, a trace is a special kind of table that tracks the values of a set of registers across time.

The values of all registers, and consequently the elements on the stack, in memory, and so on, are elements of the _B-field_, i.e., $\mathbb{F}_p$ where $p=2^{64}-2^{32}+1$.
All values of columns corresponding to one such register are elements of the B-Field as well.
The entries of a table's columns corresponding to Evaluation or Permutation Arguments are elements from the _X-Field_ $\mathbb{F}_{p^3}$.

For each table, up to three lists containing constraints of different type are given:
1. Consistency Constraints, establishing consistency within any given row,
1. Boundary Constraints, defining values in a table's first row and, in some cases, also the last, and
1. Transition Constraints, establishing the consistency of two consecutive rows in relation to each other.

Together, all these constraints constitute the AIR constraints.
