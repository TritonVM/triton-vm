# Proof of Memory Consistency

Whenever the Processor Table reads a value "from" a memory-like table, this value appears nondeterministically and is unconstrained by the main table AIR constraints. However, there is a permutation argument that links the Processor Table to the memory-like table in question. *The construction satisfies memory consistency if it guarantees that whenever a memory cell is read, its value is consistent with the last time that cell was written.*

The above is too informal to provide a meaningful proof for. Let's put formal meanings on the proposition and premises, before reducing the former to the latter.

Let $P$ denote the Processor Table and $M$ denote the memory-like table. Both have height $T$. Both have columns `clk`, `mp`, and `val`. For $P$ the column `clk` coincides with the index of the row. $P$ has another column `ci`, which contains the current instruction, which is `write`, `read`, or `any`. Obviously, `mp` and `val` are abstract names that depend on the particular memory-like table, just like `write`, `read`, and `any` are abstract instructions that depend on the type of memory being accessed. In the following math notation we use $\mathtt{col}$ to denote the column name and $\mathit{col}$ to denote the value that the column might take in a given row.

**Definition 1 (contiguity):** The memory-like table is *contiguous* iff all sublists of rows with the same memory pointer `mp` are contiguous. Specifically, for any given memory pointer $\mathit{mp}$, there are no rows with a different memory pointer $\mathit{mp}'$ in between rows with memory pointer $\mathit{mp}$.

$$ \forall i < j < k \in \lbrace 0, \ldots, T-1 \rbrace : \mathit{mp} \stackrel{\triangle}{=} M[i][\mathtt{mp}] = M[k][\mathtt{mp}] \Rightarrow M[j][\mathtt{mp}] = \mathit{mp} $$

**Definition 2 (regional sorting):** The memory-like table is *regionally sorted* iff for every contiguous region of constant memory pointer, the clock cycle increases monotonically.

$$ \forall i < j \in \lbrace 0, \ldots, T-1 \rbrace : M[i][\mathtt{mp}] = M[j][\mathtt{mp}] \Rightarrow M[i][\mathtt{clk}] <_{\mathbb{Z}} M[j][\mathtt{clk}] $$

The symbol $<_{\mathbb{Z}}$ denotes the integer less than operator, after lifting the operands from the finite field to the integers.

**Definition 3 (memory consistency):** A Processor Table $P$ has *memory consistency* if whenever a memory cell at location $\mathit{mp}$ is read, its value corresponds to the previous time the memory cell at location $\mathit{mp}$ was written. Specifically, there are no writes in between the write and the read, that give the cell a different value.

$$ \forall k \in \lbrace 0 , \ldots, T-1 \rbrace : P[k][\mathtt{ci}] = \mathit{read} \, \Rightarrow \left( (1) \, \Rightarrow \, (2) \right)$$

$$ (1) \exists i  \in \lbrace 0 , \ldots, k \rbrace : P[i][\mathtt{ci}] = \mathit{write} \, \wedge \, P[i+1][\mathtt{val}] = P[k][\mathtt{val}] \, \wedge \, P[i][\mathtt{mp}] = P[k][\mathtt{mp}]$$

$$ (2) \nexists j \in \lbrace i+1 , \ldots, k-1 \rbrace : P[j][\mathtt{ci}] = \mathit{write} \, \wedge \, P[i][\mathtt{mp}] = P[k][\mathtt{mp}] $$

**Theorem 1 (memory consistency):** Let $P$ be a Processor Table. If there exists a memory-like table $M$ such that

 - selecting for the columns `clk`, `mp`, `val`, the two tables' lists of rows are permutations of each other; and
 - $M$ is contiguous and regionally sorted; and
 - $M$ has no changes in `val` that coincide with clock jumps;

then $P$ has memory consistency.

*Proof.* For every memory pointer value $\mathit{mp}$, select the sublist of rows $P_{\mathit{mp}} \stackrel{\triangle}{=} \lbrace P[k] \, | \, P[k][\mathtt{mp}] = \mathit{mp} \rbrace$ in order. The way this sublist is constructed guarantees that it coincides with the contiguous region of $M$ where the memory pointer is also $\mathit{mp}$.

Iteratively apply the following procedure to $P_{\mathit{mp}}$: remove the bottom-most row if it does not correspond to a row $k$ that constitutes a counter-example to memory consistency. Specifically, let $i$ be the clock cycle of the previous row in $P_{\mathit{mp}}$.

 - If $i$ satisfies $(1)$ then by construction it also satisfies $(2)$. As a result, row $k$ is not part of a counter-example to memory consistency. We can therefore remove the bottom-most row and proceed to the next iteration of the outermost loop.
 - If $P[i][\mathtt{ci}] \neq \mathit{write}$ then we can safely ignore this row: if there is no clock jump, then the absence of a $\mathit{write}$-instruction guarantees that $\mathit{val}$ cannot change; and if there is a clock jump, then by assumption on $M$, $\mathit{val}$ cannot change. So set $i$ to the clock cycle of the row above it in $P_{\mathit{mp}}$ and proceed to the next iteration of the inner loop. If there are no rows left for $i$ to index, then there is no possible counterexample for $k$ and so remove the bottom-most row of $P_{\mathit{mp}}$ and proceed to the next iteration of the outermost loop.
 - The case $P[i+1][\mathtt{val}] \neq P[k][\mathtt{val}]$ cannot occur because by construction of $i$, $\mathit{val}$ cannot change.
 - The case $P[i][\mathtt{mp}] \neq P[k][\mathtt{mp}]$ cannot occur because the list was constructed by selecting only elements with the same memory pointer.
 - This list exhausts the possibilities of condition (1).

When $P_{\mathit{mp}}$ consists of only two rows, it can contain no counter-examples. By applying the above procedure, we can reduce every correctly constructed sublist $P_{\mathit{mp}}$ to a list consisting of two rows. Therefore, for every $\mathit{mp}$, the sublist $P_{\mathit{mp}}$ is free of counter-examples to memory consistency. Equivalently, $P$ is memory consistent. $\square$
