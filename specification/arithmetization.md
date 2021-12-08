# Triton VM Arithmetization

This document describes the arithmetization of Triton VM, whose instruction set architecture is defined [here](isa.md). An arithmetization defines two things: a algebraic execution tables (AETs) and arithmetic intermediate representation (AIR) constraints. The nature of Triton VM is that the execution trace is spread out over multiple tables; we treat each table separately.

Elsewhere, the acronym AET stands for algebraic execution *trace*. In the nomenclature of this note, a trace is a special kind of table that tracks the values of a set of registers across time.

## 16-bit Tables

A number of tables are used to show that certain values are in the range $[0;2^{16})$. The symbol $F$ is used to indicate an issue of *format*.

### Fixed

The Fixed 16-Bit Table contains all elements from 0 to 2^16 - 1, in that order. Every proof ships with it.

The Fixed 16-Bit Table comes with another column, whose purpose is to map the uniformly distributed 16-bit integer to a discrete Gaussian, with parameters suitable for lattice-based crypto operations.

| $F_{p16}$ | $F_{pg}$ |
|---------------|-------------|

The table is preprocessed, so no AIR is necessary. The table is used to verify copy-constraints; so there will be dynamic masks.

### Dynamic

The Dynamic 16-Bit Table consists of 2 variables. It contains all the values arising in the course of an execution whose correct format must be verified -- namely, it must be verified that these values consist of 16 bits or that their resampling to the Gaussian distribution was correct.

| $F_{d16}$ | $F_{dg}$ |
|-----------|----------|
| d | h |
| c | i |
| b | j |
| a | k |
| e | l |
| f | m |
| g | n |

A copy-constraint establishes that all the rows of the Dynamic 16-Bit Table are contained in the Fixed 16-Bit Table.

## Memory Tables

The general format of memory tables is as follows.

| Cycle | Instruction | Address | New Value | Old Value |
|-------|-------------|---------|-----------|-----------|
| - | - | - | - | - |
| - | - | - | - | - |
| - | - | - | - | - |

The rows are sorted by address, then by cycle. Consecutive rows with the same address can be verified to have consistent memory values, i.e., changing value only when the instruction is set to "write". Consecutive rows with distinct addresses represent distinct memory cells. The second row of such a pair is the first time the new memory cell is referenced, and so the Old Value of this row must be set to zero. A copy-constraint establishes that memory accesses of the register table are consistent with this table.

However, the specific memory tables used in Triton VM differ from this pattern in subtle but important ways.

### Program

Program Memory is read only, so the Old Value column can be dropped and the New Value column can be renamed to simple Value. Furthermore, since the value of the program memory does not change with time, there is nevery any need to store different values at the same address. So the address column contains simply $0$ through $\ell-1$, where $\ell$ is the program length. What's more, there already is a column in another table that counts the integers starting from 0, namely the cycle column in the Stack Register Trace. Furthermore, the Instruction and Cycle columns are not relevant any more. We are left with two columns.

| Address $P_a$ | Value $P_v$ |
|---------------|-------------|
| - | - |

The polynomials interpolate these two columns over a trace domain $\{ \omicron^i \, \vert \, 0 \leq i < 2^k \}$. The AIR constraints for this table are as follows:
 - boundary constraints:
     - address initialized to 0: $P_a(1) = 0$
     - first instruction is defined as 0: $P_v(1) = 0$;
 - transition constraints, for $X \in \{ \omicron^i \, | \, 0 \leq i < 2^k - 1\}$:
     - monotonicity of address increase $P_a(\omicron \cdot X) - P_a(X) - 1 = 0$

### Stack

The Stack Memory contains the underflow from the stack registers, i.e., the entire stack except for the top 4 elements. The difference with respect to the general format is that the instruction needs to indicate whether the stack is growing or shrinking. (When it stays the same size, there is no need to modify memory.) To this end, the table also includes the instruction bit registers. There is a low degree polynomial $f(\mathbf{X})$ that, as a function of these registers' values, indicates whether the stack grows or shrinks, by taking the zero or a nonzero value, respectively.

| Cycle $S_c$ | Instruction $S_i$ | Instr Bit 0 $S_{i0}$ | . . . | Instr Bit 5 $S_{i5}$ | Address $S_a$ | New Value $S_{nv}$ | Old Value $S_{ov}$ |
|-------|-------------|-------------|--------|-------------|---------|-----------|------|
| - | - | - |  | - | - | - |  - | - |

The polynomials interpolate these columns over a trace domain $\{\omicron^i \, \vert \, 0 \leq i < 2^k-1 \}$. The AIR constraints for this table are:
 - boundary constraints
    - boundary address: $S_a(1) = 0$
    - boundary cycle: $S_c(1) = 0$
    - initial value: $S_{ov}(1) = 0$
 - transition constraints, for $X \in \{\omicron^i \, \vert \, 0 \leq i < 2^k-1\}$
    - monotonicity of addresses: $(S_a(\omicron \cdot X) - S_a(X)) \cdot (S_a(\omicron \cdot X) - S_a(X) - 1) = 0$ 
    - conditional monotonicity of cycles: $(S_a(\omicron \cdot X) - S_a(X) - 1) \cdot (S_c(\omicron \cdot X) - S_c(X) - 1) = 0$
    - initial values of new memory cells: $(S_a(\omicron \cdot X) - S_a(X)) \cdot S_{ov}(\omicron \cdot X) = 0$
    - evolution of value in same-address cells: $(S_a(\omicron \cdot X) - S_a(X) - 1) \cdot f(\mathbf{X}) \cdot (S_{nv}(X) - S_{ov}(X)) = 0$

All columns except Old Value are involved in a copy-constraint with the Register Trace.

### RAM

The RAM is accessible in two ways: first, individual memory elements can be read to and written from the stack; second, chunks of four elements (words) can be written to and read from the SIMD register. To enable this, the address is split into the high part (everything but the least significant two bits), and low bart (least significant two bits).

| Cycle $R_c$ | Instr $R_i$ | IB0 $R_{i0}$ | ... | IB5 $R_{i5}$ | Addr $R_a$ | AddrHi $R_{ahi}$ | AddrLo $R_{alo}$ | NV0 $R_{nv0}$ | ... | NV3 $R_{nv3}$ | OV0 $R_{ov0}$ | ... | OV3 $R_{ov}$ | Val $R_{val}$ |
|------------|-------------------|----------------------|--------|---------------|------|-----|----|---------------|-------|----------------|-------|--------------|-|-|
| - | - | - |  | - | - | - | - | - |  | - | - |  | - | 

The polynomials interpolate the columns of this table over the trace domain $\{\omicron^i \, \vert \, 0 \leq i < 2^k\}$. Let $f_s(\mathbf{X})$ and $f_v(\mathbf{X})$ be the low degree polynomials that indicate write instructions from stack and SIMD registers, respectively. The AIR constraints are:
 - boundary constraints: 
    - initial values: $\forall j \in \{0, \ldots, 3\} \, . \, R_{ovj}(1) = 0$
 - consistency constraints, over $X \in \{\omicron^i \, \vert \, 0 \leq i < 2^k\}$:
    - valid decomposition of address: $4 \cdot R_{ahi}(X) + R_{alo}(X) - R_a(X) = 0$
    - valid least-significant two bits: $R_{alo}(X) \cdot ( 1 - R_{alo(X)} ) \cdot ( 2 - R_{alo}(X)) \cdot (3 - R_{alo}(X)) = 0$
 - transition constraints, for $X \in \{\omicron^i \vert \, 0 \leq i < 2^k-1\}$:
   - monotonicity of addresses: $R_a(\omicron \cdot X) - R_a(X) - 1 = 0$
   - conditional monotonicity of cycles: $(R_a(\omicron \cdot X) - R_a(X)) \cdot (R_c(\omicron \cdot X) - R_c(X)) = 0$
   - initial values of new memory cells: $\forall j \in \{0,\ldots,3\} \, . \, (R_a(\omicron \cdot X) - R_a(X)) \cdot R_{ovj}(\omicron \cdot X) = 0$
   - no changes unless instructed: $\forall j \in \{0, \ldots, 3\} \, . \, (R_{ovj}(X) - R_{nvj}(X)) \cdot ( f_s(\mathbf{X}) + f_v(\mathbf{X}) - 1 ) = 0$
   - when writing from stack, just the one register may change: $\forall j \in \{0, \ldots, 3\} \, . \, f_s(\mathbf{X}) \cdot (\prod_{j' \neq j} (R_{alo}(X) - j')) \cdot (R_{ovj}(X) - R_{nvj}(X)) = 0$

## Register Trace

The trace consists of 38 registers, Many of the names and functions are defined in the [instruction set architecture](isa.md). The remaining ones are:
 - `ibj` for `j` ranging from 0 to 5; these are the instruction bit registers. Each `ibj` should contain exactly one bit of the current instruction.
 - `if` immediate flag. The current instruction is not executed if this flag is set; instead, the previous is.

| `cir` $T_{cir}$ | `pir` $T_{pir}$ | `clk` $T_{clk}$ | `if` $T_{if}$ | `ib0` $T_{ib0}$ | ... | `ib5` $T_{ib5}$ | `ip` $T_{ip}$ | `rp` $T_{rp}$ | `sp` $T_{sp}$ | `st0` $T_{st0}$ | ... | `st3` $T_{st3}$ | `hv0` $T_{hv0}$ | ... | `hv4` $T_{hv4}$ | `sc0` $T_{sc0}$ | ... | `sc15` $T_{sc15}$ |
|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|-|

### Register Trace AIR

The columns are interpolated over the trace domain $\{\omicron^i \, | \, 0 \leq i < 2^k\}$. The AIR constraints follow.

#### Boundary constraints

Zero initial values: $T_*(1) = 0$ for all polynomials. This implies that the first instruction is zero and that the stack initially contains four zero elements.

#### Consistency constraints

The consistency constraints hold for all $X \in \{\omicron^i \, | \, 0 \leq i < 2^k\}$.
- binary instruction bits: $\forall j \in \{0, \ldots, 5\} \, . \, T_{ibj}(X) \cdot ( 1 - T_{ibj}(X) ) = 0$
- valid instruction decomposition: $T_{cir}(X) - \sum_{j=0}^{5} 2^j \cdot T_{ibj}(X) = 0$
- correct decomposition of stack top: $T_{st3}(X)) - \sum_{j=0}^{3} 2^{16 \cdot j} \cdot T_{hvj}(X) = 0$
- unique decomposition of stack top: $\big(T_{hv3}(X) \cdot (2^{16} \cdot T_{hv3}(X) + T_{hv2}(X)) - 1\big) \cdot (2^{16} \cdot T_{hv1}(X) + T_{hv0}(X)) = 0$

#### Transition Constraints Instruction Fetching and Control Flow

The transition constraints are valid for all $X \in \{\omicron^i \, | \, 0 \leq i < 2^k-1\}$.

   - monotonicity of cycle: $T_{clk}(\omicron \cdot X) - T_{clk}(X) = 0$
   - correct movement of instructions: $T_{pir}(\omicron \cdot X) - T_{cir}(X) = 0$
   - control flow, with indicator functions $f_{skiz}(\mathbf{X})$, $f_{jumpa}(X)$, and $f_{jumpr}(X)$
     - regular flow: $(1 - f_{skiz}(X) - f_{jumpa}(X) - f_{jumpr}(X)) \cdot (T_{ip}(\omicron \cdot X) - T_{ip}(X) - 1) = 0$

#### Transition Constraints for Stack

The transition constraints are valid for all $X \in \{\omicron^i \, | \, 0 \leq i < 2^k-1\}$.

   - stack
     - growth
     - shrinkage
     - manipulation

#### Transition Constraints for Native Arithmetic

The transition constraints are valid for all $X \in \{\omicron^i \, | \, 0 \leq i < 2^k-1\}$.

   - native arithmetic operations TODO
  

### Register Trace Copy-Constraints

Columns from the Register Trace are involved in the following copy-constraints:
 - For each cycle: $(T_{ip}(X), T_{cir}(X))$ with $(P_a(X), P_v(X))$ from the Program Memory Table.
 - For each of `lt`, `eq`, `and`, `or`, `xor`, `reverse`: $(T_{st1}(\omicron \cdot X), T_{st2}(\omicron \cdot X), T_{st3}(\omicron \cdot X))$ with matching rows and columns of the Uint32 Operations Table.
 - For each `div`:
   1. $(T_{st2}(X), T_{st2}(\omicron \cdot X))$ with matching rows in the Uint32 Operations Table, and the simulated column corresponding to `lt` + `eq`.
   2. $(T_{st3}(X), T_{st3}(\omicron \cdot X))$ with matching rows in the Uint32 Operations Table, and the column corresponding to `lt`.
 - For each or `read` and `write`: $(T_{clk}(X), T_{cir}(X), T_{ib0}(X), \ldots, T_{ib5}(X), T_{rp}(X), T_{st3}(X))$ with matching rows in the RAM Table.
 - For each `vread` and `vwrite`: $(T_{clk}(X), T_{pir}(X), T_{ib0}(X), \ldots, T_{ib5}(X), T_{rp}(X), T_{scj}(X), \ldots, T_{scj+3}(X))$ (where $j$ takes the value $0, 4, 8, 12$ depending on the immediate argument) with the matching rows in the RAM Table.
 - For each `split`, a batch-copy-constraint establishes that $(T_{hv0}(X), T_{hv1}(X), T_{hv2}(X), T_{hv3}(X))$ have matching rows in the Dynamic 16-Bit Table.
 - For each `vsplit`, a batch-copy-constraint establishes that $(T_{sc0}(X), T_{sc15}(X))$ have matching rows in the Dynamic 16-Bit Table.
 - For each `gauss`: $(T_{st3}(X), T_{st3}(\omicron \cdot X))$ with matching rows in the 16-Bit/Gauss Table.
 - For each `vgauss`, a batch-copy-constraint establishes that $(T_{sc0}(X),\ldots,T_{sc15}(X),T_{sc0}(\omicron \cdot X),\ldots,T_{sc15}(\omicron \cdot X))$ are in one-to-one correspondence with 16 consecutive rows of the 16-Bit/Gauss table $(G_i(X), G_o(X))$.
 - For each `vsplit`, and for each $j \in \{0, 4, 8, 12\}$: $(T_{scj}(X), \ldots, T_{scj+3}(X))$ with matching rows in the 4-Wide 32-Bit Table.
 - For each `xlix`: consecutive rows of $(T_{sc0}(X), \ldots T_{sc15}(X))$ with rows apart by 8 in the Rescue-XLIX Table.

## Auxiliary Tables

### Rescue-XLIX

The Rescue-XLIX table applies the Rescue-XLIX permutation to 16 variables, one round at a time.

|  | H0 $H_0$ | ... | H15 $H_{15}$ |
|--|----------|-----|--------------|
|0.| -        | ... | - |
|1.| -        | ... | - |
|2.| -        | ... | - |
|3.| -        | ... | - |
|4.| -        | ... | - |
|5.| -        | ... | - |
|6.| -        | ... | - |
|7.| -        | ... | - |

The AIR makes no requirements for rows that are multiples of 8. Other rows apply one round of Rescue-XLIX, with the parameters determined by the row index modulo 8.

A copy-constraint establishes that, whenever the `xlix` instruction is executed, the old and new 16-tuples in the SIMD cache correspond to rows $8k$ and $8k+7$ in the XLIX Table for some integer $k$.

### Uint32 Operations

The Uint32 Operations Table is a lookup table for 'difficult' 32-bit unsigned integer operations.

|     | LHS      | RHS      | EQ     | LT    | AND       | OR       | XOR       | REV |
|-----|----------|----------|--------|-------|-----------|----------|-----------|-----|
| 1.  | `a`      | `b`      | `a==b` | `a<b` | `a and b` | `a or b` | `a xor b` | `rev(a)` |
| 2.  | `a >> 1` | `b >> 1` | - | - | - | - | - | - |
| ... | - | - | - | - | - | - | - | - | - | - |
| 32. | `0` | `0` | `1` | `0` | `0` | `0` | `0` | `0` |
| 33. | `c` | `d` | - | - | - | - | - | - |

The AIR verifies the correct update of each consecutive pair of rows. In every row one bit is eliminated. Only when the previous row is all zeros (with a 1 in the column for `EQ`) can a new row be inserted.

The AIR constraints establish that the entire table is consistent. Copy-constraints establish that logical and bitwise operations were computed correctly.