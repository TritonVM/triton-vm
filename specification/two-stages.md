# Two-Stage STARK

The Triton STARK proof protocol consists of two stages, not counting the FRI stage that comes after them.

Triton defines several Tables, such as ProcessorTable, InstructionTable, MemoryTable, etc. Each Table has its internal AIR. There are relations between Tables, typically [permutation relations](copy-constraints.md) or [sublist relations](evaluation-argument.md). To prove these relations, every table is extended with a set of extension columns, which integrate random scalars supplied by the verifier. Each extension column computes a *terminal value*, which is identical across the two related Tables if the relation is true, and distinct with overwhelming probability if the relation is false. The *extension AIR* proves the integral computation of the terminal value.

For instance let $S$ and $T$ be two tables whose $N$ rows are permutations of one another. Without loss of generality $S$ and $T$ contain only one base column (otherwise the verifier supplies random scalars to compress the many columns into one). The prover uses a scalar $\beta$ to extend $S$ into $S^\star$ and $T$ into $T^\star$ by concatenating a column according to the following rules:
 1. $S^\star_{[0,1]} = \beta - S_{[0,0]}$
 2. for all $i \in \{1, \ldots, N-1\}$, $S^\star_{[i,1]} = S^\star_{[i-1,1]} \cdot (\beta - S_{[i,0]})$
 3. $T^\star_{[0,1]} = \beta - T_{[0,0]}$
 2. for all $i \in \{1, \ldots, N-1\}$, $T^\star_{[i,1]} = T^\star_{[i-1,1]} \cdot (\beta - T_{[i,0]})$

 Ignore cases where $\beta - S_{[i,0]} = 0$. Both extension columns should end in the terminal value

$$ \prod_{i=0}^{N-1} (\beta - S_{[i,0]}) = \prod_{i=0}^{N-1} (\beta - T_{[i,0]}) \enspace .$$

This equation holds with certainty when $T$ and $S$ are permutations of each other, and with negligible probability roughly $N / |\mathbb{F}|$ when they are not.

To prove that this equality holds, the prover supplies the terminal value. The fact that the extension columns end in this value can be verified along with other AIR constraints. The extension tables $T^\star$ and $S^\star$ therefore have several types of AIR constraints:
 - internal AIR constraints, independent of verifier challenges
 - boundary constraints on the extension columns, like numbers 1 and 3 above
 - transition constraints on the extension columns, like numbers 2 and 4 above
 - terminal constraints, that require that the last value is indeed the terminal value

Recall that every AIR constraint gives rise to a quotient polynomial. This contruction of Tables and Table-Relations therefore gives rise to a two-stage STARK.

 1. The Prover sends the polynomials $s_b(X)$ and $t_b(X)$ of degree at most $N-1$ to the Verifier. These polynomials interpolate through $S_{[:,0]}$ and $T_{[:,0]}$ on the interpolation domain.
 2. The Verifier samples a random challenge $\beta \xleftarrow{\$} \mathbb{F}$ and sends it to the Prover.
 3. The Prover computes the extension columns and their interpolants $s_e(X)$ and $t_e(X)$, which are polynomials of degree at most $N-1$, and sends them to the Verifier.
 4. For all AIR constraints $i$, the Prover computes the matching quotient $q_i(X)$ and sends it to the Verifier.
 5. The Verifier samples a vector $\{c_j\} \xleftarrow{\$} \mathbb{F}^{2n}$ where $n$ is the total number of polynomials sent by the Prover; and the Verifier sends $\{c_j\}$ to the Prover.
 6. The Prover uses $\{c_j\}$ to compute a single nonlinear combination of all polynomials, where the nonlinearity arises from duplicating each polynomial and shifting one twin by the appropriate power of $X$. The Prover sends the resulting polynomial $f(X)$ to the Verifier.
 7. The Verifier verifies the nonlinear combination.
 8. The Prover and Verifier use the FRI subprotocol to bound the degree of $f(X)$.

This protocol qualifies as two-stage because a) FRI is not counted; and b) there are only two inputs from the verifier. Specifically, if the extension columns were not present, then steps 2 and 3 could have been dropped and there would have been only one message from the verifier.

In practice, each stage comes with a whole new Merkle tree of zipped FRI codewords. If it is possible to eliminate the second stage, then this elimination boosts performance significantly. However, the problem seems to be the structure of the transition constraints that govern the extension columns. Specifically, does not seem possible to algebraically eliminate $t_e(X)$ and $s_e(X)$ and describe the quotients directly in terms of $t_b(X)$ and $s_b(X)$.
