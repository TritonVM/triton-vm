# Copy-Constraints

Triton VM, possibly in addition to other VMs, uses several different tables. Each table has its own AIR. However, the tables a *not* independent. There are constraints that require that some pairs of tables have some identical rows. This note explains how to realize these copy-constraints at the Polynomial IOP level.

## Permutation Argument

Let $T$ and $S$ be single-column tables of $2^k$ rows. We describe a protocol that proves that $S$ and $T$ are permutations of one another. The key is to take the product of all elements, after offsetting them by some random $\beta$. Clearly the polynomials

$$ (X - T_{[0]}) \cdot (X - T_{[1]}) \cdots (X - T_{[2^k-1]}) $$

and

$$ (X - S_{[0]}) \cdot (X - S_{[1]}) \cdots (X - S_{[2^k-1]}) $$

are equal iff $\{T_{[i]}\}_i = \{S_{[i]}\}_i$. The offset by $\beta$ amounts to evaluating these polynomials in a random point. The subsequent scalar equality test constitutes a randomized polynomial identity test and by the Schwartz-Zippel lemma produces a false positive with probability at most $\frac{2^k}{|\mathbb{F}|}$.

To establish that the product is correctly computed, the Prover needs to supply running product tables, $T_ p$ and $S_ p$. These pairs of tables satisfy the following AIR:
 - $T_ {p,[0]} = T_ {[0]}$ and $S_ {p,[0]} = S_ {[0]}$
 - For all $i \in \{0, \ldots, 2^k-2\}$: $T_ {p,[i+1]} = T_ {p,[i]} \cdot T_ {[i+1]}$ and $S_ {p,[i+1]} = S_ {p,[i]} \cdot S_ {[i+1]}$
 - $T_ {p, [2^k-1]} = S_ {p, [2^k-1]}$

**Protocol Permutation**
 - Verifier is already in possession of polynomial oracles $[t(X)]$ and $[s(X)]$ that agree with $T$ and $S$ on a subgroup $H$ of order $2^k$.
 - Verifier samples $\beta \xleftarrow{\\$} \mathbb{F}$ and sends it to Prover.
 - Prover computes running product polynomials, $t_ p(X)$ and $s_ p(X)$ of degree at most $|H|-1$ and sends them to Verifier.
 - Prover and Verifier run an AIR sub-protocol to establish that the AIR is satisfied for $(t(X), s(X), t_ p(X), s_ p(X))$.

What if $T$ and $S$ have multiple columns, say $r$-many? In this case the verifier simply supplies $r$ weights \(\alpha_0, \ldots, \alpha_{r-1} \xleftarrow{\$} \mathbb{F}\). The protocol is then applied to the weighted sum of columns.

What if a column in one of the tables is not explicitly available, but can only be inferred from the other columns via a low-degree polynomial? It is not a problem; the Verifier just evaluates this polynomial to simulate the column, and includes it in the random linear combination.

Therefore, in relation to copy-constraints, we can consider all original tables to have only one column.

## Subset

Let $T$ and $S$ be tables with no zero-rows. We want to show that the set of rows of $T$ are a subset of the rows of $S$. This can be done with permutation arguments, but requires two intermediate steps: *sorting* and *masking*.

Let $R$ be another table containing the same rows as $T$, but sorted to match the order of $S$. A permutation argument establishes that $R$ and $T$ are permutations.

Let $M$ be another table that contains only zeros and ones such that
 - $M_{[i]} = 1$ if $S_{[i]}$ is a row in $T$;
 - $M_{[i]} = 0$ if $S_{[i]}$ is not a row in $T$.

The table $M$ should satisfy the AIR constraint $\forall i \in \{0, \ldots, 2^k-1\}: M_{[i]} \cdot ( 1 - M_{[i]} ) = 0$. Let $M \circ S$ denote the table whose rows agree with $S$ when $M$ is set, and whose rows are zero otherwise.

At this point we have two tables, $M \circ S$ and $R$. The original subset claim is true iff the sets of nonzero rows of $M \circ S$ and $R$ are identical. Let $\{a_i\}_i$ denote the elements of this set.

In the permutation argument, we shifted both tables by $\beta$ and followed up by computing the product of all elements. We cannot apply the same trick here because 
 - the multiplicities of $(a_i- X)$ are different;
 - $M \circ S$ contains zeros, giving rise to a factor of the form $(-\beta)^\mu$.

Nevertheless, we can use $M$ write an AIR that avoids accumulating a factor $S_{[i]} - \beta$ when $M$ is not set. This running *masked* product corresponds to evaluating $\prod_i \left( a_i - X \right)$ in $\beta$.

Likewise, we can re-write the AIR for the running product to avoid accumulating a factor $R_{[i]}$ when $R_{[i]} = R_{[i-1]}$. The resulting running *square-free* product also corresponds to an evaluation of $\prod_i \left( a_i - X \right)$ in $\beta$.

**Protocol Subset**
 - Prover computes $R \leftarrow \mathsf{sort}(T)$ and $M \leftarrow \{[ [ S_{[i]} \in \mathsf{set}(R) ] ] \, \vert \, i \in \{0, \ldots, 2^k-1\}\}$ and commits to them.
 - Prover and Verifier run a Permutation subprotocol on $(T,R)$.
 - Verifier samples $\beta \xleftarrow{\\$} \mathbb{F}$ and sends it to Prover.
 - Prover computes the running square-free product table $R_ p$ of $R$, and the running masked product table of $S_ p$ of $S$ masked by $M$.
 - Prover sends $R_ p$ and $S_ p$ to Verifier.
 - Verifier checks the AIR for $(R, R_ p)$ and the AIR for $(S_ p, M \circ S)$.
 - Verifier checks that $R_ {p, [2^k-1]} = S_ {p, [2^k-1]}$.

The protocol can be extended quite naturally to the case where only a subset of the rows of $T$ are shown to be contained in $S$. This adaptation requires a second mask to indicate which rows of $T$ should be considered in this subset claim.

## Batch-Subset

What if we have a table $T$ consisting of $r$ columns and a mask $M_ T$, and we want to show that all *cells* of all indicated rows of this table are contained in the single column of another table $S$? There are two solutions. The first uses $r+1$ additional columns to compute the running product. The second uses an intermediate table of $r$ columns.

### Batch-Subset with Additional Columns

The Verifier supplies $\beta$. Then the prover concatenates to $T$ a column $T_{[:,r]}$ whose elements satisfy
 - if $M_{[0]} = 1$ then $T_{[0, r]} = (T_{[0,0]} - \beta) \cdot (T_{[0,1]} - \beta)$, and otherwise $T_{[0,r]} = 1$;
 - for $i > 1: T_{[i,r]} = T_{[i-1,r]} \cdot (T_{[i, 0]} - \beta) \cdot (T_{[i, 1]} - \beta)$.

After concatenating $r$ such columns, the last column of $T$ satisfies $T_{[i,2r-1]} = \prod_{j=0}^{r-1} (T_{[i,j]} - \beta)$ for all $i$. One more column computes the running product.

Unfortunately, this construction cannot filter out duplicates, so it cannot be linked via Permutation Argument to a duplicate-free lookup table directly. However, it *can* be linked via Permutation Argument to an intermediate table whose column contains all values sorted. This intermediate table admits a square-free running product, which can be used to complete the link to the duplicate-free lookup table $S$.

### Batch-Subset with an Intermediate Table with Linear AIR

Let $R$ be a table of $r$ columns such that
 - every next row is the same as the previous row, except for a shift to the right by one along with a new element on the left;
 - every indicated row of $T$ is present in $R$;
 - there are enough rows so that every indicated cell in $T$ appears in the rightmost column of $R$.

Then one Subset Argument establishes that all indicated rows of $T$ are present in $R$, and another Subset Argument establishes that all elements of the last column of $R$ are present in $S$.

**Example.** Let T be the table below, along with a mask.

| MT | T[:,0] | T[:,1] | T[:,2] |
|----|--------|--------|--------|
| 0  |        |        |        |
| 1  |   a    |    b   |    c   |
| 0  |        |        |        |
| 0  |        |        |        |
| 0  |        |        |        |
| 1  |   d    |    e   |   f    |

Then the intermediate table could be constructed as below.

| R[:,0] | R[:,1] | R[:,2] |
|--------|--------|--------|
| a | b | c |
| f | a | b |
| e | f | a |
| d | e | f |
| - | d | e |
| - | - | d |

Note that the indicated rows of T are present in R, and all elements from those rows are present in the last column of R.