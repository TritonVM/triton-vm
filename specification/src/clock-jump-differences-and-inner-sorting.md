# Clock Jump Differences and Inner Sorting

The previous sections show how it is proven that in the Jump Stack, Op Stack, and RAM Tables, the regions of constant memory pointer are contiguous. The next step is to prove that within each contiguous region of constant memory pointer, the rows are sorted for clock cycle. That is the topic of this section.

The problem arises from *clock jumps*, which describes the phenomenon when the clock cycle increases even though the memory pointer does not change.
If arbitrary jumps were allowed, nothing would prevent the cheating prover from using a table where higher rows correspond to later states, giving rise to an exploitable attack.
So it must be shown that every clock jump is directed forward and not backward.

Our strategy is to show that the *difference*, *i.e.*, the next clock cycle minus the current clock cycle, is itself a clock cycle.
Recall that the Processor Table's clock cycles run from 0 to $T-1$.
Therefore, a forward directed clock jump difference is in $F = \lbrace 1, \dots, T - 1 \rbrace \subseteq \mathbb{F}_p$, whereas a backward directed clock jump's difference is in $B = \lbrace -f \mid f \in F \rbrace = \lbrace 1 - T, \dots, -1 \rbrace \subseteq \mathbb{F}_p$.
No other clock jump difference can occur.
If $T \ll p/2$, there is no overlap between sets $F$ and $B$.
As a result, in this regime, showing that a clock jump difference is in $F$ guarantees that the jump is forward-directed.

The set of values in the Processor Table's clock cycle column is $F \cup \lbrace 0 \rbrace$.
A standard [Lookup Argument](lookup-argument.md) can show that the clock jump differences are elements of that column.
Since all three memory-like tables look up the same property, the [Processor Table](processor-table.md) acts as a single server as opposed to three distinct servers.
Concretely, the lookup multiplicities of the three clients, _i.e._, the memory-like tables, are recorded in a single column `cjd_mul`.
It contains the element-wise sum of the three distinct lookup multiplicities.
