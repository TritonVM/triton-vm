# Table Linking

Triton VM's tables are linked together using a variety of different cryptographic arguments.
The [section on arithmetization](arithmetization.md) incorporates all the necessary ingredients by defining the corresponding parts of the Arithmetic Execution Tables and the Arithmetic Intermediate Representation.
However, that presentation doesn't make it easy to follow and understand the specific arguments in use.
Therefor, the distinct “cross-table” or “table-linking” arguments are presented here in isolation.

## Compressing Multiple Elements

Mathematically, all arguments used in Triton VM are about single elements of the finite field $\mathbb{F}_p$.
In practice, it is very useful to argue about multiple elements.
A common trick is to “compress” multiple elements into a single one using random weights.
These weights are “supplied by the STARK verifier,” _i.e._, sampled using the Fiat-Shamir heuristic after the prover has committed to the elements in question.
For example, if $a$, $b$, and $c$ are to be incorporated into the cryptographic argument, then the prover

- commits to $a$, $b$, and $c$,
- samples weights $\alpha$, $\beta$, and $\gamma$ using the Fiat-Shamir heuristic,
- “compresses” the elements in question to $\delta = \alpha\cdot a + \beta\cdot b + \gamma\cdot c$, and
- uses $\delta$ in the cryptographic argument.

In the following, all cryptographic arguments are presented using single field elements.
Depending on the use case, this single element represents multiple, “compressed” elements.
