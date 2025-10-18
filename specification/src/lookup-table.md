# Lookup Table

The Lookup Table helps arithmetizing the lookups necessary for the split-and-lookup S-box used in the Tip5 permutation.
It works in tandem with the [Cascade Table](cascade-table.md).
In the language of the [Tip5 paper](https://eprint.iacr.org/2023/107.pdf), it is a “narrow lookup table.”
This means it is always fully populated, independent of the actual number of lookups.

Correct creation of the Lookup Table is guaranteed through a public-facing [Evaluation Argument](evaluation-argument.md):
after sampling some challenge $X$, the verifier computes the terminal of the Evaluation Argument over the list of all the expected lookup values with respect to challenge $X$.
The equality of this verifier-supplied terminal against the similarly computed, in-table part of the Evaluation Argument is checked by the Lookup Table's terminal constraint.

## Main Columns

The Lookup Table has 4 main columns:

| name                 | description                                 |
|:---------------------|:--------------------------------------------|
| `IsPadding`          | Indicator for padding rows.                 |
| `LookIn`             | The lookup input.                           |
| `LookOut`            | The lookup output.                          |
| `LookupMultiplicity` | The number of times the value is looked up. |

## Auxiliary Columns

The Lookup Table has 2 auxiliary columns:

- `CascadeTableServerLogDerivative`, the (running sum of the) logarithmic derivative for the Lookup Argument with the Cascade Table.
    In every row, accumulates the summand `LookupMultiplicity / Combo` where `Combo` is the verifier-weighted combination of `LookIn` and `LookOut`.
- `PublicEvaluationArgument`, the running sum for the public evaluation argument of the Lookup Table.
    In every row, accumulates `LookOut`.

## Padding

Each padding row is the all-zero row with the exception of `IsPadding`, which is 1.

# Arithmetic Intermediate Representation

Let all household items (🪥, 🛁, etc.) be challenges, concretely evaluation points, supplied by the verifier.
Let all fruit & vegetables (🥝, 🥥, etc.) be challenges, concretely weights to compress rows, supplied by the verifier.
Both types of challenges are X-field elements, _i.e._, elements of $\mathbb{F}_{p^3}$.

## Initial Constraints

1. `LookIn` is 0.
1. `CascadeTableServerLogDerivative` has accumulated the first row with respect to challenges 🍒 and 🍓 and indeterminate 🧺.
1. `PublicEvaluationArgument` has accumulated the first `LookOut` with respect to indeterminate 🧹.

### Initial Constraints as Polynomials

1. `LookIn`
1. `CascadeTableServerLogDerivative·(🧺 - 🍒·LookIn - 🍓·LookOut) - LookupMultiplicity`
1. `PublicEvaluationArgument - 🧹 - LookOut`

## Consistency Constraints

1. `IsPadding` is 0 or 1.

### Consistency Constraints as Polynomials

1. `IsPadding·(1 - IsPadding)`

## Transition Constraints

1. If the current row is a padding row, then the next row is a padding row.
1. If the next row is not a padding row, `LookIn` increments by 1.
    Else, `LookIn` is 0.
1. If the next row is not a padding row, `CascadeTableServerLogDerivative` accumulates the next row with respect to challenges 🍒 and 🍓 and indeterminate 🧺.
    Else, `CascadeTableServerLogDerivative` remains unchanged.
1. If the next row is not a padding row, `PublicEvaluationArgument` accumulates the next `LookOut` with respect to indeterminate 🧹.
    Else, `PublicEvaluationArgument` remains unchanged.

### Transition Constraints as Polynomials

1. `IsPadding·(1 - IsPadding')`
1. `((1 - IsPadding')·(LookIn' - LookIn - 1))`<br />
    `+ IsPadding'·LookIn'`
1. `(1 - IsPadding')·((CascadeTableServerLogDerivative' - CascadeTableServerLogDerivative)·(🧺 - 🍒·LookIn' - 🍓·LookOut') - LookupMultiplicity')`<br />
    `+ IsPadding'·(CascadeTableServerLogDerivative' - CascadeTableServerLogDerivative)`
1. `(1 - IsPadding')·(PublicEvaluationArgument' - 🧹·PublicEvaluationArgument - LookOut')`<br />
    `+ IsPadding'·(PublicEvaluationArgument' - PublicEvaluationArgument)`

## Terminal Constraints

1. `PublicEvaluationArgument` matches verifier-supplied challenge 🪠.

### Terminal Constraints as Polynomials

1. `PublicEvaluationArgument` - 🪠
