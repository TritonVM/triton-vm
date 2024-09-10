# Cascade Table

The Cascade Table helps arithmetizing the lookups necessary for the split-and-lookup S-box used in the Tip5 permutation.
The main idea is to allow the [Hash Table](hash-table.md) to look up limbs that are 16 bit wide even though the S-box is defined over limbs that are 8 bit wide.
The Cascade Table facilitates the translation of limb widths.
For the actual lookup of the 8-bit limbs, the [Lookup Table](lookup-table.md) is used.
For a more detailed explanation and in-depth analysis, see the [Tip5 paper](https://eprint.iacr.org/2023/107.pdf).

## Main Columns

The Cascade Table has 6 main columns:

| name                 | description                                                   |
|:---------------------|:--------------------------------------------------------------|
| `IsPadding`          | Indicator for padding rows.                                   |
| `LookInHi`           | The more significant bits of the lookup input.                |
| `LookInLo`           | The less significant bits of the lookup input.                |
| `LookOutHi`          | The more significant bits of the lookup output.               |
| `LookOutLo`          | The less significant bits of the lookup output.               |
| `LookupMultiplicity` | The number of times the value is looked up by the Hash Table. |

## Auxiliary Columns

The Cascade Table has 2 auxiliary columns:

- `HashTableServerLogDerivative`, the (running sum of the) logarithmic derivative for the Lookup Argument with the Hash Table.
    In every row, the sum accumulates `LookupMultiplicity / (🧺 - Combo)` where 🧺 is a verifier-supplied challenge
    and `Combo` is the weighted sum of `LookInHi · 2^8 + LookInLo` and `LookOutHi · 2^8 + LookOutLo`
    with weights 🍒 and 🍓 supplied by the verifier.
- `LookupTableClientLogDerivative`, the (running sum of the) logarithmic derivative for the Lookup Argument with the Lookup Table.
    In every row, the sum accumulates the two summands
    1. `1 / combo_hi` where `combo_hi` is the verifier-weighted combination of `LookInHi` and `LookOutHi`, and
    1. `1 / combo_lo` where `combo_lo` is the verifier-weighted combination of `LookInLo` and `LookOutLo`.

## Padding

Each padding row is the all-zero row with the exception of `IsPadding`, which is 1.

# Arithmetic Intermediate Representation

Let all household items (🪥, 🛁, etc.) be challenges, concretely evaluation points, supplied by the verifier.
Let all fruit & vegetables (🥝, 🥥, etc.) be challenges, concretely weights to compress rows, supplied by the verifier.
Both types of challenges are X-field elements, _i.e._, elements of $\mathbb{F}_{p^3}$.

## Initial Constraints

1. If the first row is not a padding row, then `HashTableServerLogDerivative` has accumulated the first row with respect to challenges 🍒 and 🍓 and indeterminate 🧺.
    Else, `HashTableServerLogDerivative` is 0.
1. If the first row is not a padding row, then `LookupTableClientLogDerivative` has accumulated the first row with respect to challenges 🥦 and 🥒 and indeterminate 🪒.
    Else, `LookupTableClientLogDerivative` is 0.

### Initial Constraints as Polynomials

1. `(1 - IsPadding)·(HashTableServerLogDerivative·(🧺 - 🍒·(2^8·LookInHi + LookInLo) - 🍓·(2^8 · LookOutHi + LookOutLo)) - LookupMultiplicity)`<br />
    `+ IsPadding · HashTableServerLogDerivative`
1. `(1 - IsPadding)·(LookupTableClientLogDerivative·(🪒 - 🥦·LookInLo - 🥒·LookOutLo)·(🪒 - 🥦·LookInHi - 🥒·LookOutHi) - 2·🪒 + 🥦·(LookInLo + LookInHi) + 🥒·(LookOutLo + LookOutHi))`<br />
    `+ IsPadding·LookupTableClientLogDerivative`

## Consistency Constraints

1. `IsPadding` is 0 or 1.

### Consistency Constraints as Polynomials

1. `IsPadding·(1 - IsPadding)`

## Transition Constraints

1. If the current row is a padding row, then the next row is a padding row.
1. If the next row is not a padding row, then `HashTableServerLogDerivative` accumulates the next row with respect to challenges 🍒 and 🍓 and indeterminate 🧺.
    Else, `HashTableServerLogDerivative` remains unchanged.
1. If the next row is not a padding row, then `LookupTableClientLogDerivative` accumulates the next row with respect to challenges 🥦 and 🥒 and indeterminate 🪒.
    Else, `LookupTableClientLogDerivative` remains unchanged.

### Transition Constraints as Polynomials

1. `IsPadding·(1 - IsPadding')`
1. `(1 - IsPadding')·((HashTableServerLogDerivative' - HashTableServerLogDerivative)·(🧺 - 🍒·(2^8·LookInHi' + LookInLo') - 🍓·(2^8 · LookOutHi' + LookOutLo')) - LookupMultiplicity')`<br />
    `+ IsPadding'·(HashTableServerLogDerivative' - HashTableServerLogDerivative)`
1. `(1 - IsPadding')·((LookupTableClientLogDerivative' - LookupTableClientLogDerivative)·(🪒 - 🥦·LookInLo' - 🥒·LookOutLo')·(🪒 - 🥦·LookInHi' - 🥒·LookOutHi') - 2·🪒 + 🥦·(LookInLo' + LookInHi') + 🥒·(LookOutLo' + LookOutHi'))`<br />
    `+ IsPadding'·(LookupTableClientLogDerivative' - LookupTableClientLogDerivative)`

## Terminal Constraints

None.
