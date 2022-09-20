# U32 Table

The U32 Operations Table is a lookup table for “difficult” 32-bit unsigned integer operations.
The two inputs to the U32 Operations Table are left-hand side (LHS) and right-hand side (RHS).

| `idc` | `bits` | `32_minus_bits_inv` | `ci`       | LHS (`st0`) | RHS (`st1`) | LT                  | AND                     | XOR                     | REV           | LHS_inv             | RHS_inv             |
|------:|-------:|--------------------:|:-----------|:------------|:------------|:--------------------|:------------------------|:------------------------|:--------------|:--------------------|:--------------------|
|     1 |      0 |           $32^{-1}$ | e.g. `and` | `a`         | `b`         | `a<b`               | `a and b`               | `a xor b`               | `rev(a)`      | `a`${}^{-1}$        | `b`${}^{-1}$        |
|     0 |      1 |           $31^{-1}$ | e.g. `and` | `a >> 1`    | `b >> 1`    | `(a >> 1)<(b >> 1)` | `(a >> 1) and (b >> 1)` | `(a >> 1) xor (b >> 1)` | `rev(a >> 1)` | `(a >> 1)`${}^{-1}$ | `(b >> 1)`${}^{-1}$ |
|     0 |      2 |           $30^{-1}$ | e.g. `and` | `a >> 2`    | `b >> 2`    | …                   | …                       | …                       | …             | …                   | …                   |
|     … |      … |                   … | …          | …           | …           | …                   | …                       | …                       | …             | …                   | …                   |
|     0 |    $n$ |       $(32-n)^{-1}$ | e.g. `and` | 0           | 0           | 2                   | 0                       | 0                       | 0             | 0                   | 0                   |
|     1 |      0 |           $32^{-1}$ | e.g. `lt`  | `c`         | `d`         | `c<d`               | `c and d`               | `c xor d`               | `rev(c)`      | `c`${}^{-1}$        | `d`${}^{-1}$        |
|     0 |      1 |           $31^{-1}$ | e.g. `lt`  | `c >> 1`    | `d >> 1`    | …                   | …                       | …                       | …             | …                   | …                   |
|     … |      … |                   … | …          | …           | …           | …                   | …                       | …                       | …             | …                   | …                   |
|     0 |    $m$ |       $(32-m)^{-1}$ | e.g. `lt`  | 0           | 0           | 2                   | 0                       | 0                       | 0             | 0                   | 0                   |
|     … |      … |                   … | …          | …           | …           | …                   | …                       | …                       | …             |                     |                     |

LT can take three possible values:
- 0 indicates LHS is definitely greater than or equal to RHS,
- 1 indicates LHS is definitely less than RHS, and
- 2 indicates that the verdict is not yet conclusive.

The AIR verifies the correct update of each consecutive pair of rows.
In every row one bit – the current least significant bit of both LHS and RHS – is eliminated.
Only when the current row equals the padding row given below can a new row with `idc = 1` be inserted.

The AIR constraints establish that the entire table is consistent.
Copy-constraints, i.e., Permutation Arguments, establish that the result of the logical and bitwise operations are correctly transferred to the Processor.

For every instruction in the `u32_op` instruction group (`lt`, `and`, `xor`, `reverse`, `div`), there is a dedicated Permutation Argument with the [Processor Table](processor-table.md).

## Padding

After the Uint32 Operations Table is filled in, its length being $l$, the table is padded until a total length of $2^{\lceil\log_2 l\rceil}$ is reached (or 0 if $l=0$).
Each padding row is the following row:

| `idc` | `bits` | `32_minus_bits_inv` | `ci` | LHS | RHS | LT | AND | XOR | REV | LHS_inv | RHS_inv |
|:------|-------:|--------------------:|-----:|----:|----:|---:|----:|----:|----:|--------:|--------:|
| 0     |      0 |           $32^{-1}$ |    0 |   0 |   0 |  2 |   0 |   0 |   0 |       0 |       0 |

## Initial Constraints

1. In the first row, the indicator `idc` is 1.

## Consistency Constraints

1. The indicator `idc` is 0 or 1.
1. If `idc` is 1, then `bits` is 0.
1. `bits` is not 32.
1. LHS_inv is the inverse of LHS if LHS is not 0, and 0 otherwise.
1. RHS_inv is the inverse of RHS if RHS is not 0, and 0 otherwise.
1. If `idc` is 0 and LHS is 0 and RHS is 0, then LT is 2.
1. If `idc` is 0 and LHS is 0 and RHS is 0, then AND is 0.
1. If `idc` is 0 and LHS is 0 and RHS is 0, then XOR is 0.
1. If `idc` is 0 and LHS is 0 and RHS is 0, then REV is 0.

Written as Disjunctive Normal Form, the same constraints can be expressed as:
1. The indicator `idc` is 0 or 1.
1. `idc` is 0 or `bits` is 0.
1. `32_minus_bits_inv` is the multiplicative inverse of (`bits` - 32).
1. LHS is 0, or LHS_inv is the inverse of LHS.
1. LHS_inv is 0, or LHS_inv is the inverse of LHS.
1. RHS is 0, or RHS_inv is the inverse of RHS.
1. RHS_inv is 0, or RHS_inv is the inverse of RHS.
1. `idc` is 1 or LHS is not 0 or RHS is not 0 or LT is 2.
1. `idc` is 1 or LHS is not 0 or RHS is not 0 or AND is 0.
1. `idc` is 1 or LHS is not 0 or RHS is not 0 or XOR is 0.
1. `idc` is 1 or LHS is not 0 or RHS is not 0 or REV is 0.

## Transition Constraints

Even though they are never explicitly represented, it is useful to alias the LHS's and RHS's _least-significant bit_, or “lsb.”
Given two consecutive rows for LHS, the (current) least significant bit can be computed by subtracting twice the next row's LHS from the current row's LHS.
The same is possible for column RHS.

1. If LHS or RHS is non-zero in the current row, then the indicator in the next row is 0.
1. If the indicator `idc` is 0 in the next row, then `ci` in the next row is `ci` in the current row.
1. If the indicator `idc` is 0 in the next row and either LHS or RHS is unequal to 0, then `bits` in the next row is `bits` in the current row plus 1.
1. If the indicator `idc` is 0 in the next row, then the lsb of LHS either 0 or 1.
1. If the indicator `idc` is 0 in the next row, then the lsb of RHS is either 0 or 1.
1. If the indicator `idc` is 0 in the next row and LT in the next row is 0, then LT in the current row is 0.
1. If the indicator `idc` is 0 in the next row and LT in the next row is 1, then LT in the current row is 1.
1. If the indicator `idc` is 0 in the next row and LT in the next row is 2 and lsb of LHS is 0 and lsb of RHS is 1, then LT in the current row is 1.
1. If the indicator `idc` is 0 in the next row and LT in the next row is 2 and lsb of LHS is 1 and lsb of RHS is 0, then LT in the current row is 0.
1. If the indicator `idc` is 0 in the next row and LT in the next row is 2 and (lsb of LHS equals lsb of RHS) and the indicator `idc` in the current row is 0, then LT in the current row is 2.
1. If the indicator `idc` is 0 in the next row and LT in the next row is 2 and (lsb of LHS equals lsb of RHS) and the indicator `idc` in the current row is 1, then LT in the current row is 0.
1. If the indicator `idc` is 0 in the next row, then AND in the current row equals twice AND in the next row plus (the product of the lsb of LHS and the lsb of the RHS).
1. If the indicator `idc` is 0 in the next row, then XOR in the current row equals twice XOR in the next row plus the lsb of LHS plus the lsb of RHS minus (twice the product of the lsb of LHS and the lsb of RHS).
1. If the indicator `idc` is 0 in the next row, then REV in the current row is (REV in the next row divided by 2, corresponding to a 1-bit right-shift) plus ($2^{31}$ times the lsb of LHS).

Written in disjunctive form, the same constraints can be expressed as:
1. LHS in the current row is 0 or `idc` in the next row is 0.
1. RHS in the current row is 0 or `idc` in the next row is 0.
1. `idc` in the next row is 1 or `ci` in the next row is `ci` in the current row.
1. `idc` in the next row is 1 or LHS is 0 or `bits` in the next row is `bits` in the current row plus 1.
1. `idc` in the next row is 1 or RHS is 0 or `bits` in the next row is `bits` in the current row plus 1.
1. `idc` in the next row is 1 or (the lsb of LHS is 0 or 1).
1. `idc` in the next row is 1 or (the lsb of RHS is 0 or 1).
1. `idc` in the next row is 1 or (LT in the next row is 1 or 2) or LT in the current row is 0.
1. `idc` in the next row is 1 or (LT in the next row is 0 or 2) or LT in the current row is 1.
1. `idc` in the next row is 1 or (LT in the next row is 0 or 1) or the lsb of LHS is 1 or the lsb of RHS is 0 or LT in the current row is 1.
1. `idc` in the next row is 1 or (LT in the next row is 0 or 1) or the lsb of LHS is 0 or the lsb of RHS is 1 or LT in the current row is 0.
1. `idc` in the next row is 1 or (LT in the next row is 0 or 1) or the lsb of LHS is unequal to the lsb of RHS or `idc` is 1 or LT in the current row is 2.
1. `idc` in the next row is 1 or (LT in the next row is 0 or 1) or the lsb of LHS is unequal to the lsb of RHS or `idc` is 0 or LT in the current row is 0.
1. `idc` in the next row is 1 or AND in the current row equals twice AND in the next row plus (the product of the lsb of LHS and the lsb of the RHS).
1. `idc` in the next row is 1 or XOR in the current row equals twice XOR in the next row plus the lsb of LHS plus the lsb of RHS minus (twice the product of the lsb of LHS and the lsb of RHS).
1. `idc` in the next row is 1 or REV in the current row is (REV in the next row divided by 2) plus ($2^{31}$ times the lsb of LHS).

## Terminal Constraints

1. In the last row, the indicator `idc` is 0.
1. In the last row, LHS is 0.
1. In the last row, RHS is 0.

## Relations to Other Tables

1. A Permutation Argument, conditioned on `ci`, establishes that the correct result is transferred to the [processor](processor-table.md).
