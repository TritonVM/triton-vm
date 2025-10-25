---
title: Deriving AIR Constraints
---

## Purpose

This guide explains how to derive Arithmetic Intermediate Representation (AIR) constraints from a textual specification of table semantics. Rather than listing final polynomials, we describe a repeatable derivation process. Concrete polynomial forms may be relegated to appendices of each table doc.

## Notation

- Current row variables use unprimed symbols (e.g., `x`).
- Next row variables use primed symbols (e.g., `x'`).
- Challenges used to compress rows or as evaluation points are denoted symbolically (e.g., 🍇, 🍅, 🧴). They reside in the X-field.
- Indicators for padding or instruction selection are boolean and constrained via `b·(1-b)`.

## Derivation Steps

1. Identify invariants and initial conditions
   - State what must hold in the first row (e.g., running products/log-derivatives initialized; zeroed columns).
   - Express each as an equality over current-row variables.

2. Encode Booleanity and range constraints (consistency)
   - For boolean bits `b`, add `b·(1-b)`.
   - For conditional-zero columns in padding rows, multiply by the padding bit.

3. Express row-to-row semantics (transition)
   - Write the high-level cases: increments, remains, conditional updates.
   - Convert OR-cases into disjunctive-normal-form-friendly polynomials by multiplying with selector bits or using “stays or updates” patterns like `(1 - sel)·(x' - x - Δ) + sel·(x' - x)`.

4. Accumulators (running products and log-derivatives)
   - Running product `rp` that absorbs a row-combination `combo` typically follows `rp' - rp·(α - combo)` for an indeterminate `α` and compressed row `combo`.
   - Log-derivative accumulators `ld` that sum inverses of linear terms typically follow `(ld' - ld)·(α - combo) - 1` in non-padding rows and `ld' - ld` in padding rows, combined via padding bit.

5. Terminal constraints
   - Constrain end-of-trace conditions, e.g., equate accumulators to a verifier-computed value or enforce final pointer positions.

6. Degree lowering considerations
   - If any constraint exceeds target degree, introduce helper variables and substitution constraints; see `triton-air::TARGET_DEGREE` and the build-time lowering in the constraint builder.

## Worked Template

For a conditional accumulator `Acc` that should update when `¬Pad`:

`(1 - Pad')·(Acc' - Acc - Update) + Pad'·(Acc' - Acc)`

For a value that must remain in padding rows:

`Pad'·Value'`

For a permutation running product `R` absorbing compressed row `combo` with indeterminate `β`:

`(1 - Pad')·(R' - R·(β - combo')) + Pad'·(R' - R)`

## Mapping Text → Constraints

When a table doc states “If condition C then update U else remain”, write two linear expressions for `Acc' - Acc - U` and `Acc' - Acc`, and gate them by the selector bit for C. Disjunctions become sums of gated clauses, each multiplied by the complement of other cases when necessary.

Refer to individual table docs for domain-specific combos and challenge wiring. The final polynomial lists are provided as appendices for verification and cross-referencing.


