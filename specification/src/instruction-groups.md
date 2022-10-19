# Instruction Groups

| group name      | description                                                                                         |
|:----------------|:----------------------------------------------------------------------------------------------------|
| `decompose_arg` | instruction's argument held in `nia` is binary decomposed into helper registers `hv0` through `hv3` |
| `step_1`        | instruction pointer `ip` increases by 1                                                             |
| `step_2`        | instruction pointer `ip` increases by 2                                                             |
| `grow_stack`    | a new element is put onto the stack, rest of the stack remains unchanged                            |
| `keep_stack`    | stack remains unchanged                                                                             |
| `shrink_stack`  | stack's top-most element is removed, rest of the stack remains unchanged. Needs `hv3`               |
| `unop`          | stack's top-most element is modified, rest of stack remains unchanged                               |
| `binop`         | stack shrinks by one element, new top of the stack is modified. Needs `hv3`                         |
| `keep_ram`      | no memory access: `ramp` and `ramv` do not change.                                                  |

A summary of all instructions and which groups they are part of is given in the following table.

| instruction      | `has_arg`* | `decompose_arg` | `step_1` | `step_2` | `grow_stack` | `keep_stack` | `shrink_stack` | `unop` | `binop` | `keep_ram` |
|:-----------------|:-----------|:----------------|:---------|:---------|:-------------|:-------------|:---------------|:-------|:--------|------------|
| `pop`            |            |                 | x        |          |              |              | x              |        |         | x          |
| `push` + `a`     | x          |                 |          | x        | x            |              |                |        |         | x          |
| `divine`         |            |                 | x        |          | x            |              |                |        |         | x          |
| `dup` + `i`      | x          | x               |          | x        | x            |              |                |        |         | x          |
| `swap` + `i`     | x          | x               |          | x        |              |              |                |        |         | x          |
| `nop`            |            |                 | x        |          |              | x            |                |        |         | x          |
| `skiz`           |            |                 |          |          |              |              | x              |        |         | x          |
| `call` + `d`     | x          |                 |          |          |              | x            |                |        |         | x          |
| `return`         |            |                 |          |          |              | x            |                |        |         | x          |
| `recurse`        |            |                 |          |          |              | x            |                |        |         | x          |
| `assert`         |            |                 | x        |          |              |              | x              |        |         | x          |
| `halt`           |            |                 | x        |          |              | x            |                |        |         | x          |
| `read_mem`       |            |                 | x        |          |              |              |                | x      |         |            |
| `write_mem`      |            |                 | x        |          |              | x            |                |        |         |            |
| `hash`           |            |                 | x        |          |              |              |                |        |         | x          |
| `divine_sibling` |            |                 | x        |          |              |              |                |        |         | x          |
| `assert_vector`  |            |                 | x        |          |              | x            |                |        |         | x          |
| `add`            |            |                 | x        |          |              |              |                |        | x       | x          |
| `mul`            |            |                 | x        |          |              |              |                |        | x       | x          |
| `invert`         |            |                 | x        |          |              |              |                | x      |         | x          |
| `split`          |            |                 | x        |          |              |              |                |        |         | x          |
| `eq`             |            |                 | x        |          |              |              |                |        | x       | x          |
| `lsb`            |            |                 | x        |          |              |              |                |        |         | x          |
| `xxadd`          |            |                 | x        |          |              |              |                |        |         | x          |
| `xxmul`          |            |                 | x        |          |              |              |                |        |         | x          |
| `xinvert`        |            |                 | x        |          |              |              |                |        |         | x          |
| `xbmul`          |            |                 | x        |          |              |              |                |        |         | x          |
| `read_io`        |            |                 | x        |          | x            |              |                |        |         | x          |
| `write_io`       |            |                 | x        |          |              |              | x              |        |         | x          |

\*
Instruction Group `has_arg` is a _virtual_ instruction group.
That is, this instruction group is not represented by the instruction bucket registers `ib`.
The virtual instruction group `has_arg` is required for correct behavior of instruction `skiz`, for which the instruction pointer `ip` needs to increment by either 1, or 2, or 3.
The concrete value depends on the top of the stack `st0` and the next instruction, held in `nia`.
If (and only if) the current instruction `ci` is instruction `skiz`, then the opcode held in register `nia` is deconstructed into helper variable registers `hv`.
This is similar to how `ci` is (always) deconstructed into instruction bucket registers `ib`.
The virtual instruction bucket `has_arg` helps in identifying optimal opcodes for all instructions during development of TritonVM.

In the following sections, a register marked with a `'` refers to the next state of that register.
For example, `st0' = st0 + 2` means that stack register `st0` is incremented by 2.
An alternative view for the same concept is that registers marked with `'` are those of the next row in the table.

## Indicator Polynomials `ind_i(hv3, hv2, hv1, hv0)`

For instructions [`dup`](instruction-specific-transition-constraints.md#instruction-dup--i) and [`swap`](instruction-specific-transition-constraints.md#instruction-swap--i), it is beneficial to have polynomials that evaluate to 1 if the instruction's argument `i` is a specific value, and to 0 otherwise.
This allows indicating which registers are constraint, and in which way they are, depending on `i`.
This is the purpose of the _indicator polynomials_ `ind_i`.
Evaluated on the binary decomposition of `i`, they show the behavior described above.

For example, take `i = 13`.
The corresponding binary decomposition is `(hv3, hv2, hv1, hv0) = (1, 1, 0, 1)`.
Indicator polynomial `ind_13(hv3, hv2, hv1, hv0)` is `hv3·hv2·(1 - hv1)·hv0`.
It evaluates to 1 on `(1, 1, 0, 1)`, i.e., `ind_13(1, 1, 0, 1) = 1`.
Any other indicator polynomial, like `ind_7`, evaluates to 0 on `(1, 1, 0, 1)`.
Likewise, the indicator polynomial for 13 evaluates to 0 for any other argument.

Below, you can find a list of all 16 indicator polynomials.

1.  `ind_0(hv3, hv2, hv1, hv0) = (1 - hv3)·(1 - hv2)·(1 - hv1)·(1 - hv0)`
1.  `ind_1(hv3, hv2, hv1, hv0) = (1 - hv3)·(1 - hv2)·(1 - hv1)·hv0`
1.  `ind_2(hv3, hv2, hv1, hv0) = (1 - hv3)·(1 - hv2)·hv1·(1 - hv0)`
1.  `ind_3(hv3, hv2, hv1, hv0) = (1 - hv3)·(1 - hv2)·hv1·hv0`
1.  `ind_4(hv3, hv2, hv1, hv0) = (1 - hv3)·hv2·(1 - hv1)·(1 - hv0)`
1.  `ind_5(hv3, hv2, hv1, hv0) = (1 - hv3)·hv2·(1 - hv1)·hv0`
1.  `ind_6(hv3, hv2, hv1, hv0) = (1 - hv3)·hv2·hv1·(1 - hv0)`
1.  `ind_7(hv3, hv2, hv1, hv0) = (1 - hv3)·hv2·hv1·hv0`
1.  `ind_8(hv3, hv2, hv1, hv0) = hv3·(1 - hv2)·(1 - hv1)·(1 - hv0)`
1.  `ind_9(hv3, hv2, hv1, hv0) = hv3·(1 - hv2)·(1 - hv1)·hv0`
1. `ind_10(hv3, hv2, hv1, hv0) = hv3·(1 - hv2)·hv1·(1 - hv0)`
1. `ind_11(hv3, hv2, hv1, hv0) = hv3·(1 - hv2)·hv1·hv0`
1. `ind_12(hv3, hv2, hv1, hv0) = hv3·hv2·(1 - hv1)·(1 - hv0)`
1. `ind_13(hv3, hv2, hv1, hv0) = hv3·hv2·(1 - hv1)·hv0`
1. `ind_14(hv3, hv2, hv1, hv0) = hv3·hv2·hv1·(1 - hv0)`
1. `ind_15(hv3, hv2, hv1, hv0) = hv3·hv2·hv1·hv0`

## Group `decompose_arg`

### Description

1. The helper variables are the decomposition of the instruction's argument, which is held in register `nia`.
1. The helper variable `hv0` is either 0 or 1.
1. The helper variable `hv1` is either 0 or 1.
1. The helper variable `hv2` is either 0 or 1.
1. The helper variable `hv3` is either 0 or 1.

### Polynomials

1. `nia - (8·hv3 + 4·hv2 + 2·hv1 + hv0)`
1. `hv0·(hv0 - 1)`
1. `hv1·(hv1 - 1)`
1. `hv2·(hv2 - 1)`
1. `hv3·(hv3 - 1)`

## Group `step_1`

### Description

1. The instruction pointer increments by 1.

### Polynomials

1. `ip' - (ip + 1)`

## Group `step_2`

### Description

1. The instruction pointer increments by 2.

### Polynomials

1. `ip' - (ip + 2)`

## Group `grow_stack`

### Description

1. The stack element in `st0` is moved into `st1`.
1. The stack element in `st1` is moved into `st2`.
1. The stack element in `st2` is moved into `st3`.
1. The stack element in `st3` is moved into `st4`.
1. The stack element in `st4` is moved into `st5`.
1. The stack element in `st5` is moved into `st6`.
1. The stack element in `st6` is moved into `st7`.
1. The stack element in `st7` is moved into `st8`.
1. The stack element in `st8` is moved into `st9`.
1. The stack element in `st9` is moved into `st10`.
1. The stack element in `st10` is moved into `st11`.
1. The stack element in `st11` is moved into `st12`.
1. The stack element in `st12` is moved into `st13`.
1. The stack element in `st13` is moved into `st14`.
1. The stack element in `st14` is moved into `st15`.
1. The stack element in `st15` is moved to the top of OpStack underflow, i.e., `osv`.
1. The OpStack pointer is incremented by 1.

### Polynomials

1. `st1' - st0`
1. `st2' - st1`
1. `st3' - st2`
1. `st4' - st3`
1. `st5' - st4`
1. `st6' - st5`
1. `st7' - st6`
1. `st8' - st7`
1. `st9' - st8`
1. `st10' - st9`
1. `st11' - st10`
1. `st12' - st11`
1. `st13' - st12`
1. `st14' - st13`
1. `st15' - st14`
1. `osv' - st15`
1. `osp' - (osp + 1)`

## Group `keep_stack`

### Description

1. The stack element in `st0` does not change.
1. The stack element in `st1` does not change.
1. The stack element in `st2` does not change.
1. The stack element in `st3` does not change.
1. The stack element in `st4` does not change.
1. The stack element in `st5` does not change.
1. The stack element in `st6` does not change.
1. The stack element in `st7` does not change.
1. The stack element in `st8` does not change.
1. The stack element in `st9` does not change.
1. The stack element in `st10` does not change.
1. The stack element in `st11` does not change.
1. The stack element in `st12` does not change.
1. The stack element in `st13` does not change.
1. The stack element in `st14` does not change.
1. The stack element in `st15` does not change.
1. The top of the OpStack underflow, i.e., `osv`, does not change.
1. The OpStack pointer does not change.

### Polynomials

1. `st0' - st0`
1. `st1' - st1`
1. `st2' - st2`
1. `st3' - st3`
1. `st4' - st4`
1. `st5' - st5`
1. `st6' - st6`
1. `st7' - st7`
1. `st8' - st8`
1. `st9' - st9`
1. `st10' - st10`
1. `st11' - st11`
1. `st12' - st12`
1. `st13' - st13`
1. `st14' - st14`
1. `st15' - st15`
1. `osv' - osv`
1. `osp' - osp`

## Group `shrink_stack`

This instruction group requires helper variable `hv3` to hold the multiplicative inverse of `(osp - 16)`.
In effect, this means that the OpStack pointer can only be decremented if it is not 16, i.e., if OpStack Underflow Memory is not empty.
Since the stack can only change by one element at a time, this prevents stack underflow.

### Description

1. The stack element in `st1` is moved into `st0`.
1. The stack element in `st2` is moved into `st1`.
1. The stack element in `st3` is moved into `st2`.
1. The stack element in `st4` is moved into `st3`.
1. The stack element in `st5` is moved into `st4`.
1. The stack element in `st6` is moved into `st5`.
1. The stack element in `st7` is moved into `st6`.
1. The stack element in `st8` is moved into `st7`.
1. The stack element in `st9` is moved into `st8`.
1. The stack element in `st10` is moved into `st9`.
1. The stack element in `st11` is moved into `st10`.
1. The stack element in `st12` is moved into `st11`.
1. The stack element in `st13` is moved into `st12`.
1. The stack element in `st14` is moved into `st13`.
1. The stack element in `st15` is moved into `st14`.
1. The stack element at the top of OpStack underflow, i.e., `osv`, is moved into `st15`.
1. The OpStack pointer is decremented by 1.
1. The helper variable register `hv3` holds the inverse of `(osp - 16)`.

### Polynomials

1. `st0' - st1`
1. `st1' - st2`
1. `st2' - st3`
1. `st3' - st4`
1. `st4' - st5`
1. `st5' - st6`
1. `st6' - st7`
1. `st7' - st8`
1. `st8' - st9`
1. `st9' - st10`
1. `st10' - st11`
1. `st11' - st12`
1. `st12' - st13`
1. `st13' - st14`
1. `st14' - st15`
1. `st15' - osv`
1. `osp' - (osp - 1)`
1. `(osp - 16)·hv3 - 1`

## Group `unop`

### Description

1. The stack element in `st1` does not change.
1. The stack element in `st2` does not change.
1. The stack element in `st3` does not change.
1. The stack element in `st4` does not change.
1. The stack element in `st5` does not change.
1. The stack element in `st6` does not change.
1. The stack element in `st7` does not change.
1. The stack element in `st8` does not change.
1. The stack element in `st9` does not change.
1. The stack element in `st10` does not change.
1. The stack element in `st11` does not change.
1. The stack element in `st12` does not change.
1. The stack element in `st13` does not change.
1. The stack element in `st14` does not change.
1. The stack element in `st15` does not change.
1. The top of the OpStack underflow, i.e., `osv`, does not change.
1. The OpStack pointer does not change.

### Polynomials

1. `st1' - st1`
1. `st2' - st2`
1. `st3' - st3`
1. `st4' - st4`
1. `st5' - st5`
1. `st6' - st6`
1. `st7' - st7`
1. `st8' - st8`
1. `st9' - st9`
1. `st10' - st10`
1. `st11' - st11`
1. `st12' - st12`
1. `st13' - st13`
1. `st14' - st14`
1. `st15' - st15`
1. `osv' - osv`
1. `osp' - osp`

## Group `binop`

### Description

1. The stack element in `st2` is moved into `st1`.
1. The stack element in `st3` is moved into `st2`.
1. The stack element in `st4` is moved into `st3`.
1. The stack element in `st5` is moved into `st4`.
1. The stack element in `st6` is moved into `st5`.
1. The stack element in `st7` is moved into `st6`.
1. The stack element in `st8` is moved into `st7`.
1. The stack element in `st9` is moved into `st8`.
1. The stack element in `st10` is moved into `st9`.
1. The stack element in `st11` is moved into `st10`.
1. The stack element in `st12` is moved into `st11`.
1. The stack element in `st13` is moved into `st12`.
1. The stack element in `st14` is moved into `st13`.
1. The stack element in `st15` is moved into `st14`.
1. The stack element at the top of OpStack underflow, i.e., `osv`, is moved into `st15`.
1. The OpStack pointer is decremented by 1.
1. The helper variable register `hv3` holds the inverse of `(osp - 16)`.

### Polynomials

1. `st1' - st2`
1. `st2' - st3`
1. `st3' - st4`
1. `st4' - st5`
1. `st5' - st6`
1. `st6' - st7`
1. `st7' - st8`
1. `st8' - st9`
1. `st9' - st10`
1. `st10' - st11`
1. `st11' - st12`
1. `st12' - st13`
1. `st13' - st14`
1. `st14' - st15`
1. `st15' - osv`
1. `osp' - (osp - 1)`
1. `(osp - 16)·hv3 - 1`

## Group `keep_ram`

### Description

1. The RAM pointer stays the same.
2. The RAM value stays the same.

### Polynomials

1. `ramp' - ramp`
2. `ramv' - ramv`
