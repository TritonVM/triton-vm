# Instruction-Specific Transition Constraints

## Instruction `push` + `a`

In addition to its [instruction groups](instruction-groups.md), this instruction has the following constraints.

### Description

1. The instruction's argument `a` is moved onto the stack.

### Polynomials

1. `st0' - nia`

## Instruction `pop` + `n`

This instruction is fully constrained by its [instruction groups](instruction-groups.md)

## Instruction `divine` + `n`

This instruction is fully constrained by its [instruction groups](instruction-groups.md)

## Instruction `pick` + `i`

This instruction makes use of [indicator polynomials](instruction-groups.md#indicator-polynomials-ind_ihv3-hv2-hv1-hv0).
In addition to its [instruction groups](instruction-groups.md), this instruction has the following constraints.

### Description

For 0 ⩽ `i` < 16:
1. Stack element `i` is moved to the top.
1. For 0 ⩽ `j` < `i`: stack element `j` is shifted down by 1.
1. For `j` > `i`: stack element `j` remains unchanged.

## Instruction `place` + `i`

This instruction makes use of [indicator polynomials](instruction-groups.md#indicator-polynomials-ind_ihv3-hv2-hv1-hv0).
In addition to its [instruction groups](instruction-groups.md), this instruction has the following constraints.

### Description

For 0 ⩽ `i` < 16:
1. Stack element 0 is moved to position `i`.
1. For 0 ⩽ `j` < `i`: stack element `j` is shifted up by 1.
1. For `j` > `i`: stack element `j` remains unchanged.

## Instruction `dup` + `i`

This instruction makes use of [indicator polynomials](instruction-groups.md#indicator-polynomials-ind_ihv3-hv2-hv1-hv0).
In addition to its [instruction groups](instruction-groups.md), this instruction has the following constraints.

### Description

1. If `i` is 0, then `st0` is put on top of the stack and <br />
    if `i` is 1, then `st1` is put on top of the stack and <br />
    if `i` is 2, then `st2` is put on top of the stack and <br />
    if `i` is 3, then `st3` is put on top of the stack and <br />
    if `i` is 4, then `st4` is put on top of the stack and <br />
    if `i` is 5, then `st5` is put on top of the stack and <br />
    if `i` is 6, then `st6` is put on top of the stack and <br />
    if `i` is 7, then `st7` is put on top of the stack and <br />
    if `i` is 8, then `st8` is put on top of the stack and <br />
    if `i` is 9, then `st9` is put on top of the stack and <br />
    if `i` is 10, then `st10` is put on top of the stack and <br />
    if `i` is 11, then `st11` is put on top of the stack and <br />
    if `i` is 12, then `st12` is put on top of the stack and <br />
    if `i` is 13, then `st13` is put on top of the stack and <br />
    if `i` is 14, then `st14` is put on top of the stack and <br />
    if `i` is 15, then `st15` is put on top of the stack.

### Polynomials

1. `ind_0(hv3, hv2, hv1, hv0)·(st0' - st0)`<br />
    `+ ind_1(hv3, hv2, hv1, hv0)·(st0' - st1)`<br />
    `+ ind_2(hv3, hv2, hv1, hv0)·(st0' - st2)`<br />
    `+ ind_3(hv3, hv2, hv1, hv0)·(st0' - st3)`<br />
    `+ ind_4(hv3, hv2, hv1, hv0)·(st0' - st4)`<br />
    `+ ind_5(hv3, hv2, hv1, hv0)·(st0' - st5)`<br />
    `+ ind_6(hv3, hv2, hv1, hv0)·(st0' - st6)`<br />
    `+ ind_7(hv3, hv2, hv1, hv0)·(st0' - st7)`<br />
    `+ ind_8(hv3, hv2, hv1, hv0)·(st0' - st8)`<br />
    `+ ind_9(hv3, hv2, hv1, hv0)·(st0' - st9)`<br />
    `+ ind_10(hv3, hv2, hv1, hv0)·(st0' - st10)`<br />
    `+ ind_11(hv3, hv2, hv1, hv0)·(st0' - st11)`<br />
    `+ ind_12(hv3, hv2, hv1, hv0)·(st0' - st12)`<br />
    `+ ind_13(hv3, hv2, hv1, hv0)·(st0' - st13)`<br />
    `+ ind_14(hv3, hv2, hv1, hv0)·(st0' - st14)`<br />
    `+ ind_15(hv3, hv2, hv1, hv0)·(st0' - st15)`

## Instruction `swap` + `i`

This instruction makes use of [indicator polynomials](instruction-groups.md#indicator-polynomials-ind_ihv3-hv2-hv1-hv0).
In addition to its [instruction groups](instruction-groups.md), this instruction has the following constraints.

### Description

For 0 ⩽ `i` < 16:
1. Stack element `i` is moved to the top.
1. The top stack element is moved to position `i`
1. For `j` ≠ `i`: stack element `j` remains unchanged.

## Instruction `nop`

This instruction is fully constrained by its [instruction groups](instruction-groups.md)

## Instruction `skiz`

For the correct behavior of instruction [`skiz`](instructions.md#skiz), the instruction pointer `ip` needs to increment by either 1, or 2, or 3.
The concrete value depends on the top of the stack `st0` and the next instruction, held in `nia`.

Helper variable `hv0` helps with identifying whether `st0` is 0.
To this end, it holds the inverse-or-zero of `st0`, _i.e._, is 0 if and only if `st0` is 0, and is the inverse of `st0` otherwise.

Efficient arithmetization of instruction `skiz` makes use of one of the properties of
[opcodes](about-instructions.md#regarding-opcodes).
Concretely, the least significant bit of an opcode is 1 if and only if the instruction takes an argument.
The arithmetization of `skiz` can incorporate this simple flag by decomposing `nia` into helper variable registers `hv`,
similarly to how `ci` is (always) deconstructed into instruction bit registers `ib`.
Correct decomposition is guaranteed by employing a [range check](https://en.wikipedia.org/wiki/Bounds_checking).

In addition to its [instruction groups](instruction-groups.md), this instruction has the following constraints.

### Description

1. Helper variable `hv1` is the inverse of `st0` or 0.
1. Helper variable `hv1` is the inverse of `st0` or `st0` is 0.
1. The next instruction `nia` is decomposed into helper variables `hv1` through `hv5`.
1. The indicator helper variable `hv1` is either 0 or 1.
    Here, `hv1 == 1` means that `nia` takes an argument.
1. The helper variable `hv2` is either 0 or 1 or 2 or 3.
1. The helper variable `hv3` is either 0 or 1 or 2 or 3.
1. The helper variable `hv4` is either 0 or 1 or 2 or 3.
1. The helper variable `hv5` is either 0 or 1 or 2 or 3.
1. If `st0` is non-zero, register `ip` is incremented by 1.
If `st0` is 0 and `nia` takes no argument, register `ip` is incremented by 2.
If `st0` is 0 and `nia` takes an argument, register `ip` is incremented by 3.

Written as Disjunctive Normal Form, the last constraint can be expressed as:

10. (Register `st0` is 0 or `ip` is incremented by 1), and
(`st0` has a multiplicative inverse or `hv1` is 1 or `ip` is incremented by 2), and
(`st0` has a multiplicative inverse or `hv1` is 0 or `ip` is incremented by 3).

Since the three cases are mutually exclusive, the three respective polynomials can be summed up into one.

### Polynomials

1. `(st0·hv0 - 1)·hv0`
1. `(st0·hv0 - 1)·st0`
1. `nia - hv1 - 2·hv2 - 8·hv3 - 32·hv4 - 128·hv5`
1. `hv1·(hv1 - 1)`
1. `hv2·(hv2 - 1)·(hv2 - 2)·(hv2 - 3)`
1. `hv3·(hv3 - 1)·(hv3 - 2)·(hv3 - 3)`
1. `hv4·(hv4 - 1)·(hv4 - 2)·(hv4 - 3)`
1. `hv5·(hv5 - 1)·(hv5 - 2)·(hv5 - 3)`
1. `(ip' - (ip + 1)·st0)`<br />
    ` + ((ip' - (ip + 2))·(st0·hv0 - 1)·(hv1 - 1))`<br />
    ` + ((ip' - (ip + 3))·(st0·hv0 - 1)·hv1)`

### Helper variable definitions for `skiz`

1. `hv0 = inverse(st0)` (if `st0 ≠ 0`)
1. `hv1 = nia mod 2`
1. `hv2 = (nia >> 1) mod 4`
1. `hv3 = (nia >> 3) mod 4`
1. `hv4 = (nia >> 5) mod 4`
1. `hv5 = nia >> 7`

## Instruction `call` + `d`

In addition to its [instruction groups](instruction-groups.md), this instruction has the following constraints.

### Description

1. The jump stack pointer `jsp` is incremented by 1.
1. The jump's origin `jso` is set to the current instruction pointer `ip` plus 2.
1. The jump's destination `jsd` is set to the instruction's argument `d`.
1. The instruction pointer `ip` is set to the instruction's argument `d`.

### Polynomials

1. `jsp' - (jsp + 1)`
1. `jso' - (ip + 2)`
1. `jsd' - nia`
1. `ip' - nia`

## Instruction `return`

In addition to its [instruction groups](instruction-groups.md), this instruction has the following constraints.

### Description

1. The jump stack pointer `jsp` is decremented by 1.
1. The instruction pointer `ip` is set to the last call's origin `jso`.

### Polynomials

1. `jsp' - (jsp - 1)`
1. `ip' - jso`

## Instruction `recurse`

In addition to its [instruction groups](instruction-groups.md), this instruction has the following constraints.

### Description

1. The jump stack pointer `jsp` does not change.
1. The last jump's origin `jso` does not change.
1. The last jump's destination `jsd` does not change.
1. The instruction pointer `ip` is set to the last jump's destination `jsd`.

### Polynomials

1. `jsp' - jsp`
1. `jso' - jso`
1. `jsd' - jsd`
1. `ip' - jsd`

## Instruction `recurse_or_return`

In addition to its [instruction groups](instruction-groups.md), this instruction has the following constraints.

### Description

1. If `ST5` equals `ST6`, then `ip` in the next row equals `jso` in the current row.
1. If `ST5` equals `ST6`, then `jsp` decrements by one.
1. If `ST5` equals `ST6`, then `hv0` in the current row is 0.
1. If `ST5` is unequal to `ST6`, then `hv0` in the current row is the inverse of `(ST6 - ST5)`.
1. If `ST5` is unequal to `ST6`, then `ip` in the next row is equal to `jsd` in the current row.
1. If `ST5` is unequal to `ST6`, then `jsp` remains unchanged.
1. If `ST5` is unequal to `ST6`, then `jso` remains unchanged.
1. If `ST5` is unequal to `ST6`, then `jsd` remains unchanged.

### Helper variable definitions for `recurse_or_return`

To help arithmetizing the equality check between `ST5` and `ST6`, helper variable `hv0` is the inverse-or-zero of `(ST6 - ST5)`.

## Instruction `assert`

In addition to its [instruction groups](instruction-groups.md), this instruction has the following constraints.

### Description

1. The current top of the stack `st0` is 1.

### Polynomials

1. `st0 - 1`

## Instruction `halt`

In addition to its [instruction groups](instruction-groups.md), this instruction has the following constraints.

### Description

1. The instruction executed in the following step is instruction `halt`.

### Polynomials

1. `ci' - ci`

## Instruction `read_mem` + `n`

In addition to its [instruction groups](instruction-groups.md), this instruction has the following constraints.

### Description

1. The RAM pointer `st0` is decremented by `n`.
1. If `n` is 1, then `st1` is moved into `st2`<br />
    else if `n` is 2, then `st1` is moved into `st3`<br />
    else if `n` is 3, then `st1` is moved into `st4`<br />
    else if `n` is 4, then `st1` is moved into `st5`<br />
    else if `n` is 5, then `st1` is moved into `st6`.
1. If `n` is 1, then `st2` is moved into `st3`<br />
    else if `n` is 2, then `st2` is moved into `st4`<br />
    else if `n` is 3, then `st2` is moved into `st5`<br />
    else if `n` is 4, then `st2` is moved into `st6`<br />
    else if `n` is 5, then `st2` is moved into `st7`.
1. If `n` is 1, then `st3` is moved into `st4`<br />
    else if `n` is 2, then `st3` is moved into `st5`<br />
    else if `n` is 3, then `st3` is moved into `st6`<br />
    else if `n` is 4, then `st3` is moved into `st7`<br />
    else if `n` is 5, then `st3` is moved into `st8`.
1. If `n` is 1, then `st4` is moved into `st5`<br />
    else if `n` is 2, then `st4` is moved into `st6`<br />
    else if `n` is 3, then `st4` is moved into `st7`<br />
    else if `n` is 4, then `st4` is moved into `st8`<br />
    else if `n` is 5, then `st4` is moved into `st9`.
1. If `n` is 1, then `st5` is moved into `st6`<br />
    else if `n` is 2, then `st5` is moved into `st7`<br />
    else if `n` is 3, then `st5` is moved into `st8`<br />
    else if `n` is 4, then `st5` is moved into `st9`<br />
    else if `n` is 5, then `st5` is moved into `st10`.
1. If `n` is 1, then `st6` is moved into `st7`<br />
    else if `n` is 2, then `st6` is moved into `st8`<br />
    else if `n` is 3, then `st6` is moved into `st9`<br />
    else if `n` is 4, then `st6` is moved into `st10`<br />
    else if `n` is 5, then `st6` is moved into `st11`.
1. If `n` is 1, then `st7` is moved into `st8`<br />
    else if `n` is 2, then `st7` is moved into `st9`<br />
    else if `n` is 3, then `st7` is moved into `st10`<br />
    else if `n` is 4, then `st7` is moved into `st11`<br />
    else if `n` is 5, then `st7` is moved into `st12`.
1. If `n` is 1, then `st8` is moved into `st9`<br />
    else if `n` is 2, then `st8` is moved into `st10`<br />
    else if `n` is 3, then `st8` is moved into `st11`<br />
    else if `n` is 4, then `st8` is moved into `st12`<br />
    else if `n` is 5, then `st8` is moved into `st13`.
1. If `n` is 1, then `st9` is moved into `st10`<br />
    else if `n` is 2, then `st9` is moved into `st11`<br />
    else if `n` is 3, then `st9` is moved into `st12`<br />
    else if `n` is 4, then `st9` is moved into `st13`<br />
    else if `n` is 5, then `st9` is moved into `st14`.
1. If `n` is 1, then `st10` is moved into `st11`<br />
    else if `n` is 2, then `st10` is moved into `st12`<br />
    else if `n` is 3, then `st10` is moved into `st13`<br />
    else if `n` is 4, then `st10` is moved into `st14`<br />
    else if `n` is 5, then `st10` is moved into `st15`.
1. If `n` is 1, then `st11` is moved into `st12`<br />
    else if `n` is 2, then `st11` is moved into `st13`<br />
    else if `n` is 3, then `st11` is moved into `st14`<br />
    else if `n` is 4, then `st11` is moved into `st15`<br />
    else if `n` is 5, then the op stack pointer grows by 5.
1. If `n` is 1, then `st12` is moved into `st13`<br />
    else if `n` is 2, then `st12` is moved into `st14`<br />
    else if `n` is 3, then `st12` is moved into `st15`<br />
    else if `n` is 4, then the op stack pointer grows by 4<br />
    else if `n` is 5, then with the Op Stack Table accumulates `st11` through `st15`.
1. If `n` is 1, then `st13` is moved into `st14`<br />
    else if `n` is 2, then `st13` is moved into `st15`<br />
    else if `n` is 3, then the op stack pointer grows by 3<br />
    else if `n` is 4, then the running product with the Op Stack Table accumulates `st12` through `st15`<br />
    else if `n` is 5, then the running product with the RAM Table accumulates next row's `st1` through `st5`.
1. If `n` is 1, then `st14` is moved into `st15`<br />
    else if `n` is 2, then the op stack pointer grows by 2<br />
    else if `n` is 3, then the running product with the Op Stack Table accumulates `st13` through `st15`<br />
    else if `n` is 4, then the running product with the RAM Table accumulates next row's `st1` through `st4`<br />
1. If `n` is 1, then the op stack pointer grows by 1<br />
    else if `n` is 2, then the running product with the Op Stack Table accumulates `st14` and `st15`<br />
    else if `n` is 3, then the running product with the RAM Table accumulates next row's `st1` through `st3`.
1. If `n` is 1, then the running product with the Op Stack Table accumulates `st15`<br />
    else if `n` is 2, then the running product with the RAM Table accumulates next row's `st1` and `st2`.
1. If `n` is 1, then the running product with the RAM Table accumulates next row's `st1`.

## Instruction `write_mem` + `n`

In addition to its [instruction groups](instruction-groups.md), this instruction has the following constraints.

### Description

1. The RAM pointer `st0` is incremented by `n`.
1. If `n` is 1, then `st2` is moved into `st1`<br />
    else if `n` is 2, then `st3` is moved into `st1`<br />
    else if `n` is 3, then `st4` is moved into `st1`<br />
    else if `n` is 4, then `st5` is moved into `st1`<br />
    else if `n` is 5, then `st6` is moved into `st1`.
1. If `n` is 1, then `st3` is moved into `st2`<br />
    else if `n` is 2, then `st4` is moved into `st2`<br />
    else if `n` is 3, then `st5` is moved into `st2`<br />
    else if `n` is 4, then `st6` is moved into `st2`<br />
    else if `n` is 5, then `st7` is moved into `st2`.
1. If `n` is 1, then `st4` is moved into `st3`<br />
    else if `n` is 2, then `st5` is moved into `st3`<br />
    else if `n` is 3, then `st6` is moved into `st3`<br />
    else if `n` is 4, then `st7` is moved into `st3`<br />
    else if `n` is 5, then `st8` is moved into `st3`.
1. If `n` is 1, then `st5` is moved into `st4`<br />
    else if `n` is 2, then `st6` is moved into `st4`<br />
    else if `n` is 3, then `st7` is moved into `st4`<br />
    else if `n` is 4, then `st8` is moved into `st4`<br />
    else if `n` is 5, then `st9` is moved into `st4`.
1. If `n` is 1, then `st6` is moved into `st5`<br />
    else if `n` is 2, then `st7` is moved into `st5`<br />
    else if `n` is 3, then `st8` is moved into `st5`<br />
    else if `n` is 4, then `st9` is moved into `st5`<br />
    else if `n` is 5, then `st10` is moved into `st5`.
1. If `n` is 1, then `st7` is moved into `st6`<br />
    else if `n` is 2, then `st8` is moved into `st6`<br />
    else if `n` is 3, then `st9` is moved into `st6`<br />
    else if `n` is 4, then `st10` is moved into `st6`<br />
    else if `n` is 5, then `st11` is moved into `st6`.
1. If `n` is 1, then `st8` is moved into `st7`<br />
    else if `n` is 2, then `st9` is moved into `st7`<br />
    else if `n` is 3, then `st10` is moved into `st7`<br />
    else if `n` is 4, then `st11` is moved into `st7`<br />
    else if `n` is 5, then `st12` is moved into `st7`.
1. If `n` is 1, then `st9` is moved into `st8`<br />
    else if `n` is 2, then `st10` is moved into `st8`<br />
    else if `n` is 3, then `st11` is moved into `st8`<br />
    else if `n` is 4, then `st12` is moved into `st8`<br />
    else if `n` is 5, then `st13` is moved into `st8`.
1. If `n` is 1, then `st10` is moved into `st9`<br />
    else if `n` is 2, then `st11` is moved into `st9`<br />
    else if `n` is 3, then `st12` is moved into `st9`<br />
    else if `n` is 4, then `st13` is moved into `st9`<br />
    else if `n` is 5, then `st14` is moved into `st9`.
1. If `n` is 1, then `st11` is moved into `st10`<br />
    else if `n` is 2, then `st12` is moved into `st10`<br />
    else if `n` is 3, then `st13` is moved into `st10`<br />
    else if `n` is 4, then `st14` is moved into `st10`<br />
    else if `n` is 5, then `st15` is moved into `st10`.
1. If `n` is 1, then `st12` is moved into `st11`<br />
    else if `n` is 2, then `st13` is moved into `st11`<br />
    else if `n` is 3, then `st14` is moved into `st11`<br />
    else if `n` is 4, then `st15` is moved into `st11`<br />
    else if `n` is 5, then the op stack pointer shrinks by 5.
1. If `n` is 1, then `st13` is moved into `st12`<br />
    else if `n` is 2, then `st14` is moved into `st12`<br />
    else if `n` is 3, then `st15` is moved into `st12`<br />
    else if `n` is 4, then the op stack pointer shrinks by 4<br />
    else if `n` is 5, then the running product with the Op Stack Table accumulates next row's `st11` through `st15`.
1. If `n` is 1, then `st14` is moved into `st13`<br />
    else if `n` is 2, then `st15` is moved into `st13`<br />
    else if `n` is 3, then the op stack pointer shrinks by 3<br />
    else if `n` is 4, then the running product with the Op Stack Table accumulates next row's `st12` through `st15`<br />
    else if `n` is 5, then the running product with the RAM Table accumulates `st1` through `st5`.
1. If `n` is 1, then `st15` is moved into `st14`<br />
    else if `n` is 2, then the op stack pointer shrinks by 2<br />
    else if `n` is 3, then the running product with the Op Stack Table accumulates next row's `st13` through `st15`<br />
    else if `n` is 4, then the running product with the RAM Table accumulates `st1` through `st4`.
1. If `n` is 1, then the op stack pointer shrinks by 1<br />
    else if `n` is 2, then the running product with the Op Stack Table accumulates next row's `st14` and `st15`<br />
    else if `n` is 3, then the running product with the RAM Table accumulates `st1` through `st2`.
1. If `n` is 1, then the running product with the Op Stack Table accumulates next row's `st15`<br />
    else if `n` is 2, then the running product with the RAM Table accumulates `st1` and `st1`.
1. If `n` is 1, then the running product with the RAM Table accumulates `st1`.

## Instruction `hash`

In addition to its [instruction groups](instruction-groups.md), this instruction has the following constraints.

### Description

1. `st10` is moved into `st5`.
1. `st11` is moved into `st6`.
1. `st12` is moved into `st7`.
1. `st13` is moved into `st8`.
1. `st14` is moved into `st9`.
1. `st15` is moved into `st10`.
1. The op stack pointer shrinks by 5.
1. The running product with the Op Stack Table accumulates next row's `st11` through `st15`.

### Polynomials

1. `st5' - st10`
1. `st6' - st11`
1. `st7' - st12`
1. `st8' - st13`
1. `st9' - st14`
1. `st10' - st15`
1. `op_stack_pointer' - op_stack_pointer + 5`
1. `RunningProductOpStackTable' - RunningProductOpStackTable·(🪤 - 🍋·clk - 🍊 - 🍉·op_stack_pointer' - 🫒·st15')`<br />
    `·(🪤 - 🍋·clk - 🍊 - 🍉·(op_stack_pointer' + 1) - 🫒·st14')`<br/>
    `·(🪤 - 🍋·clk - 🍊 - 🍉·(op_stack_pointer' + 2) - 🫒·st13')`<br/>
    `·(🪤 - 🍋·clk - 🍊 - 🍉·(op_stack_pointer' + 3) - 🫒·st12')`<br/>
    `·(🪤 - 🍋·clk - 🍊 - 🍉·(op_stack_pointer' + 4) - 🫒·st11')`

## Instruction `assert_vector`

In addition to its [instruction groups](instruction-groups.md), this instruction has the following constraints.

### Description

1. `st0` is equal to `st5`.
1. `st1` is equal to `st6`.
1. `st2` is equal to `st7`.
1. `st3` is equal to `st8`.
1. `st4` is equal to `st9`.
1. `st10` is moved into `st5`.
1. `st11` is moved into `st6`.
1. `st12` is moved into `st7`.
1. `st13` is moved into `st8`.
1. `st14` is moved into `st9`.
1. `st15` is moved into `st10`.
1. The op stack pointer shrinks by 5.
1. The running product with the Op Stack Table accumulates next row's `st11` through `st15`.

### Polynomials

1. `st5 - st0`
1. `st6 - st1`
1. `st7 - st2`
1. `st8 - st3`
1. `st9 - st4`
1. `st5' - st10`
1. `st6' - st11`
1. `st7' - st12`
1. `st8' - st13`
1. `st9' - st14`
1. `st10' - st15`
1. `op_stack_pointer' - op_stack_pointer + 5`
1. `RunningProductOpStackTable' - RunningProductOpStackTable·(🪤 - 🍋·clk - 🍊 - 🍉·op_stack_pointer' - 🫒·st15')`<br />
    `·(🪤 - 🍋·clk - 🍊 - 🍉·(op_stack_pointer' + 1) - 🫒·st14')`<br/>
    `·(🪤 - 🍋·clk - 🍊 - 🍉·(op_stack_pointer' + 2) - 🫒·st13')`<br/>
    `·(🪤 - 🍋·clk - 🍊 - 🍉·(op_stack_pointer' + 3) - 🫒·st12')`<br/>
    `·(🪤 - 🍋·clk - 🍊 - 🍉·(op_stack_pointer' + 4) - 🫒·st11')`

## Instruction `sponge_init`

This instruction is fully constrained by its [instruction groups](instruction-groups.md)
Beyond that, correct transition is guaranteed by the [Hash Table](hash-table.md).

## Instruction `sponge_absorb`

In addition to its [instruction groups](instruction-groups.md), this instruction has the following constraints.
Beyond that, correct transition is guaranteed by the [Hash Table](hash-table.md).

### Description

1. `st10` is moved into `st0`.
1. `st11` is moved into `st1`.
1. `st12` is moved into `st2`.
1. `st13` is moved into `st3`.
1. `st14` is moved into `st4`.
1. `st15` is moved into `st5`.
1. The op stack pointer shrinks by 10.
1. The running product with the Op Stack Table accumulates next row's `st6` through `st15`.

### Polynomials

1. `st0' - st10`
1. `st1' - st11`
1. `st2' - st12`
1. `st3' - st13`
1. `st4' - st14`
1. `st5' - st15`
1. `op_stack_pointer' - op_stack_pointer + 10`
1. `RunningProductOpStackTable' - RunningProductOpStackTable·(🪤 - 🍋·clk - 🍊 - 🍉·op_stack_pointer' - 🫒·st15')`<br />
    `·(🪤 - 🍋·clk - 🍊 - 🍉·(op_stack_pointer' + 1) - 🫒·st14')`<br/>
    `·(🪤 - 🍋·clk - 🍊 - 🍉·(op_stack_pointer' + 2) - 🫒·st13')`<br/>
    `·(🪤 - 🍋·clk - 🍊 - 🍉·(op_stack_pointer' + 3) - 🫒·st12')`<br/>
    `·(🪤 - 🍋·clk - 🍊 - 🍉·(op_stack_pointer' + 4) - 🫒·st11')`<br/>
    `·(🪤 - 🍋·clk - 🍊 - 🍉·(op_stack_pointer' + 5) - 🫒·st10')`<br/>
    `·(🪤 - 🍋·clk - 🍊 - 🍉·(op_stack_pointer' + 6) - 🫒·st9')`<br/>
    `·(🪤 - 🍋·clk - 🍊 - 🍉·(op_stack_pointer' + 7) - 🫒·st8')`<br/>
    `·(🪤 - 🍋·clk - 🍊 - 🍉·(op_stack_pointer' + 8) - 🫒·st7')`<br/>
    `·(🪤 - 🍋·clk - 🍊 - 🍉·(op_stack_pointer' + 9) - 🫒·st6')`

## Instruction `sponge_absorb_mem`

In addition to its [instruction groups](instruction-groups.md), this instruction has the following constraints.
Beyond that, correct transition is guaranteed by the [Hash Table](hash-table.md) and the [RAM Table](random-access-memory-table.md).

### Description

1. `st0` is incremented by 10.

## Instruction `sponge_squeeze`

In addition to its [instruction groups](instruction-groups.md), this instruction has the following constraints.
Beyond that, correct transition is guaranteed by the [Hash Table](hash-table.md).

### Description

1. `st0` is moved into `st10`.
1. `st1` is moved into `st11`.
1. `st2` is moved into `st12`.
1. `st3` is moved into `st13`.
1. `st4` is moved into `st14`.
1. `st5` is moved into `st15`.
1. The op stack pointer grows by 10.
1. The running product with the Op Stack Table accumulates `st6` through `st15`.

### Polynomials

1. `st10' - st0`
1. `st11' - st1`
1. `st12' - st2`
1. `st13' - st3`
1. `st14' - st4`
1. `st15' - st5`
1. `op_stack_pointer' - op_stack_pointer - 10`
1. `RunningProductOpStackTable' - RunningProductOpStackTable·(🪤 - 🍋·clk - 🍉·op_stack_pointer - 🫒·st15)`<br />
    `·(🪤 - 🍋·clk - 🍉·(op_stack_pointer + 1) - 🫒·st14)`<br/>
    `·(🪤 - 🍋·clk - 🍉·(op_stack_pointer + 2) - 🫒·st13)`<br/>
    `·(🪤 - 🍋·clk - 🍉·(op_stack_pointer + 3) - 🫒·st12)`<br/>
    `·(🪤 - 🍋·clk - 🍉·(op_stack_pointer + 4) - 🫒·st11)`<br/>
    `·(🪤 - 🍋·clk - 🍉·(op_stack_pointer + 5) - 🫒·st10)`<br/>
    `·(🪤 - 🍋·clk - 🍉·(op_stack_pointer + 6) - 🫒·st9)`<br/>
    `·(🪤 - 🍋·clk - 🍉·(op_stack_pointer + 7) - 🫒·st8)`<br/>
    `·(🪤 - 🍋·clk - 🍉·(op_stack_pointer + 8) - 🫒·st7)`<br/>
    `·(🪤 - 🍋·clk - 🍉·(op_stack_pointer + 9) - 🫒·st6)`

## Instruction `add`

In addition to its [instruction groups](instruction-groups.md), this instruction has the following constraints.

### Description

1. The sum of the top two stack elements is moved into the top of the stack.

### Polynomials

1. `st0' - (st0 + st1)`

## Instruction `mul`

In addition to its [instruction groups](instruction-groups.md), this instruction has the following constraints.

### Description

1. The product of the top two stack elements is moved into the top of the stack.

### Polynomials

1. `st0' - st0·st1`

## Instruction `invert`

In addition to its [instruction groups](instruction-groups.md), this instruction has the following constraints.

### Description

1. The top of the stack's inverse is moved into the top of the stack.

### Polynomials

1. `st0'·st0 - 1`

## Instruction `eq`

In addition to its [instruction groups](instruction-groups.md), this instruction has the following constraints.

### Description

1. Helper variable `hv0` is the inverse of the difference of the stack's two top-most elements or 0.
1. Helper variable `hv0` is the inverse of the difference of the stack's two top-most elements or the difference is 0.
1. The new top of the stack is 1 if the difference between the stack's two top-most elements is not invertible, 0 otherwise.

### Polynomials

1. `hv0·(hv0·(st1 - st0) - 1)`
1. `(st1 - st0)·(hv0·(st1 - st0) - 1)`
1. `st0' - (1 - hv0·(st1 - st0))`

### Helper variable definitions for `eq`

1. `hv0 = inverse(rhs - lhs)` if `rhs ≠ lhs`.
1. `hv0 = 0` if `rhs = lhs`.

## Instruction `split`

In addition to its [instruction groups](instruction-groups.md), this instruction has the following constraints.
Part of the correct transition, namely the range check on the instruction's result, is guaranteed by the [U32 Table](u32-table.md).

### Description

1. The top of the stack is decomposed as 32-bit chunks into the stack's top-most two elements.
1. Helper variable `hv0` holds the inverse of $2^{32} - 1$ subtracted from the high 32 bits or the low 32 bits are 0.

### Polynomials

1. `st0 - (2^32·st1' + st0')`
1. `st0'·(hv0·(st1' - (2^32 - 1)) - 1)`

### Helper variable definitions for `split`

Given the high 32 bits of `st0` as `hi = st0 >> 32` and the low 32 bits of `st0` as `lo = st0 & 0xffff_ffff`,

1. `hv0 = (hi - (2^32 - 1))` if `lo ≠ 0`.
1. `hv0 = 0` if `lo = 0`.

## Instruction `lt`

This instruction is fully constrained by its [instruction groups](instruction-groups.md)
Beyond that, correct transition is guaranteed by the [U32 Table](u32-table.md).

## Instruction `and`

This instruction is fully constrained by its [instruction groups](instruction-groups.md)
Beyond that, correct transition is guaranteed by the [U32 Table](u32-table.md).

## Instruction `xor`

This instruction is fully constrained by its [instruction groups](instruction-groups.md)
Beyond that, correct transition is guaranteed by the [U32 Table](u32-table.md).

## Instruction `log2floor`

This instruction is fully constrained by its [instruction groups](instruction-groups.md)
Beyond that, correct transition is guaranteed by the [U32 Table](u32-table.md).

## Instruction `pow`

This instruction is fully constrained by its [instruction groups](instruction-groups.md)
Beyond that, correct transition is guaranteed by the [U32 Table](u32-table.md).

## Instruction `div_mod`

Recall that instruction `div_mod` takes stack `_ d n` and computes `_ q r` where `n` is the numerator, `d` is the denominator, `r` is the remainder, and `q` is the quotient.
The following two properties are guaranteed by the [U32 Table](u32-table.md):
1. The remainder `r` is smaller than the denominator `d`, and
1. all four of `n`, `d`, `q`, and `r` are u32s.

In addition to its [instruction groups](instruction-groups.md), this instruction has the following constraints.

### Description

1. Numerator is quotient times denominator plus remainder: `n == q·d + r`.

### Polynomials

1. `st0 - st1·st1' - st0'`
1. `st2' - st2`

## Instruction `pop_count`

This instruction is fully constrained by its [instruction groups](instruction-groups.md)
Beyond that, correct transition is guaranteed by the [U32 Table](u32-table.md).

## Instruction `xx_add`

In addition to its [instruction groups](instruction-groups.md), this instruction has the following constraints.

### Description

1. The result of adding `st0` to `st3` is moved into `st0`.
1. The result of adding `st1` to `st4` is moved into `st1`.
1. The result of adding `st2` to `st5` is moved into `st2`.
1. `st6` is moved into `st3`.
1. `st7` is moved into `st4`.
1. `st8` is moved into `st5`.
1. `st9` is moved into `st6`.
1. `st10` is moved into `st7`.
1. `st11` is moved into `st8`.
1. `st12` is moved into `st9`.
1. `st13` is moved into `st10`.
1. `st14` is moved into `st11`.
1. `st15` is moved into `st12`.
1. The op stack pointer shrinks by 3.
1. The running product with the Op Stack Table accumulates next row's `st13` through `st15`.

### Polynomials

1. `st0' - (st0 + st3)`
1. `st1' - (st1 + st4)`
1. `st2' - (st2 + st5)`
1. `st3' - st6`
1. `st4' - st7`
1. `st5' - st8`
1. `st6' - st9`
1. `st7' - st10`
1. `st8' - st11`
1. `st9' - st12`
1. `st10' - st13`
1. `st11' - st14`
1. `st12' - st15`
1. `op_stack_pointer' - op_stack_pointer + 3`
1. `RunningProductOpStackTable' - RunningProductOpStackTable·(🪤 - 🍋·clk - 🍊 - 🍉·op_stack_pointer' - 🫒·st15')`<br />
    `·(🪤 - 🍋·clk - 🍊 - 🍉·(op_stack_pointer' + 1) - 🫒·st14')`<br/>
    `·(🪤 - 🍋·clk - 🍊 - 🍉·(op_stack_pointer' + 2) - 🫒·st13')`

## Instruction `xx_mul`

In addition to its [instruction groups](instruction-groups.md), this instruction has the following constraints.

### Description

1. The coefficient of x^0 of multiplying the two X-Field elements on the stack is moved into `st0`.
1. The coefficient of x^1 of multiplying the two X-Field elements on the stack is moved into `st1`.
1. The coefficient of x^2 of multiplying the two X-Field elements on the stack is moved into `st2`.
1. `st6` is moved into `st3`.
1. `st7` is moved into `st4`.
1. `st8` is moved into `st5`.
1. `st9` is moved into `st6`.
1. `st10` is moved into `st7`.
1. `st11` is moved into `st8`.
1. `st12` is moved into `st9`.
1. `st13` is moved into `st10`.
1. `st14` is moved into `st11`.
1. `st15` is moved into `st12`.
1. The op stack pointer shrinks by 3.
1. The running product with the Op Stack Table accumulates next row's `st13` through `st15`.

### Polynomials

1. `st0' - (st0·st3 - st2·st4 - st1·st5)`
1. `st1' - (st1·st3 + st0·st4 - st2·st5 + st2·st4 + st1·st5)`
1. `st2' - (st2·st3 + st1·st4 + st0·st5 + st2·st5)`
1. `st3' - st6`
1. `st4' - st7`
1. `st5' - st8`
1. `st6' - st9`
1. `st7' - st10`
1. `st8' - st11`
1. `st9' - st12`
1. `st10' - st13`
1. `st11' - st14`
1. `st12' - st15`
1. `op_stack_pointer' - op_stack_pointer + 3`
1. `RunningProductOpStackTable' - RunningProductOpStackTable·(🪤 - 🍋·clk - 🍊 - 🍉·op_stack_pointer' - 🫒·st15')`<br />
   `·(🪤 - 🍋·clk - 🍊 - 🍉·(op_stack_pointer' + 1) - 🫒·st14')`<br/>
   `·(🪤 - 🍋·clk - 🍊 - 🍉·(op_stack_pointer' + 2) - 🫒·st13')`

## Instruction `x_invert`

In addition to its [instruction groups](instruction-groups.md), this instruction has the following constraints.

### Description

1. The coefficient of x^0 of multiplying X-Field element on top of the current stack and on top of the next stack is 1.
1. The coefficient of x^1 of multiplying X-Field element on top of the current stack and on top of the next stack is 0.
1. The coefficient of x^2 of multiplying X-Field element on top of the current stack and on top of the next stack is 0.

### Polynomials

1. `st0·st0' - st2·st1' - st1·st2' - 1`
1. `st1·st0' + st0·st1' - st2·st2' + st2·st1' + st1·st2'`
1. `st2·st0' + st1·st1' + st0·st2' + st2·st2'`

## Instruction `xb_mul`

In addition to its [instruction groups](instruction-groups.md), this instruction has the following constraints.

### Description

1. The result of multiplying the top of the stack with the X-Field element's coefficient for x^0 is moved into `st0`.
1. The result of multiplying the top of the stack with the X-Field element's coefficient for x^1 is moved into `st1`.
1. The result of multiplying the top of the stack with the X-Field element's coefficient for x^2 is moved into `st2`.
1. `st4` is moved into `st3`.
1. `st5` is moved into `st4`.
1. `st6` is moved into `st5`.
1. `st7` is moved into `st6`.
1. `st8` is moved into `st7`.
1. `st9` is moved into `st8`.
1. `st10` is moved into `st9`.
1. `st11` is moved into `st10`.
1. `st12` is moved into `st11`.
1. `st13` is moved into `st12`.
1. `st14` is moved into `st13`.
1. `st15` is moved into `st14`.
1. The op stack pointer shrinks by 3.
1. The running product with the Op Stack Table accumulates next row's `st15`.

### Polynomials

1. `st0' - st0·st1`
1. `st1' - st0·st2`
1. `st2' - st0·st3`
1. `st3' - st4`
1. `st4' - st5`
1. `st5' - stt`
1. `st6' - st7`
1. `st7' - st8`
1. `st8' - st9`
1. `st9' - st10`
1. `st10' - st11`
1. `st11' - st12`
1. `st12' - st13`
1. `st13' - st14`
1. `st14' - st15`
1. `op_stack_pointer' - op_stack_pointer + 1`
1. `RunningProductOpStackTable' - RunningProductOpStackTable·(🪤 - 🍋·clk - 🍊 - 🍉·op_stack_pointer' - 🫒·st15')`

## Instruction `read_io` + `n`

In addition to its [instruction groups](instruction-groups.md), this instruction has the following constraints.

### Description

1. If `n` is 1, the running evaluation for standard input accumulates next row's `st0`<br />
    else if `n` is 2, the running evaluation for standard input accumulates next row's `st0` and `st1`<br />
    else if `n` is 3, the running evaluation for standard input accumulates next row's `st0` through `st2`<br />
    else if `n` is 4, the running evaluation for standard input accumulates next row's `st0` through `st3`<br />
    else if `n` is 5, the running evaluation for standard input accumulates next row's `st0` through `st4`.

### Polynomials

1. `ind_1(hv3, hv2, hv1, hv0)·(RunningEvaluationStandardInput' - 🛏·RunningEvaluationStandardInput - st0')`<br />
    `+ ind_2(hv3, hv2, hv1, hv0)·(RunningEvaluationStandardInput' - 🛏·(🛏·RunningEvaluationStandardInput - st0') - st1')`<br />
    `+ ind_3(hv3, hv2, hv1, hv0)·(RunningEvaluationStandardInput' - 🛏·(🛏·(🛏·RunningEvaluationStandardInput - st0') - st1') - st2')`<br />
    `+ ind_4(hv3, hv2, hv1, hv0)·(RunningEvaluationStandardInput' - 🛏·(🛏·(🛏·(🛏·RunningEvaluationStandardInput - st0') - st1') - st2') - st3')`<br />
    `+ ind_5(hv3, hv2, hv1, hv0)·(RunningEvaluationStandardInput' - 🛏·(🛏·(🛏·(🛏·(🛏·RunningEvaluationStandardInput - st0') - st1') - st2') - st3') - st4')`

## Instruction `write_io` + `n`

In addition to its [instruction groups](instruction-groups.md), this instruction has the following constraints.

### Description

1. If `n` is 1, the running evaluation for standard output accumulates `st0`<br />
    else if `n` is 2, the running evaluation for standard output accumulates `st0` and `st1`<br />
    else if `n` is 3, the running evaluation for standard output accumulates `st0` through `st2`<br />
    else if `n` is 4, the running evaluation for standard output accumulates `st0` through `st3`<br />
    else if `n` is 5, the running evaluation for standard output accumulates `st0` through `st4`.

### Polynomials

1. `ind_1(hv3, hv2, hv1, hv0)·(RunningEvaluationStandardOutput' - 🧯·RunningEvaluationStandardOutput - st0)`<br />
    `+ ind_2(hv3, hv2, hv1, hv0)·(RunningEvaluationStandardOutput' - 🧯·(🧯·RunningEvaluationStandardOutput - st0) - st1)`<br />
    `+ ind_3(hv3, hv2, hv1, hv0)·(RunningEvaluationStandardOutput' - 🧯·(🧯·(🧯·RunningEvaluationStandardOutput - st0) - st1) - st2)`<br />
    `+ ind_4(hv3, hv2, hv1, hv0)·(RunningEvaluationStandardOutput' - 🧯·(🧯·(🧯·(🧯·RunningEvaluationStandardOutput - st0) - st1) - st2) - st3)`<br />
    `+ ind_5(hv3, hv2, hv1, hv0)·(RunningEvaluationStandardOutput' - 🧯·(🧯·(🧯·(🧯·(🧯·RunningEvaluationStandardOutput - st0) - st1) - st2) - st3) - st4)`

## Instruction `merkle_step`

Recall that in a Merkle tree, the indices of left (respectively right) leaves have 0 (respectively 1) as their least significant bit.
This motivates the use of a helper variable to hold that least significant bit.

In addition to its [instruction groups](instruction-groups.md), this instruction has the following constraints.
The two [Evaluation Arguments also used for instruction hash](processor-table.md#transition-constraints) guarantee correct transition of the top of the stack.

### Description

1. Helper variable `hv5` is either 0 or 1.
1. `st5` is shifted by 1 bit to the right. In other words, twice `st5` in the next row plus `hv5` equals `st5` in the current row.

### Helper variable definitions for `merkle_step`

1. `hv0` through `hv4` hold the sibling digest of the node indicated by `st5`, as read from the interface for non-deterministic input.
1. `hv5` holds the result of `st5 % 2`, the Merkle tree node index's least significant bit, indicating whether it is a left or right node.

## Instruction `merkle_step_mem`

In addition to its [instruction groups](instruction-groups.md), this instruction has the following constraints.
The two [Evaluation Arguments also used for instruction hash](processor-table.md#transition-constraints) guarantee correct transition of the top of the stack.
See also [`merkle_step`](#instruction-merkle_step).

### Description

1. Stack element `st6` does not change.
1. Stack element `st7` is incremented by 5.
1. Helper variable `hv5` is either 0 or 1.
1. `st5` is shifted by 1 bit to the right. In other words, twice `st5` in the next row plus `hv5` equals `st5` in the current row.
1. The running product with the RAM Table accumulates `hv0` through `hv4` using the (appropriately adjusted) RAM pointer held in `st7`.

### Helper variable definitions for `merkle_step_mem`

1. `hv0` through `hv4` hold the sibling digest of the node indicated by `st5`, as read from RAM at the address held in `st7`.
1. `hv5` holds the result of `st5 % 2`, the Merkle tree node index's least significant bit, indicating whether it is a left or right node.

## Instruction `xx_dot_step`

In addition to its [instruction groups](instruction-groups.md), this instruction has the following constraints.

### Description

1. Store `(RAM[st0], RAM[st0+1], RAM[st0+2])` in `(hv0, hv1, hv2)`.
1. Store `(RAM[st1], RAM[st1+1], RAM[st1+2])` in `(hv3, hv4, hv5)`.
1. Add `(hv0 + hv1·x + hv2·x²) · (hv3 + hv4·x + hv5·x²)` into `(st2, st3, st4)`
1. Increase the pointers: `st0` and `st1` by 3 each.

## Instruction `xb_dot_step`

In addition to its [instruction groups](instruction-groups.md), this instruction has the following constraints.

### Description

1. Store `RAM[st0]` in `hv0`.
1. Store `(RAM[st1], RAM[st1+1], RAM[st1+2])` in `(hv1, hv2, hv3)`.
1. Add `hv0 · (hv1 + hv2·x + hv3·x²)` into `(st1, st2, st3)`
1. Increase the pointers: `st0` and `st1` by 1 and 3, respectively.
