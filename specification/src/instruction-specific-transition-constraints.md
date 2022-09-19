# Instruction-Specific Transition Constraints

## Instruction `pop`

This instruction has no additional transition constraints.

## Instruction `push` + `a`

### Description

1. The instruction's argument `a` is moved onto the stack.

### Polynomials

1. `st0' - nia`

## Instruction `divine`

This instruction has no additional transition constraints.

## Instruction `dup` + `i`

This instruction makes use of [indicator polynomials](instruction-groups.md#indicator-polynomials-ind_ihv3-hv2-hv1-hv0).
For their definition, please refer to the corresponding section.

### Description

1. If `i` is 0, then `st0` is put on top of the stack.
1. If `i` is 1, then `st1` is put on top of the stack.
1. If `i` is 2, then `st2` is put on top of the stack.
1. If `i` is 3, then `st3` is put on top of the stack.
1. If `i` is 4, then `st4` is put on top of the stack.
1. If `i` is 5, then `st5` is put on top of the stack.
1. If `i` is 6, then `st6` is put on top of the stack.
1. If `i` is 7, then `st7` is put on top of the stack.
1. If `i` is 8, then `st8` is put on top of the stack.
1. If `i` is 9, then `st9` is put on top of the stack.
1. If `i` is 10, then `st10` is put on top of the stack.
1. If `i` is 11, then `st11` is put on top of the stack.
1. If `i` is 12, then `st12` is put on top of the stack.
1. If `i` is 13, then `st13` is put on top of the stack.
1. If `i` is 14, then `st14` is put on top of the stack.
1. If `i` is 15, then `st15` is put on top of the stack.

### Polynomials

1. `ind_0(hv3, hv2, hv1, hv0)·(st0' - st0)`
1. `ind_1(hv3, hv2, hv1, hv0)·(st0' - st1)`
1. `ind_2(hv3, hv2, hv1, hv0)·(st0' - st2)`
1. `ind_3(hv3, hv2, hv1, hv0)·(st0' - st3)`
1. `ind_4(hv3, hv2, hv1, hv0)·(st0' - st4)`
1. `ind_5(hv3, hv2, hv1, hv0)·(st0' - st5)`
1. `ind_6(hv3, hv2, hv1, hv0)·(st0' - st6)`
1. `ind_7(hv3, hv2, hv1, hv0)·(st0' - st7)`
1. `ind_8(hv3, hv2, hv1, hv0)·(st0' - st8)`
1. `ind_9(hv3, hv2, hv1, hv0)·(st0' - st9)`
1. `ind_10(hv3, hv2, hv1, hv0)·(st0' - st10)`
1. `ind_11(hv3, hv2, hv1, hv0)·(st0' - st11)`
1. `ind_12(hv3, hv2, hv1, hv0)·(st0' - st12)`
1. `ind_13(hv3, hv2, hv1, hv0)·(st0' - st13)`
1. `ind_14(hv3, hv2, hv1, hv0)·(st0' - st14)`
1. `ind_15(hv3, hv2, hv1, hv0)·(st0' - st15)`

### Helper variable definitions for `dup` + `i`

For `dup` + `i`, helper variables contain the binary decomposition of `i`:

1. `hv0 = i % 2`
1. `hv1 = (i >> 1) % 2`
1. `hv2 = (i >> 2) % 2`
1. `hv3 = (i >> 3) % 2`

## Instruction `swap` + `i`

This instruction makes use of [indicator polynomials](instruction-groups.md#indicator-polynomials-ind_ihv3-hv2-hv1-hv0).
For their definition, please refer to the corresponding section.

### Description

1. Argument `i` is not 0.
1. If `i` is 1, then `st0` is moved into `st1`.
1. If `i` is 2, then `st0` is moved into `st2`.
1. If `i` is 3, then `st0` is moved into `st3`.
1. If `i` is 4, then `st0` is moved into `st4`.
1. If `i` is 5, then `st0` is moved into `st5`.
1. If `i` is 6, then `st0` is moved into `st6`.
1. If `i` is 7, then `st0` is moved into `st7`.
1. If `i` is 8, then `st0` is moved into `st8`.
1. If `i` is 9, then `st0` is moved into `st9`.
1. If `i` is 10, then `st0` is moved into `st10`.
1. If `i` is 11, then `st0` is moved into `st11`.
1. If `i` is 12, then `st0` is moved into `st12`.
1. If `i` is 13, then `st0` is moved into `st13`.
1. If `i` is 14, then `st0` is moved into `st14`.
1. If `i` is 15, then `st0` is moved into `st15`.
1. If `i` is 1, then `st1` is moved into `st0`.
1. If `i` is 2, then `st2` is moved into `st0`.
1. If `i` is 3, then `st3` is moved into `st0`.
1. If `i` is 4, then `st4` is moved into `st0`.
1. If `i` is 5, then `st5` is moved into `st0`.
1. If `i` is 6, then `st6` is moved into `st0`.
1. If `i` is 7, then `st7` is moved into `st0`.
1. If `i` is 8, then `st8` is moved into `st0`.
1. If `i` is 9, then `st9` is moved into `st0`.
1. If `i` is 10, then `st10` is moved into `st0`.
1. If `i` is 11, then `st11` is moved into `st0`.
1. If `i` is 12, then `st12` is moved into `st0`.
1. If `i` is 13, then `st13` is moved into `st0`.
1. If `i` is 14, then `st14` is moved into `st0`.
1. If `i` is 15, then `st15` is moved into `st0`.
1. If `i` is not 1, then `st1` does not change.
1. If `i` is not 2, then `st2` does not change.
1. If `i` is not 3, then `st3` does not change.
1. If `i` is not 4, then `st4` does not change.
1. If `i` is not 5, then `st5` does not change.
1. If `i` is not 6, then `st6` does not change.
1. If `i` is not 7, then `st7` does not change.
1. If `i` is not 8, then `st8` does not change.
1. If `i` is not 9, then `st9` does not change.
1. If `i` is not 10, then `st10` does not change.
1. If `i` is not 11, then `st11` does not change.
1. If `i` is not 12, then `st12` does not change.
1. If `i` is not 13, then `st13` does not change.
1. If `i` is not 14, then `st14` does not change.
1. If `i` is not 15, then `st15` does not change.
1. The top of the OpStack underflow, i.e., `osv`, does not change.
1. The OpStack pointer does not change.
1. If `i` is not 1, then the RAM value `ramv` does not change.

### Polynomials

1. `ind_0(hv3, hv2, hv1, hv0)`
1. `ind_1(hv3, hv2, hv1, hv0)·(st1' - st0)`
1. `ind_2(hv3, hv2, hv1, hv0)·(st2' - st0)`
1. `ind_3(hv3, hv2, hv1, hv0)·(st3' - st0)`
1. `ind_4(hv3, hv2, hv1, hv0)·(st4' - st0)`
1. `ind_5(hv3, hv2, hv1, hv0)·(st5' - st0)`
1. `ind_6(hv3, hv2, hv1, hv0)·(st6' - st0)`
1. `ind_7(hv3, hv2, hv1, hv0)·(st7' - st0)`
1. `ind_8(hv3, hv2, hv1, hv0)·(st8' - st0)`
1. `ind_9(hv3, hv2, hv1, hv0)·(st9' - st0)`
1. `ind_10(hv3, hv2, hv1, hv0)·(st10' - st0)`
1. `ind_11(hv3, hv2, hv1, hv0)·(st11' - st0)`
1. `ind_12(hv3, hv2, hv1, hv0)·(st12' - st0)`
1. `ind_13(hv3, hv2, hv1, hv0)·(st13' - st0)`
1. `ind_14(hv3, hv2, hv1, hv0)·(st14' - st0)`
1. `ind_15(hv3, hv2, hv1, hv0)·(st15' - st0)`
1. `ind_1(hv3, hv2, hv1, hv0)·(st0' - st1)`
1. `ind_2(hv3, hv2, hv1, hv0)·(st0' - st2)`
1. `ind_3(hv3, hv2, hv1, hv0)·(st0' - st3)`
1. `ind_4(hv3, hv2, hv1, hv0)·(st0' - st4)`
1. `ind_5(hv3, hv2, hv1, hv0)·(st0' - st5)`
1. `ind_6(hv3, hv2, hv1, hv0)·(st0' - st6)`
1. `ind_7(hv3, hv2, hv1, hv0)·(st0' - st7)`
1. `ind_8(hv3, hv2, hv1, hv0)·(st0' - st8)`
1. `ind_9(hv3, hv2, hv1, hv0)·(st0' - st9)`
1. `ind_10(hv3, hv2, hv1, hv0)·(st0' - st10)`
1. `ind_11(hv3, hv2, hv1, hv0)·(st0' - st11)`
1. `ind_12(hv3, hv2, hv1, hv0)·(st0' - st12)`
1. `ind_13(hv3, hv2, hv1, hv0)·(st0' - st13)`
1. `ind_14(hv3, hv2, hv1, hv0)·(st0' - st14)`
1. `ind_15(hv3, hv2, hv1, hv0)·(st0' - st15)`
1. `(1 - ind_1(hv3, hv2, hv1, hv0))·(st1' - st1)`
1. `(1 - ind_2(hv3, hv2, hv1, hv0))·(st2' - st2)`
1. `(1 - ind_3(hv3, hv2, hv1, hv0))·(st3' - st3)`
1. `(1 - ind_4(hv3, hv2, hv1, hv0))·(st4' - st4)`
1. `(1 - ind_5(hv3, hv2, hv1, hv0))·(st5' - st5)`
1. `(1 - ind_6(hv3, hv2, hv1, hv0))·(st6' - st6)`
1. `(1 - ind_7(hv3, hv2, hv1, hv0))·(st7' - st7)`
1. `(1 - ind_8(hv3, hv2, hv1, hv0))·(st8' - st8)`
1. `(1 - ind_9(hv3, hv2, hv1, hv0))·(st9' - st9)`
1. `(1 - ind_10(hv3, hv2, hv1, hv0))·(st10' - st10)`
1. `(1 - ind_11(hv3, hv2, hv1, hv0))·(st11' - st11)`
1. `(1 - ind_12(hv3, hv2, hv1, hv0))·(st12' - st12)`
1. `(1 - ind_13(hv3, hv2, hv1, hv0))·(st13' - st13)`
1. `(1 - ind_14(hv3, hv2, hv1, hv0))·(st14' - st14)`
1. `(1 - ind_15(hv3, hv2, hv1, hv0))·(st15' - st15)`
1. `osv' - osv`
1. `osp' - osp`
1. `(1 - ind_1(hv3, hv2, hv1, hv0))·(ramv' - ramv)`

### Helper variable definitions for `swap` + `i`

For `swap` + `i`, helper variables contain the binary decomposition of `i`:

1. `hv0 = i % 2`
1. `hv1 = (i >> 1) % 2`
1. `hv2 = (i >> 2) % 2`
1. `hv3 = (i >> 3) % 2`

## Instruction `nop`

This instruction has no additional transition constraints.

## Instruction `skiz`

### Description

1. The jump stack pointer `jsp` does not change.
1. The last jump's origin `jso` does not change.
1. The last jump's destination `jsd` does not change.
1. The next instruction `nia` is decomposed into helper variables `hv`.
1. The relevant helper variable `hv1` is either 0 or 1.
    Here, `hv1 == 1` means that `nia` takes an argument.
1. If `st0` is non-zero, register `ip` is incremented by 1.
If `st0` is 0 and `nia` takes no argument, register `ip` is incremented by 2.
If `st0` is 0 and `nia` takes an argument, register `ip` is incremented by 3.

Written as Disjunctive Normal Form, the last constraint can be expressed as:
6. (Register `st0` is 0 or `ip` is incremented by 1), and
(`st0` has a multiplicative inverse or `hv0` is 1 or `ip` is incremented by 2), and
(`st0` has a multiplicative inverse or `hv0` is 0 or `ip` is incremented by 3).

Since the three cases are mutually exclusive, the three respective polynomials can be summed up into one.

### Polynomials

1. `jsp' - jsp`
1. `jso' - jso`
1. `jsd' - jsd`
1. `nia - (hv0 + 4·hv1 + 8·hv2)`
1. `hv1·(hv1 - 1)`
1. `(ip' - (ip + 1)·st0) + ((ip' - (ip + 2))·(st0·hv2 - 1)·(hv0 - 1)) + ((ip' - (ip + 3))·(st0·hv2 - 1)·hv0)`

### Helper variable definitions for `skiz`

Note:
The concrete decomposition of `nia` into helper variables `hv` as well as the concretely relevant `hv` determining whether `nia` takes an argument (currently `hv0`) are subject to change.

1. `hv0 = nia % 2`
1. `hv1 = nia / 2`
1. `hv2 = inverse(st0)` (if `st0 ≠ 0`)

## Instruction `call` + `d`

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

### Description

1. The jump stack pointer `jsp` is decremented by 1.
1. The instruction pointer `ip` is set to the last call's origin `jso`.

### Polynomials

1. `jsp' - (jsp - 1)`
1. `ip' - jso`

## Instruction `recurse`

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

## Instruction `assert`

### Description

1. The current top of the stack `st0` is 1.

### Polynomials

1. `st0 - 1`

## Instruction `halt`

### Description

1. The instruction executed in the following step is instruction `halt`.

### Polynomials

1. `ci' - ci`

## Instruction `read_mem`

### Description

1. The top of the stack is overwritten with the RAM value.

### Polynomials

1. `st0' - ramv`

## Instruction `write_mem`

### Description

1. The RAM value is overwritten with the top of the stack.

### Polynomials

1. `ramv' - st0`

## Instruction `hash`

This instruction has no additional transition constraints.
Two Evaluation Arguments with the [Hash Table](hash-table.md) guarantee correct transition.

#### Instruction `divine_sibling`

Recall that in a Merkle tree, the indices of left (respectively right) leafs have 0 (respectively 1) as their least significant bit.
The first two polynomials achieve that helper variable `hv0` holds the result of `st10 mod 2`.
The third polynomial sets the new value of `st10` to `st10 div 2`.

##### Description

1. Helper variable `hv0` is either 0 or 1.
1. The 11th stack register is shifted by 1 bit to the right.

1. If `hv0` is 0, then `st0` does not change.
1. If `hv0` is 0, then `st1` does not change.
1. If `hv0` is 0, then `st2` does not change.
1. If `hv0` is 0, then `st3` does not change.
1. If `hv0` is 0, then `st4` does not change.
1. If `hv0` is 1, then `st0` is copied to `st5`.
1. If `hv0` is 1, then `st1` is copied to `st6`.
1. If `hv0` is 1, then `st2` is copied to `st7`.
1. If `hv0` is 1, then `st3` is copied to `st8`.
1. If `hv0` is 1, then `st4` is copied to `st9`.
1. The stack element in `st11` does not change.
1. The stack element in `st12` does not change.
1. The stack element in `st13` does not change.
1. The stack element in `st14` does not change.
1. The stack element in `st15` does not change.
1. The top of the OpStack underflow, i.e., `osv`, does not change.
1. The OpStack pointer does not change.
1. If `hv0` is 0, then the RAM value `ramv` does not change.

##### Polynomials

1. `hv0·(hv0 - 1)`
1. `st10'·2 + hv0 - st10`
1. `(1 - hv0)·(st0' - st0) + hv0·(st5' - st0)`
1. `(1 - hv0)·(st1' - st1) + hv0·(st6' - st1)`
1. `(1 - hv0)·(st2' - st2) + hv0·(st7' - st2)`
1. `(1 - hv0)·(st3' - st3) + hv0·(st8' - st3)`
1. `(1 - hv0)·(st4' - st4) + hv0·(st9' - st4)`
1. `st11' - st11`
1. `st12' - st12`
1. `st13' - st13`
1. `st14' - st14`
1. `st15' - st15`
1. `osv' - osv`
1. `osp' - osp`
1. `(1 - hv0)·(ramv' - ramv)`

##### Helper variable definitions for `divine_sibling`

Since `st10` contains the Merkle tree node index,

1. `hv0` holds the result of `st10 % 2` (the node index'es least significant bit, indicating whether it is a left/right node).

#### Instruction `assert_vector`

##### Description

1. Register `st0` is equal to `st5`.
1. Register `st1` is equal to `st6`.
1. Register `st2` is equal to `st7`.
1. Register `st3` is equal to `st8`.
1. Register `st4` is equal to `st9`.

##### Polynomials

1. `st5 - st0`
1. `st6 - st1`
1. `st7 - st2`
1. `st8 - st3`
1. `st9 - st4`

#### Instruction `add`

### Description

1. The sum of the top two stack elements is moved into the top of the stack.

### Polynomials

1. `st0' - (st0 + st1)`

## Instruction `mul`

### Description

1. The product of the top two stack elements is moved into the top of the stack.

### Polynomials

1. `st0' - st0·st1`

## Instruction `invert`

### Description

1. The top of the stack's inverse is moved into the top of the stack.

### Polynomials

1. `st0'·st0 - 1`

## Instruction `split`

### Description

1. The top of the stack is decomposed as 32-bit chunks into the stack's top-most two elements.
1. Helper variable `hv0` holds the inverse of $2^{32} - 1$ subtracted from the high 32 bits or the low 32 bits are 0.
1. Stack register `st1` is moved into `st2`
1. Stack register `st2` is moved into `st3`
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

1. `st0 - (2^32·st0' + st1')`
1. `st1'·(hv0·(st0' - (2^32 - 1)) - 1)`
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

### Helper variable definitions for `split`

Given the high 32 bits of `st0` as `hi = st0 >> 32` and the low 32 bits of `st0` as `lo = st0 & 0xffff_ffff`,

1. `hv0 = (hi - (2^32 - 1))` if `lo ≠ 0`.
1. `hv0 = 0` if `lo = 0`.

## Instruction `eq`

### Description

1. Helper variable `hv0` is the inverse of the difference of the stack's two top-most elements or 0.
1. Helper variable `hv0` is the inverse of the difference of the stack's two top-most elements or the difference is 0.
1. The new top of the stack is 1 if the difference between the stack's two top-most elements is not invertible, 0 otherwise.

### Polynomials

1. `hv0·(hv0·(st1 - st0) - 1)`
1. `(st1 - st0)·(hv0·(st1 - st0) - 1)`
1. `st0' - (1 - hv0·(st1 - st0))`

### Helper variable definitions for `eq`

1. `hv0 = inverse(rhs - lhs)` if `rhs - lhs ≠ 0`.
1. `hv0 = 0` if `rhs - lhs = 0`.

## Instruction `lt`

This instruction has no additional transition constraints.
A Permutation Argument between [Processor Table](processor-table.md)'s `st0`, `st1`, `st0'` and [Uint32 Operations Table](u32-table.md)'s `LHS`, `RHS`, `LT` guarantees correct transition.

## Instruction `and`

This instruction has no additional transition constraints.
A Permutation Argument with the [Uint32 Operations Table](u32-table.md) guarantees correct transition.
A Permutation Argument between [Processor Table](processor-table.md)'s `st0`, `st1`, `st0'` and [Uint32 Operations Table](u32-table.md)'s `LHS`, `RHS`, `AND` guarantees correct transition.

## Instruction `xor`

This instruction has no additional transition constraints.
A Permutation Argument between [Processor Table](processor-table.md)'s `st0`, `st1`, `st0'` and [Uint32 Operations Table](u32-table.md)'s `LHS`, `RHS`, `XOR` guarantees correct transition.

## Instruction `reverse`

This instruction has no additional transition constraints.
In contrast to all other 32-bit unsigned integer instructions, `reverse` only operates on the stack's top-most element, `st0`.
A Permutation Argument between [Processor Table](processor-table.md)'s `st0`, `st0'` and [Uint32 Operations Table](u32-table.md)'s `LHS`, `REV` guarantees correct transition.

## Instruction `div`

For correct division, it is required that the remainder `r` is smaller than the divisor `d`.
The result of comparing `r` to `d` is stored in helper variable `hv0`.

A Permutation Argument between [Processor Table](processor-table.md)'s `st0'`, `st0`, `hv0'` and [Uint32 Operations Table](u32-table.md)'s `LHS`, `RHS`, `LT` guarantees that `hv0 = (r < d)`.

### Description

1. Denominator `d` is not zero.
1. Result of division, i.e., quotient `q` and remainder `r`, are moved into `st1` and `st0` respectively, and match with denominator `d` and numerator `n`.
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
1. Helper variable `hv0` is 1, indicating that `r < d`.

### Polynomials

1. `st1·hv2 - 1` (i.e. `q·hv2 - 1`)
1. `st0 - st1·st1' - st0'` (i.e. `n - q·d - r`)
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
1. `hv0 - 1`

### Helper variable definitions for `div`

TODO: Fill this section when `hv0` has been removed for `div`.

```rust
hvs[0] = 1.into();
let st1 = self.op_stack.safe_peek(ST1);
if !st1.is_zero() {
    hvs[2] = st1.inverse();
}
```

## Instruction `xxadd`

### Description

1. The result of adding `st0` to `st3` is moved into `st0`.
1. The result of adding `st1` to `st4` is moved into `st1`.
1. The result of adding `st2` to `st5` is moved into `st2`.
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

1. `st0' - (st0 + st3)`
1. `st1' - (st1 + st4)`
1. `st2' - (st2 + st5)`
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

## Instruction `xxmul`

### Description

1. The coefficient of x^0 of multiplying the two X-Field elements on the stack is moved into `st0`.
1. The coefficient of x^1 of multiplying the two X-Field elements on the stack is moved into `st1`.
1. The coefficient of x^2 of multiplying the two X-Field elements on the stack is moved into `st2`.
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

1. `st0' - (st0·st3 - st2·st4 - st1·st5)`
1. `st1' - (st1·st3 + st0·st4 - st2·st5 + st2·st4 + st1·st5)`
1. `st2' - (st2·st3 + st1·st4 + st0·st5 + st2·st5)`
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

## Instruction `xinvert`

### Description

1. The coefficient of x^0 of multiplying X-Field element on top of the current stack and on top of the next stack is 1.
1. The coefficient of x^1 of multiplying X-Field element on top of the current stack and on top of the next stack is 0.
1. The coefficient of x^2 of multiplying X-Field element on top of the current stack and on top of the next stack is 0.
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

1. `st0·st0' - st2·st1' - st1·st2' - 1`
1. `st1·st0' + st0·st1' - st2·st2' + st2·st1' + st1·st2'`
1. `st2·st0' + st1·st1' + st0·st2' + st2·st2'`
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

## Instruction `xbmul`

### Description

1. The result of multiplying the top of the stack with the X-Field element's coefficient for x^0 is moved into `st0`.
1. The result of multiplying the top of the stack with the X-Field element's coefficient for x^1 is moved into `st1`.
1. The result of multiplying the top of the stack with the X-Field element's coefficient for x^2 is moved into `st2`.
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

1. `st0' - st0·st1`
1. `st1' - st0·st2`
1. `st2' - st0·st3`
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

## Instruction `read_io`

This instruction has no additional transition constraints.
An Evaluation Argument with the list of input symbols guarantees correct transition.

## Instruction `write_io`

This instruction has no additional transition constraints.
An Evaluation Argument with the list of output symbols guarantees correct transition.
