# Pseudo Instructions

Keeping the instruction set small results in a faster overall proving time due to simpler arithmetization.
Some convenient-to-have instructions are not implemented natively, but are instead simulated using existing instructions.
The following list is a comprehensive overview, including their expansion.


| Instruction    | old OpStack | new OpStack   | Description                                                                                                                                                           |
|:---------------|:------------|:--------------|:----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `neg`          | `_ a`       | `_ -a`        | Replaces the top of the stack with the field element corresponding to its additively inverse element.                                                                 |
| `sub`          | `_ b a`     | `_ a-b`       | Subtracts the stack's one-from top element from the stack's topmost element.                                                                                          |
| `is_u32`       | `_ a`       | `_ a`         | Crashes the VM if `a` cannot be represented as an unsigned 32-bit integer.                                                                                            |
| `split_assert` | `_ a`       | `_ lo hi`     | Like instruction `split`, but additionally asserts that the results `lo` and `hi` are, indeed, 32-bit integers. Should be used over `split`.                          |
| `lte`          | `_ b a`     | `_ (a <= b)`  | Tests if the top element on the stack is less than or equal to the one-from top element. Crashes the VM if `a` or `b` is not a 32-bit integer.                        |
| `lt`           | `_ b a`     | `_ (a < b)`   | Tests if the top element on the stack is less than the one-from top element. Crashes the VM if `a` or `b` is not a 32-bit integer.                                    |
| `and`          | `_ b a`     | `_ (a and b)` | Computes the bitwise-and of the top two stack elements. Crashes the VM if `a` or `b` is not a 32-bit integer.                                                         |
| `xor`          | `_ b a`     | `_ (a xor b)` | Computes the bitwise-xor of the top two stack elements. Crashes the VM if `a` or `b` is not a 32-bit integer.                                                         |
| `reverse`      | `_ a`       | `_ b`         | Reverses the bit expansion of the top stack element. Crashes the VM if `a` is not a 32-bit integer.                                                                   |
| `div`          | `_ d n`     | `_ q r`       | Computes division with remainder of the top two stack elements, assuming both arguments are unsigned 32-bit integers. The result satisfies `n == d·q + r`and `r < d`. |


## Pseudo instruction `neg`

Program length: 3.

Execution cycle count: 2.

```
push -1
mul
```

## Pseudo instruction `sub`

Program length: 6.

Execution cycle count: 4.

```
swap 1
push -1
mul
add
```

## Pseudo instruction `is_u32`

Program length: 70.

Execution cycle count: 68.

```
dup 0
for _ in 0..32 {
    lsb
    pop
}
push 0
eq
assert
```

## Pseudo instruction `split_assert`

Program length: 145.

Execution cycle count: 139.

```
split
is_u32
swap 1
is_u32
swap 1
```

## Pseudo instruction `lte`

Program length: 160.

Execution cycle count: 146.

```
push -1
mul
add
split_assert
push 0
eq
swap 1
pop
```

## Pseudo instruction `lt`

Program length: 163.

Execution cycle count: 148.

```
push 1
add
lte
```

## Pseudo instruction `and`

Program length: 426.

Execution cycle count: 295.

```
for _ in 0..32 {
    lsb
    swap 2
    lsb
    swap 2
}

for _ in 0..2 {
    push 0
    eq
    assert
}

push 0 // accumulator

for i in (0..32).rev() {
    swap 2
    mul
    push 1<<i
    mul
    add
}
```

## Pseudo instruction `xor`

Program length: 435.

Execution cycle count: 301.

```
dup 1
dup 1
and
push -2
mul
add
add
```

Credit to [Daniel Lubarov](https://github.com/dlubarov).

## Pseudo instruction `reverse`

Program length: 292.

Execution cycle count: 195.

```
for _ in 0..32 {
    lsb
    swap 1
}
push 0
eq
assert
for i in 0..32 {
    swap 1
    push 1<<i
    mul
    add
}
```

## Pseudo instruction `div`

Program length: 258.

Execution cycle count: 232.

```
divine
is_u32
dup 2
dup 1
mul
dup 2
swap 1
push -1
mul
add
dup 3
dup 1
lt
assert
swap 2
pop
swap 2
pop
```
