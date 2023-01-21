# Pseudo Instructions

Keeping the instruction set small results in a faster overall proving time due to simpler arithmetization.
Some convenient-to-have instructions are not implemented natively, but are instead simulated using existing instructions.
The following list is a comprehensive overview, including their expansion.

| Instruction | old OpStack | new OpStack  | Description                                                                                                                                |
|:------------|:------------|:-------------|:-------------------------------------------------------------------------------------------------------------------------------------------|
| `neg`       | `_ a`       | `_ -a`       | Replaces the top of the stack with the field element corresponding to its additively inverse element.                                      |
| `sub`       | `_ b a`     | `_ a-b`      | Subtracts the stack's one-from top element from the stack's topmost element.                                                               |
| `is_u32`    | `_ a`       | `_ {0,1}`    | Tests if the top of the stack is a u32, replacing it with the result.                                                                      |
| `lsb`       | `_ a`       | `_ a>>1 a%2` | Bit-shifts `a` to the right by 1 bit and pushes the least significant bit (_lsb_) of `a` to the stack. Crashes the VM if `a` is not a u32. |

## Pseudo instruction `neg`

Program length: 3.

Execution cycle count: 2.

```
         // _ a
push -1  // _ a -1
mul      // _ -a
```

## Pseudo instruction `sub`

Program length: 6.

Execution cycle count: 4.

```
         // _ b a
swap1    // _ a b
push -1  // _ a b -1
mul      // _ a -b
add      // _ a-b
```
## Pseudo instruction `is_u32`

Program length: 5.

Execution cycle count: 4.

```
       // _ a
split  // _ hi lo
pop    // _ hi
push 0 // _ hi 0
eq     // _ (hi == 0)
```

## Pseudo instruction `lsb`

Program length: 5.

Execution cycle count: 3.

```
        // _ a
push 2  // _ a 2
swap1   // _ 2 a
div     // _ a/2 a%2
```

