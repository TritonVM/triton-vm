# Pseudo Instructions

Keeping the instruction set small results in a faster overall proving time due to simpler arithmetization.
Some convenient-to-have instructions are not implemented natively, but are instead simulated using existing instructions.
The following list is a comprehensive overview, including their expansion.

| Instruction | old OpStack | new OpStack | Description                                                                                           |
|:------------|:------------|:------------|:------------------------------------------------------------------------------------------------------|
| `neg`       | `_ a`       | `_ -a`      | Replaces the top of the stack with the field element corresponding to its additively inverse element. |
| `sub`       | `_ b a`     | `_ a-b`     | Subtracts the stack's one-from top element from the stack's topmost element.                          |
| `is_u32`    | `_ a`       | `_ {0,1}`   | Tests if the top of the stack is a u32, replacing it with the result.                                 |

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

Program length: 10.

Execution cycle count: 7.

```
dup 0   // _ a a
split   // _ a lo hi
push 0  // _ a lo hi 0
eq      // _ a lo {0,1}
swap 2  // _ {0,1} lo a
eq      // _ {0,1} {0,1}
mul     // _ {0,1}
```
