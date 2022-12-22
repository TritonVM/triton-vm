# triton-opcodes

[Triton VM](https://triton-vm.org) ([GitHub](https://github.com/TritonVM/triton-vm), [Crates.io](https://crates.io/crates/triton-vm)) is a virtual machine that comes with Algebraic Execution Tables (AET) and Arithmetic Intermediate Representations (AIR) for use in combination with a STARK proof system.

The package `triton-opcodes` delivers the `Instruction` type and a parser from assembly like:

```
// Swap the top two stack elements so they're sorted.
//
// The larger element is at the top.
//
// Before: _ a b
// After: _ min(a, b) max(a, b)
minmax:
    dup1        // _ a b a
    dup1        // _ a b a b
    lt          // _ a b (b < a)
    skiz swap1  // _ min(a, b) max(a, b)
    return
```

Features:

- [pseudo-instructions](https://triton-vm.org/spec/pseudo-instructions.html) like `lt` by simple substitution.
- labelled jumps where labels look like `<name>:` and jumping to them looks like `call <name>`.
- inline comments in the form `// ...`.
