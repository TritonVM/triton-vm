# Triton VM

Triton is a virtual machine that comes with Algebraic Execution Tables (AET) and Arithmetic Intermediate Representations (AIR) for use in combination with a [STARK proof system](https://neptune.cash/learn/stark-anatomy/).
It defines a Turing complete [Instruction Set Architecture](./specification/isa.md), as well as the corresponding [arithmetization](./specification/arithmetization.md) of the VM.
The really cool thing about Triton VM is its efficient _recursive_ verification of the STARKs produced when running Triton VM.

## Recursive STARKs of Computational Integrity

Normally, when executing a machine – virtual or not – the flow of information can be regarded as follows.
The tuple of (`input`, `program`) is given to the machine, which takes the `program`, evaluates it on the `input`, and produces some `output`.

![](./specification/img/recursive-1.svg)

If the – now almost definitely virtual – machine also has an associated STARK engine, one additional output is a `proof` of computational integrity.

![](./specification/img/recursive-2.svg)

Only if `input`, `program`, and `output` correspond to one another, i.e., if `output` is indeed the result of evaluating the `program` on the `input` according to the rules defined by the virtual machine, then producing such a `proof` is easy.
Otherwise, producing a `proof` is next to impossible.

The routine that checks whether or not a `proof` is, in fact, a valid one, is called the Verifier.
It takes as input a 4-tuple (`input`, `program`, `output`, `proof`) and evaluates to `true` if and only if that 4-tuple is consistent with the rules of the virtual machine.

![](./specification/img/recursive-3.svg)

Since the Verifier is a program taking some input and producing some output, the original virtual machine can be used to perform the computation.

![](./specification/img/recursive-4.svg)

The associated STARK engine will then produce a proof of computational integrity of _verifying_ some other proof of computational integrity – recursion!
Of course, the Verifier can be a subroutine in a larger program.

Triton VM is specifically designed to allow fast recursive verification.

## Project Status

Triton VM is currently (2022-08-02) in active development.
The code implementing the specification contained in this repository is not yet publicly available, a matter that we foresee changing in the next few days.
The implementation is written in rust.
It will be published in this repository, and subject to the same [license](./LICENSE) as the specification.

Please note that the [Instruction Set Architecture](./specification/isa.md) is not to be considered final.
However, we don't currently foresee big changes.

For the time being, issues are only being tracked internally.
This will also change once the code is published.
In the meantime, please feel free to [contact the authors](mailto:ferdinand@neptune.cash).
