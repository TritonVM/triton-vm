# Triton VM

Triton is a virtual machine that comes with Algebraic Execution Tables (AET) and Arithmetic Intermediate Representations (AIR) for use in combination with a STARK proof system.

Triton VM defines an [Instruction Set Architecture](./specification/isa.md) designed for efficient recursive verification of the STARKs produced when running Triton VM, as well as the corresponding [arithmetization](./specification/arithmetization.md) of the VM.

Triton VM is currently (2022-06-27) in active, mid-stage development.
The code implementing the specification contained in this repository is not yet publicly available, a matter that we foresee changing in the next few weeks.
The implementation is written in rust.
It will be published in this repository, and subject to the same [license](./LICENSE) as the specification.

Please note that the [Instruction Set Architecture](./specification/isa.md) is not to be considered final.
However, we don't currently foresee big changes.

For the time being, issues are only being tracked internally.
This will also change once the code is published.
In the meantime, please feel free to [contact the authors](mailto:ferdinand@neptune.cash).
