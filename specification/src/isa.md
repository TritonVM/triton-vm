# Instruction Set Architecture

Triton VM is a stack machine with RAM.
It is a [Harvard architecture](https://en.wikipedia.org/wiki/Harvard_architecture) with read-only memory for the program.
The arithmetization of the VM is defined over the *B-field* $\mathbb{F}_p$ where $p=2^{64}-2^{32}+1$.
This means the registers and memory elements take values from $\mathbb{F}_p$, and the transition function gives rise to low-degree transition verification polynomials from the ring of multivariate polynomials over $\mathbb{F}_p$.

Instructions have variable width:
they either consist of one word, i.e., one B-field element, or of two words, i.e., two B-field elements.
An example for a single-word instruction is `pop`, removing the top of the stack.
An example for a double-word instruction is `push` + `arg`, pushing `arg` to the stack.

Triton VM has two interfaces for data input, one for public and one for secret data, and one interface for data output, whose data is always public.
The public interfaces differ from the private one, especially regarding their arithmetization.
