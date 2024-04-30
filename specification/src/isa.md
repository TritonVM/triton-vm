# Instruction Set Architecture

Triton VM is a stack machine with RAM.
It is a [Harvard architecture](https://en.wikipedia.org/wiki/Harvard_architecture) with read-only memory for the program.
The arithmetization of the VM is defined over the *B-field* $\mathbb{F}_p$ where $p$ is the Oxfoi prime, _i.e._, $2^{64}-2^{32}+1$.[^oxfoi]
This means the registers and memory elements take values from $\mathbb{F}_p$, and the transition function gives rise to low-degree transition verification polynomials from the ring of multivariate polynomials over $\mathbb{F}_p$.

At certain points in the course of arithmetization we need an extension field over $\mathbb{F}_p$ to ensure sufficient soundness. To this end we use $\mathbb{F}_{p^3} = \frac{\mathbb{F}_p[X]}{\langle X^3 -X +1 \rangle}$, _i.e._, the quotient ring of remainders of polynomials after division by the *Shah* polynomial, $X^3 -X +1$. We refer to this field as the *X-field* for short, and its elements as *X-field elements*.

Instructions have variable width:
they either consist of one word, i.e., one B-field element, or of two words, i.e., two B-field elements.
An example for a single-word instruction is `add`, adding the two elements on top of the stack, leaving the result as the new top of the stack.
An example for a double-word instruction is `push` + `arg`, pushing `arg` to the stack.

Triton VM has two interfaces for data input, one for public and one for secret data, and one interface for data output, whose data is always public.
The public interfaces differ from the private one, especially regarding their arithmetization.

[^oxfoi]: The name “Oxfoi” comes from the prime's hexadecimal representation `0xffffffff00000001`.
