# Introduction

Welcome to the official Triton assembly tutorial.
Here, you will learn the basics of stack machines (which [Triton VM](https://triton-vm.org/) is), as well as
non-deterministic programming.
Some of these concepts are unusual even to veteran software engineers, so we'll take things step by step.
The tutorial assumes a certain familiarity with and affinity to programming, but if some of the above words don't mean
anything to you right now, that's totally fine.

The final program this tutorial is building towards is both realistic and useful;
the early programs are intentionally simple and might not serve any purpose other than as stepping stones.
That's on purpose: each step isolates one concept so that later programs are easy to read and modify.

This tutorial is an introduction to writing programs for Triton VM, but not on how to develop Triton VM itself.
It also doesn't cover the cool cryptography behind Triton VM's zero-knowledge proof system;
if you're interested in that, take a look at the [specification](https://triton-vm.org/spec/).
Having the specification at hand might be useful in any case, not least of all because it contains a
[comprehensive overview of all instructions](https://triton-vm.org/spec/instructions.html).
If that seems somewhat overwhelming right now, don't worry, we'll explore things together.

## Follow along

One of the best ways to learn is by experimenting.
For this reason, you can execute and modify all the programs presented in this tutorial directly in your browser:

<!-- @formatter:off -->
<triton-playground>
push 42
write_io 1
halt
</triton-playground>
<!-- @formatter:on -->

If you want to observe how a program executes step by step, for example because you need to debug something, take a
look at the [Triton TUI](https://crates.io/crates/triton-tui).
I mention it here because it's been very useful to me in the past, but if you're not familiar with Triton VM yet,
it's probably better to keep reading this tutorial first, then later come back to Triton TUI.
I'll mention it again at the end of the tutorial, so you don't have to keep that browser tab open.
