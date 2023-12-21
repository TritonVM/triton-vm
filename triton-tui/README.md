# Triton TUI

A TUI to run and debug programs written for Triton VM.

<img alt="Example run of Triton TUI" src="./examples/triton-tui.gif" width="800" />

## Getting Started

Triton TUI tries to be helpful 🙂. List possible (and required) arguments with `triton-tui --help`. In the TUI, press `h` to access the help screen.

The [example program](./examples/program.tasm), serving as the tutorial, can be run with

```sh
triton-tui --program examples/program.tasm
```

## Installation

Through [crates.io](https://crates.io/crates/triton-tui):

```sh
cargo install triton-tui
```

Locally:

```sh
cargo install --path .
```
