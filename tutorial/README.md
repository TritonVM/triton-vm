# Triton Assembly Tutorial

This [mdBook](https://rust-lang.github.io/mdBook/) is a hands-on tutorial for writing Triton assembly.
For easy access, you can find it [online](https://triton-vm.org/tutorial/), or self-host it following the steps below.

Lessons progress towards an ultimately useful program, with relevant concepts introduced step by step.

> [!NOTE]
> If you want to understand the internals of Triton VM, take a look at the [specification](../specification/README.md).

## Build locally

If you don’t have it already, install [rust](https://rust-lang.org/tools/install/).
Then, install mdBook:

```sh
cargo install mdbook
```

Finally, you can serve and open the tutorial:

```sh
mdbook serve --open
```

In case you have changed some of the defaults, make sure that `~/.cargo/bin` is in your PATH.
