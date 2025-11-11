# Specification

Triton VM’s specification comes in the form of an [mdBook](https://rust-lang.github.io/mdBook/).
For easy access, you can find it [online](https://triton-vm.org/spec/).
You can also self-host it by following the steps below.

If you don’t have it already, install [rust](https://rust-lang.org/tools/install/).
Then, install mdBook as well as the
[pre-processors](https://rust-lang.github.io/mdBook/format/configuration/preprocessors.html)
we use:

```sh
cargo install mdbook mdbook-katex mdbook-linkcheck
```

Finally, you can serve and open the specification:

```sh
mdbook serve --open
```

In case you have changed some of the defaults, make sure that `~/.cargo/bin` is in your PATH.
