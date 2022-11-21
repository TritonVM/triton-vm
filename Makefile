# Treat `cargo clippy` warnings as errors.
CLIPPY_ARGS = --all-targets -- -D warnings

# Fail if `cargo fmt` changes anything.
FMT_ARGS = --all -- --check

# Primary targets

# By first building tests, and consequently building constraints, before
# running `fmt` and `clippy`, the auto-generated constraints are exposed
# to `fmt` and `clippy`.
default: build-constraints build-tests build-bench fmt-only clippy-only clean-constraints

# Alternative to `cargo build --all-targets`
build: build-constraints
	cargo build --all-targets
	make clean-constraints

# Alternative to `cargo test --all-targets`
test: build-constraints
	cargo test --all-targets
	make clean-constraints

# Alternative to `cargo bench --all-targets`
bench: build-constraints
	cargo bench --all-targets
	make clean-constraints

# Alternative to `cargo clippy ...`
clippy: build-constraints
	cargo clippy $(CLIPPY_ARGS)
	make clean-constraints

# Alternative to `cargo fmt ...`
fmt-check: build-constraints
	cargo fmt $(FMT_ARGS)
	make clean-constraints

# Alternative to `cargo clean`
clean:
	cargo clean
	make clean-constraints

# Auxiliary targets
#
# Assume constraints are compiled.

build-tests:
	cargo test --all-targets --no-run

build-bench:
	cargo bench --all-targets --no-run

build-constraints:
	cargo run --bin constraint-evaluation-generator

clean-constraints:
	git restore --staged triton-vm/src/table/constraints/
	git restore triton-vm/src/table/constraints/

fmt-only:
	cargo fmt $(FMT_ARGS)

clippy-only:
	cargo clippy $(CLIPPY_ARGS)
