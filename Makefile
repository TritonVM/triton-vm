# Treat `cargo clippy` warnings as errors.
CLIPPY_ARGS = --all-targets -- -D warnings

# Fail if `cargo fmt` changes anything.
FMT_ARGS = --all -- --check

# Treat all warnings as errors
export RUSTFLAGS = -Dwarnings

# Set another target dir than default to avoid builds from `make`
# to invalidate cache from barebones use of `cargo` commands.
# The cache is cleared when a new `RUSTFLAGS` value is encountered,
# so to prevent the two builds from interfering, we use two dirs.
export CARGO_TARGET_DIR=./makefile-target

# By first building tests, and consequently building constraints, before
# running `fmt` and `clippy`, the auto-generated constraints are exposed
# to `fmt` and `clippy`.
default: build-constraints build-tests build-bench fmt-only clippy-only clean-constraints

# Run `make all` when the constraints are already in place.
all: test build-bench fmt-only clippy-only

# Alternative to `cargo build --all-targets`
build: build-constraints
	cargo build --all-targets

# Alternative to `cargo test --all-targets`
test:
	cargo test --all-targets

# Alternative to `cargo bench --all-targets`
bench: build-constraints
	cargo bench --all-targets

# Alternative to `cargo clippy ...`
clippy: build-constraints
	cargo clippy $(CLIPPY_ARGS)

# Alternative to `cargo fmt ...`
fmt-check: build-constraints
	cargo fmt $(FMT_ARGS)

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
	git restore --staged triton-vm/src/table/constraints.rs
	git restore --staged triton-vm/src/table/degree_lowering_table.rs
	git restore triton-vm/src/table/constraints.rs
	git restore triton-vm/src/table/degree_lowering_table.rs

fmt-only:
	cargo fmt $(FMT_ARGS)

clippy-only:
	cargo clippy $(CLIPPY_ARGS)
