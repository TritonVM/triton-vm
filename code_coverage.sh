#!/bin/bash

# credit: https://rrmprogramming.com/article/code-coverage-in-rust/

# Define color variables
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

function cleanup() {
  echo -e "${YELLOW}Cleaning up previous coverages...${NC}"
  cargo clean && mkdir -p coverage/ && rm -r coverage/*
  echo -e "${GREEN}Success: Crate cleaned successfully${NC}" 
}

function run_tests() {
  echo -e "${YELLOW}Compiling and running tests with code coverage...${NC}"
  CARGO_INCREMENTAL=0 RUSTFLAGS='-Cinstrument-coverage' LLVM_PROFILE_FILE='coverage/cargo-test-%p-%m.profraw' cargo test --workspace
  if [[ $? -ne 0 ]]; then
    echo -e "${RED}Error: Tests failed to execute${NC}"
    exit 1
  fi
  echo -e "${GREEN}Success: All tests were executed correctly!${NC}"
}

function generate_coverage() {
  echo -e "${YELLOW}Generating code coverage...${NC}"
  grcov . --binary-path ./target/debug/deps/ -s . -t html --branch --ignore-not-existing --ignore '../*' --ignore "/*" -o target/coverage/ && \
  grcov . --binary-path ./target/debug/deps/ -s . -t lcov --branch --ignore-not-existing --ignore '../*' --ignore "/*" -o target/coverage/lcov.info
  if [[ $? -ne 0 ]]; then
    echo -e "${RED}Error: Failed to generate code coverage${NC}"
    exit 1
  fi
  echo -e "${GREEN}Success: Code coverage generated correctly!${NC}"
}

echo -e "${GREEN}========== Running test coverage ==========${NC}"
echo

cleanup

run_tests

generate_coverage
