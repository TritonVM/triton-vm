# Changelog

All notable changes are documented in this file.
Lines marked “(!)” indicate a breaking change.

## [1.0.0](https://github.com/TritonVM/triton-vm/compare/v0.50.0..v1.0.0) - 2025-10-23

### ⚙️ Miscellaneous

- Make environment variable name public ([#367](https://github.com/TritonVM/triton-vm/issues/367)) ([8d518985](https://github.com/TritonVM/triton-vm/commit/8d518985))
- Upgrade dependencies ([4da0e9f6](https://github.com/TritonVM/triton-vm/commit/4da0e9f6))

## [0.50.0](https://github.com/TritonVM/triton-vm/compare/v0.49.0..v0.50.0) - 2025-06-27

### ⚡️ Performance

- Avoid allocation during RAM-frugal proving ([47509643](https://github.com/TritonVM/triton-vm/commit/47509643))
- Don't allocate when hashing FRI table rows ([27f67129](https://github.com/TritonVM/triton-vm/commit/27f67129))

## [0.49.0](https://github.com/TritonVM/triton-vm/compare/v0.48.0..v0.49.0) - 2025-06-18

### ✨ Features

- Improve error messages of parser ([7ef225f1](https://github.com/TritonVM/triton-vm/commit/7ef225f1))
- (!) Report all table heights for `VM::profile` ([f05db5ee](https://github.com/TritonVM/triton-vm/commit/f05db5ee))
- *(parser)* Allow “generic types” in type hints ([f9e0666e](https://github.com/TritonVM/triton-vm/commit/f9e0666e))
- *(profiler)* Report memory consumption ([0eaac39e](https://github.com/TritonVM/triton-vm/commit/0eaac39e))

### 🐛 Bug Fixes

- Use rayon's thread count for parallelization ([b66d0f2f](https://github.com/TritonVM/triton-vm/commit/b66d0f2f))
- Correctly accept high-security proofs ([fd663411](https://github.com/TritonVM/triton-vm/commit/fd663411))

### 📚 Documentation

- Update some documentation to current lingo ([4903b6a1](https://github.com/TritonVM/triton-vm/commit/4903b6a1))
- Mention Triton CLI in README ([a736b11f](https://github.com/TritonVM/triton-vm/commit/a736b11f))
- Fix broken footnotes ([d39626d4](https://github.com/TritonVM/triton-vm/commit/d39626d4))
- Link to Triton CLI's “Profiling” section ([eec84919](https://github.com/TritonVM/triton-vm/commit/eec84919))
- Fix intra-doc links ([ac91d15d](https://github.com/TritonVM/triton-vm/commit/ac91d15d))

### ⚙️ Miscellaneous

- Drop unused dependencies ([426d70f1](https://github.com/TritonVM/triton-vm/commit/426d70f1))
- (!) Upgrade to rust edition 2024 ([3cb1528d](https://github.com/TritonVM/triton-vm/commit/3cb1528d))
- Check links ([adbce6a6](https://github.com/TritonVM/triton-vm/commit/adbce6a6))
- Build & test mdBook ([df07acae](https://github.com/TritonVM/triton-vm/commit/df07acae))
- Don't include tests in coverage percentage ([69ccd615](https://github.com/TritonVM/triton-vm/commit/69ccd615))
- Update license to MIT OR Apache-2.0 ([c012d3ad](https://github.com/TritonVM/triton-vm/commit/c012d3ad))

### ♻️ Refactor

- (!) Clarify derivation of prover's domains ([b6a1d5f3](https://github.com/TritonVM/triton-vm/commit/b6a1d5f3))

### ✅ Testing

- Increase testability of constraint failures ([bda6de5c](https://github.com/TritonVM/triton-vm/commit/bda6de5c))

## [0.48.0](https://github.com/TritonVM/triton-vm/compare/v0.45.0..v0.48.0) - 2025-02-11

### 📚 Documentation

- Fix internal intra-doc links ([d3fbc654](https://github.com/TritonVM/triton-vm/commit/d3fbc654))

### ⚙️ Miscellaneous

- Add degree lowering benchmark to codspeed ([048ea2ea](https://github.com/TritonVM/triton-vm/commit/048ea2ea))

### ♻️ Refactor

- Consistently use criterion in benchmarks ([872d42c3](https://github.com/TritonVM/triton-vm/commit/872d42c3))

## [0.45.0](https://github.com/TritonVM/triton-vm/compare/v0.44.1..v0.45.0) - 2025-01-13

### ⚡️ Performance

- *(verifier)* Remove parallelism ([baf5860f](https://github.com/TritonVM/triton-vm/commit/baf5860f))

## [0.44.1](https://github.com/TritonVM/triton-vm/compare/v0.44.0..v0.44.1) - 2025-01-07

### 🐛 Bug Fixes

- Improve error message on internal error ([9c6cc1c2](https://github.com/TritonVM/triton-vm/commit/9c6cc1c2))

### 📚 Documentation

- Add contribution guidelines ([5f17e2dc](https://github.com/TritonVM/triton-vm/commit/5f17e2dc))
- Clarify examples of instruction behavior ([875cd09e](https://github.com/TritonVM/triton-vm/commit/875cd09e))

### ⚙️ Miscellaneous

- Enable continuous benchmarks through codspeed ([b9f78adb](https://github.com/TritonVM/triton-vm/commit/b9f78adb))
- Fix typos ([595889f0](https://github.com/TritonVM/triton-vm/commit/595889f0))

## [0.44.0](https://github.com/TritonVM/triton-vm/compare/v0.43.0..v0.44.0) - 2024-12-09

### ✨ Features

- Set minimum supported rust version (MSRV) ([a6cc98a3](https://github.com/TritonVM/triton-vm/commit/a6cc98a3))

### ♻️ Refactor

- (!) Remove bracket syntax sugar for `call` ([847b513e](https://github.com/TritonVM/triton-vm/commit/847b513e))
- (!) Remove deprecated functions ([5a16da96](https://github.com/TritonVM/triton-vm/commit/5a16da96))

## [0.43.0](https://github.com/TritonVM/triton-vm/compare/v0.42.1..v0.43.0) - 2024-11-14

### ✨ Features

- *(Claim)* Accept more types for public input ([6451d764](https://github.com/TritonVM/triton-vm/commit/6451d764))
- (!) Add field `version` to `Claim` ([48f896a7](https://github.com/TritonVM/triton-vm/commit/48f896a7))
- Display jump stack when printing `VMState` ([0665f2f9](https://github.com/TritonVM/triton-vm/commit/0665f2f9))
- (!) Optionally supply prover randomness seed ([923e9e11](https://github.com/TritonVM/triton-vm/commit/923e9e11))
- (!) Introduce assertion error ([a987ebe7](https://github.com/TritonVM/triton-vm/commit/a987ebe7))

### ⚡️ Performance

- *(RAM)* Drop cached polynomials ([9bf637dd](https://github.com/TritonVM/triton-vm/commit/9bf637dd))
- *(LDE)* Only interpolate if result is used ([4133b6e4](https://github.com/TritonVM/triton-vm/commit/4133b6e4))
- Reduce prover's space requirements ([833244d0](https://github.com/TritonVM/triton-vm/commit/833244d0))

### ♻️ Refactor

- (!) Store debug information in `VMState` ([d9664e52](https://github.com/TritonVM/triton-vm/commit/d9664e52))
- Introduce structs `Prover` & `Verifier` ([2f5b7a97](https://github.com/TritonVM/triton-vm/commit/2f5b7a97))

### ✅ Testing

- Computations are independent of caching ([ecd31083](https://github.com/TritonVM/triton-vm/commit/ecd31083))
- Verify randomizers' large Hamming distance ([ba2e92be](https://github.com/TritonVM/triton-vm/commit/ba2e92be))

### ⏱ Bench

- Benchmark barycentric evaluation ([ce592a9d](https://github.com/TritonVM/triton-vm/commit/ce592a9d))

## [0.42.1](https://github.com/TritonVM/triton-vm/compare/v0.42.0..v0.42.1) - 2024-09-26

### ⚡️ Performance

- *(degree_lowering)* Cache node-degrees ([6496a740](https://github.com/TritonVM/triton-vm/commit/6496a740))
- *(TasmBackend)* Exploit mixed-type instructions ([8898bdc9](https://github.com/TritonVM/triton-vm/commit/8898bdc9))
- *(TasmBackend)* Use `addi` instruction if possible ([6d55b80d](https://github.com/TritonVM/triton-vm/commit/6d55b80d))

### ✅ Testing

- Correctness & soundness of degree lowering ([667c9103](https://github.com/TritonVM/triton-vm/commit/667c9103))

### 🛠 Build

- Set `opt-level=3` on build script execution ([5e16b087](https://github.com/TritonVM/triton-vm/commit/5e16b087))

### ⏱ Bench

- Add benchmark for degree lowering times ([86cf9623](https://github.com/TritonVM/triton-vm/commit/86cf9623))

## [0.42.0](https://github.com/TritonVM/triton-vm/compare/v0.41.0..v0.42.0) - 2024-09-16

### ✨ Features

- Include `ConstraintType` in public API ([1592459b](https://github.com/TritonVM/triton-vm/commit/1592459b))
- Detect equivalent nodes in constraint circuit ([17c5b616](https://github.com/TritonVM/triton-vm/commit/17c5b616))
- Introduce instruction for dot product ([5abf529b](https://github.com/TritonVM/triton-vm/commit/5abf529b))
- Introduce instruction `merkle_step` ([01c04e52](https://github.com/TritonVM/triton-vm/commit/01c04e52))
- Introduce memory friendly proving path ([70b740e9](https://github.com/TritonVM/triton-vm/commit/70b740e9))
- Introduce instruction `sponge_absorb_mem` ([6dd9b54a](https://github.com/TritonVM/triton-vm/commit/6dd9b54a))
- Introduce instruction `recurse_or_return` ([98dbd9ff](https://github.com/TritonVM/triton-vm/commit/98dbd9ff))
- Add instruction `addi` ([3b5bc128](https://github.com/TritonVM/triton-vm/commit/3b5bc128))
- Add dynamic counterpart to tasm code generator ([72b6f5bc](https://github.com/TritonVM/triton-vm/commit/72b6f5bc))
- Introduce instruction `merkle_step_mem` ([3b1e3590](https://github.com/TritonVM/triton-vm/commit/3b1e3590))
- Implement `Deref` for `PublicInput` ([5a521542](https://github.com/TritonVM/triton-vm/commit/5a521542))
- Introduce instructions `pick` and `place` ([b7693922](https://github.com/TritonVM/triton-vm/commit/b7693922))

### 🐛 Bug Fixes

- Don't mutate `HashSet`'s content in-place ([6e5443e5](https://github.com/TritonVM/triton-vm/commit/6e5443e5))
- Use correct domain for deep codeword ([4e52b67e](https://github.com/TritonVM/triton-vm/commit/4e52b67e))
- *(docs)* Correctly sum number of constraints ([3bf33255](https://github.com/TritonVM/triton-vm/commit/3bf33255))
- *(ZK)* Ensure ZK from quotient segment openings ([be87aefe](https://github.com/TritonVM/triton-vm/commit/be87aefe))
- *(profiler)* Correctly compute clock frequency ([e1281b0e](https://github.com/TritonVM/triton-vm/commit/e1281b0e))
- Ensure node index for `merkle_step` is u32 ([c510b163](https://github.com/TritonVM/triton-vm/commit/c510b163))
- Fix arithmetic overflow in `MemoryRegion` ([f83f8aa0](https://github.com/TritonVM/triton-vm/commit/f83f8aa0))

### ⚡️ Performance

- Faster domain-evaluation for too-large polynomial ([3905d808](https://github.com/TritonVM/triton-vm/commit/3905d808))
- Use parallelism more when evaluating domain ([8c623e82](https://github.com/TritonVM/triton-vm/commit/8c623e82))
- Profile and fix slow zero-initialization ([f7b13e74](https://github.com/TritonVM/triton-vm/commit/f7b13e74))
- Parallelize Filling of Degree-Lowering Table ([9c02c646](https://github.com/TritonVM/triton-vm/commit/9c02c646))
- Parallelize evaluation-part of quotient-LDE ([82de2994](https://github.com/TritonVM/triton-vm/commit/82de2994))
- Use fastest polynomial multiplication ([89cc89ad](https://github.com/TritonVM/triton-vm/commit/89cc89ad))
- Parallelize deep codeword inner product ([72238cf5](https://github.com/TritonVM/triton-vm/commit/72238cf5))
- (!) Halve number of combination codeword checks ([deecc224](https://github.com/TritonVM/triton-vm/commit/deecc224))
- Sum mutually exclusive constraints ([3e96faca](https://github.com/TritonVM/triton-vm/commit/3e96faca))
- (!) Simplify constraints of instruction `swap` ([62187169](https://github.com/TritonVM/triton-vm/commit/62187169))
- Parallelize table extension (#294) ([0ac5c370](https://github.com/TritonVM/triton-vm/commit/0ac5c370))
- Combine constraints of illegal `num_words` ([d74e10a9](https://github.com/TritonVM/triton-vm/commit/d74e10a9))
- Combine constraints for stack push / pop ([70361ff1](https://github.com/TritonVM/triton-vm/commit/70361ff1))
- Combine constraints for group `keep_stack` ([2ce5ff15](https://github.com/TritonVM/triton-vm/commit/2ce5ff15))
- Compress compressible constraints ([e53402eb](https://github.com/TritonVM/triton-vm/commit/e53402eb))
- Parallelize polynomial arithmetic in Bezout argument.
- *(test)* Remove super slow try-build test ([65545fe9](https://github.com/TritonVM/triton-vm/commit/65545fe9))

### 📚 Documentation

- Add arithmetization overview page ([c5b7eec1](https://github.com/TritonVM/triton-vm/commit/c5b7eec1))
- Describe cached/just-in-time low-degree-extension ([e72a4185](https://github.com/TritonVM/triton-vm/commit/e72a4185))
- Document the constraint generator ([49864e42](https://github.com/TritonVM/triton-vm/commit/49864e42))
- Document `config` module ([f05643ff](https://github.com/TritonVM/triton-vm/commit/f05643ff))
- Add overview of opcode pressure ([903a5718](https://github.com/TritonVM/triton-vm/commit/903a5718))
- Add AIR circuit node count to arithmetization overview ([c8436943](https://github.com/TritonVM/triton-vm/commit/c8436943))
- Add constraints overview table for AIR of degree 8 ([ac98c22f](https://github.com/TritonVM/triton-vm/commit/ac98c22f))
- Add column counts for various degree lowering targets ([c26cf9cc](https://github.com/TritonVM/triton-vm/commit/c26cf9cc))
- Add dynamic AIR eval cost to overview ([be9b4410](https://github.com/TritonVM/triton-vm/commit/be9b4410))

### ⚙️ Miscellaneous

- Test printing constraint circuits ([5724997f](https://github.com/TritonVM/triton-vm/commit/5724997f))
- Include `{bfe, xfe}_{array, vec}` in prelude ([4c27f360](https://github.com/TritonVM/triton-vm/commit/4c27f360))
- (!) Remove unused method `max_id` ([9e99027f](https://github.com/TritonVM/triton-vm/commit/9e99027f))
- (!) Remove deprecated functions ([d65730d8](https://github.com/TritonVM/triton-vm/commit/d65730d8))
- (!) Make instruction names more consistent ([96c92eab](https://github.com/TritonVM/triton-vm/commit/96c92eab))
- Provide API to overwrite cache decision ([0f313a7b](https://github.com/TritonVM/triton-vm/commit/0f313a7b))
- *(profiler)* Include tracing execution ([4dea54d0](https://github.com/TritonVM/triton-vm/commit/4dea54d0))
- *(test)* Fail if spec needs updating ([3e15ff9a](https://github.com/TritonVM/triton-vm/commit/3e15ff9a))
- *(bench)* Streamline Fibonacci benchmark ([3b210e67](https://github.com/TritonVM/triton-vm/commit/3b210e67))
- Use types over anonymous tuples ([4358acec](https://github.com/TritonVM/triton-vm/commit/4358acec))
- (!) Break cyclic build dependency ([f594167d](https://github.com/TritonVM/triton-vm/commit/f594167d))
- (!) Seal `InputIndicator` trait ([b803e13a](https://github.com/TritonVM/triton-vm/commit/b803e13a))
- (!) Seal trait `AIR` ([44d94848](https://github.com/TritonVM/triton-vm/commit/44d94848))

### ♻️ Refactor

- (!) Use `Polynomial` in FRI proof item ([7367c677](https://github.com/TritonVM/triton-vm/commit/7367c677))
- Compute segments directly ([d62e5587](https://github.com/TritonVM/triton-vm/commit/d62e5587))
- (!) Remove unused `JumpStackTraceRow` ([e257c358](https://github.com/TritonVM/triton-vm/commit/e257c358))
- (!) Improve internal profiler ([fa7c8b70](https://github.com/TritonVM/triton-vm/commit/fa7c8b70))
- (!) *(profiler)* Make `TritonProfiler` private ([1ecd11cd](https://github.com/TritonVM/triton-vm/commit/1ecd11cd))
- *(profiler)* Remove from optimized builds ([f4340159](https://github.com/TritonVM/triton-vm/commit/f4340159))
- *(profiler)* Accumulate loops ([195d1854](https://github.com/TritonVM/triton-vm/commit/195d1854))
- *(test)* Automatically update spec overview ([ac50fa33](https://github.com/TritonVM/triton-vm/commit/ac50fa33))
- *(test)* Simplify constraint checking, etc ([6fd207f4](https://github.com/TritonVM/triton-vm/commit/6fd207f4))
- (!) Remove generic parameter from FRI ([05c6be86](https://github.com/TritonVM/triton-vm/commit/05c6be86))

### ✅ Testing

- Verify FRI failure for too-high degree polys ([262b048e](https://github.com/TritonVM/triton-vm/commit/262b048e))
- Increase coverage of constraint generator ([06b1167b](https://github.com/TritonVM/triton-vm/commit/06b1167b))
- Ensure public types implement auto traits ([da1a99b0](https://github.com/TritonVM/triton-vm/commit/da1a99b0))
- *(bench)* Bench proving with cached / jit trace ([4bc5b9fc](https://github.com/TritonVM/triton-vm/commit/4bc5b9fc))
- Assert uniqueness of nodes on fetch by id ([51eb30a9](https://github.com/TritonVM/triton-vm/commit/51eb30a9))
- Test correct node substitution ([cfe7d093](https://github.com/TritonVM/triton-vm/commit/cfe7d093))
- Test indicator polynomial properties ([b8220690](https://github.com/TritonVM/triton-vm/commit/b8220690))
- Test FRI expansion factors > 4 ([1edecc59](https://github.com/TritonVM/triton-vm/commit/1edecc59))
- `recurse_or_return` needs jump stack content ([b68f0233](https://github.com/TritonVM/triton-vm/commit/b68f0233))
- Test transition constraints of `xb_dot_step` ([13d1fd12](https://github.com/TritonVM/triton-vm/commit/13d1fd12))
- Test transition constraints of `xx_dot_step` ([a64e8c24](https://github.com/TritonVM/triton-vm/commit/a64e8c24))
- Test constraints for every instruction ([0baff704](https://github.com/TritonVM/triton-vm/commit/0baff704))
- Verify that arguments, if any, can be changed ([e55c2474](https://github.com/TritonVM/triton-vm/commit/e55c2474))
- Deduplicate code using macros ([592d7bfa](https://github.com/TritonVM/triton-vm/commit/592d7bfa))
- *(dyn air)* Verify that dynamic and static evaluators agree ([574e407d](https://github.com/TritonVM/triton-vm/commit/574e407d))
- Add example program for `merkle_step_mem` ([d9edddd4](https://github.com/TritonVM/triton-vm/commit/d9edddd4))

### 🎨 Styling

- Enable additional lints ([dd496f71](https://github.com/TritonVM/triton-vm/commit/dd496f71))

### 🛠 Build

- Remove Makefile ([d88a7613](https://github.com/TritonVM/triton-vm/commit/d88a7613))

## [0.41.0](https://github.com/TritonVM/triton-vm/compare/v0.40.0..v0.41.0) - 2024-04-23

### ✨ Features

- Add barycentric evaluation formula ([7fe9b6de](https://github.com/TritonVM/triton-vm/commit/7fe9b6de))

### 🐛 Bug Fixes

- *(profile)* Correct trace randomization profile ([44abcda8](https://github.com/TritonVM/triton-vm/commit/44abcda8))

### ⚡️ Performance

- (!) Include last FRI polynomial into proof ([f8a59c5e](https://github.com/TritonVM/triton-vm/commit/f8a59c5e))
- (!) Use barycentric evaluation in FRI ([991688a5](https://github.com/TritonVM/triton-vm/commit/991688a5))

### 📚 Documentation

- Update links to be valid ([21ab79dd](https://github.com/TritonVM/triton-vm/commit/21ab79dd))

### ♻️ Refactor

- Use barycentric formula in verifier ([cff63b26](https://github.com/TritonVM/triton-vm/commit/cff63b26))

### ✅ Testing

- Assert table-linking arguments' properties ([6b8ffd90](https://github.com/TritonVM/triton-vm/commit/6b8ffd90))
- Test failure of incorrect last ronud poly ([0fc7b7f5](https://github.com/TritonVM/triton-vm/commit/0fc7b7f5))

## [0.40.0](https://github.com/TritonVM/triton-vm/compare/v0.38.2..v0.40.0) - 2024-04-16

### ✨ Features

- (!) Generate profiles despite unfinished tasks ([f7ebd2cb](https://github.com/TritonVM/triton-vm/commit/f7ebd2cb))
- (!) Streamline accessing AET's heights ([3f3a9fd1](https://github.com/TritonVM/triton-vm/commit/3f3a9fd1))
- (!) Track all relevant tables in VM profiler ([fa38fa8b](https://github.com/TritonVM/triton-vm/commit/fa38fa8b))

### 🐛 Bug Fixes

- (!) Don't treat randomizer polynomial special ([9bbe963b](https://github.com/TritonVM/triton-vm/commit/9bbe963b))

### ⚡️ Performance

- Minimize squeezes for combination weights ([50b803c7](https://github.com/TritonVM/triton-vm/commit/50b803c7))
- Re-organize prover steps ([86a7799f](https://github.com/TritonVM/triton-vm/commit/86a7799f))
- *(test)* Use minimal size for quotient domain ([727ff8ec](https://github.com/TritonVM/triton-vm/commit/727ff8ec))
- Use faster polynomial coset evaluation ([29849abe](https://github.com/TritonVM/triton-vm/commit/29849abe))
- Compute Bézout coefficients faster ([652b7e9c](https://github.com/TritonVM/triton-vm/commit/652b7e9c))

### 📚 Documentation

- Add rationale for performed DEEP updates ([74814884](https://github.com/TritonVM/triton-vm/commit/74814884))
- Update documentation of `MasterTable` ([d5c2049c](https://github.com/TritonVM/triton-vm/commit/d5c2049c))
- Describe computation of Bézout coefficients ([06123843](https://github.com/TritonVM/triton-vm/commit/06123843))

### ⚙️ Miscellaneous

- Use fewer glob `use`s ([9ca39513](https://github.com/TritonVM/triton-vm/commit/9ca39513))
- (!) *(circuit)* Use challenge's index ([e05e3ff3](https://github.com/TritonVM/triton-vm/commit/e05e3ff3))
- (!) *(circuit)* No `Challenges` in `evaluate` ([15a8cd7f](https://github.com/TritonVM/triton-vm/commit/15a8cd7f))
- Use constant “lookup table height” more ([22834b03](https://github.com/TritonVM/triton-vm/commit/22834b03))
- Benchmark Bézout coefficient computation ([ab135be4](https://github.com/TritonVM/triton-vm/commit/ab135be4))

### ♻️ Refactor

- Deprecate method `num_quotients()` ([363ae773](https://github.com/TritonVM/triton-vm/commit/363ae773))
- (!) Use `BFieldElement`s everywhere ([05bd271a](https://github.com/TritonVM/triton-vm/commit/05bd271a))
- (!) Make `ProofStream` non-generic ([bde928d4](https://github.com/TritonVM/triton-vm/commit/bde928d4))
- (!) Remove deprecated type aliases ([14d08ef8](https://github.com/TritonVM/triton-vm/commit/14d08ef8))
- (!) Rename `TableId` variants ([c265cf4d](https://github.com/TritonVM/triton-vm/commit/c265cf4d))
- Bypass quotient table ([ff305459](https://github.com/TritonVM/triton-vm/commit/ff305459))

### ✅ Testing

- Benchmark program with lots of memory I/O ([c7613878](https://github.com/TritonVM/triton-vm/commit/c7613878))

### 🎨 Styling

- Improve readability of `.verify()` slightly ([e0ac1096](https://github.com/TritonVM/triton-vm/commit/e0ac1096))

## [0.38.2](https://github.com/TritonVM/triton-vm/compare/v0.38.1..v0.38.2) - 2024-03-14

### ✨ Features

- Convert from `i32` to `NumberOfWords` ([b1fe8e0e](https://github.com/TritonVM/triton-vm/commit/b1fe8e0e))
- More powerful `triton_instr!` macro ([0d1d35ec](https://github.com/TritonVM/triton-vm/commit/0d1d35ec))

### ⚡️ Performance

- Emit instruction's opcodes, not instructions ([7c3de15b](https://github.com/TritonVM/triton-vm/commit/7c3de15b))

### ♻️ Refactor

- Deprecate `Challenges::count()` ([df783a04](https://github.com/TritonVM/triton-vm/commit/df783a04))
- Deprecate too-simple helper method ([bf8adf2d](https://github.com/TritonVM/triton-vm/commit/bf8adf2d))
- Deprecate some shallow methods ([df0715e7](https://github.com/TritonVM/triton-vm/commit/df0715e7))

## [0.38.1](https://github.com/TritonVM/triton-vm/compare/v0.38.0..v0.38.1) - 2024-03-11

### 🐛 Bug Fixes

- Make degree lowering deterministic ([f230ba70](https://github.com/TritonVM/triton-vm/commit/f230ba70))

### ⚙️ Miscellaneous

- Correctly name clippy warning ([568c4836](https://github.com/TritonVM/triton-vm/commit/568c4836))
- Update CI dependencies ([a06c8fa0](https://github.com/TritonVM/triton-vm/commit/a06c8fa0))

### ♻️ Refactor

- Public `NUM_*_CONSTRAINTS` constants ([abd15be8](https://github.com/TritonVM/triton-vm/commit/abd15be8))

## [0.38.0](https://github.com/TritonVM/triton-vm/compare/v0.37.0..v0.38.0) - 2024-03-07

### ✨ Features

- Evaluate AIR constraints in Triton assembly ([b740c605](https://github.com/TritonVM/triton-vm/commit/b740c605))

### ⚙️ Miscellaneous

- Generalize constraint circuit methods ([aecd75f5](https://github.com/TritonVM/triton-vm/commit/aecd75f5))
- Use `nextest` as the test runner ([651934e7](https://github.com/TritonVM/triton-vm/commit/651934e7))
- Use `cargo-llvm-cov` for code coverage ([d7b8e341](https://github.com/TritonVM/triton-vm/commit/d7b8e341))
- Report coverage of TUI's integration tests ([718b211d](https://github.com/TritonVM/triton-vm/commit/718b211d))
- Also run benchmarks as tests ([05c139f8](https://github.com/TritonVM/triton-vm/commit/05c139f8))

### ♻️ Refactor

- (!) Use `const`s for number of constraints ([eb31e6d8](https://github.com/TritonVM/triton-vm/commit/eb31e6d8))
- (!) Remove `BinOp::Sub` ([675acc69](https://github.com/TritonVM/triton-vm/commit/675acc69))
- (!) Remove unused functions ([105d9435](https://github.com/TritonVM/triton-vm/commit/105d9435))
- (!) Drop methods for storing `Proof`s ([c2974e1e](https://github.com/TritonVM/triton-vm/commit/c2974e1e))

### ✅ Testing

- Use fewer resources in constant folding test ([449426aa](https://github.com/TritonVM/triton-vm/commit/449426aa))

## [0.37.0](https://github.com/TritonVM/triton-vm/compare/v0.36.1..v0.37.0) - 2024-02-15

### ✨ Features

- Provide more `BFieldCodec`-related info for `ProofItem`s ([483d9c0d](https://github.com/TritonVM/triton-vm/commit/483d9c0d))
- *(tui)* Enable scrolling in memory widget ([96e2eaca](https://github.com/TritonVM/triton-vm/commit/96e2eaca))
- Provide Fiat-Shamir heuristic related info `ProofItemVariant`s ([02de19ff](https://github.com/TritonVM/triton-vm/commit/02de19ff))
- Simplify parsing of proof item's payload type ([5baeeaeb](https://github.com/TritonVM/triton-vm/commit/5baeeaeb))
- *(lint)* Warn if underscore bindings are being used ([8ab457a6](https://github.com/TritonVM/triton-vm/commit/8ab457a6))
- (!) Return `Err(_)`, don't `panic!`, on unsupported root of unity ([60289eb5](https://github.com/TritonVM/triton-vm/commit/60289eb5))
- Simplify constructing `Claim`s ([31694222](https://github.com/TritonVM/triton-vm/commit/31694222))

### 🐛 Bug Fixes

- *(test)* Remove failure-triggering & superfluous `as` cast ([d8b34e01](https://github.com/TritonVM/triton-vm/commit/d8b34e01))
- *(tui)* Send key, mouse, and paste events only to active component ([74e42d35](https://github.com/TritonVM/triton-vm/commit/74e42d35))

### 📚 Documentation

- Exemplify usage of Triton VM in `examples` directory ([6e4f8f0e](https://github.com/TritonVM/triton-vm/commit/6e4f8f0e))
- Update readme to point at examples folder ([2c989b3a](https://github.com/TritonVM/triton-vm/commit/2c989b3a))
- Update readme of constraint evaluation generator ([fa987f38](https://github.com/TritonVM/triton-vm/commit/fa987f38))

### ⚙️ Miscellaneous

- *(test)* Use iterator transform instead of explicit loop ([fc8b9d20](https://github.com/TritonVM/triton-vm/commit/fc8b9d20))
- In CI, check documentation builds free of warnings ([bf540685](https://github.com/TritonVM/triton-vm/commit/bf540685))
- Run _all_ tests in CI ([d6e99ccc](https://github.com/TritonVM/triton-vm/commit/d6e99ccc))
- Use `From` (not `as`) for lossless conversion ([4e8b28b7](https://github.com/TritonVM/triton-vm/commit/4e8b28b7))
- Enable additional lints ([c309d759](https://github.com/TritonVM/triton-vm/commit/c309d759))
- Don't call `Default::default()` ([cb73d220](https://github.com/TritonVM/triton-vm/commit/cb73d220))
- Avoid explicit `.(into_)iter` for loops ([02018af5](https://github.com/TritonVM/triton-vm/commit/02018af5))
- Name all `clone()`s  explicitly ([cd2e503e](https://github.com/TritonVM/triton-vm/commit/cd2e503e))
- Favor `String::new()` over `"".into()` ([4650087b](https://github.com/TritonVM/triton-vm/commit/4650087b))
- Deprecate aliases `StarkHasher`, `MTMaker` ([f9f0e288](https://github.com/TritonVM/triton-vm/commit/f9f0e288))
- Avoid manually set inclusive range bounds ([d7b5f2c6](https://github.com/TritonVM/triton-vm/commit/d7b5f2c6))
- Simplify construction of some circuits ([56d1bf0e](https://github.com/TritonVM/triton-vm/commit/56d1bf0e))
- Update dependency `twenty-first` ([49b23419](https://github.com/TritonVM/triton-vm/commit/49b23419))

### ♻️ Refactor

- *(test)* More rigorously use `proptest` framework ([a27ca6d4](https://github.com/TritonVM/triton-vm/commit/a27ca6d4))
- (!) Communicate possible FRI setup failures with `Result` ([3fe35ad1](https://github.com/TritonVM/triton-vm/commit/3fe35ad1))
- (!) Communicate possible STARK proving failures with `Result` ([5613f194](https://github.com/TritonVM/triton-vm/commit/5613f194))
- (!) Expose public (re-)exports via `triton_vm::prelude::*` ([0bb30d84](https://github.com/TritonVM/triton-vm/commit/0bb30d84))
- Simplify `use`s through prelude of dependency `twenty-first` ([75da9a17](https://github.com/TritonVM/triton-vm/commit/75da9a17))
- De-duplicate code for `ProofItem` ([a227131c](https://github.com/TritonVM/triton-vm/commit/a227131c))
- Simplify TUI layout construction with new `ratatui` features ([0054597d](https://github.com/TritonVM/triton-vm/commit/0054597d))
- (!) Make lengths of master tables' rows compile-time known ([e52f4cf0](https://github.com/TritonVM/triton-vm/commit/e52f4cf0))
- (!) Integrate `StarkParameters` into `Stark` ([0c5edc73](https://github.com/TritonVM/triton-vm/commit/0c5edc73))
- *(test)* Improve test names ([6caa0e1a](https://github.com/TritonVM/triton-vm/commit/6caa0e1a))
- (!) Make `VMState`'s `sponge` a `Tip5` ([d7b8a3f7](https://github.com/TritonVM/triton-vm/commit/d7b8a3f7))

## [0.36.1](https://github.com/TritonVM/triton-vm/compare/v0.36.0..v0.36.1) - 2024-01-15

### 🐛 Bug Fixes

- *(test)* Don't compile Triton TUI integration tests ([bf46f5ac](https://github.com/TritonVM/triton-vm/commit/bf46f5ac))
- *(visual)* Drop leading 0's from `clk` when printing VM state ([d1a61b16](https://github.com/TritonVM/triton-vm/commit/d1a61b16))

### 📚 Documentation

- Add “Getting Started” section to README.md ([80c10dd3](https://github.com/TritonVM/triton-vm/commit/80c10dd3))

### ⚙️ Miscellaneous

- Run CI on the three biggest platforms ([42ff0618](https://github.com/TritonVM/triton-vm/commit/42ff0618))
- Run code-coverage tool `tarpaulin` only on default features ([e95c7e4f](https://github.com/TritonVM/triton-vm/commit/e95c7e4f))

### ♻️ Refactor

- *(test)* De-duplicate test code for canonical input check ([e180d78e](https://github.com/TritonVM/triton-vm/commit/e180d78e))
- Generalize generic for changing call address of `instruction` ([63a5d1c1](https://github.com/TritonVM/triton-vm/commit/63a5d1c1))

## [0.36.0](https://github.com/TritonVM/triton-vm/compare/v0.35.0..v0.36.0) - 2023-12-22

### ✨ Features

- Add benchmark for execution tracing ([11b360d6](https://github.com/TritonVM/triton-vm/commit/11b360d6))
- Record opstack underflow read/write in AET ([a57ef7c3](https://github.com/TritonVM/triton-vm/commit/a57ef7c3))
- Make Op Stack Table variable length ([b606dc60](https://github.com/TritonVM/triton-vm/commit/b606dc60))
- (!) Instruction `hash` only puts digest on stack ([2e37fb2f](https://github.com/TritonVM/triton-vm/commit/2e37fb2f))
- (!) Make instruction `pop` take an argument in range 1..=5 ([81248b90](https://github.com/TritonVM/triton-vm/commit/81248b90))
- (!) Make instruction `divine` take an argument in range 1..=5 ([5bf3541a](https://github.com/TritonVM/triton-vm/commit/5bf3541a))
- (!) Instruction `divine_sibling` pushes divined digest onto stack ([4602fad8](https://github.com/TritonVM/triton-vm/commit/4602fad8))
- Sponge instructions change stack size ([0fac3fc8](https://github.com/TritonVM/triton-vm/commit/0fac3fc8))
- Extension field instructions change stack size ([f0b3ab8f](https://github.com/TritonVM/triton-vm/commit/f0b3ab8f))
- (!) Make instruction `read_io` take an argument in range 1..=5 ([e138f0a0](https://github.com/TritonVM/triton-vm/commit/e138f0a0))
- (!) Make instruction `write_io` take an argument in range 1..=5 ([b8e5f978](https://github.com/TritonVM/triton-vm/commit/b8e5f978))
- Instruction `assert_vector` shrinks stack by 5 elements ([6a0e19cc](https://github.com/TritonVM/triton-vm/commit/6a0e19cc))
- (!) Make memory instructions take an argument in range 1..=5 ([8ef132af](https://github.com/TritonVM/triton-vm/commit/8ef132af))
- Add benchmark just executing a Triton VM program ([8301d5db](https://github.com/TritonVM/triton-vm/commit/8301d5db))
- (!) Improve error reporting ([48ee1099](https://github.com/TritonVM/triton-vm/commit/48ee1099))
- Only change VM state if instruction execution will work ([d7fbb3fd](https://github.com/TritonVM/triton-vm/commit/d7fbb3fd))
- Add `triton-tui`, a TUI for debugging programs in Triton assembly ([d0d79bce](https://github.com/TritonVM/triton-vm/commit/d0d79bce))
- Allow installing triton-tui as a binary ([047bed9b](https://github.com/TritonVM/triton-vm/commit/047bed9b))
- (de)serialize `VMState` ([8df0723c](https://github.com/TritonVM/triton-vm/commit/8df0723c))

### 🐛 Bug Fixes

- Crash VM when executing `swap 0` ([215f2ede](https://github.com/TritonVM/triton-vm/commit/215f2ede))
- Overflowing subtractions when accessing op stack underflow ([2aa72e77](https://github.com/TritonVM/triton-vm/commit/2aa72e77))
- *(doc)* Correct explanations for previous designs ([4bbc2d2a](https://github.com/TritonVM/triton-vm/commit/4bbc2d2a))
- Account for op stack table length dominating the AET ([f465f756](https://github.com/TritonVM/triton-vm/commit/f465f756))
- Correct calculation of total available memory in Triton VM ([18af2b40](https://github.com/TritonVM/triton-vm/commit/18af2b40))
- Fail Sponge instructions if Sponge state is uninitialized ([881b6c0d](https://github.com/TritonVM/triton-vm/commit/881b6c0d))

### ⚡️ Performance

- Remove redundant constraint preventing op stack underflow ([6215c108](https://github.com/TritonVM/triton-vm/commit/6215c108))
- Use instruction's fast-fail for error reporting, not cloning ([08bbc41f](https://github.com/TritonVM/triton-vm/commit/08bbc41f))

### 📚 Documentation

- Add TIP-0008 “Continuations” ([4b38d01b](https://github.com/TritonVM/triton-vm/commit/4b38d01b))
- Consistently use a space in “op stack” and “jump stack” ([eb8dc840](https://github.com/TritonVM/triton-vm/commit/eb8dc840))
- Delete out-of-date cheat sheet ([69aac2dc](https://github.com/TritonVM/triton-vm/commit/69aac2dc))
- Prose and example for Op Stack Table behavior ([db01232f](https://github.com/TritonVM/triton-vm/commit/db01232f))
- Update AET relations diagram ([f177d658](https://github.com/TritonVM/triton-vm/commit/f177d658))
- Op Stack Table padding ([ad09b8d2](https://github.com/TritonVM/triton-vm/commit/ad09b8d2))
- Update Op Stack Table's AIR ([3fb003b6](https://github.com/TritonVM/triton-vm/commit/3fb003b6))
- Update Processor Table's AET and AIR ([e59eedeb](https://github.com/TritonVM/triton-vm/commit/e59eedeb))
- Reflect changes to instructions, constraints, and mechanics ([ccf123b8](https://github.com/TritonVM/triton-vm/commit/ccf123b8))
- Exemplify error handling ([90151d6c](https://github.com/TritonVM/triton-vm/commit/90151d6c))
- Add changelog ([4d1fc2c0](https://github.com/TritonVM/triton-vm/commit/4d1fc2c0))

### ⚙️ Miscellaneous

- Simplify `use`s ([51878fae](https://github.com/TritonVM/triton-vm/commit/51878fae))
- *(test)* Remove unnecessary paths ([4323b202](https://github.com/TritonVM/triton-vm/commit/4323b202))
- `read_mem` starts reading at current address ([7faad183](https://github.com/TritonVM/triton-vm/commit/7faad183))
- (!) Rename & change debugging methods of `Program` ([abd17904](https://github.com/TritonVM/triton-vm/commit/abd17904))
- Fix spelling of `collinear` (not `colinear`) ([2e9ebd7c](https://github.com/TritonVM/triton-vm/commit/2e9ebd7c))
- Improve changelog generation configuration ([9e3432f3](https://github.com/TritonVM/triton-vm/commit/9e3432f3))
- (!) Remove `Default` derivation from `Program` ([868f49d9](https://github.com/TritonVM/triton-vm/commit/868f49d9))
- Allow tracing program execution from a given starting state ([5f702d47](https://github.com/TritonVM/triton-vm/commit/5f702d47))
- Upgrade dependency `cargo-tarpaulin` ([560f2555](https://github.com/TritonVM/triton-vm/commit/560f2555))

### ♻️ Refactor

- *(examples)* Return program, not instructions ([55c731ed](https://github.com/TritonVM/triton-vm/commit/55c731ed))
- Improve API of `VMProfiler` ([202cb74b](https://github.com/TritonVM/triton-vm/commit/202cb74b))
- *(vm)* Rename `ramp` to `ram_pointer` ([612714d0](https://github.com/TritonVM/triton-vm/commit/612714d0))
- *(processor_table)* Remove never-triggered panics ([6ced006a](https://github.com/TritonVM/triton-vm/commit/6ced006a))
- *(processor_table)* Remove unused struct `ExtProcessorTraceRow` ([d39230f2](https://github.com/TritonVM/triton-vm/commit/d39230f2))
- *(test)* Use crate `test-strategy` ([01e5e229](https://github.com/TritonVM/triton-vm/commit/01e5e229))
- *(test)* Improve testing instruction's transition constraints ([77948e1a](https://github.com/TritonVM/triton-vm/commit/77948e1a))
- *(op_stack)* Simplify recording of op stack underflow I/O calls ([f3803676](https://github.com/TritonVM/triton-vm/commit/f3803676))
- Turn python script for computing opcodes into a rust test ([ddb220f2](https://github.com/TritonVM/triton-vm/commit/ddb220f2))
- *(test)* Also test transition constraints on extension table ([4bd9cf16](https://github.com/TritonVM/triton-vm/commit/4bd9cf16))
- *(test)* Split test program enumeration into individual tests ([cc79cfad](https://github.com/TritonVM/triton-vm/commit/cc79cfad))
- Abstract over legal argument range for various instructions ([a76097e9](https://github.com/TritonVM/triton-vm/commit/a76097e9))
- (!) On success, `Stark::verify` returns `Ok(())`, not `Ok(true)` ([9d3a7065](https://github.com/TritonVM/triton-vm/commit/9d3a7065))
- (!) Remove `terminal_state`, allow running a VM state instead ([fbd58f1c](https://github.com/TritonVM/triton-vm/commit/fbd58f1c))
- Simplify indexing into `OpStack` ([4b31b2fe](https://github.com/TritonVM/triton-vm/commit/4b31b2fe))

### ✅ Testing

- Op stack table row sorting ([7418502b](https://github.com/TritonVM/triton-vm/commit/7418502b))
- Factor for running product with Op Stack Table never panics ([224e7923](https://github.com/TritonVM/triton-vm/commit/224e7923))
- Turn extension field instruction tests into property tests ([067d0053](https://github.com/TritonVM/triton-vm/commit/067d0053))
- Turn `get_colinear_y` into a property test ([39bd4668](https://github.com/TritonVM/triton-vm/commit/39bd4668))
- Use `proptest`, not ad-hoc prop tests, for program parsing tests ([d2acbbf8](https://github.com/TritonVM/triton-vm/commit/d2acbbf8))
- Delete some ignored, obsolete tests ([8deb268a](https://github.com/TritonVM/triton-vm/commit/8deb268a))
- Instructions fail before they modify the state ([c680fab2](https://github.com/TritonVM/triton-vm/commit/c680fab2))

## [0.35.0](https://github.com/TritonVM/triton-vm/compare/v0.34.1..v0.35.0) – 2023-10-17

### ✨ Features

- Better error reporting for failing `assert_vector` ([ee83ab6d](https://github.com/TritonVM/triton-vm/commit/ee83ab6d))
- Include debug information when printing `Program` ([d11aa541](https://github.com/TritonVM/triton-vm/commit/d11aa541))
- (!) Replace instruction `absorb_init` with `sponge_init` ([aca87471](https://github.com/TritonVM/triton-vm/commit/aca87471))
- (!) Add debug instruction `break` ([df6dc4b5](https://github.com/TritonVM/triton-vm/commit/df6dc4b5))

### 🐛 Bug Fixes

- (!) Use `Copy`-trait of `StarkParameters` ([70f3d957](https://github.com/TritonVM/triton-vm/commit/70f3d957))
- Linter v1.73.0 warnings ([641ed393](https://github.com/TritonVM/triton-vm/commit/641ed393))
- Print all helper variables and Sponge state of VM state ([07a54f6e](https://github.com/TritonVM/triton-vm/commit/07a54f6e))
- Disallow changing argument of `swap` to 0 ([bcf61ee6](https://github.com/TritonVM/triton-vm/commit/bcf61ee6))

### 📚 Documentation

- Adapt constraints to new instruction `sponge_init` ([cde031f0](https://github.com/TritonVM/triton-vm/commit/cde031f0))
- `Program` and its new methods ([5ff137dc](https://github.com/TritonVM/triton-vm/commit/5ff137dc))
- Align specification to code ([86a501ea](https://github.com/TritonVM/triton-vm/commit/86a501ea))
- Delete cheatsheet ([e8a5b526](https://github.com/TritonVM/triton-vm/commit/e8a5b526))

### ⚙️ Miscellaneous

- (!) Rename instruction `div` to `div_mod` ([c3ad923a](https://github.com/TritonVM/triton-vm/commit/c3ad923a))
- Reduce number of `as` castings ([540cf66f](https://github.com/TritonVM/triton-vm/commit/540cf66f))
- Remove unused `impl Display for InstructionToken` ([1550f8de](https://github.com/TritonVM/triton-vm/commit/1550f8de))
- Ignore JetBrains IDE's config files ([ddf9e7ad](https://github.com/TritonVM/triton-vm/commit/ddf9e7ad))

### ♻️ Refactor

- (!) Move `padded_height()` into `AET` ([f88d94f3](https://github.com/TritonVM/triton-vm/commit/f88d94f3))
- Store debug information `address_to_label` in `Program` ([d857e838](https://github.com/TritonVM/triton-vm/commit/d857e838))
- Improve readability of `Program` decoding ([e3741d68](https://github.com/TritonVM/triton-vm/commit/e3741d68))
- Extract VM profiling logic into `VMProfiler` ([97ecef8d](https://github.com/TritonVM/triton-vm/commit/97ecef8d))

### ✅ Testing

- Lower number of test cases for some FRI tests ([0159a688](https://github.com/TritonVM/triton-vm/commit/0159a688))
- Print error of failing `assert_vector` ([9bcdac2d](https://github.com/TritonVM/triton-vm/commit/9bcdac2d))
- Separate success and failure of `ensure_eq` ([86b97993](https://github.com/TritonVM/triton-vm/commit/86b97993))
- Improve property testing of Sponge instructions ([e3d90b8e](https://github.com/TritonVM/triton-vm/commit/e3d90b8e))
- Edge case when last Hash Table row is `sponge_init` ([f5f64963](https://github.com/TritonVM/triton-vm/commit/f5f64963))
- Too many returns crash VM, not `VMProfiler` ([a109b890](https://github.com/TritonVM/triton-vm/commit/a109b890))

## [0.34.1](https://github.com/TritonVM/triton-vm/compare/v0.34.0..v0.34.1) – 2023-10-05

### ⚙️ Miscellaneous

- Remove dependency `strum_macros` ([0e53844e](https://github.com/TritonVM/triton-vm/commit/0e53844e))

## [0.34.0](https://github.com/TritonVM/triton-vm/compare/v0.33.0..v0.34.0) – 2023-10-05

### ✨ Features

- Add methods indicating instruction's effect on stack size ([2f3867d8](https://github.com/TritonVM/triton-vm/commit/2f3867d8))
- Derive `Default` for `NonDeterminism` ([954e23e3](https://github.com/TritonVM/triton-vm/commit/954e23e3))

### 📚 Documentation

- Add example uses of Triton VM to top-level documentation ([6c3537ab](https://github.com/TritonVM/triton-vm/commit/6c3537ab))
- Fix specification for instruction `pow` ([be168ef9](https://github.com/TritonVM/triton-vm/commit/be168ef9))

### ♻️ Refactor

- Refactor FRI for increased readability ([61073e2b](https://github.com/TritonVM/triton-vm/commit/61073e2b))
- (!) Don't expose profiling macros ([b2e2a600](https://github.com/TritonVM/triton-vm/commit/b2e2a600))
- Introduce profiling category for witness generation ([6362cc57](https://github.com/TritonVM/triton-vm/commit/6362cc57))

### ✅ Testing

- Use `Arbitrary` for property based tests ([74df64d5](https://github.com/TritonVM/triton-vm/commit/74df64d5))

## [0.33.0](https://github.com/TritonVM/triton-vm/compare/v0.32.1..v0.33.0) – 2023-08-10

### ✨ Features

- (!) Initialize Triton VM's RAM non-deterministically ([fb314aea](https://github.com/TritonVM/triton-vm/commit/fb314aea))

### ⚡️ Performance

- (!) Shrink FRI domain size by splitting quotients into segments ([c4f1e554](https://github.com/TritonVM/triton-vm/commit/c4f1e554))
- Parallelize random linear summing ([1b6b2d4a](https://github.com/TritonVM/triton-vm/commit/1b6b2d4a), [13dfb28f](https://github.com/TritonVM/triton-vm/commit/13dfb28f))
- Use uninitialized memory for some allocations ([b1724829](https://github.com/TritonVM/triton-vm/commit/b1724829))

### 📚 Documentation

- Add specification for TIP-0007 – Run-Time Permutation Check ([82f81f36](https://github.com/TritonVM/triton-vm/commit/82f81f36))

### ⚙️ Miscellaneous

- Upgrade dependencies ([7f24c6eb](https://github.com/TritonVM/triton-vm/commit/7f24c6eb))

### ♻️ Refactor

- Slightly improve interface for `Arithmetic Domain` ([c9907e24](https://github.com/TritonVM/triton-vm/commit/c9907e24))
- Use `.borrow()` directly instead of on explicit `.as_ref()` ([07af1875](https://github.com/TritonVM/triton-vm/commit/07af1875))
- Derive trait `GetSize` for `Proof` instead of implementing it manually ([6ffa9e3d](https://github.com/TritonVM/triton-vm/commit/6ffa9e3d))
- Simplify indexing of `Challenges` through `Index` trait ([bbbc52c8](https://github.com/TritonVM/triton-vm/commit/bbbc52c8))

### ✅ Testing

- Add various tests ([18e81ffd](https://github.com/TritonVM/triton-vm/commit/18e81ffd), [21bb1ae5](https://github.com/TritonVM/triton-vm/commit/21bb1ae5), [364a4464](https://github.com/TritonVM/triton-vm/commit/364a4464))

## [0.32.1](https://github.com/TritonVM/triton-vm/compare/v0.32.0..v0.32.1) – 2023-08-01

### ⚙️ Miscellaneous

- Upgrade dependencies ([5683ea78](https://github.com/TritonVM/triton-vm/commit/5683ea78))

## [0.32.0](https://github.com/TritonVM/triton-vm/compare/v0.31.1..v0.32.0) – 2023-08-01

### ✨ Features

- Introduce macros for writing Triton assembly: `triton_asm!` and `triton_program!`

### ⚙️ Miscellaneous

- Profile runs of the VM

### ♻️ Refactor

- Merge crates `triton-opcodes` and `triton-profiler` into `triton-vm`
- (!) Rename `simulate` to `trace_execution`

## [0.31.1](https://github.com/TritonVM/triton-vm/compare/v0.31.0..v0.31.1) – 2023-07-06

### ✨ Features

- Add helper method for program hashing ([b0555e28](https://github.com/TritonVM/triton-vm/commit/b0555e28))

### ⚙️ Miscellaneous

- Use triton-opcodes v0.31.1 ([59f2c103](https://github.com/TritonVM/triton-vm/commit/59f2c103))

## [0.31.0](https://github.com/TritonVM/triton-vm/compare/v0.30.0..v0.31.0) – 2023-07-05

### 🐛 Bug Fixes

- (!) Don't include `Claim` in `Proof` ([741c6d8f](https://github.com/TritonVM/triton-vm/commit/741c6d8f))

## [0.30.0](https://github.com/TritonVM/triton-vm/compare/v0.29.1..v0.30.0) – 2023-07-04

### ✨ Features

- (!) Attest to the program being run ([5c42531f](https://github.com/TritonVM/triton-vm/commit/5c42531f))
- Add debugging function consuming considerably less RAM ([e556535c](https://github.com/TritonVM/triton-vm/commit/e556535c))

### 🐛 Bug Fixes

- (!) Fix soundness bugs in `skiz` ([a697f3f8](https://github.com/TritonVM/triton-vm/commit/a697f3f8))

### ⚙️ Miscellaneous

- Add benchmarks for proof size ([b4517f85](https://github.com/TritonVM/triton-vm/commit/b4517f85), [618d9443](https://github.com/TritonVM/triton-vm/commit/618d9443))

## [0.29.1](https://github.com/TritonVM/triton-vm/compare/v0.29.0..v0.29.1) – 2023-06-19

### 🐛 Bug Fixes

- Correct opcodes of stack-shrinking u32 instructions ([3d9f838c](https://github.com/TritonVM/triton-vm/commit/3d9f838c))

## [0.29.0](https://github.com/TritonVM/triton-vm/compare/v0.28.0..v0.29.0) – 2023-06-15

### 🐛 Bug Fixes

- executing `lt` on operands 0 and 0 is now possible ([10360b11](https://github.com/TritonVM/triton-vm/commit/10360b11))

### ♻️ Refactor

- (!) `vm::simulate()` returns `Result<_>` ([f52e4a90](https://github.com/TritonVM/triton-vm/commit/f52e4a90))

## [0.28.0](https://github.com/TritonVM/triton-vm/compare/v0.25.1..v0.28.0) – 2023-06-13

### ✨ Features

- Add native interface for proving `Claim`s ([4f2f02ff](https://github.com/TritonVM/triton-vm/commit/4f2f02ff))

### 🐛 Bug Fixes

- (!) Include `Claim` in Fiat-Shamir heuristic ([c786c915](https://github.com/TritonVM/triton-vm/commit/c786c915))

### ⚡️ Performance

- Lower the AIR degree to 4 ([73f9e0a0](https://github.com/TritonVM/triton-vm/commit/73f9e0a0))

### ♻️ Refactor

- Derive `BFieldCodec` where possible ([dc528c41](https://github.com/TritonVM/triton-vm/commit/dc528c41))
- Derive `padded_height` from `Proof` ([0d6c1811](https://github.com/TritonVM/triton-vm/commit/0d6c1811))
- Remove the `padded_height` from the `Claim` ([4be177eb](https://github.com/TritonVM/triton-vm/commit/4be177eb))
- (!) Remove field `Uncast` from `ProofItem` ([27461b10](https://github.com/TritonVM/triton-vm/commit/27461b10))
- (!) Remove trait `MayBeUncast` ([27461b10](https://github.com/TritonVM/triton-vm/commit/27461b10))

### ✅ Testing

- Remove `TestItem`, test using actual `ProofItem`s instead ([54afb081](https://github.com/TritonVM/triton-vm/commit/54afb081))

## [0.25.1](https://github.com/TritonVM/triton-vm/compare/v0.24.0..v0.25.1) – 2023-05-24

### ⚙️ Miscellaneous

- Upgrade dependencies ([328685fb](https://github.com/TritonVM/triton-vm/commit/328685fb))

## [0.24.0](https://github.com/TritonVM/triton-vm/compare/v0.21.0..v0.24.0) – 2023-05-23

### ✨ Features

- Add `BFieldCodec` functions for `Program`, `Claim`, and `Proof` ([07f09801](https://github.com/TritonVM/triton-vm/commit/07f09801), [5e3f66be](https://github.com/TritonVM/triton-vm/commit/5e3f66be))
- Get size for `Program` and `Claim` ([2d827e9c](https://github.com/TritonVM/triton-vm/commit/2d827e9c), [77eed359](https://github.com/TritonVM/triton-vm/commit/77eed359))

### ♻️ Refactor

- (!) Move `BFieldCodec` trait and implementations to `twenty-first` ([193ffa3e](https://github.com/TritonVM/triton-vm/commit/193ffa3e))

## [0.21.0](https://github.com/TritonVM/triton-vm/compare/v0.20.0..v0.21.0) – 2023-05-09

### 🐛 Bug Fixes

- Correct number of checks performed between DEEP-ALI and FRI ([0400d0c3](https://github.com/TritonVM/triton-vm/commit/0400d0c3))

### ⚡️ Performance

- Avoid unnecessary hashing in the Fiat-Shamir heuristic ([0bc5f63d](https://github.com/TritonVM/triton-vm/commit/0bc5f63d))
- Special-case base and extension fields when evaluating the AIR ([4547b961](https://github.com/TritonVM/triton-vm/commit/4547b961))

### 📚 Documentation

- Document zk-STARK parameters ([b1740b71](https://github.com/TritonVM/triton-vm/commit/b1740b71))
- Lower the number of trace randomizers to match DEEP-ALI ([b1740b71](https://github.com/TritonVM/triton-vm/commit/b1740b71))

### ⚙️ Miscellaneous

- Upgrade dependency `twenty-first` ([f0ab8c0a](https://github.com/TritonVM/triton-vm/commit/f0ab8c0a))

### ♻️ Refactor

- (!) Remove state from the STARK's struct ([9d92e0db](https://github.com/TritonVM/triton-vm/commit/9d92e0db))
- Drop the claimed padded height from the proof ([2372461a](https://github.com/TritonVM/triton-vm/commit/2372461a))
- Sort profiled categories by their duration ([57825878](https://github.com/TritonVM/triton-vm/commit/57825878))
- Fail with specific error message instead of panic ([7cf318be](https://github.com/TritonVM/triton-vm/commit/7cf318be))
- Make `StarkParameters` (de)serializeable ([ca7cbe03](https://github.com/TritonVM/triton-vm/commit/ca7cbe03))
- Don't rely on `Digest`s being `Hashable` ([d504fc20](https://github.com/TritonVM/triton-vm/commit/d504fc20))

### ✅ Testing

- Add test for the DEEP update ([29780b06](https://github.com/TritonVM/triton-vm/commit/29780b06))
- Make some assertions already at compile-time ([564a1279](https://github.com/TritonVM/triton-vm/commit/564a1279), [e12db597](https://github.com/TritonVM/triton-vm/commit/e12db597))
- When testing the STARK, use `2^log_2_exp_factor`, not `2^(2^log_2_exp_factor)` ([304a7ea7](https://github.com/TritonVM/triton-vm/commit/304a7ea7))

## [0.20.0](https://github.com/TritonVM/triton-vm/compare/v0.19.0..v0.20.0) – 2023-04-24

### ✨ Features

- (!) Do DEEP-ALI instead of plain ALI in the zk-STARK ([96064413](https://github.com/TritonVM/triton-vm/commit/96064413))
- Add convenience functions for using Triton VM ([0dab32a2](https://github.com/TritonVM/triton-vm/commit/0dab32a2)) ([dda05e4e](https://github.com/TritonVM/triton-vm/commit/dda05e4e))
- Improve Triton profiler ([7edd1a2c](https://github.com/TritonVM/triton-vm/commit/7edd1a2c))
- Make method `debug` more powerful ([ab49df75](https://github.com/TritonVM/triton-vm/commit/ab49df75))

### ⚙️ Miscellaneous

- Add construction of AET (witness generation) to profiling ([c6e7b1e1](https://github.com/TritonVM/triton-vm/commit/c6e7b1e1))

### ♻️ Refactor

- Use `cfg(debug_assertions)`, not environment variable ([b0052f1f](https://github.com/TritonVM/triton-vm/commit/b0052f1f))

## [0.19.0](https://github.com/TritonVM/triton-vm/compare/v0.18.0..v0.19.0) – 2023-03-17

### ✨ Features

- add instruction `pop_count` ([efd90c65](https://github.com/TritonVM/triton-vm/commit/efd90c65))

### ♻️ Refactor

- (!) Parse instructions `dup` and `swap` as taking arguments ([4eecac2b](https://github.com/TritonVM/triton-vm/commit/4eecac2b))
- (!) Enforce labels to start with an alphabetic character or `_` ([5a5e6bad](https://github.com/TritonVM/triton-vm/commit/5a5e6bad))
- (!) Remove method `simulate_no_input` ([089af774](https://github.com/TritonVM/triton-vm/commit/089af774))
- (!) Rename `run` to `debug`, introduce new `run` without debug capabilities ([8bd880ff](https://github.com/TritonVM/triton-vm/commit/8bd880ff))

## [0.18.0](https://github.com/TritonVM/triton-vm/compare/v0.14.0..v0.18.0) – 2023-03-10

### ✨ Features

- (!) Change behavior of instructions `read_mem` and `write_mem` ([022245b7](https://github.com/TritonVM/triton-vm/commit/022245b7))
- (!) Move to Tip5 hash function ([d40f0b62](https://github.com/TritonVM/triton-vm/commit/d40f0b62))
- Use `nom` for parsing Triton assembly ([bbe4aa87](https://github.com/TritonVM/triton-vm/commit/bbe4aa87), [8602892f](https://github.com/TritonVM/triton-vm/commit/8602892f))

### ⚡️ Performance

- Improve constant folding in multicircuits ([c1be5bb9](https://github.com/TritonVM/triton-vm/commit/c1be5bb9))

### 📚 Documentation

- Explain the various cross-table arguments ([567efc00](https://github.com/TritonVM/triton-vm/commit/567efc00))
- Add and improve intra-document links ([a66b20dd](https://github.com/TritonVM/triton-vm/commit/a66b20dd), [9386878c](https://github.com/TritonVM/triton-vm/commit/9386878c))

### ⚙️ Miscellaneous

- Upgrade dependencies ([ff972ff8](https://github.com/TritonVM/triton-vm/commit/ff972ff8))

### ♻️ Refactor

- Replace Instruction Table by Lookup Argument ([543327a0](https://github.com/TritonVM/triton-vm/commit/543327a0))
- Rework U32 Table ([09a4c277](https://github.com/TritonVM/triton-vm/commit/09a4c277), [27190307](https://github.com/TritonVM/triton-vm/commit/27190307))
- Improve clock jump differences check ([04bb5c48](https://github.com/TritonVM/triton-vm/commit/04bb5c48), [9c2f3c0b](https://github.com/TritonVM/triton-vm/commit/9c2f3c0b))

## [0.14.0](https://github.com/TritonVM/triton-vm/compare/v0.13.0..v0.14.0) – 2023-01-20

### ✨ Features

- (!) Introduce Sponge instructions `absorb_init`, `absorb`, and `squeeze` ([af6a9e0e](https://github.com/TritonVM/triton-vm/commit/af6a9e0e))
- Add `nom` parser for Triton Assembly ([ed9e4a90](https://github.com/TritonVM/triton-vm/commit/ed9e4a90))

## [0.13.0](https://github.com/TritonVM/triton-vm/compare/v0.11.0..v0.13.0) – 2023-01-12

### ✨ Features

- (!) Add u32 instructions ([1f3eae84](https://github.com/TritonVM/triton-vm/commit/1f3eae84))

### 📚 Documentation

- Add TIP-0006: Program Attestation ([c694b4c5](https://github.com/TritonVM/triton-vm/commit/c694b4c5))

## [0.11.0](https://github.com/TritonVM/triton-vm/compare/v0.10.0..v0.11.0) – 2022-12-22

### 🐛 Bug Fixes

- (!) Enforce RAM initialization to all zero ([#155](https://github.com/TritonVM/triton-vm/issues/155))

### ⚙️ Miscellaneous

- Upgrade dependencies ([79189cb8](https://github.com/TritonVM/triton-vm/commit/79189cb8))

### ♻️ Refactor

- Represent AET as consecutive memory region ([4477d758](https://github.com/TritonVM/triton-vm/commit/4477d758))
- Distinguish AIR constraints of base and extension tables ([#119](https://github.com/TritonVM/triton-vm/issues/119))
- Reduce memory footprint ([#11](https://github.com/TritonVM/triton-vm/issues/11))
- Split Triton VM's instructions into separate sub-crate ([7bcc09ea](https://github.com/TritonVM/triton-vm/commit/7bcc09ea))

## [0.10.0](https://github.com/TritonVM/triton-vm/compare/v0.9.0..v0.10.0) – 2022-12-19

### 🐛 Bug Fixes

- (!) Adjust `::sample_weights()` and `::sample_indices()` ([cfb0fcb6](https://github.com/TritonVM/triton-vm/commit/cfb0fcb6))

## [0.9.0](https://github.com/TritonVM/triton-vm/compare/v0.8.0..v0.9.0) – 2022-12-08

### ✨ Features

- (!) Allow reading from uninitialized memory, returning zero ([444bb973](https://github.com/TritonVM/triton-vm/commit/444bb973))

## [0.8.0](https://github.com/TritonVM/triton-vm/compare/v0.7.0..v0.8.0) – 2022-12-08

### ✨ Features

- Allow comments in tasm code ([cdbcf439](https://github.com/TritonVM/triton-vm/commit/cdbcf439))

### 🐛 Bug Fixes

- Fail on duplicate labels in parser ([42c41ac2](https://github.com/TritonVM/triton-vm/commit/42c41ac2))

### ⚡️ Performance

- Use iNTT, not fast-interpolate, for polynomial interpolation ([908b7c5f](https://github.com/TritonVM/triton-vm/commit/908b7c5f))

### ♻️ Refactor

- Derive quotient domain the right way around ([d0d3c4f1](https://github.com/TritonVM/triton-vm/commit/d0d3c4f1))
- Use compile-time constants for table's width ([c4868111](https://github.com/TritonVM/triton-vm/commit/c4868111))
- Remove type parameter from arithmetic domain ([381d3643](https://github.com/TritonVM/triton-vm/commit/381d3643))
- Always use `BFieldElements` to perform low-degree extension ([b873f503](https://github.com/TritonVM/triton-vm/commit/b873f503))

## [0.7.0](https://github.com/TritonVM/triton-vm/compare/v0.3.1..v0.7.0) – 2022-11-22

### 🐛 Bug Fixes

- correctly decode `Vec<PartialAuthenticationPath>` ([e7fd6cc2](https://github.com/TritonVM/triton-vm/commit/e7fd6cc2))

### ⚡️ Performance

- (!) Use rust code, not symbolic polynomials, for the AIR ([cd62c59c](https://github.com/TritonVM/triton-vm/commit/cd62c59c))
- Use quotient domain instead of FRI domain wherever applicable ([776fa19c](https://github.com/TritonVM/triton-vm/commit/776fa19c))
- Don't multiply randomizer codeword by random weight ([b105b68d](https://github.com/TritonVM/triton-vm/commit/b105b68d))

### 📚 Documentation

- Add TIP 0004: “Drop U32 Table” ([38293c4e](https://github.com/TritonVM/triton-vm/commit/38293c4e))

### ⚙️ Miscellaneous

- Upgrade dependencies ([cc15b183](https://github.com/TritonVM/triton-vm/commit/cc15b183))
- Add prove_fib_100 benchmark for STARK proving ([1326b12d](https://github.com/TritonVM/triton-vm/commit/1326b12d))

### ♻️ Refactor

- Replace `Result<T, Box<dyn Error>>` with `anyhow::Result<T>` ([448d4cdd](https://github.com/TritonVM/triton-vm/commit/448d4cdd))
- Run `cargo fmt` after constraint-evaluation-generator ([2d183e49](https://github.com/TritonVM/triton-vm/commit/2d183e49))
- Replace TimingReporter with TritonProfiler ([c40c1bc0](https://github.com/TritonVM/triton-vm/commit/c40c1bc0))
- Drop `VecStream` in favor of `Vec<BFieldElement>` ([9668fbea](https://github.com/TritonVM/triton-vm/commit/9668fbea))

## [0.3.1](https://github.com/TritonVM/triton-vm/compare/efae926a43e3b972659cf6d756f2457cd94e4f2e..v0.3.1) – 2022-10-20

Initial release of Triton VM.
