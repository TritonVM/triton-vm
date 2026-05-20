# Hashing

In this last lesson, we'll take a look at hashing instructions.
We'll round things off with a program that's a little more relevant than the toy example we've used so far, and instead
look at proving knowledge of a hash preimage instead.

## Hashing

Triton VM has built-in support for the cryptographic hash function [Tip5](https://eprint.iacr.org/2023/107):
the instruction `hash` consumes the stack's top 10 elements, hashes them, and pushes the resulting digest onto the
stack.

<!-- @formatter:off -->
<triton-playground>
push 0 push 0
push 0 push 0
push 0 push 0
push 0 push 0
push 0 push 0
hash
write_io 5
halt
</triton-playground>
<!-- @formatter:on -->

## Tip5 Sponge

Triton VM also has built-in support for using Tip5 in [Sponge mode](https://en.wikipedia.org/wiki/Sponge_function).
There is exactly one, globally accessible Sponge state.
At program start, it is uninitialized.
The instruction `sponge_init` initializes the Sponge state, after which the instructions `sponge_absorb` and
`sponge_squeeze` perform absorption and squeezing, respectively.

## A Signature Scheme

As with any cryptographic hash function, it's impossible[^compute] to predict Tip5's input given only its output.
This allows us to build a simple signature scheme using Triton VM.[^actual]
Suppose I told you that I knew a secret set of values that hash to
`0xc55a3ddcd9428558, 0x8fc09b4e26e7febe, 0xcc6759262fdd9a89, 0x497ec4974f65e527, 0x0d0f6400210565cd`
under Tip5.
(Psst: the input is 10 zeros. Don't tell anyone!)

[^compute]: more precisely, computationally infeasible
[^actual]: It's a rather inefficient signature scheme at that, but since this is still a tutorial, that's fine.

Now, if a program included a snippet like:

<!-- @formatter:off -->
<triton-playground>
+++
non_determinism.individual_tokens = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
+++
divine 5 
divine 5
hash
push 14220746792122877272 // == 0xc55a3ddcd9428558
push 10358449902914633406 // == 0x8fc09b4e26e7febe
push 14728839126885177993 // == 0xcc6759262fdd9a89
push 05295886365985465639 // == 0x497ec4974f65e527
push 00941080798860502477 // == 0x0d0f6400210565cd
assert_vector
halt // or a continuation of the program
</triton-playground>
<!-- @formatter:on -->

then only knowledge of the secret set of values (here: a bunch of zeros) allows for error-free execution of the program.
A proof of correct execution under Triton VM's zero-knowledge proof system then means that the creator of the proof knew
the secret values at proof creation time.
If the secret values are sufficiently random (and sufficiently secret), then this is like a digital signature.

A program using this idea to sign the message “hi world” could look like this:

<!-- @formatter:off -->
<triton-playground>
+++
non_determinism.individual_tokens = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
+++
push 100 // 'd'
push 108 // 'l'
push 114 // 'r'
push 111 // 'o'
push 119 // 'w'
push  32 // ' '
push 105 // 'i'
push 104 // 'h'
write_io 5
write_io 3
call sign
halt

// this part is like in the snippet above
sign:
  divine 5 
  divine 5
  hash
  push 14220746792122877272
  push 10358449902914633406
  push 14728839126885177993
  push 05295886365985465639
  push 00941080798860502477
  assert_vector
  pop 5
  return
</triton-playground>
<!-- @formatter:on -->

<br>
