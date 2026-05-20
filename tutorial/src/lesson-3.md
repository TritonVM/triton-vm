# Input and Output

So far, we've only computed on hardcoded values and didn't communicate the result at all;
everything stayed within Triton VM.
In this lesson, we'll look at the ways in which Triton VM communicates with the outside world:
public input, public output, and – possibly most interesting – secret input.

## Public Input

A dedicated object for public input, filled with 0 or more elements, can be used to communicate information to Triton
VM.
The instruction to read from public input is `read_io`.
The instruction `read_io` takes an immediate argument in the range `1..=5`, indicating how many elements should be read.
As you would expect, the read elements are pushed onto the stack.

In the execution environment below, you can specify public input in the “frontmatter”, like so:

<!-- @formatter:off -->
<triton-playground>
+++
public_input = [17, 42]
+++

read_io 2          // _ 17 42
push 42 eq assert  // _ 17
push 17 eq assert  // _ 
halt
</triton-playground>
<!-- @formatter:on -->

## Public Output

The dual to `read_io` is the instruction `write_io`.
It also takes an immediate argument, indicating how many elements should be written to the output.

<!-- @formatter:off -->
<triton-playground>
+++
public_input = [17, 42]
+++

read_io 2
write_io 2
halt
</triton-playground>
<!-- @formatter:on -->

## Secret Input

One reason Triton VM exists is to generate zero-knowledge proofs of correct execution.
Not too many details of this proof system are relevant for this tutorial;
what is relevant is that programs need a way to consume _private_ data, _i.e._, input that is not public.
The zero-knowledge proof of correct program execution that can be generated for a Triton program always contains both
public input and public output;
it never contains secret input.

The instruction for reading secret input is called `divine`.
Just like its counterpart for public input, instruction `divine` takes an immediate argument in the range `1..=5`,
indicating how many elements should be read and pushed onto the stack.
The name of the instruction is purposefully arcane to highlight the special source of the elements that `divine` creates
on the stack.

From a program's perspective, this is a form of non-determinism:
since the proof of correct program execution is not tied to secret input in any way, the same program with the same
(public) input can behave in different ways;

<!-- @formatter:off -->
<triton-playground>
+++
public_input = []
non_determinism.individual_tokens = [0]
+++
// try changing the secret input     ↑ 

push 0    // _ 0
divine 1  // _ 0 something_from_secret_in
skiz
  assert  // ← might get executed, depending on secret input
halt
</triton-playground>
<!-- @formatter:on -->

Of course, if you can see or set the secret input yourself, the program is totally deterministic.
The name and reasoning exist (and only really make sense) in the context of the zero-knowledge proof system.

## RAM

Not only can Triton VM store elements on its stack, it also has “Random Access Memory” available.
There are two main instructions that can access RAM: `read_mem` and `write_mem`.
Both take an immediate argument in the range `1..=5`, indicating how many elements to read or write.
In both cases, the stack's top element is taken as the address to read from or write to, the “RAM pointer”.
After any single element is read (or written), the RAM pointer is decremented (or incremented, respectively).
This allows reading and writing long sequences with ease:

<!-- @formatter:off -->
<triton-playground>
+++
public_input = [8, 9]

[non_determinism.ram]
0 = 5
1 = 6
2 = 7
+++

read_io 2   // _ 8 9
push 3      // _ 8 9 write_addr
write_mem 2 // _ (write_addr+2)
pop 1       // _

push 4      // _ read_addr
read_mem 5  // _ 9 8 7 6 5 (read_addr-5)
pop 1       // _ 9 8 7 6 5
write_io 5  // _
halt
</triton-playground>
<!-- @formatter:on -->

You might have noticed that RAM is not initialized to all-zero.
Instead, RAM is another part of what makes Triton VM a non-deterministic virtual machine.
It's possible to use RAM as the same kind of input that instruction `divine` gives access to.
In particular, an address that is read from _before_ it is written to holds a non-deterministic value the _first_ time
it is read from.
The second read is guaranteed to return the same value, and an address that's written to is guaranteed to hold the
written value.

<!-- @formatter:off -->
<triton-playground>
+++
[non_determinism.ram]
0 = 42
+++
// overwrite RAM address 0 with value 0
push 0      // _ 0
push 0      // _ 0 0
write_mem 1 // _ 1
pop 1       // _

// read from RAM address 0
push 0      // _ 0
read_mem 1  // _ ? -1
pop 1       // _ ?

// assert that the read value is, in fact, 0, not 42
push 0      // _ ? 0
eq assert   // _
halt
</triton-playground>
<!-- @formatter:on -->

Non-deterministic RAM is a good way to communicate a lot of data to Triton VM without executing many `read_io` or
`divine` instructions.[^not-in-play]

[^not-in-play]: Although using this form of input for large amounts of data is probably better done when using Triton VM
programmatically, not here in your browser.

## Exercise

Let's revisit the sum-of-squares program from the previous lessons.
In the last lesson, we used various control flow instructions to build a loop instead of repeating code.
With the knowledge of `read_io` and `divine` in the toolbox, the program starts becoming more useful.

Write a program that takes 2 elements from public input.
The first argument indicates the number of elements in the sum;
the second element equals the expected sum-of-squares.
The individual elements that are to be squared, then summed, should be supplied via secret input.
The program should `assert` that the expected value matches the computed value.

A program like this can be used to prove the knowledge of certain information without revealing anything about that
information.
With the specific program you'll write here, the proof of correct execution will show that the prover knows `n` values
whose sum of squares equals the claimed result – without revealing what those values are.

Here's some example input to start you off:

<!-- @formatter:off -->
<triton-playground>
+++
public_input = [3, 49]
non_determinism.individual_tokens = [2, 3, 6]
+++
// your code here
</triton-playground>
<!-- @formatter:on -->

<details>
<summary>A possible solution could look like this.</summary>
<!-- @formatter:off -->
<triton-playground>
+++
public_input = [3, 49]
non_determinism.individual_tokens = [2, 3, 6]
+++
push 0    // _ 0
read_io 1 // _ 0 n
call square_and_sum
pop 1     // _ sum_of_squares
read_io 1
eq assert
halt

// BEFORE: _ 0 n
// AFTER:  _ sum_of_squares 0
square_and_sum:
  dup 0 push 0 eq // _ acc i (i==0)
  skiz return     // _ acc i
  divine 1        // _ acc i x
  dup 0 mul       // _ acc i x²
  pick 2 add      // _ i (acc+x²)
  place 1         // _ (acc+x²) i
  addi -1         // _ (acc+x²) (i-1)
  recurse
</triton-playground>
<!-- @formatter:on -->
</details>

<br>
