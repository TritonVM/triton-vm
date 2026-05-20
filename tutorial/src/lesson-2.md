# Routines and Control Flow

In the exercise of lesson 1, you computed the sum of squares of 2, 3, and 6.
In this lesson, we keep the same math, but introduce two big ideas:

- reusable code with `call`, `return`, and `recurse`
- conditional control flow with `skiz`

## Routines

Suppose we want to square the top stack element many times.[^pow]
It's possible to just repeat the required code in the source code.

[^pow]: And all that while still purposefully overlooking instruction
[`pow`](https://triton-vm.org/spec/instructions.html#pow) – this time, it's about showing off routines!

<!-- @formatter:off -->
<triton-playground>
pick 10 dup 0 mul
pick 10 dup 0 mul
pick 10 dup 0 mul
pick 10 dup 0 mul
// and so on
halt
</triton-playground>
<!-- @formatter:on -->

However, as you might be aware, [copy-and-paste programming](https://en.wikipedia.org/wiki/Copy-and-paste_programming)
has a number of issues.
Instead of this antipattern, you can use the instruction `call` and its counterpart `return` to benefit from routines.

Instruction `call` jumps to an address and “remembers” both where it jumped from and where it jumped to.[^jump_stack]
Executing `return` will then – surprise! – return the control flow to just beyond the last `call` executed.
This means that `call` is more powerful than a simple “go-to”.

When writing Triton assembly, these addresses don't have to be hardcoded – that'd get annoying fast.
Instead, you can introduce labels to your code in the following way:

<!-- @formatter:off -->
<triton-playground>
call my_label  // ╶───╮
halt           //  ←──│───╮
               //     │   │
my_label:      //     │   │
  // do something  ←──╯   │
  return       // ╶───────╯

</triton-playground>
<!-- @formatter:on -->

[^jump_stack]: These addresses are recorded on the
[jump stack](https://triton-vm.org/spec/data-structures.html#jump-stack).

Note that label declarations are not like scopes in other languages.
In the final program, they do not appear at all.
The following is totally valid, but probably a bug:

<!-- @formatter:off -->
<triton-playground>
routine_a:
  // do something
              // ← no return

routine_b:
  // do something else
  return
</triton-playground>
<!-- @formatter:on -->

In this program, a `call routine_a` would execute both the code implied by “do something” and by “do something else”.

## Control Flow

Now that you have an idea about how routines work, let's take a look at conditional execution.
In most programming languages, you have a construct like `if condition then { … } else { … }`.
Since Triton VM is a stack machine, the condition lives on the stack.
The corresponding conditional instruction has the following behavior:

- if the top of the stack is 0, the next instruction in the program is skipped
- if the top of the stack is not 0, the next instruction in the program is executed
- in either case, the top of the stack is removed

Because the instruction **sk**ips **i**f the top of the stack is **z**ero, the instruction is called `skiz`.
This instruction name is a bit unusual, but that's on purpose.
It highlights that at most, one instruction is skipped, and also, that it cares about zero / not zero, and nothing else.

Should you have trouble remembering the behavior of `skiz`, you can substitute the name with `if` in your mind.
Just be aware that, unlike an `if` in languages with scope, `skiz` by itself cannot skip more than one instruction.

To get an if-then-else construct going, we have observed that the following pattern works quite well:

<!-- @formatter:off -->
<triton-playground>
push 0 // this is the condition to branch on

dup 0           // _ condition condition
push 0 eq       // _ condition !condition
swap 1          // _ !condition condition
skiz call if
skiz call else
halt

if:
  pop 1  // remove “!condition” for now
  // do “if” things
  push 0 // put “!condition” back to skip the “else” branch
  return

else:
  // do “else” things
  return
</triton-playground>
<!-- @formatter:on -->

## Loops

When executing instruction `call`, Triton VM not only remembers where it jumped from, but also where it jumped _to_.
This gives rise to an interesting possibility:
we can jump to that same target again, without changing where `return` returns to once it executes.
The instruction that lets us do that is called `recurse`, because… well, it allows for recursion!

Together with `skiz` and `return`, the instruction `recurse` allows to build loops.

<!-- @formatter:off -->
<triton-playground>
           // _
push 101   // _ n
call spin  //  ╶─────╮
halt       //  ←─────│────╮
           //        │    │
// BEFORE: _ n       │    │
// AFTER:  _ 0       │    │
spin:      //  ←─────┴───────────╮
  dup 0    // _ n n       │      │
  push 0   // _ n n 0     │      │
  eq       // _ n (n==0)  │      │
  skiz     // _ n         │      │
    return //  ╶──────────╯      │
  addi -1  // ← new instruction! │
  recurse  //  ╶─────────────────╯
</triton-playground>
<!-- @formatter:on -->

## Halt

You've met all the important control flow instructions that you need to write complex programs for Triton VM.
One of them hasn't been introduced yet: `halt`.

As you might expect, `halt` marks the end of execution – when Triton VM executes `halt`, it shuts down gracefully.
Implicitly terminating a program is an error:

<!-- @formatter:off -->
<triton-playground>
push 42
pop 1
// oops, no halt
</triton-playground>
<!-- @formatter:on -->

## Exercise

Let's start generalizing the sum-of-squares from lesson 1 by using a variable number of inputs.
While for now, the inputs and the result are still hardcoded, the program becomes more amenable to future changes.
I'll get you started again:

<!-- @formatter:off -->
<triton-playground>
push 2  // _ 2
push 3  // _ 2 3
push 6  // _ 2 3 6

push 0  // _ 2 3 6 0 (← accumulator)
push 3  // _ 2 3 6 0 n (← number of iterations)
call square_and_sum
        // _ 49 0

pop 1   // _ 49
push 49 // _ 49 49
eq assert
halt

// BEFORE:    [input; n] 0 n
// AFTER:     sum_of_squares 0
square_and_sum:
  // your code here
</triton-playground>
<!-- @formatter:on -->

<details>
<summary>Hint 1</summary>
Start the loop body by checking whether you still need to square-and-sum something.
</details>
<details>
<summary>Hint 2</summary>
Make sure that between iterations, the stack retains the same general shape.
</details>
<details>
<summary>Hint 3</summary>
A good loop invariant is `// _ [input; i] acc i`, where `i` is initially `n` and counts down to 0.
</details>
<details>
<summary>Solution</summary>
<!-- @formatter:off -->
<triton-playground>
// BEFORE:    [input; n] 0 n
// INVARIANT: [input; i] acc i
// AFTER:     sum_of_squares 0
square_and_sum:
  // are we done?
  dup 0 push 0  // [input; i] acc i i 0
  eq            // [input; i] acc i (i==0)
  skiz return

  // apparently, i != 0, so there are elements left
  pick 2        // [input; i-1] acc i elem
  dup 0 mul     // [input; i-1] acc i elem²
  pick 2 add    // [input; i-1] i (acc + elem²)
  place 1       // [input; i-1] (acc + elem²) i
  addi -1       // [input; i-1] (acc + elem²) (i-1)
  recurse
</triton-playground>
<!-- @formatter:on -->
</details>

Great – you now have full control over Triton VM's flow.
In the next lesson, we'll look at output and the various forms of input, one of which makes Triton VM quite unusual.

<br>
