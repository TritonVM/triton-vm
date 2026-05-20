# The Stack

A stack machine is a computer model where instructions read from and write to a
[stack](https://en.wikipedia.org/wiki/Stack_(abstract_data_type)).
The stack is ordered, and the top element is the one each instruction touches first.

In Triton VM, the stack is the data structure that programs interact with constantly.
Because it is so central, it's what we will look at first.

## Basic Stack Manipulation

At the start of the program, the stack is empty[^empty].
You can directly add elements to the top of the stack through instruction `push`.
`push` takes an immediate argument, like this:

<!-- @formatter:off -->
<triton-playground>
push 3
push 5
halt
</triton-playground>
<!-- @formatter:on -->

[^empty]: That's a lie:
Triton VM's stack has always at least 16 elements on it.
The reason has to do with the zero-knowledge proof system and is out of scope for this tutorial.
If you are interested in more details, take a look
[here](https://triton-vm.org/spec/data-structures.html#operational-stack).
In this tutorial, you can just ignore those 16 elements the VM starts with.

To mentally keep track of the current state of the stack, we sometimes use comments like this:

<!-- @formatter:off -->
<triton-playground>
       // _     (← initially, the stack is empty)
push 3 // _ 3   (← the top of the stack is 3)  
push 5 // _ 3 5 (← the stack contains 3, then 5 on top)
pop 2  // _     (← the stack is empty once more)
halt
</triton-playground>
<!-- @formatter:on -->

You've just seen a new instruction: `pop`.
Like `push`, instruction `pop` also takes an immediate argument.
In this case, the argument specifies how many elements to pop off the stack.
You can `pop` between 1 and 5 elements at once, anything beyond that is an error.
The following program will be rejected before Triton VM even starts execution:

<!-- @formatter:off -->
<triton-playground>
push 0 push 1 push 2 // _ 0 1 2
push 3 push 4 push 5 // _ 0 1 2 3 4 5
pop 6                // _ 0 (?)
halt
</triton-playground>
<!-- @formatter:on -->

If you try to remove too many elements from the stack, Triton VM will crash:

<!-- @formatter:off -->
<triton-playground>
pop 1 // uh oh!
halt
</triton-playground>
<!-- @formatter:on -->

When the VM crashes you get a bunch of debug information to help you find the issue.
If you want to know what the various symbols mean, check out the specification's chapter on
[registers](https://triton-vm.org/spec/registers.html).
You don't need that knowledge to go on with the tutorial, though.

## Arithmetic

Arithmetic instructions, like `add`, `mul`, and `eq` consume the two elements on top of the stack and push the result.
For example, to compute the sum of 2 and 3, you can:

<!-- @formatter:off -->
<triton-playground>
push 2 push 3 // _ 2 3
add           // _ 5
halt
</triton-playground>
<!-- @formatter:on -->

## Types in Triton VM

While we've happily manipulated the stack and even done some arithmetic already, we haven't talked about the _type_ of
the elements on the stack yet.
In order to facilitate the zero-knowledge proof system, Triton VM's native type is the
[finite field](https://en.wikipedia.org/wiki/Finite_field) that contains `0xffff_ffff_0000_0001` elements.
Because of the way finite fields work, it's a fair mental model to say that arithmetic always happens modulo this
number.

For example, adding `0xffff_ffff_0000_0000` to 1 gives `0xffff_ffff_0000_0001`, which is 0.
This is similar in other languages that let you access numeric types at a low-ish level.
For example, adding Rust's `u64::MAX` to 1 also gives 0.[^overflow]

[^overflow]: You need to turn off overflow checks explicitly or crank up optimizations, but the point stands.

The only other type that Triton VM has some native support for is `u32`, _i.e._, a 32-bit unsigned integer type.
While all elements handled by Triton VM are finite field elements, the set of
[“bitwise arithmetic” instructions](https://triton-vm.org/spec/instructions.html#bitwise-arithmetic)
crash the VM if the elements they touch do not fall within the interval `[0, 0xffff_ffff]`.
Apart from being useful by themselves, these instructions can be used to guarantee the absence of overflows.

## Advanced Stack Manipulation

Sometimes, it's difficult or even impossible to keep your data organized in such a way that it is always on top of the
stack exactly when you need it to be.[^hassle]
The instructions that re-arrange the stack are the solution to that problem:

- `pick` lets you move an element from deeper in the stack to the top
- `place` does the opposite, moving the top of the stack to some deeper location
- `swap` exchanges the top of the stack with some element at a deeper location

<!-- @formatter:off -->
<triton-playground>
// set up the stack
push 7 push 6 push 5 push 4 // _ 7 6 5 4
push 3 push 2 push 1 push 0 // _ 7 6 5 4 3 2 1 0

// start manipulating
        // _ 7 6 5 4 3 2 1 0
pick 5  // _ 7 6 4 3 2 1 0 5 (← 5 is now on top, the rest got pushed down)
place 7 // _ 5 7 6 4 3 2 1 0 (← 5 is now at position 7, the rest got moved up)
swap 3  // _ 5 7 6 4 0 2 1 3 (← 0 and 3 have swapped places, the rest remains)
halt
</triton-playground>
<!-- @formatter:on -->

[^hassle]: Apart from the theoretical limitations, it can also be a major hassle to achieve this.

## Element Duplication

Often, you'll need to use some element more than once.
All the instructions we have learned so far either create a hardcoded element, move the element around, or consume it.
The instruction `dup` lets you duplicate a stack element and pushes it onto the stack.
For example, we can use this to raise some value to the third power:[^pow]

<!-- @formatter:off -->
<triton-playground>
push 2 // _ 2
dup 0  // _ 2 2   (← The element at index 0 got duplicated)
dup 1  // _ 2 2 2 (← The element at index 1 got duplicated; here, we could also have done `dup 0`)
mul    // _ 2 4
mul    // _ 8
halt
</triton-playground>
<!-- @formatter:on -->

[^pow]: Perhaps you've already spotted the dedicated [`pow`](https://triton-vm.org/spec/instructions.html#pow)
instruction. Let's just pretend we want to use `mul` instead – this is about showing off `dup`, after all!

## Assertions

Triton VM is designed for programs that make _sure_ that certain conditions hold.
Because of this, an important instruction is `assert`.
This instruction will act like `pop 1` if the top of the stack is 1.
If the top of the stack is not 1, executing this instruction will crash Triton VM.

Instruction `assert` is often paired with instruction `eq`.
Like the name might suggest, `eq` checks for equality:
it consumes the two elements on top of the stack, then pushes a 1 if they are equal, and a 0 otherwise:

<!-- @formatter:off -->
<triton-playground>
push 2 push 3 // _ 2 3
mul           // _ 6
push 6        // _ 6 6
eq            // _ 1
assert        // _
halt
</triton-playground>
<!-- @formatter:on -->

## Exercise

Let's apply what we've learned so far.
Write a program that computes the squares of 2, 3, and 6, then sums those squares.
Finally, it should assert that the result equals 49.
I'll get you started:

<!-- @formatter:off -->
<triton-playground>
push 2 // _ 2
push 3 // _ 2 3
push 6 // _ 2 3 6

// your code goes here

push 49
eq assert
halt
</triton-playground>
<!-- @formatter:on -->

<br>
<details>
<summary>If you need a hint, a possible solution waits for you in here.</summary>
<!-- @formatter:off -->
<triton-playground>
push 2 // _ 2
push 3 // _ 2 3
push 6 // _ 2 3 6

// start of the solution
pick 2 dup 0 mul // _ 3 6 4
pick 2 dup 0 mul // _ 6 4 9
pick 2 dup 0 mul // _ 4 9 36
add              // _ 4 45
add              // _ 49
// end of the solution

push 49
eq assert
halt

</triton-playground>
<!-- @formatter:on -->
</details>

Congrats, you have mastered Triton VM's stack!
In the next lesson, we'll look at control flow in Triton VM.

<br>
