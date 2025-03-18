//! # [Triton VM] Example: Factorial
//!
//! This example demonstrates how to use Triton VM to prove the correct
//! execution of a program.
//!
//! [Triton VM]: https://triton-vm.org/

use triton_vm::prelude::*;

fn main() {
    // The program that is to be run in Triton VM. Written in Triton assembly.
    // The given example program computes the factorial of the public input.
    let factorial_program = triton_program!(
                            // op stack:
        read_io 1           // n
        push 1              // n accumulator
        call factorial      // 0 accumulator!
        write_io 1          // 0
        halt

        factorial:          // n acc
            // if n == 0: return
            dup 1           // n acc n
            push 0 eq       // n acc n==0
            skiz            // n acc
                return      // 0 acc
            // else: multiply accumulator with n and recurse
            dup 1           // n acc n
            mul             // n acc·n
            swap 1          // acc·n n
            push -1 add     // acc·n n-1
            swap 1          // n-1 acc·n
            recurse
    );

    // Note that all arithmetic is in the prime field with 2^64 - 2^32 + 1
    // elements. The `bfe!` macro is used to create elements of this field.
    let public_input = PublicInput::from([bfe!(1_000)]);

    // The execution of the factorial program is already fully determined by the
    // public input. Hence, in this case, there is no need for specifying
    // non-determinism.
    let non_determinism = NonDeterminism::default();

    // Generate
    //   - the claim for the given program and input, and
    //   - the proof of correct execution.
    //
    // The claim contains the following public information:
    //   - the program's hash digest under hash function Tip5,
    //   - the program's public input, and
    //   - the program's public output.
    //
    // Triton VM is zero-knowledge with respect to almost everything else.
    // The only other piece of revealed information is an upper bound for the
    // number of steps the program was running for.
    //
    // Triton VM's default parameters give a (conjectured) security level of 160
    // bits.
    let (stark, claim, proof) =
        triton_vm::prove_program(factorial_program, public_input, non_determinism).unwrap();

    let verdict = triton_vm::verify(stark, &claim, &proof);
    assert!(verdict);

    println!("Successfully verified proof.");
    let claimed_output = claim.output.iter().map(|o| o.value());
    println!("Verifiably correct output:  {claimed_output:?}");

    let conjectured_security_level = stark.security_level;
    println!("Conjectured security level is {conjectured_security_level} bits.");

    let upper_bound_of_execution_steps = proof.padded_height().unwrap();
    println!("Executing the program took at most {upper_bound_of_execution_steps} cycles.");
}

#[test]
fn factorial() {
    main();
}
