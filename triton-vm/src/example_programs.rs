use lazy_static::lazy_static;

use crate::program::Program;
use crate::triton_program;

lazy_static! {
    pub static ref FIBONACCI_SEQUENCE: Program = fibonacci_sequence();
    pub static ref GREATEST_COMMON_DIVISOR: Program = greatest_common_divisor();
    pub static ref PROGRAM_WITH_MANY_U32_INSTRUCTIONS: Program =
        program_with_many_u32_instructions();
    pub static ref VERIFY_SUDOKU: Program = verify_sudoku();
    pub static ref CALCULATE_NEW_MMR_PEAKS_FROM_APPEND_WITH_SAFE_LISTS: Program =
        calculate_new_mmr_peaks_from_append_with_safe_lists();
    pub static ref MERKLE_TREE_AUTHENTICATION_PATH_VERIFY: Program =
        merkle_tree_authentication_path_verify();
}

fn fibonacci_sequence() -> Program {
    triton_program!(
        // initialize stack: ⊥ 0 1 i
        push 0
        push 1
        read_io 1

        // is any looping necessary?
        dup 0
        skiz
        call fib_loop

        // pop zero, write result
        pop 1
        write_io
        halt

        // before: ⊥ 0 1 i
        // after:  ⊥ fib(i-1) fib(i) 0
        fib_loop:
            push -1   // ⊥ a b j -1
            add       // ⊥ a b (j-1)
            swap 2    // ⊥ (j-1) b a
            dup 1     // ⊥ (j-1) b a b
            add       // ⊥ (j-1) b (a+b)
            swap 1    // ⊥ (j-1) (a+b) b
            swap 2    // ⊥ b (a+b) (j-1)
            dup 0     // ⊥ b (a+b) (j-1) (j-1)
            skiz      // ⊥ b (a+b) (j-1)
            recurse
            return
    )
}

fn greatest_common_divisor() -> Program {
    triton_program!(
        read_io 2    // _ a b
        dup 1        // _ a b a
        dup 1        // _ a b a b
        lt           // _ a b b<a
        skiz         // _ a b
            swap 1   // _ d n where n > d

        loop_cond:
        dup 1
        push 0
        eq
        skiz
            call terminate  // _ d n where d != 0
        dup 1               // _ d n d
        dup 1               // _ d n d n
        div_mod             // _ d n q r
        swap 2              // _ d r q n
        pop 2               // _ d r
        swap 1              // _ r d
        call loop_cond

        terminate:
            // _ d n where d == 0
            write_io // _ d
            halt
    )
}

fn program_with_many_u32_instructions() -> Program {
    triton_program!(
        push 1311768464867721216 split
        push 13387 push 78810 lt
        push     5 push     7 pow
        push 69584 push  6796 xor
        push 64972 push  3915 and
        push 98668 push 15787 div_mod
        push 15787 push 98668 div_mod
        push 98141 push  7397 and
        push 67749 push 60797 lt
        push 49528 split
        push 53483 call lsb
        push 79655 call is_u32
        push 60615 log_2_floor
        push    13 push     5 pow
        push 86323 push 37607 xor
        push 32374 push 20636 pow
        push 97416 log_2_floor
        push 14392 push 31589 div_mod
        halt
        lsb:
            push 2 swap 1 div_mod return
        is_u32:
            split pop 1 push 0 eq return
    )
}

/// Triton program to verify Merkle authentication paths.
/// - input: merkle root, number of leafs, leaf values, APs
/// - output: Result<(), VMFail>
fn merkle_tree_authentication_path_verify() -> Program {
    triton_program!(
        read_io 1                                   // number of authentication paths to test
                                                    // stack: [num]
        mt_ap_verify:                               // proper program starts here
        push 0 swap 1 write_mem pop 1               // store number of APs at RAM address 0
                                                    // stack: []
        read_io 5                                   // read Merkle root
                                                    // stack: [r4 r3 r2 r1 r0]
        call check_aps
        pop 5                                       // leave clean stack: Merkle root
                                                    // stack: []
        halt                                        // done – should be “return”

        // subroutine: check AP one at a time
        // stack before: [* r4 r3 r2 r1 r0]
        // stack after:  [* r4 r3 r2 r1 r0]
        check_aps:
        push 0 read_mem dup 0           // get number of APs left to check
                                        // stack: [* r4 r3 r2 r1 r0 0 num_left num_left]
        push 0 eq                       // see if there are authentication paths left
                                        // stack: [* r4 r3 r2 r1 r0 0 num_left num_left==0]
        skiz return                     // return if no authentication paths left
        push -1 add write_mem pop 1     // decrease number of authentication paths left to check
                                        // stack: [* r4 r3 r2 r1 r0]
        call get_idx_and_leaf
                                        // stack: [* r4 r3 r2 r1 r0 idx l4 l3 l2 l1 l0]
        call traverse_tree
                                        // stack: [* r4 r3 r2 r1 r0   1 d4 d3 d2 d1 d0]
        call assert_tree_top
                                        // stack: [* r4 r3 r2 r1 r0]
        recurse                         // check next AP

        // subroutine: read index & hash leaf
        // stack before: [*]
        // stack after:  [* idx l4 l3 l2 l1 l0]
        get_idx_and_leaf:
        read_io 1                                   // read node index
        read_io 5                                   // read leaf's value
        return

        // subroutine: go up tree
        // stack before: [* r4 r3 r2 r1 r0 idx l4 l3 l2 l1 l0]
        // stack after:  [* r4 r3 r2 r1 r0   1 d4 d3 d2 d1 d0]
        traverse_tree:
        dup 5 push 1 eq skiz return                 // break loop if node index is 1
        divine_sibling hash recurse                 // move up one level in the Merkle tree

        // subroutine: compare digests
        // stack before: [* r4 r3 r2 r1 r0   1 d4 d3 d2 d1 d0]
        // stack after:  [* r4 r3 r2 r1 r0]
        assert_tree_top:
                                                    // stack: [* r4 r3 r2 r1 r0   1 d4 d3 d2 d1 d0]
        swap 1 swap 2 swap 3 swap 4 swap 5
                                                    // stack: [* r4 r3 r2 r1 r0 d4 d3 d2 d1 d0   1]
        assert                                      // ensure the entire path was traversed
                                                    // stack: [* r4 r3 r2 r1 r0 d4 d3 d2 d1 d0]
        assert_vector                               // actually compare to root of tree
        pop 5                                       // clean up stack, leave only one root
        return
    )
}

fn verify_sudoku() -> Program {
    triton_program!(
        call initialize_primes
        call read_sudoku
        call initialize_flag
        call write_sudoku_and_check_rows
        call check_columns
        call check_squares
        push 0
        read_mem
        assert
        halt

        // For mapping legal Sudoku digits to distinct primes. Helps with checking consistency of
        // rows, columns, and boxes.
        initialize_primes:
            push 1 push  2 write_mem
            push 2 push  3 write_mem
            push 3 push  5 write_mem
            push 4 push  7 write_mem
            push 5 push 11 write_mem
            push 6 push 13 write_mem
            push 7 push 17 write_mem
            push 8 push 19 write_mem
            push 9 push 23 write_mem
            pop 5 pop 4
            return

        read_sudoku:
            call read9
            call read9
            call read9
            call read9
            call read9
            call read9
            call read9
            call read9
            call read9
            return

        read9:
            call read1
            call read1
            call read1
            call read1
            call read1
            call read1
            call read1
            call read1
            call read1
            return

        // Applies the mapping from legal Sudoku digits to distinct primes.
        read1:                            // _
            read_io 1                     // _ d
            read_mem                      // _ d p
            swap 1                        // _ p d
            pop 1                         // _ p
            return

        initialize_flag:
            push 0
            push 1
            write_mem
            pop 1
            return

        write_sudoku_and_check_rows:      // row0 row1 row2 row3 row4 row5 row6 row7 row8
            push 9                        // row0 row1 row2 row3 row4 row5 row6 row7 row8 9
            call write_and_check_one_row  // row0 row1 row2 row3 row4 row5 row6 row7 18
            call write_and_check_one_row  // row0 row1 row2 row3 row4 row5 row6 27
            call write_and_check_one_row  // row0 row1 row2 row3 row4 row5 36
            call write_and_check_one_row  // row0 row1 row2 row3 row4 45
            call write_and_check_one_row  // row0 row1 row2 row3 54
            call write_and_check_one_row  // row0 row1 row2 63
            call write_and_check_one_row  // row0 row1 72
            call write_and_check_one_row  // row0 81
            call write_and_check_one_row  // 90
            pop 1                         // ⊥
            return

        write_and_check_one_row:          // s0 s1 s2 s3 s4 s5 s6 s7 s8 mem_addr
            push 1                        // s0 s1 s2 s3 s4 s5 s6 s7 s8 mem_addr 1
            call multiply_and_write       // s0 s1 s2 s3 s4 s5 s6 s7 (mem_addr+1) s8
            call multiply_and_write       // s0 s1 s2 s3 s4 s5 s6 (mem_addr+2) (s8·s7)
            call multiply_and_write       // s0 s1 s2 s3 s4 s5 (mem_addr+3) (s8·s7·s6)
            call multiply_and_write       // s0 s1 s2 s3 s4 (mem_addr+4) (s8·s7·s6·s5)
            call multiply_and_write       // s0 s1 s2 s3 (mem_addr+5) (s8·s7·s6·s5·s4)
            call multiply_and_write       // s0 s1 s2 (mem_addr+6) (s8·s7·s6·s5·s4·s3)
            call multiply_and_write       // s0 s1 (mem_addr+7) (s8·s7·s6·s5·s4·s3·s2)
            call multiply_and_write       // s0 (mem_addr+8) (s8·s7·s6·s5·s4·s3·s2·s1)
            call multiply_and_write       // (mem_addr+9) (s8·s7·s6·s5·s4·s3·s2·s1·s0)
            push 223092870                // (mem_addr+9) (s8·s7·s6·s5·s4·s3·s2·s1·s0) 223092870
            eq                            // (mem_addr+9) (s8·s7·s6·s5·s4·s3·s2·s1·s0==223092870)
            skiz                          // (mem_addr+9)
            return
            push 0                        // (mem_addr+9) 0
            push 0                        // (mem_addr+9) 0 0
            write_mem                     // (mem_addr+9) 0
            pop 1                         // (mem_addr+9)
            return

        multiply_and_write:               // s mem_addr acc
            dup 2                         // s mem_addr acc s
            mul                           // s mem_addr (acc·s)
            swap 1                        // s (acc·s) mem_addr
            push 1                        // s (acc·s) mem_addr 1
            add                           // s (acc·s) (mem_addr+1)
            swap 1                        // s (mem_addr+1) (acc·s)
            swap 2                        // (acc·s) (mem_addr+1) s
            write_mem                     // (acc·s) (mem_addr+1)
            swap 1                        // (mem_addr+1) (acc·s)
            return

        check_columns:
            push 1
            call check_one_column
            push 2
            call check_one_column
            push 3
            call check_one_column
            push 4
            call check_one_column
            push 5
            call check_one_column
            push 6
            call check_one_column
            push 7
            call check_one_column
            push 8
            call check_one_column
            push 9
            call check_one_column
            return

        check_one_column:
            call get_column_element
            call get_column_element
            call get_column_element
            call get_column_element
            call get_column_element
            call get_column_element
            call get_column_element
            call get_column_element
            call get_column_element
            pop 1
            call check_9_numbers
            return

        get_column_element:
            push 9
            add
            read_mem
            swap 1
            return

        check_squares:
            push 10
            call check_one_square
            push 13
            call check_one_square
            push 16
            call check_one_square
            push 37
            call check_one_square
            push 40
            call check_one_square
            push 43
            call check_one_square
            push 64
            call check_one_square
            push 67
            call check_one_square
            push 70
            call check_one_square
            return

        check_one_square:
            read_mem
            swap 1
            push 1
            add
            read_mem
            swap 1
            push 1
            add
            read_mem
            swap 1
            push 7
            add
            read_mem
            swap 1
            push 1
            add
            read_mem
            swap 1
            push 1
            add
            read_mem
            swap 1
            push 7
            add
            read_mem
            swap 1
            push 1
            add
            read_mem
            swap 1
            push 1
            add
            read_mem
            swap 1
            pop 1
            call check_9_numbers
            return

        check_9_numbers:
            mul
            mul
            mul
            mul
            mul
            mul
            mul
            mul
            // 223092870 = 2·3·5·7·11·13·17·19·23
            push 223092870
            eq
            skiz
            return
            push 0
            push 0
            write_mem
            pop 1
            return
    )
}

pub(crate) fn calculate_new_mmr_peaks_from_append_with_safe_lists() -> Program {
    triton_program!(
        // Stack and memory setup
        push 0
        push 3
        push 1
        push 457470286889025784
        push 4071246825597671119
        push 17834064596403781463
        push 17484910066710486708
        push 6700794775299091393
        push 6
        push 02628975953172153832
        write_mem
        push 10
        push 01807330184488272967
        write_mem
        push 12
        push 06595477061838874830
        write_mem
        push 1
        push 2
        write_mem
        push 11
        push 10897391716490043893
        write_mem
        push 7
        push 01838589939278841373
        write_mem
        push 8
        push 05057320540678713304
        write_mem
        push 4
        push 00880730500905369322
        write_mem
        push 5
        push 06845409670928290394
        write_mem
        push 3
        push 04594396536654736100
        write_mem
        push 2
        push 64
        write_mem
        push 9
        push 05415221245149797169
        write_mem
        push 0
        push 323
        write_mem
        pop 5 pop 5 pop 3

        // Call the main function, followed by `halt`
            call tasm_mmr_calculate_new_peaks_from_append_safe
            halt

        // Main function declaration
            // BEFORE: _ old_leaf_count_hi old_leaf_count_lo *peaks [digests (new_leaf)]
            // AFTER: _ *new_peaks *auth_path
            tasm_mmr_calculate_new_peaks_from_append_safe:
                dup 5 dup 5 dup 5 dup 5 dup 5 dup 5
                call tasm_list_safe_u32_push_digest
                pop 5
                // stack: _ old_leaf_count_hi old_leaf_count_lo *peaks

                // Create auth_path return value (vector living in RAM)
                push 64 // All MMR auth paths have capacity for 64 digests
                call tasm_list_safe_u32_new_digest

                swap 1
                // stack: _ old_leaf_count_hi old_leaf_count_lo *auth_path *peaks

                dup 3 dup 3
                // stack: _ old_leaf_count_hi old_leaf_count_lo *auth_path *peaks old_leaf_count_hi old_leaf_count_lo

                call tasm_arithmetic_u64_incr
                call tasm_arithmetic_u64_index_of_last_nonzero_bit

                call tasm_mmr_calculate_new_peaks_from_append_safe_while
                // stack: _ old_leaf_count_hi old_leaf_count_lo *auth_path *peaks (rll = 0)

                pop 1
                swap 3 pop 1 swap 1 pop 1
                // stack: _ *peaks *auth_path

                return

            // Stack start and end: _ old_leaf_count_hi old_leaf_count_lo *auth_path *peaks rll
            tasm_mmr_calculate_new_peaks_from_append_safe_while:
                dup 0
                push 0
                eq
                skiz
                    return
                // Stack: _ old_leaf_count_hi old_leaf_count_lo *auth_path *peaks rll

                swap 2 swap 1
                // Stack: _ old_leaf_count_hi old_leaf_count_lo rll *auth_path *peaks

                dup 0
                dup 0
                call tasm_list_safe_u32_pop_digest
                // Stack: _ old_leaf_count_hi old_leaf_count_lo rll *auth_path *peaks *peaks [digest (new_hash)]

                dup 5
                // Stack: _ old_leaf_count_hi old_leaf_count_lo rll *auth_path *peaks *peaks [digest (new_hash)] *peaks

                call tasm_list_safe_u32_pop_digest
                // Stack: _ old_leaf_count_hi old_leaf_count_lo rll *auth_path *peaks *peaks [digest (new_hash)] [digests (previous_peak)]

                // Update authentication path with latest previous_peak
                dup 12
                dup 5 dup 5 dup 5 dup 5 dup 5
                // Stack: _ old_leaf_count_hi old_leaf_count_lo rll *auth_path *peaks *peaks [digest (new_hash)] [digests (previous_peak)] *auth_path [digests (previous_peak)]

                call tasm_list_safe_u32_push_digest
                // Stack: _ old_leaf_count_hi old_leaf_count_lo rll *auth_path *peaks *peaks [digest (new_hash)] [digests (previous_peak)]

                hash
                // Stack: _ old_leaf_count_hi old_leaf_count_lo rll *auth_path *peaks *peaks [digests (new_peak)]

                call tasm_list_safe_u32_push_digest
                // Stack: _ old_leaf_count_hi old_leaf_count_lo rll *auth_path *peaks

                swap 1 swap 2
                // Stack: _ old_leaf_count_hi old_leaf_count_lo *auth_path *peaks rll

                push -1
                add
                // Stack: _ old_leaf_count_hi old_leaf_count_lo *auth_path *peaks (rll - 1)

                recurse


            // Before: _ value_hi value_lo
            // After: _ (value + 1)_hi (value + 1)_lo
            tasm_arithmetic_u64_incr_carry:
                pop 1
                push 1
                add
                dup 0
                push 4294967296
                eq
                push 0
                eq
                assert
                push 0
                return

            tasm_arithmetic_u64_incr:
                push 1
                add
                dup 0
                push 4294967296
                eq
                skiz
                    call tasm_arithmetic_u64_incr_carry
                return

            // Before: _ *list, elem{N - 1}, elem{N - 2}, ..., elem{0}
            // After: _
            tasm_list_safe_u32_push_digest:
                dup 5
                // stack : _  *list, elem{N - 1}, elem{N - 2}, ..., elem{0}, *list

                read_mem
                // stack : _  *list, elem{N - 1}, elem{N - 2}, ..., elem{0}, *list, length

                // Verify that length < capacity (before increasing length by 1)
                    swap 1
                    push 1
                    add
                    // stack : _  *list, elem{N - 1}, elem{N - 2}, ..., elem{0}, length, (*list + 1)

                    read_mem
                    // stack : _  *list, elem{N - 1}, elem{N - 2}, ..., elem{0}, length, (*list + 1), capacity

                    dup 2 lt
                    // dup 2 eq
                    // push 0 eq
                    // stack : _  *list, elem{N - 1}, elem{N - 2}, ..., elem{0}, length, (*list + 1), capacity > length

                    assert
                    // stack : _  *list, elem{N - 1}, elem{N - 2}, ..., elem{0}, length, (*list + 1)

                    swap 1

                push 5
                mul

                // stack : _  *list, elem{N - 1}, elem{N - 2}, ..., elem{0}, (*list + 1), length * elem_size

                add
                push 1
                add
                // stack : _  *list, elem{N - 1}, elem{N - 2}, ..., elem{0}, (*list + length * elem_size + 2) -- top of stack is where we will store elements

                swap 1
                write_mem
                push 1
                add
                swap 1
                write_mem
                push 1
                add
                swap 1
                write_mem
                push 1
                add
                swap 1
                write_mem
                push 1
                add
                swap 1
                write_mem

                // stack : _  *list, address

                pop 1
                // stack : _  *list

                // Increase length indicator by one
                read_mem
                // stack : _  *list, length

                push 1
                add
                // stack : _  *list, length + 1

                write_mem
                // stack : _  *list

                pop 1
                // stack : _

                return

            tasm_list_safe_u32_new_digest:
                // _ capacity

                // Convert capacity in number of elements to number of VM words required for that list
                dup 0
                push 5
                mul

                // _ capacity (capacity_in_bfes)

                push 2
                add
                // _ capacity (words to allocate)

                call tasm_memory_dyn_malloc
                // _ capacity *list

                // Write initial length = 0 to `*list`
                push 0
                write_mem
                // _ capacity *list

                // Write capactiy to memory location `*list + 1`
                push 1
                add
                // _ capacity (*list + 1)

                swap 1
                write_mem
                // _ (*list + 1) capacity

                push -1
                add
                // _ *list

                return

            tasm_arithmetic_u64_decr:
                push -1
                add
                dup 0
                push -1
                eq
                skiz
                    call tasm_arithmetic_u64_decr_carry
                return

            tasm_arithmetic_u64_decr_carry:
                pop 1
                push -1
                add
                dup 0
                push -1
                eq
                push 0
                eq
                assert
                push 4294967295
                return

            // BEFORE: _ *list list_length
            // AFTER: _ *list
            tasm_list_safe_u32_set_length_digest:
                // Verify that new length does not exceed capacity
                dup 0
                dup 2
                push 1
                add
                read_mem
                // Stack: *list list_length list_length (*list + 1) capacity

                swap 1
                pop 1
                // Stack: *list list_length list_length capacity

                lt
                push 0
                eq
                // Stack: *list list_length list_length <= capacity

                assert
                // Stack: *list list_length

                write_mem
                // Stack: *list

                return

            // BEFORE: _ value_hi value_lo
            // AFTER: _ log2_floor(value)
            tasm_arithmetic_u64_log_2_floor:
                swap 1
                push 1
                dup 1
                // stack: _ value_lo value_hi 1 value_hi

                skiz call tasm_arithmetic_u64_log_2_floor_then
                skiz call tasm_arithmetic_u64_log_2_floor_else
                // stack: _ log2_floor(value)

                return

            tasm_arithmetic_u64_log_2_floor_then:
                // value_hi != 0
                // stack: _ value_lo value_hi 1
                swap 1
                swap 2
                pop 2
                // stack: _ value_hi

                log_2_floor
                push 32
                add
                // stack: _ (log2_floor(value_hi) + 32)

                push 0
                // stack: _ (log2_floor(value_hi) + 32) 0

                return

            tasm_arithmetic_u64_log_2_floor_else:
                // value_hi == 0
                // stack: _ value_lo value_hi
                pop 1
                log_2_floor
                return

            // Before: _ *list
            // After: _ elem{N - 1}, elem{N - 2}, ..., elem{0}
            tasm_list_safe_u32_pop_digest:
                read_mem
                // stack : _  *list, length

                // Assert that length is not 0
                dup 0
                push 0
                eq
                push 0
                eq
                assert
                // stack : _  *list, length

                // Decrease length value by one and write back to memory
                swap 1
                dup 1
                push -1
                add
                write_mem
                swap 1
                // stack : _ *list initial_length

                push 5
                mul

                // stack : _  *list, (offset_for_last_element = (N * initial_length))

                add
                push 1
                add
                // stack : _  address_for_last_element

                read_mem
                swap 1
                push -1
                add
                read_mem
                swap 1
                push -1
                add
                read_mem
                swap 1
                push -1
                add
                read_mem
                swap 1
                push -1
                add
                read_mem
                swap 1

                // Stack: _  [elements], address_for_last_unread_element

                pop 1
                // Stack: _  [elements]

                return

            // BEFORE: rhs_hi rhs_lo lhs_hi lhs_lo
            // AFTER: (rhs & lhs)_hi (rhs & lhs)_lo
            tasm_arithmetic_u64_and:
                swap 3
                and
                // stack: _ lhs_lo rhs_lo (lhs_hi & rhs_hi)

                swap 2
                and
                // stack: _ (lhs_hi & rhs_hi) (rhs_lo & lhs_lo)

                return

            // BEFORE: _ value_hi value_lo
            // AFTER: _ index_of_last_non-zero_bit
            tasm_arithmetic_u64_index_of_last_nonzero_bit:
                dup 1
                dup 1
                // _ value_hi value_lo value_hi value_lo

                call tasm_arithmetic_u64_decr
                // _ value_hi value_lo (value - 1)_hi (value - 1)_lo

                push 4294967295
                push 4294967295
                // _ value_hi value_lo (value - 1)_hi (value - 1)_lo 0xFFFFFFFF 0xFFFFFFFF

                call tasm_arithmetic_u64_xor
                // _ value_hi value_lo ~(value - 1)_hi ~(value - 1)_lo

                call tasm_arithmetic_u64_and
                // _ (value & ~(value - 1))_hi (value & ~(value - 1))_lo

                // The above value is now a power of two in u64. Calling log2_floor on this
                // value gives us the index we are looking for.
                call tasm_arithmetic_u64_log_2_floor

                return


            // Return a pointer to a free address and allocate `size` words for this pointer
            // Before: _ size
            // After: _ *next_addr
            tasm_memory_dyn_malloc:
                push 0  // _ size *free_pointer
                read_mem                   // _ size *free_pointer *next_addr'

                // add 1 iff `next_addr` was 0, i.e. uninitialized.
                dup 0                      // _ size *free_pointer *next_addr' *next_addr'
                push 0                     // _ size *free_pointer *next_addr' *next_addr' 0
                eq                         // _ size *free_pointer *next_addr' (*next_addr' == 0)
                add                        // _ size *free_pointer *next_addr

                dup 0                      // _ size *free_pointer *next_addr *next_addr
                dup 3                      // _ size *free_pointer *next_addr *next_addr size

                // Ensure that `size` does not exceed 2^32
                split
                swap 1
                push 0
                eq
                assert

                add                        // _ size *free_pointer *next_addr *(next_addr + size)

                // Ensure that no more than 2^32 words are allocated, because I don't want a wrap-around
                // in the address space
                split
                swap 1
                push 0
                eq
                assert

                swap 1                     // _ size *free_pointer *(next_addr + size) *next_addr
                swap 3                     // _ *next_addr *free_pointer *(next_addr + size) size
                pop 1                      // _ *next_addr *free_pointer *(next_addr + size)
                write_mem
                pop 1                      // _ next_addr
                return

            // BEFORE: rhs_hi rhs_lo lhs_hi lhs_lo
            // AFTER: (rhs ^ lhs)_hi (rhs ^ lhs)_lo
            tasm_arithmetic_u64_xor:
                swap 3
                xor
                // stack: _ lhs_lo rhs_lo (lhs_hi ^ rhs_hi)

                swap 2
                xor
                // stack: _ (lhs_hi ^ rhs_hi) (rhs_lo ^ lhs_lo)

                return
    )
}
