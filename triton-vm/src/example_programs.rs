use isa::program::Program;
use isa::triton_program;
use lazy_static::lazy_static;

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
    pub static ref MERKLE_TREE_UPDATE: Program = merkle_tree_update();
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
        write_io 1
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
            write_io 1      // _ d
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
        push 0 write_mem 1 pop 1                    // store number of APs at RAM address 0
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
        push 0 read_mem 1 pop 1 dup 0   // get number of APs left to check
                                        // stack: [* r4 r3 r2 r1 r0 num_left num_left]
        push 0 eq                       // see if there are authentication paths left
                                        // stack: [* r4 r3 r2 r1 r0 0 num_left num_left==0]
        skiz return                     // return if no authentication paths left
        push -1 add                     // decrease number of authentication paths left to check
                                        // stack: [* r4 r3 r2 r1 r0 num_left-1]
        push 0 write_mem 1 pop 1        // write decreased number to address 0
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
        merkle_step recurse                         // move up one level in the Merkle tree

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
        return
    )
}

/// Triton program to verifiably change a Merkle tree's leaf. That is:
/// 1. verify that the supplied `old_leaf` is indeed a leaf in the Merkle tree
///    defined by the `merkle_root` and the `tree_height`,
/// 2. update the leaf at the specified `leaf_index` with the `new_leaf`, and
/// 3. return the new Merkle root.
///
/// The authentication path for the leaf to update has to be supplied via RAM.
///
/// - input:
///     - RAM address of leaf's authentication path
///     - leaf index to update
///     - Merkle tree's height
///     - old leaf
///     - (current) merkle root
///     - new leaf
/// - output:
///     - new root
fn merkle_tree_update() -> Program {
    triton_program! {
        read_io 3           // _ *ap leaf_index tree_height
        push 2 pow add      // _ *ap node_index
        dup 1 push 1 dup 2  // _ *ap node_index *ap 1 node_index
        read_io 5           // _ *ap node_index *ap 1 node_index [old_leaf; 5]
        call compute_root   // _ *ap node_index *ap' 1 1 [root; 5]
        read_io 5           // _ *ap node_index *ap' 1 1 [root; 5] [presumed_root; 5]
        assert_vector       // _ *ap node_index *ap' 1 1 [root; 5]
        pop 5 pop 3         // _ *ap node_index
        push 1 swap 1       // _ *ap 1 node_index
        read_io 5           // _ *ap 1 node_index [new_leaf; 5]
        call compute_root   // _ *ap' 1 1 [new_root; 5]
        write_io 5          // _ *ap' 1 1
        pop 3 halt          // _

        // BEFORE: _ *ap                     1 node_index [leaf; 5]
        // AFTER:  _ (*ap + 5 * tree_height) 1          1 [root; 5]
        compute_root:
            merkle_step_mem
            recurse_or_return
    }
}

fn verify_sudoku() -> Program {
    // RAM layout:
    // 0..=8: primes for mapping digits 1..=9
    // 9: flag for whether the Sudoku is valid
    // 10..=90: the Sudoku grid
    //
    // 10 11 12  13 14 15  16 17 18
    // 19 20 21  22 23 24  25 26 27
    // 28 29 30  31 32 33  34 35 36
    //
    // 37 38 39  40 41 42  43 44 45
    // 46 47 48  49 50 51  52 53 54
    // 55 56 57  58 59 60  61 62 63
    //
    // 64 65 66  67 68 69  70 71 72
    // 73 74 75  76 77 78  79 80 81
    // 82 83 84  85 86 87  88 89 90

    triton_program!(
        call initialize_flag
        call initialize_primes
        call read_sudoku
        call write_sudoku_and_check_rows
        call check_columns
        call check_squares
        call assert_flag

        // For checking whether the Sudoku is valid. Initially `true`, set to
        // `false` if any inconsistency is found.
        initialize_flag:
            push 1                        // _ 1
            push 0                        // _ 1 0
            write_mem 1                   // _ 1
            pop 1                         // _
            return

        invalidate_flag:
            push 0                        // _ 0
            push 0                        // _ 0 0
            write_mem 1                   // _ 1
            pop 1                         // _
            return

        assert_flag:
            push 0                        // _ 0
            read_mem 1                    // _ flag -1
            pop 1                         // _ flag
            assert                        // _
            halt

        // For mapping legal Sudoku digits to distinct primes. Helps with
        // checking consistency of rows, columns, and boxes.
        initialize_primes:
            push 23 push 19 push 17
            push 13 push 11 push  7
            push  5 push  3 push  2
            push 1 write_mem 5 write_mem 4
            pop 1
            return

        read_sudoku:
            call read9 call read9 call read9
            call read9 call read9 call read9
            call read9 call read9 call read9
            return

        read9:
            call read1 call read1 call read1
            call read1 call read1 call read1
            call read1 call read1 call read1
            return

        // Applies the mapping from legal Sudoku digits to distinct primes.
        read1:                            // _
            read_io 1                     // _ d
            read_mem 1                    // _ p d-1
            pop 1                         // _ p
            return

        write_sudoku_and_check_rows:      // row0 row1 row2 row3 row4 row5 row6 row7 row8
            push 10                       // row0 row1 row2 row3 row4 row5 row6 row7 row8 10
            call write_and_check_one_row  // row0 row1 row2 row3 row4 row5 row6 row7 19
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

        write_and_check_one_row:          // row addr
            dup 9 dup 9 dup 9
            dup 9 dup 9 dup 9
            dup 9 dup 9 dup 9             // row addr row
            call check_9_numbers          // row addr
            write_mem 5 write_mem 4       // addr+9
            return

        check_columns:
            push 82 call check_one_column
            push 83 call check_one_column
            push 84 call check_one_column
            push 85 call check_one_column
            push 86 call check_one_column
            push 87 call check_one_column
            push 88 call check_one_column
            push 89 call check_one_column
            push 90 call check_one_column
            return

        check_one_column:
            read_mem 1 push -8 add read_mem 1 push -8 add read_mem 1 push -8 add
            read_mem 1 push -8 add read_mem 1 push -8 add read_mem 1 push -8 add
            read_mem 1 push -8 add read_mem 1 push -8 add read_mem 1 pop 1
            call check_9_numbers
            return

        check_squares:
            push 30 call check_one_square
            push 33 call check_one_square
            push 36 call check_one_square
            push 57 call check_one_square
            push 60 call check_one_square
            push 63 call check_one_square
            push 84 call check_one_square
            push 87 call check_one_square
            push 90 call check_one_square
            return

        check_one_square:
            read_mem 3 push -6 add
            read_mem 3 push -6 add
            read_mem 3 pop 1
            call check_9_numbers
            return

        check_9_numbers:
            mul mul mul
            mul mul mul
            mul mul
            // 223092870 = 2·3·5·7·11·13·17·19·23
            push 223092870 eq
            skiz return
            call invalidate_flag
            return
    )
}

pub(crate) fn calculate_new_mmr_peaks_from_append_with_safe_lists() -> Program {
    triton_program!(
        // Stack and memory setup
        push 0                          // _ 0
        push 3                          // _ 0 3
        push 1                          // _ 0 3 1

        push 00457470286889025784
        push 04071246825597671119
        push 17834064596403781463
        push 17484910066710486708
        push 06700794775299091393       // _ 0 3 1 [digest]

        push 06595477061838874830
        push 10897391716490043893
        push 01807330184488272967
        push 05415221245149797169
        push 05057320540678713304       // _ 0 3 1 [digest] [digest]

        push 01838589939278841373
        push 02628975953172153832
        push 06845409670928290394
        push 00880730500905369322
        push 04594396536654736100       // _ 0 3 1 [digest] [digest] [digest]

        push 64                         // _ 0 3 1 [digest] [digest] [digest] 64
        push 2                          // _ 0 3 1 [digest] [digest] [digest] 64 2
        push 323                        // _ 0 3 1 [digest] [digest] [digest] 64 2 323

        push 0                          // _ 0 3 1 [digest] [digest] [digest] 64 2 323 0
        write_mem 3                     // _ 0 3 1 [digest] [digest] [digest] 3
        write_mem 5                     // _ 0 3 1 [digest] [digest] 8
        write_mem 5                     // _ 0 3 1 [digest] 13
        pop 1                           // _ 0 3 1 [digest]

        call tasm_mmr_calculate_new_peaks_from_append_safe
        halt

        // Main function
        // BEFORE: _ [old_leaf_count: u64] *peaks [digest]
        // AFTER:  _ *new_peaks *auth_path
        tasm_mmr_calculate_new_peaks_from_append_safe:
            dup 5 dup 5 dup 5 dup 5 dup 5 dup 5
            call tasm_list_safe_u32_push_digest
            pop 5                       // _ [old_leaf_count: u64] *peaks

            // Create auth_path return value (vector living in RAM)
            // All MMR auth paths have capacity for 64 digests
            push 64                     // _ [old_leaf_count: u64] *peaks 64
            call tasm_list_safe_u32_new_digest

            swap 1
            // _ [old_leaf_count: u64] *auth_path *peaks

            dup 3 dup 3
            // _ [old_leaf_count: u64] *auth_path *peaks [old_leaf_count: u64]

            call tasm_arithmetic_u64_incr
            call tasm_arithmetic_u64_index_of_last_nonzero_bit

            call tasm_mmr_calculate_new_peaks_from_append_safe_while
            // _ [old_leaf_count: u64] *auth_path *peaks (rll = 0)

            pop 1
            swap 3 pop 1 swap 1 pop 1
            // _ *peaks *auth_path

            return

        // Stack start and end: _ *auth_path *peaks rll
        tasm_mmr_calculate_new_peaks_from_append_safe_while:
            dup 0
            push 0
            eq
            skiz
                return
            // _ *auth_path *peaks rll

            swap 2 swap 1
            // _ rll *auth_path *peaks

            dup 0
            dup 0
            call tasm_list_safe_u32_pop_digest
            // _ rll *auth_path *peaks *peaks [new: Digest]

            dup 5
            // _ rll *auth_path *peaks *peaks [new: Digest] *peaks

            call tasm_list_safe_u32_pop_digest
            // _ rll *auth_path *peaks *peaks [new: Digest] [old_peak: Digest]

            // Update authentication path with latest previous_peak
            dup 12
            // _ rll *auth_path *peaks *peaks [new: Digest] [old_peak: Digest] *auth_path

            dup 5 dup 5 dup 5 dup 5 dup 5
            call tasm_list_safe_u32_push_digest
            // _ rll *auth_path *peaks *peaks [new: Digest] [old_peak: Digest]

            hash
            // _ rll *auth_path *peaks *peaks [new_peak: Digest]

            call tasm_list_safe_u32_push_digest
            // _ rll *auth_path *peaks

            swap 1 swap 2
            // _ *auth_path *peaks rll

            push -1
            add
            // _ *auth_path *peaks (rll - 1)

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

        // Before: _ *list, elem[4], elem[3], elem[2], elem[1], elem[0]
        // After:  _
        tasm_list_safe_u32_push_digest:
            dup 5       // _ *list elem[4] elem[3] elem[2] elem[1] elem[0] *list
            push 1 add  // _ *list elem[4] elem[3] elem[2] elem[1] elem[0] *list+1
            read_mem 2  // _ *list elem[4] elem[3] elem[2] elem[1] elem[0] capacity len *list-1

            // Verify that length < capacity
            swap 2      // _ *list elem[4] elem[3] elem[2] elem[1] elem[0] *list-1 len capacity
            dup 1       // _ *list elem[4] elem[3] elem[2] elem[1] elem[0] *list-1 len capacity len
            lt          // _ *list elem[4] elem[3] elem[2] elem[1] elem[0] *list-1 len capacity>len
            assert      // _ *list elem[4] elem[3] elem[2] elem[1] elem[0] *list-1 len

            // Adjust ram pointer
            push 5      // _ *list elem[4] elem[3] elem[2] elem[1] elem[0] *list-1 len 5
            mul         // _ *list elem[4] elem[3] elem[2] elem[1] elem[0] *list-1 5·len
            add         // _ *list elem[4] elem[3] elem[2] elem[1] elem[0] *list+5·len-1
            push 3      // _ *list elem[4] elem[3] elem[2] elem[1] elem[0] *list+5·len 3
            add         // _ *list elem[4] elem[3] elem[2] elem[1] elem[0] *list+5·len+2

            // Write all elements
            write_mem 5 // _ *list *list+5·len+7

            // Remove ram pointer
            pop 1       // _ *list

            // Increase length indicator by one
            read_mem 1  // _ len *list-1
            push 1 add  // _ len *list
            swap 1      // _ *list len
            push 1      // _ *list len 1
            add         // _ *list len+1
            swap 1      // _ len+1 *list
            write_mem 1 // _ *list+1
            pop 1       // _
            return

        // BEFORE: _ capacity
        // AFTER:
        tasm_list_safe_u32_new_digest:
            // Convert capacity in number of elements to number of VM words
            // required for that list
            dup 0       // _ capacity capacity
            push 5      // _ capacity capacity 5
            mul         // _ capacity 5·capacity
                        // _ capacity capacity_in_bfes
            push 2      // _ capacity capacity_in_bfes 2
            add         // _ capacity capacity_in_bfes+2
                        // _ capacity words_to_allocate

            call tasm_memory_dyn_malloc     // _ capacity *list

            // Write initial length = 0 to `*list`, and capacity to `*list + 1`
            push 0                          // _ capacity *list 0
            swap 1                          // _ capacity 0 *list
            write_mem 2                     // _ (*list+2)
            push -2                         // _ (*list+2) -2
            add                             // _ *list
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

        // BEFORE: _ value_hi value_lo
        // AFTER: _ log2_floor(value)
        tasm_arithmetic_u64_log_2_floor:
            swap 1
            push 1
            dup 1
            // _ value_lo value_hi 1 value_hi

            skiz call tasm_arithmetic_u64_log_2_floor_then
            skiz call tasm_arithmetic_u64_log_2_floor_else
            // _ log2_floor(value)

            return

        tasm_arithmetic_u64_log_2_floor_then:
            // value_hi != 0
            // _ value_lo value_hi 1
            swap 1
            swap 2
            pop 2
            // _ value_hi

            log_2_floor
            push 32
            add
            // _ (log2_floor(value_hi) + 32)

            push 0
            // _ (log2_floor(value_hi) + 32) 0

            return

        tasm_arithmetic_u64_log_2_floor_else:
            // value_hi == 0
            // _ value_lo value_hi
            pop 1
            log_2_floor
            return

        // Before: _ *list
        // After:  _ elem{N - 1}, elem{N - 2}, ..., elem{0}
        tasm_list_safe_u32_pop_digest:
            read_mem 1      // _ len *list-1
            push 1 add      // _ len *list

            // Assert that length is not 0
            dup 1           // _ len *list len
            push 0          // _ len *list len 0
            eq              // _ len *list len==0
            push 0          // _ len *list len==0 0
            eq              // _ len *list len!=0
            assert          // _ len *list

            // Decrease length value by one and write back to memory
            dup 1           // _ len *list len
            push -1         // _ len *list len -1
            add             // _ len *list len-1
            swap 1          // _ len len-1 *list
            write_mem 1     // _ len *list+1
            push -1 add     // _ len *list

            // Read elements
            swap 1          // _ *list len
            push 5          // _ *list len 5
            mul             // _ *list 5·len
                            // _ *list offset_for_last_element
            add             // _ *list+offset_for_last_element
                            // _ address_for_last_element
            read_mem 5      // _ [elements] address_for_last_element-5
            pop 1           // _ [elements]
            return

        // BEFORE: rhs_hi rhs_lo lhs_hi lhs_lo
        // AFTER:  (rhs & lhs)_hi (rhs & lhs)_lo
        tasm_arithmetic_u64_and:
            swap 3
            and
            // _ lhs_lo rhs_lo (lhs_hi & rhs_hi)

            swap 2
            and
            // _ (lhs_hi & rhs_hi) (rhs_lo & lhs_lo)

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

            // The above value is now a power of two in u64. Calling log2_floor
            // on this value gives us the index we are looking for.
            call tasm_arithmetic_u64_log_2_floor

            return


        // Return a pointer to a free address and allocate `size` words for this
        // pointer

        // Before: _ size
        // After:  _ *next_addr
        tasm_memory_dyn_malloc:
            push 0                     // _ size *free_pointer
            read_mem 1                 // _ size *next_addr' *free_pointer-1
            pop 1                      // _ size *next_addr'

            // add 1 iff `next_addr` was 0, i.e. uninitialized.
            dup 0                      // _ size *next_addr' *next_addr'
            push 0                     // _ size *next_addr' *next_addr' 0
            eq                         // _ size *next_addr' (*next_addr' == 0)
            add                        // _ size *next_addr

            dup 0                      // _ size *next_addr *next_addr
            dup 2                      // _ size *next_addr *next_addr size

            // Ensure that `size` does not exceed 2^32
            split
            swap 1
            push 0
            eq
            assert

            add                        // _ size *free_pointer *next_addr *(next_addr + size)

            // Ensure that no more than 2^32 words are allocated, because I
            // don't want a wrap-around in the address space
            split
            swap 1
            push 0
            eq
            assert

            swap 1                     // _ size *(next_addr + size) *next_addr
            swap 2                     // _ *next_addr *(next_addr + size) size
            pop 1                      // _ *next_addr *(next_addr + size)
            push 0                     // _ *next_addr *(next_addr + size) *free_pointer
            write_mem 1                // _ *next_addr *free_pointer+1
            pop 1                      // _ *next_addr
            return

        // BEFORE: rhs_hi rhs_lo lhs_hi lhs_lo
        // AFTER: (rhs ^ lhs)_hi (rhs ^ lhs)_lo
        tasm_arithmetic_u64_xor:
            swap 3
            xor
            // _ lhs_lo rhs_lo (lhs_hi ^ rhs_hi)

            swap 2
            xor
            // _ (lhs_hi ^ rhs_hi) (rhs_lo ^ lhs_lo)

            return
    )
}
