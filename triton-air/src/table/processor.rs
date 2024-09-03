use std::cmp::max;
use std::ops::Mul;

use constraint_circuit::ConstraintCircuitBuilder;
use constraint_circuit::ConstraintCircuitMonad;
use constraint_circuit::DualRowIndicator;
use constraint_circuit::DualRowIndicator::CurrentBaseRow;
use constraint_circuit::DualRowIndicator::CurrentExtRow;
use constraint_circuit::DualRowIndicator::NextBaseRow;
use constraint_circuit::DualRowIndicator::NextExtRow;
use constraint_circuit::InputIndicator;
use constraint_circuit::SingleRowIndicator;
use constraint_circuit::SingleRowIndicator::BaseRow;
use constraint_circuit::SingleRowIndicator::ExtRow;
use isa::instruction::Instruction;
use isa::instruction::InstructionBit;
use isa::instruction::ALL_INSTRUCTIONS;
use isa::op_stack::NumberOfWords;
use isa::op_stack::OpStackElement;
use isa::op_stack::NUM_OP_STACK_REGISTERS;
use itertools::izip;
use itertools::Itertools;
use strum::EnumCount;
use twenty_first::math::x_field_element::EXTENSION_DEGREE;
use twenty_first::prelude::*;

use crate::challenge_id::ChallengeId;
use crate::challenge_id::ChallengeId::ClockJumpDifferenceLookupIndeterminate;
use crate::challenge_id::ChallengeId::CompressProgramDigestIndeterminate;
use crate::challenge_id::ChallengeId::CompressedProgramDigest;
use crate::challenge_id::ChallengeId::HashCIWeight;
use crate::challenge_id::ChallengeId::HashDigestIndeterminate;
use crate::challenge_id::ChallengeId::HashInputIndeterminate;
use crate::challenge_id::ChallengeId::InstructionLookupIndeterminate;
use crate::challenge_id::ChallengeId::JumpStackCiWeight;
use crate::challenge_id::ChallengeId::JumpStackClkWeight;
use crate::challenge_id::ChallengeId::JumpStackIndeterminate;
use crate::challenge_id::ChallengeId::JumpStackJsdWeight;
use crate::challenge_id::ChallengeId::JumpStackJsoWeight;
use crate::challenge_id::ChallengeId::JumpStackJspWeight;
use crate::challenge_id::ChallengeId::OpStackClkWeight;
use crate::challenge_id::ChallengeId::OpStackFirstUnderflowElementWeight;
use crate::challenge_id::ChallengeId::OpStackIb1Weight;
use crate::challenge_id::ChallengeId::OpStackIndeterminate;
use crate::challenge_id::ChallengeId::OpStackPointerWeight;
use crate::challenge_id::ChallengeId::ProgramAddressWeight;
use crate::challenge_id::ChallengeId::ProgramInstructionWeight;
use crate::challenge_id::ChallengeId::ProgramNextInstructionWeight;
use crate::challenge_id::ChallengeId::RamClkWeight;
use crate::challenge_id::ChallengeId::RamIndeterminate;
use crate::challenge_id::ChallengeId::RamInstructionTypeWeight;
use crate::challenge_id::ChallengeId::RamPointerWeight;
use crate::challenge_id::ChallengeId::RamValueWeight;
use crate::challenge_id::ChallengeId::SpongeIndeterminate;
use crate::challenge_id::ChallengeId::StackWeight0;
use crate::challenge_id::ChallengeId::StackWeight1;
use crate::challenge_id::ChallengeId::StackWeight10;
use crate::challenge_id::ChallengeId::StackWeight11;
use crate::challenge_id::ChallengeId::StackWeight12;
use crate::challenge_id::ChallengeId::StackWeight13;
use crate::challenge_id::ChallengeId::StackWeight14;
use crate::challenge_id::ChallengeId::StackWeight15;
use crate::challenge_id::ChallengeId::StackWeight2;
use crate::challenge_id::ChallengeId::StackWeight3;
use crate::challenge_id::ChallengeId::StackWeight4;
use crate::challenge_id::ChallengeId::StackWeight5;
use crate::challenge_id::ChallengeId::StackWeight6;
use crate::challenge_id::ChallengeId::StackWeight7;
use crate::challenge_id::ChallengeId::StackWeight8;
use crate::challenge_id::ChallengeId::StackWeight9;
use crate::challenge_id::ChallengeId::StandardInputIndeterminate;
use crate::challenge_id::ChallengeId::StandardOutputIndeterminate;
use crate::challenge_id::ChallengeId::U32CiWeight;
use crate::challenge_id::ChallengeId::U32Indeterminate;
use crate::challenge_id::ChallengeId::U32LhsWeight;
use crate::challenge_id::ChallengeId::U32ResultWeight;
use crate::challenge_id::ChallengeId::U32RhsWeight;
use crate::cross_table_argument::CrossTableArg;
use crate::cross_table_argument::EvalArg;
use crate::cross_table_argument::LookupArg;
use crate::cross_table_argument::PermArg;
use crate::table;
use crate::table_column::MasterBaseTableColumn;
use crate::table_column::MasterExtTableColumn;
use crate::table_column::ProcessorBaseTableColumn;
use crate::table_column::ProcessorBaseTableColumn::ClockJumpDifferenceLookupMultiplicity;
use crate::table_column::ProcessorBaseTableColumn::IsPadding;
use crate::table_column::ProcessorBaseTableColumn::OpStackPointer;
use crate::table_column::ProcessorBaseTableColumn::CI;
use crate::table_column::ProcessorBaseTableColumn::CLK;
use crate::table_column::ProcessorBaseTableColumn::HV0;
use crate::table_column::ProcessorBaseTableColumn::HV1;
use crate::table_column::ProcessorBaseTableColumn::HV2;
use crate::table_column::ProcessorBaseTableColumn::HV3;
use crate::table_column::ProcessorBaseTableColumn::HV4;
use crate::table_column::ProcessorBaseTableColumn::HV5;
use crate::table_column::ProcessorBaseTableColumn::IB0;
use crate::table_column::ProcessorBaseTableColumn::IB1;
use crate::table_column::ProcessorBaseTableColumn::IB2;
use crate::table_column::ProcessorBaseTableColumn::IB3;
use crate::table_column::ProcessorBaseTableColumn::IB4;
use crate::table_column::ProcessorBaseTableColumn::IB5;
use crate::table_column::ProcessorBaseTableColumn::IB6;
use crate::table_column::ProcessorBaseTableColumn::IP;
use crate::table_column::ProcessorBaseTableColumn::JSD;
use crate::table_column::ProcessorBaseTableColumn::JSO;
use crate::table_column::ProcessorBaseTableColumn::JSP;
use crate::table_column::ProcessorBaseTableColumn::NIA;
use crate::table_column::ProcessorBaseTableColumn::ST0;
use crate::table_column::ProcessorBaseTableColumn::ST1;
use crate::table_column::ProcessorBaseTableColumn::ST10;
use crate::table_column::ProcessorBaseTableColumn::ST11;
use crate::table_column::ProcessorBaseTableColumn::ST12;
use crate::table_column::ProcessorBaseTableColumn::ST13;
use crate::table_column::ProcessorBaseTableColumn::ST14;
use crate::table_column::ProcessorBaseTableColumn::ST15;
use crate::table_column::ProcessorBaseTableColumn::ST2;
use crate::table_column::ProcessorBaseTableColumn::ST3;
use crate::table_column::ProcessorBaseTableColumn::ST4;
use crate::table_column::ProcessorBaseTableColumn::ST5;
use crate::table_column::ProcessorBaseTableColumn::ST6;
use crate::table_column::ProcessorBaseTableColumn::ST7;
use crate::table_column::ProcessorBaseTableColumn::ST8;
use crate::table_column::ProcessorBaseTableColumn::ST9;
use crate::table_column::ProcessorExtTableColumn;
use crate::table_column::ProcessorExtTableColumn::ClockJumpDifferenceLookupServerLogDerivative;
use crate::table_column::ProcessorExtTableColumn::HashDigestEvalArg;
use crate::table_column::ProcessorExtTableColumn::HashInputEvalArg;
use crate::table_column::ProcessorExtTableColumn::InputTableEvalArg;
use crate::table_column::ProcessorExtTableColumn::InstructionLookupClientLogDerivative;
use crate::table_column::ProcessorExtTableColumn::JumpStackTablePermArg;
use crate::table_column::ProcessorExtTableColumn::OpStackTablePermArg;
use crate::table_column::ProcessorExtTableColumn::OutputTableEvalArg;
use crate::table_column::ProcessorExtTableColumn::RamTablePermArg;
use crate::table_column::ProcessorExtTableColumn::SpongeEvalArg;
use crate::table_column::ProcessorExtTableColumn::U32LookupClientLogDerivative;
use crate::AIR;

/// The number of helper variable registers
pub const NUM_HELPER_VARIABLE_REGISTERS: usize = 6;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct ProcessorTable;

impl ProcessorTable {
    /// # Panics
    ///
    /// Panics if the index is out of bounds.
    pub fn op_stack_column_by_index(index: usize) -> ProcessorBaseTableColumn {
        assert!(
            index < OpStackElement::COUNT,
            "Op Stack column index must be in [0, 15], not {index}"
        );

        match index {
            0 => ST0,
            1 => ST1,
            2 => ST2,
            3 => ST3,
            4 => ST4,
            5 => ST5,
            6 => ST6,
            7 => ST7,
            8 => ST8,
            9 => ST9,
            10 => ST10,
            11 => ST11,
            12 => ST12,
            13 => ST13,
            14 => ST14,
            15 => ST15,
            _ => unreachable!(),
        }
    }
}

impl AIR for ProcessorTable {
    type MainColumn = crate::table_column::ProcessorBaseTableColumn;
    type AuxColumn = crate::table_column::ProcessorExtTableColumn;

    fn initial_constraints(
        circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        let constant = |c: u32| circuit_builder.b_constant(c);
        let x_constant = |x| circuit_builder.x_constant(x);
        let challenge = |c| circuit_builder.challenge(c);
        let base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(BaseRow(col.master_base_table_index()))
        };
        let ext_row = |col: ProcessorExtTableColumn| {
            circuit_builder.input(ExtRow(col.master_ext_table_index()))
        };

        let clk_is_0 = base_row(CLK);
        let ip_is_0 = base_row(IP);
        let jsp_is_0 = base_row(JSP);
        let jso_is_0 = base_row(JSO);
        let jsd_is_0 = base_row(JSD);
        let st0_is_0 = base_row(ST0);
        let st1_is_0 = base_row(ST1);
        let st2_is_0 = base_row(ST2);
        let st3_is_0 = base_row(ST3);
        let st4_is_0 = base_row(ST4);
        let st5_is_0 = base_row(ST5);
        let st6_is_0 = base_row(ST6);
        let st7_is_0 = base_row(ST7);
        let st8_is_0 = base_row(ST8);
        let st9_is_0 = base_row(ST9);
        let st10_is_0 = base_row(ST10);
        let op_stack_pointer_is_16 = base_row(OpStackPointer) - constant(16);

        // Compress the program digest using an Evaluation Argument.
        // Lowest index in the digest corresponds to lowest index on the stack.
        let program_digest: [_; Digest::LEN] = [
            base_row(ST11),
            base_row(ST12),
            base_row(ST13),
            base_row(ST14),
            base_row(ST15),
        ];
        let compressed_program_digest = program_digest.into_iter().fold(
            circuit_builder.x_constant(EvalArg::default_initial()),
            |acc, digest_element| {
                acc * challenge(CompressProgramDigestIndeterminate) + digest_element
            },
        );
        let compressed_program_digest_is_expected_program_digest =
            compressed_program_digest - challenge(CompressedProgramDigest);

        // Permutation and Evaluation Arguments with all tables the Processor Table relates to

        // standard input
        let running_evaluation_for_standard_input_is_initialized_correctly =
            ext_row(InputTableEvalArg) - x_constant(EvalArg::default_initial());

        // program table
        let instruction_lookup_indeterminate = challenge(InstructionLookupIndeterminate);
        let instruction_ci_weight = challenge(ProgramInstructionWeight);
        let instruction_nia_weight = challenge(ProgramNextInstructionWeight);
        let compressed_row_for_instruction_lookup =
            instruction_ci_weight * base_row(CI) + instruction_nia_weight * base_row(NIA);
        let instruction_lookup_log_derivative_is_initialized_correctly =
            (ext_row(InstructionLookupClientLogDerivative)
                - x_constant(LookupArg::default_initial()))
                * (instruction_lookup_indeterminate - compressed_row_for_instruction_lookup)
                - constant(1);

        // standard output
        let running_evaluation_for_standard_output_is_initialized_correctly =
            ext_row(OutputTableEvalArg) - x_constant(EvalArg::default_initial());

        let running_product_for_op_stack_table_is_initialized_correctly =
            ext_row(OpStackTablePermArg) - x_constant(PermArg::default_initial());

        // ram table
        let running_product_for_ram_table_is_initialized_correctly =
            ext_row(RamTablePermArg) - x_constant(PermArg::default_initial());

        // jump-stack table
        let jump_stack_indeterminate = challenge(JumpStackIndeterminate);
        let jump_stack_ci_weight = challenge(JumpStackCiWeight);
        // note: `clk`, `jsp`, `jso`, and `jsd` are already constrained to be 0.
        let compressed_row_for_jump_stack_table = jump_stack_ci_weight * base_row(CI);
        let running_product_for_jump_stack_table_is_initialized_correctly =
            ext_row(JumpStackTablePermArg)
                - x_constant(PermArg::default_initial())
                    * (jump_stack_indeterminate - compressed_row_for_jump_stack_table);

        // clock jump difference lookup argument
        // The clock jump difference logarithmic derivative accumulator starts
        // off having accumulated the contribution from the first row.
        // Note that (challenge(ClockJumpDifferenceLookupIndeterminate) - base_row(CLK))
        // collapses to challenge(ClockJumpDifferenceLookupIndeterminate)
        // because base_row(CLK) = 0 is already a constraint.
        let clock_jump_diff_lookup_log_derivative_is_initialized_correctly =
            ext_row(ClockJumpDifferenceLookupServerLogDerivative)
                * challenge(ClockJumpDifferenceLookupIndeterminate)
                - base_row(ClockJumpDifferenceLookupMultiplicity);

        // from processor to hash table
        let hash_selector = base_row(CI) - constant(Instruction::Hash.opcode());
        let hash_deselector = instruction_deselector_single_row(circuit_builder, Instruction::Hash);
        let hash_input_indeterminate = challenge(HashInputIndeterminate);
        // the opStack is guaranteed to be initialized to 0 by virtue of other initial constraints
        let compressed_row = constant(0);
        let running_evaluation_hash_input_has_absorbed_first_row = ext_row(HashInputEvalArg)
            - hash_input_indeterminate * x_constant(EvalArg::default_initial())
            - compressed_row;
        let running_evaluation_hash_input_is_default_initial =
            ext_row(HashInputEvalArg) - x_constant(EvalArg::default_initial());
        let running_evaluation_hash_input_is_initialized_correctly = hash_selector
            * running_evaluation_hash_input_is_default_initial
            + hash_deselector * running_evaluation_hash_input_has_absorbed_first_row;

        // from hash table to processor
        let running_evaluation_hash_digest_is_initialized_correctly =
            ext_row(HashDigestEvalArg) - x_constant(EvalArg::default_initial());

        // Hash Table – Sponge
        let running_evaluation_sponge_absorb_is_initialized_correctly =
            ext_row(SpongeEvalArg) - x_constant(EvalArg::default_initial());

        // u32 table
        let running_sum_log_derivative_for_u32_table_is_initialized_correctly =
            ext_row(U32LookupClientLogDerivative) - x_constant(LookupArg::default_initial());

        vec![
            clk_is_0,
            ip_is_0,
            jsp_is_0,
            jso_is_0,
            jsd_is_0,
            st0_is_0,
            st1_is_0,
            st2_is_0,
            st3_is_0,
            st4_is_0,
            st5_is_0,
            st6_is_0,
            st7_is_0,
            st8_is_0,
            st9_is_0,
            st10_is_0,
            compressed_program_digest_is_expected_program_digest,
            op_stack_pointer_is_16,
            running_evaluation_for_standard_input_is_initialized_correctly,
            instruction_lookup_log_derivative_is_initialized_correctly,
            running_evaluation_for_standard_output_is_initialized_correctly,
            running_product_for_op_stack_table_is_initialized_correctly,
            running_product_for_ram_table_is_initialized_correctly,
            running_product_for_jump_stack_table_is_initialized_correctly,
            clock_jump_diff_lookup_log_derivative_is_initialized_correctly,
            running_evaluation_hash_input_is_initialized_correctly,
            running_evaluation_hash_digest_is_initialized_correctly,
            running_evaluation_sponge_absorb_is_initialized_correctly,
            running_sum_log_derivative_for_u32_table_is_initialized_correctly,
        ]
    }

    fn consistency_constraints(
        circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        let constant = |c: u32| circuit_builder.b_constant(c);
        let base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(BaseRow(col.master_base_table_index()))
        };

        // The composition of instruction bits ib0-ib7 corresponds the current instruction ci.
        let ib_composition = base_row(IB0)
            + constant(1 << 1) * base_row(IB1)
            + constant(1 << 2) * base_row(IB2)
            + constant(1 << 3) * base_row(IB3)
            + constant(1 << 4) * base_row(IB4)
            + constant(1 << 5) * base_row(IB5)
            + constant(1 << 6) * base_row(IB6);
        let ci_corresponds_to_ib0_thru_ib7 = base_row(CI) - ib_composition;

        let ib0_is_bit = base_row(IB0) * (base_row(IB0) - constant(1));
        let ib1_is_bit = base_row(IB1) * (base_row(IB1) - constant(1));
        let ib2_is_bit = base_row(IB2) * (base_row(IB2) - constant(1));
        let ib3_is_bit = base_row(IB3) * (base_row(IB3) - constant(1));
        let ib4_is_bit = base_row(IB4) * (base_row(IB4) - constant(1));
        let ib5_is_bit = base_row(IB5) * (base_row(IB5) - constant(1));
        let ib6_is_bit = base_row(IB6) * (base_row(IB6) - constant(1));
        let is_padding_is_bit = base_row(IsPadding) * (base_row(IsPadding) - constant(1));

        // In padding rows, the clock jump difference lookup multiplicity is 0. The one row
        // exempt from this rule is the row wth CLK == 1: since the memory-like tables don't have
        // an “awareness” of padding rows, they keep looking up clock jump differences of
        // magnitude 1.
        let clock_jump_diff_lookup_multiplicity_is_0_in_padding_rows = base_row(IsPadding)
            * (base_row(CLK) - constant(1))
            * base_row(ClockJumpDifferenceLookupMultiplicity);

        vec![
            ib0_is_bit,
            ib1_is_bit,
            ib2_is_bit,
            ib3_is_bit,
            ib4_is_bit,
            ib5_is_bit,
            ib6_is_bit,
            is_padding_is_bit,
            ci_corresponds_to_ib0_thru_ib7,
            clock_jump_diff_lookup_multiplicity_is_0_in_padding_rows,
        ]
    }

    fn transition_constraints(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let constant = |c: u64| circuit_builder.b_constant(c);
        let curr_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
        };
        let next_base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(NextBaseRow(col.master_base_table_index()))
        };

        // constraints common to all instructions
        let clk_increases_by_1 = next_base_row(CLK) - curr_base_row(CLK) - constant(1);
        let is_padding_is_0_or_does_not_change =
            curr_base_row(IsPadding) * (next_base_row(IsPadding) - curr_base_row(IsPadding));

        let instruction_independent_constraints =
            vec![clk_increases_by_1, is_padding_is_0_or_does_not_change];

        // instruction-specific constraints
        let transition_constraints_for_instruction =
            |instr| transition_constraints_for_instruction(circuit_builder, instr);
        let all_instructions_and_their_transition_constraints =
            ALL_INSTRUCTIONS.map(|instr| (instr, transition_constraints_for_instruction(instr)));
        let deselected_transition_constraints = combine_instruction_constraints_with_deselectors(
            circuit_builder,
            all_instructions_and_their_transition_constraints,
        );

        // if next row is padding row: disable transition constraints, enable padding constraints
        let doubly_deselected_transition_constraints =
            combine_transition_constraints_with_padding_constraints(
                circuit_builder,
                deselected_transition_constraints,
            );

        let table_linking_constraints = vec![
            log_derivative_accumulates_clk_next(circuit_builder),
            log_derivative_for_instruction_lookup_updates_correctly(circuit_builder),
            running_product_for_jump_stack_table_updates_correctly(circuit_builder),
            running_evaluation_hash_input_updates_correctly(circuit_builder),
            running_evaluation_hash_digest_updates_correctly(circuit_builder),
            running_evaluation_sponge_updates_correctly(circuit_builder),
            log_derivative_with_u32_table_updates_correctly(circuit_builder),
        ];

        [
            instruction_independent_constraints,
            doubly_deselected_transition_constraints,
            table_linking_constraints,
        ]
        .concat()
    }

    fn terminal_constraints(
        circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        let main_row =
            |col: Self::MainColumn| circuit_builder.input(BaseRow(col.master_base_table_index()));
        let constant = |c| circuit_builder.b_constant(c);

        let last_ci_is_halt =
            main_row(Self::MainColumn::CI) - constant(Instruction::Halt.opcode_b());

        vec![last_ci_is_halt]
    }
}

/// Instruction-specific transition constraints are combined with deselectors in such a way
/// that arbitrary sets of mutually exclusive combinations are summed, i.e.,
///
/// ```py
/// [ deselector_pop * tc_pop_0 + deselector_push * tc_push_0 + ...,
///   deselector_pop * tc_pop_1 + deselector_push * tc_push_1 + ...,
///   ...,
///   deselector_pop * tc_pop_i + deselector_push * tc_push_i + ...,
///   deselector_pop * 0        + deselector_push * tc_push_{i+1} + ...,
///   ...,
/// ]
/// ```
/// For instructions that have fewer transition constraints than the maximal number of
/// transition constraints among all instructions, the deselector is multiplied with a zero,
/// causing no additional terms in the final sets of combined transition constraint polynomials.
fn combine_instruction_constraints_with_deselectors(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    instr_tc_polys_tuples: [(Instruction, Vec<ConstraintCircuitMonad<DualRowIndicator>>);
        Instruction::COUNT],
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let (all_instructions, all_tc_polys_for_all_instructions): (Vec<_>, Vec<_>) =
        instr_tc_polys_tuples.into_iter().unzip();

    let all_instruction_deselectors = all_instructions
        .into_iter()
        .map(|instr| instruction_deselector_current_row(circuit_builder, instr))
        .collect_vec();

    let max_number_of_constraints = all_tc_polys_for_all_instructions
        .iter()
        .map(|tc_polys_for_instr| tc_polys_for_instr.len())
        .max()
        .unwrap();

    let zero_poly = circuit_builder.b_constant(0);
    let all_tc_polys_for_all_instructions_transposed = (0..max_number_of_constraints)
        .map(|idx| {
            all_tc_polys_for_all_instructions
                .iter()
                .map(|tc_polys_for_instr| tc_polys_for_instr.get(idx).unwrap_or(&zero_poly))
                .collect_vec()
        })
        .collect_vec();

    all_tc_polys_for_all_instructions_transposed
        .into_iter()
        .map(|row| {
            all_instruction_deselectors
                .clone()
                .into_iter()
                .zip(row)
                .map(|(deselector, instruction_tc)| deselector * instruction_tc.to_owned())
                .sum()
        })
        .collect_vec()
}

fn combine_transition_constraints_with_padding_constraints(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    instruction_transition_constraints: Vec<ConstraintCircuitMonad<DualRowIndicator>>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let constant = |c: u64| circuit_builder.b_constant(c);
    let curr_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let next_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };

    let padding_row_transition_constraints = [
        vec![
            next_base_row(IP) - curr_base_row(IP),
            next_base_row(CI) - curr_base_row(CI),
            next_base_row(NIA) - curr_base_row(NIA),
        ],
        instruction_group_keep_jump_stack(circuit_builder),
        instruction_group_keep_op_stack(circuit_builder),
        instruction_group_no_ram(circuit_builder),
        instruction_group_no_io(circuit_builder),
    ]
    .concat();

    let padding_row_deselector = constant(1) - next_base_row(IsPadding);
    let padding_row_selector = next_base_row(IsPadding);

    let max_number_of_constraints = max(
        instruction_transition_constraints.len(),
        padding_row_transition_constraints.len(),
    );

    (0..max_number_of_constraints)
        .map(|idx| {
            let instruction_constraint = instruction_transition_constraints
                .get(idx)
                .unwrap_or(&constant(0))
                .to_owned();
            let padding_constraint = padding_row_transition_constraints
                .get(idx)
                .unwrap_or(&constant(0))
                .to_owned();

            instruction_constraint * padding_row_deselector.clone()
                + padding_constraint * padding_row_selector.clone()
        })
        .collect_vec()
}

fn instruction_group_decompose_arg(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let constant = |c: u32| circuit_builder.b_constant(c);
    let curr_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };

    let hv0_is_a_bit = curr_base_row(HV0) * (curr_base_row(HV0) - constant(1));
    let hv1_is_a_bit = curr_base_row(HV1) * (curr_base_row(HV1) - constant(1));
    let hv2_is_a_bit = curr_base_row(HV2) * (curr_base_row(HV2) - constant(1));
    let hv3_is_a_bit = curr_base_row(HV3) * (curr_base_row(HV3) - constant(1));

    let helper_variables_are_binary_decomposition_of_nia = curr_base_row(NIA)
        - constant(8) * curr_base_row(HV3)
        - constant(4) * curr_base_row(HV2)
        - constant(2) * curr_base_row(HV1)
        - curr_base_row(HV0);

    vec![
        hv0_is_a_bit,
        hv1_is_a_bit,
        hv2_is_a_bit,
        hv3_is_a_bit,
        helper_variables_are_binary_decomposition_of_nia,
    ]
}

/// The permutation argument accumulator with the RAM table does
/// not change, because there is no RAM access.
fn instruction_group_no_ram(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let curr_ext_row = |col: ProcessorExtTableColumn| {
        circuit_builder.input(CurrentExtRow(col.master_ext_table_index()))
    };
    let next_ext_row = |col: ProcessorExtTableColumn| {
        circuit_builder.input(NextExtRow(col.master_ext_table_index()))
    };

    vec![next_ext_row(RamTablePermArg) - curr_ext_row(RamTablePermArg)]
}

fn instruction_group_no_io(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    vec![
        running_evaluation_for_standard_input_remains_unchanged(circuit_builder),
        running_evaluation_for_standard_output_remains_unchanged(circuit_builder),
    ]
}

/// Op Stack height does not change and except for the top n elements,
/// the values remain also.
fn instruction_group_op_stack_remains_except_top_n(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    n: usize,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    assert!(n <= NUM_OP_STACK_REGISTERS);

    let curr_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let next_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };

    let stack = (0..OpStackElement::COUNT)
        .map(ProcessorTable::op_stack_column_by_index)
        .collect_vec();
    let next_stack = stack.iter().map(|&st| next_row(st)).collect_vec();
    let curr_stack = stack.iter().map(|&st| curr_row(st)).collect_vec();

    let compress_stack_except_top_n = |stack: Vec<_>| -> ConstraintCircuitMonad<_> {
        assert_eq!(NUM_OP_STACK_REGISTERS, stack.len());
        let weight = |i| circuit_builder.challenge(stack_weight_by_index(i));
        stack
            .into_iter()
            .enumerate()
            .skip(n)
            .map(|(i, st)| weight(i) * st)
            .sum()
    };

    let all_but_n_top_elements_remain =
        compress_stack_except_top_n(next_stack) - compress_stack_except_top_n(curr_stack);

    let mut constraints = instruction_group_keep_op_stack_height(circuit_builder);
    constraints.push(all_but_n_top_elements_remain);
    constraints
}

/// Op stack does not change, _i.e._, all stack elements persist
fn instruction_group_keep_op_stack(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    instruction_group_op_stack_remains_except_top_n(circuit_builder, 0)
}

/// Op stack *height* does not change, _i.e._, the accumulator for the
/// permutation argument with the op stack table remains the same as does
/// the op stack pointer.
fn instruction_group_keep_op_stack_height(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let op_stack_pointer_curr =
        circuit_builder.input(CurrentBaseRow(OpStackPointer.master_base_table_index()));
    let op_stack_pointer_next =
        circuit_builder.input(NextBaseRow(OpStackPointer.master_base_table_index()));
    let osp_remains_unchanged = op_stack_pointer_next - op_stack_pointer_curr;

    let op_stack_table_perm_arg_curr =
        circuit_builder.input(CurrentExtRow(OpStackTablePermArg.master_ext_table_index()));
    let op_stack_table_perm_arg_next =
        circuit_builder.input(NextExtRow(OpStackTablePermArg.master_ext_table_index()));
    let perm_arg_remains_unchanged = op_stack_table_perm_arg_next - op_stack_table_perm_arg_curr;

    vec![osp_remains_unchanged, perm_arg_remains_unchanged]
}

fn instruction_group_grow_op_stack_and_top_two_elements_unconstrained(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let constant = |c: u32| circuit_builder.b_constant(c);
    let curr_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let next_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };

    vec![
        next_base_row(ST2) - curr_base_row(ST1),
        next_base_row(ST3) - curr_base_row(ST2),
        next_base_row(ST4) - curr_base_row(ST3),
        next_base_row(ST5) - curr_base_row(ST4),
        next_base_row(ST6) - curr_base_row(ST5),
        next_base_row(ST7) - curr_base_row(ST6),
        next_base_row(ST8) - curr_base_row(ST7),
        next_base_row(ST9) - curr_base_row(ST8),
        next_base_row(ST10) - curr_base_row(ST9),
        next_base_row(ST11) - curr_base_row(ST10),
        next_base_row(ST12) - curr_base_row(ST11),
        next_base_row(ST13) - curr_base_row(ST12),
        next_base_row(ST14) - curr_base_row(ST13),
        next_base_row(ST15) - curr_base_row(ST14),
        next_base_row(OpStackPointer) - curr_base_row(OpStackPointer) - constant(1),
        running_product_op_stack_accounts_for_growing_stack_by(circuit_builder, 1),
    ]
}

fn instruction_group_grow_op_stack(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let curr_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let next_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };

    let specific_constraints = vec![next_base_row(ST1) - curr_base_row(ST0)];
    let inherited_constraints =
        instruction_group_grow_op_stack_and_top_two_elements_unconstrained(circuit_builder);

    [specific_constraints, inherited_constraints].concat()
}

fn instruction_group_op_stack_shrinks_and_top_three_elements_unconstrained(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let constant = |c: u32| circuit_builder.b_constant(c);
    let curr_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let next_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };

    vec![
        next_base_row(ST3) - curr_base_row(ST4),
        next_base_row(ST4) - curr_base_row(ST5),
        next_base_row(ST5) - curr_base_row(ST6),
        next_base_row(ST6) - curr_base_row(ST7),
        next_base_row(ST7) - curr_base_row(ST8),
        next_base_row(ST8) - curr_base_row(ST9),
        next_base_row(ST9) - curr_base_row(ST10),
        next_base_row(ST10) - curr_base_row(ST11),
        next_base_row(ST11) - curr_base_row(ST12),
        next_base_row(ST12) - curr_base_row(ST13),
        next_base_row(ST13) - curr_base_row(ST14),
        next_base_row(ST14) - curr_base_row(ST15),
        next_base_row(OpStackPointer) - curr_base_row(OpStackPointer) + constant(1),
        running_product_op_stack_accounts_for_shrinking_stack_by(circuit_builder, 1),
    ]
}

fn instruction_group_binop(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let curr_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let next_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };

    let specific_constraints = vec![
        next_base_row(ST1) - curr_base_row(ST2),
        next_base_row(ST2) - curr_base_row(ST3),
    ];
    let inherited_constraints =
        instruction_group_op_stack_shrinks_and_top_three_elements_unconstrained(circuit_builder);

    [specific_constraints, inherited_constraints].concat()
}

fn instruction_group_shrink_op_stack(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let curr_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let next_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };

    let specific_constraints = vec![next_base_row(ST0) - curr_base_row(ST1)];
    let inherited_constraints = instruction_group_binop(circuit_builder);

    [specific_constraints, inherited_constraints].concat()
}

fn instruction_group_keep_jump_stack(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let curr_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let next_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };

    let jsp_does_not_change = next_base_row(JSP) - curr_base_row(JSP);
    let jso_does_not_change = next_base_row(JSO) - curr_base_row(JSO);
    let jsd_does_not_change = next_base_row(JSD) - curr_base_row(JSD);

    vec![
        jsp_does_not_change,
        jso_does_not_change,
        jsd_does_not_change,
    ]
}

/// Increase the instruction pointer by 1.
fn instruction_group_step_1(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let constant = |c: u32| circuit_builder.b_constant(c);
    let curr_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let next_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };

    let instruction_pointer_increases_by_one = next_base_row(IP) - curr_base_row(IP) - constant(1);
    [
        instruction_group_keep_jump_stack(circuit_builder),
        vec![instruction_pointer_increases_by_one],
    ]
    .concat()
}

/// Increase the instruction pointer by 2.
fn instruction_group_step_2(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let constant = |c: u32| circuit_builder.b_constant(c);
    let curr_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let next_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };

    let instruction_pointer_increases_by_two = next_base_row(IP) - curr_base_row(IP) - constant(2);
    [
        instruction_group_keep_jump_stack(circuit_builder),
        vec![instruction_pointer_increases_by_two],
    ]
    .concat()
}

/// Internal helper function to de-duplicate functionality common between the similar (but
/// different on a type level) functions for construction deselectors.
fn instruction_deselector_common_functionality<II: InputIndicator>(
    circuit_builder: &ConstraintCircuitBuilder<II>,
    instruction: Instruction,
    instruction_bit_polynomials: [ConstraintCircuitMonad<II>; InstructionBit::COUNT],
) -> ConstraintCircuitMonad<II> {
    let one = || circuit_builder.b_constant(1);

    let selector_bits: [_; InstructionBit::COUNT] = [
        instruction.ib(InstructionBit::IB0),
        instruction.ib(InstructionBit::IB1),
        instruction.ib(InstructionBit::IB2),
        instruction.ib(InstructionBit::IB3),
        instruction.ib(InstructionBit::IB4),
        instruction.ib(InstructionBit::IB5),
        instruction.ib(InstructionBit::IB6),
    ];
    let deselector_polynomials = selector_bits.map(|b| one() - circuit_builder.b_constant(b));

    instruction_bit_polynomials
        .into_iter()
        .zip_eq(deselector_polynomials)
        .map(|(instruction_bit_poly, deselector_poly)| instruction_bit_poly - deselector_poly)
        .fold(one(), ConstraintCircuitMonad::mul)
}

/// A polynomial that has no solutions when `ci` is `instruction`.
/// The number of variables in the polynomial corresponds to two rows.
fn instruction_deselector_current_row(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    instruction: Instruction,
) -> ConstraintCircuitMonad<DualRowIndicator> {
    let curr_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };

    let instruction_bit_polynomials = [
        curr_base_row(IB0),
        curr_base_row(IB1),
        curr_base_row(IB2),
        curr_base_row(IB3),
        curr_base_row(IB4),
        curr_base_row(IB5),
        curr_base_row(IB6),
    ];

    instruction_deselector_common_functionality(
        circuit_builder,
        instruction,
        instruction_bit_polynomials,
    )
}

/// A polynomial that has no solutions when `ci_next` is `instruction`.
/// The number of variables in the polynomial corresponds to two rows.
fn instruction_deselector_next_row(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    instruction: Instruction,
) -> ConstraintCircuitMonad<DualRowIndicator> {
    let next_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };

    let instruction_bit_polynomials = [
        next_base_row(IB0),
        next_base_row(IB1),
        next_base_row(IB2),
        next_base_row(IB3),
        next_base_row(IB4),
        next_base_row(IB5),
        next_base_row(IB6),
    ];

    instruction_deselector_common_functionality(
        circuit_builder,
        instruction,
        instruction_bit_polynomials,
    )
}

/// A polynomial that has no solutions when `ci` is `instruction`.
/// The number of variables in the polynomial corresponds to a single row.
fn instruction_deselector_single_row(
    circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    instruction: Instruction,
) -> ConstraintCircuitMonad<SingleRowIndicator> {
    let base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(BaseRow(col.master_base_table_index()))
    };

    let instruction_bit_polynomials = [
        base_row(IB0),
        base_row(IB1),
        base_row(IB2),
        base_row(IB3),
        base_row(IB4),
        base_row(IB5),
        base_row(IB6),
    ];

    instruction_deselector_common_functionality(
        circuit_builder,
        instruction,
        instruction_bit_polynomials,
    )
}

fn instruction_pop(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    [
        instruction_group_step_2(circuit_builder),
        instruction_group_decompose_arg(circuit_builder),
        stack_shrinks_by_any_of(circuit_builder, &NumberOfWords::legal_values()),
        prohibit_any_illegal_number_of_words(circuit_builder),
        instruction_group_no_ram(circuit_builder),
        instruction_group_no_io(circuit_builder),
    ]
    .concat()
}

fn instruction_push(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let curr_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let next_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };

    let specific_constraints = vec![next_base_row(ST0) - curr_base_row(NIA)];
    [
        specific_constraints,
        instruction_group_grow_op_stack(circuit_builder),
        instruction_group_step_2(circuit_builder),
        instruction_group_no_ram(circuit_builder),
        instruction_group_no_io(circuit_builder),
    ]
    .concat()
}

fn instruction_divine(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    [
        instruction_group_step_2(circuit_builder),
        instruction_group_decompose_arg(circuit_builder),
        stack_grows_by_any_of(circuit_builder, &NumberOfWords::legal_values()),
        prohibit_any_illegal_number_of_words(circuit_builder),
        instruction_group_no_ram(circuit_builder),
        instruction_group_no_io(circuit_builder),
    ]
    .concat()
}

fn instruction_dup(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let indicator_poly = |idx| indicator_polynomial(circuit_builder, idx);
    let curr_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let next_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };

    let st_column = ProcessorTable::op_stack_column_by_index;
    let duplicate_element = |i| indicator_poly(i) * (next_row(ST0) - curr_row(st_column(i)));
    let duplicate_indicated_element = (0..OpStackElement::COUNT).map(duplicate_element).sum();

    [
        vec![duplicate_indicated_element],
        instruction_group_decompose_arg(circuit_builder),
        instruction_group_step_2(circuit_builder),
        instruction_group_grow_op_stack(circuit_builder),
        instruction_group_no_ram(circuit_builder),
        instruction_group_no_io(circuit_builder),
    ]
    .concat()
}

fn instruction_swap(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let curr_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let next_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };

    let stack = (0..OpStackElement::COUNT)
        .map(ProcessorTable::op_stack_column_by_index)
        .collect_vec();
    let stack_with_swapped_i = |i| {
        let mut stack = stack.clone();
        stack.swap(0, i);
        stack.into_iter()
    };

    let next_stack = stack.iter().map(|&st| next_row(st)).collect_vec();
    let curr_stack_with_swapped_i = |i| stack_with_swapped_i(i).map(curr_row).collect_vec();
    let compress = |stack: Vec<_>| -> ConstraintCircuitMonad<_> {
        assert_eq!(OpStackElement::COUNT, stack.len());
        let weight = |i| circuit_builder.challenge(stack_weight_by_index(i));
        let enumerated_stack = stack.into_iter().enumerate();
        enumerated_stack.map(|(i, st)| weight(i) * st).sum()
    };

    let next_stack_is_current_stack_with_swapped_i = |i| {
        indicator_polynomial(circuit_builder, i)
            * (compress(next_stack.clone()) - compress(curr_stack_with_swapped_i(i)))
    };
    let next_stack_is_current_stack_with_correct_element_swapped = (0..OpStackElement::COUNT)
        .map(next_stack_is_current_stack_with_swapped_i)
        .sum();

    [
        vec![next_stack_is_current_stack_with_correct_element_swapped],
        instruction_group_decompose_arg(circuit_builder),
        instruction_group_step_2(circuit_builder),
        instruction_group_no_ram(circuit_builder),
        instruction_group_no_io(circuit_builder),
        instruction_group_keep_op_stack_height(circuit_builder),
    ]
    .concat()
}

fn instruction_nop(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    [
        instruction_group_step_1(circuit_builder),
        instruction_group_keep_op_stack(circuit_builder),
        instruction_group_no_ram(circuit_builder),
        instruction_group_no_io(circuit_builder),
    ]
    .concat()
}

fn instruction_skiz(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let constant = |c: u32| circuit_builder.b_constant(c);
    let one = || constant(1);
    let curr_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let next_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };

    let hv0_is_inverse_of_st0 = curr_base_row(HV0) * curr_base_row(ST0) - one();
    let hv0_is_inverse_of_st0_or_hv0_is_0 = hv0_is_inverse_of_st0.clone() * curr_base_row(HV0);
    let hv0_is_inverse_of_st0_or_st0_is_0 = hv0_is_inverse_of_st0 * curr_base_row(ST0);

    // The next instruction nia is decomposed into helper variables hv.
    let nia_decomposes_to_hvs = curr_base_row(NIA)
        - curr_base_row(HV1)
        - constant(1 << 1) * curr_base_row(HV2)
        - constant(1 << 3) * curr_base_row(HV3)
        - constant(1 << 5) * curr_base_row(HV4)
        - constant(1 << 7) * curr_base_row(HV5);

    // If `st0` is non-zero, register `ip` is incremented by 1.
    // If `st0` is 0 and `nia` takes no argument, register `ip` is incremented by 2.
    // If `st0` is 0 and `nia` takes an argument, register `ip` is incremented by 3.
    //
    // The opcodes are constructed such that hv1 == 1 means that nia takes an argument.
    //
    // Written as Disjunctive Normal Form, the constraint can be expressed as:
    // (Register `st0` is 0 or `ip` is incremented by 1), and
    // (`st0` has a multiplicative inverse or `hv1` is 1 or `ip` is incremented by 2), and
    // (`st0` has a multiplicative inverse or `hv1` is 0 or `ip` is incremented by 3).
    let ip_case_1 = (next_base_row(IP) - curr_base_row(IP) - constant(1)) * curr_base_row(ST0);
    let ip_case_2 = (next_base_row(IP) - curr_base_row(IP) - constant(2))
        * (curr_base_row(ST0) * curr_base_row(HV0) - one())
        * (curr_base_row(HV1) - one());
    let ip_case_3 = (next_base_row(IP) - curr_base_row(IP) - constant(3))
        * (curr_base_row(ST0) * curr_base_row(HV0) - one())
        * curr_base_row(HV1);
    let ip_incr_by_1_or_2_or_3 = ip_case_1 + ip_case_2 + ip_case_3;

    let specific_constraints = vec![
        hv0_is_inverse_of_st0_or_hv0_is_0,
        hv0_is_inverse_of_st0_or_st0_is_0,
        nia_decomposes_to_hvs,
        ip_incr_by_1_or_2_or_3,
    ];
    [
        specific_constraints,
        next_instruction_range_check_constraints_for_instruction_skiz(circuit_builder),
        instruction_group_keep_jump_stack(circuit_builder),
        instruction_group_shrink_op_stack(circuit_builder),
        instruction_group_no_ram(circuit_builder),
        instruction_group_no_io(circuit_builder),
    ]
    .concat()
}

fn next_instruction_range_check_constraints_for_instruction_skiz(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let constant = |c: u32| circuit_builder.b_constant(c);
    let curr_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };

    let is_0_or_1 =
        |var: ProcessorBaseTableColumn| curr_base_row(var) * (curr_base_row(var) - constant(1));
    let is_0_or_1_or_2_or_3 = |var: ProcessorBaseTableColumn| {
        curr_base_row(var)
            * (curr_base_row(var) - constant(1))
            * (curr_base_row(var) - constant(2))
            * (curr_base_row(var) - constant(3))
    };

    vec![
        is_0_or_1(HV1),
        is_0_or_1_or_2_or_3(HV2),
        is_0_or_1_or_2_or_3(HV3),
        is_0_or_1_or_2_or_3(HV4),
        is_0_or_1_or_2_or_3(HV5),
    ]
}

fn instruction_call(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let constant = |c: u32| circuit_builder.b_constant(c);
    let curr_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let next_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };

    // The jump stack pointer jsp is incremented by 1.
    let jsp_incr_1 = next_base_row(JSP) - curr_base_row(JSP) - constant(1);

    // The jump's origin jso is set to the current instruction pointer ip plus 2.
    let jso_becomes_ip_plus_2 = next_base_row(JSO) - curr_base_row(IP) - constant(2);

    // The jump's destination jsd is set to the instruction's argument.
    let jsd_becomes_nia = next_base_row(JSD) - curr_base_row(NIA);

    // The instruction pointer ip is set to the instruction's argument.
    let ip_becomes_nia = next_base_row(IP) - curr_base_row(NIA);

    let specific_constraints = vec![
        jsp_incr_1,
        jso_becomes_ip_plus_2,
        jsd_becomes_nia,
        ip_becomes_nia,
    ];
    [
        specific_constraints,
        instruction_group_keep_op_stack(circuit_builder),
        instruction_group_no_ram(circuit_builder),
        instruction_group_no_io(circuit_builder),
    ]
    .concat()
}

fn instruction_return(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let constant = |c: u32| circuit_builder.b_constant(c);
    let curr_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let next_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };

    let jsp_decrements_by_1 = next_base_row(JSP) - curr_base_row(JSP) + constant(1);
    let ip_is_set_to_jso = next_base_row(IP) - curr_base_row(JSO);
    let specific_constraints = vec![jsp_decrements_by_1, ip_is_set_to_jso];

    [
        specific_constraints,
        instruction_group_keep_op_stack(circuit_builder),
        instruction_group_no_ram(circuit_builder),
        instruction_group_no_io(circuit_builder),
    ]
    .concat()
}

fn instruction_recurse(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let curr_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let next_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };

    // The instruction pointer ip is set to the last jump's destination jsd.
    let ip_becomes_jsd = next_base_row(IP) - curr_base_row(JSD);
    let specific_constraints = vec![ip_becomes_jsd];
    [
        specific_constraints,
        instruction_group_keep_jump_stack(circuit_builder),
        instruction_group_keep_op_stack(circuit_builder),
        instruction_group_no_ram(circuit_builder),
        instruction_group_no_io(circuit_builder),
    ]
    .concat()
}

fn instruction_recurse_or_return(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let one = || circuit_builder.b_constant(1);
    let curr_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let next_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };

    // Zero if the ST5 equals ST6. One if they are not equal.
    let st5_eq_st6 = || curr_row(HV0) * (curr_row(ST6) - curr_row(ST5));
    let st5_neq_st6 = || one() - st5_eq_st6();

    let maybe_return = vec![
        // hv0 is inverse-or-zero of the difference of ST6 and ST5.
        st5_neq_st6() * curr_row(HV0),
        st5_neq_st6() * (curr_row(ST6) - curr_row(ST5)),
        st5_neq_st6() * (next_row(IP) - curr_row(JSO)),
        st5_neq_st6() * (next_row(JSP) - curr_row(JSP) + one()),
    ];
    let maybe_recurse = vec![
        // constraints are ordered to line up nicely with group “maybe_return”
        st5_eq_st6() * (next_row(JSO) - curr_row(JSO)),
        st5_eq_st6() * (next_row(JSD) - curr_row(JSD)),
        st5_eq_st6() * (next_row(IP) - curr_row(JSD)),
        st5_eq_st6() * (next_row(JSP) - curr_row(JSP)),
    ];

    // The two constraint groups are mutually exclusive: the stack element is either
    // equal to its successor or not, indicated by `st5_eq_st6` and `st5_neq_st6`.
    // Therefore, it is safe (and sound) to combine the groups into a single set of
    // constraints.
    let constraint_groups = vec![maybe_return, maybe_recurse];
    let specific_constraints =
        combine_mutually_exclusive_constraint_groups(circuit_builder, constraint_groups);

    [
        specific_constraints,
        instruction_group_keep_op_stack(circuit_builder),
        instruction_group_no_ram(circuit_builder),
        instruction_group_no_io(circuit_builder),
    ]
    .concat()
}

fn instruction_assert(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let constant = |c: u32| circuit_builder.b_constant(c);
    let curr_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };

    // The current top of the stack st0 is 1.
    let st_0_is_1 = curr_base_row(ST0) - constant(1);

    let specific_constraints = vec![st_0_is_1];
    [
        specific_constraints,
        instruction_group_step_1(circuit_builder),
        instruction_group_shrink_op_stack(circuit_builder),
        instruction_group_no_ram(circuit_builder),
        instruction_group_no_io(circuit_builder),
    ]
    .concat()
}

fn instruction_halt(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let curr_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let next_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };

    // The instruction executed in the following step is instruction halt.
    let halt_is_followed_by_halt = next_base_row(CI) - curr_base_row(CI);

    let specific_constraints = vec![halt_is_followed_by_halt];
    [
        specific_constraints,
        instruction_group_step_1(circuit_builder),
        instruction_group_keep_op_stack(circuit_builder),
        instruction_group_no_ram(circuit_builder),
        instruction_group_no_io(circuit_builder),
    ]
    .concat()
}

fn instruction_read_mem(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    [
        instruction_group_step_2(circuit_builder),
        instruction_group_decompose_arg(circuit_builder),
        read_from_ram_any_of(circuit_builder, &NumberOfWords::legal_values()),
        prohibit_any_illegal_number_of_words(circuit_builder),
        instruction_group_no_io(circuit_builder),
    ]
    .concat()
}

fn instruction_write_mem(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    [
        instruction_group_step_2(circuit_builder),
        instruction_group_decompose_arg(circuit_builder),
        write_to_ram_any_of(circuit_builder, &NumberOfWords::legal_values()),
        prohibit_any_illegal_number_of_words(circuit_builder),
        instruction_group_no_io(circuit_builder),
    ]
    .concat()
}

/// Two Evaluation Arguments with the Hash Table guarantee correct transition.
fn instruction_hash(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let constant = |c: u32| circuit_builder.b_constant(c);
    let curr_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let next_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };

    let op_stack_shrinks_by_5_and_top_5_unconstrained = vec![
        next_base_row(ST5) - curr_base_row(ST10),
        next_base_row(ST6) - curr_base_row(ST11),
        next_base_row(ST7) - curr_base_row(ST12),
        next_base_row(ST8) - curr_base_row(ST13),
        next_base_row(ST9) - curr_base_row(ST14),
        next_base_row(ST10) - curr_base_row(ST15),
        next_base_row(OpStackPointer) - curr_base_row(OpStackPointer) + constant(5),
        running_product_op_stack_accounts_for_shrinking_stack_by(circuit_builder, 5),
    ];

    [
        instruction_group_step_1(circuit_builder),
        op_stack_shrinks_by_5_and_top_5_unconstrained,
        instruction_group_no_ram(circuit_builder),
        instruction_group_no_io(circuit_builder),
    ]
    .concat()
}

fn instruction_merkle_step(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    [
        instruction_merkle_step_shared_constraints(circuit_builder),
        instruction_group_op_stack_remains_except_top_n(circuit_builder, 6),
        instruction_group_no_ram(circuit_builder),
    ]
    .concat()
}

fn instruction_merkle_step_mem(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let constant = |c: u32| circuit_builder.b_constant(c);
    let stack_weight = |i| circuit_builder.challenge(stack_weight_by_index(i));
    let curr = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let next = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };

    let ram_pointers = [0, 1, 2, 3, 4].map(|i| curr(ST7) + constant(i));
    let ram_read_destinations = [HV0, HV1, HV2, HV3, HV4].map(curr);
    let read_from_ram_to_hvs =
        read_from_ram_to(circuit_builder, ram_pointers, ram_read_destinations);

    let st6_does_not_change = next(ST6) - curr(ST6);
    let st7_increments_by_5 = next(ST7) - curr(ST7) - constant(5);
    let st6_and_st7_update_correctly =
        stack_weight(6) * st6_does_not_change + stack_weight(7) * st7_increments_by_5;

    [
        vec![st6_and_st7_update_correctly, read_from_ram_to_hvs],
        instruction_merkle_step_shared_constraints(circuit_builder),
        instruction_group_op_stack_remains_except_top_n(circuit_builder, 8),
    ]
    .concat()
}

/// Recall that in a Merkle tree, the indices of left (respectively right)
/// leaves have least-significant bit 0 (respectively 1).
///
/// Two Evaluation Arguments with the Hash Table guarantee correct transition of
/// stack elements ST0 through ST4.
fn instruction_merkle_step_shared_constraints(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let constant = |c: u32| circuit_builder.b_constant(c);
    let one = || constant(1);
    let curr = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let next = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };

    let hv5_is_0_or_1 = curr(HV5) * (curr(HV5) - one());
    let new_st5_is_previous_st5_div_2 = constant(2) * next(ST5) + curr(HV5) - curr(ST5);
    let update_merkle_tree_node_index = vec![hv5_is_0_or_1, new_st5_is_previous_st5_div_2];

    [
        update_merkle_tree_node_index,
        instruction_group_step_1(circuit_builder),
        instruction_group_no_io(circuit_builder),
    ]
    .concat()
}

fn instruction_assert_vector(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let curr_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };

    let specific_constraints = vec![
        curr_base_row(ST5) - curr_base_row(ST0),
        curr_base_row(ST6) - curr_base_row(ST1),
        curr_base_row(ST7) - curr_base_row(ST2),
        curr_base_row(ST8) - curr_base_row(ST3),
        curr_base_row(ST9) - curr_base_row(ST4),
    ];
    [
        specific_constraints,
        instruction_group_step_1(circuit_builder),
        constraints_for_shrinking_stack_by(circuit_builder, 5),
        instruction_group_no_ram(circuit_builder),
        instruction_group_no_io(circuit_builder),
    ]
    .concat()
}

fn instruction_sponge_init(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    [
        instruction_group_step_1(circuit_builder),
        instruction_group_keep_op_stack(circuit_builder),
        instruction_group_no_ram(circuit_builder),
        instruction_group_no_io(circuit_builder),
    ]
    .concat()
}

fn instruction_sponge_absorb(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    [
        instruction_group_step_1(circuit_builder),
        constraints_for_shrinking_stack_by(circuit_builder, 10),
        instruction_group_no_ram(circuit_builder),
        instruction_group_no_io(circuit_builder),
    ]
    .concat()
}

fn instruction_sponge_absorb_mem(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let curr_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let next_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };
    let constant = |c| circuit_builder.b_constant(c);

    let increment_ram_pointer =
        next_base_row(ST0) - curr_base_row(ST0) - constant(tip5::RATE as u32);

    [
        vec![increment_ram_pointer],
        instruction_group_step_1(circuit_builder),
        instruction_group_op_stack_remains_except_top_n(circuit_builder, 5),
        instruction_group_no_io(circuit_builder),
    ]
    .concat()
}

fn instruction_sponge_squeeze(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    [
        instruction_group_step_1(circuit_builder),
        constraints_for_growing_stack_by(circuit_builder, 10),
        instruction_group_no_ram(circuit_builder),
        instruction_group_no_io(circuit_builder),
    ]
    .concat()
}

fn instruction_add(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let curr_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let next_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };

    let specific_constraints = vec![next_base_row(ST0) - curr_base_row(ST0) - curr_base_row(ST1)];
    [
        specific_constraints,
        instruction_group_step_1(circuit_builder),
        instruction_group_binop(circuit_builder),
        instruction_group_no_ram(circuit_builder),
        instruction_group_no_io(circuit_builder),
    ]
    .concat()
}

fn instruction_addi(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let curr_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let next_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };

    let specific_constraints = vec![next_base_row(ST0) - curr_base_row(ST0) - curr_base_row(NIA)];
    [
        specific_constraints,
        instruction_group_step_2(circuit_builder),
        instruction_group_op_stack_remains_except_top_n(circuit_builder, 1),
        instruction_group_no_ram(circuit_builder),
        instruction_group_no_io(circuit_builder),
    ]
    .concat()
}

fn instruction_mul(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let curr_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let next_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };

    let specific_constraints = vec![next_base_row(ST0) - curr_base_row(ST0) * curr_base_row(ST1)];
    [
        specific_constraints,
        instruction_group_step_1(circuit_builder),
        instruction_group_binop(circuit_builder),
        instruction_group_no_ram(circuit_builder),
        instruction_group_no_io(circuit_builder),
    ]
    .concat()
}

fn instruction_invert(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let constant = |c: u32| circuit_builder.b_constant(c);
    let one = || constant(1);
    let curr_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let next_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };

    let specific_constraints = vec![next_base_row(ST0) * curr_base_row(ST0) - one()];
    [
        specific_constraints,
        instruction_group_step_1(circuit_builder),
        instruction_group_op_stack_remains_except_top_n(circuit_builder, 1),
        instruction_group_no_ram(circuit_builder),
        instruction_group_no_io(circuit_builder),
    ]
    .concat()
}

fn instruction_eq(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let constant = |c: u32| circuit_builder.b_constant(c);
    let one = || constant(1);
    let curr_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let next_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };

    let st0_eq_st1 = || one() - curr_base_row(HV0) * (curr_base_row(ST1) - curr_base_row(ST0));

    // Helper variable hv0 is the inverse-or-zero of the difference of the stack's two top-most
    // elements: `hv0·(1 - hv0·(st1 - st0))`
    let hv0_is_inverse_of_diff_or_hv0_is_0 = curr_base_row(HV0) * st0_eq_st1();

    // Helper variable hv0 is the inverse-or-zero of the difference of the stack's two
    // top-most elements: `(st1 - st0)·(1 - hv0·(st1 - st0))`
    let hv0_is_inverse_of_diff_or_diff_is_0 =
        (curr_base_row(ST1) - curr_base_row(ST0)) * st0_eq_st1();

    // The new top of the stack is 1 if the difference between the stack's two top-most
    // elements is not invertible, 0 otherwise: `st0' - (1 - hv0·(st1 - st0))`
    let st0_becomes_1_if_diff_is_not_invertible = next_base_row(ST0) - st0_eq_st1();

    let specific_constraints = vec![
        hv0_is_inverse_of_diff_or_hv0_is_0,
        hv0_is_inverse_of_diff_or_diff_is_0,
        st0_becomes_1_if_diff_is_not_invertible,
    ];
    [
        specific_constraints,
        instruction_group_step_1(circuit_builder),
        instruction_group_binop(circuit_builder),
        instruction_group_no_ram(circuit_builder),
        instruction_group_no_io(circuit_builder),
    ]
    .concat()
}

fn instruction_split(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let constant = |c: u64| circuit_builder.b_constant(c);
    let one = || constant(1);
    let curr_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let next_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };

    // The top of the stack is decomposed as 32-bit chunks into the stack's top-most elements:
    // st0 - (2^32·st0' + st1') = 0$
    let st0_decomposes_to_two_32_bit_chunks =
        curr_base_row(ST0) - (constant(1 << 32) * next_base_row(ST1) + next_base_row(ST0));

    // Helper variable `hv0` = 0 if either
    // 1. `hv0` is the difference between (2^32 - 1) and the high 32 bits (`st0'`), or
    // 1. the low 32 bits (`st1'`) are 0.
    //
    // st1'·(hv0·(st0' - (2^32 - 1)) - 1)
    //   lo·(hv0·(hi - 0xffff_ffff)) - 1)
    let hv0_holds_inverse_of_chunk_difference_or_low_bits_are_0 = {
        let hv0 = curr_base_row(HV0);
        let hi = next_base_row(ST1);
        let lo = next_base_row(ST0);
        let ffff_ffff = constant(0xffff_ffff);

        lo * (hv0 * (hi - ffff_ffff) - one())
    };

    let specific_constraints = vec![
        st0_decomposes_to_two_32_bit_chunks,
        hv0_holds_inverse_of_chunk_difference_or_low_bits_are_0,
    ];
    [
        specific_constraints,
        instruction_group_grow_op_stack_and_top_two_elements_unconstrained(circuit_builder),
        instruction_group_step_1(circuit_builder),
        instruction_group_no_ram(circuit_builder),
        instruction_group_no_io(circuit_builder),
    ]
    .concat()
}

fn instruction_lt(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    [
        instruction_group_step_1(circuit_builder),
        instruction_group_binop(circuit_builder),
        instruction_group_no_ram(circuit_builder),
        instruction_group_no_io(circuit_builder),
    ]
    .concat()
}

fn instruction_and(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    [
        instruction_group_step_1(circuit_builder),
        instruction_group_binop(circuit_builder),
        instruction_group_no_ram(circuit_builder),
        instruction_group_no_io(circuit_builder),
    ]
    .concat()
}

fn instruction_xor(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    [
        instruction_group_step_1(circuit_builder),
        instruction_group_binop(circuit_builder),
        instruction_group_no_ram(circuit_builder),
        instruction_group_no_io(circuit_builder),
    ]
    .concat()
}

fn instruction_log_2_floor(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    [
        instruction_group_step_1(circuit_builder),
        instruction_group_op_stack_remains_except_top_n(circuit_builder, 1),
        instruction_group_no_ram(circuit_builder),
        instruction_group_no_io(circuit_builder),
    ]
    .concat()
}

fn instruction_pow(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    [
        instruction_group_step_1(circuit_builder),
        instruction_group_binop(circuit_builder),
        instruction_group_no_ram(circuit_builder),
        instruction_group_no_io(circuit_builder),
    ]
    .concat()
}

fn instruction_div_mod(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let curr_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let next_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };

    // `n == d·q + r` means `st0 - st1·st1' - st0'`
    let numerator_is_quotient_times_denominator_plus_remainder =
        curr_base_row(ST0) - curr_base_row(ST1) * next_base_row(ST1) - next_base_row(ST0);

    let specific_constraints = vec![numerator_is_quotient_times_denominator_plus_remainder];
    [
        specific_constraints,
        instruction_group_step_1(circuit_builder),
        instruction_group_op_stack_remains_except_top_n(circuit_builder, 2),
        instruction_group_no_ram(circuit_builder),
        instruction_group_no_io(circuit_builder),
    ]
    .concat()
}

fn instruction_pop_count(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    [
        instruction_group_step_1(circuit_builder),
        instruction_group_op_stack_remains_except_top_n(circuit_builder, 1),
        instruction_group_no_ram(circuit_builder),
        instruction_group_no_io(circuit_builder),
    ]
    .concat()
}

fn instruction_xx_add(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let curr_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let next_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };

    let st0_becomes_st0_plus_st3 = next_base_row(ST0) - curr_base_row(ST0) - curr_base_row(ST3);
    let st1_becomes_st1_plus_st4 = next_base_row(ST1) - curr_base_row(ST1) - curr_base_row(ST4);
    let st2_becomes_st2_plus_st5 = next_base_row(ST2) - curr_base_row(ST2) - curr_base_row(ST5);
    let specific_constraints = vec![
        st0_becomes_st0_plus_st3,
        st1_becomes_st1_plus_st4,
        st2_becomes_st2_plus_st5,
    ];

    [
        specific_constraints,
        constraints_for_shrinking_stack_by_3_and_top_3_unconstrained(circuit_builder),
        instruction_group_step_1(circuit_builder),
        instruction_group_no_ram(circuit_builder),
        instruction_group_no_io(circuit_builder),
    ]
    .concat()
}

fn instruction_xx_mul(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let curr_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let next_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };

    let [x0, x1, x2, y0, y1, y2] = [ST0, ST1, ST2, ST3, ST4, ST5].map(curr_base_row);
    let [c0, c1, c2] = xx_product([x0, x1, x2], [y0, y1, y2]);

    let specific_constraints = vec![
        next_base_row(ST0) - c0,
        next_base_row(ST1) - c1,
        next_base_row(ST2) - c2,
    ];
    [
        specific_constraints,
        constraints_for_shrinking_stack_by_3_and_top_3_unconstrained(circuit_builder),
        instruction_group_step_1(circuit_builder),
        instruction_group_no_ram(circuit_builder),
        instruction_group_no_io(circuit_builder),
    ]
    .concat()
}

fn instruction_xinv(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let constant = |c: u64| circuit_builder.b_constant(c);
    let curr_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let next_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };

    let first_coefficient_of_product_of_element_and_inverse_is_1 = curr_base_row(ST0)
        * next_base_row(ST0)
        - curr_base_row(ST2) * next_base_row(ST1)
        - curr_base_row(ST1) * next_base_row(ST2)
        - constant(1);

    let second_coefficient_of_product_of_element_and_inverse_is_0 =
        curr_base_row(ST1) * next_base_row(ST0) + curr_base_row(ST0) * next_base_row(ST1)
            - curr_base_row(ST2) * next_base_row(ST2)
            + curr_base_row(ST2) * next_base_row(ST1)
            + curr_base_row(ST1) * next_base_row(ST2);

    let third_coefficient_of_product_of_element_and_inverse_is_0 = curr_base_row(ST2)
        * next_base_row(ST0)
        + curr_base_row(ST1) * next_base_row(ST1)
        + curr_base_row(ST0) * next_base_row(ST2)
        + curr_base_row(ST2) * next_base_row(ST2);

    let specific_constraints = vec![
        first_coefficient_of_product_of_element_and_inverse_is_1,
        second_coefficient_of_product_of_element_and_inverse_is_0,
        third_coefficient_of_product_of_element_and_inverse_is_0,
    ];
    [
        specific_constraints,
        instruction_group_op_stack_remains_except_top_n(circuit_builder, 3),
        instruction_group_step_1(circuit_builder),
        instruction_group_no_ram(circuit_builder),
        instruction_group_no_io(circuit_builder),
    ]
    .concat()
}

fn instruction_xb_mul(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let curr_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let next_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };

    let [x, y0, y1, y2] = [ST0, ST1, ST2, ST3].map(curr_base_row);
    let [c0, c1, c2] = xb_product([y0, y1, y2], x);

    let specific_constraints = vec![
        next_base_row(ST0) - c0,
        next_base_row(ST1) - c1,
        next_base_row(ST2) - c2,
    ];
    [
        specific_constraints,
        instruction_group_op_stack_shrinks_and_top_three_elements_unconstrained(circuit_builder),
        instruction_group_step_1(circuit_builder),
        instruction_group_no_ram(circuit_builder),
        instruction_group_no_io(circuit_builder),
    ]
    .concat()
}

fn instruction_read_io(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let constraint_groups_for_legal_arguments = NumberOfWords::legal_values()
        .map(|n| grow_stack_by_n_and_read_n_symbols_from_input(circuit_builder, n))
        .to_vec();
    let read_any_legal_number_of_words = combine_mutually_exclusive_constraint_groups(
        circuit_builder,
        constraint_groups_for_legal_arguments,
    );

    [
        instruction_group_step_2(circuit_builder),
        instruction_group_decompose_arg(circuit_builder),
        read_any_legal_number_of_words,
        prohibit_any_illegal_number_of_words(circuit_builder),
        instruction_group_no_ram(circuit_builder),
        vec![running_evaluation_for_standard_output_remains_unchanged(
            circuit_builder,
        )],
    ]
    .concat()
}

fn instruction_write_io(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let constraint_groups_for_legal_arguments = NumberOfWords::legal_values()
        .map(|n| shrink_stack_by_n_and_write_n_symbols_to_output(circuit_builder, n))
        .to_vec();
    let write_any_of_1_through_5_elements = combine_mutually_exclusive_constraint_groups(
        circuit_builder,
        constraint_groups_for_legal_arguments,
    );

    [
        instruction_group_step_2(circuit_builder),
        instruction_group_decompose_arg(circuit_builder),
        write_any_of_1_through_5_elements,
        prohibit_any_illegal_number_of_words(circuit_builder),
        instruction_group_no_ram(circuit_builder),
        vec![running_evaluation_for_standard_input_remains_unchanged(
            circuit_builder,
        )],
    ]
    .concat()
}

/// Update the accumulator for the Permutation Argument with the RAM table in
/// accordance with reading a bunch of words from the indicated ram pointers to
/// the indicated destination registers.
///
/// Does not constrain the op stack by default.[^stack] For that, see:
/// [`read_from_ram_any_of`].
///
/// [^stack]: Op stack registers used in arguments will be constrained.
fn read_from_ram_to<const N: usize>(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ram_pointers: [ConstraintCircuitMonad<DualRowIndicator>; N],
    destinations: [ConstraintCircuitMonad<DualRowIndicator>; N],
) -> ConstraintCircuitMonad<DualRowIndicator> {
    let curr_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let curr_ext_row = |col: ProcessorExtTableColumn| {
        circuit_builder.input(CurrentExtRow(col.master_ext_table_index()))
    };
    let next_ext_row = |col: ProcessorExtTableColumn| {
        circuit_builder.input(NextExtRow(col.master_ext_table_index()))
    };
    let challenge = |c: ChallengeId| circuit_builder.challenge(c);
    let constant = |bfe| circuit_builder.b_constant(bfe);

    let compress_row = |(ram_pointer, destination)| {
        curr_base_row(CLK) * challenge(RamClkWeight)
            + constant(table::ram::INSTRUCTION_TYPE_READ) * challenge(RamInstructionTypeWeight)
            + ram_pointer * challenge(RamPointerWeight)
            + destination * challenge(RamValueWeight)
    };

    let factor = ram_pointers
        .into_iter()
        .zip(destinations)
        .map(compress_row)
        .map(|compressed_row| challenge(RamIndeterminate) - compressed_row)
        .reduce(|l, r| l * r)
        .unwrap_or_else(|| constant(bfe!(1)));
    curr_ext_row(RamTablePermArg) * factor - next_ext_row(RamTablePermArg)
}

fn xx_product<Indicator: InputIndicator>(
    [x_0, x_1, x_2]: [ConstraintCircuitMonad<Indicator>; EXTENSION_DEGREE],
    [y_0, y_1, y_2]: [ConstraintCircuitMonad<Indicator>; EXTENSION_DEGREE],
) -> [ConstraintCircuitMonad<Indicator>; EXTENSION_DEGREE] {
    let z0 = x_0.clone() * y_0.clone();
    let z1 = x_1.clone() * y_0.clone() + x_0.clone() * y_1.clone();
    let z2 = x_2.clone() * y_0 + x_1.clone() * y_1.clone() + x_0 * y_2.clone();
    let z3 = x_2.clone() * y_1 + x_1 * y_2.clone();
    let z4 = x_2 * y_2;

    // reduce modulo x³ - x + 1
    [z0 - z3.clone(), z1 - z4.clone() + z3, z2 + z4]
}

fn xb_product<Indicator: InputIndicator>(
    [x_0, x_1, x_2]: [ConstraintCircuitMonad<Indicator>; EXTENSION_DEGREE],
    y: ConstraintCircuitMonad<Indicator>,
) -> [ConstraintCircuitMonad<Indicator>; EXTENSION_DEGREE] {
    let z0 = x_0 * y.clone();
    let z1 = x_1 * y.clone();
    let z2 = x_2 * y;
    [z0, z1, z2]
}

fn update_dotstep_accumulator(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    accumulator_indices: [ProcessorBaseTableColumn; EXTENSION_DEGREE],
    difference: [ConstraintCircuitMonad<DualRowIndicator>; EXTENSION_DEGREE],
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let curr_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let next_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };
    let curr = accumulator_indices.map(curr_base_row);
    let next = accumulator_indices.map(next_base_row);
    izip!(curr, next, difference)
        .map(|(c, n, d)| n - c - d)
        .collect()
}

fn instruction_xx_dot_step(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let curr_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let next_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };
    let constant = |c| circuit_builder.b_constant(c);

    let increment_ram_pointer_st0 = next_base_row(ST0) - curr_base_row(ST0) - constant(3);
    let increment_ram_pointer_st1 = next_base_row(ST1) - curr_base_row(ST1) - constant(3);

    let rhs_ptr0 = curr_base_row(ST0);
    let rhs_ptr1 = rhs_ptr0.clone() + constant(1);
    let rhs_ptr2 = rhs_ptr0.clone() + constant(2);
    let lhs_ptr0 = curr_base_row(ST1);
    let lhs_ptr1 = lhs_ptr0.clone() + constant(1);
    let lhs_ptr2 = lhs_ptr0.clone() + constant(2);
    let ram_read_sources = [rhs_ptr0, rhs_ptr1, rhs_ptr2, lhs_ptr0, lhs_ptr1, lhs_ptr2];
    let ram_read_destinations = [HV0, HV1, HV2, HV3, HV4, HV5].map(curr_base_row);
    let read_two_xfes_from_ram =
        read_from_ram_to(circuit_builder, ram_read_sources, ram_read_destinations);

    let ram_pointer_constraints = vec![
        increment_ram_pointer_st0,
        increment_ram_pointer_st1,
        read_two_xfes_from_ram,
    ];

    let [hv0, hv1, hv2, hv3, hv4, hv5] = [HV0, HV1, HV2, HV3, HV4, HV5].map(curr_base_row);
    let hv_product = xx_product([hv0, hv1, hv2], [hv3, hv4, hv5]);

    [
        ram_pointer_constraints,
        update_dotstep_accumulator(circuit_builder, [ST2, ST3, ST4], hv_product),
        instruction_group_step_1(circuit_builder),
        instruction_group_no_io(circuit_builder),
        instruction_group_op_stack_remains_except_top_n(circuit_builder, 5),
    ]
    .concat()
}

fn instruction_xb_dot_step(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let curr_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let next_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };
    let constant = |c| circuit_builder.b_constant(c);

    let increment_ram_pointer_st0 = next_base_row(ST0) - curr_base_row(ST0) - constant(1);
    let increment_ram_pointer_st1 = next_base_row(ST1) - curr_base_row(ST1) - constant(3);

    let rhs_ptr0 = curr_base_row(ST0);
    let lhs_ptr0 = curr_base_row(ST1);
    let lhs_ptr1 = lhs_ptr0.clone() + constant(1);
    let lhs_ptr2 = lhs_ptr0.clone() + constant(2);
    let ram_read_sources = [rhs_ptr0, lhs_ptr0, lhs_ptr1, lhs_ptr2];
    let ram_read_destinations = [HV0, HV1, HV2, HV3].map(curr_base_row);
    let read_bfe_and_xfe_from_ram =
        read_from_ram_to(circuit_builder, ram_read_sources, ram_read_destinations);

    let ram_pointer_constraints = vec![
        increment_ram_pointer_st0,
        increment_ram_pointer_st1,
        read_bfe_and_xfe_from_ram,
    ];

    let [hv0, hv1, hv2, hv3] = [HV0, HV1, HV2, HV3].map(curr_base_row);
    let hv_product = xb_product([hv1, hv2, hv3], hv0);

    [
        ram_pointer_constraints,
        update_dotstep_accumulator(circuit_builder, [ST2, ST3, ST4], hv_product),
        instruction_group_step_1(circuit_builder),
        instruction_group_no_io(circuit_builder),
        instruction_group_op_stack_remains_except_top_n(circuit_builder, 5),
    ]
    .concat()
}

#[doc(hidden)] // allows testing in different crate
pub fn transition_constraints_for_instruction(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    instruction: Instruction,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    match instruction {
        Instruction::Pop(_) => instruction_pop(circuit_builder),
        Instruction::Push(_) => instruction_push(circuit_builder),
        Instruction::Divine(_) => instruction_divine(circuit_builder),
        Instruction::Dup(_) => instruction_dup(circuit_builder),
        Instruction::Swap(_) => instruction_swap(circuit_builder),
        Instruction::Halt => instruction_halt(circuit_builder),
        Instruction::Nop => instruction_nop(circuit_builder),
        Instruction::Skiz => instruction_skiz(circuit_builder),
        Instruction::Call(_) => instruction_call(circuit_builder),
        Instruction::Return => instruction_return(circuit_builder),
        Instruction::Recurse => instruction_recurse(circuit_builder),
        Instruction::RecurseOrReturn => instruction_recurse_or_return(circuit_builder),
        Instruction::Assert => instruction_assert(circuit_builder),
        Instruction::ReadMem(_) => instruction_read_mem(circuit_builder),
        Instruction::WriteMem(_) => instruction_write_mem(circuit_builder),
        Instruction::Hash => instruction_hash(circuit_builder),
        Instruction::AssertVector => instruction_assert_vector(circuit_builder),
        Instruction::SpongeInit => instruction_sponge_init(circuit_builder),
        Instruction::SpongeAbsorb => instruction_sponge_absorb(circuit_builder),
        Instruction::SpongeAbsorbMem => instruction_sponge_absorb_mem(circuit_builder),
        Instruction::SpongeSqueeze => instruction_sponge_squeeze(circuit_builder),
        Instruction::Add => instruction_add(circuit_builder),
        Instruction::AddI(_) => instruction_addi(circuit_builder),
        Instruction::Mul => instruction_mul(circuit_builder),
        Instruction::Invert => instruction_invert(circuit_builder),
        Instruction::Eq => instruction_eq(circuit_builder),
        Instruction::Split => instruction_split(circuit_builder),
        Instruction::Lt => instruction_lt(circuit_builder),
        Instruction::And => instruction_and(circuit_builder),
        Instruction::Xor => instruction_xor(circuit_builder),
        Instruction::Log2Floor => instruction_log_2_floor(circuit_builder),
        Instruction::Pow => instruction_pow(circuit_builder),
        Instruction::DivMod => instruction_div_mod(circuit_builder),
        Instruction::PopCount => instruction_pop_count(circuit_builder),
        Instruction::XxAdd => instruction_xx_add(circuit_builder),
        Instruction::XxMul => instruction_xx_mul(circuit_builder),
        Instruction::XInvert => instruction_xinv(circuit_builder),
        Instruction::XbMul => instruction_xb_mul(circuit_builder),
        Instruction::ReadIo(_) => instruction_read_io(circuit_builder),
        Instruction::WriteIo(_) => instruction_write_io(circuit_builder),
        Instruction::MerkleStep => instruction_merkle_step(circuit_builder),
        Instruction::MerkleStepMem => instruction_merkle_step_mem(circuit_builder),
        Instruction::XxDotStep => instruction_xx_dot_step(circuit_builder),
        Instruction::XbDotStep => instruction_xb_dot_step(circuit_builder),
    }
}

/// Constrains instruction argument `nia` such that 0 < nia <= 5.
fn prohibit_any_illegal_number_of_words(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    vec![NumberOfWords::illegal_values()
        .map(|n| indicator_polynomial(circuit_builder, n))
        .into_iter()
        .sum()]
}

fn log_derivative_accumulates_clk_next(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> ConstraintCircuitMonad<DualRowIndicator> {
    let challenge = |c: ChallengeId| circuit_builder.challenge(c);
    let next_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };
    let curr_ext_row = |col: ProcessorExtTableColumn| {
        circuit_builder.input(CurrentExtRow(col.master_ext_table_index()))
    };
    let next_ext_row = |col: ProcessorExtTableColumn| {
        circuit_builder.input(NextExtRow(col.master_ext_table_index()))
    };

    (next_ext_row(ClockJumpDifferenceLookupServerLogDerivative)
        - curr_ext_row(ClockJumpDifferenceLookupServerLogDerivative))
        * (challenge(ClockJumpDifferenceLookupIndeterminate) - next_base_row(CLK))
        - next_base_row(ClockJumpDifferenceLookupMultiplicity)
}

fn running_evaluation_for_standard_input_remains_unchanged(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> ConstraintCircuitMonad<DualRowIndicator> {
    let curr_ext_row = |col: ProcessorExtTableColumn| {
        circuit_builder.input(CurrentExtRow(col.master_ext_table_index()))
    };
    let next_ext_row = |col: ProcessorExtTableColumn| {
        circuit_builder.input(NextExtRow(col.master_ext_table_index()))
    };

    next_ext_row(InputTableEvalArg) - curr_ext_row(InputTableEvalArg)
}

fn running_evaluation_for_standard_output_remains_unchanged(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> ConstraintCircuitMonad<DualRowIndicator> {
    let curr_ext_row = |col: ProcessorExtTableColumn| {
        circuit_builder.input(CurrentExtRow(col.master_ext_table_index()))
    };
    let next_ext_row = |col: ProcessorExtTableColumn| {
        circuit_builder.input(NextExtRow(col.master_ext_table_index()))
    };

    next_ext_row(OutputTableEvalArg) - curr_ext_row(OutputTableEvalArg)
}

fn grow_stack_by_n_and_read_n_symbols_from_input(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    n: usize,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let indeterminate = || circuit_builder.challenge(StandardInputIndeterminate);
    let next_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };
    let curr_ext_row = |col: ProcessorExtTableColumn| {
        circuit_builder.input(CurrentExtRow(col.master_ext_table_index()))
    };
    let next_ext_row = |col: ProcessorExtTableColumn| {
        circuit_builder.input(NextExtRow(col.master_ext_table_index()))
    };

    let mut running_evaluation = curr_ext_row(InputTableEvalArg);
    for i in (0..n).rev() {
        let stack_element = ProcessorTable::op_stack_column_by_index(i);
        running_evaluation = indeterminate() * running_evaluation + next_base_row(stack_element);
    }
    let running_evaluation_update = next_ext_row(InputTableEvalArg) - running_evaluation;
    let conditional_running_evaluation_update =
        indicator_polynomial(circuit_builder, n) * running_evaluation_update;

    let mut constraints = conditional_constraints_for_growing_stack_by(circuit_builder, n);
    constraints.push(conditional_running_evaluation_update);
    constraints
}

fn shrink_stack_by_n_and_write_n_symbols_to_output(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    n: usize,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let indeterminate = || circuit_builder.challenge(StandardOutputIndeterminate);
    let curr_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let curr_ext_row = |col: ProcessorExtTableColumn| {
        circuit_builder.input(CurrentExtRow(col.master_ext_table_index()))
    };
    let next_ext_row = |col: ProcessorExtTableColumn| {
        circuit_builder.input(NextExtRow(col.master_ext_table_index()))
    };

    let mut running_evaluation = curr_ext_row(OutputTableEvalArg);
    for i in 0..n {
        let stack_element = ProcessorTable::op_stack_column_by_index(i);
        running_evaluation = indeterminate() * running_evaluation + curr_base_row(stack_element);
    }
    let running_evaluation_update = next_ext_row(OutputTableEvalArg) - running_evaluation;
    let conditional_running_evaluation_update =
        indicator_polynomial(circuit_builder, n) * running_evaluation_update;

    let mut constraints = conditional_constraints_for_shrinking_stack_by(circuit_builder, n);
    constraints.push(conditional_running_evaluation_update);
    constraints
}

fn log_derivative_for_instruction_lookup_updates_correctly(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> ConstraintCircuitMonad<DualRowIndicator> {
    let one = || circuit_builder.b_constant(1);
    let challenge = |c: ChallengeId| circuit_builder.challenge(c);
    let next_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };
    let curr_ext_row = |col: ProcessorExtTableColumn| {
        circuit_builder.input(CurrentExtRow(col.master_ext_table_index()))
    };
    let next_ext_row = |col: ProcessorExtTableColumn| {
        circuit_builder.input(NextExtRow(col.master_ext_table_index()))
    };

    let compressed_row = challenge(ProgramAddressWeight) * next_base_row(IP)
        + challenge(ProgramInstructionWeight) * next_base_row(CI)
        + challenge(ProgramNextInstructionWeight) * next_base_row(NIA);
    let log_derivative_updates = (next_ext_row(InstructionLookupClientLogDerivative)
        - curr_ext_row(InstructionLookupClientLogDerivative))
        * (challenge(InstructionLookupIndeterminate) - compressed_row)
        - one();
    let log_derivative_remains = next_ext_row(InstructionLookupClientLogDerivative)
        - curr_ext_row(InstructionLookupClientLogDerivative);

    (one() - next_base_row(IsPadding)) * log_derivative_updates
        + next_base_row(IsPadding) * log_derivative_remains
}

fn constraints_for_shrinking_stack_by_3_and_top_3_unconstrained(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let constant = |c: u64| circuit_builder.b_constant(c);
    let curr_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let next_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };

    vec![
        next_base_row(ST3) - curr_base_row(ST6),
        next_base_row(ST4) - curr_base_row(ST7),
        next_base_row(ST5) - curr_base_row(ST8),
        next_base_row(ST6) - curr_base_row(ST9),
        next_base_row(ST7) - curr_base_row(ST10),
        next_base_row(ST8) - curr_base_row(ST11),
        next_base_row(ST9) - curr_base_row(ST12),
        next_base_row(ST10) - curr_base_row(ST13),
        next_base_row(ST11) - curr_base_row(ST14),
        next_base_row(ST12) - curr_base_row(ST15),
        next_base_row(OpStackPointer) - curr_base_row(OpStackPointer) + constant(3),
        running_product_op_stack_accounts_for_shrinking_stack_by(circuit_builder, 3),
    ]
}

fn stack_shrinks_by_any_of(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    shrinkages: &[usize],
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let all_constraints_for_all_shrinkages = shrinkages
        .iter()
        .map(|&n| conditional_constraints_for_shrinking_stack_by(circuit_builder, n))
        .collect_vec();

    combine_mutually_exclusive_constraint_groups(
        circuit_builder,
        all_constraints_for_all_shrinkages,
    )
}

fn stack_grows_by_any_of(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    growths: &[usize],
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let all_constraints_for_all_growths = growths
        .iter()
        .map(|&n| conditional_constraints_for_growing_stack_by(circuit_builder, n))
        .collect_vec();

    combine_mutually_exclusive_constraint_groups(circuit_builder, all_constraints_for_all_growths)
}

/// Reduces the number of constraints by summing mutually exclusive constraints. The mutual
/// exclusion is due to the conditional nature of the constraints, which has to be guaranteed by
/// the caller.
///
/// For example, the constraints for shrinking the stack by 2, 3, and 4 elements are:
///
/// ```markdown
/// |            shrink by 2 |            shrink by 3 |            shrink by 4 |
/// |-----------------------:|-----------------------:|-----------------------:|
/// |     ind_2·(st0' - st2) |     ind_3·(st0' - st3) |     ind_4·(st0' - st4) |
/// |     ind_2·(st1' - st3) |     ind_3·(st1' - st4) |     ind_4·(st1' - st5) |
/// |                      … |                      … |                      … |
/// |   ind_2·(st11' - st13) |   ind_3·(st11' - st14) |   ind_4·(st11' - st15) |
/// |   ind_2·(st12' - st14) |   ind_3·(st12' - st15) | ind_4·(osp' - osp + 4) |
/// |   ind_2·(st13' - st15) | ind_3·(osp' - osp + 3) | ind_4·(rp' - rp·fac_4) |
/// | ind_2·(osp' - osp + 2) | ind_3·(rp' - rp·fac_3) |                        |
/// | ind_2·(rp' - rp·fac_2) |                        |                        |
/// ```
///
/// This method sums these constraints “per row”. That is, the resulting constraints are:
///
/// ```markdown
/// |                                                  shrink by 2 or 3 or 4 |
/// |-----------------------------------------------------------------------:|
/// |           ind_2·(st0' - st2) + ind_3·(st0' - st3) + ind_4·(st0' - st4) |
/// |           ind_2·(st1' - st3) + ind_3·(st1' - st4) + ind_4·(st1' - st5) |
/// |                                                                      … |
/// |     ind_2·(st11' - st13) + ind_3·(st11' - st14) + ind_4·(st11' - st15) |
/// |   ind_2·(st12' - st14) + ind_3·(st12' - st15) + ind_4·(osp' - osp + 4) |
/// | ind_2·(st13' - st15) + ind_3·(osp' - osp + 3) + ind_4·(rp' - rp·fac_4) |
/// |                        ind_2·(osp' - osp + 2) + ind_3·(rp' - rp·fac_3) |
/// |                                                 ind_2·(rp' - rp·fac_2) |
/// ```
///
/// Syntax in above example:
/// - `ind_n` is the [indicator polynomial](indicator_polynomial) for `n`
/// - `osp` is the [op stack pointer](OpStackPointer)
/// - `rp` is the running product for the permutation argument
/// - `fac_n` is the factor for the running product
fn combine_mutually_exclusive_constraint_groups(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    all_constraint_groups: Vec<Vec<ConstraintCircuitMonad<DualRowIndicator>>>,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let constraint_group_lengths = all_constraint_groups.iter().map(|x| x.len());
    let num_constraints = constraint_group_lengths.max().unwrap_or(0);

    let zero_constraint = || circuit_builder.b_constant(0);
    let mut combined_constraints = vec![];
    for i in 0..num_constraints {
        let combined_constraint = all_constraint_groups
            .iter()
            .filter_map(|constraint_group| constraint_group.get(i))
            .fold(zero_constraint(), |acc, summand| acc + summand.clone());
        combined_constraints.push(combined_constraint);
    }
    combined_constraints
}

fn constraints_for_shrinking_stack_by(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    n: usize,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let constant = |c: usize| circuit_builder.b_constant(u64::try_from(c).unwrap());
    let curr_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let next_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };

    let stack = || (0..OpStackElement::COUNT).map(ProcessorTable::op_stack_column_by_index);
    let new_stack = stack().dropping_back(n).map(next_row).collect_vec();
    let old_stack_with_top_n_removed = stack().skip(n).map(curr_row).collect_vec();

    let compress = |stack: Vec<_>| -> ConstraintCircuitMonad<_> {
        assert_eq!(OpStackElement::COUNT - n, stack.len());
        let weight = |i| circuit_builder.challenge(stack_weight_by_index(i));
        let enumerated_stack = stack.into_iter().enumerate();
        enumerated_stack.map(|(i, st)| weight(i) * st).sum()
    };
    let compressed_new_stack = compress(new_stack);
    let compressed_old_stack = compress(old_stack_with_top_n_removed);

    let op_stack_pointer_shrinks_by_n =
        next_row(OpStackPointer) - curr_row(OpStackPointer) + constant(n);
    let new_stack_is_old_stack_with_top_n_removed = compressed_new_stack - compressed_old_stack;

    vec![
        op_stack_pointer_shrinks_by_n,
        new_stack_is_old_stack_with_top_n_removed,
        running_product_op_stack_accounts_for_shrinking_stack_by(circuit_builder, n),
    ]
}

fn constraints_for_growing_stack_by(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    n: usize,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let constant = |c: usize| circuit_builder.b_constant(u32::try_from(c).unwrap());
    let curr_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let next_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };

    let stack = || (0..OpStackElement::COUNT).map(ProcessorTable::op_stack_column_by_index);
    let new_stack = stack().skip(n).map(next_row).collect_vec();
    let old_stack_with_top_n_added = stack().map(curr_row).dropping_back(n).collect_vec();

    let compress = |stack: Vec<_>| -> ConstraintCircuitMonad<_> {
        assert_eq!(OpStackElement::COUNT - n, stack.len());
        let weight = |i| circuit_builder.challenge(stack_weight_by_index(i));
        let enumerated_stack = stack.into_iter().enumerate();
        enumerated_stack.map(|(i, st)| weight(i) * st).sum()
    };
    let compressed_new_stack = compress(new_stack);
    let compressed_old_stack = compress(old_stack_with_top_n_added);

    let op_stack_pointer_grows_by_n =
        next_row(OpStackPointer) - curr_row(OpStackPointer) - constant(n);
    let new_stack_is_old_stack_with_top_n_added = compressed_new_stack - compressed_old_stack;

    vec![
        op_stack_pointer_grows_by_n,
        new_stack_is_old_stack_with_top_n_added,
        running_product_op_stack_accounts_for_growing_stack_by(circuit_builder, n),
    ]
}

fn conditional_constraints_for_shrinking_stack_by(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    n: usize,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    constraints_for_shrinking_stack_by(circuit_builder, n)
        .into_iter()
        .map(|constraint| indicator_polynomial(circuit_builder, n) * constraint)
        .collect()
}

fn conditional_constraints_for_growing_stack_by(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    n: usize,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    constraints_for_growing_stack_by(circuit_builder, n)
        .into_iter()
        .map(|constraint| indicator_polynomial(circuit_builder, n) * constraint)
        .collect()
}

fn running_product_op_stack_accounts_for_growing_stack_by(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    n: usize,
) -> ConstraintCircuitMonad<DualRowIndicator> {
    let constant = |c: u32| circuit_builder.b_constant(c);
    let curr_ext_row = |col: ProcessorExtTableColumn| {
        circuit_builder.input(CurrentExtRow(col.master_ext_table_index()))
    };
    let next_ext_row = |col: ProcessorExtTableColumn| {
        circuit_builder.input(NextExtRow(col.master_ext_table_index()))
    };
    let single_grow_factor = |op_stack_pointer_offset| {
        single_factor_for_permutation_argument_with_op_stack_table(
            circuit_builder,
            CurrentBaseRow,
            op_stack_pointer_offset,
        )
    };

    let mut factor = constant(1);
    for op_stack_pointer_offset in 0..n {
        factor = factor * single_grow_factor(op_stack_pointer_offset);
    }

    next_ext_row(OpStackTablePermArg) - curr_ext_row(OpStackTablePermArg) * factor
}

fn running_product_op_stack_accounts_for_shrinking_stack_by(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    n: usize,
) -> ConstraintCircuitMonad<DualRowIndicator> {
    let constant = |c: u32| circuit_builder.b_constant(c);
    let curr_ext_row = |col: ProcessorExtTableColumn| {
        circuit_builder.input(CurrentExtRow(col.master_ext_table_index()))
    };
    let next_ext_row = |col: ProcessorExtTableColumn| {
        circuit_builder.input(NextExtRow(col.master_ext_table_index()))
    };
    let single_shrink_factor = |op_stack_pointer_offset| {
        single_factor_for_permutation_argument_with_op_stack_table(
            circuit_builder,
            NextBaseRow,
            op_stack_pointer_offset,
        )
    };

    let mut factor = constant(1);
    for op_stack_pointer_offset in 0..n {
        factor = factor * single_shrink_factor(op_stack_pointer_offset);
    }

    next_ext_row(OpStackTablePermArg) - curr_ext_row(OpStackTablePermArg) * factor
}

fn single_factor_for_permutation_argument_with_op_stack_table(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    row_with_shorter_stack_indicator: fn(usize) -> DualRowIndicator,
    op_stack_pointer_offset: usize,
) -> ConstraintCircuitMonad<DualRowIndicator> {
    let constant = |c: u32| circuit_builder.b_constant(c);
    let challenge = |c: ChallengeId| circuit_builder.challenge(c);
    let curr_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let row_with_shorter_stack = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(row_with_shorter_stack_indicator(
            col.master_base_table_index(),
        ))
    };

    let max_stack_element_index = OpStackElement::COUNT - 1;
    let stack_element_index = max_stack_element_index - op_stack_pointer_offset;
    let stack_element = ProcessorTable::op_stack_column_by_index(stack_element_index);
    let underflow_element = row_with_shorter_stack(stack_element);

    let op_stack_pointer = row_with_shorter_stack(OpStackPointer);
    let offset = constant(op_stack_pointer_offset as u32);
    let offset_op_stack_pointer = op_stack_pointer + offset;

    let compressed_row = challenge(OpStackClkWeight) * curr_base_row(CLK)
        + challenge(OpStackIb1Weight) * curr_base_row(IB1)
        + challenge(OpStackPointerWeight) * offset_op_stack_pointer
        + challenge(OpStackFirstUnderflowElementWeight) * underflow_element;
    challenge(OpStackIndeterminate) - compressed_row
}

/// Build constraints for popping `n` elements from the top of the stack and
/// writing them to RAM. The reciprocal of [`read_from_ram_any_of`].
fn write_to_ram_any_of(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    number_of_words: &[usize],
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let all_constraint_groups = number_of_words
        .iter()
        .map(|&n| conditional_constraints_for_writing_n_elements_to_ram(circuit_builder, n))
        .collect_vec();
    combine_mutually_exclusive_constraint_groups(circuit_builder, all_constraint_groups)
}

/// Build constraints for reading `n` elements from RAM and putting them on top
/// of the stack. The reciprocal of [`write_to_ram_any_of`].
///
/// To constrain RAM reads with more flexible target locations, see
/// [`read_from_ram_to`].
fn read_from_ram_any_of(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    number_of_words: &[usize],
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let all_constraint_groups = number_of_words
        .iter()
        .map(|&n| conditional_constraints_for_reading_n_elements_from_ram(circuit_builder, n))
        .collect_vec();
    combine_mutually_exclusive_constraint_groups(circuit_builder, all_constraint_groups)
}

fn conditional_constraints_for_writing_n_elements_to_ram(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    n: usize,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    shrink_stack_by_n_and_write_n_elements_to_ram(circuit_builder, n)
        .into_iter()
        .map(|constraint| indicator_polynomial(circuit_builder, n) * constraint)
        .collect()
}

fn conditional_constraints_for_reading_n_elements_from_ram(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    n: usize,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    grow_stack_by_n_and_read_n_elements_from_ram(circuit_builder, n)
        .into_iter()
        .map(|constraint| indicator_polynomial(circuit_builder, n) * constraint)
        .collect()
}

fn shrink_stack_by_n_and_write_n_elements_to_ram(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    n: usize,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let constant = |c: usize| circuit_builder.b_constant(u32::try_from(c).unwrap());
    let curr_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let next_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };

    let op_stack_pointer_shrinks_by_n =
        next_base_row(OpStackPointer) - curr_base_row(OpStackPointer) + constant(n);
    let ram_pointer_grows_by_n = next_base_row(ST0) - curr_base_row(ST0) - constant(n);

    let mut constraints = vec![
        op_stack_pointer_shrinks_by_n,
        ram_pointer_grows_by_n,
        running_product_op_stack_accounts_for_shrinking_stack_by(circuit_builder, n),
        running_product_ram_accounts_for_writing_n_elements(circuit_builder, n),
    ];

    let num_ram_pointers = 1;
    for i in n + num_ram_pointers..OpStackElement::COUNT {
        let curr_stack_element = ProcessorTable::op_stack_column_by_index(i);
        let next_stack_element = ProcessorTable::op_stack_column_by_index(i - n);
        let element_i_is_shifted_by_n =
            next_base_row(next_stack_element) - curr_base_row(curr_stack_element);
        constraints.push(element_i_is_shifted_by_n);
    }
    constraints
}

fn grow_stack_by_n_and_read_n_elements_from_ram(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    n: usize,
) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
    let constant = |c: usize| circuit_builder.b_constant(u64::try_from(c).unwrap());
    let curr_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let next_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };

    let op_stack_pointer_grows_by_n =
        next_base_row(OpStackPointer) - curr_base_row(OpStackPointer) - constant(n);
    let ram_pointer_shrinks_by_n = next_base_row(ST0) - curr_base_row(ST0) + constant(n);

    let mut constraints = vec![
        op_stack_pointer_grows_by_n,
        ram_pointer_shrinks_by_n,
        running_product_op_stack_accounts_for_growing_stack_by(circuit_builder, n),
        running_product_ram_accounts_for_reading_n_elements(circuit_builder, n),
    ];

    let num_ram_pointers = 1;
    for i in num_ram_pointers..OpStackElement::COUNT - n {
        let curr_stack_element = ProcessorTable::op_stack_column_by_index(i);
        let next_stack_element = ProcessorTable::op_stack_column_by_index(i + n);
        let element_i_is_shifted_by_n =
            next_base_row(next_stack_element) - curr_base_row(curr_stack_element);
        constraints.push(element_i_is_shifted_by_n);
    }
    constraints
}

fn running_product_ram_accounts_for_writing_n_elements(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    n: usize,
) -> ConstraintCircuitMonad<DualRowIndicator> {
    let constant = |c: u32| circuit_builder.b_constant(c);
    let curr_ext_row = |col: ProcessorExtTableColumn| {
        circuit_builder.input(CurrentExtRow(col.master_ext_table_index()))
    };
    let next_ext_row = |col: ProcessorExtTableColumn| {
        circuit_builder.input(NextExtRow(col.master_ext_table_index()))
    };
    let single_write_factor = |ram_pointer_offset| {
        single_factor_for_permutation_argument_with_ram_table(
            circuit_builder,
            CurrentBaseRow,
            table::ram::INSTRUCTION_TYPE_WRITE,
            ram_pointer_offset,
        )
    };

    let mut factor = constant(1);
    for ram_pointer_offset in 0..n {
        factor = factor * single_write_factor(ram_pointer_offset);
    }

    next_ext_row(RamTablePermArg) - curr_ext_row(RamTablePermArg) * factor
}

fn running_product_ram_accounts_for_reading_n_elements(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    n: usize,
) -> ConstraintCircuitMonad<DualRowIndicator> {
    let constant = |c: u32| circuit_builder.b_constant(c);
    let curr_ext_row = |col: ProcessorExtTableColumn| {
        circuit_builder.input(CurrentExtRow(col.master_ext_table_index()))
    };
    let next_ext_row = |col: ProcessorExtTableColumn| {
        circuit_builder.input(NextExtRow(col.master_ext_table_index()))
    };
    let single_read_factor = |ram_pointer_offset| {
        single_factor_for_permutation_argument_with_ram_table(
            circuit_builder,
            NextBaseRow,
            table::ram::INSTRUCTION_TYPE_READ,
            ram_pointer_offset,
        )
    };

    let mut factor = constant(1);
    for ram_pointer_offset in 0..n {
        factor = factor * single_read_factor(ram_pointer_offset);
    }

    next_ext_row(RamTablePermArg) - curr_ext_row(RamTablePermArg) * factor
}

fn single_factor_for_permutation_argument_with_ram_table(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    row_with_longer_stack_indicator: fn(usize) -> DualRowIndicator,
    instruction_type: BFieldElement,
    ram_pointer_offset: usize,
) -> ConstraintCircuitMonad<DualRowIndicator> {
    let constant = |c: u32| circuit_builder.b_constant(c);
    let b_constant = |c| circuit_builder.b_constant(c);
    let challenge = |c: ChallengeId| circuit_builder.challenge(c);
    let curr_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let row_with_longer_stack = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(row_with_longer_stack_indicator(
            col.master_base_table_index(),
        ))
    };

    let num_ram_pointers = 1;
    let ram_value_index = ram_pointer_offset + num_ram_pointers;
    let ram_value_column = ProcessorTable::op_stack_column_by_index(ram_value_index);
    let ram_value = row_with_longer_stack(ram_value_column);

    let additional_offset = match instruction_type {
        table::ram::INSTRUCTION_TYPE_READ => 1,
        table::ram::INSTRUCTION_TYPE_WRITE => 0,
        _ => panic!("Invalid instruction type"),
    };

    let ram_pointer = row_with_longer_stack(ST0);
    let offset = constant(additional_offset + ram_pointer_offset as u32);
    let offset_ram_pointer = ram_pointer + offset;

    let compressed_row = curr_base_row(CLK) * challenge(RamClkWeight)
        + b_constant(instruction_type) * challenge(RamInstructionTypeWeight)
        + offset_ram_pointer * challenge(RamPointerWeight)
        + ram_value * challenge(RamValueWeight);
    challenge(RamIndeterminate) - compressed_row
}

fn running_product_for_jump_stack_table_updates_correctly(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> ConstraintCircuitMonad<DualRowIndicator> {
    let challenge = |c: ChallengeId| circuit_builder.challenge(c);
    let next_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };
    let curr_ext_row = |col: ProcessorExtTableColumn| {
        circuit_builder.input(CurrentExtRow(col.master_ext_table_index()))
    };
    let next_ext_row = |col: ProcessorExtTableColumn| {
        circuit_builder.input(NextExtRow(col.master_ext_table_index()))
    };

    let compressed_row = challenge(JumpStackClkWeight) * next_base_row(CLK)
        + challenge(JumpStackCiWeight) * next_base_row(CI)
        + challenge(JumpStackJspWeight) * next_base_row(JSP)
        + challenge(JumpStackJsoWeight) * next_base_row(JSO)
        + challenge(JumpStackJsdWeight) * next_base_row(JSD);

    next_ext_row(JumpStackTablePermArg)
        - curr_ext_row(JumpStackTablePermArg) * (challenge(JumpStackIndeterminate) - compressed_row)
}

/// Deal with instructions `hash` and `merkle_step`. The registers from which
/// the preimage is loaded differs between the two instructions:
/// 1. `hash` always loads the stack's 10 top elements,
/// 1. `merkle_step` loads the stack's 5 top elements and helper variables 0
///    through 4. The order of those two quintuplets depends on helper variable
///    hv5.
///
/// The Hash Table does not “know” about instruction `merkle_step`.
///
/// Note that using `next_row` might be confusing at first glance; See the
/// [specification](https://triton-vm.org/spec/processor-table.html).
fn running_evaluation_hash_input_updates_correctly(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> ConstraintCircuitMonad<DualRowIndicator> {
    let constant = |c: u32| circuit_builder.b_constant(c);
    let one = || constant(1);
    let challenge = |c: ChallengeId| circuit_builder.challenge(c);
    let next_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };
    let curr_ext_row = |col: ProcessorExtTableColumn| {
        circuit_builder.input(CurrentExtRow(col.master_ext_table_index()))
    };
    let next_ext_row = |col: ProcessorExtTableColumn| {
        circuit_builder.input(NextExtRow(col.master_ext_table_index()))
    };

    let hash_deselector = instruction_deselector_next_row(circuit_builder, Instruction::Hash);
    let merkle_step_deselector =
        instruction_deselector_next_row(circuit_builder, Instruction::MerkleStep);
    let merkle_step_mem_deselector =
        instruction_deselector_next_row(circuit_builder, Instruction::MerkleStepMem);
    let hash_and_merkle_step_selector = (next_base_row(CI) - constant(Instruction::Hash.opcode()))
        * (next_base_row(CI) - constant(Instruction::MerkleStep.opcode()))
        * (next_base_row(CI) - constant(Instruction::MerkleStepMem.opcode()));

    let weights = [
        StackWeight0,
        StackWeight1,
        StackWeight2,
        StackWeight3,
        StackWeight4,
        StackWeight5,
        StackWeight6,
        StackWeight7,
        StackWeight8,
        StackWeight9,
    ]
    .map(challenge);

    // hash
    let state_for_hash = [ST0, ST1, ST2, ST3, ST4, ST5, ST6, ST7, ST8, ST9].map(next_base_row);
    let compressed_hash_row = weights
        .iter()
        .zip_eq(state_for_hash)
        .map(|(weight, state)| weight.clone() * state)
        .sum();

    // merkle step
    let is_left_sibling = || next_base_row(HV5);
    let is_right_sibling = || one() - next_base_row(HV5);
    let merkle_step_state_element =
        |l, r| is_right_sibling() * next_base_row(l) + is_left_sibling() * next_base_row(r);
    let state_for_merkle_step = [
        merkle_step_state_element(ST0, HV0),
        merkle_step_state_element(ST1, HV1),
        merkle_step_state_element(ST2, HV2),
        merkle_step_state_element(ST3, HV3),
        merkle_step_state_element(ST4, HV4),
        merkle_step_state_element(HV0, ST0),
        merkle_step_state_element(HV1, ST1),
        merkle_step_state_element(HV2, ST2),
        merkle_step_state_element(HV3, ST3),
        merkle_step_state_element(HV4, ST4),
    ];
    let compressed_merkle_step_row = weights
        .into_iter()
        .zip_eq(state_for_merkle_step)
        .map(|(weight, state)| weight * state)
        .sum::<ConstraintCircuitMonad<_>>();

    let running_evaluation_updates_with = |compressed_row| {
        next_ext_row(HashInputEvalArg)
            - challenge(HashInputIndeterminate) * curr_ext_row(HashInputEvalArg)
            - compressed_row
    };
    let running_evaluation_remains =
        next_ext_row(HashInputEvalArg) - curr_ext_row(HashInputEvalArg);

    hash_and_merkle_step_selector * running_evaluation_remains
        + hash_deselector * running_evaluation_updates_with(compressed_hash_row)
        + merkle_step_deselector
            * running_evaluation_updates_with(compressed_merkle_step_row.clone())
        + merkle_step_mem_deselector * running_evaluation_updates_with(compressed_merkle_step_row)
}

fn running_evaluation_hash_digest_updates_correctly(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> ConstraintCircuitMonad<DualRowIndicator> {
    let constant = |c: u32| circuit_builder.b_constant(c);
    let challenge = |c: ChallengeId| circuit_builder.challenge(c);
    let curr_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let next_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };
    let curr_ext_row = |col: ProcessorExtTableColumn| {
        circuit_builder.input(CurrentExtRow(col.master_ext_table_index()))
    };
    let next_ext_row = |col: ProcessorExtTableColumn| {
        circuit_builder.input(NextExtRow(col.master_ext_table_index()))
    };

    let hash_deselector = instruction_deselector_current_row(circuit_builder, Instruction::Hash);
    let merkle_step_deselector =
        instruction_deselector_current_row(circuit_builder, Instruction::MerkleStep);
    let merkle_step_mem_deselector =
        instruction_deselector_current_row(circuit_builder, Instruction::MerkleStepMem);
    let hash_and_merkle_step_selector = (curr_base_row(CI) - constant(Instruction::Hash.opcode()))
        * (curr_base_row(CI) - constant(Instruction::MerkleStep.opcode()))
        * (curr_base_row(CI) - constant(Instruction::MerkleStepMem.opcode()));

    let weights = [
        StackWeight0,
        StackWeight1,
        StackWeight2,
        StackWeight3,
        StackWeight4,
    ]
    .map(challenge);
    let state = [ST0, ST1, ST2, ST3, ST4].map(next_base_row);
    let compressed_row = weights
        .into_iter()
        .zip_eq(state)
        .map(|(weight, state)| weight * state)
        .sum();

    let running_evaluation_updates = next_ext_row(HashDigestEvalArg)
        - challenge(HashDigestIndeterminate) * curr_ext_row(HashDigestEvalArg)
        - compressed_row;
    let running_evaluation_remains =
        next_ext_row(HashDigestEvalArg) - curr_ext_row(HashDigestEvalArg);

    hash_and_merkle_step_selector * running_evaluation_remains
        + (hash_deselector + merkle_step_deselector + merkle_step_mem_deselector)
            * running_evaluation_updates
}

fn running_evaluation_sponge_updates_correctly(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> ConstraintCircuitMonad<DualRowIndicator> {
    let constant = |c: u32| circuit_builder.b_constant(c);
    let challenge = |c: ChallengeId| circuit_builder.challenge(c);
    let curr_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let next_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };
    let curr_ext_row = |col: ProcessorExtTableColumn| {
        circuit_builder.input(CurrentExtRow(col.master_ext_table_index()))
    };
    let next_ext_row = |col: ProcessorExtTableColumn| {
        circuit_builder.input(NextExtRow(col.master_ext_table_index()))
    };

    let sponge_init_deselector =
        instruction_deselector_current_row(circuit_builder, Instruction::SpongeInit);
    let sponge_absorb_deselector =
        instruction_deselector_current_row(circuit_builder, Instruction::SpongeAbsorb);
    let sponge_absorb_mem_deselector =
        instruction_deselector_current_row(circuit_builder, Instruction::SpongeAbsorbMem);
    let sponge_squeeze_deselector =
        instruction_deselector_current_row(circuit_builder, Instruction::SpongeSqueeze);

    let sponge_instruction_selector = (curr_base_row(CI)
        - constant(Instruction::SpongeInit.opcode()))
        * (curr_base_row(CI) - constant(Instruction::SpongeAbsorb.opcode()))
        * (curr_base_row(CI) - constant(Instruction::SpongeAbsorbMem.opcode()))
        * (curr_base_row(CI) - constant(Instruction::SpongeSqueeze.opcode()));

    let weighted_sum = |state| {
        let weights = [
            StackWeight0,
            StackWeight1,
            StackWeight2,
            StackWeight3,
            StackWeight4,
            StackWeight5,
            StackWeight6,
            StackWeight7,
            StackWeight8,
            StackWeight9,
        ];
        let weights = weights.map(challenge).into_iter();
        weights.zip_eq(state).map(|(w, st)| w * st).sum()
    };

    let state = [ST0, ST1, ST2, ST3, ST4, ST5, ST6, ST7, ST8, ST9];
    let compressed_row_current = weighted_sum(state.map(curr_base_row));
    let compressed_row_next = weighted_sum(state.map(next_base_row));

    // Use domain-specific knowledge: the compressed row (i.e., random linear sum)
    // of the initial Sponge state is 0.
    let running_evaluation_updates_for_sponge_init = next_ext_row(SpongeEvalArg)
        - challenge(SpongeIndeterminate) * curr_ext_row(SpongeEvalArg)
        - challenge(HashCIWeight) * curr_base_row(CI);
    let running_evaluation_updates_for_absorb =
        running_evaluation_updates_for_sponge_init.clone() - compressed_row_current;
    let running_evaluation_updates_for_squeeze =
        running_evaluation_updates_for_sponge_init.clone() - compressed_row_next;
    let running_evaluation_remains = next_ext_row(SpongeEvalArg) - curr_ext_row(SpongeEvalArg);

    // `sponge_absorb_mem`
    let stack_elements = [ST1, ST2, ST3, ST4].map(next_base_row);
    let hv_elements = [HV0, HV1, HV2, HV3, HV4, HV5].map(curr_base_row);
    let absorb_mem_elements = stack_elements.into_iter().chain(hv_elements);
    let absorb_mem_elements = absorb_mem_elements.collect_vec().try_into().unwrap();
    let compressed_row_absorb_mem = weighted_sum(absorb_mem_elements);
    let running_evaluation_updates_for_absorb_mem = next_ext_row(SpongeEvalArg)
        - challenge(SpongeIndeterminate) * curr_ext_row(SpongeEvalArg)
        - challenge(HashCIWeight) * constant(Instruction::SpongeAbsorb.opcode())
        - compressed_row_absorb_mem;

    sponge_instruction_selector * running_evaluation_remains
        + sponge_init_deselector * running_evaluation_updates_for_sponge_init
        + sponge_absorb_deselector * running_evaluation_updates_for_absorb
        + sponge_absorb_mem_deselector * running_evaluation_updates_for_absorb_mem
        + sponge_squeeze_deselector * running_evaluation_updates_for_squeeze
}

fn log_derivative_with_u32_table_updates_correctly(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
) -> ConstraintCircuitMonad<DualRowIndicator> {
    let constant = |c: u32| circuit_builder.b_constant(c);
    let one = || constant(1);
    let two_inverse = circuit_builder.b_constant(bfe!(2).inverse());
    let challenge = |c: ChallengeId| circuit_builder.challenge(c);
    let curr_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(CurrentBaseRow(col.master_base_table_index()))
    };
    let next_base_row = |col: ProcessorBaseTableColumn| {
        circuit_builder.input(NextBaseRow(col.master_base_table_index()))
    };
    let curr_ext_row = |col: ProcessorExtTableColumn| {
        circuit_builder.input(CurrentExtRow(col.master_ext_table_index()))
    };
    let next_ext_row = |col: ProcessorExtTableColumn| {
        circuit_builder.input(NextExtRow(col.master_ext_table_index()))
    };

    let split_deselector = instruction_deselector_current_row(circuit_builder, Instruction::Split);
    let lt_deselector = instruction_deselector_current_row(circuit_builder, Instruction::Lt);
    let and_deselector = instruction_deselector_current_row(circuit_builder, Instruction::And);
    let xor_deselector = instruction_deselector_current_row(circuit_builder, Instruction::Xor);
    let pow_deselector = instruction_deselector_current_row(circuit_builder, Instruction::Pow);
    let log_2_floor_deselector =
        instruction_deselector_current_row(circuit_builder, Instruction::Log2Floor);
    let div_mod_deselector =
        instruction_deselector_current_row(circuit_builder, Instruction::DivMod);
    let pop_count_deselector =
        instruction_deselector_current_row(circuit_builder, Instruction::PopCount);
    let merkle_step_deselector =
        instruction_deselector_current_row(circuit_builder, Instruction::MerkleStep);
    let merkle_step_mem_deselector =
        instruction_deselector_current_row(circuit_builder, Instruction::MerkleStepMem);

    let running_sum = curr_ext_row(U32LookupClientLogDerivative);
    let running_sum_next = next_ext_row(U32LookupClientLogDerivative);

    let split_factor = challenge(U32Indeterminate)
        - challenge(U32LhsWeight) * next_base_row(ST0)
        - challenge(U32RhsWeight) * next_base_row(ST1)
        - challenge(U32CiWeight) * curr_base_row(CI);
    let binop_factor = challenge(U32Indeterminate)
        - challenge(U32LhsWeight) * curr_base_row(ST0)
        - challenge(U32RhsWeight) * curr_base_row(ST1)
        - challenge(U32CiWeight) * curr_base_row(CI)
        - challenge(U32ResultWeight) * next_base_row(ST0);
    let xor_factor = challenge(U32Indeterminate)
        - challenge(U32LhsWeight) * curr_base_row(ST0)
        - challenge(U32RhsWeight) * curr_base_row(ST1)
        - challenge(U32CiWeight) * constant(Instruction::And.opcode())
        - challenge(U32ResultWeight)
            * (curr_base_row(ST0) + curr_base_row(ST1) - next_base_row(ST0))
            * two_inverse;
    let unop_factor = challenge(U32Indeterminate)
        - challenge(U32LhsWeight) * curr_base_row(ST0)
        - challenge(U32CiWeight) * curr_base_row(CI)
        - challenge(U32ResultWeight) * next_base_row(ST0);
    let div_mod_factor_for_lt = challenge(U32Indeterminate)
        - challenge(U32LhsWeight) * next_base_row(ST0)
        - challenge(U32RhsWeight) * curr_base_row(ST1)
        - challenge(U32CiWeight) * constant(Instruction::Lt.opcode())
        - challenge(U32ResultWeight);
    let div_mod_factor_for_range_check = challenge(U32Indeterminate)
        - challenge(U32LhsWeight) * curr_base_row(ST0)
        - challenge(U32RhsWeight) * next_base_row(ST1)
        - challenge(U32CiWeight) * constant(Instruction::Split.opcode());
    let merkle_step_range_check_factor = challenge(U32Indeterminate)
        - challenge(U32LhsWeight) * curr_base_row(ST5)
        - challenge(U32RhsWeight) * next_base_row(ST5)
        - challenge(U32CiWeight) * constant(Instruction::Split.opcode());

    let running_sum_absorbs_split_factor =
        (running_sum_next.clone() - running_sum.clone()) * split_factor - one();
    let running_sum_absorbs_binop_factor =
        (running_sum_next.clone() - running_sum.clone()) * binop_factor - one();
    let running_sum_absorb_xor_factor =
        (running_sum_next.clone() - running_sum.clone()) * xor_factor - one();
    let running_sum_absorbs_unop_factor =
        (running_sum_next.clone() - running_sum.clone()) * unop_factor - one();
    let running_sum_absorbs_merkle_step_factor =
        (running_sum_next.clone() - running_sum.clone()) * merkle_step_range_check_factor - one();

    let split_summand = split_deselector * running_sum_absorbs_split_factor;
    let lt_summand = lt_deselector * running_sum_absorbs_binop_factor.clone();
    let and_summand = and_deselector * running_sum_absorbs_binop_factor.clone();
    let xor_summand = xor_deselector * running_sum_absorb_xor_factor;
    let pow_summand = pow_deselector * running_sum_absorbs_binop_factor;
    let log_2_floor_summand = log_2_floor_deselector * running_sum_absorbs_unop_factor.clone();
    let div_mod_summand = div_mod_deselector
        * ((running_sum_next.clone() - running_sum.clone())
            * div_mod_factor_for_lt.clone()
            * div_mod_factor_for_range_check.clone()
            - div_mod_factor_for_lt
            - div_mod_factor_for_range_check);
    let pop_count_summand = pop_count_deselector * running_sum_absorbs_unop_factor;
    let merkle_step_summand =
        merkle_step_deselector * running_sum_absorbs_merkle_step_factor.clone();
    let merkle_step_mem_summand =
        merkle_step_mem_deselector * running_sum_absorbs_merkle_step_factor;
    let no_update_summand = (one() - curr_base_row(IB2)) * (running_sum_next - running_sum);

    split_summand
        + lt_summand
        + and_summand
        + xor_summand
        + pow_summand
        + log_2_floor_summand
        + div_mod_summand
        + pop_count_summand
        + merkle_step_summand
        + merkle_step_mem_summand
        + no_update_summand
}

fn stack_weight_by_index(i: usize) -> ChallengeId {
    match i {
        0 => StackWeight0,
        1 => StackWeight1,
        2 => StackWeight2,
        3 => StackWeight3,
        4 => StackWeight4,
        5 => StackWeight5,
        6 => StackWeight6,
        7 => StackWeight7,
        8 => StackWeight8,
        9 => StackWeight9,
        10 => StackWeight10,
        11 => StackWeight11,
        12 => StackWeight12,
        13 => StackWeight13,
        14 => StackWeight14,
        15 => StackWeight15,
        i => panic!("Op Stack weight index must be in [0, 15], not {i}."),
    }
}

/// A polynomial that is 1 when evaluated on the given index, and 0 otherwise.
fn indicator_polynomial(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    index: usize,
) -> ConstraintCircuitMonad<DualRowIndicator> {
    let one = || circuit_builder.b_constant(1);
    let hv = |idx| helper_variable(circuit_builder, idx);

    match index {
        0 => (one() - hv(3)) * (one() - hv(2)) * (one() - hv(1)) * (one() - hv(0)),
        1 => (one() - hv(3)) * (one() - hv(2)) * (one() - hv(1)) * hv(0),
        2 => (one() - hv(3)) * (one() - hv(2)) * hv(1) * (one() - hv(0)),
        3 => (one() - hv(3)) * (one() - hv(2)) * hv(1) * hv(0),
        4 => (one() - hv(3)) * hv(2) * (one() - hv(1)) * (one() - hv(0)),
        5 => (one() - hv(3)) * hv(2) * (one() - hv(1)) * hv(0),
        6 => (one() - hv(3)) * hv(2) * hv(1) * (one() - hv(0)),
        7 => (one() - hv(3)) * hv(2) * hv(1) * hv(0),
        8 => hv(3) * (one() - hv(2)) * (one() - hv(1)) * (one() - hv(0)),
        9 => hv(3) * (one() - hv(2)) * (one() - hv(1)) * hv(0),
        10 => hv(3) * (one() - hv(2)) * hv(1) * (one() - hv(0)),
        11 => hv(3) * (one() - hv(2)) * hv(1) * hv(0),
        12 => hv(3) * hv(2) * (one() - hv(1)) * (one() - hv(0)),
        13 => hv(3) * hv(2) * (one() - hv(1)) * hv(0),
        14 => hv(3) * hv(2) * hv(1) * (one() - hv(0)),
        15 => hv(3) * hv(2) * hv(1) * hv(0),
        i => panic!("indicator polynomial index {i} out of bounds"),
    }
}

fn helper_variable(
    circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    index: usize,
) -> ConstraintCircuitMonad<DualRowIndicator> {
    match index {
        0 => circuit_builder.input(CurrentBaseRow(HV0.master_base_table_index())),
        1 => circuit_builder.input(CurrentBaseRow(HV1.master_base_table_index())),
        2 => circuit_builder.input(CurrentBaseRow(HV2.master_base_table_index())),
        3 => circuit_builder.input(CurrentBaseRow(HV3.master_base_table_index())),
        4 => circuit_builder.input(CurrentBaseRow(HV4.master_base_table_index())),
        5 => circuit_builder.input(CurrentBaseRow(HV5.master_base_table_index())),
        i => unimplemented!("Helper variable index {i} out of bounds."),
    }
}

#[cfg(test)]
mod tests {
    use ndarray::s;
    use ndarray::Array2;
    use num_traits::identities::Zero;
    use proptest::prop_assert_eq;
    use proptest_arbitrary_interop::arb;
    use test_strategy::proptest;

    use crate::table::NUM_BASE_COLUMNS;
    use crate::table::NUM_EXT_COLUMNS;

    use super::*;

    #[test]
    fn instruction_deselector_gives_0_for_all_other_instructions() {
        let circuit_builder = ConstraintCircuitBuilder::new();

        let mut master_base_table = Array2::zeros([2, NUM_BASE_COLUMNS]);
        let master_ext_table = Array2::zeros([2, NUM_EXT_COLUMNS]);

        // For this test, dummy challenges suffice to evaluate the constraints.
        let dummy_challenges = (0..ChallengeId::COUNT)
            .map(|i| XFieldElement::from(i as u64))
            .collect_vec();
        for instruction in ALL_INSTRUCTIONS {
            use ProcessorBaseTableColumn::*;
            let deselector = instruction_deselector_current_row(&circuit_builder, instruction);

            println!("\n\nThe Deselector for instruction {instruction} is:\n{deselector}",);

            // Negative tests
            for other_instruction in ALL_INSTRUCTIONS
                .into_iter()
                .filter(|other_instruction| *other_instruction != instruction)
            {
                let mut curr_row = master_base_table.slice_mut(s![0, ..]);
                curr_row[IB0.master_base_table_index()] = other_instruction.ib(InstructionBit::IB0);
                curr_row[IB1.master_base_table_index()] = other_instruction.ib(InstructionBit::IB1);
                curr_row[IB2.master_base_table_index()] = other_instruction.ib(InstructionBit::IB2);
                curr_row[IB3.master_base_table_index()] = other_instruction.ib(InstructionBit::IB3);
                curr_row[IB4.master_base_table_index()] = other_instruction.ib(InstructionBit::IB4);
                curr_row[IB5.master_base_table_index()] = other_instruction.ib(InstructionBit::IB5);
                curr_row[IB6.master_base_table_index()] = other_instruction.ib(InstructionBit::IB6);
                let result = deselector.clone().consume().evaluate(
                    master_base_table.view(),
                    master_ext_table.view(),
                    &dummy_challenges,
                );

                assert!(
                    result.is_zero(),
                    "Deselector for {instruction} should return 0 for all other instructions, \
                    including {other_instruction} whose opcode is {}",
                    other_instruction.opcode()
                )
            }

            // Positive tests
            let mut curr_row = master_base_table.slice_mut(s![0, ..]);
            curr_row[IB0.master_base_table_index()] = instruction.ib(InstructionBit::IB0);
            curr_row[IB1.master_base_table_index()] = instruction.ib(InstructionBit::IB1);
            curr_row[IB2.master_base_table_index()] = instruction.ib(InstructionBit::IB2);
            curr_row[IB3.master_base_table_index()] = instruction.ib(InstructionBit::IB3);
            curr_row[IB4.master_base_table_index()] = instruction.ib(InstructionBit::IB4);
            curr_row[IB5.master_base_table_index()] = instruction.ib(InstructionBit::IB5);
            curr_row[IB6.master_base_table_index()] = instruction.ib(InstructionBit::IB6);
            let result = deselector.consume().evaluate(
                master_base_table.view(),
                master_ext_table.view(),
                &dummy_challenges,
            );
            assert!(
                !result.is_zero(),
                "Deselector for {instruction} should be non-zero when CI is {}",
                instruction.opcode()
            )
        }
    }

    #[test]
    fn print_number_and_degrees_of_transition_constraints_for_all_instructions() {
        println!();
        println!("| Instruction         | #polys | max deg | Degrees");
        println!("|:--------------------|-------:|--------:|:------------");
        let circuit_builder = ConstraintCircuitBuilder::new();
        for instruction in ALL_INSTRUCTIONS {
            let constraints = transition_constraints_for_instruction(&circuit_builder, instruction);
            let degrees = constraints
                .iter()
                .map(|circuit| circuit.clone().consume().degree())
                .collect_vec();
            let max_degree = degrees.iter().max().unwrap_or(&0);
            let degrees_str = degrees.iter().join(", ");
            println!(
                "| {:<19} | {:>6} | {max_degree:>7} | [{degrees_str}]",
                format!("{instruction}"),
                constraints.len(),
            );
        }
    }
    #[test]
    fn range_check_for_skiz_is_as_efficient_as_possible() {
        let range_check_constraints = next_instruction_range_check_constraints_for_instruction_skiz(
            &ConstraintCircuitBuilder::new(),
        );
        let range_check_constraints = range_check_constraints.iter();
        let all_degrees = range_check_constraints.map(|c| c.clone().consume().degree());
        let max_constraint_degree = all_degrees.max().unwrap_or(0);
        assert!(
            crate::TARGET_DEGREE <= max_constraint_degree,
            "Can the range check constraints be of a higher degree, saving columns?"
        );
    }

    #[test]
    fn helper_variables_in_bounds() {
        let circuit_builder = ConstraintCircuitBuilder::new();
        for index in 0..NUM_HELPER_VARIABLE_REGISTERS {
            helper_variable(&circuit_builder, index);
        }
    }

    #[proptest]
    #[should_panic(expected = "out of bounds")]
    fn cannot_get_helper_variable_for_out_of_range_index(
        #[strategy(NUM_HELPER_VARIABLE_REGISTERS..)] index: usize,
    ) {
        let circuit_builder = ConstraintCircuitBuilder::new();
        helper_variable(&circuit_builder, index);
    }

    #[test]
    fn indicator_polynomial_in_bounds() {
        let circuit_builder = ConstraintCircuitBuilder::new();
        for index in 0..16 {
            indicator_polynomial(&circuit_builder, index);
        }
    }

    #[proptest]
    #[should_panic(expected = "out of bounds")]
    fn cannot_get_indicator_polynomial_for_out_of_range_index(
        #[strategy(16_usize..
        )]
        index: usize,
    ) {
        let circuit_builder = ConstraintCircuitBuilder::new();
        indicator_polynomial(&circuit_builder, index);
    }

    #[proptest]
    fn indicator_polynomial_is_one_on_indicated_index_and_zero_on_all_other_indices(
        #[strategy(0_usize..16)] indicator_poly_index: usize,
        #[strategy(0_u64..16)] query_index: u64,
    ) {
        let mut base_table = Array2::ones([2, NUM_BASE_COLUMNS]);
        let aux_table = Array2::ones([2, NUM_EXT_COLUMNS]);

        base_table[[0, HV0.master_base_table_index()]] = bfe!(query_index % 2);
        base_table[[0, HV1.master_base_table_index()]] = bfe!((query_index >> 1) % 2);
        base_table[[0, HV2.master_base_table_index()]] = bfe!((query_index >> 2) % 2);
        base_table[[0, HV3.master_base_table_index()]] = bfe!((query_index >> 3) % 2);

        let builder = ConstraintCircuitBuilder::new();
        let indicator_poly = indicator_polynomial(&builder, indicator_poly_index).consume();
        let evaluation = indicator_poly.evaluate(base_table.view(), aux_table.view(), &[]);

        if indicator_poly_index as u64 == query_index {
            prop_assert_eq!(xfe!(1), evaluation);
        } else {
            prop_assert_eq!(xfe!(0), evaluation);
        }
    }

    #[test]
    fn can_get_op_stack_column_for_in_range_index() {
        for index in 0..OpStackElement::COUNT {
            let _ = ProcessorTable::op_stack_column_by_index(index);
        }
    }

    #[proptest]
    #[should_panic(expected = "[0, 15]")]
    fn cannot_get_op_stack_column_for_out_of_range_index(
        #[strategy(OpStackElement::COUNT..)] index: usize,
    ) {
        let _ = ProcessorTable::op_stack_column_by_index(index);
    }

    #[test]
    fn can_get_stack_weight_for_in_range_index() {
        for index in 0..OpStackElement::COUNT {
            let _ = stack_weight_by_index(index);
        }
    }

    #[proptest]
    #[should_panic(expected = "[0, 15]")]
    fn cannot_get_stack_weight_for_out_of_range_index(
        #[strategy(OpStackElement::COUNT..)] index: usize,
    ) {
        let _ = stack_weight_by_index(index);
    }

    #[proptest]
    fn xx_product_is_accurate(
        #[strategy(arb())] a: XFieldElement,
        #[strategy(arb())] b: XFieldElement,
    ) {
        let circuit_builder = ConstraintCircuitBuilder::new();
        let main_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(BaseRow(col.master_base_table_index()))
        };
        let [x0, x1, x2, y0, y1, y2] = [ST0, ST1, ST2, ST3, ST4, ST5].map(main_row);

        let mut base_table = Array2::zeros([1, NUM_BASE_COLUMNS]);
        let ext_table = Array2::zeros([1, NUM_EXT_COLUMNS]);
        base_table[[0, ST0.master_base_table_index()]] = a.coefficients[0];
        base_table[[0, ST1.master_base_table_index()]] = a.coefficients[1];
        base_table[[0, ST2.master_base_table_index()]] = a.coefficients[2];
        base_table[[0, ST3.master_base_table_index()]] = b.coefficients[0];
        base_table[[0, ST4.master_base_table_index()]] = b.coefficients[1];
        base_table[[0, ST5.master_base_table_index()]] = b.coefficients[2];

        let [c0, c1, c2] = xx_product([x0, x1, x2], [y0, y1, y2])
            .map(|c| c.consume())
            .map(|c| c.evaluate(base_table.view(), ext_table.view(), &[]))
            .map(|xfe| xfe.unlift().unwrap());

        let c = a * b;
        prop_assert_eq!(c.coefficients[0], c0);
        prop_assert_eq!(c.coefficients[1], c1);
        prop_assert_eq!(c.coefficients[2], c2);
    }

    #[proptest]
    fn xb_product_is_accurate(
        #[strategy(arb())] a: XFieldElement,
        #[strategy(arb())] b: BFieldElement,
    ) {
        let circuit_builder = ConstraintCircuitBuilder::new();
        let base_row = |col: ProcessorBaseTableColumn| {
            circuit_builder.input(BaseRow(col.master_base_table_index()))
        };
        let [x0, x1, x2, y] = [ST0, ST1, ST2, ST3].map(base_row);

        let mut base_table = Array2::zeros([1, NUM_BASE_COLUMNS]);
        let ext_table = Array2::zeros([1, NUM_EXT_COLUMNS]);
        base_table[[0, ST0.master_base_table_index()]] = a.coefficients[0];
        base_table[[0, ST1.master_base_table_index()]] = a.coefficients[1];
        base_table[[0, ST2.master_base_table_index()]] = a.coefficients[2];
        base_table[[0, ST3.master_base_table_index()]] = b;

        let [c0, c1, c2] = xb_product([x0, x1, x2], y)
            .map(|c| c.consume())
            .map(|c| c.evaluate(base_table.view(), ext_table.view(), &[]))
            .map(|xfe| xfe.unlift().unwrap());

        let c = a * b;
        prop_assert_eq!(c.coefficients[0], c0);
        prop_assert_eq!(c.coefficients[1], c1);
        prop_assert_eq!(c.coefficients[2], c2);
    }
}
