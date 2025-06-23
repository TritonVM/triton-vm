use air::challenge_id::ChallengeId;
use air::cross_table_argument::CrossTableArg;
use air::cross_table_argument::EvalArg;
use air::cross_table_argument::LookupArg;
use air::cross_table_argument::PermArg;
use air::table::processor::ProcessorTable;
use air::table::ram;
use air::table_column::MasterAuxColumn;
use air::table_column::MasterMainColumn;
use isa::instruction::Instruction;
use isa::op_stack::OpStackElement;
use itertools::Itertools;
use ndarray::parallel::prelude::*;
use ndarray::prelude::*;
use num_traits::ConstOne;
use num_traits::One;
use num_traits::Zero;
use strum::EnumCount;
use strum::IntoEnumIterator;
use twenty_first::math::traits::FiniteField;
use twenty_first::prelude::*;

use crate::aet::AlgebraicExecutionTrace;
use crate::challenges::Challenges;
use crate::ndarray_helper::ROW_AXIS;
use crate::ndarray_helper::contiguous_column_slices;
use crate::ndarray_helper::horizontal_multi_slice_mut;
use crate::profiler::profiler;
use crate::table::TraceTable;

type MainColumn = <ProcessorTable as air::AIR>::MainColumn;
type AuxColumn = <ProcessorTable as air::AIR>::AuxColumn;

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub(super) struct ClkJumpDiffs {
    pub op_stack: Vec<BFieldElement>,
    pub ram: Vec<BFieldElement>,
    pub jump_stack: Vec<BFieldElement>,
}

impl TraceTable for ProcessorTable {
    type FillParam = ClkJumpDiffs;
    type FillReturnInfo = ();

    fn fill(
        mut main_table: ArrayViewMut2<BFieldElement>,
        aet: &AlgebraicExecutionTrace,
        clk_jump_diffs: Self::FillParam,
    ) {
        let num_rows = aet.processor_trace.nrows();
        let mut clk_jump_diff_multiplicities = Array1::zeros([num_rows]);

        for clk_jump_diff in clk_jump_diffs
            .op_stack
            .into_iter()
            .chain(clk_jump_diffs.ram)
            .chain(clk_jump_diffs.jump_stack)
        {
            let clk = clk_jump_diff.value() as usize;
            clk_jump_diff_multiplicities[clk] += BFieldElement::ONE;
        }

        let mut processor_table = main_table.slice_mut(s![0..num_rows, ..]);
        processor_table.assign(&aet.processor_trace);
        processor_table
            .column_mut(MainColumn::ClockJumpDifferenceLookupMultiplicity.main_index())
            .assign(&clk_jump_diff_multiplicities);
    }

    fn pad(mut main_table: ArrayViewMut2<BFieldElement>, table_len: usize) {
        assert!(table_len > 0, "Processor Table must have at least one row.");
        let mut padding_template = main_table.row(table_len - 1).to_owned();
        padding_template[MainColumn::IsPadding.main_index()] = bfe!(1);
        padding_template[MainColumn::ClockJumpDifferenceLookupMultiplicity.main_index()] = bfe!(0);
        main_table
            .slice_mut(s![table_len.., ..])
            .axis_iter_mut(ROW_AXIS)
            .into_par_iter()
            .for_each(|mut row| row.assign(&padding_template));

        let clk_range = table_len..main_table.nrows();
        let clk_col = Array1::from_iter(clk_range.map(|a| bfe!(a as u64)));
        clk_col.move_into(main_table.slice_mut(s![table_len.., MainColumn::CLK.main_index()]));

        // The Jump Stack Table does not have a padding indicator. Hence, clock
        // jump differences are being looked up in its padding sections. The
        // clock jump differences in that section are always 1. The lookup
        // multiplicities of clock value 1 must be increased accordingly: one
        // per padding row.
        let num_padding_rows = main_table.nrows() - table_len;
        let num_padding_rows = bfe!(num_padding_rows as u64);
        let mut row_1 = main_table.row_mut(1);

        row_1[MainColumn::ClockJumpDifferenceLookupMultiplicity.main_index()] += num_padding_rows;
    }

    fn extend(
        main_table: ArrayView2<BFieldElement>,
        mut aux_table: ArrayViewMut2<XFieldElement>,
        challenges: &Challenges,
    ) {
        profiler!(start "processor table");
        assert_eq!(MainColumn::COUNT, main_table.ncols());
        assert_eq!(AuxColumn::COUNT, aux_table.ncols());
        assert_eq!(main_table.nrows(), aux_table.nrows());

        let all_column_indices = AuxColumn::iter()
            .map(|column| column.aux_index())
            .collect_vec();
        let all_column_slices = horizontal_multi_slice_mut(
            aux_table.view_mut(),
            &contiguous_column_slices(&all_column_indices),
        );

        let all_column_generators = [
            auxiliary_column_input_table_eval_argument,
            auxiliary_column_output_table_eval_argument,
            auxiliary_column_instruction_lookup_argument,
            auxiliary_column_op_stack_table_perm_argument,
            auxiliary_column_ram_table_perm_argument,
            auxiliary_column_jump_stack_table_perm_argument,
            auxiliary_column_hash_input_eval_argument,
            auxiliary_column_hash_digest_eval_argument,
            auxiliary_column_sponge_eval_argument,
            auxiliary_column_for_u32_lookup_argument,
            auxiliary_column_for_clock_jump_difference_lookup_argument,
        ];
        all_column_generators
            .into_par_iter()
            .zip_eq(all_column_slices)
            .for_each(|(generator, slice)| {
                generator(main_table, challenges).move_into(slice);
            });

        profiler!(stop "processor table");
    }
}

fn auxiliary_column_input_table_eval_argument(
    main_table: ArrayView2<BFieldElement>,
    challenges: &Challenges,
) -> Array2<XFieldElement> {
    let mut input_table_running_evaluation = EvalArg::default_initial();
    let mut auxiliary_column = Vec::with_capacity(main_table.nrows());
    auxiliary_column.push(input_table_running_evaluation);
    for (previous_row, current_row) in main_table.rows().into_iter().tuple_windows() {
        if let Some(Instruction::ReadIo(st)) = instruction_from_row(previous_row) {
            for i in (0..st.num_words()).rev() {
                let input_symbol_column = ProcessorTable::op_stack_column_by_index(i);
                let input_symbol = current_row[input_symbol_column.main_index()];
                input_table_running_evaluation = input_table_running_evaluation
                    * challenges[ChallengeId::StandardInputIndeterminate]
                    + input_symbol;
            }
        }
        auxiliary_column.push(input_table_running_evaluation);
    }
    Array2::from_shape_vec((main_table.nrows(), 1), auxiliary_column).unwrap()
}

fn auxiliary_column_output_table_eval_argument(
    main_table: ArrayView2<BFieldElement>,
    challenges: &Challenges,
) -> Array2<XFieldElement> {
    let mut output_table_running_evaluation = EvalArg::default_initial();
    let mut auxiliary_column = Vec::with_capacity(main_table.nrows());
    auxiliary_column.push(output_table_running_evaluation);
    for (previous_row, _) in main_table.rows().into_iter().tuple_windows() {
        if let Some(Instruction::WriteIo(st)) = instruction_from_row(previous_row) {
            for i in 0..st.num_words() {
                let output_symbol_column = ProcessorTable::op_stack_column_by_index(i);
                let output_symbol = previous_row[output_symbol_column.main_index()];
                output_table_running_evaluation = output_table_running_evaluation
                    * challenges[ChallengeId::StandardOutputIndeterminate]
                    + output_symbol;
            }
        }
        auxiliary_column.push(output_table_running_evaluation);
    }
    Array2::from_shape_vec((main_table.nrows(), 1), auxiliary_column).unwrap()
}

fn auxiliary_column_instruction_lookup_argument(
    main_table: ArrayView2<BFieldElement>,
    challenges: &Challenges,
) -> Array2<XFieldElement> {
    // collect all to-be-inverted elements for batch inversion
    let mut to_invert = vec![];
    for row in main_table.rows() {
        if row[MainColumn::IsPadding.main_index()].is_one() {
            break; // padding marks the end of the trace
        }

        let compressed_row = row[MainColumn::IP.main_index()]
            * challenges[ChallengeId::ProgramAddressWeight]
            + row[MainColumn::CI.main_index()] * challenges[ChallengeId::ProgramInstructionWeight]
            + row[MainColumn::NIA.main_index()]
                * challenges[ChallengeId::ProgramNextInstructionWeight];
        to_invert.push(challenges[ChallengeId::InstructionLookupIndeterminate] - compressed_row);
    }

    // populate auxiliary column with inverses
    let mut instruction_lookup_log_derivative = LookupArg::default_initial();
    let mut auxiliary_column = Vec::with_capacity(main_table.nrows());
    for inverse in XFieldElement::batch_inversion(to_invert) {
        instruction_lookup_log_derivative += inverse;
        auxiliary_column.push(instruction_lookup_log_derivative);
    }

    // fill padding section
    auxiliary_column.resize(main_table.nrows(), instruction_lookup_log_derivative);
    Array2::from_shape_vec((main_table.nrows(), 1), auxiliary_column).unwrap()
}

fn auxiliary_column_op_stack_table_perm_argument(
    main_table: ArrayView2<BFieldElement>,
    challenges: &Challenges,
) -> Array2<XFieldElement> {
    let mut op_stack_table_running_product = EvalArg::default_initial();
    let mut auxiliary_column = Vec::with_capacity(main_table.nrows());
    auxiliary_column.push(op_stack_table_running_product);
    for (prev, curr) in main_table.rows().into_iter().tuple_windows() {
        op_stack_table_running_product *=
            factor_for_op_stack_table_running_product(prev, curr, challenges);
        auxiliary_column.push(op_stack_table_running_product);
    }
    Array2::from_shape_vec((main_table.nrows(), 1), auxiliary_column).unwrap()
}

fn auxiliary_column_ram_table_perm_argument(
    main_table: ArrayView2<BFieldElement>,
    challenges: &Challenges,
) -> Array2<XFieldElement> {
    let mut ram_table_running_product = PermArg::default_initial();
    let mut auxiliary_column = Vec::with_capacity(main_table.nrows());
    auxiliary_column.push(ram_table_running_product);
    for (prev, curr) in main_table.rows().into_iter().tuple_windows() {
        if let Some(f) = factor_for_ram_table_running_product(prev, curr, challenges) {
            ram_table_running_product *= f;
        };
        auxiliary_column.push(ram_table_running_product);
    }
    Array2::from_shape_vec((main_table.nrows(), 1), auxiliary_column).unwrap()
}

fn auxiliary_column_jump_stack_table_perm_argument(
    main_table: ArrayView2<BFieldElement>,
    challenges: &Challenges,
) -> Array2<XFieldElement> {
    let mut jump_stack_running_product = PermArg::default_initial();
    let mut auxiliary_column = Vec::with_capacity(main_table.nrows());
    for row in main_table.rows() {
        let compressed_row = row[MainColumn::CLK.main_index()]
            * challenges[ChallengeId::JumpStackClkWeight]
            + row[MainColumn::CI.main_index()] * challenges[ChallengeId::JumpStackCiWeight]
            + row[MainColumn::JSP.main_index()] * challenges[ChallengeId::JumpStackJspWeight]
            + row[MainColumn::JSO.main_index()] * challenges[ChallengeId::JumpStackJsoWeight]
            + row[MainColumn::JSD.main_index()] * challenges[ChallengeId::JumpStackJsdWeight];
        jump_stack_running_product *=
            challenges[ChallengeId::JumpStackIndeterminate] - compressed_row;
        auxiliary_column.push(jump_stack_running_product);
    }
    Array2::from_shape_vec((main_table.nrows(), 1), auxiliary_column).unwrap()
}

/// Hash Table – `hash`'s or `merkle_step`'s input from Processor to Hash
/// Coprocessor
fn auxiliary_column_hash_input_eval_argument(
    main_table: ArrayView2<BFieldElement>,
    challenges: &Challenges,
) -> Array2<XFieldElement> {
    let st0_through_st9 = [
        MainColumn::ST0,
        MainColumn::ST1,
        MainColumn::ST2,
        MainColumn::ST3,
        MainColumn::ST4,
        MainColumn::ST5,
        MainColumn::ST6,
        MainColumn::ST7,
        MainColumn::ST8,
        MainColumn::ST9,
    ];
    let hash_state_weights = &challenges[ChallengeId::StackWeight0..ChallengeId::StackWeight10];

    let merkle_step_left_sibling = [
        MainColumn::ST0,
        MainColumn::ST1,
        MainColumn::ST2,
        MainColumn::ST3,
        MainColumn::ST4,
        MainColumn::HV0,
        MainColumn::HV1,
        MainColumn::HV2,
        MainColumn::HV3,
        MainColumn::HV4,
    ];
    let merkle_step_right_sibling = [
        MainColumn::HV0,
        MainColumn::HV1,
        MainColumn::HV2,
        MainColumn::HV3,
        MainColumn::HV4,
        MainColumn::ST0,
        MainColumn::ST1,
        MainColumn::ST2,
        MainColumn::ST3,
        MainColumn::ST4,
    ];

    let mut hash_input_running_evaluation = EvalArg::default_initial();
    let mut auxiliary_column = Vec::with_capacity(main_table.nrows());
    for row in main_table.rows() {
        let current_instruction = row[MainColumn::CI.main_index()];
        if current_instruction == Instruction::Hash.opcode_b()
            || current_instruction == Instruction::MerkleStep.opcode_b()
            || current_instruction == Instruction::MerkleStepMem.opcode_b()
        {
            let is_left_sibling = row[MainColumn::ST5.main_index()].value() % 2 == 0;
            let hash_input = match instruction_from_row(row) {
                Some(Instruction::MerkleStep | Instruction::MerkleStepMem) if is_left_sibling => {
                    merkle_step_left_sibling
                }
                Some(Instruction::MerkleStep | Instruction::MerkleStepMem) => {
                    merkle_step_right_sibling
                }
                Some(Instruction::Hash) => st0_through_st9,
                _ => unreachable!(),
            };
            let compressed_row = hash_input
                .map(|st| row[st.main_index()])
                .into_iter()
                .zip_eq(hash_state_weights.iter())
                .map(|(st, &weight)| weight * st)
                .sum::<XFieldElement>();
            hash_input_running_evaluation = hash_input_running_evaluation
                * challenges[ChallengeId::HashInputIndeterminate]
                + compressed_row;
        }
        auxiliary_column.push(hash_input_running_evaluation);
    }
    Array2::from_shape_vec((main_table.nrows(), 1), auxiliary_column).unwrap()
}

/// Hash Table – `hash`'s output from Hash Coprocessor to Processor
fn auxiliary_column_hash_digest_eval_argument(
    main_table: ArrayView2<BFieldElement>,
    challenges: &Challenges,
) -> Array2<XFieldElement> {
    let mut hash_digest_running_evaluation = EvalArg::default_initial();
    let mut auxiliary_column = Vec::with_capacity(main_table.nrows());
    auxiliary_column.push(hash_digest_running_evaluation);
    for (previous_row, current_row) in main_table.rows().into_iter().tuple_windows() {
        let previous_ci = previous_row[MainColumn::CI.main_index()];
        if previous_ci == Instruction::Hash.opcode_b()
            || previous_ci == Instruction::MerkleStep.opcode_b()
            || previous_ci == Instruction::MerkleStepMem.opcode_b()
        {
            let compressed_row = [
                MainColumn::ST0,
                MainColumn::ST1,
                MainColumn::ST2,
                MainColumn::ST3,
                MainColumn::ST4,
            ]
            .map(|st| current_row[st.main_index()])
            .into_iter()
            .zip_eq(&challenges[ChallengeId::StackWeight0..=ChallengeId::StackWeight4])
            .map(|(st, &weight)| weight * st)
            .sum::<XFieldElement>();
            hash_digest_running_evaluation = hash_digest_running_evaluation
                * challenges[ChallengeId::HashDigestIndeterminate]
                + compressed_row;
        }
        auxiliary_column.push(hash_digest_running_evaluation);
    }
    Array2::from_shape_vec((main_table.nrows(), 1), auxiliary_column).unwrap()
}

/// Hash Table – `hash`'s or `merkle_step`'s input from Processor to Hash
/// Coprocessor
fn auxiliary_column_sponge_eval_argument(
    main_table: ArrayView2<BFieldElement>,
    challenges: &Challenges,
) -> Array2<XFieldElement> {
    let st0_through_st9 = [
        MainColumn::ST0,
        MainColumn::ST1,
        MainColumn::ST2,
        MainColumn::ST3,
        MainColumn::ST4,
        MainColumn::ST5,
        MainColumn::ST6,
        MainColumn::ST7,
        MainColumn::ST8,
        MainColumn::ST9,
    ];
    let hash_state_weights = &challenges[ChallengeId::StackWeight0..ChallengeId::StackWeight10];

    let mut sponge_running_evaluation = EvalArg::default_initial();
    let mut auxiliary_column = Vec::with_capacity(main_table.nrows());
    auxiliary_column.push(sponge_running_evaluation);
    for (previous_row, current_row) in main_table.rows().into_iter().tuple_windows() {
        let previous_ci = previous_row[MainColumn::CI.main_index()];
        if previous_ci == Instruction::SpongeInit.opcode_b() {
            sponge_running_evaluation = sponge_running_evaluation
                * challenges[ChallengeId::SpongeIndeterminate]
                + challenges[ChallengeId::HashCIWeight] * Instruction::SpongeInit.opcode_b();
        } else if previous_ci == Instruction::SpongeAbsorb.opcode_b() {
            let compressed_row = st0_through_st9
                .map(|st| previous_row[st.main_index()])
                .into_iter()
                .zip_eq(hash_state_weights.iter())
                .map(|(st, &weight)| weight * st)
                .sum::<XFieldElement>();
            sponge_running_evaluation = sponge_running_evaluation
                * challenges[ChallengeId::SpongeIndeterminate]
                + challenges[ChallengeId::HashCIWeight] * Instruction::SpongeAbsorb.opcode_b()
                + compressed_row;
        } else if previous_ci == Instruction::SpongeAbsorbMem.opcode_b() {
            let stack_elements = [
                MainColumn::ST1,
                MainColumn::ST2,
                MainColumn::ST3,
                MainColumn::ST4,
            ];
            let helper_variables = [
                MainColumn::HV0,
                MainColumn::HV1,
                MainColumn::HV2,
                MainColumn::HV3,
                MainColumn::HV4,
                MainColumn::HV5,
            ];
            let compressed_row = stack_elements
                .map(|st| current_row[st.main_index()])
                .into_iter()
                .chain(helper_variables.map(|hv| previous_row[hv.main_index()]))
                .zip_eq(hash_state_weights.iter())
                .map(|(element, &weight)| weight * element)
                .sum::<XFieldElement>();
            sponge_running_evaluation = sponge_running_evaluation
                * challenges[ChallengeId::SpongeIndeterminate]
                + challenges[ChallengeId::HashCIWeight] * Instruction::SpongeAbsorb.opcode_b()
                + compressed_row;
        } else if previous_ci == Instruction::SpongeSqueeze.opcode_b() {
            let compressed_row = st0_through_st9
                .map(|st| current_row[st.main_index()])
                .into_iter()
                .zip_eq(hash_state_weights.iter())
                .map(|(st, &weight)| weight * st)
                .sum::<XFieldElement>();
            sponge_running_evaluation = sponge_running_evaluation
                * challenges[ChallengeId::SpongeIndeterminate]
                + challenges[ChallengeId::HashCIWeight] * Instruction::SpongeSqueeze.opcode_b()
                + compressed_row;
        }
        auxiliary_column.push(sponge_running_evaluation);
    }
    Array2::from_shape_vec((main_table.nrows(), 1), auxiliary_column).unwrap()
}

fn auxiliary_column_for_u32_lookup_argument(
    main_table: ArrayView2<BFieldElement>,
    challenges: &Challenges,
) -> Array2<XFieldElement> {
    // collect elements to be inverted for more performant batch inversion
    let mut to_invert = vec![];
    for (previous_row, current_row) in main_table.rows().into_iter().tuple_windows() {
        let previous_ci = previous_row[MainColumn::CI.main_index()];
        if previous_ci == Instruction::Split.opcode_b() {
            let compressed_row = current_row[MainColumn::ST0.main_index()]
                * challenges[ChallengeId::U32LhsWeight]
                + current_row[MainColumn::ST1.main_index()] * challenges[ChallengeId::U32RhsWeight]
                + previous_row[MainColumn::CI.main_index()] * challenges[ChallengeId::U32CiWeight];
            to_invert.push(challenges[ChallengeId::U32Indeterminate] - compressed_row);
        } else if previous_ci == Instruction::Lt.opcode_b()
            || previous_ci == Instruction::And.opcode_b()
            || previous_ci == Instruction::Pow.opcode_b()
        {
            let compressed_row = previous_row[MainColumn::ST0.main_index()]
                * challenges[ChallengeId::U32LhsWeight]
                + previous_row[MainColumn::ST1.main_index()]
                    * challenges[ChallengeId::U32RhsWeight]
                + previous_row[MainColumn::CI.main_index()] * challenges[ChallengeId::U32CiWeight]
                + current_row[MainColumn::ST0.main_index()]
                    * challenges[ChallengeId::U32ResultWeight];
            to_invert.push(challenges[ChallengeId::U32Indeterminate] - compressed_row);
        } else if previous_ci == Instruction::Xor.opcode_b() {
            // Triton VM uses the following equality to compute the results of
            // both the `and` and `xor` instruction using the u32 coprocessor's
            // `and` capability:
            //     a ^ b = a + b - 2 · (a & b)
            // <=> a & b = (a + b - a ^ b) / 2
            let st0_prev = previous_row[MainColumn::ST0.main_index()];
            let st1_prev = previous_row[MainColumn::ST1.main_index()];
            let st0 = current_row[MainColumn::ST0.main_index()];
            let from_xor_in_processor_to_and_in_u32_coprocessor =
                (st0_prev + st1_prev - st0) / bfe!(2);
            let compressed_row = st0_prev * challenges[ChallengeId::U32LhsWeight]
                + st1_prev * challenges[ChallengeId::U32RhsWeight]
                + Instruction::And.opcode_b() * challenges[ChallengeId::U32CiWeight]
                + from_xor_in_processor_to_and_in_u32_coprocessor
                    * challenges[ChallengeId::U32ResultWeight];
            to_invert.push(challenges[ChallengeId::U32Indeterminate] - compressed_row);
        } else if previous_ci == Instruction::Log2Floor.opcode_b()
            || previous_ci == Instruction::PopCount.opcode_b()
        {
            let compressed_row = previous_row[MainColumn::ST0.main_index()]
                * challenges[ChallengeId::U32LhsWeight]
                + previous_row[MainColumn::CI.main_index()] * challenges[ChallengeId::U32CiWeight]
                + current_row[MainColumn::ST0.main_index()]
                    * challenges[ChallengeId::U32ResultWeight];
            to_invert.push(challenges[ChallengeId::U32Indeterminate] - compressed_row);
        } else if previous_ci == Instruction::DivMod.opcode_b() {
            let compressed_row_for_lt_check = current_row[MainColumn::ST0.main_index()]
                * challenges[ChallengeId::U32LhsWeight]
                + previous_row[MainColumn::ST1.main_index()]
                    * challenges[ChallengeId::U32RhsWeight]
                + Instruction::Lt.opcode_b() * challenges[ChallengeId::U32CiWeight]
                + bfe!(1) * challenges[ChallengeId::U32ResultWeight];
            let compressed_row_for_range_check = previous_row[MainColumn::ST0.main_index()]
                * challenges[ChallengeId::U32LhsWeight]
                + current_row[MainColumn::ST1.main_index()] * challenges[ChallengeId::U32RhsWeight]
                + Instruction::Split.opcode_b() * challenges[ChallengeId::U32CiWeight];
            to_invert.push(challenges[ChallengeId::U32Indeterminate] - compressed_row_for_lt_check);
            to_invert
                .push(challenges[ChallengeId::U32Indeterminate] - compressed_row_for_range_check);
        } else if previous_ci == Instruction::MerkleStep.opcode_b()
            || previous_ci == Instruction::MerkleStepMem.opcode_b()
        {
            let compressed_row = previous_row[MainColumn::ST5.main_index()]
                * challenges[ChallengeId::U32LhsWeight]
                + current_row[MainColumn::ST5.main_index()] * challenges[ChallengeId::U32RhsWeight]
                + Instruction::Split.opcode_b() * challenges[ChallengeId::U32CiWeight];
            to_invert.push(challenges[ChallengeId::U32Indeterminate] - compressed_row);
        }
    }
    let mut inverses = XFieldElement::batch_inversion(to_invert).into_iter();

    // populate column with inverses
    let mut u32_table_running_sum_log_derivative = LookupArg::default_initial();
    let mut auxiliary_column = Vec::with_capacity(main_table.nrows());
    auxiliary_column.push(u32_table_running_sum_log_derivative);
    for (previous_row, _) in main_table.rows().into_iter().tuple_windows() {
        let previous_ci = previous_row[MainColumn::CI.main_index()];
        if Instruction::try_from(previous_ci)
            .unwrap()
            .is_u32_instruction()
        {
            u32_table_running_sum_log_derivative += inverses.next().unwrap();
        }

        // instruction `div_mod` requires a second inverse
        if previous_ci == Instruction::DivMod.opcode_b() {
            u32_table_running_sum_log_derivative += inverses.next().unwrap();
        }

        auxiliary_column.push(u32_table_running_sum_log_derivative);
    }

    Array2::from_shape_vec((main_table.nrows(), 1), auxiliary_column).unwrap()
}

fn auxiliary_column_for_clock_jump_difference_lookup_argument(
    main_table: ArrayView2<BFieldElement>,
    challenges: &Challenges,
) -> Array2<XFieldElement> {
    // collect inverses to batch invert
    let mut to_invert = vec![];
    for row in main_table.rows() {
        let lookup_multiplicity =
            row[MainColumn::ClockJumpDifferenceLookupMultiplicity.main_index()];
        if !lookup_multiplicity.is_zero() {
            let clk = row[MainColumn::CLK.main_index()];
            to_invert.push(challenges[ChallengeId::ClockJumpDifferenceLookupIndeterminate] - clk);
        }
    }
    let mut inverses = XFieldElement::batch_inversion(to_invert).into_iter();

    // populate auxiliary column with inverses
    let mut cjd_lookup_log_derivative = LookupArg::default_initial();
    let mut auxiliary_column = Vec::with_capacity(main_table.nrows());
    for row in main_table.rows() {
        let lookup_multiplicity =
            row[MainColumn::ClockJumpDifferenceLookupMultiplicity.main_index()];
        if !lookup_multiplicity.is_zero() {
            cjd_lookup_log_derivative += inverses.next().unwrap() * lookup_multiplicity;
        }
        auxiliary_column.push(cjd_lookup_log_derivative);
    }

    Array2::from_shape_vec((main_table.nrows(), 1), auxiliary_column).unwrap()
}

fn factor_for_op_stack_table_running_product(
    previous_row: ArrayView1<BFieldElement>,
    current_row: ArrayView1<BFieldElement>,
    challenges: &Challenges,
) -> XFieldElement {
    let default_factor = xfe!(1);

    let is_padding_row = current_row[MainColumn::IsPadding.main_index()].is_one();
    if is_padding_row {
        return default_factor;
    }

    let Some(previous_instruction) = instruction_from_row(previous_row) else {
        return default_factor;
    };

    // shorter stack means relevant information is on top of stack, i.e., in
    // stack registers
    let row_with_shorter_stack = if previous_instruction.op_stack_size_influence() > 0 {
        previous_row.view()
    } else {
        current_row.view()
    };
    let op_stack_delta = previous_instruction
        .op_stack_size_influence()
        .unsigned_abs() as usize;

    let mut factor = default_factor;
    for op_stack_pointer_offset in 0..op_stack_delta {
        let max_stack_element_index = OpStackElement::COUNT - 1;
        let stack_element_index = max_stack_element_index - op_stack_pointer_offset;
        let stack_element_column = ProcessorTable::op_stack_column_by_index(stack_element_index);
        let underflow_element = row_with_shorter_stack[stack_element_column.main_index()];

        let op_stack_pointer = row_with_shorter_stack[MainColumn::OpStackPointer.main_index()];
        let offset = bfe!(op_stack_pointer_offset as u64);
        let offset_op_stack_pointer = op_stack_pointer + offset;

        let clk = previous_row[MainColumn::CLK.main_index()];
        let ib1_shrink_stack = previous_row[MainColumn::IB1.main_index()];
        let compressed_row = clk * challenges[ChallengeId::OpStackClkWeight]
            + ib1_shrink_stack * challenges[ChallengeId::OpStackIb1Weight]
            + offset_op_stack_pointer * challenges[ChallengeId::OpStackPointerWeight]
            + underflow_element * challenges[ChallengeId::OpStackFirstUnderflowElementWeight];
        factor *= challenges[ChallengeId::OpStackIndeterminate] - compressed_row;
    }
    factor
}

fn factor_for_ram_table_running_product(
    prev_row: ArrayView1<BFieldElement>,
    curr_row: ArrayView1<BFieldElement>,
    challenges: &Challenges,
) -> Option<XFieldElement> {
    let is_padding_row = curr_row[MainColumn::IsPadding.main_index()].is_one();
    if is_padding_row {
        return None;
    }

    let instruction = instruction_from_row(prev_row)?;

    let clk = prev_row[MainColumn::CLK.main_index()];
    let instruction_type = match instruction {
        Instruction::ReadMem(_) => ram::INSTRUCTION_TYPE_READ,
        Instruction::WriteMem(_) => ram::INSTRUCTION_TYPE_WRITE,
        Instruction::SpongeAbsorbMem => ram::INSTRUCTION_TYPE_READ,
        Instruction::MerkleStepMem => ram::INSTRUCTION_TYPE_READ,
        Instruction::XxDotStep => ram::INSTRUCTION_TYPE_READ,
        Instruction::XbDotStep => ram::INSTRUCTION_TYPE_READ,
        _ => return None,
    };
    let mut accesses = vec![];

    match instruction {
        Instruction::ReadMem(_) | Instruction::WriteMem(_) => {
            // longer stack means relevant information is on top of stack, i.e.,
            // available in stack registers
            let row_with_longer_stack = if let Instruction::ReadMem(_) = instruction {
                curr_row.view()
            } else {
                prev_row.view()
            };
            let op_stack_delta = instruction.op_stack_size_influence().unsigned_abs() as usize;

            let num_ram_pointers = 1;
            for ram_pointer_offset in 0..op_stack_delta {
                let ram_value_index = ram_pointer_offset + num_ram_pointers;
                let ram_value_column = ProcessorTable::op_stack_column_by_index(ram_value_index);
                let ram_value = row_with_longer_stack[ram_value_column.main_index()];
                let offset_ram_pointer =
                    offset_ram_pointer(instruction, row_with_longer_stack, ram_pointer_offset);
                accesses.push((offset_ram_pointer, ram_value));
            }
        }
        Instruction::SpongeAbsorbMem => {
            let mem_ptr = prev_row[MainColumn::ST0.main_index()];
            accesses = vec![
                (mem_ptr + bfe!(0), curr_row[MainColumn::ST1.main_index()]),
                (mem_ptr + bfe!(1), curr_row[MainColumn::ST2.main_index()]),
                (mem_ptr + bfe!(2), curr_row[MainColumn::ST3.main_index()]),
                (mem_ptr + bfe!(3), curr_row[MainColumn::ST4.main_index()]),
                (mem_ptr + bfe!(4), prev_row[MainColumn::HV0.main_index()]),
                (mem_ptr + bfe!(5), prev_row[MainColumn::HV1.main_index()]),
                (mem_ptr + bfe!(6), prev_row[MainColumn::HV2.main_index()]),
                (mem_ptr + bfe!(7), prev_row[MainColumn::HV3.main_index()]),
                (mem_ptr + bfe!(8), prev_row[MainColumn::HV4.main_index()]),
                (mem_ptr + bfe!(9), prev_row[MainColumn::HV5.main_index()]),
            ];
        }
        Instruction::MerkleStepMem => {
            let mem_ptr = prev_row[MainColumn::ST7.main_index()];
            accesses = vec![
                (mem_ptr + bfe!(0), prev_row[MainColumn::HV0.main_index()]),
                (mem_ptr + bfe!(1), prev_row[MainColumn::HV1.main_index()]),
                (mem_ptr + bfe!(2), prev_row[MainColumn::HV2.main_index()]),
                (mem_ptr + bfe!(3), prev_row[MainColumn::HV3.main_index()]),
                (mem_ptr + bfe!(4), prev_row[MainColumn::HV4.main_index()]),
            ];
        }
        Instruction::XxDotStep => {
            let rhs_ptr = prev_row[MainColumn::ST0.main_index()];
            let lhs_ptr = prev_row[MainColumn::ST1.main_index()];
            accesses = vec![
                (rhs_ptr + bfe!(0), prev_row[MainColumn::HV0.main_index()]),
                (rhs_ptr + bfe!(1), prev_row[MainColumn::HV1.main_index()]),
                (rhs_ptr + bfe!(2), prev_row[MainColumn::HV2.main_index()]),
                (lhs_ptr + bfe!(0), prev_row[MainColumn::HV3.main_index()]),
                (lhs_ptr + bfe!(1), prev_row[MainColumn::HV4.main_index()]),
                (lhs_ptr + bfe!(2), prev_row[MainColumn::HV5.main_index()]),
            ];
        }
        Instruction::XbDotStep => {
            let rhs_ptr = prev_row[MainColumn::ST0.main_index()];
            let lhs_ptr = prev_row[MainColumn::ST1.main_index()];
            accesses = vec![
                (rhs_ptr + bfe!(0), prev_row[MainColumn::HV0.main_index()]),
                (lhs_ptr + bfe!(0), prev_row[MainColumn::HV1.main_index()]),
                (lhs_ptr + bfe!(1), prev_row[MainColumn::HV2.main_index()]),
                (lhs_ptr + bfe!(2), prev_row[MainColumn::HV3.main_index()]),
            ];
        }
        _ => unreachable!(),
    };

    accesses
        .into_iter()
        .map(|(ramp, ramv)| {
            clk * challenges[ChallengeId::RamClkWeight]
                + instruction_type * challenges[ChallengeId::RamInstructionTypeWeight]
                + ramp * challenges[ChallengeId::RamPointerWeight]
                + ramv * challenges[ChallengeId::RamValueWeight]
        })
        .map(|compressed_row| challenges[ChallengeId::RamIndeterminate] - compressed_row)
        .reduce(|l, r| l * r)
}

fn offset_ram_pointer(
    instruction: Instruction,
    row_with_longer_stack: ArrayView1<BFieldElement>,
    ram_pointer_offset: usize,
) -> BFieldElement {
    let ram_pointer = row_with_longer_stack[MainColumn::ST0.main_index()];
    let offset = bfe!(ram_pointer_offset as u64);

    match instruction {
        // adjust for ram_pointer pointing in front of last-read address:
        // `push 0 read_mem 1` leaves stack as `_ a -1`, with `a` read from
        // address 0.
        Instruction::ReadMem(_) => ram_pointer + offset + bfe!(1),
        Instruction::WriteMem(_) => ram_pointer + offset,
        _ => unreachable!(),
    }
}

fn instruction_from_row(row: ArrayView1<BFieldElement>) -> Option<Instruction> {
    let opcode = row[MainColumn::CI.main_index()];
    let instruction = Instruction::try_from(opcode).ok()?;

    if instruction.arg().is_some() {
        let arg = row[MainColumn::NIA.main_index()];
        return instruction.change_arg(arg).ok();
    }

    Some(instruction)
}

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
pub(crate) mod tests {
    use std::collections::HashMap;
    use std::ops::Add;

    use assert2::assert;
    use constraint_circuit::ConstraintCircuitBuilder;
    use isa::instruction::Instruction;
    use isa::op_stack::NumberOfWords;
    use isa::op_stack::OpStackElement;
    use isa::program::Program;
    use isa::triton_asm;
    use isa::triton_program;
    use ndarray::Array2;
    use proptest::collection::vec;
    use proptest_arbitrary_interop::arb;
    use rand::distr::Distribution;
    use rand::distr::StandardUniform;
    use strum::IntoEnumIterator;
    use test_strategy::proptest;

    use crate::NonDeterminism;
    use crate::error::InstructionError::DivisionByZero;
    use crate::shared_tests::TestableProgram;
    use crate::table::master_table::MasterTable;

    use super::*;

    const MAIN_WIDTH: usize = <ProcessorTable as air::AIR>::MainColumn::COUNT;

    /// Does printing recurse infinitely?
    #[test]
    fn print_simple_processor_table_row() {
        let program = triton_program!(push 2 sponge_init assert halt);
        let err = TestableProgram::new(program).run().unwrap_err();
        let state = err.vm_state;
        println!("\n{state}");
    }

    #[derive(Debug, Clone)]
    struct TestRows {
        pub challenges: Challenges,
        pub consecutive_master_main_table_rows: Array2<BFieldElement>,
        pub consecutive_master_aux_table_rows: Array2<XFieldElement>,
    }

    #[derive(Debug, Clone)]
    struct TestRowsDebugInfo {
        pub instruction: Instruction,
        pub debug_cols_curr_row: Vec<MainColumn>,
        pub debug_cols_next_row: Vec<MainColumn>,
    }

    fn test_row_from_program(program: Program, row_num: usize) -> TestRows {
        test_row_from_testable_program(TestableProgram::new(program), row_num)
    }

    fn test_row_from_testable_program(program: TestableProgram, row_num: usize) -> TestRows {
        fn slice_out_2_rows<FF>(table: impl MasterTable<Field = FF>, row_num: usize) -> Array2<FF>
        where
            FF: Clone,
            StandardUniform: Distribution<FF>,
            XFieldElement: Add<FF, Output = XFieldElement>,
        {
            table
                .trace_table()
                .slice(s![row_num..=row_num + 1, ..])
                .to_owned()
        }

        let artifacts = program.generate_proof_artifacts();
        let consecutive_master_main_table_rows =
            slice_out_2_rows(artifacts.master_main_table, row_num);
        let consecutive_master_aux_table_rows =
            slice_out_2_rows(artifacts.master_aux_table, row_num);

        TestRows {
            challenges: artifacts.challenges,
            consecutive_master_main_table_rows,
            consecutive_master_aux_table_rows,
        }
    }

    fn assert_constraints_for_rows_with_debug_info(
        test_rows: &[TestRows],
        debug_info: TestRowsDebugInfo,
    ) {
        let instruction = debug_info.instruction;
        let circuit_builder = ConstraintCircuitBuilder::new();
        let transition_constraints = air::table::processor::transition_constraints_for_instruction(
            &circuit_builder,
            instruction,
        );

        for (case_idx, rows) in test_rows.iter().enumerate() {
            let curr_row = rows.consecutive_master_main_table_rows.slice(s![0, ..]);
            let next_row = rows.consecutive_master_main_table_rows.slice(s![1, ..]);

            println!("Testing all constraints of {instruction} for test case {case_idx}…");
            for &c in &debug_info.debug_cols_curr_row {
                print!("{c}  = {}, ", curr_row[c.master_main_index()]);
            }
            println!();
            for &c in &debug_info.debug_cols_next_row {
                print!("{c}' = {}, ", next_row[c.master_main_index()]);
            }
            println!();

            assert!(
                instruction.opcode_b() == curr_row[MainColumn::CI.master_main_index()],
                "The test is trying to check the wrong transition constraint polynomials."
            );

            for (constraint_idx, constraint) in transition_constraints.iter().enumerate() {
                let evaluation_result = constraint.clone().consume().evaluate(
                    rows.consecutive_master_main_table_rows.view(),
                    rows.consecutive_master_aux_table_rows.view(),
                    &rows.challenges.challenges,
                );
                assert!(
                    evaluation_result.is_zero(),
                    "For case {case_idx}, transition constraint polynomial with \
                    index {constraint_idx} must evaluate to zero. Got {evaluation_result} instead.",
                );
            }
        }
    }

    #[proptest(cases = 20)]
    fn transition_constraints_for_instruction_pop_n(#[strategy(arb())] n: NumberOfWords) {
        let program = triton_program!(push 1 push 2 push 3 push 4 push 5 pop {n} halt);

        let test_rows = [test_row_from_program(program, 5)];
        let debug_info = TestRowsDebugInfo {
            instruction: Instruction::Pop(n),
            debug_cols_curr_row: vec![MainColumn::ST0, MainColumn::ST1, MainColumn::ST2],
            debug_cols_next_row: vec![MainColumn::ST0, MainColumn::ST1, MainColumn::ST2],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_push() {
        let test_rows = [test_row_from_program(triton_program!(push 1 halt), 0)];

        let debug_info = TestRowsDebugInfo {
            instruction: Instruction::Push(bfe!(1)),
            debug_cols_curr_row: vec![MainColumn::ST0, MainColumn::ST1, MainColumn::ST2],
            debug_cols_next_row: vec![MainColumn::ST0, MainColumn::ST1, MainColumn::ST2],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[proptest(cases = 20)]
    fn transition_constraints_for_instruction_divine_n(#[strategy(arb())] n: NumberOfWords) {
        let program = triton_program! { divine {n} halt };

        let non_determinism = (1..=16).map(|b| bfe!(b)).collect_vec();
        let program = TestableProgram::new(program).with_non_determinism(non_determinism);
        let test_rows = [test_row_from_testable_program(program, 0)];
        let debug_info = TestRowsDebugInfo {
            instruction: Instruction::Divine(n),
            debug_cols_curr_row: vec![MainColumn::ST0, MainColumn::ST1, MainColumn::ST2],
            debug_cols_next_row: vec![MainColumn::ST0, MainColumn::ST1, MainColumn::ST2],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_pick() {
        let set_up_stack = (0..OpStackElement::COUNT)
            .rev()
            .flat_map(|i| triton_asm!(push { i }))
            .collect_vec();
        let test_rows = (0..OpStackElement::COUNT)
            .map(|i| triton_program!({&set_up_stack} pick {i} push {i} eq assert halt))
            .map(|program| test_row_from_program(program, 16))
            .collect_vec();

        let debug_info = TestRowsDebugInfo {
            instruction: Instruction::Pick(OpStackElement::ST0),
            debug_cols_curr_row: vec![MainColumn::ST0, MainColumn::ST1, MainColumn::ST2],
            debug_cols_next_row: vec![MainColumn::ST0, MainColumn::ST1, MainColumn::ST2],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_place() {
        let test_rows = (0..OpStackElement::COUNT)
            .map(|i| triton_program!(push 42 place {i} dup {i} push 42 eq assert halt))
            .map(|program| test_row_from_program(program, 1))
            .collect_vec();

        let debug_info = TestRowsDebugInfo {
            instruction: Instruction::Place(OpStackElement::ST0),
            debug_cols_curr_row: vec![MainColumn::ST0, MainColumn::ST1, MainColumn::ST2],
            debug_cols_next_row: vec![MainColumn::ST0, MainColumn::ST1, MainColumn::ST2],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_dup() {
        let programs = [
            triton_program!(dup  0 halt),
            triton_program!(dup  1 halt),
            triton_program!(dup  2 halt),
            triton_program!(dup  3 halt),
            triton_program!(dup  4 halt),
            triton_program!(dup  5 halt),
            triton_program!(dup  6 halt),
            triton_program!(dup  7 halt),
            triton_program!(dup  8 halt),
            triton_program!(dup  9 halt),
            triton_program!(dup 10 halt),
            triton_program!(dup 11 halt),
            triton_program!(dup 12 halt),
            triton_program!(dup 13 halt),
            triton_program!(dup 14 halt),
            triton_program!(dup 15 halt),
        ];
        let test_rows = programs.map(|program| test_row_from_program(program, 0));

        let debug_info = TestRowsDebugInfo {
            instruction: Instruction::Dup(OpStackElement::ST0),
            debug_cols_curr_row: vec![MainColumn::ST0, MainColumn::ST1, MainColumn::ST2],
            debug_cols_next_row: vec![MainColumn::ST0, MainColumn::ST1, MainColumn::ST2],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_swap() {
        let test_rows = (0..OpStackElement::COUNT)
            .map(|i| triton_program!(swap {i} halt))
            .map(|program| test_row_from_program(program, 0))
            .collect_vec();
        let debug_info = TestRowsDebugInfo {
            instruction: Instruction::Swap(OpStackElement::ST0),
            debug_cols_curr_row: vec![MainColumn::ST0, MainColumn::ST1, MainColumn::ST2],
            debug_cols_next_row: vec![MainColumn::ST0, MainColumn::ST1, MainColumn::ST2],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_skiz() {
        let programs = [
            triton_program!(push 1 skiz halt),        // ST0 is non-zero
            triton_program!(push 0 skiz assert halt), // ST0 is zero, next instruction of size 1
            triton_program!(push 0 skiz push 1 halt), // ST0 is zero, next instruction of size 2
        ];
        let test_rows = programs.map(|program| test_row_from_program(program, 1));
        let debug_info = TestRowsDebugInfo {
            instruction: Instruction::Skiz,
            debug_cols_curr_row: vec![MainColumn::IP, MainColumn::NIA, MainColumn::ST0],
            debug_cols_next_row: vec![MainColumn::IP],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_call() {
        let programs = [triton_program!(call label label: halt)];
        let test_rows = programs.map(|program| test_row_from_program(program, 0));
        let debug_cols = [
            MainColumn::IP,
            MainColumn::CI,
            MainColumn::NIA,
            MainColumn::JSP,
            MainColumn::JSO,
            MainColumn::JSD,
        ];
        let debug_info = TestRowsDebugInfo {
            instruction: Instruction::Call(BFieldElement::default()),
            debug_cols_curr_row: debug_cols.to_vec(),
            debug_cols_next_row: debug_cols.to_vec(),
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_return() {
        let programs = [triton_program!(call label halt label: return)];
        let test_rows = programs.map(|program| test_row_from_program(program, 1));
        let debug_cols = [
            MainColumn::IP,
            MainColumn::JSP,
            MainColumn::JSO,
            MainColumn::JSD,
        ];
        let debug_info = TestRowsDebugInfo {
            instruction: Instruction::Return,
            debug_cols_curr_row: debug_cols.to_vec(),
            debug_cols_next_row: debug_cols.to_vec(),
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_recurse() {
        let programs =
            [triton_program!(push 2 call label halt label: push -1 add dup 0 skiz recurse return)];
        let test_rows = programs.map(|program| test_row_from_program(program, 6));
        let debug_cols = [
            MainColumn::IP,
            MainColumn::JSP,
            MainColumn::JSO,
            MainColumn::JSD,
        ];
        let debug_info = TestRowsDebugInfo {
            instruction: Instruction::Recurse,
            debug_cols_curr_row: debug_cols.to_vec(),
            debug_cols_next_row: debug_cols.to_vec(),
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_recurse_or_return() {
        let program = triton_program! {
            push 2 swap 6
            call loop halt
            loop:
                swap 5 push 1 add swap 5
                recurse_or_return
        };
        let test_rows = [
            test_row_from_program(program.clone(), 7), // recurse
            test_row_from_program(program, 12),        // return
        ];
        let debug_info = TestRowsDebugInfo {
            instruction: Instruction::RecurseOrReturn,
            debug_cols_curr_row: vec![
                MainColumn::IP,
                MainColumn::JSP,
                MainColumn::JSO,
                MainColumn::JSD,
                MainColumn::ST5,
                MainColumn::ST6,
                MainColumn::HV4,
            ],
            debug_cols_next_row: vec![
                MainColumn::IP,
                MainColumn::JSP,
                MainColumn::JSO,
                MainColumn::JSD,
            ],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_read_mem() {
        let programs = [
            triton_program!(push 1 read_mem 1 push 0 eq assert assert halt),
            triton_program!(push 2 read_mem 2 push 0 eq assert swap 1 push 2 eq assert halt),
            triton_program!(push 3 read_mem 3 push 0 eq assert swap 2 push 3 eq assert halt),
            triton_program!(push 4 read_mem 4 push 0 eq assert swap 3 push 4 eq assert halt),
            triton_program!(push 5 read_mem 5 push 0 eq assert swap 4 push 5 eq assert halt),
        ];
        let initial_ram = (1..=5)
            .map(|i| (bfe!(i), bfe!(i)))
            .collect::<HashMap<_, _>>();
        let non_determinism = NonDeterminism::default().with_ram(initial_ram);
        let programs_with_input = programs.map(|program| {
            TestableProgram::new(program).with_non_determinism(non_determinism.clone())
        });
        let test_rows = programs_with_input.map(|p_w_i| test_row_from_testable_program(p_w_i, 1));
        let debug_info = TestRowsDebugInfo {
            instruction: Instruction::ReadMem(NumberOfWords::N1),
            debug_cols_curr_row: vec![MainColumn::ST0, MainColumn::ST1],
            debug_cols_next_row: vec![MainColumn::ST0, MainColumn::ST1],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_write_mem() {
        let push_10_elements = triton_asm![push 2; 10];
        let programs = [
            triton_program!({&push_10_elements} write_mem 1 halt),
            triton_program!({&push_10_elements} write_mem 2 halt),
            triton_program!({&push_10_elements} write_mem 3 halt),
            triton_program!({&push_10_elements} write_mem 4 halt),
            triton_program!({&push_10_elements} write_mem 5 halt),
        ];
        let test_rows = programs.map(|program| test_row_from_program(program, 10));
        let debug_info = TestRowsDebugInfo {
            instruction: Instruction::WriteMem(NumberOfWords::N1),
            debug_cols_curr_row: vec![MainColumn::ST0, MainColumn::ST1],
            debug_cols_next_row: vec![MainColumn::ST0, MainColumn::ST1],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_merkle_step() {
        let programs = [
            triton_program!(push 2 swap 5 merkle_step halt),
            triton_program!(push 3 swap 5 merkle_step halt),
        ];
        let dummy_digest = Digest::new(bfe_array![1, 2, 3, 4, 5]);
        let non_determinism = NonDeterminism::default().with_digests(vec![dummy_digest]);
        let programs_with_input = programs.map(|program| {
            TestableProgram::new(program).with_non_determinism(non_determinism.clone())
        });
        let test_rows = programs_with_input.map(|p_w_i| test_row_from_testable_program(p_w_i, 2));

        let debug_info = TestRowsDebugInfo {
            instruction: Instruction::MerkleStep,
            debug_cols_curr_row: vec![],
            debug_cols_next_row: vec![],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_merkle_step_mem() {
        let sibling_digest = bfe_array![1, 2, 3, 4, 5];
        let acc_digest = bfe_array![11, 12, 13, 14, 15];
        let test_program = |node_index: u32| {
            triton_program! {
                push 42             // RAM pointer
                push  1             // dummy
                push {node_index}
                push {acc_digest[0]}
                push {acc_digest[1]}
                push {acc_digest[2]}
                push {acc_digest[3]}
                push {acc_digest[4]}
                merkle_step_mem
                halt
            }
        };
        let mut ram = HashMap::new();
        ram.insert(bfe!(42), sibling_digest[0]);
        ram.insert(bfe!(43), sibling_digest[1]);
        ram.insert(bfe!(44), sibling_digest[2]);
        ram.insert(bfe!(45), sibling_digest[3]);
        ram.insert(bfe!(46), sibling_digest[4]);
        let non_determinism = NonDeterminism::default().with_ram(ram);

        let node_indices = [2, 3];
        let test_rows = node_indices
            .map(test_program)
            .map(TestableProgram::new)
            .map(|p| p.with_non_determinism(non_determinism.clone()))
            .map(|p| test_row_from_testable_program(p, 8));

        let debug_info = TestRowsDebugInfo {
            instruction: Instruction::MerkleStepMem,
            debug_cols_curr_row: vec![],
            debug_cols_next_row: vec![],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_sponge_init() {
        let programs = [triton_program!(sponge_init halt)];
        let test_rows = programs.map(|program| test_row_from_program(program, 0));
        let debug_info = TestRowsDebugInfo {
            instruction: Instruction::SpongeInit,
            debug_cols_curr_row: vec![],
            debug_cols_next_row: vec![],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_sponge_absorb() {
        let push_10_zeros = triton_asm![push 0; 10];
        let push_10_ones = triton_asm![push 1; 10];
        let programs = [
            triton_program!(sponge_init {&push_10_zeros} sponge_absorb halt),
            triton_program!(sponge_init {&push_10_ones} sponge_absorb halt),
        ];
        let test_rows = programs.map(|program| test_row_from_program(program, 11));
        let debug_info = TestRowsDebugInfo {
            instruction: Instruction::SpongeAbsorb,
            debug_cols_curr_row: vec![],
            debug_cols_next_row: vec![],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_sponge_absorb_mem() {
        let programs = [triton_program!(sponge_init push 0 sponge_absorb_mem halt)];
        let test_rows = programs.map(|program| test_row_from_program(program, 2));
        let debug_info = TestRowsDebugInfo {
            instruction: Instruction::SpongeAbsorbMem,
            debug_cols_curr_row: vec![],
            debug_cols_next_row: vec![],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_sponge_squeeze() {
        let programs = [triton_program!(sponge_init sponge_squeeze halt)];
        let test_rows = programs.map(|program| test_row_from_program(program, 1));
        let debug_info = TestRowsDebugInfo {
            instruction: Instruction::SpongeSqueeze,
            debug_cols_curr_row: vec![],
            debug_cols_next_row: vec![],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_eq() {
        let programs = [
            triton_program!(push 3 push 3 eq assert halt),
            triton_program!(push 3 push 2 eq push 0 eq assert halt),
        ];
        let test_rows = programs.map(|program| test_row_from_program(program, 2));
        let debug_info = TestRowsDebugInfo {
            instruction: Instruction::Eq,
            debug_cols_curr_row: vec![MainColumn::ST0, MainColumn::ST1, MainColumn::HV0],
            debug_cols_next_row: vec![MainColumn::ST0],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_split() {
        let programs = [
            triton_program!(push -1 split halt),
            triton_program!(push  0 split halt),
            triton_program!(push  1 split halt),
            triton_program!(push  2 split halt),
            triton_program!(push  3 split halt),
            // test pushing push 2^32 +- 1
            triton_program!(push 4294967295 split halt),
            triton_program!(push 4294967296 split halt),
            triton_program!(push 4294967297 split halt),
        ];
        let test_rows = programs.map(|program| test_row_from_program(program, 1));
        let debug_cols = [MainColumn::ST0, MainColumn::ST1, MainColumn::HV0];
        let debug_info = TestRowsDebugInfo {
            instruction: Instruction::Split,
            debug_cols_curr_row: debug_cols.to_vec(),
            debug_cols_next_row: debug_cols.to_vec(),
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_lt() {
        let programs = [
            triton_program!(push   3 push   3 lt push 0 eq assert halt),
            triton_program!(push   3 push   2 lt push 1 eq assert halt),
            triton_program!(push   2 push   3 lt push 0 eq assert halt),
            triton_program!(push 512 push 513 lt push 0 eq assert halt),
        ];
        let test_rows = programs.map(|program| test_row_from_program(program, 2));
        let debug_info = TestRowsDebugInfo {
            instruction: Instruction::Lt,
            debug_cols_curr_row: vec![MainColumn::ST0, MainColumn::ST1],
            debug_cols_next_row: vec![MainColumn::ST0],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_and() {
        let test_rows = [test_row_from_program(
            triton_program!(push 5 push 12 and push 4 eq assert halt),
            2,
        )];
        let debug_info = TestRowsDebugInfo {
            instruction: Instruction::And,
            debug_cols_curr_row: vec![MainColumn::ST0, MainColumn::ST1],
            debug_cols_next_row: vec![MainColumn::ST0],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_xor() {
        let test_rows = [test_row_from_program(
            triton_program!(push 5 push 12 xor push 9 eq assert halt),
            2,
        )];
        let debug_info = TestRowsDebugInfo {
            instruction: Instruction::Xor,
            debug_cols_curr_row: vec![MainColumn::ST0, MainColumn::ST1],
            debug_cols_next_row: vec![MainColumn::ST0],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_log2floor() {
        let programs = [
            triton_program!(push  1 log_2_floor push  0 eq assert halt),
            triton_program!(push  2 log_2_floor push  1 eq assert halt),
            triton_program!(push  3 log_2_floor push  1 eq assert halt),
            triton_program!(push  4 log_2_floor push  2 eq assert halt),
            triton_program!(push  5 log_2_floor push  2 eq assert halt),
            triton_program!(push  6 log_2_floor push  2 eq assert halt),
            triton_program!(push  7 log_2_floor push  2 eq assert halt),
            triton_program!(push  8 log_2_floor push  3 eq assert halt),
            triton_program!(push  9 log_2_floor push  3 eq assert halt),
            triton_program!(push 10 log_2_floor push  3 eq assert halt),
            triton_program!(push 11 log_2_floor push  3 eq assert halt),
            triton_program!(push 12 log_2_floor push  3 eq assert halt),
            triton_program!(push 13 log_2_floor push  3 eq assert halt),
            triton_program!(push 14 log_2_floor push  3 eq assert halt),
            triton_program!(push 15 log_2_floor push  3 eq assert halt),
            triton_program!(push 16 log_2_floor push  4 eq assert halt),
            triton_program!(push 17 log_2_floor push  4 eq assert halt),
        ];

        let test_rows = programs.map(|program| test_row_from_program(program, 1));
        let debug_info = TestRowsDebugInfo {
            instruction: Instruction::Log2Floor,
            debug_cols_curr_row: vec![MainColumn::ST0, MainColumn::ST1],
            debug_cols_next_row: vec![MainColumn::ST0],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_pow() {
        let programs = [
            triton_program!(push 0 push  0 pow push   1 eq assert halt),
            triton_program!(push 1 push  0 pow push   0 eq assert halt),
            triton_program!(push 2 push  0 pow push   0 eq assert halt),
            triton_program!(push 0 push  1 pow push   1 eq assert halt),
            triton_program!(push 1 push  1 pow push   1 eq assert halt),
            triton_program!(push 2 push  1 pow push   1 eq assert halt),
            triton_program!(push 0 push  2 pow push   1 eq assert halt),
            triton_program!(push 1 push  2 pow push   2 eq assert halt),
            triton_program!(push 2 push  2 pow push   4 eq assert halt),
            triton_program!(push 3 push  2 pow push   8 eq assert halt),
            triton_program!(push 4 push  2 pow push  16 eq assert halt),
            triton_program!(push 5 push  2 pow push  32 eq assert halt),
            triton_program!(push 0 push  3 pow push   1 eq assert halt),
            triton_program!(push 1 push  3 pow push   3 eq assert halt),
            triton_program!(push 2 push  3 pow push   9 eq assert halt),
            triton_program!(push 3 push  3 pow push  27 eq assert halt),
            triton_program!(push 4 push  3 pow push  81 eq assert halt),
            triton_program!(push 0 push 17 pow push   1 eq assert halt),
            triton_program!(push 1 push 17 pow push  17 eq assert halt),
            triton_program!(push 2 push 17 pow push 289 eq assert halt),
        ];

        let test_rows = programs.map(|program| test_row_from_program(program, 2));
        let debug_info = TestRowsDebugInfo {
            instruction: Instruction::Pow,
            debug_cols_curr_row: vec![MainColumn::ST0, MainColumn::ST1],
            debug_cols_next_row: vec![MainColumn::ST0],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_div_mod() {
        let programs = [
            triton_program!(push 2 push 3 div_mod push 1 eq assert push 1 eq assert halt),
            triton_program!(push 3 push 7 div_mod push 1 eq assert push 2 eq assert halt),
            triton_program!(push 4 push 7 div_mod push 3 eq assert push 1 eq assert halt),
        ];
        let test_rows = programs.map(|program| test_row_from_program(program, 2));
        let debug_info = TestRowsDebugInfo {
            instruction: Instruction::DivMod,
            debug_cols_curr_row: vec![MainColumn::ST0, MainColumn::ST1],
            debug_cols_next_row: vec![MainColumn::ST0, MainColumn::ST1],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn division_by_zero_is_impossible() {
        let program = TestableProgram::new(triton_program! { div_mod });
        let err = program.run().unwrap_err();
        assert_eq!(DivisionByZero, err.source);
    }

    #[test]
    fn transition_constraints_for_instruction_xx_add() {
        let programs = [
            triton_program!(push 5 push 6 push 7 push 8 push 9 push 10 xx_add halt),
            triton_program!(push 2 push 3 push 4 push -2 push -3 push -4 xx_add halt),
        ];
        let test_rows = programs.map(|program| test_row_from_program(program, 6));
        let debug_cols = [
            MainColumn::ST0,
            MainColumn::ST1,
            MainColumn::ST2,
            MainColumn::ST3,
            MainColumn::ST4,
            MainColumn::ST5,
        ];
        let debug_info = TestRowsDebugInfo {
            instruction: Instruction::XxAdd,
            debug_cols_curr_row: debug_cols.to_vec(),
            debug_cols_next_row: debug_cols.to_vec(),
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_xx_mul() {
        let programs = [
            triton_program!(push 5 push 6 push 7 push 8 push 9 push 10 xx_mul halt),
            triton_program!(push 2 push 3 push 4 push -2 push -3 push -4 xx_mul halt),
        ];
        let test_rows = programs.map(|program| test_row_from_program(program, 6));
        let debug_cols = [
            MainColumn::ST0,
            MainColumn::ST1,
            MainColumn::ST2,
            MainColumn::ST3,
            MainColumn::ST4,
            MainColumn::ST5,
        ];
        let debug_info = TestRowsDebugInfo {
            instruction: Instruction::XxMul,
            debug_cols_curr_row: debug_cols.to_vec(),
            debug_cols_next_row: debug_cols.to_vec(),
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_x_invert() {
        let programs = [
            triton_program!(push 5 push 6 push 7 x_invert halt),
            triton_program!(push -2 push -3 push -4 x_invert halt),
        ];
        let test_rows = programs.map(|program| test_row_from_program(program, 3));
        let debug_cols = [MainColumn::ST0, MainColumn::ST1, MainColumn::ST2];
        let debug_info = TestRowsDebugInfo {
            instruction: Instruction::XInvert,
            debug_cols_curr_row: debug_cols.to_vec(),
            debug_cols_next_row: debug_cols.to_vec(),
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_xb_mul() {
        let programs = [
            triton_program!(push 5 push 6 push 7 push 2 xb_mul halt),
            triton_program!(push 2 push 3 push 4 push -2 xb_mul halt),
        ];
        let test_rows = programs.map(|program| test_row_from_program(program, 4));
        let debug_cols = [
            MainColumn::ST0,
            MainColumn::ST1,
            MainColumn::ST2,
            MainColumn::ST3,
            MainColumn::OpStackPointer,
            MainColumn::HV0,
        ];
        let debug_info = TestRowsDebugInfo {
            instruction: Instruction::XbMul,
            debug_cols_curr_row: debug_cols.to_vec(),
            debug_cols_next_row: debug_cols.to_vec(),
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[proptest(cases = 20)]
    fn transition_constraints_for_instruction_read_io_n(#[strategy(arb())] n: NumberOfWords) {
        let program = triton_program! {read_io {n} halt};

        let public_input = (1..=16).map(|i| bfe!(i)).collect_vec();
        let program = TestableProgram::new(program).with_input(public_input);
        let test_rows = [test_row_from_testable_program(program, 0)];
        let debug_cols = [MainColumn::ST0, MainColumn::ST1, MainColumn::ST2];
        let debug_info = TestRowsDebugInfo {
            instruction: Instruction::ReadIo(n),
            debug_cols_curr_row: debug_cols.to_vec(),
            debug_cols_next_row: debug_cols.to_vec(),
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[proptest(cases = 20)]
    fn transition_constraints_for_instruction_write_io_n(#[strategy(arb())] n: NumberOfWords) {
        let program = triton_program! {divine 5 write_io {n} halt};

        let non_determinism = (1..=16).map(|b| bfe!(b)).collect_vec();
        let program = TestableProgram::new(program).with_non_determinism(non_determinism);
        let test_rows = [test_row_from_testable_program(program, 1)];
        let debug_cols = [
            MainColumn::ST0,
            MainColumn::ST1,
            MainColumn::ST2,
            MainColumn::ST3,
            MainColumn::ST4,
            MainColumn::ST5,
        ];
        let debug_info = TestRowsDebugInfo {
            instruction: Instruction::WriteIo(n),
            debug_cols_curr_row: debug_cols.to_vec(),
            debug_cols_next_row: debug_cols.to_vec(),
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_xb_dot_step() {
        let program = triton_program! {
            push 10 push 20 push 30     // accumulator `[30, 20, 10]`
            push 96                     // pointer to extension-field element `[3, 5, 7]`
            push 42                     // pointer to base-field element `2`
            xb_dot_step
            push 43 eq assert
            push 99 eq assert
            push {30 + 2 * 3} eq assert
            push {20 + 2 * 5} eq assert
            push {10 + 2 * 7} eq assert
            halt
        };

        let mut ram = HashMap::new();
        ram.insert(bfe!(42), bfe!(2));
        ram.insert(bfe!(96), bfe!(3));
        ram.insert(bfe!(97), bfe!(5));
        ram.insert(bfe!(98), bfe!(7));
        let non_determinism = NonDeterminism::default().with_ram(ram);
        let program = TestableProgram::new(program).with_non_determinism(non_determinism);
        let test_rows = [test_row_from_testable_program(program, 5)];
        let debug_info = TestRowsDebugInfo {
            instruction: Instruction::XbDotStep,
            debug_cols_curr_row: vec![
                MainColumn::ST0,
                MainColumn::ST1,
                MainColumn::ST2,
                MainColumn::ST3,
                MainColumn::ST4,
                MainColumn::HV0,
                MainColumn::HV1,
                MainColumn::HV2,
                MainColumn::HV3,
            ],
            debug_cols_next_row: vec![
                MainColumn::ST0,
                MainColumn::ST1,
                MainColumn::ST2,
                MainColumn::ST3,
                MainColumn::ST4,
            ],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn transition_constraints_for_instruction_xx_dot_step() {
        let operand_0 = xfe!([3, 5, 7]);
        let operand_1 = xfe!([11, 13, 17]);
        let product = operand_0 * operand_1;

        let program = triton_program! {
            push 10 push 20 push 30     // accumulator `[30, 20, 10]`
            push 96                     // pointer to `operand_1`
            push 42                     // pointer to `operand_0`
            xx_dot_step
            push 45 eq assert
            push 99 eq assert
            push {bfe!(30) + product.coefficients[0]} eq assert
            push {bfe!(20) + product.coefficients[1]} eq assert
            push {bfe!(10) + product.coefficients[2]} eq assert
            halt
        };

        let mut ram = HashMap::new();
        ram.insert(bfe!(42), operand_0.coefficients[0]);
        ram.insert(bfe!(43), operand_0.coefficients[1]);
        ram.insert(bfe!(44), operand_0.coefficients[2]);
        ram.insert(bfe!(96), operand_1.coefficients[0]);
        ram.insert(bfe!(97), operand_1.coefficients[1]);
        ram.insert(bfe!(98), operand_1.coefficients[2]);
        let non_determinism = NonDeterminism::default().with_ram(ram);
        let program = TestableProgram::new(program).with_non_determinism(non_determinism);
        let test_rows = [test_row_from_testable_program(program, 5)];
        let debug_info = TestRowsDebugInfo {
            instruction: Instruction::XxDotStep,
            debug_cols_curr_row: vec![
                MainColumn::ST0,
                MainColumn::ST1,
                MainColumn::ST2,
                MainColumn::ST3,
                MainColumn::ST4,
                MainColumn::HV0,
                MainColumn::HV1,
                MainColumn::HV2,
                MainColumn::HV3,
                MainColumn::HV4,
                MainColumn::HV5,
            ],
            debug_cols_next_row: vec![
                MainColumn::ST0,
                MainColumn::ST1,
                MainColumn::ST2,
                MainColumn::ST3,
                MainColumn::ST4,
            ],
        };
        assert_constraints_for_rows_with_debug_info(&test_rows, debug_info);
    }

    #[test]
    fn opcode_decomposition_for_skiz_is_unique() {
        let max_value_of_skiz_constraint_for_nia_decomposition =
            (3 << 7) * (3 << 5) * (3 << 3) * (3 << 1) * 2;
        for instruction in Instruction::iter() {
            assert!(
                instruction.opcode() < max_value_of_skiz_constraint_for_nia_decomposition,
                "Opcode for {instruction} is too high."
            );
        }
    }

    #[proptest]
    fn constructing_factor_for_op_stack_table_running_product_never_panics(
        #[strategy(vec(arb(), MAIN_WIDTH))] previous_row: Vec<BFieldElement>,
        #[strategy(vec(arb(), MAIN_WIDTH))] current_row: Vec<BFieldElement>,
        #[strategy(arb())] challenges: Challenges,
    ) {
        let previous_row = Array1::from(previous_row);
        let current_row = Array1::from(current_row);
        let _ = factor_for_op_stack_table_running_product(
            previous_row.view(),
            current_row.view(),
            &challenges,
        );
    }

    #[proptest]
    fn constructing_factor_for_ram_table_running_product_never_panics(
        #[strategy(vec(arb(), MAIN_WIDTH))] previous_row: Vec<BFieldElement>,
        #[strategy(vec(arb(), MAIN_WIDTH))] current_row: Vec<BFieldElement>,
        #[strategy(arb())] challenges: Challenges,
    ) {
        let previous_row = Array1::from(previous_row);
        let current_row = Array1::from(current_row);
        let _ = factor_for_ram_table_running_product(
            previous_row.view(),
            current_row.view(),
            &challenges,
        );
    }
}
