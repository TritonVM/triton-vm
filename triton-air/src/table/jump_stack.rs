use constraint_circuit::ConstraintCircuitBuilder;
use constraint_circuit::ConstraintCircuitMonad;
use constraint_circuit::DualRowIndicator;
use constraint_circuit::DualRowIndicator::CurrentBaseRow;
use constraint_circuit::DualRowIndicator::CurrentExtRow;
use constraint_circuit::DualRowIndicator::NextBaseRow;
use constraint_circuit::DualRowIndicator::NextExtRow;
use constraint_circuit::SingleRowIndicator;
use constraint_circuit::SingleRowIndicator::BaseRow;
use constraint_circuit::SingleRowIndicator::ExtRow;
use isa::instruction::Instruction;
use twenty_first::prelude::BFieldElement;

use crate::challenge_id::ChallengeId::ClockJumpDifferenceLookupIndeterminate;
use crate::challenge_id::ChallengeId::JumpStackCiWeight;
use crate::challenge_id::ChallengeId::JumpStackClkWeight;
use crate::challenge_id::ChallengeId::JumpStackIndeterminate;
use crate::challenge_id::ChallengeId::JumpStackJsdWeight;
use crate::challenge_id::ChallengeId::JumpStackJsoWeight;
use crate::challenge_id::ChallengeId::JumpStackJspWeight;
use crate::cross_table_argument::CrossTableArg;
use crate::cross_table_argument::LookupArg;
use crate::table_column::JumpStackBaseTableColumn::CI;
use crate::table_column::JumpStackBaseTableColumn::CLK;
use crate::table_column::JumpStackBaseTableColumn::JSD;
use crate::table_column::JumpStackBaseTableColumn::JSO;
use crate::table_column::JumpStackBaseTableColumn::JSP;
use crate::table_column::JumpStackExtTableColumn::ClockJumpDifferenceLookupClientLogDerivative;
use crate::table_column::JumpStackExtTableColumn::RunningProductPermArg;
use crate::table_column::MasterBaseTableColumn;
use crate::table_column::MasterExtTableColumn;
use crate::AIR;

#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct JumpStackTable;

impl AIR for JumpStackTable {
    type MainColumn = crate::table_column::JumpStackBaseTableColumn;
    type AuxColumn = crate::table_column::JumpStackExtTableColumn;

    fn initial_constraints(
        circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        let clk = circuit_builder.input(BaseRow(CLK.master_base_table_index()));
        let jsp = circuit_builder.input(BaseRow(JSP.master_base_table_index()));
        let jso = circuit_builder.input(BaseRow(JSO.master_base_table_index()));
        let jsd = circuit_builder.input(BaseRow(JSD.master_base_table_index()));
        let ci = circuit_builder.input(BaseRow(CI.master_base_table_index()));
        let rppa = circuit_builder.input(ExtRow(RunningProductPermArg.master_ext_table_index()));
        let clock_jump_diff_log_derivative = circuit_builder.input(ExtRow(
            ClockJumpDifferenceLookupClientLogDerivative.master_ext_table_index(),
        ));

        let processor_perm_indeterminate = circuit_builder.challenge(JumpStackIndeterminate);
        // note: `clk`, `jsp`, `jso`, and `jsd` are all constrained to be 0 and can thus be omitted.
        let compressed_row = circuit_builder.challenge(JumpStackCiWeight) * ci;
        let rppa_starts_correctly = rppa - (processor_perm_indeterminate - compressed_row);

        // A clock jump difference of 0 is not allowed. Hence, the initial is recorded.
        let clock_jump_diff_log_derivative_starts_correctly = clock_jump_diff_log_derivative
            - circuit_builder.x_constant(LookupArg::default_initial());

        vec![
            clk,
            jsp,
            jso,
            jsd,
            rppa_starts_correctly,
            clock_jump_diff_log_derivative_starts_correctly,
        ]
    }

    fn consistency_constraints(
        _circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        // no further constraints
        vec![]
    }

    fn transition_constraints(
        circuit_builder: &ConstraintCircuitBuilder<DualRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<DualRowIndicator>> {
        let one = || circuit_builder.b_constant(1);
        let call_opcode =
            circuit_builder.b_constant(Instruction::Call(BFieldElement::default()).opcode_b());
        let return_opcode = circuit_builder.b_constant(Instruction::Return.opcode_b());
        let recurse_or_return_opcode =
            circuit_builder.b_constant(Instruction::RecurseOrReturn.opcode_b());

        let clk = circuit_builder.input(CurrentBaseRow(CLK.master_base_table_index()));
        let ci = circuit_builder.input(CurrentBaseRow(CI.master_base_table_index()));
        let jsp = circuit_builder.input(CurrentBaseRow(JSP.master_base_table_index()));
        let jso = circuit_builder.input(CurrentBaseRow(JSO.master_base_table_index()));
        let jsd = circuit_builder.input(CurrentBaseRow(JSD.master_base_table_index()));
        let rppa = circuit_builder.input(CurrentExtRow(
            RunningProductPermArg.master_ext_table_index(),
        ));
        let clock_jump_diff_log_derivative = circuit_builder.input(CurrentExtRow(
            ClockJumpDifferenceLookupClientLogDerivative.master_ext_table_index(),
        ));

        let clk_next = circuit_builder.input(NextBaseRow(CLK.master_base_table_index()));
        let ci_next = circuit_builder.input(NextBaseRow(CI.master_base_table_index()));
        let jsp_next = circuit_builder.input(NextBaseRow(JSP.master_base_table_index()));
        let jso_next = circuit_builder.input(NextBaseRow(JSO.master_base_table_index()));
        let jsd_next = circuit_builder.input(NextBaseRow(JSD.master_base_table_index()));
        let rppa_next =
            circuit_builder.input(NextExtRow(RunningProductPermArg.master_ext_table_index()));
        let clock_jump_diff_log_derivative_next = circuit_builder.input(NextExtRow(
            ClockJumpDifferenceLookupClientLogDerivative.master_ext_table_index(),
        ));

        let jsp_inc_or_stays =
            (jsp_next.clone() - jsp.clone() - one()) * (jsp_next.clone() - jsp.clone());

        let jsp_inc_by_one_or_ci_can_return = (jsp_next.clone() - jsp.clone() - one())
            * (ci.clone() - return_opcode)
            * (ci.clone() - recurse_or_return_opcode);
        let jsp_inc_or_jso_stays_or_ci_can_ret =
            jsp_inc_by_one_or_ci_can_return.clone() * (jso_next.clone() - jso);

        let jsp_inc_or_jsd_stays_or_ci_can_ret =
            jsp_inc_by_one_or_ci_can_return.clone() * (jsd_next.clone() - jsd);

        let jsp_inc_or_clk_inc_or_ci_call_or_ci_can_ret = jsp_inc_by_one_or_ci_can_return
            * (clk_next.clone() - clk.clone() - one())
            * (ci.clone() - call_opcode);

        let compressed_row = circuit_builder.challenge(JumpStackClkWeight) * clk_next.clone()
            + circuit_builder.challenge(JumpStackCiWeight) * ci_next
            + circuit_builder.challenge(JumpStackJspWeight) * jsp_next.clone()
            + circuit_builder.challenge(JumpStackJsoWeight) * jso_next
            + circuit_builder.challenge(JumpStackJsdWeight) * jsd_next;
        let rppa_updates_correctly =
            rppa_next - rppa * (circuit_builder.challenge(JumpStackIndeterminate) - compressed_row);

        let log_derivative_remains =
            clock_jump_diff_log_derivative_next.clone() - clock_jump_diff_log_derivative.clone();
        let clk_diff = clk_next - clk;
        let log_derivative_accumulates = (clock_jump_diff_log_derivative_next
            - clock_jump_diff_log_derivative)
            * (circuit_builder.challenge(ClockJumpDifferenceLookupIndeterminate) - clk_diff)
            - one();
        let log_derivative_updates_correctly = (jsp_next.clone() - jsp.clone() - one())
            * log_derivative_accumulates
            + (jsp_next - jsp) * log_derivative_remains;

        vec![
            jsp_inc_or_stays,
            jsp_inc_or_jso_stays_or_ci_can_ret,
            jsp_inc_or_jsd_stays_or_ci_can_ret,
            jsp_inc_or_clk_inc_or_ci_call_or_ci_can_ret,
            rppa_updates_correctly,
            log_derivative_updates_correctly,
        ]
    }

    fn terminal_constraints(
        _circuit_builder: &ConstraintCircuitBuilder<SingleRowIndicator>,
    ) -> Vec<ConstraintCircuitMonad<SingleRowIndicator>> {
        // no further constraints
        vec![]
    }
}
