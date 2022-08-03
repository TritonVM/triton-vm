use itertools::Itertools;

use super::hash_table::{HashTableChallenges, HashTableEndpoints};
use super::instruction_table::{InstructionTableChallenges, InstructionTableEndpoints};
use super::jump_stack_table::{JumpStackTableChallenges, JumpStackTableEndpoints};
use super::op_stack_table::{OpStackTableChallenges, OpStackTableEndpoints};
use super::processor_table::IOChallenges;
use super::processor_table::{ProcessorTableChallenges, ProcessorTableEndpoints};
use super::program_table::{ProgramTableChallenges, ProgramTableEndpoints};
use super::ram_table::{RamTableChallenges, RamTableEndpoints};
use super::u32_op_table::{U32OpTableChallenges, U32OpTableEndpoints};
use crate::shared_math::b_field_element::BFieldElement;
use crate::shared_math::stark::triton::state::DIGEST_LEN;
use crate::shared_math::x_field_element::XFieldElement;

#[derive(Debug, Clone)]
pub struct AllChallenges {
    pub program_table_challenges: ProgramTableChallenges,
    pub instruction_table_challenges: InstructionTableChallenges,
    pub input_challenges: IOChallenges,
    pub output_challenges: IOChallenges,
    pub processor_table_challenges: ProcessorTableChallenges,
    pub op_stack_table_challenges: OpStackTableChallenges,
    pub ram_table_challenges: RamTableChallenges,
    pub jump_stack_table_challenges: JumpStackTableChallenges,
    pub hash_table_challenges: HashTableChallenges,
    pub u32_op_table_challenges: U32OpTableChallenges,
}

impl AllChallenges {
    pub const TOTAL_CHALLENGES: usize = 127;

    pub fn create_challenges(weights: &[XFieldElement]) -> Self {
        let mut weights = weights.to_vec();

        let processor_table_challenges = ProcessorTableChallenges {
            input_table_eval_row_weight: weights.pop().unwrap(),
            output_table_eval_row_weight: weights.pop().unwrap(),
            to_hash_table_eval_row_weight: weights.pop().unwrap(),
            from_hash_table_eval_row_weight: weights.pop().unwrap(),
            instruction_perm_row_weight: weights.pop().unwrap(),
            op_stack_perm_row_weight: weights.pop().unwrap(),
            ram_perm_row_weight: weights.pop().unwrap(),
            jump_stack_perm_row_weight: weights.pop().unwrap(),
            u32_lt_perm_row_weight: weights.pop().unwrap(),
            u32_and_perm_row_weight: weights.pop().unwrap(),
            u32_xor_perm_row_weight: weights.pop().unwrap(),
            u32_reverse_perm_row_weight: weights.pop().unwrap(),
            u32_div_perm_row_weight: weights.pop().unwrap(),

            instruction_table_ip_weight: weights.pop().unwrap(),
            instruction_table_ci_processor_weight: weights.pop().unwrap(),
            instruction_table_nia_weight: weights.pop().unwrap(),

            op_stack_table_clk_weight: weights.pop().unwrap(),
            op_stack_table_ci_weight: weights.pop().unwrap(),
            op_stack_table_osv_weight: weights.pop().unwrap(),
            op_stack_table_osp_weight: weights.pop().unwrap(),

            ram_table_clk_weight: weights.pop().unwrap(),
            ram_table_ramv_weight: weights.pop().unwrap(),
            ram_table_ramp_weight: weights.pop().unwrap(),

            jump_stack_table_clk_weight: weights.pop().unwrap(),
            jump_stack_table_ci_weight: weights.pop().unwrap(),
            jump_stack_table_jsp_weight: weights.pop().unwrap(),
            jump_stack_table_jso_weight: weights.pop().unwrap(),
            jump_stack_table_jsd_weight: weights.pop().unwrap(),

            hash_table_stack_input_weights: weights
                .drain(0..2 * DIGEST_LEN)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
            hash_table_digest_output_weights: weights
                .drain(0..DIGEST_LEN)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),

            u32_op_table_lt_lhs_weight: weights.pop().unwrap(),
            u32_op_table_lt_rhs_weight: weights.pop().unwrap(),
            u32_op_table_lt_result_weight: weights.pop().unwrap(),
            u32_op_table_and_lhs_weight: weights.pop().unwrap(),
            u32_op_table_and_rhs_weight: weights.pop().unwrap(),
            u32_op_table_and_result_weight: weights.pop().unwrap(),
            u32_op_table_xor_lhs_weight: weights.pop().unwrap(),
            u32_op_table_xor_rhs_weight: weights.pop().unwrap(),
            u32_op_table_xor_result_weight: weights.pop().unwrap(),
            u32_op_table_reverse_lhs_weight: weights.pop().unwrap(),
            u32_op_table_reverse_result_weight: weights.pop().unwrap(),
            u32_op_table_div_divisor_weight: weights.pop().unwrap(),
            u32_op_table_div_remainder_weight: weights.pop().unwrap(),
            u32_op_table_div_result_weight: weights.pop().unwrap(),
        };

        let program_table_challenges = ProgramTableChallenges {
            instruction_eval_row_weight: weights.pop().unwrap(),
            address_weight: weights.pop().unwrap(),
            instruction_weight: weights.pop().unwrap(),
            next_instruction_weight: weights.pop().unwrap(),
        };

        let instruction_table_challenges = InstructionTableChallenges {
            processor_perm_row_weight: processor_table_challenges.instruction_perm_row_weight,
            ip_processor_weight: processor_table_challenges.instruction_table_ip_weight,
            ci_processor_weight: processor_table_challenges.instruction_table_ci_processor_weight,
            nia_processor_weight: processor_table_challenges.instruction_table_nia_weight,

            program_eval_row_weight: program_table_challenges.instruction_eval_row_weight,
            address_weight: program_table_challenges.address_weight,
            instruction_weight: program_table_challenges.instruction_weight,
            next_instruction_weight: program_table_challenges.next_instruction_weight,
        };

        let input_challenges = IOChallenges {
            processor_eval_row_weight: processor_table_challenges.input_table_eval_row_weight,
        };

        let output_challenges = IOChallenges {
            processor_eval_row_weight: processor_table_challenges.output_table_eval_row_weight,
        };

        let op_stack_table_challenges = OpStackTableChallenges {
            processor_perm_row_weight: processor_table_challenges.op_stack_perm_row_weight,
            clk_weight: processor_table_challenges.op_stack_table_clk_weight,
            ci_weight: processor_table_challenges.op_stack_table_ci_weight,
            osv_weight: processor_table_challenges.op_stack_table_osv_weight,
            osp_weight: processor_table_challenges.op_stack_table_osp_weight,
        };

        let ram_table_challenges = RamTableChallenges {
            processor_perm_row_weight: processor_table_challenges.ram_perm_row_weight,
            clk_weight: processor_table_challenges.ram_table_clk_weight,
            ramv_weight: processor_table_challenges.ram_table_ramv_weight,
            ramp_weight: processor_table_challenges.ram_table_ramp_weight,
        };

        let jump_stack_table_challenges = JumpStackTableChallenges {
            processor_perm_row_weight: processor_table_challenges.jump_stack_perm_row_weight,
            clk_weight: processor_table_challenges.jump_stack_table_clk_weight,
            ci_weight: processor_table_challenges.jump_stack_table_ci_weight,
            jsp_weight: processor_table_challenges.jump_stack_table_jsp_weight,
            jso_weight: processor_table_challenges.jump_stack_table_jso_weight,
            jsd_weight: processor_table_challenges.jump_stack_table_jsd_weight,
        };

        let stack_input_weights = weights
            .drain(0..2 * DIGEST_LEN)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let digest_output_weights = weights
            .drain(0..DIGEST_LEN)
            .collect::<Vec<_>>()
            .try_into()
            .unwrap();

        let hash_table_challenges = HashTableChallenges {
            from_processor_eval_row_weight: processor_table_challenges
                .to_hash_table_eval_row_weight,
            to_processor_eval_row_weight: processor_table_challenges
                .from_hash_table_eval_row_weight,

            stack_input_weights,
            digest_output_weights,
        };

        let u32_op_table_challenges = U32OpTableChallenges {
            processor_lt_perm_row_weight: processor_table_challenges.u32_lt_perm_row_weight,
            processor_and_perm_row_weight: processor_table_challenges.u32_and_perm_row_weight,
            processor_xor_perm_row_weight: processor_table_challenges.u32_xor_perm_row_weight,
            processor_reverse_perm_row_weight: processor_table_challenges
                .u32_reverse_perm_row_weight,
            processor_div_perm_row_weight: processor_table_challenges.u32_div_perm_row_weight,

            lt_lhs_weight: processor_table_challenges.u32_op_table_lt_lhs_weight,
            lt_rhs_weight: processor_table_challenges.u32_op_table_lt_rhs_weight,
            lt_result_weight: processor_table_challenges.u32_op_table_lt_result_weight,

            and_lhs_weight: processor_table_challenges.u32_op_table_and_lhs_weight,
            and_rhs_weight: processor_table_challenges.u32_op_table_and_rhs_weight,
            and_result_weight: processor_table_challenges.u32_op_table_and_result_weight,

            xor_lhs_weight: processor_table_challenges.u32_op_table_xor_lhs_weight,
            xor_rhs_weight: processor_table_challenges.u32_op_table_xor_rhs_weight,
            xor_result_weight: processor_table_challenges.u32_op_table_xor_result_weight,

            reverse_lhs_weight: processor_table_challenges.u32_op_table_reverse_lhs_weight,
            reverse_result_weight: processor_table_challenges.u32_op_table_reverse_result_weight,

            div_divisor_weight: processor_table_challenges.u32_op_table_div_divisor_weight,
            div_remainder_weight: processor_table_challenges.u32_op_table_div_remainder_weight,
            div_result_weight: processor_table_challenges.u32_op_table_div_result_weight,
        };

        AllChallenges {
            program_table_challenges,
            instruction_table_challenges,
            input_challenges,
            output_challenges,
            processor_table_challenges,
            op_stack_table_challenges,
            ram_table_challenges,
            jump_stack_table_challenges,
            hash_table_challenges,
            u32_op_table_challenges,
        }
    }

    pub fn dummy() -> Self {
        let zero: XFieldElement = XFieldElement::new_const(0.into());
        let zeros = vec![zero; Self::TOTAL_CHALLENGES];

        Self::create_challenges(&zeros)
    }
}

/// An *endpoint* is the collective term for *initials* and *terminals*.
#[derive(Debug, Clone)]
pub struct AllEndpoints {
    pub program_table_endpoints: ProgramTableEndpoints,
    pub instruction_table_endpoints: InstructionTableEndpoints,
    pub processor_table_endpoints: ProcessorTableEndpoints,
    pub op_stack_table_endpoints: OpStackTableEndpoints,
    pub ram_table_endpoints: RamTableEndpoints,
    pub jump_stack_table_endpoints: JumpStackTableEndpoints,
    pub hash_table_endpoints: HashTableEndpoints,
    pub u32_op_table_endpoints: U32OpTableEndpoints,
}

impl AllEndpoints {
    pub const TOTAL_ENDPOINTS: usize = 14;

    pub fn create_initials(weights: &[XFieldElement]) -> Self {
        let mut weights = weights.to_vec();

        let processor_table_initials = ProcessorTableEndpoints {
            input_table_eval_sum: XFieldElement::ring_zero(),
            output_table_eval_sum: XFieldElement::ring_zero(),
            instruction_table_perm_product: weights.pop().unwrap(),
            opstack_table_perm_product: weights.pop().unwrap(),
            ram_table_perm_product: weights.pop().unwrap(),
            jump_stack_perm_product: weights.pop().unwrap(),
            to_hash_table_eval_sum: weights.pop().unwrap(),
            from_hash_table_eval_sum: weights.pop().unwrap(),
            u32_table_lt_perm_product: weights.pop().unwrap(),
            u32_table_and_perm_product: weights.pop().unwrap(),
            u32_table_xor_perm_product: weights.pop().unwrap(),
            u32_table_reverse_perm_product: weights.pop().unwrap(),
            u32_table_div_perm_product: weights.pop().unwrap(),
        };

        let program_table_initials = ProgramTableEndpoints {
            instruction_eval_sum: weights.pop().unwrap(),
        };

        let instruction_table_initials = InstructionTableEndpoints {
            processor_perm_product: processor_table_initials.instruction_table_perm_product,
            program_eval_sum: program_table_initials.instruction_eval_sum,
        };

        let op_stack_table_initials = OpStackTableEndpoints {
            processor_perm_product: processor_table_initials.opstack_table_perm_product,
        };

        let ram_table_initials = RamTableEndpoints {
            processor_perm_product: processor_table_initials.ram_table_perm_product,
        };

        let jump_stack_table_initials = JumpStackTableEndpoints {
            processor_perm_product: processor_table_initials.jump_stack_perm_product,
        };

        // hash_table.from_processor <-> processor_table.to_hash, and
        // hash_table.to_processor   <-> processor_table.from_hash
        let hash_table_initials = HashTableEndpoints {
            from_processor_eval_sum: processor_table_initials.to_hash_table_eval_sum,
            to_processor_eval_sum: processor_table_initials.from_hash_table_eval_sum,
        };

        let u32_op_table_initials = U32OpTableEndpoints {
            processor_lt_perm_product: processor_table_initials.u32_table_lt_perm_product,
            processor_and_perm_product: processor_table_initials.u32_table_and_perm_product,
            processor_xor_perm_product: processor_table_initials.u32_table_xor_perm_product,
            processor_reverse_perm_product: processor_table_initials.u32_table_reverse_perm_product,
            processor_div_perm_product: processor_table_initials.u32_table_div_perm_product,
        };

        AllEndpoints {
            program_table_endpoints: program_table_initials,
            instruction_table_endpoints: instruction_table_initials,
            processor_table_endpoints: processor_table_initials,
            op_stack_table_endpoints: op_stack_table_initials,
            ram_table_endpoints: ram_table_initials,
            jump_stack_table_endpoints: jump_stack_table_initials,
            hash_table_endpoints: hash_table_initials,
            u32_op_table_endpoints: u32_op_table_initials,
        }
    }
}

/// Make terminals iterable for ProofStream
///
/// In order for `Stark::verify()` to receive all terminals via `ProofStream`,
/// they must serialise to a stream of `BFieldElement`s.
impl IntoIterator for AllEndpoints {
    type Item = BFieldElement;

    type IntoIter = std::vec::IntoIter<BFieldElement>;

    fn into_iter(self) -> Self::IntoIter {
        vec![
            &self.program_table_endpoints.instruction_eval_sum,
            &self.instruction_table_endpoints.processor_perm_product,
            &self.instruction_table_endpoints.program_eval_sum,
            &self
                .processor_table_endpoints
                .instruction_table_perm_product,
            &self.processor_table_endpoints.opstack_table_perm_product,
            &self.processor_table_endpoints.ram_table_perm_product,
            &self.processor_table_endpoints.jump_stack_perm_product,
            &self.processor_table_endpoints.to_hash_table_eval_sum,
            &self.processor_table_endpoints.from_hash_table_eval_sum,
            &self.processor_table_endpoints.u32_table_lt_perm_product,
            &self.processor_table_endpoints.u32_table_and_perm_product,
            &self.processor_table_endpoints.u32_table_xor_perm_product,
            &self
                .processor_table_endpoints
                .u32_table_reverse_perm_product,
            &self.processor_table_endpoints.u32_table_div_perm_product,
            &self.op_stack_table_endpoints.processor_perm_product,
            &self.ram_table_endpoints.processor_perm_product,
            &self.jump_stack_table_endpoints.processor_perm_product,
            &self.hash_table_endpoints.from_processor_eval_sum,
            &self.hash_table_endpoints.to_processor_eval_sum,
            &self.u32_op_table_endpoints.processor_lt_perm_product,
            &self.u32_op_table_endpoints.processor_and_perm_product,
            &self.u32_op_table_endpoints.processor_xor_perm_product,
            &self.u32_op_table_endpoints.processor_reverse_perm_product,
            &self.u32_op_table_endpoints.processor_div_perm_product,
        ]
        .into_iter()
        .map(|endpoint| endpoint.coefficients.to_vec())
        .concat()
        .into_iter()
    }
}

#[cfg(test)]
mod challenges_endpoints_tests {
    use crate::shared_math::stark::triton::table::processor_table;
    use crate::shared_math::stark::triton::table::program_table;

    use super::*;

    #[test]
    fn total_challenges_equal_permutation_and_evaluation_args_test() {
        assert_eq!(
            processor_table::PROCESSOR_TABLE_INITIALS_COUNT
                + program_table::PROGRAM_TABLE_INITIALS_COUNT,
            AllEndpoints::TOTAL_ENDPOINTS,
        );
    }
}
