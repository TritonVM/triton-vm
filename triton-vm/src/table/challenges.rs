use twenty_first::shared_math::other::random_elements;
use twenty_first::shared_math::rescue_prime_regular::DIGEST_LENGTH;
use twenty_first::shared_math::x_field_element::XFieldElement;

use super::hash_table::HashTableChallenges;
use super::instruction_table::InstructionTableChallenges;
use super::jump_stack_table::JumpStackTableChallenges;
use super::op_stack_table::OpStackTableChallenges;
use super::processor_table::IOChallenges;
use super::processor_table::ProcessorTableChallenges;
use super::program_table::ProgramTableChallenges;
use super::ram_table::RamTableChallenges;

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
}

impl AllChallenges {
    pub const TOTAL_CHALLENGES: usize = 128;

    pub fn create_challenges(mut weights: Vec<XFieldElement>) -> Self {
        let processor_table_challenges = ProcessorTableChallenges {
            input_table_eval_row_weight: weights.pop().unwrap(),
            output_table_eval_row_weight: weights.pop().unwrap(),
            to_hash_table_eval_row_weight: weights.pop().unwrap(),
            from_hash_table_eval_row_weight: weights.pop().unwrap(),
            instruction_perm_row_weight: weights.pop().unwrap(),
            op_stack_perm_row_weight: weights.pop().unwrap(),
            ram_perm_row_weight: weights.pop().unwrap(),
            jump_stack_perm_row_weight: weights.pop().unwrap(),

            instruction_table_ip_weight: weights.pop().unwrap(),
            instruction_table_ci_processor_weight: weights.pop().unwrap(),
            instruction_table_nia_weight: weights.pop().unwrap(),

            op_stack_table_clk_weight: weights.pop().unwrap(),
            op_stack_table_ib1_weight: weights.pop().unwrap(),
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
                .drain(0..2 * DIGEST_LENGTH)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
            hash_table_digest_output_weights: weights
                .drain(0..DIGEST_LENGTH)
                .collect::<Vec<_>>()
                .try_into()
                .unwrap(),
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
            ib1_weight: processor_table_challenges.op_stack_table_ib1_weight,
            osv_weight: processor_table_challenges.op_stack_table_osv_weight,
            osp_weight: processor_table_challenges.op_stack_table_osp_weight,
        };

        let ram_table_challenges = RamTableChallenges {
            bezout_relation_sample_point: weights.pop().unwrap(),
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

        let hash_table_challenges = HashTableChallenges {
            from_processor_eval_row_weight: processor_table_challenges
                .to_hash_table_eval_row_weight,
            to_processor_eval_row_weight: processor_table_challenges
                .from_hash_table_eval_row_weight,

            stack_input_weights: processor_table_challenges.hash_table_stack_input_weights,
            digest_output_weights: processor_table_challenges.hash_table_digest_output_weights,
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
        }
    }

    /// Stand-in challenges. Can be used for deriving degree bounds and in tests. For non-
    /// interactive STARKs, use Fiat-Shamir to derive the actual challenges.
    pub fn placeholder() -> Self {
        let random_challenges = random_elements(Self::TOTAL_CHALLENGES);

        Self::create_challenges(random_challenges)
    }
}
