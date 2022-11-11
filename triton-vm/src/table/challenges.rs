use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;
use strum::EnumCount;
use strum::IntoEnumIterator;
use twenty_first::shared_math::other::random_elements;
use twenty_first::shared_math::x_field_element::XFieldElement;

use super::hash_table::HashTableChallenges;
use super::instruction_table::InstructionTableChallenges;
use super::jump_stack_table::JumpStackTableChallenges;
use super::op_stack_table::OpStackTableChallenges;
use super::processor_table::IOChallenges;
use super::processor_table::ProcessorTableChallenges;
use super::program_table::ProgramTableChallenges;
use super::ram_table::RamTableChallenges;

pub trait TableChallenges: Clone + Debug {
    type Id: Display
        + Clone
        + Copy
        + std::fmt::Debug
        + EnumCount
        + IntoEnumIterator
        + Into<usize>
        + PartialEq
        + Eq
        + Hash;

    fn count() -> usize {
        Self::Id::COUNT
    }

    fn get_challenge(&self, id: Self::Id) -> XFieldElement;

    fn to_vec(&self) -> Vec<XFieldElement> {
        let mut ret: Vec<XFieldElement> = vec![];
        for id in Self::Id::iter() {
            ret.push(self.get_challenge(id));
        }

        ret
    }
}

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
    pub const TOTAL_CHALLENGES: usize = 130;

    pub fn create_challenges(mut weights: Vec<XFieldElement>) -> Self {
        let processor_table_challenges = ProcessorTableChallenges {
            standard_input_eval_indeterminate: weights.pop().unwrap(),
            standard_output_eval_indeterminate: weights.pop().unwrap(),
            to_hash_table_eval_indeterminate: weights.pop().unwrap(),
            from_hash_table_eval_indeterminate: weights.pop().unwrap(),
            instruction_perm_indeterminate: weights.pop().unwrap(),
            op_stack_perm_indeterminate: weights.pop().unwrap(),
            ram_perm_indeterminate: weights.pop().unwrap(),
            jump_stack_perm_indeterminate: weights.pop().unwrap(),

            instruction_table_ip_weight: weights.pop().unwrap(),
            instruction_table_ci_processor_weight: weights.pop().unwrap(),
            instruction_table_nia_weight: weights.pop().unwrap(),

            op_stack_table_clk_weight: weights.pop().unwrap(),
            op_stack_table_ib1_weight: weights.pop().unwrap(),
            op_stack_table_osp_weight: weights.pop().unwrap(),
            op_stack_table_osv_weight: weights.pop().unwrap(),

            ram_table_clk_weight: weights.pop().unwrap(),
            ram_table_ramp_weight: weights.pop().unwrap(),
            ram_table_ramv_weight: weights.pop().unwrap(),

            jump_stack_table_clk_weight: weights.pop().unwrap(),
            jump_stack_table_ci_weight: weights.pop().unwrap(),
            jump_stack_table_jsp_weight: weights.pop().unwrap(),
            jump_stack_table_jso_weight: weights.pop().unwrap(),
            jump_stack_table_jsd_weight: weights.pop().unwrap(),

            unique_clock_jump_differences_eval_indeterminate: weights.pop().unwrap(),
            all_clock_jump_differences_multi_perm_indeterminate: weights.pop().unwrap(),

            hash_table_stack_input_weights0: weights.pop().unwrap(),
            hash_table_stack_input_weights1: weights.pop().unwrap(),
            hash_table_stack_input_weights2: weights.pop().unwrap(),
            hash_table_stack_input_weights3: weights.pop().unwrap(),
            hash_table_stack_input_weights4: weights.pop().unwrap(),
            hash_table_stack_input_weights5: weights.pop().unwrap(),
            hash_table_stack_input_weights6: weights.pop().unwrap(),
            hash_table_stack_input_weights7: weights.pop().unwrap(),
            hash_table_stack_input_weights8: weights.pop().unwrap(),
            hash_table_stack_input_weights9: weights.pop().unwrap(),

            hash_table_digest_output_weights0: weights.pop().unwrap(),
            hash_table_digest_output_weights1: weights.pop().unwrap(),
            hash_table_digest_output_weights2: weights.pop().unwrap(),
            hash_table_digest_output_weights3: weights.pop().unwrap(),
            hash_table_digest_output_weights4: weights.pop().unwrap(),
        };

        let program_table_challenges = ProgramTableChallenges {
            instruction_eval_indeterminate: weights.pop().unwrap(),
            address_weight: weights.pop().unwrap(),
            instruction_weight: weights.pop().unwrap(),
            next_instruction_weight: weights.pop().unwrap(),
        };

        let instruction_table_challenges = InstructionTableChallenges {
            processor_perm_indeterminate: processor_table_challenges.instruction_perm_indeterminate,
            ip_processor_weight: processor_table_challenges.instruction_table_ip_weight,
            ci_processor_weight: processor_table_challenges.instruction_table_ci_processor_weight,
            nia_processor_weight: processor_table_challenges.instruction_table_nia_weight,

            program_eval_indeterminate: program_table_challenges.instruction_eval_indeterminate,
            address_weight: program_table_challenges.address_weight,
            instruction_weight: program_table_challenges.instruction_weight,
            next_instruction_weight: program_table_challenges.next_instruction_weight,
        };

        let input_challenges = IOChallenges {
            processor_eval_indeterminate: processor_table_challenges
                .standard_input_eval_indeterminate,
        };

        let output_challenges = IOChallenges {
            processor_eval_indeterminate: processor_table_challenges
                .standard_output_eval_indeterminate,
        };

        let op_stack_table_challenges = OpStackTableChallenges {
            processor_perm_indeterminate: processor_table_challenges.op_stack_perm_indeterminate,
            clk_weight: processor_table_challenges.op_stack_table_clk_weight,
            ib1_weight: processor_table_challenges.op_stack_table_ib1_weight,
            osv_weight: processor_table_challenges.op_stack_table_osv_weight,
            osp_weight: processor_table_challenges.op_stack_table_osp_weight,
            all_clock_jump_differences_multi_perm_indeterminate: processor_table_challenges
                .all_clock_jump_differences_multi_perm_indeterminate,
        };

        let ram_table_challenges = RamTableChallenges {
            bezout_relation_indeterminate: weights.pop().unwrap(),
            processor_perm_indeterminate: processor_table_challenges.ram_perm_indeterminate,
            clk_weight: processor_table_challenges.ram_table_clk_weight,
            ramv_weight: processor_table_challenges.ram_table_ramv_weight,
            ramp_weight: processor_table_challenges.ram_table_ramp_weight,
            all_clock_jump_differences_multi_perm_indeterminate: processor_table_challenges
                .all_clock_jump_differences_multi_perm_indeterminate,
        };

        let jump_stack_table_challenges = JumpStackTableChallenges {
            processor_perm_indeterminate: processor_table_challenges.jump_stack_perm_indeterminate,
            clk_weight: processor_table_challenges.jump_stack_table_clk_weight,
            ci_weight: processor_table_challenges.jump_stack_table_ci_weight,
            jsp_weight: processor_table_challenges.jump_stack_table_jsp_weight,
            jso_weight: processor_table_challenges.jump_stack_table_jso_weight,
            jsd_weight: processor_table_challenges.jump_stack_table_jsd_weight,
            all_clock_jump_differences_multi_perm_indeterminate: processor_table_challenges
                .all_clock_jump_differences_multi_perm_indeterminate,
        };

        let hash_table_challenges = HashTableChallenges {
            from_processor_eval_indeterminate: processor_table_challenges
                .to_hash_table_eval_indeterminate,
            to_processor_eval_indeterminate: processor_table_challenges
                .from_hash_table_eval_indeterminate,

            stack_input_weights0: processor_table_challenges.hash_table_stack_input_weights0,
            stack_input_weights1: processor_table_challenges.hash_table_stack_input_weights1,
            stack_input_weights2: processor_table_challenges.hash_table_stack_input_weights2,
            stack_input_weights3: processor_table_challenges.hash_table_stack_input_weights3,
            stack_input_weights4: processor_table_challenges.hash_table_stack_input_weights4,
            stack_input_weights5: processor_table_challenges.hash_table_stack_input_weights5,
            stack_input_weights6: processor_table_challenges.hash_table_stack_input_weights6,
            stack_input_weights7: processor_table_challenges.hash_table_stack_input_weights7,
            stack_input_weights8: processor_table_challenges.hash_table_stack_input_weights8,
            stack_input_weights9: processor_table_challenges.hash_table_stack_input_weights9,

            digest_output_weights0: processor_table_challenges.hash_table_digest_output_weights0,
            digest_output_weights1: processor_table_challenges.hash_table_digest_output_weights1,
            digest_output_weights2: processor_table_challenges.hash_table_digest_output_weights2,
            digest_output_weights3: processor_table_challenges.hash_table_digest_output_weights3,
            digest_output_weights4: processor_table_challenges.hash_table_digest_output_weights4,
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
