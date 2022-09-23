use std::marker::PhantomData;

use itertools::Itertools;
use num_traits::Zero;
use rand::thread_rng;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::rescue_prime_regular::DIGEST_LENGTH;
use twenty_first::shared_math::traits::GetRandomElements;
use twenty_first::shared_math::x_field_element::XFieldElement;
use twenty_first::util_types::simple_hasher::{Hashable, Hasher};

use super::hash_table::{HashTableChallenges, HashTableTerminals};
use super::instruction_table::{InstructionTableChallenges, InstructionTableTerminals};
use super::jump_stack_table::{JumpStackTableChallenges, JumpStackTableTerminals};
use super::op_stack_table::{OpStackTableChallenges, OpStackTableTerminals};
use super::processor_table::IOChallenges;
use super::processor_table::{ProcessorTableChallenges, ProcessorTableTerminals};
use super::program_table::{ProgramTableChallenges, ProgramTableTerminals};
use super::ram_table::{RamTableChallenges, RamTableTerminals};
use super::u32_op_table::{U32OpTableChallenges, U32OpTableTerminals};

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
            u32_perm_row_weight: weights.pop().unwrap(),

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

            u32_op_table_ci_weight: weights.pop().unwrap(),
            u32_op_table_lhs_weight: weights.pop().unwrap(),
            u32_op_table_rhs_weight: weights.pop().unwrap(),
            u32_op_table_result_weight: weights.pop().unwrap(),
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

        let u32_op_table_challenges = U32OpTableChallenges {
            processor_perm_row_weight: processor_table_challenges.u32_perm_row_weight,

            ci_weight: processor_table_challenges.u32_op_table_ci_weight,
            lhs_weight: processor_table_challenges.u32_op_table_lhs_weight,
            rhs_weight: processor_table_challenges.u32_op_table_rhs_weight,
            result_weight: processor_table_challenges.u32_op_table_result_weight,
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

    /// Only intended for debugging purposes. In a production STARK, use Fiat-Shamir instead.
    pub fn dummy() -> Self {
        let mut rng = thread_rng();
        let random_challenges = XFieldElement::random_elements(Self::TOTAL_CHALLENGES, &mut rng);

        Self::create_challenges(random_challenges)
    }
}

#[derive(Debug, Clone)]
pub struct AllTerminals<H: Hasher>
where
    BFieldElement: Hashable<H::T>,
{
    pub program_table_terminals: ProgramTableTerminals,
    pub instruction_table_terminals: InstructionTableTerminals,
    pub processor_table_terminals: ProcessorTableTerminals,
    pub op_stack_table_terminals: OpStackTableTerminals,
    pub ram_table_terminals: RamTableTerminals,
    pub jump_stack_table_terminals: JumpStackTableTerminals,
    pub hash_table_terminals: HashTableTerminals,
    pub u32_op_table_terminals: U32OpTableTerminals,
    pub(crate) phantom: PhantomData<H>,
}

impl<H: Hasher> AllTerminals<H>
where
    BFieldElement: Hashable<H::T>,
{
    pub const TOTAL_TERMINALS: usize = 10;

    /// Only intended for debugging purposes. In a production STARK, don't use this for deriving
    /// terminals.
    pub fn dummy() -> Self {
        let mut rng = thread_rng();
        let random_challenges = XFieldElement::random_elements(Self::TOTAL_TERMINALS, &mut rng);

        let mut weights = random_challenges;
        let processor_table_terminals = ProcessorTableTerminals {
            input_table_eval_arg: XFieldElement::zero(),
            output_table_eval_arg: XFieldElement::zero(),
            instruction_table_perm_product: weights.pop().unwrap(),
            opstack_table_perm_product: weights.pop().unwrap(),
            ram_table_perm_product: weights.pop().unwrap(),
            jump_stack_perm_product: weights.pop().unwrap(),
            to_hash_table_eval_arg: weights.pop().unwrap(),
            from_hash_table_eval_arg: weights.pop().unwrap(),
            u32_table_perm_product: weights.pop().unwrap(),
        };
        let program_table_terminals = ProgramTableTerminals {
            instruction_eval_arg: weights.pop().unwrap(),
        };
        let instruction_table_terminals = InstructionTableTerminals {
            processor_perm_product: processor_table_terminals.instruction_table_perm_product,
            program_eval_arg: program_table_terminals.instruction_eval_arg,
        };
        let op_stack_table_terminals = OpStackTableTerminals {
            processor_perm_product: processor_table_terminals.opstack_table_perm_product,
        };
        let ram_table_terminals = RamTableTerminals {
            processor_perm_product: processor_table_terminals.ram_table_perm_product,
        };
        let jump_stack_table_terminals = JumpStackTableTerminals {
            processor_perm_product: processor_table_terminals.jump_stack_perm_product,
        };
        let hash_table_terminals = HashTableTerminals {
            from_processor_eval_arg: processor_table_terminals.to_hash_table_eval_arg,
            to_processor_eval_arg: processor_table_terminals.from_hash_table_eval_arg,
        };
        let u32_op_table_terminals = U32OpTableTerminals {
            processor_perm_product: processor_table_terminals.u32_table_perm_product,
        };
        AllTerminals {
            program_table_terminals,
            instruction_table_terminals,
            processor_table_terminals,
            op_stack_table_terminals,
            ram_table_terminals,
            jump_stack_table_terminals,
            hash_table_terminals,
            u32_op_table_terminals,
            phantom: PhantomData,
        }
    }
}

/// Make terminals iterable for ProofStream
///
/// In order for `Stark::verify()` to receive all terminals via `ProofStream`,
/// they must serialise to a stream of `BFieldElement`s.
impl<H: Hasher> IntoIterator for AllTerminals<H>
where
    BFieldElement: Hashable<H::T>,
{
    type Item = H::T;

    type IntoIter = std::vec::IntoIter<H::T>;

    fn into_iter(self) -> Self::IntoIter
    where
        BFieldElement: Hashable<H::T>,
    {
        vec![
            &self.program_table_terminals.instruction_eval_arg,
            &self.instruction_table_terminals.processor_perm_product,
            &self.instruction_table_terminals.program_eval_arg,
            &self
                .processor_table_terminals
                .instruction_table_perm_product,
            &self.processor_table_terminals.opstack_table_perm_product,
            &self.processor_table_terminals.ram_table_perm_product,
            &self.processor_table_terminals.jump_stack_perm_product,
            &self.processor_table_terminals.to_hash_table_eval_arg,
            &self.processor_table_terminals.from_hash_table_eval_arg,
            &self.processor_table_terminals.u32_table_perm_product,
            &self.op_stack_table_terminals.processor_perm_product,
            &self.ram_table_terminals.processor_perm_product,
            &self.jump_stack_table_terminals.processor_perm_product,
            &self.hash_table_terminals.from_processor_eval_arg,
            &self.hash_table_terminals.to_processor_eval_arg,
            &self.u32_op_table_terminals.processor_perm_product,
        ]
        .into_iter()
        .map(|terminal| terminal.coefficients.to_vec())
        .concat()
        .iter()
        .flat_map(|b| b.to_sequence())
        .collect_vec()
        .into_iter()
    }
}

#[cfg(test)]
mod challenges_terminals_tests {
    use twenty_first::shared_math::rescue_prime_regular::RescuePrimeRegular;

    use crate::table::challenges_terminals::AllTerminals;
    use crate::table::processor_table;
    use crate::table::program_table;

    #[test]
    fn total_challenges_equal_permutation_and_evaluation_args_test() {
        type AEP = AllTerminals<RescuePrimeRegular>;
        assert_eq!(
            processor_table::PROCESSOR_TABLE_INITIALS_COUNT
                + program_table::PROGRAM_TABLE_INITIALS_COUNT,
            AEP::TOTAL_TERMINALS,
        );
    }
}
