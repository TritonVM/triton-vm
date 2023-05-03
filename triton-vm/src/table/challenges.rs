//! Challenges are needed for the [cross-table arguments](CrossTableArg), _i.e._,
//! [Permutation Arguments](crate::table::cross_table_argument::PermArg),
//! [Evaluation Arguments](crate::table::cross_table_argument::EvalArg), and
//! [Lookup Arguments](crate::table::cross_table_argument::LookupArg),
//! as well as for the RAM Table's Contiguity Argument.
//!
//! There are three types of challenges:
//! - **Weights**. Weights are used to linearly combine multiple elements into one element. The
//! resulting single element can then be used in a cross-table argument.
//! - **Indeterminates**. All cross-table arguments work by checking the equality of polynomials (or
//! rational functions). Through the Schwartz-Zippel lemma, this equality check can be performed
//! by evaluating the polynomials (or rational functions) in a single point. The challenges that
//! are indeterminates are exactly this evaluation point. The polynomials (or rational functions)
//! are never stored explicitly. Instead, they are directly evaluated at the point indicated by a
//! challenge of “type” `Indeterminate`, giving rise to “running products”, “running
//! evaluations”, _et cetera_.
//! - **Terminals**. The public input (respectively output) of the program is not stored in any
//! table. Instead, the terminal of the Evaluation Argument is computed directly from the
//! public input (respectively output) and the indeterminate.

use std::fmt::Debug;
use std::hash::Hash;

use strum::EnumCount;
use strum_macros::Display;
use strum_macros::EnumCount as EnumCountMacro;
use strum_macros::EnumIter;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::other::random_elements;
use twenty_first::shared_math::tip5::LOOKUP_TABLE;
use twenty_first::shared_math::x_field_element::XFieldElement;

use crate::table::challenges::ChallengeId::*;
use crate::table::cross_table_argument::CrossTableArg;
use crate::table::cross_table_argument::EvalArg;

/// A `ChallengeId` is a unique, symbolic identifier for a challenge used in Triton VM. The
/// `ChallengeId` enum works in tandem with the struct [`Challenges`], which can be
/// instantiated to hold actual challenges that can be indexed by some `ChallengeId`.
///
/// Since almost all challenges relate to the Processor Table in some form, the words “Processor
/// Table” are usually omitted from the `ChallengeId`'s name.
#[repr(usize)]
#[derive(Display, Debug, Clone, Copy, PartialEq, Eq, EnumIter, EnumCountMacro, Hash)]
pub enum ChallengeId {
    /// The indeterminate for the [Evaluation Argument](EvalArg) with standard input.
    StandardInputIndeterminate,

    /// The indeterminate for the [Evaluation Argument](EvalArg) with standard output.
    StandardOutputIndeterminate,

    /// The indeterminate for the instruction
    /// [Lookup Argument](crate::table::cross_table_argument::LookupArg)
    /// between the [Processor Table](crate::table::processor_table) and the
    /// [Program Table](crate::table::program_table) guaranteeing that the instructions and their
    /// arguments are copied correctly.
    InstructionLookupIndeterminate,

    HashInputIndeterminate,
    HashDigestIndeterminate,
    SpongeIndeterminate,

    OpStackIndeterminate,
    RamIndeterminate,
    JumpStackIndeterminate,

    U32Indeterminate,

    /// The indeterminate for the Lookup Argument between the Processor Table and all memory-like
    /// tables, _i.e._, the OpStack Table, the Ram Table, and the JumpStack Table, guaranteeing
    /// that all clock jump differences are directed forward.
    ClockJumpDifferenceLookupIndeterminate,

    /// The indeterminate for the Contiguity Argument within the Ram Table.
    RamTableBezoutRelationIndeterminate,

    /// A weight for linearly combining multiple elements. Applies to
    /// - `Address` in the Program Table
    /// - `IP` in the Processor Table
    ProgramAddressWeight,

    /// A weight for linearly combining multiple elements. Applies to
    /// - `Instruction` in the Program Table
    /// - `CI` in the Processor Table
    ProgramInstructionWeight,

    /// A weight for linearly combining multiple elements. Applies to
    /// - `Instruction'` (_i.e._, in the next row) in the Program Table
    /// - `NIA` in the Processor Table
    ProgramNextInstructionWeight,

    OpStackClkWeight,
    OpStackIb1Weight,
    OpStackOspWeight,
    OpStackOsvWeight,

    RamClkWeight,
    RamRampWeight,
    RamRamvWeight,
    RamPreviousInstructionWeight,

    JumpStackClkWeight,
    JumpStackCiWeight,
    JumpStackJspWeight,
    JumpStackJsoWeight,
    JumpStackJsdWeight,

    HashCIWeight,
    HashStateWeight0,
    HashStateWeight1,
    HashStateWeight2,
    HashStateWeight3,
    HashStateWeight4,
    HashStateWeight5,
    HashStateWeight6,
    HashStateWeight7,
    HashStateWeight8,
    HashStateWeight9,
    HashStateWeight10,
    HashStateWeight11,
    HashStateWeight12,
    HashStateWeight13,
    HashStateWeight14,
    HashStateWeight15,

    /// The indeterminate for the Lookup Argument between the Hash Table and the Cascade Table.
    HashCascadeLookupIndeterminate,

    /// A weight for linearly combining multiple elements. Applies to
    /// - `*LkIn` in the Hash Table, and
    /// - `2^16·LookInHi + LookInLo` in the Cascade Table.
    HashCascadeLookInWeight,

    /// A weight for linearly combining multiple elements. Applies to
    /// - `*LkOut` in the Hash Table, and
    /// - `2^16·LookOutHi + LookOutLo` in the Cascade Table.
    HashCascadeLookOutWeight,

    /// The indeterminate for the Lookup Argument between the Cascade Table and the Lookup Table.
    CascadeLookupIndeterminate,

    /// A weight for linearly combining multiple elements. Applies to
    /// - `LkIn*` in the Cascade Table, and
    /// - `LookIn` in the Lookup Table.
    LookupTableInputWeight,

    /// A weight for linearly combining multiple elements. Applies to
    /// - `LkOut*` in the Cascade Table, and
    /// - `LookOut` in the Lookup Table.
    LookupTableOutputWeight,

    /// The indeterminate for the public Evaluation Argument establishing correctness of the
    /// Lookup Table.
    LookupTablePublicIndeterminate,

    U32LhsWeight,
    U32RhsWeight,
    U32CiWeight,
    U32ResultWeight,

    /// The terminal for the Evaluation Argument with standard input.
    StandardInputTerminal,

    /// The terminal for the Evaluation Argument with standard output.
    StandardOutputTerminal,

    /// The terminal for the Evaluation Argument establishing correctness of the
    /// [Lookup Table](crate::table::lookup_table).
    LookupTablePublicTerminal,
}

impl ChallengeId {
    pub const fn index(&self) -> usize {
        *self as usize
    }
}

/// The `Challenges` struct holds the challenges used in Triton VM. The concrete challenges are
/// known only at runtime. The challenges are indexed using enum [`ChallengeId`]. The `Challenges`
/// struct is essentially a thin wrapper around an array of [`XFieldElement`]s, providing
/// convenience methods.
pub struct Challenges {
    pub challenges: [XFieldElement; Self::count()],
}

impl Challenges {
    pub const fn count() -> usize {
        ChallengeId::COUNT
    }

    /// The number of weights to sample using the Fiat-Shamir heuristic. This number is lower
    /// than the number of challenges because several challenges are not sampled, but computed
    /// from publicly known values and other, sampled challenges.
    ///
    /// Concretely:
    /// - The [`StandardInputTerminal`] is computed from Triton VM's public input and the sampled
    /// indeterminate [`StandardInputIndeterminate`].
    /// - The [`StandardOutputTerminal`] is computed from Triton VM's public output and the sampled
    /// indeterminate [`StandardOutputIndeterminate`].
    /// - The [`LookupTablePublicTerminal`] is computed from the publicly known and constant
    /// lookup table and the sampled indeterminate [`LookupTablePublicIndeterminate`].
    pub const fn num_challenges_to_sample() -> usize {
        // When modifying this, be sure to add to the compile-time assertions of the form
        // `const _: () = assert!(…);`
        // at the end of this file.
        Self::count() - 3
    }

    pub fn new(
        mut challenges: Vec<XFieldElement>,
        public_input: &[BFieldElement],
        public_output: &[BFieldElement],
    ) -> Self {
        assert_eq!(Self::num_challenges_to_sample(), challenges.len());

        let input_terminal = EvalArg::compute_terminal(
            public_input,
            EvalArg::default_initial(),
            challenges[StandardInputIndeterminate.index()],
        );
        let output_terminal = EvalArg::compute_terminal(
            public_output,
            EvalArg::default_initial(),
            challenges[StandardOutputIndeterminate.index()],
        );
        let lookup_terminal = EvalArg::compute_terminal(
            &LOOKUP_TABLE.map(|i| BFieldElement::new(i as u64)),
            EvalArg::default_initial(),
            challenges[LookupTablePublicIndeterminate.index()],
        );

        challenges.insert(StandardInputTerminal.index(), input_terminal);
        challenges.insert(StandardOutputTerminal.index(), output_terminal);
        challenges.insert(LookupTablePublicTerminal.index(), lookup_terminal);
        assert_eq!(Self::count(), challenges.len());
        let challenges = challenges.try_into().unwrap();

        Self { challenges }
    }

    /// Stand-in challenges. Can be used in tests. For non-interactive STARKs, use the
    /// Fiat-Shamir heuristic to derive the actual challenges.
    pub fn placeholder(public_input: &[BFieldElement], public_output: &[BFieldElement]) -> Self {
        let stand_in_challenges = random_elements(Self::num_challenges_to_sample());
        Self::new(stand_in_challenges, public_input, public_output)
    }

    #[inline(always)]
    pub fn get_challenge(&self, id: ChallengeId) -> XFieldElement {
        self.challenges[id.index()]
    }
}

#[cfg(test)]
mod challenge_tests {
    use super::*;

    #[test]
    const fn compile_time_index_assertions() {
        // Terminal challenges are computed from public information, such as public input or
        // public output, and other challenges. Because these other challenges are used to compute
        // the terminal challenges, the terminal challenges must be inserted into the challenges
        // vector after the used challenges.
        assert!(StandardInputIndeterminate.index() < StandardInputTerminal.index());
        assert!(StandardInputIndeterminate.index() < StandardOutputTerminal.index());
        assert!(StandardInputIndeterminate.index() < LookupTablePublicTerminal.index());

        assert!(StandardOutputIndeterminate.index() < StandardInputTerminal.index());
        assert!(StandardOutputIndeterminate.index() < StandardOutputTerminal.index());
        assert!(StandardOutputIndeterminate.index() < LookupTablePublicTerminal.index());

        assert!(LookupTablePublicIndeterminate.index() < StandardInputTerminal.index());
        assert!(LookupTablePublicIndeterminate.index() < StandardOutputTerminal.index());
        assert!(LookupTablePublicIndeterminate.index() < LookupTablePublicTerminal.index());
    }
    // Ensure the compile-time assertions are actually executed by the compiler.
    const _: () = compile_time_index_assertions();
}
