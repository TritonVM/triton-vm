//! A parametric harness that mechanically hunts for the "accumulator column
//! unpinned on padding / boundary rows" soundness bug class.
//!
//! # What this checks
//!
//! Running-argument accumulator columns (log-derivatives, permutation products,
//! evaluation arguments) have transition constraints that are *linear* in their
//! target column `X'` (the column's value in the next row):
//!
//! ```text
//! C(X') = coeff · (X' − X) + const
//! ```
//!
//! The column is *pinned* on a given two-row window iff `coeff ≠ 0`. If `coeff`
//! can be driven to zero by an adversary while every *other* transition
//! constraint is still satisfiable, the prover gains a free cell whose freedom
//! flows into the table's terminal / cross-table argument — a soundness break.
//!
//! Rather than extract `coeff` symbolically, this harness probes it numerically:
//! for a fixed two-row window it evaluates *all* transition constraints with the
//! target aux cell of the next row set to `c`, then again with it set to
//! `c + δ`. If *no* constraint distinguishes the two values (every constraint is
//! invariant under the change), then `coeff = 0` on that window and the column is
//! unpinned there.
//!
//! The art is in choosing *adversarial* windows. A column whose update is gated
//! on "the next row is a padding row" is only at risk on a window that enters
//! padding; and the bug only bites when the *other* terms that would otherwise
//! pin the column (the honest "remains" / "accumulates" branches) are
//! simultaneously made vacuous by adversarial — not honest — guard values. The
//! window builders below therefore drive every competing guard to zero and set
//! genuinely-unconstrained quantities (RAM/op-stack pointer deltas, clock
//! differences, round numbers, modes, instruction selectors) to the specific
//! coefficient-cancelling values an adversary would pick, including a RAM /
//! op-stack pointer delta of `−2`.
//!
//! # Scope (read me before trusting a green run)
//!
//! This is a *targeted adversarial-window* harness, not a symbolic proof.
//!
//! - It only inspects the **transition** constraints. Guard columns that are
//!   confined to a finite set by *consistency* or *initial* constraints (e.g.
//!   the boolean padding indicators of the Lookup / Cascade tables, `copy_flag`
//!   of the U32 table) are set to their **legal in-domain edge values** here,
//!   because off-domain values are independently rejected by constraints this
//!   harness does not evaluate. Treating such a column as fully adversarial would
//!   be a false positive, not a finding.
//! - It only probes the **enumerated windows** below. A clean run means
//!   "no pinning gap found on the covered windows", **not** "proven pinned on all
//!   reachable windows".
//! - Covered tables / columns and known residual gaps are documented at the
//!   bottom of this file (see `coverage_notes`).
//!
//! # Validation
//!
//! The harness has been validated to *fail* (flagging the right column) when
//! either the RAM clock-jump-difference fix or the Hash cascade fix is reverted:
//!
//! - reverting the RAM fix (drop the `* next_row_is_padding_row` factor on
//!   `log_derivative_remains_or_ram_pointer_doesnt_change...` in
//!   `triton-air/src/table/ram.rs`) makes `ram_accumulators_are_pinned` flag the
//!   RAM `ClockJumpDifferenceLookupClientLogDerivative` column on the
//!   `ram: * -> padding, ptr delta -2` windows;
//! - reverting the Hash cascade fix (drop the `+ next_row_is_padding_row *
//!   cascade_log_derivative_remains` term in
//!   `HashTable::cascade_log_derivative_update_circuit`) makes
//!   `hash_accumulators_are_pinned` flag all sixteen cascade client
//!   log-derivative columns on the `hash: * -> Pad, rn'=0, ci'=hash` windows.

use ndarray::Array2;
use twenty_first::prelude::*;

use crate::AIR;
use crate::challenge_id::ChallengeId;
use crate::table::NUM_AUX_COLUMNS;
use crate::table::NUM_MAIN_COLUMNS;
use crate::table_column::MasterAuxColumn;
use crate::table_column::MasterMainColumn;

use constraint_circuit::ConstraintCircuitBuilder;
use strum::EnumCount;

/// A single adversarial two-row window: a human-readable label plus the main and
/// current-row-aux cells (addressed by *master* index) that define it.
///
/// Cells left unset default to zero. The harness fills the *next-row* value of
/// the accumulator column under test itself; everything else here is the fixed
/// context against which pinning is probed.
struct Window {
    label: &'static str,
    /// `(master_main_index, current_row?, value)` — `current_row == true` sets
    /// row 0 (the "current" row), `false` sets row 1 (the "next" row).
    main: Vec<(usize, bool, BFieldElement)>,
    /// `(master_aux_index, value)` for the *current* row only. The next-row value
    /// of each accumulator column is supplied by the prober.
    aux_curr: Vec<(usize, XFieldElement)>,
}

impl Window {
    fn new(label: &'static str) -> Self {
        Self {
            label,
            main: vec![],
            aux_curr: vec![],
        }
    }

    fn curr_main<C: MasterMainColumn>(mut self, col: C, value: BFieldElement) -> Self {
        self.main.push((col.master_main_index(), true, value));
        self
    }

    fn next_main<C: MasterMainColumn>(mut self, col: C, value: BFieldElement) -> Self {
        self.main.push((col.master_main_index(), false, value));
        self
    }

    /// Set a current-row value of an accumulator column. Currently unused by the
    /// windows below (every covered gap manifests with a zero current-row
    /// accumulator), but retained because some accumulators are *bilinear*
    /// (`X · X'`) and future windows may need a nonzero anchor to expose them.
    #[allow(dead_code)]
    fn curr_aux<C: MasterAuxColumn>(mut self, col: C, value: XFieldElement) -> Self {
        self.aux_curr.push((col.master_aux_index(), value));
        self
    }
}

/// Deterministic, distinct challenge values. Any fixed nonzero assignment works:
/// the pinning question is whether *some* constraint's value moves when the cell
/// moves, and that is a generic (Zariski-dense) property — a single sufficiently
/// generic challenge vector witnesses it. Using `i + 1` keeps every challenge
/// nonzero and pairwise distinct, matching the existing regression tests.
fn generic_challenges() -> Vec<XFieldElement> {
    (0..ChallengeId::COUNT)
        .map(|i| xfe!([i as u64 + 1, i as u64 + 2, i as u64 + 3]))
        .collect()
}

/// Probe pinning of every accumulator aux column of table `T` on every supplied
/// window. Returns the list of `(window_label, master_aux_index)` pairs that are
/// *unpinned* (a gap). An empty result means: no gap found on these windows.
///
/// `aux_range` is the half-open master-aux-index range owned by table `T`; only
/// columns in that range are probed (a table's transition constraints reference
/// only its own columns, so foreign columns are trivially "invariant" and would
/// be spurious).
fn unpinned_columns<T: AIR>(
    aux_range: std::ops::Range<usize>,
    windows: &[Window],
) -> Vec<(&'static str, usize)> {
    let builder = ConstraintCircuitBuilder::new();
    let constraints = T::transition_constraints(&builder);
    let circuits = constraints
        .into_iter()
        .map(|c| c.consume())
        .collect::<Vec<_>>();
    let challenges = generic_challenges();

    // Two probe values for the cell under test and their (nonzero) difference.
    let base = xfe!([42, 43, 44]);
    let perturbed = xfe!([1000, 7, 13]);
    assert_ne!(base, perturbed);

    // Two distinct, fully-populated aux tables used to recognise constraints that
    // depend *only* on the main columns. Such constraints encode reachability of
    // the window's main part (e.g. "the stack pointer changes by 0 or 1"); if any
    // of them is nonzero on a window, that window is *not* constraint-satisfying
    // and an apparent pinning gap there is a false positive (the adversary could
    // never reach it). This is exactly what separates the genuine RAM
    // pointer-delta `−2` bug (RAM pointers may jump arbitrarily) from the
    // non-reachable Op-Stack delta `−2` (forbidden by the stack-pointer
    // constraint).
    let aux_probe_a = Array2::<XFieldElement>::from_shape_fn([2, NUM_AUX_COLUMNS], |(r, c)| {
        xfe!([(r * NUM_AUX_COLUMNS + c + 1) as u64, 5, 9])
    });
    let aux_probe_b = Array2::<XFieldElement>::from_shape_fn([2, NUM_AUX_COLUMNS], |(r, c)| {
        xfe!([(2 * (r * NUM_AUX_COLUMNS + c) + 3) as u64, 11, 1])
    });

    let mut gaps = vec![];
    for window in windows {
        // Build the fixed main table for this window.
        let mut main = Array2::<BFieldElement>::zeros([2, NUM_MAIN_COLUMNS]);
        for &(idx, is_curr, value) in &window.main {
            main[[usize::from(!is_curr), idx]] = value;
        }

        // Reject windows whose main part is itself rejected by a main-only
        // transition constraint: such windows are unreachable, so a pinning gap
        // there would be spurious.
        let window_is_reachable = circuits.iter().all(|circuit| {
            let va = circuit.evaluate(main.view(), aux_probe_a.view(), &challenges);
            let vb = circuit.evaluate(main.view(), aux_probe_b.view(), &challenges);
            let is_main_only = va == vb;
            !(is_main_only && va != xfe!(0))
        });
        if !window_is_reachable {
            continue;
        }

        for aux_idx in aux_range.clone() {
            // Build the auxiliary table twice, differing only in the next-row
            // value of the single column under test. Current-row aux cells take
            // the window's specified values; everything else is zero. The
            // perturbed cell is exactly the cell whose pinning we test.
            let build_aux = |next_value: XFieldElement| {
                let mut aux = Array2::<XFieldElement>::zeros([2, NUM_AUX_COLUMNS]);
                for &(idx, value) in &window.aux_curr {
                    aux[[0, idx]] = value;
                }
                aux[[1, aux_idx]] = next_value;
                aux
            };
            let aux_base = build_aux(base);
            let aux_perturbed = build_aux(perturbed);

            let distinguished = circuits.iter().any(|circuit| {
                let v0 = circuit.evaluate(main.view(), aux_base.view(), &challenges);
                let v1 = circuit.evaluate(main.view(), aux_perturbed.view(), &challenges);
                v0 != v1
            });

            if !distinguished {
                gaps.push((window.label, aux_idx));
            }
        }
    }
    gaps
}

/// Assert that table `T`'s accumulator columns are pinned on every supplied
/// window, producing a clear, column-naming failure otherwise.
fn assert_all_pinned<T: AIR>(
    table_name: &str,
    aux_range: std::ops::Range<usize>,
    windows: &[Window],
) {
    let gaps = unpinned_columns::<T>(aux_range, windows);
    assert!(
        gaps.is_empty(),
        "Accumulator pinning gap(s) in the {table_name} table: the following \
         (window, master-aux-column-index) pairs leave the aux column FREE — \
         perturbing only that next-row cell satisfies every transition \
         constraint, so its terminal / cross-table value is forgeable:\n{gaps:#?}",
    );
}

// ===========================================================================
//                       Per-table adversarial windows
// ===========================================================================
//
// Each builder returns windows that (a) enter / stay in padding and (b) zero out
// every competing pinning guard, so that only the padding-row update gate stands
// between the accumulator and total freedom. Where a guard variable is genuinely
// adversary-controlled (RAM / op-stack pointer deltas, clock differences, round
// numbers) we set it to the coefficient-cancelling value an attacker would pick.

mod program_windows {
    use super::*;
    use crate::table_column::ProgramMainColumn as M;

    /// On the Program table the instruction-lookup server log-derivative is gated
    /// by `is_hash_input_padding` (a bit). Honest values pin it; we exercise both
    /// edges. (Also the two chunk evaluation arguments, which are pinned by
    /// linear update relations independent of padding.)
    pub fn windows() -> Vec<Window> {
        let one = bfe!(1);
        let zero = bfe!(0);
        vec![
            // current row is hash-input padding -> log-derivative must "remain".
            Window::new("program: into hash-input padding")
                .curr_main(M::IsHashInputPadding, one)
                .next_main(M::IsHashInputPadding, one),
            // current row is NOT padding -> log-derivative must "update".
            Window::new("program: active row (not padding)")
                .curr_main(M::IsHashInputPadding, zero)
                .next_main(M::IsHashInputPadding, zero),
        ]
    }
}

mod op_stack_windows {
    use super::*;
    use crate::table::op_stack::PADDING_VALUE;
    use crate::table_column::OpStackMainColumn as M;

    /// The op-stack clock-jump-difference log-derivative is gated like RAM's: it
    /// only "accumulates" when the stack pointer changes and the next row is
    /// padding, and "remains" otherwise. The danger window is entering padding
    /// with an adversarial pointer delta. (Op-stack pointers may only change by
    /// 0 or 1 honestly, but the transition constraints alone do not forbid other
    /// deltas, so we probe `−2` as well.)
    pub fn windows() -> Vec<Window> {
        let padding = PADDING_VALUE;
        let mut out = vec![];
        for (label, sp_curr, sp_next) in [
            ("op_stack: into padding, ptr delta 0", bfe!(5), bfe!(5)),
            ("op_stack: into padding, ptr delta -2", bfe!(5), bfe!(3)),
            ("op_stack: into padding, ptr delta +1", bfe!(5), bfe!(6)),
        ] {
            out.push(
                Window::new(label)
                    // next row is padding: IB1ShrinkStack == PADDING_VALUE.
                    .curr_main(M::IB1ShrinkStack, bfe!(0))
                    .next_main(M::IB1ShrinkStack, padding)
                    .curr_main(M::StackPointer, sp_curr)
                    .next_main(M::StackPointer, sp_next),
            );
        }
        // padding -> padding.
        out.push(
            Window::new("op_stack: padding -> padding, ptr delta -2")
                .curr_main(M::IB1ShrinkStack, padding)
                .next_main(M::IB1ShrinkStack, padding)
                .curr_main(M::StackPointer, bfe!(5))
                .next_main(M::StackPointer, bfe!(3)),
        );
        out
    }
}

mod ram_windows {
    use super::*;
    use crate::table::ram::INSTRUCTION_TYPE_READ;
    use crate::table::ram::INSTRUCTION_TYPE_WRITE;
    use crate::table::ram::PADDING_INDICATOR;
    use crate::table_column::RamMainColumn as M;

    /// The RAM clock-jump-difference log-derivative bug: entering a padding row
    /// the prover may decrease the RAM pointer by two. On the *buggy* tree the
    /// "accumulate" guard vanishes (next row is padding) and the "remains" guard
    /// `ram_pointer_difference + it(it−1)` becomes `−2 + 2 = 0`, freeing the
    /// column. RAM pointers may jump arbitrarily, so nothing else forbids the
    /// `−2` delta. This is the window that distinguishes the fixed and buggy
    /// trees, so it must be present.
    pub fn windows() -> Vec<Window> {
        let mut out = vec![];
        for (label, rp_curr, rp_next) in [
            ("ram: real -> padding, ptr delta -2", bfe!(5), bfe!(3)),
            ("ram: real -> padding, ptr delta 0", bfe!(5), bfe!(5)),
            ("ram: real -> padding, ptr delta +1", bfe!(5), bfe!(6)),
        ] {
            out.push(
                Window::new(label)
                    .curr_main(M::InstructionType, INSTRUCTION_TYPE_READ)
                    .next_main(M::InstructionType, PADDING_INDICATOR)
                    .curr_main(M::RamPointer, rp_curr)
                    .next_main(M::RamPointer, rp_next)
                    // keep `iord` consistent with the chosen pointer delta so the
                    // companion iord/pointer constraints don't accidentally
                    // distinguish the perturbation for the wrong reason.
                    .curr_main(M::InverseOfRampDifference, (rp_next - rp_curr).inverse_or_zero()),
            );
        }
        // padding -> padding with an adversarial pointer delta of -2.
        out.push(
            Window::new("ram: padding -> padding, ptr delta -2")
                .curr_main(M::InstructionType, PADDING_INDICATOR)
                .next_main(M::InstructionType, PADDING_INDICATOR)
                .curr_main(M::RamPointer, bfe!(5))
                .next_main(M::RamPointer, bfe!(3))
                .curr_main(M::InverseOfRampDifference, (bfe!(3) - bfe!(5)).inverse_or_zero()),
        );
        // a write->padding variant (different current instruction type).
        out.push(
            Window::new("ram: write -> padding, ptr delta -2")
                .curr_main(M::InstructionType, INSTRUCTION_TYPE_WRITE)
                .next_main(M::InstructionType, PADDING_INDICATOR)
                .curr_main(M::RamPointer, bfe!(5))
                .next_main(M::RamPointer, bfe!(3))
                .curr_main(M::InverseOfRampDifference, (bfe!(3) - bfe!(5)).inverse_or_zero()),
        );
        out
    }
}

mod jump_stack_windows {
    use super::*;
    use crate::table_column::JumpStackMainColumn as M;

    /// The jump-stack clock-jump-difference log-derivative is gated by whether
    /// the jump-stack pointer increases by one. Its "remains" branch is guarded
    /// by `jsp' − jsp`, so the dangerous window is `jsp' − jsp = 0` (no
    /// increase). There is no padding indicator; this exercises the boundary
    /// where the pointer does not advance.
    pub fn windows() -> Vec<Window> {
        vec![
            Window::new("jump_stack: jsp unchanged")
                .curr_main(M::JSP, bfe!(5))
                .next_main(M::JSP, bfe!(5)),
            Window::new("jump_stack: jsp +1")
                .curr_main(M::JSP, bfe!(5))
                .next_main(M::JSP, bfe!(6)),
        ]
    }
}

mod hash_windows {
    use super::*;
    use crate::table::hash::HashTableMode;
    use crate::table_column::HashMainColumn as M;
    use isa::instruction::Instruction;
    use twenty_first::prelude::tip5::NUM_ROUNDS;

    /// The Hash table's cascade client log-derivatives bug. On a window entering
    /// padding (`mode' = Pad`) the cascade-update constraint reduces to
    ///
    /// ```text
    ///   select_mode(Pad, mode')·(rn'−NUM_ROUNDS)·(ci'−init)·updates
    ///   + deselect(rn' = NUM_ROUNDS)·remains
    ///   + deselect(ci' = sponge_init)·remains
    ///   [ + mode_deselector(Pad, mode')·remains   <-- the fix ]
    /// ```
    ///
    /// With `mode' = Pad` the `select_mode(Pad, mode')` factor is `0` (Pad's mode
    /// value is `0`), so the "updates" term drops out. With `rn' = 0` and
    /// `ci' = hash` the two remaining `deselect(...)` coefficients are *also* `0`
    /// (each deselector is zero unless its argument hits the selected value). On
    /// the *buggy* tree the coefficient of the cascade column is then `0` — it is
    /// free. The fix restores pinning by adding `mode_deselector(Pad)·remains`,
    /// whose coefficient is nonzero exactly when `mode' = Pad`.
    ///
    /// For the window to be *reachable* (so the gap is real, not spurious) the
    /// current row must satisfy the round-number / mode transition constraints.
    /// Several of those are gated by `(round_number − NUM_ROUNDS)`; setting the
    /// current `RoundNumber = NUM_ROUNDS` makes them vacuous, which is also the
    /// only round number from which the mode may legally change.
    pub fn windows() -> Vec<Window> {
        let mode_pad = bfe!(u64::from(HashTableMode::Pad));
        let mode_hash = bfe!(u64::from(HashTableMode::Hash));
        let hash_op = Instruction::Hash.opcode_b();
        let num_rounds = bfe!(NUM_ROUNDS as u64);
        let mut windows = vec![
            // The coefficient-cancelling into-padding window. Current row:
            // mode = Hash, round_number = NUM_ROUNDS (mode may change), ci = hash.
            // Next row: mode = Pad, round_number = 0, ci = hash — both deselector
            // coefficients vanish, so only the fix's term pins the column.
            Window::new("hash: Hash(rn=NR) -> Pad, rn'=0, ci'=hash")
                .curr_main(M::Mode, mode_hash)
                .curr_main(M::RoundNumber, num_rounds)
                .curr_main(M::CI, hash_op)
                .next_main(M::Mode, mode_pad)
                .next_main(M::RoundNumber, bfe!(0))
                .next_main(M::CI, hash_op),
            // padding -> padding (already in Pad; round_number 0 -> 0).
            Window::new("hash: Pad(rn=NR) -> Pad, rn'=0, ci'=hash")
                .curr_main(M::Mode, mode_pad)
                .curr_main(M::RoundNumber, num_rounds)
                .next_main(M::Mode, mode_pad)
                .next_main(M::RoundNumber, bfe!(0))
                .next_main(M::CI, hash_op),
        ];
        // INTERIOR sweep. Padding/boundary windows alone miss accumulators whose
        // update/remains guards cancel additively at non-boundary rows. Sweep
        // every (mode, round-number) interior transition `rn -> rn+1` within a
        // permutation. This catches e.g. `HashInputRunningEvaluation`, whose
        // remains-guard coefficient `round_number_next + (mode_next - Hash)` is
        // zero at the *reachable* rows (ProgramHashing, rn'=2) and (Sponge, rn'=1)
        // — additive cancellation away from the update point (rn'=0, mode'=Hash).
        let sponge_absorb = Instruction::SpongeAbsorb.opcode_b();
        let mode_prog = bfe!(u64::from(HashTableMode::ProgramHashing));
        let mode_sponge = bfe!(u64::from(HashTableMode::Sponge));
        for (mode, ci) in [
            (mode_prog, hash_op),
            (mode_sponge, sponge_absorb),
            (mode_hash, hash_op),
        ] {
            for curr_rn in 0..NUM_ROUNDS as u64 {
                windows.push(
                    Window::new("hash: interior (mode, rn -> rn+1)")
                        .curr_main(M::Mode, mode)
                        .curr_main(M::RoundNumber, bfe!(curr_rn))
                        .curr_main(M::CI, ci)
                        .next_main(M::Mode, mode)
                        .next_main(M::RoundNumber, bfe!(curr_rn + 1))
                        .next_main(M::CI, ci),
                );
            }
        }
        windows
    }
}

mod cascade_windows {
    use super::*;
    use crate::table_column::CascadeMainColumn as M;

    /// The Cascade table's two log-derivatives are gated by `is_padding_next`, a
    /// bit confined to `{0, 1}` by the consistency constraints. Both edges pin
    /// (the bit-complement guard is robust), so we exercise both. Off-domain
    /// values of `is_padding_next` are rejected by constraints this harness does
    /// not evaluate, so they are not adversarial here.
    pub fn windows() -> Vec<Window> {
        vec![
            Window::new("cascade: into padding (is_padding'=1)")
                .next_main(M::IsPadding, bfe!(1)),
            Window::new("cascade: active (is_padding'=0)").next_main(M::IsPadding, bfe!(0)),
            Window::new("cascade: padding -> padding")
                .curr_main(M::IsPadding, bfe!(1))
                .next_main(M::IsPadding, bfe!(1)),
        ]
    }
}

mod lookup_windows {
    use super::*;
    use crate::table_column::LookupMainColumn as M;

    /// The Lookup table's cascade-server log-derivative and public evaluation
    /// argument are gated by `is_padding_next` (a bit). Same reasoning as the
    /// Cascade table: both edges pin.
    pub fn windows() -> Vec<Window> {
        vec![
            Window::new("lookup: into padding (is_padding'=1)")
                .next_main(M::IsPadding, bfe!(1)),
            Window::new("lookup: active (is_padding'=0)").next_main(M::IsPadding, bfe!(0)),
            Window::new("lookup: padding -> padding")
                .curr_main(M::IsPadding, bfe!(1))
                .next_main(M::IsPadding, bfe!(1)),
        ]
    }
}

mod u32_windows {
    use super::*;
    use crate::table_column::U32MainColumn as M;

    /// The U32 lookup server log-derivative is gated by `copy_flag_next`, a bit
    /// confined to `{0, 1}` by the consistency constraints. The bit-complement
    /// guard `(cf'−1)·remains + cf'·accumulates` pins on both edges, so we
    /// exercise both.
    pub fn windows() -> Vec<Window> {
        vec![
            Window::new("u32: copy_flag'=0 (mid-section, remains)")
                .next_main(M::CopyFlag, bfe!(0)),
            Window::new("u32: copy_flag'=1 (new section, accumulates)")
                .next_main(M::CopyFlag, bfe!(1)),
        ]
    }
}

mod processor_windows {
    use super::*;
    use crate::table_column::ProcessorMainColumn as M;

    /// Best-effort coverage of the Processor table. The Processor's accumulators
    /// (the standard-input/output evaluation arguments, the three table
    /// permutation products, the instruction-lookup, hash, sponge, U32 and
    /// clock-jump-difference arguments) are gated by per-instruction *deselectors*
    /// rather than a single padding indicator. Pinning them on adversarial
    /// windows requires reconstructing each instruction's deselector to zero, a
    /// table-specific effort out of scope for this padding-boundary harness.
    ///
    /// We include the padding window (`IsPadding` current = next = 1) as a smoke
    /// check; see `coverage_notes` for the residual, *unverified* Processor
    /// columns.
    pub fn windows() -> Vec<Window> {
        vec![
            Window::new("processor: padding -> padding")
                .curr_main(M::IsPadding, bfe!(1))
                .next_main(M::IsPadding, bfe!(1)),
        ]
    }
}

// ===========================================================================
//                                  Tests
// ===========================================================================

use crate::table::AUX_CASCADE_TABLE_END;
use crate::table::AUX_CASCADE_TABLE_START;
use crate::table::AUX_HASH_TABLE_END;
use crate::table::AUX_HASH_TABLE_START;
use crate::table::AUX_JUMP_STACK_TABLE_END;
use crate::table::AUX_JUMP_STACK_TABLE_START;
use crate::table::AUX_LOOKUP_TABLE_END;
use crate::table::AUX_LOOKUP_TABLE_START;
use crate::table::AUX_OP_STACK_TABLE_END;
use crate::table::AUX_OP_STACK_TABLE_START;
use crate::table::AUX_PROGRAM_TABLE_END;
use crate::table::AUX_PROGRAM_TABLE_START;
use crate::table::AUX_RAM_TABLE_END;
use crate::table::AUX_RAM_TABLE_START;
use crate::table::AUX_PROCESSOR_TABLE_END;
use crate::table::AUX_PROCESSOR_TABLE_START;
use crate::table::AUX_U32_TABLE_END;
use crate::table::AUX_U32_TABLE_START;
use crate::table::cascade::CascadeTable;
use crate::table::hash::HashTable;
use crate::table::jump_stack::JumpStackTable;
use crate::table::lookup::LookupTable;
use crate::table::op_stack::OpStackTable;
use crate::table::processor::ProcessorTable;
use crate::table::program::ProgramTable;
use crate::table::ram::RamTable;
use crate::table::u32::U32Table;

#[test]
fn program_accumulators_are_pinned() {
    assert_all_pinned::<ProgramTable>(
        "Program",
        AUX_PROGRAM_TABLE_START..AUX_PROGRAM_TABLE_END,
        &program_windows::windows(),
    );
}

#[test]
fn op_stack_accumulators_are_pinned() {
    assert_all_pinned::<OpStackTable>(
        "OpStack",
        AUX_OP_STACK_TABLE_START..AUX_OP_STACK_TABLE_END,
        &op_stack_windows::windows(),
    );
}

#[test]
fn ram_accumulators_are_pinned() {
    assert_all_pinned::<RamTable>(
        "Ram",
        AUX_RAM_TABLE_START..AUX_RAM_TABLE_END,
        &ram_windows::windows(),
    );
}

#[test]
fn jump_stack_accumulators_are_pinned() {
    assert_all_pinned::<JumpStackTable>(
        "JumpStack",
        AUX_JUMP_STACK_TABLE_START..AUX_JUMP_STACK_TABLE_END,
        &jump_stack_windows::windows(),
    );
}

#[test]
fn hash_accumulators_are_pinned() {
    assert_all_pinned::<HashTable>(
        "Hash",
        AUX_HASH_TABLE_START..AUX_HASH_TABLE_END,
        &hash_windows::windows(),
    );
}

#[test]
fn cascade_accumulators_are_pinned() {
    assert_all_pinned::<CascadeTable>(
        "Cascade",
        AUX_CASCADE_TABLE_START..AUX_CASCADE_TABLE_END,
        &cascade_windows::windows(),
    );
}

#[test]
fn lookup_accumulators_are_pinned() {
    assert_all_pinned::<LookupTable>(
        "Lookup",
        AUX_LOOKUP_TABLE_START..AUX_LOOKUP_TABLE_END,
        &lookup_windows::windows(),
    );
}

#[test]
fn u32_accumulators_are_pinned() {
    assert_all_pinned::<U32Table>(
        "U32",
        AUX_U32_TABLE_START..AUX_U32_TABLE_END,
        &u32_windows::windows(),
    );
}

/// Best-effort smoke check for the Processor table; see `coverage_notes` and
/// the `processor_windows` module documentation for the residual gap.
#[test]
fn processor_accumulators_pinned_on_padding_smoke() {
    assert_all_pinned::<ProcessorTable>(
        "Processor",
        AUX_PROCESSOR_TABLE_START..AUX_PROCESSOR_TABLE_END,
        &processor_windows::windows(),
    );
}

/// A meta-test documenting which columns the harness *intends* to cover and
/// asserting the per-table window lists are non-empty. This is a guard against
/// silently shipping an empty window set (which would make a table's pinning
/// test vacuously pass).
#[test]
fn coverage_notes() {
    // Tables whose accumulators are gated by an explicit padding indicator and
    // are fully covered by adversarial into-padding / padding->padding windows
    // (including pointer delta = -2 where the pointer is adversary-controlled):
    //
    //   Program   - InstructionLookupServerLogDerivative + chunk eval args
    //   OpStack   - RunningProductPermArg, ClockJumpDifferenceLookupClient...
    //   Ram       - all six aux columns, incl. ClockJumpDifferenceLookupClient...
    //   JumpStack - RunningProductPermArg, ClockJumpDifferenceLookupClient...
    //   Hash      - all four cascade-client log-derivatives + eval args
    //   Cascade   - HashTableServer..., LookupTableClient... (bit-gated)
    //   Lookup    - CascadeTableServer..., PublicEvaluationArgument (bit-gated)
    //   U32       - LookupServerLogDerivative (bit-gated by copy_flag)
    //
    // RESIDUAL / UNVERIFIED:
    //
    //   Processor - the eleven accumulator columns are gated by per-instruction
    //               deselectors, not a single padding indicator. Only the
    //               padding->padding smoke window is exercised here; per-column
    //               adversarial coverage requires reconstructing each
    //               instruction deselector and is left as future work. A clean
    //               Processor result below therefore means "the padding row does
    //               not free these columns", NOT "pinned on all reachable
    //               windows".
    assert!(!program_windows::windows().is_empty());
    assert!(!op_stack_windows::windows().is_empty());
    assert!(!ram_windows::windows().is_empty());
    assert!(!jump_stack_windows::windows().is_empty());
    assert!(!hash_windows::windows().is_empty());
    assert!(!cascade_windows::windows().is_empty());
    assert!(!lookup_windows::windows().is_empty());
    assert!(!u32_windows::windows().is_empty());
    assert!(!processor_windows::windows().is_empty());
}
