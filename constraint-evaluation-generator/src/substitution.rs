use std::collections::HashSet;

use itertools::Itertools;
use proc_macro2::TokenStream;
use quote::format_ident;
use quote::quote;

use triton_vm::table;
use triton_vm::table::constraint_circuit::BinOp;
use triton_vm::table::constraint_circuit::CircuitExpression;
use triton_vm::table::constraint_circuit::ConstraintCircuit;
use triton_vm::table::constraint_circuit::ConstraintCircuitMonad;
use triton_vm::table::constraint_circuit::DualRowIndicator;
use triton_vm::table::constraint_circuit::InputIndicator;
use triton_vm::table::constraint_circuit::SingleRowIndicator;
use triton_vm::table::degree_lowering_table;

use crate::codegen::RustBackend;

pub(crate) struct AllSubstitutions {
    pub base: Substitutions,
    pub ext: Substitutions,
}

pub(crate) struct Substitutions {
    pub init: Vec<ConstraintCircuitMonad<SingleRowIndicator>>,
    pub cons: Vec<ConstraintCircuitMonad<SingleRowIndicator>>,
    pub tran: Vec<ConstraintCircuitMonad<DualRowIndicator>>,
    pub term: Vec<ConstraintCircuitMonad<SingleRowIndicator>>,
}

impl AllSubstitutions {
    /// Generate code that evaluates all substitution rules in order.
    /// This includes generating the columns that are to be filled using the substitution rules.
    pub fn generate_degree_lowering_table_code(&self) -> TokenStream {
        let num_new_base_cols = self.base.len();
        let num_new_ext_cols = self.ext.len();

        // A zero-variant enum cannot be annotated with `repr(usize)`.
        let base_repr_usize = match num_new_base_cols == 0 {
            true => quote!(),
            false => quote!(#[repr(usize)]),
        };
        let ext_repr_usize = match num_new_ext_cols == 0 {
            true => quote!(),
            false => quote!(#[repr(usize)]),
        };
        let use_challenge_ids = match num_new_ext_cols == 0 {
            true => quote!(),
            false => quote!(
                use crate::table::challenges::ChallengeId::*;
            ),
        };

        let base_columns = (0..num_new_base_cols)
            .map(|i| format_ident!("DegreeLoweringBaseCol{i}"))
            .map(|ident| quote!(#ident))
            .collect_vec();
        let ext_columns = (0..num_new_ext_cols)
            .map(|i| format_ident!("DegreeLoweringExtCol{i}"))
            .map(|ident| quote!(#ident))
            .collect_vec();

        let fill_base_columns_code = self.base.generate_fill_base_columns_code();
        let fill_ext_columns_code = self.ext.generate_fill_ext_columns_code();

        quote!(
            //! The degree lowering table contains the introduced variables that allow
            //! lowering the degree of the AIR. See
            //! [`crate::table::master_table::AIR_TARGET_DEGREE`]
            //! for additional information.
            //!
            //! This file has been auto-generated. Any modifications _will_ be lost.
            //! To re-generate, execute:
            //! `cargo run --bin constraint-evaluation-generator`

            use ndarray::s;
            use ndarray::ArrayView2;
            use ndarray::ArrayViewMut2;
            use strum::Display;
            use strum::EnumCount;
            use strum::EnumIter;
            use twenty_first::prelude::BFieldElement;
            use twenty_first::prelude::XFieldElement;

            #use_challenge_ids
            use crate::table::challenges::Challenges;
            use crate::table::master_table::NUM_BASE_COLUMNS;
            use crate::table::master_table::NUM_EXT_COLUMNS;

            pub const BASE_WIDTH: usize = DegreeLoweringBaseTableColumn::COUNT;
            pub const EXT_WIDTH: usize = DegreeLoweringExtTableColumn::COUNT;
            pub const FULL_WIDTH: usize = BASE_WIDTH + EXT_WIDTH;

            #base_repr_usize
            #[derive(Debug, Display, Copy, Clone, Eq, PartialEq, Hash, EnumCount, EnumIter)]
            pub enum DegreeLoweringBaseTableColumn {
                #(#base_columns),*
            }

            #ext_repr_usize
            #[derive(Debug, Display, Copy, Clone, Eq, PartialEq, Hash, EnumCount, EnumIter)]
            pub enum DegreeLoweringExtTableColumn {
                #(#ext_columns),*
            }

            #[derive(Debug, Copy, Clone, Eq, PartialEq)]
            pub struct DegreeLoweringTable;

            impl DegreeLoweringTable {
                #fill_base_columns_code
                #fill_ext_columns_code
            }
        )
    }
}

impl Substitutions {
    fn len(&self) -> usize {
        self.init.len() + self.cons.len() + self.tran.len() + self.term.len()
    }

    fn generate_fill_base_columns_code(&self) -> TokenStream {
        let derived_section_init_start =
            table::NUM_BASE_COLUMNS - degree_lowering_table::BASE_WIDTH;
        let derived_section_cons_start = derived_section_init_start + self.init.len();
        let derived_section_tran_start = derived_section_cons_start + self.cons.len();
        let derived_section_term_start = derived_section_tran_start + self.tran.len();

        let init_col_indices = (0..self.init.len())
            .map(|i| i + derived_section_init_start)
            .collect_vec();
        let cons_col_indices = (0..self.cons.len())
            .map(|i| i + derived_section_cons_start)
            .collect_vec();
        let tran_col_indices = (0..self.tran.len())
            .map(|i| i + derived_section_tran_start)
            .collect_vec();
        let term_col_indices = (0..self.term.len())
            .map(|i| i + derived_section_term_start)
            .collect_vec();

        let init_substitutions = Self::several_substitution_rules_to_code(&self.init);
        let cons_substitutions = Self::several_substitution_rules_to_code(&self.cons);
        let tran_substitutions = Self::several_substitution_rules_to_code(&self.tran);
        let term_substitutions = Self::several_substitution_rules_to_code(&self.term);

        let init_substitutions =
            Self::base_single_row_substitutions(&init_col_indices, &init_substitutions);
        let cons_substitutions =
            Self::base_single_row_substitutions(&cons_col_indices, &cons_substitutions);
        let tran_substitutions =
            Self::base_dual_row_substitutions(&tran_col_indices, &tran_substitutions);
        let term_substitutions =
            Self::base_single_row_substitutions(&term_col_indices, &term_substitutions);

        quote!(
        #[allow(unused_variables)]
        pub fn fill_derived_base_columns(mut master_base_table: ArrayViewMut2<BFieldElement>) {
            assert_eq!(NUM_BASE_COLUMNS, master_base_table.ncols());
            #init_substitutions
            #cons_substitutions
            #tran_substitutions
            #term_substitutions
        }
        )
    }

    fn generate_fill_ext_columns_code(&self) -> TokenStream {
        let derived_section_init_start = table::NUM_EXT_COLUMNS - degree_lowering_table::EXT_WIDTH;
        let derived_section_cons_start = derived_section_init_start + self.init.len();
        let derived_section_tran_start = derived_section_cons_start + self.cons.len();
        let derived_section_term_start = derived_section_tran_start + self.tran.len();

        let init_col_indices = (0..self.init.len())
            .map(|i| i + derived_section_init_start)
            .collect_vec();
        let cons_col_indices = (0..self.cons.len())
            .map(|i| i + derived_section_cons_start)
            .collect_vec();
        let tran_col_indices = (0..self.tran.len())
            .map(|i| i + derived_section_tran_start)
            .collect_vec();
        let term_col_indices = (0..self.term.len())
            .map(|i| i + derived_section_term_start)
            .collect_vec();

        let init_substitutions = Self::several_substitution_rules_to_code(&self.init);
        let cons_substitutions = Self::several_substitution_rules_to_code(&self.cons);
        let tran_substitutions = Self::several_substitution_rules_to_code(&self.tran);
        let term_substitutions = Self::several_substitution_rules_to_code(&self.term);

        let init_substitutions =
            Self::ext_single_row_substitutions(&init_col_indices, &init_substitutions);
        let cons_substitutions =
            Self::ext_single_row_substitutions(&cons_col_indices, &cons_substitutions);
        let tran_substitutions =
            Self::ext_dual_row_substitutions(&tran_col_indices, &tran_substitutions);
        let term_substitutions =
            Self::ext_single_row_substitutions(&term_col_indices, &term_substitutions);

        quote!(
            #[allow(unused_variables)]
            #[allow(unused_mut)]
            pub fn fill_derived_ext_columns(
                master_base_table: ArrayView2<BFieldElement>,
                mut master_ext_table: ArrayViewMut2<XFieldElement>,
                challenges: &Challenges,
            ) {
                assert_eq!(NUM_BASE_COLUMNS, master_base_table.ncols());
                assert_eq!(NUM_EXT_COLUMNS, master_ext_table.ncols());
                assert_eq!(master_base_table.nrows(), master_ext_table.nrows());
                #init_substitutions
                #cons_substitutions
                #tran_substitutions
                #term_substitutions
            }
        )
    }

    fn several_substitution_rules_to_code<II: InputIndicator>(
        substitution_rules: &[ConstraintCircuitMonad<II>],
    ) -> Vec<TokenStream> {
        substitution_rules
            .iter()
            .map(|c| Self::substitution_rule_to_code(c.circuit.as_ref().borrow().to_owned()))
            .collect()
    }

    /// Given a substitution rule, i.e., a `ConstraintCircuit` of the form `x - expr`, generate code
    /// that evaluates `expr`.
    fn substitution_rule_to_code<II: InputIndicator>(
        circuit: ConstraintCircuit<II>,
    ) -> TokenStream {
        let CircuitExpression::BinaryOperation(BinOp::Sub, new_var, expr) = circuit.expression
        else {
            panic!("Substitution rule must be a subtraction.");
        };
        let CircuitExpression::Input(_) = new_var.as_ref().borrow().expression else {
            panic!("Substitution rule must be a simple substitution.");
        };

        let expr = expr.as_ref().borrow().to_owned();
        RustBackend::evaluate_single_node(usize::MAX, &expr, &HashSet::new())
    }

    fn base_single_row_substitutions(
        indices: &[usize],
        substitutions: &[TokenStream],
    ) -> TokenStream {
        assert_eq!(indices.len(), substitutions.len());
        if indices.is_empty() {
            return quote!();
        }
        quote!(
            master_base_table.rows_mut().into_iter().for_each(|mut row| {
            #(
            let (base_row, mut det_col) =
                row.multi_slice_mut((s![..#indices],s![#indices..=#indices]));
            det_col[0] = #substitutions;
            )*
            });
        )
    }

    fn base_dual_row_substitutions(
        indices: &[usize],
        substitutions: &[TokenStream],
    ) -> TokenStream {
        assert_eq!(indices.len(), substitutions.len());
        if indices.is_empty() {
            return quote!();
        }
        quote!(
            for curr_row_idx in 0..master_base_table.nrows() - 1 {
                let next_row_idx = curr_row_idx + 1;
                let (mut curr_base_row, next_base_row) = master_base_table.multi_slice_mut((
                    s![curr_row_idx..=curr_row_idx, ..],
                    s![next_row_idx..=next_row_idx, ..],
                ));
                let mut curr_base_row = curr_base_row.row_mut(0);
                let next_base_row = next_base_row.row(0);
                #(
                let (current_base_row, mut det_col) =
                    curr_base_row.multi_slice_mut((s![..#indices], s![#indices..=#indices]));
                det_col[0] = #substitutions;
                )*
            }
        )
    }

    fn ext_single_row_substitutions(
        indices: &[usize],
        substitutions: &[TokenStream],
    ) -> TokenStream {
        assert_eq!(indices.len(), substitutions.len());
        if indices.is_empty() {
            return quote!();
        }
        quote!(
            for row_idx in 0..master_base_table.nrows() - 1 {
                let base_row = master_base_table.row(row_idx);
                let mut extension_row = master_ext_table.row_mut(row_idx);
                #(
                let (ext_row, mut det_col) =
                    extension_row.multi_slice_mut((s![..#indices],s![#indices..=#indices]));
                det_col[0] = #substitutions;
                )*
            }
        )
    }

    fn ext_dual_row_substitutions(indices: &[usize], substitutions: &[TokenStream]) -> TokenStream {
        assert_eq!(indices.len(), substitutions.len());
        if indices.is_empty() {
            return quote!();
        }
        quote!(
            for curr_row_idx in 0..master_base_table.nrows() - 1 {
                let next_row_idx = curr_row_idx + 1;
                let current_base_row = master_base_table.row(curr_row_idx);
                let next_base_row = master_base_table.row(next_row_idx);
                let (mut curr_ext_row, next_ext_row) = master_ext_table.multi_slice_mut((
                    s![curr_row_idx..=curr_row_idx, ..],
                    s![next_row_idx..=next_row_idx, ..],
                ));
                let mut curr_ext_row = curr_ext_row.row_mut(0);
                let next_ext_row = next_ext_row.row(0);
                #(
                let (current_ext_row, mut det_col) =
                    curr_ext_row.multi_slice_mut((s![..#indices], s![#indices..=#indices]));
                det_col[0] = #substitutions;
                )*
            }
        )
    }
}
