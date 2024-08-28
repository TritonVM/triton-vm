use constraint_builder::BinOp;
use constraint_builder::CircuitExpression;
use constraint_builder::ConstraintCircuit;
use constraint_builder::ConstraintCircuitMonad;
use constraint_builder::DegreeLoweringInfo;
use constraint_builder::DualRowIndicator;
use constraint_builder::InputIndicator;
use constraint_builder::SingleRowIndicator;
use itertools::Itertools;
use proc_macro2::TokenStream;
use quote::format_ident;
use quote::quote;

use crate::codegen::constraints::RustBackend;

pub(crate) struct AllSubstitutions {
    pub base: Substitutions,
    pub ext: Substitutions,
}

pub(crate) struct Substitutions {
    pub lowering_info: DegreeLoweringInfo,
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
        let base_repr_usize = match num_new_base_cols {
            0 => quote!(),
            _ => quote!(#[repr(usize)]),
        };
        let ext_repr_usize = match num_new_ext_cols {
            0 => quote!(),
            _ => quote!(#[repr(usize)]),
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
            //! [`table::master_table::AIR_TARGET_DEGREE`]
            //! for additional information.
            //!
            //! This file has been auto-generated. Any modifications _will_ be lost.
            //! To re-generate, execute:
            //! `cargo run --bin constraint-evaluation-generator`

            use ndarray::Array1;
            use ndarray::s;
            use ndarray::ArrayView2;
            use ndarray::ArrayViewMut2;
            use ndarray::Axis;
            use ndarray::Zip;
            use strum::Display;
            use strum::EnumCount;
            use strum::EnumIter;
            use twenty_first::prelude::BFieldElement;
            use twenty_first::prelude::XFieldElement;

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
        let derived_section_init_start = self.lowering_info.num_base_cols;
        let derived_section_cons_start = derived_section_init_start + self.init.len();
        let derived_section_tran_start = derived_section_cons_start + self.cons.len();
        let derived_section_term_start = derived_section_tran_start + self.tran.len();

        let init_substitutions = Self::several_substitution_rules_to_code(&self.init);
        let cons_substitutions = Self::several_substitution_rules_to_code(&self.cons);
        let tran_substitutions = Self::several_substitution_rules_to_code(&self.tran);
        let term_substitutions = Self::several_substitution_rules_to_code(&self.term);

        let init_substitutions =
            Self::base_single_row_substitutions(derived_section_init_start, &init_substitutions);
        let cons_substitutions =
            Self::base_single_row_substitutions(derived_section_cons_start, &cons_substitutions);
        let tran_substitutions =
            Self::base_dual_row_substitutions(derived_section_tran_start, &tran_substitutions);
        let term_substitutions =
            Self::base_single_row_substitutions(derived_section_term_start, &term_substitutions);

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
        let derived_section_init_start = self.lowering_info.num_ext_cols;
        let derived_section_cons_start = derived_section_init_start + self.init.len();
        let derived_section_tran_start = derived_section_cons_start + self.cons.len();
        let derived_section_term_start = derived_section_tran_start + self.tran.len();

        let init_substitutions = Self::several_substitution_rules_to_code(&self.init);
        let cons_substitutions = Self::several_substitution_rules_to_code(&self.cons);
        let tran_substitutions = Self::several_substitution_rules_to_code(&self.tran);
        let term_substitutions = Self::several_substitution_rules_to_code(&self.term);

        let init_substitutions =
            Self::ext_single_row_substitutions(derived_section_init_start, &init_substitutions);
        let cons_substitutions =
            Self::ext_single_row_substitutions(derived_section_cons_start, &cons_substitutions);
        let tran_substitutions =
            Self::ext_dual_row_substitutions(derived_section_tran_start, &tran_substitutions);
        let term_substitutions =
            Self::ext_single_row_substitutions(derived_section_term_start, &term_substitutions);

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
            .map(|c| Self::substitution_rule_to_code(c.circuit.borrow().to_owned()))
            .collect()
    }

    /// Given a substitution rule, i.e., a `ConstraintCircuit` of the form `x - expr`, generate code
    /// that evaluates `expr`.
    fn substitution_rule_to_code<II: InputIndicator>(
        circuit: ConstraintCircuit<II>,
    ) -> TokenStream {
        let CircuitExpression::BinaryOperation(BinOp::Add, new_var, expr) = circuit.expression
        else {
            panic!("Substitution rule must be a subtraction, i.e., addition of `x` and `-expr`.");
        };
        let CircuitExpression::Input(_) = new_var.borrow().expression else {
            panic!("Substitution rule must be a simple substitution.");
        };
        let expr = expr.borrow();
        let CircuitExpression::BinaryOperation(BinOp::Mul, neg_one, expr) = &expr.expression else {
            panic!("Substitution rule must be a subtraction.");
        };
        assert!(neg_one.borrow().is_neg_one());

        let expr = expr.borrow();
        RustBackend::default().evaluate_single_node(&expr)
    }

    fn base_single_row_substitutions(
        section_start_index: usize,
        substitutions: &[TokenStream],
    ) -> TokenStream {
        let num_substitutions = substitutions.len();
        let indices = (0..num_substitutions).collect_vec();
        if indices.is_empty() {
            return quote!();
        }
        quote!(
            let (original_part, mut current_section) =
                master_base_table.multi_slice_mut(
                    (
                        s![.., 0..#section_start_index],
                        s![.., #section_start_index..#section_start_index+#num_substitutions],
                    )
                );
            Zip::from(original_part.rows())
                .and(current_section.rows_mut())
                .par_for_each(|original_row, mut section_row| {
                    let mut base_row = original_row.to_owned();
                    #(
                        section_row[#indices] = #substitutions;
                        base_row.push(Axis(0), section_row.slice(s![#indices])).unwrap();
                    )*
                });
        )
    }

    fn base_dual_row_substitutions(
        section_start_index: usize,
        substitutions: &[TokenStream],
    ) -> TokenStream {
        let num_substitutions = substitutions.len();
        let indices = (0..substitutions.len()).collect_vec();
        if indices.is_empty() {
            return quote!();
        }
        quote!(
            let num_rows = master_base_table.nrows();
            let (original_part, mut current_section) =
                master_base_table.multi_slice_mut(
                    (
                        s![.., 0..#section_start_index],
                        s![.., #section_start_index..#section_start_index+#num_substitutions],
                    )
                );
            let row_indices = Array1::from_vec((0..num_rows - 1).collect::<Vec<_>>());
            Zip::from(current_section.slice_mut(s![0..num_rows-1, ..]).rows_mut())
                .and(row_indices.view())
                .par_for_each( |mut section_row, &current_row_index| {
                    let next_row_index = current_row_index + 1;
                    let current_base_row_slice = original_part.slice(s![current_row_index..=current_row_index, ..]);
                    let next_base_row_slice = original_part.slice(s![next_row_index..=next_row_index, ..]);
                    let mut current_base_row = current_base_row_slice.row(0).to_owned();
                    let next_base_row = next_base_row_slice.row(0);
                    #(
                        section_row[#indices] = #substitutions;
                        current_base_row.push(Axis(0), section_row.slice(s![#indices])).unwrap();
                    )*
                });
        )
    }

    fn ext_single_row_substitutions(
        section_start_index: usize,
        substitutions: &[TokenStream],
    ) -> TokenStream {
        let num_substitutions = substitutions.len();
        let indices = (0..substitutions.len()).collect_vec();
        if indices.is_empty() {
            return quote!();
        }
        quote!(
            let (original_part, mut current_section) = master_ext_table.multi_slice_mut(
                (
                    s![.., 0..#section_start_index],
                    s![.., #section_start_index..#section_start_index+#num_substitutions],
                )
            );
            Zip::from(master_base_table.rows())
                .and(original_part.rows())
                .and(current_section.rows_mut())
                .par_for_each(
                    |base_table_row, original_row, mut section_row| {
                        let mut extension_row = original_row.to_owned();
                        #(
                            let (original_row_extension_row, mut det_col) =
                                section_row.multi_slice_mut((s![..#indices],s![#indices..=#indices]));
                            det_col[0] = #substitutions;
                            extension_row.push(Axis(0), det_col.slice(s![0])).unwrap();
                        )*
                    }
                );
        )
    }

    fn ext_dual_row_substitutions(
        section_start_index: usize,
        substitutions: &[TokenStream],
    ) -> TokenStream {
        let num_substitutions = substitutions.len();
        let indices = (0..substitutions.len()).collect_vec();
        if indices.is_empty() {
            return quote!();
        }
        quote!(
            let num_rows = master_base_table.nrows();
            let (original_part, mut current_section) = master_ext_table.multi_slice_mut(
                (
                    s![.., 0..#section_start_index],
                    s![.., #section_start_index..#section_start_index+#num_substitutions],
                )
            );
            let row_indices = Array1::from_vec((0..num_rows - 1).collect::<Vec<_>>());
            Zip::from(current_section.slice_mut(s![0..num_rows-1, ..]).rows_mut())
                .and(row_indices.view())
                .par_for_each(|mut section_row, &current_row_index| {
                    let next_row_index = current_row_index + 1;
                    let current_base_row = master_base_table.row(current_row_index);
                    let next_base_row = master_base_table.row(next_row_index);
                    let mut current_ext_row = original_part.row(current_row_index).to_owned();
                    let next_ext_row = original_part.row(next_row_index);
                    #(
                        section_row[#indices]= #substitutions;
                        current_ext_row.push(Axis(0), section_row.slice(s![#indices])).unwrap();
                    )*
                });
        )
    }
}
