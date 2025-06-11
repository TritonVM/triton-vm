use constraint_circuit::BinOp;
use constraint_circuit::CircuitExpression;
use constraint_circuit::ConstraintCircuit;
use constraint_circuit::ConstraintCircuitMonad;
use constraint_circuit::DegreeLoweringInfo;
use constraint_circuit::DualRowIndicator;
use constraint_circuit::InputIndicator;
use constraint_circuit::SingleRowIndicator;
use itertools::Itertools;
use proc_macro2::TokenStream;
use quote::format_ident;
use quote::quote;

use crate::codegen::RustBackend;

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct AllSubstitutions {
    pub main: Substitutions,
    pub aux: Substitutions,
}

#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub struct Substitutions {
    pub lowering_info: DegreeLoweringInfo,
    pub init: Vec<ConstraintCircuitMonad<SingleRowIndicator>>,
    pub cons: Vec<ConstraintCircuitMonad<SingleRowIndicator>>,
    pub tran: Vec<ConstraintCircuitMonad<DualRowIndicator>>,
    pub term: Vec<ConstraintCircuitMonad<SingleRowIndicator>>,
}

impl AllSubstitutions {
    /// Generate code that evaluates all substitution rules in order.
    /// This includes generating the columns that are to be filled using the
    /// substitution rules.
    pub fn generate_degree_lowering_table_code(&self) -> TokenStream {
        let num_new_main_cols = self.main.len();
        let num_new_aux_cols = self.aux.len();

        // A zero-variant enum cannot be annotated with `repr(usize)`.
        let main_repr_usize = match num_new_main_cols {
            0 => quote!(),
            _ => quote!(#[repr(usize)]),
        };
        let aux_repr_usize = match num_new_aux_cols {
            0 => quote!(),
            _ => quote!(#[repr(usize)]),
        };

        let main_columns = (0..num_new_main_cols)
            .map(|i| format_ident!("DegreeLoweringMainCol{i}"))
            .map(|ident| quote!(#ident))
            .collect_vec();
        let aux_columns = (0..num_new_aux_cols)
            .map(|i| format_ident!("DegreeLoweringAuxCol{i}"))
            .map(|ident| quote!(#ident))
            .collect_vec();

        let fill_main_columns_code = self.main.generate_fill_main_columns_code();
        let fill_aux_columns_code = self.aux.generate_fill_aux_columns_code();

        quote!(
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
            use air::table_column::MasterMainColumn;
            use air::table_column::MasterAuxColumn;

            use crate::challenges::Challenges;

            #main_repr_usize
            #[derive(Debug, Display, Copy, Clone, Eq, PartialEq, Hash, EnumCount, EnumIter)]
            pub enum DegreeLoweringMainColumn {
                #(#main_columns),*
            }

            impl MasterMainColumn for DegreeLoweringMainColumn {
                fn main_index(&self) -> usize {
                    (*self) as usize
                }

                fn master_main_index(&self) -> usize {
                    // hardcore domain-specific knowledge, and bad style
                    air::table::U32_TABLE_END + self.main_index()
                }
            }

            #aux_repr_usize
            #[derive(Debug, Display, Copy, Clone, Eq, PartialEq, Hash, EnumCount, EnumIter)]
            pub enum DegreeLoweringAuxColumn {
                #(#aux_columns),*
            }

            impl MasterAuxColumn for DegreeLoweringAuxColumn {
                fn aux_index(&self) -> usize {
                    (*self) as usize
                }

                fn master_aux_index(&self) -> usize {
                    // hardcore domain-specific knowledge, and bad style
                    air::table::AUX_U32_TABLE_END + self.aux_index()
                }
            }

            #[derive(Debug, Copy, Clone, Eq, PartialEq)]
            pub struct DegreeLoweringTable;

            impl DegreeLoweringTable {
                #fill_main_columns_code
                #fill_aux_columns_code
            }
        )
    }
}

impl Substitutions {
    fn len(&self) -> usize {
        self.init.len() + self.cons.len() + self.tran.len() + self.term.len()
    }

    fn generate_fill_main_columns_code(&self) -> TokenStream {
        let derived_section_init_start = self.lowering_info.num_main_cols;
        let derived_section_cons_start = derived_section_init_start + self.init.len();
        let derived_section_tran_start = derived_section_cons_start + self.cons.len();
        let derived_section_term_start = derived_section_tran_start + self.tran.len();

        let init_substitutions = Self::several_substitution_rules_to_code(&self.init);
        let cons_substitutions = Self::several_substitution_rules_to_code(&self.cons);
        let tran_substitutions = Self::several_substitution_rules_to_code(&self.tran);
        let term_substitutions = Self::several_substitution_rules_to_code(&self.term);

        let init_substitutions =
            Self::main_single_row_substitutions(derived_section_init_start, &init_substitutions);
        let cons_substitutions =
            Self::main_single_row_substitutions(derived_section_cons_start, &cons_substitutions);
        let tran_substitutions =
            Self::main_dual_row_substitutions(derived_section_tran_start, &tran_substitutions);
        let term_substitutions =
            Self::main_single_row_substitutions(derived_section_term_start, &term_substitutions);

        quote!(
            #[allow(unused_variables)]
            pub fn fill_derived_main_columns(
                mut master_main_table: ArrayViewMut2<BFieldElement>
            ) {
                let num_expected_columns =
                    crate::table::master_table::MasterMainTable::NUM_COLUMNS;
                assert_eq!(num_expected_columns, master_main_table.ncols());
                #init_substitutions
                #cons_substitutions
                #tran_substitutions
                #term_substitutions
            }
        )
    }

    fn generate_fill_aux_columns_code(&self) -> TokenStream {
        let derived_section_init_start = self.lowering_info.num_aux_cols;
        let derived_section_cons_start = derived_section_init_start + self.init.len();
        let derived_section_tran_start = derived_section_cons_start + self.cons.len();
        let derived_section_term_start = derived_section_tran_start + self.tran.len();

        let init_substitutions = Self::several_substitution_rules_to_code(&self.init);
        let cons_substitutions = Self::several_substitution_rules_to_code(&self.cons);
        let tran_substitutions = Self::several_substitution_rules_to_code(&self.tran);
        let term_substitutions = Self::several_substitution_rules_to_code(&self.term);

        let init_substitutions =
            Self::aux_single_row_substitutions(derived_section_init_start, &init_substitutions);
        let cons_substitutions =
            Self::aux_single_row_substitutions(derived_section_cons_start, &cons_substitutions);
        let tran_substitutions =
            Self::aux_dual_row_substitutions(derived_section_tran_start, &tran_substitutions);
        let term_substitutions =
            Self::aux_single_row_substitutions(derived_section_term_start, &term_substitutions);

        quote!(
            #[allow(unused_variables)]
            #[allow(unused_mut)]
            pub fn fill_derived_aux_columns(
                master_main_table: ArrayView2<BFieldElement>,
                mut master_aux_table: ArrayViewMut2<XFieldElement>,
                challenges: &Challenges,
            ) {
                let num_expected_main_columns =
                    crate::table::master_table::MasterMainTable::NUM_COLUMNS;
                let num_expected_aux_columns =
                    crate::table::master_table::MasterAuxTable::NUM_COLUMNS;
                assert_eq!(num_expected_main_columns, master_main_table.ncols());
                assert_eq!(num_expected_aux_columns, master_aux_table.ncols());
                assert_eq!(master_main_table.nrows(), master_aux_table.nrows());
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

    /// Given a substitution rule, i.e., a `ConstraintCircuit` of the form `x -
    /// expr`, generate code that evaluates `expr`.
    fn substitution_rule_to_code<II: InputIndicator>(
        circuit: ConstraintCircuit<II>,
    ) -> TokenStream {
        let CircuitExpression::BinOp(BinOp::Add, new_var, expr) = circuit.expression else {
            panic!("Substitution rule must be a subtraction, i.e., addition of `x` and `-expr`.");
        };
        let CircuitExpression::Input(_) = new_var.borrow().expression else {
            panic!("Substitution rule must be a simple substitution.");
        };
        let expr = expr.borrow();
        let CircuitExpression::BinOp(BinOp::Mul, neg_one, expr) = &expr.expression else {
            panic!("Substitution rule must be a subtraction.");
        };
        assert!(neg_one.borrow().is_neg_one());

        let expr = expr.borrow();
        RustBackend::default().evaluate_single_node(&expr)
    }

    fn main_single_row_substitutions(
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
                master_main_table.multi_slice_mut(
                    (
                        s![.., 0..#section_start_index],
                        s![.., #section_start_index..#section_start_index+#num_substitutions],
                    )
                );
            Zip::from(original_part.rows())
                .and(current_section.rows_mut())
                .par_for_each(|original_row, mut section_row| {
                    let mut main_row = original_row.to_owned();
                    #(
                        section_row[#indices] = #substitutions;
                        main_row.push(Axis(0), section_row.slice(s![#indices])).unwrap();
                    )*
                });
        )
    }

    fn main_dual_row_substitutions(
        section_start_index: usize,
        substitutions: &[TokenStream],
    ) -> TokenStream {
        let num_substitutions = substitutions.len();
        let indices = (0..substitutions.len()).collect_vec();
        if indices.is_empty() {
            return quote!();
        }
        quote!(
            let num_rows = master_main_table.nrows();
            let (original_part, mut current_section) =
                master_main_table.multi_slice_mut(
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
                    let current_main_row_slice =
                        original_part.slice(s![current_row_index..=current_row_index, ..]);
                    let next_main_row_slice =
                        original_part.slice(s![next_row_index..=next_row_index, ..]);
                    let mut current_main_row = current_main_row_slice.row(0).to_owned();
                    let next_main_row = next_main_row_slice.row(0);
                    #(
                        section_row[#indices] = #substitutions;
                        current_main_row.push(Axis(0), section_row.slice(s![#indices])).unwrap();
                    )*
            });
        )
    }

    fn aux_single_row_substitutions(
        section_start_index: usize,
        substitutions: &[TokenStream],
    ) -> TokenStream {
        let num_substitutions = substitutions.len();
        let indices = (0..substitutions.len()).collect_vec();
        if indices.is_empty() {
            return quote!();
        }
        quote!(
            let (original_part, mut current_section) = master_aux_table.multi_slice_mut(
                (
                    s![.., 0..#section_start_index],
                    s![.., #section_start_index..#section_start_index+#num_substitutions],
                )
            );
            Zip::from(master_main_table.rows())
                .and(original_part.rows())
                .and(current_section.rows_mut())
                .par_for_each(
                    |main_table_row, original_row, mut section_row| {
                        let mut auxiliary_row = original_row.to_owned();
                        #(
                        let (original_row_auxiliary_row, mut det_col) =
                            section_row.multi_slice_mut((s![..#indices],s![#indices..=#indices]));
                        det_col[0] = #substitutions;
                        auxiliary_row.push(Axis(0), det_col.slice(s![0])).unwrap();
                        )*
                    }
                );
        )
    }

    fn aux_dual_row_substitutions(
        section_start_index: usize,
        substitutions: &[TokenStream],
    ) -> TokenStream {
        let num_substitutions = substitutions.len();
        let indices = (0..substitutions.len()).collect_vec();
        if indices.is_empty() {
            return quote!();
        }
        quote!(
            let num_rows = master_main_table.nrows();
            let (original_part, mut current_section) = master_aux_table.multi_slice_mut(
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
                    let current_main_row = master_main_table.row(current_row_index);
                    let next_main_row = master_main_table.row(next_row_index);
                    let mut current_aux_row = original_part.row(current_row_index).to_owned();
                    let next_aux_row = original_part.row(next_row_index);
                    #(
                        section_row[#indices]= #substitutions;
                        current_aux_row.push(Axis(0), section_row.slice(s![#indices])).unwrap();
                    )*
                });
        )
    }
}
