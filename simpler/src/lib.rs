//! Constraint circuits are a way to represent constraint polynomials in a way
//! that is amenable to optimizations. The constraint circuit is a directed
//! acyclic graph (DAG) of [`CircuitExpression`]s, where each
//! `CircuitExpression` is a node in the graph. The edges of the graph are
//! labeled with [`BinOp`]s. The leafs of the graph are the inputs to the
//! constraint polynomial, and the (multiple) roots of the graph are the outputs
//! of all the constraint polynomials, with each root corresponding to a
//! different constraint polynomial. Because the graph has multiple roots, it is
//! called a “multitree.”

use std::cmp;

use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FmtResult;
use std::hash::Hash;
use std::hash::Hasher;

use std::ops::Add;

use std::ops::Mul;




use arbitrary::Arbitrary;
use ndarray::ArrayView2;
use num_traits::One;
use num_traits::Zero;
use quote::ToTokens;
use quote::quote;
use twenty_first::prelude::*;

mod private {
    // A public but un-nameable type for sealing traits.
    pub trait Seal {}
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub struct DegreeLoweringInfo {
    /// The degree after degree lowering. Must be greater than 1.
    pub target_degree: isize,

    /// The total number of main columns _before_ degree lowering has happened.
    pub num_main_cols: usize,

    /// The total number of auxiliary columns _before_ degree lowering has
    /// happened.
    pub num_aux_cols: usize,
}

#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum BinOp {
    Add,
    Mul,
}

impl Display for BinOp {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match self {
            BinOp::Add => write!(f, "+"),
            BinOp::Mul => write!(f, "*"),
        }
    }
}

impl ToTokens for BinOp {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        match self {
            BinOp::Add => tokens.extend(quote!(+)),
            BinOp::Mul => tokens.extend(quote!(*)),
        }
    }
}

impl BinOp {
    pub fn operation<L, R, O>(&self, lhs: L, rhs: R) -> O
    where
        L: Add<R, Output = O> + Mul<R, Output = O>,
    {
        match self {
            BinOp::Add => lhs + rhs,
            BinOp::Mul => lhs * rhs,
        }
    }
}

/// Describes the position of a variable in a constraint polynomial in the row
/// layout applicable for a certain kind of constraint polynomial.
///
/// The position of variable in a constraint polynomial is, in principle, a
/// `usize`. However, depending on the type of the constraint polynomial, this
/// index may be an index into a single row (for initial, consistency and
/// terminal constraints), or a pair of adjacent rows (for transition
/// constraints). Additionally, the index may refer to a column in the main
/// table, or a column in the auxiliary table. This trait abstracts over these
/// possibilities, and provides a uniform interface for accessing the index.
///
/// Having `Copy + Hash + Eq` helps to put `InputIndicator`s into containers.
///
/// This is a _sealed_ trait. It is not intended (or possible) to implement this
/// trait outside the crate defining it.
pub trait InputIndicator: Debug + Display + Copy + Hash + Eq + ToTokens + private::Seal {
    /// `true` iff `self` refers to a column in the main table.
    fn is_main_table_column(&self) -> bool;

    /// `true` iff `self` refers to the current row.
    fn is_current_row(&self) -> bool;

    /// The index of the indicated (main or auxiliary) column.
    fn column(&self) -> usize;

    fn main_table_input(index: usize) -> Self;
    fn aux_table_input(index: usize) -> Self;

    fn evaluate(
        &self,
        main_table: ArrayView2<BFieldElement>,
        aux_table: ArrayView2<XFieldElement>,
    ) -> XFieldElement;
}

/// The position of a variable in a constraint polynomial that operates on a
/// single row of the execution trace.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Arbitrary)]
pub enum SingleRowIndicator {
    Main(usize),
    Aux(usize),
}

impl private::Seal for SingleRowIndicator {}

impl Display for SingleRowIndicator {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let input_indicator: String = match self {
            Self::Main(i) => format!("main_row[{i}]"),
            Self::Aux(i) => format!("aux_row[{i}]"),
        };

        write!(f, "{input_indicator}")
    }
}

impl ToTokens for SingleRowIndicator {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        match self {
            Self::Main(i) => tokens.extend(quote!(main_row[#i])),
            Self::Aux(i) => tokens.extend(quote!(aux_row[#i])),
        }
    }
}

impl InputIndicator for SingleRowIndicator {
    fn is_main_table_column(&self) -> bool {
        matches!(self, Self::Main(_))
    }

    fn is_current_row(&self) -> bool {
        true
    }

    fn column(&self) -> usize {
        match self {
            Self::Main(i) | Self::Aux(i) => *i,
        }
    }

    fn main_table_input(index: usize) -> Self {
        Self::Main(index)
    }

    fn aux_table_input(index: usize) -> Self {
        Self::Aux(index)
    }

    fn evaluate(
        &self,
        main_table: ArrayView2<BFieldElement>,
        aux_table: ArrayView2<XFieldElement>,
    ) -> XFieldElement {
        match self {
            Self::Main(i) => main_table[[0, *i]].lift(),
            Self::Aux(i) => aux_table[[0, *i]],
        }
    }
}

/// The position of a variable in a constraint polynomial that operates on two
/// rows (current and next) of the execution trace.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, Arbitrary)]
pub enum DualRowIndicator {
    CurrentMain(usize),
    CurrentAux(usize),
    NextMain(usize),
    NextAux(usize),
}

impl private::Seal for DualRowIndicator {}

impl Display for DualRowIndicator {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let input_indicator: String = match self {
            Self::CurrentMain(i) => format!("current_main_row[{i}]"),
            Self::CurrentAux(i) => format!("current_aux_row[{i}]"),
            Self::NextMain(i) => format!("next_main_row[{i}]"),
            Self::NextAux(i) => format!("next_aux_row[{i}]"),
        };

        write!(f, "{input_indicator}")
    }
}

impl ToTokens for DualRowIndicator {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        match self {
            Self::CurrentMain(i) => tokens.extend(quote!(current_main_row[#i])),
            Self::CurrentAux(i) => tokens.extend(quote!(current_aux_row[#i])),
            Self::NextMain(i) => tokens.extend(quote!(next_main_row[#i])),
            Self::NextAux(i) => tokens.extend(quote!(next_aux_row[#i])),
        }
    }
}

impl InputIndicator for DualRowIndicator {
    fn is_main_table_column(&self) -> bool {
        matches!(self, Self::CurrentMain(_) | Self::NextMain(_))
    }

    fn is_current_row(&self) -> bool {
        matches!(self, Self::CurrentMain(_) | Self::CurrentAux(_))
    }

    fn column(&self) -> usize {
        match self {
            Self::CurrentMain(i) | Self::NextMain(i) | Self::CurrentAux(i) | Self::NextAux(i) => *i,
        }
    }

    fn main_table_input(index: usize) -> Self {
        // It seems that the choice between `CurrentMain` and `NextMain` is
        // arbitrary: any transition constraint polynomial is evaluated on both
        // the current and the next row. Hence, both rows are in scope.
        Self::CurrentMain(index)
    }

    fn aux_table_input(index: usize) -> Self {
        Self::CurrentAux(index)
    }

    fn evaluate(
        &self,
        main_table: ArrayView2<BFieldElement>,
        aux_table: ArrayView2<XFieldElement>,
    ) -> XFieldElement {
        match self {
            Self::CurrentMain(i) => main_table[[0, *i]].lift(),
            Self::CurrentAux(i) => aux_table[[0, *i]],
            Self::NextMain(i) => main_table[[1, *i]].lift(),
            Self::NextAux(i) => aux_table[[1, *i]],
        }
    }
}

/// A circuit expression is the recursive data structure that represents the
/// constraint circuit. It is a directed, acyclic graph of binary operations on
/// a) the variables corresponding to columns in the AET, b) constants, and c)
/// challenges. It has multiple roots, making it a “multitree.” Each root
/// corresponds to one constraint.
///
/// The leafs of the tree are
/// - constants in the base field, _i.e._, [`BFieldElement`]s,
/// - constants in the extension field, _i.e._, [`XFieldElement`]s,
/// - input variables, _i.e._, entries from the Algebraic Execution Trace, main
///   or aux, and
/// - challenges, _i.e._, (pseudo-)random values sampled through the Fiat-Shamir
///   heuristic.
///
/// An internal node, representing some binary operation, is either addition or
/// multiplication. The left and right children of the node are the operands of
/// the binary operation. The left and right children are node IDs (indices into
/// the builder's Vec).
#[derive(Debug, Clone)]
pub enum CircuitExpression<II: InputIndicator> {
    BConst(BFieldElement),
    XConst(XFieldElement),
    Input(II),
    Challenge(usize),
    BinOp(BinOp, usize, usize),
}

impl<II: InputIndicator> Hash for CircuitExpression<II> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        match self {
            Self::BConst(bfe) => {
                "bfe".hash(state);
                bfe.hash(state);
            }
            Self::XConst(xfe) => {
                "xfe".hash(state);
                xfe.hash(state);
            }
            Self::Input(index) => {
                "input".hash(state);
                index.hash(state);
            }
            Self::Challenge(table_challenge_id) => {
                "challenge".hash(state);
                table_challenge_id.hash(state);
            }
            Self::BinOp(binop, lhs, rhs) => {
                "binop".hash(state);
                binop.hash(state);
                lhs.hash(state);
                rhs.hash(state);
            }
        }
    }
}

impl<II: InputIndicator> PartialEq for CircuitExpression<II> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (Self::BConst(b), Self::BConst(b_o)) => b == b_o,
            (Self::XConst(x), Self::XConst(x_o)) => x == x_o,
            (Self::Input(i), Self::Input(i_o)) => i == i_o,
            (Self::Challenge(c), Self::Challenge(c_o)) => c == c_o,
            (Self::BinOp(op, l, r), Self::BinOp(op_o, l_o, r_o)) => {
                op == op_o && l == l_o && r == r_o
            }
            _ => false,
        }
    }
}

impl<II: InputIndicator> Hash for ConstraintCircuit<II> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.expression.hash(state)
    }
}

// Hash implementation for ConstraintCircuitMonad removed

/// A wrapper around a [`CircuitExpression`] that manages additional bookkeeping
/// information, such as node id and visited counter.
///
/// In contrast to [`ConstraintCircuitMonad`], this struct cannot manage the
/// state required to insert new nodes.
#[derive(Debug, Clone)]
pub struct ConstraintCircuit<II: InputIndicator> {
    pub id: usize,
    pub ref_count: usize,
    pub expression: CircuitExpression<II>,
}

impl<II: InputIndicator> Eq for ConstraintCircuit<II> {}

impl<II: InputIndicator> PartialEq for ConstraintCircuit<II> {
    /// Calculate equality of circuits. In particular, this function does *not*
    /// attempt to simplify or reduce neutral terms or products. So this
    /// comparison will return false for `a == a + 0`. It will also return
    /// false for `XFieldElement(7) == BFieldElement(7)`
    fn eq(&self, other: &Self) -> bool {
        self.expression == other.expression
    }
}

impl<II: InputIndicator> Display for ConstraintCircuit<II> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match &self.expression {
            CircuitExpression::XConst(xfe) => write!(f, "{xfe}"),
            CircuitExpression::BConst(bfe) => write!(f, "{bfe}"),
            CircuitExpression::Input(input) => write!(f, "{input} "),
            CircuitExpression::Challenge(self_challenge_idx) => write!(f, "{self_challenge_idx}"),
            CircuitExpression::BinOp(operation, lhs, rhs) => {
                write!(f, "(node_{}) {operation} (node_{})", lhs, rhs)
            }
        }
    }
}

impl<II: InputIndicator> ConstraintCircuit<II> {
    fn new(id: usize, expression: CircuitExpression<II>) -> Self {
        Self {
            id,
            ref_count: 0,
            expression,
        }
    }

    /// Reset the reference counters for the entire subtree
    fn reset_ref_count_for_tree(&mut self) {
        self.ref_count = 0;
    }

    /// Assert that all IDs in the builder are valid indices
    pub fn assert_valid_ids(builder: &ConstraintCircuitBuilder<II>, root_ids: &[usize]) {
        for &root_id in root_ids {
            Self::assert_valid_ids_recursive(builder, root_id, &mut std::collections::HashSet::new());
        }
    }

    fn assert_valid_ids_recursive(builder: &ConstraintCircuitBuilder<II>, node_id: usize, visited: &mut std::collections::HashSet<usize>) {
        if visited.contains(&node_id) {
            return;
        }
        visited.insert(node_id);

        let node = builder.get_node(node_id).expect(&format!("Invalid node ID: {}", node_id));
        assert_eq!(node.id, node_id, "Node ID mismatch: expected {}, got {}", node_id, node.id);

        if let CircuitExpression::BinOp(_, lhs, rhs) = &node.expression {
            Self::assert_valid_ids_recursive(builder, *lhs, visited);
            Self::assert_valid_ids_recursive(builder, *rhs, visited);
        }
    }

    /// Return degree of the multivariate polynomial represented by this circuit
    pub fn degree(&self, builder: &ConstraintCircuitBuilder<II>) -> isize {
        if self.is_zero() {
            return -1;
        }

        match &self.expression {
            CircuitExpression::BinOp(binop, lhs, rhs) => {
                let lhs_node = builder.get_node(*lhs).expect("Invalid left node ID");
                let rhs_node = builder.get_node(*rhs).expect("Invalid right node ID");
                let degree_lhs = lhs_node.degree(builder);
                let degree_rhs = rhs_node.degree(builder);
                let degree_additive = cmp::max(degree_lhs, degree_rhs);
                let degree_multiplicative = if cmp::min(degree_lhs, degree_rhs) <= -1 {
                    -1
                } else {
                    degree_lhs + degree_rhs
                };
                match binop {
                    BinOp::Add => degree_additive,
                    BinOp::Mul => degree_multiplicative,
                }
            }
            CircuitExpression::Input(_) => 1,
            CircuitExpression::BConst(_)
            | CircuitExpression::XConst(_)
            | CircuitExpression::Challenge(_) => 0,
        }
    }

    /// All unique reference counters in the subtree, sorted.
    pub fn all_ref_counters(&self, builder: &ConstraintCircuitBuilder<II>) -> Vec<usize> {
        let mut ref_counters = vec![self.ref_count];
        if let CircuitExpression::BinOp(_, lhs, rhs) = &self.expression {
            if let Some(lhs_node) = builder.get_node(*lhs) {
                ref_counters.extend(lhs_node.all_ref_counters(builder));
            }
            if let Some(rhs_node) = builder.get_node(*rhs) {
                ref_counters.extend(rhs_node.all_ref_counters(builder));
            }
        };
        ref_counters.sort_unstable();
        ref_counters.dedup();
        ref_counters
    }

    /// Is the node the constant 0?
    /// Does not catch composite expressions that will always evaluate to zero,
    /// like `0·a`.
    pub fn is_zero(&self) -> bool {
        match self.expression {
            CircuitExpression::BConst(bfe) => bfe.is_zero(),
            CircuitExpression::XConst(xfe) => xfe.is_zero(),
            _ => false,
        }
    }

    /// Is the node the constant 1?
    /// Does not catch composite expressions that will always evaluate to one,
    /// like `1·1`.
    pub fn is_one(&self) -> bool {
        match self.expression {
            CircuitExpression::BConst(bfe) => bfe.is_one(),
            CircuitExpression::XConst(xfe) => xfe.is_one(),
            _ => false,
        }
    }

    pub fn is_neg_one(&self) -> bool {
        match self.expression {
            CircuitExpression::BConst(bfe) => (-bfe).is_one(),
            CircuitExpression::XConst(xfe) => (-xfe).is_one(),
            _ => false,
        }
    }

    /// Recursively check whether this node is composed of only BFieldElements,
    /// i.e., only uses
    /// 1. inputs from main rows,
    /// 2. constants from the B-field, and
    /// 3. binary operations on BFieldElements.
    pub fn evaluates_to_base_element(&self, builder: &ConstraintCircuitBuilder<II>) -> bool {
        match &self.expression {
            CircuitExpression::BConst(_) => true,
            CircuitExpression::XConst(_) => false,
            CircuitExpression::Input(indicator) => indicator.is_main_table_column(),
            CircuitExpression::Challenge(_) => false,
            CircuitExpression::BinOp(_, lhs, rhs) => {
                let lhs_node = builder.get_node(*lhs).expect("Invalid left node ID");
                let rhs_node = builder.get_node(*rhs).expect("Invalid right node ID");
                lhs_node.evaluates_to_base_element(builder) && rhs_node.evaluates_to_base_element(builder)
            }
        }
    }

    pub fn evaluate(
        &self,
        builder: &ConstraintCircuitBuilder<II>,
        main_table: ArrayView2<BFieldElement>,
        aux_table: ArrayView2<XFieldElement>,
        challenges: &[XFieldElement],
    ) -> XFieldElement {
        match &self.expression {
            CircuitExpression::BConst(bfe) => bfe.lift(),
            CircuitExpression::XConst(xfe) => *xfe,
            CircuitExpression::Input(input) => input.evaluate(main_table, aux_table),
            CircuitExpression::Challenge(challenge_id) => challenges[*challenge_id],
            CircuitExpression::BinOp(binop, lhs, rhs) => {
                let lhs_node = builder.get_node(*lhs).expect("Invalid left node ID");
                let rhs_node = builder.get_node(*rhs).expect("Invalid right node ID");
                let lhs_value = lhs_node.evaluate(builder, main_table, aux_table, challenges);
                let rhs_value = rhs_node.evaluate(builder, main_table, aux_table, challenges);
                binop.operation(lhs_value, rhs_value)
            }
        }
    }
}

/// [`ConstraintCircuit`] with extra context pertaining to the whole
/// multicircuit.
///
/// This context is needed to ensure that two equal nodes (meaning: same
/// expression) are not added to the multicircuit. It also enables a rudimentary
/// check for node equivalence (commutation + constant folding), in which case
/// the existing expression is used instead.
///
// ConstraintCircuitMonad has been removed in favor of the Vec-based approach
// where node IDs (usize) are used instead of monads

// Old binop functions removed - functionality moved to ConstraintCircuitBuilder methods

// Arithmetic operations removed - use ConstraintCircuitBuilder methods instead

struct EvolvingMainConstraintsNumber(usize);
impl From<EvolvingMainConstraintsNumber> for usize {
    fn from(value: EvolvingMainConstraintsNumber) -> Self {
        value.0
    }
}

struct EvolvingAuxConstraintsNumber(usize);
impl From<EvolvingAuxConstraintsNumber> for usize {
    fn from(value: EvolvingAuxConstraintsNumber) -> Self {
        value.0
    }
}

// TODO: Update or remove ConstraintCircuitMonad implementation
/*
impl<II: InputIndicator> ConstraintCircuitMonad<II> {
    /// Unwrap a ConstraintCircuitMonad to reveal its inner ConstraintCircuit
    pub fn consume(&self) -> ConstraintCircuit<II> {
        self.circuit.borrow().to_owned()
    }

    /// Lower the degree of a given multicircuit to the target degree.
    /// This is achieved by introducing additional variables and constraints.
    /// The appropriate substitutions are applied to the given multicircuit.
    /// The target degree must be greater than 1.
    ///
    /// The new constraints are returned as two vector of
    /// ConstraintCircuitMonads: the first corresponds to main columns and
    /// constraints, the second to auxiliary columns and constraints. The
    /// modifications are applied to the function argument in-place.
    ///
    /// Each returned constraint is guaranteed to correspond to some
    /// `CircuitExpression::BinaryOperation(BinOp::Sub, lhs, rhs)` where
    /// - `lhs` is the new variable, and
    /// - `rhs` is the (sub)circuit replaced by `lhs`. These can then be used to
    ///   construct new columns, as well as derivation rules for filling those
    ///   new columns.
    ///
    /// For example, starting with the constraint set {x^4}, we insert
    /// {y - x^2} and modify in-place (x^4) --> (y^2).
    ///
    /// The highest index of main and auxiliary columns used by the multicircuit
    /// have to be provided. The uniqueness of the new columns' indices
    /// depends on these provided values. Note that these indices are
    /// generally not equal to the number of used columns, especially when a
    /// tables' constraints are built using the master table's column indices.
    pub fn lower_to_degree(
        multicircuit: &mut [Self],
        info: DegreeLoweringInfo,
    ) -> (Vec<Self>, Vec<Self>) {
        let target_degree = info.target_degree;
        assert!(
            target_degree > 1,
            "Target degree must be greater than 1. Got {target_degree}."
        );

        let mut main_constraints = vec![];
        let mut aux_constraints = vec![];

        if multicircuit.is_empty() {
            return (main_constraints, aux_constraints);
        }

        while Self::multicircuit_degree(multicircuit) > target_degree {
            let chosen_node_id = Self::pick_node_to_substitute(multicircuit, target_degree);

            let new_constraint = Self::apply_substitution(
                multicircuit,
                info,
                chosen_node_id,
                EvolvingMainConstraintsNumber(main_constraints.len()),
                EvolvingAuxConstraintsNumber(aux_constraints.len()),
            );

            if new_constraint.circuit.borrow().evaluates_to_base_element() {
                main_constraints.push(new_constraint)
            } else {
                aux_constraints.push(new_constraint)
            }
        }

        (main_constraints, aux_constraints)
    }

    /// Apply a substitution:
    ///  - create a new variable to replaces the chosen node;
    ///  - make all nodes that point to the chosen node point to the new
    ///    variable instead;
    ///  - return the new constraint that makes it sound: new variable minus
    ///    chosen node's expression.
    fn apply_substitution(
        multicircuit: &mut [Self],
        info: DegreeLoweringInfo,
        chosen_node_id: usize,
        new_main_constraints_count: EvolvingMainConstraintsNumber,
        new_aux_constraints_count: EvolvingAuxConstraintsNumber,
    ) -> ConstraintCircuitMonad<II> {
        let builder = multicircuit[0].builder.clone();

        // Create a new variable.
        let chosen_node = builder.all_nodes.borrow()[&chosen_node_id].clone();
        let chosen_node_is_main_col = chosen_node.circuit.borrow().evaluates_to_base_element();
        let new_input_indicator = if chosen_node_is_main_col {
            let new_main_col_idx = info.num_main_cols + usize::from(new_main_constraints_count);
            II::main_table_input(new_main_col_idx)
        } else {
            let new_aux_col_idx = info.num_aux_cols + usize::from(new_aux_constraints_count);
            II::aux_table_input(new_aux_col_idx)
        };
        let new_variable = builder.input(new_input_indicator);

        // Point all descendants of the chosen node to the new variable instead
        builder.redirect_all_references_to_node(chosen_node_id, new_variable.clone());

        // Treat roots of the multicircuit explicitly.
        for circuit in multicircuit.iter_mut() {
            if circuit.circuit.borrow().id == chosen_node_id {
                circuit.circuit = new_variable.circuit.clone();
            }
        }

        // return substitution equation
        new_variable - chosen_node
    }

    /// Heuristically pick a node from the given multicircuit that is to be
    /// substituted with a new variable. The ID of the chosen node is
    /// returned.
    fn pick_node_to_substitute(
        multicircuit: &[ConstraintCircuitMonad<II>],
        target_degree: isize,
    ) -> usize {
        assert!(!multicircuit.is_empty());
        let multicircuit = multicircuit
            .iter()
            .map(|c| c.clone().consume())
            .collect_vec();

        // Computing all node degree is slow; this cache de-duplicates work.
        let node_degrees = Self::all_nodes_in_multicircuit(&multicircuit)
            .into_iter()
            .map(|node| (node.id, node.degree()))
            .collect::<HashMap<_, _>>();

        // Only nodes with degree > target_degree need changing.
        let high_degree_nodes = Self::all_nodes_in_multicircuit(&multicircuit)
            .into_iter()
            .filter(|node| node_degrees[&node.id] > target_degree)
            .unique()
            .collect_vec();

        // Collect all candidates for substitution, i.e., descendents of
        // high_degree_nodes with degree <= target_degree. Substituting a node
        // of degree 1 is both pointless and can lead to infinite iteration.
        let low_degree_nodes = Self::all_nodes_in_multicircuit(&high_degree_nodes)
            .into_iter()
            .filter(|node| 1 < node_degrees[&node.id] && node_degrees[&node.id] <= target_degree)
            .map(|node| node.id)
            .collect_vec();

        // If the resulting list is empty, there is no way forward.
        assert!(!low_degree_nodes.is_empty(), "Cannot lower degree.");

        // Of the remaining nodes, keep the ones occurring the most often.
        let mut nodes_and_occurrences = HashMap::new();
        for node in low_degree_nodes {
            *nodes_and_occurrences.entry(node).or_insert(0) += 1;
        }
        let max_occurrences = nodes_and_occurrences.iter().map(|(_, &c)| c).max().unwrap();
        nodes_and_occurrences.retain(|_, &mut count| count == max_occurrences);
        let mut candidate_node_ids = nodes_and_occurrences.keys().copied().collect_vec();

        // If there are still multiple nodes, pick the one with the highest
        // degree.
        let max_degree = candidate_node_ids
            .iter()
            .map(|node_id| node_degrees[node_id])
            .max()
            .unwrap();
        candidate_node_ids.retain(|node_id| node_degrees[node_id] == max_degree);

        candidate_node_ids.sort_unstable();

        // If there are still multiple nodes, pick any one – but
        // deterministically so.
        candidate_node_ids.into_iter().min().unwrap()
    }

    /// Returns all nodes used in the multicircuit.
    /// This is distinct from `ConstraintCircuitBuilder::all_nodes` because it
    /// 1. only considers nodes used in the given multicircuit, not all nodes in
    ///    the builder,
    /// 2. returns the nodes as [`ConstraintCircuit`]s, not as
    ///    [`ConstraintCircuitMonad`]s, and
    /// 3. keeps duplicates, allowing to count how often a node occurs.
    pub fn all_nodes_in_multicircuit(
        multicircuit: &[ConstraintCircuit<II>],
    ) -> Vec<ConstraintCircuit<II>> {
        multicircuit
            .iter()
            .flat_map(Self::all_nodes_in_circuit)
            .collect()
    }

    /// Internal helper function to recursively find all nodes in a circuit.
    fn all_nodes_in_circuit(circuit: &ConstraintCircuit<II>) -> Vec<ConstraintCircuit<II>> {
        let mut all_nodes = vec![];
        if let CircuitExpression::BinOp(_, lhs, rhs) = circuit.expression.clone() {
            let lhs_nodes = Self::all_nodes_in_circuit(&lhs.borrow());
            let rhs_nodes = Self::all_nodes_in_circuit(&rhs.borrow());
            all_nodes.extend(lhs_nodes);
            all_nodes.extend(rhs_nodes);
        };
        all_nodes.push(circuit.to_owned());
        all_nodes
    }

    /// Counts the number of nodes in this multicircuit. Only counts nodes that
    /// are used; not nodes that have been forgotten.
    pub fn num_visible_nodes(constraints: &[Self]) -> usize {
        constraints
            .iter()
            .flat_map(|ccm| Self::all_nodes_in_circuit(&ccm.circuit.borrow()))
            .unique()
            .count()
    }

    /// Returns the maximum degree of all circuits in the multicircuit.
    pub fn multicircuit_degree(multicircuit: &[ConstraintCircuitMonad<II>]) -> isize {
        multicircuit
            .iter()
            .map(|circuit| circuit.circuit.borrow().degree())
            .max()
            .unwrap_or(-1)
    }
}
*/

/// Helper struct to construct new leaf nodes (*i.e.*, input or challenge or
/// constant) in the circuit multitree. Ensures that newly created nodes, even
/// non-leaf nodes created through joining two other nodes using an arithmetic
/// operation, get a unique ID.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ConstraintCircuitBuilder<II: InputIndicator> {
    nodes: Vec<ConstraintCircuit<II>>,
}

impl<II: InputIndicator> Default for ConstraintCircuitBuilder<II> {
    fn default() -> Self {
        Self::new()
    }
}

impl<II: InputIndicator> ConstraintCircuitBuilder<II> {
    pub fn new() -> Self {
        Self {
            nodes: Vec::new(),
        }
    }

    /// Get the number of nodes in the builder
    pub fn len(&self) -> usize {
        self.nodes.len()
    }

    /// Check if the builder is empty
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Get a reference to a node by its ID (index)
    pub fn get_node(&self, id: usize) -> Option<&ConstraintCircuit<II>> {
        self.nodes.get(id)
    }

    /// Get a mutable reference to a node by its ID (index)
    pub fn get_node_mut(&mut self, id: usize) -> Option<&mut ConstraintCircuit<II>> {
        self.nodes.get_mut(id)
    }

    /// Reset reference counts for all nodes
    pub fn reset_all_ref_counts(&mut self) {
        for node in &mut self.nodes {
            node.reset_ref_count_for_tree();
        }
    }

    /// Get the degree of a node
    pub fn degree(&self, node_id: usize) -> isize {
        if let Some(node) = self.get_node(node_id) {
            node.degree(self)
        } else {
            panic!("Invalid node ID: {}", node_id);
        }
    }

    /// The node ID representing the constant value 0.
    pub fn zero(&mut self) -> usize {
        self.b_constant(0)
    }

    /// The node ID representing the constant value 1.
    pub fn one(&mut self) -> usize {
        self.b_constant(1)
    }

    /// The node ID representing the constant value -1.
    pub fn minus_one(&mut self) -> usize {
        self.b_constant(-1)
    }

    /// Create a leaf node with constant over the [base field][BFieldElement].
    pub fn b_constant<B>(&mut self, bfe: B) -> usize
    where
        B: Into<BFieldElement>,
    {
        self.make_leaf(CircuitExpression::BConst(bfe.into()))
    }

    /// Create a leaf node with constant over the [extension field][XFieldElement].
    pub fn x_constant<X>(&mut self, xfe: X) -> usize
    where
        X: Into<XFieldElement>,
    {
        self.make_leaf(CircuitExpression::XConst(xfe.into()))
    }

    /// Create deterministic input leaf node.
    pub fn input(&mut self, input: II) -> usize {
        self.make_leaf(CircuitExpression::Input(input))
    }

    /// Create challenge leaf node.
    pub fn challenge<C>(&mut self, challenge: C) -> usize
    where
        C: Into<usize>,
    {
        self.make_leaf(CircuitExpression::Challenge(challenge.into()))
    }

    fn make_leaf(&mut self, mut expression: CircuitExpression<II>) -> usize {
        assert!(
            !matches!(expression, CircuitExpression::BinOp(_, _, _)),
            "`make_leaf` is intended for anything but `BinOp`s"
        );

        // don't use X field if the B field suffices
        if let CircuitExpression::XConst(xfe) = expression {
            if let Some(bfe) = xfe.unlift() {
                expression = CircuitExpression::BConst(bfe);
            }
        }

        // Check if we already have this expression
        for (id, node) in self.nodes.iter().enumerate() {
            if node.expression == expression {
                return id;
            }
        }

        let id = self.nodes.len();
        let circuit = ConstraintCircuit::new(id, expression);
        self.nodes.push(circuit);
        id
    }

    /// Add two nodes
    pub fn add(&mut self, lhs: usize, rhs: usize) -> usize {
        self.binop(BinOp::Add, lhs, rhs)
    }

    /// Multiply two nodes
    pub fn mul(&mut self, lhs: usize, rhs: usize) -> usize {
        self.binop(BinOp::Mul, lhs, rhs)
    }

    /// Subtract two nodes (lhs - rhs)
    pub fn sub(&mut self, lhs: usize, rhs: usize) -> usize {
        let neg_rhs = self.neg(rhs);
        self.add(lhs, neg_rhs)
    }

    /// Negate a node
    pub fn neg(&mut self, node: usize) -> usize {
        let minus_one = self.minus_one();
        self.mul(minus_one, node)
    }

    fn binop(&mut self, binop: BinOp, lhs: usize, rhs: usize) -> usize {
        // Optimization: handle special cases
        if let (Some(lhs_node), Some(rhs_node)) = (self.nodes.get(lhs), self.nodes.get(rhs)) {
            match binop {
                BinOp::Add => {
                    if lhs_node.is_zero() { return rhs; }
                    if rhs_node.is_zero() { return lhs; }
                }
                BinOp::Mul => {
                    if lhs_node.is_one() { return rhs; }
                    if rhs_node.is_one() { return lhs; }
                    if lhs_node.is_zero() || rhs_node.is_zero() {
                        return self.zero();
                    }
                }
            }

            // Constant folding
            match (&lhs_node.expression, &rhs_node.expression) {
                (CircuitExpression::BConst(l), CircuitExpression::BConst(r)) => {
                    return self.b_constant(binop.operation(*l, *r));
                }
                (CircuitExpression::BConst(l), CircuitExpression::XConst(r)) => {
                    return self.x_constant(binop.operation(*l, *r));
                }
                (CircuitExpression::XConst(l), CircuitExpression::BConst(r)) => {
                    return self.x_constant(binop.operation(*l, *r));
                }
                (CircuitExpression::XConst(l), CircuitExpression::XConst(r)) => {
                    return self.x_constant(binop.operation(*l, *r));
                }
                _ => {}
            }
        }

        let expression = CircuitExpression::BinOp(binop, lhs, rhs);
        
        // Check if we already have this expression (try both orders for commutative ops)
        for (id, node) in self.nodes.iter().enumerate() {
            if node.expression == expression {
                return id;
            }
            // Try commuted version
            if let CircuitExpression::BinOp(op, l, r) = &expression {
                let commuted = CircuitExpression::BinOp(*op, *r, *l);
                if node.expression == commuted {
                    return id;
                }
            }
        }

        let id = self.nodes.len();
        let circuit = ConstraintCircuit::new(id, expression);
        self.nodes.push(circuit);
        id
    }

    /// Replace all references to a given node with a new node ID
    pub fn redirect_all_references_to_node(&mut self, old_id: usize, new_id: usize) {
        for node in &mut self.nodes {
            if let CircuitExpression::BinOp(_op, left, right) = &mut node.expression {
                if *left == old_id {
                    *left = new_id;
                }
                if *right == old_id {
                    *right = new_id;
                }
            }
        }
    }
}

// TODO: Update Arbitrary implementation for new architecture
/*
impl<'a, II: InputIndicator + Arbitrary<'a>> Arbitrary<'a> for ConstraintCircuitMonad<II> {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let mut builder = ConstraintCircuitBuilder::new();
        let mut random_circuit = random_circuit_leaf(&mut builder, u)?;

        let num_nodes_in_circuit = u.arbitrary_len::<Self>()?;
        for _ in 0..num_nodes_in_circuit {
            let leaf = random_circuit_leaf(&mut builder, u)?;
            match u.int_in_range(0..=5)? {
                0 => random_circuit = builder.mul(random_circuit, leaf),
                1 => random_circuit = builder.add(random_circuit, leaf),
                2 => random_circuit = builder.sub(random_circuit, leaf),
                3 => random_circuit = builder.mul(leaf, random_circuit),
                4 => random_circuit = builder.add(leaf, random_circuit),
                5 => random_circuit = builder.sub(leaf, random_circuit),
                _ => unreachable!(),
            }
        }

        Ok(random_circuit)
    }
}
*/

/*
fn random_circuit_leaf<'a, II: InputIndicator + Arbitrary<'a>>(
    builder: &mut ConstraintCircuitBuilder<II>,
    u: &mut Unstructured<'a>,
) -> arbitrary::Result<usize> {
    let leaf = match u.int_in_range(0..=5)? {
        0 => builder.input(u.arbitrary()?),
        1 => builder.challenge(u.arbitrary::<usize>()?),
        2 => builder.b_constant(u.arbitrary::<BFieldElement>()?),
        3 => builder.x_constant(u.arbitrary::<XFieldElement>()?),
        4 => builder.one(),
        5 => builder.zero(),
        _ => unreachable!(),
    };
    Ok(leaf)
}
*/

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests;