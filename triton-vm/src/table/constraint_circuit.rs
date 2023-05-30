//! Constraint circuits are a way to represent constraint polynomials in a way that is amenable
//! to optimizations. The constraint circuit is a directed acyclic graph (DAG) of
//! [`CircuitExpression`]s, where each `CircuitExpression` is a node in the graph. The edges of the
//! graph are labeled with [`BinOp`]s. The leafs of the graph are the inputs to the constraint
//! polynomial, and the (multiple) roots of the graph are the outputs of all the
//! constraint polynomials, with each root corresponding to a different constraint polynomial.
//! Because the graph has multiple roots, it is called a “multitree.”

use std::borrow::BorrowMut;
use std::cell::RefCell;
use std::cmp;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::Debug;
use std::fmt::Display;
use std::hash::Hash;
use std::iter::Sum;
use std::ops::Add;
use std::ops::Mul;
use std::ops::Sub;
use std::rc::Rc;

use ndarray::ArrayView2;
use num_traits::One;
use num_traits::Zero;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::mpolynomial::Degree;
use twenty_first::shared_math::x_field_element::XFieldElement;

use quote::quote;
use quote::ToTokens;
use CircuitExpression::*;

use crate::table::challenges::ChallengeId;
use crate::table::challenges::Challenges;

#[derive(Debug, Clone, Copy, PartialEq, Hash, Eq)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
}

impl Display for BinOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinOp::Add => write!(f, "+"),
            BinOp::Sub => write!(f, "-"),
            BinOp::Mul => write!(f, "*"),
        }
    }
}

impl ToTokens for BinOp {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        match self {
            BinOp::Add => tokens.extend(quote!(+)),
            BinOp::Sub => tokens.extend(quote!(-)),
            BinOp::Mul => tokens.extend(quote!(*)),
        }
    }
}

impl BinOp {
    pub fn operation<L, R, O>(&self, lhs: L, rhs: R) -> O
    where
        L: Add<R, Output = O> + Sub<R, Output = O> + Mul<R, Output = O>,
    {
        match self {
            BinOp::Add => lhs + rhs,
            BinOp::Sub => lhs - rhs,
            BinOp::Mul => lhs * rhs,
        }
    }
}

/// Describes the position of a variable in a constraint polynomial in the row layout applicable
/// for a certain kind of constraint polynomial.
///
/// The position of variable in a constraint polynomial is, in principle, a `usize`. However,
/// depending on the type of the constraint polynomial, this index may be an index into a single
/// row (for initial, consistency and terminal constraints), or a pair of adjacent rows (for
/// transition constraints). Additionally, the index may refer to a column in the base table, or
/// a column in the extension table. This trait abstracts over these possibilities, and provides
/// a uniform interface for accessing the index.
///
/// Having `Clone + Copy + Hash + PartialEq + Eq` helps putting `InputIndicator`s into containers.
pub trait InputIndicator:
    Debug + Clone + Copy + Hash + PartialEq + Eq + Display + ToTokens
{
    /// `true` iff `self` refers to a column in the base table.
    fn is_base_table_column(&self) -> bool;

    fn base_col_index(&self) -> usize;
    fn ext_col_index(&self) -> usize;

    fn base_table_input(index: usize) -> Self;
    fn ext_table_input(index: usize) -> Self;

    fn evaluate(
        &self,
        base_table: ArrayView2<BFieldElement>,
        ext_table: ArrayView2<XFieldElement>,
    ) -> XFieldElement;
}

/// The position of a variable in a constraint polynomial that operates on a single row of the
/// execution trace.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum SingleRowIndicator {
    BaseRow(usize),
    ExtRow(usize),
}

impl Display for SingleRowIndicator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use SingleRowIndicator::*;
        let input_indicator: String = match self {
            BaseRow(i) => format!("base_row[{i}]"),
            ExtRow(i) => format!("ext_row[{i}]"),
        };

        write!(f, "{input_indicator}")
    }
}

impl ToTokens for SingleRowIndicator {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        use SingleRowIndicator::*;
        match self {
            BaseRow(i) => tokens.extend(quote!(base_row[#i])),
            ExtRow(i) => tokens.extend(quote!(ext_row[#i])),
        }
    }
}

impl InputIndicator for SingleRowIndicator {
    fn is_base_table_column(&self) -> bool {
        use SingleRowIndicator::*;
        matches!(self, BaseRow(_))
    }

    fn base_col_index(&self) -> usize {
        use SingleRowIndicator::*;
        match self {
            BaseRow(i) => *i,
            ExtRow(_) => panic!("not a base row"),
        }
    }

    fn ext_col_index(&self) -> usize {
        use SingleRowIndicator::*;
        match self {
            BaseRow(_) => panic!("not an ext row"),
            ExtRow(i) => *i,
        }
    }

    fn base_table_input(index: usize) -> Self {
        Self::BaseRow(index)
    }

    fn ext_table_input(index: usize) -> Self {
        Self::ExtRow(index)
    }

    fn evaluate(
        &self,
        base_table: ArrayView2<BFieldElement>,
        ext_table: ArrayView2<XFieldElement>,
    ) -> XFieldElement {
        use SingleRowIndicator::*;
        match self {
            BaseRow(i) => base_table[[0, *i]].lift(),
            ExtRow(i) => ext_table[[0, *i]],
        }
    }
}

/// The position of a variable in a constraint polynomial that operates on two rows (current and
/// next) of the execution trace.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum DualRowIndicator {
    CurrentBaseRow(usize),
    CurrentExtRow(usize),
    NextBaseRow(usize),
    NextExtRow(usize),
}

impl Display for DualRowIndicator {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        use DualRowIndicator::*;
        let input_indicator: String = match self {
            CurrentBaseRow(i) => format!("current_base_row[{i}]"),
            CurrentExtRow(i) => format!("current_ext_row[{i}]"),
            NextBaseRow(i) => format!("next_base_row[{i}]"),
            NextExtRow(i) => format!("next_ext_row[{i}]"),
        };

        write!(f, "{input_indicator}")
    }
}

impl ToTokens for DualRowIndicator {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        use DualRowIndicator::*;
        match self {
            CurrentBaseRow(i) => tokens.extend(quote!(current_base_row[#i])),
            CurrentExtRow(i) => tokens.extend(quote!(current_ext_row[#i])),
            NextBaseRow(i) => tokens.extend(quote!(next_base_row[#i])),
            NextExtRow(i) => tokens.extend(quote!(next_ext_row[#i])),
        }
    }
}

impl InputIndicator for DualRowIndicator {
    fn is_base_table_column(&self) -> bool {
        use DualRowIndicator::*;
        matches!(self, CurrentBaseRow(_) | NextBaseRow(_))
    }

    fn base_col_index(&self) -> usize {
        use DualRowIndicator::*;
        match self {
            CurrentBaseRow(i) | NextBaseRow(i) => *i,
            CurrentExtRow(_) | NextExtRow(_) => panic!("not a base row"),
        }
    }

    fn ext_col_index(&self) -> usize {
        use DualRowIndicator::*;
        match self {
            CurrentBaseRow(_) | NextBaseRow(_) => panic!("not an ext row"),
            CurrentExtRow(i) | NextExtRow(i) => *i,
        }
    }

    fn base_table_input(index: usize) -> Self {
        // It seems that the choice between `CurrentBaseRow` and `NextBaseRow` is arbitrary:
        // any transition constraint polynomial is evaluated on both the current and the next row.
        // Hence, both rows are in scope.
        Self::CurrentBaseRow(index)
    }

    fn ext_table_input(index: usize) -> Self {
        Self::CurrentExtRow(index)
    }

    fn evaluate(
        &self,
        base_table: ArrayView2<BFieldElement>,
        ext_table: ArrayView2<XFieldElement>,
    ) -> XFieldElement {
        use DualRowIndicator::*;
        match self {
            CurrentBaseRow(i) => base_table[[0, *i]].lift(),
            CurrentExtRow(i) => ext_table[[0, *i]],
            NextBaseRow(i) => base_table[[1, *i]].lift(),
            NextExtRow(i) => ext_table[[1, *i]],
        }
    }
}

/// A circuit expression is the recursive data structure that represents the constraint polynomials.
/// It is a directed, acyclic graph of binary operations on the variables of the constraint
/// polynomials, constants, and challenges. It has multiple roots, making it a “multitree.” Each
/// root corresponds to one constraint polynomial.
///
/// The leafs of the tree are
/// - constants in the base field, _i.e._, [`BFieldElement`]s,
/// - constants in the extension field, _i.e._, [`XFieldElement`]s,
/// - input variables, _i.e._, entries from the Algebraic Execution Trace, and
/// - challenges, _i.e._, (pseudo-)random values sampled through the Fiat-Shamir heuristic.
///
/// An inner node, representing some binary operation, is either addition, multiplication, or
/// subtraction. The left and right children of the node are the operands of the binary operation.
/// The left and right children are not themselves `CircuitExpression`s, but rather
/// [`ConstraintCircuit`]s, which is a wrapper around `CircuitExpression` that manages additional
/// bookkeeping information.
#[derive(Debug, Clone)]
pub enum CircuitExpression<II: InputIndicator> {
    XConstant(XFieldElement),
    BConstant(BFieldElement),
    Input(II),
    Challenge(ChallengeId),
    BinaryOperation(
        BinOp,
        Rc<RefCell<ConstraintCircuit<II>>>,
        Rc<RefCell<ConstraintCircuit<II>>>,
    ),
}

impl<II: InputIndicator> Hash for CircuitExpression<II> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            BConstant(bfe) => {
                "bfe".hash(state);
                bfe.hash(state);
            }
            XConstant(xfe) => {
                "xfe".hash(state);
                xfe.hash(state);
            }
            Input(index) => {
                "input".hash(state);
                index.hash(state);
            }
            Challenge(table_challenge_id) => {
                "challenge".hash(state);
                table_challenge_id.hash(state);
            }
            BinaryOperation(binop, lhs, rhs) => {
                "binop".hash(state);
                binop.hash(state);
                lhs.as_ref().borrow().hash(state);
                rhs.as_ref().borrow().hash(state);
            }
        }
    }
}

impl<II: InputIndicator> PartialEq for CircuitExpression<II> {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            (BConstant(bfe_self), BConstant(bfe_other)) => bfe_self == bfe_other,
            (XConstant(xfe_self), XConstant(xfe_other)) => xfe_self == xfe_other,
            (Input(input_self), Input(input_other)) => input_self == input_other,
            (Challenge(id_self), Challenge(id_other)) => id_self == id_other,
            (BinaryOperation(op_s, lhs_s, rhs_s), BinaryOperation(op_o, lhs_o, rhs_o)) => {
                op_s == op_o && lhs_s == lhs_o && rhs_s == rhs_o
            }
            _ => false,
        }
    }
}

impl<II: InputIndicator> Hash for ConstraintCircuit<II> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.expression.hash(state)
    }
}

impl<II: InputIndicator> Hash for ConstraintCircuitMonad<II> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.circuit.as_ref().borrow().hash(state)
    }
}

/// A wrapper around a [`CircuitExpression`] that manages additional bookkeeping information.
#[derive(Clone, Debug)]
pub struct ConstraintCircuit<II: InputIndicator> {
    pub id: usize,
    pub visited_counter: usize,
    pub expression: CircuitExpression<II>,
}

impl<II: InputIndicator> Eq for ConstraintCircuit<II> {}

impl<II: InputIndicator> PartialEq for ConstraintCircuit<II> {
    /// Calculate equality of circuits. In particular, this function does *not* attempt to
    /// simplify or reduce neutral terms or products. So this comparison will return false for
    /// `a == a + 0`. It will also return false for `XFieldElement(7) == BFieldElement(7)`
    fn eq(&self, other: &Self) -> bool {
        self.expression == other.expression
    }
}

impl<II: InputIndicator> Display for ConstraintCircuit<II> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.expression {
            XConstant(xfe) => {
                write!(f, "{xfe}")
            }
            BConstant(bfe) => {
                write!(f, "{bfe}")
            }
            Input(input) => write!(f, "{input} "),
            Challenge(self_challenge_id) => {
                write!(f, "#{self_challenge_id}")
            }
            BinaryOperation(operation, lhs, rhs) => {
                write!(
                    f,
                    "({}) {} ({})",
                    lhs.as_ref().borrow(),
                    operation,
                    rhs.as_ref().borrow()
                )
            }
        }
    }
}

impl<II: InputIndicator> ConstraintCircuit<II> {
    /// Increment `visited_counter` by one for each reachable node
    fn increment_visit_count_for_tree(&mut self) {
        self.visited_counter += 1;
        if let BinaryOperation(_, lhs, rhs) = self.expression.borrow_mut() {
            lhs.as_ref().borrow_mut().increment_visit_count_for_tree();
            rhs.as_ref().borrow_mut().increment_visit_count_for_tree();
        }
    }

    /// Count how often each node is referenced when traversing from the given starting points.
    /// The result is stored in the `visited_counter` field of each node.
    pub fn refresh_visit_counters(ccs: &mut [ConstraintCircuit<II>]) {
        for cc in ccs.iter_mut() {
            cc.reset_visit_count_for_tree();
        }
        for cc in ccs.iter_mut() {
            cc.increment_visit_count_for_tree();
        }
    }

    /// Reset the visited counters for the entire subtree
    fn reset_visit_count_for_tree(&mut self) {
        self.visited_counter = 0;

        if let BinaryOperation(_, lhs, rhs) = &self.expression {
            lhs.as_ref().borrow_mut().reset_visit_count_for_tree();
            rhs.as_ref().borrow_mut().reset_visit_count_for_tree();
        }
    }

    /// Verify that all IDs in the subtree are unique. Panics otherwise.
    fn inner_has_unique_ids(&mut self, ids: &mut HashMap<usize, ConstraintCircuit<II>>) {
        let self_id = self.id;

        // Try to detect duplicate IDs only once for this node.
        let maybe_other_node = if self.visited_counter == 0 {
            ids.insert(self_id, self.clone())
        } else {
            None
        };
        if let Some(other) = maybe_other_node {
            panic!("ID {self_id} was repeated. Self: {self:?}. Other: {other:?}.");
        }

        self.visited_counter += 1;
        if let BinaryOperation(_, lhs, rhs) = &self.expression {
            lhs.as_ref().borrow_mut().inner_has_unique_ids(ids);
            rhs.as_ref().borrow_mut().inner_has_unique_ids(ids);
        }
    }

    /// Verify that a multicircuit has unique IDs. Panics otherwise.
    /// Also determines how often each node is referenced and stores the result in the
    /// `visited_counter` field of each node.
    /// This makes this function very similar to
    /// [`refresh_visit_counters`](ConstraintCircuit::refresh_visit_counters),
    /// but it also checks for duplicate IDs.
    pub fn assert_has_unique_ids(constraints: &mut [ConstraintCircuit<II>]) {
        let mut ids: HashMap<usize, ConstraintCircuit<II>> = HashMap::new();
        // The inner uniqueness checks relies on visit counters being 0 for unseen nodes.
        // Hence, they are reset here.
        for circuit in constraints.iter_mut() {
            circuit.reset_visit_count_for_tree();
        }
        for circuit in constraints.iter_mut() {
            circuit.inner_has_unique_ids(&mut ids);
        }
    }

    /// Return degree of the multivariate polynomial represented by this circuit
    pub fn degree(&self) -> Degree {
        match &self.expression {
            BinaryOperation(binop, lhs, rhs) => {
                let lhs_degree = lhs.borrow().degree();
                let rhs_degree = rhs.borrow().degree();
                match binop {
                    BinOp::Add | BinOp::Sub => cmp::max(lhs_degree, rhs_degree),
                    BinOp::Mul => match lhs_degree == -1 || rhs_degree == -1 {
                        true => -1,
                        false => lhs_degree + rhs_degree,
                    },
                }
            }
            Input(_) => 1,
            XConstant(xfe) => match xfe.is_zero() {
                true => -1,
                false => 0,
            },
            BConstant(bfe) => match bfe.is_zero() {
                true => -1,
                false => 0,
            },
            Challenge(_) => 0,
        }
    }

    /// Return all visited counters in the subtree
    pub fn get_all_visited_counters(&self) -> Vec<usize> {
        // Maybe this could be solved smarter with dynamic programming
        // but we probably don't need that as our circuits aren't too big.
        match &self.expression {
            BinaryOperation(_, lhs, rhs) => {
                let lhs_counters = lhs.as_ref().borrow().get_all_visited_counters();
                let rhs_counters = rhs.as_ref().borrow().get_all_visited_counters();
                let own_counter = vec![self.visited_counter];
                let mut all_counters = vec![own_counter, lhs_counters, rhs_counters].concat();
                all_counters.sort_unstable();
                all_counters.dedup();
                all_counters
            }
            _ => vec![self.visited_counter],
        }
    }

    /// Return true if the contained multivariate polynomial consists of only a single term. This
    /// means that it can be pretty-printed without parentheses.
    pub fn print_without_parentheses(&self) -> bool {
        !matches!(&self.expression, BinaryOperation(_, _, _))
    }

    /// Return true if this node represents a constant value of zero, does not catch composite
    /// expressions that will always evaluate to zero.
    pub fn is_zero(&self) -> bool {
        match self.expression {
            BConstant(bfe) => bfe.is_zero(),
            XConstant(xfe) => xfe.is_zero(),
            _ => false,
        }
    }

    /// Return true if this node represents a constant value of one, does not catch composite
    /// expressions that will always evaluate to one.
    pub fn is_one(&self) -> bool {
        match self.expression {
            XConstant(xfe) => xfe.is_one(),
            BConstant(bfe) => bfe.is_one(),
            _ => false,
        }
    }

    /// Return true iff the evaluation value of this node depends on a challenge.
    pub fn is_randomized(&self) -> bool {
        match &self.expression {
            Challenge(_) => true,
            BinaryOperation(_, lhs, rhs) => {
                lhs.as_ref().borrow().is_randomized() || rhs.as_ref().borrow().is_randomized()
            }
            _ => false,
        }
    }

    /// Recursively check whether this node is composed of only BFieldElements, i.e., only uses
    /// 1. inputs from base rows,
    /// 2. constants from the B-field, and
    /// 3. binary operations on BFieldElements.
    pub fn evaluates_to_base_element(&self) -> bool {
        match &self.expression {
            BConstant(_) => true,
            XConstant(_) => false,
            Input(indicator) => indicator.is_base_table_column(),
            Challenge(_) => false,
            BinaryOperation(_, lhs, rhs) => {
                let lhs = lhs.as_ref().borrow();
                let rhs = rhs.as_ref().borrow();
                lhs.evaluates_to_base_element() && rhs.evaluates_to_base_element()
            }
        }
    }

    pub fn evaluate(
        &self,
        base_table: ArrayView2<BFieldElement>,
        ext_table: ArrayView2<XFieldElement>,
        challenges: &Challenges,
    ) -> XFieldElement {
        match self.clone().expression {
            XConstant(xfe) => xfe,
            BConstant(bfe) => bfe.lift(),
            Input(input) => input.evaluate(base_table, ext_table),
            Challenge(challenge_id) => challenges.get_challenge(challenge_id),
            BinaryOperation(binop, lhs, rhs) => {
                let lhs_value = lhs
                    .as_ref()
                    .borrow()
                    .evaluate(base_table, ext_table, challenges);
                let rhs_value = rhs
                    .as_ref()
                    .borrow()
                    .evaluate(base_table, ext_table, challenges);
                binop.operation(lhs_value, rhs_value)
            }
        }
    }

    /// Returns the number of unvisited nodes in the subtree of the given node, which includes
    /// the node itself. Increments the visit counter of each visited node.
    fn count_nodes_inner(constraint: &mut ConstraintCircuit<II>) -> usize {
        let num_unvisited_self = match constraint.visited_counter {
            0 => 1,
            _ => 0,
        };
        constraint.visited_counter += 1;
        let num_unvisited_children = match &constraint.expression {
            BinaryOperation(_, lhs, rhs) => {
                let num_left = Self::count_nodes_inner(&mut lhs.as_ref().borrow_mut());
                let num_right = Self::count_nodes_inner(&mut rhs.as_ref().borrow_mut());
                num_left + num_right
            }
            _ => 0,
        };

        num_unvisited_self + num_unvisited_children
    }

    /// Count the total number of unique nodes in the given multicircuit.
    /// Also refreshes the visit counter for each node, similar to
    /// [`refresh_visit_counters`](ConstraintCircuit::refresh_visit_counters).
    pub fn count_nodes(constraints: &mut [ConstraintCircuit<II>]) -> usize {
        // The uniqueness of nodes is determined by their visit count.
        // To ensure a correct node count, the visit count must be reset before counting nodes.
        for constraint in constraints.iter_mut() {
            ConstraintCircuit::reset_visit_count_for_tree(constraint);
        }
        constraints
            .iter_mut()
            .map(|c| Self::count_nodes_inner(c))
            .sum()
    }
}

/// Constraint expressions, with context needed to ensure that two equal nodes are not added to
/// the multicircuit.
#[derive(Clone)]
pub struct ConstraintCircuitMonad<II: InputIndicator> {
    pub circuit: Rc<RefCell<ConstraintCircuit<II>>>,
    pub builder: ConstraintCircuitBuilder<II>,
}

impl<II: InputIndicator> Debug for ConstraintCircuitMonad<II> {
    // We cannot derive `Debug` as `all_nodes` contains itself which a derived `Debug` will
    // attempt to print as well, thus leading to infinite recursion.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConstraintCircuitMonad")
            .field("id", &self.circuit)
            .field(
                "all_nodes length: ",
                &self.builder.all_nodes.as_ref().borrow().len(),
            )
            .field(
                "id_counter_ref value: ",
                &self.builder.id_counter.as_ref().borrow(),
            )
            .finish()
    }
}

impl<II: InputIndicator> Display for ConstraintCircuitMonad<II> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.circuit.as_ref().borrow())
    }
}

impl<II: InputIndicator> PartialEq for ConstraintCircuitMonad<II> {
    // Equality for the ConstraintCircuitMonad is defined by the circuit, not the
    // other metadata (e.g. ID) that it carries around.
    fn eq(&self, other: &Self) -> bool {
        self.circuit == other.circuit
    }
}

impl<II: InputIndicator> Eq for ConstraintCircuitMonad<II> {}

/// Helper function for binary operations that are used to generate new parent nodes in the
/// multitree that represents the algebraic circuit. Ensures that each newly created node has a
/// unique ID.
fn binop<II: InputIndicator>(
    binop: BinOp,
    lhs: ConstraintCircuitMonad<II>,
    rhs: ConstraintCircuitMonad<II>,
) -> ConstraintCircuitMonad<II> {
    let id = lhs.builder.id_counter.as_ref().borrow().to_owned();
    let expression = BinaryOperation(binop, lhs.circuit.clone(), rhs.circuit.clone());
    let circuit = ConstraintCircuit {
        id,
        visited_counter: 0,
        expression,
    };
    let circuit = Rc::new(RefCell::new(circuit));
    let new_node = ConstraintCircuitMonad {
        circuit,
        builder: lhs.builder.clone(),
    };

    let mut all_nodes = lhs.builder.all_nodes.as_ref().borrow_mut();
    if let Some(same_node) = all_nodes.get(&new_node) {
        return same_node.to_owned();
    }

    // If the operator commutes, check if the switched node has already been constructed.
    // If it has, return it instead. Do not allow a new one to be built.
    if matches!(binop, BinOp::Add | BinOp::Mul) {
        let expression_switched = BinaryOperation(binop, rhs.circuit, lhs.circuit);
        let circuit_switched = ConstraintCircuit {
            id,
            visited_counter: 0,
            expression: expression_switched,
        };
        let circuit_switched = Rc::new(RefCell::new(circuit_switched));
        let new_node_switched = ConstraintCircuitMonad {
            circuit: circuit_switched,
            builder: lhs.builder.clone(),
        };
        if let Some(same_node) = all_nodes.get(&new_node_switched) {
            return same_node.to_owned();
        }
    }

    *lhs.builder.id_counter.as_ref().borrow_mut() += 1;
    let was_inserted = all_nodes.insert(new_node.clone());
    assert!(was_inserted, "Binop-created value must be new");
    new_node
}

impl<II: InputIndicator> Add for ConstraintCircuitMonad<II> {
    type Output = ConstraintCircuitMonad<II>;

    fn add(self, rhs: Self) -> Self::Output {
        binop(BinOp::Add, self, rhs)
    }
}

impl<II: InputIndicator> Sub for ConstraintCircuitMonad<II> {
    type Output = ConstraintCircuitMonad<II>;

    fn sub(self, rhs: Self) -> Self::Output {
        binop(BinOp::Sub, self, rhs)
    }
}

impl<II: InputIndicator> Mul for ConstraintCircuitMonad<II> {
    type Output = ConstraintCircuitMonad<II>;

    fn mul(self, rhs: Self) -> Self::Output {
        binop(BinOp::Mul, self, rhs)
    }
}

/// This will panic if the iterator is empty because the neutral element needs a unique ID, and
/// we have no way of getting that here.
impl<II: InputIndicator> Sum for ConstraintCircuitMonad<II> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|accum, item| accum + item)
            .expect("ConstraintCircuitMonad Iterator was empty")
    }
}

impl<II: InputIndicator> ConstraintCircuitMonad<II> {
    /// Unwrap a ConstraintCircuitMonad to reveal its inner ConstraintCircuit
    pub fn consume(self) -> ConstraintCircuit<II> {
        self.circuit.try_borrow().unwrap().to_owned()
    }

    pub fn max_id(&self) -> usize {
        let max_from_hash_map = self
            .builder
            .all_nodes
            .as_ref()
            .borrow()
            .iter()
            .map(|x| x.circuit.as_ref().borrow().id)
            .max()
            .unwrap();

        let id_ref_value = *self.builder.id_counter.borrow();
        assert_eq!(id_ref_value - 1, max_from_hash_map);
        max_from_hash_map
    }

    fn find_equivalent_expression(&self) -> Option<Rc<RefCell<ConstraintCircuit<II>>>> {
        if let BinaryOperation(op, lhs, rhs) = &self.circuit.as_ref().borrow().expression {
            // a + 0 = a ∧ a - 0 = a
            if matches!(op, BinOp::Add | BinOp::Sub) && rhs.borrow().is_zero() {
                return Some(lhs.clone());
            }

            // 0 + a = a
            if op == &BinOp::Add && lhs.borrow().is_zero() {
                return Some(rhs.clone());
            }

            if op == &BinOp::Mul {
                // a * 1 = a
                if rhs.borrow().is_one() {
                    return Some(lhs.clone());
                }
                // 1 * a = a
                if lhs.borrow().is_one() {
                    return Some(rhs.clone());
                }
                // 0 * a = 0
                if lhs.borrow().is_zero() {
                    return Some(lhs.clone());
                }
                // a * 0 = 0
                if rhs.borrow().is_zero() {
                    return Some(rhs.clone());
                }
            }

            // if both left and right hand sides are constants, simplify
            let maybe_new_const = match (&lhs.borrow().expression, &rhs.borrow().expression) {
                (&BConstant(l), &BConstant(r)) => Some(BConstant(op.operation(l, r))),
                (&BConstant(l), &XConstant(r)) => Some(XConstant(op.operation(l, r))),
                (&XConstant(l), &BConstant(r)) => Some(XConstant(op.operation(l, r))),
                (&XConstant(l), &XConstant(r)) => Some(XConstant(op.operation(l, r))),
                _ => None,
            };

            if let Some(new_const) = maybe_new_const {
                let new_const = self.builder.make_leaf(new_const).consume();
                let new_const = Rc::new(RefCell::new(new_const));
                return Some(new_const);
            }
        }
        None
    }

    /// Apply constant folding to simplify the (sub)tree.
    /// If the subtree is a leaf: no change.
    /// If the subtree is a binary operation on:
    ///
    ///  - constant x constant => fold
    ///  - anything else       => can't fold
    ///
    /// This operation mutates self and returns true if a change was applied anywhere in the tree.
    fn constant_fold_inner(&mut self) -> (bool, Option<Rc<RefCell<ConstraintCircuit<II>>>>) {
        let mut change_tracker = false;
        let self_expr = self.circuit.as_ref().borrow().expression.clone();
        if let BinaryOperation(_, lhs, rhs) = &self_expr {
            let mut lhs_as_monadic_value = ConstraintCircuitMonad {
                circuit: lhs.clone(),
                builder: self.builder.clone(),
            };
            let (change_in_lhs, _) = lhs_as_monadic_value.constant_fold_inner();
            change_tracker |= change_in_lhs;
            let mut rhs_as_monadic_value = ConstraintCircuitMonad {
                circuit: rhs.clone(),
                builder: self.builder.clone(),
            };
            let (change_in_rhs, _) = rhs_as_monadic_value.constant_fold_inner();
            change_tracker |= change_in_rhs;
        }

        let equivalent_circuit = self.find_equivalent_expression();
        change_tracker |= equivalent_circuit.is_some();

        if equivalent_circuit.is_some() {
            let equivalent_circuit = equivalent_circuit.as_ref().unwrap().clone();
            let id_to_remove = self.circuit.borrow().id;
            self.builder.substitute(id_to_remove, equivalent_circuit);
            self.builder.all_nodes.as_ref().borrow_mut().remove(self);
        }

        (change_tracker, equivalent_circuit)
    }

    /// Reduce size of multitree by simplifying constant expressions such as `1 * MPol(_,_)`
    pub fn constant_folding(circuits: &mut [ConstraintCircuitMonad<II>]) {
        for circuit in circuits.iter_mut() {
            let mut mutated = true;
            while mutated {
                let (mutated_inner, maybe_new_root) = circuit.constant_fold_inner();
                mutated = mutated_inner;
                if let Some(new_root) = maybe_new_root {
                    *circuit = ConstraintCircuitMonad {
                        circuit: new_root,
                        builder: circuit.builder.clone(),
                    };
                }
            }
        }
    }

    /// Lowers the degree of a given multicircuit to the target degree.
    /// This is achieved by introducing additional variables and constraints.
    /// The appropriate substitutions are applied to the given multicircuit.
    /// The target degree must be greater than 1.
    ///
    /// The new constraints are returned as two vector of ConstraintCircuitMonads:
    /// the first corresponds to base columns and constraints,
    /// the second to extension columns and constraints.
    ///
    /// Each returned constraint is guaranteed to correspond to some
    /// `CircuitExpression::BinaryOperation(BinOp::Sub, lhs, rhs)` where
    /// - `lhs` is the new variable, and
    /// - `rhs` is the (sub)circuit replaced by `lhs`.
    /// These can then be used to construct new columns,
    /// as well as derivation rules for filling those new columns.
    ///
    /// The highest index of base and extension columns used by the multicircuit have to be
    /// provided. The uniqueness of the new columns' indices depends on these provided values.
    /// Note that these indices are generally not equal to the number of used columns, especially
    /// when a tables' constraints are built using the master table's column indices.
    pub fn lower_to_degree(
        multicircuit: &mut [Self],
        target_degree: Degree,
        num_base_cols: usize,
        num_ext_cols: usize,
    ) -> (Vec<Self>, Vec<Self>) {
        if target_degree <= 1 {
            panic!("Target degree must be greater than 1. Got {target_degree}.");
        }

        let mut base_constraints = vec![];
        let mut ext_constraints = vec![];

        if multicircuit.is_empty() {
            return (base_constraints, ext_constraints);
        }

        let builder = multicircuit[0].builder.clone();

        while Self::multicircuit_degree(multicircuit) > target_degree {
            let chosen_node = Self::pick_node_to_substitute(target_degree, &builder);

            // Create a new variable.
            let chosen_node_id = chosen_node.as_ref().borrow().id;
            let chosen_node_is_base_col = chosen_node.as_ref().borrow().evaluates_to_base_element();
            let new_col_idx = match chosen_node_is_base_col {
                true => num_base_cols + base_constraints.len(),
                false => num_ext_cols + ext_constraints.len(),
            };
            let new_input_indicator = match chosen_node_is_base_col {
                true => II::base_table_input(new_col_idx),
                false => II::ext_table_input(new_col_idx),
            };
            let new_variable = builder.input(new_input_indicator);
            let new_circuit = new_variable.circuit.clone();

            // Substitute the chosen circuit with the new variable.
            builder.substitute(chosen_node_id, new_circuit.clone());

            // Create a new constraint: new_var - chosen_node
            let chosen_node_monad = ConstraintCircuitMonad {
                circuit: chosen_node,
                builder: builder.clone(),
            };
            let new_constraint = new_variable - chosen_node_monad;

            // Put the new constraint into the appropriate return vector.
            match chosen_node_is_base_col {
                true => base_constraints.push(new_constraint),
                false => ext_constraints.push(new_constraint),
            }

            // Treat roots of the multicircuit explicitly.
            for circuit in multicircuit.iter_mut() {
                if circuit.circuit.as_ref().borrow().id == chosen_node_id {
                    circuit.circuit = new_circuit.clone();
                }
            }
        }

        (base_constraints, ext_constraints)
    }

    fn pick_node_to_substitute(
        target_degree: Degree,
        builder: &ConstraintCircuitBuilder<II>,
    ) -> Rc<RefCell<ConstraintCircuit<II>>> {
        // Only nodes with degree > target_degree need changing.
        let all_nodes = builder.all_nodes.as_ref().borrow();
        let high_degree_nodes = all_nodes
            .iter()
            .filter(|node| node.circuit.as_ref().borrow().degree() > target_degree);

        // Of those nodes, get all the children with degree <= target_degree.
        let mut barely_low_degree_nodes = vec![];
        for node in high_degree_nodes {
            // Constants, inputs, and challenges are of degree <= 1, which is not high.
            // Addition and subtraction don't affect the degree. Hence, they are uninteresting.
            if let BinaryOperation(BinOp::Mul, lhs, rhs) =
                &node.circuit.as_ref().borrow().expression
            {
                if lhs.as_ref().borrow().degree() <= target_degree {
                    barely_low_degree_nodes.push(lhs.clone());
                }
                if rhs.as_ref().borrow().degree() <= target_degree {
                    barely_low_degree_nodes.push(rhs.clone());
                }
            }
        }

        // Substituting a node of degree 1 is both pointless and can lead to infinite iteration.
        barely_low_degree_nodes.retain(|node| node.as_ref().borrow().degree() > 1);
        let max_degree = barely_low_degree_nodes
            .iter()
            .map(|node| node.as_ref().borrow().degree())
            .max()
            .unwrap_or(-1);
        barely_low_degree_nodes.retain(|node| node.as_ref().borrow().degree() == max_degree);
        let barely_low_degree_nodes = barely_low_degree_nodes;

        // If the resulting list is empty, there is no way forward. Stop – panic time!
        assert!(
            !barely_low_degree_nodes.is_empty(),
            "Could not lower degree of circuit to target degree. This is a bug."
        );

        // Of the remaining nodes, pick the one occurring the most often.
        let mut occurrences = HashMap::new();
        for node in barely_low_degree_nodes.iter() {
            let node_id = node.as_ref().borrow().id;
            *occurrences.entry(node_id).or_insert(0) += 1;
        }
        let (&chosen_node_id, _) = occurrences.iter().max_by_key(|(_, &count)| count).unwrap();
        barely_low_degree_nodes
            .into_iter()
            .find(|node| node.as_ref().borrow().id == chosen_node_id)
            .unwrap()
    }

    /// Returns the maximum degree of all circuits in the multicircuit.
    fn multicircuit_degree(multicircuit: &[ConstraintCircuitMonad<II>]) -> Degree {
        multicircuit
            .iter()
            .map(|circuit| circuit.circuit.as_ref().borrow().degree())
            .max()
            .unwrap_or(-1)
    }
}

#[derive(Debug, Clone)]
/// Helper struct to construct new leaf nodes in the circuit multitree. Ensures that each newly
/// created node gets a unique ID.
pub struct ConstraintCircuitBuilder<II: InputIndicator> {
    id_counter: Rc<RefCell<usize>>,
    all_nodes: Rc<RefCell<HashSet<ConstraintCircuitMonad<II>>>>,
}

impl<II: InputIndicator> Default for ConstraintCircuitBuilder<II> {
    fn default() -> Self {
        Self::new()
    }
}

impl<II: InputIndicator> ConstraintCircuitBuilder<II> {
    pub fn new() -> Self {
        Self {
            id_counter: Rc::new(RefCell::new(0)),
            all_nodes: Rc::new(RefCell::new(HashSet::default())),
        }
    }

    pub fn get_node_by_id(&self, id: usize) -> Option<ConstraintCircuitMonad<II>> {
        for node in self.all_nodes.as_ref().borrow().iter() {
            if node.circuit.as_ref().borrow().id == id {
                return Some(node.clone());
            }
        }
        None
    }

    /// Create constant leaf node.
    pub fn x_constant(&self, xfe: XFieldElement) -> ConstraintCircuitMonad<II> {
        let expression = XConstant(xfe);
        self.make_leaf(expression)
    }

    /// Create constant leaf node.
    pub fn b_constant(&self, bfe: BFieldElement) -> ConstraintCircuitMonad<II> {
        let expression = BConstant(bfe);
        self.make_leaf(expression)
    }

    /// Create deterministic input leaf node.
    pub fn input(&self, input: II) -> ConstraintCircuitMonad<II> {
        let expression = Input(input);
        self.make_leaf(expression)
    }

    /// Create challenge leaf node.
    pub fn challenge(&self, challenge_id: ChallengeId) -> ConstraintCircuitMonad<II> {
        let expression = Challenge(challenge_id);
        self.make_leaf(expression)
    }

    fn make_leaf(&self, mut expression: CircuitExpression<II>) -> ConstraintCircuitMonad<II> {
        // Don't generate an X field leaf if it can be expressed as a B field leaf
        if let XConstant(xfe) = expression {
            if let Some(bfe) = xfe.unlift() {
                expression = BConstant(bfe);
            }
        }

        let id = self.id_counter.as_ref().borrow().to_owned();
        let circuit = ConstraintCircuit {
            id,
            visited_counter: 0,
            expression,
        };
        let circuit = Rc::new(RefCell::new(circuit));
        let new_node = ConstraintCircuitMonad {
            circuit,
            builder: self.clone(),
        };

        let mut all_nodes = self.all_nodes.as_ref().borrow_mut();
        if let Some(same_node) = all_nodes.get(&new_node) {
            same_node.to_owned()
        } else {
            *self.id_counter.as_ref().borrow_mut() += 1;
            all_nodes.insert(new_node.clone());
            new_node
        }
    }

    /// Substitute all nodes with ID `old_id` with the given `new` node.
    pub fn substitute(&self, old_id: usize, new: Rc<RefCell<ConstraintCircuit<II>>>) {
        for node in self.all_nodes.as_ref().borrow().clone().into_iter() {
            if node.circuit.as_ref().borrow().id == old_id {
                continue;
            }

            if let BinaryOperation(_, ref mut lhs, ref mut rhs) =
                node.circuit.as_ref().borrow_mut().expression
            {
                if lhs.as_ref().borrow().id == old_id {
                    *lhs = new.clone();
                }
                if rhs.as_ref().borrow().id == old_id {
                    *rhs = new.clone();
                }
            }
        }
    }
}

#[cfg(test)]
mod constraint_circuit_tests {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;

    use itertools::Itertools;
    use ndarray::Array2;
    use rand::random;
    use rand::rngs::StdRng;
    use rand::thread_rng;
    use rand::Rng;
    use rand::RngCore;
    use rand::SeedableRng;
    use twenty_first::shared_math::other::random_elements;

    use crate::table::cascade_table::ExtCascadeTable;
    use crate::table::challenges::ChallengeId::U32Indeterminate;
    use crate::table::challenges::Challenges;
    use crate::table::constraint_circuit::SingleRowIndicator::*;
    use crate::table::hash_table::ExtHashTable;
    use crate::table::jump_stack_table::ExtJumpStackTable;
    use crate::table::lookup_table::ExtLookupTable;
    use crate::table::master_table;
    use crate::table::master_table::*;
    use crate::table::op_stack_table::ExtOpStackTable;
    use crate::table::processor_table::ExtProcessorTable;
    use crate::table::program_table::ExtProgramTable;
    use crate::table::ram_table::ExtRamTable;
    use crate::table::u32_table::ExtU32Table;

    use super::*;

    fn random_circuit_builder() -> (
        ConstraintCircuitMonad<DualRowIndicator>,
        ConstraintCircuitBuilder<DualRowIndicator>,
    ) {
        let mut rng = thread_rng();
        let num_base_columns = 50;
        let num_ext_columns = 40;
        let var_count = 2 * (num_base_columns + num_ext_columns);
        let circuit_builder = ConstraintCircuitBuilder::new();
        let b_constants: Vec<BFieldElement> = random_elements(var_count);
        let x_constants: Vec<XFieldElement> = random_elements(var_count);
        let circuit_input =
            DualRowIndicator::NextBaseRow(rng.next_u64() as usize % num_base_columns);
        let mut ret_circuit = circuit_builder.input(circuit_input);
        for _ in 0..100 {
            let circuit = match rng.next_u64() % 6 {
                0 => {
                    // p(x, y, z) = x
                    circuit_builder.input(DualRowIndicator::CurrentBaseRow(
                        rng.next_u64() as usize % num_base_columns,
                    ))
                }
                1 => {
                    // p(x, y, z) = xfe
                    circuit_builder.x_constant(x_constants[rng.next_u64() as usize % var_count])
                }
                2 => {
                    // p(x, y, z) = rand_i
                    circuit_builder.challenge(U32Indeterminate)
                }
                3 => {
                    // p(x, y, z) = 0
                    circuit_builder.x_constant(XFieldElement::zero())
                }
                4 => {
                    // p(x, y, z) = bfe
                    circuit_builder.b_constant(b_constants[rng.next_u64() as usize % var_count])
                }
                5 => {
                    // p(x, y, z) = rand_i * x
                    let input_value =
                        DualRowIndicator::CurrentExtRow(rng.next_u64() as usize % num_ext_columns);
                    let challenge_id = U32Indeterminate;
                    circuit_builder.input(input_value) * circuit_builder.challenge(challenge_id)
                }
                _ => unreachable!(),
            };
            match rng.next_u32() % 3 {
                0 => ret_circuit = ret_circuit * circuit,
                1 => ret_circuit = ret_circuit + circuit,
                2 => ret_circuit = ret_circuit - circuit,
                _ => unreachable!(),
            }
        }

        (ret_circuit, circuit_builder)
    }

    // Make a deep copy of a Multicircuit and return it as a ConstraintCircuitMonad
    fn deep_copy_inner<II: InputIndicator>(
        val: &ConstraintCircuit<II>,
        builder: &mut ConstraintCircuitBuilder<II>,
    ) -> ConstraintCircuitMonad<II> {
        match &val.expression {
            BinaryOperation(op, lhs, rhs) => {
                let lhs_ref = deep_copy_inner(&lhs.as_ref().borrow(), builder);
                let rhs_ref = deep_copy_inner(&rhs.as_ref().borrow(), builder);
                binop(*op, lhs_ref, rhs_ref)
            }
            XConstant(xfe) => builder.x_constant(*xfe),
            BConstant(bfe) => builder.b_constant(*bfe),
            Input(input_index) => builder.input(*input_index),
            Challenge(challenge_id) => builder.challenge(*challenge_id),
        }
    }

    fn deep_copy<II: InputIndicator>(val: &ConstraintCircuit<II>) -> ConstraintCircuitMonad<II> {
        let mut builder = ConstraintCircuitBuilder::new();
        deep_copy_inner(val, &mut builder)
    }

    #[test]
    fn equality_and_hash_agree_test() {
        // The Multicircuits are put into a hash set. Hence, it is important that `Eq` and `Hash`
        // agree whether two nodes are equal: k1 == k2 => h(k1) == h(k2)
        for _ in 0..100 {
            let (circuit, circuit_builder) = random_circuit_builder();
            let mut hasher0 = DefaultHasher::new();
            circuit.hash(&mut hasher0);
            let hash0 = hasher0.finish();
            assert_eq!(circuit, circuit);

            let zero = circuit_builder.x_constant(0.into());
            let same_circuit = circuit.clone() + zero;
            let mut hasher1 = DefaultHasher::new();
            same_circuit.hash(&mut hasher1);
            let hash1 = hasher1.finish();
            let eq_eq = circuit == same_circuit;
            let hash_eq = hash0 == hash1;

            assert_eq!(eq_eq, hash_eq);
        }
    }

    #[test]
    fn multi_circuit_hash_is_unchanged_by_meta_data_test() {
        // From https://doc.rust-lang.org/std/collections/struct.HashSet.html
        // "It is a logic error for a key to be modified in such a way that the key’s hash, as
        // determined by the Hash trait, or its equality, as determined by the Eq trait, changes
        // while it is in the map. This is normally only possible through Cell, RefCell, global
        // state, I/O, or unsafe code. The behavior resulting from such a logic error is not
        // specified, but will be encapsulated to the HashSet that observed the logic error and not
        // result in undefined behavior. This could include panics, incorrect results, aborts,
        // memory leaks, and non-termination."
        // This means that the hash of a node may not depend on: `visited_counter`, `counter`,
        // `id_counter_ref`, or `all_nodes`. The reason for this constraint is that `all_nodes`
        // contains the digest of all nodes in the multi tree.
        let (circuit, _circuit_builder) = random_circuit_builder();
        let mut hasher0 = DefaultHasher::new();
        circuit.hash(&mut hasher0);
        let digest_prior = hasher0.finish();

        // Increase visited counter and verify digest is unchanged
        circuit
            .circuit
            .as_ref()
            .borrow_mut()
            .increment_visit_count_for_tree();
        let mut hasher1 = DefaultHasher::new();
        circuit.hash(&mut hasher1);
        let digest_after = hasher1.finish();
        assert_eq!(
            digest_prior, digest_after,
            "Digest must be unchanged by traversal"
        );

        // id counter and verify digest is unchanged
        let _dummy = circuit.clone() + circuit.clone();
        let mut hasher2 = DefaultHasher::new();
        circuit.hash(&mut hasher2);
        let digest_after2 = hasher2.finish();
        assert_eq!(
            digest_prior, digest_after2,
            "Digest must be unchanged by Id counter increase"
        );
    }

    #[test]
    fn circuit_equality_check_and_constant_folding_test() {
        let circuit_builder: ConstraintCircuitBuilder<DualRowIndicator> =
            ConstraintCircuitBuilder::new();
        let var_0 = circuit_builder.input(DualRowIndicator::CurrentBaseRow(0));
        let var_4 = circuit_builder.input(DualRowIndicator::NextBaseRow(4));
        let four = circuit_builder.x_constant(4.into());
        let one = circuit_builder.x_constant(1.into());
        let zero = circuit_builder.x_constant(0.into());

        assert_ne!(var_0, var_4);
        assert_ne!(var_0, four);
        assert_ne!(one, four);
        assert_ne!(one, zero);
        assert_ne!(zero, one);

        // Verify that constant folding can handle a = a * 1
        let var_0_copy_0 = deep_copy(&var_0.circuit.as_ref().borrow());
        let var_0_mul_one_0 = var_0_copy_0.clone() * one.clone();
        assert_ne!(var_0_copy_0, var_0_mul_one_0);
        let mut circuits = [var_0_copy_0, var_0_mul_one_0];
        ConstraintCircuitMonad::constant_folding(&mut circuits);
        assert_eq!(
            circuits[0], circuits[1],
            "{} != {}",
            circuits[0], circuits[1]
        );
        assert_eq!(
            circuits[1], circuits[0],
            "{} != {}",
            circuits[1], circuits[0]
        );

        // Verify that constant folding can handle a = 1 * a
        let var_0_copy_1 = deep_copy(&var_0.circuit.as_ref().borrow());
        let var_0_one_mul_1 = one.clone() * var_0_copy_1.clone();
        assert_ne!(var_0_copy_1, var_0_one_mul_1);
        let mut circuits = [var_0_copy_1, var_0_one_mul_1];
        ConstraintCircuitMonad::constant_folding(&mut circuits);
        assert_eq!(
            circuits[0], circuits[1],
            "{} != {}",
            circuits[0], circuits[1]
        );
        assert_eq!(
            circuits[1], circuits[0],
            "{} != {}",
            circuits[1], circuits[0]
        );

        // Verify that constant folding can handle a = 1 * a * 1
        let var_0_copy_2 = deep_copy(&var_0.circuit.as_ref().borrow());
        let var_0_one_mul_2 = one.clone() * var_0_copy_2.clone() * one;
        assert_ne!(var_0_copy_2, var_0_one_mul_2);
        let mut circuits = [var_0_copy_2, var_0_one_mul_2];
        ConstraintCircuitMonad::constant_folding(&mut circuits);
        assert_eq!(
            circuits[0], circuits[1],
            "{} != {}",
            circuits[0], circuits[1]
        );
        assert_eq!(
            circuits[1], circuits[0],
            "{} != {}",
            circuits[1], circuits[0]
        );

        // Verify that constant folding handles a + 0 = a
        let var_0_copy_3 = deep_copy(&var_0.circuit.as_ref().borrow());
        let var_0_plus_zero_3 = var_0_copy_3.clone() + zero.clone();
        assert_ne!(var_0_copy_3, var_0_plus_zero_3);
        let mut circuits = [var_0_copy_3, var_0_plus_zero_3];
        ConstraintCircuitMonad::constant_folding(&mut circuits);
        assert_eq!(
            circuits[0], circuits[1],
            "{} != {}",
            circuits[0], circuits[1]
        );
        assert_eq!(
            circuits[1], circuits[0],
            "{} != {}",
            circuits[1], circuits[0]
        );

        // Verify that constant folding handles a + (a * 0) = a
        let var_0_copy_4 = deep_copy(&var_0.circuit.as_ref().borrow());
        let var_0_plus_zero_4 = var_0_copy_4.clone() + var_0_copy_4.clone() * zero.clone();
        assert_ne!(var_0_copy_4, var_0_plus_zero_4);
        let mut circuits = [var_0_copy_4, var_0_plus_zero_4];
        ConstraintCircuitMonad::constant_folding(&mut circuits);
        assert_eq!(
            circuits[0], circuits[1],
            "{} != {}",
            circuits[0], circuits[1]
        );
        assert_eq!(
            circuits[1], circuits[0],
            "{} != {}",
            circuits[1], circuits[0]
        );

        // Verify that constant folding does not equate `0 - a` with `a`
        let var_0_copy_5 = deep_copy(&var_0.circuit.as_ref().borrow());
        let zero_minus_var_0 = zero - var_0_copy_5.clone();
        assert_ne!(var_0_copy_5, zero_minus_var_0);
        let mut circuits = [var_0_copy_5, zero_minus_var_0];
        ConstraintCircuitMonad::constant_folding(&mut circuits);
        assert_ne!(
            circuits[0], circuits[1],
            "{} == {}",
            circuits[0], circuits[1]
        );
        assert_ne!(
            circuits[1], circuits[0],
            "{} == {}",
            circuits[1], circuits[0]
        );
    }

    #[test]
    fn constant_folding_pbt() {
        for _ in 0..200 {
            let (circuit, circuit_builder) = random_circuit_builder();
            let one = circuit_builder.x_constant(1.into());
            let zero = circuit_builder.x_constant(0.into());

            // Verify that constant folding can handle a = a * 1
            let copy_0 = deep_copy(&circuit.circuit.as_ref().borrow());
            let copy_0_alt = copy_0.clone() * one.clone();
            assert_ne!(copy_0, copy_0_alt);
            let mut circuits = [copy_0.clone(), copy_0_alt.clone()];
            ConstraintCircuitMonad::constant_folding(&mut circuits);
            assert_eq!(
                circuits[0], circuits[1],
                "{} != {}",
                circuits[0], circuits[1]
            );
            assert_eq!(
                circuits[1], circuits[0],
                "{} != {}",
                circuits[1], circuits[0]
            );

            // Verify that constant folding can handle a = 1 * a
            let copy_1 = deep_copy(&circuit.circuit.as_ref().borrow());
            let copy_1_alt = one.clone() * copy_1.clone();
            assert_ne!(copy_1, copy_1_alt);
            let mut circuits = [copy_1, copy_1_alt];
            ConstraintCircuitMonad::constant_folding(&mut circuits);
            assert_eq!(
                circuits[0], circuits[1],
                "{} != {}",
                circuits[0], circuits[1]
            );
            assert_eq!(
                circuits[1], circuits[0],
                "{} != {}",
                circuits[1], circuits[0]
            );

            // Verify that constant folding can handle a = 1 * a * 1
            let copy_2 = deep_copy(&circuit.circuit.as_ref().borrow());
            let copy_2_alt = one.clone() * copy_2.clone() * one.clone();
            assert_ne!(copy_2, copy_2_alt);
            let mut circuits = [copy_2, copy_2_alt];
            ConstraintCircuitMonad::constant_folding(&mut circuits);
            assert_eq!(
                circuits[0], circuits[1],
                "{} != {}",
                circuits[0], circuits[1]
            );
            assert_eq!(
                circuits[1], circuits[0],
                "{} != {}",
                circuits[1], circuits[0]
            );

            // Verify that constant folding handles a + 0 = a
            let copy_3 = deep_copy(&circuit.circuit.as_ref().borrow());
            let copy_3_alt = copy_3.clone() + zero.clone();
            assert_ne!(copy_3, copy_3_alt);
            let mut circuits = [copy_3, copy_3_alt];
            ConstraintCircuitMonad::constant_folding(&mut circuits);
            assert_eq!(
                circuits[0], circuits[1],
                "{} != {}",
                circuits[0], circuits[1]
            );
            assert_eq!(
                circuits[1], circuits[0],
                "{} != {}",
                circuits[1], circuits[0]
            );

            // Verify that constant folding handles a + (a * 0) = a
            let copy_4 = deep_copy(&circuit.circuit.as_ref().borrow());
            let copy_4_alt = copy_4.clone() + copy_4.clone() * zero.clone();
            assert_ne!(copy_4, copy_4_alt);
            let mut circuits = [copy_4, copy_4_alt];
            ConstraintCircuitMonad::constant_folding(&mut circuits);
            assert_eq!(
                circuits[0], circuits[1],
                "{} != {}",
                circuits[0], circuits[1]
            );
            assert_eq!(
                circuits[1], circuits[0],
                "{} != {}",
                circuits[1], circuits[0]
            );

            // Verify that constant folding handles a + (0 * a) = a
            let copy_5 = deep_copy(&circuit.circuit.as_ref().borrow());
            let copy_5_alt = copy_5.clone() + copy_5.clone() * zero.clone();
            assert_ne!(copy_5, copy_5_alt);
            let mut circuits = [copy_5, copy_5_alt];
            ConstraintCircuitMonad::constant_folding(&mut circuits);
            assert_eq!(
                circuits[0], circuits[1],
                "{} != {}",
                circuits[0], circuits[1]
            );
            assert_eq!(
                circuits[1], circuits[0],
                "{} != {}",
                circuits[1], circuits[0]
            );

            // Verify that constant folding does not equate `0 - a` with `a`
            // But only if `a != 0`
            let copy_6 = deep_copy(&circuit.circuit.as_ref().borrow());
            let zero_minus_copy_6 = zero.clone() - copy_6.clone();
            assert_ne!(copy_6, zero_minus_copy_6);
            let mut circuits = [copy_6, zero_minus_copy_6];
            ConstraintCircuitMonad::constant_folding(&mut circuits);
            let copy_6_is_zero = circuits[0].circuit.as_ref().borrow().is_zero();
            let copy_6_expr = circuits[0].circuit.as_ref().borrow().expression.clone();
            let zero_minus_copy_6_expr = circuits[1].circuit.as_ref().borrow().expression.clone();

            // An X field and a B field leaf will never be equal
            if copy_6_is_zero
                && (matches!(copy_6_expr, CircuitExpression::BConstant(_))
                    && matches!(zero_minus_copy_6_expr, CircuitExpression::BConstant(_))
                    || matches!(copy_6_expr, CircuitExpression::XConstant(_))
                        && matches!(zero_minus_copy_6_expr, CircuitExpression::XConstant(_)))
            {
                assert_eq!(
                    circuits[0], circuits[1],
                    "{} != {}",
                    circuits[0], circuits[1]
                );
                assert_eq!(
                    circuits[1], circuits[0],
                    "{} != {}",
                    circuits[1], circuits[0]
                );
            } else {
                assert_ne!(
                    circuits[0], circuits[1],
                    "{} == {}",
                    circuits[0], circuits[1]
                );
                assert_ne!(
                    circuits[1], circuits[0],
                    "{} == {}",
                    circuits[1], circuits[0]
                );
            }

            // Verify that constant folding handles a - 0 = a
            let copy_7 = deep_copy(&circuit.circuit.as_ref().borrow());
            let copy_7_alt = copy_7.clone() - zero.clone();
            assert_ne!(copy_7, copy_7_alt);
            let mut circuits = [copy_7, copy_7_alt];
            ConstraintCircuitMonad::constant_folding(&mut circuits);
            assert_eq!(
                circuits[0], circuits[1],
                "{} != {}",
                circuits[0], circuits[1]
            );
            assert_eq!(
                circuits[1], circuits[0],
                "{} != {}",
                circuits[1], circuits[0]
            );
        }
    }

    /// Recursively evaluates the given constraint circuit and its sub-circuits on the given
    /// base and extension table, and returns the result of the evaluation.
    /// At each recursive step, updates the given HashMap with the result of the evaluation.
    /// If the HashMap already contains the result of the evaluation, panics.
    /// This function is used to assert that the evaluation of a constraint circuit
    /// and its sub-circuits is unique.
    /// It is used to identify redundant constraints or sub-circuits.
    /// The employed method is the Schwartz-Zippel lemma.
    fn evaluate_assert_unique<II: InputIndicator>(
        constraint: &ConstraintCircuit<II>,
        challenges: &Challenges,
        base_rows: ArrayView2<BFieldElement>,
        ext_rows: ArrayView2<XFieldElement>,
        values: &mut HashMap<XFieldElement, (usize, ConstraintCircuit<II>)>,
    ) -> XFieldElement {
        let value = match &constraint.expression {
            BinaryOperation(binop, lhs, rhs) => {
                let lhs = lhs.as_ref().borrow();
                let rhs = rhs.as_ref().borrow();
                let lhs = evaluate_assert_unique(&lhs, challenges, base_rows, ext_rows, values);
                let rhs = evaluate_assert_unique(&rhs, challenges, base_rows, ext_rows, values);
                binop.operation(lhs, rhs)
            }
            _ => constraint.evaluate(base_rows, ext_rows, challenges),
        };

        let own_id = constraint.id.to_owned();
        let maybe_entry = values.insert(value, (own_id, constraint.clone()));
        if let Some((other_id, other_circuit)) = maybe_entry {
            assert_eq!(
                own_id, other_id,
                "Circuit ID {other_id} and circuit ID {own_id} are not unique. \
                Collision on:\n\
                ID {other_id} – {other_circuit}\n\
                ID {own_id} – {constraint}\n\
                Both evaluate to {value}.",
            );
        }

        value
    }

    /// Verify that all nodes evaluate to a unique value when given a randomized input.
    /// If this is not the case two nodes that are not equal evaluate to the same value.
    fn table_constraints_prop<II: InputIndicator>(
        constraints: &[ConstraintCircuit<II>],
        table_name: &str,
    ) {
        let seed = random();
        let mut rng = StdRng::seed_from_u64(seed);
        println!("seed: {seed}");

        let challenges: [XFieldElement; Challenges::num_challenges_to_sample()] = rng.gen();
        let challenges = challenges.to_vec();
        let challenges = Challenges::new(challenges, &[], &[]);

        let num_rows = 2;
        let base_shape = [num_rows, master_table::NUM_BASE_COLUMNS];
        let ext_shape = [num_rows, master_table::NUM_EXT_COLUMNS];
        let base_rows = Array2::from_shape_simple_fn(base_shape, || rng.gen::<BFieldElement>());
        let ext_rows = Array2::from_shape_simple_fn(ext_shape, || rng.gen::<XFieldElement>());
        let base_rows = base_rows.view();
        let ext_rows = ext_rows.view();

        let mut values = HashMap::new();
        for c in constraints.iter() {
            evaluate_assert_unique(c, &challenges, base_rows, ext_rows, &mut values);
        }

        let circuit_degree = constraints.iter().map(|c| c.degree()).max().unwrap_or(-1);
        println!("Max degree constraint for {table_name} table: {circuit_degree}");
    }

    fn build_constraints<II: InputIndicator>(
        multicircuit_builder: &dyn Fn(
            &ConstraintCircuitBuilder<II>,
        ) -> Vec<ConstraintCircuitMonad<II>>,
    ) -> Vec<ConstraintCircuit<II>> {
        let multicircuit = build_multicircuit(multicircuit_builder);
        let mut constraints = multicircuit.into_iter().map(|c| c.consume()).collect_vec();
        ConstraintCircuit::assert_has_unique_ids(&mut constraints);
        constraints
    }

    fn build_multicircuit<II: InputIndicator>(
        multicircuit_builder: &dyn Fn(
            &ConstraintCircuitBuilder<II>,
        ) -> Vec<ConstraintCircuitMonad<II>>,
    ) -> Vec<ConstraintCircuitMonad<II>> {
        let circuit_builder = ConstraintCircuitBuilder::new();
        let mut multicircuit = multicircuit_builder(&circuit_builder);
        ConstraintCircuitMonad::constant_folding(&mut multicircuit);
        multicircuit
    }

    #[test]
    fn constant_folding_processor_table_test() {
        let init = build_constraints(&ExtProcessorTable::initial_constraints);
        let cons = build_constraints(&ExtProcessorTable::consistency_constraints);
        let tran = build_constraints(&ExtProcessorTable::transition_constraints);
        let term = build_constraints(&ExtProcessorTable::terminal_constraints);
        table_constraints_prop(&init, "processor initial");
        table_constraints_prop(&cons, "processor consistency");
        table_constraints_prop(&tran, "processor transition");
        table_constraints_prop(&term, "processor terminal");
    }

    #[test]
    fn constant_folding_program_table_test() {
        let init = build_constraints(&ExtProgramTable::initial_constraints);
        let cons = build_constraints(&ExtProgramTable::consistency_constraints);
        let tran = build_constraints(&ExtProgramTable::transition_constraints);
        let term = build_constraints(&ExtProgramTable::terminal_constraints);
        table_constraints_prop(&init, "program initial");
        table_constraints_prop(&cons, "program consistency");
        table_constraints_prop(&tran, "program transition");
        table_constraints_prop(&term, "program terminal");
    }

    #[test]
    fn constant_folding_jump_stack_table_test() {
        let init = build_constraints(&ExtJumpStackTable::initial_constraints);
        let cons = build_constraints(&ExtJumpStackTable::consistency_constraints);
        let tran = build_constraints(&ExtJumpStackTable::transition_constraints);
        let term = build_constraints(&ExtJumpStackTable::terminal_constraints);
        table_constraints_prop(&init, "jump stack initial");
        table_constraints_prop(&cons, "jump stack consistency");
        table_constraints_prop(&tran, "jump stack transition");
        table_constraints_prop(&term, "jump stack terminal");
    }

    #[test]
    fn constant_folding_op_stack_table_test() {
        let init = build_constraints(&ExtOpStackTable::initial_constraints);
        let cons = build_constraints(&ExtOpStackTable::consistency_constraints);
        let tran = build_constraints(&ExtOpStackTable::transition_constraints);
        let term = build_constraints(&ExtOpStackTable::terminal_constraints);
        table_constraints_prop(&init, "op stack initial");
        table_constraints_prop(&cons, "op stack consistency");
        table_constraints_prop(&tran, "op stack transition");
        table_constraints_prop(&term, "op stack terminal");
    }

    #[test]
    fn constant_folding_ram_table_test() {
        let init = build_constraints(&ExtRamTable::initial_constraints);
        let cons = build_constraints(&ExtRamTable::consistency_constraints);
        let tran = build_constraints(&ExtRamTable::transition_constraints);
        let term = build_constraints(&ExtRamTable::terminal_constraints);
        table_constraints_prop(&init, "ram initial");
        table_constraints_prop(&cons, "ram consistency");
        table_constraints_prop(&tran, "ram transition");
        table_constraints_prop(&term, "ram terminal");
    }

    #[test]
    fn constant_folding_hash_table_test() {
        let init = build_constraints(&ExtHashTable::initial_constraints);
        let cons = build_constraints(&ExtHashTable::consistency_constraints);
        let tran = build_constraints(&ExtHashTable::transition_constraints);
        let term = build_constraints(&ExtHashTable::terminal_constraints);
        table_constraints_prop(&init, "hash initial");
        table_constraints_prop(&cons, "hash consistency");
        table_constraints_prop(&tran, "hash transition");
        table_constraints_prop(&term, "hash terminal");
    }

    #[test]
    fn constant_folding_u32_table_test() {
        let init = build_constraints(&ExtU32Table::initial_constraints);
        let cons = build_constraints(&ExtU32Table::consistency_constraints);
        let tran = build_constraints(&ExtU32Table::transition_constraints);
        let term = build_constraints(&ExtU32Table::terminal_constraints);
        table_constraints_prop(&init, "u32 initial");
        table_constraints_prop(&cons, "u32 consistency");
        table_constraints_prop(&tran, "u32 transition");
        table_constraints_prop(&term, "u32 terminal");
    }

    #[test]
    fn constant_folding_cascade_table_test() {
        let init = build_constraints(&ExtCascadeTable::initial_constraints);
        let cons = build_constraints(&ExtCascadeTable::consistency_constraints);
        let tran = build_constraints(&ExtCascadeTable::transition_constraints);
        let term = build_constraints(&ExtCascadeTable::terminal_constraints);
        table_constraints_prop(&init, "cascade initial");
        table_constraints_prop(&cons, "cascade consistency");
        table_constraints_prop(&tran, "cascade transition");
        table_constraints_prop(&term, "cascade terminal");
    }

    #[test]
    fn constant_folding_lookup_table_test() {
        let init = build_constraints(&ExtLookupTable::initial_constraints);
        let cons = build_constraints(&ExtLookupTable::consistency_constraints);
        let tran = build_constraints(&ExtLookupTable::transition_constraints);
        let term = build_constraints(&ExtLookupTable::terminal_constraints);
        table_constraints_prop(&init, "lookup initial");
        table_constraints_prop(&cons, "lookup consistency");
        table_constraints_prop(&tran, "lookup transition");
        table_constraints_prop(&term, "lookup terminal");
    }

    #[test]
    fn simple_degree_lowering_test() {
        let builder = ConstraintCircuitBuilder::new();
        let x = || builder.input(BaseRow(0));
        let x_pow_3 = x() * x() * x();
        let x_pow_5 = x() * x() * x() * x() * x();
        let mut multicircuit = [x_pow_5, x_pow_3];

        let target_degree = 3;
        let num_base_cols = 1;
        let num_ext_cols = 0;
        let (new_base_constraints, new_ext_constraints) = lower_degree_and_assert_properties(
            &mut multicircuit,
            target_degree,
            num_base_cols,
            num_ext_cols,
        );

        assert!(new_ext_constraints.is_empty());
        assert_eq!(1, new_base_constraints.len());
    }

    #[test]
    fn somewhat_simple_degree_lowering_test() {
        let builder = ConstraintCircuitBuilder::new();
        let x = |i| builder.input(BaseRow(i));
        let y = |i| builder.input(ExtRow(i));
        let b_con = |i: u64| builder.b_constant(i.into());

        let constraint_0 = x(0) * x(0) * (x(1) - x(2)) - x(0) * x(2) - b_con(42);
        let constraint_1 = x(1) * (x(1) - b_con(5)) * x(2) * (x(2) - b_con(1));
        let constraint_2 = y(0)
            * (b_con(2) * x(0) + b_con(3) * x(1) + b_con(4) * x(2))
            * (b_con(4) * x(0) + b_con(8) * x(1) + b_con(16) * x(2))
            - y(1);

        let mut multicircuit = [constraint_0, constraint_1, constraint_2];

        let target_degree = 2;
        let num_base_cols = 3;
        let num_ext_cols = 2;
        let (new_base_constraints, new_ext_constraints) = lower_degree_and_assert_properties(
            &mut multicircuit,
            target_degree,
            num_base_cols,
            num_ext_cols,
        );

        assert!(new_base_constraints.len() <= 3);
        assert!(new_ext_constraints.len() <= 1);
    }

    #[test]
    fn program_table_initial_constraints_degree_lowering_test() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtProgramTable::initial_constraints),
            AIR_TARGET_DEGREE,
            PROGRAM_TABLE_END,
            EXT_PROGRAM_TABLE_END,
        );
    }

    #[test]
    fn program_table_consistency_constraints_degree_lowering_test() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtProgramTable::consistency_constraints),
            AIR_TARGET_DEGREE,
            PROGRAM_TABLE_END,
            EXT_PROGRAM_TABLE_END,
        );
    }

    #[test]
    fn program_table_transition_constraints_degree_lowering_test() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtProgramTable::transition_constraints),
            AIR_TARGET_DEGREE,
            PROGRAM_TABLE_END,
            EXT_PROGRAM_TABLE_END,
        );
    }

    #[test]
    fn program_table_terminal_constraints_degree_lowering_test() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtProgramTable::terminal_constraints),
            AIR_TARGET_DEGREE,
            PROGRAM_TABLE_END,
            EXT_PROGRAM_TABLE_END,
        );
    }

    #[test]
    fn processor_table_initial_constraints_degree_lowering_test() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtProcessorTable::initial_constraints),
            AIR_TARGET_DEGREE,
            PROCESSOR_TABLE_END,
            EXT_PROCESSOR_TABLE_END,
        );
    }

    #[test]
    fn processor_table_consistency_constraints_degree_lowering_test() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtProcessorTable::consistency_constraints),
            AIR_TARGET_DEGREE,
            PROCESSOR_TABLE_END,
            EXT_PROCESSOR_TABLE_END,
        );
    }

    #[test]
    fn processor_table_transition_constraints_degree_lowering_test() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtProcessorTable::transition_constraints),
            AIR_TARGET_DEGREE,
            PROCESSOR_TABLE_END,
            EXT_PROCESSOR_TABLE_END,
        );
    }

    #[test]
    fn processor_table_terminal_constraints_degree_lowering_test() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtProcessorTable::terminal_constraints),
            AIR_TARGET_DEGREE,
            PROCESSOR_TABLE_END,
            EXT_PROCESSOR_TABLE_END,
        );
    }

    #[test]
    fn op_stack_table_initial_constraints_degree_lowering_test() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtOpStackTable::initial_constraints),
            AIR_TARGET_DEGREE,
            OP_STACK_TABLE_END,
            EXT_OP_STACK_TABLE_END,
        );
    }

    #[test]
    fn op_stack_table_consistency_constraints_degree_lowering_test() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtOpStackTable::consistency_constraints),
            AIR_TARGET_DEGREE,
            OP_STACK_TABLE_END,
            EXT_OP_STACK_TABLE_END,
        );
    }

    #[test]
    fn op_stack_table_transition_constraints_degree_lowering_test() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtOpStackTable::transition_constraints),
            AIR_TARGET_DEGREE,
            OP_STACK_TABLE_END,
            EXT_OP_STACK_TABLE_END,
        );
    }

    #[test]
    fn op_stack_table_terminal_constraints_degree_lowering_test() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtOpStackTable::terminal_constraints),
            AIR_TARGET_DEGREE,
            OP_STACK_TABLE_END,
            EXT_OP_STACK_TABLE_END,
        );
    }

    #[test]
    fn ram_table_initial_constraints_degree_lowering_test() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtRamTable::initial_constraints),
            AIR_TARGET_DEGREE,
            RAM_TABLE_END,
            EXT_RAM_TABLE_END,
        );
    }

    #[test]
    fn ram_table_consistency_constraints_degree_lowering_test() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtRamTable::consistency_constraints),
            AIR_TARGET_DEGREE,
            RAM_TABLE_END,
            EXT_RAM_TABLE_END,
        );
    }

    #[test]
    fn ram_table_transition_constraints_degree_lowering_test() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtRamTable::transition_constraints),
            AIR_TARGET_DEGREE,
            RAM_TABLE_END,
            EXT_RAM_TABLE_END,
        );
    }

    #[test]
    fn ram_table_terminal_constraints_degree_lowering_test() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtRamTable::terminal_constraints),
            AIR_TARGET_DEGREE,
            RAM_TABLE_END,
            EXT_RAM_TABLE_END,
        );
    }

    #[test]
    fn jump_stack_table_initial_constraints_degree_lowering_test() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtJumpStackTable::initial_constraints),
            AIR_TARGET_DEGREE,
            JUMP_STACK_TABLE_END,
            EXT_JUMP_STACK_TABLE_END,
        );
    }

    #[test]
    fn jump_stack_table_consistency_constraints_degree_lowering_test() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtJumpStackTable::consistency_constraints),
            AIR_TARGET_DEGREE,
            JUMP_STACK_TABLE_END,
            EXT_JUMP_STACK_TABLE_END,
        );
    }

    #[test]
    fn jump_stack_table_transition_constraints_degree_lowering_test() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtJumpStackTable::transition_constraints),
            AIR_TARGET_DEGREE,
            JUMP_STACK_TABLE_END,
            EXT_JUMP_STACK_TABLE_END,
        );
    }

    #[test]
    fn jump_stack_table_terminal_constraints_degree_lowering_test() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtJumpStackTable::terminal_constraints),
            AIR_TARGET_DEGREE,
            JUMP_STACK_TABLE_END,
            EXT_JUMP_STACK_TABLE_END,
        );
    }

    #[test]
    fn hash_table_initial_constraints_degree_lowering_test() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtHashTable::initial_constraints),
            AIR_TARGET_DEGREE,
            HASH_TABLE_END,
            EXT_HASH_TABLE_END,
        );
    }

    #[test]
    fn hash_table_consistency_constraints_degree_lowering_test() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtHashTable::consistency_constraints),
            AIR_TARGET_DEGREE,
            HASH_TABLE_END,
            EXT_HASH_TABLE_END,
        );
    }

    #[test]
    fn hash_table_transition_constraints_degree_lowering_test() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtHashTable::transition_constraints),
            AIR_TARGET_DEGREE,
            HASH_TABLE_END,
            EXT_HASH_TABLE_END,
        );
    }

    #[test]
    fn hash_table_terminal_constraints_degree_lowering_test() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtHashTable::terminal_constraints),
            AIR_TARGET_DEGREE,
            HASH_TABLE_END,
            EXT_HASH_TABLE_END,
        );
    }

    #[test]
    fn cascade_table_initial_constraints_degree_lowering_test() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtCascadeTable::initial_constraints),
            AIR_TARGET_DEGREE,
            CASCADE_TABLE_END,
            EXT_CASCADE_TABLE_END,
        );
    }

    #[test]
    fn cascade_table_consistency_constraints_degree_lowering_test() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtCascadeTable::consistency_constraints),
            AIR_TARGET_DEGREE,
            CASCADE_TABLE_END,
            EXT_CASCADE_TABLE_END,
        );
    }

    #[test]
    fn cascade_table_transition_constraints_degree_lowering_test() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtCascadeTable::transition_constraints),
            AIR_TARGET_DEGREE,
            CASCADE_TABLE_END,
            EXT_CASCADE_TABLE_END,
        );
    }

    #[test]
    fn cascade_table_terminal_constraints_degree_lowering_test() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtCascadeTable::terminal_constraints),
            AIR_TARGET_DEGREE,
            CASCADE_TABLE_END,
            EXT_CASCADE_TABLE_END,
        );
    }

    #[test]
    fn lookup_table_initial_constraints_degree_lowering_test() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtLookupTable::initial_constraints),
            AIR_TARGET_DEGREE,
            LOOKUP_TABLE_END,
            EXT_LOOKUP_TABLE_END,
        );
    }

    #[test]
    fn lookup_table_consistency_constraints_degree_lowering_test() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtLookupTable::consistency_constraints),
            AIR_TARGET_DEGREE,
            LOOKUP_TABLE_END,
            EXT_LOOKUP_TABLE_END,
        );
    }

    #[test]
    fn lookup_table_transition_constraints_degree_lowering_test() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtLookupTable::transition_constraints),
            AIR_TARGET_DEGREE,
            LOOKUP_TABLE_END,
            EXT_LOOKUP_TABLE_END,
        );
    }

    #[test]
    fn lookup_table_terminal_constraints_degree_lowering_test() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtLookupTable::terminal_constraints),
            AIR_TARGET_DEGREE,
            LOOKUP_TABLE_END,
            EXT_LOOKUP_TABLE_END,
        );
    }

    #[test]
    fn u32_table_initial_constraints_degree_lowering_test() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtU32Table::initial_constraints),
            AIR_TARGET_DEGREE,
            U32_TABLE_END,
            EXT_U32_TABLE_END,
        );
    }

    #[test]
    fn u32_table_consistency_constraints_degree_lowering_test() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtU32Table::consistency_constraints),
            AIR_TARGET_DEGREE,
            U32_TABLE_END,
            EXT_U32_TABLE_END,
        );
    }

    #[test]
    fn u32_table_transition_constraints_degree_lowering_test() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtU32Table::transition_constraints),
            AIR_TARGET_DEGREE,
            U32_TABLE_END,
            EXT_U32_TABLE_END,
        );
    }

    #[test]
    fn u32_table_terminal_constraints_degree_lowering_test() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtU32Table::terminal_constraints),
            AIR_TARGET_DEGREE,
            U32_TABLE_END,
            EXT_U32_TABLE_END,
        );
    }

    /// Like [`ConstraintCircuitMonad::lower_to_degree`] with additional assertion of expected
    /// properties. Also prints:
    /// - the given multicircuit prior to degree lowering
    /// - the multicircuit after degree lowering
    /// - the new base constraints
    /// - the new extension constraints
    /// - the numbers of original and new constraints
    fn lower_degree_and_assert_properties<II: InputIndicator>(
        multicircuit: &mut [ConstraintCircuitMonad<II>],
        target_deg: Degree,
        num_base_cols: usize,
        num_ext_cols: usize,
    ) -> (
        Vec<ConstraintCircuitMonad<II>>,
        Vec<ConstraintCircuitMonad<II>>,
    ) {
        let seed = random();
        let mut rng = StdRng::seed_from_u64(seed);
        println!("seed: {seed}");

        let num_constraints = multicircuit.len();
        println!("original multicircuit:");
        for circuit in multicircuit.iter() {
            println!("  {circuit}");
        }

        let (new_base_constraints, new_ext_constraints) = ConstraintCircuitMonad::lower_to_degree(
            multicircuit,
            target_deg,
            num_base_cols,
            num_ext_cols,
        );

        assert_eq!(num_constraints, multicircuit.len());
        assert!(ConstraintCircuitMonad::multicircuit_degree(multicircuit) <= target_deg);
        assert!(ConstraintCircuitMonad::multicircuit_degree(&new_base_constraints) <= target_deg);
        assert!(ConstraintCircuitMonad::multicircuit_degree(&new_ext_constraints) <= target_deg);

        // Check that the new constraints are simple substitutions.
        let mut substitution_rules = vec![];
        for (constraint_type, constraints) in [
            ("base", &new_base_constraints),
            ("ext", &new_ext_constraints),
        ] {
            for (i, constraint) in constraints.iter().enumerate() {
                let expression = constraint.circuit.as_ref().borrow().expression.clone();
                let BinaryOperation(BinOp::Sub, lhs, rhs) = expression else {
                    panic!("New {constraint_type} constraint {i} must be a subtraction.");
                };
                let Input(input_indicator) = lhs.as_ref().borrow().expression.clone() else {
                    panic!("New {constraint_type} constraint {i} must be a simple substitution.");
                };
                let substitution_rule = rhs.as_ref().borrow().clone();
                assert_substitution_rule_uses_legal_variables(input_indicator, &substitution_rule);
                substitution_rules.push(substitution_rule);
            }
        }

        // Use the Schwartz-Zippel lemma to check no two substitution rules are equal.
        let challenges: [XFieldElement; Challenges::num_challenges_to_sample()] = rng.gen();
        let challenges = challenges.to_vec();
        let challenges = Challenges::new(challenges, &[], &[]);

        let num_rows = 2;
        let num_new_base_constraints = new_base_constraints.len();
        let num_new_ext_constraints = new_ext_constraints.len();
        let num_base_cols = master_table::NUM_BASE_COLUMNS + num_new_base_constraints;
        let num_ext_cols = master_table::NUM_EXT_COLUMNS + num_new_ext_constraints;
        let base_shape = [num_rows, num_base_cols];
        let ext_shape = [num_rows, num_ext_cols];
        let base_rows = Array2::from_shape_simple_fn(base_shape, || rng.gen::<BFieldElement>());
        let ext_rows = Array2::from_shape_simple_fn(ext_shape, || rng.gen::<XFieldElement>());
        let base_rows = base_rows.view();
        let ext_rows = ext_rows.view();

        let evaluated_substitution_rules = substitution_rules
            .iter()
            .map(|c| c.evaluate(base_rows, ext_rows, &challenges));

        let mut values_to_index = HashMap::new();
        for (idx, value) in evaluated_substitution_rules.enumerate() {
            if let Some(index) = values_to_index.get(&value) {
                panic!("Substitution {idx} must be distinct from substitution {index}.");
            } else {
                values_to_index.insert(value, idx);
            }
        }

        // Print the multicircuit and new constraints after degree lowering.
        println!("new multicircuit:");
        for circuit in multicircuit.iter() {
            println!("  {circuit}");
        }
        println!("new base constraints:");
        for constraint in new_base_constraints.iter() {
            println!("  {constraint}");
        }
        println!("new ext constraints:");
        for constraint in new_ext_constraints.iter() {
            println!("  {constraint}");
        }

        println!(
            "Started with {num_constraints} constraints. \
            Derived {num_new_base_constraints} new base, \
            {num_new_ext_constraints} new extension constraints."
        );

        (new_base_constraints, new_ext_constraints)
    }

    /// Panics if the given substitution rule uses variables with an index greater than (or equal)
    /// to the given index. In practice, this given index corresponds to a newly introduced
    /// variable.
    fn assert_substitution_rule_uses_legal_variables<II: InputIndicator>(
        new_var: II,
        substitution_rule: &ConstraintCircuit<II>,
    ) {
        match substitution_rule.expression.clone() {
            BinaryOperation(_, lhs, rhs) => {
                let lhs = lhs.as_ref().borrow();
                let rhs = rhs.as_ref().borrow();
                assert_substitution_rule_uses_legal_variables(new_var, &lhs);
                assert_substitution_rule_uses_legal_variables(new_var, &rhs);
            }
            Input(old_var) => {
                let new_var_is_base = new_var.is_base_table_column();
                let old_var_is_base = old_var.is_base_table_column();
                let legal_substitute = match (new_var_is_base, old_var_is_base) {
                    (true, true) => old_var.base_col_index() < new_var.base_col_index(),
                    (true, false) => false,
                    (false, true) => true,
                    (false, false) => old_var.ext_col_index() < new_var.ext_col_index(),
                };
                assert!(legal_substitute, "Cannot replace {old_var} with {new_var}.");
            }
            _ => (),
        };
    }
}
