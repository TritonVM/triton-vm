//! Constraint circuits are a way to represent constraint polynomials in a way
//! that is amenable to optimizations. The constraint circuit is a directed
//! acyclic graph (DAG) of [`CircuitExpression`]s, where each
//! `CircuitExpression` is a node in the graph. The edges of the graph are
//! labeled with [`BinOp`]s. The leafs of the graph are the inputs to the
//! constraint polynomial, and the (multiple) roots of the graph are the outputs
//! of all the constraint polynomials, with each root corresponding to a
//! different constraint polynomial. Because the graph has multiple roots, it is
//! called a “multitree.”

// See the corresponding attribute in triton_vm/lib.rs
#![cfg_attr(coverage_nightly, feature(coverage_attribute))]

use std::cell::RefCell;
use std::cmp;
use std::collections::HashMap;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FmtResult;
use std::hash::Hash;
use std::hash::Hasher;
use std::iter::Sum;
use std::ops::Add;
use std::ops::AddAssign;
use std::ops::Mul;
use std::ops::Neg;
use std::ops::Sub;
use std::rc::Rc;

use arbitrary::Arbitrary;
use arbitrary::Unstructured;
use itertools::Itertools;
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
/// the binary operation. The left and right children are not themselves
/// `CircuitExpression`s, but rather [`ConstraintCircuit`]s, which is a wrapper
/// around `CircuitExpression` that manages additional bookkeeping information.
#[derive(Debug, Clone)]
pub enum CircuitExpression<II: InputIndicator> {
    BConst(BFieldElement),
    XConst(XFieldElement),
    Input(II),
    Challenge(usize),
    BinOp(
        BinOp,
        Rc<RefCell<ConstraintCircuit<II>>>,
        Rc<RefCell<ConstraintCircuit<II>>>,
    ),
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
                lhs.borrow().hash(state);
                rhs.borrow().hash(state);
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

impl<II: InputIndicator> Hash for ConstraintCircuitMonad<II> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.circuit.borrow().hash(state)
    }
}

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
                write!(f, "({}) {operation} ({})", lhs.borrow(), rhs.borrow())
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

        if let CircuitExpression::BinOp(_, lhs, rhs) = &self.expression {
            lhs.borrow_mut().reset_ref_count_for_tree();
            rhs.borrow_mut().reset_ref_count_for_tree();
        }
    }

    /// Assert that all IDs in the subtree are unique.
    ///
    /// # Panics
    ///
    /// Panics if a duplicate ID is found.
    fn assert_unique_ids_inner(&mut self, ids: &mut HashMap<usize, ConstraintCircuit<II>>) {
        self.ref_count += 1;

        // Try to detect duplicate IDs only once for this node.
        if self.ref_count > 1 {
            return;
        }

        let self_id = self.id;
        if let Some(other) = ids.insert(self_id, self.clone()) {
            panic!("Repeated ID: {self_id}\nSelf:\n{self}\n{self:?}\nOther:\n{other}\n{other:?}");
        }

        if let CircuitExpression::BinOp(_, lhs, rhs) = &self.expression {
            lhs.borrow_mut().assert_unique_ids_inner(ids);
            rhs.borrow_mut().assert_unique_ids_inner(ids);
        }
    }

    /// Assert that a multicircuit has unique IDs.
    /// Also determines how often each node is referenced, updating the
    /// respective `ref_count`s.
    ///
    /// # Panics
    ///
    /// Panics if a duplicate ID is found.
    pub fn assert_unique_ids(constraints: &mut [ConstraintCircuit<II>]) {
        // inner uniqueness checks relies on reference counters being 0 for
        // unseen nodes
        for circuit in constraints.iter_mut() {
            circuit.reset_ref_count_for_tree();
        }
        let mut ids: HashMap<usize, ConstraintCircuit<II>> = HashMap::new();
        for circuit in constraints.iter_mut() {
            circuit.assert_unique_ids_inner(&mut ids);
        }
    }

    /// Return degree of the multivariate polynomial represented by this circuit
    pub fn degree(&self) -> isize {
        if self.is_zero() {
            return -1;
        }

        match &self.expression {
            CircuitExpression::BinOp(binop, lhs, rhs) => {
                let degree_lhs = lhs.borrow().degree();
                let degree_rhs = rhs.borrow().degree();
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
    pub fn all_ref_counters(&self) -> Vec<usize> {
        let mut ref_counters = vec![self.ref_count];
        if let CircuitExpression::BinOp(_, lhs, rhs) = &self.expression {
            ref_counters.extend(lhs.borrow().all_ref_counters());
            ref_counters.extend(rhs.borrow().all_ref_counters());
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
    pub fn evaluates_to_base_element(&self) -> bool {
        match &self.expression {
            CircuitExpression::BConst(_) => true,
            CircuitExpression::XConst(_) => false,
            CircuitExpression::Input(indicator) => indicator.is_main_table_column(),
            CircuitExpression::Challenge(_) => false,
            CircuitExpression::BinOp(_, lhs, rhs) => {
                lhs.borrow().evaluates_to_base_element() && rhs.borrow().evaluates_to_base_element()
            }
        }
    }

    pub fn evaluate(
        &self,
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
                let lhs_value = lhs.borrow().evaluate(main_table, aux_table, challenges);
                let rhs_value = rhs.borrow().evaluate(main_table, aux_table, challenges);
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
/// One can create new instances of [`ConstraintCircuitMonad`] by applying
/// arithmetic operations to existing instances, *e.g.*, `let c = a * b;`.
#[derive(Clone)]
pub struct ConstraintCircuitMonad<II: InputIndicator> {
    pub circuit: Rc<RefCell<ConstraintCircuit<II>>>,
    pub builder: ConstraintCircuitBuilder<II>,
}

impl<II: InputIndicator> Debug for ConstraintCircuitMonad<II> {
    // `all_nodes` contains itself, leading to infinite recursion during `Debug`
    // printing. Hence, this manual implementation.
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        f.debug_struct("ConstraintCircuitMonad")
            .field("id", &self.circuit)
            .field("all_nodes length: ", &self.builder.all_nodes.borrow().len())
            .field("id_counter_ref value: ", &self.builder.id_counter.borrow())
            .finish()
    }
}

impl<II: InputIndicator> Display for ConstraintCircuitMonad<II> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        write!(f, "{}", self.circuit.borrow())
    }
}

impl<II: InputIndicator> PartialEq for ConstraintCircuitMonad<II> {
    // Equality for the ConstraintCircuitMonad is defined by the circuit, not
    // the other metadata (e.g. ID) that it carries around.
    fn eq(&self, other: &Self) -> bool {
        self.circuit == other.circuit
    }
}

impl<II: InputIndicator> Eq for ConstraintCircuitMonad<II> {}

/// Helper function for binary operations that are used to generate new parent
/// nodes in the multitree that represents the algebraic circuit. Ensures that
/// each newly created node has a unique ID.
///
/// This function does not (currently) catch expressions of the form ((x+1)+1).
fn binop<II: InputIndicator>(
    binop: BinOp,
    lhs: ConstraintCircuitMonad<II>,
    rhs: ConstraintCircuitMonad<II>,
) -> ConstraintCircuitMonad<II> {
    assert!(lhs.builder.is_same_as(&rhs.builder));

    match (binop, &lhs, &rhs) {
        (BinOp::Add, _, zero) if zero.circuit.borrow().is_zero() => return lhs,
        (BinOp::Add, zero, _) if zero.circuit.borrow().is_zero() => return rhs,
        (BinOp::Mul, _, one) if one.circuit.borrow().is_one() => return lhs,
        (BinOp::Mul, one, _) if one.circuit.borrow().is_one() => return rhs,
        (BinOp::Mul, _, zero) if zero.circuit.borrow().is_zero() => return rhs,
        (BinOp::Mul, zero, _) if zero.circuit.borrow().is_zero() => return lhs,
        _ => (),
    };

    match (
        &lhs.circuit.borrow().expression,
        &rhs.circuit.borrow().expression,
    ) {
        (&CircuitExpression::BConst(l), &CircuitExpression::BConst(r)) => {
            return lhs.builder.b_constant(binop.operation(l, r));
        }
        (&CircuitExpression::BConst(l), &CircuitExpression::XConst(r)) => {
            return lhs.builder.x_constant(binop.operation(l, r));
        }
        (&CircuitExpression::XConst(l), &CircuitExpression::BConst(r)) => {
            return lhs.builder.x_constant(binop.operation(l, r));
        }
        (&CircuitExpression::XConst(l), &CircuitExpression::XConst(r)) => {
            return lhs.builder.x_constant(binop.operation(l, r));
        }
        _ => (),
    };

    // all `BinOp`s are commutative – try both orders of the operands
    let all_nodes = &mut lhs.builder.all_nodes.borrow_mut();
    let new_node = binop_new_node(binop, &rhs, &lhs);
    if let Some(node) = all_nodes.values().find(|&n| n == &new_node) {
        return node.to_owned();
    }

    let new_node = binop_new_node(binop, &lhs, &rhs);
    if let Some(node) = all_nodes.values().find(|&n| n == &new_node) {
        return node.to_owned();
    }

    let new_id = new_node.circuit.borrow().id;
    let maybe_existing_node = all_nodes.insert(new_id, new_node.clone());
    let new_node_is_new = maybe_existing_node.is_none();
    assert!(new_node_is_new, "new node must not overwrite existing node");
    lhs.builder.id_counter.borrow_mut().add_assign(1);
    new_node
}

fn binop_new_node<II: InputIndicator>(
    binop: BinOp,
    lhs: &ConstraintCircuitMonad<II>,
    rhs: &ConstraintCircuitMonad<II>,
) -> ConstraintCircuitMonad<II> {
    let id = lhs.builder.id_counter.borrow().to_owned();
    let expression = CircuitExpression::BinOp(binop, lhs.circuit.clone(), rhs.circuit.clone());
    let circuit = ConstraintCircuit::new(id, expression);
    lhs.builder.new_monad(circuit)
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
        binop(BinOp::Add, self, -rhs)
    }
}

impl<II: InputIndicator> Mul for ConstraintCircuitMonad<II> {
    type Output = ConstraintCircuitMonad<II>;

    fn mul(self, rhs: Self) -> Self::Output {
        binop(BinOp::Mul, self, rhs)
    }
}

impl<II: InputIndicator> Neg for ConstraintCircuitMonad<II> {
    type Output = ConstraintCircuitMonad<II>;

    fn neg(self) -> Self::Output {
        binop(BinOp::Mul, self.builder.minus_one(), self)
    }
}

/// This will panic if the iterator is empty because the neutral element needs a
/// unique ID, and we have no way of getting that here.
impl<II: InputIndicator> Sum for ConstraintCircuitMonad<II> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|accum, item| accum + item)
            .expect("ConstraintCircuitMonad Iterator was empty")
    }
}

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

/// Helper struct to construct new leaf nodes (*i.e.*, input or challenge or
/// constant) in the circuit multitree. Ensures that newly created nodes, even
/// non-leaf nodes created through joining two other nodes using an arithmetic
/// operation, get a unique ID.
#[derive(Debug, Clone, Eq, PartialEq)]
pub struct ConstraintCircuitBuilder<II: InputIndicator> {
    id_counter: Rc<RefCell<usize>>,
    all_nodes: Rc<RefCell<HashMap<usize, ConstraintCircuitMonad<II>>>>,
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
            all_nodes: Rc::new(RefCell::new(HashMap::new())),
        }
    }

    /// Check whether two builders are the same.
    ///
    /// Notably, this is distinct from checking equality: two builders are equal
    /// if they are in the same state. Two builders are the same if they are
    /// the same instance.
    pub fn is_same_as(&self, other: &Self) -> bool {
        Rc::ptr_eq(&self.id_counter, &other.id_counter)
            && Rc::ptr_eq(&self.all_nodes, &other.all_nodes)
    }

    fn new_monad(&self, circuit: ConstraintCircuit<II>) -> ConstraintCircuitMonad<II> {
        let circuit = Rc::new(RefCell::new(circuit));
        ConstraintCircuitMonad {
            circuit,
            builder: self.clone(),
        }
    }

    /// The unique monad representing the constant value 0.
    pub fn zero(&self) -> ConstraintCircuitMonad<II> {
        self.b_constant(0)
    }

    /// The unique monad representing the constant value 1.
    pub fn one(&self) -> ConstraintCircuitMonad<II> {
        self.b_constant(1)
    }

    /// The unique monad representing the constant value -1.
    pub fn minus_one(&self) -> ConstraintCircuitMonad<II> {
        self.b_constant(-1)
    }

    /// Leaf node with constant over the [base field][BFieldElement].
    pub fn b_constant<B>(&self, bfe: B) -> ConstraintCircuitMonad<II>
    where
        B: Into<BFieldElement>,
    {
        self.make_leaf(CircuitExpression::BConst(bfe.into()))
    }

    /// Leaf node with constant over the [extension field][XFieldElement].
    pub fn x_constant<X>(&self, xfe: X) -> ConstraintCircuitMonad<II>
    where
        X: Into<XFieldElement>,
    {
        self.make_leaf(CircuitExpression::XConst(xfe.into()))
    }

    /// Create deterministic input leaf node.
    pub fn input(&self, input: II) -> ConstraintCircuitMonad<II> {
        self.make_leaf(CircuitExpression::Input(input))
    }

    /// Create challenge leaf node.
    pub fn challenge<C>(&self, challenge: C) -> ConstraintCircuitMonad<II>
    where
        C: Into<usize>,
    {
        self.make_leaf(CircuitExpression::Challenge(challenge.into()))
    }

    fn make_leaf(&self, mut expression: CircuitExpression<II>) -> ConstraintCircuitMonad<II> {
        assert!(
            !matches!(expression, CircuitExpression::BinOp(_, _, _)),
            "`make_leaf` is intended for anything but `BinOp`s"
        );

        // don't use X field if the B field suffices
        if let CircuitExpression::XConst(xfe) = expression
            && let Some(bfe) = xfe.unlift()
        {
            expression = CircuitExpression::BConst(bfe);
        }

        let id = self.id_counter.borrow().to_owned();
        let circuit = ConstraintCircuit::new(id, expression);
        let new_node = self.new_monad(circuit);

        if let Some(same_node) = self.all_nodes.borrow().values().find(|&n| n == &new_node) {
            return same_node.to_owned();
        }

        let maybe_previous_node = self.all_nodes.borrow_mut().insert(id, new_node.clone());
        let new_node_is_new = maybe_previous_node.is_none();
        assert!(new_node_is_new, "Leaf-created node must be new… {new_node}");
        self.id_counter.borrow_mut().add_assign(1);
        new_node
    }

    /// Replace all pointers to a given node (identified by `old_id`) by one
    /// to the new node.
    ///
    /// A circuit's root node cannot be substituted with this method. Manual
    /// care must be taken to update the root node if necessary.
    fn redirect_all_references_to_node(&self, old_id: usize, new: ConstraintCircuitMonad<II>) {
        self.all_nodes.borrow_mut().remove(&old_id);
        for node in self.all_nodes.borrow_mut().values_mut() {
            let CircuitExpression::BinOp(_, ref mut left, ref mut right) =
                node.circuit.borrow_mut().expression
            else {
                continue;
            };
            if left.borrow().id == old_id {
                *left = new.circuit.clone();
            }
            if right.borrow().id == old_id {
                *right = new.circuit.clone();
            }
        }
    }
}

impl<'a, II: InputIndicator + Arbitrary<'a>> Arbitrary<'a> for ConstraintCircuitMonad<II> {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let builder = ConstraintCircuitBuilder::new();
        let mut random_circuit = random_circuit_leaf(&builder, u)?;

        let num_nodes_in_circuit = u.arbitrary_len::<Self>()?;
        for _ in 0..num_nodes_in_circuit {
            let leaf = random_circuit_leaf(&builder, u)?;
            match u.int_in_range(0..=5)? {
                0 => random_circuit = random_circuit * leaf,
                1 => random_circuit = random_circuit + leaf,
                2 => random_circuit = random_circuit - leaf,
                3 => random_circuit = leaf * random_circuit,
                4 => random_circuit = leaf + random_circuit,
                5 => random_circuit = leaf - random_circuit,
                _ => unreachable!(),
            }
        }

        Ok(random_circuit)
    }
}

fn random_circuit_leaf<'a, II: InputIndicator + Arbitrary<'a>>(
    builder: &ConstraintCircuitBuilder<II>,
    u: &mut Unstructured<'a>,
) -> arbitrary::Result<ConstraintCircuitMonad<II>> {
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

#[cfg(test)]
#[cfg_attr(coverage_nightly, coverage(off))]
mod tests {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;
    use std::ops::Not;

    use itertools::Itertools;
    use ndarray::Array2;
    use ndarray::Axis;
    use proptest::arbitrary::Arbitrary;
    use proptest::collection::vec;
    use proptest::prelude::*;
    use proptest_arbitrary_interop::arb;
    use test_strategy::proptest;

    use super::*;

    impl<II: InputIndicator> ConstraintCircuitMonad<II> {
        /// Check whether the given node is contained in this circuit.
        fn contains(&self, other: &Self) -> bool {
            let self_expression = &self.circuit.borrow().expression;
            let other_expression = &other.circuit.borrow().expression;
            self_expression.contains(other_expression)
        }

        /// Produces an iter over all nodes in the multicircuit, if it is
        /// non-empty.
        ///
        /// Helper function for counting the number of nodes of a specific type.
        fn iter_nodes(
            constraints: &[Self],
        ) -> std::vec::IntoIter<(usize, ConstraintCircuitMonad<II>)> {
            let Some(first) = constraints.first() else {
                return vec![].into_iter();
            };

            first
                .builder
                .all_nodes
                .borrow()
                .iter()
                .map(|(n, m)| (*n, m.clone()))
                .collect_vec()
                .into_iter()
        }

        /// The total number of nodes in the multicircuit
        fn num_nodes(constraints: &[Self]) -> usize {
            Self::iter_nodes(constraints).count()
        }

        /// Determine if the constraint circuit monad corresponds to a main
        /// table column.
        fn is_main_table_column(&self) -> bool {
            let CircuitExpression::Input(ii) = self.circuit.borrow().expression else {
                return false;
            };

            ii.is_main_table_column()
        }

        /// The number of inputs from the main table
        fn num_main_inputs(constraints: &[Self]) -> usize {
            Self::iter_nodes(constraints)
                .filter(|(_, cc)| cc.is_main_table_column())
                .filter(|(_, cc)| cc.circuit.borrow().evaluates_to_base_element())
                .count()
        }

        /// The number of inputs from the aux table
        fn num_aux_inputs(constraints: &[Self]) -> usize {
            Self::iter_nodes(constraints)
                .filter(|(_, cc)| !cc.is_main_table_column())
                .filter(|(_, cc)| {
                    matches!(cc.circuit.borrow().expression, CircuitExpression::Input(_))
                })
                .count()
        }

        /// The number of total (*i.e.*, main + aux) inputs
        fn num_inputs(constraints: &[Self]) -> usize {
            Self::num_main_inputs(constraints) + Self::num_aux_inputs(constraints)
        }

        /// The number of challenges
        fn num_challenges(constraints: &[Self]) -> usize {
            Self::iter_nodes(constraints)
                .filter(|(_, cc)| {
                    matches!(
                        cc.circuit.borrow().expression,
                        CircuitExpression::Challenge(_)
                    )
                })
                .count()
        }

        /// The number of `BinOp`s
        fn num_binops(constraints: &[Self]) -> usize {
            Self::iter_nodes(constraints)
                .filter(|(_, cc)| {
                    matches!(
                        cc.circuit.borrow().expression,
                        CircuitExpression::BinOp(_, _, _)
                    )
                })
                .count()
        }

        /// The number of BFE constants
        fn num_bfield_constants(constraints: &[Self]) -> usize {
            Self::iter_nodes(constraints)
                .filter(|(_, cc)| {
                    matches!(cc.circuit.borrow().expression, CircuitExpression::BConst(_))
                })
                .count()
        }

        /// The number of XFE constants
        fn num_xfield_constants(constraints: &[Self]) -> usize {
            Self::iter_nodes(constraints)
                .filter(|(_, cc)| {
                    matches!(
                        cc.circuit.as_ref().borrow().expression,
                        CircuitExpression::XConst(_)
                    )
                })
                .count()
        }
    }

    impl<II: InputIndicator> CircuitExpression<II> {
        /// Check whether the given node is contained in this circuit.
        fn contains(&self, other: &Self) -> bool {
            if self == other {
                return true;
            }
            let CircuitExpression::BinOp(_, lhs, rhs) = self else {
                return false;
            };

            lhs.borrow().expression.contains(other) || rhs.borrow().expression.contains(other)
        }
    }

    /// The [`Hash`] trait requires:
    /// circuit_0 == circuit_1 => hash(circuit_0) == hash(circuit_1)
    ///
    /// By De-Morgan's law, this is equivalent to the more meaningful test:
    /// hash(circuit_0) != hash(circuit_1) => circuit_0 != circuit_1
    #[proptest]
    fn unequal_hash_implies_unequal_constraint_circuit_monad(
        #[strategy(arb())] circuit_0: ConstraintCircuitMonad<SingleRowIndicator>,
        #[strategy(arb())] circuit_1: ConstraintCircuitMonad<SingleRowIndicator>,
    ) {
        if hash_circuit(&circuit_0) != hash_circuit(&circuit_1) {
            prop_assert_ne!(circuit_0, circuit_1);
        }
    }

    /// The hash of a node may not depend on `ref_count`, `counter`,
    /// `id_counter_ref`, or `all_nodes`, since `all_nodes` contains the
    /// digest of all nodes in the multi tree. For more details, see
    /// [`HashSet`].
    #[proptest]
    fn multi_circuit_hash_is_unchanged_by_meta_data(
        #[strategy(arb())] circuit: ConstraintCircuitMonad<DualRowIndicator>,
        new_ref_count: usize,
        new_id_counter: usize,
    ) {
        let original_digest = hash_circuit(&circuit);

        circuit.circuit.borrow_mut().ref_count = new_ref_count;
        prop_assert_eq!(original_digest, hash_circuit(&circuit));

        circuit.builder.id_counter.replace(new_id_counter);
        prop_assert_eq!(original_digest, hash_circuit(&circuit));
    }

    fn hash_circuit<II: InputIndicator>(circuit: &ConstraintCircuitMonad<II>) -> u64 {
        let mut hasher = DefaultHasher::new();
        circuit.hash(&mut hasher);
        hasher.finish()
    }

    #[test]
    fn printing_constraint_circuit_gives_expected_strings() {
        let builder = ConstraintCircuitBuilder::new();
        assert_eq!("1", builder.b_constant(1).to_string());
        assert_eq!(
            "main_row[5] ",
            builder.input(SingleRowIndicator::Main(5)).to_string()
        );
        assert_eq!("6", builder.challenge(6_usize).to_string());

        let xfe_str = builder.x_constant([2, 3, 4]).to_string();
        assert_eq!("(4·x² + 3·x + 2)", xfe_str);
    }

    #[proptest]
    fn constant_folding_can_deal_with_multiplication_by_one(
        #[strategy(arb())] c: ConstraintCircuitMonad<DualRowIndicator>,
    ) {
        let one = || c.builder.one();
        prop_assert_eq!(c.clone(), c.clone() * one());
        prop_assert_eq!(c.clone(), one() * c.clone());
        prop_assert_eq!(c.clone(), one() * c.clone() * one());
    }

    #[proptest]
    fn constant_folding_can_deal_with_adding_zero(
        #[strategy(arb())] c: ConstraintCircuitMonad<DualRowIndicator>,
    ) {
        let zero = || c.builder.zero();
        prop_assert_eq!(c.clone(), c.clone() + zero());
        prop_assert_eq!(c.clone(), zero() + c.clone());
        prop_assert_eq!(c.clone(), zero() + c.clone() + zero());
    }

    #[proptest]
    fn constant_folding_can_deal_with_subtracting_zero(
        #[strategy(arb())] c: ConstraintCircuitMonad<DualRowIndicator>,
    ) {
        prop_assert_eq!(c.clone(), c.clone() - c.builder.zero());
    }

    #[proptest]
    fn constant_folding_can_deal_with_adding_effectively_zero_term(
        #[strategy(arb())] c: ConstraintCircuitMonad<DualRowIndicator>,
        modification_selectors: [bool; 4],
    ) {
        let zero = || c.builder.zero();
        let mut redundant_circuit = c.clone();
        if modification_selectors[0] {
            redundant_circuit = redundant_circuit + (c.clone() * zero());
        }
        if modification_selectors[1] {
            redundant_circuit = redundant_circuit + (zero() * c.clone());
        }
        if modification_selectors[2] {
            redundant_circuit = (c.clone() * zero()) + redundant_circuit;
        }
        if modification_selectors[3] {
            redundant_circuit = (zero() * c.clone()) + redundant_circuit;
        }
        prop_assert_eq!(c, redundant_circuit);
    }

    #[proptest]
    fn constant_folding_does_not_replace_0_minus_circuit_with_the_circuit(
        #[strategy(arb())] circuit: ConstraintCircuitMonad<DualRowIndicator>,
    ) {
        if circuit.circuit.borrow().is_zero() {
            return Err(TestCaseError::Reject("0 - 0 actually is 0".into()));
        }
        let zero_minus_circuit = circuit.builder.zero() - circuit.clone();
        prop_assert_ne!(&circuit, &zero_minus_circuit);
    }

    #[test]
    fn pointer_redirection_obliviates_a_node_in_a_circuit() {
        let builder = ConstraintCircuitBuilder::new();
        let x = |i| builder.input(SingleRowIndicator::Main(i));
        let constant = |c: u32| builder.b_constant(c);
        let challenge = |i: usize| builder.challenge(i);

        let part = x(0) + x(1);
        let substitute_me = x(0) * part.clone();
        let root_0 = part.clone() + challenge(1) - constant(84);
        let root_1 = substitute_me.clone() + challenge(0) - constant(42);
        let root_2 = x(2) * substitute_me.clone() - challenge(1);

        assert!(!root_0.contains(&substitute_me));
        assert!(root_1.contains(&substitute_me));
        assert!(root_2.contains(&substitute_me));

        let new_variable = x(3);
        builder.redirect_all_references_to_node(
            substitute_me.circuit.borrow().id,
            new_variable.clone(),
        );

        assert!(!root_0.contains(&substitute_me));
        assert!(!root_1.contains(&substitute_me));
        assert!(!root_2.contains(&substitute_me));

        assert!(root_0.contains(&part));
        assert!(root_1.contains(&new_variable));
        assert!(root_2.contains(&new_variable));
    }

    #[test]
    fn simple_degree_lowering() {
        let builder = ConstraintCircuitBuilder::new();
        let x = || builder.input(SingleRowIndicator::Main(0));
        let x_pow_3 = x() * x() * x();
        let x_pow_5 = x() * x() * x() * x() * x();
        let mut multicircuit = [x_pow_5, x_pow_3];

        let degree_lowering_info = DegreeLoweringInfo {
            target_degree: 3,
            num_main_cols: 1,
            num_aux_cols: 0,
        };
        let (new_main_constraints, new_aux_constraints) =
            ConstraintCircuitMonad::lower_to_degree(&mut multicircuit, degree_lowering_info);

        assert_eq!(1, new_main_constraints.len());
        assert!(new_aux_constraints.is_empty());
    }

    #[test]
    fn somewhat_simple_degree_lowering() {
        let builder = ConstraintCircuitBuilder::new();
        let x = |i| builder.input(SingleRowIndicator::Main(i));
        let y = |i| builder.input(SingleRowIndicator::Aux(i));
        let b_con = |i: u64| builder.b_constant(i);

        let constraint_0 = x(0) * x(0) * (x(1) - x(2)) - x(0) * x(2) - b_con(42);
        let constraint_1 = x(1) * (x(1) - b_con(5)) * x(2) * (x(2) - b_con(1));
        let constraint_2 = y(0)
            * (b_con(2) * x(0) + b_con(3) * x(1) + b_con(4) * x(2))
            * (b_con(4) * x(0) + b_con(8) * x(1) + b_con(16) * x(2))
            - y(1);

        let mut multicircuit = [constraint_0, constraint_1, constraint_2];

        let degree_lowering_info = DegreeLoweringInfo {
            target_degree: 2,
            num_main_cols: 3,
            num_aux_cols: 2,
        };
        let (new_main_constraints, new_aux_constraints) =
            ConstraintCircuitMonad::lower_to_degree(&mut multicircuit, degree_lowering_info);

        assert!(new_main_constraints.len() <= 3);
        assert!(new_aux_constraints.len() <= 1);
    }

    #[test]
    fn less_simple_degree_lowering() {
        let builder = ConstraintCircuitBuilder::new();
        let x = |i| builder.input(SingleRowIndicator::Main(i));

        let constraint_0 = (x(0) * x(1) * x(2)) * (x(3) * x(4)) * x(5);
        let constraint_1 = (x(6) * x(7)) * (x(3) * x(4)) * x(8);

        let mut multicircuit = [constraint_0, constraint_1];

        let degree_lowering_info = DegreeLoweringInfo {
            target_degree: 3,
            num_main_cols: 9,
            num_aux_cols: 0,
        };
        let (new_main_constraints, new_aux_constraints) =
            ConstraintCircuitMonad::lower_to_degree(&mut multicircuit, degree_lowering_info);

        assert!(new_main_constraints.len() <= 3);
        assert!(new_aux_constraints.is_empty());
    }

    fn circuit_with_multiple_options_for_degree_lowering_to_degree_2()
    -> [ConstraintCircuitMonad<SingleRowIndicator>; 2] {
        let builder = ConstraintCircuitBuilder::new();
        let x = |i| builder.input(SingleRowIndicator::Main(i));

        let constraint_0 = x(0) * x(0) * x(0);
        let constraint_1 = x(1) * x(1) * x(1);

        [constraint_0, constraint_1]
    }

    #[test]
    fn pick_node_to_substitute_is_deterministic() {
        let multicircuit = circuit_with_multiple_options_for_degree_lowering_to_degree_2();
        let first_node_id = ConstraintCircuitMonad::pick_node_to_substitute(&multicircuit, 2);

        for _ in 0..20 {
            let node_id_again = ConstraintCircuitMonad::pick_node_to_substitute(&multicircuit, 2);
            assert_eq!(first_node_id, node_id_again);
        }
    }

    #[test]
    fn degree_lowering_specific_simple_circuit_is_deterministic() {
        let degree_lowering_info = DegreeLoweringInfo {
            target_degree: 2,
            num_main_cols: 2,
            num_aux_cols: 0,
        };

        let mut original_multicircuit =
            circuit_with_multiple_options_for_degree_lowering_to_degree_2();
        let (new_main_constraints, _) = ConstraintCircuitMonad::lower_to_degree(
            &mut original_multicircuit,
            degree_lowering_info,
        );

        for _ in 0..20 {
            let mut new_multicircuit =
                circuit_with_multiple_options_for_degree_lowering_to_degree_2();
            let (new_main_constraints_again, _) = ConstraintCircuitMonad::lower_to_degree(
                &mut new_multicircuit,
                degree_lowering_info,
            );
            assert_eq!(new_main_constraints, new_main_constraints_again);
            assert_eq!(original_multicircuit, new_multicircuit);
        }
    }

    #[test]
    fn all_nodes_in_multicircuit_are_identified_correctly() {
        let builder = ConstraintCircuitBuilder::new();

        let x = |i| builder.input(SingleRowIndicator::Main(i));
        let b_con = |i: u64| builder.b_constant(i);

        let sub_tree_0 = x(0) * x(1) * (x(2) - b_con(1)) * x(3) * x(4);
        let sub_tree_1 = x(0) * x(1) * (x(2) - b_con(1)) * x(3) * x(5);
        let sub_tree_2 = x(10) * x(10) * x(2) * x(13);
        let sub_tree_3 = x(10) * x(10) * x(2) * x(14);

        let circuit_0 = sub_tree_0.clone() + sub_tree_1.clone();
        let circuit_1 = sub_tree_2.clone() + sub_tree_3.clone();
        let circuit_2 = sub_tree_0 + sub_tree_2;
        let circuit_3 = sub_tree_1 + sub_tree_3;

        let multicircuit = [circuit_0, circuit_1, circuit_2, circuit_3].map(|c| c.consume());

        let all_nodes = ConstraintCircuitMonad::all_nodes_in_multicircuit(&multicircuit);
        let count_node = |node| all_nodes.iter().filter(|&n| n == &node).count();

        let x0 = x(0).consume();
        assert_eq!(4, count_node(x0));

        let x2 = x(2).consume();
        assert_eq!(8, count_node(x2));

        let x10 = x(10).consume();
        assert_eq!(8, count_node(x10));

        let x4 = x(4).consume();
        assert_eq!(2, count_node(x4));

        let x6 = x(6).consume();
        assert_eq!(0, count_node(x6));

        let x0_x1 = (x(0) * x(1)).consume();
        assert_eq!(4, count_node(x0_x1));

        let tree = (x(0) * x(1) * (x(2) - b_con(1))).consume();
        assert_eq!(4, count_node(tree));

        let max_occurrences = all_nodes
            .iter()
            .map(|node| all_nodes.iter().filter(|&n| n == node).count())
            .max()
            .unwrap();
        assert_eq!(8, max_occurrences);

        let most_frequent_nodes = all_nodes
            .iter()
            .filter(|&node| all_nodes.iter().filter(|&n| n == node).count() == max_occurrences)
            .unique()
            .collect_vec();
        assert_eq!(2, most_frequent_nodes.len());
        assert!(most_frequent_nodes.contains(&&x(2).consume()));
        assert!(most_frequent_nodes.contains(&&x(10).consume()));
    }

    #[derive(Debug, Copy, Clone, Eq, PartialEq, test_strategy::Arbitrary)]
    enum CircuitOperationChoice {
        Add(usize, usize),
        Mul(usize, usize),
    }

    #[derive(Debug, Copy, Clone, Eq, PartialEq, test_strategy::Arbitrary)]
    enum CircuitInputType {
        Main,
        Aux,
    }

    #[derive(Debug, Copy, Clone, Eq, PartialEq, test_strategy::Arbitrary)]
    enum CircuitConstantType {
        Base(#[strategy(arb())] BFieldElement),
        Extension(#[strategy(arb())] XFieldElement),
    }

    fn arbitrary_circuit_monad<II: InputIndicator>(
        num_inputs: usize,
        num_challenges: usize,
        num_constants: usize,
        num_operations: usize,
        num_outputs: usize,
    ) -> BoxedStrategy<Vec<ConstraintCircuitMonad<II>>> {
        (
            vec(CircuitInputType::arbitrary(), num_inputs),
            vec(CircuitConstantType::arbitrary(), num_constants),
            vec(CircuitOperationChoice::arbitrary(), num_operations),
            vec(arb::<usize>(), num_outputs),
        )
            .prop_map(move |(inputs, constants, operations, outputs)| {
                let builder = ConstraintCircuitBuilder::<II>::new();

                assert_eq!(0, *builder.id_counter.borrow());
                assert!(
                    builder.all_nodes.borrow().is_empty(),
                    "fresh hashmap should be empty"
                );

                let mut num_main_inputs = 0;
                let mut num_aux_inputs = 0;
                let mut all_nodes = vec![];
                let mut output_nodes = vec![];

                for input in inputs {
                    match input {
                        CircuitInputType::Main => {
                            let node = builder.input(II::main_table_input(num_main_inputs));
                            num_main_inputs += 1;
                            all_nodes.push(node);
                        }
                        CircuitInputType::Aux => {
                            let node = builder.input(II::aux_table_input(num_aux_inputs));
                            num_aux_inputs += 1;
                            all_nodes.push(node);
                        }
                    }
                }

                for i in 0..num_challenges {
                    let node = builder.challenge(i);
                    all_nodes.push(node);
                }

                for constant in constants {
                    let node = match constant {
                        CircuitConstantType::Base(bfe) => builder.b_constant(bfe),
                        CircuitConstantType::Extension(xfe) => builder.x_constant(xfe),
                    };
                    all_nodes.push(node);
                }

                if all_nodes.is_empty() {
                    return vec![];
                }

                for operation in operations {
                    let (lhs, rhs) = match operation {
                        CircuitOperationChoice::Add(lhs, rhs) => (lhs, rhs),
                        CircuitOperationChoice::Mul(lhs, rhs) => (lhs, rhs),
                    };

                    let lhs_index = lhs % all_nodes.len();
                    let rhs_index = rhs % all_nodes.len();

                    let lhs_node = all_nodes[lhs_index].clone();
                    let rhs_node = all_nodes[rhs_index].clone();

                    let node = match operation {
                        CircuitOperationChoice::Add(_, _) => lhs_node + rhs_node,
                        CircuitOperationChoice::Mul(_, _) => lhs_node * rhs_node,
                    };
                    all_nodes.push(node);
                }

                for output in outputs {
                    let index = output % all_nodes.len();
                    output_nodes.push(all_nodes[index].clone());
                }

                output_nodes
            })
            .boxed()
    }

    #[proptest]
    fn node_type_counts_add_up(
        #[strategy(arbitrary_circuit_monad(10, 10, 10, 60, 10))] multicircuit_monad: Vec<
            ConstraintCircuitMonad<SingleRowIndicator>,
        >,
    ) {
        prop_assert_eq!(
            ConstraintCircuitMonad::num_nodes(&multicircuit_monad),
            ConstraintCircuitMonad::num_main_inputs(&multicircuit_monad)
                + ConstraintCircuitMonad::num_aux_inputs(&multicircuit_monad)
                + ConstraintCircuitMonad::num_challenges(&multicircuit_monad)
                + ConstraintCircuitMonad::num_bfield_constants(&multicircuit_monad)
                + ConstraintCircuitMonad::num_xfield_constants(&multicircuit_monad)
                + ConstraintCircuitMonad::num_binops(&multicircuit_monad)
        );

        prop_assert_eq!(10, ConstraintCircuitMonad::num_inputs(&multicircuit_monad));
    }

    /// Test the completeness and soundness of the `apply_substitution`
    /// function, which substitutes a single node.
    ///
    /// In this context, completeness means: "given a satisfying assignment to
    /// the circuit before degree lowering, one can derive a satisfying
    /// assignment to the circuit after degree lowering." Soundness means
    /// the converse.
    ///
    /// We test these features using random input vectors. Naturally, the output
    /// is not the zero vector (with high probability) and so the given input is
    /// *not* a satisfying assignment (with high probability). However, the
    /// circuit can be extended by way of thought experiment into one that
    /// subtracts a fixed constant from the original output. For the right
    /// choice of subtrahend, the random input now *is* a satisfying
    /// assignment to the circuit.
    ///
    /// Specifically, let `input` denote the original (before degree lowering)
    /// input, and `C` the circuit. Then `input` is a satisfying input for
    /// the new circuit `C'(X) = C(X) - C(input)`
    ///
    /// After applying a substitution to obtain circuit `C || k` from `C`, where
    /// `k = Z - some_expr(X)` and `Z` is the introduced variable, the length
    /// of the input and output increases by 1. Moreover, if `input` is a
    /// satisfying input to `C'` then `input || some_expr(input)` is* a
    /// satisfying input to `C' || k`.
    ///
    /// (*: If the transformation is complete.)
    ///
    /// To establish the converse, we want to start from a satisfying input to
    /// `C" || k` and reduce it to a satisfying input to `C"`. The requirement,
    /// implied by "satisfying input", that `k(X || Z) == 0` implies `Z ==
    /// some_expr(X)`. Therefore, in order to sample a random satisfying
    /// input to `C" || k`, it suffices to select `input` at random, define
    /// `C"(X) = C(X) - C(input)`, and evaluate `some_expr(input)`. Then
    /// `input || some_expr(input)` is a random satisfying input to `C" ||
    /// k`. It follows** that `input` is a satisfying input to `C"`.
    ///
    /// (**: If the transformation is sound.)
    ///
    /// This description makes use of the following commutative diagram.
    ///
    /// ```text
    ///          C ───── degree-lowering ────> C || k
    ///          │                               │
    /// subtract │                      subtract │
    ///    fixed │                         fixed │
    ///   output │                        output │
    ///          │                               │
    ///          v                               v
    ///          C* ─── degree-lowering ────> C* || k
    /// ```
    ///
    /// The point of this elaboration is that in this particular case, testing
    /// completeness and soundness require the same code path. If `input`
    /// and `input || some_expr(input)` work for circuits before and after
    /// degree lowering, this fact establishes both completeness and
    /// soundness simultaneously.
    //
    // Shrinking on this test is disabled because we noticed some weird ass
    // behavior. In short, shrinking does not play ball with the arbitrary
    // circuit generator; it seems to make the generated circuits *more*
    // complex, not less so.
    #[proptest(cases = 1000, max_shrink_iters = 0)]
    fn node_substitution_is_complete_and_sound(
        #[strategy(arbitrary_circuit_monad(10, 10, 10, 160, 10))] mut multicircuit_monad: Vec<
            ConstraintCircuitMonad<SingleRowIndicator>,
        >,
        #[strategy(vec(arb(), ConstraintCircuitMonad::num_main_inputs(&#multicircuit_monad)))]
        #[filter(!#main_input.is_empty())]
        main_input: Vec<BFieldElement>,
        #[strategy(vec(arb(), ConstraintCircuitMonad::num_aux_inputs(&#multicircuit_monad)))]
        #[filter(!#aux_input.is_empty())]
        aux_input: Vec<XFieldElement>,
        #[strategy(vec(arb(), ConstraintCircuitMonad::num_challenges(&#multicircuit_monad)))]
        challenges: Vec<XFieldElement>,
        #[strategy(arb())] substitution_node_index: usize,
    ) {
        let mut main_input = Array2::from_shape_vec((1, main_input.len()), main_input).unwrap();
        let mut aux_input = Array2::from_shape_vec((1, aux_input.len()), aux_input).unwrap();

        let output_before_lowering = multicircuit_monad
            .iter()
            .map(|m| m.circuit.borrow())
            .map(|c| c.evaluate(main_input.view(), aux_input.view(), &challenges))
            .collect_vec();

        // apply one step of degree-lowering
        let num_nodes = ConstraintCircuitMonad::num_nodes(&multicircuit_monad);
        let &substitution_node_id = multicircuit_monad[0]
            .builder
            .all_nodes
            .borrow()
            .iter()
            .cycle()
            .skip(substitution_node_index % num_nodes)
            .take(num_nodes)
            .find_map(|(id, monad)| monad.circuit.borrow().is_zero().not().then_some(id))
            .expect("no suitable nodes to substitute");

        let degree_lowering_info = DegreeLoweringInfo {
            target_degree: 2,
            num_main_cols: main_input.len(),
            num_aux_cols: aux_input.len(),
        };
        let substitution_constraint = ConstraintCircuitMonad::apply_substitution(
            &mut multicircuit_monad,
            degree_lowering_info,
            substitution_node_id,
            EvolvingMainConstraintsNumber(0),
            EvolvingAuxConstraintsNumber(0),
        );

        // extract substituted constraint
        let CircuitExpression::BinOp(BinOp::Add, variable, neg_expression) =
            &substitution_constraint.circuit.borrow().expression
        else {
            panic!();
        };
        let extra_input =
            match &neg_expression.borrow().expression {
                CircuitExpression::BinOp(BinOp::Mul, _neg_one, circuit) => circuit
                    .borrow()
                    .evaluate(main_input.view(), aux_input.view(), &challenges),
                CircuitExpression::BConst(c) => -c.lift(),
                CircuitExpression::XConst(c) => -*c,
                _ => panic!(),
            };
        if variable.borrow().evaluates_to_base_element() {
            let extra_input = extra_input.unlift().unwrap();
            let extra_input = Array2::from_shape_vec([1, 1], vec![extra_input]).unwrap();
            main_input.append(Axis(1), extra_input.view()).unwrap();
        } else {
            let extra_input = Array2::from_shape_vec([1, 1], vec![extra_input]).unwrap();
            aux_input.append(Axis(1), extra_input.view()).unwrap();
        }

        // evaluate again
        let output_after_lowering = multicircuit_monad
            .iter()
            .map(|m| m.circuit.borrow())
            .map(|c| c.evaluate(main_input.view(), aux_input.view(), &challenges))
            .collect_vec();
        prop_assert_eq!(output_before_lowering, output_after_lowering);

        let evaluated_constraint = substitution_constraint.circuit.borrow().evaluate(
            main_input.view(),
            aux_input.view(),
            &challenges,
        );
        prop_assert!(evaluated_constraint.is_zero());
    }
}
