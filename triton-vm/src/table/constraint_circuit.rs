//! Constraint circuits are a way to represent constraint polynomials in a way that is amenable
//! to optimizations. The constraint circuit is a directed acyclic graph (DAG) of
//! [`CircuitExpression`]s, where each `CircuitExpression` is a node in the graph. The edges of the
//! graph are labeled with [`BinOp`]s. The leafs of the graph are the inputs to the constraint
//! polynomial, and the (multiple) roots of the graph are the outputs of all the
//! constraint polynomials, with each root corresponding to a different constraint polynomial.
//! Because the graph has multiple roots, it is called a “multitree.”

use std::cell::RefCell;
use std::cmp;
use std::collections::HashMap;
use std::collections::HashSet;
use std::fmt::Debug;
use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FmtResult;
use std::hash::Hash;
use std::hash::Hasher;
use std::iter::Sum;
use std::ops::Add;
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
use quote::quote;
use quote::ToTokens;
use rand::thread_rng;
use rand::Rng;
use twenty_first::prelude::*;

use CircuitExpression::*;

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
/// Having `Copy + Hash + Eq` helps to put `InputIndicator`s into containers.
pub trait InputIndicator: Debug + Display + Copy + Hash + Eq + ToTokens {
    /// `true` iff `self` refers to a column in the base table.
    fn is_base_table_column(&self) -> bool;

    /// `true` iff `self` refers to the current row.
    fn is_current_row(&self) -> bool;

    /// The index of the indicated (base or extension) column.
    fn column(&self) -> usize;

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
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum SingleRowIndicator {
    BaseRow(usize),
    ExtRow(usize),
}

impl Display for SingleRowIndicator {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let input_indicator: String = match self {
            Self::BaseRow(i) => format!("base_row[{i}]"),
            Self::ExtRow(i) => format!("ext_row[{i}]"),
        };

        write!(f, "{input_indicator}")
    }
}

impl ToTokens for SingleRowIndicator {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        match self {
            Self::BaseRow(i) => tokens.extend(quote!(base_row[#i])),
            Self::ExtRow(i) => tokens.extend(quote!(ext_row[#i])),
        }
    }
}

impl InputIndicator for SingleRowIndicator {
    fn is_base_table_column(&self) -> bool {
        matches!(self, Self::BaseRow(_))
    }

    fn is_current_row(&self) -> bool {
        true
    }

    fn column(&self) -> usize {
        match self {
            Self::BaseRow(i) | Self::ExtRow(i) => *i,
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
        match self {
            Self::BaseRow(i) => base_table[[0, *i]].lift(),
            Self::ExtRow(i) => ext_table[[0, *i]],
        }
    }
}

/// The position of a variable in a constraint polynomial that operates on two rows (current and
/// next) of the execution trace.
#[derive(Debug, Copy, Clone, Eq, PartialEq, Hash)]
pub enum DualRowIndicator {
    CurrentBaseRow(usize),
    CurrentExtRow(usize),
    NextBaseRow(usize),
    NextExtRow(usize),
}

impl Display for DualRowIndicator {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let input_indicator: String = match self {
            Self::CurrentBaseRow(i) => format!("current_base_row[{i}]"),
            Self::CurrentExtRow(i) => format!("current_ext_row[{i}]"),
            Self::NextBaseRow(i) => format!("next_base_row[{i}]"),
            Self::NextExtRow(i) => format!("next_ext_row[{i}]"),
        };

        write!(f, "{input_indicator}")
    }
}

impl ToTokens for DualRowIndicator {
    fn to_tokens(&self, tokens: &mut proc_macro2::TokenStream) {
        match self {
            Self::CurrentBaseRow(i) => tokens.extend(quote!(current_base_row[#i])),
            Self::CurrentExtRow(i) => tokens.extend(quote!(current_ext_row[#i])),
            Self::NextBaseRow(i) => tokens.extend(quote!(next_base_row[#i])),
            Self::NextExtRow(i) => tokens.extend(quote!(next_ext_row[#i])),
        }
    }
}

impl InputIndicator for DualRowIndicator {
    fn is_base_table_column(&self) -> bool {
        matches!(self, Self::CurrentBaseRow(_) | Self::NextBaseRow(_))
    }

    fn is_current_row(&self) -> bool {
        matches!(self, Self::CurrentBaseRow(_) | Self::CurrentExtRow(_))
    }

    fn column(&self) -> usize {
        match self {
            Self::CurrentBaseRow(i)
            | Self::NextBaseRow(i)
            | Self::CurrentExtRow(i)
            | Self::NextExtRow(i) => *i,
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
        match self {
            Self::CurrentBaseRow(i) => base_table[[0, *i]].lift(),
            Self::CurrentExtRow(i) => ext_table[[0, *i]],
            Self::NextBaseRow(i) => base_table[[1, *i]].lift(),
            Self::NextExtRow(i) => ext_table[[1, *i]],
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
    BConstant(BFieldElement),
    XConstant(XFieldElement),
    Input(II),
    Challenge(usize),
    BinaryOperation(
        BinOp,
        Rc<RefCell<ConstraintCircuit<II>>>,
        Rc<RefCell<ConstraintCircuit<II>>>,
    ),
}

impl<II: InputIndicator> Hash for CircuitExpression<II> {
    fn hash<H: Hasher>(&self, state: &mut H) {
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
                lhs.borrow().hash(state);
                rhs.borrow().hash(state);
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
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.expression.hash(state)
    }
}

impl<II: InputIndicator> Hash for ConstraintCircuitMonad<II> {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.circuit.borrow().hash(state)
    }
}

/// A wrapper around a [`CircuitExpression`] that manages additional bookkeeping information.
#[derive(Debug, Clone)]
pub struct ConstraintCircuit<II: InputIndicator> {
    pub id: usize,
    pub ref_count: usize,
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
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        match &self.expression {
            XConstant(xfe) => write!(f, "{xfe}"),
            BConstant(bfe) => write!(f, "{bfe}"),
            Input(input) => write!(f, "{input} "),
            Challenge(self_challenge_idx) => write!(f, "{self_challenge_idx}"),
            BinaryOperation(operation, lhs, rhs) => {
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

        if let BinaryOperation(_, lhs, rhs) = &self.expression {
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

        if let BinaryOperation(_, lhs, rhs) = &self.expression {
            lhs.borrow_mut().assert_unique_ids_inner(ids);
            rhs.borrow_mut().assert_unique_ids_inner(ids);
        }
    }

    /// Assert that a multicircuit has unique IDs.
    /// Also determines how often each node is referenced, updating the respective `ref_count`s.
    ///
    /// # Panics
    ///
    /// Panics if a duplicate ID is found.
    pub fn assert_unique_ids(constraints: &mut [ConstraintCircuit<II>]) {
        // inner uniqueness checks relies on reference counters being 0 for unseen nodes
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
            BinaryOperation(binop, lhs, rhs) => {
                let degree_lhs = lhs.borrow().degree();
                let degree_rhs = rhs.borrow().degree();
                let degree_additive = cmp::max(degree_lhs, degree_rhs);
                let degree_multiplicative = match degree_lhs == -1 || degree_rhs == -1 {
                    true => -1,
                    false => degree_lhs + degree_rhs,
                };
                match binop {
                    BinOp::Add => degree_additive,
                    BinOp::Mul => degree_multiplicative,
                }
            }
            Input(_) => 1,
            BConstant(_) | XConstant(_) | Challenge(_) => 0,
        }
    }

    /// All unique reference counters in the subtree, sorted.
    pub fn all_ref_counters(&self) -> Vec<usize> {
        let mut ref_counters = vec![self.ref_count];
        if let BinaryOperation(_, lhs, rhs) = &self.expression {
            ref_counters.extend(lhs.borrow().all_ref_counters());
            ref_counters.extend(rhs.borrow().all_ref_counters());
        };
        ref_counters.sort_unstable();
        ref_counters.dedup();
        ref_counters
    }

    /// Is the node the constant 0?
    /// Does not catch composite expressions that will always evaluate to zero, like `0·a`.
    pub fn is_zero(&self) -> bool {
        match self.expression {
            BConstant(bfe) => bfe.is_zero(),
            XConstant(xfe) => xfe.is_zero(),
            _ => false,
        }
    }

    /// Is the node the constant 1?
    /// Does not catch composite expressions that will always evaluate to one, like `1·1`.
    pub fn is_one(&self) -> bool {
        match self.expression {
            BConstant(bfe) => bfe.is_one(),
            XConstant(xfe) => xfe.is_one(),
            _ => false,
        }
    }

    pub fn is_neg_one(&self) -> bool {
        match self.expression {
            BConstant(bfe) => (-bfe).is_one(),
            XConstant(xfe) => (-xfe).is_one(),
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
                lhs.borrow().evaluates_to_base_element() && rhs.borrow().evaluates_to_base_element()
            }
        }
    }

    pub fn evaluate(
        &self,
        base_table: ArrayView2<BFieldElement>,
        ext_table: ArrayView2<XFieldElement>,
        challenges: &[XFieldElement],
    ) -> XFieldElement {
        match &self.expression {
            BConstant(bfe) => bfe.lift(),
            XConstant(xfe) => *xfe,
            Input(input) => input.evaluate(base_table, ext_table),
            Challenge(challenge_id) => challenges[*challenge_id],
            BinaryOperation(binop, lhs, rhs) => {
                let lhs_value = lhs.borrow().evaluate(base_table, ext_table, challenges);
                let rhs_value = rhs.borrow().evaluate(base_table, ext_table, challenges);
                binop.operation(lhs_value, rhs_value)
            }
        }
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
    // `all_nodes` contains itself, leading to infinite recursion during `Debug` printing.
    // Hence, this manual implementation.
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
        (&BConstant(l), &BConstant(r)) => return lhs.builder.b_constant(binop.operation(l, r)),
        (&BConstant(l), &XConstant(r)) => return lhs.builder.x_constant(binop.operation(l, r)),
        (&XConstant(l), &BConstant(r)) => return lhs.builder.x_constant(binop.operation(l, r)),
        (&XConstant(l), &XConstant(r)) => return lhs.builder.x_constant(binop.operation(l, r)),
        _ => (),
    };

    // all `BinOp`s are commutative – try both orders of the operands
    let new_node = binop_new_node(binop, &rhs, &lhs);
    if let Some(node) = lhs.builder.all_nodes.borrow().get(&new_node) {
        return node.to_owned();
    }

    let new_node = binop_new_node(binop, &lhs, &rhs);
    if let Some(node) = lhs.builder.all_nodes.borrow().get(&new_node) {
        return node.to_owned();
    }

    *lhs.builder.id_counter.borrow_mut() += 1;
    let was_inserted = lhs.builder.all_nodes.borrow_mut().insert(new_node.clone());
    assert!(was_inserted, "Binop-created value must be new");
    new_node
}

fn binop_new_node<II: InputIndicator>(
    binop: BinOp,
    lhs: &ConstraintCircuitMonad<II>,
    rhs: &ConstraintCircuitMonad<II>,
) -> ConstraintCircuitMonad<II> {
    let id = lhs.builder.id_counter.borrow().to_owned();
    let expression = BinaryOperation(binop, lhs.circuit.clone(), rhs.circuit.clone());
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
    pub fn consume(&self) -> ConstraintCircuit<II> {
        self.circuit.borrow().to_owned()
    }

    /// Traverse the circuit and find all nodes that are equivalent. Note that
    /// two nodes are equivalent if they compute the same value on all identical
    /// inputs. Equivalence is different from identity, which is when two nodes
    /// connect the same set of neighbors in the same way. (There may be two
    /// different ways to compute the same result; they are equivalent but
    /// unequal.)
    ///
    /// This function returns a list of lists of equivalent nodes such that
    /// every inner list can be reduced to a single node without changing the
    /// circuit's function.
    ///
    /// Equivalent nodes are detected probabilistically using the multivariate
    /// Schwartz-Zippel lemma. The false positive probability is zero (we can be
    /// certain that equivalent nodes will be found). The false negative
    /// probability is bounded by max_degree / (2^64 - 2^32 + 1)^3.
    pub fn find_equivalent_nodes(&self) -> Vec<Vec<Rc<RefCell<ConstraintCircuit<II>>>>> {
        let mut values: HashMap<usize, XFieldElement> = HashMap::new();
        let mut ids: HashMap<XFieldElement, Vec<usize>> = HashMap::new();
        let mut nodes: HashMap<usize, Rc<RefCell<ConstraintCircuit<II>>>> = HashMap::new();
        let seed: [u8; 32] = thread_rng().gen();
        Self::probe_random(
            self.circuit.clone(),
            &mut values,
            &mut ids,
            &mut nodes,
            seed,
        );

        ids.values()
            .filter(|l| l.len() >= 2)
            .cloned()
            .map(|l| l.iter().map(|i| nodes[i].clone()).collect_vec())
            .collect_vec()
    }

    /// Populate the dictionaries such that they associate with every node in
    /// the circuit its evaluation in a random point. The inputs are assigned
    /// random values. Equivalent nodes are detected based on evaluating to the
    /// same value using the Schwartz-Zippel lemma.
    fn probe_random(
        circuit: Rc<RefCell<ConstraintCircuit<II>>>,
        values: &mut HashMap<usize, XFieldElement>,
        ids: &mut HashMap<XFieldElement, Vec<usize>>,
        nodes: &mut HashMap<usize, Rc<RefCell<ConstraintCircuit<II>>>>,
        master_seed: [u8; 32],
    ) {
        // the node was already touched; nothing to do
        if values.contains_key(&circuit.borrow().id) {
            return;
        }

        // compute the node's value; recurse if necessary
        let value = match &circuit.borrow().expression {
            BConstant(bfe) => bfe.lift(),
            XConstant(xfe) => *xfe,
            Input(input) => {
                let mut hasher = blake3::Hasher::new();
                hasher.update(&master_seed);
                hasher.update(b"input");
                hasher.update(&usize::from(input.is_base_table_column()).to_ne_bytes());
                hasher.update(&input.column().to_ne_bytes());
                let mut output = [0u8; 24];
                hasher.finalize_xof().fill(&mut output);
                let x0 = BFieldElement::from_ne_bytes(&output[0..8]);
                let x1 = BFieldElement::from_ne_bytes(&output[8..16]);
                let x2 = BFieldElement::from_ne_bytes(&output[16..24]);

                XFieldElement::new([x0, x1, x2])
            }
            Challenge(challenge) => {
                let mut hasher = blake3::Hasher::new();
                hasher.update(&master_seed);
                hasher.update(b"challenge");
                hasher.update(&challenge.to_ne_bytes());
                let mut output = [0u8; 24];
                hasher.finalize_xof().fill(&mut output);
                let x0 = BFieldElement::from_ne_bytes(&output[0..8]);
                let x1 = BFieldElement::from_ne_bytes(&output[8..16]);
                let x2 = BFieldElement::from_ne_bytes(&output[16..24]);

                XFieldElement::new([x0, x1, x2])
            }
            BinaryOperation(operation, lhs, rhs) => {
                // if lhs or rhs wasn't touched yet, recurse
                if !values.contains_key(&lhs.borrow().id) {
                    Self::probe_random(lhs.clone(), values, ids, nodes, master_seed);
                }
                if !values.contains_key(&rhs.borrow().id) {
                    Self::probe_random(rhs.clone(), values, ids, nodes, master_seed);
                }

                // lookup values
                let lhs_value = *values.get(&lhs.borrow().id).unwrap();
                let rhs_value = *values.get(&rhs.borrow().id).unwrap();

                // combine using appropriate operator
                match operation {
                    BinOp::Add => lhs_value + rhs_value,
                    BinOp::Mul => lhs_value * rhs_value,
                }
            }
        };

        // value already exists; keep books
        if let Some(peers) = ids.get_mut(&value) {
            values.insert(circuit.borrow().id, value);
            peers.push(circuit.borrow().id);
            nodes.insert(circuit.borrow().id, circuit.clone());
        }
        // value is new; keep books
        else {
            values.insert(circuit.borrow().id, value);
            ids.insert(value, vec![circuit.borrow().id]);
            nodes.insert(circuit.borrow().id, circuit.clone());
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
        target_degree: isize,
        num_base_cols: usize,
        num_ext_cols: usize,
    ) -> (Vec<Self>, Vec<Self>) {
        assert!(
            target_degree > 1,
            "Target degree must be greater than 1. Got {target_degree}."
        );

        let mut base_constraints = vec![];
        let mut ext_constraints = vec![];

        if multicircuit.is_empty() {
            return (base_constraints, ext_constraints);
        }

        let builder = multicircuit[0].builder.clone();

        while Self::multicircuit_degree(multicircuit) > target_degree {
            let chosen_node_id = Self::pick_node_to_substitute(multicircuit, target_degree);

            // Create a new variable.
            let chosen_node = builder.get_node_by_id(chosen_node_id).unwrap();
            let chosen_node_is_base_col = chosen_node.circuit.borrow().evaluates_to_base_element();
            let new_input_indicator = if chosen_node_is_base_col {
                let new_base_col_idx = num_base_cols + base_constraints.len();
                II::base_table_input(new_base_col_idx)
            } else {
                let new_ext_col_idx = num_ext_cols + ext_constraints.len();
                II::ext_table_input(new_ext_col_idx)
            };
            let new_variable = builder.input(new_input_indicator);
            let new_circuit = new_variable.circuit.clone();

            // Substitute the chosen circuit with the new variable.
            builder.substitute(chosen_node_id, &new_circuit);

            // Create new constraint and put it into the appropriate return vector.
            let new_constraint = new_variable - chosen_node;
            match chosen_node_is_base_col {
                true => base_constraints.push(new_constraint),
                false => ext_constraints.push(new_constraint),
            }

            // Treat roots of the multicircuit explicitly.
            for circuit in multicircuit.iter_mut() {
                if circuit.circuit.borrow().id == chosen_node_id {
                    circuit.circuit = new_circuit.clone();
                }
            }
        }

        (base_constraints, ext_constraints)
    }

    /// Heuristically pick a node from the given multicircuit that is to be substituted with a new
    /// variable. The ID of the chosen node is returned.
    fn pick_node_to_substitute(
        multicircuit: &[ConstraintCircuitMonad<II>],
        target_degree: isize,
    ) -> usize {
        assert!(!multicircuit.is_empty());

        let multicircuit = multicircuit
            .iter()
            .map(|c| c.clone().consume())
            .collect_vec();
        let all_nodes = Self::all_nodes_in_multicircuit(&multicircuit);
        let all_nodes: HashSet<_> = HashSet::from_iter(all_nodes);

        // Only nodes with degree > target_degree need changing.
        let high_degree_nodes = all_nodes
            .into_iter()
            .filter(|node| node.degree() > target_degree)
            .collect_vec();

        // Collect all candidates for substitution, i.e., descendents of high_degree_nodes
        // with degree <= target_degree.
        // Substituting a node of degree 1 is both pointless and can lead to infinite iteration.
        let low_degree_nodes = Self::all_nodes_in_multicircuit(&high_degree_nodes)
            .into_iter()
            .filter(|node| 1 < node.degree() && node.degree() <= target_degree)
            .collect_vec();

        // If the resulting list is empty, there is no way forward. Stop – panic time!
        assert!(!low_degree_nodes.is_empty(), "Cannot lower degree.");

        // Of the remaining nodes, keep the ones occurring the most often.
        let mut nodes_and_occurrences = HashMap::new();
        for node in &low_degree_nodes {
            *nodes_and_occurrences.entry(node).or_insert(0) += 1;
        }
        let max_occurrences = nodes_and_occurrences.iter().map(|(_, &c)| c).max().unwrap();
        nodes_and_occurrences.retain(|_, &mut count| count == max_occurrences);
        let mut candidate_nodes = nodes_and_occurrences.keys().copied().collect_vec();

        // If there are still multiple nodes, pick the one with the highest degree.
        let max_degree = candidate_nodes.iter().map(|n| n.degree()).max().unwrap();
        candidate_nodes.retain(|node| node.degree() == max_degree);

        // If there are still multiple nodes, pick any one – but deterministically so.
        candidate_nodes.sort_by_key(|node| node.id);
        candidate_nodes[0].id
    }

    /// Returns all nodes used in the multicircuit.
    /// This is distinct from `ConstraintCircuitBuilder::all_nodes` because it
    /// 1. only considers nodes used in the given multicircuit, not all nodes in the builder,
    /// 2. returns the nodes as [`ConstraintCircuit`]s, not as [`ConstraintCircuitMonad`]s, and
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
        if let BinaryOperation(_, lhs, rhs) = circuit.expression.clone() {
            let lhs_nodes = Self::all_nodes_in_circuit(&lhs.borrow());
            let rhs_nodes = Self::all_nodes_in_circuit(&rhs.borrow());
            all_nodes.extend(lhs_nodes);
            all_nodes.extend(rhs_nodes);
        };
        all_nodes.push(circuit.to_owned());
        all_nodes
    }

    /// Returns the maximum degree of all circuits in the multicircuit.
    fn multicircuit_degree(multicircuit: &[ConstraintCircuitMonad<II>]) -> isize {
        multicircuit
            .iter()
            .map(|circuit| circuit.circuit.borrow().degree())
            .max()
            .unwrap_or(-1)
    }
}

#[derive(Debug, Clone, Eq, PartialEq)]
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

    /// Check whether two builders are the same.
    ///
    /// Notably, this is distinct from checking equality: two builders are equal if they are in the
    /// same state. Two builders are the same if they are the same instance.
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

    pub fn get_node_by_id(&self, id: usize) -> Option<ConstraintCircuitMonad<II>> {
        self.all_nodes
            .borrow()
            .iter()
            .find(|node| node.circuit.borrow().id == id)
            .cloned()
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
        self.make_leaf(BConstant(bfe.into()))
    }

    /// Leaf node with constant over the [extension field][XFieldElement].
    pub fn x_constant<X>(&self, xfe: X) -> ConstraintCircuitMonad<II>
    where
        X: Into<XFieldElement>,
    {
        self.make_leaf(XConstant(xfe.into()))
    }

    /// Create deterministic input leaf node.
    pub fn input(&self, input: II) -> ConstraintCircuitMonad<II> {
        self.make_leaf(Input(input))
    }

    /// Create challenge leaf node.
    pub fn challenge<C>(&self, challenge: C) -> ConstraintCircuitMonad<II>
    where
        C: Into<usize>,
    {
        self.make_leaf(Challenge(challenge.into()))
    }

    fn make_leaf(&self, mut expression: CircuitExpression<II>) -> ConstraintCircuitMonad<II> {
        // Don't generate an X field leaf if it can be expressed as a B field leaf
        if let XConstant(xfe) = expression {
            if let Some(bfe) = xfe.unlift() {
                expression = BConstant(bfe);
            }
        }

        let id = self.id_counter.borrow().to_owned();
        let circuit = ConstraintCircuit::new(id, expression);
        let new_node = self.new_monad(circuit);

        if let Some(same_node) = self.all_nodes.borrow().get(&new_node) {
            return same_node.to_owned();
        }

        *self.id_counter.borrow_mut() += 1;
        let was_inserted = self.all_nodes.borrow_mut().insert(new_node.clone());
        assert!(was_inserted, "Leaf-created value must be new… {new_node}");
        new_node
    }

    /// Substitute all nodes with ID `old_id` with the given `new` node.
    pub fn substitute(&self, old_id: usize, new: &Rc<RefCell<ConstraintCircuit<II>>>) {
        for node in self.all_nodes.borrow().iter() {
            if node.circuit.borrow().id == old_id {
                continue;
            }

            let BinaryOperation(_, ref mut lhs, ref mut rhs) = node.circuit.borrow_mut().expression
            else {
                continue;
            };

            if lhs.borrow().id == old_id {
                *lhs = new.clone();
            }
            if rhs.borrow().id == old_id {
                *rhs = new.clone();
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

impl<'a> Arbitrary<'a> for SingleRowIndicator {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let col_idx = u.arbitrary()?;
        let indicator = match u.arbitrary()? {
            true => Self::BaseRow(col_idx),
            false => Self::ExtRow(col_idx),
        };
        Ok(indicator)
    }
}

impl<'a> Arbitrary<'a> for DualRowIndicator {
    fn arbitrary(u: &mut Unstructured<'a>) -> arbitrary::Result<Self> {
        let col_idx = u.arbitrary()?;
        let indicator = match u.int_in_range(0..=3)? {
            0 => Self::CurrentBaseRow(col_idx),
            1 => Self::CurrentExtRow(col_idx),
            2 => Self::NextBaseRow(col_idx),
            3 => Self::NextExtRow(col_idx),
            _ => unreachable!(),
        };
        Ok(indicator)
    }
}

#[cfg(test)]
mod tests {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;

    use itertools::Itertools;
    use ndarray::Array2;
    use proptest::prelude::*;
    use proptest_arbitrary_interop::arb;
    use rand::random;
    use rand::rngs::StdRng;
    use rand::Rng;
    use rand::SeedableRng;
    use test_strategy::proptest;

    use crate::table::cascade_table::ExtCascadeTable;
    use crate::table::challenges::Challenges;
    use crate::table::constraint_circuit::SingleRowIndicator::*;
    use crate::table::degree_lowering_table::DegreeLoweringTable;
    use crate::table::hash_table::ExtHashTable;
    use crate::table::jump_stack_table::ExtJumpStackTable;
    use crate::table::lookup_table::ExtLookupTable;
    use crate::table::master_table::*;
    use crate::table::op_stack_table::ExtOpStackTable;
    use crate::table::processor_table::ExtProcessorTable;
    use crate::table::program_table::ExtProgramTable;
    use crate::table::ram_table::ExtRamTable;
    use crate::table::u32_table::ExtU32Table;
    use crate::Claim;

    use super::*;

    /// Circuit monads are put into hash sets. Hence, it is important that `Eq` and `Hash`
    /// agree whether two nodes are equal: k1 == k2 => h(k1) == h(k2)
    #[proptest]
    fn equality_and_hash_agree(
        #[strategy(arb())] circuit: ConstraintCircuitMonad<SingleRowIndicator>,
    ) {
        let hash0 = hash_circuit(&circuit);
        let other_circuit = circuit.clone() + circuit.builder.zero();
        let hash1 = hash_circuit(&other_circuit);
        prop_assert_eq!(circuit == other_circuit, hash0 == hash1);
    }

    /// The hash of a node may not depend on `ref_count`, `counter`, `id_counter_ref`, or
    /// `all_nodes`, since `all_nodes` contains the digest of all nodes in the multi tree.
    /// For more details, see [`HashSet`].
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

        let xfe_str = builder.x_constant([2, 3, 4]).to_string();
        assert_eq!("(4·x² + 3·x + 2)", xfe_str);
        assert_eq!("base_row[5] ", builder.input(BaseRow(5)).to_string());
        assert_eq!("6", builder.challenge(6_usize).to_string());
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

    /// Terribly confusing, super rare bug that's extremely difficult to reproduce or pin down:
    /// 1. apply constant folding
    /// 1. introduce a new redundant circuit,
    /// 1. apply constant folding again.
    ///
    /// As a workaround, only _one_ redundant circuit is produced below.
    ///
    /// If you, dear reader, feel like diving into a rabbit hole of confusion and frustration,
    /// try checking the constant-folding property of all 4 possible combinations in the same test.
    #[proptest]
    fn constant_folding_can_deal_with_adding_effectively_zero_term(
        #[strategy(arb())] c: ConstraintCircuitMonad<DualRowIndicator>,
        #[strategy(0_usize..4)] test_case: usize,
    ) {
        let zero = || c.builder.zero();
        let redundant_circuit = match test_case {
            0 => c.clone() + (c.clone() * zero()),
            1 => c.clone() + (zero() * c.clone()),
            2 => (c.clone() * zero()) + c.clone(),
            3 => (zero() * c.clone()) + c.clone(),
            _ => unreachable!(),
        };

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
        challenges: &[XFieldElement],
        base_rows: ArrayView2<BFieldElement>,
        ext_rows: ArrayView2<XFieldElement>,
        values: &mut HashMap<XFieldElement, (usize, ConstraintCircuit<II>)>,
    ) -> XFieldElement {
        let value = match &constraint.expression {
            BinaryOperation(binop, lhs, rhs) => {
                let lhs = lhs.borrow();
                let rhs = rhs.borrow();
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

        let dummy_claim = Claim::default();
        let challenges: [XFieldElement; Challenges::SAMPLE_COUNT] = rng.gen();
        let challenges = challenges.to_vec();
        let challenges = Challenges::new(challenges, &dummy_claim);
        let challenges = &challenges.challenges;

        let num_rows = 2;
        let base_shape = [num_rows, NUM_BASE_COLUMNS];
        let ext_shape = [num_rows, NUM_EXT_COLUMNS];
        let base_rows = Array2::from_shape_simple_fn(base_shape, || rng.gen::<BFieldElement>());
        let ext_rows = Array2::from_shape_simple_fn(ext_shape, || rng.gen::<XFieldElement>());
        let base_rows = base_rows.view();
        let ext_rows = ext_rows.view();

        let mut values = HashMap::new();
        for c in constraints {
            evaluate_assert_unique(c, challenges, base_rows, ext_rows, &mut values);
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
        ConstraintCircuit::assert_unique_ids(&mut constraints);
        constraints
    }

    fn build_multicircuit<II: InputIndicator>(
        multicircuit_builder: &dyn Fn(
            &ConstraintCircuitBuilder<II>,
        ) -> Vec<ConstraintCircuitMonad<II>>,
    ) -> Vec<ConstraintCircuitMonad<II>> {
        let circuit_builder = ConstraintCircuitBuilder::new();
        multicircuit_builder(&circuit_builder)
    }

    #[test]
    fn constant_folding_processor_table() {
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
    fn constant_folding_program_table() {
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
    fn constant_folding_jump_stack_table() {
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
    fn constant_folding_op_stack_table() {
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
    fn constant_folding_ram_table() {
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
    fn constant_folding_hash_table() {
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
    fn constant_folding_u32_table() {
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
    fn constant_folding_cascade_table() {
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
    fn constant_folding_lookup_table() {
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
    fn simple_degree_lowering() {
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
    fn somewhat_simple_degree_lowering() {
        let builder = ConstraintCircuitBuilder::new();
        let x = |i| builder.input(BaseRow(i));
        let y = |i| builder.input(ExtRow(i));
        let b_con = |i: u64| builder.b_constant(i);

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
    fn less_simple_degree_lowering() {
        let builder = ConstraintCircuitBuilder::new();
        let x = |i| builder.input(BaseRow(i));

        let constraint_0 = (x(0) * x(1) * x(2)) * (x(3) * x(4)) * x(5);
        let constraint_1 = (x(6) * x(7)) * (x(3) * x(4)) * x(8);

        let mut multicircuit = [constraint_0, constraint_1];

        let target_degree = 3;
        let num_base_cols = 9;
        let num_ext_cols = 0;
        let (new_base_constraints, new_ext_constraints) = lower_degree_and_assert_properties(
            &mut multicircuit,
            target_degree,
            num_base_cols,
            num_ext_cols,
        );

        assert!(new_base_constraints.len() <= 3);
        assert!(new_ext_constraints.is_empty());
    }

    #[test]
    fn program_table_initial_constraints_degree_lowering() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtProgramTable::initial_constraints),
            AIR_TARGET_DEGREE,
            PROGRAM_TABLE_END,
            EXT_PROGRAM_TABLE_END,
        );
    }

    #[test]
    fn program_table_consistency_constraints_degree_lowering() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtProgramTable::consistency_constraints),
            AIR_TARGET_DEGREE,
            PROGRAM_TABLE_END,
            EXT_PROGRAM_TABLE_END,
        );
    }

    #[test]
    fn program_table_transition_constraints_degree_lowering() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtProgramTable::transition_constraints),
            AIR_TARGET_DEGREE,
            PROGRAM_TABLE_END,
            EXT_PROGRAM_TABLE_END,
        );
    }

    #[test]
    fn program_table_terminal_constraints_degree_lowering() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtProgramTable::terminal_constraints),
            AIR_TARGET_DEGREE,
            PROGRAM_TABLE_END,
            EXT_PROGRAM_TABLE_END,
        );
    }

    #[test]
    fn processor_table_initial_constraints_degree_lowering() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtProcessorTable::initial_constraints),
            AIR_TARGET_DEGREE,
            PROCESSOR_TABLE_END,
            EXT_PROCESSOR_TABLE_END,
        );
    }

    #[test]
    fn processor_table_consistency_constraints_degree_lowering() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtProcessorTable::consistency_constraints),
            AIR_TARGET_DEGREE,
            PROCESSOR_TABLE_END,
            EXT_PROCESSOR_TABLE_END,
        );
    }

    #[test]
    fn processor_table_transition_constraints_degree_lowering() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtProcessorTable::transition_constraints),
            AIR_TARGET_DEGREE,
            PROCESSOR_TABLE_END,
            EXT_PROCESSOR_TABLE_END,
        );
    }

    #[test]
    fn processor_table_terminal_constraints_degree_lowering() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtProcessorTable::terminal_constraints),
            AIR_TARGET_DEGREE,
            PROCESSOR_TABLE_END,
            EXT_PROCESSOR_TABLE_END,
        );
    }

    #[test]
    fn op_stack_table_initial_constraints_degree_lowering() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtOpStackTable::initial_constraints),
            AIR_TARGET_DEGREE,
            OP_STACK_TABLE_END,
            EXT_OP_STACK_TABLE_END,
        );
    }

    #[test]
    fn op_stack_table_consistency_constraints_degree_lowering() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtOpStackTable::consistency_constraints),
            AIR_TARGET_DEGREE,
            OP_STACK_TABLE_END,
            EXT_OP_STACK_TABLE_END,
        );
    }

    #[test]
    fn op_stack_table_transition_constraints_degree_lowering() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtOpStackTable::transition_constraints),
            AIR_TARGET_DEGREE,
            OP_STACK_TABLE_END,
            EXT_OP_STACK_TABLE_END,
        );
    }

    #[test]
    fn op_stack_table_terminal_constraints_degree_lowering() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtOpStackTable::terminal_constraints),
            AIR_TARGET_DEGREE,
            OP_STACK_TABLE_END,
            EXT_OP_STACK_TABLE_END,
        );
    }

    #[test]
    fn ram_table_initial_constraints_degree_lowering() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtRamTable::initial_constraints),
            AIR_TARGET_DEGREE,
            RAM_TABLE_END,
            EXT_RAM_TABLE_END,
        );
    }

    #[test]
    fn ram_table_consistency_constraints_degree_lowering() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtRamTable::consistency_constraints),
            AIR_TARGET_DEGREE,
            RAM_TABLE_END,
            EXT_RAM_TABLE_END,
        );
    }

    #[test]
    fn ram_table_transition_constraints_degree_lowering() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtRamTable::transition_constraints),
            AIR_TARGET_DEGREE,
            RAM_TABLE_END,
            EXT_RAM_TABLE_END,
        );
    }

    #[test]
    fn ram_table_terminal_constraints_degree_lowering() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtRamTable::terminal_constraints),
            AIR_TARGET_DEGREE,
            RAM_TABLE_END,
            EXT_RAM_TABLE_END,
        );
    }

    #[test]
    fn jump_stack_table_initial_constraints_degree_lowering() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtJumpStackTable::initial_constraints),
            AIR_TARGET_DEGREE,
            JUMP_STACK_TABLE_END,
            EXT_JUMP_STACK_TABLE_END,
        );
    }

    #[test]
    fn jump_stack_table_consistency_constraints_degree_lowering() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtJumpStackTable::consistency_constraints),
            AIR_TARGET_DEGREE,
            JUMP_STACK_TABLE_END,
            EXT_JUMP_STACK_TABLE_END,
        );
    }

    #[test]
    fn jump_stack_table_transition_constraints_degree_lowering() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtJumpStackTable::transition_constraints),
            AIR_TARGET_DEGREE,
            JUMP_STACK_TABLE_END,
            EXT_JUMP_STACK_TABLE_END,
        );
    }

    #[test]
    fn jump_stack_table_terminal_constraints_degree_lowering() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtJumpStackTable::terminal_constraints),
            AIR_TARGET_DEGREE,
            JUMP_STACK_TABLE_END,
            EXT_JUMP_STACK_TABLE_END,
        );
    }

    #[test]
    fn hash_table_initial_constraints_degree_lowering() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtHashTable::initial_constraints),
            AIR_TARGET_DEGREE,
            HASH_TABLE_END,
            EXT_HASH_TABLE_END,
        );
    }

    #[test]
    fn hash_table_consistency_constraints_degree_lowering() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtHashTable::consistency_constraints),
            AIR_TARGET_DEGREE,
            HASH_TABLE_END,
            EXT_HASH_TABLE_END,
        );
    }

    #[test]
    fn hash_table_transition_constraints_degree_lowering() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtHashTable::transition_constraints),
            AIR_TARGET_DEGREE,
            HASH_TABLE_END,
            EXT_HASH_TABLE_END,
        );
    }

    #[test]
    fn hash_table_terminal_constraints_degree_lowering() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtHashTable::terminal_constraints),
            AIR_TARGET_DEGREE,
            HASH_TABLE_END,
            EXT_HASH_TABLE_END,
        );
    }

    #[test]
    fn cascade_table_initial_constraints_degree_lowering() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtCascadeTable::initial_constraints),
            AIR_TARGET_DEGREE,
            CASCADE_TABLE_END,
            EXT_CASCADE_TABLE_END,
        );
    }

    #[test]
    fn cascade_table_consistency_constraints_degree_lowering() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtCascadeTable::consistency_constraints),
            AIR_TARGET_DEGREE,
            CASCADE_TABLE_END,
            EXT_CASCADE_TABLE_END,
        );
    }

    #[test]
    fn cascade_table_transition_constraints_degree_lowering() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtCascadeTable::transition_constraints),
            AIR_TARGET_DEGREE,
            CASCADE_TABLE_END,
            EXT_CASCADE_TABLE_END,
        );
    }

    #[test]
    fn cascade_table_terminal_constraints_degree_lowering() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtCascadeTable::terminal_constraints),
            AIR_TARGET_DEGREE,
            CASCADE_TABLE_END,
            EXT_CASCADE_TABLE_END,
        );
    }

    #[test]
    fn lookup_table_initial_constraints_degree_lowering() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtLookupTable::initial_constraints),
            AIR_TARGET_DEGREE,
            LOOKUP_TABLE_END,
            EXT_LOOKUP_TABLE_END,
        );
    }

    #[test]
    fn lookup_table_consistency_constraints_degree_lowering() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtLookupTable::consistency_constraints),
            AIR_TARGET_DEGREE,
            LOOKUP_TABLE_END,
            EXT_LOOKUP_TABLE_END,
        );
    }

    #[test]
    fn lookup_table_transition_constraints_degree_lowering() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtLookupTable::transition_constraints),
            AIR_TARGET_DEGREE,
            LOOKUP_TABLE_END,
            EXT_LOOKUP_TABLE_END,
        );
    }

    #[test]
    fn lookup_table_terminal_constraints_degree_lowering() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtLookupTable::terminal_constraints),
            AIR_TARGET_DEGREE,
            LOOKUP_TABLE_END,
            EXT_LOOKUP_TABLE_END,
        );
    }

    #[test]
    fn u32_table_initial_constraints_degree_lowering() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtU32Table::initial_constraints),
            AIR_TARGET_DEGREE,
            U32_TABLE_END,
            EXT_U32_TABLE_END,
        );
    }

    #[test]
    fn u32_table_consistency_constraints_degree_lowering() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtU32Table::consistency_constraints),
            AIR_TARGET_DEGREE,
            U32_TABLE_END,
            EXT_U32_TABLE_END,
        );
    }

    #[test]
    fn u32_table_transition_constraints_degree_lowering() {
        lower_degree_and_assert_properties(
            &mut build_multicircuit(&ExtU32Table::transition_constraints),
            AIR_TARGET_DEGREE,
            U32_TABLE_END,
            EXT_U32_TABLE_END,
        );
    }

    #[test]
    fn u32_table_terminal_constraints_degree_lowering() {
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
        target_deg: isize,
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
                let expression = constraint.circuit.borrow().expression.clone();
                let BinaryOperation(BinOp::Add, lhs, rhs) = expression else {
                    panic!("New {constraint_type} constraint {i} must be a subtraction.");
                };
                let Input(input_indicator) = lhs.borrow().expression.clone() else {
                    panic!("New {constraint_type} constraint {i} must be a simple substitution.");
                };
                let substitution_rule = rhs.borrow().clone();
                assert_substitution_rule_uses_legal_variables(input_indicator, &substitution_rule);
                substitution_rules.push(substitution_rule);
            }
        }

        // Use the Schwartz-Zippel lemma to check no two substitution rules are equal.
        let dummy_claim = Claim::default();
        let challenges: [XFieldElement; Challenges::SAMPLE_COUNT] = rng.gen();
        let challenges = challenges.to_vec();
        let challenges = Challenges::new(challenges, &dummy_claim);
        let challenges = &challenges.challenges;

        let num_rows = 2;
        let num_new_base_constraints = new_base_constraints.len();
        let num_new_ext_constraints = new_ext_constraints.len();
        let num_base_cols = NUM_BASE_COLUMNS + num_new_base_constraints;
        let num_ext_cols = NUM_EXT_COLUMNS + num_new_ext_constraints;
        let base_shape = [num_rows, num_base_cols];
        let ext_shape = [num_rows, num_ext_cols];
        let base_rows = Array2::from_shape_simple_fn(base_shape, || rng.gen::<BFieldElement>());
        let ext_rows = Array2::from_shape_simple_fn(ext_shape, || rng.gen::<XFieldElement>());
        let base_rows = base_rows.view();
        let ext_rows = ext_rows.view();

        let evaluated_substitution_rules = substitution_rules
            .iter()
            .map(|c| c.evaluate(base_rows, ext_rows, challenges));

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
        for constraint in &new_base_constraints {
            println!("  {constraint}");
        }
        println!("new ext constraints:");
        for constraint in &new_ext_constraints {
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
                let lhs = lhs.borrow();
                let rhs = rhs.borrow();
                assert_substitution_rule_uses_legal_variables(new_var, &lhs);
                assert_substitution_rule_uses_legal_variables(new_var, &rhs);
            }
            Input(old_var) => {
                let new_var_is_base = new_var.is_base_table_column();
                let old_var_is_base = old_var.is_base_table_column();
                let legal_substitute = match (new_var_is_base, old_var_is_base) {
                    (true, false) => false,
                    (false, true) => true,
                    _ => old_var.column() < new_var.column(),
                };
                assert!(legal_substitute, "Cannot replace {old_var} with {new_var}.");
            }
            _ => (),
        };
    }

    #[test]
    fn all_nodes_in_multicircuit_are_identified_correctly() {
        let builder = ConstraintCircuitBuilder::new();

        let x = |i| builder.input(BaseRow(i));
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

        let max_occurences = all_nodes
            .iter()
            .map(|node| all_nodes.iter().filter(|&n| n == node).count())
            .max()
            .unwrap();
        assert_eq!(8, max_occurences);

        let most_frequent_nodes = all_nodes
            .iter()
            .filter(|&node| all_nodes.iter().filter(|&n| n == node).count() == max_occurences)
            .collect::<HashSet<_>>();
        assert_eq!(2, most_frequent_nodes.len());
        assert!(most_frequent_nodes.contains(&x(2).consume()));
        assert!(most_frequent_nodes.contains(&x(10).consume()));
    }

    /// Fills the derived columns of the degree-lowering table using randomly generated rows and
    /// checks the resulting values for uniqueness. The described method corresponds to an
    /// application of the Schwartz-Zippel lemma to check uniqueness of the substitution rules
    /// generated during degree lowering.
    #[test]
    #[ignore = "(probably) requires normalization of circuit expressions"]
    fn substitution_rules_are_unique() {
        let challenges = Challenges::default();
        let mut base_table_rows = Array2::from_shape_fn((2, NUM_BASE_COLUMNS), |_| random());
        let mut ext_table_rows = Array2::from_shape_fn((2, NUM_EXT_COLUMNS), |_| random());

        DegreeLoweringTable::fill_derived_base_columns(base_table_rows.view_mut());
        DegreeLoweringTable::fill_derived_ext_columns(
            base_table_rows.view(),
            ext_table_rows.view_mut(),
            &challenges,
        );

        let mut encountered_values = HashMap::new();
        for col_idx in 0..NUM_BASE_COLUMNS {
            let val = base_table_rows[(0, col_idx)].lift();
            let other_entry = encountered_values.insert(val, col_idx);
            if let Some(other_idx) = other_entry {
                panic!("Duplicate value {val} in derived base column {other_idx} and {col_idx}.");
            }
        }
        println!("Now comparing extension columns…");
        for col_idx in 0..NUM_EXT_COLUMNS {
            let val = ext_table_rows[(0, col_idx)];
            let other_entry = encountered_values.insert(val, col_idx);
            if let Some(other_idx) = other_entry {
                panic!("Duplicate value {val} in derived ext column {other_idx} and {col_idx}.");
            }
        }
    }

    #[test]
    fn equivalent_nodes_are_detected_when_present() {
        let builder = ConstraintCircuitBuilder::new();

        let x = |i| builder.input(BaseRow(i));
        let ch = |i: usize| builder.challenge(i);

        let u0 = x(0) + x(1);
        let u1 = x(2) + x(3);
        let v = u0 * u1;

        let z0 = x(0) * x(2);
        let z2 = x(1) * x(3);

        let z1 = x(1) * x(2) + x(0) * x(3);
        let w = v - z0 - z2;
        assert!(w.find_equivalent_nodes().is_empty());

        let o = ch(0) * z1 - ch(1) * w;
        assert!(!o.find_equivalent_nodes().is_empty());
    }
}
