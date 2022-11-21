use std::borrow::BorrowMut;
use std::cell::RefCell;
use std::cmp::{self};
use std::collections::HashSet;
use std::fmt::{Debug, Display};
use std::hash::Hash;
use std::iter::Sum;
use std::marker::PhantomData;
use std::ops::{Add, Mul, Sub};
use std::rc::Rc;

use num_traits::{One, Zero};
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::mpolynomial::MPolynomial;
use twenty_first::shared_math::x_field_element::XFieldElement;

use CircuitExpression::*;

use super::challenges::TableChallenges;

#[derive(Debug, Clone, Copy, PartialEq, Hash)]
pub enum BinOp {
    Add,
    Sub,
    Mul,
}

impl Eq for BinOp {}

impl Display for BinOp {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BinOp::Add => write!(f, "+"),
            BinOp::Sub => write!(f, "-"),
            BinOp::Mul => write!(f, "*"),
        }
    }
}

/// Data structure for uniquely identifying each node
#[derive(Debug, Clone, Hash, PartialEq)]
pub struct CircuitId(usize);

impl Eq for CircuitId {}

impl Display for CircuitId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

/// An `InputIndicator` is a type that describes the position of a variable in
/// a constraint polynomial in the row layout applicable for a certain kind of
/// constraint polynomial.
///
/// A variable in a constraint polynomial comes in the shape of a `usize`, but
/// depending on the type of constraint polynomial, this index may be an index
/// into a single row (for initial, consistency and terminal constraints), or
/// a pair of adjacent rows (for transition constraints), or some other layout
/// for a third type of constraint.
///
/// `From<usize>` and `Into<usize>` occur for the purpose of this conversion.
///
/// Having `Clone + Copy + Hash + PartialEq + Eq` help put these in containers.
pub trait InputIndicator:
    Debug + Clone + Copy + Hash + PartialEq + Eq + Display + From<usize> + Into<usize>
{
}

/// A `SingleRowIndicator<COLUMN_COUNT>` describes the position of a variable in
/// a constraint polynomial that operates on a single execution trace table at a
/// time.
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum SingleRowIndicator<const COLUMN_COUNT: usize> {
    Row(usize),
}

impl<const COLUMN_COUNT: usize> Display for SingleRowIndicator<COLUMN_COUNT> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let input_indicator: String = match self {
            SingleRowIndicator::Row(i) => format!("row[{i}]"),
        };

        writeln!(f, "{input_indicator}")
    }
}

impl<const COLUMN_COUNT: usize> From<usize> for SingleRowIndicator<COLUMN_COUNT> {
    fn from(val: usize) -> Self {
        assert!(val < COLUMN_COUNT, "Cannot index out of width of table");
        SingleRowIndicator::Row(val)
    }
}

impl<const COLUMN_COUNT: usize> From<SingleRowIndicator<COLUMN_COUNT>> for usize {
    fn from(val: SingleRowIndicator<COLUMN_COUNT>) -> usize {
        match val {
            SingleRowIndicator::Row(i) => i,
        }
    }
}

impl<const COLUMN_COUNT: usize> InputIndicator for SingleRowIndicator<COLUMN_COUNT> {}

/// A `DualRowIndicator<COLUMN_COUNT>` describes the position of a variable in
/// a constraint polynomial that operates on pairs of rows (current and next).
#[derive(Debug, Clone, Copy, Hash, PartialEq, Eq)]
pub enum DualRowIndicator<const COLUMN_COUNT: usize> {
    CurrentRow(usize),
    NextRow(usize),
}

impl<const COLUMN_COUNT: usize> Display for DualRowIndicator<COLUMN_COUNT> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let input_indicator: String = match self {
            DualRowIndicator::CurrentRow(i) => format!("current_row[{i}]"),
            DualRowIndicator::NextRow(i) => format!("next_row[{i}]"),
        };

        writeln!(f, "{input_indicator}")
    }
}

impl<const COLUMN_COUNT: usize> InputIndicator for DualRowIndicator<COLUMN_COUNT> {}

impl<const COLUMN_COUNT: usize> From<usize> for DualRowIndicator<COLUMN_COUNT> {
    fn from(val: usize) -> Self {
        assert!(
            val < 2 * COLUMN_COUNT,
            "Cannot index out of two times the width of the table"
        );
        if val < COLUMN_COUNT {
            DualRowIndicator::CurrentRow(val)
        } else {
            DualRowIndicator::NextRow(val - COLUMN_COUNT)
        }
    }
}

impl<const COLUMN_COUNT: usize> From<DualRowIndicator<COLUMN_COUNT>> for usize {
    fn from(val: DualRowIndicator<COLUMN_COUNT>) -> Self {
        match val {
            DualRowIndicator::CurrentRow(i) => i,
            DualRowIndicator::NextRow(i) => COLUMN_COUNT + i,
        }
    }
}

#[derive(Debug, Clone)]
pub enum CircuitExpression<T: TableChallenges, II: InputIndicator> {
    XConstant(XFieldElement),
    BConstant(BFieldElement),
    Input(II),
    Challenge(T::Id),
    BinaryOperation(
        BinOp,
        Rc<RefCell<ConstraintCircuit<T, II>>>,
        Rc<RefCell<ConstraintCircuit<T, II>>>,
    ),
}

impl<T: TableChallenges, II: InputIndicator> Hash for CircuitExpression<T, II> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            BConstant(bfe) => bfe.hash(state),
            XConstant(xfe) => xfe.hash(state),
            Input(index) => index.hash(state),
            Challenge(table_challenge_id) => {
                table_challenge_id.hash(state);
            }
            BinaryOperation(binop, lhs, rhs) => {
                binop.hash(state);
                lhs.as_ref().borrow().hash(state);
                rhs.as_ref().borrow().hash(state);
            }
        }
    }
}

impl<T: TableChallenges, II: InputIndicator> Hash for ConstraintCircuit<T, II> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.expression.hash(state)
    }
}

impl<T: TableChallenges, II: InputIndicator> Hash for ConstraintCircuitMonad<T, II> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.circuit.as_ref().borrow().expression.hash(state)
    }
}

#[derive(Clone, Debug)]
pub struct ConstraintCircuit<T: TableChallenges, II: InputIndicator> {
    pub id: CircuitId,
    pub visited_counter: usize,
    pub expression: CircuitExpression<T, II>,
    pub var_count: usize,
}

impl<T: TableChallenges, II: InputIndicator> Eq for ConstraintCircuit<T, II> {}

impl<T: TableChallenges, II: InputIndicator> PartialEq for ConstraintCircuit<T, II> {
    /// Calculate equality of circuits.
    /// In particular, this function does *not* attempt to simplify
    /// or reduce neutral terms or products. So this comparison will
    /// return false for `a == a + 0`. It will also return false for
    /// `XFieldElement(7) == BFieldElement(7)`
    fn eq(&self, other: &Self) -> bool {
        match &self.expression {
            XConstant(self_xfe) => match &other.expression {
                XConstant(other_xfe) => self_xfe == other_xfe,
                _ => false,
            },
            BConstant(self_bfe) => match &other.expression {
                BConstant(other_bfe) => self_bfe == other_bfe,
                _ => false,
            },
            Input(self_input) => match &other.expression {
                Input(other_input) => self_input == other_input,
                _ => false,
            },
            Challenge(self_challenge_id) => match &other.expression {
                Challenge(other_challenge_id) => self_challenge_id == other_challenge_id,
                _ => false,
            },
            BinaryOperation(binop_self, lhs_self, rhs_self) => {
                match &other.expression {
                    BinaryOperation(binop_other, lhs_other, rhs_other) => {
                        // a = b `op0` c,
                        // d = e `op1` f =>
                        // a = d <= op0 == op1 && b == e && c ==f
                        binop_self == binop_other && lhs_self == lhs_other && rhs_self == rhs_other
                    }

                    _ => false,
                }
            }
        }
    }
}

impl<T: TableChallenges, II: InputIndicator> Display for ConstraintCircuit<T, II> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.expression {
            XConstant(xfe) => {
                write!(f, "{}", xfe)
            }
            BConstant(bfe) => {
                write!(f, "{}", bfe)
            }
            Input(input) => write!(f, "{} ", input),
            Challenge(self_challenge_id) => {
                write!(f, "#{}", self_challenge_id)
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

impl<T: TableChallenges, II: InputIndicator> ConstraintCircuit<T, II> {
    /// Increment `visited_counter` by one for each reachable node
    fn traverse_single(&mut self) {
        self.visited_counter += 1;
        if let BinaryOperation(_, lhs, rhs) = self.expression.borrow_mut() {
            lhs.as_ref().borrow_mut().traverse_single();
            rhs.as_ref().borrow_mut().traverse_single();
        }
    }

    /// Count how many times each reachable node is reached when traversing from
    /// the starting points that are given as input. The result is stored in the
    /// `visited_counter` field in each node.
    pub fn traverse_multiple(mpols: &mut [ConstraintCircuit<T, II>]) {
        for mpol in mpols.iter_mut() {
            assert!(
                mpol.visited_counter.is_zero(),
                "visited counter must be zero before starting count"
            );
            mpol.traverse_single();
        }
    }

    /// Reset the visited counters for the entire subtree
    fn reset_visited_counters(&mut self) {
        self.visited_counter = 0;

        if let BinaryOperation(_, lhs, rhs) = &self.expression {
            lhs.as_ref().borrow_mut().reset_visited_counters();
            rhs.as_ref().borrow_mut().reset_visited_counters();
        }
    }

    /// Verify that all IDs in the subtree are unique. Panics otherwise.
    fn inner_has_unique_ids(&mut self, ids: &mut HashSet<usize>) {
        let new_value = ids.insert(self.id.0);
        assert!(
            !self.visited_counter.is_zero() || new_value,
            "ID = {} was repeated",
            self.id
        );
        self.visited_counter += 1;
        if let BinaryOperation(_, lhs, rhs) = &self.expression {
            lhs.as_ref().borrow_mut().inner_has_unique_ids(ids);
            rhs.as_ref().borrow_mut().inner_has_unique_ids(ids);
        }
    }

    // Verify that a multitree has unique IDs. Otherwise panic.
    pub fn assert_has_unique_ids(constraints: &mut [ConstraintCircuit<T, II>]) {
        let mut ids: HashSet<usize> = HashSet::new();

        for mpol in constraints.iter_mut() {
            mpol.inner_has_unique_ids(&mut ids);
        }

        for mpol in constraints.iter_mut() {
            mpol.reset_visited_counters();
        }
    }

    /// Apply constant folding to simplify the (sub)tree.
    /// If the subtree is a leaf (terminal), no change.
    /// If the subtree is a binary operation on:
    ///
    ///  - one constant x one constant   => fold
    ///  - one constant x one expr       => can't
    ///  - one expr x one constant       => can't
    ///  - one expr x one expr           => can't
    ///
    /// This operation mutates self and returns true if a change was
    /// applied anywhere in the tree.
    fn constant_fold_inner(&mut self) -> bool {
        let mut change_tracker = false;
        if let BinaryOperation(_, lhs, rhs) = &self.expression {
            change_tracker |= lhs.clone().as_ref().borrow_mut().constant_fold_inner();
            change_tracker |= rhs.clone().as_ref().borrow_mut().constant_fold_inner();
        }

        match &self.expression.clone() {
            BinaryOperation(binop, lhs, rhs) => {
                // a + 0 = a ∧ a - 0 = a
                if matches!(binop, BinOp::Add | BinOp::Sub) && rhs.as_ref().borrow().is_zero() {
                    *self.expression.borrow_mut() = lhs.as_ref().borrow().expression.clone();
                    return true;
                }

                // 0 + a = a
                if *binop == BinOp::Add && lhs.as_ref().borrow().is_zero() {
                    *self.expression.borrow_mut() = rhs.as_ref().borrow().expression.clone();
                    return true;
                }

                if matches!(binop, BinOp::Mul) {
                    // a * 1 = a
                    if rhs.as_ref().borrow().is_one() {
                        *self.expression.borrow_mut() = lhs.as_ref().borrow().expression.clone();
                        return true;
                    }

                    // 1 * a = a
                    if lhs.as_ref().borrow().is_one() {
                        *self.expression.borrow_mut() = rhs.as_ref().borrow().expression.clone();
                        return true;
                    }

                    // 0 * a = a * 0 = 0
                    if lhs.as_ref().borrow().is_zero() || rhs.as_ref().borrow().is_zero() {
                        *self.expression.borrow_mut() = BConstant(0u64.into());
                        return true;
                    }
                }

                // if left and right hand sides are both constants
                if let XConstant(lhs_xfe) = lhs.as_ref().borrow().expression {
                    if let XConstant(rhs_xfe) = rhs.as_ref().borrow().expression {
                        *self.expression.borrow_mut() = match binop {
                            BinOp::Add => XConstant(lhs_xfe + rhs_xfe),
                            BinOp::Sub => XConstant(lhs_xfe - rhs_xfe),
                            BinOp::Mul => XConstant(lhs_xfe * rhs_xfe),
                        };
                        return true;
                    }

                    if let BConstant(rhs_bfe) = rhs.as_ref().borrow().expression {
                        *self.expression.borrow_mut() = match binop {
                            BinOp::Add => XConstant(lhs_xfe + rhs_bfe.lift()),
                            BinOp::Sub => XConstant(lhs_xfe - rhs_bfe.lift()),
                            BinOp::Mul => XConstant(lhs_xfe * rhs_bfe),
                        };
                        return true;
                    }
                }

                if let BConstant(lhs_bfe) = lhs.as_ref().borrow().expression {
                    if let XConstant(rhs_xfe) = rhs.as_ref().borrow().expression {
                        *self.expression.borrow_mut() = match binop {
                            BinOp::Add => XConstant(lhs_bfe.lift() + rhs_xfe),
                            BinOp::Sub => XConstant(lhs_bfe.lift() - rhs_xfe),
                            BinOp::Mul => XConstant(rhs_xfe * lhs_bfe),
                        };
                        return true;
                    }

                    if let BConstant(rhs_bfe) = rhs.as_ref().borrow().expression {
                        *self.expression.borrow_mut() = match binop {
                            BinOp::Add => BConstant(lhs_bfe + rhs_bfe),
                            BinOp::Sub => BConstant(lhs_bfe - rhs_bfe),
                            BinOp::Mul => BConstant(lhs_bfe * rhs_bfe),
                        };
                        return true;
                    }
                }

                change_tracker
            }
            _ => change_tracker,
        }
    }

    /// Reduce size of multitree by simplifying constant expressions such as `1 * MPol(_,_)`
    pub fn constant_folding(circuits: &mut [&mut ConstraintCircuit<T, II>]) {
        for circuit in circuits.iter_mut() {
            let mut mutated = circuit.constant_fold_inner();
            while mutated {
                mutated = circuit.constant_fold_inner();
            }
        }
    }

    /// Return max degree after evaluating the circuit with an input of specified degree
    pub fn symbolic_degree_bound(&self, max_degrees: &[i64]) -> i64 {
        assert_eq!(
            self.var_count,
            max_degrees.len(),
            "max_degrees length and var_count must match. Got: {}, {}.",
            max_degrees.len(),
            self.var_count
        );
        match &self.expression {
            BinaryOperation(binop, lhs, rhs) => {
                let lhs_degree = lhs.borrow().symbolic_degree_bound(max_degrees);
                let rhs_degree = rhs.borrow().symbolic_degree_bound(max_degrees);
                match binop {
                    BinOp::Add | BinOp::Sub => cmp::max(lhs_degree, rhs_degree),
                    BinOp::Mul => {
                        // If either operand is zero, product is zero so degree is -1
                        if lhs_degree == -1 || rhs_degree == -1 {
                            -1
                        } else {
                            lhs_degree + rhs_degree
                        }
                    }
                }
            }
            Input(input) => {
                let index: usize = (*input).into();
                max_degrees[index]
            }
            XConstant(xfe) => {
                if xfe.is_zero() {
                    -1
                } else {
                    0
                }
            }
            BConstant(bfe) => {
                if bfe.is_zero() {
                    -1
                } else {
                    0
                }
            }
            Challenge(_) => 0,
        }
    }

    /// Return degree of the multivariate polynomial represented by this circuit
    pub fn degree(&self) -> i64 {
        match &self.expression {
            BinaryOperation(binop, lhs, rhs) => {
                let lhs_degree = lhs.borrow().degree();
                let rhs_degree = rhs.borrow().degree();
                match binop {
                    BinOp::Add | BinOp::Sub => cmp::max(lhs_degree, rhs_degree),
                    BinOp::Mul => {
                        if lhs_degree == -1 || rhs_degree == -1 {
                            -1
                        } else {
                            lhs_degree + rhs_degree
                        }
                    }
                }
            }
            Input(_) => 1,
            XConstant(xfe) => {
                if xfe.is_zero() {
                    -1
                } else {
                    0
                }
            }
            BConstant(bfe) => {
                if bfe.is_zero() {
                    -1
                } else {
                    0
                }
            }
            Challenge(_) => 0,
        }
    }

    /// Return all visited counters in the subtree
    pub fn get_all_visited_counters(&self) -> Vec<usize> {
        // Maybe this could be solved smarter with dynamic programming
        // but we probably don't need that as our circuits aren't too big.
        match &self.expression {
            // The highest number will always be in a leaf so we only
            // need to check those.
            BinaryOperation(_, lhs, rhs) => {
                let lhs_counters = lhs.as_ref().borrow().get_all_visited_counters();
                let rhs_counters = rhs.as_ref().borrow().get_all_visited_counters();
                let own_counter = self.visited_counter;
                let mut all = vec![vec![own_counter], lhs_counters, rhs_counters].concat();
                all.sort_unstable();
                all.dedup();
                all.reverse();
                all
            }
            _ => vec![self.visited_counter],
        }
    }

    /// Return true if the contained multivariate polynomial consists of only a single term. This means that it can be
    /// pretty-printed without parentheses.
    pub fn print_without_parentheses(&self) -> bool {
        !matches!(&self.expression, BinaryOperation(_, _, _))
    }

    /// Return true if this node represents a constant value of zero, does not
    /// catch composite expressions that will always evaluate to zero.
    pub fn is_zero(&self) -> bool {
        match self.expression {
            BConstant(bfe) => bfe.is_zero(),
            XConstant(xfe) => xfe.is_zero(),
            _ => false,
        }
    }

    /// Return true if this node represents a constant value of one, does not
    /// catch composite expressions that will always evaluate to one.
    pub fn is_one(&self) -> bool {
        match self.expression {
            XConstant(xfe) => xfe.is_one(),
            BConstant(bfe) => bfe.is_one(),
            _ => false,
        }
    }

    /// Return Some(index) iff the circuit node represents a linear function with one
    /// term and a coefficient of one. Returns the index in which the multivariate
    /// polynomial is linear. Returns None otherwise.
    pub fn get_linear_one_index(&self) -> Option<usize> {
        if let Input(input) = self.expression {
            let index: usize = input.into();
            Some(index)
        } else {
            None
        }
    }

    /// Return true iff the evaluation value of this node depends on a challenge
    pub fn is_randomized(&self) -> bool {
        match &self.expression {
            Challenge(_) => true,
            BinaryOperation(_, lhs, rhs) => {
                lhs.as_ref().borrow().is_randomized() || rhs.as_ref().borrow().is_randomized()
            }
            _ => false,
        }
    }

    /// Return the flattened multivariate polynomial that this node
    /// represents, given the challenges.
    pub fn partial_evaluate(&self, challenges: &T) -> MPolynomial<XFieldElement> {
        let mut polynomial = self.flatten(challenges);
        polynomial.normalize();
        polynomial
    }

    /// Return the flat multivariate polynomial that computes the
    /// same value as this circuit. Do this by recursively applying
    /// the multivariate polynomial binary operations, and by
    /// replacing the inputs by variables.
    fn flatten(&self, challenges: &T) -> MPolynomial<XFieldElement> {
        match &self.expression {
            XConstant(xfe) => MPolynomial::<XFieldElement>::from_constant(*xfe, self.var_count),
            BConstant(bfe) => {
                MPolynomial::<XFieldElement>::from_constant(bfe.lift(), self.var_count)
            }
            Input(input) => {
                let mpol_index: usize = (*input).into();
                MPolynomial::<XFieldElement>::variables(self.var_count)[mpol_index].clone()
            }
            Challenge(challenge_id) => MPolynomial::<XFieldElement>::from_constant(
                challenges.get_challenge(*challenge_id),
                self.var_count,
            ),
            BinaryOperation(binop, lhs, rhs) => match binop {
                BinOp::Add => {
                    lhs.as_ref().borrow().flatten(challenges)
                        + rhs.as_ref().borrow().flatten(challenges)
                }
                BinOp::Sub => {
                    lhs.as_ref().borrow().flatten(challenges)
                        - rhs.as_ref().borrow().flatten(challenges)
                }
                BinOp::Mul => {
                    lhs.as_ref().borrow().flatten(challenges)
                        * rhs.as_ref().borrow().flatten(challenges)
                }
            },
        }
    }

    /// Replace all challenges with constants in subtree
    fn apply_challenges_to_one_root(&mut self, challenges: &T) {
        match &self.expression {
            Challenge(challenge_id) => {
                *self.expression.borrow_mut() = XConstant(challenges.get_challenge(*challenge_id))
            }
            BinaryOperation(_, lhs, rhs) => {
                lhs.as_ref()
                    .borrow_mut()
                    .apply_challenges_to_one_root(challenges);
                rhs.as_ref()
                    .borrow_mut()
                    .apply_challenges_to_one_root(challenges);
            }
            _ => (),
        }
    }

    /// Simplify the circuit constraints by replacing the known challenges with roots
    pub fn apply_challenges(constraints: &mut [ConstraintCircuit<T, II>], challenges: &T) {
        for circuit in constraints.iter_mut() {
            circuit.apply_challenges_to_one_root(challenges);
        }
    }
}

#[derive(Clone)]
pub struct ConstraintCircuitMonad<T: TableChallenges, II: InputIndicator> {
    pub circuit: Rc<RefCell<ConstraintCircuit<T, II>>>,
    pub all_nodes: Rc<RefCell<HashSet<ConstraintCircuitMonad<T, II>>>>,
    pub id_counter_ref: Rc<RefCell<usize>>,
}

impl<T: TableChallenges, II: InputIndicator> Debug for ConstraintCircuitMonad<T, II> {
    // We cannot derive `Debug` as `all_nodes` contains itself which a derived `Debug` will
    // attempt to print as well, thus leading to infinite recursion.
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ConstraintCircuitMonad")
            .field("id", &self.circuit)
            .field(
                "all_nodes length: ",
                &self.all_nodes.as_ref().borrow().len(),
            )
            .field(
                "id_counter_ref value: ",
                &self.id_counter_ref.as_ref().borrow(),
            )
            .finish()
    }
}

impl<T: TableChallenges, II: InputIndicator> Display for ConstraintCircuitMonad<T, II> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.circuit.as_ref().borrow())
    }
}

impl<T: TableChallenges, II: InputIndicator> PartialEq for ConstraintCircuitMonad<T, II> {
    // Equality for the ConstraintCircuitMonad is defined by the circuit, not the
    // other metadata (e.g. ID) that it carries around.
    fn eq(&self, other: &Self) -> bool {
        self.circuit == other.circuit
    }
}

impl<T: TableChallenges, II: InputIndicator> Eq for ConstraintCircuitMonad<T, II> {}

/// Helper function for binary operations that are used to generate new parent
/// nodes in the multitree that represents the algebraic circuit. Ensures that
/// each newly created node has a unique ID.
fn binop<T: TableChallenges, II: InputIndicator>(
    binop: BinOp,
    lhs: ConstraintCircuitMonad<T, II>,
    rhs: ConstraintCircuitMonad<T, II>,
) -> ConstraintCircuitMonad<T, II> {
    // Get ID for the new node
    let new_index = lhs.id_counter_ref.as_ref().borrow().to_owned();

    let new_node = ConstraintCircuitMonad {
        circuit: Rc::new(RefCell::new(ConstraintCircuit {
            visited_counter: 0,
            expression: BinaryOperation(binop, Rc::clone(&lhs.circuit), Rc::clone(&rhs.circuit)),
            id: CircuitId(new_index),
            var_count: lhs.circuit.as_ref().borrow().var_count,
        })),
        id_counter_ref: Rc::clone(&lhs.id_counter_ref),
        all_nodes: Rc::clone(&lhs.all_nodes),
    };

    // check if node already exists
    let contained = lhs.all_nodes.as_ref().borrow().contains(&new_node);
    if contained {
        let ret0 = &lhs.all_nodes.as_ref().borrow();
        let ret1 = &(*ret0.get(&new_node).as_ref().unwrap()).clone();
        return ret1.to_owned();
    }

    // If the operator commutes, check if the inverse node has already been constructed.
    // If it has, return this instead. Do not allow a new one to be built.
    if matches!(binop, BinOp::Add | BinOp::Mul) {
        let new_node_inverted = ConstraintCircuitMonad {
            circuit: Rc::new(RefCell::new(ConstraintCircuit {
                visited_counter: 0,
                expression: BinaryOperation(
                    binop,
                    // Switch rhs and lhs for symmetric operators to check for membership in hash set
                    Rc::clone(&rhs.circuit),
                    Rc::clone(&lhs.circuit),
                ),
                id: CircuitId(new_index),
                var_count: lhs.circuit.as_ref().borrow().var_count,
            })),
            id_counter_ref: Rc::clone(&lhs.id_counter_ref),
            all_nodes: Rc::clone(&lhs.all_nodes),
        };

        // check if node already exists
        let inverted_contained = lhs.all_nodes.as_ref().borrow().contains(&new_node_inverted);
        if inverted_contained {
            let ret0 = &lhs.all_nodes.as_ref().borrow();
            let ret1 = &(*ret0.get(&new_node_inverted).as_ref().unwrap()).clone();
            return ret1.to_owned();
        }
    }

    // Increment counter index
    *lhs.id_counter_ref.as_ref().borrow_mut() = new_index + 1;

    // Store new node in HashSet
    new_node
        .all_nodes
        .as_ref()
        .borrow_mut()
        .insert(new_node.clone());

    new_node
}

impl<T: TableChallenges, II: InputIndicator> Add for ConstraintCircuitMonad<T, II> {
    type Output = ConstraintCircuitMonad<T, II>;

    fn add(self, rhs: Self) -> Self::Output {
        binop(BinOp::Add, self, rhs)
    }
}

impl<T: TableChallenges, II: InputIndicator> Sub for ConstraintCircuitMonad<T, II> {
    type Output = ConstraintCircuitMonad<T, II>;

    fn sub(self, rhs: Self) -> Self::Output {
        binop(BinOp::Sub, self, rhs)
    }
}

impl<T: TableChallenges, II: InputIndicator> Mul for ConstraintCircuitMonad<T, II> {
    type Output = ConstraintCircuitMonad<T, II>;

    fn mul(self, rhs: Self) -> Self::Output {
        binop(BinOp::Mul, self, rhs)
    }
}

/// This will panic if the iterator is empty because the neutral
/// element needs a unique ID, and we have no way of getting that
/// here.
impl<T: TableChallenges, II: InputIndicator> Sum for ConstraintCircuitMonad<T, II> {
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|accum, item| accum + item)
            .expect("ConstraintCircuitMonad Iterator was empty")
    }
}

impl<T: TableChallenges, II: InputIndicator> ConstraintCircuitMonad<T, II> {
    /// Flatten a circuit to reveal its equivalent multivariate polynomial
    pub fn partial_evaluate(&self, challenges: &T) -> MPolynomial<XFieldElement> {
        self.circuit.as_ref().borrow().partial_evaluate(challenges)
    }

    /// Unwrap a ConstraintCircuitMonad to reveal its inner ConstraintCircuit
    pub fn consume(self) -> ConstraintCircuit<T, II> {
        self.circuit.try_borrow().unwrap().to_owned()
    }
}

#[derive(Debug, Clone)]
/// Helper struct to construct new leaf nodes in the circuit multitree. Ensures that each newly
/// created node gets a unique ID.
pub struct ConstraintCircuitBuilder<T: TableChallenges, II: InputIndicator> {
    id_counter: Rc<RefCell<usize>>,
    all_nodes: Rc<RefCell<HashSet<ConstraintCircuitMonad<T, II>>>>,
    _table_type: PhantomData<T>,
    var_count: usize,
}

impl<T: TableChallenges, II: InputIndicator> ConstraintCircuitBuilder<T, II> {
    pub fn new(var_count: usize) -> Self {
        Self {
            id_counter: Rc::new(RefCell::new(0)),
            all_nodes: Rc::new(RefCell::new(HashSet::default())),
            _table_type: PhantomData,
            var_count,
        }
    }

    /// Create constant leaf node
    pub fn x_constant(&self, xfe: XFieldElement) -> ConstraintCircuitMonad<T, II> {
        let expression = XConstant(xfe);
        self.make_leaf(expression)
    }

    /// Create constant leaf node
    pub fn b_constant(&self, bfe: BFieldElement) -> ConstraintCircuitMonad<T, II> {
        let expression = BConstant(bfe);
        self.make_leaf(expression)
    }

    /// Create deterministic input leaf node
    pub fn input(&self, input: II) -> ConstraintCircuitMonad<T, II> {
        let expression = Input(input);
        self.make_leaf(expression)
    }

    /// Create challenge leaf node
    pub fn challenge(&self, challenge_id: T::Id) -> ConstraintCircuitMonad<T, II> {
        let expression = Challenge(challenge_id);
        self.make_leaf(expression)
    }

    fn make_leaf(&self, expression: CircuitExpression<T, II>) -> ConstraintCircuitMonad<T, II> {
        let new_id = self.id_counter.as_ref().borrow().to_owned();
        let new_node = ConstraintCircuitMonad {
            circuit: Rc::new(RefCell::new(ConstraintCircuit {
                visited_counter: 0usize,
                expression,
                id: CircuitId(new_id),
                var_count: self.var_count,
            })),
            id_counter_ref: Rc::clone(&self.id_counter),
            all_nodes: Rc::clone(&self.all_nodes),
        };

        // Check if node already exists, return the existing one if it does
        let contained = self.all_nodes.as_ref().borrow().contains(&new_node);
        if contained {
            let ret0 = &self.all_nodes.as_ref().borrow();
            let ret1 = &(*ret0.get(&new_node).as_ref().unwrap()).clone();
            return ret1.to_owned();
        }

        // If node did not already exist, increment counter and insert node into hash set
        *self.id_counter.as_ref().borrow_mut() = new_id + 1;
        self.all_nodes
            .as_ref()
            .borrow_mut()
            .insert(new_node.clone());

        new_node
    }
}

#[cfg(test)]
mod constraint_circuit_tests {
    use std::collections::hash_map::DefaultHasher;
    use std::hash::Hasher;

    use itertools::Itertools;
    use rand::{thread_rng, RngCore};
    use twenty_first::shared_math::mpolynomial::MPolynomial;
    use twenty_first::shared_math::other::random_elements;

    use crate::table::challenges::AllChallenges;
    use crate::table::instruction_table::{
        ExtInstructionTable, InstructionTableChallengeId, InstructionTableChallenges,
    };
    use crate::table::jump_stack_table::ExtJumpStackTable;
    use crate::table::op_stack_table::ExtOpStackTable;
    use crate::table::processor_table::ExtProcessorTable;
    use crate::table::program_table::ExtProgramTable;
    use crate::table::ram_table::ExtRamTable;

    use super::*;

    fn node_counter_inner<T: TableChallenges, II: InputIndicator>(
        constraint: &mut ConstraintCircuit<T, II>,
        counter: &mut usize,
    ) {
        if constraint.visited_counter == 0 {
            *counter += 1;
            constraint.visited_counter = 1;

            if let BinaryOperation(_, lhs, rhs) = &constraint.expression {
                node_counter_inner(&mut lhs.as_ref().borrow_mut(), counter);
                node_counter_inner(&mut rhs.as_ref().borrow_mut(), counter);
            }
        }
    }

    /// Count the total number of nodes in call constraints
    fn node_counter<T: TableChallenges, II: InputIndicator>(
        constraints: &mut [ConstraintCircuit<T, II>],
    ) -> usize {
        let mut counter = 0;

        for constraint in constraints.iter_mut() {
            node_counter_inner(constraint, &mut counter);
        }

        for constraint in constraints.iter_mut() {
            ConstraintCircuit::reset_visited_counters(constraint);
        }

        counter
    }

    fn circuit_mpol_builder(
        challenges: &InstructionTableChallenges,
    ) -> (
        ConstraintCircuitMonad<InstructionTableChallenges, DualRowIndicator<50>>,
        MPolynomial<XFieldElement>,
        ConstraintCircuitBuilder<InstructionTableChallenges, DualRowIndicator<50>>,
    ) {
        let var_count = 100;
        let circuit_builder: ConstraintCircuitBuilder<
            InstructionTableChallenges,
            DualRowIndicator<50>,
        > = ConstraintCircuitBuilder::new(var_count);
        let mpol_variables = MPolynomial::<XFieldElement>::variables(var_count);
        let b_constants: Vec<BFieldElement> = random_elements(var_count);
        let x_constants: Vec<XFieldElement> = random_elements(var_count);
        let zero = MPolynomial::from_constant(XFieldElement::zero(), var_count);
        let mut rng = thread_rng();
        let rand: usize = rng.next_u64() as usize;
        let mut ret_mpol = mpol_variables[rand % var_count].clone();
        let circuit_input: DualRowIndicator<50> = (rand % var_count).into();
        let mut ret_circuit = circuit_builder.input(circuit_input);
        for _ in 0..100 {
            let rand: usize = rng.next_u64() as usize;
            let choices = 6;
            let (mpol, circuit): (
                MPolynomial<XFieldElement>,
                ConstraintCircuitMonad<InstructionTableChallenges, DualRowIndicator<50>>,
            ) = if rand % choices == 0 {
                // p(x, y, z) = x
                let mp = mpol_variables[rand % var_count].clone();
                let input_value: DualRowIndicator<50> = (rand % var_count).into();
                (mp.clone(), circuit_builder.input(input_value))
            } else if rand % choices == 1 {
                // p(x, y, z) = c
                (
                    MPolynomial::from_constant(x_constants[rand % var_count], var_count),
                    circuit_builder.x_constant(x_constants[rand % var_count]),
                )
            } else if rand % choices == 2 {
                // p(x, y, z) = rand_i
                (
                    MPolynomial::from_constant(challenges.processor_perm_indeterminate, var_count),
                    circuit_builder
                        .challenge(InstructionTableChallengeId::ProcessorPermIndeterminate),
                )
            } else if rand % choices == 3 {
                // p(x, y, z) = 0
                (
                    zero.clone(),
                    circuit_builder.x_constant(XFieldElement::zero()),
                )
            } else if rand % choices == 4 {
                // p(x, y, z) = bfe
                (
                    MPolynomial::from_constant(b_constants[rand % var_count].lift(), var_count),
                    circuit_builder.b_constant(b_constants[rand % var_count]),
                )
            } else {
                // p(x, y, z) = rand_i * x
                let input_value: DualRowIndicator<50> = (rand % var_count).into();
                (
                    mpol_variables[rand % var_count]
                        .clone()
                        .scalar_mul(challenges.processor_perm_indeterminate),
                    circuit_builder.input(input_value)
                        * circuit_builder
                            .challenge(InstructionTableChallengeId::ProcessorPermIndeterminate),
                )
            };
            let operation_indicator = rand % 3;
            match operation_indicator {
                0 => {
                    ret_mpol = ret_mpol.clone() * mpol;
                    ret_circuit = ret_circuit * circuit;
                }
                1 => {
                    ret_mpol = ret_mpol.clone() + mpol;
                    ret_circuit = ret_circuit + circuit;
                }
                2 => {
                    ret_mpol = ret_mpol.clone() - mpol;
                    ret_circuit = ret_circuit - circuit;
                }
                _ => panic!(),
            }
        }

        (ret_circuit, ret_mpol, circuit_builder)
    }

    // Make a deep copy of a MPolCircuit and return it as a MPolCircuitRef
    fn deep_copy_inner<T: TableChallenges, II: InputIndicator>(
        val: &ConstraintCircuit<T, II>,
        builder: &mut ConstraintCircuitBuilder<T, II>,
    ) -> ConstraintCircuitMonad<T, II> {
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

    fn deep_copy<T: TableChallenges, II: InputIndicator>(
        val: &ConstraintCircuit<T, II>,
    ) -> ConstraintCircuitMonad<T, II> {
        let mut builder = ConstraintCircuitBuilder::new(val.var_count);
        deep_copy_inner(val, &mut builder)
    }

    #[test]
    fn equality_and_hash_agree_test() {
        // Since the MPolCircuits are put into a hash set, I think it's important
        // that `Eq` and `Hash` agree whether two nodes are equal or not. So if
        // k1 == k2 => h(k1) == h(k2)
        for _ in 0..100 {
            let challenges = AllChallenges::placeholder();
            let (circuit, _mpol, circuit_builder) =
                circuit_mpol_builder(&challenges.instruction_table_challenges);
            let mut hasher0 = DefaultHasher::new();
            circuit.hash(&mut hasher0);
            let hash0 = hasher0.finish();
            assert_eq!(circuit, circuit);

            // let zero = circuit_builder.deterministic_input(MPolynomial::zero(100));
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
    fn mpol_circuit_hash_is_unchanged_by_meta_data_test() {
        // From https://doc.rust-lang.org/std/collections/struct.HashSet.html
        // "It is a logic error for a key to be modified in such a way that the key’s hash, as determined by the Hash
        // trait, or its equality, as determined by the Eq trait, changes while it is in the map. This is normally only
        // possible through Cell, RefCell, global state, I/O, or unsafe code. The behavior resulting from such a logic
        // error is not specified, but will be encapsulated to the HashSet that observed the logic error and not result
        // in undefined behavior. This could include panics, incorrect results, aborts, memory leaks, and
        // non-termination."
        // This means that the hash of a node may not depend on: `visited_counter`, `counter`,
        // `id_counter_ref`, or `all_nodes`. The reason for this constraint is that `all_nodes` contains
        // the digest of all nodes in the multi tree.
        let challenges = AllChallenges::placeholder();
        let (circuit, _mpol, _circuit_builder) =
            circuit_mpol_builder(&challenges.instruction_table_challenges);
        let mut hasher0 = DefaultHasher::new();
        circuit.hash(&mut hasher0);
        let digest_prior = hasher0.finish();

        // Increase visited counter and verify digest is unchanged
        circuit.circuit.as_ref().borrow_mut().traverse_single();
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
    fn circuit_and_mpol_equivalence_check() {
        for i in 0..1000 {
            let challenges = AllChallenges::placeholder();
            let (circuit, mpol, circuit_builder) =
                circuit_mpol_builder(&challenges.instruction_table_challenges);
            assert_eq!(
                mpol,
                circuit.partial_evaluate(&challenges.instruction_table_challenges),
                "Partial evaluate and constructed mpol must agree"
            );

            assert_eq!(
                circuit.circuit.as_ref().borrow().degree(),
                mpol.degree(),
                "circuit degree and equivalent mpol degree must match before constant folding. circuit: {}\n\n mpol: {mpol}.\n iteration {i}", circuit.circuit.as_ref().borrow()
            );

            // Also compare with symbolic evaluation
            let rand_degree = (thread_rng().next_u32() % 200) as i64;
            let interpolated_degrees = vec![rand_degree; circuit_builder.var_count];
            assert_eq!(
                circuit.circuit.as_ref().borrow().symbolic_degree_bound(&interpolated_degrees),
                mpol.symbolic_degree_bound(&interpolated_degrees),
                "symbolic degree bounds must match before constant folding. circuit: {}\n\n mpol: {mpol}.\n interpolated degree: {rand_degree}\niteration {i}",
                circuit.circuit.as_ref().borrow()
            );

            // Also verify equality after constant folding of the circuit
            let copied_circuit = deep_copy(&circuit.circuit.as_ref().borrow());
            let mut circuits = vec![circuit.consume()];
            ConstraintCircuit::constant_folding(&mut circuits.iter_mut().collect_vec());
            let partial_evaluated =
                circuits[0].partial_evaluate(&challenges.instruction_table_challenges);
            assert_eq!(
                mpol,
                partial_evaluated, "Circuit before and after constant folding must agree after parital evaluate.\n before: {copied_circuit}\nafter: {}", circuits[0]
            );
            assert_eq!(
                circuits[0].degree(),
                mpol.degree(),
                "circuit degree and equivalent mpol degree must match after constant folding. circuit: {}\n\n mpol: {mpol}.\n iteration {i}", circuits[0]
            );
            assert_eq!(
                circuits[0].degree(),
                partial_evaluated.degree(),
                "circuit degree and the degree of its partial evaluation must agree. circuit: {}\n\n mpol: {mpol}.\n iteration {i}", circuits[0]
            );

            // Also compare with symbolic evaluation
            let interpolated_degrees = vec![rand_degree; circuit_builder.var_count];
            assert_eq!(
                circuits[0].symbolic_degree_bound(&interpolated_degrees),
                partial_evaluated.symbolic_degree_bound(&interpolated_degrees),
                "symbolic degree bounds must match before constant folding. circuit: {}\n\n mpol: {mpol}.\n iteration {i}", circuits[0]
            );
        }
    }

    #[test]
    fn circuit_equality_check_and_constant_folding_test() {
        let var_count = 10;
        let circuit_builder: ConstraintCircuitBuilder<
            InstructionTableChallenges,
            DualRowIndicator<5>,
        > = ConstraintCircuitBuilder::new(var_count);
        let var_0 = circuit_builder.input(DualRowIndicator::CurrentRow(0));
        let var_4 = circuit_builder.input(DualRowIndicator::NextRow(4));
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
        let mut var_0_circuit_0 = var_0_copy_0.consume();
        let mut var_0_same_circuit_0 = var_0_mul_one_0.consume();
        ConstraintCircuit::constant_folding(&mut [&mut var_0_circuit_0, &mut var_0_same_circuit_0]);
        assert_eq!(var_0_circuit_0, var_0_same_circuit_0);
        assert_eq!(var_0_same_circuit_0, var_0_circuit_0);

        // Verify that constant folding can handle a = 1 * a
        let var_0_copy_1 = deep_copy(&var_0.circuit.as_ref().borrow());
        let var_0_one_mul_1 = one.clone() * var_0_copy_1.clone();
        assert_ne!(var_0_copy_1, var_0_one_mul_1);
        let mut var_0_circuit_1 = var_0_copy_1.consume();
        let mut var_0_same_circuit_1 = var_0_one_mul_1.consume();
        ConstraintCircuit::constant_folding(&mut [&mut var_0_circuit_1, &mut var_0_same_circuit_1]);
        assert_eq!(var_0_circuit_1, var_0_same_circuit_1);
        assert_eq!(var_0_same_circuit_1, var_0_circuit_1);

        // Verify that constant folding can handle a = 1 * a * 1
        let var_0_copy_2 = deep_copy(&var_0.circuit.as_ref().borrow());
        let var_0_one_mul_2 = one.clone() * var_0_copy_2.clone() * one;
        assert_ne!(var_0_copy_2, var_0_one_mul_2);
        let mut var_0_circuit_2 = var_0_copy_2.consume();
        let mut var_0_same_circuit_2 = var_0_one_mul_2.consume();
        ConstraintCircuit::constant_folding(&mut [&mut var_0_circuit_2, &mut var_0_same_circuit_2]);
        assert_eq!(var_0_circuit_2, var_0_same_circuit_2);
        assert_eq!(var_0_same_circuit_2, var_0_circuit_2);

        // Verify that constant folding handles a + 0 = a
        let var_0_copy_3 = deep_copy(&var_0.circuit.as_ref().borrow());
        let var_0_plus_zero_3 = var_0_copy_3.clone() + zero.clone();
        assert_ne!(var_0_copy_3, var_0_plus_zero_3);
        let mut var_0_circuit_3 = var_0_copy_3.consume();
        let mut var_0_same_circuit_3 = var_0_plus_zero_3.consume();
        ConstraintCircuit::constant_folding(&mut [&mut var_0_circuit_3, &mut var_0_same_circuit_3]);
        assert_eq!(var_0_circuit_3, var_0_same_circuit_3);
        assert_eq!(var_0_same_circuit_3, var_0_circuit_3);

        // Verify that constant folding handles a + (a * 0) = a
        let var_0_copy_4 = deep_copy(&var_0.circuit.as_ref().borrow());
        let var_0_plus_zero_4 = var_0_copy_4.clone() + var_0_copy_4.clone() * zero.clone();
        assert_ne!(var_0_copy_4, var_0_plus_zero_4);
        let mut var_0_circuit_4 = var_0_copy_4.consume();
        let mut var_0_same_circuit_4 = var_0_plus_zero_4.consume();
        ConstraintCircuit::constant_folding(&mut [&mut var_0_circuit_4, &mut var_0_same_circuit_4]);
        assert_eq!(var_0_circuit_4, var_0_same_circuit_4);
        assert_eq!(var_0_same_circuit_4, var_0_circuit_4);

        // Verify that constant folding does not equate `0 - a` with `a`
        let var_0_copy_5 = deep_copy(&var_0.circuit.as_ref().borrow());
        let zero_minus_var_0 = zero - var_0_copy_5.clone();
        assert_ne!(var_0_copy_5, zero_minus_var_0);
        let mut var_0_circuit_5 = var_0_copy_5.consume();
        let mut var_0_not_same_circuit_5 = zero_minus_var_0.consume();
        ConstraintCircuit::constant_folding(&mut [
            &mut var_0_circuit_5,
            &mut var_0_not_same_circuit_5,
        ]);
        assert_ne!(var_0_circuit_5, var_0_not_same_circuit_5);
        assert_ne!(var_0_not_same_circuit_5, var_0_circuit_5);
    }

    #[test]
    fn constant_folding_pbt() {
        for _ in 0..1000 {
            let challenges = AllChallenges::placeholder();
            let (circuit, _mpol, circuit_builder) =
                circuit_mpol_builder(&challenges.instruction_table_challenges);
            let one = circuit_builder.x_constant(1.into());
            let zero = circuit_builder.x_constant(0.into());

            // Verify that constant folding can handle a = a * 1
            let copy_0 = deep_copy(&circuit.circuit.as_ref().borrow());
            let copy_0_alt = copy_0.clone() * one.clone();
            assert_ne!(copy_0, copy_0_alt);
            let mut circuit_0 = copy_0.consume();
            let mut same_circuit_0 = copy_0_alt.consume();
            ConstraintCircuit::constant_folding(&mut [&mut circuit_0, &mut same_circuit_0]);
            assert_eq!(circuit_0, same_circuit_0);
            assert_eq!(same_circuit_0, circuit_0);

            // Verify that constant folding can handle a = 1 * a
            let copy_1 = deep_copy(&circuit.circuit.as_ref().borrow());
            let copy_1_alt = one.clone() * copy_1.clone();
            assert_ne!(copy_1, copy_1_alt);
            let mut circuit_1 = copy_1.consume();
            let mut circuit_1_alt = copy_1_alt.consume();
            ConstraintCircuit::constant_folding(&mut [&mut circuit_1, &mut circuit_1_alt]);
            assert_eq!(circuit_1, circuit_1_alt);
            assert_eq!(circuit_1_alt, circuit_1);

            // Verify that constant folding can handle a = 1 * a * 1
            let copy_2 = deep_copy(&circuit.circuit.as_ref().borrow());
            let copy_2_alt = one.clone() * copy_2.clone() * one.clone();
            assert_ne!(copy_2, copy_2_alt);
            let mut circuit_1 = copy_2.consume();
            let mut circuit_1_alt = copy_2_alt.consume();
            ConstraintCircuit::constant_folding(&mut [&mut circuit_1, &mut circuit_1_alt]);
            assert_eq!(circuit_1, circuit_1_alt);
            assert_eq!(circuit_1_alt, circuit_1);

            // Verify that constant folding handles a + 0 = a
            let copy_3 = deep_copy(&circuit.circuit.as_ref().borrow());
            let copy_3_alt = copy_3.clone() + zero.clone();
            assert_ne!(copy_3, copy_3_alt);
            let mut circuit_3 = copy_3.consume();
            let mut circuit_3_alt = copy_3_alt.consume();
            ConstraintCircuit::constant_folding(&mut [&mut circuit_3, &mut circuit_3_alt]);
            assert_eq!(circuit_3, circuit_3_alt);
            assert_eq!(circuit_3_alt, circuit_3);

            // Verify that constant folding handles a + (a * 0) = a
            let copy_4 = deep_copy(&circuit.circuit.as_ref().borrow());
            let copy_4_alt = copy_4.clone() + copy_4.clone() * zero.clone();
            assert_ne!(copy_4, copy_4_alt);
            let mut circuit_4 = copy_4.consume();
            let mut circuit_4_alt = copy_4_alt.consume();
            ConstraintCircuit::constant_folding(&mut [&mut circuit_4, &mut circuit_4_alt]);
            assert_eq!(circuit_4, circuit_4_alt);
            assert_eq!(circuit_4_alt, circuit_4);

            // Verify that constant folding handles a + (0 * a) = a
            let copy_5 = deep_copy(&circuit.circuit.as_ref().borrow());
            let copy_5_alt = copy_5.clone() + copy_5.clone() * zero.clone();
            assert_ne!(copy_5, copy_5_alt);
            let mut circuit_5 = copy_5.consume();
            let mut circuit_5_alt = copy_5_alt.consume();
            ConstraintCircuit::constant_folding(&mut [&mut circuit_5, &mut circuit_5_alt]);
            assert_eq!(circuit_5, circuit_5_alt);
            assert_eq!(circuit_5_alt, circuit_5);

            // Verify that constant folding does not equate `0 - a` with `a`
            // But only if `a != 0`
            let copy_6 = deep_copy(&circuit.circuit.as_ref().borrow());
            let zero_minus_copy_6 = zero.clone() - copy_6.clone();
            assert_ne!(copy_6, zero_minus_copy_6);
            let mut var_0_circuit_6 = copy_6.consume();
            let mut var_0_not_same_circuit_6 = zero_minus_copy_6.consume();
            ConstraintCircuit::constant_folding(&mut [
                &mut var_0_circuit_6,
                &mut var_0_not_same_circuit_6,
            ]);

            // An X field and a B field leaf will never be equal
            if var_0_circuit_6.is_zero()
                && (matches!(var_0_circuit_6.expression, CircuitExpression::BConstant(_))
                    && matches!(
                        var_0_not_same_circuit_6.expression,
                        CircuitExpression::BConstant(_)
                    )
                    || matches!(var_0_circuit_6.expression, CircuitExpression::XConstant(_))
                        && matches!(
                            var_0_not_same_circuit_6.expression,
                            CircuitExpression::XConstant(_)
                        ))
            {
                assert_eq!(var_0_circuit_6, var_0_not_same_circuit_6);
                assert_eq!(var_0_not_same_circuit_6, var_0_circuit_6);
            } else {
                assert_ne!(var_0_circuit_6, var_0_not_same_circuit_6);
                assert_ne!(var_0_not_same_circuit_6, var_0_circuit_6);
            }

            // Verify that constant folding handles a - 0 = a
            let copy_7 = deep_copy(&circuit.circuit.as_ref().borrow());
            let copy_7_alt = copy_7.clone() - zero.clone();
            assert_ne!(copy_7, copy_7_alt);
            let mut circuit_7 = copy_7.consume();
            let mut circuit_7_alt = copy_7_alt.consume();
            ConstraintCircuit::constant_folding(&mut [&mut circuit_7, &mut circuit_7_alt]);
            assert_eq!(circuit_7, circuit_7_alt);
            assert_eq!(circuit_7_alt, circuit_7);
        }
    }

    #[test]
    fn mpol_algebra_and_circuit_building_is_equivalent_simple_test() {
        let var_count = 10;
        let variables = MPolynomial::<XFieldElement>::variables(10);
        let four_mpol = MPolynomial::<XFieldElement>::from_constant(
            XFieldElement::new_const(4u64.into()),
            var_count,
        );

        let expr_mpol = (variables[0].clone() + variables[4].clone())
            * (variables[8].clone() - variables[9].clone())
            * four_mpol.clone()
            * four_mpol;

        let circuit_builder: ConstraintCircuitBuilder<
            InstructionTableChallenges,
            DualRowIndicator<5>,
        > = ConstraintCircuitBuilder::new(var_count);
        let var_0 = circuit_builder.input(DualRowIndicator::CurrentRow(0));
        let var_4 = circuit_builder.input(DualRowIndicator::CurrentRow(4));
        let var_8 = circuit_builder.input(DualRowIndicator::NextRow(3));
        let var_9 = circuit_builder.input(DualRowIndicator::NextRow(4));

        let four = circuit_builder.x_constant(4.into());

        let expr_circuit = (var_0 + var_4) * (var_8 - var_9) * four.clone() * four;

        // Verify that IDs are unique
        ConstraintCircuit::<InstructionTableChallenges, DualRowIndicator<5>>::assert_has_unique_ids(
            &mut [expr_circuit.clone().consume()],
        );

        // Verify that partial evaluation agrees with the flat polynomial representation
        let expr_circuit_partial_evaluated = expr_circuit
            .partial_evaluate(&AllChallenges::placeholder().instruction_table_challenges);
        assert_eq!(expr_mpol, expr_circuit_partial_evaluated);
    }

    fn constant_folding_of_table_constraints_test<T: TableChallenges, II: InputIndicator>(
        mut constraints: Vec<ConstraintCircuit<T, II>>,
        challenges: T,
        table_name: &str,
    ) {
        println!(
            "nodes in {table_name} table constraint multitree prior to constant folding: {}",
            node_counter(&mut constraints)
        );

        let mut before_fold: Vec<MPolynomial<XFieldElement>> = vec![];

        for circuit in constraints.iter() {
            let partial_evaluated = circuit.partial_evaluate(&challenges);
            assert_eq!(
                partial_evaluated.degree(),
                circuit.degree(),
                "Degree of partial evaluated and circuit must agree before constant folding"
            );
            before_fold.push(partial_evaluated);
        }

        ConstraintCircuit::constant_folding(&mut constraints.iter_mut().collect_vec());
        println!(
            "nodes in {table_name} constraint multitree after constant folding: {}",
            node_counter(&mut constraints)
        );

        let mut after_fold: Vec<MPolynomial<XFieldElement>> = vec![];
        for circuit in constraints.iter() {
            let partial_evaluated = circuit.partial_evaluate(&challenges);
            assert_eq!(
                partial_evaluated.degree(),
                circuit.degree(),
                "Degree of partial evaluated and circuit must agree after constant folding"
            );
            after_fold.push(partial_evaluated);
        }

        for (i, (before, after)) in before_fold.iter().zip_eq(after_fold.iter()).enumerate() {
            assert_eq!(before, after, "Constant folding must leave partially evaluated constraints unchanged for {table_name} table constraint {i}");
        }

        assert!(
            constraints
                .iter()
                .any(|constraint| constraint.is_randomized()),
            "Constraint must contain randomness before challenges have been applied"
        );

        // apply challenges and verify that subtree no longer contains randomness
        ConstraintCircuit::apply_challenges(&mut constraints, &challenges);
        assert!(
            constraints
                .iter()
                .all(|constraint| !constraint.is_randomized()),
            "Constraint may not contain randomness after challenges have been applied"
        );

        ConstraintCircuit::constant_folding(&mut constraints.iter_mut().collect_vec());
        println!(
            "nodes in {table_name} constraint multitree after applying challenges and constant folding again: {}",
            node_counter(&mut constraints)
        );
        let circuit_degree = constraints.iter().map(|c| c.degree()).max().unwrap();

        println!("Max degree constraint for {table_name} table: {circuit_degree}");
    }

    #[test]
    fn constant_folding_instruction_table_test() {
        let challenges = AllChallenges::placeholder();
        let constraint_circuits = ExtInstructionTable::ext_transition_constraints_as_circuits();
        constant_folding_of_table_constraints_test(
            constraint_circuits,
            challenges.instruction_table_challenges,
            "instruction",
        );
    }

    #[test]
    fn constant_folding_processor_table_test() {
        let challenges = AllChallenges::placeholder();
        let constraint_circuits = ExtProcessorTable::ext_transition_constraints_as_circuits();
        constant_folding_of_table_constraints_test(
            constraint_circuits,
            challenges.processor_table_challenges,
            "processor",
        );
    }

    #[test]
    fn constant_folding_program_table_test() {
        let challenges = AllChallenges::placeholder();
        let constraint_circuits = ExtProgramTable::ext_transition_constraints_as_circuits();
        constant_folding_of_table_constraints_test(
            constraint_circuits,
            challenges.program_table_challenges,
            "program",
        );
    }

    #[test]
    fn constant_folding_jump_stack_table_test() {
        let challenges = AllChallenges::placeholder();
        let constraint_circuits = ExtJumpStackTable::ext_transition_constraints_as_circuits();
        constant_folding_of_table_constraints_test(
            constraint_circuits,
            challenges.jump_stack_table_challenges,
            "jump stack",
        );
    }

    #[test]
    fn constant_folding_op_stack_table_test() {
        let challenges = AllChallenges::placeholder();
        let constraint_circuits = ExtOpStackTable::ext_transition_constraints_as_circuits();
        constant_folding_of_table_constraints_test(
            constraint_circuits,
            challenges.op_stack_table_challenges,
            "op stack",
        );
    }

    #[test]
    fn constant_folding_ram_stack_table_test() {
        let challenges = AllChallenges::placeholder();
        let constraint_circuits = ExtRamTable::ext_transition_constraints_as_circuits();
        constant_folding_of_table_constraints_test(
            constraint_circuits,
            challenges.ram_table_challenges,
            "ram",
        );
    }
}
