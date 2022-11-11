use std::{
    borrow::BorrowMut,
    cell::RefCell,
    cmp::{self},
    collections::HashSet,
    fmt::{Debug, Display},
    iter::Sum,
    marker::PhantomData,
    ops::{Add, Mul, Sub},
    rc::Rc,
};

use num_traits::{One, Zero};
use std::hash::Hash;
use twenty_first::shared_math::{mpolynomial::MPolynomial, x_field_element::XFieldElement};

use super::challenges::TableChallenges;

#[derive(Debug, Clone, Copy)]
pub enum ConstraintType<T: TableChallenges> {
    Deterministic,
    Randomized(T::Id),
}

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
#[derive(Debug, Clone, std::hash::Hash, PartialEq)]
pub struct CircuitId(usize);

impl Eq for CircuitId {}

impl Display for CircuitId {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

#[derive(Debug, Clone)]
pub enum CircuitExpression<T: TableChallenges> {
    // MPol(MPolynomial<XFieldElement>, ConstraintType<T>),
    Constant(XFieldElement),
    Input(usize, ConstraintType<T>),
    Challenge(T::Id),
    BinaryOperation(
        BinOp,
        Rc<RefCell<ConstraintCircuit<T>>>,
        Rc<RefCell<ConstraintCircuit<T>>>,
    ),
}

impl<T: TableChallenges> Hash for CircuitExpression<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        match self {
            CircuitExpression::Constant(xfe) => xfe.hash(state),
            CircuitExpression::Input(index, ConstraintType::Deterministic) => index.hash(state),
            CircuitExpression::Input(index, ConstraintType::Randomized(challenge_id)) => {
                index.hash(state);
                challenge_id.hash(state);
            }

            CircuitExpression::Challenge(table_challenge_id) => {
                table_challenge_id.hash(state);
            }

            CircuitExpression::BinaryOperation(binop, lhs, rhs) => {
                binop.hash(state);
                lhs.as_ref().borrow().hash(state);
                rhs.as_ref().borrow().hash(state);
            }
        }
    }
}

impl<T: TableChallenges> Hash for ConstraintCircuit<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.expression.hash(state)
    }
}

impl<T: TableChallenges> Hash for ConstraintCircuitMonad<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.circuit.as_ref().borrow().expression.hash(state)
    }
}

#[derive(Clone, Debug)]
pub struct ConstraintCircuit<T: TableChallenges> {
    pub id: CircuitId,
    pub visited_counter: usize,
    pub expression: CircuitExpression<T>,
    var_count: usize,
}

impl<T: TableChallenges> Eq for ConstraintCircuit<T> {}

impl<T: TableChallenges> PartialEq for ConstraintCircuit<T> {
    /// Calculate equality of circuits.
    /// In particular, this function does *not* attempt to simplify
    /// or reduce neutral terms or products. So this comparison will
    /// return false for `a == a + 0`.
    fn eq(&self, other: &Self) -> bool {
        match &self.expression {
            CircuitExpression::Constant(self_xfe) => match &other.expression {
                CircuitExpression::Constant(other_xfe) => self_xfe == other_xfe,
                _ => false,
            },
            CircuitExpression::Input(self_input_index, ConstraintType::Deterministic) => {
                match &other.expression {
                    CircuitExpression::Input(other_input_index, ConstraintType::Deterministic) => {
                        self_input_index == other_input_index
                    }

                    // for all inputs: randomized != deterministic
                    _ => false,
                }
            }
            CircuitExpression::Input(
                self_input_index,
                ConstraintType::Randomized(self_challenge_id),
            ) => {
                match &other.expression {
                    // for all inputs: randomized != deterministic

                    // randomized is equal to other randomized if randomized and contained input index
                    // are both the same
                    CircuitExpression::Input(
                        other_input_index,
                        ConstraintType::Randomized(other_challenge_id),
                    ) => {
                        other_challenge_id == self_challenge_id
                            && other_input_index == self_input_index
                    }

                    _ => false,
                }
            }
            CircuitExpression::Challenge(self_challenge_id) => match &other.expression {
                CircuitExpression::Challenge(other_challenge_id) => {
                    self_challenge_id == other_challenge_id
                }
                _ => false,
            },
            CircuitExpression::BinaryOperation(binop_self, lhs_self, rhs_self) => {
                match &other.expression {
                    CircuitExpression::BinaryOperation(binop_other, lhs_other, rhs_other) => {
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

impl<T: TableChallenges> Display for ConstraintCircuit<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.expression {
            CircuitExpression::Constant(xfe) => {
                write!(f, "{}", xfe)
            }
            // CircuitExpression::MPol(normalized_pol, challenge) => match challenge {
            CircuitExpression::Input(self_input_index, constraint_type) => match constraint_type {
                ConstraintType::Deterministic => write!(f, "${} ", self_input_index),
                ConstraintType::Randomized(constraint_id) => {
                    write!(f, "#{}${}", constraint_id, self_input_index)
                }
            },
            CircuitExpression::Challenge(self_challenge_id) => {
                write!(f, "#{}", self_challenge_id)
            }
            CircuitExpression::BinaryOperation(operation, lhs, rhs) => {
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

impl<T: TableChallenges> ConstraintCircuit<T> {
    /// Increment `visited_counter` by one for each reachable node
    fn traverse_single(&mut self) {
        self.visited_counter += 1;
        match self.expression.borrow_mut() {
            CircuitExpression::Constant(_) => (),
            CircuitExpression::Input(_, _) => (),
            CircuitExpression::Challenge(_) => (),
            CircuitExpression::BinaryOperation(_, lhs, rhs) => {
                lhs.as_ref().borrow_mut().traverse_single();
                rhs.as_ref().borrow_mut().traverse_single();
            }
        }
    }

    /// Count how many times each reachable node is reached when traversing from
    /// the starting points that are given as input. The result is stored in the
    /// `visited_counter` field in each node.
    pub fn traverse_multiple(mpols: &mut [ConstraintCircuit<T>]) {
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

        if let CircuitExpression::BinaryOperation(_, lhs, rhs) = &self.expression {
            lhs.as_ref().borrow_mut().reset_visited_counters();
            rhs.as_ref().borrow_mut().reset_visited_counters();
        }
    }

    /// Verify that all IDs in the subtree are unique. Panics otherwise.
    fn inner_has_unique_ids(&mut self, ids: &mut HashSet<usize>) {
        let new_value = ids.insert(self.id.0);
        assert!(!self.visited_counter.is_zero() || new_value);
        self.visited_counter += 1;
        match &self.expression {
            CircuitExpression::Constant(_) => (),
            CircuitExpression::Input(_, _) => (),
            CircuitExpression::Challenge(_) => (),
            CircuitExpression::BinaryOperation(_, lhs, rhs) => {
                lhs.as_ref().borrow_mut().inner_has_unique_ids(ids);
                rhs.as_ref().borrow_mut().inner_has_unique_ids(ids);
            }
        }
    }

    // Verify that a multitree has unique IDs. Otherwise panic.
    pub fn assert_has_unique_ids(mpols: &mut [ConstraintCircuit<T>]) {
        let mut ids: HashSet<usize> = HashSet::new();

        for mpol in mpols.iter_mut() {
            mpol.inner_has_unique_ids(&mut ids);
        }

        for mpol in mpols.iter_mut() {
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
        if let CircuitExpression::BinaryOperation(_, lhs, rhs) = &self.expression {
            change_tracker |= lhs.clone().as_ref().borrow_mut().constant_fold_inner();
            change_tracker |= rhs.clone().as_ref().borrow_mut().constant_fold_inner();
        }

        match &self.expression.clone() {
            CircuitExpression::Constant(_) => change_tracker,
            CircuitExpression::Input(_, _) => change_tracker,
            CircuitExpression::Challenge(_) => change_tracker,
            CircuitExpression::BinaryOperation(binop, lhs, rhs) => {
                // a + 0 = a /\ a - 0 = a
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
                    if rhs.as_ref().borrow().is_one() {
                        *self.expression.borrow_mut() = lhs.as_ref().borrow().expression.clone();
                        return true;
                    }

                    if lhs.as_ref().borrow().is_one() {
                        *self.expression.borrow_mut() = rhs.as_ref().borrow().expression.clone();
                        return true;
                    }

                    if lhs.as_ref().borrow().is_zero() || rhs.as_ref().borrow().is_zero() {
                        *self.expression.borrow_mut() = CircuitExpression::Constant(0.into());
                        return true;
                    }
                }

                // if left and right hand sides are both constants
                if let Some(lhs_const) = lhs.as_ref().borrow().get_constant_value() {
                    if let Some(rhs_const) = rhs.as_ref().borrow().get_constant_value() {
                        *self.expression.borrow_mut() = match binop {
                            BinOp::Add => CircuitExpression::Constant(lhs_const + rhs_const),
                            BinOp::Sub => CircuitExpression::Constant(lhs_const - rhs_const),
                            BinOp::Mul => CircuitExpression::Constant(lhs_const * rhs_const),
                        };
                        return true;
                    }
                }

                change_tracker
            }
        }
    }

    /// Reduce size of multitree by simplifying constant expressions such as `1 * MPol(_,_)`
    pub fn constant_folding(circuits: &mut [&mut ConstraintCircuit<T>]) {
        for mpol in circuits.iter_mut() {
            let mut mutated = mpol.constant_fold_inner();
            while mutated {
                mutated = mpol.constant_fold_inner();
            }
        }
    }

    /// Return the highest counter value encountered in this subtree
    pub fn get_max_visited_counter(&self) -> usize {
        // Maybe this could be solved smarter with dynamic programming
        // but we probably don't need that as our circuits aren't too big.
        match &self.expression {
            CircuitExpression::Constant(_) => self.visited_counter,
            CircuitExpression::Input(_, _) => self.visited_counter,
            CircuitExpression::Challenge(_) => self.visited_counter,
            // The highest number will always be in a leaf so we only
            // need to check those.
            CircuitExpression::BinaryOperation(_, lhs, rhs) => cmp::max(
                lhs.as_ref().borrow().get_max_visited_counter(),
                rhs.as_ref().borrow().get_max_visited_counter(),
            ),
        }
    }

    /// Return true if the contained multivariate polynomial consists of only a single term. This means that it can be
    /// pretty-printed without parentheses.
    pub fn is_single_term(&self) -> bool {
        match &self.expression {
            CircuitExpression::Constant(_) => true,
            CircuitExpression::Input(_, _) => true,
            CircuitExpression::Challenge(_) => true,
            CircuitExpression::BinaryOperation(_, _, _) => false,
        }
    }

    /// Return true if this node represents a constant value of zero, does not
    /// catch composite expressions that will always evaluate to zero.
    pub fn is_zero(&self) -> bool {
        match self.get_constant_value() {
            Some(val) => val.is_zero(),
            None => false,
        }
    }

    /// Return true if this node represents a constant value of one, does not
    /// catch composite expressions that will always evaluate to one.
    pub fn is_one(&self) -> bool {
        match self.get_constant_value() {
            Some(val) => val.is_one(),
            None => false,
        }
    }

    /// If node is a constant value, then return the constant. Otherwise return None.
    pub fn get_constant_value(&self) -> Option<XFieldElement> {
        match &self.expression {
            CircuitExpression::Constant(xfe) => Some(*xfe),

            // This assumes that the circuit is built in a somewhat sane manner, or that
            // constant folding has been applied on the multitree
            _ => None,
        }
    }

    /// Return Some(index) iff the circuit node represents a linear function with one
    /// term and a coefficient of one. Returns the index in which the multivariate
    /// polynomial is linear. Returns None otherwise.
    pub fn get_linear_one_index(&self) -> Option<usize> {
        if let CircuitExpression::Input(input_index, ConstraintType::Deterministic) =
            self.expression
        {
            Some(input_index)
        } else {
            None
        }
    }

    /// Return true iff the evaluation value of this node depends on a challenge
    pub fn is_randomized(&self) -> bool {
        match &self.expression {
            CircuitExpression::Input(_, constaint_type) => match constaint_type {
                ConstraintType::Deterministic => false,
                ConstraintType::Randomized(_) => true,
            },
            CircuitExpression::Constant(_) => false,
            CircuitExpression::Challenge(_) => true,
            CircuitExpression::BinaryOperation(_, lhs, rhs) => {
                lhs.as_ref().borrow().is_randomized() || rhs.as_ref().borrow().is_randomized()
            }
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
            CircuitExpression::Constant(xfe) => {
                MPolynomial::<XFieldElement>::from_constant(*xfe, self.var_count)
            }
            CircuitExpression::Input(input_index, constraint_type) => match constraint_type {
                ConstraintType::Deterministic => {
                    MPolynomial::<XFieldElement>::variables(self.var_count)[*input_index].clone()
                }
                ConstraintType::Randomized(challenge_id) => {
                    MPolynomial::<XFieldElement>::variables(self.var_count)[*input_index]
                        .scalar_mul(challenges.get_challenge(*challenge_id))
                }
            },
            CircuitExpression::Challenge(challenge_id) => {
                MPolynomial::<XFieldElement>::from_constant(
                    challenges.get_challenge(*challenge_id),
                    self.var_count,
                )
            }
            CircuitExpression::BinaryOperation(binop, lhs, rhs) => match binop {
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
}

#[derive(Clone)]
pub struct ConstraintCircuitMonad<T: TableChallenges> {
    pub circuit: Rc<RefCell<ConstraintCircuit<T>>>,
    pub all_nodes: Rc<RefCell<HashSet<ConstraintCircuitMonad<T>>>>,
    pub id_counter_ref: Rc<RefCell<usize>>,
}

impl<T: TableChallenges> Debug for ConstraintCircuitMonad<T> {
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

impl<T: TableChallenges> Display for ConstraintCircuitMonad<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.circuit.as_ref().borrow())
    }
}

impl<T: TableChallenges> PartialEq for ConstraintCircuitMonad<T> {
    // Equality for the ConstraintCircuitMonad is defined by the circuit, not the
    // other metadata (e.g. ID) that it carries around.
    fn eq(&self, other: &Self) -> bool {
        self.circuit == other.circuit
    }
}

impl<T: TableChallenges> Eq for ConstraintCircuitMonad<T> {}

/// Helper function for binary operations that are used to generate new parent
/// nodes in the multitree that represents the algebraic circuit. Ensures that
/// each newly created node has a unique ID.
fn binop<T: TableChallenges>(
    binop: BinOp,
    lhs: ConstraintCircuitMonad<T>,
    rhs: ConstraintCircuitMonad<T>,
) -> ConstraintCircuitMonad<T> {
    // Get ID for the new node
    let new_index = lhs.id_counter_ref.as_ref().borrow().to_owned();

    let new_node = ConstraintCircuitMonad {
        circuit: Rc::new(RefCell::new(ConstraintCircuit {
            visited_counter: 0,
            expression: CircuitExpression::BinaryOperation(
                binop,
                Rc::clone(&lhs.circuit),
                Rc::clone(&rhs.circuit),
            ),
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

impl<T: TableChallenges> Add for ConstraintCircuitMonad<T> {
    type Output = ConstraintCircuitMonad<T>;

    fn add(self, rhs: Self) -> Self::Output {
        binop(BinOp::Add, self, rhs)
    }
}

impl<T: TableChallenges> Sub for ConstraintCircuitMonad<T> {
    type Output = ConstraintCircuitMonad<T>;

    fn sub(self, rhs: Self) -> Self::Output {
        binop(BinOp::Sub, self, rhs)
    }
}

impl<T: TableChallenges> Mul for ConstraintCircuitMonad<T> {
    type Output = ConstraintCircuitMonad<T>;

    fn mul(self, rhs: Self) -> Self::Output {
        binop(BinOp::Mul, self, rhs)
    }
}

impl<T: TableChallenges> Sum for ConstraintCircuitMonad<T> {
    // TODO: This will panic if the iterator is empty! Can or should we avoid that?
    fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
        iter.reduce(|accum, item| accum + item).unwrap()
    }
}

#[derive(Debug, Clone)]
/// Helper struct to construct new leaf nodes in the circuit multitree. Ensures that each newly
/// created node gets a unique ID.
pub struct ConstraintCircuitBuilder<T: TableChallenges> {
    id_counter: Rc<RefCell<usize>>,
    all_nodes: Rc<RefCell<HashSet<ConstraintCircuitMonad<T>>>>,
    _table_type: PhantomData<T>,
    var_count: usize,
}

impl<T: TableChallenges> ConstraintCircuitBuilder<T> {
    pub fn new(var_count: usize) -> Self {
        Self {
            id_counter: Rc::new(RefCell::new(0)),
            all_nodes: Rc::new(RefCell::new(HashSet::default())),
            _table_type: PhantomData,
            var_count,
        }
    }
    /// Create constant leaf node
    pub fn constant(&mut self, xfe: XFieldElement) -> ConstraintCircuitMonad<T> {
        let expression = CircuitExpression::Constant(xfe);
        self.make_leaf(expression)
    }

    /// Create deterministic input leaf node
    pub fn deterministic_input(&mut self, input_index: usize) -> ConstraintCircuitMonad<T> {
        let expression = CircuitExpression::Input(input_index, ConstraintType::Deterministic);
        self.make_leaf(expression)
    }

    /// Create randomized input leaf node
    pub fn randomized_input(
        &mut self,
        input_index: usize,
        challenge_id: T::Id,
    ) -> ConstraintCircuitMonad<T> {
        let expression =
            CircuitExpression::Input(input_index, ConstraintType::Randomized(challenge_id));
        self.make_leaf(expression)
    }

    /// Create challenge leaf node
    pub fn challenge(&mut self, challenge_id: T::Id) -> ConstraintCircuitMonad<T> {
        let expression = CircuitExpression::Challenge(challenge_id);
        self.make_leaf(expression)
    }

    fn make_leaf(&self, expression: CircuitExpression<T>) -> ConstraintCircuitMonad<T> {
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

impl<T: TableChallenges> ConstraintCircuitMonad<T> {
    pub fn partial_evaluate(&self, challenges: &T) -> MPolynomial<XFieldElement> {
        self.circuit.as_ref().borrow().partial_evaluate(challenges)
    }

    /// Unwrap a ConstraintCircuitMonad to reveal its inner ConstraintCircuit
    pub fn consume(self) -> ConstraintCircuit<T> {
        self.circuit.try_borrow().unwrap().to_owned()
    }
}

#[cfg(test)]
mod constraint_circuit_tests {
    use itertools::Itertools;
    use std::{collections::hash_map::DefaultHasher, hash::Hasher};

    use rand::{thread_rng, RngCore};
    use twenty_first::shared_math::mpolynomial::MPolynomial;

    use crate::table::{
        challenges::AllChallenges,
        instruction_table::{
            ExtInstructionTable, InstructionTableChallengeId, InstructionTableChallenges,
        },
        processor_table::ExtProcessorTable,
        program_table::ExtProgramTable,
    };

    use super::*;

    fn circuit_mpol_builder(
        challenges: &InstructionTableChallenges,
    ) -> (
        ConstraintCircuitMonad<InstructionTableChallenges>,
        MPolynomial<XFieldElement>,
        ConstraintCircuitBuilder<InstructionTableChallenges>,
    ) {
        let var_count = 100;
        let mut circuit_builder: ConstraintCircuitBuilder<InstructionTableChallenges> =
            ConstraintCircuitBuilder::new(var_count);
        let mpol_variables = MPolynomial::<XFieldElement>::variables(var_count);
        let constants: Vec<XFieldElement> = (140u64..140 + var_count as u64)
            .map(|x| XFieldElement::new_const(x.into()))
            .collect_vec();
        let zero = MPolynomial::from_constant(XFieldElement::zero(), var_count);
        let mut rng = thread_rng();
        let rand: usize = rng.next_u64() as usize;
        let mut ret_mpol = mpol_variables[rand % var_count].clone();
        let mut ret_circuit = circuit_builder.deterministic_input(rand % var_count);
        for _ in 0..100 {
            let rand: usize = rng.next_u64() as usize;
            let (mpol, circuit) = if rand % 5 == 0 {
                // p(x, y, z) = x
                let mp = mpol_variables[rand % var_count].clone();
                (
                    mp.clone(),
                    circuit_builder.deterministic_input(rand % var_count),
                )
            } else if rand % 5 == 1 {
                // p(x, y, z) = c
                (
                    MPolynomial::from_constant(constants[rand % var_count], var_count),
                    circuit_builder.constant(constants[rand % var_count]),
                )
            } else if rand % 5 == 2 {
                // p(x, y, z) = rand_i
                (
                    MPolynomial::from_constant(challenges.processor_perm_indeterminate, var_count),
                    circuit_builder
                        .challenge(InstructionTableChallengeId::ProcessorPermIndeterminate),
                )
            } else if rand % 5 == 3 {
                // p(x, y, z) = 0
                (
                    zero.clone(),
                    circuit_builder.constant(XFieldElement::zero()),
                )
            } else {
                // p(x, y, z) = rand_i * x
                (
                    mpol_variables[rand % var_count]
                        .clone()
                        .scalar_mul(challenges.processor_perm_indeterminate),
                    circuit_builder.randomized_input(
                        rand % var_count,
                        InstructionTableChallengeId::ProcessorPermIndeterminate,
                    ),
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
    fn deep_copy_inner<T: TableChallenges>(
        val: &ConstraintCircuit<T>,
        builder: &mut ConstraintCircuitBuilder<T>,
    ) -> ConstraintCircuitMonad<T> {
        match &val.expression {
            CircuitExpression::BinaryOperation(op, lhs, rhs) => {
                let lhs_ref = deep_copy_inner(&lhs.as_ref().borrow(), builder);
                let rhs_ref = deep_copy_inner(&rhs.as_ref().borrow(), builder);
                binop(*op, lhs_ref, rhs_ref)
            }
            CircuitExpression::Constant(xfe) => builder.constant(*xfe),
            CircuitExpression::Input(input_index, ConstraintType::Deterministic) => {
                builder.deterministic_input(*input_index)
            }
            CircuitExpression::Input(input_index, ConstraintType::Randomized(challenge_id)) => {
                builder.randomized_input(*input_index, *challenge_id)
            }
            CircuitExpression::Challenge(challenge_id) => builder.challenge(*challenge_id),
        }
    }

    fn deep_copy<T: TableChallenges>(val: &ConstraintCircuit<T>) -> ConstraintCircuitMonad<T> {
        let mut builder = ConstraintCircuitBuilder::new(val.var_count);
        deep_copy_inner(val, &mut builder)
    }

    #[test]
    fn equality_and_hash_agree_test() {
        // Since the MPolCircuits are put into a hash set, I think it's important
        // that `Eq` and `Hash` agree whether two nodes are equal or not. So if
        // k1 == k2 => h(k1) == h(k2)
        for _ in 0..10 {
            let challenges = AllChallenges::placeholder();
            let (circuit, _mpol, mut circuit_builder) =
                circuit_mpol_builder(&challenges.instruction_table_challenges);
            let mut hasher0 = DefaultHasher::new();
            circuit.hash(&mut hasher0);
            let hash0 = hasher0.finish();
            assert_eq!(circuit, circuit);

            // let zero = circuit_builder.deterministic_input(MPolynomial::zero(100));
            let zero = circuit_builder.constant(0.into());
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
        for _ in 0..100 {
            let challenges = AllChallenges::placeholder();
            let (circuit, mpol, _) = circuit_mpol_builder(&challenges.instruction_table_challenges);
            assert_eq!(
                mpol,
                circuit.partial_evaluate(&challenges.instruction_table_challenges)
            );

            // Also verify equality after constant folding of the circuit
            let mut circuits = vec![circuit.consume()];
            ConstraintCircuit::constant_folding(&mut circuits.iter_mut().collect_vec());
            assert_eq!(
                mpol,
                circuits[0].partial_evaluate(&challenges.instruction_table_challenges)
            );
        }
    }

    #[test]
    fn circuit_equality_check_and_constant_folding_test() {
        let var_count = 10;
        let mut circuit_builder: ConstraintCircuitBuilder<InstructionTableChallenges> =
            ConstraintCircuitBuilder::new(var_count);
        let var_0 = circuit_builder.deterministic_input(0);
        let var_4 = circuit_builder.deterministic_input(4);
        let four = circuit_builder.constant(4.into());
        let one = circuit_builder.constant(1.into());
        let zero = circuit_builder.constant(0.into());

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
            let (circuit, _mpol, mut circuit_builder) =
                circuit_mpol_builder(&challenges.instruction_table_challenges);
            let one = circuit_builder.constant(1.into());
            let zero = circuit_builder.constant(0.into());

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

            if var_0_circuit_6.is_zero() {
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
            * four_mpol.clone();

        let mut circuit_builder: ConstraintCircuitBuilder<InstructionTableChallenges> =
            ConstraintCircuitBuilder::new(var_count);
        let var_0 = circuit_builder.deterministic_input(0);
        let var_4 = circuit_builder.deterministic_input(4);
        let var_8 = circuit_builder.deterministic_input(8);
        let var_9 = circuit_builder.deterministic_input(9);

        let four = circuit_builder.constant(4.into());

        let expr_circuit = (var_0 + var_4) * (var_8 - var_9) * four.clone() * four;

        // Verify that IDs are unique
        ConstraintCircuit::<InstructionTableChallenges>::assert_has_unique_ids(&mut [expr_circuit
            .clone()
            .consume()]);

        // Verify that partial evaluation agrees with the flat polynomial representation
        let expr_circuit_partial_evaluated = expr_circuit
            .partial_evaluate(&AllChallenges::placeholder().instruction_table_challenges);
        assert_eq!(expr_mpol, expr_circuit_partial_evaluated);
    }

    #[test]
    fn constant_folding_instruction_table_test() {
        let mut constraint_circuits = ExtInstructionTable::ext_transition_constraints_as_circuits();
        let mut before_fold: Vec<MPolynomial<XFieldElement>> = vec![];
        let challenges = AllChallenges::placeholder();
        for circuit in constraint_circuits.iter() {
            before_fold.push(circuit.partial_evaluate(&challenges.instruction_table_challenges));
        }

        ConstraintCircuit::constant_folding(&mut constraint_circuits.iter_mut().collect_vec());
        let mut after_fold: Vec<MPolynomial<XFieldElement>> = vec![];
        for circuit in constraint_circuits.iter() {
            after_fold.push(circuit.partial_evaluate(&challenges.instruction_table_challenges));
        }

        for (i, (before, after)) in before_fold.iter().zip_eq(after_fold.iter()).enumerate() {
            assert_eq!(before, after, "Constant folding must leave partially evaluated constraints unchanged for instruction table constraint {i}");
        }
    }

    #[test]
    fn constant_folding_processor_table_test() {
        let mut constraint_circuits = ExtProcessorTable::ext_transition_constraints_as_circuits();
        let mut before_fold: Vec<MPolynomial<XFieldElement>> = vec![];
        let challenges = AllChallenges::placeholder();
        for circuit in constraint_circuits.iter() {
            before_fold.push(circuit.partial_evaluate(&challenges.processor_table_challenges));
        }

        ConstraintCircuit::constant_folding(&mut constraint_circuits.iter_mut().collect_vec());
        let mut after_fold: Vec<MPolynomial<XFieldElement>> = vec![];
        for circuit in constraint_circuits.iter() {
            after_fold.push(circuit.partial_evaluate(&challenges.processor_table_challenges));
        }

        for (i, (before, after)) in before_fold.iter().zip_eq(after_fold.iter()).enumerate() {
            assert_eq!(before, after, "Constant folding must leave partially evaluated constraints unchanged for processor table constraint {i}");
        }
    }

    #[test]
    fn constant_folding_program_table_test() {
        let mut constraint_circuits = ExtProgramTable::ext_transition_constraints_as_circuits();
        let mut before_fold: Vec<MPolynomial<XFieldElement>> = vec![];
        let challenges = AllChallenges::placeholder();
        for circuit in constraint_circuits.iter() {
            before_fold.push(circuit.partial_evaluate(&challenges.program_table_challenges));
        }

        ConstraintCircuit::constant_folding(&mut constraint_circuits.iter_mut().collect_vec());
        let mut after_fold: Vec<MPolynomial<XFieldElement>> = vec![];
        for circuit in constraint_circuits.iter() {
            after_fold.push(circuit.partial_evaluate(&challenges.program_table_challenges));
        }

        for (i, (before, after)) in before_fold.iter().zip_eq(after_fold.iter()).enumerate() {
            assert_eq!(before, after, "Constant folding must leave partially evaluated constraints unchanged for processor table constraint {i}");
        }
    }
}
