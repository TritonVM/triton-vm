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
use rand::random;
use test_strategy::proptest;

use super::*;

impl<II: InputIndicator> ConstraintCircuitBuilder<II> {
//     // this is used in asserts in a function -- can be postponed
//     pub fn find_equivalent_nodes(&self) -> Vec<Vec<Rc<RefCell<ConstraintCircuit<II>>>>> {
//         let mut id_to_eval = HashMap::new();
//         let mut eval_to_ids = HashMap::new();
//         let mut id_to_node = HashMap::new();
//         Self::probe_random(
//             &self.circuit,
//             &mut id_to_eval,
//             &mut eval_to_ids,
//             &mut id_to_node,
//             random(),
//         );

//         eval_to_ids
//             .values()
//             .filter(|ids| ids.len() >= 2)
//             .map(|ids| ids.iter().map(|i| id_to_node[i].clone()).collect_vec())
//             .collect_vec()
//     }

//     /// Populate the dictionaries such that they associate with every node
//     /// in the circuit its evaluation in a random point. The inputs
//     /// are assigned random values. Equivalent nodes are detected
//     /// based on evaluating to the same value using the
//     /// Schwartz-Zippel lemma.
//     fn probe_random(
//         circuit: &Rc<RefCell<ConstraintCircuit<II>>>,
//         id_to_eval: &mut HashMap<usize, XFieldElement>,
//         eval_to_ids: &mut HashMap<XFieldElement, Vec<usize>>,
//         id_to_node: &mut HashMap<usize, Rc<RefCell<ConstraintCircuit<II>>>>,
//         master_seed: XFieldElement,
//     ) -> XFieldElement {
//         const DOMAIN_SEPARATOR_CURR_ROW: BFieldElement = BFieldElement::new(0);
//         const DOMAIN_SEPARATOR_NEXT_ROW: BFieldElement = BFieldElement::new(1);
//         const DOMAIN_SEPARATOR_CHALLENGE: BFieldElement = BFieldElement::new(2);

//         let circuit_id = circuit.borrow().id;
//         if let Some(&xfe) = id_to_eval.get(&circuit_id) {
//             return xfe;
//         }

//         let evaluation = match &circuit.borrow().expression {
//             CircuitExpression::BConst(bfe) => bfe.lift(),
//             CircuitExpression::XConst(xfe) => *xfe,
//             CircuitExpression::Input(input) => {
//                 let [s0, s1, s2] = master_seed.coefficients;
//                 let dom_sep = if input.is_current_row() {
//                     DOMAIN_SEPARATOR_CURR_ROW
//                 } else {
//                     DOMAIN_SEPARATOR_NEXT_ROW
//                 };
//                 let i = bfe!(u64::try_from(input.column()).unwrap());
//                 let Digest([d0, d1, d2, _, _]) = Tip5::hash_varlen(&[s0, s1, s2, dom_sep, i]);
//                 xfe!([d0, d1, d2])
//             }
//             CircuitExpression::Challenge(challenge) => {
//                 let [s0, s1, s2] = master_seed.coefficients;
//                 let dom_sep = DOMAIN_SEPARATOR_CHALLENGE;
//                 let ch = bfe!(u64::try_from(*challenge).unwrap());
//                 let Digest([d0, d1, d2, _, _]) = Tip5::hash_varlen(&[s0, s1, s2, dom_sep, ch]);
//                 xfe!([d0, d1, d2])
//             }
//             CircuitExpression::BinOp(bin_op, lhs, rhs) => {
//                 let l =
//                     Self::probe_random(lhs, id_to_eval, eval_to_ids, id_to_node, master_seed);
//                 let r =
//                     Self::probe_random(rhs, id_to_eval, eval_to_ids, id_to_node, master_seed);
//                 bin_op.operation(l, r)
//             }
//         };

//         id_to_eval.insert(circuit_id, evaluation);
//         eval_to_ids.entry(evaluation).or_default().push(circuit_id);
//         id_to_node.insert(circuit_id, circuit.clone());

//         evaluation
//     }
//     /// Produces an iter over all nodes in the multicircuit, if it is
//     /// non-empty.
//     ///
//     /// Helper function for counting the number of nodes of a specific type.
//     fn iter_nodes(
//         constraints: &[Self],
//     ) -> std::vec::IntoIter<(usize, ConstraintCircuitMonad<II>)> {
//         let Some(first) = constraints.first() else {
//             return vec![].into_iter();
//         };

//         first
//             .builder
//             .all_nodes
//             .borrow()
//             .iter()
//             .map(|(n, m)| (*n, m.clone()))
//             .collect_vec()
//             .into_iter()
//     }

//     /// The total number of nodes in the multicircuit
//     fn num_nodes(constraints: &[Self]) -> usize {
//         Self::iter_nodes(constraints).count()
//     }

//     /// Determine if the constraint circuit monad corresponds to a main
//     /// table column.
//     fn is_main_table_column(&self) -> bool {
//         let CircuitExpression::Input(ii) = self.circuit.borrow().expression else {
//             return false;
//         };

//         ii.is_main_table_column()
//     }

//     /// The number of inputs from the main table
//     fn num_main_inputs(constraints: &[Self]) -> usize {
//         Self::iter_nodes(constraints)
//             .filter(|(_, cc)| cc.is_main_table_column())
//             .filter(|(_, cc)| cc.circuit.borrow().evaluates_to_base_element())
//             .count()
//     }

//     /// The number of inputs from the aux table
//     fn num_aux_inputs(constraints: &[Self]) -> usize {
//         Self::iter_nodes(constraints)
//             .filter(|(_, cc)| !cc.is_main_table_column())
//             .filter(|(_, cc)| {
//                 matches!(cc.circuit.borrow().expression, CircuitExpression::Input(_))
//             })
//             .count()
//     }

//     /// The number of total (*i.e.*, main + aux) inputs
//     fn num_inputs(constraints: &[Self]) -> usize {
//         Self::num_main_inputs(constraints) + Self::num_aux_inputs(constraints)
//     }

//     /// The number of challenges
//     fn num_challenges(constraints: &[Self]) -> usize {
//         Self::iter_nodes(constraints)
//             .filter(|(_, cc)| {
//                 matches!(
//                     cc.circuit.borrow().expression,
//                     CircuitExpression::Challenge(_)
//                 )
//             })
//             .count()
//     }

//     /// The number of `BinOp`s
//     fn num_binops(constraints: &[Self]) -> usize {
//         Self::iter_nodes(constraints)
//             .filter(|(_, cc)| {
//                 matches!(
//                     cc.circuit.borrow().expression,
//                     CircuitExpression::BinOp(_, _, _)
//                 )
//             })
//             .count()
//     }

//     /// The number of BFE constants
//     fn num_bfield_constants(constraints: &[Self]) -> usize {
//         Self::iter_nodes(constraints)
//             .filter(|(_, cc)| {
//                 matches!(cc.circuit.borrow().expression, CircuitExpression::BConst(_))
//             })
//             .count()
//     }

//     /// The number of XFE constants
//     fn num_xfield_constants(constraints: &[Self]) -> usize {
//         Self::iter_nodes(constraints)
//             .filter(|(_, cc)| {
//                 matches!(
//                     cc.circuit.as_ref().borrow().expression,
//                     CircuitExpression::XConst(_)
//                 )
//             })
//             .count()
//     }
    fn contains(&self, self_id: usize, other_id: usize) -> bool {
        let self_expression = &self.nodes[self_id].expression;
        let other_expression = &self.nodes[other_id].expression;
        
        if self_expression == other_expression {
            return true;
        }
        let CircuitExpression::BinOp(_, lhs, rhs) = self_expression else {
            return false;
        };

        self.contains(*lhs, other_id) || self.contains(*rhs, other_id)
    }
}

/// The hash of a node may not depend on `ref_count`, `counter`, `id_counter_ref`, or `all_nodes`, since `all_nodes` contains the
/// digest of all nodes in the multi tree. For more details, see [`HashSet`].
// #[proptest]
// fn multi_circuit_hash_is_unchanged_by_meta_data(
//     #[strategy(arb())] circuit: ConstraintCircuitMonad<DualRowIndicator>,
//     new_ref_count: usize,
//     new_id_counter: usize,
// ) {
//     let original_digest = hash_circuit(&circuit);

//     circuit.circuit.borrow_mut().ref_count = new_ref_count;
//     prop_assert_eq!(original_digest, hash_circuit(&circuit));

//     circuit.builder.id_counter.replace(new_id_counter);
//     prop_assert_eq!(original_digest, hash_circuit(&circuit));
// }
// fn hash_circuit<II: InputIndicator>(circuit: &ConstraintCircuitMonad<II>) -> u64 {
//     let mut hasher = DefaultHasher::new();
//     circuit.hash(&mut hasher);
//     hasher.finish()
// }

// #[proptest]
// fn constant_folding_can_deal_with_multiplication_by_one(
//     #[strategy(arb())] c: ConstraintCircuit<DualRowIndicator>,
// ) {
//     let one = || c.builder.one();
//     prop_assert_eq!(c.clone(), c.clone() * one());
//     prop_assert_eq!(c.clone(), one() * c.clone());
//     prop_assert_eq!(c.clone(), one() * c.clone() * one());
// }

// #[proptest]
// fn constant_folding_can_deal_with_adding_zero(
//     #[strategy(arb())] c: ConstraintCircuitMonad<DualRowIndicator>,
// ) {
//     let zero = || c.builder.zero();
//     prop_assert_eq!(c.clone(), c.clone() + zero());
//     prop_assert_eq!(c.clone(), zero() + c.clone());
//     prop_assert_eq!(c.clone(), zero() + c.clone() + zero());
// }

// #[proptest]
// fn constant_folding_can_deal_with_subtracting_zero(
//     #[strategy(arb())] c: ConstraintCircuitMonad<DualRowIndicator>,
// ) {
//     prop_assert_eq!(c.clone(), c.clone() - c.builder.zero());
// }

// #[proptest]
// fn constant_folding_can_deal_with_adding_effectively_zero_term(
//     #[strategy(arb())] c: ConstraintCircuitMonad<DualRowIndicator>,
//     modification_selectors: [bool; 4],
// ) {
//     let zero = || c.builder.zero();
//     let mut redundant_circuit = c.clone();
//     if modification_selectors[0] {
//         redundant_circuit = redundant_circuit + (c.clone() * zero());
//     }
//     if modification_selectors[1] {
//         redundant_circuit = redundant_circuit + (zero() * c.clone());
//     }
//     if modification_selectors[2] {
//         redundant_circuit = (c.clone() * zero()) + redundant_circuit;
//     }
//     if modification_selectors[3] {
//         redundant_circuit = (zero() * c.clone()) + redundant_circuit;
//     }
//     prop_assert_eq!(c, redundant_circuit);
// }

// #[proptest]
// fn constant_folding_does_not_replace_0_minus_circuit_with_the_circuit(
//     #[strategy(arb())] circuit: ConstraintCircuitMonad<DualRowIndicator>,
// ) {
//     if circuit.circuit.borrow().is_zero() {
//         return Err(TestCaseError::Reject("0 - 0 actually is 0".into()));
//     }
//     let zero_minus_circuit = circuit.builder.zero() - circuit.clone();
//     prop_assert_ne!(&circuit, &zero_minus_circuit);
// }

/* show that degrades readability (and I don't see how to fix this)
but improves parallelization, usage of compiler safety checks, reuse of constants (like $-1$ or another) */
#[test]
fn pointer_redirection_obliviates_a_node_in_a_circuit() {
    let mut builder = ConstraintCircuitBuilder{nodes: Vec::new()};
    // let mut x_id = |i| builder.input(SingleRowIndicator::Main(i));
    // let mut constant_id = |c: u32| builder.b_constant(c);
    // let mut challenge_id = |i: usize| builder.challenge(i);

    let lhs = builder.input(SingleRowIndicator::Main(0));
    let rhs = builder.input(SingleRowIndicator::Main(1));
    let part_id = builder.binop(BinOp::Add, lhs, rhs);
    
    let lhs = builder.input(SingleRowIndicator::Main(0));
    let substitute_me_id = builder.binop(BinOp::Mul, lhs, part_id);

    let minus_one = builder.minus_one();
    
    let rhs = builder.challenge(1usize);
    let lhs = builder.binop(BinOp::Add, part_id, rhs);
    let rhs_inner = builder.b_constant(84);
    let rhs = builder.binop(BinOp::Mul, minus_one, rhs_inner);
    let root_0_id = builder.binop(BinOp::Add, lhs, rhs);
    
    let rhs = builder.challenge(0usize);
    let lhs = builder.binop(BinOp::Add, substitute_me_id, rhs);
    let rhs_inner = builder.b_constant(42);
    let rhs = builder.binop(BinOp::Mul, minus_one, rhs_inner);
    let root_1_id = builder.binop(BinOp::Add, lhs, rhs);

    let lhs_inner = builder.input(SingleRowIndicator::Main(2));
    let lhs = builder.binop(BinOp::Mul, lhs_inner, substitute_me_id);
    let rhs_inner = builder.challenge(1usize);
    let rhs = builder.binop(BinOp::Mul, minus_one, rhs_inner);
    let root_2_id = builder.binop(BinOp::Add, lhs, rhs);

    assert!(!builder.contains(root_0_id, substitute_me_id));
    assert!(builder.contains(root_1_id, substitute_me_id));
    assert!(builder.contains(root_2_id, substitute_me_id));

    let new_variable_id = builder.input(SingleRowIndicator::Main(3));
    builder.redirect_all_references_to_node(
        substitute_me_id,
        new_variable_id,
    );

    assert!(!builder.contains(root_0_id, substitute_me_id));
    assert!(!builder.contains(root_1_id, substitute_me_id));
    assert!(!builder.contains(root_2_id, substitute_me_id));

    assert!(builder.contains(root_0_id, part_id));
    assert!(builder.contains(root_1_id, new_variable_id));
    assert!(builder.contains(root_2_id, new_variable_id));
}

// #[test]
// fn simple_degree_lowering() {
//     let builder = ConstraintCircuitBuilder::new();
//     let x = || builder.input(SingleRowIndicator::Main(0));
//     let x_pow_3 = x() * x() * x();
//     let x_pow_5 = x() * x() * x() * x() * x();
//     let mut multicircuit = [x_pow_5, x_pow_3];

//     let degree_lowering_info = DegreeLoweringInfo {
//         target_degree: 3,
//         num_main_cols: 1,
//         num_aux_cols: 0,
//     };
//     let (new_main_constraints, new_aux_constraints) =
//         ConstraintCircuitMonad::lower_to_degree(&mut multicircuit, degree_lowering_info);

//     assert_eq!(1, new_main_constraints.len());
//     assert!(new_aux_constraints.is_empty());
// }

// #[test]
// fn somewhat_simple_degree_lowering() {
//     let builder = ConstraintCircuitBuilder::new();
//     let x = |i| builder.input(SingleRowIndicator::Main(i));
//     let y = |i| builder.input(SingleRowIndicator::Aux(i));
//     let b_con = |i: u64| builder.b_constant(i);

//     let constraint_0 = x(0) * x(0) * (x(1) - x(2)) - x(0) * x(2) - b_con(42);
//     let constraint_1 = x(1) * (x(1) - b_con(5)) * x(2) * (x(2) - b_con(1));
//     let constraint_2 = y(0)
//         * (b_con(2) * x(0) + b_con(3) * x(1) + b_con(4) * x(2))
//         * (b_con(4) * x(0) + b_con(8) * x(1) + b_con(16) * x(2))
//         - y(1);

//     let mut multicircuit = [constraint_0, constraint_1, constraint_2];

//     let degree_lowering_info = DegreeLoweringInfo {
//         target_degree: 2,
//         num_main_cols: 3,
//         num_aux_cols: 2,
//     };
//     let (new_main_constraints, new_aux_constraints) =
//         ConstraintCircuitMonad::lower_to_degree(&mut multicircuit, degree_lowering_info);

//     assert!(new_main_constraints.len() <= 3);
//     assert!(new_aux_constraints.len() <= 1);
// }

// #[test]
// fn less_simple_degree_lowering() {
//     let builder = ConstraintCircuitBuilder::new();
//     let x = |i| builder.input(SingleRowIndicator::Main(i));

//     let constraint_0 = (x(0) * x(1) * x(2)) * (x(3) * x(4)) * x(5);
//     let constraint_1 = (x(6) * x(7)) * (x(3) * x(4)) * x(8);

//     let mut multicircuit = [constraint_0, constraint_1];

//     let degree_lowering_info = DegreeLoweringInfo {
//         target_degree: 3,
//         num_main_cols: 9,
//         num_aux_cols: 0,
//     };
//     let (new_main_constraints, new_aux_constraints) =
//         ConstraintCircuitMonad::lower_to_degree(&mut multicircuit, degree_lowering_info);

//     assert!(new_main_constraints.len() <= 3);
//     assert!(new_aux_constraints.is_empty());
// }

// fn circuit_with_multiple_options_for_degree_lowering_to_degree_2()
// -> [ConstraintCircuitMonad<SingleRowIndicator>; 2] {
//     let builder = ConstraintCircuitBuilder::new();
//     let x = |i| builder.input(SingleRowIndicator::Main(i));

//     let constraint_0 = x(0) * x(0) * x(0);
//     let constraint_1 = x(1) * x(1) * x(1);

//     [constraint_0, constraint_1]
// }

// #[test]
// fn pick_node_to_substitute_is_deterministic() {
//     let multicircuit = circuit_with_multiple_options_for_degree_lowering_to_degree_2();
//     let first_node_id = ConstraintCircuitMonad::pick_node_to_substitute(&multicircuit, 2);

//     for _ in 0..20 {
//         let node_id_again = ConstraintCircuitMonad::pick_node_to_substitute(&multicircuit, 2);
//         assert_eq!(first_node_id, node_id_again);
//     }
// }

// #[test]
// fn degree_lowering_specific_simple_circuit_is_deterministic() {
//     let degree_lowering_info = DegreeLoweringInfo {
//         target_degree: 2,
//         num_main_cols: 2,
//         num_aux_cols: 0,
//     };

//     let mut original_multicircuit =
//         circuit_with_multiple_options_for_degree_lowering_to_degree_2();
//     let (new_main_constraints, _) = ConstraintCircuitMonad::lower_to_degree(
//         &mut original_multicircuit,
//         degree_lowering_info,
//     );

//     for _ in 0..20 {
//         let mut new_multicircuit =
//             circuit_with_multiple_options_for_degree_lowering_to_degree_2();
//         let (new_main_constraints_again, _) = ConstraintCircuitMonad::lower_to_degree(
//             &mut new_multicircuit,
//             degree_lowering_info,
//         );
//         assert_eq!(new_main_constraints, new_main_constraints_again);
//         assert_eq!(original_multicircuit, new_multicircuit);
//     }
// }

// #[test]
// fn all_nodes_in_multicircuit_are_identified_correctly() {
//     let builder = ConstraintCircuitBuilder::new();

//     let x = |i| builder.input(SingleRowIndicator::Main(i));
//     let b_con = |i: u64| builder.b_constant(i);

//     let sub_tree_0 = x(0) * x(1) * (x(2) - b_con(1)) * x(3) * x(4);
//     let sub_tree_1 = x(0) * x(1) * (x(2) - b_con(1)) * x(3) * x(5);
//     let sub_tree_2 = x(10) * x(10) * x(2) * x(13);
//     let sub_tree_3 = x(10) * x(10) * x(2) * x(14);

//     let circuit_0 = sub_tree_0.clone() + sub_tree_1.clone();
//     let circuit_1 = sub_tree_2.clone() + sub_tree_3.clone();
//     let circuit_2 = sub_tree_0 + sub_tree_2;
//     let circuit_3 = sub_tree_1 + sub_tree_3;

//     let multicircuit = [circuit_0, circuit_1, circuit_2, circuit_3].map(|c| c.consume());

//     let all_nodes = ConstraintCircuitMonad::all_nodes_in_multicircuit(&multicircuit);
//     let count_node = |node| all_nodes.iter().filter(|&n| n == &node).count();

//     let x0 = x(0).consume();
//     assert_eq!(4, count_node(x0));

//     let x2 = x(2).consume();
//     assert_eq!(8, count_node(x2));

//     let x10 = x(10).consume();
//     assert_eq!(8, count_node(x10));

//     let x4 = x(4).consume();
//     assert_eq!(2, count_node(x4));

//     let x6 = x(6).consume();
//     assert_eq!(0, count_node(x6));

//     let x0_x1 = (x(0) * x(1)).consume();
//     assert_eq!(4, count_node(x0_x1));

//     let tree = (x(0) * x(1) * (x(2) - b_con(1))).consume();
//     assert_eq!(4, count_node(tree));

//     let max_occurrences = all_nodes
//         .iter()
//         .map(|node| all_nodes.iter().filter(|&n| n == node).count())
//         .max()
//         .unwrap();
//     assert_eq!(8, max_occurrences);

//     let most_frequent_nodes = all_nodes
//         .iter()
//         .filter(|&node| all_nodes.iter().filter(|&n| n == node).count() == max_occurrences)
//         .unique()
//         .collect_vec();
//     assert_eq!(2, most_frequent_nodes.len());
//     assert!(most_frequent_nodes.contains(&&x(2).consume()));
//     assert!(most_frequent_nodes.contains(&&x(10).consume()));
// }

// #[test]
// fn equivalent_nodes_are_detected_when_present() {
//     let builder = ConstraintCircuitBuilder::new();

//     let x = |i| builder.input(SingleRowIndicator::Main(i));
//     let ch = |i: usize| builder.challenge(i);

//     let u0 = x(0) + x(1);
//     let u1 = x(2) + x(3);
//     let v = u0 * u1;

//     let z0 = x(0) * x(2);
//     let z2 = x(1) * x(3);

//     let z1 = x(1) * x(2) + x(0) * x(3);
//     let w = v - z0 - z2;
//     assert!(w.find_equivalent_nodes().is_empty());

//     let o = ch(0) * z1 - ch(1) * w;
//     assert!(!o.find_equivalent_nodes().is_empty());
// }

// #[derive(Debug, Copy, Clone, Eq, PartialEq, test_strategy::Arbitrary)]
// enum CircuitOperationChoice {
//     Add(usize, usize),
//     Mul(usize, usize),
// }

// #[derive(Debug, Copy, Clone, Eq, PartialEq, test_strategy::Arbitrary)]
// enum CircuitInputType {
//     Main,
//     Aux,
// }

// #[derive(Debug, Copy, Clone, Eq, PartialEq, test_strategy::Arbitrary)]
// enum CircuitConstantType {
//     Base(#[strategy(arb())] BFieldElement),
//     Extension(#[strategy(arb())] XFieldElement),
// }

// fn arbitrary_circuit_monad<II: InputIndicator>(
//     num_inputs: usize,
//     num_challenges: usize,
//     num_constants: usize,
//     num_operations: usize,
//     num_outputs: usize,
// ) -> BoxedStrategy<Vec<ConstraintCircuitMonad<II>>> {
//     (
//         vec(CircuitInputType::arbitrary(), num_inputs),
//         vec(CircuitConstantType::arbitrary(), num_constants),
//         vec(CircuitOperationChoice::arbitrary(), num_operations),
//         vec(arb::<usize>(), num_outputs),
//     )
//         .prop_map(move |(inputs, constants, operations, outputs)| {
//             let builder = ConstraintCircuitBuilder::<II>::new();

//             assert_eq!(0, *builder.id_counter.borrow());
//             assert!(
//                 builder.all_nodes.borrow().is_empty(),
//                 "fresh hashmap should be empty"
//             );

//             let mut num_main_inputs = 0;
//             let mut num_aux_inputs = 0;
//             let mut all_nodes = vec![];
//             let mut output_nodes = vec![];

//             for input in inputs {
//                 match input {
//                     CircuitInputType::Main => {
//                         let node = builder.input(II::main_table_input(num_main_inputs));
//                         num_main_inputs += 1;
//                         all_nodes.push(node);
//                     }
//                     CircuitInputType::Aux => {
//                         let node = builder.input(II::aux_table_input(num_aux_inputs));
//                         num_aux_inputs += 1;
//                         all_nodes.push(node);
//                     }
//                 }
//             }

//             for i in 0..num_challenges {
//                 let node = builder.challenge(i);
//                 all_nodes.push(node);
//             }

//             for constant in constants {
//                 let node = match constant {
//                     CircuitConstantType::Base(bfe) => builder.b_constant(bfe),
//                     CircuitConstantType::Extension(xfe) => builder.x_constant(xfe),
//                 };
//                 all_nodes.push(node);
//             }

//             if all_nodes.is_empty() {
//                 return vec![];
//             }

//             for operation in operations {
//                 let (lhs, rhs) = match operation {
//                     CircuitOperationChoice::Add(lhs, rhs) => (lhs, rhs),
//                     CircuitOperationChoice::Mul(lhs, rhs) => (lhs, rhs),
//                 };

//                 let lhs_index = lhs % all_nodes.len();
//                 let rhs_index = rhs % all_nodes.len();

//                 let lhs_node = all_nodes[lhs_index].clone();
//                 let rhs_node = all_nodes[rhs_index].clone();

//                 let node = match operation {
//                     CircuitOperationChoice::Add(_, _) => lhs_node + rhs_node,
//                     CircuitOperationChoice::Mul(_, _) => lhs_node * rhs_node,
//                 };
//                 all_nodes.push(node);
//             }

//             for output in outputs {
//                 let index = output % all_nodes.len();
//                 output_nodes.push(all_nodes[index].clone());
//             }

//             output_nodes
//         })
//         .boxed()
// }

// #[proptest]
// fn node_type_counts_add_up(
//     #[strategy(arbitrary_circuit_monad(10, 10, 10, 60, 10))] multicircuit_monad: Vec<
//         ConstraintCircuitMonad<SingleRowIndicator>,
//     >,
// ) {
//     prop_assert_eq!(
//         ConstraintCircuitMonad::num_nodes(&multicircuit_monad),
//         ConstraintCircuitMonad::num_main_inputs(&multicircuit_monad)
//             + ConstraintCircuitMonad::num_aux_inputs(&multicircuit_monad)
//             + ConstraintCircuitMonad::num_challenges(&multicircuit_monad)
//             + ConstraintCircuitMonad::num_bfield_constants(&multicircuit_monad)
//             + ConstraintCircuitMonad::num_xfield_constants(&multicircuit_monad)
//             + ConstraintCircuitMonad::num_binops(&multicircuit_monad)
//     );

//     prop_assert_eq!(10, ConstraintCircuitMonad::num_inputs(&multicircuit_monad));
// }

// /// Test the completeness and soundness of the `apply_substitution`
// /// function, which substitutes a single node.
// ///
// /// In this context, completeness means: "given a satisfying assignment to
// /// the circuit before degree lowering, one can derive a satisfying
// /// assignment to the circuit after degree lowering." Soundness means
// /// the converse.
// ///
// /// We test these features using random input vectors. Naturally, the output
// /// is not the zero vector (with high probability) and so the given input is
// /// *not* a satisfying assignment (with high probability). However, the
// /// circuit can be extended by way of thought experiment into one that
// /// subtracts a fixed constant from the original output. For the right
// /// choice of subtrahend, the random input now *is* a satisfying
// /// assignment to the circuit.
// ///
// /// Specifically, let `input` denote the original (before degree lowering)
// /// input, and `C` the circuit. Then `input` is a satisfying input for
// /// the new circuit `C'(X) = C(X) - C(input)`
// ///
// /// After applying a substitution to obtain circuit `C || k` from `C`, where
// /// `k = Z - some_expr(X)` and `Z` is the introduced variable, the length
// /// of the input and output increases by 1. Moreover, if `input` is a
// /// satisfying input to `C'` then `input || some_expr(input)` is a
// /// satisfying input to `C' || k` (given the transformation is complete).
// ///
// /// To establish the converse, we want to start from a satisfying input to
// /// `C" || k` and reduce it to a satisfying input to `C"`. The requirement,
// /// implied by "satisfying input", that `k(X || Z) == 0` implies `Z ==
// /// some_expr(X)`. Therefore, in order to sample a random satisfying
// /// input to `C" || k`, it suffices to select `input` at random, define
// /// `C"(X) = C(X) - C(input)`, and evaluate `some_expr(input)`. Then
// /// `input || some_expr(input)` is a random satisfying input to `C" ||
// /// k`. It follows** that `input` is a satisfying input to `C"` (given the transformation is sound).
// ///
// /// This description makes use of the following commutative diagram.
// ///
// /// ```text
// ///          C ───── degree-lowering ────> C || k
// ///          │                               │
// /// subtract │                      subtract │
// ///    fixed │                         fixed │
// ///   output │                        output │
// ///          │                               │
// ///          v                               v
// ///          C* ─── degree-lowering ────> C* || k
// /// ```
// ///
// /// The point of this elaboration is that in this particular case, testing
// /// completeness and soundness require the same code path. If `input`
// /// and `input || some_expr(input)` work for circuits before and after
// /// degree lowering, this fact establishes both completeness and
// /// soundness simultaneously.
// //
// // Shrinking on this test is disabled because we noticed some weird ass
// // behavior. In short, shrinking does not play ball with the arbitrary
// // circuit generator; it seems to make the generated circuits *more*
// // complex, not less so.
// #[proptest(cases = 1000, max_shrink_iters = 0)]
// fn node_substitution_is_complete_and_sound(
//     #[strategy(arbitrary_circuit_monad(10, 10, 10, 160, 10))] mut multicircuit_monad: Vec<
//         ConstraintCircuitMonad<SingleRowIndicator>,
//     >,
//     #[strategy(vec(arb(), ConstraintCircuitMonad::num_main_inputs(&#multicircuit_monad)))]
//     #[filter(!#main_input.is_empty())]
//     main_input: Vec<BFieldElement>,
//     #[strategy(vec(arb(), ConstraintCircuitMonad::num_aux_inputs(&#multicircuit_monad)))]
//     #[filter(!#aux_input.is_empty())]
//     aux_input: Vec<XFieldElement>,
//     #[strategy(vec(arb(), ConstraintCircuitMonad::num_challenges(&#multicircuit_monad)))]
//     challenges: Vec<XFieldElement>,
//     #[strategy(arb())] substitution_node_index: usize,
// ) {
//     let mut main_input = Array2::from_shape_vec((1, main_input.len()), main_input).unwrap();
//     let mut aux_input = Array2::from_shape_vec((1, aux_input.len()), aux_input).unwrap();

//     let output_before_lowering = multicircuit_monad
//         .iter()
//         .map(|m| m.circuit.borrow())
//         .map(|c| c.evaluate(main_input.view(), aux_input.view(), &challenges))
//         .collect_vec();

//     // apply one step of degree-lowering
//     let num_nodes = ConstraintCircuitMonad::num_nodes(&multicircuit_monad);
//     let &substitution_node_id = multicircuit_monad[0]
//         .builder
//         .all_nodes
//         .borrow()
//         .iter()
//         .cycle()
//         .skip(substitution_node_index % num_nodes)
//         .take(num_nodes)
//         .find_map(|(id, monad)| monad.circuit.borrow().is_zero().not().then_some(id))
//         .expect("no suitable nodes to substitute");

//     let degree_lowering_info = DegreeLoweringInfo {
//         target_degree: 2,
//         num_main_cols: main_input.len(),
//         num_aux_cols: aux_input.len(),
//     };
//     let substitution_constraint = ConstraintCircuitMonad::apply_substitution(
//         &mut multicircuit_monad,
//         degree_lowering_info,
//         substitution_node_id,
//         EvolvingMainConstraintsNumber(0),
//         EvolvingAuxConstraintsNumber(0),
//     );

//     // extract substituted constraint
//     let CircuitExpression::BinOp(BinOp::Add, variable, neg_expression) =
//         &substitution_constraint.circuit.borrow().expression
//     else {
//         panic!();
//     };
//     let extra_input =
//         match &neg_expression.borrow().expression {
//             CircuitExpression::BinOp(BinOp::Mul, _neg_one, circuit) => circuit
//                 .borrow()
//                 .evaluate(main_input.view(), aux_input.view(), &challenges),
//             CircuitExpression::BConst(c) => -c.lift(),
//             CircuitExpression::XConst(c) => -*c,
//             _ => panic!(),
//         };
//     if variable.borrow().evaluates_to_base_element() {
//         let extra_input = extra_input.unlift().unwrap();
//         let extra_input = Array2::from_shape_vec([1, 1], vec![extra_input]).unwrap();
//         main_input.append(Axis(1), extra_input.view()).unwrap();
//     } else {
//         let extra_input = Array2::from_shape_vec([1, 1], vec![extra_input]).unwrap();
//         aux_input.append(Axis(1), extra_input.view()).unwrap();
//     }

//     // evaluate again
//     let output_after_lowering = multicircuit_monad
//         .iter()
//         .map(|m| m.circuit.borrow())
//         .map(|c| c.evaluate(main_input.view(), aux_input.view(), &challenges))
//         .collect_vec();
//     prop_assert_eq!(output_before_lowering, output_after_lowering);

//     let evaluated_constraint = substitution_constraint.circuit.borrow().evaluate(
//         main_input.view(),
//         aux_input.view(),
//         &challenges,
//     );
//     prop_assert!(evaluated_constraint.is_zero());
// }