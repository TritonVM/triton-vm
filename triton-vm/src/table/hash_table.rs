use super::base_table::{self, InheritsFromTable, Table, TableLike};
use super::challenges_endpoints::{AllChallenges, AllEndpoints};
use super::extension_table::{ExtensionTable, Quotientable, QuotientableExtensionTable};
use super::table_column::HashTableColumn;
use crate::fri_domain::FriDomain;
use crate::state::DIGEST_LEN;
use crate::table::base_table::Extendable;
use crate::table::extension_table::Evaluable;
use crate::table::table_column::HashTableColumn::*;
use itertools::Itertools;
use std::ops::Mul;
use twenty_first::shared_math::b_field_element::BFieldElement;
use twenty_first::shared_math::mpolynomial::{Degree, MPolynomial};
use twenty_first::shared_math::polynomial::Polynomial;
use twenty_first::shared_math::x_field_element::XFieldElement;

pub const HASH_TABLE_PERMUTATION_ARGUMENTS_COUNT: usize = 0;
pub const HASH_TABLE_EVALUATION_ARGUMENT_COUNT: usize = 2;
pub const HASH_TABLE_INITIALS_COUNT: usize =
    HASH_TABLE_PERMUTATION_ARGUMENTS_COUNT + HASH_TABLE_EVALUATION_ARGUMENT_COUNT;

/// This is 18 because it combines: 12 stack_input_weights and 6 digest_output_weights.
pub const HASH_TABLE_EXTENSION_CHALLENGE_COUNT: usize = 18;

/// The number of constants used in each round of the permutation. Since Rescue Prime uses one round
/// constant per half-round, this number is twice the number of state elements.
pub const NUM_ROUND_CONSTANTS: usize = 32;

/// The number of rounds for Rescue Prime
pub const NUM_ROUNDS: usize = 8;

/// Capacity of Rescue Prime
pub const CAPACITY: usize = 4;

pub const BASE_WIDTH: usize = 49;
pub const FULL_WIDTH: usize = 53; // BASE_WIDTH + 2 * INITIALS_COUNT

pub const TOTAL_NUM_CONSTANTS: usize = NUM_ROUND_CONSTANTS * NUM_ROUNDS;
pub const ROUND_CONSTANTS: [u64; TOTAL_NUM_CONSTANTS] = [
    15139912583685767368,
    8372387753867525709,
    2183680717104184380,
    3244606960098905893,
    3147881977364597901,
    9452780775072264938,
    1082537058754139762,
    10970853375448447283,
    3062104324741241281,
    18009675040823690122,
    9709134112189744652,
    15857062738397365943,
    5016225506033072343,
    5216859395468346115,
    6462263614532073214,
    1493656999465165663,
    828768000476271392,
    262568612853428171,
    10113927829938360011,
    3228404466757125020,
    7320852123907649631,
    13627426656786462355,
    7964883404857794874,
    1407934150297697997,
    17336604982330804394,
    17906014506034551057,
    4632709206831589562,
    12999797106063314512,
    17162978498471467904,
    6827540927719713380,
    4753504633679017533,
    17716852809995758525,
    8549423660797843647,
    2362390356169006813,
    16716828864075537528,
    2740683348482332949,
    7756193835844677826,
    17543799665801483121,
    15002804793384601632,
    7902645524886711764,
    15165733099428544473,
    4077635361197762831,
    15132376188215154091,
    10741861618481937993,
    13707397012333257757,
    14226034480467186519,
    18245513484961172378,
    13273670281248631122,
    18251304196568320201,
    18190580491784411188,
    6118572220412064319,
    5630770511111509423,
    7970516069264861936,
    13449271048822160788,
    6851697376735269367,
    17987627018199535376,
    5294172762355915266,
    13844513406523115704,
    14597636171777994036,
    6061614115452481739,
    8186070796010445225,
    2327693164544063482,
    855935718254855095,
    10009207201287677622,
    10381177680823887718,
    18166133947715927863,
    17760506907335165396,
    3370764898316519938,
    5201580129905804035,
    1620223121525450629,
    14461318317868382163,
    1250929940922089768,
    13370612866774614255,
    7175470036866504098,
    16421684582717699126,
    16644320598987600726,
    17802798266780789487,
    6974241949143442442,
    17591712720223212489,
    16201733676622149735,
    286099893890784288,
    8057298197517276497,
    6444512502860040579,
    8347461167435943315,
    17352444114675313421,
    13535064425127211380,
    4772591666336791434,
    427985333074531866,
    14141791479819390154,
    7028809244427084468,
    9426904145082569174,
    6166111020063614179,
    8951223707117953234,
    3431064000345231130,
    1944155315841337325,
    6285600810995398496,
    16897111123465175857,
    4660909896474179791,
    18192626343736320364,
    5057838432340191471,
    14014302776583938723,
    9925254923879301551,
    6829435345780265556,
    8968794115294201104,
    17778545491689490446,
    18017797995365371861,
    18060766500386119579,
    12896732587303423715,
    4187616244444972880,
    10797712368247465599,
    5551515461716974377,
    5987237400880775150,
    8306936493309794552,
    10555482202024602033,
    16045656883318709119,
    14224667772707921698,
    7464515010550790466,
    14683637456755672385,
    8606694398702844028,
    12783325878688361611,
    10135605311909694521,
    6036681888442161456,
    13502595716772524386,
    17837288544072949135,
    16970790481274575856,
    12771951327386638665,
    7953144665513487435,
    10232601596097265370,
    7142562723872426447,
    7061326483481627814,
    2700322576799317485,
    6623246769381195291,
    16825539912038364772,
    17345255259493544461,
    3655344217194071236,
    4906781818047525714,
    14897453143374918047,
    12697105275305687091,
    6365510487307614865,
    16389921370395602280,
    6184292348425681997,
    1625734039805583227,
    7926303851971506844,
    6764450482313517598,
    12861725371095466098,
    1457318443242363431,
    6401144276852156944,
    11758577537140385015,
    7035279949079298611,
    17490109387633149109,
    9028549762556146425,
    14629064429955990677,
    7345978731773547933,
    2380447650891770049,
    13946626261179506153,
    14112757565552107369,
    18323048004349754740,
    3761840715850313303,
    2423761811055022202,
    4043073367058340954,
    4714747831891079272,
    9903324717460101691,
    16489681373737990564,
    12205142203164019145,
    7650721966187356479,
    13176636867741415622,
    8725940740195977648,
    7850051922002287223,
    7013216436240322065,
    7521500899942431357,
    17948709915499568560,
    12709968715340313663,
    12864870176028239567,
    13835492971050856940,
    14117813659377608536,
    17930389253653738705,
    16665999642411270287,
    8522764273244228281,
    17022385114623716012,
    17792533099449144220,
    9666141708192493561,
    4101243295111900354,
    11110149680511328320,
    15833373900081216881,
    2858902809543644288,
    15185937040593697757,
    1229742010402781808,
    12488915253710643809,
    14449445461821352645,
    11702325210632962260,
    7390229042372607295,
    13724660230648496560,
    16370078900053649525,
    6897898366117786971,
    12564585209779431146,
    15916465850680923114,
    3497319829092809455,
    3681935191724738445,
    17269401177087593182,
    14149218837807091766,
    13453529877505970461,
    15298165362714239682,
    14728462634044980354,
    14409721890326796259,
    17353894810846356075,
    16857127813837277773,
    11187357872695367332,
    15533140707195072093,
    1163405869960896591,
    15296392010875874377,
    17872716265685676772,
    14706935000063347212,
    14502717840925123585,
    1458466805797611569,
    2849079512899132391,
    14109081278228167673,
    8933669600131241369,
    8173386480957668450,
    15252826729106121549,
    10128993114764423519,
    11364771171604097376,
    14762095736262922188,
    13319725258546020263,
    16948750294723703018,
    10039494505766092885,
    14730563960989205668,
    16314543682302146762,
    13412588491336542421,
    5973689466852663000,
    673906515894578274,
    4039316712345686736,
    2031308080490921066,
    2907338798762025874,
    12316517814797934964,
    9307548410347506674,
    9351070955954520832,
    5794230072435402060,
    7922269617708021679,
    9708384153023840180,
    16472577099676318887,
    5244055413069805590,
    18123735486382626662,
    6519538476295982160,
    14228372996780660309,
    7960505044283116493,
    13993750470080027634,
    11478414004339098168,
    5009409638864158506,
    15807366605352652129,
    10685686439628572285,
    6800403862825412390,
    13138657193944784618,
    6448410590255081786,
    4381763274661386195,
    3646572817684127401,
    2916928929409428212,
];

#[derive(Debug, Clone)]
pub struct HashTable {
    inherited_table: Table<BFieldElement>,
}

impl InheritsFromTable<BFieldElement> for HashTable {
    fn inherited_table(&self) -> &Table<BFieldElement> {
        &self.inherited_table
    }

    fn mut_inherited_table(&mut self) -> &mut Table<BFieldElement> {
        &mut self.inherited_table
    }
}

#[derive(Debug, Clone)]
pub struct ExtHashTable {
    inherited_table: Table<XFieldElement>,
}

impl Evaluable for ExtHashTable {
    fn evaluate_consistency_constraints(
        &self,
        evaluation_point: &[XFieldElement],
    ) -> Vec<XFieldElement> {
        let round_number = evaluation_point[ROUNDNUMBER as usize];
        let state12 = evaluation_point[STATE12 as usize];
        let state13 = evaluation_point[STATE13 as usize];
        let state14 = evaluation_point[STATE14 as usize];
        let state15 = evaluation_point[STATE15 as usize];

        let round_number_is_not_1_or = (0..=8)
            .filter(|&r| r != 1)
            .map(|r| round_number - r.into())
            .fold(1.into(), XFieldElement::mul);

        let mut evaluated_consistency_constraints = vec![
            round_number_is_not_1_or * state12,
            round_number_is_not_1_or * state13,
            round_number_is_not_1_or * state14,
            round_number_is_not_1_or * state15,
        ];

        let round_constant_offset = CONSTANT0A as usize;
        for round_constant_idx in 0..NUM_ROUND_CONSTANTS {
            let round_constant_column: HashTableColumn =
                (round_constant_idx + round_constant_offset).into();
            evaluated_consistency_constraints.push(
                round_number
                    * (Self::round_constants_interpolant(round_constant_column)
                        .evaluate(&evaluation_point[ROUNDNUMBER as usize])
                        - evaluation_point[round_constant_column as usize]),
            );
        }

        evaluated_consistency_constraints
    }
}

impl Quotientable for ExtHashTable {
    fn get_consistency_quotient_degree_bounds(&self) -> Vec<Degree> {
        let capacity_degree_bounds =
            vec![self.interpolant_degree() * (NUM_ROUNDS + 1) as Degree; CAPACITY];
        let round_constant_degree_bounds =
            vec![self.interpolant_degree() * (NUM_ROUNDS + 1) as Degree; NUM_ROUND_CONSTANTS];
        [capacity_degree_bounds, round_constant_degree_bounds].concat()
    }
}

impl QuotientableExtensionTable for ExtHashTable {}

impl InheritsFromTable<XFieldElement> for ExtHashTable {
    fn inherited_table(&self) -> &Table<XFieldElement> {
        &self.inherited_table
    }

    fn mut_inherited_table(&mut self) -> &mut Table<XFieldElement> {
        &mut self.inherited_table
    }
}

impl TableLike<BFieldElement> for HashTable {}

impl Extendable for HashTable {
    fn get_padding_rows(&self) -> (Option<usize>, Vec<Vec<BFieldElement>>) {
        (None, vec![vec![0.into(); BASE_WIDTH]])
    }
}

impl TableLike<XFieldElement> for ExtHashTable {}

impl ExtHashTable {
    fn ext_boundary_constraints() -> Vec<MPolynomial<XFieldElement>> {
        let one = MPolynomial::from_constant(1.into(), FULL_WIDTH);
        let variables = MPolynomial::variables(FULL_WIDTH, 1.into());

        let round_number = variables[ROUNDNUMBER as usize].clone();
        let round_number_is_0_or_1 = round_number.clone() * (round_number - one);
        vec![round_number_is_0_or_1]
    }

    /// The implementation below is kept around for debugging purposes. This table evaluates the
    /// corresponding constraints directly by implementing the respective method in trait
    /// `Evaluable`, and does not use the polynomials below.
    fn ext_consistency_constraints() -> Vec<MPolynomial<XFieldElement>> {
        let constant = |c: u32| MPolynomial::from_constant(c.into(), FULL_WIDTH);
        let variables = MPolynomial::variables(FULL_WIDTH, 1.into());

        let round_number = variables[ROUNDNUMBER as usize].clone();
        let state12 = variables[STATE12 as usize].clone();
        let state13 = variables[STATE13 as usize].clone();
        let state14 = variables[STATE14 as usize].clone();
        let state15 = variables[STATE15 as usize].clone();

        let round_number_is_not_1_or = (0..=8)
            .filter(|&r| r != 1)
            .map(|r| round_number.clone() - constant(r))
            .fold(constant(1), MPolynomial::mul);

        let mut consistency_polynomials = vec![
            round_number_is_not_1_or.clone() * state12,
            round_number_is_not_1_or.clone() * state13,
            round_number_is_not_1_or.clone() * state14,
            round_number_is_not_1_or * state15,
        ];

        let round_constant_offset = CONSTANT0A as usize;
        for round_constant_idx in 0..NUM_ROUND_CONSTANTS {
            let round_constant_column: HashTableColumn =
                (round_constant_idx + round_constant_offset).into();
            let interpolant = Self::round_constants_interpolant(round_constant_column);
            let multivariate_interpolant =
                MPolynomial::lift(interpolant, ROUNDNUMBER as usize, FULL_WIDTH);
            consistency_polynomials.push(
                round_number.clone()
                    * (multivariate_interpolant
                        - variables[round_constant_column as usize].clone()),
            );
        }

        consistency_polynomials
    }

    fn round_constants_interpolant(round_constant: HashTableColumn) -> Polynomial<XFieldElement> {
        let round_constant_idx = (round_constant as usize) - (CONSTANT0A as usize);
        let domain = (1..=NUM_ROUNDS)
            .map(|x| BFieldElement::new(x as u64).lift())
            .collect_vec();
        let abscissae = (1..=NUM_ROUNDS)
            .map(|i| ROUND_CONSTANTS[NUM_ROUND_CONSTANTS * (i - 1) + round_constant_idx])
            .map(|x| BFieldElement::new(x).lift())
            .collect_vec();
        Polynomial::lagrange_interpolate(&domain, &abscissae)
    }

    fn ext_transition_constraints(
        _challenges: &HashTableChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        let constant = |c: u32| MPolynomial::from_constant(c.into(), 2 * FULL_WIDTH);
        let variables = MPolynomial::variables(2 * FULL_WIDTH, 1.into());

        let round_number = variables[ROUNDNUMBER as usize].clone();
        let round_number_next = variables[FULL_WIDTH + ROUNDNUMBER as usize].clone();

        let if_round_number_is_x_then_round_number_next_is_y = |x, y| {
            (0..=8)
                .filter(|&r| r != x)
                .map(|r| round_number.clone() - constant(r))
                .fold(round_number_next.clone() - constant(y), MPolynomial::mul)
        };

        vec![
            if_round_number_is_x_then_round_number_next_is_y(0, 0),
            if_round_number_is_x_then_round_number_next_is_y(1, 2),
            if_round_number_is_x_then_round_number_next_is_y(2, 3),
            if_round_number_is_x_then_round_number_next_is_y(3, 4),
            if_round_number_is_x_then_round_number_next_is_y(4, 5),
            if_round_number_is_x_then_round_number_next_is_y(5, 6),
            if_round_number_is_x_then_round_number_next_is_y(6, 7),
            if_round_number_is_x_then_round_number_next_is_y(7, 8),
            // if round number is 8, then round number next is 0 or 1
            if_round_number_is_x_then_round_number_next_is_y(8, 0)
                * (round_number_next - constant(1)),
            // todo: The remaining 7·16 = 112 constraints are left as an exercise to the reader.
        ]
    }

    fn ext_terminal_constraints(
        _challenges: &HashTableChallenges,
        _terminals: &HashTableEndpoints,
    ) -> Vec<MPolynomial<XFieldElement>> {
        vec![]
    }
}

impl HashTable {
    pub fn new_prover(num_trace_randomizers: usize, matrix: Vec<Vec<BFieldElement>>) -> Self {
        let unpadded_height = matrix.len();
        let padded_height = base_table::padded_height(unpadded_height);

        let omicron = base_table::derive_omicron(padded_height as u64);
        let inherited_table = Table::new(
            BASE_WIDTH,
            FULL_WIDTH,
            padded_height,
            num_trace_randomizers,
            omicron,
            matrix,
            "HashTable".to_string(),
        );

        Self { inherited_table }
    }

    pub fn codeword_table(&self, fri_domain: &FriDomain<BFieldElement>) -> Self {
        let base_columns = 0..self.base_width();
        let codewords = self.low_degree_extension(fri_domain, base_columns);

        let inherited_table = self.inherited_table.with_data(codewords);
        Self { inherited_table }
    }

    pub fn extend(
        &self,
        challenges: &HashTableChallenges,
        initials: &HashTableEndpoints,
    ) -> (ExtHashTable, HashTableEndpoints) {
        let mut from_processor_running_sum = initials.from_processor_eval_sum;
        let mut to_processor_running_sum = initials.to_processor_eval_sum;

        let mut extension_matrix: Vec<Vec<XFieldElement>> = Vec::with_capacity(self.data().len());
        for row in self.data().iter() {
            let mut extension_row = Vec::with_capacity(FULL_WIDTH);
            extension_row.extend(row.iter().map(|elem| elem.lift()));

            // Compress input values into single value (independent of round index)
            let state_for_input = [
                extension_row[HashTableColumn::STATE0 as usize],
                extension_row[HashTableColumn::STATE1 as usize],
                extension_row[HashTableColumn::STATE2 as usize],
                extension_row[HashTableColumn::STATE3 as usize],
                extension_row[HashTableColumn::STATE4 as usize],
                extension_row[HashTableColumn::STATE5 as usize],
                extension_row[HashTableColumn::STATE6 as usize],
                extension_row[HashTableColumn::STATE7 as usize],
                extension_row[HashTableColumn::STATE8 as usize],
                extension_row[HashTableColumn::STATE9 as usize],
                extension_row[HashTableColumn::STATE10 as usize],
                extension_row[HashTableColumn::STATE11 as usize],
            ];
            let compressed_state_for_input = state_for_input
                .iter()
                .zip(challenges.stack_input_weights.iter())
                .map(|(state, weight)| *weight * *state)
                .fold(XFieldElement::ring_zero(), |sum, summand| sum + summand);
            extension_row.push(compressed_state_for_input);

            // Add compressed input to running sum if round index marks beginning of hashing
            extension_row.push(from_processor_running_sum);
            if row[HashTableColumn::ROUNDNUMBER as usize].value() == 1 {
                from_processor_running_sum = from_processor_running_sum
                    * challenges.from_processor_eval_row_weight
                    + compressed_state_for_input;
            }

            // Compress digest values into single value (independent of round index)
            let state_for_output = [
                extension_row[HashTableColumn::STATE0 as usize],
                extension_row[HashTableColumn::STATE1 as usize],
                extension_row[HashTableColumn::STATE2 as usize],
                extension_row[HashTableColumn::STATE3 as usize],
                extension_row[HashTableColumn::STATE4 as usize],
                extension_row[HashTableColumn::STATE5 as usize],
            ];
            let compressed_state_for_output = state_for_output
                .iter()
                .zip(challenges.digest_output_weights.iter())
                .map(|(state, weight)| *weight * *state)
                .fold(XFieldElement::ring_zero(), |sum, summand| sum + summand);
            extension_row.push(compressed_state_for_output);

            // Add compressed digest to running sum if round index marks end of hashing
            extension_row.push(to_processor_running_sum);
            if row[HashTableColumn::ROUNDNUMBER as usize].value() == 8 {
                to_processor_running_sum = to_processor_running_sum
                    * challenges.to_processor_eval_row_weight
                    + compressed_state_for_output;
            }

            extension_matrix.push(extension_row);
        }

        let terminals = HashTableEndpoints {
            from_processor_eval_sum: from_processor_running_sum,
            to_processor_eval_sum: to_processor_running_sum,
        };

        let extension_table = self.extension(
            extension_matrix,
            ExtHashTable::ext_boundary_constraints(),
            ExtHashTable::ext_transition_constraints(challenges),
            ExtHashTable::ext_consistency_constraints(),
            ExtHashTable::ext_terminal_constraints(challenges, &terminals),
        );

        (
            ExtHashTable {
                inherited_table: extension_table,
            },
            terminals,
        )
    }

    pub fn for_verifier(
        num_trace_randomizers: usize,
        padded_height: usize,
        all_challenges: &AllChallenges,
        all_terminals: &AllEndpoints,
    ) -> ExtHashTable {
        let omicron = base_table::derive_omicron(padded_height as u64);
        let inherited_table = Table::new(
            BASE_WIDTH,
            FULL_WIDTH,
            padded_height,
            num_trace_randomizers,
            omicron,
            vec![],
            "ExtHashTable".to_string(),
        );
        let base_table = Self { inherited_table };
        let empty_matrix: Vec<Vec<XFieldElement>> = vec![];
        let extension_table = base_table.extension(
            empty_matrix,
            ExtHashTable::ext_boundary_constraints(),
            ExtHashTable::ext_transition_constraints(&all_challenges.hash_table_challenges),
            ExtHashTable::ext_consistency_constraints(),
            ExtHashTable::ext_terminal_constraints(
                &all_challenges.hash_table_challenges,
                &all_terminals.hash_table_endpoints,
            ),
        );

        ExtHashTable {
            inherited_table: extension_table,
        }
    }
}

impl ExtHashTable {
    pub fn with_padded_height(num_trace_randomizers: usize, padded_height: usize) -> Self {
        let matrix: Vec<Vec<XFieldElement>> = vec![];

        let omicron = base_table::derive_omicron(padded_height as u64);
        let inherited_table = Table::new(
            BASE_WIDTH,
            FULL_WIDTH,
            padded_height,
            num_trace_randomizers,
            omicron,
            matrix,
            "ExtHashTable".to_string(),
        );

        Self { inherited_table }
    }

    pub fn ext_codeword_table(
        &self,
        fri_domain: &FriDomain<XFieldElement>,
        base_codewords: &[Vec<BFieldElement>],
    ) -> Self {
        let ext_columns = self.base_width()..self.full_width();
        let ext_codewords = self.low_degree_extension(fri_domain, ext_columns);

        let lifted_base_codewords = base_codewords
            .iter()
            .map(|base_codeword| base_codeword.iter().map(|bfe| bfe.lift()).collect_vec())
            .collect_vec();
        let all_codewords = vec![lifted_base_codewords, ext_codewords].concat();
        assert_eq!(self.full_width(), all_codewords.len());

        let inherited_table = self.inherited_table.with_data(all_codewords);
        ExtHashTable { inherited_table }
    }
}

#[derive(Debug, Clone)]
pub struct HashTableChallenges {
    /// The weight that combines two consecutive rows in the
    /// permutation/evaluation column of the hash table.
    pub from_processor_eval_row_weight: XFieldElement,
    pub to_processor_eval_row_weight: XFieldElement,

    /// Weights for condensing part of a row into a single column. (Related to processor table.)
    pub stack_input_weights: [XFieldElement; 2 * DIGEST_LEN],
    pub digest_output_weights: [XFieldElement; DIGEST_LEN],
}

#[derive(Debug, Clone)]
pub struct HashTableEndpoints {
    /// Values randomly generated by the prover for zero-knowledge.
    pub from_processor_eval_sum: XFieldElement,
    pub to_processor_eval_sum: XFieldElement,
}

impl ExtensionTable for ExtHashTable {
    fn dynamic_boundary_constraints(&self) -> Vec<MPolynomial<XFieldElement>> {
        ExtHashTable::ext_boundary_constraints()
    }

    fn dynamic_transition_constraints(
        &self,
        challenges: &super::challenges_endpoints::AllChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtHashTable::ext_transition_constraints(&challenges.hash_table_challenges)
    }

    fn dynamic_consistency_constraints(&self) -> Vec<MPolynomial<XFieldElement>> {
        ExtHashTable::ext_consistency_constraints()
    }

    fn dynamic_terminal_constraints(
        &self,
        challenges: &super::challenges_endpoints::AllChallenges,
        terminals: &super::challenges_endpoints::AllEndpoints,
    ) -> Vec<MPolynomial<XFieldElement>> {
        ExtHashTable::ext_terminal_constraints(
            &challenges.hash_table_challenges,
            &terminals.hash_table_endpoints,
        )
    }
}
