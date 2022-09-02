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
use std::ops::Add;
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

/// Sponge parameters of Rescue Prime
pub const CAPACITY: usize = 4;
pub const RATE: usize = 12;
pub const STATE_SIZE: usize = 16;

pub const BASE_WIDTH: usize = 49;
pub const FULL_WIDTH: usize = 53; // BASE_WIDTH + 2 * INITIALS_COUNT

pub const TOTAL_NUM_CONSTANTS: usize = NUM_ROUND_CONSTANTS * NUM_ROUNDS;
pub const MDS: [u64; STATE_SIZE * STATE_SIZE] = [
    5910257123858819639,
    3449115226714951713,
    16770055338049327985,
    610399731775780810,
    7363016345531076300,
    16174724756564259629,
    8736587794472183152,
    12699016954477470956,
    13948112026909862966,
    18015813124076612987,
    9568929147539067610,
    14859461777592116402,
    18169364738825153183,
    18221568702798258352,
    1524268296724555606,
    5538821761600,
    1649528676200182784,
    336497118937017052,
    15805000027048028625,
    15709375513998678646,
    14837031240173858084,
    11366298206428370494,
    15698532768527519720,
    5911577595727321095,
    16676030327621016157,
    16537624251746851423,
    13325141695736654367,
    9337952653454313447,
    9090375522091353302,
    5605636660979522224,
    6357222834896114791,
    7776871531164456679,
    8264739868177574620,
    12732288338686680125,
    13022293791945187811,
    17403057736098613442,
    2871266924987061743,
    13286707530570640459,
    9229362695439112266,
    815317759014579856,
    7447771153889267897,
    2209002535000750347,
    3280506473249596174,
    13756142018694965622,
    10518080861296830621,
    16578355848983066277,
    12732532221704648123,
    3426526797578099186,
    8563516248221808333,
    13079317959606236131,
    15645458946300428515,
    9958819147895829140,
    13028053188247480206,
    6789511720078828478,
    6583246594815170294,
    4423695887326249884,
    9751139665897711642,
    10039202025292797758,
    12208726994829996150,
    6238795140281096003,
    9113696057226188857,
    9898705245385052191,
    4213712701625520075,
    8038355032286280912,
    426685147605824917,
    7673465577918025498,
    8452867379070564008,
    10827610229277395180,
    16155539332955658546,
    1575428636717115288,
    8765972548498757598,
    8405996249707890526,
    14855028677418679455,
    17878170012428694685,
    16572621079016066883,
    5311046098447994501,
    10635376800783355348,
    14205668690430323921,
    1181422971831412672,
    4651053123208915543,
    12465667489477238576,
    7300129031676503132,
    13458544786180633209,
    8946801771555977477,
    14203890406114400141,
    8219081892380458635,
    6035067543134909245,
    15140374581570897616,
    4514006299509426029,
    16757530089801321524,
    13202061911440346802,
    11227558237427129334,
    315998614524336401,
    11280705904396606227,
    5798516367202621128,
    17154761698338453414,
    13574436947400004837,
    3126509266905053998,
    10740979484255925394,
    9273322683773825324,
    15349096509718845737,
    14694022445619674948,
    8733857890739087596,
    3198488337424282101,
    9521016570828679381,
    11267736037298472148,
    14825280481028844943,
    1326588754335738002,
    6200834522767914499,
    1070210996042416038,
    9140190343656907671,
    15531381283521001952,
    253143295675927354,
    11977331414401291539,
    13941376566367813256,
    469904915148256197,
    10873951860155749104,
    3939719938926157877,
    2271392376641547055,
    4725974756185387075,
    14827835543640648161,
    17663273767033351157,
    12440960700789890843,
    16589620022628590428,
    12838889473653138505,
    11170336581460183657,
    7583333056198317221,
    6006908286410425140,
    15648567098514276013,
    188901633101859949,
    12256163716419861419,
    17319784688409668747,
    9648971065289440425,
    11370683735445551679,
    11265203235776280908,
    1737672785338087677,
    5225587291780939578,
    4739055740469849012,
    1212344601223444182,
    12958616893209019599,
    7922060480554370635,
    14661420107595710445,
    11744359917257111592,
    9674559564931202709,
    8326110231976411065,
    16856751238353701757,
    7515652322254196544,
    2062531989536141174,
    3875321171362100965,
    1164854003752487518,
    3997098993859160292,
    4074090397542250057,
    3050858158567944540,
    4568245569065883863,
    14559440781022773799,
    5401845794552358815,
    6544584366002554176,
    2511522072283652847,
    9759884967674698659,
    16411672358681189856,
    11392578809073737776,
    8013631514034873271,
    11439549174997471674,
    6373021446442411366,
    12491600135569477757,
    1017093281401495736,
    663547836518863091,
    16157302719777897692,
    11208801522915446640,
    10058178191286215107,
    5521712058210208094,
    3611681474253815005,
    4864578569041337696,
    12270319000993569289,
    7347066511426336318,
    6696546239958933736,
    3335469193383486908,
    12719366334180058014,
    14123019207894489639,
    11418186023060178542,
    2042199956854124583,
    17539253100488345226,
    16240833881391672847,
    11712520063241304909,
    6456900719511754234,
    1819022137223501306,
    7371152900053879920,
    6521878675261223812,
    2050999666988944811,
    8262038465464898064,
    13303819303390508091,
    12657292926928303663,
    8794128680724662595,
    4068577832515945116,
    758247715040138478,
    5600369601992438532,
    3369463178350382224,
    13763645328734311418,
    9685701761982837416,
    2711119809520557835,
    11680482056777716424,
    10958223503056770518,
    4168390070510137163,
    10823375744683484459,
    5613197991565754677,
    11781942063118564684,
    9352512500813609723,
    15997830646514778986,
    7407352006524266457,
    15312663387608602775,
    3026364159907661789,
    5698531403379362946,
    2544271242593770624,
    13104502948897878458,
    7840062700088318710,
    6028743588538970215,
    6144415809411296980,
    468368941216390216,
    3638618405705274008,
    11105401941482704573,
    1850274872877725129,
    1011155312563349004,
    3234620948537841909,
    3818372677739507813,
    4863130691592118581,
    8942166964590283171,
    3639677194051371072,
    15477372418124081864,
    10322228711752830209,
    9139111778956611066,
    202171733050704358,
    11982413146686512577,
    11001000478006340870,
    5491471715020327065,
    6969114856449768266,
    11088492421847219924,
    12913509272810999025,
    17366506887360149369,
    7036328554328346102,
    11139255730689011050,
    2844974929907956457,
    6488525141985913483,
    2860098796699131680,
    10366343151884073105,
    844875652557703984,
    1053177270393416978,
    5189466196833763142,
    1024738234713107670,
    8846741799369572841,
    14490406830213564822,
    10577371742628912722,
    3276210642025060502,
    2605621719516949928,
    5417148926702080639,
    11100652475866543814,
    5247366835775169839,
];
pub const MDS_INV: [u64; STATE_SIZE * STATE_SIZE] = [
    1572742562154761373,
    11904188991461183391,
    16702037635100780588,
    10395027733616703929,
    8130016957979279389,
    12091057987196709719,
    14570460902390750822,
    13452497170858892918,
    7302470671584418296,
    12930709087691977410,
    6940810864055149191,
    15479085069460687984,
    15273989414499187903,
    8742532579937987008,
    78143684950290654,
    10454925311792498315,
    7789818152192856725,
    3486011543032592030,
    17188770042768805161,
    10490412495468775616,
    298640180115056798,
    12895819509602002088,
    1755013598313843104,
    17242416429764373372,
    993835663551930043,
    17604339535769584753,
    17954116481891390155,
    332811330083846624,
    14730023810555747819,
    435413210797820565,
    1781261080337413422,
    4148505421656051973,
    980199695323775177,
    4706730905557535223,
    12734714246714791746,
    14273996233795959868,
    7921735635146743134,
    14772166129594741813,
    2171393332099124215,
    11431591906353698662,
    1968460689143086961,
    12435956952300281356,
    18203712123938736914,
    13226878153002754824,
    4722189513468037980,
    14552059159516237140,
    2186026037853355566,
    11286141841507813990,
    565856028734827369,
    13655906686104936396,
    8559867348362880285,
    2797343365604350633,
    4465794635391355875,
    10602340776590577912,
    6532765362293732644,
    9971594382705594993,
    8246981798349136173,
    4260734168634971109,
    3096607081570771,
    823237991393038853,
    17532689952600815755,
    12134755733102166916,
    10570439735096051664,
    18403803913856082900,
    13128404168847275462,
    16663835358650929116,
    16546671721888068220,
    4685011688485137218,
    1959001578540316019,
    16340711608595843821,
    9460495021221259854,
    3858517940845573321,
    9427670160758976948,
    18064975260450261693,
    4905506444249847758,
    15986418616213903133,
    9282818778268010424,
    9769107232941785010,
    8521948467436343364,
    7419602577337727529,
    5926710664024036226,
    11667040483862285999,
    12291037072726747355,
    12257844845576909578,
    5216888292865522221,
    4949589496388892504,
    6571373688631618567,
    10091372984903831417,
    6240610640427541397,
    6328690792776976228,
    11836184983048970818,
    12710419323566440454,
    10374451385652807364,
    8254232795575550118,
    9866490979395302091,
    12991014125893242232,
    1063347186953727863,
    2952135743830082310,
    17315974856538709017,
    14554512349953922358,
    14134347382797855179,
    17882046380988406016,
    17463193400175360824,
    3726957756828900632,
    17604631050958608669,
    7585987025945897953,
    14470977033142357695,
    10643295498661723800,
    8871197056529643534,
    8384208064507509379,
    9280566467635869786,
    87319369282683875,
    1100172740622998121,
    622721254307916221,
    16843330035110191506,
    13024130485811341782,
    12334996107415540952,
    461552745543935046,
    8140793910765831499,
    9008477689109468885,
    17409910369122253035,
    1804565454784197696,
    5310948951638903141,
    12531953612536647976,
    6147853502869470889,
    1125351356112285953,
    6467901683012265601,
    16792548587138841945,
    14092833521360698433,
    13651748079341829335,
    10688258556205752814,
    1823953496327460008,
    2558053704584850519,
    13269131806718310421,
    4608410977522599149,
    9221187654763620553,
    4611978991500182874,
    8855429001286425455,
    5696709580182222832,
    17579496245625003067,
    5267934104348282564,
    1835676094870249003,
    3542280417783105151,
    11824126253481498070,
    9504622962336320170,
    17887320494921151801,
    6574518722274623914,
    16658124633332643846,
    13808019273382263890,
    13092903038683672100,
    501471167473345282,
    11161560208140424921,
    13001827442679699140,
    14739684132127818993,
    2868223407847949089,
    1726410909424820290,
    6794531346610991076,
    6698331109000773276,
    3680934785728193940,
    8875468921351982841,
    5477651765997654015,
    12280771278642823764,
    3619998794343148112,
    6883119128428826230,
    13512760119042878827,
    3675597821767844913,
    5414638790278102151,
    3587251244316549755,
    17100313981528550060,
    11048426899172804713,
    1396562484529002856,
    2252873797267794672,
    14201526079271439737,
    16618356769072634008,
    144564843743666734,
    11912794688498369701,
    10937102025343594422,
    15432144252435329607,
    2221546737981282133,
    6015808993571140081,
    7447996510907844453,
    7039231904611782781,
    2218118803134364409,
    9472427559993341443,
    11066826455107746221,
    6223571389973384864,
    13615228926415811268,
    10241352486499609335,
    12605380114102527595,
    11403123666082872720,
    9771232158486004346,
    11862860570670038891,
    10489319728736503343,
    588166220336712628,
    524399652036013851,
    2215268375273320892,
    1424724725807107497,
    2223952838426612865,
    1901666565705039600,
    14666084855112001547,
    16529527081633002035,
    3475787534446449190,
    17395838083455569055,
    10036301139275236437,
    5830062976180250577,
    6201110308815839738,
    3908827014617539568,
    13269427316630307104,
    1104974093011983663,
    335137437077264843,
    13411663683768112565,
    7907493007733959147,
    17240291213488173803,
    6357405277112016289,
    7875258449007392338,
    16100900298327085499,
    13542432207857463387,
    9466802464896264825,
    9221606791343926561,
    10417300838622453849,
    13201838829839066427,
    9833345239958202067,
    16688814355354359676,
    13315432437333533951,
    378443609734580293,
    14654525144709164243,
    1967217494445269914,
    16045947041840686058,
    18049263629128746044,
    1957063364541610677,
    16123386013589472221,
    5923137592664329389,
    12399617421793397670,
    3403518680407886401,
    6416516714555000604,
    13286977196258324106,
    17641011370212535641,
    14823578540420219384,
    11909888788340877523,
    11040604022089158722,
    14682783085930648838,
    7896655986299558210,
    9328642557612914244,
    6213125364180629684,
    16259136970573308007,
    12025260496935037210,
    1512031407150257270,
    1295709332547428576,
    13851880110872460625,
    6734559515296147531,
    17720805166223714561,
    11264121550751120724,
    7210341680607060660,
    17759718475616004694,
    610155440804635364,
    3209025413915748371,
];
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

    fn get_transition_quotient_degree_bounds(&self) -> Vec<Degree> {
        let round_number_bounds = vec![
            self.interpolant_degree() * 9,
            self.interpolant_degree() * 9,
            self.interpolant_degree() * 3,
        ];
        let state_evolution_bounds = vec![self.interpolant_degree() * 8; STATE_SIZE];
        [round_number_bounds, state_evolution_bounds].concat()
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
    /// consistency constraints directly by implementing the respective method in trait
    /// `Evaluable`, and does not use the polynomials below.
    fn ext_consistency_constraints() -> Vec<MPolynomial<XFieldElement>> {
        let constant = |c: u32| MPolynomial::from_constant(c.into(), FULL_WIDTH);
        let variables = MPolynomial::variables(FULL_WIDTH, 1.into());

        let round_number = variables[ROUNDNUMBER as usize].clone();
        let state12 = variables[STATE12 as usize].clone();
        let state13 = variables[STATE13 as usize].clone();
        let state14 = variables[STATE14 as usize].clone();
        let state15 = variables[STATE15 as usize].clone();

        // 1. if round number is 1, then capacity is zero
        // DNF: rn =/= 1 \/ cap = 0
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

        // 2. round number is in {0, ..., 8}
        let polynomial = (0..=8)
            .map(|r| constant(r) - round_number.clone())
            .fold(constant(1), MPolynomial::mul);
        consistency_polynomials.push(polynomial);

        // 3. round constants
        // if round number is zero, we don't care
        // otherwise, make sure the constant is correct
        let round_constant_offset = CONSTANT0A as usize;
        for round_constant_idx in 0..NUM_ROUND_CONSTANTS {
            let round_constant_column: HashTableColumn =
                (round_constant_idx + round_constant_offset).into();
            let round_constant = &variables[round_constant_column as usize];
            let interpolant = Self::round_constants_interpolant(round_constant_column);
            let multivariate_interpolant =
                MPolynomial::lift(interpolant, ROUNDNUMBER as usize, FULL_WIDTH);
            consistency_polynomials
                .push(round_number.clone() * (multivariate_interpolant - round_constant.clone()));
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

    /// The implementation below is kept around for debugging purposes. This table evaluates the
    /// transition constraints directly by implementing the respective method in trait
    /// `Evaluable`, and does not use the polynomials below.
    fn ext_transition_constraints(
        _challenges: &HashTableChallenges,
    ) -> Vec<MPolynomial<XFieldElement>> {
        let constant =
            |c: u64| MPolynomial::from_constant(BFieldElement::new(c).lift(), 2 * FULL_WIDTH);
        let variables = MPolynomial::variables(2 * FULL_WIDTH, 1.into());

        let round_number = variables[ROUNDNUMBER as usize].clone();
        let round_number_next = variables[FULL_WIDTH + ROUNDNUMBER as usize].clone();

        let mut constraint_polynomials: Vec<MPolynomial<XFieldElement>> = vec![];

        // round number
        // round numbers evolve as
        // 1 -> 2 -> 3 -> 4 -> 5 -> 6 -> 7 -> 8, and
        // 8 -> 1 or 8 -> 0, and
        // 0 -> 0

        // 1. round number belongs to {0, ..., 8}
        // => consistency constraint

        // 2. if round number is 0, then next round number is 0
        // DNF: rn in {1, ..., 8} \/ rn* = 0
        let mut polynomial = (1..=8)
            .map(|r| constant(r) - round_number.clone())
            .fold(constant(1), MPolynomial::mul);
        polynomial *= round_number_next.clone();
        constraint_polynomials.push(polynomial);

        // 3. if round number is 8, then next round number is 0 or 1
        // DNF: rn =/= 8 \/ rn* = 0 \/ rn* = 1
        polynomial = (0..=7)
            .map(|r| constant(r) - round_number.clone())
            .fold(constant(1), MPolynomial::mul);
        polynomial *= constant(1) - round_number_next.clone();
        polynomial *= round_number_next.clone();
        constraint_polynomials.push(polynomial);

        // 4. if round number is in {1, ..., 7} then next round number is +1
        // DNF: (rn == 0 \/ rn == 8) \/ rn* = rn + 1
        polynomial = round_number.clone()
            * (constant(8) - round_number.clone())
            * (round_number_next.clone() - round_number.clone() - constant(1));
        constraint_polynomials.push(polynomial);

        // Rescue-XLIX

        // left-hand-side, starting at current round and going forward
        let current_state: Vec<MPolynomial<XFieldElement>> = (0..STATE_SIZE)
            .map(|i| variables[HashTableColumn::STATE0 as usize + i].clone())
            .collect_vec();
        let after_sbox = current_state
            .iter()
            .map(|c| c.mod_pow(7.into(), XFieldElement::ring_one()))
            .collect_vec();
        let after_mds = (0..STATE_SIZE)
            .map(|i| {
                (0..STATE_SIZE)
                    .map(|j| constant(MDS[i * STATE_SIZE + j]) * after_sbox[j].clone())
                    .fold(constant(1), MPolynomial::add)
            })
            .collect_vec();
        let round_constants = variables
            [(HashTableColumn::CONSTANT0A as usize)..=(HashTableColumn::CONSTANT15B as usize)]
            .to_vec();
        let after_constants = after_mds
            .into_iter()
            .zip_eq(&round_constants[..(NUM_ROUND_CONSTANTS / 2)])
            .map(|(st, rndc)| st + rndc.to_owned())
            .collect_vec();

        // right hand side; move backwards
        let next_state: Vec<MPolynomial<XFieldElement>> = (0..STATE_SIZE)
            .map(|i| variables[FULL_WIDTH + HashTableColumn::STATE0 as usize + i].clone())
            .collect_vec();
        let before_constants = next_state
            .into_iter()
            .zip_eq(&round_constants[(NUM_ROUND_CONSTANTS / 2)..])
            .map(|(st, rndc)| st - rndc.to_owned())
            .collect_vec();
        let before_mds = (0..STATE_SIZE)
            .map(|i| {
                (0..STATE_SIZE)
                    .map(|j| constant(MDS_INV[i * STATE_SIZE + j]) * before_constants[j].clone())
                    .fold(constant(1), MPolynomial::add)
            })
            .collect_vec();
        let before_sbox = before_mds
            .iter()
            .map(|c| c.mod_pow(7.into(), XFieldElement::ring_one()))
            .collect_vec();

        // equate left hand side to right hand side
        // (and ignore if padding row)
        constraint_polynomials.append(
            &mut after_constants
                .into_iter()
                .zip_eq(before_sbox.into_iter())
                .map(|(lhs, rhs)| round_number.clone() * (lhs - rhs))
                .collect_vec(),
        );

        constraint_polynomials
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
