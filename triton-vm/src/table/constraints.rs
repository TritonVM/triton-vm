use ndarray::ArrayView1;
use twenty_first::prelude::BFieldElement;
use twenty_first::prelude::XFieldElement;
use crate::table::challenges::Challenges;
use crate::table::extension_table::Evaluable;
use crate::table::extension_table::Quotientable;
use crate::table::master_table::MasterExtTable;
impl Evaluable<BFieldElement> for MasterExtTable {
    #[allow(unused_variables)]
    fn evaluate_initial_constraints(
        base_row: ArrayView1<BFieldElement>,
        ext_row: ArrayView1<XFieldElement>,
        challenges: &Challenges,
    ) -> Vec<XFieldElement> {
        let node_468 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (base_row[129usize]));
        let node_474 = ((challenges[52usize]) * (base_row[131usize]))
            + ((challenges[53usize]) * (base_row[133usize]));
        let node_477 = ((challenges[52usize]) * (base_row[130usize]))
            + ((challenges[53usize]) * (base_row[132usize]));
        let base_constraints = [
            base_row[0usize],
            base_row[3usize],
            base_row[5usize],
            base_row[7usize],
            base_row[9usize],
            base_row[19usize],
            base_row[20usize],
            base_row[21usize],
            base_row[22usize],
            base_row[23usize],
            base_row[24usize],
            base_row[25usize],
            base_row[26usize],
            base_row[27usize],
            base_row[28usize],
            base_row[29usize],
            base_row[30usize],
            base_row[31usize],
            base_row[32usize],
            (base_row[38usize]) + (BFieldElement::from_raw_u64(18446744000695107601u64)),
            (base_row[48usize]) + (BFieldElement::from_raw_u64(18446744000695107601u64)),
            base_row[55usize],
            base_row[57usize],
            base_row[59usize],
            base_row[60usize],
            base_row[61usize],
            (base_row[62usize]) + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            base_row[64usize],
            base_row[136usize],
            (base_row[149usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((((base_row[12usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                        * (base_row[13usize]))
                        * ((base_row[14usize])
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                        * ((base_row[15usize])
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))))),
            (base_row[150usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((base_row[149usize]) * (base_row[16usize]))
                        * ((base_row[17usize])
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                        * ((base_row[18usize])
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))))),
        ];
        let ext_constraints = [
            ext_row[0usize],
            ((ext_row[1usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (challenges[29usize])))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (base_row[1usize])),
            (ext_row[2usize]) + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((((((((((challenges[0usize]) + (base_row[33usize])) * (challenges[0usize]))
                + (base_row[34usize])) * (challenges[0usize])) + (base_row[35usize]))
                * (challenges[0usize])) + (base_row[36usize])) * (challenges[0usize]))
                + (base_row[37usize]))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (challenges[62usize])),
            (ext_row[3usize]) + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[5usize])
                * ((challenges[3usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[14usize]) * (base_row[10usize]))
                            + ((challenges[15usize]) * (base_row[11usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            (ext_row[4usize]) + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            (ext_row[6usize]) + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            (ext_row[7usize]) + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            (ext_row[8usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((challenges[9usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((challenges[25usize]) * (base_row[10usize]))))),
            ((ext_row[13usize]) * (challenges[11usize]))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (base_row[45usize])),
            (((base_row[10usize])
                + (BFieldElement::from_raw_u64(18446743992105173011u64)))
                * ((ext_row[9usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((base_row[150usize])
                    * ((ext_row[9usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (challenges[4usize])))),
            (ext_row[10usize]) + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            (ext_row[11usize]) + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ext_row[12usize],
            (((ext_row[14usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((challenges[7usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((challenges[16usize]) * (base_row[46usize]))
                                + ((challenges[17usize]) * (base_row[47usize])))
                                + ((challenges[18usize])
                                    * (BFieldElement::from_raw_u64(68719476720u64))))
                                + ((challenges[19usize]) * (base_row[49usize])))))))
                * ((base_row[47usize])
                    + (BFieldElement::from_raw_u64(18446744060824649731u64))))
                + (((ext_row[14usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                    * ((base_row[47usize])
                        * ((base_row[47usize])
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))))),
            ext_row[15usize],
            ext_row[18usize],
            (ext_row[19usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (base_row[56usize])),
            ((ext_row[16usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (challenges[12usize]))) + (base_row[52usize]),
            (ext_row[17usize]) + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((((ext_row[20usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (challenges[8usize])))
                + (((((base_row[50usize]) * (challenges[20usize]))
                    + ((base_row[51usize]) * (challenges[23usize])))
                    + ((base_row[52usize]) * (challenges[21usize])))
                    + ((base_row[53usize]) * (challenges[22usize]))))
                * ((base_row[51usize])
                    + (BFieldElement::from_raw_u64(18446744060824649731u64))))
                + (((ext_row[20usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                    * (((base_row[51usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                        * (base_row[51usize]))),
            ext_row[21usize],
            (ext_row[22usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((challenges[9usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((challenges[25usize]) * (base_row[58usize]))))),
            ext_row[23usize],
            (ext_row[25usize]) + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            (ext_row[26usize]) + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            (ext_row[27usize]) + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[24usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (challenges[30usize])))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((((((((((((((((((challenges[29usize])
                        + ((((((base_row[65usize])
                            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
                            + ((base_row[66usize])
                                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
                            + ((base_row[67usize])
                                * (BFieldElement::from_raw_u64(281474976645120u64))))
                            + (base_row[68usize]))
                            * (BFieldElement::from_raw_u64(1u64))))
                        * (challenges[29usize]))
                        + ((((((base_row[69usize])
                            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
                            + ((base_row[70usize])
                                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
                            + ((base_row[71usize])
                                * (BFieldElement::from_raw_u64(281474976645120u64))))
                            + (base_row[72usize]))
                            * (BFieldElement::from_raw_u64(1u64))))
                        * (challenges[29usize]))
                        + ((((((base_row[73usize])
                            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
                            + ((base_row[74usize])
                                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
                            + ((base_row[75usize])
                                * (BFieldElement::from_raw_u64(281474976645120u64))))
                            + (base_row[76usize]))
                            * (BFieldElement::from_raw_u64(1u64))))
                        * (challenges[29usize]))
                        + ((((((base_row[77usize])
                            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
                            + ((base_row[78usize])
                                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
                            + ((base_row[79usize])
                                * (BFieldElement::from_raw_u64(281474976645120u64))))
                            + (base_row[80usize]))
                            * (BFieldElement::from_raw_u64(1u64))))
                        * (challenges[29usize])) + (base_row[97usize]))
                        * (challenges[29usize])) + (base_row[98usize]))
                        * (challenges[29usize])) + (base_row[99usize]))
                        * (challenges[29usize])) + (base_row[100usize]))
                        * (challenges[29usize])) + (base_row[101usize]))
                        * (challenges[29usize])) + (base_row[102usize]))),
            ((ext_row[28usize])
                * ((challenges[48usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[49usize]) * (base_row[65usize]))
                            + ((challenges[50usize]) * (base_row[81usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[29usize])
                * ((challenges[48usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[49usize]) * (base_row[66usize]))
                            + ((challenges[50usize]) * (base_row[82usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[30usize])
                * ((challenges[48usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[49usize]) * (base_row[67usize]))
                            + ((challenges[50usize]) * (base_row[83usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[31usize])
                * ((challenges[48usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[49usize]) * (base_row[68usize]))
                            + ((challenges[50usize]) * (base_row[84usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[32usize])
                * ((challenges[48usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[49usize]) * (base_row[69usize]))
                            + ((challenges[50usize]) * (base_row[85usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[33usize])
                * ((challenges[48usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[49usize]) * (base_row[70usize]))
                            + ((challenges[50usize]) * (base_row[86usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[34usize])
                * ((challenges[48usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[49usize]) * (base_row[71usize]))
                            + ((challenges[50usize]) * (base_row[87usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[35usize])
                * ((challenges[48usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[49usize]) * (base_row[72usize]))
                            + ((challenges[50usize]) * (base_row[88usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[36usize])
                * ((challenges[48usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[49usize]) * (base_row[73usize]))
                            + ((challenges[50usize]) * (base_row[89usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[37usize])
                * ((challenges[48usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[49usize]) * (base_row[74usize]))
                            + ((challenges[50usize]) * (base_row[90usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[38usize])
                * ((challenges[48usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[49usize]) * (base_row[75usize]))
                            + ((challenges[50usize]) * (base_row[91usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[39usize])
                * ((challenges[48usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[49usize]) * (base_row[76usize]))
                            + ((challenges[50usize]) * (base_row[92usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[40usize])
                * ((challenges[48usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[49usize]) * (base_row[77usize]))
                            + ((challenges[50usize]) * (base_row[93usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[41usize])
                * ((challenges[48usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[49usize]) * (base_row[78usize]))
                            + ((challenges[50usize]) * (base_row[94usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[42usize])
                * ((challenges[48usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[49usize]) * (base_row[79usize]))
                            + ((challenges[50usize]) * (base_row[95usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[43usize])
                * ((challenges[48usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[49usize]) * (base_row[80usize]))
                            + ((challenges[50usize]) * (base_row[96usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((node_468)
                * (((ext_row[44usize])
                    * ((challenges[48usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[49usize])
                                * (((BFieldElement::from_raw_u64(1099511627520u64))
                                    * (base_row[130usize])) + (base_row[131usize])))
                                + ((challenges[50usize])
                                    * (((BFieldElement::from_raw_u64(1099511627520u64))
                                        * (base_row[132usize])) + (base_row[133usize])))))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (base_row[134usize]))))
                + ((base_row[129usize]) * (ext_row[44usize])),
            ((node_468)
                * ((((((ext_row[45usize])
                    * ((challenges[51usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_474))))
                    * ((challenges[51usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_477))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((BFieldElement::from_raw_u64(8589934590u64))
                            * (challenges[51usize])))) + (node_474)) + (node_477)))
                + ((base_row[129usize]) * (ext_row[45usize])),
            ((ext_row[46usize])
                * ((challenges[51usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((base_row[137usize]) * (challenges[53usize])))))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (base_row[138usize])),
            ((ext_row[47usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (challenges[54usize])))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (base_row[137usize])),
            (((base_row[139usize])
                + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                * (ext_row[48usize]))
                + ((base_row[139usize])
                    * (((ext_row[48usize])
                        * ((challenges[10usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((((challenges[55usize]) * (base_row[143usize]))
                                    + ((challenges[56usize]) * (base_row[145usize])))
                                    + ((challenges[57usize]) * (base_row[142usize])))
                                    + ((challenges[58usize]) * (base_row[147usize]))))))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (base_row[148usize])))),
        ];
        base_constraints
            .into_iter()
            .map(|bfe| bfe.lift())
            .chain(ext_constraints)
            .collect()
    }
    #[allow(unused_variables)]
    fn evaluate_consistency_constraints(
        base_row: ArrayView1<BFieldElement>,
        ext_row: ArrayView1<XFieldElement>,
        challenges: &Challenges,
    ) -> Vec<XFieldElement> {
        let node_102 = (base_row[152usize])
            * ((base_row[64usize])
                + (BFieldElement::from_raw_u64(18446744047939747846u64)));
        let node_221 = (base_row[153usize])
            * ((base_row[64usize])
                + (BFieldElement::from_raw_u64(18446744047939747846u64)));
        let node_238 = ((base_row[154usize])
            * ((base_row[64usize])
                + (BFieldElement::from_raw_u64(18446744052234715141u64))))
            * ((base_row[64usize])
                + (BFieldElement::from_raw_u64(18446744047939747846u64)));
        let node_245 = ((base_row[154usize])
            * ((base_row[64usize])
                + (BFieldElement::from_raw_u64(18446744056529682436u64))))
            * ((base_row[64usize])
                + (BFieldElement::from_raw_u64(18446744047939747846u64)));
        let node_655 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (base_row[157usize]));
        let node_114 = (((base_row[63usize])
            + (BFieldElement::from_raw_u64(18446743992105173011u64)))
            * ((base_row[63usize])
                + (BFieldElement::from_raw_u64(18446743923385696291u64))))
            * ((base_row[63usize])
                + (BFieldElement::from_raw_u64(18446743828896415801u64)));
        let node_116 = (node_102) * (base_row[161usize]);
        let node_660 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (base_row[160usize]));
        let node_101 = (base_row[64usize])
            + (BFieldElement::from_raw_u64(18446744047939747846u64));
        let node_678 = (base_row[142usize])
            + (BFieldElement::from_raw_u64(18446743949155500061u64));
        let node_674 = (base_row[142usize])
            + (BFieldElement::from_raw_u64(18446743940565565471u64));
        let node_94 = (base_row[64usize])
            + (BFieldElement::from_raw_u64(18446744056529682436u64));
        let node_97 = (base_row[64usize])
            + (BFieldElement::from_raw_u64(18446744052234715141u64));
        let node_153 = ((((BFieldElement::from_raw_u64(18446744065119617025u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((base_row[65usize])
                    * (BFieldElement::from_raw_u64(281474976645120u64)))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (base_row[66usize]))) * (base_row[109usize]))
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_155 = ((((BFieldElement::from_raw_u64(18446744065119617025u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((base_row[69usize])
                    * (BFieldElement::from_raw_u64(281474976645120u64)))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (base_row[70usize]))) * (base_row[110usize]))
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_157 = ((((BFieldElement::from_raw_u64(18446744065119617025u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((base_row[73usize])
                    * (BFieldElement::from_raw_u64(281474976645120u64)))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (base_row[74usize]))) * (base_row[111usize]))
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_159 = ((((BFieldElement::from_raw_u64(18446744065119617025u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((base_row[77usize])
                    * (BFieldElement::from_raw_u64(281474976645120u64)))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (base_row[78usize]))) * (base_row[112usize]))
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_680 = (base_row[139usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_90 = (base_row[64usize])
            + (BFieldElement::from_raw_u64(18446744060824649731u64));
        let node_670 = (base_row[142usize])
            + (BFieldElement::from_raw_u64(18446744017874976781u64));
        let node_11 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (((BFieldElement::from_raw_u64(38654705655u64))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (base_row[3usize]))) * (base_row[4usize])));
        let node_8 = (BFieldElement::from_raw_u64(38654705655u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (base_row[3usize]));
        let node_104 = (((base_row[62usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64)))
            * ((base_row[62usize])
                + (BFieldElement::from_raw_u64(18446744060824649731u64))))
            * ((base_row[62usize])
                + (BFieldElement::from_raw_u64(18446744056529682436u64)));
        let node_85 = (base_row[62usize])
            + (BFieldElement::from_raw_u64(18446744060824649731u64));
        let node_73 = (base_row[63usize])
            + (BFieldElement::from_raw_u64(18446743992105173011u64));
        let node_79 = (base_row[63usize])
            + (BFieldElement::from_raw_u64(18446743923385696291u64));
        let node_82 = (base_row[63usize])
            + (BFieldElement::from_raw_u64(18446743828896415801u64));
        let node_126 = ((BFieldElement::from_raw_u64(18446744065119617025u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((base_row[65usize])
                    * (BFieldElement::from_raw_u64(281474976645120u64)))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (base_row[66usize]));
        let node_133 = ((BFieldElement::from_raw_u64(18446744065119617025u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((base_row[69usize])
                    * (BFieldElement::from_raw_u64(281474976645120u64)))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (base_row[70usize]));
        let node_140 = ((BFieldElement::from_raw_u64(18446744065119617025u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((base_row[73usize])
                    * (BFieldElement::from_raw_u64(281474976645120u64)))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (base_row[74usize]));
        let node_147 = ((BFieldElement::from_raw_u64(18446744065119617025u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((base_row[77usize])
                    * (BFieldElement::from_raw_u64(281474976645120u64)))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (base_row[78usize]));
        let node_89 = (base_row[64usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_663 = (base_row[142usize])
            + (BFieldElement::from_raw_u64(18446744052234715141u64));
        let node_666 = (base_row[142usize])
            + (BFieldElement::from_raw_u64(18446744009285042191u64));
        let node_86 = ((base_row[62usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64))) * (node_85);
        let node_83 = (base_row[62usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_103 = (base_row[62usize])
            + (BFieldElement::from_raw_u64(18446744056529682436u64));
        let base_constraints = [
            (node_11) * (base_row[4usize]),
            (node_11) * (node_8),
            (base_row[5usize])
                * ((base_row[5usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (base_row[6usize])
                * ((base_row[6usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (base_row[12usize])
                * ((base_row[12usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (base_row[13usize])
                * ((base_row[13usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (base_row[14usize])
                * ((base_row[14usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (base_row[15usize])
                * ((base_row[15usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (base_row[16usize])
                * ((base_row[16usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (base_row[17usize])
                * ((base_row[17usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (base_row[18usize])
                * ((base_row[18usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (base_row[8usize])
                * ((base_row[8usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (base_row[10usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((((((base_row[12usize])
                        + ((BFieldElement::from_raw_u64(8589934590u64))
                            * (base_row[13usize])))
                        + ((BFieldElement::from_raw_u64(17179869180u64))
                            * (base_row[14usize])))
                        + ((BFieldElement::from_raw_u64(34359738360u64))
                            * (base_row[15usize])))
                        + ((BFieldElement::from_raw_u64(68719476720u64))
                            * (base_row[16usize])))
                        + ((BFieldElement::from_raw_u64(137438953440u64))
                            * (base_row[17usize])))
                        + ((BFieldElement::from_raw_u64(274877906880u64))
                            * (base_row[18usize])))),
            ((base_row[8usize])
                * ((base_row[7usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                * (base_row[45usize]),
            (node_104) * (base_row[62usize]),
            (node_85) * (node_73),
            ((base_row[165usize]) * (node_79)) * (node_82),
            (node_104) * (base_row[64usize]),
            (node_114) * (base_row[64usize]),
            (node_153) * (base_row[109usize]),
            (node_155) * (base_row[110usize]),
            (node_157) * (base_row[111usize]),
            (node_159) * (base_row[112usize]),
            (node_153) * (node_126),
            (node_155) * (node_133),
            (node_157) * (node_140),
            (node_159) * (node_147),
            (node_153)
                * (((base_row[67usize])
                    * (BFieldElement::from_raw_u64(281474976645120u64)))
                    + (base_row[68usize])),
            (node_155)
                * (((base_row[71usize])
                    * (BFieldElement::from_raw_u64(281474976645120u64)))
                    + (base_row[72usize])),
            (node_157)
                * (((base_row[75usize])
                    * (BFieldElement::from_raw_u64(281474976645120u64)))
                    + (base_row[76usize])),
            (node_159)
                * (((base_row[79usize])
                    * (BFieldElement::from_raw_u64(281474976645120u64)))
                    + (base_row[80usize])),
            (node_114) * (base_row[103usize]),
            (node_114) * (base_row[104usize]),
            (node_114) * (base_row[105usize]),
            (node_114) * (base_row[106usize]),
            (node_114) * (base_row[107usize]),
            (node_114) * (base_row[108usize]),
            (node_116)
                * ((base_row[103usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (node_116)
                * ((base_row[104usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (node_116)
                * ((base_row[105usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (node_116)
                * ((base_row[106usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (node_116)
                * ((base_row[107usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (node_116)
                * ((base_row[108usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (((((node_102)
                * ((base_row[113usize])
                    + (BFieldElement::from_raw_u64(11408918724931329738u64))))
                + ((node_221)
                    * ((base_row[113usize])
                        + (BFieldElement::from_raw_u64(16073625066478178581u64)))))
                + ((base_row[155usize])
                    * ((base_row[113usize])
                        + (BFieldElement::from_raw_u64(12231462398569191607u64)))))
                + ((node_238)
                    * ((base_row[113usize])
                        + (BFieldElement::from_raw_u64(9408518518620565480u64)))))
                + ((node_245)
                    * ((base_row[113usize])
                        + (BFieldElement::from_raw_u64(11492978409391175103u64)))),
            (((((node_102)
                * ((base_row[114usize])
                    + (BFieldElement::from_raw_u64(2786462832312611053u64))))
                + ((node_221)
                    * ((base_row[114usize])
                        + (BFieldElement::from_raw_u64(11837051899140380443u64)))))
                + ((base_row[155usize])
                    * ((base_row[114usize])
                        + (BFieldElement::from_raw_u64(11546487907579866869u64)))))
                + ((node_238)
                    * ((base_row[114usize])
                        + (BFieldElement::from_raw_u64(1785884128667671832u64)))))
                + ((node_245)
                    * ((base_row[114usize])
                        + (BFieldElement::from_raw_u64(17615222217495663839u64)))),
            (((((node_102)
                * ((base_row[115usize])
                    + (BFieldElement::from_raw_u64(6782977121958050999u64))))
                + ((node_221)
                    * ((base_row[115usize])
                        + (BFieldElement::from_raw_u64(15625104599191418968u64)))))
                + ((base_row[155usize])
                    * ((base_row[115usize])
                        + (BFieldElement::from_raw_u64(14006427992450931468u64)))))
                + ((node_238)
                    * ((base_row[115usize])
                        + (BFieldElement::from_raw_u64(1188899344229954938u64)))))
                + ((node_245)
                    * ((base_row[115usize])
                        + (BFieldElement::from_raw_u64(5864349944556149748u64)))),
            (((((node_102)
                * ((base_row[116usize])
                    + (BFieldElement::from_raw_u64(8688421733879975670u64))))
                + ((node_221)
                    * ((base_row[116usize])
                        + (BFieldElement::from_raw_u64(12819157612210448391u64)))))
                + ((base_row[155usize])
                    * ((base_row[116usize])
                        + (BFieldElement::from_raw_u64(11770003398407723041u64)))))
                + ((node_238)
                    * ((base_row[116usize])
                        + (BFieldElement::from_raw_u64(14740727267735052728u64)))))
                + ((node_245)
                    * ((base_row[116usize])
                        + (BFieldElement::from_raw_u64(2745609811140253793u64)))),
            (((((node_102)
                * ((base_row[117usize])
                    + (BFieldElement::from_raw_u64(8602724563769480463u64))))
                + ((node_221)
                    * ((base_row[117usize])
                        + (BFieldElement::from_raw_u64(6235256903503367222u64)))))
                + ((base_row[155usize])
                    * ((base_row[117usize])
                        + (BFieldElement::from_raw_u64(15124190001489436038u64)))))
                + ((node_238)
                    * ((base_row[117usize])
                        + (BFieldElement::from_raw_u64(880257844992994007u64)))))
                + ((node_245)
                    * ((base_row[117usize])
                        + (BFieldElement::from_raw_u64(15189664869386394185u64)))),
            (((((node_102)
                * ((base_row[118usize])
                    + (BFieldElement::from_raw_u64(13589155570211330507u64))))
                + ((node_221)
                    * ((base_row[118usize])
                        + (BFieldElement::from_raw_u64(11242082964257948320u64)))))
                + ((base_row[155usize])
                    * ((base_row[118usize])
                        + (BFieldElement::from_raw_u64(14834674155811570980u64)))))
                + ((node_238)
                    * ((base_row[118usize])
                        + (BFieldElement::from_raw_u64(10737952517017171197u64)))))
                + ((node_245)
                    * ((base_row[118usize])
                        + (BFieldElement::from_raw_u64(5192963426821415349u64)))),
            (((((node_102)
                * ((base_row[119usize])
                    + (BFieldElement::from_raw_u64(10263462378312899510u64))))
                + ((node_221)
                    * ((base_row[119usize])
                        + (BFieldElement::from_raw_u64(5820425254787221108u64)))))
                + ((base_row[155usize])
                    * ((base_row[119usize])
                        + (BFieldElement::from_raw_u64(13004675752386552573u64)))))
                + ((node_238)
                    * ((base_row[119usize])
                        + (BFieldElement::from_raw_u64(15757222735741919824u64)))))
                + ((node_245)
                    * ((base_row[119usize])
                        + (BFieldElement::from_raw_u64(11971160388083607515u64)))),
            (((((node_102)
                * ((base_row[120usize])
                    + (BFieldElement::from_raw_u64(3264875873073042616u64))))
                + ((node_221)
                    * ((base_row[120usize])
                        + (BFieldElement::from_raw_u64(12019227591549292608u64)))))
                + ((base_row[155usize])
                    * ((base_row[120usize])
                        + (BFieldElement::from_raw_u64(1475232519215872482u64)))))
                + ((node_238)
                    * ((base_row[120usize])
                        + (BFieldElement::from_raw_u64(14382578632612566479u64)))))
                + ((node_245)
                    * ((base_row[120usize])
                        + (BFieldElement::from_raw_u64(11608544217838050708u64)))),
            (((((node_102)
                * ((base_row[121usize])
                    + (BFieldElement::from_raw_u64(3133435276616064683u64))))
                + ((node_221)
                    * ((base_row[121usize])
                        + (BFieldElement::from_raw_u64(4625353063880731092u64)))))
                + ((base_row[155usize])
                    * ((base_row[121usize])
                        + (BFieldElement::from_raw_u64(4883869161905122316u64)))))
                + ((node_238)
                    * ((base_row[121usize])
                        + (BFieldElement::from_raw_u64(3305272539067787726u64)))))
                + ((node_245)
                    * ((base_row[121usize])
                        + (BFieldElement::from_raw_u64(674972795234232729u64)))),
            (((((node_102)
                * ((base_row[122usize])
                    + (BFieldElement::from_raw_u64(13508500531157332153u64))))
                + ((node_221)
                    * ((base_row[122usize])
                        + (BFieldElement::from_raw_u64(3723900760706330287u64)))))
                + ((base_row[155usize])
                    * ((base_row[122usize])
                        + (BFieldElement::from_raw_u64(12579737103870920763u64)))))
                + ((node_238)
                    * ((base_row[122usize])
                        + (BFieldElement::from_raw_u64(17082569335437832789u64)))))
                + ((node_245)
                    * ((base_row[122usize])
                        + (BFieldElement::from_raw_u64(14165256104883557753u64)))),
            (((((node_102)
                * ((base_row[123usize])
                    + (BFieldElement::from_raw_u64(6968886508437513677u64))))
                + ((node_221)
                    * ((base_row[123usize])
                        + (BFieldElement::from_raw_u64(615596267195055952u64)))))
                + ((base_row[155usize])
                    * ((base_row[123usize])
                        + (BFieldElement::from_raw_u64(10119826060478909841u64)))))
                + ((node_238)
                    * ((base_row[123usize])
                        + (BFieldElement::from_raw_u64(229051680548583225u64)))))
                + ((node_245)
                    * ((base_row[123usize])
                        + (BFieldElement::from_raw_u64(15283356519694111298u64)))),
            (((((node_102)
                * ((base_row[124usize])
                    + (BFieldElement::from_raw_u64(9713264609690967820u64))))
                + ((node_221)
                    * ((base_row[124usize])
                        + (BFieldElement::from_raw_u64(18227830850447556704u64)))))
                + ((base_row[155usize])
                    * ((base_row[124usize])
                        + (BFieldElement::from_raw_u64(1528714547662620921u64)))))
                + ((node_238)
                    * ((base_row[124usize])
                        + (BFieldElement::from_raw_u64(2943254981416254648u64)))))
                + ((node_245)
                    * ((base_row[124usize])
                        + (BFieldElement::from_raw_u64(2306049938060341466u64)))),
            (((((node_102)
                * ((base_row[125usize])
                    + (BFieldElement::from_raw_u64(12482374976099749513u64))))
                + ((node_221)
                    * ((base_row[125usize])
                        + (BFieldElement::from_raw_u64(15609691041895848348u64)))))
                + ((base_row[155usize])
                    * ((base_row[125usize])
                        + (BFieldElement::from_raw_u64(12972275929555275935u64)))))
                + ((node_238)
                    * ((base_row[125usize])
                        + (BFieldElement::from_raw_u64(5767629304344025219u64)))))
                + ((node_245)
                    * ((base_row[125usize])
                        + (BFieldElement::from_raw_u64(11578793764462375094u64)))),
            (((((node_102)
                * ((base_row[126usize])
                    + (BFieldElement::from_raw_u64(13209711277645656680u64))))
                + ((node_221)
                    * ((base_row[126usize])
                        + (BFieldElement::from_raw_u64(15235800289984546486u64)))))
                + ((base_row[155usize])
                    * ((base_row[126usize])
                        + (BFieldElement::from_raw_u64(15992731669612695172u64)))))
                + ((node_238)
                    * ((base_row[126usize])
                        + (BFieldElement::from_raw_u64(16721422493821450473u64)))))
                + ((node_245)
                    * ((base_row[126usize])
                        + (BFieldElement::from_raw_u64(7511767364422267184u64)))),
            (((((node_102)
                * ((base_row[127usize])
                    + (BFieldElement::from_raw_u64(87705059284758253u64))))
                + ((node_221)
                    * ((base_row[127usize])
                        + (BFieldElement::from_raw_u64(11392407538241985753u64)))))
                + ((base_row[155usize])
                    * ((base_row[127usize])
                        + (BFieldElement::from_raw_u64(17877154195438905917u64)))))
                + ((node_238)
                    * ((base_row[127usize])
                        + (BFieldElement::from_raw_u64(5753720429376839714u64)))))
                + ((node_245)
                    * ((base_row[127usize])
                        + (BFieldElement::from_raw_u64(16999805755930336630u64)))),
            (((((node_102)
                * ((base_row[128usize])
                    + (BFieldElement::from_raw_u64(330155256278907084u64))))
                + ((node_221)
                    * ((base_row[128usize])
                        + (BFieldElement::from_raw_u64(11776128816341368822u64)))))
                + ((base_row[155usize])
                    * ((base_row[128usize])
                        + (BFieldElement::from_raw_u64(939319986782105612u64)))))
                + ((node_238)
                    * ((base_row[128usize])
                        + (BFieldElement::from_raw_u64(2063756830275051942u64)))))
                + ((node_245)
                    * ((base_row[128usize])
                        + (BFieldElement::from_raw_u64(940614108343834936u64)))),
            (base_row[129usize])
                * ((BFieldElement::from_raw_u64(4294967295u64))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (base_row[129usize]))),
            (base_row[135usize])
                * ((BFieldElement::from_raw_u64(4294967295u64))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (base_row[135usize]))),
            (base_row[139usize])
                * ((BFieldElement::from_raw_u64(4294967295u64))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (base_row[139usize]))),
            (base_row[139usize]) * (base_row[140usize]),
            (BFieldElement::from_raw_u64(4294967295u64))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((base_row[141usize])
                        * ((base_row[140usize])
                            + (BFieldElement::from_raw_u64(18446743927680663586u64))))),
            (base_row[144usize]) * (node_655),
            (base_row[143usize]) * (node_655),
            (base_row[146usize]) * (node_660),
            (base_row[145usize]) * (node_660),
            (base_row[167usize])
                * ((base_row[147usize])
                    + (BFieldElement::from_raw_u64(18446744060824649731u64))),
            (base_row[168usize]) * (base_row[147usize]),
            (((base_row[163usize]) * (node_655)) * (node_660)) * (base_row[147usize]),
            (((base_row[166usize]) * (node_678)) * (node_660))
                * ((base_row[147usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (((base_row[164usize]) * (node_680)) * (node_655))
                * ((base_row[147usize]) + (BFieldElement::from_raw_u64(4294967295u64))),
            (((base_row[166usize]) * (node_674)) * (node_655)) * (base_row[147usize]),
            ((base_row[164usize]) * (base_row[139usize])) * (node_655),
            (node_680) * (base_row[148usize]),
            (base_row[151usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((base_row[64usize]) * (node_89))),
            (base_row[152usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((node_89) * (node_90)) * (node_94)) * (node_97))),
            (base_row[153usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((base_row[64usize]) * (node_90)) * (node_94)) * (node_97))),
            (base_row[154usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((base_row[151usize]) * (node_90))),
            (base_row[155usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((base_row[151usize]) * (node_94)) * (node_97)) * (node_101))),
            (base_row[156usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_663)
                        * ((base_row[142usize])
                            + (BFieldElement::from_raw_u64(18446744043644780551u64))))),
            (base_row[157usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((base_row[143usize]) * (base_row[144usize]))),
            (base_row[158usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((node_663) * (node_666)) * (node_670)) * (node_674))),
            (base_row[159usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((base_row[156usize]) * (node_666))),
            (base_row[160usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((base_row[145usize]) * (base_row[146usize]))),
            (base_row[161usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_86) * (base_row[62usize]))),
            (base_row[162usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((base_row[158usize]) * (node_678))),
            (base_row[163usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((base_row[156usize]) * (node_670)) * (node_674)) * (node_678))),
            (base_row[164usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((base_row[159usize]) * (node_674)) * (node_678))),
            (base_row[165usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((node_83) * (node_103)) * (base_row[62usize]))
                        * ((base_row[63usize])
                            + (BFieldElement::from_raw_u64(18446743897615892521u64))))),
            (base_row[166usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((base_row[159usize]) * (node_670))),
            (base_row[167usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((base_row[162usize]) * (node_680)) * (node_655)) * (node_660))),
            (base_row[168usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((base_row[162usize]) * (base_row[139usize])) * (node_655))
                        * (node_660))),
        ];
        let ext_constraints = [];
        base_constraints
            .into_iter()
            .map(|bfe| bfe.lift())
            .chain(ext_constraints)
            .collect()
    }
    #[allow(unused_variables)]
    fn evaluate_transition_constraints(
        current_base_row: ArrayView1<BFieldElement>,
        current_ext_row: ArrayView1<XFieldElement>,
        next_base_row: ArrayView1<BFieldElement>,
        next_ext_row: ArrayView1<XFieldElement>,
        challenges: &Challenges,
    ) -> Vec<XFieldElement> {
        let node_120 = (next_base_row[19usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[19usize]));
        let node_517 = (next_ext_row[3usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[3usize]));
        let node_521 = (next_ext_row[4usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[4usize]));
        let node_124 = (next_base_row[20usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[20usize]));
        let node_4096 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (next_base_row[8usize]));
        let node_128 = (next_base_row[21usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[21usize]));
        let node_513 = (next_ext_row[7usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[7usize]));
        let node_1848 = (current_base_row[18usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_158 = (next_base_row[38usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[38usize]));
        let node_1224 = ((next_base_row[9usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[9usize])))
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_1846 = (current_base_row[17usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_785 = (next_base_row[22usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[22usize]));
        let node_167 = ((challenges[16usize]) * (current_base_row[7usize]))
            + ((challenges[17usize]) * (current_base_row[13usize]));
        let node_1219 = (next_ext_row[6usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[6usize]));
        let node_1230 = (next_base_row[28usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[28usize]));
        let node_1231 = (next_base_row[29usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[29usize]));
        let node_1232 = (next_base_row[30usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[30usize]));
        let node_1233 = (next_base_row[31usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[31usize]));
        let node_1234 = (next_base_row[32usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[32usize]));
        let node_1235 = (next_base_row[33usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[33usize]));
        let node_1236 = (next_base_row[34usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[34usize]));
        let node_1237 = (next_base_row[35usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[35usize]));
        let node_1238 = (next_base_row[36usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[36usize]));
        let node_1239 = (next_base_row[37usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[37usize]));
        let node_5778 = (current_base_row[280usize])
            * ((next_base_row[64usize])
                + (BFieldElement::from_raw_u64(18446744052234715141u64)));
        let node_868 = ((((((((((((((((challenges[32usize]) * (next_base_row[22usize]))
            + ((challenges[33usize]) * (next_base_row[23usize])))
            + ((challenges[34usize]) * (next_base_row[24usize])))
            + ((challenges[35usize]) * (next_base_row[25usize])))
            + ((challenges[36usize]) * (next_base_row[26usize])))
            + ((challenges[37usize]) * (next_base_row[27usize])))
            + ((challenges[38usize]) * (next_base_row[28usize])))
            + ((challenges[39usize]) * (next_base_row[29usize])))
            + ((challenges[40usize]) * (next_base_row[30usize])))
            + ((challenges[41usize]) * (next_base_row[31usize])))
            + ((challenges[42usize]) * (next_base_row[32usize])))
            + ((challenges[43usize]) * (next_base_row[33usize])))
            + ((challenges[44usize]) * (next_base_row[34usize])))
            + ((challenges[45usize]) * (next_base_row[35usize])))
            + ((challenges[46usize]) * (next_base_row[36usize])))
            + ((challenges[47usize]) * (next_base_row[37usize]));
        let node_1229 = (next_base_row[27usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[27usize]));
        let node_5855 = (((next_base_row[63usize])
            + (BFieldElement::from_raw_u64(18446743992105173011u64)))
            * ((next_base_row[63usize])
                + (BFieldElement::from_raw_u64(18446743923385696291u64))))
            * ((next_base_row[63usize])
                + (BFieldElement::from_raw_u64(18446743828896415801u64)));
        let node_4832 = (((((current_base_row[81usize])
            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
            + ((current_base_row[82usize])
                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
            + ((current_base_row[83usize])
                * (BFieldElement::from_raw_u64(281474976645120u64))))
            + (current_base_row[84usize])) * (BFieldElement::from_raw_u64(1u64));
        let node_4843 = (((((current_base_row[85usize])
            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
            + ((current_base_row[86usize])
                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
            + ((current_base_row[87usize])
                * (BFieldElement::from_raw_u64(281474976645120u64))))
            + (current_base_row[88usize])) * (BFieldElement::from_raw_u64(1u64));
        let node_4854 = (((((current_base_row[89usize])
            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
            + ((current_base_row[90usize])
                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
            + ((current_base_row[91usize])
                * (BFieldElement::from_raw_u64(281474976645120u64))))
            + (current_base_row[92usize])) * (BFieldElement::from_raw_u64(1u64));
        let node_4865 = (((((current_base_row[93usize])
            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
            + ((current_base_row[94usize])
                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
            + ((current_base_row[95usize])
                * (BFieldElement::from_raw_u64(281474976645120u64))))
            + (current_base_row[96usize])) * (BFieldElement::from_raw_u64(1u64));
        let node_200 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[27usize]);
        let node_870 = (challenges[33usize]) * (current_base_row[23usize]);
        let node_872 = (challenges[34usize]) * (current_base_row[24usize]);
        let node_874 = (challenges[35usize]) * (current_base_row[25usize]);
        let node_876 = (challenges[36usize]) * (current_base_row[26usize]);
        let node_878 = (challenges[37usize]) * (current_base_row[27usize]);
        let node_880 = (challenges[38usize]) * (current_base_row[28usize]);
        let node_882 = (challenges[39usize]) * (current_base_row[29usize]);
        let node_884 = (challenges[40usize]) * (current_base_row[30usize]);
        let node_886 = (challenges[41usize]) * (current_base_row[31usize]);
        let node_888 = (challenges[42usize]) * (current_base_row[32usize]);
        let node_890 = (challenges[43usize]) * (current_base_row[33usize]);
        let node_892 = (challenges[44usize]) * (current_base_row[34usize]);
        let node_894 = (challenges[45usize]) * (current_base_row[35usize]);
        let node_896 = (challenges[46usize]) * (current_base_row[36usize]);
        let node_898 = (challenges[47usize]) * (current_base_row[37usize]);
        let node_1226 = (next_base_row[24usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[24usize]));
        let node_1227 = (next_base_row[25usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[25usize]));
        let node_1228 = (next_base_row[26usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[26usize]));
        let node_196 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[26usize]);
        let node_1225 = (next_base_row[23usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[23usize]));
        let node_192 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[25usize]);
        let node_204 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[28usize]);
        let node_208 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[29usize]);
        let node_212 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[30usize]);
        let node_216 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[31usize]);
        let node_220 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[32usize]);
        let node_224 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[33usize]);
        let node_1844 = (current_base_row[16usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_228 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[34usize]);
        let node_197 = (next_base_row[25usize]) + (node_196);
        let node_201 = (next_base_row[26usize]) + (node_200);
        let node_205 = (next_base_row[27usize]) + (node_204);
        let node_209 = (next_base_row[28usize]) + (node_208);
        let node_213 = (next_base_row[29usize]) + (node_212);
        let node_217 = (next_base_row[30usize]) + (node_216);
        let node_159 = (node_158) + (BFieldElement::from_raw_u64(4294967295u64));
        let node_221 = (next_base_row[31usize]) + (node_220);
        let node_225 = (next_base_row[32usize]) + (node_224);
        let node_229 = (next_base_row[33usize]) + (node_228);
        let node_233 = (next_base_row[34usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[35usize]));
        let node_237 = (next_base_row[35usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[36usize]));
        let node_241 = (next_base_row[36usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[37usize]));
        let node_181 = (next_ext_row[6usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((current_ext_row[6usize])
                    * ((challenges[7usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((node_167)
                                + ((challenges[18usize]) * (next_base_row[38usize])))
                                + ((challenges[19usize]) * (next_base_row[37usize])))))));
        let node_6244 = (next_base_row[139usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_1317 = ((current_base_row[7usize]) * (challenges[20usize]))
            + (challenges[23usize]);
        let node_184 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[23usize]);
        let node_188 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[24usize]);
        let node_232 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[35usize]);
        let node_116 = ((next_base_row[9usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[9usize])))
            + (BFieldElement::from_raw_u64(18446744060824649731u64));
        let node_189 = (next_base_row[23usize]) + (node_188);
        let node_193 = (next_base_row[24usize]) + (node_192);
        let node_4274 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (next_base_row[44usize]));
        let node_4601 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (((next_base_row[52usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[52usize]))) * (current_base_row[54usize])));
        let node_236 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[36usize]);
        let node_525 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[22usize]);
        let node_240 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[37usize]);
        let node_154 = ((((current_base_row[11usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((BFieldElement::from_raw_u64(34359738360u64))
                    * (current_base_row[42usize]))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((BFieldElement::from_raw_u64(17179869180u64))
                    * (current_base_row[41usize]))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((BFieldElement::from_raw_u64(8589934590u64))
                    * (current_base_row[40usize]))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[39usize]));
        let node_295 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[39usize]));
        let node_4598 = (next_base_row[52usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[52usize]));
        let node_5621 = (next_base_row[62usize])
            + (BFieldElement::from_raw_u64(18446744056529682436u64));
        let node_6240 = (current_base_row[145usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((BFieldElement::from_raw_u64(8589934590u64))
                    * (next_base_row[145usize])));
        let node_4476 = (next_ext_row[12usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[12usize]));
        let node_6237 = (current_base_row[143usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((BFieldElement::from_raw_u64(8589934590u64))
                    * (next_base_row[143usize])));
        let node_1842 = (current_base_row[15usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_1315 = (current_base_row[7usize]) * (challenges[20usize]);
        let node_1665 = (challenges[1usize]) * (current_ext_row[3usize]);
        let node_5625 = (next_base_row[63usize])
            + (BFieldElement::from_raw_u64(18446743897615892521u64));
        let node_522 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[11usize]);
        let node_1241 = (current_base_row[276usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_1298 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[275usize]));
        let node_113 = (next_base_row[9usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[9usize]));
        let node_527 = (next_base_row[24usize]) + (node_184);
        let node_528 = (next_base_row[25usize]) + (node_188);
        let node_529 = (next_base_row[26usize]) + (node_192);
        let node_530 = (next_base_row[27usize]) + (node_196);
        let node_531 = (next_base_row[28usize]) + (node_200);
        let node_532 = (next_base_row[29usize]) + (node_204);
        let node_533 = (next_base_row[30usize]) + (node_208);
        let node_534 = (next_base_row[31usize]) + (node_212);
        let node_541 = (node_158)
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_535 = (next_base_row[32usize]) + (node_216);
        let node_536 = (next_base_row[33usize]) + (node_220);
        let node_537 = (next_base_row[34usize]) + (node_224);
        let node_538 = (next_base_row[35usize]) + (node_228);
        let node_539 = (next_base_row[36usize]) + (node_232);
        let node_540 = (next_base_row[37usize]) + (node_236);
        let node_550 = (next_ext_row[6usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((current_ext_row[6usize])
                    * ((challenges[7usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((node_167)
                                + ((challenges[18usize]) * (current_base_row[38usize])))
                                + ((challenges[19usize])
                                    * (current_base_row[37usize])))))));
        let node_4705 = ((next_base_row[59usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[59usize])))
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_5492 = (((((next_base_row[65usize])
            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
            + ((next_base_row[66usize])
                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
            + ((next_base_row[67usize])
                * (BFieldElement::from_raw_u64(281474976645120u64))))
            + (next_base_row[68usize])) * (BFieldElement::from_raw_u64(1u64));
        let node_5503 = (((((next_base_row[69usize])
            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
            + ((next_base_row[70usize])
                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
            + ((next_base_row[71usize])
                * (BFieldElement::from_raw_u64(281474976645120u64))))
            + (next_base_row[72usize])) * (BFieldElement::from_raw_u64(1u64));
        let node_5514 = (((((next_base_row[73usize])
            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
            + ((next_base_row[74usize])
                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
            + ((next_base_row[75usize])
                * (BFieldElement::from_raw_u64(281474976645120u64))))
            + (next_base_row[76usize])) * (BFieldElement::from_raw_u64(1u64));
        let node_5525 = (((((next_base_row[77usize])
            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
            + ((next_base_row[78usize])
                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
            + ((next_base_row[79usize])
                * (BFieldElement::from_raw_u64(281474976645120u64))))
            + (next_base_row[80usize])) * (BFieldElement::from_raw_u64(1u64));
        let node_5616 = (next_base_row[62usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_6184 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (next_base_row[135usize]));
        let node_6271 = (next_base_row[142usize])
            + (BFieldElement::from_raw_u64(18446743940565565471u64));
        let node_1840 = (current_base_row[14usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_248 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[40usize]));
        let node_6275 = (next_base_row[142usize])
            + (BFieldElement::from_raw_u64(18446743949155500061u64));
        let node_34 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((current_base_row[4usize])
                    * ((BFieldElement::from_raw_u64(38654705655u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[3usize])))));
        let node_185 = (next_base_row[22usize]) + (node_184);
        let node_1590 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[317usize]));
        let node_526 = (next_base_row[23usize]) + (node_525);
        let node_329 = (next_base_row[25usize]) + (node_204);
        let node_414 = (next_base_row[27usize]) + (node_220);
        let node_330 = (next_base_row[26usize]) + (node_208);
        let node_415 = (next_base_row[28usize]) + (node_224);
        let node_331 = (next_base_row[27usize]) + (node_212);
        let node_416 = (next_base_row[29usize]) + (node_228);
        let node_332 = (next_base_row[28usize]) + (node_216);
        let node_417 = (next_base_row[30usize]) + (node_232);
        let node_333 = (next_base_row[29usize]) + (node_220);
        let node_418 = (next_base_row[31usize]) + (node_236);
        let node_334 = (next_base_row[30usize]) + (node_224);
        let node_437 = (((((current_base_row[185usize]) * (node_159))
            + ((current_base_row[186usize])
                * ((node_158) + (BFieldElement::from_raw_u64(8589934590u64)))))
            + ((current_base_row[187usize])
                * ((node_158) + (BFieldElement::from_raw_u64(12884901885u64)))))
            + ((current_base_row[189usize])
                * ((node_158) + (BFieldElement::from_raw_u64(17179869180u64)))))
            + ((current_base_row[190usize])
                * ((node_158) + (BFieldElement::from_raw_u64(21474836475u64))));
        let node_730 = (((((current_base_row[185usize]) * (node_541))
            + ((current_base_row[186usize])
                * ((node_158) + (BFieldElement::from_raw_u64(18446744060824649731u64)))))
            + ((current_base_row[187usize])
                * ((node_158) + (BFieldElement::from_raw_u64(18446744056529682436u64)))))
            + ((current_base_row[189usize])
                * ((node_158) + (BFieldElement::from_raw_u64(18446744052234715141u64)))))
            + ((current_base_row[190usize])
                * ((node_158) + (BFieldElement::from_raw_u64(18446744047939747846u64))));
        let node_419 = (next_base_row[32usize]) + (node_240);
        let node_397 = (node_158) + (BFieldElement::from_raw_u64(21474836475u64));
        let node_335 = (next_base_row[31usize]) + (node_228);
        let node_441 = ((((current_ext_row[73usize]) + (current_ext_row[74usize]))
            + (current_ext_row[75usize])) + (current_ext_row[76usize]))
            + ((current_base_row[190usize])
                * ((next_ext_row[6usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_ext_row[59usize]))));
        let node_734 = ((((current_ext_row[77usize]) + (current_ext_row[78usize]))
            + (current_ext_row[79usize])) + (current_ext_row[80usize]))
            + (current_ext_row[81usize]);
        let node_408 = (next_ext_row[6usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[59usize]));
        let node_336 = (next_base_row[32usize]) + (node_232);
        let node_337 = (next_base_row[33usize]) + (node_236);
        let node_449 = ((((current_base_row[290usize]) + (current_base_row[291usize]))
            + (current_base_row[292usize])) + (current_base_row[293usize]))
            + (current_base_row[294usize]);
        let node_742 = ((((current_base_row[285usize]) + (current_base_row[286usize]))
            + (current_base_row[287usize])) + (current_base_row[288usize]))
            + (current_base_row[289usize]);
        let node_338 = (next_base_row[34usize]) + (node_240);
        let node_453 = (((((current_base_row[185usize]) * (node_193))
            + ((current_base_row[186usize]) * ((next_base_row[24usize]) + (node_196))))
            + ((current_base_row[187usize]) * ((next_base_row[24usize]) + (node_200))))
            + ((current_base_row[189usize]) * ((next_base_row[24usize]) + (node_204))))
            + ((current_base_row[190usize]) * ((next_base_row[24usize]) + (node_208)));
        let node_746 = (((((current_base_row[185usize]) * (node_528))
            + ((current_base_row[186usize]) * ((next_base_row[26usize]) + (node_188))))
            + ((current_base_row[187usize]) * ((next_base_row[27usize]) + (node_188))))
            + ((current_base_row[189usize]) * ((next_base_row[28usize]) + (node_188))))
            + ((current_base_row[190usize]) * ((next_base_row[29usize]) + (node_188)));
        let node_314 = (node_158) + (BFieldElement::from_raw_u64(12884901885u64));
        let node_457 = (((((current_base_row[185usize]) * (node_197))
            + ((current_base_row[186usize]) * ((next_base_row[25usize]) + (node_200))))
            + ((current_base_row[187usize]) * (node_329)))
            + ((current_base_row[189usize]) * ((next_base_row[25usize]) + (node_208))))
            + ((current_base_row[190usize]) * ((next_base_row[25usize]) + (node_212)));
        let node_750 = (((((current_base_row[185usize]) * (node_529))
            + ((current_base_row[186usize]) * ((next_base_row[27usize]) + (node_192))))
            + ((current_base_row[187usize]) * ((next_base_row[28usize]) + (node_192))))
            + ((current_base_row[189usize]) * ((next_base_row[29usize]) + (node_192))))
            + ((current_base_row[190usize]) * ((next_base_row[30usize]) + (node_192)));
        let node_325 = (next_ext_row[6usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((current_ext_row[6usize]) * (current_ext_row[52usize])));
        let node_461 = (((((current_base_row[185usize]) * (node_201))
            + ((current_base_row[186usize]) * ((next_base_row[26usize]) + (node_204))))
            + ((current_base_row[187usize]) * (node_330)))
            + ((current_base_row[189usize]) * ((next_base_row[26usize]) + (node_212))))
            + ((current_base_row[190usize]) * ((next_base_row[26usize]) + (node_216)));
        let node_754 = (((((current_base_row[185usize]) * (node_530))
            + ((current_base_row[186usize]) * ((next_base_row[28usize]) + (node_196))))
            + ((current_base_row[187usize]) * ((next_base_row[29usize]) + (node_196))))
            + ((current_base_row[189usize]) * ((next_base_row[30usize]) + (node_196))))
            + ((current_base_row[190usize]) * ((next_base_row[31usize]) + (node_196)));
        let node_465 = (((((current_base_row[185usize]) * (node_205))
            + ((current_base_row[186usize]) * ((next_base_row[27usize]) + (node_208))))
            + ((current_base_row[187usize]) * (node_331)))
            + ((current_base_row[189usize]) * ((next_base_row[27usize]) + (node_216))))
            + ((current_base_row[190usize]) * (node_414));
        let node_758 = (((((current_base_row[185usize]) * (node_531))
            + ((current_base_row[186usize]) * ((next_base_row[29usize]) + (node_200))))
            + ((current_base_row[187usize]) * ((next_base_row[30usize]) + (node_200))))
            + ((current_base_row[189usize]) * ((next_base_row[31usize]) + (node_200))))
            + ((current_base_row[190usize]) * ((next_base_row[32usize]) + (node_200)));
        let node_469 = (((((current_base_row[185usize]) * (node_209))
            + ((current_base_row[186usize]) * ((next_base_row[28usize]) + (node_212))))
            + ((current_base_row[187usize]) * (node_332)))
            + ((current_base_row[189usize]) * ((next_base_row[28usize]) + (node_220))))
            + ((current_base_row[190usize]) * (node_415));
        let node_762 = (((((current_base_row[185usize]) * (node_532))
            + ((current_base_row[186usize]) * ((next_base_row[30usize]) + (node_204))))
            + ((current_base_row[187usize]) * ((next_base_row[31usize]) + (node_204))))
            + ((current_base_row[189usize]) * ((next_base_row[32usize]) + (node_204))))
            + ((current_base_row[190usize]) * ((next_base_row[33usize]) + (node_204)));
        let node_473 = (((((current_base_row[185usize]) * (node_213))
            + ((current_base_row[186usize]) * ((next_base_row[29usize]) + (node_216))))
            + ((current_base_row[187usize]) * (node_333)))
            + ((current_base_row[189usize]) * ((next_base_row[29usize]) + (node_224))))
            + ((current_base_row[190usize]) * (node_416));
        let node_766 = (((((current_base_row[185usize]) * (node_533))
            + ((current_base_row[186usize]) * ((next_base_row[31usize]) + (node_208))))
            + ((current_base_row[187usize]) * ((next_base_row[32usize]) + (node_208))))
            + ((current_base_row[189usize]) * ((next_base_row[33usize]) + (node_208))))
            + ((current_base_row[190usize]) * ((next_base_row[34usize]) + (node_208)));
        let node_477 = (((((current_base_row[185usize]) * (node_217))
            + ((current_base_row[186usize]) * ((next_base_row[30usize]) + (node_220))))
            + ((current_base_row[187usize]) * (node_334)))
            + ((current_base_row[189usize]) * ((next_base_row[30usize]) + (node_228))))
            + ((current_base_row[190usize]) * (node_417));
        let node_770 = (((((current_base_row[185usize]) * (node_534))
            + ((current_base_row[186usize]) * ((next_base_row[32usize]) + (node_212))))
            + ((current_base_row[187usize]) * ((next_base_row[33usize]) + (node_212))))
            + ((current_base_row[189usize]) * ((next_base_row[34usize]) + (node_212))))
            + ((current_base_row[190usize]) * ((next_base_row[35usize]) + (node_212)));
        let node_481 = (((((current_base_row[185usize]) * (node_221))
            + ((current_base_row[186usize]) * ((next_base_row[31usize]) + (node_224))))
            + ((current_base_row[187usize]) * (node_335)))
            + ((current_base_row[189usize]) * ((next_base_row[31usize]) + (node_232))))
            + ((current_base_row[190usize]) * (node_418));
        let node_774 = (((((current_base_row[185usize]) * (node_535))
            + ((current_base_row[186usize]) * ((next_base_row[33usize]) + (node_216))))
            + ((current_base_row[187usize]) * ((next_base_row[34usize]) + (node_216))))
            + ((current_base_row[189usize]) * ((next_base_row[35usize]) + (node_216))))
            + ((current_base_row[190usize]) * ((next_base_row[36usize]) + (node_216)));
        let node_485 = (((((current_base_row[185usize]) * (node_225))
            + ((current_base_row[186usize]) * ((next_base_row[32usize]) + (node_228))))
            + ((current_base_row[187usize]) * (node_336)))
            + ((current_base_row[189usize]) * ((next_base_row[32usize]) + (node_236))))
            + ((current_base_row[190usize]) * (node_419));
        let node_778 = (((((current_base_row[185usize]) * (node_536))
            + ((current_base_row[186usize]) * ((next_base_row[34usize]) + (node_220))))
            + ((current_base_row[187usize]) * ((next_base_row[35usize]) + (node_220))))
            + ((current_base_row[189usize]) * ((next_base_row[36usize]) + (node_220))))
            + ((current_base_row[190usize]) * ((next_base_row[37usize]) + (node_220)));
        let node_488 = ((((current_base_row[185usize]) * (node_229))
            + ((current_base_row[186usize]) * ((next_base_row[33usize]) + (node_232))))
            + ((current_base_row[187usize]) * (node_337)))
            + ((current_base_row[189usize]) * ((next_base_row[33usize]) + (node_240)));
        let node_781 = ((((current_base_row[185usize]) * (node_537))
            + ((current_base_row[186usize]) * ((next_base_row[35usize]) + (node_224))))
            + ((current_base_row[187usize]) * ((next_base_row[36usize]) + (node_224))))
            + ((current_base_row[189usize]) * ((next_base_row[37usize]) + (node_224)));
        let node_490 = (((current_base_row[185usize]) * (node_233))
            + ((current_base_row[186usize]) * ((next_base_row[34usize]) + (node_236))))
            + ((current_base_row[187usize]) * (node_338));
        let node_783 = (((current_base_row[185usize]) * (node_538))
            + ((current_base_row[186usize]) * ((next_base_row[36usize]) + (node_228))))
            + ((current_base_row[187usize]) * ((next_base_row[37usize]) + (node_228)));
        let node_491 = ((current_base_row[185usize]) * (node_237))
            + ((current_base_row[186usize]) * ((next_base_row[35usize]) + (node_240)));
        let node_784 = ((current_base_row[185usize]) * (node_539))
            + ((current_base_row[186usize]) * ((next_base_row[37usize]) + (node_232)));
        let node_267 = (current_base_row[185usize]) * (node_241);
        let node_567 = (current_base_row[185usize]) * (node_540);
        let node_4387 = ((next_ext_row[11usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((challenges[6usize]) * (current_ext_row[11usize]))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((challenges[31usize]) * (current_base_row[10usize])));
        let node_4440 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * ((challenges[57usize]) * (current_base_row[10usize]));
        let node_4480 = ((node_4476)
            * (((((challenges[10usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((challenges[55usize]) * (current_base_row[22usize]))))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((challenges[56usize]) * (current_base_row[23usize]))))
                + (node_4440))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((challenges[58usize]) * (next_base_row[22usize])))))
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_4444 = (challenges[10usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((challenges[55usize]) * (current_base_row[22usize])));
        let node_4524 = ((next_base_row[48usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[48usize])))
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_4523 = (next_base_row[48usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[48usize]));
        let node_4530 = (next_base_row[47usize])
            + (BFieldElement::from_raw_u64(18446744060824649731u64));
        let node_4556 = (next_ext_row[15usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[15usize]));
        let node_4593 = (next_base_row[51usize])
            + (BFieldElement::from_raw_u64(18446744060824649731u64));
        let node_4676 = (next_ext_row[21usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[21usize]));
        let node_4704 = (next_base_row[59usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[59usize]));
        let node_5613 = (current_base_row[62usize])
            + (BFieldElement::from_raw_u64(18446744056529682436u64));
        let node_5677 = (next_base_row[64usize])
            + (BFieldElement::from_raw_u64(18446744047939747846u64));
        let node_5728 = (next_base_row[63usize])
            + (BFieldElement::from_raw_u64(18446743992105173011u64));
        let node_5730 = (next_base_row[63usize])
            + (BFieldElement::from_raw_u64(18446743923385696291u64));
        let node_5847 = (next_ext_row[28usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[28usize]));
        let node_5868 = (next_ext_row[29usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[29usize]));
        let node_5885 = (next_ext_row[30usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[30usize]));
        let node_5902 = (next_ext_row[31usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[31usize]));
        let node_5919 = (next_ext_row[32usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[32usize]));
        let node_5936 = (next_ext_row[33usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[33usize]));
        let node_5953 = (next_ext_row[34usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[34usize]));
        let node_5970 = (next_ext_row[35usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[35usize]));
        let node_5987 = (next_ext_row[36usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[36usize]));
        let node_6004 = (next_ext_row[37usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[37usize]));
        let node_6021 = (next_ext_row[38usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[38usize]));
        let node_6038 = (next_ext_row[39usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[39usize]));
        let node_6055 = (next_ext_row[40usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[40usize]));
        let node_6072 = (next_ext_row[41usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[41usize]));
        let node_6089 = (next_ext_row[42usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[42usize]));
        let node_6106 = (next_ext_row[43usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[43usize]));
        let node_6132 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (next_base_row[129usize]));
        let node_6234 = (current_base_row[142usize])
            + (BFieldElement::from_raw_u64(18446743940565565471u64));
        let node_6261 = (node_6240)
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_6269 = (next_base_row[142usize])
            + (BFieldElement::from_raw_u64(18446744017874976781u64));
        let node_5637 = (next_base_row[62usize])
            + (BFieldElement::from_raw_u64(18446744060824649731u64));
        let node_30 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[3usize]);
        let node_47 = (next_base_row[6usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_51 = (next_ext_row[0usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[0usize]));
        let node_31 = (BFieldElement::from_raw_u64(38654705655u64)) + (node_30);
        let node_74 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (next_base_row[1usize]);
        let node_90 = (BFieldElement::from_raw_u64(38654705655u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (next_base_row[3usize]));
        let node_88 = (next_ext_row[2usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[2usize]));
        let node_253 = (current_base_row[185usize]) * (node_185);
        let node_299 = (current_base_row[186usize])
            * ((next_base_row[22usize]) + (node_188));
        let node_342 = (current_base_row[187usize])
            * ((next_base_row[22usize]) + (node_192));
        let node_384 = (current_base_row[189usize])
            * ((next_base_row[22usize]) + (node_196));
        let node_423 = (current_base_row[190usize])
            * ((next_base_row[22usize]) + (node_200));
        let node_804 = (next_base_row[22usize]) + (node_220);
        let node_887 = ((((((((((challenges[32usize]) * (current_base_row[22usize]))
            + (node_870)) + (node_872)) + (node_874)) + (node_876)) + (node_878))
            + (node_880)) + (node_882)) + (node_884)) + (node_886);
        let node_1223 = (next_base_row[10usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[10usize]));
        let node_1292 = (node_120) + (BFieldElement::from_raw_u64(4294967295u64));
        let node_1294 = (next_base_row[9usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[21usize]));
        let node_1584 = (next_base_row[22usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((current_base_row[22usize]) * (current_base_row[23usize])));
        let node_1585 = (next_base_row[22usize]) * (current_base_row[22usize]);
        let node_1609 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (next_base_row[22usize]);
        let node_1625 = ((current_base_row[24usize]) * (current_base_row[26usize]))
            + ((current_base_row[23usize]) * (current_base_row[27usize]));
        let node_1639 = (current_base_row[24usize]) * (next_base_row[23usize]);
        let node_1642 = (current_base_row[23usize]) * (next_base_row[24usize]);
        let node_1422 = (node_785)
            + (BFieldElement::from_raw_u64(18446744056529682436u64));
        let node_1396 = (node_785)
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_112 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[9usize]);
        let node_1293 = (next_base_row[9usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[20usize]));
        let node_1295 = (current_base_row[28usize]) + (node_200);
        let node_1587 = (current_base_row[23usize]) + (node_525);
        let node_1743 = (node_1225)
            + (BFieldElement::from_raw_u64(18446744056529682436u64));
        let node_247 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[40usize]);
        let node_144 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * ((BFieldElement::from_raw_u64(34359738360u64))
                * (current_base_row[42usize]));
        let node_1785 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (next_ext_row[7usize]);
        let node_1797 = ((current_base_row[41usize]) * (current_base_row[43usize]))
            + ((current_base_row[40usize]) * (current_base_row[44usize]));
        let node_1798 = (current_base_row[41usize]) * (current_base_row[44usize]);
        let node_445 = ((((node_253) + (node_299)) + (node_342)) + (node_384))
            + (node_423);
        let node_738 = (((((current_base_row[185usize]) * (node_526))
            + ((current_base_row[186usize]) * ((next_base_row[24usize]) + (node_525))))
            + ((current_base_row[187usize]) * ((next_base_row[25usize]) + (node_525))))
            + ((current_base_row[189usize]) * ((next_base_row[26usize]) + (node_525))))
            + ((current_base_row[190usize]) * ((next_base_row[27usize]) + (node_525)));
        let node_409 = (next_base_row[22usize]) + (node_200);
        let node_410 = (next_base_row[23usize]) + (node_204);
        let node_411 = (next_base_row[24usize]) + (node_208);
        let node_412 = (next_base_row[25usize]) + (node_212);
        let node_413 = (next_base_row[26usize]) + (node_216);
        let node_1727 = ((challenges[2usize])
            * (((challenges[2usize])
                * (((challenges[2usize])
                    * (((challenges[2usize]) * (current_ext_row[4usize]))
                        + (current_base_row[22usize]))) + (current_base_row[23usize])))
                + (current_base_row[24usize]))) + (current_base_row[25usize]);
        let node_1722 = ((challenges[2usize])
            * (((challenges[2usize])
                * (((challenges[2usize]) * (current_ext_row[4usize]))
                    + (current_base_row[22usize]))) + (current_base_row[23usize])))
            + (current_base_row[24usize]);
        let node_1717 = ((challenges[2usize])
            * (((challenges[2usize]) * (current_ext_row[4usize]))
                + (current_base_row[22usize]))) + (current_base_row[23usize]);
        let node_1712 = ((challenges[2usize]) * (current_ext_row[4usize]))
            + (current_base_row[22usize]);
        let node_4210 = (next_ext_row[5usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[5usize]));
        let node_4334 = (next_ext_row[9usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((challenges[4usize]) * (current_ext_row[9usize])));
        let node_4335 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (((((((((((challenges[32usize]) * (next_base_row[22usize]))
                + ((challenges[33usize]) * (next_base_row[23usize])))
                + ((challenges[34usize]) * (next_base_row[24usize])))
                + ((challenges[35usize]) * (next_base_row[25usize])))
                + ((challenges[36usize]) * (next_base_row[26usize])))
                + ((challenges[37usize]) * (next_base_row[27usize])))
                + ((challenges[38usize]) * (next_base_row[28usize])))
                + ((challenges[39usize]) * (next_base_row[29usize])))
                + ((challenges[40usize]) * (next_base_row[30usize])))
                + ((challenges[41usize]) * (next_base_row[31usize])));
        let node_846 = (((((challenges[32usize]) * (next_base_row[22usize]))
            + ((challenges[33usize]) * (next_base_row[23usize])))
            + ((challenges[34usize]) * (next_base_row[24usize])))
            + ((challenges[35usize]) * (next_base_row[25usize])))
            + ((challenges[36usize]) * (next_base_row[26usize]));
        let node_4383 = (next_ext_row[11usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((challenges[6usize]) * (current_ext_row[11usize])));
        let node_4433 = (challenges[10usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((challenges[55usize]) * (next_base_row[22usize])));
        let node_4436 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * ((challenges[56usize]) * (next_base_row[23usize]));
        let node_4447 = (node_4444)
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((challenges[56usize]) * (current_base_row[23usize])));
        let node_4484 = ((node_4476)
            * (((node_4444) + (node_4440))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((challenges[58usize]) * (next_base_row[22usize])))))
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_4470 = (((node_4433)
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((challenges[56usize]) * (current_base_row[23usize]))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((challenges[57usize])
                    * (BFieldElement::from_raw_u64(25769803770u64)))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (challenges[58usize]));
        let node_4474 = ((node_4444) + (node_4436))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((challenges[57usize])
                    * (BFieldElement::from_raw_u64(17179869180u64))));
        let node_4547 = (next_base_row[47usize])
            * ((next_base_row[47usize])
                + (BFieldElement::from_raw_u64(18446744065119617026u64)));
        let node_4616 = (challenges[12usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (next_base_row[52usize]));
        let node_4621 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_ext_row[16usize]);
        let node_4667 = ((next_base_row[51usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64)))
            * (next_base_row[51usize]);
        let node_4709 = (node_4705)
            * ((current_base_row[58usize])
                + (BFieldElement::from_raw_u64(18446744000695107601u64)));
        let node_4717 = (next_base_row[57usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[57usize]));
        let node_4708 = (current_base_row[58usize])
            + (BFieldElement::from_raw_u64(18446744000695107601u64));
        let node_4739 = (next_ext_row[23usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[23usize]));
        let node_5595 = (current_base_row[63usize])
            + (BFieldElement::from_raw_u64(18446743897615892521u64));
        let node_5597 = (current_base_row[64usize])
            + (BFieldElement::from_raw_u64(18446744047939747846u64));
        let node_5828 = (next_ext_row[24usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[24usize]));
        let node_4776 = (((((current_base_row[65usize])
            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
            + ((current_base_row[66usize])
                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
            + ((current_base_row[67usize])
                * (BFieldElement::from_raw_u64(281474976645120u64))))
            + (current_base_row[68usize])) * (BFieldElement::from_raw_u64(1u64));
        let node_4787 = (((((current_base_row[69usize])
            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
            + ((current_base_row[70usize])
                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
            + ((current_base_row[71usize])
                * (BFieldElement::from_raw_u64(281474976645120u64))))
            + (current_base_row[72usize])) * (BFieldElement::from_raw_u64(1u64));
        let node_4798 = (((((current_base_row[73usize])
            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
            + ((current_base_row[74usize])
                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
            + ((current_base_row[75usize])
                * (BFieldElement::from_raw_u64(281474976645120u64))))
            + (current_base_row[76usize])) * (BFieldElement::from_raw_u64(1u64));
        let node_4809 = (((((current_base_row[77usize])
            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
            + ((current_base_row[78usize])
                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
            + ((current_base_row[79usize])
                * (BFieldElement::from_raw_u64(281474976645120u64))))
            + (current_base_row[80usize])) * (BFieldElement::from_raw_u64(1u64));
        let node_5627 = (node_5597) * (node_5595);
        let node_5641 = ((current_base_row[62usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64)))
            * ((current_base_row[62usize])
                + (BFieldElement::from_raw_u64(18446744060824649731u64)));
        let node_5659 = (challenges[42usize])
            * ((next_base_row[103usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (current_base_row[103usize])));
        let node_5660 = (challenges[43usize])
            * ((next_base_row[104usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (current_base_row[104usize])));
        let node_5662 = (challenges[44usize])
            * ((next_base_row[105usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (current_base_row[105usize])));
        let node_5664 = (challenges[45usize])
            * ((next_base_row[106usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (current_base_row[106usize])));
        let node_5666 = (challenges[46usize])
            * ((next_base_row[107usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (current_base_row[107usize])));
        let node_5668 = (challenges[47usize])
            * ((next_base_row[108usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (current_base_row[108usize])));
        let node_5758 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (((((((((((challenges[32usize]) * (node_5492))
                + ((challenges[33usize]) * (node_5503)))
                + ((challenges[34usize]) * (node_5514)))
                + ((challenges[35usize]) * (node_5525)))
                + ((challenges[36usize]) * (next_base_row[97usize])))
                + ((challenges[37usize]) * (next_base_row[98usize])))
                + ((challenges[38usize]) * (next_base_row[99usize])))
                + ((challenges[39usize]) * (next_base_row[100usize])))
                + ((challenges[40usize]) * (next_base_row[101usize])))
                + ((challenges[41usize]) * (next_base_row[102usize])));
        let node_5735 = (next_ext_row[25usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[25usize]));
        let node_5744 = (((((challenges[32usize]) * (node_5492))
            + ((challenges[33usize]) * (node_5503)))
            + ((challenges[34usize]) * (node_5514)))
            + ((challenges[35usize]) * (node_5525)))
            + ((challenges[36usize]) * (next_base_row[97usize]));
        let node_5769 = (next_ext_row[26usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[26usize]));
        let node_5795 = (next_ext_row[27usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[27usize]));
        let node_5798 = (next_base_row[63usize])
            + (BFieldElement::from_raw_u64(18446743828896415801u64));
        let node_6142 = (next_ext_row[44usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[44usize]));
        let node_6158 = (next_ext_row[45usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[45usize]));
        let node_6153 = ((challenges[52usize]) * (next_base_row[131usize]))
            + ((challenges[53usize]) * (next_base_row[133usize]));
        let node_6156 = ((challenges[52usize]) * (next_base_row[130usize]))
            + ((challenges[53usize]) * (next_base_row[132usize]));
        let node_6196 = (next_ext_row[46usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[46usize]));
        let node_6252 = ((next_base_row[140usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[140usize])))
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_6258 = (node_6237)
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_6278 = (next_base_row[147usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_6280 = (next_base_row[147usize])
            + (BFieldElement::from_raw_u64(18446744060824649731u64));
        let node_6283 = (current_base_row[313usize]) * (next_base_row[147usize]);
        let node_6285 = (current_base_row[147usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_6250 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[140usize]);
        let node_6344 = (next_base_row[147usize]) * (next_base_row[147usize]);
        let node_6294 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (node_6237);
        let node_6360 = (next_ext_row[48usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[48usize]));
        let node_1867 = (current_base_row[12usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_1850 = (current_base_row[13usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_243 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[42usize]));
        let node_245 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[41usize]));
        let node_133 = (current_base_row[40usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_5670 = (next_base_row[64usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_5671 = (next_base_row[64usize])
            + (BFieldElement::from_raw_u64(18446744060824649731u64));
        let node_5673 = (next_base_row[64usize])
            + (BFieldElement::from_raw_u64(18446744056529682436u64));
        let node_6263 = (next_base_row[142usize])
            + (BFieldElement::from_raw_u64(18446744052234715141u64));
        let node_6265 = (next_base_row[142usize])
            + (BFieldElement::from_raw_u64(18446744009285042191u64));
        let node_5675 = (next_base_row[64usize])
            + (BFieldElement::from_raw_u64(18446744052234715141u64));
        let node_4247 = (next_base_row[12usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_4249 = (next_base_row[14usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_4251 = (next_base_row[15usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_4254 = (next_base_row[17usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_5612 = (current_base_row[62usize])
            + (BFieldElement::from_raw_u64(18446744060824649731u64));
        let node_5634 = (current_base_row[62usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_176 = (challenges[7usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (((node_167) + ((challenges[18usize]) * (next_base_row[38usize])))
                    + ((challenges[19usize]) * (next_base_row[37usize]))));
        let node_547 = (challenges[7usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (((node_167) + ((challenges[18usize]) * (current_base_row[38usize])))
                    + ((challenges[19usize]) * (current_base_row[37usize]))));
        let node_1326 = (challenges[8usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (((node_1317)
                    + (((next_base_row[22usize])
                        + (BFieldElement::from_raw_u64(4294967295u64)))
                        * (challenges[21usize])))
                    + ((next_base_row[23usize]) * (challenges[22usize]))));
        let node_1402 = (challenges[8usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (((node_1315) + ((current_base_row[22usize]) * (challenges[21usize])))
                    + ((current_base_row[23usize]) * (challenges[22usize]))));
        let node_1410 = ((current_base_row[22usize])
            + (BFieldElement::from_raw_u64(4294967295u64))) * (challenges[21usize]);
        let node_1424 = ((current_base_row[22usize])
            + (BFieldElement::from_raw_u64(8589934590u64))) * (challenges[21usize]);
        let node_1750 = (challenges[8usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (((node_1317) + ((current_base_row[22usize]) * (challenges[21usize])))
                    + ((current_base_row[39usize]) * (challenges[22usize]))));
        let node_1752 = (current_base_row[40usize]) * (challenges[22usize]);
        let node_1758 = (current_base_row[41usize]) * (challenges[22usize]);
        let node_1764 = (node_1317)
            + ((current_base_row[23usize]) * (challenges[21usize]));
        let node_1765 = (current_base_row[42usize]) * (challenges[22usize]);
        let node_1771 = (node_1317)
            + (((current_base_row[23usize])
                + (BFieldElement::from_raw_u64(4294967295u64))) * (challenges[21usize]));
        let node_1778 = (node_1317)
            + (((current_base_row[23usize])
                + (BFieldElement::from_raw_u64(8589934590u64))) * (challenges[21usize]));
        let base_constraints = [
            (next_base_row[0usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[0usize])
                        + (BFieldElement::from_raw_u64(4294967295u64)))),
            (current_base_row[6usize])
                * ((next_base_row[6usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[6usize]))),
            ((node_34) * (next_base_row[3usize]))
                + ((current_base_row[4usize])
                    * (((next_base_row[3usize]) + (node_30))
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)))),
            (current_base_row[5usize])
                * ((next_base_row[5usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (((current_base_row[5usize])
                + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                * (next_base_row[5usize]))
                * ((next_base_row[1usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (current_base_row[5usize]) * (next_base_row[1usize]),
            ((current_base_row[5usize]) * (node_34)) * (node_47),
            ((next_base_row[7usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (current_base_row[7usize])))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            (current_base_row[8usize])
                * ((next_base_row[8usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[8usize]))),
            ((((((((((((((((((((((((((((((((((((((((((((current_base_row[207usize])
                * (node_124)) + ((current_base_row[213usize]) * (node_526)))
                + ((current_base_row[209usize]) * (node_124)))
                + ((current_base_row[211usize]) * (current_base_row[328usize])))
                + ((current_base_row[232usize]) * (current_base_row[328usize])))
                + ((current_base_row[216usize]) * (node_120)))
                + ((current_base_row[221usize]) * (node_124)))
                + ((current_base_row[214usize])
                    * ((node_1241) * (current_base_row[22usize]))))
                + ((current_base_row[224usize])
                    * (((next_base_row[20usize]) + (node_112))
                        + (BFieldElement::from_raw_u64(18446744060824649731u64)))))
                + ((current_base_row[228usize]) * (node_1293)))
                + ((current_base_row[227usize]) * (node_120)))
                + ((current_base_row[229usize])
                    * (((node_1298) * (node_1295))
                        + ((current_base_row[275usize]) * (node_128)))))
                + ((current_base_row[230usize]) * (node_120)))
                + ((current_base_row[219usize]) * (node_124)))
                + ((current_base_row[226usize]) * (node_124)))
                + ((current_base_row[248usize]) * (node_124)))
                + ((current_base_row[233usize])
                    * ((current_base_row[28usize]) + (node_184))))
                + ((current_base_row[234usize]) * (node_124)))
                + ((current_base_row[250usize]) * (node_124)))
                + ((current_base_row[244usize]) * (node_120)))
                + ((current_base_row[254usize]) * (node_124)))
                + ((current_base_row[237usize]) * (node_120)))
                + ((current_base_row[239usize]) * (node_120)))
                + ((current_base_row[240usize]) * (node_120)))
                + ((current_base_row[241usize]) * ((node_1587) * (node_1590))))
                + (current_base_row[344usize]))
                + ((current_base_row[242usize]) * (node_124)))
                + ((current_base_row[243usize]) * (node_124)))
                + ((current_base_row[245usize]) * (node_124)))
                + ((current_base_row[246usize]) * (node_124)))
                + ((current_base_row[247usize]) * (node_124)))
                + ((current_base_row[249usize]) * (node_120)))
                + ((current_base_row[251usize]) * (node_124)))
                + ((current_base_row[277usize]) * ((node_1225) + (node_196))))
                + ((current_base_row[278usize])
                    * ((next_base_row[23usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((current_base_row[23usize])
                                * (current_base_row[25usize]))
                                + ((current_base_row[22usize])
                                    * (current_base_row[26usize])))
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (current_base_row[322usize]))) + (node_1625))))))
                + ((current_base_row[279usize])
                    * ((((((current_base_row[23usize]) * (next_base_row[22usize]))
                        + ((current_base_row[22usize]) * (next_base_row[23usize])))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[323usize]))) + (node_1639))
                        + (node_1642))))
                + ((current_base_row[281usize])
                    * ((next_base_row[23usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((current_base_row[24usize])
                                * (current_base_row[22usize]))))))
                + ((current_base_row[255usize]) * (node_124)))
                + ((current_base_row[258usize]) * (node_124)))
                + ((current_base_row[295usize])
                    * ((((next_base_row[27usize])
                        * (BFieldElement::from_raw_u64(8589934590u64)))
                        + (current_base_row[44usize])) + (node_200))))
                + ((current_base_row[296usize]) * (node_1743)))
                + ((current_base_row[297usize]) * (node_1743))) * (node_4096))
                + ((node_1223) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((((((current_base_row[207usize])
                * (node_116)) + ((current_base_row[213usize]) * (node_528)))
                + ((current_base_row[209usize]) * (node_116)))
                + ((current_base_row[211usize]) * (current_base_row[252usize])))
                + ((current_base_row[232usize]) * (current_base_row[252usize])))
                + ((current_base_row[216usize]) * (node_128)))
                + ((current_base_row[221usize]) * (node_1224)))
                + (current_base_row[345usize]))
                + ((current_base_row[224usize])
                    * ((next_base_row[9usize]) + (node_522))))
                + ((current_base_row[228usize]) * (node_1225)))
                + ((current_base_row[227usize]) * (node_128)))
                + ((current_base_row[229usize])
                    * (((node_1298) * (node_1292))
                        + ((current_base_row[275usize]) * (node_120)))))
                + ((current_base_row[230usize]) * (node_128)))
                + ((current_base_row[219usize]) * (node_116)))
                + ((current_base_row[226usize]) * (node_116)))
                + ((current_base_row[248usize]) * (node_1224)))
                + ((current_base_row[233usize])
                    * ((current_base_row[30usize]) + (node_192))))
                + ((current_base_row[234usize]) * (node_1224)))
                + ((current_base_row[250usize]) * (node_1224)))
                + ((current_base_row[244usize]) * (node_128)))
                + ((current_base_row[254usize]) * (node_1224)))
                + ((current_base_row[237usize]) * (node_128)))
                + ((current_base_row[239usize]) * (node_128)))
                + ((current_base_row[240usize]) * (node_128)))
                + ((current_base_row[241usize]) * (node_120)))
                + ((current_base_row[238usize]) * (node_528)))
                + ((current_base_row[242usize]) * (node_1224)))
                + ((current_base_row[243usize]) * (node_1224)))
                + ((current_base_row[245usize]) * (node_1224)))
                + ((current_base_row[246usize]) * (node_1224)))
                + ((current_base_row[247usize]) * (node_1224)))
                + ((current_base_row[249usize]) * (node_128)))
                + ((current_base_row[251usize]) * (node_1224)))
                + ((current_base_row[277usize]) * (node_329)))
                + ((current_base_row[278usize]) * (node_329)))
                + ((current_base_row[279usize]) * (node_1227)))
                + ((current_base_row[281usize]) * (node_197)))
                + ((current_base_row[255usize]) * (node_116)))
                + ((current_base_row[258usize]) * (node_116)))
                + ((current_base_row[295usize]) * (node_1231)))
                + ((current_base_row[296usize])
                    * ((node_1226)
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((current_base_row[340usize])
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (node_1797)))))))
                + ((current_base_row[297usize])
                    * ((node_1226)
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((current_base_row[40usize])
                                * (current_base_row[39usize])))))) * (node_4096))
                + ((node_120) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((((((current_base_row[207usize])
                * (current_base_row[328usize]))
                + ((current_base_row[213usize]) * (node_529)))
                + ((current_base_row[209usize]) * (current_base_row[328usize])))
                + ((current_base_row[211usize]) * (current_base_row[261usize])))
                + ((current_base_row[232usize]) * (current_base_row[261usize])))
                + ((current_base_row[216usize]) * (node_1224)))
                + ((current_base_row[221usize]) * (node_785)))
                + ((current_base_row[214usize]) * (current_base_row[271usize])))
                + ((current_base_row[224usize]) * (node_785)))
                + ((current_base_row[228usize]) * (node_1226)))
                + ((current_base_row[227usize]) * (node_785)))
                + ((current_base_row[229usize]) * (node_785)))
                + ((current_base_row[230usize]) * (node_1224)))
                + ((current_base_row[219usize]) * (current_base_row[328usize])))
                + ((current_base_row[226usize]) * (current_base_row[328usize])))
                + ((current_base_row[248usize]) * (node_414)))
                + ((current_base_row[233usize])
                    * ((current_base_row[31usize]) + (node_196))))
                + ((current_base_row[234usize]) * (node_785)))
                + ((current_base_row[250usize])
                    * ((node_158) + (BFieldElement::from_raw_u64(42949672950u64)))))
                + ((current_base_row[244usize]) * (node_1224)))
                + ((current_base_row[254usize])
                    * ((node_158)
                        + (BFieldElement::from_raw_u64(18446744026464911371u64)))))
                + ((current_base_row[237usize]) * (node_1224)))
                + ((current_base_row[239usize]) * (node_1224)))
                + ((current_base_row[240usize]) * (node_1224)))
                + ((current_base_row[241usize]) * (node_124)))
                + ((current_base_row[238usize]) * (node_529)))
                + ((current_base_row[242usize]) * (node_189)))
                + ((current_base_row[243usize]) * (node_189)))
                + ((current_base_row[245usize]) * (node_189)))
                + ((current_base_row[246usize]) * (node_1225)))
                + ((current_base_row[247usize]) * (node_189)))
                + ((current_base_row[249usize]) * (node_1224)))
                + ((current_base_row[251usize]) * (node_1225)))
                + ((current_base_row[277usize]) * (node_330)))
                + ((current_base_row[278usize]) * (node_330)))
                + ((current_base_row[279usize]) * (node_1228)))
                + ((current_base_row[281usize]) * (node_201)))
                + ((current_base_row[255usize]) * (current_base_row[328usize])))
                + ((current_base_row[258usize]) * (current_base_row[328usize])))
                + ((current_base_row[295usize]) * (node_1232)))
                + ((current_base_row[296usize])
                    * ((node_1227)
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((current_base_row[40usize])
                                * (current_base_row[42usize]))
                                + ((current_base_row[39usize])
                                    * (current_base_row[43usize])))
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (node_1798))) + (node_1797))))))
                + ((current_base_row[297usize])
                    * ((node_1227)
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((current_base_row[41usize])
                                * (current_base_row[39usize])))))) * (node_4096))
                + ((node_124) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((((((current_base_row[207usize])
                * (current_base_row[252usize]))
                + ((current_base_row[213usize]) * (node_531)))
                + ((current_base_row[209usize]) * (current_base_row[252usize])))
                + ((current_base_row[211usize]) * (node_120)))
                + ((current_base_row[232usize]) * (node_120)))
                + ((current_base_row[216usize]) * (node_1225)))
                + ((current_base_row[221usize]) * (node_1226)))
                + (current_base_row[347usize]))
                + ((current_base_row[224usize]) * (node_1226)))
                + ((current_base_row[228usize]) * (node_1228)))
                + ((current_base_row[227usize]) * (node_1226)))
                + ((current_base_row[229usize]) * (node_1226)))
                + ((current_base_row[230usize]) * (node_189)))
                + ((current_base_row[219usize]) * (current_base_row[252usize])))
                + ((current_base_row[226usize]) * (current_base_row[252usize])))
                + ((current_base_row[248usize]) * (node_416)))
                + ((current_base_row[233usize]) * (node_124)))
                + ((current_base_row[234usize]) * (node_1226)))
                + ((current_base_row[250usize]) * (node_804)))
                + ((current_base_row[244usize]) * (node_1230)))
                + ((current_base_row[254usize])
                    * ((next_base_row[32usize]) + (node_525))))
                + ((current_base_row[237usize]) * (node_193)))
                + ((current_base_row[239usize]) * (node_193)))
                + ((current_base_row[240usize]) * (node_1226)))
                + ((current_base_row[241usize]) * (node_1224)))
                + ((current_base_row[238usize]) * (node_531)))
                + ((current_base_row[242usize]) * (node_197)))
                + ((current_base_row[243usize]) * (node_197)))
                + ((current_base_row[245usize]) * (node_197)))
                + ((current_base_row[246usize]) * (node_1227)))
                + ((current_base_row[247usize]) * (node_197)))
                + ((current_base_row[249usize]) * (node_1227)))
                + ((current_base_row[251usize]) * (node_1227)))
                + ((current_base_row[277usize]) * (node_332)))
                + ((current_base_row[278usize]) * (node_332)))
                + ((current_base_row[279usize]) * (node_1230)))
                + ((current_base_row[281usize]) * (node_209)))
                + ((current_base_row[255usize]) * (current_base_row[252usize])))
                + ((current_base_row[258usize]) * (current_base_row[252usize])))
                + ((current_base_row[295usize]) * (node_1234)))
                + ((current_base_row[296usize]) * (node_120)))
                + ((current_base_row[297usize]) * (node_120))) * (node_4096))
                + ((node_785) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((((((current_base_row[207usize])
                * (current_base_row[261usize]))
                + ((current_base_row[213usize]) * (node_532)))
                + ((current_base_row[209usize]) * (current_base_row[261usize])))
                + ((current_base_row[211usize]) * (node_124)))
                + ((current_base_row[232usize]) * (node_124)))
                + ((current_base_row[216usize]) * (node_1226)))
                + ((current_base_row[221usize]) * (node_1227)))
                + ((current_base_row[214usize]) * (current_base_row[284usize])))
                + ((current_base_row[224usize]) * (node_1227)))
                + ((current_base_row[228usize]) * (node_1229)))
                + ((current_base_row[227usize]) * (node_1227)))
                + ((current_base_row[229usize]) * (node_1227)))
                + ((current_base_row[230usize]) * (node_193)))
                + ((current_base_row[219usize]) * (current_base_row[261usize])))
                + ((current_base_row[226usize]) * (current_base_row[261usize])))
                + ((current_base_row[248usize]) * (node_417)))
                + ((current_base_row[233usize]) * (node_128)))
                + ((current_base_row[234usize]) * (node_1227)))
                + ((current_base_row[250usize])
                    * ((next_base_row[23usize]) + (node_224))))
                + ((current_base_row[244usize]) * (node_1231)))
                + ((current_base_row[254usize])
                    * ((next_base_row[33usize]) + (node_184))))
                + ((current_base_row[237usize]) * (node_197)))
                + ((current_base_row[239usize]) * (node_197)))
                + ((current_base_row[240usize]) * (node_1227)))
                + ((current_base_row[241usize]) * (node_189)))
                + ((current_base_row[238usize]) * (node_532)))
                + ((current_base_row[242usize]) * (node_201)))
                + ((current_base_row[243usize]) * (node_201)))
                + ((current_base_row[245usize]) * (node_201)))
                + ((current_base_row[246usize]) * (node_1228)))
                + ((current_base_row[247usize]) * (node_201)))
                + ((current_base_row[249usize]) * (node_1228)))
                + ((current_base_row[251usize]) * (node_1228)))
                + ((current_base_row[277usize]) * (node_333)))
                + ((current_base_row[278usize]) * (node_333)))
                + ((current_base_row[279usize]) * (node_1231)))
                + ((current_base_row[281usize]) * (node_213)))
                + ((current_base_row[255usize]) * (current_base_row[261usize])))
                + ((current_base_row[258usize]) * (current_base_row[261usize])))
                + ((current_base_row[295usize]) * (node_1235)))
                + ((current_base_row[296usize]) * (node_124)))
                + ((current_base_row[297usize]) * (node_124))) * (node_4096))
                + ((node_1225) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((((((current_base_row[207usize])
                * (node_154)) + ((current_base_row[213usize]) * (node_533)))
                + ((current_base_row[209usize]) * (node_154)))
                + ((current_base_row[211usize]) * (node_128)))
                + ((current_base_row[232usize]) * (node_128)))
                + ((current_base_row[216usize]) * (node_1227)))
                + ((current_base_row[221usize]) * (node_1228)))
                + (current_base_row[348usize]))
                + ((current_base_row[224usize]) * (node_1228)))
                + ((current_base_row[228usize]) * (node_1230)))
                + ((current_base_row[227usize]) * (node_1228)))
                + ((current_base_row[229usize]) * (node_1228)))
                + ((current_base_row[230usize]) * (node_197)))
                + ((current_base_row[219usize]) * (node_154)))
                + ((current_base_row[226usize]) * (node_154)))
                + ((current_base_row[248usize]) * (node_418)))
                + ((current_base_row[233usize]) * (node_1224)))
                + ((current_base_row[234usize]) * (node_1228)))
                + ((current_base_row[250usize])
                    * ((next_base_row[24usize]) + (node_228))))
                + ((current_base_row[244usize]) * (node_1232)))
                + ((current_base_row[254usize])
                    * ((next_base_row[34usize]) + (node_188))))
                + ((current_base_row[237usize]) * (node_201)))
                + ((current_base_row[239usize]) * (node_201)))
                + ((current_base_row[240usize]) * (node_1228)))
                + ((current_base_row[241usize]) * (node_193)))
                + ((current_base_row[238usize]) * (node_533)))
                + ((current_base_row[242usize]) * (node_205)))
                + ((current_base_row[243usize]) * (node_205)))
                + ((current_base_row[245usize]) * (node_205)))
                + ((current_base_row[246usize]) * (node_1229)))
                + ((current_base_row[247usize]) * (node_205)))
                + ((current_base_row[249usize]) * (node_1229)))
                + ((current_base_row[251usize]) * (node_1229)))
                + ((current_base_row[277usize]) * (node_334)))
                + ((current_base_row[278usize]) * (node_334)))
                + ((current_base_row[279usize]) * (node_1232)))
                + ((current_base_row[281usize]) * (node_217)))
                + ((current_base_row[255usize]) * (node_154)))
                + ((current_base_row[258usize]) * (node_154)))
                + ((current_base_row[295usize]) * (node_1236)))
                + ((current_base_row[296usize]) * (node_128)))
                + ((current_base_row[297usize]) * (node_128))) * (node_4096))
                + ((node_1226) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((((((current_base_row[207usize])
                * (node_437)) + ((current_base_row[213usize]) * (node_534)))
                + ((current_base_row[209usize]) * (node_730)))
                + ((current_base_row[211usize]) * (node_116)))
                + ((current_base_row[232usize]) * (node_116)))
                + ((current_base_row[216usize]) * (node_1228)))
                + ((current_base_row[221usize]) * (node_1229)))
                + ((current_base_row[214usize]) * (node_120)))
                + ((current_base_row[224usize]) * (node_1229)))
                + ((current_base_row[228usize]) * (node_1231)))
                + ((current_base_row[227usize]) * (node_1229)))
                + ((current_base_row[229usize]) * (node_1229)))
                + ((current_base_row[230usize]) * (node_201)))
                + ((current_base_row[219usize]) * (node_730)))
                + ((current_base_row[226usize]) * (node_437)))
                + ((current_base_row[248usize]) * (node_419)))
                + ((current_base_row[233usize]) * (node_397)))
                + ((current_base_row[234usize]) * (node_1229)))
                + ((current_base_row[250usize])
                    * ((next_base_row[25usize]) + (node_232))))
                + ((current_base_row[244usize]) * (node_1233)))
                + ((current_base_row[254usize])
                    * ((next_base_row[35usize]) + (node_192))))
                + ((current_base_row[237usize]) * (node_205)))
                + ((current_base_row[239usize]) * (node_205)))
                + ((current_base_row[240usize]) * (node_1229)))
                + ((current_base_row[241usize]) * (node_197)))
                + ((current_base_row[238usize]) * (node_534)))
                + ((current_base_row[242usize]) * (node_209)))
                + ((current_base_row[243usize]) * (node_209)))
                + ((current_base_row[245usize]) * (node_209)))
                + ((current_base_row[246usize]) * (node_1230)))
                + ((current_base_row[247usize]) * (node_209)))
                + ((current_base_row[249usize]) * (node_1230)))
                + ((current_base_row[251usize]) * (node_1230)))
                + ((current_base_row[277usize]) * (node_335)))
                + ((current_base_row[278usize]) * (node_335)))
                + ((current_base_row[279usize]) * (node_1233)))
                + ((current_base_row[281usize]) * (node_221)))
                + ((current_base_row[255usize]) * (node_730)))
                + ((current_base_row[258usize]) * (node_437)))
                + ((current_base_row[295usize]) * (node_1237)))
                + ((current_base_row[296usize]) * (node_1224)))
                + ((current_base_row[297usize]) * (node_1224))) * (node_4096))
                + ((node_1227) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((current_base_row[207usize])
                * (node_461)) + ((current_base_row[213usize]) * (node_540)))
                + ((current_base_row[209usize]) * (node_754)))
                + ((current_base_row[211usize]) * (node_531)))
                + ((current_base_row[216usize]) * (node_1234)))
                + ((current_base_row[221usize]) * (node_1235)))
                + ((current_base_row[214usize]) * (node_197)))
                + ((current_base_row[224usize]) * (node_1235)))
                + ((current_base_row[228usize]) * (node_1237)))
                + ((current_base_row[227usize]) * (node_1235)))
                + ((current_base_row[229usize]) * (node_1235)))
                + ((current_base_row[230usize]) * (node_225)))
                + ((current_base_row[219usize]) * (node_750)))
                + ((current_base_row[226usize]) * (node_457)))
                + ((current_base_row[233usize]) * (node_413)))
                + ((current_base_row[234usize]) * (node_1235)))
                + ((current_base_row[244usize]) * (node_1239)))
                + ((current_base_row[237usize]) * (node_229)))
                + ((current_base_row[239usize]) * (node_229)))
                + ((current_base_row[240usize]) * (node_1235)))
                + ((current_base_row[241usize]) * (node_221)))
                + ((current_base_row[238usize]) * (node_540)))
                + ((current_base_row[242usize]) * (node_233)))
                + ((current_base_row[243usize]) * (node_233)))
                + ((current_base_row[245usize]) * (node_233)))
                + ((current_base_row[246usize]) * (node_1236)))
                + ((current_base_row[247usize]) * (node_233)))
                + ((current_base_row[249usize]) * (node_1236)))
                + ((current_base_row[251usize]) * (node_1236)))
                + ((current_base_row[277usize]) * (node_120)))
                + ((current_base_row[278usize]) * (node_120)))
                + ((current_base_row[279usize]) * (node_1239)))
                + ((current_base_row[281usize]) * (node_159)))
                + ((current_base_row[255usize]) * (node_754)))
                + ((current_base_row[258usize]) * (node_461)))
                + ((current_base_row[295usize]) * (node_124)))
                + ((current_base_row[296usize]) * (node_1232)))
                + ((current_base_row[297usize]) * (node_1232))) * (node_4096))
                + ((node_1233) * (next_base_row[8usize])),
            (((((((current_base_row[207usize]) * (current_base_row[335usize]))
                + ((current_base_row[209usize]) * (current_base_row[335usize])))
                + ((current_base_row[219usize]) * (current_base_row[334usize])))
                + ((current_base_row[226usize]) * (current_base_row[334usize])))
                + ((current_base_row[255usize]) * (current_base_row[334usize])))
                + ((current_base_row[258usize]) * (current_base_row[334usize])))
                * (node_4096),
            (((((((current_base_row[207usize]) * (current_base_row[336usize]))
                + ((current_base_row[209usize]) * (current_base_row[336usize])))
                + ((current_base_row[219usize]) * (current_base_row[335usize])))
                + ((current_base_row[226usize]) * (current_base_row[335usize])))
                + ((current_base_row[255usize]) * (current_base_row[335usize])))
                + ((current_base_row[258usize]) * (current_base_row[335usize])))
                * (node_4096),
            (((((((current_base_row[207usize]) * (current_base_row[337usize]))
                + ((current_base_row[209usize]) * (current_base_row[337usize])))
                + ((current_base_row[219usize]) * (current_base_row[336usize])))
                + ((current_base_row[226usize]) * (current_base_row[336usize])))
                + ((current_base_row[255usize]) * (current_base_row[336usize])))
                + ((current_base_row[258usize]) * (current_base_row[336usize])))
                * (node_4096),
            (((((((current_base_row[207usize]) * (current_base_row[338usize]))
                + ((current_base_row[209usize]) * (current_base_row[338usize])))
                + ((current_base_row[219usize]) * (current_base_row[337usize])))
                + ((current_base_row[226usize]) * (current_base_row[337usize])))
                + ((current_base_row[255usize]) * (current_base_row[337usize])))
                + ((current_base_row[258usize]) * (current_base_row[337usize])))
                * (node_4096),
            (((((((current_base_row[207usize]) * (current_base_row[339usize]))
                + ((current_base_row[209usize]) * (current_base_row[339usize])))
                + ((current_base_row[219usize]) * (current_base_row[338usize])))
                + ((current_base_row[226usize]) * (current_base_row[338usize])))
                + ((current_base_row[255usize]) * (current_base_row[338usize])))
                + ((current_base_row[258usize]) * (current_base_row[338usize])))
                * (node_4096),
            (node_4524) * (node_4523),
            ((node_4524)
                * ((next_base_row[49usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[49usize])))) * (next_base_row[47usize]),
            ((current_base_row[47usize])
                * ((current_base_row[47usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                * (node_4530),
            (((current_base_row[51usize])
                + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                * (current_base_row[51usize])) * (node_4593),
            (current_base_row[54usize]) * (node_4601),
            (node_4598) * (node_4601),
            ((node_4601)
                * ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (next_base_row[51usize])))
                * ((next_base_row[53usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[53usize]))),
            (node_4601)
                * ((next_base_row[55usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[55usize]))),
            (node_4601)
                * ((next_base_row[56usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[56usize]))),
            (node_4705) * (node_4704),
            (node_4709)
                * ((next_base_row[60usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[60usize]))),
            (node_4709)
                * ((next_base_row[61usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[61usize]))),
            (((node_4705)
                * ((node_4717) + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                * ((current_base_row[58usize])
                    + (BFieldElement::from_raw_u64(18446743927680663586u64))))
                * (node_4708),
            ((current_base_row[350usize])
                * ((current_base_row[64usize])
                    + (BFieldElement::from_raw_u64(18446744052234715141u64))))
                * (next_base_row[64usize]),
            (((next_base_row[62usize]) * (node_5595)) * (node_5597))
                * (((next_base_row[64usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[64usize])))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (current_base_row[355usize]) * (node_5625),
            (node_5627)
                * ((next_base_row[63usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[63usize]))),
            (node_5627)
                * ((next_base_row[62usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[62usize]))),
            ((current_base_row[351usize]) * (node_5621)) * (next_base_row[62usize]),
            (current_base_row[356usize]) * (next_base_row[62usize]),
            ((node_5641) * (node_5613)) * (next_base_row[62usize]),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(263719581847590u64))
                    * (node_4832))
                    + ((BFieldElement::from_raw_u64(76643691379275u64)) * (node_4843)))
                    + ((BFieldElement::from_raw_u64(115096533571410u64)) * (node_4854)))
                    + ((BFieldElement::from_raw_u64(256362302871255u64)) * (node_4865)))
                    + ((BFieldElement::from_raw_u64(51629801853195u64))
                        * (current_base_row[301usize])))
                    + ((BFieldElement::from_raw_u64(175668457332795u64))
                        * (current_base_row[302usize])))
                    + ((BFieldElement::from_raw_u64(177601192615545u64))
                        * (current_base_row[303usize])))
                    + ((BFieldElement::from_raw_u64(118201794925695u64))
                        * (current_base_row[304usize])))
                    + ((BFieldElement::from_raw_u64(244602682417545u64))
                        * (current_base_row[305usize])))
                    + ((BFieldElement::from_raw_u64(51685636428030u64))
                        * (current_base_row[306usize])))
                    + ((BFieldElement::from_raw_u64(231348413345175u64))
                        * (current_base_row[307usize])))
                    + ((BFieldElement::from_raw_u64(185731565704980u64))
                        * (current_base_row[308usize])))
                    + ((BFieldElement::from_raw_u64(32014686216930u64))
                        * (current_base_row[309usize])))
                    + ((BFieldElement::from_raw_u64(145268678818785u64))
                        * (current_base_row[310usize])))
                    + ((BFieldElement::from_raw_u64(123480309731250u64))
                        * (current_base_row[311usize])))
                    + ((BFieldElement::from_raw_u64(4758823762860u64))
                        * (current_base_row[312usize]))) + (current_base_row[113usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (node_5492))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(4758823762860u64))
                    * (node_4832))
                    + ((BFieldElement::from_raw_u64(263719581847590u64)) * (node_4843)))
                    + ((BFieldElement::from_raw_u64(76643691379275u64)) * (node_4854)))
                    + ((BFieldElement::from_raw_u64(115096533571410u64)) * (node_4865)))
                    + ((BFieldElement::from_raw_u64(256362302871255u64))
                        * (current_base_row[301usize])))
                    + ((BFieldElement::from_raw_u64(51629801853195u64))
                        * (current_base_row[302usize])))
                    + ((BFieldElement::from_raw_u64(175668457332795u64))
                        * (current_base_row[303usize])))
                    + ((BFieldElement::from_raw_u64(177601192615545u64))
                        * (current_base_row[304usize])))
                    + ((BFieldElement::from_raw_u64(118201794925695u64))
                        * (current_base_row[305usize])))
                    + ((BFieldElement::from_raw_u64(244602682417545u64))
                        * (current_base_row[306usize])))
                    + ((BFieldElement::from_raw_u64(51685636428030u64))
                        * (current_base_row[307usize])))
                    + ((BFieldElement::from_raw_u64(231348413345175u64))
                        * (current_base_row[308usize])))
                    + ((BFieldElement::from_raw_u64(185731565704980u64))
                        * (current_base_row[309usize])))
                    + ((BFieldElement::from_raw_u64(32014686216930u64))
                        * (current_base_row[310usize])))
                    + ((BFieldElement::from_raw_u64(145268678818785u64))
                        * (current_base_row[311usize])))
                    + ((BFieldElement::from_raw_u64(123480309731250u64))
                        * (current_base_row[312usize]))) + (current_base_row[114usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (node_5503))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(123480309731250u64))
                    * (node_4832))
                    + ((BFieldElement::from_raw_u64(4758823762860u64)) * (node_4843)))
                    + ((BFieldElement::from_raw_u64(263719581847590u64)) * (node_4854)))
                    + ((BFieldElement::from_raw_u64(76643691379275u64)) * (node_4865)))
                    + ((BFieldElement::from_raw_u64(115096533571410u64))
                        * (current_base_row[301usize])))
                    + ((BFieldElement::from_raw_u64(256362302871255u64))
                        * (current_base_row[302usize])))
                    + ((BFieldElement::from_raw_u64(51629801853195u64))
                        * (current_base_row[303usize])))
                    + ((BFieldElement::from_raw_u64(175668457332795u64))
                        * (current_base_row[304usize])))
                    + ((BFieldElement::from_raw_u64(177601192615545u64))
                        * (current_base_row[305usize])))
                    + ((BFieldElement::from_raw_u64(118201794925695u64))
                        * (current_base_row[306usize])))
                    + ((BFieldElement::from_raw_u64(244602682417545u64))
                        * (current_base_row[307usize])))
                    + ((BFieldElement::from_raw_u64(51685636428030u64))
                        * (current_base_row[308usize])))
                    + ((BFieldElement::from_raw_u64(231348413345175u64))
                        * (current_base_row[309usize])))
                    + ((BFieldElement::from_raw_u64(185731565704980u64))
                        * (current_base_row[310usize])))
                    + ((BFieldElement::from_raw_u64(32014686216930u64))
                        * (current_base_row[311usize])))
                    + ((BFieldElement::from_raw_u64(145268678818785u64))
                        * (current_base_row[312usize]))) + (current_base_row[115usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (node_5514))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(145268678818785u64))
                    * (node_4832))
                    + ((BFieldElement::from_raw_u64(123480309731250u64)) * (node_4843)))
                    + ((BFieldElement::from_raw_u64(4758823762860u64)) * (node_4854)))
                    + ((BFieldElement::from_raw_u64(263719581847590u64)) * (node_4865)))
                    + ((BFieldElement::from_raw_u64(76643691379275u64))
                        * (current_base_row[301usize])))
                    + ((BFieldElement::from_raw_u64(115096533571410u64))
                        * (current_base_row[302usize])))
                    + ((BFieldElement::from_raw_u64(256362302871255u64))
                        * (current_base_row[303usize])))
                    + ((BFieldElement::from_raw_u64(51629801853195u64))
                        * (current_base_row[304usize])))
                    + ((BFieldElement::from_raw_u64(175668457332795u64))
                        * (current_base_row[305usize])))
                    + ((BFieldElement::from_raw_u64(177601192615545u64))
                        * (current_base_row[306usize])))
                    + ((BFieldElement::from_raw_u64(118201794925695u64))
                        * (current_base_row[307usize])))
                    + ((BFieldElement::from_raw_u64(244602682417545u64))
                        * (current_base_row[308usize])))
                    + ((BFieldElement::from_raw_u64(51685636428030u64))
                        * (current_base_row[309usize])))
                    + ((BFieldElement::from_raw_u64(231348413345175u64))
                        * (current_base_row[310usize])))
                    + ((BFieldElement::from_raw_u64(185731565704980u64))
                        * (current_base_row[311usize])))
                    + ((BFieldElement::from_raw_u64(32014686216930u64))
                        * (current_base_row[312usize]))) + (current_base_row[116usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (node_5525))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(32014686216930u64))
                    * (node_4832))
                    + ((BFieldElement::from_raw_u64(145268678818785u64)) * (node_4843)))
                    + ((BFieldElement::from_raw_u64(123480309731250u64)) * (node_4854)))
                    + ((BFieldElement::from_raw_u64(4758823762860u64)) * (node_4865)))
                    + ((BFieldElement::from_raw_u64(263719581847590u64))
                        * (current_base_row[301usize])))
                    + ((BFieldElement::from_raw_u64(76643691379275u64))
                        * (current_base_row[302usize])))
                    + ((BFieldElement::from_raw_u64(115096533571410u64))
                        * (current_base_row[303usize])))
                    + ((BFieldElement::from_raw_u64(256362302871255u64))
                        * (current_base_row[304usize])))
                    + ((BFieldElement::from_raw_u64(51629801853195u64))
                        * (current_base_row[305usize])))
                    + ((BFieldElement::from_raw_u64(175668457332795u64))
                        * (current_base_row[306usize])))
                    + ((BFieldElement::from_raw_u64(177601192615545u64))
                        * (current_base_row[307usize])))
                    + ((BFieldElement::from_raw_u64(118201794925695u64))
                        * (current_base_row[308usize])))
                    + ((BFieldElement::from_raw_u64(244602682417545u64))
                        * (current_base_row[309usize])))
                    + ((BFieldElement::from_raw_u64(51685636428030u64))
                        * (current_base_row[310usize])))
                    + ((BFieldElement::from_raw_u64(231348413345175u64))
                        * (current_base_row[311usize])))
                    + ((BFieldElement::from_raw_u64(185731565704980u64))
                        * (current_base_row[312usize]))) + (current_base_row[117usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[97usize]))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(185731565704980u64))
                    * (node_4832))
                    + ((BFieldElement::from_raw_u64(32014686216930u64)) * (node_4843)))
                    + ((BFieldElement::from_raw_u64(145268678818785u64)) * (node_4854)))
                    + ((BFieldElement::from_raw_u64(123480309731250u64)) * (node_4865)))
                    + ((BFieldElement::from_raw_u64(4758823762860u64))
                        * (current_base_row[301usize])))
                    + ((BFieldElement::from_raw_u64(263719581847590u64))
                        * (current_base_row[302usize])))
                    + ((BFieldElement::from_raw_u64(76643691379275u64))
                        * (current_base_row[303usize])))
                    + ((BFieldElement::from_raw_u64(115096533571410u64))
                        * (current_base_row[304usize])))
                    + ((BFieldElement::from_raw_u64(256362302871255u64))
                        * (current_base_row[305usize])))
                    + ((BFieldElement::from_raw_u64(51629801853195u64))
                        * (current_base_row[306usize])))
                    + ((BFieldElement::from_raw_u64(175668457332795u64))
                        * (current_base_row[307usize])))
                    + ((BFieldElement::from_raw_u64(177601192615545u64))
                        * (current_base_row[308usize])))
                    + ((BFieldElement::from_raw_u64(118201794925695u64))
                        * (current_base_row[309usize])))
                    + ((BFieldElement::from_raw_u64(244602682417545u64))
                        * (current_base_row[310usize])))
                    + ((BFieldElement::from_raw_u64(51685636428030u64))
                        * (current_base_row[311usize])))
                    + ((BFieldElement::from_raw_u64(231348413345175u64))
                        * (current_base_row[312usize]))) + (current_base_row[118usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[98usize]))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(231348413345175u64))
                    * (node_4832))
                    + ((BFieldElement::from_raw_u64(185731565704980u64)) * (node_4843)))
                    + ((BFieldElement::from_raw_u64(32014686216930u64)) * (node_4854)))
                    + ((BFieldElement::from_raw_u64(145268678818785u64)) * (node_4865)))
                    + ((BFieldElement::from_raw_u64(123480309731250u64))
                        * (current_base_row[301usize])))
                    + ((BFieldElement::from_raw_u64(4758823762860u64))
                        * (current_base_row[302usize])))
                    + ((BFieldElement::from_raw_u64(263719581847590u64))
                        * (current_base_row[303usize])))
                    + ((BFieldElement::from_raw_u64(76643691379275u64))
                        * (current_base_row[304usize])))
                    + ((BFieldElement::from_raw_u64(115096533571410u64))
                        * (current_base_row[305usize])))
                    + ((BFieldElement::from_raw_u64(256362302871255u64))
                        * (current_base_row[306usize])))
                    + ((BFieldElement::from_raw_u64(51629801853195u64))
                        * (current_base_row[307usize])))
                    + ((BFieldElement::from_raw_u64(175668457332795u64))
                        * (current_base_row[308usize])))
                    + ((BFieldElement::from_raw_u64(177601192615545u64))
                        * (current_base_row[309usize])))
                    + ((BFieldElement::from_raw_u64(118201794925695u64))
                        * (current_base_row[310usize])))
                    + ((BFieldElement::from_raw_u64(244602682417545u64))
                        * (current_base_row[311usize])))
                    + ((BFieldElement::from_raw_u64(51685636428030u64))
                        * (current_base_row[312usize]))) + (current_base_row[119usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[99usize]))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(51685636428030u64))
                    * (node_4832))
                    + ((BFieldElement::from_raw_u64(231348413345175u64)) * (node_4843)))
                    + ((BFieldElement::from_raw_u64(185731565704980u64)) * (node_4854)))
                    + ((BFieldElement::from_raw_u64(32014686216930u64)) * (node_4865)))
                    + ((BFieldElement::from_raw_u64(145268678818785u64))
                        * (current_base_row[301usize])))
                    + ((BFieldElement::from_raw_u64(123480309731250u64))
                        * (current_base_row[302usize])))
                    + ((BFieldElement::from_raw_u64(4758823762860u64))
                        * (current_base_row[303usize])))
                    + ((BFieldElement::from_raw_u64(263719581847590u64))
                        * (current_base_row[304usize])))
                    + ((BFieldElement::from_raw_u64(76643691379275u64))
                        * (current_base_row[305usize])))
                    + ((BFieldElement::from_raw_u64(115096533571410u64))
                        * (current_base_row[306usize])))
                    + ((BFieldElement::from_raw_u64(256362302871255u64))
                        * (current_base_row[307usize])))
                    + ((BFieldElement::from_raw_u64(51629801853195u64))
                        * (current_base_row[308usize])))
                    + ((BFieldElement::from_raw_u64(175668457332795u64))
                        * (current_base_row[309usize])))
                    + ((BFieldElement::from_raw_u64(177601192615545u64))
                        * (current_base_row[310usize])))
                    + ((BFieldElement::from_raw_u64(118201794925695u64))
                        * (current_base_row[311usize])))
                    + ((BFieldElement::from_raw_u64(244602682417545u64))
                        * (current_base_row[312usize]))) + (current_base_row[120usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[100usize]))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(244602682417545u64))
                    * (node_4832))
                    + ((BFieldElement::from_raw_u64(51685636428030u64)) * (node_4843)))
                    + ((BFieldElement::from_raw_u64(231348413345175u64)) * (node_4854)))
                    + ((BFieldElement::from_raw_u64(185731565704980u64)) * (node_4865)))
                    + ((BFieldElement::from_raw_u64(32014686216930u64))
                        * (current_base_row[301usize])))
                    + ((BFieldElement::from_raw_u64(145268678818785u64))
                        * (current_base_row[302usize])))
                    + ((BFieldElement::from_raw_u64(123480309731250u64))
                        * (current_base_row[303usize])))
                    + ((BFieldElement::from_raw_u64(4758823762860u64))
                        * (current_base_row[304usize])))
                    + ((BFieldElement::from_raw_u64(263719581847590u64))
                        * (current_base_row[305usize])))
                    + ((BFieldElement::from_raw_u64(76643691379275u64))
                        * (current_base_row[306usize])))
                    + ((BFieldElement::from_raw_u64(115096533571410u64))
                        * (current_base_row[307usize])))
                    + ((BFieldElement::from_raw_u64(256362302871255u64))
                        * (current_base_row[308usize])))
                    + ((BFieldElement::from_raw_u64(51629801853195u64))
                        * (current_base_row[309usize])))
                    + ((BFieldElement::from_raw_u64(175668457332795u64))
                        * (current_base_row[310usize])))
                    + ((BFieldElement::from_raw_u64(177601192615545u64))
                        * (current_base_row[311usize])))
                    + ((BFieldElement::from_raw_u64(118201794925695u64))
                        * (current_base_row[312usize]))) + (current_base_row[121usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[101usize]))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(118201794925695u64))
                    * (node_4832))
                    + ((BFieldElement::from_raw_u64(244602682417545u64)) * (node_4843)))
                    + ((BFieldElement::from_raw_u64(51685636428030u64)) * (node_4854)))
                    + ((BFieldElement::from_raw_u64(231348413345175u64)) * (node_4865)))
                    + ((BFieldElement::from_raw_u64(185731565704980u64))
                        * (current_base_row[301usize])))
                    + ((BFieldElement::from_raw_u64(32014686216930u64))
                        * (current_base_row[302usize])))
                    + ((BFieldElement::from_raw_u64(145268678818785u64))
                        * (current_base_row[303usize])))
                    + ((BFieldElement::from_raw_u64(123480309731250u64))
                        * (current_base_row[304usize])))
                    + ((BFieldElement::from_raw_u64(4758823762860u64))
                        * (current_base_row[305usize])))
                    + ((BFieldElement::from_raw_u64(263719581847590u64))
                        * (current_base_row[306usize])))
                    + ((BFieldElement::from_raw_u64(76643691379275u64))
                        * (current_base_row[307usize])))
                    + ((BFieldElement::from_raw_u64(115096533571410u64))
                        * (current_base_row[308usize])))
                    + ((BFieldElement::from_raw_u64(256362302871255u64))
                        * (current_base_row[309usize])))
                    + ((BFieldElement::from_raw_u64(51629801853195u64))
                        * (current_base_row[310usize])))
                    + ((BFieldElement::from_raw_u64(175668457332795u64))
                        * (current_base_row[311usize])))
                    + ((BFieldElement::from_raw_u64(177601192615545u64))
                        * (current_base_row[312usize]))) + (current_base_row[122usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[102usize]))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(177601192615545u64))
                    * (node_4832))
                    + ((BFieldElement::from_raw_u64(118201794925695u64)) * (node_4843)))
                    + ((BFieldElement::from_raw_u64(244602682417545u64)) * (node_4854)))
                    + ((BFieldElement::from_raw_u64(51685636428030u64)) * (node_4865)))
                    + ((BFieldElement::from_raw_u64(231348413345175u64))
                        * (current_base_row[301usize])))
                    + ((BFieldElement::from_raw_u64(185731565704980u64))
                        * (current_base_row[302usize])))
                    + ((BFieldElement::from_raw_u64(32014686216930u64))
                        * (current_base_row[303usize])))
                    + ((BFieldElement::from_raw_u64(145268678818785u64))
                        * (current_base_row[304usize])))
                    + ((BFieldElement::from_raw_u64(123480309731250u64))
                        * (current_base_row[305usize])))
                    + ((BFieldElement::from_raw_u64(4758823762860u64))
                        * (current_base_row[306usize])))
                    + ((BFieldElement::from_raw_u64(263719581847590u64))
                        * (current_base_row[307usize])))
                    + ((BFieldElement::from_raw_u64(76643691379275u64))
                        * (current_base_row[308usize])))
                    + ((BFieldElement::from_raw_u64(115096533571410u64))
                        * (current_base_row[309usize])))
                    + ((BFieldElement::from_raw_u64(256362302871255u64))
                        * (current_base_row[310usize])))
                    + ((BFieldElement::from_raw_u64(51629801853195u64))
                        * (current_base_row[311usize])))
                    + ((BFieldElement::from_raw_u64(175668457332795u64))
                        * (current_base_row[312usize]))) + (current_base_row[123usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[103usize]))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(175668457332795u64))
                    * (node_4832))
                    + ((BFieldElement::from_raw_u64(177601192615545u64)) * (node_4843)))
                    + ((BFieldElement::from_raw_u64(118201794925695u64)) * (node_4854)))
                    + ((BFieldElement::from_raw_u64(244602682417545u64)) * (node_4865)))
                    + ((BFieldElement::from_raw_u64(51685636428030u64))
                        * (current_base_row[301usize])))
                    + ((BFieldElement::from_raw_u64(231348413345175u64))
                        * (current_base_row[302usize])))
                    + ((BFieldElement::from_raw_u64(185731565704980u64))
                        * (current_base_row[303usize])))
                    + ((BFieldElement::from_raw_u64(32014686216930u64))
                        * (current_base_row[304usize])))
                    + ((BFieldElement::from_raw_u64(145268678818785u64))
                        * (current_base_row[305usize])))
                    + ((BFieldElement::from_raw_u64(123480309731250u64))
                        * (current_base_row[306usize])))
                    + ((BFieldElement::from_raw_u64(4758823762860u64))
                        * (current_base_row[307usize])))
                    + ((BFieldElement::from_raw_u64(263719581847590u64))
                        * (current_base_row[308usize])))
                    + ((BFieldElement::from_raw_u64(76643691379275u64))
                        * (current_base_row[309usize])))
                    + ((BFieldElement::from_raw_u64(115096533571410u64))
                        * (current_base_row[310usize])))
                    + ((BFieldElement::from_raw_u64(256362302871255u64))
                        * (current_base_row[311usize])))
                    + ((BFieldElement::from_raw_u64(51629801853195u64))
                        * (current_base_row[312usize]))) + (current_base_row[124usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[104usize]))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(51629801853195u64))
                    * (node_4832))
                    + ((BFieldElement::from_raw_u64(175668457332795u64)) * (node_4843)))
                    + ((BFieldElement::from_raw_u64(177601192615545u64)) * (node_4854)))
                    + ((BFieldElement::from_raw_u64(118201794925695u64)) * (node_4865)))
                    + ((BFieldElement::from_raw_u64(244602682417545u64))
                        * (current_base_row[301usize])))
                    + ((BFieldElement::from_raw_u64(51685636428030u64))
                        * (current_base_row[302usize])))
                    + ((BFieldElement::from_raw_u64(231348413345175u64))
                        * (current_base_row[303usize])))
                    + ((BFieldElement::from_raw_u64(185731565704980u64))
                        * (current_base_row[304usize])))
                    + ((BFieldElement::from_raw_u64(32014686216930u64))
                        * (current_base_row[305usize])))
                    + ((BFieldElement::from_raw_u64(145268678818785u64))
                        * (current_base_row[306usize])))
                    + ((BFieldElement::from_raw_u64(123480309731250u64))
                        * (current_base_row[307usize])))
                    + ((BFieldElement::from_raw_u64(4758823762860u64))
                        * (current_base_row[308usize])))
                    + ((BFieldElement::from_raw_u64(263719581847590u64))
                        * (current_base_row[309usize])))
                    + ((BFieldElement::from_raw_u64(76643691379275u64))
                        * (current_base_row[310usize])))
                    + ((BFieldElement::from_raw_u64(115096533571410u64))
                        * (current_base_row[311usize])))
                    + ((BFieldElement::from_raw_u64(256362302871255u64))
                        * (current_base_row[312usize]))) + (current_base_row[125usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[105usize]))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(256362302871255u64))
                    * (node_4832))
                    + ((BFieldElement::from_raw_u64(51629801853195u64)) * (node_4843)))
                    + ((BFieldElement::from_raw_u64(175668457332795u64)) * (node_4854)))
                    + ((BFieldElement::from_raw_u64(177601192615545u64)) * (node_4865)))
                    + ((BFieldElement::from_raw_u64(118201794925695u64))
                        * (current_base_row[301usize])))
                    + ((BFieldElement::from_raw_u64(244602682417545u64))
                        * (current_base_row[302usize])))
                    + ((BFieldElement::from_raw_u64(51685636428030u64))
                        * (current_base_row[303usize])))
                    + ((BFieldElement::from_raw_u64(231348413345175u64))
                        * (current_base_row[304usize])))
                    + ((BFieldElement::from_raw_u64(185731565704980u64))
                        * (current_base_row[305usize])))
                    + ((BFieldElement::from_raw_u64(32014686216930u64))
                        * (current_base_row[306usize])))
                    + ((BFieldElement::from_raw_u64(145268678818785u64))
                        * (current_base_row[307usize])))
                    + ((BFieldElement::from_raw_u64(123480309731250u64))
                        * (current_base_row[308usize])))
                    + ((BFieldElement::from_raw_u64(4758823762860u64))
                        * (current_base_row[309usize])))
                    + ((BFieldElement::from_raw_u64(263719581847590u64))
                        * (current_base_row[310usize])))
                    + ((BFieldElement::from_raw_u64(76643691379275u64))
                        * (current_base_row[311usize])))
                    + ((BFieldElement::from_raw_u64(115096533571410u64))
                        * (current_base_row[312usize]))) + (current_base_row[126usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[106usize]))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(115096533571410u64))
                    * (node_4832))
                    + ((BFieldElement::from_raw_u64(256362302871255u64)) * (node_4843)))
                    + ((BFieldElement::from_raw_u64(51629801853195u64)) * (node_4854)))
                    + ((BFieldElement::from_raw_u64(175668457332795u64)) * (node_4865)))
                    + ((BFieldElement::from_raw_u64(177601192615545u64))
                        * (current_base_row[301usize])))
                    + ((BFieldElement::from_raw_u64(118201794925695u64))
                        * (current_base_row[302usize])))
                    + ((BFieldElement::from_raw_u64(244602682417545u64))
                        * (current_base_row[303usize])))
                    + ((BFieldElement::from_raw_u64(51685636428030u64))
                        * (current_base_row[304usize])))
                    + ((BFieldElement::from_raw_u64(231348413345175u64))
                        * (current_base_row[305usize])))
                    + ((BFieldElement::from_raw_u64(185731565704980u64))
                        * (current_base_row[306usize])))
                    + ((BFieldElement::from_raw_u64(32014686216930u64))
                        * (current_base_row[307usize])))
                    + ((BFieldElement::from_raw_u64(145268678818785u64))
                        * (current_base_row[308usize])))
                    + ((BFieldElement::from_raw_u64(123480309731250u64))
                        * (current_base_row[309usize])))
                    + ((BFieldElement::from_raw_u64(4758823762860u64))
                        * (current_base_row[310usize])))
                    + ((BFieldElement::from_raw_u64(263719581847590u64))
                        * (current_base_row[311usize])))
                    + ((BFieldElement::from_raw_u64(76643691379275u64))
                        * (current_base_row[312usize]))) + (current_base_row[127usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[107usize]))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(76643691379275u64))
                    * (node_4832))
                    + ((BFieldElement::from_raw_u64(115096533571410u64)) * (node_4843)))
                    + ((BFieldElement::from_raw_u64(256362302871255u64)) * (node_4854)))
                    + ((BFieldElement::from_raw_u64(51629801853195u64)) * (node_4865)))
                    + ((BFieldElement::from_raw_u64(175668457332795u64))
                        * (current_base_row[301usize])))
                    + ((BFieldElement::from_raw_u64(177601192615545u64))
                        * (current_base_row[302usize])))
                    + ((BFieldElement::from_raw_u64(118201794925695u64))
                        * (current_base_row[303usize])))
                    + ((BFieldElement::from_raw_u64(244602682417545u64))
                        * (current_base_row[304usize])))
                    + ((BFieldElement::from_raw_u64(51685636428030u64))
                        * (current_base_row[305usize])))
                    + ((BFieldElement::from_raw_u64(231348413345175u64))
                        * (current_base_row[306usize])))
                    + ((BFieldElement::from_raw_u64(185731565704980u64))
                        * (current_base_row[307usize])))
                    + ((BFieldElement::from_raw_u64(32014686216930u64))
                        * (current_base_row[308usize])))
                    + ((BFieldElement::from_raw_u64(145268678818785u64))
                        * (current_base_row[309usize])))
                    + ((BFieldElement::from_raw_u64(123480309731250u64))
                        * (current_base_row[310usize])))
                    + ((BFieldElement::from_raw_u64(4758823762860u64))
                        * (current_base_row[311usize])))
                    + ((BFieldElement::from_raw_u64(263719581847590u64))
                        * (current_base_row[312usize]))) + (current_base_row[128usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[108usize]))),
            (current_base_row[129usize]) * (node_6132),
            (current_base_row[135usize]) * (node_6184),
            ((next_base_row[135usize]) * (next_base_row[136usize]))
                + ((node_6184)
                    * (((next_base_row[136usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[136usize])))
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)))),
            ((next_base_row[139usize]) * (current_base_row[143usize])) * (node_6234),
            (next_base_row[139usize]) * (current_base_row[145usize]),
            (node_6244)
                * ((next_base_row[142usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[142usize]))),
            (((node_6244) * (current_base_row[143usize])) * (node_6234)) * (node_6252),
            ((node_6244) * (current_base_row[145usize])) * (node_6252),
            (((node_6244) * (node_6234)) * (node_6237)) * (node_6258),
            ((node_6244) * (node_6240)) * (node_6261),
            (((current_base_row[313usize]) * (node_6278)) * (node_6280))
                * (current_base_row[147usize]),
            ((node_6283) * (node_6280)) * (node_6285),
            (((current_base_row[316usize]) * (node_6258)) * (node_6240)) * (node_6285),
            (((current_base_row[316usize]) * (node_6237)) * (node_6261))
                * (current_base_row[147usize]),
            ((current_base_row[353usize])
                * ((current_base_row[139usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                * ((current_base_row[147usize])
                    + (BFieldElement::from_raw_u64(18446744060824649731u64))),
            ((current_base_row[353usize]) * (current_base_row[139usize]))
                * (current_base_row[147usize]),
            ((node_6244) * (current_base_row[352usize]))
                * (((current_base_row[147usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((BFieldElement::from_raw_u64(8589934590u64))
                            * (next_base_row[147usize]))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((node_6237) * (node_6240)))),
            (current_base_row[359usize]) * ((current_base_row[147usize]) + (node_6250)),
            ((current_base_row[327usize]) * (next_base_row[143usize]))
                * ((next_base_row[147usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[147usize]))),
            (current_base_row[354usize])
                * ((next_base_row[143usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[143usize]))),
            ((current_base_row[354usize]) * (node_6261))
                * ((current_base_row[147usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (node_6344))),
            ((current_base_row[354usize]) * (node_6240))
                * ((current_base_row[147usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[360usize]))),
            ((node_6244) * ((current_base_row[324usize]) * (node_6271)))
                * (((current_base_row[147usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[147usize]))) + (node_6294)),
            (current_base_row[169usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_1867) * (node_1850))),
            (current_base_row[170usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_243) * (node_245))),
            (current_base_row[171usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_1867) * (current_base_row[13usize]))),
            (current_base_row[172usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[169usize]) * (node_1840))),
            (current_base_row[173usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[12usize]) * (node_1850)) * (node_1840))),
            (current_base_row[174usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[171usize]) * (node_1840))),
            (current_base_row[175usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[170usize]) * (current_base_row[40usize]))),
            (current_base_row[176usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_243) * (current_base_row[41usize]))),
            (current_base_row[177usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[176usize]) * (node_248))),
            (current_base_row[178usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[172usize]) * (node_1842))),
            (current_base_row[179usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[173usize]) * (node_1842))),
            (current_base_row[180usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[172usize]) * (current_base_row[15usize]))),
            (current_base_row[181usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[174usize]) * (node_1842))),
            (current_base_row[182usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[174usize]) * (current_base_row[15usize]))),
            (current_base_row[183usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[170usize]) * (node_248))),
            (current_base_row[184usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[173usize]) * (current_base_row[15usize]))),
            (current_base_row[185usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[183usize]) * (current_base_row[39usize]))),
            (current_base_row[186usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[175usize]) * (node_295))),
            (current_base_row[187usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[175usize]) * (current_base_row[39usize]))),
            (current_base_row[188usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[178usize]) * (node_1844))),
            (current_base_row[189usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[177usize]) * (node_295))),
            (current_base_row[190usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[177usize]) * (current_base_row[39usize]))),
            (current_base_row[191usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[12usize]) * (current_base_row[13usize]))
                        * (node_1840))),
            (current_base_row[192usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[184usize]) * (node_1844))),
            (current_base_row[193usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[179usize]) * (node_1844))),
            (current_base_row[194usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[180usize]) * (node_1844))),
            (current_base_row[195usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[181usize]) * (node_1844))),
            (current_base_row[196usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[171usize]) * (current_base_row[14usize]))),
            (current_base_row[197usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[182usize]) * (node_1844))),
            (current_base_row[198usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[169usize]) * (current_base_row[14usize]))),
            (current_base_row[199usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[178usize]) * (current_base_row[16usize]))),
            (current_base_row[200usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[188usize]) * (node_1846))),
            (current_base_row[201usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[191usize]) * (node_1842))),
            (current_base_row[202usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[180usize]) * (current_base_row[16usize]))),
            (current_base_row[203usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[179usize]) * (current_base_row[16usize]))),
            (current_base_row[204usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[181usize]) * (current_base_row[16usize]))),
            (current_base_row[205usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[182usize]) * (current_base_row[16usize]))),
            (current_base_row[206usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[195usize]) * (node_1846))),
            (current_base_row[207usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[201usize]) * (node_1844)) * (node_1846))
                        * (node_1848))),
            (current_base_row[208usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[194usize]) * (node_1846))),
            (current_base_row[209usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[192usize]) * (node_1846)) * (node_1848))),
            (current_base_row[210usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[197usize]) * (node_1846))),
            (current_base_row[211usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[203usize]) * (node_1846)) * (node_1848))),
            (current_base_row[212usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[202usize]) * (node_1846))),
            (current_base_row[213usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[193usize]) * (node_1846)) * (node_1848))),
            (current_base_row[214usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[206usize]) * (node_1848))),
            (current_base_row[215usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[199usize]) * (node_1846))),
            (current_base_row[216usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[200usize]) * (node_1848))),
            (current_base_row[217usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[196usize]) * (node_1842))),
            (current_base_row[218usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[188usize]) * (current_base_row[17usize]))),
            (current_base_row[219usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[192usize]) * (current_base_row[17usize]))
                        * (node_1848))),
            (current_base_row[220usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[198usize]) * (node_1842))),
            (current_base_row[221usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[208usize]) * (node_1848))),
            (current_base_row[222usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[191usize]) * (current_base_row[15usize]))
                        * (node_1844)) * (node_1846))),
            (current_base_row[223usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[196usize]) * (current_base_row[15usize]))),
            (current_base_row[224usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[193usize]) * (current_base_row[17usize]))
                        * (node_1848))),
            (current_base_row[225usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[198usize]) * (current_base_row[15usize]))),
            (current_base_row[226usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[222usize]) * (node_1848))),
            (current_base_row[227usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[212usize]) * (node_1848))),
            (current_base_row[228usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[215usize]) * (node_1848))),
            (current_base_row[229usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[218usize]) * (node_1848))),
            (current_base_row[230usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[210usize]) * (node_1848))),
            (current_base_row[231usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[204usize]) * (node_1846))),
            (current_base_row[232usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[184usize]) * (current_base_row[16usize]))
                        * (node_1846)) * (node_1848))),
            (current_base_row[233usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[205usize]) * (node_1846)) * (node_1848))),
            (current_base_row[234usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[194usize]) * (current_base_row[17usize]))
                        * (node_1848))),
            (current_base_row[235usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[42usize]) * (node_245))),
            (current_base_row[236usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[42usize]) * (current_base_row[41usize]))),
            (current_base_row[237usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[197usize]) * (current_base_row[17usize]))
                        * (node_1848))),
            (current_base_row[238usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[220usize]) * (node_1844)) * (node_1846))
                        * (node_1848))),
            (current_base_row[239usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[204usize]) * (current_base_row[17usize]))
                        * (node_1848))),
            (current_base_row[240usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[200usize]) * (current_base_row[18usize]))),
            (current_base_row[241usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[205usize]) * (current_base_row[17usize]))
                        * (node_1848))),
            (current_base_row[242usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[217usize]) * (node_1844)) * (node_1846))
                        * (node_1848))),
            (current_base_row[243usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[223usize]) * (node_1844)) * (node_1846))
                        * (node_1848))),
            (current_base_row[244usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[199usize]) * (current_base_row[17usize]))
                        * (node_1848))),
            (current_base_row[245usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[217usize]) * (current_base_row[16usize]))
                        * (node_1846)) * (node_1848))),
            (current_base_row[246usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[225usize]) * (node_1844)) * (node_1846))
                        * (node_1848))),
            (current_base_row[247usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[223usize]) * (current_base_row[16usize]))
                        * (node_1846)) * (node_1848))),
            (current_base_row[248usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[231usize]) * (node_1848))),
            (current_base_row[249usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[220usize]) * (current_base_row[16usize]))
                        * (node_1846)) * (node_1848))),
            (current_base_row[250usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[195usize]) * (current_base_row[17usize]))
                        * (node_1848))),
            (current_base_row[251usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[225usize]) * (current_base_row[16usize]))
                        * (node_1846)) * (node_1848))),
            (current_base_row[252usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[41usize])
                        * ((current_base_row[41usize])
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))))),
            (current_base_row[253usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[97usize]) * (current_base_row[97usize]))
                        * (current_base_row[97usize])) * (current_base_row[97usize]))),
            (current_base_row[254usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[202usize]) * (current_base_row[17usize]))
                        * (node_1848))),
            (current_base_row[255usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[203usize]) * (current_base_row[17usize]))
                        * (node_1848))),
            (current_base_row[256usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[98usize]) * (current_base_row[98usize]))
                        * (current_base_row[98usize])) * (current_base_row[98usize]))),
            (current_base_row[257usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[99usize]) * (current_base_row[99usize]))
                        * (current_base_row[99usize])) * (current_base_row[99usize]))),
            (current_base_row[258usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[201usize]) * (current_base_row[16usize]))
                        * (node_1846)) * (node_1848))),
            (current_base_row[259usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[100usize]) * (current_base_row[100usize]))
                        * (current_base_row[100usize])) * (current_base_row[100usize]))),
            (current_base_row[260usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[101usize]) * (current_base_row[101usize]))
                        * (current_base_row[101usize])) * (current_base_row[101usize]))),
            (current_base_row[261usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[42usize])
                        * ((current_base_row[42usize])
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))))),
            (current_base_row[262usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[102usize]) * (current_base_row[102usize]))
                        * (current_base_row[102usize])) * (current_base_row[102usize]))),
            (current_base_row[263usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[176usize]) * (current_base_row[40usize]))),
            (current_base_row[264usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[235usize]) * (node_248))),
            (current_base_row[265usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[235usize]) * (current_base_row[40usize]))),
            (current_base_row[266usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[236usize]) * (node_248))),
            (current_base_row[267usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[236usize]) * (current_base_row[40usize]))),
            (current_base_row[268usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[103usize]) * (current_base_row[103usize]))
                        * (current_base_row[103usize])) * (current_base_row[103usize]))),
            (current_base_row[269usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[104usize]) * (current_base_row[104usize]))
                        * (current_base_row[104usize])) * (current_base_row[104usize]))),
            (current_base_row[270usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[105usize]) * (current_base_row[105usize]))
                        * (current_base_row[105usize])) * (current_base_row[105usize]))),
            (current_base_row[271usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[40usize]) * (node_133))),
            (current_base_row[272usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[106usize]) * (current_base_row[106usize]))
                        * (current_base_row[106usize])) * (current_base_row[106usize]))),
            (current_base_row[273usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[107usize]) * (current_base_row[107usize]))
                        * (current_base_row[107usize])) * (current_base_row[107usize]))),
            (current_base_row[274usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[108usize]) * (current_base_row[108usize]))
                        * (current_base_row[108usize])) * (current_base_row[108usize]))),
            (current_base_row[275usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[39usize]) * (node_1295))),
            (current_base_row[276usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[39usize]) * (current_base_row[22usize]))),
            (current_base_row[277usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[206usize]) * (current_base_row[18usize]))),
            (current_base_row[278usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[210usize]) * (current_base_row[18usize]))),
            (current_base_row[279usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[208usize]) * (current_base_row[18usize]))),
            (current_base_row[280usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((next_base_row[64usize]) * (node_5670)) * (node_5671))
                        * (node_5673))),
            (current_base_row[281usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[231usize]) * (current_base_row[18usize]))),
            (current_base_row[282usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((next_base_row[62usize]) * (node_5677)) * (node_5625))),
            (current_base_row[283usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[44usize])
                        * ((current_base_row[44usize])
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))))),
            (current_base_row[284usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[43usize])
                        * ((current_base_row[43usize])
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                        * ((current_base_row[43usize])
                            + (BFieldElement::from_raw_u64(18446744060824649731u64))))
                        * ((current_base_row[43usize])
                            + (BFieldElement::from_raw_u64(18446744056529682436u64))))),
            (current_base_row[285usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[185usize]) * (node_527))),
            (current_base_row[286usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[186usize])
                        * ((next_base_row[25usize]) + (node_184)))),
            (current_base_row[287usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[187usize])
                        * ((next_base_row[26usize]) + (node_184)))),
            (current_base_row[288usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[189usize])
                        * ((next_base_row[27usize]) + (node_184)))),
            (current_base_row[289usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[190usize])
                        * ((next_base_row[28usize]) + (node_184)))),
            (current_base_row[290usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[185usize]) * (node_189))),
            (current_base_row[291usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[186usize])
                        * ((next_base_row[23usize]) + (node_192)))),
            (current_base_row[292usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[187usize])
                        * ((next_base_row[23usize]) + (node_196)))),
            (current_base_row[293usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[189usize])
                        * ((next_base_row[23usize]) + (node_200)))),
            (current_base_row[294usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[190usize]) * (node_410))),
            (current_base_row[295usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[215usize]) * (current_base_row[18usize]))),
            (current_base_row[296usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[212usize]) * (current_base_row[18usize]))),
            (current_base_row[297usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[218usize]) * (current_base_row[18usize]))),
            (current_base_row[298usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((node_6263) * (node_6265)) * (node_6269)) * (node_6271))),
            (current_base_row[299usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_6263)
                        * ((next_base_row[142usize])
                            + (BFieldElement::from_raw_u64(18446744043644780551u64))))),
            (current_base_row[300usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((node_5670) * (node_5671)) * (node_5673)) * (node_5675))),
            (current_base_row[301usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[253usize]) * (current_base_row[97usize]))
                        * (current_base_row[97usize])) * (current_base_row[97usize]))),
            (current_base_row[302usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[256usize]) * (current_base_row[98usize]))
                        * (current_base_row[98usize])) * (current_base_row[98usize]))),
            (current_base_row[303usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[257usize]) * (current_base_row[99usize]))
                        * (current_base_row[99usize])) * (current_base_row[99usize]))),
            (current_base_row[304usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[259usize]) * (current_base_row[100usize]))
                        * (current_base_row[100usize])) * (current_base_row[100usize]))),
            (current_base_row[305usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[260usize]) * (current_base_row[101usize]))
                        * (current_base_row[101usize])) * (current_base_row[101usize]))),
            (current_base_row[306usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[262usize]) * (current_base_row[102usize]))
                        * (current_base_row[102usize])) * (current_base_row[102usize]))),
            (current_base_row[307usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[268usize]) * (current_base_row[103usize]))
                        * (current_base_row[103usize])) * (current_base_row[103usize]))),
            (current_base_row[308usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[269usize]) * (current_base_row[104usize]))
                        * (current_base_row[104usize])) * (current_base_row[104usize]))),
            (current_base_row[309usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[270usize]) * (current_base_row[105usize]))
                        * (current_base_row[105usize])) * (current_base_row[105usize]))),
            (current_base_row[310usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[272usize]) * (current_base_row[106usize]))
                        * (current_base_row[106usize])) * (current_base_row[106usize]))),
            (current_base_row[311usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[273usize]) * (current_base_row[107usize]))
                        * (current_base_row[107usize])) * (current_base_row[107usize]))),
            (current_base_row[312usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[274usize]) * (current_base_row[108usize]))
                        * (current_base_row[108usize])) * (current_base_row[108usize]))),
            (current_base_row[313usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_6244) * ((current_base_row[298usize]) * (node_6275)))),
            (current_base_row[314usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[299usize]) * (node_6265))),
            (current_base_row[315usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[300usize]) * (node_5677))),
            (current_base_row[316usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_6283) * (node_6278))),
            (current_base_row[317usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[39usize]) * (node_1587))),
            (current_base_row[318usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((node_5616) * (node_5637)) * (next_base_row[62usize]))),
            (current_base_row[319usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((node_4247) * (next_base_row[13usize])) * (node_4249))
                        * (node_4251))),
            (current_base_row[320usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[23usize]) * (next_base_row[23usize]))),
            (current_base_row[321usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[22usize]) * (current_base_row[25usize]))),
            (current_base_row[322usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[24usize]) * (current_base_row[27usize]))),
            (current_base_row[323usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[24usize]) * (next_base_row[24usize]))),
            (current_base_row[324usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[314usize]) * (node_6269))),
            (current_base_row[325usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((node_4247)
                        * ((next_base_row[13usize])
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                        * (node_4249)) * (node_4251))),
            (current_base_row[326usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((((current_base_row[10usize])
                        + (BFieldElement::from_raw_u64(18446743897615892521u64)))
                        * ((current_base_row[10usize])
                            + (BFieldElement::from_raw_u64(18446743923385696291u64))))
                        * ((current_base_row[10usize])
                            + (BFieldElement::from_raw_u64(18446743863256154161u64))))
                        * ((current_base_row[10usize])
                            + (BFieldElement::from_raw_u64(18446743828896415801u64))))),
            (current_base_row[327usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_6244)
                        * (((current_base_row[314usize]) * (node_6271)) * (node_6275)))),
            (current_base_row[328usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[39usize])
                        * ((current_base_row[39usize])
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))))),
            (current_base_row[329usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[183usize]) * (node_295))),
            (current_base_row[330usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[263usize]) * (node_295))),
            (current_base_row[331usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[263usize]) * (current_base_row[39usize]))),
            (current_base_row[332usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[264usize]) * (node_295))),
            (current_base_row[333usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[264usize]) * (current_base_row[39usize]))),
            (current_base_row[334usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[265usize]) * (node_295))),
            (current_base_row[335usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[265usize]) * (current_base_row[39usize]))),
            (current_base_row[336usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[266usize]) * (node_295))),
            (current_base_row[337usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[266usize]) * (current_base_row[39usize]))),
            (current_base_row[338usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[267usize]) * (node_295))),
            (current_base_row[339usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[267usize]) * (current_base_row[39usize]))),
            (current_base_row[340usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[39usize]) * (current_base_row[42usize]))),
            (current_base_row[341usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[319usize]) * (next_base_row[16usize]))
                        * (node_4254))
                        * ((next_base_row[18usize])
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))))),
            (current_base_row[342usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[315usize])
                        * (((node_5637) * (node_5621)) * (next_base_row[62usize])))),
            (current_base_row[343usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((node_5612) * (node_5613)) * (current_base_row[62usize]))),
            (current_base_row[344usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[238usize])
                        * ((next_base_row[22usize])
                            * (((current_base_row[39usize])
                                * ((next_base_row[23usize])
                                    + (BFieldElement::from_raw_u64(4294967296u64))))
                                + (BFieldElement::from_raw_u64(
                                    18446744065119617026u64,
                                )))))),
            (current_base_row[345usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[214usize])
                        * ((((node_1224) * (current_base_row[22usize]))
                            + (((node_116) * (node_1241)) * (node_133)))
                            + ((((node_113)
                                + (BFieldElement::from_raw_u64(18446744056529682436u64)))
                                * (node_1241)) * (current_base_row[40usize]))))),
            (current_base_row[346usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[214usize])
                        * (((current_base_row[252usize])
                            * ((current_base_row[41usize])
                                + (BFieldElement::from_raw_u64(18446744060824649731u64))))
                            * ((current_base_row[41usize])
                                + (BFieldElement::from_raw_u64(
                                    18446744056529682436u64,
                                )))))),
            (current_base_row[347usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[214usize])
                        * (((current_base_row[261usize])
                            * ((current_base_row[42usize])
                                + (BFieldElement::from_raw_u64(18446744060824649731u64))))
                            * ((current_base_row[42usize])
                                + (BFieldElement::from_raw_u64(
                                    18446744056529682436u64,
                                )))))),
            (current_base_row[348usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[214usize])
                        * (((current_base_row[283usize])
                            * ((current_base_row[44usize])
                                + (BFieldElement::from_raw_u64(18446744060824649731u64))))
                            * ((current_base_row[44usize])
                                + (BFieldElement::from_raw_u64(
                                    18446744056529682436u64,
                                )))))),
            (current_base_row[349usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[325usize]) * (next_base_row[16usize]))
                        * (node_4254)) * (next_base_row[18usize]))),
            (current_base_row[350usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[64usize])
                        * ((current_base_row[64usize])
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                        * ((current_base_row[64usize])
                            + (BFieldElement::from_raw_u64(18446744060824649731u64))))
                        * ((current_base_row[64usize])
                            + (BFieldElement::from_raw_u64(18446744056529682436u64))))),
            (current_base_row[351usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((node_5634) * (node_5613)) * (current_base_row[62usize]))
                        * (node_5637))),
            (current_base_row[352usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[299usize]) * (node_6269)) * (node_6271))
                        * (node_6275))),
            (current_base_row[353usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[316usize])
                        * ((((BFieldElement::from_raw_u64(4294967295u64)) + (node_6294))
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (node_6240)))
                            + (((BFieldElement::from_raw_u64(8589934590u64))
                                * (node_6237)) * (node_6240))))),
            (current_base_row[354usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_6244) * ((current_base_row[324usize]) * (node_6275)))),
            (current_base_row[355usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[343usize])
                        * (((node_5616) * (node_5621)) * (next_base_row[62usize])))),
            (current_base_row[356usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((node_5641) * (current_base_row[62usize])) * (node_5621))),
            (current_base_row[357usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[315usize]) * (node_5621))
                        * (next_base_row[62usize])) * (node_5625))),
            (current_base_row[358usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[315usize])
                        * (((node_5728) * (node_5625)) * (node_5730)))),
            (current_base_row[359usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[327usize])
                        * ((BFieldElement::from_raw_u64(4294967295u64))
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((next_base_row[143usize]) * (next_base_row[144usize])))))
                        * (current_base_row[143usize]))),
            (current_base_row[360usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_6344) * (current_base_row[143usize]))),
        ];
        let ext_constraints = [
            (((BFieldElement::from_raw_u64(4294967295u64))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (current_base_row[5usize])))
                * (((node_51)
                    * ((challenges[3usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((((challenges[13usize]) * (current_base_row[0usize]))
                                + ((challenges[14usize]) * (current_base_row[1usize])))
                                + ((challenges[15usize]) * (next_base_row[1usize]))))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[2usize]))))
                + ((current_base_row[5usize]) * (node_51)),
            ((node_31)
                * (((next_ext_row[1usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((challenges[29usize]) * (current_ext_row[1usize]))))
                    + (node_74)))
                + ((node_34)
                    * (((next_ext_row[1usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (challenges[29usize]))) + (node_74))),
            ((((((next_ext_row[2usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((challenges[30usize]) * (current_ext_row[2usize]))))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (next_ext_row[1usize]))) * (node_47))
                * ((BFieldElement::from_raw_u64(4294967295u64))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((next_base_row[4usize]) * (node_90)))))
                + ((node_88) * (next_base_row[6usize]))) + ((node_88) * (node_90)),
            ((((((((((((((((((((((((((((((((((((((((((((current_base_row[207usize])
                * (node_120))
                + ((current_base_row[213usize])
                    * ((next_base_row[22usize]) + (node_522))))
                + ((current_base_row[209usize]) * (node_120)))
                + ((current_base_row[211usize])
                    * (((((((((((((((((current_base_row[329usize]) * (node_785))
                        + (node_253)) + (node_299)) + (node_342)) + (node_384))
                        + (node_423))
                        + ((current_base_row[330usize])
                            * ((next_base_row[22usize]) + (node_204))))
                        + ((current_base_row[331usize])
                            * ((next_base_row[22usize]) + (node_208))))
                        + ((current_base_row[332usize])
                            * ((next_base_row[22usize]) + (node_212))))
                        + ((current_base_row[333usize])
                            * ((next_base_row[22usize]) + (node_216))))
                        + ((current_base_row[334usize]) * (node_804)))
                        + ((current_base_row[335usize])
                            * ((next_base_row[22usize]) + (node_224))))
                        + ((current_base_row[336usize])
                            * ((next_base_row[22usize]) + (node_228))))
                        + ((current_base_row[337usize])
                            * ((next_base_row[22usize]) + (node_232))))
                        + ((current_base_row[338usize])
                            * ((next_base_row[22usize]) + (node_236))))
                        + ((current_base_row[339usize])
                            * ((next_base_row[22usize]) + (node_240))))))
                + ((current_base_row[232usize])
                    * (((((((((((((((((current_base_row[329usize])
                        * ((node_868)
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((((((node_887) + (node_888)) + (node_890)) + (node_892))
                                    + (node_894)) + (node_896)) + (node_898)))))
                        + ((current_base_row[185usize])
                            * ((node_868)
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (((((((((((((((((challenges[32usize])
                                        * (current_base_row[23usize]))
                                        + ((challenges[33usize]) * (current_base_row[22usize])))
                                        + (node_872)) + (node_874)) + (node_876)) + (node_878))
                                        + (node_880)) + (node_882)) + (node_884)) + (node_886))
                                        + (node_888)) + (node_890)) + (node_892)) + (node_894))
                                        + (node_896)) + (node_898))))))
                        + ((current_base_row[186usize])
                            * ((node_868)
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (((((((((((((((((challenges[32usize])
                                        * (current_base_row[24usize])) + (node_870))
                                        + ((challenges[34usize]) * (current_base_row[22usize])))
                                        + (node_874)) + (node_876)) + (node_878)) + (node_880))
                                        + (node_882)) + (node_884)) + (node_886)) + (node_888))
                                        + (node_890)) + (node_892)) + (node_894)) + (node_896))
                                        + (node_898))))))
                        + ((current_base_row[187usize])
                            * ((node_868)
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (((((((((((((((((challenges[32usize])
                                        * (current_base_row[25usize])) + (node_870)) + (node_872))
                                        + ((challenges[35usize]) * (current_base_row[22usize])))
                                        + (node_876)) + (node_878)) + (node_880)) + (node_882))
                                        + (node_884)) + (node_886)) + (node_888)) + (node_890))
                                        + (node_892)) + (node_894)) + (node_896)) + (node_898))))))
                        + ((current_base_row[189usize])
                            * ((node_868)
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (((((((((((((((((challenges[32usize])
                                        * (current_base_row[26usize])) + (node_870)) + (node_872))
                                        + (node_874))
                                        + ((challenges[36usize]) * (current_base_row[22usize])))
                                        + (node_878)) + (node_880)) + (node_882)) + (node_884))
                                        + (node_886)) + (node_888)) + (node_890)) + (node_892))
                                        + (node_894)) + (node_896)) + (node_898))))))
                        + ((current_base_row[190usize])
                            * ((node_868)
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (((((((((((((((((challenges[32usize])
                                        * (current_base_row[27usize])) + (node_870)) + (node_872))
                                        + (node_874)) + (node_876))
                                        + ((challenges[37usize]) * (current_base_row[22usize])))
                                        + (node_880)) + (node_882)) + (node_884)) + (node_886))
                                        + (node_888)) + (node_890)) + (node_892)) + (node_894))
                                        + (node_896)) + (node_898))))))
                        + ((current_base_row[330usize])
                            * ((node_868)
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (((((((((((((((((challenges[32usize])
                                        * (current_base_row[28usize])) + (node_870)) + (node_872))
                                        + (node_874)) + (node_876)) + (node_878))
                                        + ((challenges[38usize]) * (current_base_row[22usize])))
                                        + (node_882)) + (node_884)) + (node_886)) + (node_888))
                                        + (node_890)) + (node_892)) + (node_894)) + (node_896))
                                        + (node_898))))))
                        + ((current_base_row[331usize])
                            * ((node_868)
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (((((((((((((((((challenges[32usize])
                                        * (current_base_row[29usize])) + (node_870)) + (node_872))
                                        + (node_874)) + (node_876)) + (node_878)) + (node_880))
                                        + ((challenges[39usize]) * (current_base_row[22usize])))
                                        + (node_884)) + (node_886)) + (node_888)) + (node_890))
                                        + (node_892)) + (node_894)) + (node_896)) + (node_898))))))
                        + ((current_base_row[332usize])
                            * ((node_868)
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (((((((((((((((((challenges[32usize])
                                        * (current_base_row[30usize])) + (node_870)) + (node_872))
                                        + (node_874)) + (node_876)) + (node_878)) + (node_880))
                                        + (node_882))
                                        + ((challenges[40usize]) * (current_base_row[22usize])))
                                        + (node_886)) + (node_888)) + (node_890)) + (node_892))
                                        + (node_894)) + (node_896)) + (node_898))))))
                        + ((current_base_row[333usize])
                            * ((node_868)
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (((((((((((((((((challenges[32usize])
                                        * (current_base_row[31usize])) + (node_870)) + (node_872))
                                        + (node_874)) + (node_876)) + (node_878)) + (node_880))
                                        + (node_882)) + (node_884))
                                        + ((challenges[41usize]) * (current_base_row[22usize])))
                                        + (node_888)) + (node_890)) + (node_892)) + (node_894))
                                        + (node_896)) + (node_898))))))
                        + ((current_base_row[334usize])
                            * ((node_868)
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (((((((((((((((((challenges[32usize])
                                        * (current_base_row[32usize])) + (node_870)) + (node_872))
                                        + (node_874)) + (node_876)) + (node_878)) + (node_880))
                                        + (node_882)) + (node_884)) + (node_886))
                                        + ((challenges[42usize]) * (current_base_row[22usize])))
                                        + (node_890)) + (node_892)) + (node_894)) + (node_896))
                                        + (node_898))))))
                        + ((current_base_row[335usize])
                            * ((node_868)
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (((((((((((((((((challenges[32usize])
                                        * (current_base_row[33usize])) + (node_870)) + (node_872))
                                        + (node_874)) + (node_876)) + (node_878)) + (node_880))
                                        + (node_882)) + (node_884)) + (node_886)) + (node_888))
                                        + ((challenges[43usize]) * (current_base_row[22usize])))
                                        + (node_892)) + (node_894)) + (node_896)) + (node_898))))))
                        + ((current_base_row[336usize])
                            * ((node_868)
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (((((((((((((((((challenges[32usize])
                                        * (current_base_row[34usize])) + (node_870)) + (node_872))
                                        + (node_874)) + (node_876)) + (node_878)) + (node_880))
                                        + (node_882)) + (node_884)) + (node_886)) + (node_888))
                                        + (node_890))
                                        + ((challenges[44usize]) * (current_base_row[22usize])))
                                        + (node_894)) + (node_896)) + (node_898))))))
                        + ((current_base_row[337usize])
                            * ((node_868)
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (((((((((((((((((challenges[32usize])
                                        * (current_base_row[35usize])) + (node_870)) + (node_872))
                                        + (node_874)) + (node_876)) + (node_878)) + (node_880))
                                        + (node_882)) + (node_884)) + (node_886)) + (node_888))
                                        + (node_890)) + (node_892))
                                        + ((challenges[45usize]) * (current_base_row[22usize])))
                                        + (node_896)) + (node_898))))))
                        + ((current_base_row[338usize])
                            * ((node_868)
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (((((((((((((((((challenges[32usize])
                                        * (current_base_row[36usize])) + (node_870)) + (node_872))
                                        + (node_874)) + (node_876)) + (node_878)) + (node_880))
                                        + (node_882)) + (node_884)) + (node_886)) + (node_888))
                                        + (node_890)) + (node_892)) + (node_894))
                                        + ((challenges[46usize]) * (current_base_row[22usize])))
                                        + (node_898))))))
                        + ((current_base_row[339usize])
                            * ((node_868)
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (((((((((((((((((challenges[32usize])
                                        * (current_base_row[37usize])) + (node_870)) + (node_872))
                                        + (node_874)) + (node_876)) + (node_878)) + (node_880))
                                        + (node_882)) + (node_884)) + (node_886)) + (node_888))
                                        + (node_890)) + (node_892)) + (node_894)) + (node_896))
                                        + ((challenges[47usize])
                                            * (current_base_row[22usize])))))))))
                + ((current_base_row[216usize]) * (node_1223)))
                + ((current_base_row[221usize]) * (node_120)))
                + ((current_base_row[214usize])
                    * ((node_1241) * (current_base_row[39usize]))))
                + ((current_base_row[224usize])
                    * ((node_120)
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)))))
                + ((current_base_row[228usize]) * (node_1292)))
                + ((current_base_row[227usize]) * (node_1294)))
                + ((current_base_row[229usize])
                    * (((node_1298) * (current_base_row[39usize]))
                        + ((current_base_row[275usize]) * (node_124)))))
                + ((current_base_row[230usize])
                    * ((current_base_row[22usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)))))
                + ((current_base_row[219usize]) * (node_120)))
                + ((current_base_row[226usize]) * (node_120)))
                + ((current_base_row[248usize]) * (node_120)))
                + ((current_base_row[233usize])
                    * ((current_base_row[27usize]) + (node_525))))
                + ((current_base_row[234usize]) * (node_120)))
                + ((current_base_row[250usize]) * (node_120)))
                + ((current_base_row[244usize])
                    * ((node_785)
                        + (BFieldElement::from_raw_u64(18446744026464911371u64)))))
                + ((current_base_row[254usize]) * (node_120)))
                + ((current_base_row[237usize]) * ((node_785) + (node_184))))
                + ((current_base_row[239usize]) * (node_1584)))
                + ((current_base_row[240usize])
                    * ((node_1585)
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)))))
                + ((current_base_row[241usize])
                    * ((current_base_row[39usize]) * (node_1590))))
                + ((current_base_row[238usize])
                    * ((current_base_row[22usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((BFieldElement::from_raw_u64(18446744069414584320u64))
                                * (next_base_row[23usize])) + (next_base_row[22usize]))))))
                + ((current_base_row[242usize]) * (node_120)))
                + ((current_base_row[243usize]) * (node_120)))
                + ((current_base_row[245usize]) * (node_120)))
                + ((current_base_row[246usize]) * (node_120)))
                + ((current_base_row[247usize]) * (node_120)))
                + ((current_base_row[249usize])
                    * (((current_base_row[22usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[320usize]))) + (node_1609))))
                + ((current_base_row[251usize]) * (node_120)))
                + ((current_base_row[277usize]) * ((node_785) + (node_192))))
                + ((current_base_row[278usize])
                    * ((next_base_row[22usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((current_base_row[321usize])
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (node_1625)))))))
                + ((current_base_row[279usize])
                    * ((((node_1585)
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_1639)))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_1642)))
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)))))
                + ((current_base_row[281usize]) * (node_1584)))
                + ((current_base_row[255usize]) * (node_120)))
                + ((current_base_row[258usize]) * (node_120)))
                + ((current_base_row[295usize]) * (current_base_row[283usize])))
                + ((current_base_row[296usize]) * (node_1422)))
                + ((current_base_row[297usize]) * (node_1396))) * (node_4096))
                + ((node_113) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((((((current_base_row[207usize])
                * (node_128)) + ((current_base_row[213usize]) * (node_527)))
                + ((current_base_row[209usize]) * (node_128)))
                + ((current_base_row[211usize]) * (current_base_row[271usize])))
                + ((current_base_row[232usize]) * (current_base_row[271usize])))
                + ((current_base_row[216usize]) * (node_124)))
                + ((current_base_row[221usize]) * (node_128)))
                + ((current_base_row[214usize])
                    * ((((((current_base_row[11usize]) + (node_247))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((BFieldElement::from_raw_u64(8589934590u64))
                                * (current_base_row[41usize])))) + (node_144))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((BFieldElement::from_raw_u64(137438953440u64))
                                * (current_base_row[43usize]))))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((BFieldElement::from_raw_u64(549755813760u64))
                                * (current_base_row[44usize]))))))
                + ((current_base_row[224usize])
                    * ((next_base_row[21usize]) + (node_522))))
                + ((current_base_row[228usize]) * (node_785)))
                + ((current_base_row[227usize]) * (node_124)))
                + ((current_base_row[229usize])
                    * (((node_1298) * (node_1293))
                        + ((current_base_row[275usize]) * (node_1294)))))
                + ((current_base_row[230usize]) * (node_124)))
                + ((current_base_row[219usize]) * (node_128)))
                + ((current_base_row[226usize]) * (node_128)))
                + ((current_base_row[248usize]) * (node_128)))
                + ((current_base_row[233usize])
                    * ((current_base_row[29usize]) + (node_188))))
                + ((current_base_row[234usize]) * (node_128)))
                + ((current_base_row[250usize]) * (node_128)))
                + ((current_base_row[244usize]) * (node_124)))
                + ((current_base_row[254usize]) * (node_128)))
                + ((current_base_row[237usize]) * (node_124)))
                + ((current_base_row[239usize]) * (node_124)))
                + ((current_base_row[240usize]) * (node_124)))
                + ((current_base_row[241usize])
                    * ((next_base_row[22usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_1590)))))
                + ((current_base_row[238usize]) * (node_527)))
                + ((current_base_row[242usize]) * (node_128)))
                + ((current_base_row[243usize]) * (node_128)))
                + ((current_base_row[245usize]) * (node_128)))
                + ((current_base_row[246usize]) * (node_128)))
                + ((current_base_row[247usize]) * (node_128)))
                + ((current_base_row[249usize]) * (node_124)))
                + ((current_base_row[251usize]) * (node_128)))
                + ((current_base_row[277usize]) * ((node_1226) + (node_200))))
                + ((current_base_row[278usize])
                    * ((next_base_row[24usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((current_base_row[24usize])
                                * (current_base_row[25usize]))
                                + ((current_base_row[23usize])
                                    * (current_base_row[26usize])))
                                + ((current_base_row[22usize])
                                    * (current_base_row[27usize])))
                                + (current_base_row[322usize]))))))
                + ((current_base_row[279usize])
                    * (((((current_base_row[24usize]) * (next_base_row[22usize]))
                        + (current_base_row[320usize]))
                        + ((current_base_row[22usize]) * (next_base_row[24usize])))
                        + (current_base_row[323usize]))))
                + ((current_base_row[281usize])
                    * ((next_base_row[24usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[321usize])))))
                + ((current_base_row[255usize]) * (node_128)))
                + ((current_base_row[258usize]) * (node_128)))
                + ((current_base_row[295usize]) * (node_1230)))
                + ((current_base_row[296usize])
                    * ((current_ext_row[82usize]) + (node_1785))))
                + ((current_base_row[297usize])
                    * (((current_ext_row[7usize]) * (current_ext_row[72usize]))
                        + (node_1785)))) * (node_4096))
                + (((next_base_row[11usize]) + (node_522)) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((((((current_base_row[207usize])
                * (current_base_row[271usize]))
                + ((current_base_row[213usize]) * (node_530)))
                + ((current_base_row[209usize]) * (current_base_row[271usize])))
                + ((current_base_row[211usize]) * (node_154)))
                + ((current_base_row[232usize]) * (node_154)))
                + ((current_base_row[216usize]) * (node_785)))
                + ((current_base_row[221usize]) * (node_1225)))
                + (current_base_row[346usize]))
                + ((current_base_row[224usize]) * (node_1225)))
                + ((current_base_row[228usize]) * (node_1227)))
                + ((current_base_row[227usize]) * (node_1225)))
                + ((current_base_row[229usize]) * (node_1225)))
                + ((current_base_row[230usize]) * (node_185)))
                + ((current_base_row[219usize]) * (current_base_row[271usize])))
                + ((current_base_row[226usize]) * (current_base_row[271usize])))
                + ((current_base_row[248usize]) * (node_415)))
                + ((current_base_row[233usize]) * (node_120)))
                + ((current_base_row[234usize]) * (node_1225)))
                + ((current_base_row[250usize])
                    * ((next_ext_row[6usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_ext_row[69usize])))))
                + ((current_base_row[244usize]) * (node_1229)))
                + ((current_base_row[254usize])
                    * ((next_ext_row[6usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_ext_row[70usize])))))
                + ((current_base_row[237usize]) * (node_189)))
                + ((current_base_row[239usize]) * (node_189)))
                + ((current_base_row[240usize]) * (node_1225)))
                + ((current_base_row[241usize]) * (node_128)))
                + ((current_base_row[238usize]) * (node_530)))
                + ((current_base_row[242usize]) * (node_193)))
                + ((current_base_row[243usize]) * (node_193)))
                + ((current_base_row[245usize]) * (node_193)))
                + ((current_base_row[246usize]) * (node_1226)))
                + ((current_base_row[247usize]) * (node_193)))
                + ((current_base_row[249usize]) * (node_1226)))
                + ((current_base_row[251usize]) * (node_1226)))
                + ((current_base_row[277usize]) * (node_331)))
                + ((current_base_row[278usize]) * (node_331)))
                + ((current_base_row[279usize]) * (node_1229)))
                + ((current_base_row[281usize]) * (node_205)))
                + ((current_base_row[255usize]) * (current_base_row[271usize])))
                + ((current_base_row[258usize]) * (current_base_row[271usize])))
                + ((current_base_row[295usize]) * (node_1233)))
                + ((current_base_row[296usize])
                    * ((node_1228)
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((((current_base_row[236usize])
                                + ((current_base_row[40usize])
                                    * (current_base_row[43usize])))
                                + ((current_base_row[39usize])
                                    * (current_base_row[44usize]))) + (node_1798))))))
                + ((current_base_row[297usize])
                    * ((node_1228)
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[340usize]))))) * (node_4096))
                + ((node_128) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((((((current_base_row[207usize])
                * (node_441)) + ((current_base_row[213usize]) * (node_535)))
                + ((current_base_row[209usize]) * (node_734)))
                + ((current_base_row[211usize]) * (node_526)))
                + ((current_base_row[232usize]) * (node_513)))
                + ((current_base_row[216usize]) * (node_1229)))
                + ((current_base_row[221usize]) * (node_1230)))
                + ((current_base_row[214usize]) * (node_124)))
                + ((current_base_row[224usize]) * (node_1230)))
                + ((current_base_row[228usize]) * (node_1232)))
                + ((current_base_row[227usize]) * (node_1230)))
                + ((current_base_row[229usize]) * (node_1230)))
                + ((current_base_row[230usize]) * (node_205)))
                + ((current_base_row[219usize])
                    * ((((((current_base_row[185usize])
                        * ((node_785) + (BFieldElement::from_raw_u64(4294967295u64))))
                        + ((current_base_row[186usize])
                            * ((node_785)
                                + (BFieldElement::from_raw_u64(8589934590u64)))))
                        + ((current_base_row[187usize])
                            * ((node_785)
                                + (BFieldElement::from_raw_u64(12884901885u64)))))
                        + ((current_base_row[189usize])
                            * ((node_785)
                                + (BFieldElement::from_raw_u64(17179869180u64)))))
                        + ((current_base_row[190usize])
                            * ((node_785)
                                + (BFieldElement::from_raw_u64(21474836475u64)))))))
                + ((current_base_row[226usize])
                    * ((((((current_base_row[185usize]) * (node_1396))
                        + ((current_base_row[186usize])
                            * ((node_785)
                                + (BFieldElement::from_raw_u64(18446744060824649731u64)))))
                        + ((current_base_row[187usize]) * (node_1422)))
                        + ((current_base_row[189usize])
                            * ((node_785)
                                + (BFieldElement::from_raw_u64(18446744052234715141u64)))))
                        + ((current_base_row[190usize])
                            * ((node_785)
                                + (BFieldElement::from_raw_u64(
                                    18446744047939747846u64,
                                ))))))) + ((current_base_row[248usize]) * (node_397)))
                + ((current_base_row[233usize]) * (node_408)))
                + ((current_base_row[234usize]) * (node_1230)))
                + ((current_base_row[250usize])
                    * ((next_base_row[26usize]) + (node_236))))
                + ((current_base_row[244usize]) * (node_1234)))
                + ((current_base_row[254usize])
                    * ((next_base_row[36usize]) + (node_196))))
                + ((current_base_row[237usize]) * (node_209)))
                + ((current_base_row[239usize]) * (node_209)))
                + ((current_base_row[240usize]) * (node_1230)))
                + ((current_base_row[241usize]) * (node_201)))
                + ((current_base_row[238usize]) * (node_535)))
                + ((current_base_row[242usize]) * (node_213)))
                + ((current_base_row[243usize]) * (node_213)))
                + ((current_base_row[245usize]) * (node_213)))
                + ((current_base_row[246usize]) * (node_1231)))
                + ((current_base_row[247usize]) * (node_213)))
                + ((current_base_row[249usize]) * (node_1231)))
                + ((current_base_row[251usize]) * (node_1231)))
                + ((current_base_row[277usize]) * (node_336)))
                + ((current_base_row[278usize]) * (node_336)))
                + ((current_base_row[279usize]) * (node_1234)))
                + ((current_base_row[281usize]) * (node_225)))
                + ((current_base_row[255usize]) * (node_734)))
                + ((current_base_row[258usize]) * (node_441)))
                + ((current_base_row[295usize]) * (node_1238)))
                + ((current_base_row[296usize]) * (node_517)))
                + ((current_base_row[297usize]) * (node_517))) * (node_4096))
                + ((node_1228) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((((((current_base_row[207usize])
                * (node_445)) + ((current_base_row[213usize]) * (node_536)))
                + ((current_base_row[209usize]) * (node_738)))
                + ((current_base_row[211usize]) * (node_527)))
                + ((current_base_row[232usize]) * (node_517)))
                + ((current_base_row[216usize]) * (node_1230)))
                + ((current_base_row[221usize]) * (node_1231)))
                + ((current_base_row[214usize]) * (node_128)))
                + ((current_base_row[224usize]) * (node_1231)))
                + ((current_base_row[228usize]) * (node_1233)))
                + ((current_base_row[227usize]) * (node_1231)))
                + ((current_base_row[229usize]) * (node_1231)))
                + ((current_base_row[230usize]) * (node_209)))
                + ((current_base_row[219usize]) * (node_734)))
                + ((current_base_row[226usize]) * (node_441)))
                + ((current_base_row[248usize]) * (node_408)))
                + ((current_base_row[233usize]) * (node_409)))
                + ((current_base_row[234usize]) * (node_1231)))
                + ((current_base_row[250usize])
                    * ((next_base_row[27usize]) + (node_240))))
                + ((current_base_row[244usize]) * (node_1235)))
                + ((current_base_row[254usize])
                    * ((next_base_row[37usize]) + (node_200))))
                + ((current_base_row[237usize]) * (node_213)))
                + ((current_base_row[239usize]) * (node_213)))
                + ((current_base_row[240usize]) * (node_1231)))
                + ((current_base_row[241usize]) * (node_205)))
                + ((current_base_row[238usize]) * (node_536)))
                + ((current_base_row[242usize]) * (node_217)))
                + ((current_base_row[243usize]) * (node_217)))
                + ((current_base_row[245usize]) * (node_217)))
                + ((current_base_row[246usize]) * (node_1232)))
                + ((current_base_row[247usize]) * (node_217)))
                + ((current_base_row[249usize]) * (node_1232)))
                + ((current_base_row[251usize]) * (node_1232)))
                + ((current_base_row[277usize]) * (node_337)))
                + ((current_base_row[278usize]) * (node_337)))
                + ((current_base_row[279usize]) * (node_1235)))
                + ((current_base_row[281usize]) * (node_229)))
                + ((current_base_row[255usize]) * (node_738)))
                + ((current_base_row[258usize]) * (node_445)))
                + ((current_base_row[295usize]) * (node_1239)))
                + ((current_base_row[296usize]) * (node_521)))
                + ((current_base_row[297usize]) * (node_521))) * (node_4096))
                + ((node_1229) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((((((current_base_row[207usize])
                * (node_449)) + ((current_base_row[213usize]) * (node_537)))
                + ((current_base_row[209usize]) * (node_742)))
                + ((current_base_row[211usize]) * (node_528)))
                + ((current_base_row[232usize]) * (node_521)))
                + ((current_base_row[216usize]) * (node_1231)))
                + ((current_base_row[221usize]) * (node_1232)))
                + ((current_base_row[214usize]) * (node_185)))
                + ((current_base_row[224usize]) * (node_1232)))
                + ((current_base_row[228usize]) * (node_1234)))
                + ((current_base_row[227usize]) * (node_1232)))
                + ((current_base_row[229usize]) * (node_1232)))
                + ((current_base_row[230usize]) * (node_213)))
                + (current_ext_row[84usize])) + (current_ext_row[85usize]))
                + ((current_base_row[248usize]) * (node_513)))
                + ((current_base_row[233usize]) * (node_410)))
                + ((current_base_row[234usize]) * (node_1232)))
                + ((current_base_row[250usize]) * (node_513)))
                + ((current_base_row[244usize]) * (node_1236)))
                + ((current_base_row[254usize]) * (node_513)))
                + ((current_base_row[237usize]) * (node_217)))
                + ((current_base_row[239usize]) * (node_217)))
                + ((current_base_row[240usize]) * (node_1232)))
                + ((current_base_row[241usize]) * (node_209)))
                + ((current_base_row[238usize]) * (node_537)))
                + ((current_base_row[242usize]) * (node_221)))
                + ((current_base_row[243usize]) * (node_221)))
                + ((current_base_row[245usize]) * (node_221)))
                + ((current_base_row[246usize]) * (node_1233)))
                + ((current_base_row[247usize]) * (node_221)))
                + ((current_base_row[249usize]) * (node_1233)))
                + ((current_base_row[251usize]) * (node_1233)))
                + ((current_base_row[277usize]) * (node_338)))
                + ((current_base_row[278usize]) * (node_338)))
                + ((current_base_row[279usize]) * (node_1236)))
                + ((current_base_row[281usize]) * (node_233)))
                + ((current_base_row[255usize]) * (node_742)))
                + ((current_base_row[258usize]) * (node_449)))
                + ((current_base_row[295usize]) * (node_1219)))
                + ((current_base_row[296usize]) * (node_1229)))
                + ((current_base_row[297usize]) * (node_1229))) * (node_4096))
                + ((node_1230) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((((((current_base_row[207usize])
                * (node_453)) + ((current_base_row[213usize]) * (node_538)))
                + ((current_base_row[209usize]) * (node_746)))
                + ((current_base_row[211usize]) * (node_529)))
                + ((current_base_row[232usize]) * (node_1219)))
                + ((current_base_row[216usize]) * (node_1232)))
                + ((current_base_row[221usize]) * (node_1233)))
                + ((current_base_row[214usize]) * (node_189)))
                + ((current_base_row[224usize]) * (node_1233)))
                + ((current_base_row[228usize]) * (node_1235)))
                + ((current_base_row[227usize]) * (node_1233)))
                + ((current_base_row[229usize]) * (node_1233)))
                + ((current_base_row[230usize]) * (node_217)))
                + ((current_base_row[219usize]) * (node_742)))
                + ((current_base_row[226usize]) * (node_449)))
                + ((current_base_row[248usize]) * (node_517)))
                + ((current_base_row[233usize]) * (node_411)))
                + ((current_base_row[234usize]) * (node_1233)))
                + ((current_base_row[250usize]) * (node_517)))
                + ((current_base_row[244usize]) * (node_1237)))
                + ((current_base_row[254usize]) * (node_517)))
                + ((current_base_row[237usize]) * (node_221)))
                + ((current_base_row[239usize]) * (node_221)))
                + ((current_base_row[240usize]) * (node_1233)))
                + ((current_base_row[241usize]) * (node_213)))
                + ((current_base_row[238usize]) * (node_538)))
                + ((current_base_row[242usize]) * (node_225)))
                + ((current_base_row[243usize]) * (node_225)))
                + ((current_base_row[245usize]) * (node_225)))
                + ((current_base_row[246usize]) * (node_1234)))
                + ((current_base_row[247usize]) * (node_225)))
                + ((current_base_row[249usize]) * (node_1234)))
                + ((current_base_row[251usize]) * (node_1234)))
                + ((current_base_row[277usize]) * (node_314)))
                + ((current_base_row[278usize]) * (node_314)))
                + ((current_base_row[279usize]) * (node_1237)))
                + ((current_base_row[281usize]) * (node_237)))
                + ((current_base_row[255usize]) * (node_746)))
                + ((current_base_row[258usize]) * (node_453)))
                + ((current_base_row[295usize]) * (node_158)))
                + ((current_base_row[296usize]) * (node_1230)))
                + ((current_base_row[297usize]) * (node_1230))) * (node_4096))
                + ((node_1231) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((((((current_base_row[207usize])
                * (node_457)) + ((current_base_row[213usize]) * (node_539)))
                + ((current_base_row[209usize]) * (node_750)))
                + ((current_base_row[211usize]) * (node_530)))
                + ((current_base_row[232usize]) * (node_158)))
                + ((current_base_row[216usize]) * (node_1233)))
                + ((current_base_row[221usize]) * (node_1234)))
                + ((current_base_row[214usize]) * (node_193)))
                + ((current_base_row[224usize]) * (node_1234)))
                + ((current_base_row[228usize]) * (node_1236)))
                + ((current_base_row[227usize]) * (node_1234)))
                + ((current_base_row[229usize]) * (node_1234)))
                + ((current_base_row[230usize]) * (node_221)))
                + ((current_base_row[219usize]) * (node_746)))
                + ((current_base_row[226usize]) * (node_453)))
                + ((current_base_row[248usize]) * (node_521)))
                + ((current_base_row[233usize]) * (node_412)))
                + ((current_base_row[234usize]) * (node_1234)))
                + ((current_base_row[250usize]) * (node_521)))
                + ((current_base_row[244usize]) * (node_1238)))
                + ((current_base_row[254usize]) * (node_521)))
                + ((current_base_row[237usize]) * (node_225)))
                + ((current_base_row[239usize]) * (node_225)))
                + ((current_base_row[240usize]) * (node_1234)))
                + ((current_base_row[241usize]) * (node_217)))
                + ((current_base_row[238usize]) * (node_539)))
                + ((current_base_row[242usize]) * (node_229)))
                + ((current_base_row[243usize]) * (node_229)))
                + ((current_base_row[245usize]) * (node_229)))
                + ((current_base_row[246usize]) * (node_1235)))
                + ((current_base_row[247usize]) * (node_229)))
                + ((current_base_row[249usize]) * (node_1235)))
                + ((current_base_row[251usize]) * (node_1235)))
                + ((current_base_row[277usize]) * (node_325)))
                + ((current_base_row[278usize]) * (node_325)))
                + ((current_base_row[279usize]) * (node_1238)))
                + ((current_base_row[281usize]) * (node_241)))
                + ((current_base_row[255usize]) * (node_750)))
                + ((current_base_row[258usize]) * (node_457)))
                + ((current_base_row[295usize]) * (node_120)))
                + ((current_base_row[296usize]) * (node_1231)))
                + ((current_base_row[297usize]) * (node_1231))) * (node_4096))
                + ((node_1232) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((current_base_row[207usize])
                * (node_465)) + ((current_base_row[213usize]) * (node_541)))
                + ((current_base_row[209usize]) * (node_758)))
                + ((current_base_row[211usize]) * (node_532)))
                + ((current_base_row[216usize]) * (node_1235)))
                + ((current_base_row[221usize]) * (node_1236)))
                + ((current_base_row[214usize]) * (node_201)))
                + ((current_base_row[224usize]) * (node_1236)))
                + ((current_base_row[228usize]) * (node_1238)))
                + ((current_base_row[227usize]) * (node_1236)))
                + ((current_base_row[229usize]) * (node_1236)))
                + ((current_base_row[230usize]) * (node_229)))
                + ((current_base_row[219usize]) * (node_754)))
                + ((current_base_row[226usize]) * (node_461)))
                + ((current_base_row[233usize]) * (node_414)))
                + ((current_base_row[234usize]) * (node_1236)))
                + ((current_base_row[244usize]) * (node_1219)))
                + ((current_base_row[237usize]) * (node_233)))
                + ((current_base_row[239usize]) * (node_233)))
                + ((current_base_row[240usize]) * (node_1236)))
                + ((current_base_row[241usize]) * (node_225)))
                + ((current_base_row[238usize]) * (node_541)))
                + ((current_base_row[242usize]) * (node_237)))
                + ((current_base_row[243usize]) * (node_237)))
                + ((current_base_row[245usize]) * (node_237)))
                + ((current_base_row[246usize]) * (node_1237)))
                + ((current_base_row[247usize]) * (node_237)))
                + ((current_base_row[249usize]) * (node_1237)))
                + ((current_base_row[251usize]) * (node_1237)))
                + ((current_base_row[277usize]) * (node_124)))
                + ((current_base_row[278usize]) * (node_124)))
                + ((current_base_row[279usize]) * (node_1219)))
                + ((current_base_row[281usize]) * (node_181)))
                + ((current_base_row[255usize]) * (node_758)))
                + ((current_base_row[258usize]) * (node_465)))
                + ((current_base_row[295usize]) * (node_128)))
                + ((current_base_row[296usize]) * (node_1233)))
                + ((current_base_row[297usize]) * (node_1233))) * (node_4096))
                + ((node_1234) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((current_base_row[207usize])
                * (node_469)) + ((current_base_row[213usize]) * (node_550)))
                + ((current_base_row[209usize]) * (node_762)))
                + ((current_base_row[211usize]) * (node_533)))
                + ((current_base_row[216usize]) * (node_1236)))
                + ((current_base_row[221usize]) * (node_1237)))
                + ((current_base_row[214usize]) * (node_205)))
                + ((current_base_row[224usize]) * (node_1237)))
                + ((current_base_row[228usize]) * (node_1239)))
                + ((current_base_row[227usize]) * (node_1237)))
                + ((current_base_row[229usize]) * (node_1237)))
                + ((current_base_row[230usize]) * (node_233)))
                + ((current_base_row[219usize]) * (node_758)))
                + ((current_base_row[226usize]) * (node_465)))
                + ((current_base_row[233usize]) * (node_415)))
                + ((current_base_row[234usize]) * (node_1237)))
                + ((current_base_row[244usize]) * (node_158)))
                + ((current_base_row[237usize]) * (node_237)))
                + ((current_base_row[239usize]) * (node_237)))
                + ((current_base_row[240usize]) * (node_1237)))
                + ((current_base_row[241usize]) * (node_229)))
                + ((current_base_row[238usize]) * (node_550)))
                + ((current_base_row[242usize]) * (node_241)))
                + ((current_base_row[243usize]) * (node_241)))
                + ((current_base_row[245usize]) * (node_241)))
                + ((current_base_row[246usize]) * (node_1238)))
                + ((current_base_row[247usize]) * (node_241)))
                + ((current_base_row[249usize]) * (node_1238)))
                + ((current_base_row[251usize]) * (node_1238)))
                + ((current_base_row[277usize]) * (node_128)))
                + ((current_base_row[278usize]) * (node_128)))
                + ((current_base_row[279usize]) * (node_158)))
                + ((current_base_row[281usize]) * (node_120)))
                + ((current_base_row[255usize]) * (node_762)))
                + ((current_base_row[258usize]) * (node_469)))
                + ((current_base_row[295usize]) * (node_1224)))
                + ((current_base_row[296usize]) * (node_1234)))
                + ((current_base_row[297usize]) * (node_1234))) * (node_4096))
                + ((node_1235) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((current_base_row[207usize])
                * (node_473)) + ((current_base_row[213usize]) * (node_120)))
                + ((current_base_row[209usize]) * (node_766)))
                + ((current_base_row[211usize]) * (node_534)))
                + ((current_base_row[216usize]) * (node_1237)))
                + ((current_base_row[221usize]) * (node_1238)))
                + ((current_base_row[214usize]) * (node_209)))
                + ((current_base_row[224usize]) * (node_1238)))
                + ((current_base_row[228usize]) * (node_1219)))
                + ((current_base_row[227usize]) * (node_1238)))
                + ((current_base_row[229usize]) * (node_1238)))
                + ((current_base_row[230usize]) * (node_237)))
                + ((current_base_row[219usize]) * (node_762)))
                + ((current_base_row[226usize]) * (node_469)))
                + ((current_base_row[233usize]) * (node_416)))
                + ((current_base_row[234usize]) * (node_1238)))
                + ((current_base_row[244usize]) * (node_517)))
                + ((current_base_row[237usize]) * (node_241)))
                + ((current_base_row[239usize]) * (node_241)))
                + ((current_base_row[240usize]) * (node_1238)))
                + ((current_base_row[241usize]) * (node_233)))
                + ((current_base_row[238usize]) * (node_120)))
                + ((current_base_row[242usize]) * (node_159)))
                + ((current_base_row[243usize]) * (node_159)))
                + ((current_base_row[245usize]) * (node_159)))
                + ((current_base_row[246usize]) * (node_1239)))
                + ((current_base_row[247usize]) * (node_159)))
                + ((current_base_row[249usize]) * (node_1239)))
                + ((current_base_row[251usize]) * (node_1239)))
                + ((current_base_row[277usize]) * (node_1224)))
                + ((current_base_row[278usize]) * (node_1224)))
                + ((current_base_row[279usize]) * (node_120)))
                + ((current_base_row[281usize]) * (node_124)))
                + ((current_base_row[255usize]) * (node_766)))
                + ((current_base_row[258usize]) * (node_473)))
                + ((current_base_row[295usize]) * (node_513)))
                + ((current_base_row[296usize]) * (node_1235)))
                + ((current_base_row[297usize]) * (node_1235))) * (node_4096))
                + ((node_1236) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((current_base_row[207usize])
                * (node_477)) + ((current_base_row[213usize]) * (node_124)))
                + ((current_base_row[209usize]) * (node_770)))
                + ((current_base_row[211usize]) * (node_535)))
                + ((current_base_row[216usize]) * (node_1238)))
                + ((current_base_row[221usize]) * (node_1239)))
                + ((current_base_row[214usize]) * (node_213)))
                + ((current_base_row[224usize]) * (node_1239)))
                + ((current_base_row[228usize]) * (node_158)))
                + ((current_base_row[227usize]) * (node_1239)))
                + ((current_base_row[229usize]) * (node_1239)))
                + ((current_base_row[230usize]) * (node_241)))
                + ((current_base_row[219usize]) * (node_766)))
                + ((current_base_row[226usize]) * (node_473)))
                + ((current_base_row[233usize]) * (node_417)))
                + ((current_base_row[234usize]) * (node_1239)))
                + ((current_base_row[244usize]) * (node_521)))
                + ((current_base_row[237usize]) * (node_159)))
                + ((current_base_row[239usize]) * (node_159)))
                + ((current_base_row[240usize]) * (node_1239)))
                + ((current_base_row[241usize]) * (node_237)))
                + ((current_base_row[238usize]) * (node_124)))
                + ((current_base_row[242usize]) * (node_181)))
                + ((current_base_row[243usize]) * (node_181)))
                + ((current_base_row[245usize]) * (node_181)))
                + ((current_base_row[246usize]) * (node_1219)))
                + ((current_base_row[247usize]) * (node_181)))
                + ((current_base_row[249usize]) * (node_1219)))
                + ((current_base_row[251usize]) * (node_1219)))
                + ((current_base_row[277usize]) * (node_513)))
                + ((current_base_row[278usize]) * (node_513)))
                + ((current_base_row[279usize]) * (node_124)))
                + ((current_base_row[281usize]) * (node_128)))
                + ((current_base_row[255usize]) * (node_770)))
                + ((current_base_row[258usize]) * (node_477)))
                + ((current_base_row[295usize]) * (node_517)))
                + ((current_base_row[296usize]) * (node_1236)))
                + ((current_base_row[297usize]) * (node_1236))) * (node_4096))
                + ((node_1237) * (next_base_row[8usize])),
            (((((((((((((((((((((((((((((((((((((((current_base_row[207usize])
                * (node_481)) + ((current_base_row[213usize]) * (node_128)))
                + ((current_base_row[209usize]) * (node_774)))
                + ((current_base_row[211usize]) * (node_536)))
                + ((current_base_row[216usize]) * (node_1239)))
                + ((current_base_row[221usize]) * (node_1219)))
                + ((current_base_row[214usize]) * (node_217)))
                + ((current_base_row[224usize]) * (node_1219)))
                + ((current_base_row[228usize]) * (node_513)))
                + ((current_base_row[227usize]) * (node_1219)))
                + ((current_base_row[229usize]) * (node_1219)))
                + ((current_base_row[230usize]) * (node_159)))
                + ((current_base_row[219usize]) * (node_770)))
                + ((current_base_row[226usize]) * (node_477)))
                + ((current_base_row[233usize]) * (node_418)))
                + ((current_base_row[234usize]) * (node_1219)))
                + ((current_base_row[237usize]) * (node_181)))
                + ((current_base_row[239usize]) * (node_181)))
                + ((current_base_row[240usize]) * (node_1219)))
                + ((current_base_row[241usize]) * (node_241)))
                + ((current_base_row[238usize]) * (node_128)))
                + ((current_base_row[242usize]) * (node_513)))
                + ((current_base_row[243usize]) * (node_513)))
                + ((current_base_row[245usize]) * (node_513)))
                + ((current_base_row[246usize]) * (node_158)))
                + ((current_base_row[247usize]) * (node_513)))
                + ((current_base_row[249usize]) * (node_158)))
                + ((current_base_row[251usize]) * (node_158)))
                + ((current_base_row[277usize]) * (node_517)))
                + ((current_base_row[278usize]) * (node_517)))
                + ((current_base_row[279usize]) * (node_128)))
                + ((current_base_row[281usize]) * (node_1224)))
                + ((current_base_row[255usize]) * (node_774)))
                + ((current_base_row[258usize]) * (node_481)))
                + ((current_base_row[295usize]) * (node_521)))
                + ((current_base_row[296usize]) * (node_1237)))
                + ((current_base_row[297usize]) * (node_1237))) * (node_4096))
                + ((node_1238) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((current_base_row[207usize])
                * (node_485)) + ((current_base_row[213usize]) * (node_116)))
                + ((current_base_row[209usize]) * (node_778)))
                + ((current_base_row[211usize]) * (node_537)))
                + ((current_base_row[216usize]) * (node_1219)))
                + ((current_base_row[221usize]) * (node_158)))
                + ((current_base_row[214usize]) * (node_221)))
                + ((current_base_row[224usize]) * (node_158)))
                + ((current_base_row[228usize]) * (node_517)))
                + ((current_base_row[227usize]) * (node_158)))
                + ((current_base_row[229usize]) * (node_158)))
                + ((current_base_row[230usize]) * (node_181)))
                + ((current_base_row[219usize]) * (node_774)))
                + ((current_base_row[226usize]) * (node_481)))
                + ((current_base_row[233usize]) * (node_419)))
                + ((current_base_row[234usize]) * (node_158)))
                + ((current_base_row[237usize]) * (node_513)))
                + ((current_base_row[239usize]) * (node_513)))
                + ((current_base_row[240usize]) * (node_158)))
                + ((current_base_row[241usize]) * (node_159)))
                + ((current_base_row[238usize]) * (node_1224)))
                + ((current_base_row[242usize]) * (node_517)))
                + ((current_base_row[243usize]) * (node_517)))
                + ((current_base_row[245usize]) * (node_517)))
                + ((current_base_row[246usize]) * (node_513)))
                + ((current_base_row[247usize]) * (node_517)))
                + ((current_base_row[249usize]) * (node_513)))
                + ((current_base_row[251usize]) * (node_513)))
                + ((current_base_row[277usize]) * (node_521)))
                + ((current_base_row[278usize]) * (node_521)))
                + ((current_base_row[279usize]) * (node_1224)))
                + ((current_base_row[281usize]) * (node_513)))
                + ((current_base_row[255usize]) * (node_778)))
                + ((current_base_row[258usize]) * (node_485)))
                + ((current_base_row[296usize]) * (node_1238)))
                + ((current_base_row[297usize]) * (node_1238))) * (node_4096))
                + ((node_1239) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((current_base_row[207usize]) * (node_488))
                + ((current_base_row[213usize]) * (node_513)))
                + ((current_base_row[209usize]) * (node_781)))
                + ((current_base_row[211usize]) * (node_538)))
                + ((current_base_row[216usize]) * (node_158)))
                + ((current_base_row[221usize]) * (node_513)))
                + ((current_base_row[214usize]) * (node_225)))
                + ((current_base_row[224usize]) * (node_513)))
                + ((current_base_row[228usize]) * (node_521)))
                + ((current_base_row[227usize]) * (node_513)))
                + ((current_base_row[229usize]) * (node_513)))
                + ((current_base_row[230usize]) * (node_513)))
                + ((current_base_row[219usize]) * (node_778)))
                + ((current_base_row[226usize]) * (node_485)))
                + ((current_base_row[233usize]) * (node_513)))
                + ((current_base_row[234usize]) * (node_513)))
                + ((current_base_row[237usize]) * (node_517)))
                + ((current_base_row[239usize]) * (node_517)))
                + ((current_base_row[240usize]) * (node_513)))
                + ((current_base_row[241usize]) * (node_181)))
                + ((current_base_row[238usize]) * (node_513)))
                + ((current_base_row[242usize]) * (node_521)))
                + ((current_base_row[243usize]) * (node_521)))
                + ((current_base_row[245usize]) * (node_521)))
                + ((current_base_row[246usize]) * (node_517)))
                + ((current_base_row[247usize]) * (node_521)))
                + ((current_base_row[249usize]) * (node_517)))
                + ((current_base_row[251usize]) * (node_517)))
                + ((current_base_row[279usize]) * (node_513)))
                + ((current_base_row[281usize]) * (node_517)))
                + ((current_base_row[255usize])
                    * ((node_781)
                        + (((next_ext_row[3usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((challenges[1usize])
                                    * (((challenges[1usize])
                                        * (((challenges[1usize])
                                            * (((challenges[1usize])
                                                * ((node_1665) + (next_base_row[26usize])))
                                                + (next_base_row[25usize]))) + (next_base_row[24usize])))
                                        + (next_base_row[23usize]))) + (next_base_row[22usize]))))
                            * (current_base_row[190usize])))))
                + ((current_base_row[258usize])
                    * ((node_488)
                        + (((next_ext_row[4usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((challenges[2usize]) * (node_1727))
                                    + (current_base_row[26usize]))))
                            * (current_base_row[190usize])))))
                + ((current_base_row[296usize]) * (node_1239)))
                + ((current_base_row[297usize]) * (node_1239))) * (node_4096))
                + ((node_1219) * (next_base_row[8usize])),
            (((((((((((((((((((((((((((((((current_base_row[207usize]) * (node_490))
                + ((current_base_row[213usize]) * (node_517)))
                + ((current_base_row[209usize]) * (node_783)))
                + ((current_base_row[211usize]) * (node_539)))
                + ((current_base_row[216usize]) * (node_513)))
                + ((current_base_row[221usize]) * (node_517)))
                + ((current_base_row[214usize]) * (node_229)))
                + ((current_base_row[224usize]) * (node_517)))
                + ((current_base_row[227usize]) * (node_517)))
                + ((current_base_row[229usize]) * (node_517)))
                + ((current_base_row[230usize]) * (node_517)))
                + ((current_base_row[219usize]) * (node_781)))
                + ((current_base_row[226usize]) * (node_488)))
                + ((current_base_row[233usize]) * (node_517)))
                + ((current_base_row[234usize]) * (node_517)))
                + ((current_base_row[237usize]) * (node_521)))
                + ((current_base_row[239usize]) * (node_521)))
                + ((current_base_row[240usize]) * (node_517)))
                + ((current_base_row[241usize]) * (node_513)))
                + ((current_base_row[238usize]) * (node_517)))
                + ((current_base_row[246usize]) * (node_521)))
                + ((current_base_row[249usize]) * (node_521)))
                + ((current_base_row[251usize]) * (node_521)))
                + ((current_base_row[279usize]) * (node_517)))
                + ((current_base_row[281usize]) * (node_521)))
                + ((current_base_row[255usize])
                    * ((node_783)
                        + (((next_ext_row[3usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((challenges[1usize])
                                    * (((challenges[1usize])
                                        * (((challenges[1usize])
                                            * ((node_1665) + (next_base_row[25usize])))
                                            + (next_base_row[24usize]))) + (next_base_row[23usize])))
                                    + (next_base_row[22usize]))))
                            * (current_base_row[189usize])))))
                + ((current_base_row[258usize])
                    * ((node_490)
                        + (((next_ext_row[4usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (node_1727))) * (current_base_row[189usize])))))
                + ((current_base_row[296usize]) * (node_1219)))
                + ((current_base_row[297usize]) * (node_1219))) * (node_4096))
                + ((node_158) * (next_base_row[8usize])),
            (((((((((((((((((((((((((current_base_row[207usize]) * (node_491))
                + ((current_base_row[213usize]) * (node_521)))
                + ((current_base_row[209usize]) * (node_784)))
                + ((current_base_row[211usize]) * (node_540)))
                + ((current_base_row[216usize]) * (node_517)))
                + ((current_base_row[221usize]) * (node_521)))
                + ((current_base_row[214usize]) * (node_233)))
                + ((current_base_row[224usize]) * (node_521)))
                + ((current_base_row[227usize]) * (node_521)))
                + ((current_base_row[229usize]) * (node_521)))
                + ((current_base_row[230usize]) * (node_521)))
                + ((current_base_row[219usize]) * (node_783)))
                + ((current_base_row[226usize]) * (node_490)))
                + ((current_base_row[233usize]) * (node_521)))
                + ((current_base_row[234usize]) * (node_521)))
                + ((current_base_row[240usize]) * (node_521)))
                + ((current_base_row[241usize]) * (node_517)))
                + ((current_base_row[238usize]) * (node_521)))
                + ((current_base_row[279usize]) * (node_521)))
                + ((current_base_row[255usize])
                    * ((node_784)
                        + (((next_ext_row[3usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((challenges[1usize])
                                    * (((challenges[1usize])
                                        * ((node_1665) + (next_base_row[24usize])))
                                        + (next_base_row[23usize]))) + (next_base_row[22usize]))))
                            * (current_base_row[187usize])))))
                + ((current_base_row[258usize])
                    * ((node_491)
                        + (((next_ext_row[4usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (node_1722))) * (current_base_row[187usize])))))
                + ((current_base_row[296usize]) * (node_158)))
                + ((current_base_row[297usize]) * (node_158))) * (node_4096))
                + ((node_513) * (next_base_row[8usize])),
            ((((((((((((current_base_row[207usize]) * (node_267))
                + ((current_base_row[209usize]) * (node_567)))
                + ((current_base_row[211usize]) * (node_541)))
                + ((current_base_row[216usize]) * (node_521)))
                + ((current_base_row[214usize]) * (node_237)))
                + ((current_base_row[219usize]) * (node_784)))
                + ((current_base_row[226usize]) * (node_491)))
                + ((current_base_row[241usize]) * (node_521)))
                + ((current_base_row[255usize])
                    * ((node_567)
                        + (((next_ext_row[3usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((challenges[1usize])
                                    * ((node_1665) + (next_base_row[23usize])))
                                    + (next_base_row[22usize]))))
                            * (current_base_row[186usize])))))
                + ((current_base_row[258usize])
                    * ((node_267)
                        + (((next_ext_row[4usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (node_1717))) * (current_base_row[186usize])))))
                * (node_4096)) + ((node_517) * (next_base_row[8usize])),
            ((((((((((current_base_row[207usize]) * (current_base_row[329usize]))
                + ((current_base_row[209usize]) * (current_base_row[329usize])))
                + ((current_base_row[211usize]) * (node_550)))
                + ((current_base_row[214usize]) * (node_241)))
                + ((current_base_row[219usize]) * (node_567)))
                + ((current_base_row[226usize]) * (node_267)))
                + ((current_base_row[255usize])
                    * (((next_ext_row[3usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((node_1665) + (next_base_row[22usize]))))
                        * (current_base_row[185usize]))))
                + ((current_base_row[258usize])
                    * (((next_ext_row[4usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_1712))) * (current_base_row[185usize]))))
                * (node_4096)) + ((node_521) * (next_base_row[8usize])),
            (((((((((current_base_row[207usize]) * (current_base_row[330usize]))
                + ((current_base_row[209usize]) * (current_base_row[330usize])))
                + ((current_base_row[211usize]) * (node_513)))
                + ((current_base_row[214usize]) * (node_159)))
                + ((current_base_row[219usize]) * (current_base_row[329usize])))
                + ((current_base_row[226usize]) * (current_base_row[329usize])))
                + ((current_base_row[255usize]) * (current_base_row[329usize])))
                + ((current_base_row[258usize]) * (current_base_row[329usize])))
                * (node_4096),
            (((((((((current_base_row[207usize]) * (current_base_row[331usize]))
                + ((current_base_row[209usize]) * (current_base_row[331usize])))
                + ((current_base_row[211usize]) * (node_517)))
                + ((current_base_row[214usize]) * (node_181)))
                + ((current_base_row[219usize]) * (current_base_row[330usize])))
                + ((current_base_row[226usize]) * (current_base_row[330usize])))
                + ((current_base_row[255usize]) * (current_base_row[330usize])))
                + ((current_base_row[258usize]) * (current_base_row[330usize])))
                * (node_4096),
            (((((((((current_base_row[207usize]) * (current_base_row[332usize]))
                + ((current_base_row[209usize]) * (current_base_row[332usize])))
                + ((current_base_row[211usize]) * (node_521)))
                + ((current_base_row[214usize]) * (node_513)))
                + ((current_base_row[219usize]) * (current_base_row[331usize])))
                + ((current_base_row[226usize]) * (current_base_row[331usize])))
                + ((current_base_row[255usize]) * (current_base_row[331usize])))
                + ((current_base_row[258usize]) * (current_base_row[331usize])))
                * (node_4096),
            ((((((((current_base_row[207usize]) * (current_base_row[333usize]))
                + ((current_base_row[209usize]) * (current_base_row[333usize])))
                + ((current_base_row[214usize]) * (node_517)))
                + ((current_base_row[219usize]) * (current_base_row[332usize])))
                + ((current_base_row[226usize]) * (current_base_row[332usize])))
                + ((current_base_row[255usize]) * (current_base_row[332usize])))
                + ((current_base_row[258usize]) * (current_base_row[332usize])))
                * (node_4096),
            ((((((((current_base_row[207usize]) * (current_base_row[334usize]))
                + ((current_base_row[209usize]) * (current_base_row[334usize])))
                + ((current_base_row[214usize]) * (node_521)))
                + ((current_base_row[219usize]) * (current_base_row[333usize])))
                + ((current_base_row[226usize]) * (current_base_row[333usize])))
                + ((current_base_row[255usize]) * (current_base_row[333usize])))
                + ((current_base_row[258usize]) * (current_base_row[333usize])))
                * (node_4096),
            (((((((current_base_row[207usize]) * (node_513))
                + ((current_base_row[209usize]) * (node_513)))
                + ((current_base_row[219usize]) * (current_base_row[339usize])))
                + ((current_base_row[226usize]) * (current_base_row[339usize])))
                + ((current_base_row[255usize]) * (current_base_row[339usize])))
                + ((current_base_row[258usize]) * (current_base_row[339usize])))
                * (node_4096),
            (((((((current_base_row[207usize]) * (node_517))
                + ((current_base_row[209usize]) * (node_517)))
                + ((current_base_row[219usize]) * (node_517)))
                + ((current_base_row[226usize]) * (node_517)))
                + ((current_base_row[255usize]) * (node_513)))
                + ((current_base_row[258usize]) * (node_513))) * (node_4096),
            (((((((current_base_row[207usize]) * (node_521))
                + ((current_base_row[209usize]) * (node_521)))
                + ((current_base_row[219usize]) * (node_521)))
                + ((current_base_row[226usize]) * (node_521)))
                + ((current_base_row[255usize]) * (node_521)))
                + ((current_base_row[258usize]) * (node_517))) * (node_4096),
            (((next_ext_row[13usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (current_ext_row[13usize])))
                * ((challenges[11usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[7usize]))))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (next_base_row[45usize])),
            ((node_4096)
                * (((node_4210)
                    * ((challenges[3usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((((challenges[13usize]) * (next_base_row[9usize]))
                                + ((challenges[14usize]) * (next_base_row[10usize])))
                                + ((challenges[15usize]) * (next_base_row[11usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((next_base_row[8usize]) * (node_4210)),
            (next_ext_row[8usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[8usize])
                        * ((challenges[9usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((((((challenges[24usize]) * (next_base_row[7usize]))
                                    + ((challenges[25usize]) * (next_base_row[10usize])))
                                    + ((challenges[26usize]) * (next_base_row[19usize])))
                                    + ((challenges[27usize]) * (next_base_row[20usize])))
                                    + ((challenges[28usize]) * (next_base_row[21usize]))))))),
            (((((next_base_row[10usize])
                + (BFieldElement::from_raw_u64(18446743992105173011u64)))
                * ((next_base_row[10usize])
                    + (BFieldElement::from_raw_u64(18446743725817200721u64))))
                * ((next_ext_row[9usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_ext_row[9usize]))))
                + ((current_base_row[341usize]) * ((node_4334) + (node_4335))))
                + ((current_base_row[349usize])
                    * ((node_4334)
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((((((((challenges[32usize])
                                * (((node_4274) * (next_base_row[22usize]))
                                    + ((next_base_row[44usize]) * (next_base_row[39usize]))))
                                + ((challenges[33usize])
                                    * (((node_4274) * (next_base_row[23usize]))
                                        + ((next_base_row[44usize]) * (next_base_row[40usize])))))
                                + ((challenges[34usize])
                                    * (((node_4274) * (next_base_row[24usize]))
                                        + ((next_base_row[44usize]) * (next_base_row[41usize])))))
                                + ((challenges[35usize])
                                    * (((node_4274) * (next_base_row[25usize]))
                                        + ((next_base_row[44usize]) * (next_base_row[42usize])))))
                                + ((challenges[36usize])
                                    * (((node_4274) * (next_base_row[26usize]))
                                        + ((next_base_row[44usize]) * (next_base_row[43usize])))))
                                + ((challenges[37usize])
                                    * (((node_4274) * (next_base_row[39usize]))
                                        + ((next_base_row[44usize]) * (next_base_row[22usize])))))
                                + ((challenges[38usize])
                                    * (((node_4274) * (next_base_row[40usize]))
                                        + ((next_base_row[44usize]) * (next_base_row[23usize])))))
                                + ((challenges[39usize])
                                    * (((node_4274) * (next_base_row[41usize]))
                                        + ((next_base_row[44usize]) * (next_base_row[24usize])))))
                                + ((challenges[40usize])
                                    * (((node_4274) * (next_base_row[42usize]))
                                        + ((next_base_row[44usize]) * (next_base_row[25usize])))))
                                + ((challenges[41usize])
                                    * (((node_4274) * (next_base_row[43usize]))
                                        + ((next_base_row[44usize])
                                            * (next_base_row[26usize])))))))),
            ((((current_base_row[10usize])
                + (BFieldElement::from_raw_u64(18446743992105173011u64)))
                * ((current_base_row[10usize])
                    + (BFieldElement::from_raw_u64(18446743725817200721u64))))
                * ((next_ext_row[10usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_ext_row[10usize]))))
                + (((current_base_row[248usize]) + (current_base_row[295usize]))
                    * (((next_ext_row[10usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((challenges[5usize]) * (current_ext_row[10usize]))))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_846)))),
            (((((current_base_row[326usize])
                * ((next_ext_row[11usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_ext_row[11usize]))))
                + ((current_base_row[234usize]) * (node_4387)))
                + ((current_base_row[250usize])
                    * ((node_4387)
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_887)))))
                + ((current_base_row[244usize])
                    * (((node_4383)
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((challenges[31usize])
                                * (BFieldElement::from_raw_u64(146028888030u64)))))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((((((((challenges[32usize]) * (next_base_row[23usize]))
                                + ((challenges[33usize]) * (next_base_row[24usize])))
                                + ((challenges[34usize]) * (next_base_row[25usize])))
                                + ((challenges[35usize]) * (next_base_row[26usize])))
                                + ((challenges[36usize]) * (current_base_row[39usize])))
                                + ((challenges[37usize]) * (current_base_row[40usize])))
                                + ((challenges[38usize]) * (current_base_row[41usize])))
                                + ((challenges[39usize]) * (current_base_row[42usize])))
                                + ((challenges[40usize]) * (current_base_row[43usize])))
                                + ((challenges[41usize]) * (current_base_row[44usize])))))))
                + ((current_base_row[254usize]) * ((node_4387) + (node_4335))),
            (((((((((current_base_row[238usize])
                * (((node_4476) * (((node_4433) + (node_4436)) + (node_4440)))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((current_base_row[242usize]) * (node_4480)))
                + ((current_base_row[243usize]) * (node_4480)))
                + ((current_base_row[245usize])
                    * (((node_4476)
                        * (((node_4447)
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((challenges[57usize])
                                    * (BFieldElement::from_raw_u64(60129542130u64)))))
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((challenges[58usize])
                                    * (((current_base_row[22usize])
                                        + (current_base_row[23usize])) + (node_1609)))
                                    * (BFieldElement::from_raw_u64(9223372036854775808u64))))))
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)))))
                + ((current_base_row[247usize]) * (node_4480)))
                + ((current_base_row[246usize]) * (node_4484)))
                + ((current_base_row[249usize])
                    * (((((node_4476) * (node_4470)) * (node_4474))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_4470)))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_4474)))))
                + ((current_base_row[251usize]) * (node_4484)))
                + (((BFieldElement::from_raw_u64(4294967295u64))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[14usize]))) * (node_4476)),
            (((next_ext_row[14usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[14usize])
                        * ((challenges[7usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((((challenges[16usize]) * (next_base_row[46usize]))
                                    + ((challenges[17usize]) * (next_base_row[47usize])))
                                    + ((challenges[18usize]) * (next_base_row[48usize])))
                                    + ((challenges[19usize]) * (next_base_row[49usize]))))))))
                * (node_4530))
                + (((next_ext_row[14usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_ext_row[14usize]))) * (node_4547)),
            ((((((node_4556)
                * ((challenges[11usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((next_base_row[46usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (current_base_row[46usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64))) * (node_4524))
                * (node_4530)) + ((node_4556) * (node_4523)))
                + ((node_4556) * (node_4547)),
            ((node_4598)
                * ((next_ext_row[16usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((current_ext_row[16usize]) * (node_4616)))))
                + ((node_4601) * ((next_ext_row[16usize]) + (node_4621))),
            ((node_4598)
                * (((next_ext_row[17usize]) + (node_4621))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((node_4616) * (current_ext_row[17usize])))))
                + ((node_4601)
                    * ((next_ext_row[17usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_ext_row[17usize])))),
            ((node_4598)
                * (((next_ext_row[18usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((challenges[12usize]) * (current_ext_row[18usize]))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[55usize]))))
                + ((node_4601)
                    * ((next_ext_row[18usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_ext_row[18usize])))),
            ((node_4598)
                * (((next_ext_row[19usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((challenges[12usize]) * (current_ext_row[19usize]))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[56usize]))))
                + ((node_4601)
                    * ((next_ext_row[19usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_ext_row[19usize])))),
            (((next_ext_row[20usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[20usize])
                        * ((challenges[8usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((((next_base_row[50usize]) * (challenges[20usize]))
                                    + ((next_base_row[52usize]) * (challenges[21usize])))
                                    + ((next_base_row[53usize]) * (challenges[22usize])))
                                    + ((next_base_row[51usize]) * (challenges[23usize]))))))))
                * (node_4593))
                + (((next_ext_row[20usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_ext_row[20usize]))) * (node_4667)),
            (((current_ext_row[83usize]) * (node_4593)) + ((node_4676) * (node_4598)))
                + ((node_4676) * (node_4667)),
            (next_ext_row[22usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[22usize])
                        * ((challenges[9usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((((((challenges[24usize]) * (next_base_row[57usize]))
                                    + ((challenges[25usize]) * (next_base_row[58usize])))
                                    + ((challenges[26usize]) * (next_base_row[59usize])))
                                    + ((challenges[27usize]) * (next_base_row[60usize])))
                                    + ((challenges[28usize]) * (next_base_row[61usize]))))))),
            ((node_4705)
                * (((node_4739)
                    * ((challenges[11usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_4717))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_4704) * (node_4739)),
            (((current_base_row[342usize])
                * (((next_ext_row[24usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((challenges[30usize]) * (current_ext_row[24usize]))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((((((((((((((((((((challenges[29usize]) + (node_5492))
                            * (challenges[29usize])) + (node_5503))
                            * (challenges[29usize])) + (node_5514))
                            * (challenges[29usize])) + (node_5525))
                            * (challenges[29usize])) + (next_base_row[97usize]))
                            * (challenges[29usize])) + (next_base_row[98usize]))
                            * (challenges[29usize])) + (next_base_row[99usize]))
                            * (challenges[29usize])) + (next_base_row[100usize]))
                            * (challenges[29usize])) + (next_base_row[101usize]))
                            * (challenges[29usize])) + (next_base_row[102usize])))))
                + ((next_base_row[64usize]) * (node_5828)))
                + ((node_5616) * (node_5828)),
            ((current_base_row[343usize]) * (node_5616))
                * (((((((((((challenges[0usize]) + (node_4776)) * (challenges[0usize]))
                    + (node_4787)) * (challenges[0usize])) + (node_4798))
                    * (challenges[0usize])) + (node_4809)) * (challenges[0usize]))
                    + (current_base_row[97usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (challenges[62usize]))),
            (current_base_row[357usize])
                * ((((((node_5659) + (node_5660)) + (node_5662)) + (node_5664))
                    + (node_5666)) + (node_5668)),
            (current_base_row[358usize])
                * (((((((((((((((((challenges[32usize])
                    * ((node_5492)
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_4776))))
                    + ((challenges[33usize])
                        * ((node_5503)
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (node_4787)))))
                    + ((challenges[34usize])
                        * ((node_5514)
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (node_4798)))))
                    + ((challenges[35usize])
                        * ((node_5525)
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (node_4809)))))
                    + ((challenges[36usize])
                        * ((next_base_row[97usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (current_base_row[97usize])))))
                    + ((challenges[37usize])
                        * ((next_base_row[98usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (current_base_row[98usize])))))
                    + ((challenges[38usize])
                        * ((next_base_row[99usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (current_base_row[99usize])))))
                    + ((challenges[39usize])
                        * ((next_base_row[100usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (current_base_row[100usize])))))
                    + ((challenges[40usize])
                        * ((next_base_row[101usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (current_base_row[101usize])))))
                    + ((challenges[41usize])
                        * ((next_base_row[102usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (current_base_row[102usize]))))) + (node_5659))
                    + (node_5660)) + (node_5662)) + (node_5664)) + (node_5666))
                    + (node_5668)),
            ((((current_base_row[315usize]) * (current_base_row[318usize]))
                * (((next_ext_row[25usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((challenges[4usize]) * (current_ext_row[25usize]))))
                    + (node_5758))) + ((next_base_row[64usize]) * (node_5735)))
                + ((node_5621) * (node_5735)),
            ((((node_5778) * (current_base_row[318usize]))
                * (((next_ext_row[26usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((challenges[5usize]) * (current_ext_row[26usize]))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (node_5744)))) + ((node_5677) * (node_5769)))
                + ((node_5621) * (node_5769)),
            ((((current_base_row[315usize]) * (node_5728))
                * ((((next_ext_row[27usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((challenges[6usize]) * (current_ext_row[27usize]))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((challenges[31usize]) * (next_base_row[63usize]))))
                    + (node_5758))) + ((next_base_row[64usize]) * (node_5795)))
                + ((((node_5625) * (node_5730)) * (node_5798)) * (node_5795)),
            (((current_base_row[282usize])
                * (((node_5847)
                    * ((challenges[48usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[49usize]) * (next_base_row[65usize]))
                                + ((challenges[50usize]) * (next_base_row[81usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_5778) * (node_5847))) + ((node_5855) * (node_5847)),
            (((current_base_row[282usize])
                * (((node_5868)
                    * ((challenges[48usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[49usize]) * (next_base_row[66usize]))
                                + ((challenges[50usize]) * (next_base_row[82usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_5778) * (node_5868))) + ((node_5855) * (node_5868)),
            (((current_base_row[282usize])
                * (((node_5885)
                    * ((challenges[48usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[49usize]) * (next_base_row[67usize]))
                                + ((challenges[50usize]) * (next_base_row[83usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_5778) * (node_5885))) + ((node_5855) * (node_5885)),
            (((current_base_row[282usize])
                * (((node_5902)
                    * ((challenges[48usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[49usize]) * (next_base_row[68usize]))
                                + ((challenges[50usize]) * (next_base_row[84usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_5778) * (node_5902))) + ((node_5855) * (node_5902)),
            (((current_base_row[282usize])
                * (((node_5919)
                    * ((challenges[48usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[49usize]) * (next_base_row[69usize]))
                                + ((challenges[50usize]) * (next_base_row[85usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_5778) * (node_5919))) + ((node_5855) * (node_5919)),
            (((current_base_row[282usize])
                * (((node_5936)
                    * ((challenges[48usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[49usize]) * (next_base_row[70usize]))
                                + ((challenges[50usize]) * (next_base_row[86usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_5778) * (node_5936))) + ((node_5855) * (node_5936)),
            (((current_base_row[282usize])
                * (((node_5953)
                    * ((challenges[48usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[49usize]) * (next_base_row[71usize]))
                                + ((challenges[50usize]) * (next_base_row[87usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_5778) * (node_5953))) + ((node_5855) * (node_5953)),
            (((current_base_row[282usize])
                * (((node_5970)
                    * ((challenges[48usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[49usize]) * (next_base_row[72usize]))
                                + ((challenges[50usize]) * (next_base_row[88usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_5778) * (node_5970))) + ((node_5855) * (node_5970)),
            (((current_base_row[282usize])
                * (((node_5987)
                    * ((challenges[48usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[49usize]) * (next_base_row[73usize]))
                                + ((challenges[50usize]) * (next_base_row[89usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_5778) * (node_5987))) + ((node_5855) * (node_5987)),
            (((current_base_row[282usize])
                * (((node_6004)
                    * ((challenges[48usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[49usize]) * (next_base_row[74usize]))
                                + ((challenges[50usize]) * (next_base_row[90usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_5778) * (node_6004))) + ((node_5855) * (node_6004)),
            (((current_base_row[282usize])
                * (((node_6021)
                    * ((challenges[48usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[49usize]) * (next_base_row[75usize]))
                                + ((challenges[50usize]) * (next_base_row[91usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_5778) * (node_6021))) + ((node_5855) * (node_6021)),
            (((current_base_row[282usize])
                * (((node_6038)
                    * ((challenges[48usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[49usize]) * (next_base_row[76usize]))
                                + ((challenges[50usize]) * (next_base_row[92usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_5778) * (node_6038))) + ((node_5855) * (node_6038)),
            (((current_base_row[282usize])
                * (((node_6055)
                    * ((challenges[48usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[49usize]) * (next_base_row[77usize]))
                                + ((challenges[50usize]) * (next_base_row[93usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_5778) * (node_6055))) + ((node_5855) * (node_6055)),
            (((current_base_row[282usize])
                * (((node_6072)
                    * ((challenges[48usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[49usize]) * (next_base_row[78usize]))
                                + ((challenges[50usize]) * (next_base_row[94usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_5778) * (node_6072))) + ((node_5855) * (node_6072)),
            (((current_base_row[282usize])
                * (((node_6089)
                    * ((challenges[48usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[49usize]) * (next_base_row[79usize]))
                                + ((challenges[50usize]) * (next_base_row[95usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_5778) * (node_6089))) + ((node_5855) * (node_6089)),
            (((current_base_row[282usize])
                * (((node_6106)
                    * ((challenges[48usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[49usize]) * (next_base_row[80usize]))
                                + ((challenges[50usize]) * (next_base_row[96usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_5778) * (node_6106))) + ((node_5855) * (node_6106)),
            ((node_6132)
                * (((node_6142)
                    * ((challenges[48usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[49usize])
                                * (((BFieldElement::from_raw_u64(1099511627520u64))
                                    * (next_base_row[130usize])) + (next_base_row[131usize])))
                                + ((challenges[50usize])
                                    * (((BFieldElement::from_raw_u64(1099511627520u64))
                                        * (next_base_row[132usize]))
                                        + (next_base_row[133usize])))))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[134usize]))))
                + ((next_base_row[129usize]) * (node_6142)),
            ((node_6132)
                * ((((((node_6158)
                    * ((challenges[51usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_6153))))
                    * ((challenges[51usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_6156))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((BFieldElement::from_raw_u64(8589934590u64))
                            * (challenges[51usize])))) + (node_6153)) + (node_6156)))
                + ((next_base_row[129usize]) * (node_6158)),
            ((node_6184)
                * (((node_6196)
                    * ((challenges[51usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((next_base_row[136usize]) * (challenges[52usize]))
                                + ((next_base_row[137usize]) * (challenges[53usize]))))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[138usize]))))
                + ((next_base_row[135usize]) * (node_6196)),
            ((node_6184)
                * (((next_ext_row[47usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((current_ext_row[47usize]) * (challenges[54usize]))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[137usize]))))
                + ((next_base_row[135usize])
                    * ((next_ext_row[47usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_ext_row[47usize])))),
            (node_6244) * (node_6360),
            (next_base_row[139usize])
                * (((node_6360)
                    * ((challenges[10usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((challenges[57usize]) * (next_base_row[142usize]))
                                + ((challenges[55usize]) * (next_base_row[143usize])))
                                + ((challenges[56usize]) * (next_base_row[145usize])))
                                + ((challenges[58usize]) * (next_base_row[147usize]))))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[148usize]))),
            (current_ext_row[50usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_176)
                        * ((challenges[7usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_167)
                                    + ((challenges[18usize])
                                        * ((next_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(4294967295u64)))))
                                    + ((challenges[19usize]) * (next_base_row[36usize]))))))),
            (current_ext_row[51usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_547)
                        * ((challenges[7usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_167)
                                    + ((challenges[18usize])
                                        * ((current_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(4294967295u64)))))
                                    + ((challenges[19usize])
                                        * (current_base_row[36usize]))))))),
            (current_ext_row[52usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[50usize])
                        * ((challenges[7usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_167)
                                    + ((challenges[18usize])
                                        * ((next_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(8589934590u64)))))
                                    + ((challenges[19usize]) * (next_base_row[35usize]))))))),
            (current_ext_row[53usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[51usize])
                        * ((challenges[7usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_167)
                                    + ((challenges[18usize])
                                        * ((current_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(8589934590u64)))))
                                    + ((challenges[19usize])
                                        * (current_base_row[35usize]))))))),
            (current_ext_row[54usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[52usize])
                        * ((challenges[7usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_167)
                                    + ((challenges[18usize])
                                        * ((next_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(12884901885u64)))))
                                    + ((challenges[19usize]) * (next_base_row[34usize]))))))),
            (current_ext_row[55usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[53usize])
                        * ((challenges[7usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_167)
                                    + ((challenges[18usize])
                                        * ((current_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(12884901885u64)))))
                                    + ((challenges[19usize])
                                        * (current_base_row[34usize]))))))),
            (current_ext_row[56usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[54usize])
                        * ((challenges[7usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_167)
                                    + ((challenges[18usize])
                                        * ((next_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(17179869180u64)))))
                                    + ((challenges[19usize]) * (next_base_row[33usize]))))))),
            (current_ext_row[57usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_1326)
                        * ((challenges[8usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_1317)
                                    + (((next_base_row[22usize])
                                        + (BFieldElement::from_raw_u64(8589934590u64)))
                                        * (challenges[21usize])))
                                    + ((next_base_row[24usize]) * (challenges[22usize]))))))),
            (current_ext_row[58usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_1402)
                        * ((challenges[8usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_1315) + (node_1410))
                                    + ((current_base_row[24usize])
                                        * (challenges[22usize]))))))),
            (current_ext_row[59usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[6usize]) * (current_ext_row[56usize]))),
            (current_ext_row[60usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[55usize])
                        * ((challenges[7usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_167)
                                    + ((challenges[18usize])
                                        * ((current_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(17179869180u64)))))
                                    + ((challenges[19usize])
                                        * (current_base_row[33usize]))))))),
            (current_ext_row[61usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[57usize])
                        * ((challenges[8usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_1317)
                                    + (((next_base_row[22usize])
                                        + (BFieldElement::from_raw_u64(12884901885u64)))
                                        * (challenges[21usize])))
                                    + ((next_base_row[25usize]) * (challenges[22usize]))))))),
            (current_ext_row[62usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[58usize])
                        * ((challenges[8usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_1315) + (node_1424))
                                    + ((current_base_row[25usize])
                                        * (challenges[22usize]))))))),
            (current_ext_row[63usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[61usize])
                        * ((challenges[8usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_1317)
                                    + (((next_base_row[22usize])
                                        + (BFieldElement::from_raw_u64(17179869180u64)))
                                        * (challenges[21usize])))
                                    + ((next_base_row[26usize]) * (challenges[22usize]))))))),
            (current_ext_row[64usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[62usize])
                        * ((challenges[8usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_1315)
                                    + (((current_base_row[22usize])
                                        + (BFieldElement::from_raw_u64(12884901885u64)))
                                        * (challenges[21usize])))
                                    + ((current_base_row[26usize])
                                        * (challenges[22usize]))))))),
            (current_ext_row[65usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[190usize])
                        * ((next_ext_row[7usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((current_ext_row[7usize])
                                    * ((current_ext_row[63usize])
                                        * ((challenges[8usize])
                                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                                * (((node_1317)
                                                    + (((next_base_row[22usize])
                                                        + (BFieldElement::from_raw_u64(21474836475u64)))
                                                        * (challenges[21usize])))
                                                    + ((next_base_row[27usize])
                                                        * (challenges[22usize]))))))))))),
            (current_ext_row[66usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_ext_row[56usize])
                        * ((challenges[7usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_167)
                                    + ((challenges[18usize])
                                        * ((next_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(21474836475u64)))))
                                    + ((challenges[19usize]) * (next_base_row[32usize]))))))
                        * ((challenges[7usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_167)
                                    + ((challenges[18usize])
                                        * ((next_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(25769803770u64)))))
                                    + ((challenges[19usize]) * (next_base_row[31usize]))))))
                        * ((challenges[7usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_167)
                                    + ((challenges[18usize])
                                        * ((next_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(30064771065u64)))))
                                    + ((challenges[19usize]) * (next_base_row[30usize]))))))),
            (current_ext_row[67usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[190usize])
                        * ((next_ext_row[7usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((current_ext_row[7usize])
                                    * ((current_ext_row[64usize])
                                        * ((challenges[8usize])
                                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                                * (((node_1315)
                                                    + (((current_base_row[22usize])
                                                        + (BFieldElement::from_raw_u64(17179869180u64)))
                                                        * (challenges[21usize])))
                                                    + ((current_base_row[27usize])
                                                        * (challenges[22usize]))))))))))),
            (current_ext_row[68usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_ext_row[60usize])
                        * ((challenges[7usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_167)
                                    + ((challenges[18usize])
                                        * ((current_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(21474836475u64)))))
                                    + ((challenges[19usize]) * (current_base_row[32usize]))))))
                        * ((challenges[7usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_167)
                                    + ((challenges[18usize])
                                        * ((current_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(25769803770u64)))))
                                    + ((challenges[19usize]) * (current_base_row[31usize]))))))
                        * ((challenges[7usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_167)
                                    + ((challenges[18usize])
                                        * ((current_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(30064771065u64)))))
                                    + ((challenges[19usize])
                                        * (current_base_row[30usize]))))))),
            (current_ext_row[69usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[6usize])
                        * (((current_ext_row[66usize])
                            * ((challenges[7usize])
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (((node_167)
                                        + ((challenges[18usize])
                                            * ((next_base_row[38usize])
                                                + (BFieldElement::from_raw_u64(34359738360u64)))))
                                        + ((challenges[19usize]) * (next_base_row[29usize]))))))
                            * ((challenges[7usize])
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (((node_167)
                                        + ((challenges[18usize])
                                            * ((next_base_row[38usize])
                                                + (BFieldElement::from_raw_u64(38654705655u64)))))
                                        + ((challenges[19usize]) * (next_base_row[28usize])))))))),
            (current_ext_row[70usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[6usize])
                        * (((current_ext_row[68usize])
                            * ((challenges[7usize])
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (((node_167)
                                        + ((challenges[18usize])
                                            * ((current_base_row[38usize])
                                                + (BFieldElement::from_raw_u64(34359738360u64)))))
                                        + ((challenges[19usize]) * (current_base_row[29usize]))))))
                            * ((challenges[7usize])
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (((node_167)
                                        + ((challenges[18usize])
                                            * ((current_base_row[38usize])
                                                + (BFieldElement::from_raw_u64(38654705655u64)))))
                                        + ((challenges[19usize])
                                            * (current_base_row[28usize])))))))),
            (current_ext_row[71usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((node_1750)
                        * ((challenges[8usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_1317) + (node_1410)) + (node_1752)))))
                        * ((challenges[8usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_1317) + (node_1424)) + (node_1758)))))
                        * ((challenges[8usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((node_1764) + (node_1765)))))),
            (current_ext_row[72usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((node_1750)
                        * ((challenges[8usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((node_1764) + (node_1752)))))
                        * ((challenges[8usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((node_1771) + (node_1758)))))
                        * ((challenges[8usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((node_1778) + (node_1765)))))),
            (current_ext_row[73usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[185usize]) * (node_181))),
            (current_ext_row[74usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[186usize])
                        * ((next_ext_row[6usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((current_ext_row[6usize])
                                    * (current_ext_row[50usize])))))),
            (current_ext_row[75usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[187usize]) * (node_325))),
            (current_ext_row[76usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[189usize])
                        * ((next_ext_row[6usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((current_ext_row[6usize])
                                    * (current_ext_row[54usize])))))),
            (current_ext_row[77usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[185usize]) * (node_550))),
            (current_ext_row[78usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[186usize])
                        * ((next_ext_row[6usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((current_ext_row[6usize])
                                    * (current_ext_row[51usize])))))),
            (current_ext_row[79usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[187usize])
                        * ((next_ext_row[6usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((current_ext_row[6usize])
                                    * (current_ext_row[53usize])))))),
            (current_ext_row[80usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[189usize])
                        * ((next_ext_row[6usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((current_ext_row[6usize])
                                    * (current_ext_row[55usize])))))),
            (current_ext_row[81usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[190usize])
                        * ((next_ext_row[6usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((current_ext_row[6usize])
                                    * (current_ext_row[60usize])))))),
            (current_ext_row[82usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[7usize])
                        * (((current_ext_row[71usize])
                            * ((challenges[8usize])
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * ((node_1771)
                                        + ((current_base_row[43usize]) * (challenges[22usize]))))))
                            * ((challenges[8usize])
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * ((node_1778)
                                        + ((current_base_row[44usize])
                                            * (challenges[22usize])))))))),
            (current_ext_row[83usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((node_4676)
                        * ((challenges[11usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((next_base_row[50usize])
                                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                        * (current_base_row[50usize]))))))
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                        * (node_4601))),
            (current_ext_row[84usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[219usize])
                        * ((((((current_base_row[185usize])
                            * ((next_ext_row[7usize])
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * ((current_ext_row[7usize]) * (node_1326)))))
                            + ((current_base_row[186usize])
                                * ((next_ext_row[7usize])
                                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                        * ((current_ext_row[7usize])
                                            * (current_ext_row[57usize]))))))
                            + ((current_base_row[187usize])
                                * ((next_ext_row[7usize])
                                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                        * ((current_ext_row[7usize])
                                            * (current_ext_row[61usize]))))))
                            + ((current_base_row[189usize])
                                * ((next_ext_row[7usize])
                                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                        * ((current_ext_row[7usize])
                                            * (current_ext_row[63usize]))))))
                            + (current_ext_row[65usize])))),
            (current_ext_row[85usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[226usize])
                        * ((((((current_base_row[185usize])
                            * ((next_ext_row[7usize])
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * ((current_ext_row[7usize]) * (node_1402)))))
                            + ((current_base_row[186usize])
                                * ((next_ext_row[7usize])
                                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                        * ((current_ext_row[7usize])
                                            * (current_ext_row[58usize]))))))
                            + ((current_base_row[187usize])
                                * ((next_ext_row[7usize])
                                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                        * ((current_ext_row[7usize])
                                            * (current_ext_row[62usize]))))))
                            + ((current_base_row[189usize])
                                * ((next_ext_row[7usize])
                                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                        * ((current_ext_row[7usize])
                                            * (current_ext_row[64usize]))))))
                            + (current_ext_row[67usize])))),
        ];
        base_constraints
            .into_iter()
            .map(|bfe| bfe.lift())
            .chain(ext_constraints)
            .collect()
    }
    #[allow(unused_variables)]
    fn evaluate_terminal_constraints(
        base_row: ArrayView1<BFieldElement>,
        ext_row: ArrayView1<XFieldElement>,
        challenges: &Challenges,
    ) -> Vec<XFieldElement> {
        let base_constraints = [
            (base_row[5usize]) + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((base_row[3usize]) + (BFieldElement::from_raw_u64(18446744030759878666u64)))
                * ((base_row[6usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            base_row[10usize],
            ((base_row[62usize])
                * ((base_row[63usize])
                    + (BFieldElement::from_raw_u64(18446743897615892521u64))))
                * ((base_row[64usize])
                    + (BFieldElement::from_raw_u64(18446744047939747846u64))),
            (base_row[143usize])
                * ((base_row[142usize])
                    + (BFieldElement::from_raw_u64(18446743940565565471u64))),
            base_row[145usize],
        ];
        let ext_constraints = [
            (((ext_row[18usize]) * (ext_row[16usize]))
                + ((ext_row[19usize]) * (ext_row[17usize])))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((((base_row[62usize])
                + (BFieldElement::from_raw_u64(18446744060824649731u64)))
                * ((base_row[62usize])
                    + (BFieldElement::from_raw_u64(18446744056529682436u64))))
                * (base_row[62usize]))
                * (((((((((((challenges[0usize])
                    + ((((((base_row[65usize])
                        * (BFieldElement::from_raw_u64(18446744069414518785u64)))
                        + ((base_row[66usize])
                            * (BFieldElement::from_raw_u64(18446744069414584320u64))))
                        + ((base_row[67usize])
                            * (BFieldElement::from_raw_u64(281474976645120u64))))
                        + (base_row[68usize])) * (BFieldElement::from_raw_u64(1u64))))
                    * (challenges[0usize]))
                    + ((((((base_row[69usize])
                        * (BFieldElement::from_raw_u64(18446744069414518785u64)))
                        + ((base_row[70usize])
                            * (BFieldElement::from_raw_u64(18446744069414584320u64))))
                        + ((base_row[71usize])
                            * (BFieldElement::from_raw_u64(281474976645120u64))))
                        + (base_row[72usize])) * (BFieldElement::from_raw_u64(1u64))))
                    * (challenges[0usize]))
                    + ((((((base_row[73usize])
                        * (BFieldElement::from_raw_u64(18446744069414518785u64)))
                        + ((base_row[74usize])
                            * (BFieldElement::from_raw_u64(18446744069414584320u64))))
                        + ((base_row[75usize])
                            * (BFieldElement::from_raw_u64(281474976645120u64))))
                        + (base_row[76usize])) * (BFieldElement::from_raw_u64(1u64))))
                    * (challenges[0usize]))
                    + ((((((base_row[77usize])
                        * (BFieldElement::from_raw_u64(18446744069414518785u64)))
                        + ((base_row[78usize])
                            * (BFieldElement::from_raw_u64(18446744069414584320u64))))
                        + ((base_row[79usize])
                            * (BFieldElement::from_raw_u64(281474976645120u64))))
                        + (base_row[80usize])) * (BFieldElement::from_raw_u64(1u64))))
                    * (challenges[0usize])) + (base_row[97usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (challenges[62usize]))),
            (ext_row[47usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (challenges[61usize])),
            (ext_row[2usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[24usize])),
            (challenges[59usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[3usize])),
            (ext_row[4usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (challenges[60usize])),
            (ext_row[5usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[0usize])),
            (ext_row[6usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[14usize])),
            (ext_row[7usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[20usize])),
            (ext_row[8usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[22usize])),
            (ext_row[9usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[25usize])),
            (ext_row[26usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[10usize])),
            (ext_row[11usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[27usize])),
            ((((((((((((((((ext_row[44usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[28usize])))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[29usize])))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[30usize])))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[31usize])))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[32usize])))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[33usize])))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[34usize])))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[35usize])))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[36usize])))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[37usize])))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[38usize])))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[39usize])))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[40usize])))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[41usize])))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[42usize])))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[43usize])),
            (ext_row[45usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[46usize])),
            (ext_row[12usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[48usize])),
            (((ext_row[13usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[15usize])))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[21usize])))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[23usize])),
        ];
        base_constraints
            .into_iter()
            .map(|bfe| bfe.lift())
            .chain(ext_constraints)
            .collect()
    }
}
impl Evaluable<XFieldElement> for MasterExtTable {
    #[allow(unused_variables)]
    fn evaluate_initial_constraints(
        base_row: ArrayView1<XFieldElement>,
        ext_row: ArrayView1<XFieldElement>,
        challenges: &Challenges,
    ) -> Vec<XFieldElement> {
        let node_468 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (base_row[129usize]));
        let node_474 = ((challenges[52usize]) * (base_row[131usize]))
            + ((challenges[53usize]) * (base_row[133usize]));
        let node_477 = ((challenges[52usize]) * (base_row[130usize]))
            + ((challenges[53usize]) * (base_row[132usize]));
        let base_constraints = [
            base_row[0usize],
            base_row[3usize],
            base_row[5usize],
            base_row[7usize],
            base_row[9usize],
            base_row[19usize],
            base_row[20usize],
            base_row[21usize],
            base_row[22usize],
            base_row[23usize],
            base_row[24usize],
            base_row[25usize],
            base_row[26usize],
            base_row[27usize],
            base_row[28usize],
            base_row[29usize],
            base_row[30usize],
            base_row[31usize],
            base_row[32usize],
            (base_row[38usize]) + (BFieldElement::from_raw_u64(18446744000695107601u64)),
            (base_row[48usize]) + (BFieldElement::from_raw_u64(18446744000695107601u64)),
            base_row[55usize],
            base_row[57usize],
            base_row[59usize],
            base_row[60usize],
            base_row[61usize],
            (base_row[62usize]) + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            base_row[64usize],
            base_row[136usize],
            (base_row[149usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((((base_row[12usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                        * (base_row[13usize]))
                        * ((base_row[14usize])
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                        * ((base_row[15usize])
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))))),
            (base_row[150usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((base_row[149usize]) * (base_row[16usize]))
                        * ((base_row[17usize])
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                        * ((base_row[18usize])
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))))),
        ];
        let ext_constraints = [
            ext_row[0usize],
            ((ext_row[1usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (challenges[29usize])))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (base_row[1usize])),
            (ext_row[2usize]) + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((((((((((challenges[0usize]) + (base_row[33usize])) * (challenges[0usize]))
                + (base_row[34usize])) * (challenges[0usize])) + (base_row[35usize]))
                * (challenges[0usize])) + (base_row[36usize])) * (challenges[0usize]))
                + (base_row[37usize]))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (challenges[62usize])),
            (ext_row[3usize]) + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[5usize])
                * ((challenges[3usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[14usize]) * (base_row[10usize]))
                            + ((challenges[15usize]) * (base_row[11usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            (ext_row[4usize]) + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            (ext_row[6usize]) + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            (ext_row[7usize]) + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            (ext_row[8usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((challenges[9usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((challenges[25usize]) * (base_row[10usize]))))),
            ((ext_row[13usize]) * (challenges[11usize]))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (base_row[45usize])),
            (((base_row[10usize])
                + (BFieldElement::from_raw_u64(18446743992105173011u64)))
                * ((ext_row[9usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((base_row[150usize])
                    * ((ext_row[9usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (challenges[4usize])))),
            (ext_row[10usize]) + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            (ext_row[11usize]) + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ext_row[12usize],
            (((ext_row[14usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((challenges[7usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((challenges[16usize]) * (base_row[46usize]))
                                + ((challenges[17usize]) * (base_row[47usize])))
                                + ((challenges[18usize])
                                    * (BFieldElement::from_raw_u64(68719476720u64))))
                                + ((challenges[19usize]) * (base_row[49usize])))))))
                * ((base_row[47usize])
                    + (BFieldElement::from_raw_u64(18446744060824649731u64))))
                + (((ext_row[14usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                    * ((base_row[47usize])
                        * ((base_row[47usize])
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))))),
            ext_row[15usize],
            ext_row[18usize],
            (ext_row[19usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (base_row[56usize])),
            ((ext_row[16usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (challenges[12usize]))) + (base_row[52usize]),
            (ext_row[17usize]) + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((((ext_row[20usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (challenges[8usize])))
                + (((((base_row[50usize]) * (challenges[20usize]))
                    + ((base_row[51usize]) * (challenges[23usize])))
                    + ((base_row[52usize]) * (challenges[21usize])))
                    + ((base_row[53usize]) * (challenges[22usize]))))
                * ((base_row[51usize])
                    + (BFieldElement::from_raw_u64(18446744060824649731u64))))
                + (((ext_row[20usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                    * (((base_row[51usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                        * (base_row[51usize]))),
            ext_row[21usize],
            (ext_row[22usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((challenges[9usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((challenges[25usize]) * (base_row[58usize]))))),
            ext_row[23usize],
            (ext_row[25usize]) + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            (ext_row[26usize]) + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            (ext_row[27usize]) + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[24usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (challenges[30usize])))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((((((((((((((((((challenges[29usize])
                        + ((((((base_row[65usize])
                            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
                            + ((base_row[66usize])
                                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
                            + ((base_row[67usize])
                                * (BFieldElement::from_raw_u64(281474976645120u64))))
                            + (base_row[68usize]))
                            * (BFieldElement::from_raw_u64(1u64))))
                        * (challenges[29usize]))
                        + ((((((base_row[69usize])
                            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
                            + ((base_row[70usize])
                                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
                            + ((base_row[71usize])
                                * (BFieldElement::from_raw_u64(281474976645120u64))))
                            + (base_row[72usize]))
                            * (BFieldElement::from_raw_u64(1u64))))
                        * (challenges[29usize]))
                        + ((((((base_row[73usize])
                            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
                            + ((base_row[74usize])
                                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
                            + ((base_row[75usize])
                                * (BFieldElement::from_raw_u64(281474976645120u64))))
                            + (base_row[76usize]))
                            * (BFieldElement::from_raw_u64(1u64))))
                        * (challenges[29usize]))
                        + ((((((base_row[77usize])
                            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
                            + ((base_row[78usize])
                                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
                            + ((base_row[79usize])
                                * (BFieldElement::from_raw_u64(281474976645120u64))))
                            + (base_row[80usize]))
                            * (BFieldElement::from_raw_u64(1u64))))
                        * (challenges[29usize])) + (base_row[97usize]))
                        * (challenges[29usize])) + (base_row[98usize]))
                        * (challenges[29usize])) + (base_row[99usize]))
                        * (challenges[29usize])) + (base_row[100usize]))
                        * (challenges[29usize])) + (base_row[101usize]))
                        * (challenges[29usize])) + (base_row[102usize]))),
            ((ext_row[28usize])
                * ((challenges[48usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[49usize]) * (base_row[65usize]))
                            + ((challenges[50usize]) * (base_row[81usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[29usize])
                * ((challenges[48usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[49usize]) * (base_row[66usize]))
                            + ((challenges[50usize]) * (base_row[82usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[30usize])
                * ((challenges[48usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[49usize]) * (base_row[67usize]))
                            + ((challenges[50usize]) * (base_row[83usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[31usize])
                * ((challenges[48usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[49usize]) * (base_row[68usize]))
                            + ((challenges[50usize]) * (base_row[84usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[32usize])
                * ((challenges[48usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[49usize]) * (base_row[69usize]))
                            + ((challenges[50usize]) * (base_row[85usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[33usize])
                * ((challenges[48usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[49usize]) * (base_row[70usize]))
                            + ((challenges[50usize]) * (base_row[86usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[34usize])
                * ((challenges[48usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[49usize]) * (base_row[71usize]))
                            + ((challenges[50usize]) * (base_row[87usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[35usize])
                * ((challenges[48usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[49usize]) * (base_row[72usize]))
                            + ((challenges[50usize]) * (base_row[88usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[36usize])
                * ((challenges[48usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[49usize]) * (base_row[73usize]))
                            + ((challenges[50usize]) * (base_row[89usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[37usize])
                * ((challenges[48usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[49usize]) * (base_row[74usize]))
                            + ((challenges[50usize]) * (base_row[90usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[38usize])
                * ((challenges[48usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[49usize]) * (base_row[75usize]))
                            + ((challenges[50usize]) * (base_row[91usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[39usize])
                * ((challenges[48usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[49usize]) * (base_row[76usize]))
                            + ((challenges[50usize]) * (base_row[92usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[40usize])
                * ((challenges[48usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[49usize]) * (base_row[77usize]))
                            + ((challenges[50usize]) * (base_row[93usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[41usize])
                * ((challenges[48usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[49usize]) * (base_row[78usize]))
                            + ((challenges[50usize]) * (base_row[94usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[42usize])
                * ((challenges[48usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[49usize]) * (base_row[79usize]))
                            + ((challenges[50usize]) * (base_row[95usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[43usize])
                * ((challenges[48usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[49usize]) * (base_row[80usize]))
                            + ((challenges[50usize]) * (base_row[96usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((node_468)
                * (((ext_row[44usize])
                    * ((challenges[48usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[49usize])
                                * (((BFieldElement::from_raw_u64(1099511627520u64))
                                    * (base_row[130usize])) + (base_row[131usize])))
                                + ((challenges[50usize])
                                    * (((BFieldElement::from_raw_u64(1099511627520u64))
                                        * (base_row[132usize])) + (base_row[133usize])))))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (base_row[134usize]))))
                + ((base_row[129usize]) * (ext_row[44usize])),
            ((node_468)
                * ((((((ext_row[45usize])
                    * ((challenges[51usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_474))))
                    * ((challenges[51usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_477))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((BFieldElement::from_raw_u64(8589934590u64))
                            * (challenges[51usize])))) + (node_474)) + (node_477)))
                + ((base_row[129usize]) * (ext_row[45usize])),
            ((ext_row[46usize])
                * ((challenges[51usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((base_row[137usize]) * (challenges[53usize])))))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (base_row[138usize])),
            ((ext_row[47usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (challenges[54usize])))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (base_row[137usize])),
            (((base_row[139usize])
                + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                * (ext_row[48usize]))
                + ((base_row[139usize])
                    * (((ext_row[48usize])
                        * ((challenges[10usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((((challenges[55usize]) * (base_row[143usize]))
                                    + ((challenges[56usize]) * (base_row[145usize])))
                                    + ((challenges[57usize]) * (base_row[142usize])))
                                    + ((challenges[58usize]) * (base_row[147usize]))))))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (base_row[148usize])))),
        ];
        base_constraints.into_iter().chain(ext_constraints).collect()
    }
    #[allow(unused_variables)]
    fn evaluate_consistency_constraints(
        base_row: ArrayView1<XFieldElement>,
        ext_row: ArrayView1<XFieldElement>,
        challenges: &Challenges,
    ) -> Vec<XFieldElement> {
        let node_102 = (base_row[152usize])
            * ((base_row[64usize])
                + (BFieldElement::from_raw_u64(18446744047939747846u64)));
        let node_221 = (base_row[153usize])
            * ((base_row[64usize])
                + (BFieldElement::from_raw_u64(18446744047939747846u64)));
        let node_238 = ((base_row[154usize])
            * ((base_row[64usize])
                + (BFieldElement::from_raw_u64(18446744052234715141u64))))
            * ((base_row[64usize])
                + (BFieldElement::from_raw_u64(18446744047939747846u64)));
        let node_245 = ((base_row[154usize])
            * ((base_row[64usize])
                + (BFieldElement::from_raw_u64(18446744056529682436u64))))
            * ((base_row[64usize])
                + (BFieldElement::from_raw_u64(18446744047939747846u64)));
        let node_655 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (base_row[157usize]));
        let node_114 = (((base_row[63usize])
            + (BFieldElement::from_raw_u64(18446743992105173011u64)))
            * ((base_row[63usize])
                + (BFieldElement::from_raw_u64(18446743923385696291u64))))
            * ((base_row[63usize])
                + (BFieldElement::from_raw_u64(18446743828896415801u64)));
        let node_116 = (node_102) * (base_row[161usize]);
        let node_660 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (base_row[160usize]));
        let node_101 = (base_row[64usize])
            + (BFieldElement::from_raw_u64(18446744047939747846u64));
        let node_678 = (base_row[142usize])
            + (BFieldElement::from_raw_u64(18446743949155500061u64));
        let node_674 = (base_row[142usize])
            + (BFieldElement::from_raw_u64(18446743940565565471u64));
        let node_94 = (base_row[64usize])
            + (BFieldElement::from_raw_u64(18446744056529682436u64));
        let node_97 = (base_row[64usize])
            + (BFieldElement::from_raw_u64(18446744052234715141u64));
        let node_153 = ((((BFieldElement::from_raw_u64(18446744065119617025u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((base_row[65usize])
                    * (BFieldElement::from_raw_u64(281474976645120u64)))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (base_row[66usize]))) * (base_row[109usize]))
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_155 = ((((BFieldElement::from_raw_u64(18446744065119617025u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((base_row[69usize])
                    * (BFieldElement::from_raw_u64(281474976645120u64)))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (base_row[70usize]))) * (base_row[110usize]))
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_157 = ((((BFieldElement::from_raw_u64(18446744065119617025u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((base_row[73usize])
                    * (BFieldElement::from_raw_u64(281474976645120u64)))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (base_row[74usize]))) * (base_row[111usize]))
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_159 = ((((BFieldElement::from_raw_u64(18446744065119617025u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((base_row[77usize])
                    * (BFieldElement::from_raw_u64(281474976645120u64)))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (base_row[78usize]))) * (base_row[112usize]))
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_680 = (base_row[139usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_90 = (base_row[64usize])
            + (BFieldElement::from_raw_u64(18446744060824649731u64));
        let node_670 = (base_row[142usize])
            + (BFieldElement::from_raw_u64(18446744017874976781u64));
        let node_11 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (((BFieldElement::from_raw_u64(38654705655u64))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (base_row[3usize]))) * (base_row[4usize])));
        let node_8 = (BFieldElement::from_raw_u64(38654705655u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (base_row[3usize]));
        let node_104 = (((base_row[62usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64)))
            * ((base_row[62usize])
                + (BFieldElement::from_raw_u64(18446744060824649731u64))))
            * ((base_row[62usize])
                + (BFieldElement::from_raw_u64(18446744056529682436u64)));
        let node_85 = (base_row[62usize])
            + (BFieldElement::from_raw_u64(18446744060824649731u64));
        let node_73 = (base_row[63usize])
            + (BFieldElement::from_raw_u64(18446743992105173011u64));
        let node_79 = (base_row[63usize])
            + (BFieldElement::from_raw_u64(18446743923385696291u64));
        let node_82 = (base_row[63usize])
            + (BFieldElement::from_raw_u64(18446743828896415801u64));
        let node_126 = ((BFieldElement::from_raw_u64(18446744065119617025u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((base_row[65usize])
                    * (BFieldElement::from_raw_u64(281474976645120u64)))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (base_row[66usize]));
        let node_133 = ((BFieldElement::from_raw_u64(18446744065119617025u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((base_row[69usize])
                    * (BFieldElement::from_raw_u64(281474976645120u64)))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (base_row[70usize]));
        let node_140 = ((BFieldElement::from_raw_u64(18446744065119617025u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((base_row[73usize])
                    * (BFieldElement::from_raw_u64(281474976645120u64)))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (base_row[74usize]));
        let node_147 = ((BFieldElement::from_raw_u64(18446744065119617025u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((base_row[77usize])
                    * (BFieldElement::from_raw_u64(281474976645120u64)))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (base_row[78usize]));
        let node_89 = (base_row[64usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_663 = (base_row[142usize])
            + (BFieldElement::from_raw_u64(18446744052234715141u64));
        let node_666 = (base_row[142usize])
            + (BFieldElement::from_raw_u64(18446744009285042191u64));
        let node_86 = ((base_row[62usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64))) * (node_85);
        let node_83 = (base_row[62usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_103 = (base_row[62usize])
            + (BFieldElement::from_raw_u64(18446744056529682436u64));
        let base_constraints = [
            (node_11) * (base_row[4usize]),
            (node_11) * (node_8),
            (base_row[5usize])
                * ((base_row[5usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (base_row[6usize])
                * ((base_row[6usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (base_row[12usize])
                * ((base_row[12usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (base_row[13usize])
                * ((base_row[13usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (base_row[14usize])
                * ((base_row[14usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (base_row[15usize])
                * ((base_row[15usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (base_row[16usize])
                * ((base_row[16usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (base_row[17usize])
                * ((base_row[17usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (base_row[18usize])
                * ((base_row[18usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (base_row[8usize])
                * ((base_row[8usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (base_row[10usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((((((base_row[12usize])
                        + ((BFieldElement::from_raw_u64(8589934590u64))
                            * (base_row[13usize])))
                        + ((BFieldElement::from_raw_u64(17179869180u64))
                            * (base_row[14usize])))
                        + ((BFieldElement::from_raw_u64(34359738360u64))
                            * (base_row[15usize])))
                        + ((BFieldElement::from_raw_u64(68719476720u64))
                            * (base_row[16usize])))
                        + ((BFieldElement::from_raw_u64(137438953440u64))
                            * (base_row[17usize])))
                        + ((BFieldElement::from_raw_u64(274877906880u64))
                            * (base_row[18usize])))),
            ((base_row[8usize])
                * ((base_row[7usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                * (base_row[45usize]),
            (node_104) * (base_row[62usize]),
            (node_85) * (node_73),
            ((base_row[165usize]) * (node_79)) * (node_82),
            (node_104) * (base_row[64usize]),
            (node_114) * (base_row[64usize]),
            (node_153) * (base_row[109usize]),
            (node_155) * (base_row[110usize]),
            (node_157) * (base_row[111usize]),
            (node_159) * (base_row[112usize]),
            (node_153) * (node_126),
            (node_155) * (node_133),
            (node_157) * (node_140),
            (node_159) * (node_147),
            (node_153)
                * (((base_row[67usize])
                    * (BFieldElement::from_raw_u64(281474976645120u64)))
                    + (base_row[68usize])),
            (node_155)
                * (((base_row[71usize])
                    * (BFieldElement::from_raw_u64(281474976645120u64)))
                    + (base_row[72usize])),
            (node_157)
                * (((base_row[75usize])
                    * (BFieldElement::from_raw_u64(281474976645120u64)))
                    + (base_row[76usize])),
            (node_159)
                * (((base_row[79usize])
                    * (BFieldElement::from_raw_u64(281474976645120u64)))
                    + (base_row[80usize])),
            (node_114) * (base_row[103usize]),
            (node_114) * (base_row[104usize]),
            (node_114) * (base_row[105usize]),
            (node_114) * (base_row[106usize]),
            (node_114) * (base_row[107usize]),
            (node_114) * (base_row[108usize]),
            (node_116)
                * ((base_row[103usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (node_116)
                * ((base_row[104usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (node_116)
                * ((base_row[105usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (node_116)
                * ((base_row[106usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (node_116)
                * ((base_row[107usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (node_116)
                * ((base_row[108usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (((((node_102)
                * ((base_row[113usize])
                    + (BFieldElement::from_raw_u64(11408918724931329738u64))))
                + ((node_221)
                    * ((base_row[113usize])
                        + (BFieldElement::from_raw_u64(16073625066478178581u64)))))
                + ((base_row[155usize])
                    * ((base_row[113usize])
                        + (BFieldElement::from_raw_u64(12231462398569191607u64)))))
                + ((node_238)
                    * ((base_row[113usize])
                        + (BFieldElement::from_raw_u64(9408518518620565480u64)))))
                + ((node_245)
                    * ((base_row[113usize])
                        + (BFieldElement::from_raw_u64(11492978409391175103u64)))),
            (((((node_102)
                * ((base_row[114usize])
                    + (BFieldElement::from_raw_u64(2786462832312611053u64))))
                + ((node_221)
                    * ((base_row[114usize])
                        + (BFieldElement::from_raw_u64(11837051899140380443u64)))))
                + ((base_row[155usize])
                    * ((base_row[114usize])
                        + (BFieldElement::from_raw_u64(11546487907579866869u64)))))
                + ((node_238)
                    * ((base_row[114usize])
                        + (BFieldElement::from_raw_u64(1785884128667671832u64)))))
                + ((node_245)
                    * ((base_row[114usize])
                        + (BFieldElement::from_raw_u64(17615222217495663839u64)))),
            (((((node_102)
                * ((base_row[115usize])
                    + (BFieldElement::from_raw_u64(6782977121958050999u64))))
                + ((node_221)
                    * ((base_row[115usize])
                        + (BFieldElement::from_raw_u64(15625104599191418968u64)))))
                + ((base_row[155usize])
                    * ((base_row[115usize])
                        + (BFieldElement::from_raw_u64(14006427992450931468u64)))))
                + ((node_238)
                    * ((base_row[115usize])
                        + (BFieldElement::from_raw_u64(1188899344229954938u64)))))
                + ((node_245)
                    * ((base_row[115usize])
                        + (BFieldElement::from_raw_u64(5864349944556149748u64)))),
            (((((node_102)
                * ((base_row[116usize])
                    + (BFieldElement::from_raw_u64(8688421733879975670u64))))
                + ((node_221)
                    * ((base_row[116usize])
                        + (BFieldElement::from_raw_u64(12819157612210448391u64)))))
                + ((base_row[155usize])
                    * ((base_row[116usize])
                        + (BFieldElement::from_raw_u64(11770003398407723041u64)))))
                + ((node_238)
                    * ((base_row[116usize])
                        + (BFieldElement::from_raw_u64(14740727267735052728u64)))))
                + ((node_245)
                    * ((base_row[116usize])
                        + (BFieldElement::from_raw_u64(2745609811140253793u64)))),
            (((((node_102)
                * ((base_row[117usize])
                    + (BFieldElement::from_raw_u64(8602724563769480463u64))))
                + ((node_221)
                    * ((base_row[117usize])
                        + (BFieldElement::from_raw_u64(6235256903503367222u64)))))
                + ((base_row[155usize])
                    * ((base_row[117usize])
                        + (BFieldElement::from_raw_u64(15124190001489436038u64)))))
                + ((node_238)
                    * ((base_row[117usize])
                        + (BFieldElement::from_raw_u64(880257844992994007u64)))))
                + ((node_245)
                    * ((base_row[117usize])
                        + (BFieldElement::from_raw_u64(15189664869386394185u64)))),
            (((((node_102)
                * ((base_row[118usize])
                    + (BFieldElement::from_raw_u64(13589155570211330507u64))))
                + ((node_221)
                    * ((base_row[118usize])
                        + (BFieldElement::from_raw_u64(11242082964257948320u64)))))
                + ((base_row[155usize])
                    * ((base_row[118usize])
                        + (BFieldElement::from_raw_u64(14834674155811570980u64)))))
                + ((node_238)
                    * ((base_row[118usize])
                        + (BFieldElement::from_raw_u64(10737952517017171197u64)))))
                + ((node_245)
                    * ((base_row[118usize])
                        + (BFieldElement::from_raw_u64(5192963426821415349u64)))),
            (((((node_102)
                * ((base_row[119usize])
                    + (BFieldElement::from_raw_u64(10263462378312899510u64))))
                + ((node_221)
                    * ((base_row[119usize])
                        + (BFieldElement::from_raw_u64(5820425254787221108u64)))))
                + ((base_row[155usize])
                    * ((base_row[119usize])
                        + (BFieldElement::from_raw_u64(13004675752386552573u64)))))
                + ((node_238)
                    * ((base_row[119usize])
                        + (BFieldElement::from_raw_u64(15757222735741919824u64)))))
                + ((node_245)
                    * ((base_row[119usize])
                        + (BFieldElement::from_raw_u64(11971160388083607515u64)))),
            (((((node_102)
                * ((base_row[120usize])
                    + (BFieldElement::from_raw_u64(3264875873073042616u64))))
                + ((node_221)
                    * ((base_row[120usize])
                        + (BFieldElement::from_raw_u64(12019227591549292608u64)))))
                + ((base_row[155usize])
                    * ((base_row[120usize])
                        + (BFieldElement::from_raw_u64(1475232519215872482u64)))))
                + ((node_238)
                    * ((base_row[120usize])
                        + (BFieldElement::from_raw_u64(14382578632612566479u64)))))
                + ((node_245)
                    * ((base_row[120usize])
                        + (BFieldElement::from_raw_u64(11608544217838050708u64)))),
            (((((node_102)
                * ((base_row[121usize])
                    + (BFieldElement::from_raw_u64(3133435276616064683u64))))
                + ((node_221)
                    * ((base_row[121usize])
                        + (BFieldElement::from_raw_u64(4625353063880731092u64)))))
                + ((base_row[155usize])
                    * ((base_row[121usize])
                        + (BFieldElement::from_raw_u64(4883869161905122316u64)))))
                + ((node_238)
                    * ((base_row[121usize])
                        + (BFieldElement::from_raw_u64(3305272539067787726u64)))))
                + ((node_245)
                    * ((base_row[121usize])
                        + (BFieldElement::from_raw_u64(674972795234232729u64)))),
            (((((node_102)
                * ((base_row[122usize])
                    + (BFieldElement::from_raw_u64(13508500531157332153u64))))
                + ((node_221)
                    * ((base_row[122usize])
                        + (BFieldElement::from_raw_u64(3723900760706330287u64)))))
                + ((base_row[155usize])
                    * ((base_row[122usize])
                        + (BFieldElement::from_raw_u64(12579737103870920763u64)))))
                + ((node_238)
                    * ((base_row[122usize])
                        + (BFieldElement::from_raw_u64(17082569335437832789u64)))))
                + ((node_245)
                    * ((base_row[122usize])
                        + (BFieldElement::from_raw_u64(14165256104883557753u64)))),
            (((((node_102)
                * ((base_row[123usize])
                    + (BFieldElement::from_raw_u64(6968886508437513677u64))))
                + ((node_221)
                    * ((base_row[123usize])
                        + (BFieldElement::from_raw_u64(615596267195055952u64)))))
                + ((base_row[155usize])
                    * ((base_row[123usize])
                        + (BFieldElement::from_raw_u64(10119826060478909841u64)))))
                + ((node_238)
                    * ((base_row[123usize])
                        + (BFieldElement::from_raw_u64(229051680548583225u64)))))
                + ((node_245)
                    * ((base_row[123usize])
                        + (BFieldElement::from_raw_u64(15283356519694111298u64)))),
            (((((node_102)
                * ((base_row[124usize])
                    + (BFieldElement::from_raw_u64(9713264609690967820u64))))
                + ((node_221)
                    * ((base_row[124usize])
                        + (BFieldElement::from_raw_u64(18227830850447556704u64)))))
                + ((base_row[155usize])
                    * ((base_row[124usize])
                        + (BFieldElement::from_raw_u64(1528714547662620921u64)))))
                + ((node_238)
                    * ((base_row[124usize])
                        + (BFieldElement::from_raw_u64(2943254981416254648u64)))))
                + ((node_245)
                    * ((base_row[124usize])
                        + (BFieldElement::from_raw_u64(2306049938060341466u64)))),
            (((((node_102)
                * ((base_row[125usize])
                    + (BFieldElement::from_raw_u64(12482374976099749513u64))))
                + ((node_221)
                    * ((base_row[125usize])
                        + (BFieldElement::from_raw_u64(15609691041895848348u64)))))
                + ((base_row[155usize])
                    * ((base_row[125usize])
                        + (BFieldElement::from_raw_u64(12972275929555275935u64)))))
                + ((node_238)
                    * ((base_row[125usize])
                        + (BFieldElement::from_raw_u64(5767629304344025219u64)))))
                + ((node_245)
                    * ((base_row[125usize])
                        + (BFieldElement::from_raw_u64(11578793764462375094u64)))),
            (((((node_102)
                * ((base_row[126usize])
                    + (BFieldElement::from_raw_u64(13209711277645656680u64))))
                + ((node_221)
                    * ((base_row[126usize])
                        + (BFieldElement::from_raw_u64(15235800289984546486u64)))))
                + ((base_row[155usize])
                    * ((base_row[126usize])
                        + (BFieldElement::from_raw_u64(15992731669612695172u64)))))
                + ((node_238)
                    * ((base_row[126usize])
                        + (BFieldElement::from_raw_u64(16721422493821450473u64)))))
                + ((node_245)
                    * ((base_row[126usize])
                        + (BFieldElement::from_raw_u64(7511767364422267184u64)))),
            (((((node_102)
                * ((base_row[127usize])
                    + (BFieldElement::from_raw_u64(87705059284758253u64))))
                + ((node_221)
                    * ((base_row[127usize])
                        + (BFieldElement::from_raw_u64(11392407538241985753u64)))))
                + ((base_row[155usize])
                    * ((base_row[127usize])
                        + (BFieldElement::from_raw_u64(17877154195438905917u64)))))
                + ((node_238)
                    * ((base_row[127usize])
                        + (BFieldElement::from_raw_u64(5753720429376839714u64)))))
                + ((node_245)
                    * ((base_row[127usize])
                        + (BFieldElement::from_raw_u64(16999805755930336630u64)))),
            (((((node_102)
                * ((base_row[128usize])
                    + (BFieldElement::from_raw_u64(330155256278907084u64))))
                + ((node_221)
                    * ((base_row[128usize])
                        + (BFieldElement::from_raw_u64(11776128816341368822u64)))))
                + ((base_row[155usize])
                    * ((base_row[128usize])
                        + (BFieldElement::from_raw_u64(939319986782105612u64)))))
                + ((node_238)
                    * ((base_row[128usize])
                        + (BFieldElement::from_raw_u64(2063756830275051942u64)))))
                + ((node_245)
                    * ((base_row[128usize])
                        + (BFieldElement::from_raw_u64(940614108343834936u64)))),
            (base_row[129usize])
                * ((BFieldElement::from_raw_u64(4294967295u64))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (base_row[129usize]))),
            (base_row[135usize])
                * ((BFieldElement::from_raw_u64(4294967295u64))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (base_row[135usize]))),
            (base_row[139usize])
                * ((BFieldElement::from_raw_u64(4294967295u64))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (base_row[139usize]))),
            (base_row[139usize]) * (base_row[140usize]),
            (BFieldElement::from_raw_u64(4294967295u64))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((base_row[141usize])
                        * ((base_row[140usize])
                            + (BFieldElement::from_raw_u64(18446743927680663586u64))))),
            (base_row[144usize]) * (node_655),
            (base_row[143usize]) * (node_655),
            (base_row[146usize]) * (node_660),
            (base_row[145usize]) * (node_660),
            (base_row[167usize])
                * ((base_row[147usize])
                    + (BFieldElement::from_raw_u64(18446744060824649731u64))),
            (base_row[168usize]) * (base_row[147usize]),
            (((base_row[163usize]) * (node_655)) * (node_660)) * (base_row[147usize]),
            (((base_row[166usize]) * (node_678)) * (node_660))
                * ((base_row[147usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (((base_row[164usize]) * (node_680)) * (node_655))
                * ((base_row[147usize]) + (BFieldElement::from_raw_u64(4294967295u64))),
            (((base_row[166usize]) * (node_674)) * (node_655)) * (base_row[147usize]),
            ((base_row[164usize]) * (base_row[139usize])) * (node_655),
            (node_680) * (base_row[148usize]),
            (base_row[151usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((base_row[64usize]) * (node_89))),
            (base_row[152usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((node_89) * (node_90)) * (node_94)) * (node_97))),
            (base_row[153usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((base_row[64usize]) * (node_90)) * (node_94)) * (node_97))),
            (base_row[154usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((base_row[151usize]) * (node_90))),
            (base_row[155usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((base_row[151usize]) * (node_94)) * (node_97)) * (node_101))),
            (base_row[156usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_663)
                        * ((base_row[142usize])
                            + (BFieldElement::from_raw_u64(18446744043644780551u64))))),
            (base_row[157usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((base_row[143usize]) * (base_row[144usize]))),
            (base_row[158usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((node_663) * (node_666)) * (node_670)) * (node_674))),
            (base_row[159usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((base_row[156usize]) * (node_666))),
            (base_row[160usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((base_row[145usize]) * (base_row[146usize]))),
            (base_row[161usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_86) * (base_row[62usize]))),
            (base_row[162usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((base_row[158usize]) * (node_678))),
            (base_row[163usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((base_row[156usize]) * (node_670)) * (node_674)) * (node_678))),
            (base_row[164usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((base_row[159usize]) * (node_674)) * (node_678))),
            (base_row[165usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((node_83) * (node_103)) * (base_row[62usize]))
                        * ((base_row[63usize])
                            + (BFieldElement::from_raw_u64(18446743897615892521u64))))),
            (base_row[166usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((base_row[159usize]) * (node_670))),
            (base_row[167usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((base_row[162usize]) * (node_680)) * (node_655)) * (node_660))),
            (base_row[168usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((base_row[162usize]) * (base_row[139usize])) * (node_655))
                        * (node_660))),
        ];
        let ext_constraints = [];
        base_constraints.into_iter().chain(ext_constraints).collect()
    }
    #[allow(unused_variables)]
    fn evaluate_transition_constraints(
        current_base_row: ArrayView1<XFieldElement>,
        current_ext_row: ArrayView1<XFieldElement>,
        next_base_row: ArrayView1<XFieldElement>,
        next_ext_row: ArrayView1<XFieldElement>,
        challenges: &Challenges,
    ) -> Vec<XFieldElement> {
        let node_120 = (next_base_row[19usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[19usize]));
        let node_517 = (next_ext_row[3usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[3usize]));
        let node_521 = (next_ext_row[4usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[4usize]));
        let node_124 = (next_base_row[20usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[20usize]));
        let node_4096 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (next_base_row[8usize]));
        let node_128 = (next_base_row[21usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[21usize]));
        let node_513 = (next_ext_row[7usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[7usize]));
        let node_1848 = (current_base_row[18usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_158 = (next_base_row[38usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[38usize]));
        let node_1224 = ((next_base_row[9usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[9usize])))
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_1846 = (current_base_row[17usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_785 = (next_base_row[22usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[22usize]));
        let node_167 = ((challenges[16usize]) * (current_base_row[7usize]))
            + ((challenges[17usize]) * (current_base_row[13usize]));
        let node_1219 = (next_ext_row[6usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[6usize]));
        let node_1230 = (next_base_row[28usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[28usize]));
        let node_1231 = (next_base_row[29usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[29usize]));
        let node_1232 = (next_base_row[30usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[30usize]));
        let node_1233 = (next_base_row[31usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[31usize]));
        let node_1234 = (next_base_row[32usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[32usize]));
        let node_1235 = (next_base_row[33usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[33usize]));
        let node_1236 = (next_base_row[34usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[34usize]));
        let node_1237 = (next_base_row[35usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[35usize]));
        let node_1238 = (next_base_row[36usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[36usize]));
        let node_1239 = (next_base_row[37usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[37usize]));
        let node_5778 = (current_base_row[280usize])
            * ((next_base_row[64usize])
                + (BFieldElement::from_raw_u64(18446744052234715141u64)));
        let node_868 = ((((((((((((((((challenges[32usize]) * (next_base_row[22usize]))
            + ((challenges[33usize]) * (next_base_row[23usize])))
            + ((challenges[34usize]) * (next_base_row[24usize])))
            + ((challenges[35usize]) * (next_base_row[25usize])))
            + ((challenges[36usize]) * (next_base_row[26usize])))
            + ((challenges[37usize]) * (next_base_row[27usize])))
            + ((challenges[38usize]) * (next_base_row[28usize])))
            + ((challenges[39usize]) * (next_base_row[29usize])))
            + ((challenges[40usize]) * (next_base_row[30usize])))
            + ((challenges[41usize]) * (next_base_row[31usize])))
            + ((challenges[42usize]) * (next_base_row[32usize])))
            + ((challenges[43usize]) * (next_base_row[33usize])))
            + ((challenges[44usize]) * (next_base_row[34usize])))
            + ((challenges[45usize]) * (next_base_row[35usize])))
            + ((challenges[46usize]) * (next_base_row[36usize])))
            + ((challenges[47usize]) * (next_base_row[37usize]));
        let node_1229 = (next_base_row[27usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[27usize]));
        let node_5855 = (((next_base_row[63usize])
            + (BFieldElement::from_raw_u64(18446743992105173011u64)))
            * ((next_base_row[63usize])
                + (BFieldElement::from_raw_u64(18446743923385696291u64))))
            * ((next_base_row[63usize])
                + (BFieldElement::from_raw_u64(18446743828896415801u64)));
        let node_4832 = (((((current_base_row[81usize])
            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
            + ((current_base_row[82usize])
                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
            + ((current_base_row[83usize])
                * (BFieldElement::from_raw_u64(281474976645120u64))))
            + (current_base_row[84usize])) * (BFieldElement::from_raw_u64(1u64));
        let node_4843 = (((((current_base_row[85usize])
            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
            + ((current_base_row[86usize])
                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
            + ((current_base_row[87usize])
                * (BFieldElement::from_raw_u64(281474976645120u64))))
            + (current_base_row[88usize])) * (BFieldElement::from_raw_u64(1u64));
        let node_4854 = (((((current_base_row[89usize])
            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
            + ((current_base_row[90usize])
                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
            + ((current_base_row[91usize])
                * (BFieldElement::from_raw_u64(281474976645120u64))))
            + (current_base_row[92usize])) * (BFieldElement::from_raw_u64(1u64));
        let node_4865 = (((((current_base_row[93usize])
            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
            + ((current_base_row[94usize])
                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
            + ((current_base_row[95usize])
                * (BFieldElement::from_raw_u64(281474976645120u64))))
            + (current_base_row[96usize])) * (BFieldElement::from_raw_u64(1u64));
        let node_200 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[27usize]);
        let node_870 = (challenges[33usize]) * (current_base_row[23usize]);
        let node_872 = (challenges[34usize]) * (current_base_row[24usize]);
        let node_874 = (challenges[35usize]) * (current_base_row[25usize]);
        let node_876 = (challenges[36usize]) * (current_base_row[26usize]);
        let node_878 = (challenges[37usize]) * (current_base_row[27usize]);
        let node_880 = (challenges[38usize]) * (current_base_row[28usize]);
        let node_882 = (challenges[39usize]) * (current_base_row[29usize]);
        let node_884 = (challenges[40usize]) * (current_base_row[30usize]);
        let node_886 = (challenges[41usize]) * (current_base_row[31usize]);
        let node_888 = (challenges[42usize]) * (current_base_row[32usize]);
        let node_890 = (challenges[43usize]) * (current_base_row[33usize]);
        let node_892 = (challenges[44usize]) * (current_base_row[34usize]);
        let node_894 = (challenges[45usize]) * (current_base_row[35usize]);
        let node_896 = (challenges[46usize]) * (current_base_row[36usize]);
        let node_898 = (challenges[47usize]) * (current_base_row[37usize]);
        let node_1226 = (next_base_row[24usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[24usize]));
        let node_1227 = (next_base_row[25usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[25usize]));
        let node_1228 = (next_base_row[26usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[26usize]));
        let node_196 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[26usize]);
        let node_1225 = (next_base_row[23usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[23usize]));
        let node_192 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[25usize]);
        let node_204 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[28usize]);
        let node_208 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[29usize]);
        let node_212 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[30usize]);
        let node_216 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[31usize]);
        let node_220 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[32usize]);
        let node_224 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[33usize]);
        let node_1844 = (current_base_row[16usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_228 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[34usize]);
        let node_197 = (next_base_row[25usize]) + (node_196);
        let node_201 = (next_base_row[26usize]) + (node_200);
        let node_205 = (next_base_row[27usize]) + (node_204);
        let node_209 = (next_base_row[28usize]) + (node_208);
        let node_213 = (next_base_row[29usize]) + (node_212);
        let node_217 = (next_base_row[30usize]) + (node_216);
        let node_159 = (node_158) + (BFieldElement::from_raw_u64(4294967295u64));
        let node_221 = (next_base_row[31usize]) + (node_220);
        let node_225 = (next_base_row[32usize]) + (node_224);
        let node_229 = (next_base_row[33usize]) + (node_228);
        let node_233 = (next_base_row[34usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[35usize]));
        let node_237 = (next_base_row[35usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[36usize]));
        let node_241 = (next_base_row[36usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[37usize]));
        let node_181 = (next_ext_row[6usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((current_ext_row[6usize])
                    * ((challenges[7usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((node_167)
                                + ((challenges[18usize]) * (next_base_row[38usize])))
                                + ((challenges[19usize]) * (next_base_row[37usize])))))));
        let node_6244 = (next_base_row[139usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_1317 = ((current_base_row[7usize]) * (challenges[20usize]))
            + (challenges[23usize]);
        let node_184 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[23usize]);
        let node_188 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[24usize]);
        let node_232 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[35usize]);
        let node_116 = ((next_base_row[9usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[9usize])))
            + (BFieldElement::from_raw_u64(18446744060824649731u64));
        let node_189 = (next_base_row[23usize]) + (node_188);
        let node_193 = (next_base_row[24usize]) + (node_192);
        let node_4274 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (next_base_row[44usize]));
        let node_4601 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (((next_base_row[52usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[52usize]))) * (current_base_row[54usize])));
        let node_236 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[36usize]);
        let node_525 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[22usize]);
        let node_240 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[37usize]);
        let node_154 = ((((current_base_row[11usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((BFieldElement::from_raw_u64(34359738360u64))
                    * (current_base_row[42usize]))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((BFieldElement::from_raw_u64(17179869180u64))
                    * (current_base_row[41usize]))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((BFieldElement::from_raw_u64(8589934590u64))
                    * (current_base_row[40usize]))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[39usize]));
        let node_295 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[39usize]));
        let node_4598 = (next_base_row[52usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[52usize]));
        let node_5621 = (next_base_row[62usize])
            + (BFieldElement::from_raw_u64(18446744056529682436u64));
        let node_6240 = (current_base_row[145usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((BFieldElement::from_raw_u64(8589934590u64))
                    * (next_base_row[145usize])));
        let node_4476 = (next_ext_row[12usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[12usize]));
        let node_6237 = (current_base_row[143usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((BFieldElement::from_raw_u64(8589934590u64))
                    * (next_base_row[143usize])));
        let node_1842 = (current_base_row[15usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_1315 = (current_base_row[7usize]) * (challenges[20usize]);
        let node_1665 = (challenges[1usize]) * (current_ext_row[3usize]);
        let node_5625 = (next_base_row[63usize])
            + (BFieldElement::from_raw_u64(18446743897615892521u64));
        let node_522 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[11usize]);
        let node_1241 = (current_base_row[276usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_1298 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[275usize]));
        let node_113 = (next_base_row[9usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[9usize]));
        let node_527 = (next_base_row[24usize]) + (node_184);
        let node_528 = (next_base_row[25usize]) + (node_188);
        let node_529 = (next_base_row[26usize]) + (node_192);
        let node_530 = (next_base_row[27usize]) + (node_196);
        let node_531 = (next_base_row[28usize]) + (node_200);
        let node_532 = (next_base_row[29usize]) + (node_204);
        let node_533 = (next_base_row[30usize]) + (node_208);
        let node_534 = (next_base_row[31usize]) + (node_212);
        let node_541 = (node_158)
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_535 = (next_base_row[32usize]) + (node_216);
        let node_536 = (next_base_row[33usize]) + (node_220);
        let node_537 = (next_base_row[34usize]) + (node_224);
        let node_538 = (next_base_row[35usize]) + (node_228);
        let node_539 = (next_base_row[36usize]) + (node_232);
        let node_540 = (next_base_row[37usize]) + (node_236);
        let node_550 = (next_ext_row[6usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((current_ext_row[6usize])
                    * ((challenges[7usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((node_167)
                                + ((challenges[18usize]) * (current_base_row[38usize])))
                                + ((challenges[19usize])
                                    * (current_base_row[37usize])))))));
        let node_4705 = ((next_base_row[59usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[59usize])))
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_5492 = (((((next_base_row[65usize])
            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
            + ((next_base_row[66usize])
                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
            + ((next_base_row[67usize])
                * (BFieldElement::from_raw_u64(281474976645120u64))))
            + (next_base_row[68usize])) * (BFieldElement::from_raw_u64(1u64));
        let node_5503 = (((((next_base_row[69usize])
            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
            + ((next_base_row[70usize])
                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
            + ((next_base_row[71usize])
                * (BFieldElement::from_raw_u64(281474976645120u64))))
            + (next_base_row[72usize])) * (BFieldElement::from_raw_u64(1u64));
        let node_5514 = (((((next_base_row[73usize])
            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
            + ((next_base_row[74usize])
                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
            + ((next_base_row[75usize])
                * (BFieldElement::from_raw_u64(281474976645120u64))))
            + (next_base_row[76usize])) * (BFieldElement::from_raw_u64(1u64));
        let node_5525 = (((((next_base_row[77usize])
            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
            + ((next_base_row[78usize])
                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
            + ((next_base_row[79usize])
                * (BFieldElement::from_raw_u64(281474976645120u64))))
            + (next_base_row[80usize])) * (BFieldElement::from_raw_u64(1u64));
        let node_5616 = (next_base_row[62usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_6184 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (next_base_row[135usize]));
        let node_6271 = (next_base_row[142usize])
            + (BFieldElement::from_raw_u64(18446743940565565471u64));
        let node_1840 = (current_base_row[14usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_248 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[40usize]));
        let node_6275 = (next_base_row[142usize])
            + (BFieldElement::from_raw_u64(18446743949155500061u64));
        let node_34 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((current_base_row[4usize])
                    * ((BFieldElement::from_raw_u64(38654705655u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[3usize])))));
        let node_185 = (next_base_row[22usize]) + (node_184);
        let node_1590 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[317usize]));
        let node_526 = (next_base_row[23usize]) + (node_525);
        let node_329 = (next_base_row[25usize]) + (node_204);
        let node_414 = (next_base_row[27usize]) + (node_220);
        let node_330 = (next_base_row[26usize]) + (node_208);
        let node_415 = (next_base_row[28usize]) + (node_224);
        let node_331 = (next_base_row[27usize]) + (node_212);
        let node_416 = (next_base_row[29usize]) + (node_228);
        let node_332 = (next_base_row[28usize]) + (node_216);
        let node_417 = (next_base_row[30usize]) + (node_232);
        let node_333 = (next_base_row[29usize]) + (node_220);
        let node_418 = (next_base_row[31usize]) + (node_236);
        let node_334 = (next_base_row[30usize]) + (node_224);
        let node_437 = (((((current_base_row[185usize]) * (node_159))
            + ((current_base_row[186usize])
                * ((node_158) + (BFieldElement::from_raw_u64(8589934590u64)))))
            + ((current_base_row[187usize])
                * ((node_158) + (BFieldElement::from_raw_u64(12884901885u64)))))
            + ((current_base_row[189usize])
                * ((node_158) + (BFieldElement::from_raw_u64(17179869180u64)))))
            + ((current_base_row[190usize])
                * ((node_158) + (BFieldElement::from_raw_u64(21474836475u64))));
        let node_730 = (((((current_base_row[185usize]) * (node_541))
            + ((current_base_row[186usize])
                * ((node_158) + (BFieldElement::from_raw_u64(18446744060824649731u64)))))
            + ((current_base_row[187usize])
                * ((node_158) + (BFieldElement::from_raw_u64(18446744056529682436u64)))))
            + ((current_base_row[189usize])
                * ((node_158) + (BFieldElement::from_raw_u64(18446744052234715141u64)))))
            + ((current_base_row[190usize])
                * ((node_158) + (BFieldElement::from_raw_u64(18446744047939747846u64))));
        let node_419 = (next_base_row[32usize]) + (node_240);
        let node_397 = (node_158) + (BFieldElement::from_raw_u64(21474836475u64));
        let node_335 = (next_base_row[31usize]) + (node_228);
        let node_441 = ((((current_ext_row[73usize]) + (current_ext_row[74usize]))
            + (current_ext_row[75usize])) + (current_ext_row[76usize]))
            + ((current_base_row[190usize])
                * ((next_ext_row[6usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_ext_row[59usize]))));
        let node_734 = ((((current_ext_row[77usize]) + (current_ext_row[78usize]))
            + (current_ext_row[79usize])) + (current_ext_row[80usize]))
            + (current_ext_row[81usize]);
        let node_408 = (next_ext_row[6usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[59usize]));
        let node_336 = (next_base_row[32usize]) + (node_232);
        let node_337 = (next_base_row[33usize]) + (node_236);
        let node_449 = ((((current_base_row[290usize]) + (current_base_row[291usize]))
            + (current_base_row[292usize])) + (current_base_row[293usize]))
            + (current_base_row[294usize]);
        let node_742 = ((((current_base_row[285usize]) + (current_base_row[286usize]))
            + (current_base_row[287usize])) + (current_base_row[288usize]))
            + (current_base_row[289usize]);
        let node_338 = (next_base_row[34usize]) + (node_240);
        let node_453 = (((((current_base_row[185usize]) * (node_193))
            + ((current_base_row[186usize]) * ((next_base_row[24usize]) + (node_196))))
            + ((current_base_row[187usize]) * ((next_base_row[24usize]) + (node_200))))
            + ((current_base_row[189usize]) * ((next_base_row[24usize]) + (node_204))))
            + ((current_base_row[190usize]) * ((next_base_row[24usize]) + (node_208)));
        let node_746 = (((((current_base_row[185usize]) * (node_528))
            + ((current_base_row[186usize]) * ((next_base_row[26usize]) + (node_188))))
            + ((current_base_row[187usize]) * ((next_base_row[27usize]) + (node_188))))
            + ((current_base_row[189usize]) * ((next_base_row[28usize]) + (node_188))))
            + ((current_base_row[190usize]) * ((next_base_row[29usize]) + (node_188)));
        let node_314 = (node_158) + (BFieldElement::from_raw_u64(12884901885u64));
        let node_457 = (((((current_base_row[185usize]) * (node_197))
            + ((current_base_row[186usize]) * ((next_base_row[25usize]) + (node_200))))
            + ((current_base_row[187usize]) * (node_329)))
            + ((current_base_row[189usize]) * ((next_base_row[25usize]) + (node_208))))
            + ((current_base_row[190usize]) * ((next_base_row[25usize]) + (node_212)));
        let node_750 = (((((current_base_row[185usize]) * (node_529))
            + ((current_base_row[186usize]) * ((next_base_row[27usize]) + (node_192))))
            + ((current_base_row[187usize]) * ((next_base_row[28usize]) + (node_192))))
            + ((current_base_row[189usize]) * ((next_base_row[29usize]) + (node_192))))
            + ((current_base_row[190usize]) * ((next_base_row[30usize]) + (node_192)));
        let node_325 = (next_ext_row[6usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((current_ext_row[6usize]) * (current_ext_row[52usize])));
        let node_461 = (((((current_base_row[185usize]) * (node_201))
            + ((current_base_row[186usize]) * ((next_base_row[26usize]) + (node_204))))
            + ((current_base_row[187usize]) * (node_330)))
            + ((current_base_row[189usize]) * ((next_base_row[26usize]) + (node_212))))
            + ((current_base_row[190usize]) * ((next_base_row[26usize]) + (node_216)));
        let node_754 = (((((current_base_row[185usize]) * (node_530))
            + ((current_base_row[186usize]) * ((next_base_row[28usize]) + (node_196))))
            + ((current_base_row[187usize]) * ((next_base_row[29usize]) + (node_196))))
            + ((current_base_row[189usize]) * ((next_base_row[30usize]) + (node_196))))
            + ((current_base_row[190usize]) * ((next_base_row[31usize]) + (node_196)));
        let node_465 = (((((current_base_row[185usize]) * (node_205))
            + ((current_base_row[186usize]) * ((next_base_row[27usize]) + (node_208))))
            + ((current_base_row[187usize]) * (node_331)))
            + ((current_base_row[189usize]) * ((next_base_row[27usize]) + (node_216))))
            + ((current_base_row[190usize]) * (node_414));
        let node_758 = (((((current_base_row[185usize]) * (node_531))
            + ((current_base_row[186usize]) * ((next_base_row[29usize]) + (node_200))))
            + ((current_base_row[187usize]) * ((next_base_row[30usize]) + (node_200))))
            + ((current_base_row[189usize]) * ((next_base_row[31usize]) + (node_200))))
            + ((current_base_row[190usize]) * ((next_base_row[32usize]) + (node_200)));
        let node_469 = (((((current_base_row[185usize]) * (node_209))
            + ((current_base_row[186usize]) * ((next_base_row[28usize]) + (node_212))))
            + ((current_base_row[187usize]) * (node_332)))
            + ((current_base_row[189usize]) * ((next_base_row[28usize]) + (node_220))))
            + ((current_base_row[190usize]) * (node_415));
        let node_762 = (((((current_base_row[185usize]) * (node_532))
            + ((current_base_row[186usize]) * ((next_base_row[30usize]) + (node_204))))
            + ((current_base_row[187usize]) * ((next_base_row[31usize]) + (node_204))))
            + ((current_base_row[189usize]) * ((next_base_row[32usize]) + (node_204))))
            + ((current_base_row[190usize]) * ((next_base_row[33usize]) + (node_204)));
        let node_473 = (((((current_base_row[185usize]) * (node_213))
            + ((current_base_row[186usize]) * ((next_base_row[29usize]) + (node_216))))
            + ((current_base_row[187usize]) * (node_333)))
            + ((current_base_row[189usize]) * ((next_base_row[29usize]) + (node_224))))
            + ((current_base_row[190usize]) * (node_416));
        let node_766 = (((((current_base_row[185usize]) * (node_533))
            + ((current_base_row[186usize]) * ((next_base_row[31usize]) + (node_208))))
            + ((current_base_row[187usize]) * ((next_base_row[32usize]) + (node_208))))
            + ((current_base_row[189usize]) * ((next_base_row[33usize]) + (node_208))))
            + ((current_base_row[190usize]) * ((next_base_row[34usize]) + (node_208)));
        let node_477 = (((((current_base_row[185usize]) * (node_217))
            + ((current_base_row[186usize]) * ((next_base_row[30usize]) + (node_220))))
            + ((current_base_row[187usize]) * (node_334)))
            + ((current_base_row[189usize]) * ((next_base_row[30usize]) + (node_228))))
            + ((current_base_row[190usize]) * (node_417));
        let node_770 = (((((current_base_row[185usize]) * (node_534))
            + ((current_base_row[186usize]) * ((next_base_row[32usize]) + (node_212))))
            + ((current_base_row[187usize]) * ((next_base_row[33usize]) + (node_212))))
            + ((current_base_row[189usize]) * ((next_base_row[34usize]) + (node_212))))
            + ((current_base_row[190usize]) * ((next_base_row[35usize]) + (node_212)));
        let node_481 = (((((current_base_row[185usize]) * (node_221))
            + ((current_base_row[186usize]) * ((next_base_row[31usize]) + (node_224))))
            + ((current_base_row[187usize]) * (node_335)))
            + ((current_base_row[189usize]) * ((next_base_row[31usize]) + (node_232))))
            + ((current_base_row[190usize]) * (node_418));
        let node_774 = (((((current_base_row[185usize]) * (node_535))
            + ((current_base_row[186usize]) * ((next_base_row[33usize]) + (node_216))))
            + ((current_base_row[187usize]) * ((next_base_row[34usize]) + (node_216))))
            + ((current_base_row[189usize]) * ((next_base_row[35usize]) + (node_216))))
            + ((current_base_row[190usize]) * ((next_base_row[36usize]) + (node_216)));
        let node_485 = (((((current_base_row[185usize]) * (node_225))
            + ((current_base_row[186usize]) * ((next_base_row[32usize]) + (node_228))))
            + ((current_base_row[187usize]) * (node_336)))
            + ((current_base_row[189usize]) * ((next_base_row[32usize]) + (node_236))))
            + ((current_base_row[190usize]) * (node_419));
        let node_778 = (((((current_base_row[185usize]) * (node_536))
            + ((current_base_row[186usize]) * ((next_base_row[34usize]) + (node_220))))
            + ((current_base_row[187usize]) * ((next_base_row[35usize]) + (node_220))))
            + ((current_base_row[189usize]) * ((next_base_row[36usize]) + (node_220))))
            + ((current_base_row[190usize]) * ((next_base_row[37usize]) + (node_220)));
        let node_488 = ((((current_base_row[185usize]) * (node_229))
            + ((current_base_row[186usize]) * ((next_base_row[33usize]) + (node_232))))
            + ((current_base_row[187usize]) * (node_337)))
            + ((current_base_row[189usize]) * ((next_base_row[33usize]) + (node_240)));
        let node_781 = ((((current_base_row[185usize]) * (node_537))
            + ((current_base_row[186usize]) * ((next_base_row[35usize]) + (node_224))))
            + ((current_base_row[187usize]) * ((next_base_row[36usize]) + (node_224))))
            + ((current_base_row[189usize]) * ((next_base_row[37usize]) + (node_224)));
        let node_490 = (((current_base_row[185usize]) * (node_233))
            + ((current_base_row[186usize]) * ((next_base_row[34usize]) + (node_236))))
            + ((current_base_row[187usize]) * (node_338));
        let node_783 = (((current_base_row[185usize]) * (node_538))
            + ((current_base_row[186usize]) * ((next_base_row[36usize]) + (node_228))))
            + ((current_base_row[187usize]) * ((next_base_row[37usize]) + (node_228)));
        let node_491 = ((current_base_row[185usize]) * (node_237))
            + ((current_base_row[186usize]) * ((next_base_row[35usize]) + (node_240)));
        let node_784 = ((current_base_row[185usize]) * (node_539))
            + ((current_base_row[186usize]) * ((next_base_row[37usize]) + (node_232)));
        let node_267 = (current_base_row[185usize]) * (node_241);
        let node_567 = (current_base_row[185usize]) * (node_540);
        let node_4387 = ((next_ext_row[11usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((challenges[6usize]) * (current_ext_row[11usize]))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((challenges[31usize]) * (current_base_row[10usize])));
        let node_4440 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * ((challenges[57usize]) * (current_base_row[10usize]));
        let node_4480 = ((node_4476)
            * (((((challenges[10usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((challenges[55usize]) * (current_base_row[22usize]))))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((challenges[56usize]) * (current_base_row[23usize]))))
                + (node_4440))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((challenges[58usize]) * (next_base_row[22usize])))))
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_4444 = (challenges[10usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((challenges[55usize]) * (current_base_row[22usize])));
        let node_4524 = ((next_base_row[48usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[48usize])))
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_4523 = (next_base_row[48usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[48usize]));
        let node_4530 = (next_base_row[47usize])
            + (BFieldElement::from_raw_u64(18446744060824649731u64));
        let node_4556 = (next_ext_row[15usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[15usize]));
        let node_4593 = (next_base_row[51usize])
            + (BFieldElement::from_raw_u64(18446744060824649731u64));
        let node_4676 = (next_ext_row[21usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[21usize]));
        let node_4704 = (next_base_row[59usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[59usize]));
        let node_5613 = (current_base_row[62usize])
            + (BFieldElement::from_raw_u64(18446744056529682436u64));
        let node_5677 = (next_base_row[64usize])
            + (BFieldElement::from_raw_u64(18446744047939747846u64));
        let node_5728 = (next_base_row[63usize])
            + (BFieldElement::from_raw_u64(18446743992105173011u64));
        let node_5730 = (next_base_row[63usize])
            + (BFieldElement::from_raw_u64(18446743923385696291u64));
        let node_5847 = (next_ext_row[28usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[28usize]));
        let node_5868 = (next_ext_row[29usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[29usize]));
        let node_5885 = (next_ext_row[30usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[30usize]));
        let node_5902 = (next_ext_row[31usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[31usize]));
        let node_5919 = (next_ext_row[32usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[32usize]));
        let node_5936 = (next_ext_row[33usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[33usize]));
        let node_5953 = (next_ext_row[34usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[34usize]));
        let node_5970 = (next_ext_row[35usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[35usize]));
        let node_5987 = (next_ext_row[36usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[36usize]));
        let node_6004 = (next_ext_row[37usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[37usize]));
        let node_6021 = (next_ext_row[38usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[38usize]));
        let node_6038 = (next_ext_row[39usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[39usize]));
        let node_6055 = (next_ext_row[40usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[40usize]));
        let node_6072 = (next_ext_row[41usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[41usize]));
        let node_6089 = (next_ext_row[42usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[42usize]));
        let node_6106 = (next_ext_row[43usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[43usize]));
        let node_6132 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (next_base_row[129usize]));
        let node_6234 = (current_base_row[142usize])
            + (BFieldElement::from_raw_u64(18446743940565565471u64));
        let node_6261 = (node_6240)
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_6269 = (next_base_row[142usize])
            + (BFieldElement::from_raw_u64(18446744017874976781u64));
        let node_5637 = (next_base_row[62usize])
            + (BFieldElement::from_raw_u64(18446744060824649731u64));
        let node_30 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[3usize]);
        let node_47 = (next_base_row[6usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_51 = (next_ext_row[0usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[0usize]));
        let node_31 = (BFieldElement::from_raw_u64(38654705655u64)) + (node_30);
        let node_74 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (next_base_row[1usize]);
        let node_90 = (BFieldElement::from_raw_u64(38654705655u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (next_base_row[3usize]));
        let node_88 = (next_ext_row[2usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[2usize]));
        let node_253 = (current_base_row[185usize]) * (node_185);
        let node_299 = (current_base_row[186usize])
            * ((next_base_row[22usize]) + (node_188));
        let node_342 = (current_base_row[187usize])
            * ((next_base_row[22usize]) + (node_192));
        let node_384 = (current_base_row[189usize])
            * ((next_base_row[22usize]) + (node_196));
        let node_423 = (current_base_row[190usize])
            * ((next_base_row[22usize]) + (node_200));
        let node_804 = (next_base_row[22usize]) + (node_220);
        let node_887 = ((((((((((challenges[32usize]) * (current_base_row[22usize]))
            + (node_870)) + (node_872)) + (node_874)) + (node_876)) + (node_878))
            + (node_880)) + (node_882)) + (node_884)) + (node_886);
        let node_1223 = (next_base_row[10usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[10usize]));
        let node_1292 = (node_120) + (BFieldElement::from_raw_u64(4294967295u64));
        let node_1294 = (next_base_row[9usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[21usize]));
        let node_1584 = (next_base_row[22usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((current_base_row[22usize]) * (current_base_row[23usize])));
        let node_1585 = (next_base_row[22usize]) * (current_base_row[22usize]);
        let node_1609 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (next_base_row[22usize]);
        let node_1625 = ((current_base_row[24usize]) * (current_base_row[26usize]))
            + ((current_base_row[23usize]) * (current_base_row[27usize]));
        let node_1639 = (current_base_row[24usize]) * (next_base_row[23usize]);
        let node_1642 = (current_base_row[23usize]) * (next_base_row[24usize]);
        let node_1422 = (node_785)
            + (BFieldElement::from_raw_u64(18446744056529682436u64));
        let node_1396 = (node_785)
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_112 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[9usize]);
        let node_1293 = (next_base_row[9usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[20usize]));
        let node_1295 = (current_base_row[28usize]) + (node_200);
        let node_1587 = (current_base_row[23usize]) + (node_525);
        let node_1743 = (node_1225)
            + (BFieldElement::from_raw_u64(18446744056529682436u64));
        let node_247 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[40usize]);
        let node_144 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * ((BFieldElement::from_raw_u64(34359738360u64))
                * (current_base_row[42usize]));
        let node_1785 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (next_ext_row[7usize]);
        let node_1797 = ((current_base_row[41usize]) * (current_base_row[43usize]))
            + ((current_base_row[40usize]) * (current_base_row[44usize]));
        let node_1798 = (current_base_row[41usize]) * (current_base_row[44usize]);
        let node_445 = ((((node_253) + (node_299)) + (node_342)) + (node_384))
            + (node_423);
        let node_738 = (((((current_base_row[185usize]) * (node_526))
            + ((current_base_row[186usize]) * ((next_base_row[24usize]) + (node_525))))
            + ((current_base_row[187usize]) * ((next_base_row[25usize]) + (node_525))))
            + ((current_base_row[189usize]) * ((next_base_row[26usize]) + (node_525))))
            + ((current_base_row[190usize]) * ((next_base_row[27usize]) + (node_525)));
        let node_409 = (next_base_row[22usize]) + (node_200);
        let node_410 = (next_base_row[23usize]) + (node_204);
        let node_411 = (next_base_row[24usize]) + (node_208);
        let node_412 = (next_base_row[25usize]) + (node_212);
        let node_413 = (next_base_row[26usize]) + (node_216);
        let node_1727 = ((challenges[2usize])
            * (((challenges[2usize])
                * (((challenges[2usize])
                    * (((challenges[2usize]) * (current_ext_row[4usize]))
                        + (current_base_row[22usize]))) + (current_base_row[23usize])))
                + (current_base_row[24usize]))) + (current_base_row[25usize]);
        let node_1722 = ((challenges[2usize])
            * (((challenges[2usize])
                * (((challenges[2usize]) * (current_ext_row[4usize]))
                    + (current_base_row[22usize]))) + (current_base_row[23usize])))
            + (current_base_row[24usize]);
        let node_1717 = ((challenges[2usize])
            * (((challenges[2usize]) * (current_ext_row[4usize]))
                + (current_base_row[22usize]))) + (current_base_row[23usize]);
        let node_1712 = ((challenges[2usize]) * (current_ext_row[4usize]))
            + (current_base_row[22usize]);
        let node_4210 = (next_ext_row[5usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[5usize]));
        let node_4334 = (next_ext_row[9usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((challenges[4usize]) * (current_ext_row[9usize])));
        let node_4335 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (((((((((((challenges[32usize]) * (next_base_row[22usize]))
                + ((challenges[33usize]) * (next_base_row[23usize])))
                + ((challenges[34usize]) * (next_base_row[24usize])))
                + ((challenges[35usize]) * (next_base_row[25usize])))
                + ((challenges[36usize]) * (next_base_row[26usize])))
                + ((challenges[37usize]) * (next_base_row[27usize])))
                + ((challenges[38usize]) * (next_base_row[28usize])))
                + ((challenges[39usize]) * (next_base_row[29usize])))
                + ((challenges[40usize]) * (next_base_row[30usize])))
                + ((challenges[41usize]) * (next_base_row[31usize])));
        let node_846 = (((((challenges[32usize]) * (next_base_row[22usize]))
            + ((challenges[33usize]) * (next_base_row[23usize])))
            + ((challenges[34usize]) * (next_base_row[24usize])))
            + ((challenges[35usize]) * (next_base_row[25usize])))
            + ((challenges[36usize]) * (next_base_row[26usize]));
        let node_4383 = (next_ext_row[11usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((challenges[6usize]) * (current_ext_row[11usize])));
        let node_4433 = (challenges[10usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((challenges[55usize]) * (next_base_row[22usize])));
        let node_4436 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * ((challenges[56usize]) * (next_base_row[23usize]));
        let node_4447 = (node_4444)
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((challenges[56usize]) * (current_base_row[23usize])));
        let node_4484 = ((node_4476)
            * (((node_4444) + (node_4440))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((challenges[58usize]) * (next_base_row[22usize])))))
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_4470 = (((node_4433)
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((challenges[56usize]) * (current_base_row[23usize]))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((challenges[57usize])
                    * (BFieldElement::from_raw_u64(25769803770u64)))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (challenges[58usize]));
        let node_4474 = ((node_4444) + (node_4436))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((challenges[57usize])
                    * (BFieldElement::from_raw_u64(17179869180u64))));
        let node_4547 = (next_base_row[47usize])
            * ((next_base_row[47usize])
                + (BFieldElement::from_raw_u64(18446744065119617026u64)));
        let node_4616 = (challenges[12usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (next_base_row[52usize]));
        let node_4621 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_ext_row[16usize]);
        let node_4667 = ((next_base_row[51usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64)))
            * (next_base_row[51usize]);
        let node_4709 = (node_4705)
            * ((current_base_row[58usize])
                + (BFieldElement::from_raw_u64(18446744000695107601u64)));
        let node_4717 = (next_base_row[57usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[57usize]));
        let node_4708 = (current_base_row[58usize])
            + (BFieldElement::from_raw_u64(18446744000695107601u64));
        let node_4739 = (next_ext_row[23usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[23usize]));
        let node_5595 = (current_base_row[63usize])
            + (BFieldElement::from_raw_u64(18446743897615892521u64));
        let node_5597 = (current_base_row[64usize])
            + (BFieldElement::from_raw_u64(18446744047939747846u64));
        let node_5828 = (next_ext_row[24usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[24usize]));
        let node_4776 = (((((current_base_row[65usize])
            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
            + ((current_base_row[66usize])
                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
            + ((current_base_row[67usize])
                * (BFieldElement::from_raw_u64(281474976645120u64))))
            + (current_base_row[68usize])) * (BFieldElement::from_raw_u64(1u64));
        let node_4787 = (((((current_base_row[69usize])
            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
            + ((current_base_row[70usize])
                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
            + ((current_base_row[71usize])
                * (BFieldElement::from_raw_u64(281474976645120u64))))
            + (current_base_row[72usize])) * (BFieldElement::from_raw_u64(1u64));
        let node_4798 = (((((current_base_row[73usize])
            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
            + ((current_base_row[74usize])
                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
            + ((current_base_row[75usize])
                * (BFieldElement::from_raw_u64(281474976645120u64))))
            + (current_base_row[76usize])) * (BFieldElement::from_raw_u64(1u64));
        let node_4809 = (((((current_base_row[77usize])
            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
            + ((current_base_row[78usize])
                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
            + ((current_base_row[79usize])
                * (BFieldElement::from_raw_u64(281474976645120u64))))
            + (current_base_row[80usize])) * (BFieldElement::from_raw_u64(1u64));
        let node_5627 = (node_5597) * (node_5595);
        let node_5641 = ((current_base_row[62usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64)))
            * ((current_base_row[62usize])
                + (BFieldElement::from_raw_u64(18446744060824649731u64)));
        let node_5659 = (challenges[42usize])
            * ((next_base_row[103usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (current_base_row[103usize])));
        let node_5660 = (challenges[43usize])
            * ((next_base_row[104usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (current_base_row[104usize])));
        let node_5662 = (challenges[44usize])
            * ((next_base_row[105usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (current_base_row[105usize])));
        let node_5664 = (challenges[45usize])
            * ((next_base_row[106usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (current_base_row[106usize])));
        let node_5666 = (challenges[46usize])
            * ((next_base_row[107usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (current_base_row[107usize])));
        let node_5668 = (challenges[47usize])
            * ((next_base_row[108usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (current_base_row[108usize])));
        let node_5758 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (((((((((((challenges[32usize]) * (node_5492))
                + ((challenges[33usize]) * (node_5503)))
                + ((challenges[34usize]) * (node_5514)))
                + ((challenges[35usize]) * (node_5525)))
                + ((challenges[36usize]) * (next_base_row[97usize])))
                + ((challenges[37usize]) * (next_base_row[98usize])))
                + ((challenges[38usize]) * (next_base_row[99usize])))
                + ((challenges[39usize]) * (next_base_row[100usize])))
                + ((challenges[40usize]) * (next_base_row[101usize])))
                + ((challenges[41usize]) * (next_base_row[102usize])));
        let node_5735 = (next_ext_row[25usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[25usize]));
        let node_5744 = (((((challenges[32usize]) * (node_5492))
            + ((challenges[33usize]) * (node_5503)))
            + ((challenges[34usize]) * (node_5514)))
            + ((challenges[35usize]) * (node_5525)))
            + ((challenges[36usize]) * (next_base_row[97usize]));
        let node_5769 = (next_ext_row[26usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[26usize]));
        let node_5795 = (next_ext_row[27usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[27usize]));
        let node_5798 = (next_base_row[63usize])
            + (BFieldElement::from_raw_u64(18446743828896415801u64));
        let node_6142 = (next_ext_row[44usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[44usize]));
        let node_6158 = (next_ext_row[45usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[45usize]));
        let node_6153 = ((challenges[52usize]) * (next_base_row[131usize]))
            + ((challenges[53usize]) * (next_base_row[133usize]));
        let node_6156 = ((challenges[52usize]) * (next_base_row[130usize]))
            + ((challenges[53usize]) * (next_base_row[132usize]));
        let node_6196 = (next_ext_row[46usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[46usize]));
        let node_6252 = ((next_base_row[140usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[140usize])))
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_6258 = (node_6237)
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_6278 = (next_base_row[147usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_6280 = (next_base_row[147usize])
            + (BFieldElement::from_raw_u64(18446744060824649731u64));
        let node_6283 = (current_base_row[313usize]) * (next_base_row[147usize]);
        let node_6285 = (current_base_row[147usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_6250 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[140usize]);
        let node_6344 = (next_base_row[147usize]) * (next_base_row[147usize]);
        let node_6294 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (node_6237);
        let node_6360 = (next_ext_row[48usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[48usize]));
        let node_1867 = (current_base_row[12usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_1850 = (current_base_row[13usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_243 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[42usize]));
        let node_245 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[41usize]));
        let node_133 = (current_base_row[40usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_5670 = (next_base_row[64usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_5671 = (next_base_row[64usize])
            + (BFieldElement::from_raw_u64(18446744060824649731u64));
        let node_5673 = (next_base_row[64usize])
            + (BFieldElement::from_raw_u64(18446744056529682436u64));
        let node_6263 = (next_base_row[142usize])
            + (BFieldElement::from_raw_u64(18446744052234715141u64));
        let node_6265 = (next_base_row[142usize])
            + (BFieldElement::from_raw_u64(18446744009285042191u64));
        let node_5675 = (next_base_row[64usize])
            + (BFieldElement::from_raw_u64(18446744052234715141u64));
        let node_4247 = (next_base_row[12usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_4249 = (next_base_row[14usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_4251 = (next_base_row[15usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_4254 = (next_base_row[17usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_5612 = (current_base_row[62usize])
            + (BFieldElement::from_raw_u64(18446744060824649731u64));
        let node_5634 = (current_base_row[62usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_176 = (challenges[7usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (((node_167) + ((challenges[18usize]) * (next_base_row[38usize])))
                    + ((challenges[19usize]) * (next_base_row[37usize]))));
        let node_547 = (challenges[7usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (((node_167) + ((challenges[18usize]) * (current_base_row[38usize])))
                    + ((challenges[19usize]) * (current_base_row[37usize]))));
        let node_1326 = (challenges[8usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (((node_1317)
                    + (((next_base_row[22usize])
                        + (BFieldElement::from_raw_u64(4294967295u64)))
                        * (challenges[21usize])))
                    + ((next_base_row[23usize]) * (challenges[22usize]))));
        let node_1402 = (challenges[8usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (((node_1315) + ((current_base_row[22usize]) * (challenges[21usize])))
                    + ((current_base_row[23usize]) * (challenges[22usize]))));
        let node_1410 = ((current_base_row[22usize])
            + (BFieldElement::from_raw_u64(4294967295u64))) * (challenges[21usize]);
        let node_1424 = ((current_base_row[22usize])
            + (BFieldElement::from_raw_u64(8589934590u64))) * (challenges[21usize]);
        let node_1750 = (challenges[8usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (((node_1317) + ((current_base_row[22usize]) * (challenges[21usize])))
                    + ((current_base_row[39usize]) * (challenges[22usize]))));
        let node_1752 = (current_base_row[40usize]) * (challenges[22usize]);
        let node_1758 = (current_base_row[41usize]) * (challenges[22usize]);
        let node_1764 = (node_1317)
            + ((current_base_row[23usize]) * (challenges[21usize]));
        let node_1765 = (current_base_row[42usize]) * (challenges[22usize]);
        let node_1771 = (node_1317)
            + (((current_base_row[23usize])
                + (BFieldElement::from_raw_u64(4294967295u64))) * (challenges[21usize]));
        let node_1778 = (node_1317)
            + (((current_base_row[23usize])
                + (BFieldElement::from_raw_u64(8589934590u64))) * (challenges[21usize]));
        let base_constraints = [
            (next_base_row[0usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[0usize])
                        + (BFieldElement::from_raw_u64(4294967295u64)))),
            (current_base_row[6usize])
                * ((next_base_row[6usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[6usize]))),
            ((node_34) * (next_base_row[3usize]))
                + ((current_base_row[4usize])
                    * (((next_base_row[3usize]) + (node_30))
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)))),
            (current_base_row[5usize])
                * ((next_base_row[5usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (((current_base_row[5usize])
                + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                * (next_base_row[5usize]))
                * ((next_base_row[1usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (current_base_row[5usize]) * (next_base_row[1usize]),
            ((current_base_row[5usize]) * (node_34)) * (node_47),
            ((next_base_row[7usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (current_base_row[7usize])))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            (current_base_row[8usize])
                * ((next_base_row[8usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[8usize]))),
            ((((((((((((((((((((((((((((((((((((((((((((current_base_row[207usize])
                * (node_124)) + ((current_base_row[213usize]) * (node_526)))
                + ((current_base_row[209usize]) * (node_124)))
                + ((current_base_row[211usize]) * (current_base_row[328usize])))
                + ((current_base_row[232usize]) * (current_base_row[328usize])))
                + ((current_base_row[216usize]) * (node_120)))
                + ((current_base_row[221usize]) * (node_124)))
                + ((current_base_row[214usize])
                    * ((node_1241) * (current_base_row[22usize]))))
                + ((current_base_row[224usize])
                    * (((next_base_row[20usize]) + (node_112))
                        + (BFieldElement::from_raw_u64(18446744060824649731u64)))))
                + ((current_base_row[228usize]) * (node_1293)))
                + ((current_base_row[227usize]) * (node_120)))
                + ((current_base_row[229usize])
                    * (((node_1298) * (node_1295))
                        + ((current_base_row[275usize]) * (node_128)))))
                + ((current_base_row[230usize]) * (node_120)))
                + ((current_base_row[219usize]) * (node_124)))
                + ((current_base_row[226usize]) * (node_124)))
                + ((current_base_row[248usize]) * (node_124)))
                + ((current_base_row[233usize])
                    * ((current_base_row[28usize]) + (node_184))))
                + ((current_base_row[234usize]) * (node_124)))
                + ((current_base_row[250usize]) * (node_124)))
                + ((current_base_row[244usize]) * (node_120)))
                + ((current_base_row[254usize]) * (node_124)))
                + ((current_base_row[237usize]) * (node_120)))
                + ((current_base_row[239usize]) * (node_120)))
                + ((current_base_row[240usize]) * (node_120)))
                + ((current_base_row[241usize]) * ((node_1587) * (node_1590))))
                + (current_base_row[344usize]))
                + ((current_base_row[242usize]) * (node_124)))
                + ((current_base_row[243usize]) * (node_124)))
                + ((current_base_row[245usize]) * (node_124)))
                + ((current_base_row[246usize]) * (node_124)))
                + ((current_base_row[247usize]) * (node_124)))
                + ((current_base_row[249usize]) * (node_120)))
                + ((current_base_row[251usize]) * (node_124)))
                + ((current_base_row[277usize]) * ((node_1225) + (node_196))))
                + ((current_base_row[278usize])
                    * ((next_base_row[23usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((current_base_row[23usize])
                                * (current_base_row[25usize]))
                                + ((current_base_row[22usize])
                                    * (current_base_row[26usize])))
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (current_base_row[322usize]))) + (node_1625))))))
                + ((current_base_row[279usize])
                    * ((((((current_base_row[23usize]) * (next_base_row[22usize]))
                        + ((current_base_row[22usize]) * (next_base_row[23usize])))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[323usize]))) + (node_1639))
                        + (node_1642))))
                + ((current_base_row[281usize])
                    * ((next_base_row[23usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((current_base_row[24usize])
                                * (current_base_row[22usize]))))))
                + ((current_base_row[255usize]) * (node_124)))
                + ((current_base_row[258usize]) * (node_124)))
                + ((current_base_row[295usize])
                    * ((((next_base_row[27usize])
                        * (BFieldElement::from_raw_u64(8589934590u64)))
                        + (current_base_row[44usize])) + (node_200))))
                + ((current_base_row[296usize]) * (node_1743)))
                + ((current_base_row[297usize]) * (node_1743))) * (node_4096))
                + ((node_1223) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((((((current_base_row[207usize])
                * (node_116)) + ((current_base_row[213usize]) * (node_528)))
                + ((current_base_row[209usize]) * (node_116)))
                + ((current_base_row[211usize]) * (current_base_row[252usize])))
                + ((current_base_row[232usize]) * (current_base_row[252usize])))
                + ((current_base_row[216usize]) * (node_128)))
                + ((current_base_row[221usize]) * (node_1224)))
                + (current_base_row[345usize]))
                + ((current_base_row[224usize])
                    * ((next_base_row[9usize]) + (node_522))))
                + ((current_base_row[228usize]) * (node_1225)))
                + ((current_base_row[227usize]) * (node_128)))
                + ((current_base_row[229usize])
                    * (((node_1298) * (node_1292))
                        + ((current_base_row[275usize]) * (node_120)))))
                + ((current_base_row[230usize]) * (node_128)))
                + ((current_base_row[219usize]) * (node_116)))
                + ((current_base_row[226usize]) * (node_116)))
                + ((current_base_row[248usize]) * (node_1224)))
                + ((current_base_row[233usize])
                    * ((current_base_row[30usize]) + (node_192))))
                + ((current_base_row[234usize]) * (node_1224)))
                + ((current_base_row[250usize]) * (node_1224)))
                + ((current_base_row[244usize]) * (node_128)))
                + ((current_base_row[254usize]) * (node_1224)))
                + ((current_base_row[237usize]) * (node_128)))
                + ((current_base_row[239usize]) * (node_128)))
                + ((current_base_row[240usize]) * (node_128)))
                + ((current_base_row[241usize]) * (node_120)))
                + ((current_base_row[238usize]) * (node_528)))
                + ((current_base_row[242usize]) * (node_1224)))
                + ((current_base_row[243usize]) * (node_1224)))
                + ((current_base_row[245usize]) * (node_1224)))
                + ((current_base_row[246usize]) * (node_1224)))
                + ((current_base_row[247usize]) * (node_1224)))
                + ((current_base_row[249usize]) * (node_128)))
                + ((current_base_row[251usize]) * (node_1224)))
                + ((current_base_row[277usize]) * (node_329)))
                + ((current_base_row[278usize]) * (node_329)))
                + ((current_base_row[279usize]) * (node_1227)))
                + ((current_base_row[281usize]) * (node_197)))
                + ((current_base_row[255usize]) * (node_116)))
                + ((current_base_row[258usize]) * (node_116)))
                + ((current_base_row[295usize]) * (node_1231)))
                + ((current_base_row[296usize])
                    * ((node_1226)
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((current_base_row[340usize])
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (node_1797)))))))
                + ((current_base_row[297usize])
                    * ((node_1226)
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((current_base_row[40usize])
                                * (current_base_row[39usize])))))) * (node_4096))
                + ((node_120) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((((((current_base_row[207usize])
                * (current_base_row[328usize]))
                + ((current_base_row[213usize]) * (node_529)))
                + ((current_base_row[209usize]) * (current_base_row[328usize])))
                + ((current_base_row[211usize]) * (current_base_row[261usize])))
                + ((current_base_row[232usize]) * (current_base_row[261usize])))
                + ((current_base_row[216usize]) * (node_1224)))
                + ((current_base_row[221usize]) * (node_785)))
                + ((current_base_row[214usize]) * (current_base_row[271usize])))
                + ((current_base_row[224usize]) * (node_785)))
                + ((current_base_row[228usize]) * (node_1226)))
                + ((current_base_row[227usize]) * (node_785)))
                + ((current_base_row[229usize]) * (node_785)))
                + ((current_base_row[230usize]) * (node_1224)))
                + ((current_base_row[219usize]) * (current_base_row[328usize])))
                + ((current_base_row[226usize]) * (current_base_row[328usize])))
                + ((current_base_row[248usize]) * (node_414)))
                + ((current_base_row[233usize])
                    * ((current_base_row[31usize]) + (node_196))))
                + ((current_base_row[234usize]) * (node_785)))
                + ((current_base_row[250usize])
                    * ((node_158) + (BFieldElement::from_raw_u64(42949672950u64)))))
                + ((current_base_row[244usize]) * (node_1224)))
                + ((current_base_row[254usize])
                    * ((node_158)
                        + (BFieldElement::from_raw_u64(18446744026464911371u64)))))
                + ((current_base_row[237usize]) * (node_1224)))
                + ((current_base_row[239usize]) * (node_1224)))
                + ((current_base_row[240usize]) * (node_1224)))
                + ((current_base_row[241usize]) * (node_124)))
                + ((current_base_row[238usize]) * (node_529)))
                + ((current_base_row[242usize]) * (node_189)))
                + ((current_base_row[243usize]) * (node_189)))
                + ((current_base_row[245usize]) * (node_189)))
                + ((current_base_row[246usize]) * (node_1225)))
                + ((current_base_row[247usize]) * (node_189)))
                + ((current_base_row[249usize]) * (node_1224)))
                + ((current_base_row[251usize]) * (node_1225)))
                + ((current_base_row[277usize]) * (node_330)))
                + ((current_base_row[278usize]) * (node_330)))
                + ((current_base_row[279usize]) * (node_1228)))
                + ((current_base_row[281usize]) * (node_201)))
                + ((current_base_row[255usize]) * (current_base_row[328usize])))
                + ((current_base_row[258usize]) * (current_base_row[328usize])))
                + ((current_base_row[295usize]) * (node_1232)))
                + ((current_base_row[296usize])
                    * ((node_1227)
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((current_base_row[40usize])
                                * (current_base_row[42usize]))
                                + ((current_base_row[39usize])
                                    * (current_base_row[43usize])))
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (node_1798))) + (node_1797))))))
                + ((current_base_row[297usize])
                    * ((node_1227)
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((current_base_row[41usize])
                                * (current_base_row[39usize])))))) * (node_4096))
                + ((node_124) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((((((current_base_row[207usize])
                * (current_base_row[252usize]))
                + ((current_base_row[213usize]) * (node_531)))
                + ((current_base_row[209usize]) * (current_base_row[252usize])))
                + ((current_base_row[211usize]) * (node_120)))
                + ((current_base_row[232usize]) * (node_120)))
                + ((current_base_row[216usize]) * (node_1225)))
                + ((current_base_row[221usize]) * (node_1226)))
                + (current_base_row[347usize]))
                + ((current_base_row[224usize]) * (node_1226)))
                + ((current_base_row[228usize]) * (node_1228)))
                + ((current_base_row[227usize]) * (node_1226)))
                + ((current_base_row[229usize]) * (node_1226)))
                + ((current_base_row[230usize]) * (node_189)))
                + ((current_base_row[219usize]) * (current_base_row[252usize])))
                + ((current_base_row[226usize]) * (current_base_row[252usize])))
                + ((current_base_row[248usize]) * (node_416)))
                + ((current_base_row[233usize]) * (node_124)))
                + ((current_base_row[234usize]) * (node_1226)))
                + ((current_base_row[250usize]) * (node_804)))
                + ((current_base_row[244usize]) * (node_1230)))
                + ((current_base_row[254usize])
                    * ((next_base_row[32usize]) + (node_525))))
                + ((current_base_row[237usize]) * (node_193)))
                + ((current_base_row[239usize]) * (node_193)))
                + ((current_base_row[240usize]) * (node_1226)))
                + ((current_base_row[241usize]) * (node_1224)))
                + ((current_base_row[238usize]) * (node_531)))
                + ((current_base_row[242usize]) * (node_197)))
                + ((current_base_row[243usize]) * (node_197)))
                + ((current_base_row[245usize]) * (node_197)))
                + ((current_base_row[246usize]) * (node_1227)))
                + ((current_base_row[247usize]) * (node_197)))
                + ((current_base_row[249usize]) * (node_1227)))
                + ((current_base_row[251usize]) * (node_1227)))
                + ((current_base_row[277usize]) * (node_332)))
                + ((current_base_row[278usize]) * (node_332)))
                + ((current_base_row[279usize]) * (node_1230)))
                + ((current_base_row[281usize]) * (node_209)))
                + ((current_base_row[255usize]) * (current_base_row[252usize])))
                + ((current_base_row[258usize]) * (current_base_row[252usize])))
                + ((current_base_row[295usize]) * (node_1234)))
                + ((current_base_row[296usize]) * (node_120)))
                + ((current_base_row[297usize]) * (node_120))) * (node_4096))
                + ((node_785) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((((((current_base_row[207usize])
                * (current_base_row[261usize]))
                + ((current_base_row[213usize]) * (node_532)))
                + ((current_base_row[209usize]) * (current_base_row[261usize])))
                + ((current_base_row[211usize]) * (node_124)))
                + ((current_base_row[232usize]) * (node_124)))
                + ((current_base_row[216usize]) * (node_1226)))
                + ((current_base_row[221usize]) * (node_1227)))
                + ((current_base_row[214usize]) * (current_base_row[284usize])))
                + ((current_base_row[224usize]) * (node_1227)))
                + ((current_base_row[228usize]) * (node_1229)))
                + ((current_base_row[227usize]) * (node_1227)))
                + ((current_base_row[229usize]) * (node_1227)))
                + ((current_base_row[230usize]) * (node_193)))
                + ((current_base_row[219usize]) * (current_base_row[261usize])))
                + ((current_base_row[226usize]) * (current_base_row[261usize])))
                + ((current_base_row[248usize]) * (node_417)))
                + ((current_base_row[233usize]) * (node_128)))
                + ((current_base_row[234usize]) * (node_1227)))
                + ((current_base_row[250usize])
                    * ((next_base_row[23usize]) + (node_224))))
                + ((current_base_row[244usize]) * (node_1231)))
                + ((current_base_row[254usize])
                    * ((next_base_row[33usize]) + (node_184))))
                + ((current_base_row[237usize]) * (node_197)))
                + ((current_base_row[239usize]) * (node_197)))
                + ((current_base_row[240usize]) * (node_1227)))
                + ((current_base_row[241usize]) * (node_189)))
                + ((current_base_row[238usize]) * (node_532)))
                + ((current_base_row[242usize]) * (node_201)))
                + ((current_base_row[243usize]) * (node_201)))
                + ((current_base_row[245usize]) * (node_201)))
                + ((current_base_row[246usize]) * (node_1228)))
                + ((current_base_row[247usize]) * (node_201)))
                + ((current_base_row[249usize]) * (node_1228)))
                + ((current_base_row[251usize]) * (node_1228)))
                + ((current_base_row[277usize]) * (node_333)))
                + ((current_base_row[278usize]) * (node_333)))
                + ((current_base_row[279usize]) * (node_1231)))
                + ((current_base_row[281usize]) * (node_213)))
                + ((current_base_row[255usize]) * (current_base_row[261usize])))
                + ((current_base_row[258usize]) * (current_base_row[261usize])))
                + ((current_base_row[295usize]) * (node_1235)))
                + ((current_base_row[296usize]) * (node_124)))
                + ((current_base_row[297usize]) * (node_124))) * (node_4096))
                + ((node_1225) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((((((current_base_row[207usize])
                * (node_154)) + ((current_base_row[213usize]) * (node_533)))
                + ((current_base_row[209usize]) * (node_154)))
                + ((current_base_row[211usize]) * (node_128)))
                + ((current_base_row[232usize]) * (node_128)))
                + ((current_base_row[216usize]) * (node_1227)))
                + ((current_base_row[221usize]) * (node_1228)))
                + (current_base_row[348usize]))
                + ((current_base_row[224usize]) * (node_1228)))
                + ((current_base_row[228usize]) * (node_1230)))
                + ((current_base_row[227usize]) * (node_1228)))
                + ((current_base_row[229usize]) * (node_1228)))
                + ((current_base_row[230usize]) * (node_197)))
                + ((current_base_row[219usize]) * (node_154)))
                + ((current_base_row[226usize]) * (node_154)))
                + ((current_base_row[248usize]) * (node_418)))
                + ((current_base_row[233usize]) * (node_1224)))
                + ((current_base_row[234usize]) * (node_1228)))
                + ((current_base_row[250usize])
                    * ((next_base_row[24usize]) + (node_228))))
                + ((current_base_row[244usize]) * (node_1232)))
                + ((current_base_row[254usize])
                    * ((next_base_row[34usize]) + (node_188))))
                + ((current_base_row[237usize]) * (node_201)))
                + ((current_base_row[239usize]) * (node_201)))
                + ((current_base_row[240usize]) * (node_1228)))
                + ((current_base_row[241usize]) * (node_193)))
                + ((current_base_row[238usize]) * (node_533)))
                + ((current_base_row[242usize]) * (node_205)))
                + ((current_base_row[243usize]) * (node_205)))
                + ((current_base_row[245usize]) * (node_205)))
                + ((current_base_row[246usize]) * (node_1229)))
                + ((current_base_row[247usize]) * (node_205)))
                + ((current_base_row[249usize]) * (node_1229)))
                + ((current_base_row[251usize]) * (node_1229)))
                + ((current_base_row[277usize]) * (node_334)))
                + ((current_base_row[278usize]) * (node_334)))
                + ((current_base_row[279usize]) * (node_1232)))
                + ((current_base_row[281usize]) * (node_217)))
                + ((current_base_row[255usize]) * (node_154)))
                + ((current_base_row[258usize]) * (node_154)))
                + ((current_base_row[295usize]) * (node_1236)))
                + ((current_base_row[296usize]) * (node_128)))
                + ((current_base_row[297usize]) * (node_128))) * (node_4096))
                + ((node_1226) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((((((current_base_row[207usize])
                * (node_437)) + ((current_base_row[213usize]) * (node_534)))
                + ((current_base_row[209usize]) * (node_730)))
                + ((current_base_row[211usize]) * (node_116)))
                + ((current_base_row[232usize]) * (node_116)))
                + ((current_base_row[216usize]) * (node_1228)))
                + ((current_base_row[221usize]) * (node_1229)))
                + ((current_base_row[214usize]) * (node_120)))
                + ((current_base_row[224usize]) * (node_1229)))
                + ((current_base_row[228usize]) * (node_1231)))
                + ((current_base_row[227usize]) * (node_1229)))
                + ((current_base_row[229usize]) * (node_1229)))
                + ((current_base_row[230usize]) * (node_201)))
                + ((current_base_row[219usize]) * (node_730)))
                + ((current_base_row[226usize]) * (node_437)))
                + ((current_base_row[248usize]) * (node_419)))
                + ((current_base_row[233usize]) * (node_397)))
                + ((current_base_row[234usize]) * (node_1229)))
                + ((current_base_row[250usize])
                    * ((next_base_row[25usize]) + (node_232))))
                + ((current_base_row[244usize]) * (node_1233)))
                + ((current_base_row[254usize])
                    * ((next_base_row[35usize]) + (node_192))))
                + ((current_base_row[237usize]) * (node_205)))
                + ((current_base_row[239usize]) * (node_205)))
                + ((current_base_row[240usize]) * (node_1229)))
                + ((current_base_row[241usize]) * (node_197)))
                + ((current_base_row[238usize]) * (node_534)))
                + ((current_base_row[242usize]) * (node_209)))
                + ((current_base_row[243usize]) * (node_209)))
                + ((current_base_row[245usize]) * (node_209)))
                + ((current_base_row[246usize]) * (node_1230)))
                + ((current_base_row[247usize]) * (node_209)))
                + ((current_base_row[249usize]) * (node_1230)))
                + ((current_base_row[251usize]) * (node_1230)))
                + ((current_base_row[277usize]) * (node_335)))
                + ((current_base_row[278usize]) * (node_335)))
                + ((current_base_row[279usize]) * (node_1233)))
                + ((current_base_row[281usize]) * (node_221)))
                + ((current_base_row[255usize]) * (node_730)))
                + ((current_base_row[258usize]) * (node_437)))
                + ((current_base_row[295usize]) * (node_1237)))
                + ((current_base_row[296usize]) * (node_1224)))
                + ((current_base_row[297usize]) * (node_1224))) * (node_4096))
                + ((node_1227) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((current_base_row[207usize])
                * (node_461)) + ((current_base_row[213usize]) * (node_540)))
                + ((current_base_row[209usize]) * (node_754)))
                + ((current_base_row[211usize]) * (node_531)))
                + ((current_base_row[216usize]) * (node_1234)))
                + ((current_base_row[221usize]) * (node_1235)))
                + ((current_base_row[214usize]) * (node_197)))
                + ((current_base_row[224usize]) * (node_1235)))
                + ((current_base_row[228usize]) * (node_1237)))
                + ((current_base_row[227usize]) * (node_1235)))
                + ((current_base_row[229usize]) * (node_1235)))
                + ((current_base_row[230usize]) * (node_225)))
                + ((current_base_row[219usize]) * (node_750)))
                + ((current_base_row[226usize]) * (node_457)))
                + ((current_base_row[233usize]) * (node_413)))
                + ((current_base_row[234usize]) * (node_1235)))
                + ((current_base_row[244usize]) * (node_1239)))
                + ((current_base_row[237usize]) * (node_229)))
                + ((current_base_row[239usize]) * (node_229)))
                + ((current_base_row[240usize]) * (node_1235)))
                + ((current_base_row[241usize]) * (node_221)))
                + ((current_base_row[238usize]) * (node_540)))
                + ((current_base_row[242usize]) * (node_233)))
                + ((current_base_row[243usize]) * (node_233)))
                + ((current_base_row[245usize]) * (node_233)))
                + ((current_base_row[246usize]) * (node_1236)))
                + ((current_base_row[247usize]) * (node_233)))
                + ((current_base_row[249usize]) * (node_1236)))
                + ((current_base_row[251usize]) * (node_1236)))
                + ((current_base_row[277usize]) * (node_120)))
                + ((current_base_row[278usize]) * (node_120)))
                + ((current_base_row[279usize]) * (node_1239)))
                + ((current_base_row[281usize]) * (node_159)))
                + ((current_base_row[255usize]) * (node_754)))
                + ((current_base_row[258usize]) * (node_461)))
                + ((current_base_row[295usize]) * (node_124)))
                + ((current_base_row[296usize]) * (node_1232)))
                + ((current_base_row[297usize]) * (node_1232))) * (node_4096))
                + ((node_1233) * (next_base_row[8usize])),
            (((((((current_base_row[207usize]) * (current_base_row[335usize]))
                + ((current_base_row[209usize]) * (current_base_row[335usize])))
                + ((current_base_row[219usize]) * (current_base_row[334usize])))
                + ((current_base_row[226usize]) * (current_base_row[334usize])))
                + ((current_base_row[255usize]) * (current_base_row[334usize])))
                + ((current_base_row[258usize]) * (current_base_row[334usize])))
                * (node_4096),
            (((((((current_base_row[207usize]) * (current_base_row[336usize]))
                + ((current_base_row[209usize]) * (current_base_row[336usize])))
                + ((current_base_row[219usize]) * (current_base_row[335usize])))
                + ((current_base_row[226usize]) * (current_base_row[335usize])))
                + ((current_base_row[255usize]) * (current_base_row[335usize])))
                + ((current_base_row[258usize]) * (current_base_row[335usize])))
                * (node_4096),
            (((((((current_base_row[207usize]) * (current_base_row[337usize]))
                + ((current_base_row[209usize]) * (current_base_row[337usize])))
                + ((current_base_row[219usize]) * (current_base_row[336usize])))
                + ((current_base_row[226usize]) * (current_base_row[336usize])))
                + ((current_base_row[255usize]) * (current_base_row[336usize])))
                + ((current_base_row[258usize]) * (current_base_row[336usize])))
                * (node_4096),
            (((((((current_base_row[207usize]) * (current_base_row[338usize]))
                + ((current_base_row[209usize]) * (current_base_row[338usize])))
                + ((current_base_row[219usize]) * (current_base_row[337usize])))
                + ((current_base_row[226usize]) * (current_base_row[337usize])))
                + ((current_base_row[255usize]) * (current_base_row[337usize])))
                + ((current_base_row[258usize]) * (current_base_row[337usize])))
                * (node_4096),
            (((((((current_base_row[207usize]) * (current_base_row[339usize]))
                + ((current_base_row[209usize]) * (current_base_row[339usize])))
                + ((current_base_row[219usize]) * (current_base_row[338usize])))
                + ((current_base_row[226usize]) * (current_base_row[338usize])))
                + ((current_base_row[255usize]) * (current_base_row[338usize])))
                + ((current_base_row[258usize]) * (current_base_row[338usize])))
                * (node_4096),
            (node_4524) * (node_4523),
            ((node_4524)
                * ((next_base_row[49usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[49usize])))) * (next_base_row[47usize]),
            ((current_base_row[47usize])
                * ((current_base_row[47usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                * (node_4530),
            (((current_base_row[51usize])
                + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                * (current_base_row[51usize])) * (node_4593),
            (current_base_row[54usize]) * (node_4601),
            (node_4598) * (node_4601),
            ((node_4601)
                * ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (next_base_row[51usize])))
                * ((next_base_row[53usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[53usize]))),
            (node_4601)
                * ((next_base_row[55usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[55usize]))),
            (node_4601)
                * ((next_base_row[56usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[56usize]))),
            (node_4705) * (node_4704),
            (node_4709)
                * ((next_base_row[60usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[60usize]))),
            (node_4709)
                * ((next_base_row[61usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[61usize]))),
            (((node_4705)
                * ((node_4717) + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                * ((current_base_row[58usize])
                    + (BFieldElement::from_raw_u64(18446743927680663586u64))))
                * (node_4708),
            ((current_base_row[350usize])
                * ((current_base_row[64usize])
                    + (BFieldElement::from_raw_u64(18446744052234715141u64))))
                * (next_base_row[64usize]),
            (((next_base_row[62usize]) * (node_5595)) * (node_5597))
                * (((next_base_row[64usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[64usize])))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (current_base_row[355usize]) * (node_5625),
            (node_5627)
                * ((next_base_row[63usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[63usize]))),
            (node_5627)
                * ((next_base_row[62usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[62usize]))),
            ((current_base_row[351usize]) * (node_5621)) * (next_base_row[62usize]),
            (current_base_row[356usize]) * (next_base_row[62usize]),
            ((node_5641) * (node_5613)) * (next_base_row[62usize]),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(263719581847590u64))
                    * (node_4832))
                    + ((BFieldElement::from_raw_u64(76643691379275u64)) * (node_4843)))
                    + ((BFieldElement::from_raw_u64(115096533571410u64)) * (node_4854)))
                    + ((BFieldElement::from_raw_u64(256362302871255u64)) * (node_4865)))
                    + ((BFieldElement::from_raw_u64(51629801853195u64))
                        * (current_base_row[301usize])))
                    + ((BFieldElement::from_raw_u64(175668457332795u64))
                        * (current_base_row[302usize])))
                    + ((BFieldElement::from_raw_u64(177601192615545u64))
                        * (current_base_row[303usize])))
                    + ((BFieldElement::from_raw_u64(118201794925695u64))
                        * (current_base_row[304usize])))
                    + ((BFieldElement::from_raw_u64(244602682417545u64))
                        * (current_base_row[305usize])))
                    + ((BFieldElement::from_raw_u64(51685636428030u64))
                        * (current_base_row[306usize])))
                    + ((BFieldElement::from_raw_u64(231348413345175u64))
                        * (current_base_row[307usize])))
                    + ((BFieldElement::from_raw_u64(185731565704980u64))
                        * (current_base_row[308usize])))
                    + ((BFieldElement::from_raw_u64(32014686216930u64))
                        * (current_base_row[309usize])))
                    + ((BFieldElement::from_raw_u64(145268678818785u64))
                        * (current_base_row[310usize])))
                    + ((BFieldElement::from_raw_u64(123480309731250u64))
                        * (current_base_row[311usize])))
                    + ((BFieldElement::from_raw_u64(4758823762860u64))
                        * (current_base_row[312usize]))) + (current_base_row[113usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (node_5492))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(4758823762860u64))
                    * (node_4832))
                    + ((BFieldElement::from_raw_u64(263719581847590u64)) * (node_4843)))
                    + ((BFieldElement::from_raw_u64(76643691379275u64)) * (node_4854)))
                    + ((BFieldElement::from_raw_u64(115096533571410u64)) * (node_4865)))
                    + ((BFieldElement::from_raw_u64(256362302871255u64))
                        * (current_base_row[301usize])))
                    + ((BFieldElement::from_raw_u64(51629801853195u64))
                        * (current_base_row[302usize])))
                    + ((BFieldElement::from_raw_u64(175668457332795u64))
                        * (current_base_row[303usize])))
                    + ((BFieldElement::from_raw_u64(177601192615545u64))
                        * (current_base_row[304usize])))
                    + ((BFieldElement::from_raw_u64(118201794925695u64))
                        * (current_base_row[305usize])))
                    + ((BFieldElement::from_raw_u64(244602682417545u64))
                        * (current_base_row[306usize])))
                    + ((BFieldElement::from_raw_u64(51685636428030u64))
                        * (current_base_row[307usize])))
                    + ((BFieldElement::from_raw_u64(231348413345175u64))
                        * (current_base_row[308usize])))
                    + ((BFieldElement::from_raw_u64(185731565704980u64))
                        * (current_base_row[309usize])))
                    + ((BFieldElement::from_raw_u64(32014686216930u64))
                        * (current_base_row[310usize])))
                    + ((BFieldElement::from_raw_u64(145268678818785u64))
                        * (current_base_row[311usize])))
                    + ((BFieldElement::from_raw_u64(123480309731250u64))
                        * (current_base_row[312usize]))) + (current_base_row[114usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (node_5503))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(123480309731250u64))
                    * (node_4832))
                    + ((BFieldElement::from_raw_u64(4758823762860u64)) * (node_4843)))
                    + ((BFieldElement::from_raw_u64(263719581847590u64)) * (node_4854)))
                    + ((BFieldElement::from_raw_u64(76643691379275u64)) * (node_4865)))
                    + ((BFieldElement::from_raw_u64(115096533571410u64))
                        * (current_base_row[301usize])))
                    + ((BFieldElement::from_raw_u64(256362302871255u64))
                        * (current_base_row[302usize])))
                    + ((BFieldElement::from_raw_u64(51629801853195u64))
                        * (current_base_row[303usize])))
                    + ((BFieldElement::from_raw_u64(175668457332795u64))
                        * (current_base_row[304usize])))
                    + ((BFieldElement::from_raw_u64(177601192615545u64))
                        * (current_base_row[305usize])))
                    + ((BFieldElement::from_raw_u64(118201794925695u64))
                        * (current_base_row[306usize])))
                    + ((BFieldElement::from_raw_u64(244602682417545u64))
                        * (current_base_row[307usize])))
                    + ((BFieldElement::from_raw_u64(51685636428030u64))
                        * (current_base_row[308usize])))
                    + ((BFieldElement::from_raw_u64(231348413345175u64))
                        * (current_base_row[309usize])))
                    + ((BFieldElement::from_raw_u64(185731565704980u64))
                        * (current_base_row[310usize])))
                    + ((BFieldElement::from_raw_u64(32014686216930u64))
                        * (current_base_row[311usize])))
                    + ((BFieldElement::from_raw_u64(145268678818785u64))
                        * (current_base_row[312usize]))) + (current_base_row[115usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (node_5514))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(145268678818785u64))
                    * (node_4832))
                    + ((BFieldElement::from_raw_u64(123480309731250u64)) * (node_4843)))
                    + ((BFieldElement::from_raw_u64(4758823762860u64)) * (node_4854)))
                    + ((BFieldElement::from_raw_u64(263719581847590u64)) * (node_4865)))
                    + ((BFieldElement::from_raw_u64(76643691379275u64))
                        * (current_base_row[301usize])))
                    + ((BFieldElement::from_raw_u64(115096533571410u64))
                        * (current_base_row[302usize])))
                    + ((BFieldElement::from_raw_u64(256362302871255u64))
                        * (current_base_row[303usize])))
                    + ((BFieldElement::from_raw_u64(51629801853195u64))
                        * (current_base_row[304usize])))
                    + ((BFieldElement::from_raw_u64(175668457332795u64))
                        * (current_base_row[305usize])))
                    + ((BFieldElement::from_raw_u64(177601192615545u64))
                        * (current_base_row[306usize])))
                    + ((BFieldElement::from_raw_u64(118201794925695u64))
                        * (current_base_row[307usize])))
                    + ((BFieldElement::from_raw_u64(244602682417545u64))
                        * (current_base_row[308usize])))
                    + ((BFieldElement::from_raw_u64(51685636428030u64))
                        * (current_base_row[309usize])))
                    + ((BFieldElement::from_raw_u64(231348413345175u64))
                        * (current_base_row[310usize])))
                    + ((BFieldElement::from_raw_u64(185731565704980u64))
                        * (current_base_row[311usize])))
                    + ((BFieldElement::from_raw_u64(32014686216930u64))
                        * (current_base_row[312usize]))) + (current_base_row[116usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (node_5525))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(32014686216930u64))
                    * (node_4832))
                    + ((BFieldElement::from_raw_u64(145268678818785u64)) * (node_4843)))
                    + ((BFieldElement::from_raw_u64(123480309731250u64)) * (node_4854)))
                    + ((BFieldElement::from_raw_u64(4758823762860u64)) * (node_4865)))
                    + ((BFieldElement::from_raw_u64(263719581847590u64))
                        * (current_base_row[301usize])))
                    + ((BFieldElement::from_raw_u64(76643691379275u64))
                        * (current_base_row[302usize])))
                    + ((BFieldElement::from_raw_u64(115096533571410u64))
                        * (current_base_row[303usize])))
                    + ((BFieldElement::from_raw_u64(256362302871255u64))
                        * (current_base_row[304usize])))
                    + ((BFieldElement::from_raw_u64(51629801853195u64))
                        * (current_base_row[305usize])))
                    + ((BFieldElement::from_raw_u64(175668457332795u64))
                        * (current_base_row[306usize])))
                    + ((BFieldElement::from_raw_u64(177601192615545u64))
                        * (current_base_row[307usize])))
                    + ((BFieldElement::from_raw_u64(118201794925695u64))
                        * (current_base_row[308usize])))
                    + ((BFieldElement::from_raw_u64(244602682417545u64))
                        * (current_base_row[309usize])))
                    + ((BFieldElement::from_raw_u64(51685636428030u64))
                        * (current_base_row[310usize])))
                    + ((BFieldElement::from_raw_u64(231348413345175u64))
                        * (current_base_row[311usize])))
                    + ((BFieldElement::from_raw_u64(185731565704980u64))
                        * (current_base_row[312usize]))) + (current_base_row[117usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[97usize]))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(185731565704980u64))
                    * (node_4832))
                    + ((BFieldElement::from_raw_u64(32014686216930u64)) * (node_4843)))
                    + ((BFieldElement::from_raw_u64(145268678818785u64)) * (node_4854)))
                    + ((BFieldElement::from_raw_u64(123480309731250u64)) * (node_4865)))
                    + ((BFieldElement::from_raw_u64(4758823762860u64))
                        * (current_base_row[301usize])))
                    + ((BFieldElement::from_raw_u64(263719581847590u64))
                        * (current_base_row[302usize])))
                    + ((BFieldElement::from_raw_u64(76643691379275u64))
                        * (current_base_row[303usize])))
                    + ((BFieldElement::from_raw_u64(115096533571410u64))
                        * (current_base_row[304usize])))
                    + ((BFieldElement::from_raw_u64(256362302871255u64))
                        * (current_base_row[305usize])))
                    + ((BFieldElement::from_raw_u64(51629801853195u64))
                        * (current_base_row[306usize])))
                    + ((BFieldElement::from_raw_u64(175668457332795u64))
                        * (current_base_row[307usize])))
                    + ((BFieldElement::from_raw_u64(177601192615545u64))
                        * (current_base_row[308usize])))
                    + ((BFieldElement::from_raw_u64(118201794925695u64))
                        * (current_base_row[309usize])))
                    + ((BFieldElement::from_raw_u64(244602682417545u64))
                        * (current_base_row[310usize])))
                    + ((BFieldElement::from_raw_u64(51685636428030u64))
                        * (current_base_row[311usize])))
                    + ((BFieldElement::from_raw_u64(231348413345175u64))
                        * (current_base_row[312usize]))) + (current_base_row[118usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[98usize]))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(231348413345175u64))
                    * (node_4832))
                    + ((BFieldElement::from_raw_u64(185731565704980u64)) * (node_4843)))
                    + ((BFieldElement::from_raw_u64(32014686216930u64)) * (node_4854)))
                    + ((BFieldElement::from_raw_u64(145268678818785u64)) * (node_4865)))
                    + ((BFieldElement::from_raw_u64(123480309731250u64))
                        * (current_base_row[301usize])))
                    + ((BFieldElement::from_raw_u64(4758823762860u64))
                        * (current_base_row[302usize])))
                    + ((BFieldElement::from_raw_u64(263719581847590u64))
                        * (current_base_row[303usize])))
                    + ((BFieldElement::from_raw_u64(76643691379275u64))
                        * (current_base_row[304usize])))
                    + ((BFieldElement::from_raw_u64(115096533571410u64))
                        * (current_base_row[305usize])))
                    + ((BFieldElement::from_raw_u64(256362302871255u64))
                        * (current_base_row[306usize])))
                    + ((BFieldElement::from_raw_u64(51629801853195u64))
                        * (current_base_row[307usize])))
                    + ((BFieldElement::from_raw_u64(175668457332795u64))
                        * (current_base_row[308usize])))
                    + ((BFieldElement::from_raw_u64(177601192615545u64))
                        * (current_base_row[309usize])))
                    + ((BFieldElement::from_raw_u64(118201794925695u64))
                        * (current_base_row[310usize])))
                    + ((BFieldElement::from_raw_u64(244602682417545u64))
                        * (current_base_row[311usize])))
                    + ((BFieldElement::from_raw_u64(51685636428030u64))
                        * (current_base_row[312usize]))) + (current_base_row[119usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[99usize]))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(51685636428030u64))
                    * (node_4832))
                    + ((BFieldElement::from_raw_u64(231348413345175u64)) * (node_4843)))
                    + ((BFieldElement::from_raw_u64(185731565704980u64)) * (node_4854)))
                    + ((BFieldElement::from_raw_u64(32014686216930u64)) * (node_4865)))
                    + ((BFieldElement::from_raw_u64(145268678818785u64))
                        * (current_base_row[301usize])))
                    + ((BFieldElement::from_raw_u64(123480309731250u64))
                        * (current_base_row[302usize])))
                    + ((BFieldElement::from_raw_u64(4758823762860u64))
                        * (current_base_row[303usize])))
                    + ((BFieldElement::from_raw_u64(263719581847590u64))
                        * (current_base_row[304usize])))
                    + ((BFieldElement::from_raw_u64(76643691379275u64))
                        * (current_base_row[305usize])))
                    + ((BFieldElement::from_raw_u64(115096533571410u64))
                        * (current_base_row[306usize])))
                    + ((BFieldElement::from_raw_u64(256362302871255u64))
                        * (current_base_row[307usize])))
                    + ((BFieldElement::from_raw_u64(51629801853195u64))
                        * (current_base_row[308usize])))
                    + ((BFieldElement::from_raw_u64(175668457332795u64))
                        * (current_base_row[309usize])))
                    + ((BFieldElement::from_raw_u64(177601192615545u64))
                        * (current_base_row[310usize])))
                    + ((BFieldElement::from_raw_u64(118201794925695u64))
                        * (current_base_row[311usize])))
                    + ((BFieldElement::from_raw_u64(244602682417545u64))
                        * (current_base_row[312usize]))) + (current_base_row[120usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[100usize]))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(244602682417545u64))
                    * (node_4832))
                    + ((BFieldElement::from_raw_u64(51685636428030u64)) * (node_4843)))
                    + ((BFieldElement::from_raw_u64(231348413345175u64)) * (node_4854)))
                    + ((BFieldElement::from_raw_u64(185731565704980u64)) * (node_4865)))
                    + ((BFieldElement::from_raw_u64(32014686216930u64))
                        * (current_base_row[301usize])))
                    + ((BFieldElement::from_raw_u64(145268678818785u64))
                        * (current_base_row[302usize])))
                    + ((BFieldElement::from_raw_u64(123480309731250u64))
                        * (current_base_row[303usize])))
                    + ((BFieldElement::from_raw_u64(4758823762860u64))
                        * (current_base_row[304usize])))
                    + ((BFieldElement::from_raw_u64(263719581847590u64))
                        * (current_base_row[305usize])))
                    + ((BFieldElement::from_raw_u64(76643691379275u64))
                        * (current_base_row[306usize])))
                    + ((BFieldElement::from_raw_u64(115096533571410u64))
                        * (current_base_row[307usize])))
                    + ((BFieldElement::from_raw_u64(256362302871255u64))
                        * (current_base_row[308usize])))
                    + ((BFieldElement::from_raw_u64(51629801853195u64))
                        * (current_base_row[309usize])))
                    + ((BFieldElement::from_raw_u64(175668457332795u64))
                        * (current_base_row[310usize])))
                    + ((BFieldElement::from_raw_u64(177601192615545u64))
                        * (current_base_row[311usize])))
                    + ((BFieldElement::from_raw_u64(118201794925695u64))
                        * (current_base_row[312usize]))) + (current_base_row[121usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[101usize]))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(118201794925695u64))
                    * (node_4832))
                    + ((BFieldElement::from_raw_u64(244602682417545u64)) * (node_4843)))
                    + ((BFieldElement::from_raw_u64(51685636428030u64)) * (node_4854)))
                    + ((BFieldElement::from_raw_u64(231348413345175u64)) * (node_4865)))
                    + ((BFieldElement::from_raw_u64(185731565704980u64))
                        * (current_base_row[301usize])))
                    + ((BFieldElement::from_raw_u64(32014686216930u64))
                        * (current_base_row[302usize])))
                    + ((BFieldElement::from_raw_u64(145268678818785u64))
                        * (current_base_row[303usize])))
                    + ((BFieldElement::from_raw_u64(123480309731250u64))
                        * (current_base_row[304usize])))
                    + ((BFieldElement::from_raw_u64(4758823762860u64))
                        * (current_base_row[305usize])))
                    + ((BFieldElement::from_raw_u64(263719581847590u64))
                        * (current_base_row[306usize])))
                    + ((BFieldElement::from_raw_u64(76643691379275u64))
                        * (current_base_row[307usize])))
                    + ((BFieldElement::from_raw_u64(115096533571410u64))
                        * (current_base_row[308usize])))
                    + ((BFieldElement::from_raw_u64(256362302871255u64))
                        * (current_base_row[309usize])))
                    + ((BFieldElement::from_raw_u64(51629801853195u64))
                        * (current_base_row[310usize])))
                    + ((BFieldElement::from_raw_u64(175668457332795u64))
                        * (current_base_row[311usize])))
                    + ((BFieldElement::from_raw_u64(177601192615545u64))
                        * (current_base_row[312usize]))) + (current_base_row[122usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[102usize]))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(177601192615545u64))
                    * (node_4832))
                    + ((BFieldElement::from_raw_u64(118201794925695u64)) * (node_4843)))
                    + ((BFieldElement::from_raw_u64(244602682417545u64)) * (node_4854)))
                    + ((BFieldElement::from_raw_u64(51685636428030u64)) * (node_4865)))
                    + ((BFieldElement::from_raw_u64(231348413345175u64))
                        * (current_base_row[301usize])))
                    + ((BFieldElement::from_raw_u64(185731565704980u64))
                        * (current_base_row[302usize])))
                    + ((BFieldElement::from_raw_u64(32014686216930u64))
                        * (current_base_row[303usize])))
                    + ((BFieldElement::from_raw_u64(145268678818785u64))
                        * (current_base_row[304usize])))
                    + ((BFieldElement::from_raw_u64(123480309731250u64))
                        * (current_base_row[305usize])))
                    + ((BFieldElement::from_raw_u64(4758823762860u64))
                        * (current_base_row[306usize])))
                    + ((BFieldElement::from_raw_u64(263719581847590u64))
                        * (current_base_row[307usize])))
                    + ((BFieldElement::from_raw_u64(76643691379275u64))
                        * (current_base_row[308usize])))
                    + ((BFieldElement::from_raw_u64(115096533571410u64))
                        * (current_base_row[309usize])))
                    + ((BFieldElement::from_raw_u64(256362302871255u64))
                        * (current_base_row[310usize])))
                    + ((BFieldElement::from_raw_u64(51629801853195u64))
                        * (current_base_row[311usize])))
                    + ((BFieldElement::from_raw_u64(175668457332795u64))
                        * (current_base_row[312usize]))) + (current_base_row[123usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[103usize]))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(175668457332795u64))
                    * (node_4832))
                    + ((BFieldElement::from_raw_u64(177601192615545u64)) * (node_4843)))
                    + ((BFieldElement::from_raw_u64(118201794925695u64)) * (node_4854)))
                    + ((BFieldElement::from_raw_u64(244602682417545u64)) * (node_4865)))
                    + ((BFieldElement::from_raw_u64(51685636428030u64))
                        * (current_base_row[301usize])))
                    + ((BFieldElement::from_raw_u64(231348413345175u64))
                        * (current_base_row[302usize])))
                    + ((BFieldElement::from_raw_u64(185731565704980u64))
                        * (current_base_row[303usize])))
                    + ((BFieldElement::from_raw_u64(32014686216930u64))
                        * (current_base_row[304usize])))
                    + ((BFieldElement::from_raw_u64(145268678818785u64))
                        * (current_base_row[305usize])))
                    + ((BFieldElement::from_raw_u64(123480309731250u64))
                        * (current_base_row[306usize])))
                    + ((BFieldElement::from_raw_u64(4758823762860u64))
                        * (current_base_row[307usize])))
                    + ((BFieldElement::from_raw_u64(263719581847590u64))
                        * (current_base_row[308usize])))
                    + ((BFieldElement::from_raw_u64(76643691379275u64))
                        * (current_base_row[309usize])))
                    + ((BFieldElement::from_raw_u64(115096533571410u64))
                        * (current_base_row[310usize])))
                    + ((BFieldElement::from_raw_u64(256362302871255u64))
                        * (current_base_row[311usize])))
                    + ((BFieldElement::from_raw_u64(51629801853195u64))
                        * (current_base_row[312usize]))) + (current_base_row[124usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[104usize]))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(51629801853195u64))
                    * (node_4832))
                    + ((BFieldElement::from_raw_u64(175668457332795u64)) * (node_4843)))
                    + ((BFieldElement::from_raw_u64(177601192615545u64)) * (node_4854)))
                    + ((BFieldElement::from_raw_u64(118201794925695u64)) * (node_4865)))
                    + ((BFieldElement::from_raw_u64(244602682417545u64))
                        * (current_base_row[301usize])))
                    + ((BFieldElement::from_raw_u64(51685636428030u64))
                        * (current_base_row[302usize])))
                    + ((BFieldElement::from_raw_u64(231348413345175u64))
                        * (current_base_row[303usize])))
                    + ((BFieldElement::from_raw_u64(185731565704980u64))
                        * (current_base_row[304usize])))
                    + ((BFieldElement::from_raw_u64(32014686216930u64))
                        * (current_base_row[305usize])))
                    + ((BFieldElement::from_raw_u64(145268678818785u64))
                        * (current_base_row[306usize])))
                    + ((BFieldElement::from_raw_u64(123480309731250u64))
                        * (current_base_row[307usize])))
                    + ((BFieldElement::from_raw_u64(4758823762860u64))
                        * (current_base_row[308usize])))
                    + ((BFieldElement::from_raw_u64(263719581847590u64))
                        * (current_base_row[309usize])))
                    + ((BFieldElement::from_raw_u64(76643691379275u64))
                        * (current_base_row[310usize])))
                    + ((BFieldElement::from_raw_u64(115096533571410u64))
                        * (current_base_row[311usize])))
                    + ((BFieldElement::from_raw_u64(256362302871255u64))
                        * (current_base_row[312usize]))) + (current_base_row[125usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[105usize]))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(256362302871255u64))
                    * (node_4832))
                    + ((BFieldElement::from_raw_u64(51629801853195u64)) * (node_4843)))
                    + ((BFieldElement::from_raw_u64(175668457332795u64)) * (node_4854)))
                    + ((BFieldElement::from_raw_u64(177601192615545u64)) * (node_4865)))
                    + ((BFieldElement::from_raw_u64(118201794925695u64))
                        * (current_base_row[301usize])))
                    + ((BFieldElement::from_raw_u64(244602682417545u64))
                        * (current_base_row[302usize])))
                    + ((BFieldElement::from_raw_u64(51685636428030u64))
                        * (current_base_row[303usize])))
                    + ((BFieldElement::from_raw_u64(231348413345175u64))
                        * (current_base_row[304usize])))
                    + ((BFieldElement::from_raw_u64(185731565704980u64))
                        * (current_base_row[305usize])))
                    + ((BFieldElement::from_raw_u64(32014686216930u64))
                        * (current_base_row[306usize])))
                    + ((BFieldElement::from_raw_u64(145268678818785u64))
                        * (current_base_row[307usize])))
                    + ((BFieldElement::from_raw_u64(123480309731250u64))
                        * (current_base_row[308usize])))
                    + ((BFieldElement::from_raw_u64(4758823762860u64))
                        * (current_base_row[309usize])))
                    + ((BFieldElement::from_raw_u64(263719581847590u64))
                        * (current_base_row[310usize])))
                    + ((BFieldElement::from_raw_u64(76643691379275u64))
                        * (current_base_row[311usize])))
                    + ((BFieldElement::from_raw_u64(115096533571410u64))
                        * (current_base_row[312usize]))) + (current_base_row[126usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[106usize]))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(115096533571410u64))
                    * (node_4832))
                    + ((BFieldElement::from_raw_u64(256362302871255u64)) * (node_4843)))
                    + ((BFieldElement::from_raw_u64(51629801853195u64)) * (node_4854)))
                    + ((BFieldElement::from_raw_u64(175668457332795u64)) * (node_4865)))
                    + ((BFieldElement::from_raw_u64(177601192615545u64))
                        * (current_base_row[301usize])))
                    + ((BFieldElement::from_raw_u64(118201794925695u64))
                        * (current_base_row[302usize])))
                    + ((BFieldElement::from_raw_u64(244602682417545u64))
                        * (current_base_row[303usize])))
                    + ((BFieldElement::from_raw_u64(51685636428030u64))
                        * (current_base_row[304usize])))
                    + ((BFieldElement::from_raw_u64(231348413345175u64))
                        * (current_base_row[305usize])))
                    + ((BFieldElement::from_raw_u64(185731565704980u64))
                        * (current_base_row[306usize])))
                    + ((BFieldElement::from_raw_u64(32014686216930u64))
                        * (current_base_row[307usize])))
                    + ((BFieldElement::from_raw_u64(145268678818785u64))
                        * (current_base_row[308usize])))
                    + ((BFieldElement::from_raw_u64(123480309731250u64))
                        * (current_base_row[309usize])))
                    + ((BFieldElement::from_raw_u64(4758823762860u64))
                        * (current_base_row[310usize])))
                    + ((BFieldElement::from_raw_u64(263719581847590u64))
                        * (current_base_row[311usize])))
                    + ((BFieldElement::from_raw_u64(76643691379275u64))
                        * (current_base_row[312usize]))) + (current_base_row[127usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[107usize]))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(76643691379275u64))
                    * (node_4832))
                    + ((BFieldElement::from_raw_u64(115096533571410u64)) * (node_4843)))
                    + ((BFieldElement::from_raw_u64(256362302871255u64)) * (node_4854)))
                    + ((BFieldElement::from_raw_u64(51629801853195u64)) * (node_4865)))
                    + ((BFieldElement::from_raw_u64(175668457332795u64))
                        * (current_base_row[301usize])))
                    + ((BFieldElement::from_raw_u64(177601192615545u64))
                        * (current_base_row[302usize])))
                    + ((BFieldElement::from_raw_u64(118201794925695u64))
                        * (current_base_row[303usize])))
                    + ((BFieldElement::from_raw_u64(244602682417545u64))
                        * (current_base_row[304usize])))
                    + ((BFieldElement::from_raw_u64(51685636428030u64))
                        * (current_base_row[305usize])))
                    + ((BFieldElement::from_raw_u64(231348413345175u64))
                        * (current_base_row[306usize])))
                    + ((BFieldElement::from_raw_u64(185731565704980u64))
                        * (current_base_row[307usize])))
                    + ((BFieldElement::from_raw_u64(32014686216930u64))
                        * (current_base_row[308usize])))
                    + ((BFieldElement::from_raw_u64(145268678818785u64))
                        * (current_base_row[309usize])))
                    + ((BFieldElement::from_raw_u64(123480309731250u64))
                        * (current_base_row[310usize])))
                    + ((BFieldElement::from_raw_u64(4758823762860u64))
                        * (current_base_row[311usize])))
                    + ((BFieldElement::from_raw_u64(263719581847590u64))
                        * (current_base_row[312usize]))) + (current_base_row[128usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[108usize]))),
            (current_base_row[129usize]) * (node_6132),
            (current_base_row[135usize]) * (node_6184),
            ((next_base_row[135usize]) * (next_base_row[136usize]))
                + ((node_6184)
                    * (((next_base_row[136usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[136usize])))
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)))),
            ((next_base_row[139usize]) * (current_base_row[143usize])) * (node_6234),
            (next_base_row[139usize]) * (current_base_row[145usize]),
            (node_6244)
                * ((next_base_row[142usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[142usize]))),
            (((node_6244) * (current_base_row[143usize])) * (node_6234)) * (node_6252),
            ((node_6244) * (current_base_row[145usize])) * (node_6252),
            (((node_6244) * (node_6234)) * (node_6237)) * (node_6258),
            ((node_6244) * (node_6240)) * (node_6261),
            (((current_base_row[313usize]) * (node_6278)) * (node_6280))
                * (current_base_row[147usize]),
            ((node_6283) * (node_6280)) * (node_6285),
            (((current_base_row[316usize]) * (node_6258)) * (node_6240)) * (node_6285),
            (((current_base_row[316usize]) * (node_6237)) * (node_6261))
                * (current_base_row[147usize]),
            ((current_base_row[353usize])
                * ((current_base_row[139usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                * ((current_base_row[147usize])
                    + (BFieldElement::from_raw_u64(18446744060824649731u64))),
            ((current_base_row[353usize]) * (current_base_row[139usize]))
                * (current_base_row[147usize]),
            ((node_6244) * (current_base_row[352usize]))
                * (((current_base_row[147usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((BFieldElement::from_raw_u64(8589934590u64))
                            * (next_base_row[147usize]))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((node_6237) * (node_6240)))),
            (current_base_row[359usize]) * ((current_base_row[147usize]) + (node_6250)),
            ((current_base_row[327usize]) * (next_base_row[143usize]))
                * ((next_base_row[147usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[147usize]))),
            (current_base_row[354usize])
                * ((next_base_row[143usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[143usize]))),
            ((current_base_row[354usize]) * (node_6261))
                * ((current_base_row[147usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (node_6344))),
            ((current_base_row[354usize]) * (node_6240))
                * ((current_base_row[147usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[360usize]))),
            ((node_6244) * ((current_base_row[324usize]) * (node_6271)))
                * (((current_base_row[147usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[147usize]))) + (node_6294)),
            (current_base_row[169usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_1867) * (node_1850))),
            (current_base_row[170usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_243) * (node_245))),
            (current_base_row[171usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_1867) * (current_base_row[13usize]))),
            (current_base_row[172usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[169usize]) * (node_1840))),
            (current_base_row[173usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[12usize]) * (node_1850)) * (node_1840))),
            (current_base_row[174usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[171usize]) * (node_1840))),
            (current_base_row[175usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[170usize]) * (current_base_row[40usize]))),
            (current_base_row[176usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_243) * (current_base_row[41usize]))),
            (current_base_row[177usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[176usize]) * (node_248))),
            (current_base_row[178usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[172usize]) * (node_1842))),
            (current_base_row[179usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[173usize]) * (node_1842))),
            (current_base_row[180usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[172usize]) * (current_base_row[15usize]))),
            (current_base_row[181usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[174usize]) * (node_1842))),
            (current_base_row[182usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[174usize]) * (current_base_row[15usize]))),
            (current_base_row[183usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[170usize]) * (node_248))),
            (current_base_row[184usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[173usize]) * (current_base_row[15usize]))),
            (current_base_row[185usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[183usize]) * (current_base_row[39usize]))),
            (current_base_row[186usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[175usize]) * (node_295))),
            (current_base_row[187usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[175usize]) * (current_base_row[39usize]))),
            (current_base_row[188usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[178usize]) * (node_1844))),
            (current_base_row[189usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[177usize]) * (node_295))),
            (current_base_row[190usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[177usize]) * (current_base_row[39usize]))),
            (current_base_row[191usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[12usize]) * (current_base_row[13usize]))
                        * (node_1840))),
            (current_base_row[192usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[184usize]) * (node_1844))),
            (current_base_row[193usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[179usize]) * (node_1844))),
            (current_base_row[194usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[180usize]) * (node_1844))),
            (current_base_row[195usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[181usize]) * (node_1844))),
            (current_base_row[196usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[171usize]) * (current_base_row[14usize]))),
            (current_base_row[197usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[182usize]) * (node_1844))),
            (current_base_row[198usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[169usize]) * (current_base_row[14usize]))),
            (current_base_row[199usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[178usize]) * (current_base_row[16usize]))),
            (current_base_row[200usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[188usize]) * (node_1846))),
            (current_base_row[201usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[191usize]) * (node_1842))),
            (current_base_row[202usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[180usize]) * (current_base_row[16usize]))),
            (current_base_row[203usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[179usize]) * (current_base_row[16usize]))),
            (current_base_row[204usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[181usize]) * (current_base_row[16usize]))),
            (current_base_row[205usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[182usize]) * (current_base_row[16usize]))),
            (current_base_row[206usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[195usize]) * (node_1846))),
            (current_base_row[207usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[201usize]) * (node_1844)) * (node_1846))
                        * (node_1848))),
            (current_base_row[208usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[194usize]) * (node_1846))),
            (current_base_row[209usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[192usize]) * (node_1846)) * (node_1848))),
            (current_base_row[210usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[197usize]) * (node_1846))),
            (current_base_row[211usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[203usize]) * (node_1846)) * (node_1848))),
            (current_base_row[212usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[202usize]) * (node_1846))),
            (current_base_row[213usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[193usize]) * (node_1846)) * (node_1848))),
            (current_base_row[214usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[206usize]) * (node_1848))),
            (current_base_row[215usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[199usize]) * (node_1846))),
            (current_base_row[216usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[200usize]) * (node_1848))),
            (current_base_row[217usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[196usize]) * (node_1842))),
            (current_base_row[218usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[188usize]) * (current_base_row[17usize]))),
            (current_base_row[219usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[192usize]) * (current_base_row[17usize]))
                        * (node_1848))),
            (current_base_row[220usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[198usize]) * (node_1842))),
            (current_base_row[221usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[208usize]) * (node_1848))),
            (current_base_row[222usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[191usize]) * (current_base_row[15usize]))
                        * (node_1844)) * (node_1846))),
            (current_base_row[223usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[196usize]) * (current_base_row[15usize]))),
            (current_base_row[224usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[193usize]) * (current_base_row[17usize]))
                        * (node_1848))),
            (current_base_row[225usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[198usize]) * (current_base_row[15usize]))),
            (current_base_row[226usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[222usize]) * (node_1848))),
            (current_base_row[227usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[212usize]) * (node_1848))),
            (current_base_row[228usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[215usize]) * (node_1848))),
            (current_base_row[229usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[218usize]) * (node_1848))),
            (current_base_row[230usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[210usize]) * (node_1848))),
            (current_base_row[231usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[204usize]) * (node_1846))),
            (current_base_row[232usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[184usize]) * (current_base_row[16usize]))
                        * (node_1846)) * (node_1848))),
            (current_base_row[233usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[205usize]) * (node_1846)) * (node_1848))),
            (current_base_row[234usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[194usize]) * (current_base_row[17usize]))
                        * (node_1848))),
            (current_base_row[235usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[42usize]) * (node_245))),
            (current_base_row[236usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[42usize]) * (current_base_row[41usize]))),
            (current_base_row[237usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[197usize]) * (current_base_row[17usize]))
                        * (node_1848))),
            (current_base_row[238usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[220usize]) * (node_1844)) * (node_1846))
                        * (node_1848))),
            (current_base_row[239usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[204usize]) * (current_base_row[17usize]))
                        * (node_1848))),
            (current_base_row[240usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[200usize]) * (current_base_row[18usize]))),
            (current_base_row[241usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[205usize]) * (current_base_row[17usize]))
                        * (node_1848))),
            (current_base_row[242usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[217usize]) * (node_1844)) * (node_1846))
                        * (node_1848))),
            (current_base_row[243usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[223usize]) * (node_1844)) * (node_1846))
                        * (node_1848))),
            (current_base_row[244usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[199usize]) * (current_base_row[17usize]))
                        * (node_1848))),
            (current_base_row[245usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[217usize]) * (current_base_row[16usize]))
                        * (node_1846)) * (node_1848))),
            (current_base_row[246usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[225usize]) * (node_1844)) * (node_1846))
                        * (node_1848))),
            (current_base_row[247usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[223usize]) * (current_base_row[16usize]))
                        * (node_1846)) * (node_1848))),
            (current_base_row[248usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[231usize]) * (node_1848))),
            (current_base_row[249usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[220usize]) * (current_base_row[16usize]))
                        * (node_1846)) * (node_1848))),
            (current_base_row[250usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[195usize]) * (current_base_row[17usize]))
                        * (node_1848))),
            (current_base_row[251usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[225usize]) * (current_base_row[16usize]))
                        * (node_1846)) * (node_1848))),
            (current_base_row[252usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[41usize])
                        * ((current_base_row[41usize])
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))))),
            (current_base_row[253usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[97usize]) * (current_base_row[97usize]))
                        * (current_base_row[97usize])) * (current_base_row[97usize]))),
            (current_base_row[254usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[202usize]) * (current_base_row[17usize]))
                        * (node_1848))),
            (current_base_row[255usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[203usize]) * (current_base_row[17usize]))
                        * (node_1848))),
            (current_base_row[256usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[98usize]) * (current_base_row[98usize]))
                        * (current_base_row[98usize])) * (current_base_row[98usize]))),
            (current_base_row[257usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[99usize]) * (current_base_row[99usize]))
                        * (current_base_row[99usize])) * (current_base_row[99usize]))),
            (current_base_row[258usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[201usize]) * (current_base_row[16usize]))
                        * (node_1846)) * (node_1848))),
            (current_base_row[259usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[100usize]) * (current_base_row[100usize]))
                        * (current_base_row[100usize])) * (current_base_row[100usize]))),
            (current_base_row[260usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[101usize]) * (current_base_row[101usize]))
                        * (current_base_row[101usize])) * (current_base_row[101usize]))),
            (current_base_row[261usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[42usize])
                        * ((current_base_row[42usize])
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))))),
            (current_base_row[262usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[102usize]) * (current_base_row[102usize]))
                        * (current_base_row[102usize])) * (current_base_row[102usize]))),
            (current_base_row[263usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[176usize]) * (current_base_row[40usize]))),
            (current_base_row[264usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[235usize]) * (node_248))),
            (current_base_row[265usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[235usize]) * (current_base_row[40usize]))),
            (current_base_row[266usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[236usize]) * (node_248))),
            (current_base_row[267usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[236usize]) * (current_base_row[40usize]))),
            (current_base_row[268usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[103usize]) * (current_base_row[103usize]))
                        * (current_base_row[103usize])) * (current_base_row[103usize]))),
            (current_base_row[269usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[104usize]) * (current_base_row[104usize]))
                        * (current_base_row[104usize])) * (current_base_row[104usize]))),
            (current_base_row[270usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[105usize]) * (current_base_row[105usize]))
                        * (current_base_row[105usize])) * (current_base_row[105usize]))),
            (current_base_row[271usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[40usize]) * (node_133))),
            (current_base_row[272usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[106usize]) * (current_base_row[106usize]))
                        * (current_base_row[106usize])) * (current_base_row[106usize]))),
            (current_base_row[273usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[107usize]) * (current_base_row[107usize]))
                        * (current_base_row[107usize])) * (current_base_row[107usize]))),
            (current_base_row[274usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[108usize]) * (current_base_row[108usize]))
                        * (current_base_row[108usize])) * (current_base_row[108usize]))),
            (current_base_row[275usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[39usize]) * (node_1295))),
            (current_base_row[276usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[39usize]) * (current_base_row[22usize]))),
            (current_base_row[277usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[206usize]) * (current_base_row[18usize]))),
            (current_base_row[278usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[210usize]) * (current_base_row[18usize]))),
            (current_base_row[279usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[208usize]) * (current_base_row[18usize]))),
            (current_base_row[280usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((next_base_row[64usize]) * (node_5670)) * (node_5671))
                        * (node_5673))),
            (current_base_row[281usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[231usize]) * (current_base_row[18usize]))),
            (current_base_row[282usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((next_base_row[62usize]) * (node_5677)) * (node_5625))),
            (current_base_row[283usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[44usize])
                        * ((current_base_row[44usize])
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))))),
            (current_base_row[284usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[43usize])
                        * ((current_base_row[43usize])
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                        * ((current_base_row[43usize])
                            + (BFieldElement::from_raw_u64(18446744060824649731u64))))
                        * ((current_base_row[43usize])
                            + (BFieldElement::from_raw_u64(18446744056529682436u64))))),
            (current_base_row[285usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[185usize]) * (node_527))),
            (current_base_row[286usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[186usize])
                        * ((next_base_row[25usize]) + (node_184)))),
            (current_base_row[287usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[187usize])
                        * ((next_base_row[26usize]) + (node_184)))),
            (current_base_row[288usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[189usize])
                        * ((next_base_row[27usize]) + (node_184)))),
            (current_base_row[289usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[190usize])
                        * ((next_base_row[28usize]) + (node_184)))),
            (current_base_row[290usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[185usize]) * (node_189))),
            (current_base_row[291usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[186usize])
                        * ((next_base_row[23usize]) + (node_192)))),
            (current_base_row[292usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[187usize])
                        * ((next_base_row[23usize]) + (node_196)))),
            (current_base_row[293usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[189usize])
                        * ((next_base_row[23usize]) + (node_200)))),
            (current_base_row[294usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[190usize]) * (node_410))),
            (current_base_row[295usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[215usize]) * (current_base_row[18usize]))),
            (current_base_row[296usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[212usize]) * (current_base_row[18usize]))),
            (current_base_row[297usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[218usize]) * (current_base_row[18usize]))),
            (current_base_row[298usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((node_6263) * (node_6265)) * (node_6269)) * (node_6271))),
            (current_base_row[299usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_6263)
                        * ((next_base_row[142usize])
                            + (BFieldElement::from_raw_u64(18446744043644780551u64))))),
            (current_base_row[300usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((node_5670) * (node_5671)) * (node_5673)) * (node_5675))),
            (current_base_row[301usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[253usize]) * (current_base_row[97usize]))
                        * (current_base_row[97usize])) * (current_base_row[97usize]))),
            (current_base_row[302usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[256usize]) * (current_base_row[98usize]))
                        * (current_base_row[98usize])) * (current_base_row[98usize]))),
            (current_base_row[303usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[257usize]) * (current_base_row[99usize]))
                        * (current_base_row[99usize])) * (current_base_row[99usize]))),
            (current_base_row[304usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[259usize]) * (current_base_row[100usize]))
                        * (current_base_row[100usize])) * (current_base_row[100usize]))),
            (current_base_row[305usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[260usize]) * (current_base_row[101usize]))
                        * (current_base_row[101usize])) * (current_base_row[101usize]))),
            (current_base_row[306usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[262usize]) * (current_base_row[102usize]))
                        * (current_base_row[102usize])) * (current_base_row[102usize]))),
            (current_base_row[307usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[268usize]) * (current_base_row[103usize]))
                        * (current_base_row[103usize])) * (current_base_row[103usize]))),
            (current_base_row[308usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[269usize]) * (current_base_row[104usize]))
                        * (current_base_row[104usize])) * (current_base_row[104usize]))),
            (current_base_row[309usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[270usize]) * (current_base_row[105usize]))
                        * (current_base_row[105usize])) * (current_base_row[105usize]))),
            (current_base_row[310usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[272usize]) * (current_base_row[106usize]))
                        * (current_base_row[106usize])) * (current_base_row[106usize]))),
            (current_base_row[311usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[273usize]) * (current_base_row[107usize]))
                        * (current_base_row[107usize])) * (current_base_row[107usize]))),
            (current_base_row[312usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[274usize]) * (current_base_row[108usize]))
                        * (current_base_row[108usize])) * (current_base_row[108usize]))),
            (current_base_row[313usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_6244) * ((current_base_row[298usize]) * (node_6275)))),
            (current_base_row[314usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[299usize]) * (node_6265))),
            (current_base_row[315usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[300usize]) * (node_5677))),
            (current_base_row[316usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_6283) * (node_6278))),
            (current_base_row[317usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[39usize]) * (node_1587))),
            (current_base_row[318usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((node_5616) * (node_5637)) * (next_base_row[62usize]))),
            (current_base_row[319usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((node_4247) * (next_base_row[13usize])) * (node_4249))
                        * (node_4251))),
            (current_base_row[320usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[23usize]) * (next_base_row[23usize]))),
            (current_base_row[321usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[22usize]) * (current_base_row[25usize]))),
            (current_base_row[322usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[24usize]) * (current_base_row[27usize]))),
            (current_base_row[323usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[24usize]) * (next_base_row[24usize]))),
            (current_base_row[324usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[314usize]) * (node_6269))),
            (current_base_row[325usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((node_4247)
                        * ((next_base_row[13usize])
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                        * (node_4249)) * (node_4251))),
            (current_base_row[326usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((((current_base_row[10usize])
                        + (BFieldElement::from_raw_u64(18446743897615892521u64)))
                        * ((current_base_row[10usize])
                            + (BFieldElement::from_raw_u64(18446743923385696291u64))))
                        * ((current_base_row[10usize])
                            + (BFieldElement::from_raw_u64(18446743863256154161u64))))
                        * ((current_base_row[10usize])
                            + (BFieldElement::from_raw_u64(18446743828896415801u64))))),
            (current_base_row[327usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_6244)
                        * (((current_base_row[314usize]) * (node_6271)) * (node_6275)))),
            (current_base_row[328usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[39usize])
                        * ((current_base_row[39usize])
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))))),
            (current_base_row[329usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[183usize]) * (node_295))),
            (current_base_row[330usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[263usize]) * (node_295))),
            (current_base_row[331usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[263usize]) * (current_base_row[39usize]))),
            (current_base_row[332usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[264usize]) * (node_295))),
            (current_base_row[333usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[264usize]) * (current_base_row[39usize]))),
            (current_base_row[334usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[265usize]) * (node_295))),
            (current_base_row[335usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[265usize]) * (current_base_row[39usize]))),
            (current_base_row[336usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[266usize]) * (node_295))),
            (current_base_row[337usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[266usize]) * (current_base_row[39usize]))),
            (current_base_row[338usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[267usize]) * (node_295))),
            (current_base_row[339usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[267usize]) * (current_base_row[39usize]))),
            (current_base_row[340usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[39usize]) * (current_base_row[42usize]))),
            (current_base_row[341usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[319usize]) * (next_base_row[16usize]))
                        * (node_4254))
                        * ((next_base_row[18usize])
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))))),
            (current_base_row[342usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[315usize])
                        * (((node_5637) * (node_5621)) * (next_base_row[62usize])))),
            (current_base_row[343usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((node_5612) * (node_5613)) * (current_base_row[62usize]))),
            (current_base_row[344usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[238usize])
                        * ((next_base_row[22usize])
                            * (((current_base_row[39usize])
                                * ((next_base_row[23usize])
                                    + (BFieldElement::from_raw_u64(4294967296u64))))
                                + (BFieldElement::from_raw_u64(
                                    18446744065119617026u64,
                                )))))),
            (current_base_row[345usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[214usize])
                        * ((((node_1224) * (current_base_row[22usize]))
                            + (((node_116) * (node_1241)) * (node_133)))
                            + ((((node_113)
                                + (BFieldElement::from_raw_u64(18446744056529682436u64)))
                                * (node_1241)) * (current_base_row[40usize]))))),
            (current_base_row[346usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[214usize])
                        * (((current_base_row[252usize])
                            * ((current_base_row[41usize])
                                + (BFieldElement::from_raw_u64(18446744060824649731u64))))
                            * ((current_base_row[41usize])
                                + (BFieldElement::from_raw_u64(
                                    18446744056529682436u64,
                                )))))),
            (current_base_row[347usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[214usize])
                        * (((current_base_row[261usize])
                            * ((current_base_row[42usize])
                                + (BFieldElement::from_raw_u64(18446744060824649731u64))))
                            * ((current_base_row[42usize])
                                + (BFieldElement::from_raw_u64(
                                    18446744056529682436u64,
                                )))))),
            (current_base_row[348usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[214usize])
                        * (((current_base_row[283usize])
                            * ((current_base_row[44usize])
                                + (BFieldElement::from_raw_u64(18446744060824649731u64))))
                            * ((current_base_row[44usize])
                                + (BFieldElement::from_raw_u64(
                                    18446744056529682436u64,
                                )))))),
            (current_base_row[349usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[325usize]) * (next_base_row[16usize]))
                        * (node_4254)) * (next_base_row[18usize]))),
            (current_base_row[350usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[64usize])
                        * ((current_base_row[64usize])
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                        * ((current_base_row[64usize])
                            + (BFieldElement::from_raw_u64(18446744060824649731u64))))
                        * ((current_base_row[64usize])
                            + (BFieldElement::from_raw_u64(18446744056529682436u64))))),
            (current_base_row[351usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((node_5634) * (node_5613)) * (current_base_row[62usize]))
                        * (node_5637))),
            (current_base_row[352usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[299usize]) * (node_6269)) * (node_6271))
                        * (node_6275))),
            (current_base_row[353usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[316usize])
                        * ((((BFieldElement::from_raw_u64(4294967295u64)) + (node_6294))
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (node_6240)))
                            + (((BFieldElement::from_raw_u64(8589934590u64))
                                * (node_6237)) * (node_6240))))),
            (current_base_row[354usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_6244) * ((current_base_row[324usize]) * (node_6275)))),
            (current_base_row[355usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[343usize])
                        * (((node_5616) * (node_5621)) * (next_base_row[62usize])))),
            (current_base_row[356usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((node_5641) * (current_base_row[62usize])) * (node_5621))),
            (current_base_row[357usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[315usize]) * (node_5621))
                        * (next_base_row[62usize])) * (node_5625))),
            (current_base_row[358usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[315usize])
                        * (((node_5728) * (node_5625)) * (node_5730)))),
            (current_base_row[359usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[327usize])
                        * ((BFieldElement::from_raw_u64(4294967295u64))
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((next_base_row[143usize]) * (next_base_row[144usize])))))
                        * (current_base_row[143usize]))),
            (current_base_row[360usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_6344) * (current_base_row[143usize]))),
        ];
        let ext_constraints = [
            (((BFieldElement::from_raw_u64(4294967295u64))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (current_base_row[5usize])))
                * (((node_51)
                    * ((challenges[3usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((((challenges[13usize]) * (current_base_row[0usize]))
                                + ((challenges[14usize]) * (current_base_row[1usize])))
                                + ((challenges[15usize]) * (next_base_row[1usize]))))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[2usize]))))
                + ((current_base_row[5usize]) * (node_51)),
            ((node_31)
                * (((next_ext_row[1usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((challenges[29usize]) * (current_ext_row[1usize]))))
                    + (node_74)))
                + ((node_34)
                    * (((next_ext_row[1usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (challenges[29usize]))) + (node_74))),
            ((((((next_ext_row[2usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((challenges[30usize]) * (current_ext_row[2usize]))))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (next_ext_row[1usize]))) * (node_47))
                * ((BFieldElement::from_raw_u64(4294967295u64))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((next_base_row[4usize]) * (node_90)))))
                + ((node_88) * (next_base_row[6usize]))) + ((node_88) * (node_90)),
            ((((((((((((((((((((((((((((((((((((((((((((current_base_row[207usize])
                * (node_120))
                + ((current_base_row[213usize])
                    * ((next_base_row[22usize]) + (node_522))))
                + ((current_base_row[209usize]) * (node_120)))
                + ((current_base_row[211usize])
                    * (((((((((((((((((current_base_row[329usize]) * (node_785))
                        + (node_253)) + (node_299)) + (node_342)) + (node_384))
                        + (node_423))
                        + ((current_base_row[330usize])
                            * ((next_base_row[22usize]) + (node_204))))
                        + ((current_base_row[331usize])
                            * ((next_base_row[22usize]) + (node_208))))
                        + ((current_base_row[332usize])
                            * ((next_base_row[22usize]) + (node_212))))
                        + ((current_base_row[333usize])
                            * ((next_base_row[22usize]) + (node_216))))
                        + ((current_base_row[334usize]) * (node_804)))
                        + ((current_base_row[335usize])
                            * ((next_base_row[22usize]) + (node_224))))
                        + ((current_base_row[336usize])
                            * ((next_base_row[22usize]) + (node_228))))
                        + ((current_base_row[337usize])
                            * ((next_base_row[22usize]) + (node_232))))
                        + ((current_base_row[338usize])
                            * ((next_base_row[22usize]) + (node_236))))
                        + ((current_base_row[339usize])
                            * ((next_base_row[22usize]) + (node_240))))))
                + ((current_base_row[232usize])
                    * (((((((((((((((((current_base_row[329usize])
                        * ((node_868)
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((((((node_887) + (node_888)) + (node_890)) + (node_892))
                                    + (node_894)) + (node_896)) + (node_898)))))
                        + ((current_base_row[185usize])
                            * ((node_868)
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (((((((((((((((((challenges[32usize])
                                        * (current_base_row[23usize]))
                                        + ((challenges[33usize]) * (current_base_row[22usize])))
                                        + (node_872)) + (node_874)) + (node_876)) + (node_878))
                                        + (node_880)) + (node_882)) + (node_884)) + (node_886))
                                        + (node_888)) + (node_890)) + (node_892)) + (node_894))
                                        + (node_896)) + (node_898))))))
                        + ((current_base_row[186usize])
                            * ((node_868)
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (((((((((((((((((challenges[32usize])
                                        * (current_base_row[24usize])) + (node_870))
                                        + ((challenges[34usize]) * (current_base_row[22usize])))
                                        + (node_874)) + (node_876)) + (node_878)) + (node_880))
                                        + (node_882)) + (node_884)) + (node_886)) + (node_888))
                                        + (node_890)) + (node_892)) + (node_894)) + (node_896))
                                        + (node_898))))))
                        + ((current_base_row[187usize])
                            * ((node_868)
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (((((((((((((((((challenges[32usize])
                                        * (current_base_row[25usize])) + (node_870)) + (node_872))
                                        + ((challenges[35usize]) * (current_base_row[22usize])))
                                        + (node_876)) + (node_878)) + (node_880)) + (node_882))
                                        + (node_884)) + (node_886)) + (node_888)) + (node_890))
                                        + (node_892)) + (node_894)) + (node_896)) + (node_898))))))
                        + ((current_base_row[189usize])
                            * ((node_868)
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (((((((((((((((((challenges[32usize])
                                        * (current_base_row[26usize])) + (node_870)) + (node_872))
                                        + (node_874))
                                        + ((challenges[36usize]) * (current_base_row[22usize])))
                                        + (node_878)) + (node_880)) + (node_882)) + (node_884))
                                        + (node_886)) + (node_888)) + (node_890)) + (node_892))
                                        + (node_894)) + (node_896)) + (node_898))))))
                        + ((current_base_row[190usize])
                            * ((node_868)
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (((((((((((((((((challenges[32usize])
                                        * (current_base_row[27usize])) + (node_870)) + (node_872))
                                        + (node_874)) + (node_876))
                                        + ((challenges[37usize]) * (current_base_row[22usize])))
                                        + (node_880)) + (node_882)) + (node_884)) + (node_886))
                                        + (node_888)) + (node_890)) + (node_892)) + (node_894))
                                        + (node_896)) + (node_898))))))
                        + ((current_base_row[330usize])
                            * ((node_868)
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (((((((((((((((((challenges[32usize])
                                        * (current_base_row[28usize])) + (node_870)) + (node_872))
                                        + (node_874)) + (node_876)) + (node_878))
                                        + ((challenges[38usize]) * (current_base_row[22usize])))
                                        + (node_882)) + (node_884)) + (node_886)) + (node_888))
                                        + (node_890)) + (node_892)) + (node_894)) + (node_896))
                                        + (node_898))))))
                        + ((current_base_row[331usize])
                            * ((node_868)
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (((((((((((((((((challenges[32usize])
                                        * (current_base_row[29usize])) + (node_870)) + (node_872))
                                        + (node_874)) + (node_876)) + (node_878)) + (node_880))
                                        + ((challenges[39usize]) * (current_base_row[22usize])))
                                        + (node_884)) + (node_886)) + (node_888)) + (node_890))
                                        + (node_892)) + (node_894)) + (node_896)) + (node_898))))))
                        + ((current_base_row[332usize])
                            * ((node_868)
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (((((((((((((((((challenges[32usize])
                                        * (current_base_row[30usize])) + (node_870)) + (node_872))
                                        + (node_874)) + (node_876)) + (node_878)) + (node_880))
                                        + (node_882))
                                        + ((challenges[40usize]) * (current_base_row[22usize])))
                                        + (node_886)) + (node_888)) + (node_890)) + (node_892))
                                        + (node_894)) + (node_896)) + (node_898))))))
                        + ((current_base_row[333usize])
                            * ((node_868)
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (((((((((((((((((challenges[32usize])
                                        * (current_base_row[31usize])) + (node_870)) + (node_872))
                                        + (node_874)) + (node_876)) + (node_878)) + (node_880))
                                        + (node_882)) + (node_884))
                                        + ((challenges[41usize]) * (current_base_row[22usize])))
                                        + (node_888)) + (node_890)) + (node_892)) + (node_894))
                                        + (node_896)) + (node_898))))))
                        + ((current_base_row[334usize])
                            * ((node_868)
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (((((((((((((((((challenges[32usize])
                                        * (current_base_row[32usize])) + (node_870)) + (node_872))
                                        + (node_874)) + (node_876)) + (node_878)) + (node_880))
                                        + (node_882)) + (node_884)) + (node_886))
                                        + ((challenges[42usize]) * (current_base_row[22usize])))
                                        + (node_890)) + (node_892)) + (node_894)) + (node_896))
                                        + (node_898))))))
                        + ((current_base_row[335usize])
                            * ((node_868)
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (((((((((((((((((challenges[32usize])
                                        * (current_base_row[33usize])) + (node_870)) + (node_872))
                                        + (node_874)) + (node_876)) + (node_878)) + (node_880))
                                        + (node_882)) + (node_884)) + (node_886)) + (node_888))
                                        + ((challenges[43usize]) * (current_base_row[22usize])))
                                        + (node_892)) + (node_894)) + (node_896)) + (node_898))))))
                        + ((current_base_row[336usize])
                            * ((node_868)
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (((((((((((((((((challenges[32usize])
                                        * (current_base_row[34usize])) + (node_870)) + (node_872))
                                        + (node_874)) + (node_876)) + (node_878)) + (node_880))
                                        + (node_882)) + (node_884)) + (node_886)) + (node_888))
                                        + (node_890))
                                        + ((challenges[44usize]) * (current_base_row[22usize])))
                                        + (node_894)) + (node_896)) + (node_898))))))
                        + ((current_base_row[337usize])
                            * ((node_868)
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (((((((((((((((((challenges[32usize])
                                        * (current_base_row[35usize])) + (node_870)) + (node_872))
                                        + (node_874)) + (node_876)) + (node_878)) + (node_880))
                                        + (node_882)) + (node_884)) + (node_886)) + (node_888))
                                        + (node_890)) + (node_892))
                                        + ((challenges[45usize]) * (current_base_row[22usize])))
                                        + (node_896)) + (node_898))))))
                        + ((current_base_row[338usize])
                            * ((node_868)
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (((((((((((((((((challenges[32usize])
                                        * (current_base_row[36usize])) + (node_870)) + (node_872))
                                        + (node_874)) + (node_876)) + (node_878)) + (node_880))
                                        + (node_882)) + (node_884)) + (node_886)) + (node_888))
                                        + (node_890)) + (node_892)) + (node_894))
                                        + ((challenges[46usize]) * (current_base_row[22usize])))
                                        + (node_898))))))
                        + ((current_base_row[339usize])
                            * ((node_868)
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (((((((((((((((((challenges[32usize])
                                        * (current_base_row[37usize])) + (node_870)) + (node_872))
                                        + (node_874)) + (node_876)) + (node_878)) + (node_880))
                                        + (node_882)) + (node_884)) + (node_886)) + (node_888))
                                        + (node_890)) + (node_892)) + (node_894)) + (node_896))
                                        + ((challenges[47usize])
                                            * (current_base_row[22usize])))))))))
                + ((current_base_row[216usize]) * (node_1223)))
                + ((current_base_row[221usize]) * (node_120)))
                + ((current_base_row[214usize])
                    * ((node_1241) * (current_base_row[39usize]))))
                + ((current_base_row[224usize])
                    * ((node_120)
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)))))
                + ((current_base_row[228usize]) * (node_1292)))
                + ((current_base_row[227usize]) * (node_1294)))
                + ((current_base_row[229usize])
                    * (((node_1298) * (current_base_row[39usize]))
                        + ((current_base_row[275usize]) * (node_124)))))
                + ((current_base_row[230usize])
                    * ((current_base_row[22usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)))))
                + ((current_base_row[219usize]) * (node_120)))
                + ((current_base_row[226usize]) * (node_120)))
                + ((current_base_row[248usize]) * (node_120)))
                + ((current_base_row[233usize])
                    * ((current_base_row[27usize]) + (node_525))))
                + ((current_base_row[234usize]) * (node_120)))
                + ((current_base_row[250usize]) * (node_120)))
                + ((current_base_row[244usize])
                    * ((node_785)
                        + (BFieldElement::from_raw_u64(18446744026464911371u64)))))
                + ((current_base_row[254usize]) * (node_120)))
                + ((current_base_row[237usize]) * ((node_785) + (node_184))))
                + ((current_base_row[239usize]) * (node_1584)))
                + ((current_base_row[240usize])
                    * ((node_1585)
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)))))
                + ((current_base_row[241usize])
                    * ((current_base_row[39usize]) * (node_1590))))
                + ((current_base_row[238usize])
                    * ((current_base_row[22usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((BFieldElement::from_raw_u64(18446744069414584320u64))
                                * (next_base_row[23usize])) + (next_base_row[22usize]))))))
                + ((current_base_row[242usize]) * (node_120)))
                + ((current_base_row[243usize]) * (node_120)))
                + ((current_base_row[245usize]) * (node_120)))
                + ((current_base_row[246usize]) * (node_120)))
                + ((current_base_row[247usize]) * (node_120)))
                + ((current_base_row[249usize])
                    * (((current_base_row[22usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[320usize]))) + (node_1609))))
                + ((current_base_row[251usize]) * (node_120)))
                + ((current_base_row[277usize]) * ((node_785) + (node_192))))
                + ((current_base_row[278usize])
                    * ((next_base_row[22usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((current_base_row[321usize])
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (node_1625)))))))
                + ((current_base_row[279usize])
                    * ((((node_1585)
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_1639)))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_1642)))
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)))))
                + ((current_base_row[281usize]) * (node_1584)))
                + ((current_base_row[255usize]) * (node_120)))
                + ((current_base_row[258usize]) * (node_120)))
                + ((current_base_row[295usize]) * (current_base_row[283usize])))
                + ((current_base_row[296usize]) * (node_1422)))
                + ((current_base_row[297usize]) * (node_1396))) * (node_4096))
                + ((node_113) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((((((current_base_row[207usize])
                * (node_128)) + ((current_base_row[213usize]) * (node_527)))
                + ((current_base_row[209usize]) * (node_128)))
                + ((current_base_row[211usize]) * (current_base_row[271usize])))
                + ((current_base_row[232usize]) * (current_base_row[271usize])))
                + ((current_base_row[216usize]) * (node_124)))
                + ((current_base_row[221usize]) * (node_128)))
                + ((current_base_row[214usize])
                    * ((((((current_base_row[11usize]) + (node_247))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((BFieldElement::from_raw_u64(8589934590u64))
                                * (current_base_row[41usize])))) + (node_144))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((BFieldElement::from_raw_u64(137438953440u64))
                                * (current_base_row[43usize]))))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((BFieldElement::from_raw_u64(549755813760u64))
                                * (current_base_row[44usize]))))))
                + ((current_base_row[224usize])
                    * ((next_base_row[21usize]) + (node_522))))
                + ((current_base_row[228usize]) * (node_785)))
                + ((current_base_row[227usize]) * (node_124)))
                + ((current_base_row[229usize])
                    * (((node_1298) * (node_1293))
                        + ((current_base_row[275usize]) * (node_1294)))))
                + ((current_base_row[230usize]) * (node_124)))
                + ((current_base_row[219usize]) * (node_128)))
                + ((current_base_row[226usize]) * (node_128)))
                + ((current_base_row[248usize]) * (node_128)))
                + ((current_base_row[233usize])
                    * ((current_base_row[29usize]) + (node_188))))
                + ((current_base_row[234usize]) * (node_128)))
                + ((current_base_row[250usize]) * (node_128)))
                + ((current_base_row[244usize]) * (node_124)))
                + ((current_base_row[254usize]) * (node_128)))
                + ((current_base_row[237usize]) * (node_124)))
                + ((current_base_row[239usize]) * (node_124)))
                + ((current_base_row[240usize]) * (node_124)))
                + ((current_base_row[241usize])
                    * ((next_base_row[22usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_1590)))))
                + ((current_base_row[238usize]) * (node_527)))
                + ((current_base_row[242usize]) * (node_128)))
                + ((current_base_row[243usize]) * (node_128)))
                + ((current_base_row[245usize]) * (node_128)))
                + ((current_base_row[246usize]) * (node_128)))
                + ((current_base_row[247usize]) * (node_128)))
                + ((current_base_row[249usize]) * (node_124)))
                + ((current_base_row[251usize]) * (node_128)))
                + ((current_base_row[277usize]) * ((node_1226) + (node_200))))
                + ((current_base_row[278usize])
                    * ((next_base_row[24usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((current_base_row[24usize])
                                * (current_base_row[25usize]))
                                + ((current_base_row[23usize])
                                    * (current_base_row[26usize])))
                                + ((current_base_row[22usize])
                                    * (current_base_row[27usize])))
                                + (current_base_row[322usize]))))))
                + ((current_base_row[279usize])
                    * (((((current_base_row[24usize]) * (next_base_row[22usize]))
                        + (current_base_row[320usize]))
                        + ((current_base_row[22usize]) * (next_base_row[24usize])))
                        + (current_base_row[323usize]))))
                + ((current_base_row[281usize])
                    * ((next_base_row[24usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[321usize])))))
                + ((current_base_row[255usize]) * (node_128)))
                + ((current_base_row[258usize]) * (node_128)))
                + ((current_base_row[295usize]) * (node_1230)))
                + ((current_base_row[296usize])
                    * ((current_ext_row[82usize]) + (node_1785))))
                + ((current_base_row[297usize])
                    * (((current_ext_row[7usize]) * (current_ext_row[72usize]))
                        + (node_1785)))) * (node_4096))
                + (((next_base_row[11usize]) + (node_522)) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((((((current_base_row[207usize])
                * (current_base_row[271usize]))
                + ((current_base_row[213usize]) * (node_530)))
                + ((current_base_row[209usize]) * (current_base_row[271usize])))
                + ((current_base_row[211usize]) * (node_154)))
                + ((current_base_row[232usize]) * (node_154)))
                + ((current_base_row[216usize]) * (node_785)))
                + ((current_base_row[221usize]) * (node_1225)))
                + (current_base_row[346usize]))
                + ((current_base_row[224usize]) * (node_1225)))
                + ((current_base_row[228usize]) * (node_1227)))
                + ((current_base_row[227usize]) * (node_1225)))
                + ((current_base_row[229usize]) * (node_1225)))
                + ((current_base_row[230usize]) * (node_185)))
                + ((current_base_row[219usize]) * (current_base_row[271usize])))
                + ((current_base_row[226usize]) * (current_base_row[271usize])))
                + ((current_base_row[248usize]) * (node_415)))
                + ((current_base_row[233usize]) * (node_120)))
                + ((current_base_row[234usize]) * (node_1225)))
                + ((current_base_row[250usize])
                    * ((next_ext_row[6usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_ext_row[69usize])))))
                + ((current_base_row[244usize]) * (node_1229)))
                + ((current_base_row[254usize])
                    * ((next_ext_row[6usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_ext_row[70usize])))))
                + ((current_base_row[237usize]) * (node_189)))
                + ((current_base_row[239usize]) * (node_189)))
                + ((current_base_row[240usize]) * (node_1225)))
                + ((current_base_row[241usize]) * (node_128)))
                + ((current_base_row[238usize]) * (node_530)))
                + ((current_base_row[242usize]) * (node_193)))
                + ((current_base_row[243usize]) * (node_193)))
                + ((current_base_row[245usize]) * (node_193)))
                + ((current_base_row[246usize]) * (node_1226)))
                + ((current_base_row[247usize]) * (node_193)))
                + ((current_base_row[249usize]) * (node_1226)))
                + ((current_base_row[251usize]) * (node_1226)))
                + ((current_base_row[277usize]) * (node_331)))
                + ((current_base_row[278usize]) * (node_331)))
                + ((current_base_row[279usize]) * (node_1229)))
                + ((current_base_row[281usize]) * (node_205)))
                + ((current_base_row[255usize]) * (current_base_row[271usize])))
                + ((current_base_row[258usize]) * (current_base_row[271usize])))
                + ((current_base_row[295usize]) * (node_1233)))
                + ((current_base_row[296usize])
                    * ((node_1228)
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((((current_base_row[236usize])
                                + ((current_base_row[40usize])
                                    * (current_base_row[43usize])))
                                + ((current_base_row[39usize])
                                    * (current_base_row[44usize]))) + (node_1798))))))
                + ((current_base_row[297usize])
                    * ((node_1228)
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[340usize]))))) * (node_4096))
                + ((node_128) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((((((current_base_row[207usize])
                * (node_441)) + ((current_base_row[213usize]) * (node_535)))
                + ((current_base_row[209usize]) * (node_734)))
                + ((current_base_row[211usize]) * (node_526)))
                + ((current_base_row[232usize]) * (node_513)))
                + ((current_base_row[216usize]) * (node_1229)))
                + ((current_base_row[221usize]) * (node_1230)))
                + ((current_base_row[214usize]) * (node_124)))
                + ((current_base_row[224usize]) * (node_1230)))
                + ((current_base_row[228usize]) * (node_1232)))
                + ((current_base_row[227usize]) * (node_1230)))
                + ((current_base_row[229usize]) * (node_1230)))
                + ((current_base_row[230usize]) * (node_205)))
                + ((current_base_row[219usize])
                    * ((((((current_base_row[185usize])
                        * ((node_785) + (BFieldElement::from_raw_u64(4294967295u64))))
                        + ((current_base_row[186usize])
                            * ((node_785)
                                + (BFieldElement::from_raw_u64(8589934590u64)))))
                        + ((current_base_row[187usize])
                            * ((node_785)
                                + (BFieldElement::from_raw_u64(12884901885u64)))))
                        + ((current_base_row[189usize])
                            * ((node_785)
                                + (BFieldElement::from_raw_u64(17179869180u64)))))
                        + ((current_base_row[190usize])
                            * ((node_785)
                                + (BFieldElement::from_raw_u64(21474836475u64)))))))
                + ((current_base_row[226usize])
                    * ((((((current_base_row[185usize]) * (node_1396))
                        + ((current_base_row[186usize])
                            * ((node_785)
                                + (BFieldElement::from_raw_u64(18446744060824649731u64)))))
                        + ((current_base_row[187usize]) * (node_1422)))
                        + ((current_base_row[189usize])
                            * ((node_785)
                                + (BFieldElement::from_raw_u64(18446744052234715141u64)))))
                        + ((current_base_row[190usize])
                            * ((node_785)
                                + (BFieldElement::from_raw_u64(
                                    18446744047939747846u64,
                                ))))))) + ((current_base_row[248usize]) * (node_397)))
                + ((current_base_row[233usize]) * (node_408)))
                + ((current_base_row[234usize]) * (node_1230)))
                + ((current_base_row[250usize])
                    * ((next_base_row[26usize]) + (node_236))))
                + ((current_base_row[244usize]) * (node_1234)))
                + ((current_base_row[254usize])
                    * ((next_base_row[36usize]) + (node_196))))
                + ((current_base_row[237usize]) * (node_209)))
                + ((current_base_row[239usize]) * (node_209)))
                + ((current_base_row[240usize]) * (node_1230)))
                + ((current_base_row[241usize]) * (node_201)))
                + ((current_base_row[238usize]) * (node_535)))
                + ((current_base_row[242usize]) * (node_213)))
                + ((current_base_row[243usize]) * (node_213)))
                + ((current_base_row[245usize]) * (node_213)))
                + ((current_base_row[246usize]) * (node_1231)))
                + ((current_base_row[247usize]) * (node_213)))
                + ((current_base_row[249usize]) * (node_1231)))
                + ((current_base_row[251usize]) * (node_1231)))
                + ((current_base_row[277usize]) * (node_336)))
                + ((current_base_row[278usize]) * (node_336)))
                + ((current_base_row[279usize]) * (node_1234)))
                + ((current_base_row[281usize]) * (node_225)))
                + ((current_base_row[255usize]) * (node_734)))
                + ((current_base_row[258usize]) * (node_441)))
                + ((current_base_row[295usize]) * (node_1238)))
                + ((current_base_row[296usize]) * (node_517)))
                + ((current_base_row[297usize]) * (node_517))) * (node_4096))
                + ((node_1228) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((((((current_base_row[207usize])
                * (node_445)) + ((current_base_row[213usize]) * (node_536)))
                + ((current_base_row[209usize]) * (node_738)))
                + ((current_base_row[211usize]) * (node_527)))
                + ((current_base_row[232usize]) * (node_517)))
                + ((current_base_row[216usize]) * (node_1230)))
                + ((current_base_row[221usize]) * (node_1231)))
                + ((current_base_row[214usize]) * (node_128)))
                + ((current_base_row[224usize]) * (node_1231)))
                + ((current_base_row[228usize]) * (node_1233)))
                + ((current_base_row[227usize]) * (node_1231)))
                + ((current_base_row[229usize]) * (node_1231)))
                + ((current_base_row[230usize]) * (node_209)))
                + ((current_base_row[219usize]) * (node_734)))
                + ((current_base_row[226usize]) * (node_441)))
                + ((current_base_row[248usize]) * (node_408)))
                + ((current_base_row[233usize]) * (node_409)))
                + ((current_base_row[234usize]) * (node_1231)))
                + ((current_base_row[250usize])
                    * ((next_base_row[27usize]) + (node_240))))
                + ((current_base_row[244usize]) * (node_1235)))
                + ((current_base_row[254usize])
                    * ((next_base_row[37usize]) + (node_200))))
                + ((current_base_row[237usize]) * (node_213)))
                + ((current_base_row[239usize]) * (node_213)))
                + ((current_base_row[240usize]) * (node_1231)))
                + ((current_base_row[241usize]) * (node_205)))
                + ((current_base_row[238usize]) * (node_536)))
                + ((current_base_row[242usize]) * (node_217)))
                + ((current_base_row[243usize]) * (node_217)))
                + ((current_base_row[245usize]) * (node_217)))
                + ((current_base_row[246usize]) * (node_1232)))
                + ((current_base_row[247usize]) * (node_217)))
                + ((current_base_row[249usize]) * (node_1232)))
                + ((current_base_row[251usize]) * (node_1232)))
                + ((current_base_row[277usize]) * (node_337)))
                + ((current_base_row[278usize]) * (node_337)))
                + ((current_base_row[279usize]) * (node_1235)))
                + ((current_base_row[281usize]) * (node_229)))
                + ((current_base_row[255usize]) * (node_738)))
                + ((current_base_row[258usize]) * (node_445)))
                + ((current_base_row[295usize]) * (node_1239)))
                + ((current_base_row[296usize]) * (node_521)))
                + ((current_base_row[297usize]) * (node_521))) * (node_4096))
                + ((node_1229) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((((((current_base_row[207usize])
                * (node_449)) + ((current_base_row[213usize]) * (node_537)))
                + ((current_base_row[209usize]) * (node_742)))
                + ((current_base_row[211usize]) * (node_528)))
                + ((current_base_row[232usize]) * (node_521)))
                + ((current_base_row[216usize]) * (node_1231)))
                + ((current_base_row[221usize]) * (node_1232)))
                + ((current_base_row[214usize]) * (node_185)))
                + ((current_base_row[224usize]) * (node_1232)))
                + ((current_base_row[228usize]) * (node_1234)))
                + ((current_base_row[227usize]) * (node_1232)))
                + ((current_base_row[229usize]) * (node_1232)))
                + ((current_base_row[230usize]) * (node_213)))
                + (current_ext_row[84usize])) + (current_ext_row[85usize]))
                + ((current_base_row[248usize]) * (node_513)))
                + ((current_base_row[233usize]) * (node_410)))
                + ((current_base_row[234usize]) * (node_1232)))
                + ((current_base_row[250usize]) * (node_513)))
                + ((current_base_row[244usize]) * (node_1236)))
                + ((current_base_row[254usize]) * (node_513)))
                + ((current_base_row[237usize]) * (node_217)))
                + ((current_base_row[239usize]) * (node_217)))
                + ((current_base_row[240usize]) * (node_1232)))
                + ((current_base_row[241usize]) * (node_209)))
                + ((current_base_row[238usize]) * (node_537)))
                + ((current_base_row[242usize]) * (node_221)))
                + ((current_base_row[243usize]) * (node_221)))
                + ((current_base_row[245usize]) * (node_221)))
                + ((current_base_row[246usize]) * (node_1233)))
                + ((current_base_row[247usize]) * (node_221)))
                + ((current_base_row[249usize]) * (node_1233)))
                + ((current_base_row[251usize]) * (node_1233)))
                + ((current_base_row[277usize]) * (node_338)))
                + ((current_base_row[278usize]) * (node_338)))
                + ((current_base_row[279usize]) * (node_1236)))
                + ((current_base_row[281usize]) * (node_233)))
                + ((current_base_row[255usize]) * (node_742)))
                + ((current_base_row[258usize]) * (node_449)))
                + ((current_base_row[295usize]) * (node_1219)))
                + ((current_base_row[296usize]) * (node_1229)))
                + ((current_base_row[297usize]) * (node_1229))) * (node_4096))
                + ((node_1230) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((((((current_base_row[207usize])
                * (node_453)) + ((current_base_row[213usize]) * (node_538)))
                + ((current_base_row[209usize]) * (node_746)))
                + ((current_base_row[211usize]) * (node_529)))
                + ((current_base_row[232usize]) * (node_1219)))
                + ((current_base_row[216usize]) * (node_1232)))
                + ((current_base_row[221usize]) * (node_1233)))
                + ((current_base_row[214usize]) * (node_189)))
                + ((current_base_row[224usize]) * (node_1233)))
                + ((current_base_row[228usize]) * (node_1235)))
                + ((current_base_row[227usize]) * (node_1233)))
                + ((current_base_row[229usize]) * (node_1233)))
                + ((current_base_row[230usize]) * (node_217)))
                + ((current_base_row[219usize]) * (node_742)))
                + ((current_base_row[226usize]) * (node_449)))
                + ((current_base_row[248usize]) * (node_517)))
                + ((current_base_row[233usize]) * (node_411)))
                + ((current_base_row[234usize]) * (node_1233)))
                + ((current_base_row[250usize]) * (node_517)))
                + ((current_base_row[244usize]) * (node_1237)))
                + ((current_base_row[254usize]) * (node_517)))
                + ((current_base_row[237usize]) * (node_221)))
                + ((current_base_row[239usize]) * (node_221)))
                + ((current_base_row[240usize]) * (node_1233)))
                + ((current_base_row[241usize]) * (node_213)))
                + ((current_base_row[238usize]) * (node_538)))
                + ((current_base_row[242usize]) * (node_225)))
                + ((current_base_row[243usize]) * (node_225)))
                + ((current_base_row[245usize]) * (node_225)))
                + ((current_base_row[246usize]) * (node_1234)))
                + ((current_base_row[247usize]) * (node_225)))
                + ((current_base_row[249usize]) * (node_1234)))
                + ((current_base_row[251usize]) * (node_1234)))
                + ((current_base_row[277usize]) * (node_314)))
                + ((current_base_row[278usize]) * (node_314)))
                + ((current_base_row[279usize]) * (node_1237)))
                + ((current_base_row[281usize]) * (node_237)))
                + ((current_base_row[255usize]) * (node_746)))
                + ((current_base_row[258usize]) * (node_453)))
                + ((current_base_row[295usize]) * (node_158)))
                + ((current_base_row[296usize]) * (node_1230)))
                + ((current_base_row[297usize]) * (node_1230))) * (node_4096))
                + ((node_1231) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((((((current_base_row[207usize])
                * (node_457)) + ((current_base_row[213usize]) * (node_539)))
                + ((current_base_row[209usize]) * (node_750)))
                + ((current_base_row[211usize]) * (node_530)))
                + ((current_base_row[232usize]) * (node_158)))
                + ((current_base_row[216usize]) * (node_1233)))
                + ((current_base_row[221usize]) * (node_1234)))
                + ((current_base_row[214usize]) * (node_193)))
                + ((current_base_row[224usize]) * (node_1234)))
                + ((current_base_row[228usize]) * (node_1236)))
                + ((current_base_row[227usize]) * (node_1234)))
                + ((current_base_row[229usize]) * (node_1234)))
                + ((current_base_row[230usize]) * (node_221)))
                + ((current_base_row[219usize]) * (node_746)))
                + ((current_base_row[226usize]) * (node_453)))
                + ((current_base_row[248usize]) * (node_521)))
                + ((current_base_row[233usize]) * (node_412)))
                + ((current_base_row[234usize]) * (node_1234)))
                + ((current_base_row[250usize]) * (node_521)))
                + ((current_base_row[244usize]) * (node_1238)))
                + ((current_base_row[254usize]) * (node_521)))
                + ((current_base_row[237usize]) * (node_225)))
                + ((current_base_row[239usize]) * (node_225)))
                + ((current_base_row[240usize]) * (node_1234)))
                + ((current_base_row[241usize]) * (node_217)))
                + ((current_base_row[238usize]) * (node_539)))
                + ((current_base_row[242usize]) * (node_229)))
                + ((current_base_row[243usize]) * (node_229)))
                + ((current_base_row[245usize]) * (node_229)))
                + ((current_base_row[246usize]) * (node_1235)))
                + ((current_base_row[247usize]) * (node_229)))
                + ((current_base_row[249usize]) * (node_1235)))
                + ((current_base_row[251usize]) * (node_1235)))
                + ((current_base_row[277usize]) * (node_325)))
                + ((current_base_row[278usize]) * (node_325)))
                + ((current_base_row[279usize]) * (node_1238)))
                + ((current_base_row[281usize]) * (node_241)))
                + ((current_base_row[255usize]) * (node_750)))
                + ((current_base_row[258usize]) * (node_457)))
                + ((current_base_row[295usize]) * (node_120)))
                + ((current_base_row[296usize]) * (node_1231)))
                + ((current_base_row[297usize]) * (node_1231))) * (node_4096))
                + ((node_1232) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((current_base_row[207usize])
                * (node_465)) + ((current_base_row[213usize]) * (node_541)))
                + ((current_base_row[209usize]) * (node_758)))
                + ((current_base_row[211usize]) * (node_532)))
                + ((current_base_row[216usize]) * (node_1235)))
                + ((current_base_row[221usize]) * (node_1236)))
                + ((current_base_row[214usize]) * (node_201)))
                + ((current_base_row[224usize]) * (node_1236)))
                + ((current_base_row[228usize]) * (node_1238)))
                + ((current_base_row[227usize]) * (node_1236)))
                + ((current_base_row[229usize]) * (node_1236)))
                + ((current_base_row[230usize]) * (node_229)))
                + ((current_base_row[219usize]) * (node_754)))
                + ((current_base_row[226usize]) * (node_461)))
                + ((current_base_row[233usize]) * (node_414)))
                + ((current_base_row[234usize]) * (node_1236)))
                + ((current_base_row[244usize]) * (node_1219)))
                + ((current_base_row[237usize]) * (node_233)))
                + ((current_base_row[239usize]) * (node_233)))
                + ((current_base_row[240usize]) * (node_1236)))
                + ((current_base_row[241usize]) * (node_225)))
                + ((current_base_row[238usize]) * (node_541)))
                + ((current_base_row[242usize]) * (node_237)))
                + ((current_base_row[243usize]) * (node_237)))
                + ((current_base_row[245usize]) * (node_237)))
                + ((current_base_row[246usize]) * (node_1237)))
                + ((current_base_row[247usize]) * (node_237)))
                + ((current_base_row[249usize]) * (node_1237)))
                + ((current_base_row[251usize]) * (node_1237)))
                + ((current_base_row[277usize]) * (node_124)))
                + ((current_base_row[278usize]) * (node_124)))
                + ((current_base_row[279usize]) * (node_1219)))
                + ((current_base_row[281usize]) * (node_181)))
                + ((current_base_row[255usize]) * (node_758)))
                + ((current_base_row[258usize]) * (node_465)))
                + ((current_base_row[295usize]) * (node_128)))
                + ((current_base_row[296usize]) * (node_1233)))
                + ((current_base_row[297usize]) * (node_1233))) * (node_4096))
                + ((node_1234) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((current_base_row[207usize])
                * (node_469)) + ((current_base_row[213usize]) * (node_550)))
                + ((current_base_row[209usize]) * (node_762)))
                + ((current_base_row[211usize]) * (node_533)))
                + ((current_base_row[216usize]) * (node_1236)))
                + ((current_base_row[221usize]) * (node_1237)))
                + ((current_base_row[214usize]) * (node_205)))
                + ((current_base_row[224usize]) * (node_1237)))
                + ((current_base_row[228usize]) * (node_1239)))
                + ((current_base_row[227usize]) * (node_1237)))
                + ((current_base_row[229usize]) * (node_1237)))
                + ((current_base_row[230usize]) * (node_233)))
                + ((current_base_row[219usize]) * (node_758)))
                + ((current_base_row[226usize]) * (node_465)))
                + ((current_base_row[233usize]) * (node_415)))
                + ((current_base_row[234usize]) * (node_1237)))
                + ((current_base_row[244usize]) * (node_158)))
                + ((current_base_row[237usize]) * (node_237)))
                + ((current_base_row[239usize]) * (node_237)))
                + ((current_base_row[240usize]) * (node_1237)))
                + ((current_base_row[241usize]) * (node_229)))
                + ((current_base_row[238usize]) * (node_550)))
                + ((current_base_row[242usize]) * (node_241)))
                + ((current_base_row[243usize]) * (node_241)))
                + ((current_base_row[245usize]) * (node_241)))
                + ((current_base_row[246usize]) * (node_1238)))
                + ((current_base_row[247usize]) * (node_241)))
                + ((current_base_row[249usize]) * (node_1238)))
                + ((current_base_row[251usize]) * (node_1238)))
                + ((current_base_row[277usize]) * (node_128)))
                + ((current_base_row[278usize]) * (node_128)))
                + ((current_base_row[279usize]) * (node_158)))
                + ((current_base_row[281usize]) * (node_120)))
                + ((current_base_row[255usize]) * (node_762)))
                + ((current_base_row[258usize]) * (node_469)))
                + ((current_base_row[295usize]) * (node_1224)))
                + ((current_base_row[296usize]) * (node_1234)))
                + ((current_base_row[297usize]) * (node_1234))) * (node_4096))
                + ((node_1235) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((current_base_row[207usize])
                * (node_473)) + ((current_base_row[213usize]) * (node_120)))
                + ((current_base_row[209usize]) * (node_766)))
                + ((current_base_row[211usize]) * (node_534)))
                + ((current_base_row[216usize]) * (node_1237)))
                + ((current_base_row[221usize]) * (node_1238)))
                + ((current_base_row[214usize]) * (node_209)))
                + ((current_base_row[224usize]) * (node_1238)))
                + ((current_base_row[228usize]) * (node_1219)))
                + ((current_base_row[227usize]) * (node_1238)))
                + ((current_base_row[229usize]) * (node_1238)))
                + ((current_base_row[230usize]) * (node_237)))
                + ((current_base_row[219usize]) * (node_762)))
                + ((current_base_row[226usize]) * (node_469)))
                + ((current_base_row[233usize]) * (node_416)))
                + ((current_base_row[234usize]) * (node_1238)))
                + ((current_base_row[244usize]) * (node_517)))
                + ((current_base_row[237usize]) * (node_241)))
                + ((current_base_row[239usize]) * (node_241)))
                + ((current_base_row[240usize]) * (node_1238)))
                + ((current_base_row[241usize]) * (node_233)))
                + ((current_base_row[238usize]) * (node_120)))
                + ((current_base_row[242usize]) * (node_159)))
                + ((current_base_row[243usize]) * (node_159)))
                + ((current_base_row[245usize]) * (node_159)))
                + ((current_base_row[246usize]) * (node_1239)))
                + ((current_base_row[247usize]) * (node_159)))
                + ((current_base_row[249usize]) * (node_1239)))
                + ((current_base_row[251usize]) * (node_1239)))
                + ((current_base_row[277usize]) * (node_1224)))
                + ((current_base_row[278usize]) * (node_1224)))
                + ((current_base_row[279usize]) * (node_120)))
                + ((current_base_row[281usize]) * (node_124)))
                + ((current_base_row[255usize]) * (node_766)))
                + ((current_base_row[258usize]) * (node_473)))
                + ((current_base_row[295usize]) * (node_513)))
                + ((current_base_row[296usize]) * (node_1235)))
                + ((current_base_row[297usize]) * (node_1235))) * (node_4096))
                + ((node_1236) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((current_base_row[207usize])
                * (node_477)) + ((current_base_row[213usize]) * (node_124)))
                + ((current_base_row[209usize]) * (node_770)))
                + ((current_base_row[211usize]) * (node_535)))
                + ((current_base_row[216usize]) * (node_1238)))
                + ((current_base_row[221usize]) * (node_1239)))
                + ((current_base_row[214usize]) * (node_213)))
                + ((current_base_row[224usize]) * (node_1239)))
                + ((current_base_row[228usize]) * (node_158)))
                + ((current_base_row[227usize]) * (node_1239)))
                + ((current_base_row[229usize]) * (node_1239)))
                + ((current_base_row[230usize]) * (node_241)))
                + ((current_base_row[219usize]) * (node_766)))
                + ((current_base_row[226usize]) * (node_473)))
                + ((current_base_row[233usize]) * (node_417)))
                + ((current_base_row[234usize]) * (node_1239)))
                + ((current_base_row[244usize]) * (node_521)))
                + ((current_base_row[237usize]) * (node_159)))
                + ((current_base_row[239usize]) * (node_159)))
                + ((current_base_row[240usize]) * (node_1239)))
                + ((current_base_row[241usize]) * (node_237)))
                + ((current_base_row[238usize]) * (node_124)))
                + ((current_base_row[242usize]) * (node_181)))
                + ((current_base_row[243usize]) * (node_181)))
                + ((current_base_row[245usize]) * (node_181)))
                + ((current_base_row[246usize]) * (node_1219)))
                + ((current_base_row[247usize]) * (node_181)))
                + ((current_base_row[249usize]) * (node_1219)))
                + ((current_base_row[251usize]) * (node_1219)))
                + ((current_base_row[277usize]) * (node_513)))
                + ((current_base_row[278usize]) * (node_513)))
                + ((current_base_row[279usize]) * (node_124)))
                + ((current_base_row[281usize]) * (node_128)))
                + ((current_base_row[255usize]) * (node_770)))
                + ((current_base_row[258usize]) * (node_477)))
                + ((current_base_row[295usize]) * (node_517)))
                + ((current_base_row[296usize]) * (node_1236)))
                + ((current_base_row[297usize]) * (node_1236))) * (node_4096))
                + ((node_1237) * (next_base_row[8usize])),
            (((((((((((((((((((((((((((((((((((((((current_base_row[207usize])
                * (node_481)) + ((current_base_row[213usize]) * (node_128)))
                + ((current_base_row[209usize]) * (node_774)))
                + ((current_base_row[211usize]) * (node_536)))
                + ((current_base_row[216usize]) * (node_1239)))
                + ((current_base_row[221usize]) * (node_1219)))
                + ((current_base_row[214usize]) * (node_217)))
                + ((current_base_row[224usize]) * (node_1219)))
                + ((current_base_row[228usize]) * (node_513)))
                + ((current_base_row[227usize]) * (node_1219)))
                + ((current_base_row[229usize]) * (node_1219)))
                + ((current_base_row[230usize]) * (node_159)))
                + ((current_base_row[219usize]) * (node_770)))
                + ((current_base_row[226usize]) * (node_477)))
                + ((current_base_row[233usize]) * (node_418)))
                + ((current_base_row[234usize]) * (node_1219)))
                + ((current_base_row[237usize]) * (node_181)))
                + ((current_base_row[239usize]) * (node_181)))
                + ((current_base_row[240usize]) * (node_1219)))
                + ((current_base_row[241usize]) * (node_241)))
                + ((current_base_row[238usize]) * (node_128)))
                + ((current_base_row[242usize]) * (node_513)))
                + ((current_base_row[243usize]) * (node_513)))
                + ((current_base_row[245usize]) * (node_513)))
                + ((current_base_row[246usize]) * (node_158)))
                + ((current_base_row[247usize]) * (node_513)))
                + ((current_base_row[249usize]) * (node_158)))
                + ((current_base_row[251usize]) * (node_158)))
                + ((current_base_row[277usize]) * (node_517)))
                + ((current_base_row[278usize]) * (node_517)))
                + ((current_base_row[279usize]) * (node_128)))
                + ((current_base_row[281usize]) * (node_1224)))
                + ((current_base_row[255usize]) * (node_774)))
                + ((current_base_row[258usize]) * (node_481)))
                + ((current_base_row[295usize]) * (node_521)))
                + ((current_base_row[296usize]) * (node_1237)))
                + ((current_base_row[297usize]) * (node_1237))) * (node_4096))
                + ((node_1238) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((current_base_row[207usize])
                * (node_485)) + ((current_base_row[213usize]) * (node_116)))
                + ((current_base_row[209usize]) * (node_778)))
                + ((current_base_row[211usize]) * (node_537)))
                + ((current_base_row[216usize]) * (node_1219)))
                + ((current_base_row[221usize]) * (node_158)))
                + ((current_base_row[214usize]) * (node_221)))
                + ((current_base_row[224usize]) * (node_158)))
                + ((current_base_row[228usize]) * (node_517)))
                + ((current_base_row[227usize]) * (node_158)))
                + ((current_base_row[229usize]) * (node_158)))
                + ((current_base_row[230usize]) * (node_181)))
                + ((current_base_row[219usize]) * (node_774)))
                + ((current_base_row[226usize]) * (node_481)))
                + ((current_base_row[233usize]) * (node_419)))
                + ((current_base_row[234usize]) * (node_158)))
                + ((current_base_row[237usize]) * (node_513)))
                + ((current_base_row[239usize]) * (node_513)))
                + ((current_base_row[240usize]) * (node_158)))
                + ((current_base_row[241usize]) * (node_159)))
                + ((current_base_row[238usize]) * (node_1224)))
                + ((current_base_row[242usize]) * (node_517)))
                + ((current_base_row[243usize]) * (node_517)))
                + ((current_base_row[245usize]) * (node_517)))
                + ((current_base_row[246usize]) * (node_513)))
                + ((current_base_row[247usize]) * (node_517)))
                + ((current_base_row[249usize]) * (node_513)))
                + ((current_base_row[251usize]) * (node_513)))
                + ((current_base_row[277usize]) * (node_521)))
                + ((current_base_row[278usize]) * (node_521)))
                + ((current_base_row[279usize]) * (node_1224)))
                + ((current_base_row[281usize]) * (node_513)))
                + ((current_base_row[255usize]) * (node_778)))
                + ((current_base_row[258usize]) * (node_485)))
                + ((current_base_row[296usize]) * (node_1238)))
                + ((current_base_row[297usize]) * (node_1238))) * (node_4096))
                + ((node_1239) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((current_base_row[207usize]) * (node_488))
                + ((current_base_row[213usize]) * (node_513)))
                + ((current_base_row[209usize]) * (node_781)))
                + ((current_base_row[211usize]) * (node_538)))
                + ((current_base_row[216usize]) * (node_158)))
                + ((current_base_row[221usize]) * (node_513)))
                + ((current_base_row[214usize]) * (node_225)))
                + ((current_base_row[224usize]) * (node_513)))
                + ((current_base_row[228usize]) * (node_521)))
                + ((current_base_row[227usize]) * (node_513)))
                + ((current_base_row[229usize]) * (node_513)))
                + ((current_base_row[230usize]) * (node_513)))
                + ((current_base_row[219usize]) * (node_778)))
                + ((current_base_row[226usize]) * (node_485)))
                + ((current_base_row[233usize]) * (node_513)))
                + ((current_base_row[234usize]) * (node_513)))
                + ((current_base_row[237usize]) * (node_517)))
                + ((current_base_row[239usize]) * (node_517)))
                + ((current_base_row[240usize]) * (node_513)))
                + ((current_base_row[241usize]) * (node_181)))
                + ((current_base_row[238usize]) * (node_513)))
                + ((current_base_row[242usize]) * (node_521)))
                + ((current_base_row[243usize]) * (node_521)))
                + ((current_base_row[245usize]) * (node_521)))
                + ((current_base_row[246usize]) * (node_517)))
                + ((current_base_row[247usize]) * (node_521)))
                + ((current_base_row[249usize]) * (node_517)))
                + ((current_base_row[251usize]) * (node_517)))
                + ((current_base_row[279usize]) * (node_513)))
                + ((current_base_row[281usize]) * (node_517)))
                + ((current_base_row[255usize])
                    * ((node_781)
                        + (((next_ext_row[3usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((challenges[1usize])
                                    * (((challenges[1usize])
                                        * (((challenges[1usize])
                                            * (((challenges[1usize])
                                                * ((node_1665) + (next_base_row[26usize])))
                                                + (next_base_row[25usize]))) + (next_base_row[24usize])))
                                        + (next_base_row[23usize]))) + (next_base_row[22usize]))))
                            * (current_base_row[190usize])))))
                + ((current_base_row[258usize])
                    * ((node_488)
                        + (((next_ext_row[4usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((challenges[2usize]) * (node_1727))
                                    + (current_base_row[26usize]))))
                            * (current_base_row[190usize])))))
                + ((current_base_row[296usize]) * (node_1239)))
                + ((current_base_row[297usize]) * (node_1239))) * (node_4096))
                + ((node_1219) * (next_base_row[8usize])),
            (((((((((((((((((((((((((((((((current_base_row[207usize]) * (node_490))
                + ((current_base_row[213usize]) * (node_517)))
                + ((current_base_row[209usize]) * (node_783)))
                + ((current_base_row[211usize]) * (node_539)))
                + ((current_base_row[216usize]) * (node_513)))
                + ((current_base_row[221usize]) * (node_517)))
                + ((current_base_row[214usize]) * (node_229)))
                + ((current_base_row[224usize]) * (node_517)))
                + ((current_base_row[227usize]) * (node_517)))
                + ((current_base_row[229usize]) * (node_517)))
                + ((current_base_row[230usize]) * (node_517)))
                + ((current_base_row[219usize]) * (node_781)))
                + ((current_base_row[226usize]) * (node_488)))
                + ((current_base_row[233usize]) * (node_517)))
                + ((current_base_row[234usize]) * (node_517)))
                + ((current_base_row[237usize]) * (node_521)))
                + ((current_base_row[239usize]) * (node_521)))
                + ((current_base_row[240usize]) * (node_517)))
                + ((current_base_row[241usize]) * (node_513)))
                + ((current_base_row[238usize]) * (node_517)))
                + ((current_base_row[246usize]) * (node_521)))
                + ((current_base_row[249usize]) * (node_521)))
                + ((current_base_row[251usize]) * (node_521)))
                + ((current_base_row[279usize]) * (node_517)))
                + ((current_base_row[281usize]) * (node_521)))
                + ((current_base_row[255usize])
                    * ((node_783)
                        + (((next_ext_row[3usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((challenges[1usize])
                                    * (((challenges[1usize])
                                        * (((challenges[1usize])
                                            * ((node_1665) + (next_base_row[25usize])))
                                            + (next_base_row[24usize]))) + (next_base_row[23usize])))
                                    + (next_base_row[22usize]))))
                            * (current_base_row[189usize])))))
                + ((current_base_row[258usize])
                    * ((node_490)
                        + (((next_ext_row[4usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (node_1727))) * (current_base_row[189usize])))))
                + ((current_base_row[296usize]) * (node_1219)))
                + ((current_base_row[297usize]) * (node_1219))) * (node_4096))
                + ((node_158) * (next_base_row[8usize])),
            (((((((((((((((((((((((((current_base_row[207usize]) * (node_491))
                + ((current_base_row[213usize]) * (node_521)))
                + ((current_base_row[209usize]) * (node_784)))
                + ((current_base_row[211usize]) * (node_540)))
                + ((current_base_row[216usize]) * (node_517)))
                + ((current_base_row[221usize]) * (node_521)))
                + ((current_base_row[214usize]) * (node_233)))
                + ((current_base_row[224usize]) * (node_521)))
                + ((current_base_row[227usize]) * (node_521)))
                + ((current_base_row[229usize]) * (node_521)))
                + ((current_base_row[230usize]) * (node_521)))
                + ((current_base_row[219usize]) * (node_783)))
                + ((current_base_row[226usize]) * (node_490)))
                + ((current_base_row[233usize]) * (node_521)))
                + ((current_base_row[234usize]) * (node_521)))
                + ((current_base_row[240usize]) * (node_521)))
                + ((current_base_row[241usize]) * (node_517)))
                + ((current_base_row[238usize]) * (node_521)))
                + ((current_base_row[279usize]) * (node_521)))
                + ((current_base_row[255usize])
                    * ((node_784)
                        + (((next_ext_row[3usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((challenges[1usize])
                                    * (((challenges[1usize])
                                        * ((node_1665) + (next_base_row[24usize])))
                                        + (next_base_row[23usize]))) + (next_base_row[22usize]))))
                            * (current_base_row[187usize])))))
                + ((current_base_row[258usize])
                    * ((node_491)
                        + (((next_ext_row[4usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (node_1722))) * (current_base_row[187usize])))))
                + ((current_base_row[296usize]) * (node_158)))
                + ((current_base_row[297usize]) * (node_158))) * (node_4096))
                + ((node_513) * (next_base_row[8usize])),
            ((((((((((((current_base_row[207usize]) * (node_267))
                + ((current_base_row[209usize]) * (node_567)))
                + ((current_base_row[211usize]) * (node_541)))
                + ((current_base_row[216usize]) * (node_521)))
                + ((current_base_row[214usize]) * (node_237)))
                + ((current_base_row[219usize]) * (node_784)))
                + ((current_base_row[226usize]) * (node_491)))
                + ((current_base_row[241usize]) * (node_521)))
                + ((current_base_row[255usize])
                    * ((node_567)
                        + (((next_ext_row[3usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((challenges[1usize])
                                    * ((node_1665) + (next_base_row[23usize])))
                                    + (next_base_row[22usize]))))
                            * (current_base_row[186usize])))))
                + ((current_base_row[258usize])
                    * ((node_267)
                        + (((next_ext_row[4usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (node_1717))) * (current_base_row[186usize])))))
                * (node_4096)) + ((node_517) * (next_base_row[8usize])),
            ((((((((((current_base_row[207usize]) * (current_base_row[329usize]))
                + ((current_base_row[209usize]) * (current_base_row[329usize])))
                + ((current_base_row[211usize]) * (node_550)))
                + ((current_base_row[214usize]) * (node_241)))
                + ((current_base_row[219usize]) * (node_567)))
                + ((current_base_row[226usize]) * (node_267)))
                + ((current_base_row[255usize])
                    * (((next_ext_row[3usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((node_1665) + (next_base_row[22usize]))))
                        * (current_base_row[185usize]))))
                + ((current_base_row[258usize])
                    * (((next_ext_row[4usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_1712))) * (current_base_row[185usize]))))
                * (node_4096)) + ((node_521) * (next_base_row[8usize])),
            (((((((((current_base_row[207usize]) * (current_base_row[330usize]))
                + ((current_base_row[209usize]) * (current_base_row[330usize])))
                + ((current_base_row[211usize]) * (node_513)))
                + ((current_base_row[214usize]) * (node_159)))
                + ((current_base_row[219usize]) * (current_base_row[329usize])))
                + ((current_base_row[226usize]) * (current_base_row[329usize])))
                + ((current_base_row[255usize]) * (current_base_row[329usize])))
                + ((current_base_row[258usize]) * (current_base_row[329usize])))
                * (node_4096),
            (((((((((current_base_row[207usize]) * (current_base_row[331usize]))
                + ((current_base_row[209usize]) * (current_base_row[331usize])))
                + ((current_base_row[211usize]) * (node_517)))
                + ((current_base_row[214usize]) * (node_181)))
                + ((current_base_row[219usize]) * (current_base_row[330usize])))
                + ((current_base_row[226usize]) * (current_base_row[330usize])))
                + ((current_base_row[255usize]) * (current_base_row[330usize])))
                + ((current_base_row[258usize]) * (current_base_row[330usize])))
                * (node_4096),
            (((((((((current_base_row[207usize]) * (current_base_row[332usize]))
                + ((current_base_row[209usize]) * (current_base_row[332usize])))
                + ((current_base_row[211usize]) * (node_521)))
                + ((current_base_row[214usize]) * (node_513)))
                + ((current_base_row[219usize]) * (current_base_row[331usize])))
                + ((current_base_row[226usize]) * (current_base_row[331usize])))
                + ((current_base_row[255usize]) * (current_base_row[331usize])))
                + ((current_base_row[258usize]) * (current_base_row[331usize])))
                * (node_4096),
            ((((((((current_base_row[207usize]) * (current_base_row[333usize]))
                + ((current_base_row[209usize]) * (current_base_row[333usize])))
                + ((current_base_row[214usize]) * (node_517)))
                + ((current_base_row[219usize]) * (current_base_row[332usize])))
                + ((current_base_row[226usize]) * (current_base_row[332usize])))
                + ((current_base_row[255usize]) * (current_base_row[332usize])))
                + ((current_base_row[258usize]) * (current_base_row[332usize])))
                * (node_4096),
            ((((((((current_base_row[207usize]) * (current_base_row[334usize]))
                + ((current_base_row[209usize]) * (current_base_row[334usize])))
                + ((current_base_row[214usize]) * (node_521)))
                + ((current_base_row[219usize]) * (current_base_row[333usize])))
                + ((current_base_row[226usize]) * (current_base_row[333usize])))
                + ((current_base_row[255usize]) * (current_base_row[333usize])))
                + ((current_base_row[258usize]) * (current_base_row[333usize])))
                * (node_4096),
            (((((((current_base_row[207usize]) * (node_513))
                + ((current_base_row[209usize]) * (node_513)))
                + ((current_base_row[219usize]) * (current_base_row[339usize])))
                + ((current_base_row[226usize]) * (current_base_row[339usize])))
                + ((current_base_row[255usize]) * (current_base_row[339usize])))
                + ((current_base_row[258usize]) * (current_base_row[339usize])))
                * (node_4096),
            (((((((current_base_row[207usize]) * (node_517))
                + ((current_base_row[209usize]) * (node_517)))
                + ((current_base_row[219usize]) * (node_517)))
                + ((current_base_row[226usize]) * (node_517)))
                + ((current_base_row[255usize]) * (node_513)))
                + ((current_base_row[258usize]) * (node_513))) * (node_4096),
            (((((((current_base_row[207usize]) * (node_521))
                + ((current_base_row[209usize]) * (node_521)))
                + ((current_base_row[219usize]) * (node_521)))
                + ((current_base_row[226usize]) * (node_521)))
                + ((current_base_row[255usize]) * (node_521)))
                + ((current_base_row[258usize]) * (node_517))) * (node_4096),
            (((next_ext_row[13usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (current_ext_row[13usize])))
                * ((challenges[11usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[7usize]))))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (next_base_row[45usize])),
            ((node_4096)
                * (((node_4210)
                    * ((challenges[3usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((((challenges[13usize]) * (next_base_row[9usize]))
                                + ((challenges[14usize]) * (next_base_row[10usize])))
                                + ((challenges[15usize]) * (next_base_row[11usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((next_base_row[8usize]) * (node_4210)),
            (next_ext_row[8usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[8usize])
                        * ((challenges[9usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((((((challenges[24usize]) * (next_base_row[7usize]))
                                    + ((challenges[25usize]) * (next_base_row[10usize])))
                                    + ((challenges[26usize]) * (next_base_row[19usize])))
                                    + ((challenges[27usize]) * (next_base_row[20usize])))
                                    + ((challenges[28usize]) * (next_base_row[21usize]))))))),
            (((((next_base_row[10usize])
                + (BFieldElement::from_raw_u64(18446743992105173011u64)))
                * ((next_base_row[10usize])
                    + (BFieldElement::from_raw_u64(18446743725817200721u64))))
                * ((next_ext_row[9usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_ext_row[9usize]))))
                + ((current_base_row[341usize]) * ((node_4334) + (node_4335))))
                + ((current_base_row[349usize])
                    * ((node_4334)
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((((((((challenges[32usize])
                                * (((node_4274) * (next_base_row[22usize]))
                                    + ((next_base_row[44usize]) * (next_base_row[39usize]))))
                                + ((challenges[33usize])
                                    * (((node_4274) * (next_base_row[23usize]))
                                        + ((next_base_row[44usize]) * (next_base_row[40usize])))))
                                + ((challenges[34usize])
                                    * (((node_4274) * (next_base_row[24usize]))
                                        + ((next_base_row[44usize]) * (next_base_row[41usize])))))
                                + ((challenges[35usize])
                                    * (((node_4274) * (next_base_row[25usize]))
                                        + ((next_base_row[44usize]) * (next_base_row[42usize])))))
                                + ((challenges[36usize])
                                    * (((node_4274) * (next_base_row[26usize]))
                                        + ((next_base_row[44usize]) * (next_base_row[43usize])))))
                                + ((challenges[37usize])
                                    * (((node_4274) * (next_base_row[39usize]))
                                        + ((next_base_row[44usize]) * (next_base_row[22usize])))))
                                + ((challenges[38usize])
                                    * (((node_4274) * (next_base_row[40usize]))
                                        + ((next_base_row[44usize]) * (next_base_row[23usize])))))
                                + ((challenges[39usize])
                                    * (((node_4274) * (next_base_row[41usize]))
                                        + ((next_base_row[44usize]) * (next_base_row[24usize])))))
                                + ((challenges[40usize])
                                    * (((node_4274) * (next_base_row[42usize]))
                                        + ((next_base_row[44usize]) * (next_base_row[25usize])))))
                                + ((challenges[41usize])
                                    * (((node_4274) * (next_base_row[43usize]))
                                        + ((next_base_row[44usize])
                                            * (next_base_row[26usize])))))))),
            ((((current_base_row[10usize])
                + (BFieldElement::from_raw_u64(18446743992105173011u64)))
                * ((current_base_row[10usize])
                    + (BFieldElement::from_raw_u64(18446743725817200721u64))))
                * ((next_ext_row[10usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_ext_row[10usize]))))
                + (((current_base_row[248usize]) + (current_base_row[295usize]))
                    * (((next_ext_row[10usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((challenges[5usize]) * (current_ext_row[10usize]))))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_846)))),
            (((((current_base_row[326usize])
                * ((next_ext_row[11usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_ext_row[11usize]))))
                + ((current_base_row[234usize]) * (node_4387)))
                + ((current_base_row[250usize])
                    * ((node_4387)
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_887)))))
                + ((current_base_row[244usize])
                    * (((node_4383)
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((challenges[31usize])
                                * (BFieldElement::from_raw_u64(146028888030u64)))))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((((((((challenges[32usize]) * (next_base_row[23usize]))
                                + ((challenges[33usize]) * (next_base_row[24usize])))
                                + ((challenges[34usize]) * (next_base_row[25usize])))
                                + ((challenges[35usize]) * (next_base_row[26usize])))
                                + ((challenges[36usize]) * (current_base_row[39usize])))
                                + ((challenges[37usize]) * (current_base_row[40usize])))
                                + ((challenges[38usize]) * (current_base_row[41usize])))
                                + ((challenges[39usize]) * (current_base_row[42usize])))
                                + ((challenges[40usize]) * (current_base_row[43usize])))
                                + ((challenges[41usize]) * (current_base_row[44usize])))))))
                + ((current_base_row[254usize]) * ((node_4387) + (node_4335))),
            (((((((((current_base_row[238usize])
                * (((node_4476) * (((node_4433) + (node_4436)) + (node_4440)))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((current_base_row[242usize]) * (node_4480)))
                + ((current_base_row[243usize]) * (node_4480)))
                + ((current_base_row[245usize])
                    * (((node_4476)
                        * (((node_4447)
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((challenges[57usize])
                                    * (BFieldElement::from_raw_u64(60129542130u64)))))
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((challenges[58usize])
                                    * (((current_base_row[22usize])
                                        + (current_base_row[23usize])) + (node_1609)))
                                    * (BFieldElement::from_raw_u64(9223372036854775808u64))))))
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)))))
                + ((current_base_row[247usize]) * (node_4480)))
                + ((current_base_row[246usize]) * (node_4484)))
                + ((current_base_row[249usize])
                    * (((((node_4476) * (node_4470)) * (node_4474))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_4470)))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_4474)))))
                + ((current_base_row[251usize]) * (node_4484)))
                + (((BFieldElement::from_raw_u64(4294967295u64))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[14usize]))) * (node_4476)),
            (((next_ext_row[14usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[14usize])
                        * ((challenges[7usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((((challenges[16usize]) * (next_base_row[46usize]))
                                    + ((challenges[17usize]) * (next_base_row[47usize])))
                                    + ((challenges[18usize]) * (next_base_row[48usize])))
                                    + ((challenges[19usize]) * (next_base_row[49usize]))))))))
                * (node_4530))
                + (((next_ext_row[14usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_ext_row[14usize]))) * (node_4547)),
            ((((((node_4556)
                * ((challenges[11usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((next_base_row[46usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (current_base_row[46usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64))) * (node_4524))
                * (node_4530)) + ((node_4556) * (node_4523)))
                + ((node_4556) * (node_4547)),
            ((node_4598)
                * ((next_ext_row[16usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((current_ext_row[16usize]) * (node_4616)))))
                + ((node_4601) * ((next_ext_row[16usize]) + (node_4621))),
            ((node_4598)
                * (((next_ext_row[17usize]) + (node_4621))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((node_4616) * (current_ext_row[17usize])))))
                + ((node_4601)
                    * ((next_ext_row[17usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_ext_row[17usize])))),
            ((node_4598)
                * (((next_ext_row[18usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((challenges[12usize]) * (current_ext_row[18usize]))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[55usize]))))
                + ((node_4601)
                    * ((next_ext_row[18usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_ext_row[18usize])))),
            ((node_4598)
                * (((next_ext_row[19usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((challenges[12usize]) * (current_ext_row[19usize]))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[56usize]))))
                + ((node_4601)
                    * ((next_ext_row[19usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_ext_row[19usize])))),
            (((next_ext_row[20usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[20usize])
                        * ((challenges[8usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((((next_base_row[50usize]) * (challenges[20usize]))
                                    + ((next_base_row[52usize]) * (challenges[21usize])))
                                    + ((next_base_row[53usize]) * (challenges[22usize])))
                                    + ((next_base_row[51usize]) * (challenges[23usize]))))))))
                * (node_4593))
                + (((next_ext_row[20usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_ext_row[20usize]))) * (node_4667)),
            (((current_ext_row[83usize]) * (node_4593)) + ((node_4676) * (node_4598)))
                + ((node_4676) * (node_4667)),
            (next_ext_row[22usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[22usize])
                        * ((challenges[9usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((((((challenges[24usize]) * (next_base_row[57usize]))
                                    + ((challenges[25usize]) * (next_base_row[58usize])))
                                    + ((challenges[26usize]) * (next_base_row[59usize])))
                                    + ((challenges[27usize]) * (next_base_row[60usize])))
                                    + ((challenges[28usize]) * (next_base_row[61usize]))))))),
            ((node_4705)
                * (((node_4739)
                    * ((challenges[11usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_4717))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_4704) * (node_4739)),
            (((current_base_row[342usize])
                * (((next_ext_row[24usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((challenges[30usize]) * (current_ext_row[24usize]))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((((((((((((((((((((challenges[29usize]) + (node_5492))
                            * (challenges[29usize])) + (node_5503))
                            * (challenges[29usize])) + (node_5514))
                            * (challenges[29usize])) + (node_5525))
                            * (challenges[29usize])) + (next_base_row[97usize]))
                            * (challenges[29usize])) + (next_base_row[98usize]))
                            * (challenges[29usize])) + (next_base_row[99usize]))
                            * (challenges[29usize])) + (next_base_row[100usize]))
                            * (challenges[29usize])) + (next_base_row[101usize]))
                            * (challenges[29usize])) + (next_base_row[102usize])))))
                + ((next_base_row[64usize]) * (node_5828)))
                + ((node_5616) * (node_5828)),
            ((current_base_row[343usize]) * (node_5616))
                * (((((((((((challenges[0usize]) + (node_4776)) * (challenges[0usize]))
                    + (node_4787)) * (challenges[0usize])) + (node_4798))
                    * (challenges[0usize])) + (node_4809)) * (challenges[0usize]))
                    + (current_base_row[97usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (challenges[62usize]))),
            (current_base_row[357usize])
                * ((((((node_5659) + (node_5660)) + (node_5662)) + (node_5664))
                    + (node_5666)) + (node_5668)),
            (current_base_row[358usize])
                * (((((((((((((((((challenges[32usize])
                    * ((node_5492)
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_4776))))
                    + ((challenges[33usize])
                        * ((node_5503)
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (node_4787)))))
                    + ((challenges[34usize])
                        * ((node_5514)
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (node_4798)))))
                    + ((challenges[35usize])
                        * ((node_5525)
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (node_4809)))))
                    + ((challenges[36usize])
                        * ((next_base_row[97usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (current_base_row[97usize])))))
                    + ((challenges[37usize])
                        * ((next_base_row[98usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (current_base_row[98usize])))))
                    + ((challenges[38usize])
                        * ((next_base_row[99usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (current_base_row[99usize])))))
                    + ((challenges[39usize])
                        * ((next_base_row[100usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (current_base_row[100usize])))))
                    + ((challenges[40usize])
                        * ((next_base_row[101usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (current_base_row[101usize])))))
                    + ((challenges[41usize])
                        * ((next_base_row[102usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (current_base_row[102usize]))))) + (node_5659))
                    + (node_5660)) + (node_5662)) + (node_5664)) + (node_5666))
                    + (node_5668)),
            ((((current_base_row[315usize]) * (current_base_row[318usize]))
                * (((next_ext_row[25usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((challenges[4usize]) * (current_ext_row[25usize]))))
                    + (node_5758))) + ((next_base_row[64usize]) * (node_5735)))
                + ((node_5621) * (node_5735)),
            ((((node_5778) * (current_base_row[318usize]))
                * (((next_ext_row[26usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((challenges[5usize]) * (current_ext_row[26usize]))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (node_5744)))) + ((node_5677) * (node_5769)))
                + ((node_5621) * (node_5769)),
            ((((current_base_row[315usize]) * (node_5728))
                * ((((next_ext_row[27usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((challenges[6usize]) * (current_ext_row[27usize]))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((challenges[31usize]) * (next_base_row[63usize]))))
                    + (node_5758))) + ((next_base_row[64usize]) * (node_5795)))
                + ((((node_5625) * (node_5730)) * (node_5798)) * (node_5795)),
            (((current_base_row[282usize])
                * (((node_5847)
                    * ((challenges[48usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[49usize]) * (next_base_row[65usize]))
                                + ((challenges[50usize]) * (next_base_row[81usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_5778) * (node_5847))) + ((node_5855) * (node_5847)),
            (((current_base_row[282usize])
                * (((node_5868)
                    * ((challenges[48usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[49usize]) * (next_base_row[66usize]))
                                + ((challenges[50usize]) * (next_base_row[82usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_5778) * (node_5868))) + ((node_5855) * (node_5868)),
            (((current_base_row[282usize])
                * (((node_5885)
                    * ((challenges[48usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[49usize]) * (next_base_row[67usize]))
                                + ((challenges[50usize]) * (next_base_row[83usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_5778) * (node_5885))) + ((node_5855) * (node_5885)),
            (((current_base_row[282usize])
                * (((node_5902)
                    * ((challenges[48usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[49usize]) * (next_base_row[68usize]))
                                + ((challenges[50usize]) * (next_base_row[84usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_5778) * (node_5902))) + ((node_5855) * (node_5902)),
            (((current_base_row[282usize])
                * (((node_5919)
                    * ((challenges[48usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[49usize]) * (next_base_row[69usize]))
                                + ((challenges[50usize]) * (next_base_row[85usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_5778) * (node_5919))) + ((node_5855) * (node_5919)),
            (((current_base_row[282usize])
                * (((node_5936)
                    * ((challenges[48usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[49usize]) * (next_base_row[70usize]))
                                + ((challenges[50usize]) * (next_base_row[86usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_5778) * (node_5936))) + ((node_5855) * (node_5936)),
            (((current_base_row[282usize])
                * (((node_5953)
                    * ((challenges[48usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[49usize]) * (next_base_row[71usize]))
                                + ((challenges[50usize]) * (next_base_row[87usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_5778) * (node_5953))) + ((node_5855) * (node_5953)),
            (((current_base_row[282usize])
                * (((node_5970)
                    * ((challenges[48usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[49usize]) * (next_base_row[72usize]))
                                + ((challenges[50usize]) * (next_base_row[88usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_5778) * (node_5970))) + ((node_5855) * (node_5970)),
            (((current_base_row[282usize])
                * (((node_5987)
                    * ((challenges[48usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[49usize]) * (next_base_row[73usize]))
                                + ((challenges[50usize]) * (next_base_row[89usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_5778) * (node_5987))) + ((node_5855) * (node_5987)),
            (((current_base_row[282usize])
                * (((node_6004)
                    * ((challenges[48usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[49usize]) * (next_base_row[74usize]))
                                + ((challenges[50usize]) * (next_base_row[90usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_5778) * (node_6004))) + ((node_5855) * (node_6004)),
            (((current_base_row[282usize])
                * (((node_6021)
                    * ((challenges[48usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[49usize]) * (next_base_row[75usize]))
                                + ((challenges[50usize]) * (next_base_row[91usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_5778) * (node_6021))) + ((node_5855) * (node_6021)),
            (((current_base_row[282usize])
                * (((node_6038)
                    * ((challenges[48usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[49usize]) * (next_base_row[76usize]))
                                + ((challenges[50usize]) * (next_base_row[92usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_5778) * (node_6038))) + ((node_5855) * (node_6038)),
            (((current_base_row[282usize])
                * (((node_6055)
                    * ((challenges[48usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[49usize]) * (next_base_row[77usize]))
                                + ((challenges[50usize]) * (next_base_row[93usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_5778) * (node_6055))) + ((node_5855) * (node_6055)),
            (((current_base_row[282usize])
                * (((node_6072)
                    * ((challenges[48usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[49usize]) * (next_base_row[78usize]))
                                + ((challenges[50usize]) * (next_base_row[94usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_5778) * (node_6072))) + ((node_5855) * (node_6072)),
            (((current_base_row[282usize])
                * (((node_6089)
                    * ((challenges[48usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[49usize]) * (next_base_row[79usize]))
                                + ((challenges[50usize]) * (next_base_row[95usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_5778) * (node_6089))) + ((node_5855) * (node_6089)),
            (((current_base_row[282usize])
                * (((node_6106)
                    * ((challenges[48usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[49usize]) * (next_base_row[80usize]))
                                + ((challenges[50usize]) * (next_base_row[96usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_5778) * (node_6106))) + ((node_5855) * (node_6106)),
            ((node_6132)
                * (((node_6142)
                    * ((challenges[48usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[49usize])
                                * (((BFieldElement::from_raw_u64(1099511627520u64))
                                    * (next_base_row[130usize])) + (next_base_row[131usize])))
                                + ((challenges[50usize])
                                    * (((BFieldElement::from_raw_u64(1099511627520u64))
                                        * (next_base_row[132usize]))
                                        + (next_base_row[133usize])))))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[134usize]))))
                + ((next_base_row[129usize]) * (node_6142)),
            ((node_6132)
                * ((((((node_6158)
                    * ((challenges[51usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_6153))))
                    * ((challenges[51usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_6156))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((BFieldElement::from_raw_u64(8589934590u64))
                            * (challenges[51usize])))) + (node_6153)) + (node_6156)))
                + ((next_base_row[129usize]) * (node_6158)),
            ((node_6184)
                * (((node_6196)
                    * ((challenges[51usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((next_base_row[136usize]) * (challenges[52usize]))
                                + ((next_base_row[137usize]) * (challenges[53usize]))))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[138usize]))))
                + ((next_base_row[135usize]) * (node_6196)),
            ((node_6184)
                * (((next_ext_row[47usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((current_ext_row[47usize]) * (challenges[54usize]))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[137usize]))))
                + ((next_base_row[135usize])
                    * ((next_ext_row[47usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_ext_row[47usize])))),
            (node_6244) * (node_6360),
            (next_base_row[139usize])
                * (((node_6360)
                    * ((challenges[10usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((challenges[57usize]) * (next_base_row[142usize]))
                                + ((challenges[55usize]) * (next_base_row[143usize])))
                                + ((challenges[56usize]) * (next_base_row[145usize])))
                                + ((challenges[58usize]) * (next_base_row[147usize]))))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[148usize]))),
            (current_ext_row[50usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_176)
                        * ((challenges[7usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_167)
                                    + ((challenges[18usize])
                                        * ((next_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(4294967295u64)))))
                                    + ((challenges[19usize]) * (next_base_row[36usize]))))))),
            (current_ext_row[51usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_547)
                        * ((challenges[7usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_167)
                                    + ((challenges[18usize])
                                        * ((current_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(4294967295u64)))))
                                    + ((challenges[19usize])
                                        * (current_base_row[36usize]))))))),
            (current_ext_row[52usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[50usize])
                        * ((challenges[7usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_167)
                                    + ((challenges[18usize])
                                        * ((next_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(8589934590u64)))))
                                    + ((challenges[19usize]) * (next_base_row[35usize]))))))),
            (current_ext_row[53usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[51usize])
                        * ((challenges[7usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_167)
                                    + ((challenges[18usize])
                                        * ((current_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(8589934590u64)))))
                                    + ((challenges[19usize])
                                        * (current_base_row[35usize]))))))),
            (current_ext_row[54usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[52usize])
                        * ((challenges[7usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_167)
                                    + ((challenges[18usize])
                                        * ((next_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(12884901885u64)))))
                                    + ((challenges[19usize]) * (next_base_row[34usize]))))))),
            (current_ext_row[55usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[53usize])
                        * ((challenges[7usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_167)
                                    + ((challenges[18usize])
                                        * ((current_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(12884901885u64)))))
                                    + ((challenges[19usize])
                                        * (current_base_row[34usize]))))))),
            (current_ext_row[56usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[54usize])
                        * ((challenges[7usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_167)
                                    + ((challenges[18usize])
                                        * ((next_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(17179869180u64)))))
                                    + ((challenges[19usize]) * (next_base_row[33usize]))))))),
            (current_ext_row[57usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_1326)
                        * ((challenges[8usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_1317)
                                    + (((next_base_row[22usize])
                                        + (BFieldElement::from_raw_u64(8589934590u64)))
                                        * (challenges[21usize])))
                                    + ((next_base_row[24usize]) * (challenges[22usize]))))))),
            (current_ext_row[58usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_1402)
                        * ((challenges[8usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_1315) + (node_1410))
                                    + ((current_base_row[24usize])
                                        * (challenges[22usize]))))))),
            (current_ext_row[59usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[6usize]) * (current_ext_row[56usize]))),
            (current_ext_row[60usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[55usize])
                        * ((challenges[7usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_167)
                                    + ((challenges[18usize])
                                        * ((current_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(17179869180u64)))))
                                    + ((challenges[19usize])
                                        * (current_base_row[33usize]))))))),
            (current_ext_row[61usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[57usize])
                        * ((challenges[8usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_1317)
                                    + (((next_base_row[22usize])
                                        + (BFieldElement::from_raw_u64(12884901885u64)))
                                        * (challenges[21usize])))
                                    + ((next_base_row[25usize]) * (challenges[22usize]))))))),
            (current_ext_row[62usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[58usize])
                        * ((challenges[8usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_1315) + (node_1424))
                                    + ((current_base_row[25usize])
                                        * (challenges[22usize]))))))),
            (current_ext_row[63usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[61usize])
                        * ((challenges[8usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_1317)
                                    + (((next_base_row[22usize])
                                        + (BFieldElement::from_raw_u64(17179869180u64)))
                                        * (challenges[21usize])))
                                    + ((next_base_row[26usize]) * (challenges[22usize]))))))),
            (current_ext_row[64usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[62usize])
                        * ((challenges[8usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_1315)
                                    + (((current_base_row[22usize])
                                        + (BFieldElement::from_raw_u64(12884901885u64)))
                                        * (challenges[21usize])))
                                    + ((current_base_row[26usize])
                                        * (challenges[22usize]))))))),
            (current_ext_row[65usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[190usize])
                        * ((next_ext_row[7usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((current_ext_row[7usize])
                                    * ((current_ext_row[63usize])
                                        * ((challenges[8usize])
                                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                                * (((node_1317)
                                                    + (((next_base_row[22usize])
                                                        + (BFieldElement::from_raw_u64(21474836475u64)))
                                                        * (challenges[21usize])))
                                                    + ((next_base_row[27usize])
                                                        * (challenges[22usize]))))))))))),
            (current_ext_row[66usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_ext_row[56usize])
                        * ((challenges[7usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_167)
                                    + ((challenges[18usize])
                                        * ((next_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(21474836475u64)))))
                                    + ((challenges[19usize]) * (next_base_row[32usize]))))))
                        * ((challenges[7usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_167)
                                    + ((challenges[18usize])
                                        * ((next_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(25769803770u64)))))
                                    + ((challenges[19usize]) * (next_base_row[31usize]))))))
                        * ((challenges[7usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_167)
                                    + ((challenges[18usize])
                                        * ((next_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(30064771065u64)))))
                                    + ((challenges[19usize]) * (next_base_row[30usize]))))))),
            (current_ext_row[67usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[190usize])
                        * ((next_ext_row[7usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((current_ext_row[7usize])
                                    * ((current_ext_row[64usize])
                                        * ((challenges[8usize])
                                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                                * (((node_1315)
                                                    + (((current_base_row[22usize])
                                                        + (BFieldElement::from_raw_u64(17179869180u64)))
                                                        * (challenges[21usize])))
                                                    + ((current_base_row[27usize])
                                                        * (challenges[22usize]))))))))))),
            (current_ext_row[68usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_ext_row[60usize])
                        * ((challenges[7usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_167)
                                    + ((challenges[18usize])
                                        * ((current_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(21474836475u64)))))
                                    + ((challenges[19usize]) * (current_base_row[32usize]))))))
                        * ((challenges[7usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_167)
                                    + ((challenges[18usize])
                                        * ((current_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(25769803770u64)))))
                                    + ((challenges[19usize]) * (current_base_row[31usize]))))))
                        * ((challenges[7usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_167)
                                    + ((challenges[18usize])
                                        * ((current_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(30064771065u64)))))
                                    + ((challenges[19usize])
                                        * (current_base_row[30usize]))))))),
            (current_ext_row[69usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[6usize])
                        * (((current_ext_row[66usize])
                            * ((challenges[7usize])
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (((node_167)
                                        + ((challenges[18usize])
                                            * ((next_base_row[38usize])
                                                + (BFieldElement::from_raw_u64(34359738360u64)))))
                                        + ((challenges[19usize]) * (next_base_row[29usize]))))))
                            * ((challenges[7usize])
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (((node_167)
                                        + ((challenges[18usize])
                                            * ((next_base_row[38usize])
                                                + (BFieldElement::from_raw_u64(38654705655u64)))))
                                        + ((challenges[19usize]) * (next_base_row[28usize])))))))),
            (current_ext_row[70usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[6usize])
                        * (((current_ext_row[68usize])
                            * ((challenges[7usize])
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (((node_167)
                                        + ((challenges[18usize])
                                            * ((current_base_row[38usize])
                                                + (BFieldElement::from_raw_u64(34359738360u64)))))
                                        + ((challenges[19usize]) * (current_base_row[29usize]))))))
                            * ((challenges[7usize])
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (((node_167)
                                        + ((challenges[18usize])
                                            * ((current_base_row[38usize])
                                                + (BFieldElement::from_raw_u64(38654705655u64)))))
                                        + ((challenges[19usize])
                                            * (current_base_row[28usize])))))))),
            (current_ext_row[71usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((node_1750)
                        * ((challenges[8usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_1317) + (node_1410)) + (node_1752)))))
                        * ((challenges[8usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_1317) + (node_1424)) + (node_1758)))))
                        * ((challenges[8usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((node_1764) + (node_1765)))))),
            (current_ext_row[72usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((node_1750)
                        * ((challenges[8usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((node_1764) + (node_1752)))))
                        * ((challenges[8usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((node_1771) + (node_1758)))))
                        * ((challenges[8usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((node_1778) + (node_1765)))))),
            (current_ext_row[73usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[185usize]) * (node_181))),
            (current_ext_row[74usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[186usize])
                        * ((next_ext_row[6usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((current_ext_row[6usize])
                                    * (current_ext_row[50usize])))))),
            (current_ext_row[75usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[187usize]) * (node_325))),
            (current_ext_row[76usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[189usize])
                        * ((next_ext_row[6usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((current_ext_row[6usize])
                                    * (current_ext_row[54usize])))))),
            (current_ext_row[77usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[185usize]) * (node_550))),
            (current_ext_row[78usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[186usize])
                        * ((next_ext_row[6usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((current_ext_row[6usize])
                                    * (current_ext_row[51usize])))))),
            (current_ext_row[79usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[187usize])
                        * ((next_ext_row[6usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((current_ext_row[6usize])
                                    * (current_ext_row[53usize])))))),
            (current_ext_row[80usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[189usize])
                        * ((next_ext_row[6usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((current_ext_row[6usize])
                                    * (current_ext_row[55usize])))))),
            (current_ext_row[81usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[190usize])
                        * ((next_ext_row[6usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((current_ext_row[6usize])
                                    * (current_ext_row[60usize])))))),
            (current_ext_row[82usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[7usize])
                        * (((current_ext_row[71usize])
                            * ((challenges[8usize])
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * ((node_1771)
                                        + ((current_base_row[43usize]) * (challenges[22usize]))))))
                            * ((challenges[8usize])
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * ((node_1778)
                                        + ((current_base_row[44usize])
                                            * (challenges[22usize])))))))),
            (current_ext_row[83usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((node_4676)
                        * ((challenges[11usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((next_base_row[50usize])
                                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                        * (current_base_row[50usize]))))))
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                        * (node_4601))),
            (current_ext_row[84usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[219usize])
                        * ((((((current_base_row[185usize])
                            * ((next_ext_row[7usize])
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * ((current_ext_row[7usize]) * (node_1326)))))
                            + ((current_base_row[186usize])
                                * ((next_ext_row[7usize])
                                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                        * ((current_ext_row[7usize])
                                            * (current_ext_row[57usize]))))))
                            + ((current_base_row[187usize])
                                * ((next_ext_row[7usize])
                                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                        * ((current_ext_row[7usize])
                                            * (current_ext_row[61usize]))))))
                            + ((current_base_row[189usize])
                                * ((next_ext_row[7usize])
                                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                        * ((current_ext_row[7usize])
                                            * (current_ext_row[63usize]))))))
                            + (current_ext_row[65usize])))),
            (current_ext_row[85usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[226usize])
                        * ((((((current_base_row[185usize])
                            * ((next_ext_row[7usize])
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * ((current_ext_row[7usize]) * (node_1402)))))
                            + ((current_base_row[186usize])
                                * ((next_ext_row[7usize])
                                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                        * ((current_ext_row[7usize])
                                            * (current_ext_row[58usize]))))))
                            + ((current_base_row[187usize])
                                * ((next_ext_row[7usize])
                                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                        * ((current_ext_row[7usize])
                                            * (current_ext_row[62usize]))))))
                            + ((current_base_row[189usize])
                                * ((next_ext_row[7usize])
                                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                        * ((current_ext_row[7usize])
                                            * (current_ext_row[64usize]))))))
                            + (current_ext_row[67usize])))),
        ];
        base_constraints.into_iter().chain(ext_constraints).collect()
    }
    #[allow(unused_variables)]
    fn evaluate_terminal_constraints(
        base_row: ArrayView1<XFieldElement>,
        ext_row: ArrayView1<XFieldElement>,
        challenges: &Challenges,
    ) -> Vec<XFieldElement> {
        let base_constraints = [
            (base_row[5usize]) + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((base_row[3usize]) + (BFieldElement::from_raw_u64(18446744030759878666u64)))
                * ((base_row[6usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            base_row[10usize],
            ((base_row[62usize])
                * ((base_row[63usize])
                    + (BFieldElement::from_raw_u64(18446743897615892521u64))))
                * ((base_row[64usize])
                    + (BFieldElement::from_raw_u64(18446744047939747846u64))),
            (base_row[143usize])
                * ((base_row[142usize])
                    + (BFieldElement::from_raw_u64(18446743940565565471u64))),
            base_row[145usize],
        ];
        let ext_constraints = [
            (((ext_row[18usize]) * (ext_row[16usize]))
                + ((ext_row[19usize]) * (ext_row[17usize])))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((((base_row[62usize])
                + (BFieldElement::from_raw_u64(18446744060824649731u64)))
                * ((base_row[62usize])
                    + (BFieldElement::from_raw_u64(18446744056529682436u64))))
                * (base_row[62usize]))
                * (((((((((((challenges[0usize])
                    + ((((((base_row[65usize])
                        * (BFieldElement::from_raw_u64(18446744069414518785u64)))
                        + ((base_row[66usize])
                            * (BFieldElement::from_raw_u64(18446744069414584320u64))))
                        + ((base_row[67usize])
                            * (BFieldElement::from_raw_u64(281474976645120u64))))
                        + (base_row[68usize])) * (BFieldElement::from_raw_u64(1u64))))
                    * (challenges[0usize]))
                    + ((((((base_row[69usize])
                        * (BFieldElement::from_raw_u64(18446744069414518785u64)))
                        + ((base_row[70usize])
                            * (BFieldElement::from_raw_u64(18446744069414584320u64))))
                        + ((base_row[71usize])
                            * (BFieldElement::from_raw_u64(281474976645120u64))))
                        + (base_row[72usize])) * (BFieldElement::from_raw_u64(1u64))))
                    * (challenges[0usize]))
                    + ((((((base_row[73usize])
                        * (BFieldElement::from_raw_u64(18446744069414518785u64)))
                        + ((base_row[74usize])
                            * (BFieldElement::from_raw_u64(18446744069414584320u64))))
                        + ((base_row[75usize])
                            * (BFieldElement::from_raw_u64(281474976645120u64))))
                        + (base_row[76usize])) * (BFieldElement::from_raw_u64(1u64))))
                    * (challenges[0usize]))
                    + ((((((base_row[77usize])
                        * (BFieldElement::from_raw_u64(18446744069414518785u64)))
                        + ((base_row[78usize])
                            * (BFieldElement::from_raw_u64(18446744069414584320u64))))
                        + ((base_row[79usize])
                            * (BFieldElement::from_raw_u64(281474976645120u64))))
                        + (base_row[80usize])) * (BFieldElement::from_raw_u64(1u64))))
                    * (challenges[0usize])) + (base_row[97usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (challenges[62usize]))),
            (ext_row[47usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (challenges[61usize])),
            (ext_row[2usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[24usize])),
            (challenges[59usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[3usize])),
            (ext_row[4usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (challenges[60usize])),
            (ext_row[5usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[0usize])),
            (ext_row[6usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[14usize])),
            (ext_row[7usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[20usize])),
            (ext_row[8usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[22usize])),
            (ext_row[9usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[25usize])),
            (ext_row[26usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[10usize])),
            (ext_row[11usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[27usize])),
            ((((((((((((((((ext_row[44usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[28usize])))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[29usize])))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[30usize])))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[31usize])))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[32usize])))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[33usize])))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[34usize])))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[35usize])))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[36usize])))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[37usize])))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[38usize])))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[39usize])))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[40usize])))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[41usize])))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[42usize])))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[43usize])),
            (ext_row[45usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[46usize])),
            (ext_row[12usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[48usize])),
            (((ext_row[13usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[15usize])))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[21usize])))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[23usize])),
        ];
        base_constraints.into_iter().chain(ext_constraints).collect()
    }
}
impl Quotientable for MasterExtTable {
    const NUM_INITIAL_CONSTRAINTS: usize = 81usize;
    const NUM_CONSISTENCY_CONSTRAINTS: usize = 94usize;
    const NUM_TRANSITION_CONSTRAINTS: usize = 386usize;
    const NUM_TERMINAL_CONSTRAINTS: usize = 23usize;
    #[allow(unused_variables)]
    fn initial_quotient_degree_bounds(interpolant_degree: isize) -> Vec<isize> {
        let zerofier_degree = 1;
        [
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
        ]
            .to_vec()
    }
    #[allow(unused_variables)]
    fn consistency_quotient_degree_bounds(
        interpolant_degree: isize,
        padded_height: usize,
    ) -> Vec<isize> {
        let zerofier_degree = padded_height as isize;
        [
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
        ]
            .to_vec()
    }
    #[allow(unused_variables)]
    fn transition_quotient_degree_bounds(
        interpolant_degree: isize,
        padded_height: usize,
    ) -> Vec<isize> {
        let zerofier_degree = padded_height as isize - 1;
        [
            interpolant_degree - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
        ]
            .to_vec()
    }
    #[allow(unused_variables)]
    fn terminal_quotient_degree_bounds(interpolant_degree: isize) -> Vec<isize> {
        let zerofier_degree = 1;
        [
            interpolant_degree - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree * 3isize - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree * 2isize - zerofier_degree,
            interpolant_degree * 4isize - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
        ]
            .to_vec()
    }
}
