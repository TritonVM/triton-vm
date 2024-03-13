use ndarray::ArrayView1;
use twenty_first::prelude::BFieldElement;
use twenty_first::prelude::XFieldElement;
use twenty_first::shared_math::mpolynomial::Degree;
use crate::table::challenges::Challenges;
use crate::table::challenges::ChallengeId::*;
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
        let node_503 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (base_row[129usize]));
        let node_509 = ((challenges[LookupTableInputWeight]) * (base_row[131usize]))
            + ((challenges[LookupTableOutputWeight]) * (base_row[133usize]));
        let node_512 = ((challenges[LookupTableInputWeight]) * (base_row[130usize]))
            + ((challenges[LookupTableOutputWeight]) * (base_row[132usize]));
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
                    * (challenges[ProgramAttestationPrepareChunkIndeterminate])))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (base_row[1usize])),
            (ext_row[2usize]) + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((((((((((challenges[CompressProgramDigestIndeterminate])
                + (base_row[33usize]))
                * (challenges[CompressProgramDigestIndeterminate]))
                + (base_row[34usize]))
                * (challenges[CompressProgramDigestIndeterminate]))
                + (base_row[35usize]))
                * (challenges[CompressProgramDigestIndeterminate]))
                + (base_row[36usize]))
                * (challenges[CompressProgramDigestIndeterminate]))
                + (base_row[37usize]))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (challenges[CompressedProgramDigest])),
            (ext_row[3usize]) + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[5usize])
                * ((challenges[InstructionLookupIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[ProgramInstructionWeight]) * (base_row[10usize]))
                            + ((challenges[ProgramNextInstructionWeight])
                                * (base_row[11usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            (ext_row[4usize]) + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            (ext_row[6usize]) + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            (ext_row[7usize]) + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            (ext_row[8usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((challenges[JumpStackIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((challenges[JumpStackCiWeight]) * (base_row[10usize]))))),
            ext_row[13usize],
            (((base_row[10usize])
                + (BFieldElement::from_raw_u64(18446743992105173011u64)))
                * ((ext_row[9usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((base_row[150usize])
                    * ((ext_row[9usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (challenges[HashInputIndeterminate])))),
            (ext_row[10usize]) + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            (ext_row[11usize]) + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ext_row[12usize],
            (((ext_row[14usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((challenges[OpStackIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((challenges[OpStackClkWeight]) * (base_row[46usize]))
                                + ((challenges[OpStackIb1Weight]) * (base_row[47usize])))
                                + ((challenges[OpStackPointerWeight])
                                    * (BFieldElement::from_raw_u64(68719476720u64))))
                                + ((challenges[OpStackFirstUnderflowElementWeight])
                                    * (base_row[49usize])))))))
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
                    * (challenges[RamTableBezoutRelationIndeterminate])))
                + (base_row[52usize]),
            (ext_row[17usize]) + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((((ext_row[20usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (challenges[RamIndeterminate])))
                + (((((base_row[50usize]) * (challenges[RamClkWeight]))
                    + ((base_row[51usize]) * (challenges[RamInstructionTypeWeight])))
                    + ((base_row[52usize]) * (challenges[RamPointerWeight])))
                    + ((base_row[53usize]) * (challenges[RamValueWeight]))))
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
                    * ((challenges[JumpStackIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((challenges[JumpStackCiWeight]) * (base_row[58usize]))))),
            ext_row[23usize],
            (ext_row[25usize]) + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            (ext_row[26usize]) + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            (ext_row[27usize]) + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[24usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (challenges[ProgramAttestationSendChunkIndeterminate])))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((((((((((((((((((challenges[ProgramAttestationPrepareChunkIndeterminate])
                        + ((((((base_row[65usize])
                            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
                            + ((base_row[66usize])
                                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
                            + ((base_row[67usize])
                                * (BFieldElement::from_raw_u64(281474976645120u64))))
                            + (base_row[68usize]))
                            * (BFieldElement::from_raw_u64(1u64))))
                        * (challenges[ProgramAttestationPrepareChunkIndeterminate]))
                        + ((((((base_row[69usize])
                            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
                            + ((base_row[70usize])
                                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
                            + ((base_row[71usize])
                                * (BFieldElement::from_raw_u64(281474976645120u64))))
                            + (base_row[72usize]))
                            * (BFieldElement::from_raw_u64(1u64))))
                        * (challenges[ProgramAttestationPrepareChunkIndeterminate]))
                        + ((((((base_row[73usize])
                            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
                            + ((base_row[74usize])
                                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
                            + ((base_row[75usize])
                                * (BFieldElement::from_raw_u64(281474976645120u64))))
                            + (base_row[76usize]))
                            * (BFieldElement::from_raw_u64(1u64))))
                        * (challenges[ProgramAttestationPrepareChunkIndeterminate]))
                        + ((((((base_row[77usize])
                            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
                            + ((base_row[78usize])
                                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
                            + ((base_row[79usize])
                                * (BFieldElement::from_raw_u64(281474976645120u64))))
                            + (base_row[80usize]))
                            * (BFieldElement::from_raw_u64(1u64))))
                        * (challenges[ProgramAttestationPrepareChunkIndeterminate]))
                        + (base_row[97usize]))
                        * (challenges[ProgramAttestationPrepareChunkIndeterminate]))
                        + (base_row[98usize]))
                        * (challenges[ProgramAttestationPrepareChunkIndeterminate]))
                        + (base_row[99usize]))
                        * (challenges[ProgramAttestationPrepareChunkIndeterminate]))
                        + (base_row[100usize]))
                        * (challenges[ProgramAttestationPrepareChunkIndeterminate]))
                        + (base_row[101usize]))
                        * (challenges[ProgramAttestationPrepareChunkIndeterminate]))
                        + (base_row[102usize]))),
            ((ext_row[28usize])
                * ((challenges[HashCascadeLookupIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[HashCascadeLookInWeight]) * (base_row[65usize]))
                            + ((challenges[HashCascadeLookOutWeight])
                                * (base_row[81usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[29usize])
                * ((challenges[HashCascadeLookupIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[HashCascadeLookInWeight]) * (base_row[66usize]))
                            + ((challenges[HashCascadeLookOutWeight])
                                * (base_row[82usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[30usize])
                * ((challenges[HashCascadeLookupIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[HashCascadeLookInWeight]) * (base_row[67usize]))
                            + ((challenges[HashCascadeLookOutWeight])
                                * (base_row[83usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[31usize])
                * ((challenges[HashCascadeLookupIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[HashCascadeLookInWeight]) * (base_row[68usize]))
                            + ((challenges[HashCascadeLookOutWeight])
                                * (base_row[84usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[32usize])
                * ((challenges[HashCascadeLookupIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[HashCascadeLookInWeight]) * (base_row[69usize]))
                            + ((challenges[HashCascadeLookOutWeight])
                                * (base_row[85usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[33usize])
                * ((challenges[HashCascadeLookupIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[HashCascadeLookInWeight]) * (base_row[70usize]))
                            + ((challenges[HashCascadeLookOutWeight])
                                * (base_row[86usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[34usize])
                * ((challenges[HashCascadeLookupIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[HashCascadeLookInWeight]) * (base_row[71usize]))
                            + ((challenges[HashCascadeLookOutWeight])
                                * (base_row[87usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[35usize])
                * ((challenges[HashCascadeLookupIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[HashCascadeLookInWeight]) * (base_row[72usize]))
                            + ((challenges[HashCascadeLookOutWeight])
                                * (base_row[88usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[36usize])
                * ((challenges[HashCascadeLookupIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[HashCascadeLookInWeight]) * (base_row[73usize]))
                            + ((challenges[HashCascadeLookOutWeight])
                                * (base_row[89usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[37usize])
                * ((challenges[HashCascadeLookupIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[HashCascadeLookInWeight]) * (base_row[74usize]))
                            + ((challenges[HashCascadeLookOutWeight])
                                * (base_row[90usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[38usize])
                * ((challenges[HashCascadeLookupIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[HashCascadeLookInWeight]) * (base_row[75usize]))
                            + ((challenges[HashCascadeLookOutWeight])
                                * (base_row[91usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[39usize])
                * ((challenges[HashCascadeLookupIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[HashCascadeLookInWeight]) * (base_row[76usize]))
                            + ((challenges[HashCascadeLookOutWeight])
                                * (base_row[92usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[40usize])
                * ((challenges[HashCascadeLookupIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[HashCascadeLookInWeight]) * (base_row[77usize]))
                            + ((challenges[HashCascadeLookOutWeight])
                                * (base_row[93usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[41usize])
                * ((challenges[HashCascadeLookupIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[HashCascadeLookInWeight]) * (base_row[78usize]))
                            + ((challenges[HashCascadeLookOutWeight])
                                * (base_row[94usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[42usize])
                * ((challenges[HashCascadeLookupIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[HashCascadeLookInWeight]) * (base_row[79usize]))
                            + ((challenges[HashCascadeLookOutWeight])
                                * (base_row[95usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[43usize])
                * ((challenges[HashCascadeLookupIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[HashCascadeLookInWeight]) * (base_row[80usize]))
                            + ((challenges[HashCascadeLookOutWeight])
                                * (base_row[96usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((node_503)
                * (((ext_row[44usize])
                    * ((challenges[HashCascadeLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[HashCascadeLookInWeight])
                                * (((BFieldElement::from_raw_u64(1099511627520u64))
                                    * (base_row[130usize])) + (base_row[131usize])))
                                + ((challenges[HashCascadeLookOutWeight])
                                    * (((BFieldElement::from_raw_u64(1099511627520u64))
                                        * (base_row[132usize])) + (base_row[133usize])))))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (base_row[134usize]))))
                + ((base_row[129usize]) * (ext_row[44usize])),
            ((node_503)
                * ((((((ext_row[45usize])
                    * ((challenges[CascadeLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_509))))
                    * ((challenges[CascadeLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_512))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((BFieldElement::from_raw_u64(8589934590u64))
                            * (challenges[CascadeLookupIndeterminate])))) + (node_509))
                    + (node_512))) + ((base_row[129usize]) * (ext_row[45usize])),
            ((ext_row[46usize])
                * ((challenges[CascadeLookupIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((base_row[137usize])
                            * (challenges[LookupTableOutputWeight])))))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (base_row[138usize])),
            ((ext_row[47usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (challenges[LookupTablePublicIndeterminate])))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (base_row[137usize])),
            (((base_row[139usize])
                + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                * (ext_row[48usize]))
                + ((base_row[139usize])
                    * (((ext_row[48usize])
                        * ((challenges[U32Indeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((((challenges[U32LhsWeight]) * (base_row[143usize]))
                                    + ((challenges[U32RhsWeight]) * (base_row[145usize])))
                                    + ((challenges[U32CiWeight]) * (base_row[142usize])))
                                    + ((challenges[U32ResultWeight]) * (base_row[147usize]))))))
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
        let node_107 = (base_row[152usize])
            * ((base_row[64usize])
                + (BFieldElement::from_raw_u64(18446744047939747846u64)));
        let node_227 = (base_row[153usize])
            * ((base_row[64usize])
                + (BFieldElement::from_raw_u64(18446744047939747846u64)));
        let node_244 = ((base_row[154usize])
            * ((base_row[64usize])
                + (BFieldElement::from_raw_u64(18446744052234715141u64))))
            * ((base_row[64usize])
                + (BFieldElement::from_raw_u64(18446744047939747846u64)));
        let node_251 = ((base_row[154usize])
            * ((base_row[64usize])
                + (BFieldElement::from_raw_u64(18446744056529682436u64))))
            * ((base_row[64usize])
                + (BFieldElement::from_raw_u64(18446744047939747846u64)));
        let node_676 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (base_row[157usize]));
        let node_119 = (((base_row[63usize])
            + (BFieldElement::from_raw_u64(18446743992105173011u64)))
            * ((base_row[63usize])
                + (BFieldElement::from_raw_u64(18446743923385696291u64))))
            * ((base_row[63usize])
                + (BFieldElement::from_raw_u64(18446743863256154161u64)));
        let node_121 = (node_107) * (base_row[161usize]);
        let node_681 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (base_row[160usize]));
        let node_106 = (base_row[64usize])
            + (BFieldElement::from_raw_u64(18446744047939747846u64));
        let node_700 = (base_row[142usize])
            + (BFieldElement::from_raw_u64(18446743949155500061u64));
        let node_696 = (base_row[142usize])
            + (BFieldElement::from_raw_u64(18446743940565565471u64));
        let node_99 = (base_row[64usize])
            + (BFieldElement::from_raw_u64(18446744056529682436u64));
        let node_102 = (base_row[64usize])
            + (BFieldElement::from_raw_u64(18446744052234715141u64));
        let node_158 = ((((BFieldElement::from_raw_u64(18446744065119617025u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((base_row[65usize])
                    * (BFieldElement::from_raw_u64(281474976645120u64)))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (base_row[66usize]))) * (base_row[109usize]))
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_160 = ((((BFieldElement::from_raw_u64(18446744065119617025u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((base_row[69usize])
                    * (BFieldElement::from_raw_u64(281474976645120u64)))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (base_row[70usize]))) * (base_row[110usize]))
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_162 = ((((BFieldElement::from_raw_u64(18446744065119617025u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((base_row[73usize])
                    * (BFieldElement::from_raw_u64(281474976645120u64)))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (base_row[74usize]))) * (base_row[111usize]))
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_164 = ((((BFieldElement::from_raw_u64(18446744065119617025u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((base_row[77usize])
                    * (BFieldElement::from_raw_u64(281474976645120u64)))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (base_row[78usize]))) * (base_row[112usize]))
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_702 = (base_row[139usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_95 = (base_row[64usize])
            + (BFieldElement::from_raw_u64(18446744060824649731u64));
        let node_692 = (base_row[142usize])
            + (BFieldElement::from_raw_u64(18446744017874976781u64));
        let node_11 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (((BFieldElement::from_raw_u64(38654705655u64))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (base_row[3usize]))) * (base_row[4usize])));
        let node_8 = (BFieldElement::from_raw_u64(38654705655u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (base_row[3usize]));
        let node_109 = (((base_row[62usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64)))
            * ((base_row[62usize])
                + (BFieldElement::from_raw_u64(18446744060824649731u64))))
            * ((base_row[62usize])
                + (BFieldElement::from_raw_u64(18446744056529682436u64)));
        let node_87 = (base_row[62usize])
            + (BFieldElement::from_raw_u64(18446744060824649731u64));
        let node_74 = (base_row[63usize])
            + (BFieldElement::from_raw_u64(18446743992105173011u64));
        let node_80 = (base_row[63usize])
            + (BFieldElement::from_raw_u64(18446743923385696291u64));
        let node_83 = (base_row[63usize])
            + (BFieldElement::from_raw_u64(18446743863256154161u64));
        let node_131 = ((BFieldElement::from_raw_u64(18446744065119617025u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((base_row[65usize])
                    * (BFieldElement::from_raw_u64(281474976645120u64)))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (base_row[66usize]));
        let node_138 = ((BFieldElement::from_raw_u64(18446744065119617025u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((base_row[69usize])
                    * (BFieldElement::from_raw_u64(281474976645120u64)))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (base_row[70usize]));
        let node_145 = ((BFieldElement::from_raw_u64(18446744065119617025u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((base_row[73usize])
                    * (BFieldElement::from_raw_u64(281474976645120u64)))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (base_row[74usize]));
        let node_152 = ((BFieldElement::from_raw_u64(18446744065119617025u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((base_row[77usize])
                    * (BFieldElement::from_raw_u64(281474976645120u64)))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (base_row[78usize]));
        let node_93 = (base_row[64usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_684 = (base_row[142usize])
            + (BFieldElement::from_raw_u64(18446744052234715141u64));
        let node_688 = (base_row[142usize])
            + (BFieldElement::from_raw_u64(18446744009285042191u64));
        let node_88 = ((base_row[62usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64))) * (node_87);
        let node_84 = (base_row[62usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_108 = (base_row[62usize])
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
            (node_109) * (base_row[62usize]),
            (node_87) * (node_74),
            ((base_row[165usize]) * (node_80)) * (node_83),
            (node_109) * (base_row[64usize]),
            (node_119) * (base_row[64usize]),
            (node_158) * (base_row[109usize]),
            (node_160) * (base_row[110usize]),
            (node_162) * (base_row[111usize]),
            (node_164) * (base_row[112usize]),
            (node_158) * (node_131),
            (node_160) * (node_138),
            (node_162) * (node_145),
            (node_164) * (node_152),
            (node_158)
                * (((base_row[67usize])
                    * (BFieldElement::from_raw_u64(281474976645120u64)))
                    + (base_row[68usize])),
            (node_160)
                * (((base_row[71usize])
                    * (BFieldElement::from_raw_u64(281474976645120u64)))
                    + (base_row[72usize])),
            (node_162)
                * (((base_row[75usize])
                    * (BFieldElement::from_raw_u64(281474976645120u64)))
                    + (base_row[76usize])),
            (node_164)
                * (((base_row[79usize])
                    * (BFieldElement::from_raw_u64(281474976645120u64)))
                    + (base_row[80usize])),
            (node_119) * (base_row[103usize]),
            (node_119) * (base_row[104usize]),
            (node_119) * (base_row[105usize]),
            (node_119) * (base_row[106usize]),
            (node_119) * (base_row[107usize]),
            (node_119) * (base_row[108usize]),
            (node_121)
                * ((base_row[103usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (node_121)
                * ((base_row[104usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (node_121)
                * ((base_row[105usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (node_121)
                * ((base_row[106usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (node_121)
                * ((base_row[107usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (node_121)
                * ((base_row[108usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (((((node_107)
                * ((base_row[113usize])
                    + (BFieldElement::from_raw_u64(11408918724931329738u64))))
                + ((node_227)
                    * ((base_row[113usize])
                        + (BFieldElement::from_raw_u64(16073625066478178581u64)))))
                + ((base_row[155usize])
                    * ((base_row[113usize])
                        + (BFieldElement::from_raw_u64(12231462398569191607u64)))))
                + ((node_244)
                    * ((base_row[113usize])
                        + (BFieldElement::from_raw_u64(9408518518620565480u64)))))
                + ((node_251)
                    * ((base_row[113usize])
                        + (BFieldElement::from_raw_u64(11492978409391175103u64)))),
            (((((node_107)
                * ((base_row[114usize])
                    + (BFieldElement::from_raw_u64(2786462832312611053u64))))
                + ((node_227)
                    * ((base_row[114usize])
                        + (BFieldElement::from_raw_u64(11837051899140380443u64)))))
                + ((base_row[155usize])
                    * ((base_row[114usize])
                        + (BFieldElement::from_raw_u64(11546487907579866869u64)))))
                + ((node_244)
                    * ((base_row[114usize])
                        + (BFieldElement::from_raw_u64(1785884128667671832u64)))))
                + ((node_251)
                    * ((base_row[114usize])
                        + (BFieldElement::from_raw_u64(17615222217495663839u64)))),
            (((((node_107)
                * ((base_row[115usize])
                    + (BFieldElement::from_raw_u64(6782977121958050999u64))))
                + ((node_227)
                    * ((base_row[115usize])
                        + (BFieldElement::from_raw_u64(15625104599191418968u64)))))
                + ((base_row[155usize])
                    * ((base_row[115usize])
                        + (BFieldElement::from_raw_u64(14006427992450931468u64)))))
                + ((node_244)
                    * ((base_row[115usize])
                        + (BFieldElement::from_raw_u64(1188899344229954938u64)))))
                + ((node_251)
                    * ((base_row[115usize])
                        + (BFieldElement::from_raw_u64(5864349944556149748u64)))),
            (((((node_107)
                * ((base_row[116usize])
                    + (BFieldElement::from_raw_u64(8688421733879975670u64))))
                + ((node_227)
                    * ((base_row[116usize])
                        + (BFieldElement::from_raw_u64(12819157612210448391u64)))))
                + ((base_row[155usize])
                    * ((base_row[116usize])
                        + (BFieldElement::from_raw_u64(11770003398407723041u64)))))
                + ((node_244)
                    * ((base_row[116usize])
                        + (BFieldElement::from_raw_u64(14740727267735052728u64)))))
                + ((node_251)
                    * ((base_row[116usize])
                        + (BFieldElement::from_raw_u64(2745609811140253793u64)))),
            (((((node_107)
                * ((base_row[117usize])
                    + (BFieldElement::from_raw_u64(8602724563769480463u64))))
                + ((node_227)
                    * ((base_row[117usize])
                        + (BFieldElement::from_raw_u64(6235256903503367222u64)))))
                + ((base_row[155usize])
                    * ((base_row[117usize])
                        + (BFieldElement::from_raw_u64(15124190001489436038u64)))))
                + ((node_244)
                    * ((base_row[117usize])
                        + (BFieldElement::from_raw_u64(880257844992994007u64)))))
                + ((node_251)
                    * ((base_row[117usize])
                        + (BFieldElement::from_raw_u64(15189664869386394185u64)))),
            (((((node_107)
                * ((base_row[118usize])
                    + (BFieldElement::from_raw_u64(13589155570211330507u64))))
                + ((node_227)
                    * ((base_row[118usize])
                        + (BFieldElement::from_raw_u64(11242082964257948320u64)))))
                + ((base_row[155usize])
                    * ((base_row[118usize])
                        + (BFieldElement::from_raw_u64(14834674155811570980u64)))))
                + ((node_244)
                    * ((base_row[118usize])
                        + (BFieldElement::from_raw_u64(10737952517017171197u64)))))
                + ((node_251)
                    * ((base_row[118usize])
                        + (BFieldElement::from_raw_u64(5192963426821415349u64)))),
            (((((node_107)
                * ((base_row[119usize])
                    + (BFieldElement::from_raw_u64(10263462378312899510u64))))
                + ((node_227)
                    * ((base_row[119usize])
                        + (BFieldElement::from_raw_u64(5820425254787221108u64)))))
                + ((base_row[155usize])
                    * ((base_row[119usize])
                        + (BFieldElement::from_raw_u64(13004675752386552573u64)))))
                + ((node_244)
                    * ((base_row[119usize])
                        + (BFieldElement::from_raw_u64(15757222735741919824u64)))))
                + ((node_251)
                    * ((base_row[119usize])
                        + (BFieldElement::from_raw_u64(11971160388083607515u64)))),
            (((((node_107)
                * ((base_row[120usize])
                    + (BFieldElement::from_raw_u64(3264875873073042616u64))))
                + ((node_227)
                    * ((base_row[120usize])
                        + (BFieldElement::from_raw_u64(12019227591549292608u64)))))
                + ((base_row[155usize])
                    * ((base_row[120usize])
                        + (BFieldElement::from_raw_u64(1475232519215872482u64)))))
                + ((node_244)
                    * ((base_row[120usize])
                        + (BFieldElement::from_raw_u64(14382578632612566479u64)))))
                + ((node_251)
                    * ((base_row[120usize])
                        + (BFieldElement::from_raw_u64(11608544217838050708u64)))),
            (((((node_107)
                * ((base_row[121usize])
                    + (BFieldElement::from_raw_u64(3133435276616064683u64))))
                + ((node_227)
                    * ((base_row[121usize])
                        + (BFieldElement::from_raw_u64(4625353063880731092u64)))))
                + ((base_row[155usize])
                    * ((base_row[121usize])
                        + (BFieldElement::from_raw_u64(4883869161905122316u64)))))
                + ((node_244)
                    * ((base_row[121usize])
                        + (BFieldElement::from_raw_u64(3305272539067787726u64)))))
                + ((node_251)
                    * ((base_row[121usize])
                        + (BFieldElement::from_raw_u64(674972795234232729u64)))),
            (((((node_107)
                * ((base_row[122usize])
                    + (BFieldElement::from_raw_u64(13508500531157332153u64))))
                + ((node_227)
                    * ((base_row[122usize])
                        + (BFieldElement::from_raw_u64(3723900760706330287u64)))))
                + ((base_row[155usize])
                    * ((base_row[122usize])
                        + (BFieldElement::from_raw_u64(12579737103870920763u64)))))
                + ((node_244)
                    * ((base_row[122usize])
                        + (BFieldElement::from_raw_u64(17082569335437832789u64)))))
                + ((node_251)
                    * ((base_row[122usize])
                        + (BFieldElement::from_raw_u64(14165256104883557753u64)))),
            (((((node_107)
                * ((base_row[123usize])
                    + (BFieldElement::from_raw_u64(6968886508437513677u64))))
                + ((node_227)
                    * ((base_row[123usize])
                        + (BFieldElement::from_raw_u64(615596267195055952u64)))))
                + ((base_row[155usize])
                    * ((base_row[123usize])
                        + (BFieldElement::from_raw_u64(10119826060478909841u64)))))
                + ((node_244)
                    * ((base_row[123usize])
                        + (BFieldElement::from_raw_u64(229051680548583225u64)))))
                + ((node_251)
                    * ((base_row[123usize])
                        + (BFieldElement::from_raw_u64(15283356519694111298u64)))),
            (((((node_107)
                * ((base_row[124usize])
                    + (BFieldElement::from_raw_u64(9713264609690967820u64))))
                + ((node_227)
                    * ((base_row[124usize])
                        + (BFieldElement::from_raw_u64(18227830850447556704u64)))))
                + ((base_row[155usize])
                    * ((base_row[124usize])
                        + (BFieldElement::from_raw_u64(1528714547662620921u64)))))
                + ((node_244)
                    * ((base_row[124usize])
                        + (BFieldElement::from_raw_u64(2943254981416254648u64)))))
                + ((node_251)
                    * ((base_row[124usize])
                        + (BFieldElement::from_raw_u64(2306049938060341466u64)))),
            (((((node_107)
                * ((base_row[125usize])
                    + (BFieldElement::from_raw_u64(12482374976099749513u64))))
                + ((node_227)
                    * ((base_row[125usize])
                        + (BFieldElement::from_raw_u64(15609691041895848348u64)))))
                + ((base_row[155usize])
                    * ((base_row[125usize])
                        + (BFieldElement::from_raw_u64(12972275929555275935u64)))))
                + ((node_244)
                    * ((base_row[125usize])
                        + (BFieldElement::from_raw_u64(5767629304344025219u64)))))
                + ((node_251)
                    * ((base_row[125usize])
                        + (BFieldElement::from_raw_u64(11578793764462375094u64)))),
            (((((node_107)
                * ((base_row[126usize])
                    + (BFieldElement::from_raw_u64(13209711277645656680u64))))
                + ((node_227)
                    * ((base_row[126usize])
                        + (BFieldElement::from_raw_u64(15235800289984546486u64)))))
                + ((base_row[155usize])
                    * ((base_row[126usize])
                        + (BFieldElement::from_raw_u64(15992731669612695172u64)))))
                + ((node_244)
                    * ((base_row[126usize])
                        + (BFieldElement::from_raw_u64(16721422493821450473u64)))))
                + ((node_251)
                    * ((base_row[126usize])
                        + (BFieldElement::from_raw_u64(7511767364422267184u64)))),
            (((((node_107)
                * ((base_row[127usize])
                    + (BFieldElement::from_raw_u64(87705059284758253u64))))
                + ((node_227)
                    * ((base_row[127usize])
                        + (BFieldElement::from_raw_u64(11392407538241985753u64)))))
                + ((base_row[155usize])
                    * ((base_row[127usize])
                        + (BFieldElement::from_raw_u64(17877154195438905917u64)))))
                + ((node_244)
                    * ((base_row[127usize])
                        + (BFieldElement::from_raw_u64(5753720429376839714u64)))))
                + ((node_251)
                    * ((base_row[127usize])
                        + (BFieldElement::from_raw_u64(16999805755930336630u64)))),
            (((((node_107)
                * ((base_row[128usize])
                    + (BFieldElement::from_raw_u64(330155256278907084u64))))
                + ((node_227)
                    * ((base_row[128usize])
                        + (BFieldElement::from_raw_u64(11776128816341368822u64)))))
                + ((base_row[155usize])
                    * ((base_row[128usize])
                        + (BFieldElement::from_raw_u64(939319986782105612u64)))))
                + ((node_244)
                    * ((base_row[128usize])
                        + (BFieldElement::from_raw_u64(2063756830275051942u64)))))
                + ((node_251)
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
            (base_row[144usize]) * (node_676),
            (base_row[143usize]) * (node_676),
            (base_row[146usize]) * (node_681),
            (base_row[145usize]) * (node_681),
            (base_row[167usize])
                * ((base_row[147usize])
                    + (BFieldElement::from_raw_u64(18446744060824649731u64))),
            (base_row[168usize]) * (base_row[147usize]),
            (((base_row[163usize]) * (node_676)) * (node_681)) * (base_row[147usize]),
            (((base_row[166usize]) * (node_700)) * (node_681))
                * ((base_row[147usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (((base_row[164usize]) * (node_702)) * (node_676))
                * ((base_row[147usize]) + (BFieldElement::from_raw_u64(4294967295u64))),
            (((base_row[166usize]) * (node_696)) * (node_676)) * (base_row[147usize]),
            ((base_row[164usize]) * (base_row[139usize])) * (node_676),
            (node_702) * (base_row[148usize]),
            (base_row[151usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((base_row[64usize]) * (node_93))),
            (base_row[152usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((node_93) * (node_95)) * (node_99)) * (node_102))),
            (base_row[153usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((base_row[64usize]) * (node_95)) * (node_99)) * (node_102))),
            (base_row[154usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((base_row[151usize]) * (node_95))),
            (base_row[155usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((base_row[151usize]) * (node_99)) * (node_102)) * (node_106))),
            (base_row[156usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_684)
                        * ((base_row[142usize])
                            + (BFieldElement::from_raw_u64(18446744043644780551u64))))),
            (base_row[157usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((base_row[143usize]) * (base_row[144usize]))),
            (base_row[158usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((node_684) * (node_688)) * (node_692)) * (node_696))),
            (base_row[159usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((base_row[156usize]) * (node_688))),
            (base_row[160usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((base_row[145usize]) * (base_row[146usize]))),
            (base_row[161usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_88) * (base_row[62usize]))),
            (base_row[162usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((base_row[158usize]) * (node_700))),
            (base_row[163usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((base_row[156usize]) * (node_692)) * (node_696)) * (node_700))),
            (base_row[164usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((base_row[159usize]) * (node_696)) * (node_700))),
            (base_row[165usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((node_84) * (node_108)) * (base_row[62usize]))
                        * ((base_row[63usize])
                            + (BFieldElement::from_raw_u64(18446743897615892521u64))))),
            (base_row[166usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((base_row[159usize]) * (node_692))),
            (base_row[167usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((base_row[162usize]) * (node_702)) * (node_676)) * (node_681))),
            (base_row[168usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((base_row[162usize]) * (base_row[139usize])) * (node_676))
                        * (node_681))),
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
        let node_4849 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (next_base_row[8usize]));
        let node_121 = (next_base_row[19usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[19usize]));
        let node_537 = (next_ext_row[3usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[3usize]));
        let node_541 = (next_ext_row[4usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[4usize]));
        let node_125 = (next_base_row[20usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[20usize]));
        let node_129 = (next_base_row[21usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[21usize]));
        let node_533 = (next_ext_row[7usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[7usize]));
        let node_1472 = (current_base_row[18usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_932 = ((next_base_row[9usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[9usize])))
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_159 = (next_base_row[38usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[38usize]));
        let node_1470 = (current_base_row[17usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_824 = (next_base_row[22usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[22usize]));
        let node_169 = ((challenges[OpStackClkWeight]) * (current_base_row[7usize]))
            + ((challenges[OpStackIb1Weight]) * (current_base_row[13usize]));
        let node_545 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[22usize]);
        let node_6564 = (current_base_row[276usize])
            * ((next_base_row[64usize])
                + (BFieldElement::from_raw_u64(18446744052234715141u64)));
        let node_6643 = (((next_base_row[63usize])
            + (BFieldElement::from_raw_u64(18446743992105173011u64)))
            * ((next_base_row[63usize])
                + (BFieldElement::from_raw_u64(18446743923385696291u64))))
            * ((next_base_row[63usize])
                + (BFieldElement::from_raw_u64(18446743863256154161u64)));
        let node_5588 = (((((current_base_row[81usize])
            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
            + ((current_base_row[82usize])
                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
            + ((current_base_row[83usize])
                * (BFieldElement::from_raw_u64(281474976645120u64))))
            + (current_base_row[84usize])) * (BFieldElement::from_raw_u64(1u64));
        let node_5599 = (((((current_base_row[85usize])
            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
            + ((current_base_row[86usize])
                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
            + ((current_base_row[87usize])
                * (BFieldElement::from_raw_u64(281474976645120u64))))
            + (current_base_row[88usize])) * (BFieldElement::from_raw_u64(1u64));
        let node_5610 = (((((current_base_row[89usize])
            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
            + ((current_base_row[90usize])
                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
            + ((current_base_row[91usize])
                * (BFieldElement::from_raw_u64(281474976645120u64))))
            + (current_base_row[92usize])) * (BFieldElement::from_raw_u64(1u64));
        let node_5621 = (((((current_base_row[93usize])
            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
            + ((current_base_row[94usize])
                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
            + ((current_base_row[95usize])
                * (BFieldElement::from_raw_u64(281474976645120u64))))
            + (current_base_row[96usize])) * (BFieldElement::from_raw_u64(1u64));
        let node_203 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[27usize]);
        let node_872 = (next_base_row[24usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[24usize]));
        let node_876 = (next_base_row[25usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[25usize]));
        let node_880 = (next_base_row[26usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[26usize]));
        let node_868 = (next_base_row[23usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[23usize]));
        let node_199 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[26usize]);
        let node_298 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[39usize]));
        let node_884 = (next_base_row[27usize]) + (node_203);
        let node_888 = (next_base_row[28usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[28usize]));
        let node_892 = (next_base_row[29usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[29usize]));
        let node_896 = (next_base_row[30usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[30usize]));
        let node_900 = (next_base_row[31usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[31usize]));
        let node_904 = (next_base_row[32usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[32usize]));
        let node_908 = (next_base_row[33usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[33usize]));
        let node_912 = (next_base_row[34usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[34usize]));
        let node_916 = (next_base_row[35usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[35usize]));
        let node_920 = (next_base_row[36usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[36usize]));
        let node_924 = (next_base_row[37usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[37usize]));
        let node_927 = (next_ext_row[6usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[6usize]));
        let node_195 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[25usize]);
        let node_207 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[28usize]);
        let node_223 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[32usize]);
        let node_211 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[29usize]);
        let node_227 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[33usize]);
        let node_215 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[30usize]);
        let node_219 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[31usize]);
        let node_1468 = (current_base_row[16usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_200 = (next_base_row[25usize]) + (node_199);
        let node_204 = (next_base_row[26usize]) + (node_203);
        let node_208 = (next_base_row[27usize]) + (node_207);
        let node_231 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[34usize]);
        let node_212 = (next_base_row[28usize]) + (node_211);
        let node_216 = (next_base_row[29usize]) + (node_215);
        let node_220 = (next_base_row[30usize]) + (node_219);
        let node_160 = (node_159) + (BFieldElement::from_raw_u64(4294967295u64));
        let node_224 = (next_base_row[31usize]) + (node_223);
        let node_184 = (next_ext_row[6usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[73usize]));
        let node_228 = (next_base_row[32usize]) + (node_227);
        let node_232 = (next_base_row[33usize]) + (node_231);
        let node_236 = (next_base_row[34usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[35usize]));
        let node_240 = (next_base_row[35usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[36usize]));
        let node_244 = (next_base_row[36usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[37usize]));
        let node_7032 = (next_base_row[139usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_187 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[23usize]);
        let node_191 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[24usize]);
        let node_117 = ((next_base_row[9usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[9usize])))
            + (BFieldElement::from_raw_u64(18446744060824649731u64));
        let node_192 = (next_base_row[23usize]) + (node_191);
        let node_196 = (next_base_row[24usize]) + (node_195);
        let node_235 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[35usize]);
        let node_5355 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (((next_base_row[52usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[52usize]))) * (current_base_row[54usize])));
        let node_132 = (current_base_row[39usize])
            * ((current_base_row[39usize])
                + (BFieldElement::from_raw_u64(18446744065119617026u64)));
        let node_239 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[36usize]);
        let node_155 = ((((current_base_row[11usize])
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
        let node_243 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[37usize]);
        let node_5352 = (next_base_row[52usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[52usize]));
        let node_6404 = (next_base_row[62usize])
            + (BFieldElement::from_raw_u64(18446744056529682436u64));
        let node_7028 = (current_base_row[145usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((BFieldElement::from_raw_u64(8589934590u64))
                    * (next_base_row[145usize])));
        let node_5229 = (next_ext_row[12usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[12usize]));
        let node_7025 = (current_base_row[143usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((BFieldElement::from_raw_u64(8589934590u64))
                    * (next_base_row[143usize])));
        let node_1466 = (current_base_row[15usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_994 = (current_base_row[7usize]) * (challenges[RamClkWeight]);
        let node_1372 = (challenges[StandardInputIndeterminate])
            * (current_ext_row[3usize]);
        let node_6408 = (next_base_row[63usize])
            + (BFieldElement::from_raw_u64(18446743897615892521u64));
        let node_997 = (node_994) + (challenges[RamInstructionTypeWeight]);
        let node_542 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[11usize]);
        let node_934 = (current_base_row[272usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_114 = (next_base_row[9usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[9usize]));
        let node_547 = (next_base_row[24usize]) + (node_187);
        let node_548 = (next_base_row[25usize]) + (node_191);
        let node_549 = (next_base_row[26usize]) + (node_195);
        let node_550 = (next_base_row[27usize]) + (node_199);
        let node_551 = (next_base_row[28usize]) + (node_203);
        let node_552 = (next_base_row[29usize]) + (node_207);
        let node_553 = (next_base_row[30usize]) + (node_211);
        let node_554 = (next_base_row[31usize]) + (node_215);
        let node_561 = (node_159)
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_555 = (next_base_row[32usize]) + (node_219);
        let node_556 = (next_base_row[33usize]) + (node_223);
        let node_557 = (next_base_row[34usize]) + (node_227);
        let node_558 = (next_base_row[35usize]) + (node_231);
        let node_559 = (next_base_row[36usize]) + (node_235);
        let node_560 = (next_base_row[37usize]) + (node_239);
        let node_572 = (next_ext_row[6usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((current_ext_row[6usize])
                    * ((challenges[OpStackIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((node_169)
                                + ((challenges[OpStackPointerWeight])
                                    * (current_base_row[38usize])))
                                + ((challenges[OpStackFirstUnderflowElementWeight])
                                    * (current_base_row[37usize])))))));
        let node_5461 = ((next_base_row[59usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[59usize])))
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_6264 = (((((next_base_row[65usize])
            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
            + ((next_base_row[66usize])
                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
            + ((next_base_row[67usize])
                * (BFieldElement::from_raw_u64(281474976645120u64))))
            + (next_base_row[68usize])) * (BFieldElement::from_raw_u64(1u64));
        let node_6275 = (((((next_base_row[69usize])
            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
            + ((next_base_row[70usize])
                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
            + ((next_base_row[71usize])
                * (BFieldElement::from_raw_u64(281474976645120u64))))
            + (next_base_row[72usize])) * (BFieldElement::from_raw_u64(1u64));
        let node_6286 = (((((next_base_row[73usize])
            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
            + ((next_base_row[74usize])
                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
            + ((next_base_row[75usize])
                * (BFieldElement::from_raw_u64(281474976645120u64))))
            + (next_base_row[76usize])) * (BFieldElement::from_raw_u64(1u64));
        let node_6297 = (((((next_base_row[77usize])
            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
            + ((next_base_row[78usize])
                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
            + ((next_base_row[79usize])
                * (BFieldElement::from_raw_u64(281474976645120u64))))
            + (next_base_row[80usize])) * (BFieldElement::from_raw_u64(1u64));
        let node_6398 = (next_base_row[62usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_6972 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (next_base_row[135usize]));
        let node_7060 = (next_base_row[142usize])
            + (BFieldElement::from_raw_u64(18446743940565565471u64));
        let node_1464 = (current_base_row[14usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_251 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[40usize]));
        let node_7064 = (next_base_row[142usize])
            + (BFieldElement::from_raw_u64(18446743949155500061u64));
        let node_34 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((current_base_row[4usize])
                    * ((BFieldElement::from_raw_u64(38654705655u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[3usize])))));
        let node_1295 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[337usize]));
        let node_546 = (next_base_row[23usize]) + (node_545);
        let node_256 = (current_base_row[185usize])
            * ((next_base_row[22usize]) + (node_187));
        let node_302 = (current_base_row[186usize])
            * ((next_base_row[22usize]) + (node_191));
        let node_345 = (current_base_row[187usize])
            * ((next_base_row[22usize]) + (node_195));
        let node_332 = (next_base_row[25usize]) + (node_207);
        let node_387 = (current_base_row[188usize])
            * ((next_base_row[22usize]) + (node_199));
        let node_417 = (next_base_row[27usize]) + (node_223);
        let node_333 = (next_base_row[26usize]) + (node_211);
        let node_188 = (next_base_row[22usize]) + (node_187);
        let node_418 = (next_base_row[28usize]) + (node_227);
        let node_334 = (next_base_row[27usize]) + (node_215);
        let node_419 = (next_base_row[29usize]) + (node_231);
        let node_335 = (next_base_row[28usize]) + (node_219);
        let node_420 = (next_base_row[30usize]) + (node_235);
        let node_336 = (next_base_row[29usize]) + (node_223);
        let node_421 = (next_base_row[31usize]) + (node_239);
        let node_337 = (next_base_row[30usize]) + (node_227);
        let node_441 = (((((current_base_row[185usize]) * (node_160))
            + ((current_base_row[186usize])
                * ((node_159) + (BFieldElement::from_raw_u64(8589934590u64)))))
            + ((current_base_row[187usize])
                * ((node_159) + (BFieldElement::from_raw_u64(12884901885u64)))))
            + ((current_base_row[188usize])
                * ((node_159) + (BFieldElement::from_raw_u64(17179869180u64)))))
            + ((current_base_row[189usize])
                * ((node_159) + (BFieldElement::from_raw_u64(21474836475u64))));
        let node_753 = (((((current_base_row[185usize]) * (node_561))
            + ((current_base_row[186usize])
                * ((node_159) + (BFieldElement::from_raw_u64(18446744060824649731u64)))))
            + ((current_base_row[187usize])
                * ((node_159) + (BFieldElement::from_raw_u64(18446744056529682436u64)))))
            + ((current_base_row[188usize])
                * ((node_159) + (BFieldElement::from_raw_u64(18446744052234715141u64)))))
            + ((current_base_row[189usize])
                * ((node_159) + (BFieldElement::from_raw_u64(18446744047939747846u64))));
        let node_422 = (next_base_row[32usize]) + (node_243);
        let node_400 = (node_159) + (BFieldElement::from_raw_u64(21474836475u64));
        let node_338 = (next_base_row[31usize]) + (node_231);
        let node_446 = (((((current_base_row[185usize]) * (node_184))
            + (current_ext_row[74usize]))
            + ((current_base_row[187usize])
                * ((next_ext_row[6usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_ext_row[71usize]))))) + (current_ext_row[75usize]))
            + ((current_base_row[189usize])
                * ((next_ext_row[6usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_ext_row[59usize]))));
        let node_758 = ((((current_ext_row[76usize]) + (current_ext_row[77usize]))
            + (current_ext_row[78usize])) + (current_ext_row[79usize]))
            + ((current_base_row[189usize])
                * ((next_ext_row[6usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_ext_row[72usize]))));
        let node_411 = (next_ext_row[6usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[59usize]));
        let node_339 = (next_base_row[32usize]) + (node_235);
        let node_340 = (next_base_row[33usize]) + (node_239);
        let node_456 = ((((current_base_row[287usize]) + (current_base_row[288usize]))
            + (current_base_row[289usize])) + (current_base_row[290usize]))
            + (current_base_row[291usize]);
        let node_768 = ((((current_base_row[282usize]) + (current_base_row[283usize]))
            + (current_base_row[284usize])) + (current_base_row[285usize]))
            + (current_base_row[286usize]);
        let node_341 = (next_base_row[34usize]) + (node_243);
        let node_461 = ((((current_base_row[319usize]) + (current_base_row[321usize]))
            + (current_base_row[323usize])) + (current_base_row[325usize]))
            + (current_base_row[327usize]);
        let node_773 = (((((current_base_row[185usize]) * (node_548))
            + ((current_base_row[186usize]) * ((next_base_row[26usize]) + (node_191))))
            + ((current_base_row[187usize]) * ((next_base_row[27usize]) + (node_191))))
            + ((current_base_row[188usize]) * ((next_base_row[28usize]) + (node_191))))
            + ((current_base_row[189usize]) * ((next_base_row[29usize]) + (node_191)));
        let node_317 = (node_159) + (BFieldElement::from_raw_u64(12884901885u64));
        let node_466 = ((((current_base_row[320usize]) + (current_base_row[322usize]))
            + (current_base_row[324usize])) + (current_base_row[326usize]))
            + (current_base_row[328usize]);
        let node_778 = (((((current_base_row[185usize]) * (node_549))
            + ((current_base_row[186usize]) * ((next_base_row[27usize]) + (node_195))))
            + ((current_base_row[187usize]) * ((next_base_row[28usize]) + (node_195))))
            + ((current_base_row[188usize]) * ((next_base_row[29usize]) + (node_195))))
            + ((current_base_row[189usize]) * ((next_base_row[30usize]) + (node_195)));
        let node_328 = (next_ext_row[6usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[71usize]));
        let node_471 = (((((current_base_row[185usize]) * (node_204))
            + ((current_base_row[186usize]) * ((next_base_row[26usize]) + (node_207))))
            + ((current_base_row[187usize]) * (node_333)))
            + ((current_base_row[188usize]) * ((next_base_row[26usize]) + (node_215))))
            + ((current_base_row[189usize]) * ((next_base_row[26usize]) + (node_219)));
        let node_783 = (((((current_base_row[185usize]) * (node_550))
            + ((current_base_row[186usize]) * ((next_base_row[28usize]) + (node_199))))
            + ((current_base_row[187usize]) * ((next_base_row[29usize]) + (node_199))))
            + ((current_base_row[188usize]) * ((next_base_row[30usize]) + (node_199))))
            + ((current_base_row[189usize]) * ((next_base_row[31usize]) + (node_199)));
        let node_476 = (((((current_base_row[185usize]) * (node_208))
            + ((current_base_row[186usize]) * ((next_base_row[27usize]) + (node_211))))
            + ((current_base_row[187usize]) * (node_334)))
            + ((current_base_row[188usize]) * ((next_base_row[27usize]) + (node_219))))
            + ((current_base_row[189usize]) * (node_417));
        let node_788 = (((((current_base_row[185usize]) * (node_551))
            + ((current_base_row[186usize]) * ((next_base_row[29usize]) + (node_203))))
            + ((current_base_row[187usize]) * ((next_base_row[30usize]) + (node_203))))
            + ((current_base_row[188usize]) * ((next_base_row[31usize]) + (node_203))))
            + ((current_base_row[189usize]) * ((next_base_row[32usize]) + (node_203)));
        let node_481 = (((((current_base_row[185usize]) * (node_212))
            + ((current_base_row[186usize]) * ((next_base_row[28usize]) + (node_215))))
            + ((current_base_row[187usize]) * (node_335)))
            + ((current_base_row[188usize]) * ((next_base_row[28usize]) + (node_223))))
            + ((current_base_row[189usize]) * (node_418));
        let node_793 = (((((current_base_row[185usize]) * (node_552))
            + ((current_base_row[186usize]) * ((next_base_row[30usize]) + (node_207))))
            + ((current_base_row[187usize]) * ((next_base_row[31usize]) + (node_207))))
            + ((current_base_row[188usize]) * ((next_base_row[32usize]) + (node_207))))
            + ((current_base_row[189usize]) * ((next_base_row[33usize]) + (node_207)));
        let node_486 = (((((current_base_row[185usize]) * (node_216))
            + ((current_base_row[186usize]) * ((next_base_row[29usize]) + (node_219))))
            + ((current_base_row[187usize]) * (node_336)))
            + ((current_base_row[188usize]) * ((next_base_row[29usize]) + (node_227))))
            + ((current_base_row[189usize]) * (node_419));
        let node_798 = (((((current_base_row[185usize]) * (node_553))
            + ((current_base_row[186usize]) * ((next_base_row[31usize]) + (node_211))))
            + ((current_base_row[187usize]) * ((next_base_row[32usize]) + (node_211))))
            + ((current_base_row[188usize]) * ((next_base_row[33usize]) + (node_211))))
            + ((current_base_row[189usize]) * ((next_base_row[34usize]) + (node_211)));
        let node_491 = (((((current_base_row[185usize]) * (node_220))
            + ((current_base_row[186usize]) * ((next_base_row[30usize]) + (node_223))))
            + ((current_base_row[187usize]) * (node_337)))
            + ((current_base_row[188usize]) * ((next_base_row[30usize]) + (node_231))))
            + ((current_base_row[189usize]) * (node_420));
        let node_803 = (((((current_base_row[185usize]) * (node_554))
            + ((current_base_row[186usize]) * ((next_base_row[32usize]) + (node_215))))
            + ((current_base_row[187usize]) * ((next_base_row[33usize]) + (node_215))))
            + ((current_base_row[188usize]) * ((next_base_row[34usize]) + (node_215))))
            + ((current_base_row[189usize]) * ((next_base_row[35usize]) + (node_215)));
        let node_496 = (((((current_base_row[185usize]) * (node_224))
            + ((current_base_row[186usize]) * ((next_base_row[31usize]) + (node_227))))
            + ((current_base_row[187usize]) * (node_338)))
            + ((current_base_row[188usize]) * ((next_base_row[31usize]) + (node_235))))
            + ((current_base_row[189usize]) * (node_421));
        let node_808 = (((((current_base_row[185usize]) * (node_555))
            + ((current_base_row[186usize]) * ((next_base_row[33usize]) + (node_219))))
            + ((current_base_row[187usize]) * ((next_base_row[34usize]) + (node_219))))
            + ((current_base_row[188usize]) * ((next_base_row[35usize]) + (node_219))))
            + ((current_base_row[189usize]) * ((next_base_row[36usize]) + (node_219)));
        let node_501 = (((((current_base_row[185usize]) * (node_228))
            + ((current_base_row[186usize]) * ((next_base_row[32usize]) + (node_231))))
            + ((current_base_row[187usize]) * (node_339)))
            + ((current_base_row[188usize]) * ((next_base_row[32usize]) + (node_239))))
            + ((current_base_row[189usize]) * (node_422));
        let node_813 = (((((current_base_row[185usize]) * (node_556))
            + ((current_base_row[186usize]) * ((next_base_row[34usize]) + (node_223))))
            + ((current_base_row[187usize]) * ((next_base_row[35usize]) + (node_223))))
            + ((current_base_row[188usize]) * ((next_base_row[36usize]) + (node_223))))
            + ((current_base_row[189usize]) * ((next_base_row[37usize]) + (node_223)));
        let node_505 = ((((current_base_row[185usize]) * (node_232))
            + ((current_base_row[186usize]) * ((next_base_row[33usize]) + (node_235))))
            + ((current_base_row[187usize]) * (node_340)))
            + ((current_base_row[188usize]) * ((next_base_row[33usize]) + (node_243)));
        let node_817 = ((((current_base_row[185usize]) * (node_557))
            + ((current_base_row[186usize]) * ((next_base_row[35usize]) + (node_227))))
            + ((current_base_row[187usize]) * ((next_base_row[36usize]) + (node_227))))
            + ((current_base_row[188usize]) * ((next_base_row[37usize]) + (node_227)));
        let node_508 = (((current_base_row[185usize]) * (node_236))
            + ((current_base_row[186usize]) * ((next_base_row[34usize]) + (node_239))))
            + ((current_base_row[187usize]) * (node_341));
        let node_820 = (((current_base_row[185usize]) * (node_558))
            + ((current_base_row[186usize]) * ((next_base_row[36usize]) + (node_231))))
            + ((current_base_row[187usize]) * ((next_base_row[37usize]) + (node_231)));
        let node_510 = ((current_base_row[185usize]) * (node_240))
            + ((current_base_row[186usize]) * ((next_base_row[35usize]) + (node_243)));
        let node_822 = ((current_base_row[185usize]) * (node_559))
            + ((current_base_row[186usize]) * ((next_base_row[37usize]) + (node_235)));
        let node_270 = (current_base_row[185usize]) * (node_244);
        let node_589 = (current_base_row[185usize]) * (node_560);
        let node_5166 = ((next_ext_row[11usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((challenges[SpongeIndeterminate]) * (current_ext_row[11usize]))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((challenges[HashCIWeight]) * (current_base_row[10usize])));
        let node_5193 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * ((challenges[U32CiWeight]) * (current_base_row[10usize]));
        let node_5233 = ((node_5229)
            * (((((challenges[U32Indeterminate])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((challenges[U32LhsWeight]) * (current_base_row[22usize]))))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((challenges[U32RhsWeight]) * (current_base_row[23usize]))))
                + (node_5193))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((challenges[U32ResultWeight]) * (next_base_row[22usize])))))
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_5197 = (challenges[U32Indeterminate])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((challenges[U32LhsWeight]) * (current_base_row[22usize])));
        let node_5277 = ((next_base_row[48usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[48usize])))
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_5276 = (next_base_row[48usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[48usize]));
        let node_5283 = (next_base_row[47usize])
            + (BFieldElement::from_raw_u64(18446744060824649731u64));
        let node_5309 = (next_ext_row[15usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[15usize]));
        let node_5346 = (next_base_row[51usize])
            + (BFieldElement::from_raw_u64(18446744060824649731u64));
        let node_5432 = (next_ext_row[21usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[21usize]));
        let node_5460 = (next_base_row[59usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[59usize]));
        let node_6394 = (current_base_row[62usize])
            + (BFieldElement::from_raw_u64(18446744056529682436u64));
        let node_6462 = (next_base_row[64usize])
            + (BFieldElement::from_raw_u64(18446744047939747846u64));
        let node_6513 = (next_base_row[63usize])
            + (BFieldElement::from_raw_u64(18446743992105173011u64));
        let node_6516 = (next_base_row[63usize])
            + (BFieldElement::from_raw_u64(18446743923385696291u64));
        let node_6635 = (next_ext_row[28usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[28usize]));
        let node_6656 = (next_ext_row[29usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[29usize]));
        let node_6673 = (next_ext_row[30usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[30usize]));
        let node_6690 = (next_ext_row[31usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[31usize]));
        let node_6707 = (next_ext_row[32usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[32usize]));
        let node_6724 = (next_ext_row[33usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[33usize]));
        let node_6741 = (next_ext_row[34usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[34usize]));
        let node_6758 = (next_ext_row[35usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[35usize]));
        let node_6775 = (next_ext_row[36usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[36usize]));
        let node_6792 = (next_ext_row[37usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[37usize]));
        let node_6809 = (next_ext_row[38usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[38usize]));
        let node_6826 = (next_ext_row[39usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[39usize]));
        let node_6843 = (next_ext_row[40usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[40usize]));
        let node_6860 = (next_ext_row[41usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[41usize]));
        let node_6877 = (next_ext_row[42usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[42usize]));
        let node_6894 = (next_ext_row[43usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[43usize]));
        let node_6920 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (next_base_row[129usize]));
        let node_7022 = (current_base_row[142usize])
            + (BFieldElement::from_raw_u64(18446743940565565471u64));
        let node_7049 = (node_7028)
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_7058 = (next_base_row[142usize])
            + (BFieldElement::from_raw_u64(18446744017874976781u64));
        let node_6421 = (next_base_row[62usize])
            + (BFieldElement::from_raw_u64(18446744060824649731u64));
        let node_30 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[3usize]);
        let node_48 = (next_base_row[6usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_52 = (next_ext_row[0usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[0usize]));
        let node_31 = (BFieldElement::from_raw_u64(38654705655u64)) + (node_30);
        let node_75 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (next_base_row[1usize]);
        let node_91 = (BFieldElement::from_raw_u64(38654705655u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (next_base_row[3usize]));
        let node_89 = (next_ext_row[2usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[2usize]));
        let node_931 = (next_base_row[10usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[10usize]));
        let node_128 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[21usize]);
        let node_1289 = (next_base_row[22usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[335usize]));
        let node_1311 = (current_base_row[23usize]) * (next_base_row[23usize]);
        let node_1314 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (next_base_row[22usize]);
        let node_1319 = (current_base_row[22usize]) * (current_base_row[25usize]);
        let node_1320 = (current_base_row[24usize]) * (current_base_row[26usize]);
        let node_1323 = (current_base_row[23usize]) * (current_base_row[27usize]);
        let node_1346 = (current_base_row[24usize]) * (next_base_row[23usize]);
        let node_1349 = (current_base_row[23usize]) * (next_base_row[24usize]);
        let node_575 = (current_base_row[185usize]) * (node_546);
        let node_113 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[9usize]);
        let node_124 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[20usize]);
        let node_1292 = (current_base_row[23usize]) + (node_545);
        let node_1331 = (current_base_row[24usize]) * (current_base_row[27usize]);
        let node_1356 = (current_base_row[24usize]) * (next_base_row[24usize]);
        let node_618 = (current_base_row[186usize])
            * ((next_base_row[24usize]) + (node_545));
        let node_250 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[40usize]);
        let node_145 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * ((BFieldElement::from_raw_u64(34359738360u64))
                * (current_base_row[42usize]));
        let node_725 = (next_base_row[27usize]) + (node_545);
        let node_660 = (current_base_row[187usize])
            * ((next_base_row[25usize]) + (node_545));
        let node_726 = (next_base_row[28usize]) + (node_187);
        let node_700 = (current_base_row[188usize])
            * ((next_base_row[26usize]) + (node_545));
        let node_727 = (next_base_row[29usize]) + (node_191);
        let node_728 = (next_base_row[30usize]) + (node_195);
        let node_827 = (current_base_row[312usize])
            * ((next_base_row[22usize]) + (node_207));
        let node_729 = (next_base_row[31usize]) + (node_199);
        let node_834 = (next_base_row[22usize]) + (node_223);
        let node_854 = (next_base_row[32usize]) + (node_545);
        let node_829 = (current_base_row[270usize])
            * ((next_base_row[22usize]) + (node_211));
        let node_731 = (next_base_row[33usize]) + (node_207);
        let node_831 = (current_base_row[271usize])
            * ((next_base_row[22usize]) + (node_215));
        let node_732 = (next_base_row[34usize]) + (node_211);
        let node_833 = (current_base_row[314usize])
            * ((next_base_row[22usize]) + (node_219));
        let node_733 = (next_base_row[35usize]) + (node_215);
        let node_835 = (current_base_row[317usize]) * (node_834);
        let node_734 = (next_base_row[36usize]) + (node_219);
        let node_451 = ((((node_256) + (node_302)) + (node_345)) + (node_387))
            + (current_base_row[292usize]);
        let node_763 = ((((node_575) + (node_618)) + (node_660)) + (node_700))
            + (current_base_row[293usize]);
        let node_837 = (current_base_row[318usize])
            * ((next_base_row[22usize]) + (node_227));
        let node_735 = (next_base_row[37usize]) + (node_223);
        let node_412 = (next_base_row[22usize]) + (node_203);
        let node_839 = (current_base_row[273usize])
            * ((next_base_row[22usize]) + (node_231));
        let node_713 = (node_159)
            + (BFieldElement::from_raw_u64(18446744047939747846u64));
        let node_413 = (next_base_row[23usize]) + (node_207);
        let node_841 = (current_base_row[329usize])
            * ((next_base_row[22usize]) + (node_235));
        let node_724 = (next_ext_row[6usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[72usize]));
        let node_414 = (next_base_row[24usize]) + (node_211);
        let node_843 = (current_base_row[330usize])
            * ((next_base_row[22usize]) + (node_239));
        let node_415 = (next_base_row[25usize]) + (node_215);
        let node_845 = (current_base_row[331usize])
            * ((next_base_row[22usize]) + (node_243));
        let node_416 = (next_base_row[26usize]) + (node_219);
        let node_1435 = ((challenges[StandardOutputIndeterminate])
            * (((challenges[StandardOutputIndeterminate])
                * (((challenges[StandardOutputIndeterminate])
                    * (((challenges[StandardOutputIndeterminate])
                        * (current_ext_row[4usize])) + (current_base_row[22usize])))
                    + (current_base_row[23usize]))) + (current_base_row[24usize])))
            + (current_base_row[25usize]);
        let node_1430 = ((challenges[StandardOutputIndeterminate])
            * (((challenges[StandardOutputIndeterminate])
                * (((challenges[StandardOutputIndeterminate])
                    * (current_ext_row[4usize])) + (current_base_row[22usize])))
                + (current_base_row[23usize]))) + (current_base_row[24usize]);
        let node_1425 = ((challenges[StandardOutputIndeterminate])
            * (((challenges[StandardOutputIndeterminate]) * (current_ext_row[4usize]))
                + (current_base_row[22usize]))) + (current_base_row[23usize]);
        let node_1420 = ((challenges[StandardOutputIndeterminate])
            * (current_ext_row[4usize])) + (current_base_row[22usize]);
        let node_5017 = (next_ext_row[5usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[5usize]));
        let node_5106 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (((((((((((challenges[HashStateWeight0]) * (next_base_row[22usize]))
                + ((challenges[HashStateWeight1]) * (next_base_row[23usize])))
                + ((challenges[HashStateWeight2]) * (next_base_row[24usize])))
                + ((challenges[HashStateWeight3]) * (next_base_row[25usize])))
                + ((challenges[HashStateWeight4]) * (next_base_row[26usize])))
                + ((challenges[HashStateWeight5]) * (next_base_row[27usize])))
                + ((challenges[HashStateWeight6]) * (next_base_row[28usize])))
                + ((challenges[HashStateWeight7]) * (next_base_row[29usize])))
                + ((challenges[HashStateWeight8]) * (next_base_row[30usize])))
                + ((challenges[HashStateWeight9]) * (next_base_row[31usize])));
        let node_5089 = (((((challenges[HashStateWeight0]) * (next_base_row[22usize]))
            + ((challenges[HashStateWeight1]) * (next_base_row[23usize])))
            + ((challenges[HashStateWeight2]) * (next_base_row[24usize])))
            + ((challenges[HashStateWeight3]) * (next_base_row[25usize])))
            + ((challenges[HashStateWeight4]) * (next_base_row[26usize]));
        let node_5186 = (challenges[U32Indeterminate])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((challenges[U32LhsWeight]) * (next_base_row[22usize])));
        let node_5189 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * ((challenges[U32RhsWeight]) * (next_base_row[23usize]));
        let node_5200 = (node_5197)
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((challenges[U32RhsWeight]) * (current_base_row[23usize])));
        let node_5237 = ((node_5229)
            * (((node_5197) + (node_5193))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((challenges[U32ResultWeight]) * (next_base_row[22usize])))))
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_5223 = (((node_5186)
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((challenges[U32RhsWeight]) * (current_base_row[23usize]))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((challenges[U32CiWeight])
                    * (BFieldElement::from_raw_u64(25769803770u64)))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (challenges[U32ResultWeight]));
        let node_5227 = ((node_5197) + (node_5189))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((challenges[U32CiWeight])
                    * (BFieldElement::from_raw_u64(17179869180u64))));
        let node_5300 = (next_base_row[47usize])
            * ((next_base_row[47usize])
                + (BFieldElement::from_raw_u64(18446744065119617026u64)));
        let node_5371 = (challenges[RamTableBezoutRelationIndeterminate])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (next_base_row[52usize]));
        let node_5376 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_ext_row[16usize]);
        let node_5423 = ((next_base_row[51usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64)))
            * (next_base_row[51usize]);
        let node_5465 = (node_5461)
            * ((current_base_row[58usize])
                + (BFieldElement::from_raw_u64(18446744000695107601u64)));
        let node_5473 = (next_base_row[57usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[57usize]));
        let node_5464 = (current_base_row[58usize])
            + (BFieldElement::from_raw_u64(18446744000695107601u64));
        let node_5495 = (next_ext_row[23usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[23usize]));
        let node_6374 = (current_base_row[63usize])
            + (BFieldElement::from_raw_u64(18446743897615892521u64));
        let node_6376 = (current_base_row[64usize])
            + (BFieldElement::from_raw_u64(18446744047939747846u64));
        let node_6615 = (next_ext_row[24usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[24usize]));
        let node_5532 = (((((current_base_row[65usize])
            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
            + ((current_base_row[66usize])
                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
            + ((current_base_row[67usize])
                * (BFieldElement::from_raw_u64(281474976645120u64))))
            + (current_base_row[68usize])) * (BFieldElement::from_raw_u64(1u64));
        let node_5543 = (((((current_base_row[69usize])
            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
            + ((current_base_row[70usize])
                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
            + ((current_base_row[71usize])
                * (BFieldElement::from_raw_u64(281474976645120u64))))
            + (current_base_row[72usize])) * (BFieldElement::from_raw_u64(1u64));
        let node_5554 = (((((current_base_row[73usize])
            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
            + ((current_base_row[74usize])
                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
            + ((current_base_row[75usize])
                * (BFieldElement::from_raw_u64(281474976645120u64))))
            + (current_base_row[76usize])) * (BFieldElement::from_raw_u64(1u64));
        let node_5565 = (((((current_base_row[77usize])
            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
            + ((current_base_row[78usize])
                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
            + ((current_base_row[79usize])
                * (BFieldElement::from_raw_u64(281474976645120u64))))
            + (current_base_row[80usize])) * (BFieldElement::from_raw_u64(1u64));
        let node_6410 = (node_6376) * (node_6374);
        let node_6425 = ((current_base_row[62usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64)))
            * ((current_base_row[62usize])
                + (BFieldElement::from_raw_u64(18446744060824649731u64)));
        let node_6443 = (challenges[HashStateWeight10])
            * ((next_base_row[103usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (current_base_row[103usize])));
        let node_6444 = (challenges[HashStateWeight11])
            * ((next_base_row[104usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (current_base_row[104usize])));
        let node_6446 = (challenges[HashStateWeight12])
            * ((next_base_row[105usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (current_base_row[105usize])));
        let node_6448 = (challenges[HashStateWeight13])
            * ((next_base_row[106usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (current_base_row[106usize])));
        let node_6450 = (challenges[HashStateWeight14])
            * ((next_base_row[107usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (current_base_row[107usize])));
        let node_6452 = (challenges[HashStateWeight15])
            * ((next_base_row[108usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (current_base_row[108usize])));
        let node_6544 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (((((((((((challenges[HashStateWeight0]) * (node_6264))
                + ((challenges[HashStateWeight1]) * (node_6275)))
                + ((challenges[HashStateWeight2]) * (node_6286)))
                + ((challenges[HashStateWeight3]) * (node_6297)))
                + ((challenges[HashStateWeight4]) * (next_base_row[97usize])))
                + ((challenges[HashStateWeight5]) * (next_base_row[98usize])))
                + ((challenges[HashStateWeight6]) * (next_base_row[99usize])))
                + ((challenges[HashStateWeight7]) * (next_base_row[100usize])))
                + ((challenges[HashStateWeight8]) * (next_base_row[101usize])))
                + ((challenges[HashStateWeight9]) * (next_base_row[102usize])));
        let node_6521 = (next_ext_row[25usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[25usize]));
        let node_6530 = (((((challenges[HashStateWeight0]) * (node_6264))
            + ((challenges[HashStateWeight1]) * (node_6275)))
            + ((challenges[HashStateWeight2]) * (node_6286)))
            + ((challenges[HashStateWeight3]) * (node_6297)))
            + ((challenges[HashStateWeight4]) * (next_base_row[97usize]));
        let node_6555 = (next_ext_row[26usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[26usize]));
        let node_6581 = (next_ext_row[27usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[27usize]));
        let node_6584 = (next_base_row[63usize])
            + (BFieldElement::from_raw_u64(18446743863256154161u64));
        let node_6930 = (next_ext_row[44usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[44usize]));
        let node_6946 = (next_ext_row[45usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[45usize]));
        let node_6941 = ((challenges[LookupTableInputWeight])
            * (next_base_row[131usize]))
            + ((challenges[LookupTableOutputWeight]) * (next_base_row[133usize]));
        let node_6944 = ((challenges[LookupTableInputWeight])
            * (next_base_row[130usize]))
            + ((challenges[LookupTableOutputWeight]) * (next_base_row[132usize]));
        let node_6984 = (next_ext_row[46usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[46usize]));
        let node_7040 = ((next_base_row[140usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[140usize])))
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_7046 = (node_7025)
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_7067 = (next_base_row[147usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_7069 = (next_base_row[147usize])
            + (BFieldElement::from_raw_u64(18446744060824649731u64));
        let node_7072 = (current_base_row[309usize]) * (next_base_row[147usize]);
        let node_7074 = (current_base_row[147usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_7038 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[140usize]);
        let node_7133 = (next_base_row[147usize]) * (next_base_row[147usize]);
        let node_7083 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (node_7025);
        let node_7149 = (next_ext_row[48usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[48usize]));
        let node_246 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[42usize]));
        let node_248 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[41usize]));
        let node_1493 = (current_base_row[12usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_1474 = (current_base_row[13usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_134 = (current_base_row[40usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_6454 = (next_base_row[64usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_6456 = (next_base_row[64usize])
            + (BFieldElement::from_raw_u64(18446744060824649731u64));
        let node_6458 = (next_base_row[64usize])
            + (BFieldElement::from_raw_u64(18446744056529682436u64));
        let node_7051 = (next_base_row[142usize])
            + (BFieldElement::from_raw_u64(18446744052234715141u64));
        let node_7054 = (next_base_row[142usize])
            + (BFieldElement::from_raw_u64(18446744009285042191u64));
        let node_6460 = (next_base_row[64usize])
            + (BFieldElement::from_raw_u64(18446744052234715141u64));
        let node_6392 = (current_base_row[62usize])
            + (BFieldElement::from_raw_u64(18446744060824649731u64));
        let node_6417 = (current_base_row[62usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_178 = (challenges[OpStackIndeterminate])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (((node_169)
                    + ((challenges[OpStackPointerWeight]) * (next_base_row[38usize])))
                    + ((challenges[OpStackFirstUnderflowElementWeight])
                        * (next_base_row[37usize]))));
        let node_568 = (challenges[OpStackIndeterminate])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (((node_169)
                    + ((challenges[OpStackPointerWeight]) * (current_base_row[38usize])))
                    + ((challenges[OpStackFirstUnderflowElementWeight])
                        * (current_base_row[37usize]))));
        let node_1006 = (challenges[RamIndeterminate])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (((node_997)
                    + (((next_base_row[22usize])
                        + (BFieldElement::from_raw_u64(4294967295u64)))
                        * (challenges[RamPointerWeight])))
                    + ((next_base_row[23usize]) * (challenges[RamValueWeight]))));
        let node_1088 = (challenges[RamIndeterminate])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (((node_994)
                    + ((current_base_row[22usize]) * (challenges[RamPointerWeight])))
                    + ((current_base_row[23usize]) * (challenges[RamValueWeight]))));
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
            ((current_base_row[5usize]) * (node_34)) * (node_48),
            ((next_base_row[7usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (current_base_row[7usize])))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            (current_base_row[8usize])
                * ((next_base_row[8usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[8usize]))),
            ((((((((((((((((((((((((((((((((((((((((current_base_row[202usize])
                * (node_121))
                + ((current_base_row[212usize])
                    * ((next_base_row[22usize]) + (node_542))))
                + ((current_base_row[203usize]) * (node_121)))
                + ((current_base_row[208usize])
                    * ((current_base_row[315usize]) * (node_824))))
                + ((current_base_row[205usize]) * (current_base_row[315usize])))
                + ((current_base_row[214usize]) * (node_931)))
                + ((current_base_row[216usize]) * (node_121)))
                + ((current_base_row[213usize])
                    * ((node_934) * (current_base_row[39usize]))))
                + ((current_base_row[220usize])
                    * ((node_121)
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)))))
                + ((current_base_row[225usize])
                    * ((next_base_row[19usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((current_base_row[19usize])
                                + (BFieldElement::from_raw_u64(
                                    18446744065119617026u64,
                                )))))))
                + ((current_base_row[223usize])
                    * ((next_base_row[9usize]) + (node_128))))
                + ((current_base_row[224usize])
                    * ((current_base_row[22usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)))))
                + ((current_base_row[215usize]) * (node_121)))
                + ((current_base_row[221usize]) * (node_121)))
                + ((current_base_row[240usize]) * (node_121)))
                + ((current_base_row[232usize]) * (node_132)))
                + ((current_base_row[228usize])
                    * ((current_base_row[27usize]) + (node_545))))
                + ((current_base_row[229usize]) * (node_121)))
                + ((current_base_row[244usize]) * (node_121)))
                + ((current_base_row[246usize]) * (node_121)))
                + ((current_base_row[233usize]) * ((node_824) + (node_187))))
                + ((current_base_row[234usize]) * (node_1289)))
                + ((current_base_row[235usize])
                    * ((current_base_row[336usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)))))
                + ((current_base_row[237usize])
                    * ((current_base_row[39usize]) * (node_1295))))
                + ((current_base_row[236usize])
                    * ((current_base_row[22usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((BFieldElement::from_raw_u64(18446744069414584320u64))
                                * (next_base_row[23usize])) + (next_base_row[22usize]))))))
                + ((current_base_row[238usize]) * (node_121)))
                + ((current_base_row[239usize]) * (node_121)))
                + ((current_base_row[241usize]) * (node_121)))
                + ((current_base_row[242usize]) * (node_121)))
                + ((current_base_row[245usize]) * (node_121)))
                + ((current_base_row[248usize])
                    * (((current_base_row[22usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_1311))) + (node_1314))))
                + ((current_base_row[251usize]) * (node_121)))
                + ((current_base_row[274usize]) * ((node_824) + (node_195))))
                + ((current_base_row[275usize])
                    * ((next_base_row[22usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((node_1319)
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (node_1320)))
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (node_1323)))))))
                + ((current_base_row[277usize])
                    * ((((current_base_row[336usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_1346)))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_1349)))
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)))))
                + ((current_base_row[279usize]) * (node_1289)))
                + ((current_base_row[259usize]) * (node_121)))
                + ((current_base_row[263usize]) * (node_121))) * (node_4849))
                + ((node_114) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((current_base_row[202usize])
                * (node_125)) + ((current_base_row[212usize]) * (node_546)))
                + ((current_base_row[203usize]) * (node_125)))
                + ((current_base_row[208usize]) * (node_256)))
                + ((current_base_row[205usize]) * (node_575)))
                + ((current_base_row[214usize]) * (node_121)))
                + ((current_base_row[216usize]) * (node_125)))
                + ((current_base_row[213usize])
                    * ((node_934) * (current_base_row[22usize]))))
                + ((current_base_row[220usize])
                    * (((next_base_row[20usize]) + (node_113))
                        + (BFieldElement::from_raw_u64(18446744060824649731u64)))))
                + ((current_base_row[225usize])
                    * ((next_base_row[9usize]) + (node_124))))
                + ((current_base_row[223usize]) * (node_121)))
                + ((current_base_row[224usize]) * (node_121)))
                + ((current_base_row[215usize]) * (node_125)))
                + ((current_base_row[221usize]) * (node_125)))
                + ((current_base_row[240usize]) * (node_125)))
                + ((current_base_row[232usize])
                    * ((((next_base_row[32usize])
                        * (BFieldElement::from_raw_u64(8589934590u64)))
                        + (current_base_row[39usize])) + (node_203))))
                + ((current_base_row[228usize])
                    * ((current_base_row[28usize]) + (node_187))))
                + ((current_base_row[229usize]) * (node_125)))
                + ((current_base_row[244usize]) * (node_125)))
                + ((current_base_row[246usize]) * (node_125)))
                + ((current_base_row[233usize]) * (node_121)))
                + ((current_base_row[234usize]) * (node_121)))
                + ((current_base_row[235usize]) * (node_121)))
                + ((current_base_row[237usize]) * ((node_1292) * (node_1295))))
                + (current_base_row[340usize]))
                + ((current_base_row[238usize]) * (node_125)))
                + ((current_base_row[239usize]) * (node_125)))
                + ((current_base_row[241usize]) * (node_125)))
                + ((current_base_row[242usize]) * (node_125)))
                + ((current_base_row[245usize]) * (node_125)))
                + ((current_base_row[248usize]) * (node_872)))
                + ((current_base_row[251usize]) * (node_125)))
                + ((current_base_row[274usize]) * ((node_868) + (node_199))))
                + ((current_base_row[275usize])
                    * ((next_base_row[23usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((((((current_base_row[23usize])
                                * (current_base_row[25usize]))
                                + ((current_base_row[22usize])
                                    * (current_base_row[26usize])))
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (node_1331))) + (node_1320)) + (node_1323))))))
                + ((current_base_row[277usize])
                    * ((((((current_base_row[23usize]) * (next_base_row[22usize]))
                        + ((current_base_row[22usize]) * (next_base_row[23usize])))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_1356))) + (node_1346)) + (node_1349))))
                + ((current_base_row[279usize])
                    * ((next_base_row[23usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((current_base_row[22usize])
                                * (current_base_row[24usize]))))))
                + ((current_base_row[259usize]) * (node_125)))
                + ((current_base_row[263usize]) * (node_125))) * (node_4849))
                + ((node_931) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((current_base_row[202usize])
                * (node_129)) + ((current_base_row[212usize]) * (node_547)))
                + ((current_base_row[203usize]) * (node_129)))
                + ((current_base_row[208usize]) * (node_302)))
                + ((current_base_row[205usize]) * (node_618)))
                + ((current_base_row[214usize]) * (node_125)))
                + ((current_base_row[216usize]) * (node_129)))
                + ((current_base_row[213usize])
                    * ((((((current_base_row[11usize]) + (node_250))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((BFieldElement::from_raw_u64(8589934590u64))
                                * (current_base_row[41usize])))) + (node_145))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((BFieldElement::from_raw_u64(137438953440u64))
                                * (current_base_row[43usize]))))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((BFieldElement::from_raw_u64(549755813760u64))
                                * (current_base_row[44usize]))))))
                + ((current_base_row[220usize])
                    * ((next_base_row[21usize]) + (node_542))))
                + ((current_base_row[225usize]) * (node_824)))
                + ((current_base_row[223usize]) * (node_125)))
                + ((current_base_row[224usize]) * (node_125)))
                + ((current_base_row[215usize]) * (node_129)))
                + ((current_base_row[221usize]) * (node_129)))
                + ((current_base_row[240usize]) * (node_129)))
                + ((current_base_row[232usize])
                    * (((node_298) * (node_824))
                        + ((current_base_row[39usize]) * (node_725)))))
                + ((current_base_row[228usize])
                    * ((current_base_row[29usize]) + (node_191))))
                + ((current_base_row[229usize]) * (node_129)))
                + ((current_base_row[244usize]) * (node_129)))
                + ((current_base_row[246usize]) * (node_129)))
                + ((current_base_row[233usize]) * (node_125)))
                + ((current_base_row[234usize]) * (node_125)))
                + ((current_base_row[235usize]) * (node_125)))
                + ((current_base_row[237usize])
                    * ((next_base_row[22usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_1295)))))
                + ((current_base_row[236usize]) * (node_547)))
                + ((current_base_row[238usize]) * (node_129)))
                + ((current_base_row[239usize]) * (node_129)))
                + ((current_base_row[241usize]) * (node_129)))
                + ((current_base_row[242usize]) * (node_129)))
                + ((current_base_row[245usize]) * (node_129)))
                + ((current_base_row[248usize]) * (node_121)))
                + ((current_base_row[251usize]) * (node_129)))
                + ((current_base_row[274usize]) * ((node_872) + (node_203))))
                + ((current_base_row[275usize])
                    * ((next_base_row[24usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((current_base_row[24usize])
                                * (current_base_row[25usize]))
                                + ((current_base_row[23usize])
                                    * (current_base_row[26usize])))
                                + ((current_base_row[22usize])
                                    * (current_base_row[27usize]))) + (node_1331))))))
                + ((current_base_row[277usize])
                    * (((((current_base_row[24usize]) * (next_base_row[22usize]))
                        + (node_1311))
                        + ((current_base_row[22usize]) * (next_base_row[24usize])))
                        + (node_1356))))
                + ((current_base_row[279usize])
                    * ((next_base_row[24usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_1319)))))
                + ((current_base_row[259usize]) * (node_129)))
                + ((current_base_row[263usize]) * (node_129))) * (node_4849))
                + (((next_base_row[11usize]) + (node_542)) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((current_base_row[202usize])
                * (node_117)) + ((current_base_row[212usize]) * (node_548)))
                + ((current_base_row[203usize]) * (node_117)))
                + ((current_base_row[208usize]) * (node_345)))
                + ((current_base_row[205usize]) * (node_660)))
                + ((current_base_row[214usize]) * (node_129)))
                + ((current_base_row[216usize]) * (node_932)))
                + (current_base_row[341usize]))
                + ((current_base_row[220usize])
                    * ((next_base_row[9usize]) + (node_542))))
                + ((current_base_row[225usize]) * (node_868)))
                + ((current_base_row[223usize]) * (node_129)))
                + ((current_base_row[224usize]) * (node_129)))
                + ((current_base_row[215usize]) * (node_117)))
                + ((current_base_row[221usize]) * (node_117)))
                + ((current_base_row[240usize]) * (node_932)))
                + ((current_base_row[232usize])
                    * (((node_298) * (node_868))
                        + ((current_base_row[39usize]) * (node_726)))))
                + ((current_base_row[228usize])
                    * ((current_base_row[30usize]) + (node_195))))
                + ((current_base_row[229usize]) * (node_932)))
                + ((current_base_row[244usize]) * (node_932)))
                + ((current_base_row[246usize]) * (node_932)))
                + ((current_base_row[233usize]) * (node_129)))
                + ((current_base_row[234usize]) * (node_129)))
                + ((current_base_row[235usize]) * (node_129)))
                + ((current_base_row[237usize]) * (node_121)))
                + ((current_base_row[236usize]) * (node_548)))
                + ((current_base_row[238usize]) * (node_932)))
                + ((current_base_row[239usize]) * (node_932)))
                + ((current_base_row[241usize]) * (node_932)))
                + ((current_base_row[242usize]) * (node_932)))
                + ((current_base_row[245usize]) * (node_932)))
                + ((current_base_row[248usize]) * (node_125)))
                + ((current_base_row[251usize]) * (node_932)))
                + ((current_base_row[274usize]) * (node_332)))
                + ((current_base_row[275usize]) * (node_332)))
                + ((current_base_row[277usize]) * (node_876)))
                + ((current_base_row[279usize]) * (node_200)))
                + ((current_base_row[259usize]) * (node_117)))
                + ((current_base_row[263usize]) * (node_117))) * (node_4849))
                + ((node_121) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((current_base_row[202usize])
                * (node_132)) + ((current_base_row[212usize]) * (node_549)))
                + ((current_base_row[203usize]) * (node_132)))
                + ((current_base_row[208usize]) * (node_387)))
                + ((current_base_row[205usize]) * (node_700)))
                + ((current_base_row[214usize]) * (node_932)))
                + ((current_base_row[216usize]) * (node_824)))
                + ((current_base_row[213usize]) * (current_base_row[268usize])))
                + ((current_base_row[220usize]) * (node_824)))
                + ((current_base_row[225usize]) * (node_872)))
                + ((current_base_row[223usize]) * (node_824)))
                + ((current_base_row[224usize]) * (node_932)))
                + ((current_base_row[215usize]) * (node_132)))
                + ((current_base_row[221usize]) * (node_132)))
                + ((current_base_row[240usize]) * (node_417)))
                + ((current_base_row[232usize])
                    * (((node_298) * (node_872))
                        + ((current_base_row[39usize]) * (node_727)))))
                + ((current_base_row[228usize])
                    * ((current_base_row[31usize]) + (node_199))))
                + ((current_base_row[229usize]) * (node_824)))
                + ((current_base_row[244usize])
                    * ((node_159) + (BFieldElement::from_raw_u64(42949672950u64)))))
                + ((current_base_row[246usize])
                    * ((node_159)
                        + (BFieldElement::from_raw_u64(18446744026464911371u64)))))
                + ((current_base_row[233usize]) * (node_932)))
                + ((current_base_row[234usize]) * (node_932)))
                + ((current_base_row[235usize]) * (node_932)))
                + ((current_base_row[237usize]) * (node_125)))
                + ((current_base_row[236usize]) * (node_549)))
                + ((current_base_row[238usize]) * (node_192)))
                + ((current_base_row[239usize]) * (node_192)))
                + ((current_base_row[241usize]) * (node_192)))
                + ((current_base_row[242usize]) * (node_868)))
                + ((current_base_row[245usize]) * (node_192)))
                + ((current_base_row[248usize]) * (node_129)))
                + ((current_base_row[251usize]) * (node_868)))
                + ((current_base_row[274usize]) * (node_333)))
                + ((current_base_row[275usize]) * (node_333)))
                + ((current_base_row[277usize]) * (node_880)))
                + ((current_base_row[279usize]) * (node_204)))
                + ((current_base_row[259usize]) * (node_132)))
                + ((current_base_row[263usize]) * (node_132))) * (node_4849))
                + ((node_125) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((current_base_row[202usize])
                * (current_base_row[260usize]))
                + ((current_base_row[212usize]) * (node_551)))
                + ((current_base_row[203usize]) * (current_base_row[260usize])))
                + ((current_base_row[208usize]) * (node_827)))
                + ((current_base_row[205usize])
                    * ((current_base_row[312usize])
                        * ((next_base_row[28usize]) + (node_545)))))
                + ((current_base_row[214usize]) * (node_868)))
                + ((current_base_row[216usize]) * (node_872)))
                + (current_base_row[343usize]))
                + ((current_base_row[220usize]) * (node_872)))
                + ((current_base_row[225usize]) * (node_880)))
                + ((current_base_row[223usize]) * (node_872)))
                + ((current_base_row[224usize]) * (node_192)))
                + ((current_base_row[215usize]) * (current_base_row[260usize])))
                + ((current_base_row[221usize]) * (current_base_row[260usize])))
                + ((current_base_row[240usize]) * (node_419)))
                + ((current_base_row[232usize])
                    * (((node_298) * (node_880))
                        + ((current_base_row[39usize]) * (node_729)))))
                + ((current_base_row[228usize]) * (node_125)))
                + ((current_base_row[229usize]) * (node_872)))
                + ((current_base_row[244usize]) * (node_834)))
                + ((current_base_row[246usize]) * (node_854)))
                + ((current_base_row[233usize]) * (node_196)))
                + ((current_base_row[234usize]) * (node_196)))
                + ((current_base_row[235usize]) * (node_872)))
                + ((current_base_row[237usize]) * (node_932)))
                + ((current_base_row[236usize]) * (node_551)))
                + ((current_base_row[238usize]) * (node_200)))
                + ((current_base_row[239usize]) * (node_200)))
                + ((current_base_row[241usize]) * (node_200)))
                + ((current_base_row[242usize]) * (node_876)))
                + ((current_base_row[245usize]) * (node_200)))
                + ((current_base_row[248usize]) * (node_876)))
                + ((current_base_row[251usize]) * (node_876)))
                + ((current_base_row[274usize]) * (node_335)))
                + ((current_base_row[275usize]) * (node_335)))
                + ((current_base_row[277usize]) * (node_888)))
                + ((current_base_row[279usize]) * (node_212)))
                + ((current_base_row[259usize]) * (current_base_row[260usize])))
                + ((current_base_row[263usize]) * (current_base_row[260usize])))
                * (node_4849)) + ((node_824) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((current_base_row[202usize])
                * (current_base_row[261usize]))
                + ((current_base_row[212usize]) * (node_552)))
                + ((current_base_row[203usize]) * (current_base_row[261usize])))
                + ((current_base_row[208usize]) * (node_829)))
                + ((current_base_row[205usize])
                    * ((current_base_row[270usize])
                        * ((next_base_row[29usize]) + (node_545)))))
                + ((current_base_row[214usize]) * (node_872)))
                + ((current_base_row[216usize]) * (node_876)))
                + ((current_base_row[213usize]) * (current_base_row[280usize])))
                + ((current_base_row[220usize]) * (node_876)))
                + ((current_base_row[225usize]) * (node_884)))
                + ((current_base_row[223usize]) * (node_876)))
                + ((current_base_row[224usize]) * (node_196)))
                + ((current_base_row[215usize]) * (current_base_row[261usize])))
                + ((current_base_row[221usize]) * (current_base_row[261usize])))
                + ((current_base_row[240usize]) * (node_420)))
                + ((current_base_row[232usize]) * (node_731)))
                + ((current_base_row[228usize]) * (node_129)))
                + ((current_base_row[229usize]) * (node_876)))
                + ((current_base_row[244usize])
                    * ((next_base_row[23usize]) + (node_227))))
                + ((current_base_row[246usize])
                    * ((next_base_row[33usize]) + (node_187))))
                + ((current_base_row[233usize]) * (node_200)))
                + ((current_base_row[234usize]) * (node_200)))
                + ((current_base_row[235usize]) * (node_876)))
                + ((current_base_row[237usize]) * (node_192)))
                + ((current_base_row[236usize]) * (node_552)))
                + ((current_base_row[238usize]) * (node_204)))
                + ((current_base_row[239usize]) * (node_204)))
                + ((current_base_row[241usize]) * (node_204)))
                + ((current_base_row[242usize]) * (node_880)))
                + ((current_base_row[245usize]) * (node_204)))
                + ((current_base_row[248usize]) * (node_880)))
                + ((current_base_row[251usize]) * (node_880)))
                + ((current_base_row[274usize]) * (node_336)))
                + ((current_base_row[275usize]) * (node_336)))
                + ((current_base_row[277usize]) * (node_892)))
                + ((current_base_row[279usize]) * (node_216)))
                + ((current_base_row[259usize]) * (current_base_row[261usize])))
                + ((current_base_row[263usize]) * (current_base_row[261usize])))
                * (node_4849)) + ((node_868) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((current_base_row[202usize])
                * (node_155)) + ((current_base_row[212usize]) * (node_553)))
                + ((current_base_row[203usize]) * (node_155)))
                + ((current_base_row[208usize]) * (node_831)))
                + ((current_base_row[205usize])
                    * ((current_base_row[271usize])
                        * ((next_base_row[30usize]) + (node_545)))))
                + ((current_base_row[214usize]) * (node_876)))
                + ((current_base_row[216usize]) * (node_880)))
                + ((current_base_row[213usize]) * (current_base_row[281usize])))
                + ((current_base_row[220usize]) * (node_880)))
                + ((current_base_row[225usize]) * (node_888)))
                + ((current_base_row[223usize]) * (node_880)))
                + ((current_base_row[224usize]) * (node_200)))
                + ((current_base_row[215usize]) * (node_155)))
                + ((current_base_row[221usize]) * (node_155)))
                + ((current_base_row[240usize]) * (node_421)))
                + ((current_base_row[232usize]) * (node_732)))
                + ((current_base_row[228usize]) * (node_932)))
                + ((current_base_row[229usize]) * (node_880)))
                + ((current_base_row[244usize])
                    * ((next_base_row[24usize]) + (node_231))))
                + ((current_base_row[246usize])
                    * ((next_base_row[34usize]) + (node_191))))
                + ((current_base_row[233usize]) * (node_204)))
                + ((current_base_row[234usize]) * (node_204)))
                + ((current_base_row[235usize]) * (node_880)))
                + ((current_base_row[237usize]) * (node_196)))
                + ((current_base_row[236usize]) * (node_553)))
                + ((current_base_row[238usize]) * (node_208)))
                + ((current_base_row[239usize]) * (node_208)))
                + ((current_base_row[241usize]) * (node_208)))
                + ((current_base_row[242usize]) * (node_884)))
                + ((current_base_row[245usize]) * (node_208)))
                + ((current_base_row[248usize]) * (node_884)))
                + ((current_base_row[251usize]) * (node_884)))
                + ((current_base_row[274usize]) * (node_337)))
                + ((current_base_row[275usize]) * (node_337)))
                + ((current_base_row[277usize]) * (node_896)))
                + ((current_base_row[279usize]) * (node_220)))
                + ((current_base_row[259usize]) * (node_155)))
                + ((current_base_row[263usize]) * (node_155))) * (node_4849))
                + ((node_872) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((current_base_row[202usize])
                * (node_441)) + ((current_base_row[212usize]) * (node_554)))
                + ((current_base_row[203usize]) * (node_753)))
                + ((current_base_row[208usize]) * (node_833)))
                + ((current_base_row[205usize])
                    * ((current_base_row[314usize])
                        * ((next_base_row[31usize]) + (node_545)))))
                + ((current_base_row[214usize]) * (node_880)))
                + ((current_base_row[216usize]) * (node_884)))
                + ((current_base_row[213usize]) * (node_121)))
                + ((current_base_row[220usize]) * (node_884)))
                + ((current_base_row[225usize]) * (node_892)))
                + ((current_base_row[223usize]) * (node_884)))
                + ((current_base_row[224usize]) * (node_204)))
                + ((current_base_row[215usize]) * (node_753)))
                + ((current_base_row[221usize]) * (node_441)))
                + ((current_base_row[240usize]) * (node_422)))
                + ((current_base_row[232usize]) * (node_733)))
                + ((current_base_row[228usize]) * (node_400)))
                + ((current_base_row[229usize]) * (node_884)))
                + ((current_base_row[244usize])
                    * ((next_base_row[25usize]) + (node_235))))
                + ((current_base_row[246usize])
                    * ((next_base_row[35usize]) + (node_195))))
                + ((current_base_row[233usize]) * (node_208)))
                + ((current_base_row[234usize]) * (node_208)))
                + ((current_base_row[235usize]) * (node_884)))
                + ((current_base_row[237usize]) * (node_200)))
                + ((current_base_row[236usize]) * (node_554)))
                + ((current_base_row[238usize]) * (node_212)))
                + ((current_base_row[239usize]) * (node_212)))
                + ((current_base_row[241usize]) * (node_212)))
                + ((current_base_row[242usize]) * (node_888)))
                + ((current_base_row[245usize]) * (node_212)))
                + ((current_base_row[248usize]) * (node_888)))
                + ((current_base_row[251usize]) * (node_888)))
                + ((current_base_row[274usize]) * (node_338)))
                + ((current_base_row[275usize]) * (node_338)))
                + ((current_base_row[277usize]) * (node_900)))
                + ((current_base_row[279usize]) * (node_224)))
                + ((current_base_row[259usize]) * (node_753)))
                + ((current_base_row[263usize]) * (node_441))) * (node_4849))
                + ((node_876) * (next_base_row[8usize])),
            (((((((((((((((((((((((((((((((((((((current_base_row[202usize])
                * (node_471)) + ((current_base_row[212usize]) * (node_560)))
                + ((current_base_row[203usize]) * (node_783)))
                + ((current_base_row[208usize]) * (node_845)))
                + ((current_base_row[205usize])
                    * ((current_base_row[331usize])
                        * ((next_base_row[37usize]) + (node_545)))))
                + ((current_base_row[214usize]) * (node_904)))
                + ((current_base_row[216usize]) * (node_908)))
                + ((current_base_row[213usize]) * (node_200)))
                + ((current_base_row[220usize]) * (node_908)))
                + ((current_base_row[225usize]) * (node_916)))
                + ((current_base_row[223usize]) * (node_908)))
                + ((current_base_row[224usize]) * (node_228)))
                + ((current_base_row[215usize]) * (node_778)))
                + ((current_base_row[221usize]) * (node_466)))
                + ((current_base_row[232usize]) * (node_125)))
                + ((current_base_row[228usize]) * (node_416)))
                + ((current_base_row[229usize]) * (node_908)))
                + ((current_base_row[233usize]) * (node_232)))
                + ((current_base_row[234usize]) * (node_232)))
                + ((current_base_row[235usize]) * (node_908)))
                + ((current_base_row[237usize]) * (node_224)))
                + ((current_base_row[236usize]) * (node_560)))
                + ((current_base_row[238usize]) * (node_236)))
                + ((current_base_row[239usize]) * (node_236)))
                + ((current_base_row[241usize]) * (node_236)))
                + ((current_base_row[242usize]) * (node_912)))
                + ((current_base_row[245usize]) * (node_236)))
                + ((current_base_row[248usize]) * (node_912)))
                + ((current_base_row[251usize]) * (node_912)))
                + ((current_base_row[274usize]) * (node_121)))
                + ((current_base_row[275usize]) * (node_121)))
                + ((current_base_row[277usize]) * (node_924)))
                + ((current_base_row[279usize]) * (node_160)))
                + ((current_base_row[259usize]) * (node_783)))
                + ((current_base_row[263usize]) * (node_471))) * (node_4849))
                + ((node_900) * (next_base_row[8usize])),
            ((((((((((current_base_row[202usize]) * (current_base_row[312usize]))
                + ((current_base_row[203usize]) * (current_base_row[312usize])))
                + ((current_base_row[208usize]) * (node_548)))
                + ((current_base_row[205usize]) * (node_839)))
                + ((current_base_row[213usize]) * (node_160)))
                + ((current_base_row[215usize]) * (current_base_row[315usize])))
                + ((current_base_row[221usize]) * (current_base_row[315usize])))
                + ((current_base_row[259usize]) * (current_base_row[315usize])))
                + ((current_base_row[263usize]) * (current_base_row[315usize])))
                * (node_4849),
            (((((((((current_base_row[202usize]) * (current_base_row[318usize]))
                + ((current_base_row[203usize]) * (current_base_row[318usize])))
                + ((current_base_row[208usize]) * (node_553)))
                + ((current_base_row[205usize])
                    * (((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[186usize]))) * (node_872))))
                + ((current_base_row[215usize]) * (current_base_row[317usize])))
                + ((current_base_row[221usize]) * (current_base_row[317usize])))
                + ((current_base_row[259usize]) * (current_base_row[317usize])))
                + ((current_base_row[263usize]) * (current_base_row[317usize])))
                * (node_4849),
            (((((((((current_base_row[202usize]) * (current_base_row[273usize]))
                + ((current_base_row[203usize]) * (current_base_row[273usize])))
                + ((current_base_row[208usize]) * (node_554)))
                + ((current_base_row[205usize])
                    * (((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[187usize]))) * (node_876))))
                + ((current_base_row[215usize]) * (current_base_row[318usize])))
                + ((current_base_row[221usize]) * (current_base_row[318usize])))
                + ((current_base_row[259usize]) * (current_base_row[318usize])))
                + ((current_base_row[263usize]) * (current_base_row[318usize])))
                * (node_4849),
            (((((((((current_base_row[202usize]) * (current_base_row[329usize]))
                + ((current_base_row[203usize]) * (current_base_row[329usize])))
                + ((current_base_row[208usize]) * (node_555)))
                + ((current_base_row[205usize])
                    * (((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[188usize]))) * (node_880))))
                + ((current_base_row[215usize]) * (current_base_row[273usize])))
                + ((current_base_row[221usize]) * (current_base_row[273usize])))
                + ((current_base_row[259usize]) * (current_base_row[273usize])))
                + ((current_base_row[263usize]) * (current_base_row[273usize])))
                * (node_4849),
            (((((((((current_base_row[202usize]) * (current_base_row[330usize]))
                + ((current_base_row[203usize]) * (current_base_row[330usize])))
                + ((current_base_row[208usize]) * (node_556)))
                + ((current_base_row[205usize])
                    * (((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[189usize]))) * (node_884))))
                + ((current_base_row[215usize]) * (current_base_row[329usize])))
                + ((current_base_row[221usize]) * (current_base_row[329usize])))
                + ((current_base_row[259usize]) * (current_base_row[329usize])))
                + ((current_base_row[263usize]) * (current_base_row[329usize])))
                * (node_4849),
            (((((((((current_base_row[202usize]) * (current_base_row[331usize]))
                + ((current_base_row[203usize]) * (current_base_row[331usize])))
                + ((current_base_row[208usize]) * (node_557)))
                + ((current_base_row[205usize])
                    * (((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[312usize]))) * (node_888))))
                + ((current_base_row[215usize]) * (current_base_row[330usize])))
                + ((current_base_row[221usize]) * (current_base_row[330usize])))
                + ((current_base_row[259usize]) * (current_base_row[330usize])))
                + ((current_base_row[263usize]) * (current_base_row[330usize])))
                * (node_4849),
            (((current_base_row[208usize]) * (node_561))
                + ((current_base_row[205usize])
                    * (((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[317usize]))) * (node_904))))
                * (node_4849),
            ((current_base_row[205usize])
                * (((BFieldElement::from_raw_u64(4294967295u64))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[331usize]))) * (node_924))) * (node_4849),
            ((current_base_row[205usize]) * (node_159)) * (node_4849),
            ((current_base_row[205usize]) * (node_132)) * (node_4849),
            ((current_base_row[205usize]) * (current_base_row[268usize])) * (node_4849),
            ((current_base_row[205usize]) * (current_base_row[260usize])) * (node_4849),
            ((current_base_row[205usize]) * (current_base_row[261usize])) * (node_4849),
            ((current_base_row[205usize]) * (node_155)) * (node_4849),
            ((current_base_row[205usize]) * (node_121)) * (node_4849),
            ((current_base_row[205usize]) * (node_125)) * (node_4849),
            ((current_base_row[205usize]) * (node_129)) * (node_4849),
            ((current_base_row[205usize]) * (node_117)) * (node_4849),
            (node_5277) * (node_5276),
            ((node_5277)
                * ((next_base_row[49usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[49usize])))) * (next_base_row[47usize]),
            ((current_base_row[47usize])
                * ((current_base_row[47usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                * (node_5283),
            (((current_base_row[51usize])
                + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                * (current_base_row[51usize])) * (node_5346),
            (current_base_row[54usize]) * (node_5355),
            (node_5352) * (node_5355),
            ((node_5355)
                * ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (next_base_row[51usize])))
                * ((next_base_row[53usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[53usize]))),
            (node_5355)
                * ((next_base_row[55usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[55usize]))),
            (node_5355)
                * ((next_base_row[56usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[56usize]))),
            (node_5461) * (node_5460),
            (node_5465)
                * ((next_base_row[60usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[60usize]))),
            (node_5465)
                * ((next_base_row[61usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[61usize]))),
            (((node_5461)
                * ((node_5473) + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                * ((current_base_row[58usize])
                    + (BFieldElement::from_raw_u64(18446743927680663586u64))))
                * (node_5464),
            ((current_base_row[345usize])
                * ((current_base_row[64usize])
                    + (BFieldElement::from_raw_u64(18446744052234715141u64))))
                * (next_base_row[64usize]),
            (((next_base_row[62usize]) * (node_6374)) * (node_6376))
                * (((next_base_row[64usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[64usize])))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (current_base_row[350usize]) * (node_6408),
            (node_6410)
                * ((next_base_row[63usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[63usize]))),
            (node_6410)
                * ((next_base_row[62usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[62usize]))),
            ((current_base_row[346usize]) * (node_6404)) * (next_base_row[62usize]),
            (current_base_row[351usize]) * (next_base_row[62usize]),
            ((node_6425) * (node_6394)) * (next_base_row[62usize]),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(263719581847590u64))
                    * (node_5588))
                    + ((BFieldElement::from_raw_u64(76643691379275u64)) * (node_5599)))
                    + ((BFieldElement::from_raw_u64(115096533571410u64)) * (node_5610)))
                    + ((BFieldElement::from_raw_u64(256362302871255u64)) * (node_5621)))
                    + ((BFieldElement::from_raw_u64(51629801853195u64))
                        * (current_base_row[297usize])))
                    + ((BFieldElement::from_raw_u64(175668457332795u64))
                        * (current_base_row[298usize])))
                    + ((BFieldElement::from_raw_u64(177601192615545u64))
                        * (current_base_row[299usize])))
                    + ((BFieldElement::from_raw_u64(118201794925695u64))
                        * (current_base_row[300usize])))
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
                        * (current_base_row[308usize]))) + (current_base_row[113usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (node_6264))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(4758823762860u64))
                    * (node_5588))
                    + ((BFieldElement::from_raw_u64(263719581847590u64)) * (node_5599)))
                    + ((BFieldElement::from_raw_u64(76643691379275u64)) * (node_5610)))
                    + ((BFieldElement::from_raw_u64(115096533571410u64)) * (node_5621)))
                    + ((BFieldElement::from_raw_u64(256362302871255u64))
                        * (current_base_row[297usize])))
                    + ((BFieldElement::from_raw_u64(51629801853195u64))
                        * (current_base_row[298usize])))
                    + ((BFieldElement::from_raw_u64(175668457332795u64))
                        * (current_base_row[299usize])))
                    + ((BFieldElement::from_raw_u64(177601192615545u64))
                        * (current_base_row[300usize])))
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
                        * (current_base_row[308usize]))) + (current_base_row[114usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (node_6275))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(123480309731250u64))
                    * (node_5588))
                    + ((BFieldElement::from_raw_u64(4758823762860u64)) * (node_5599)))
                    + ((BFieldElement::from_raw_u64(263719581847590u64)) * (node_5610)))
                    + ((BFieldElement::from_raw_u64(76643691379275u64)) * (node_5621)))
                    + ((BFieldElement::from_raw_u64(115096533571410u64))
                        * (current_base_row[297usize])))
                    + ((BFieldElement::from_raw_u64(256362302871255u64))
                        * (current_base_row[298usize])))
                    + ((BFieldElement::from_raw_u64(51629801853195u64))
                        * (current_base_row[299usize])))
                    + ((BFieldElement::from_raw_u64(175668457332795u64))
                        * (current_base_row[300usize])))
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
                        * (current_base_row[308usize]))) + (current_base_row[115usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (node_6286))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(145268678818785u64))
                    * (node_5588))
                    + ((BFieldElement::from_raw_u64(123480309731250u64)) * (node_5599)))
                    + ((BFieldElement::from_raw_u64(4758823762860u64)) * (node_5610)))
                    + ((BFieldElement::from_raw_u64(263719581847590u64)) * (node_5621)))
                    + ((BFieldElement::from_raw_u64(76643691379275u64))
                        * (current_base_row[297usize])))
                    + ((BFieldElement::from_raw_u64(115096533571410u64))
                        * (current_base_row[298usize])))
                    + ((BFieldElement::from_raw_u64(256362302871255u64))
                        * (current_base_row[299usize])))
                    + ((BFieldElement::from_raw_u64(51629801853195u64))
                        * (current_base_row[300usize])))
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
                        * (current_base_row[308usize]))) + (current_base_row[116usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (node_6297))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(32014686216930u64))
                    * (node_5588))
                    + ((BFieldElement::from_raw_u64(145268678818785u64)) * (node_5599)))
                    + ((BFieldElement::from_raw_u64(123480309731250u64)) * (node_5610)))
                    + ((BFieldElement::from_raw_u64(4758823762860u64)) * (node_5621)))
                    + ((BFieldElement::from_raw_u64(263719581847590u64))
                        * (current_base_row[297usize])))
                    + ((BFieldElement::from_raw_u64(76643691379275u64))
                        * (current_base_row[298usize])))
                    + ((BFieldElement::from_raw_u64(115096533571410u64))
                        * (current_base_row[299usize])))
                    + ((BFieldElement::from_raw_u64(256362302871255u64))
                        * (current_base_row[300usize])))
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
                        * (current_base_row[308usize]))) + (current_base_row[117usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[97usize]))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(185731565704980u64))
                    * (node_5588))
                    + ((BFieldElement::from_raw_u64(32014686216930u64)) * (node_5599)))
                    + ((BFieldElement::from_raw_u64(145268678818785u64)) * (node_5610)))
                    + ((BFieldElement::from_raw_u64(123480309731250u64)) * (node_5621)))
                    + ((BFieldElement::from_raw_u64(4758823762860u64))
                        * (current_base_row[297usize])))
                    + ((BFieldElement::from_raw_u64(263719581847590u64))
                        * (current_base_row[298usize])))
                    + ((BFieldElement::from_raw_u64(76643691379275u64))
                        * (current_base_row[299usize])))
                    + ((BFieldElement::from_raw_u64(115096533571410u64))
                        * (current_base_row[300usize])))
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
                        * (current_base_row[308usize]))) + (current_base_row[118usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[98usize]))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(231348413345175u64))
                    * (node_5588))
                    + ((BFieldElement::from_raw_u64(185731565704980u64)) * (node_5599)))
                    + ((BFieldElement::from_raw_u64(32014686216930u64)) * (node_5610)))
                    + ((BFieldElement::from_raw_u64(145268678818785u64)) * (node_5621)))
                    + ((BFieldElement::from_raw_u64(123480309731250u64))
                        * (current_base_row[297usize])))
                    + ((BFieldElement::from_raw_u64(4758823762860u64))
                        * (current_base_row[298usize])))
                    + ((BFieldElement::from_raw_u64(263719581847590u64))
                        * (current_base_row[299usize])))
                    + ((BFieldElement::from_raw_u64(76643691379275u64))
                        * (current_base_row[300usize])))
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
                        * (current_base_row[308usize]))) + (current_base_row[119usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[99usize]))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(51685636428030u64))
                    * (node_5588))
                    + ((BFieldElement::from_raw_u64(231348413345175u64)) * (node_5599)))
                    + ((BFieldElement::from_raw_u64(185731565704980u64)) * (node_5610)))
                    + ((BFieldElement::from_raw_u64(32014686216930u64)) * (node_5621)))
                    + ((BFieldElement::from_raw_u64(145268678818785u64))
                        * (current_base_row[297usize])))
                    + ((BFieldElement::from_raw_u64(123480309731250u64))
                        * (current_base_row[298usize])))
                    + ((BFieldElement::from_raw_u64(4758823762860u64))
                        * (current_base_row[299usize])))
                    + ((BFieldElement::from_raw_u64(263719581847590u64))
                        * (current_base_row[300usize])))
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
                        * (current_base_row[308usize]))) + (current_base_row[120usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[100usize]))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(244602682417545u64))
                    * (node_5588))
                    + ((BFieldElement::from_raw_u64(51685636428030u64)) * (node_5599)))
                    + ((BFieldElement::from_raw_u64(231348413345175u64)) * (node_5610)))
                    + ((BFieldElement::from_raw_u64(185731565704980u64)) * (node_5621)))
                    + ((BFieldElement::from_raw_u64(32014686216930u64))
                        * (current_base_row[297usize])))
                    + ((BFieldElement::from_raw_u64(145268678818785u64))
                        * (current_base_row[298usize])))
                    + ((BFieldElement::from_raw_u64(123480309731250u64))
                        * (current_base_row[299usize])))
                    + ((BFieldElement::from_raw_u64(4758823762860u64))
                        * (current_base_row[300usize])))
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
                        * (current_base_row[308usize]))) + (current_base_row[121usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[101usize]))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(118201794925695u64))
                    * (node_5588))
                    + ((BFieldElement::from_raw_u64(244602682417545u64)) * (node_5599)))
                    + ((BFieldElement::from_raw_u64(51685636428030u64)) * (node_5610)))
                    + ((BFieldElement::from_raw_u64(231348413345175u64)) * (node_5621)))
                    + ((BFieldElement::from_raw_u64(185731565704980u64))
                        * (current_base_row[297usize])))
                    + ((BFieldElement::from_raw_u64(32014686216930u64))
                        * (current_base_row[298usize])))
                    + ((BFieldElement::from_raw_u64(145268678818785u64))
                        * (current_base_row[299usize])))
                    + ((BFieldElement::from_raw_u64(123480309731250u64))
                        * (current_base_row[300usize])))
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
                        * (current_base_row[308usize]))) + (current_base_row[122usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[102usize]))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(177601192615545u64))
                    * (node_5588))
                    + ((BFieldElement::from_raw_u64(118201794925695u64)) * (node_5599)))
                    + ((BFieldElement::from_raw_u64(244602682417545u64)) * (node_5610)))
                    + ((BFieldElement::from_raw_u64(51685636428030u64)) * (node_5621)))
                    + ((BFieldElement::from_raw_u64(231348413345175u64))
                        * (current_base_row[297usize])))
                    + ((BFieldElement::from_raw_u64(185731565704980u64))
                        * (current_base_row[298usize])))
                    + ((BFieldElement::from_raw_u64(32014686216930u64))
                        * (current_base_row[299usize])))
                    + ((BFieldElement::from_raw_u64(145268678818785u64))
                        * (current_base_row[300usize])))
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
                        * (current_base_row[308usize]))) + (current_base_row[123usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[103usize]))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(175668457332795u64))
                    * (node_5588))
                    + ((BFieldElement::from_raw_u64(177601192615545u64)) * (node_5599)))
                    + ((BFieldElement::from_raw_u64(118201794925695u64)) * (node_5610)))
                    + ((BFieldElement::from_raw_u64(244602682417545u64)) * (node_5621)))
                    + ((BFieldElement::from_raw_u64(51685636428030u64))
                        * (current_base_row[297usize])))
                    + ((BFieldElement::from_raw_u64(231348413345175u64))
                        * (current_base_row[298usize])))
                    + ((BFieldElement::from_raw_u64(185731565704980u64))
                        * (current_base_row[299usize])))
                    + ((BFieldElement::from_raw_u64(32014686216930u64))
                        * (current_base_row[300usize])))
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
                        * (current_base_row[308usize]))) + (current_base_row[124usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[104usize]))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(51629801853195u64))
                    * (node_5588))
                    + ((BFieldElement::from_raw_u64(175668457332795u64)) * (node_5599)))
                    + ((BFieldElement::from_raw_u64(177601192615545u64)) * (node_5610)))
                    + ((BFieldElement::from_raw_u64(118201794925695u64)) * (node_5621)))
                    + ((BFieldElement::from_raw_u64(244602682417545u64))
                        * (current_base_row[297usize])))
                    + ((BFieldElement::from_raw_u64(51685636428030u64))
                        * (current_base_row[298usize])))
                    + ((BFieldElement::from_raw_u64(231348413345175u64))
                        * (current_base_row[299usize])))
                    + ((BFieldElement::from_raw_u64(185731565704980u64))
                        * (current_base_row[300usize])))
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
                        * (current_base_row[308usize]))) + (current_base_row[125usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[105usize]))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(256362302871255u64))
                    * (node_5588))
                    + ((BFieldElement::from_raw_u64(51629801853195u64)) * (node_5599)))
                    + ((BFieldElement::from_raw_u64(175668457332795u64)) * (node_5610)))
                    + ((BFieldElement::from_raw_u64(177601192615545u64)) * (node_5621)))
                    + ((BFieldElement::from_raw_u64(118201794925695u64))
                        * (current_base_row[297usize])))
                    + ((BFieldElement::from_raw_u64(244602682417545u64))
                        * (current_base_row[298usize])))
                    + ((BFieldElement::from_raw_u64(51685636428030u64))
                        * (current_base_row[299usize])))
                    + ((BFieldElement::from_raw_u64(231348413345175u64))
                        * (current_base_row[300usize])))
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
                        * (current_base_row[308usize]))) + (current_base_row[126usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[106usize]))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(115096533571410u64))
                    * (node_5588))
                    + ((BFieldElement::from_raw_u64(256362302871255u64)) * (node_5599)))
                    + ((BFieldElement::from_raw_u64(51629801853195u64)) * (node_5610)))
                    + ((BFieldElement::from_raw_u64(175668457332795u64)) * (node_5621)))
                    + ((BFieldElement::from_raw_u64(177601192615545u64))
                        * (current_base_row[297usize])))
                    + ((BFieldElement::from_raw_u64(118201794925695u64))
                        * (current_base_row[298usize])))
                    + ((BFieldElement::from_raw_u64(244602682417545u64))
                        * (current_base_row[299usize])))
                    + ((BFieldElement::from_raw_u64(51685636428030u64))
                        * (current_base_row[300usize])))
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
                        * (current_base_row[308usize]))) + (current_base_row[127usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[107usize]))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(76643691379275u64))
                    * (node_5588))
                    + ((BFieldElement::from_raw_u64(115096533571410u64)) * (node_5599)))
                    + ((BFieldElement::from_raw_u64(256362302871255u64)) * (node_5610)))
                    + ((BFieldElement::from_raw_u64(51629801853195u64)) * (node_5621)))
                    + ((BFieldElement::from_raw_u64(175668457332795u64))
                        * (current_base_row[297usize])))
                    + ((BFieldElement::from_raw_u64(177601192615545u64))
                        * (current_base_row[298usize])))
                    + ((BFieldElement::from_raw_u64(118201794925695u64))
                        * (current_base_row[299usize])))
                    + ((BFieldElement::from_raw_u64(244602682417545u64))
                        * (current_base_row[300usize])))
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
                        * (current_base_row[308usize]))) + (current_base_row[128usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[108usize]))),
            (current_base_row[129usize]) * (node_6920),
            (current_base_row[135usize]) * (node_6972),
            ((next_base_row[135usize]) * (next_base_row[136usize]))
                + ((node_6972)
                    * (((next_base_row[136usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[136usize])))
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)))),
            ((next_base_row[139usize]) * (current_base_row[143usize])) * (node_7022),
            (next_base_row[139usize]) * (current_base_row[145usize]),
            (node_7032)
                * ((next_base_row[142usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[142usize]))),
            (((node_7032) * (current_base_row[143usize])) * (node_7022)) * (node_7040),
            ((node_7032) * (current_base_row[145usize])) * (node_7040),
            (((node_7032) * (node_7022)) * (node_7025)) * (node_7046),
            ((node_7032) * (node_7028)) * (node_7049),
            (((current_base_row[309usize]) * (node_7067)) * (node_7069))
                * (current_base_row[147usize]),
            ((node_7072) * (node_7069)) * (node_7074),
            (((current_base_row[313usize]) * (node_7046)) * (node_7028)) * (node_7074),
            (((current_base_row[313usize]) * (node_7025)) * (node_7049))
                * (current_base_row[147usize]),
            ((current_base_row[348usize])
                * ((current_base_row[139usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                * ((current_base_row[147usize])
                    + (BFieldElement::from_raw_u64(18446744060824649731u64))),
            ((current_base_row[348usize]) * (current_base_row[139usize]))
                * (current_base_row[147usize]),
            ((node_7032) * (current_base_row[347usize]))
                * (((current_base_row[147usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((BFieldElement::from_raw_u64(8589934590u64))
                            * (next_base_row[147usize]))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((node_7025) * (node_7028)))),
            (current_base_row[354usize]) * ((current_base_row[147usize]) + (node_7038)),
            ((current_base_row[334usize]) * (next_base_row[143usize]))
                * ((next_base_row[147usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[147usize]))),
            (current_base_row[349usize])
                * ((next_base_row[143usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[143usize]))),
            ((current_base_row[349usize]) * (node_7049))
                * ((current_base_row[147usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (node_7133))),
            ((current_base_row[349usize]) * (node_7028))
                * ((current_base_row[147usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[355usize]))),
            ((node_7032) * ((current_base_row[332usize]) * (node_7060)))
                * (((current_base_row[147usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[147usize]))) + (node_7083)),
            (current_base_row[169usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_246) * (node_248))),
            (current_base_row[170usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_1493) * (node_1474))),
            (current_base_row[171usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_1493) * (current_base_row[13usize]))),
            (current_base_row[172usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[12usize]) * (node_1474)) * (node_1464))),
            (current_base_row[173usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[170usize]) * (node_1464))),
            (current_base_row[174usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[171usize]) * (node_1464))),
            (current_base_row[175usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[169usize]) * (current_base_row[40usize]))),
            (current_base_row[176usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_246) * (current_base_row[41usize]))),
            (current_base_row[177usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[176usize]) * (node_251))),
            (current_base_row[178usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[172usize]) * (node_1466))),
            (current_base_row[179usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[172usize]) * (current_base_row[15usize]))),
            (current_base_row[180usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[173usize]) * (node_1466))),
            (current_base_row[181usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[173usize]) * (current_base_row[15usize]))),
            (current_base_row[182usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[169usize]) * (node_251))),
            (current_base_row[183usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[174usize]) * (node_1466))),
            (current_base_row[184usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[174usize]) * (current_base_row[15usize]))),
            (current_base_row[185usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[182usize]) * (current_base_row[39usize]))),
            (current_base_row[186usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[175usize]) * (node_298))),
            (current_base_row[187usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[175usize]) * (current_base_row[39usize]))),
            (current_base_row[188usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[177usize]) * (node_298))),
            (current_base_row[189usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[177usize]) * (current_base_row[39usize]))),
            (current_base_row[190usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[12usize]) * (current_base_row[13usize]))
                        * (node_1464))),
            (current_base_row[191usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[179usize]) * (node_1468))),
            (current_base_row[192usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[178usize]) * (node_1468))),
            (current_base_row[193usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[180usize]) * (node_1468))),
            (current_base_row[194usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[183usize]) * (node_1468))),
            (current_base_row[195usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[181usize]) * (node_1468))),
            (current_base_row[196usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[184usize]) * (node_1468))),
            (current_base_row[197usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[171usize]) * (current_base_row[14usize]))),
            (current_base_row[198usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[190usize]) * (node_1466))),
            (current_base_row[199usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[170usize]) * (current_base_row[14usize]))),
            (current_base_row[200usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[178usize]) * (current_base_row[16usize]))),
            (current_base_row[201usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[181usize]) * (current_base_row[16usize]))),
            (current_base_row[202usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[198usize]) * (node_1468)) * (node_1470))
                        * (node_1472))),
            (current_base_row[203usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[191usize]) * (node_1470)) * (node_1472))),
            (current_base_row[204usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[180usize]) * (current_base_row[16usize]))),
            (current_base_row[205usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[179usize]) * (current_base_row[16usize]))
                        * (node_1470)) * (node_1472))),
            (current_base_row[206usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[183usize]) * (current_base_row[16usize]))),
            (current_base_row[207usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[184usize]) * (current_base_row[16usize]))),
            (current_base_row[208usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[200usize]) * (node_1470)) * (node_1472))),
            (current_base_row[209usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[194usize]) * (node_1470))),
            (current_base_row[210usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[193usize]) * (node_1470))),
            (current_base_row[211usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[196usize]) * (node_1470))),
            (current_base_row[212usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[192usize]) * (node_1470)) * (node_1472))),
            (current_base_row[213usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[209usize]) * (node_1472))),
            (current_base_row[214usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[210usize]) * (node_1472))),
            (current_base_row[215usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[191usize]) * (current_base_row[17usize]))
                        * (node_1472))),
            (current_base_row[216usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[195usize]) * (node_1470)) * (node_1472))),
            (current_base_row[217usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[190usize]) * (current_base_row[15usize]))
                        * (node_1468)) * (node_1470))),
            (current_base_row[218usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[197usize]) * (node_1466))),
            (current_base_row[219usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[199usize]) * (node_1466))),
            (current_base_row[220usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[192usize]) * (current_base_row[17usize]))
                        * (node_1472))),
            (current_base_row[221usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[217usize]) * (node_1472))),
            (current_base_row[222usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[197usize]) * (current_base_row[15usize]))),
            (current_base_row[223usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[201usize]) * (node_1470)) * (node_1472))),
            (current_base_row[224usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[211usize]) * (node_1472))),
            (current_base_row[225usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[204usize]) * (node_1470)) * (node_1472))),
            (current_base_row[226usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[199usize]) * (current_base_row[15usize]))),
            (current_base_row[227usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[42usize]) * (node_248))),
            (current_base_row[228usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[207usize]) * (node_1470)) * (node_1472))),
            (current_base_row[229usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[195usize]) * (current_base_row[17usize]))
                        * (node_1472))),
            (current_base_row[230usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[206usize]) * (node_1470))),
            (current_base_row[231usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[42usize]) * (current_base_row[41usize]))),
            (current_base_row[232usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[193usize]) * (current_base_row[17usize]))
                        * (node_1472))),
            (current_base_row[233usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[196usize]) * (current_base_row[17usize]))
                        * (node_1472))),
            (current_base_row[234usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[206usize]) * (current_base_row[17usize]))
                        * (node_1472))),
            (current_base_row[235usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[201usize]) * (current_base_row[17usize]))
                        * (node_1472))),
            (current_base_row[236usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[219usize]) * (node_1468)) * (node_1470))
                        * (node_1472))),
            (current_base_row[237usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[207usize]) * (current_base_row[17usize]))
                        * (node_1472))),
            (current_base_row[238usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[218usize]) * (node_1468)) * (node_1470))
                        * (node_1472))),
            (current_base_row[239usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[222usize]) * (node_1468)) * (node_1470))
                        * (node_1472))),
            (current_base_row[240usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[230usize]) * (node_1472))),
            (current_base_row[241usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[218usize]) * (current_base_row[16usize]))
                        * (node_1470)) * (node_1472))),
            (current_base_row[242usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[226usize]) * (node_1468)) * (node_1470))
                        * (node_1472))),
            (current_base_row[243usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[176usize]) * (current_base_row[40usize]))),
            (current_base_row[244usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[194usize]) * (current_base_row[17usize]))
                        * (node_1472))),
            (current_base_row[245usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[222usize]) * (current_base_row[16usize]))
                        * (node_1470)) * (node_1472))),
            (current_base_row[246usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[204usize]) * (current_base_row[17usize]))
                        * (node_1472))),
            (current_base_row[247usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[227usize]) * (node_251))),
            (current_base_row[248usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[219usize]) * (current_base_row[16usize]))
                        * (node_1470)) * (node_1472))),
            (current_base_row[249usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[97usize]) * (current_base_row[97usize]))
                        * (current_base_row[97usize])) * (current_base_row[97usize]))),
            (current_base_row[250usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[98usize]) * (current_base_row[98usize]))
                        * (current_base_row[98usize])) * (current_base_row[98usize]))),
            (current_base_row[251usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[226usize]) * (current_base_row[16usize]))
                        * (node_1470)) * (node_1472))),
            (current_base_row[252usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[99usize]) * (current_base_row[99usize]))
                        * (current_base_row[99usize])) * (current_base_row[99usize]))),
            (current_base_row[253usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[227usize]) * (current_base_row[40usize]))),
            (current_base_row[254usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[100usize]) * (current_base_row[100usize]))
                        * (current_base_row[100usize])) * (current_base_row[100usize]))),
            (current_base_row[255usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[231usize]) * (node_251))),
            (current_base_row[256usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[231usize]) * (current_base_row[40usize]))),
            (current_base_row[257usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[101usize]) * (current_base_row[101usize]))
                        * (current_base_row[101usize])) * (current_base_row[101usize]))),
            (current_base_row[258usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[102usize]) * (current_base_row[102usize]))
                        * (current_base_row[102usize])) * (current_base_row[102usize]))),
            (current_base_row[259usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[200usize]) * (current_base_row[17usize]))
                        * (node_1472))),
            (current_base_row[260usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[41usize])
                        * ((current_base_row[41usize])
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))))),
            (current_base_row[261usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[42usize])
                        * ((current_base_row[42usize])
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))))),
            (current_base_row[262usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[103usize]) * (current_base_row[103usize]))
                        * (current_base_row[103usize])) * (current_base_row[103usize]))),
            (current_base_row[263usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[198usize]) * (current_base_row[16usize]))
                        * (node_1470)) * (node_1472))),
            (current_base_row[264usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[104usize]) * (current_base_row[104usize]))
                        * (current_base_row[104usize])) * (current_base_row[104usize]))),
            (current_base_row[265usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[105usize]) * (current_base_row[105usize]))
                        * (current_base_row[105usize])) * (current_base_row[105usize]))),
            (current_base_row[266usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[106usize]) * (current_base_row[106usize]))
                        * (current_base_row[106usize])) * (current_base_row[106usize]))),
            (current_base_row[267usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[107usize]) * (current_base_row[107usize]))
                        * (current_base_row[107usize])) * (current_base_row[107usize]))),
            (current_base_row[268usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[40usize]) * (node_134))),
            (current_base_row[269usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[108usize]) * (current_base_row[108usize]))
                        * (current_base_row[108usize])) * (current_base_row[108usize]))),
            (current_base_row[270usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[243usize]) * (current_base_row[39usize]))),
            (current_base_row[271usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[247usize]) * (node_298))),
            (current_base_row[272usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[39usize]) * (current_base_row[22usize]))),
            (current_base_row[273usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[255usize]) * (node_298))),
            (current_base_row[274usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[209usize]) * (current_base_row[18usize]))),
            (current_base_row[275usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[211usize]) * (current_base_row[18usize]))),
            (current_base_row[276usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((next_base_row[64usize]) * (node_6454)) * (node_6456))
                        * (node_6458))),
            (current_base_row[277usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[210usize]) * (current_base_row[18usize]))),
            (current_base_row[278usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((next_base_row[62usize]) * (node_6462)) * (node_6408))),
            (current_base_row[279usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[230usize]) * (current_base_row[18usize]))),
            (current_base_row[280usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[43usize])
                        * ((current_base_row[43usize])
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                        * ((current_base_row[43usize])
                            + (BFieldElement::from_raw_u64(18446744060824649731u64))))
                        * ((current_base_row[43usize])
                            + (BFieldElement::from_raw_u64(18446744056529682436u64))))),
            (current_base_row[281usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[44usize])
                        * ((current_base_row[44usize])
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                        * ((current_base_row[44usize])
                            + (BFieldElement::from_raw_u64(18446744060824649731u64))))
                        * ((current_base_row[44usize])
                            + (BFieldElement::from_raw_u64(18446744056529682436u64))))),
            (current_base_row[282usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[185usize]) * (node_547))),
            (current_base_row[283usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[186usize])
                        * ((next_base_row[25usize]) + (node_187)))),
            (current_base_row[284usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[187usize])
                        * ((next_base_row[26usize]) + (node_187)))),
            (current_base_row[285usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[188usize])
                        * ((next_base_row[27usize]) + (node_187)))),
            (current_base_row[286usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[189usize]) * (node_726))),
            (current_base_row[287usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[185usize]) * (node_192))),
            (current_base_row[288usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[186usize])
                        * ((next_base_row[23usize]) + (node_195)))),
            (current_base_row[289usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[187usize])
                        * ((next_base_row[23usize]) + (node_199)))),
            (current_base_row[290usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[188usize])
                        * ((next_base_row[23usize]) + (node_203)))),
            (current_base_row[291usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[189usize]) * (node_413))),
            (current_base_row[292usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[189usize]) * (node_412))),
            (current_base_row[293usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[189usize]) * (node_725))),
            (current_base_row[294usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((node_7051) * (node_7054)) * (node_7058)) * (node_7060))),
            (current_base_row[295usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_7051)
                        * ((next_base_row[142usize])
                            + (BFieldElement::from_raw_u64(18446744043644780551u64))))),
            (current_base_row[296usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((node_6454) * (node_6456)) * (node_6458)) * (node_6460))),
            (current_base_row[297usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[249usize]) * (current_base_row[97usize]))
                        * (current_base_row[97usize])) * (current_base_row[97usize]))),
            (current_base_row[298usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[250usize]) * (current_base_row[98usize]))
                        * (current_base_row[98usize])) * (current_base_row[98usize]))),
            (current_base_row[299usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[252usize]) * (current_base_row[99usize]))
                        * (current_base_row[99usize])) * (current_base_row[99usize]))),
            (current_base_row[300usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[254usize]) * (current_base_row[100usize]))
                        * (current_base_row[100usize])) * (current_base_row[100usize]))),
            (current_base_row[301usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[257usize]) * (current_base_row[101usize]))
                        * (current_base_row[101usize])) * (current_base_row[101usize]))),
            (current_base_row[302usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[258usize]) * (current_base_row[102usize]))
                        * (current_base_row[102usize])) * (current_base_row[102usize]))),
            (current_base_row[303usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[262usize]) * (current_base_row[103usize]))
                        * (current_base_row[103usize])) * (current_base_row[103usize]))),
            (current_base_row[304usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[264usize]) * (current_base_row[104usize]))
                        * (current_base_row[104usize])) * (current_base_row[104usize]))),
            (current_base_row[305usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[265usize]) * (current_base_row[105usize]))
                        * (current_base_row[105usize])) * (current_base_row[105usize]))),
            (current_base_row[306usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[266usize]) * (current_base_row[106usize]))
                        * (current_base_row[106usize])) * (current_base_row[106usize]))),
            (current_base_row[307usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[267usize]) * (current_base_row[107usize]))
                        * (current_base_row[107usize])) * (current_base_row[107usize]))),
            (current_base_row[308usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[269usize]) * (current_base_row[108usize]))
                        * (current_base_row[108usize])) * (current_base_row[108usize]))),
            (current_base_row[309usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_7032) * ((current_base_row[294usize]) * (node_7064)))),
            (current_base_row[310usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[295usize]) * (node_7054))),
            (current_base_row[311usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[296usize]) * (node_6462))),
            (current_base_row[312usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[243usize]) * (node_298))),
            (current_base_row[313usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_7072) * (node_7067))),
            (current_base_row[314usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[247usize]) * (current_base_row[39usize]))),
            (current_base_row[315usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[182usize]) * (node_298))),
            (current_base_row[316usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((node_6398) * (node_6421)) * (next_base_row[62usize]))),
            (current_base_row[317usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[253usize]) * (node_298))),
            (current_base_row[318usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[253usize]) * (current_base_row[39usize]))),
            (current_base_row[319usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[185usize]) * (node_196))),
            (current_base_row[320usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[185usize]) * (node_200))),
            (current_base_row[321usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[186usize])
                        * ((next_base_row[24usize]) + (node_199)))),
            (current_base_row[322usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[186usize])
                        * ((next_base_row[25usize]) + (node_203)))),
            (current_base_row[323usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[187usize])
                        * ((next_base_row[24usize]) + (node_203)))),
            (current_base_row[324usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[187usize]) * (node_332))),
            (current_base_row[325usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[188usize])
                        * ((next_base_row[24usize]) + (node_207)))),
            (current_base_row[326usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[188usize])
                        * ((next_base_row[25usize]) + (node_211)))),
            (current_base_row[327usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[189usize]) * (node_414))),
            (current_base_row[328usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[189usize]) * (node_415))),
            (current_base_row[329usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[255usize]) * (current_base_row[39usize]))),
            (current_base_row[330usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[256usize]) * (node_298))),
            (current_base_row[331usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[256usize]) * (current_base_row[39usize]))),
            (current_base_row[332usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[310usize]) * (node_7058))),
            (current_base_row[333usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((((next_base_row[12usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                        * (next_base_row[13usize]))
                        * ((next_base_row[14usize])
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                        * ((next_base_row[15usize])
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))))),
            (current_base_row[334usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_7032)
                        * (((current_base_row[310usize]) * (node_7060)) * (node_7064)))),
            (current_base_row[335usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[22usize]) * (current_base_row[23usize]))),
            (current_base_row[336usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((next_base_row[22usize]) * (current_base_row[22usize]))),
            (current_base_row[337usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[39usize]) * (node_1292))),
            (current_base_row[338usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[311usize])
                        * (((node_6421) * (node_6404)) * (next_base_row[62usize])))),
            (current_base_row[339usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((node_6392) * (node_6394)) * (current_base_row[62usize]))),
            (current_base_row[340usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[236usize])
                        * ((next_base_row[22usize])
                            * (((current_base_row[39usize])
                                * ((next_base_row[23usize])
                                    + (BFieldElement::from_raw_u64(4294967296u64))))
                                + (BFieldElement::from_raw_u64(
                                    18446744065119617026u64,
                                )))))),
            (current_base_row[341usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[213usize])
                        * ((((node_932) * (current_base_row[22usize]))
                            + (((node_117) * (node_934)) * (node_134)))
                            + ((((node_114)
                                + (BFieldElement::from_raw_u64(18446744056529682436u64)))
                                * (node_934)) * (current_base_row[40usize]))))),
            (current_base_row[342usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[213usize])
                        * (((current_base_row[260usize])
                            * ((current_base_row[41usize])
                                + (BFieldElement::from_raw_u64(18446744060824649731u64))))
                            * ((current_base_row[41usize])
                                + (BFieldElement::from_raw_u64(
                                    18446744056529682436u64,
                                )))))),
            (current_base_row[343usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[213usize])
                        * (((current_base_row[261usize])
                            * ((current_base_row[42usize])
                                + (BFieldElement::from_raw_u64(18446744060824649731u64))))
                            * ((current_base_row[42usize])
                                + (BFieldElement::from_raw_u64(
                                    18446744056529682436u64,
                                )))))),
            (current_base_row[344usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[333usize]) * (next_base_row[16usize]))
                        * ((next_base_row[17usize])
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                        * ((next_base_row[18usize])
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))))),
            (current_base_row[345usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[64usize])
                        * ((current_base_row[64usize])
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                        * ((current_base_row[64usize])
                            + (BFieldElement::from_raw_u64(18446744060824649731u64))))
                        * ((current_base_row[64usize])
                            + (BFieldElement::from_raw_u64(18446744056529682436u64))))),
            (current_base_row[346usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((node_6417) * (node_6394)) * (current_base_row[62usize]))
                        * (node_6421))),
            (current_base_row[347usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[295usize]) * (node_7058)) * (node_7060))
                        * (node_7064))),
            (current_base_row[348usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[313usize])
                        * ((((BFieldElement::from_raw_u64(4294967295u64)) + (node_7083))
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (node_7028)))
                            + (((BFieldElement::from_raw_u64(8589934590u64))
                                * (node_7025)) * (node_7028))))),
            (current_base_row[349usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_7032) * ((current_base_row[332usize]) * (node_7064)))),
            (current_base_row[350usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[339usize])
                        * (((node_6398) * (node_6404)) * (next_base_row[62usize])))),
            (current_base_row[351usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((node_6425) * (current_base_row[62usize])) * (node_6404))),
            (current_base_row[352usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[311usize]) * (node_6404))
                        * (next_base_row[62usize])) * (node_6408))),
            (current_base_row[353usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[311usize])
                        * (((node_6513) * (node_6408)) * (node_6516)))),
            (current_base_row[354usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[334usize])
                        * ((BFieldElement::from_raw_u64(4294967295u64))
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((next_base_row[143usize]) * (next_base_row[144usize])))))
                        * (current_base_row[143usize]))),
            (current_base_row[355usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_7133) * (current_base_row[143usize]))),
        ];
        let ext_constraints = [
            (((BFieldElement::from_raw_u64(4294967295u64))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (current_base_row[5usize])))
                * (((node_52)
                    * ((challenges[InstructionLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((((challenges[ProgramAddressWeight])
                                * (current_base_row[0usize]))
                                + ((challenges[ProgramInstructionWeight])
                                    * (current_base_row[1usize])))
                                + ((challenges[ProgramNextInstructionWeight])
                                    * (next_base_row[1usize]))))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[2usize]))))
                + ((current_base_row[5usize]) * (node_52)),
            ((node_31)
                * (((next_ext_row[1usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((challenges[ProgramAttestationPrepareChunkIndeterminate])
                            * (current_ext_row[1usize])))) + (node_75)))
                + ((node_34)
                    * (((next_ext_row[1usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (challenges[ProgramAttestationPrepareChunkIndeterminate])))
                        + (node_75))),
            ((((((next_ext_row[2usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((challenges[ProgramAttestationSendChunkIndeterminate])
                        * (current_ext_row[2usize]))))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (next_ext_row[1usize]))) * (node_48))
                * ((BFieldElement::from_raw_u64(4294967295u64))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((next_base_row[4usize]) * (node_91)))))
                + ((node_89) * (next_base_row[6usize]))) + ((node_89) * (node_91)),
            ((((((((((((((((((((((((((((((((((((((((current_base_row[202usize])
                * (current_base_row[268usize]))
                + ((current_base_row[212usize]) * (node_550)))
                + ((current_base_row[203usize]) * (current_base_row[268usize])))
                + ((current_base_row[208usize]) * (current_base_row[292usize])))
                + ((current_base_row[205usize]) * (current_base_row[293usize])))
                + ((current_base_row[214usize]) * (node_824)))
                + ((current_base_row[216usize]) * (node_868)))
                + (current_base_row[342usize]))
                + ((current_base_row[220usize]) * (node_868)))
                + ((current_base_row[225usize]) * (node_876)))
                + ((current_base_row[223usize]) * (node_868)))
                + ((current_base_row[224usize]) * (node_188)))
                + ((current_base_row[215usize]) * (current_base_row[268usize])))
                + ((current_base_row[221usize]) * (current_base_row[268usize])))
                + ((current_base_row[240usize]) * (node_418)))
                + ((current_base_row[232usize])
                    * (((node_298) * (node_876))
                        + ((current_base_row[39usize]) * (node_728)))))
                + ((current_base_row[228usize]) * (node_121)))
                + ((current_base_row[229usize]) * (node_868)))
                + ((current_base_row[244usize])
                    * ((next_ext_row[6usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_ext_row[69usize])))))
                + ((current_base_row[246usize])
                    * ((next_ext_row[6usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_ext_row[70usize])))))
                + ((current_base_row[233usize]) * (node_192)))
                + ((current_base_row[234usize]) * (node_192)))
                + ((current_base_row[235usize]) * (node_868)))
                + ((current_base_row[237usize]) * (node_129)))
                + ((current_base_row[236usize]) * (node_550)))
                + ((current_base_row[238usize]) * (node_196)))
                + ((current_base_row[239usize]) * (node_196)))
                + ((current_base_row[241usize]) * (node_196)))
                + ((current_base_row[242usize]) * (node_872)))
                + ((current_base_row[245usize]) * (node_196)))
                + ((current_base_row[248usize]) * (node_932)))
                + ((current_base_row[251usize]) * (node_872)))
                + ((current_base_row[274usize]) * (node_334)))
                + ((current_base_row[275usize]) * (node_334)))
                + ((current_base_row[277usize]) * (node_884)))
                + ((current_base_row[279usize]) * (node_208)))
                + ((current_base_row[259usize]) * (current_base_row[268usize])))
                + ((current_base_row[263usize]) * (current_base_row[268usize])))
                * (node_4849)) + ((node_129) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((current_base_row[202usize])
                * (node_446)) + ((current_base_row[212usize]) * (node_555)))
                + ((current_base_row[203usize]) * (node_758)))
                + ((current_base_row[208usize]) * (node_835)))
                + ((current_base_row[205usize])
                    * ((current_base_row[317usize]) * (node_854))))
                + ((current_base_row[214usize]) * (node_884)))
                + ((current_base_row[216usize]) * (node_888)))
                + ((current_base_row[213usize]) * (node_125)))
                + ((current_base_row[220usize]) * (node_888)))
                + ((current_base_row[225usize]) * (node_896)))
                + ((current_base_row[223usize]) * (node_888)))
                + ((current_base_row[224usize]) * (node_208)))
                + ((current_base_row[215usize])
                    * ((((((current_base_row[185usize])
                        * ((node_824) + (BFieldElement::from_raw_u64(4294967295u64))))
                        + ((current_base_row[186usize])
                            * ((node_824)
                                + (BFieldElement::from_raw_u64(8589934590u64)))))
                        + ((current_base_row[187usize])
                            * ((node_824)
                                + (BFieldElement::from_raw_u64(12884901885u64)))))
                        + ((current_base_row[188usize])
                            * ((node_824)
                                + (BFieldElement::from_raw_u64(17179869180u64)))))
                        + ((current_base_row[189usize])
                            * ((node_824)
                                + (BFieldElement::from_raw_u64(21474836475u64)))))))
                + ((current_base_row[221usize])
                    * ((((((current_base_row[185usize])
                        * ((node_824)
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                        + ((current_base_row[186usize])
                            * ((node_824)
                                + (BFieldElement::from_raw_u64(18446744060824649731u64)))))
                        + ((current_base_row[187usize])
                            * ((node_824)
                                + (BFieldElement::from_raw_u64(18446744056529682436u64)))))
                        + ((current_base_row[188usize])
                            * ((node_824)
                                + (BFieldElement::from_raw_u64(18446744052234715141u64)))))
                        + ((current_base_row[189usize])
                            * ((node_824)
                                + (BFieldElement::from_raw_u64(
                                    18446744047939747846u64,
                                ))))))) + ((current_base_row[240usize]) * (node_400)))
                + ((current_base_row[232usize]) * (node_734)))
                + ((current_base_row[228usize]) * (node_411)))
                + ((current_base_row[229usize]) * (node_888)))
                + ((current_base_row[244usize])
                    * ((next_base_row[26usize]) + (node_239))))
                + ((current_base_row[246usize])
                    * ((next_base_row[36usize]) + (node_199))))
                + ((current_base_row[233usize]) * (node_212)))
                + ((current_base_row[234usize]) * (node_212)))
                + ((current_base_row[235usize]) * (node_888)))
                + ((current_base_row[237usize]) * (node_204)))
                + ((current_base_row[236usize]) * (node_555)))
                + ((current_base_row[238usize]) * (node_216)))
                + ((current_base_row[239usize]) * (node_216)))
                + ((current_base_row[241usize]) * (node_216)))
                + ((current_base_row[242usize]) * (node_892)))
                + ((current_base_row[245usize]) * (node_216)))
                + ((current_base_row[248usize]) * (node_892)))
                + ((current_base_row[251usize]) * (node_892)))
                + ((current_base_row[274usize]) * (node_339)))
                + ((current_base_row[275usize]) * (node_339)))
                + ((current_base_row[277usize]) * (node_904)))
                + ((current_base_row[279usize]) * (node_228)))
                + ((current_base_row[259usize]) * (node_758)))
                + ((current_base_row[263usize]) * (node_446))) * (node_4849))
                + ((node_880) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((current_base_row[202usize])
                * (node_451)) + ((current_base_row[212usize]) * (node_556)))
                + ((current_base_row[203usize]) * (node_763)))
                + ((current_base_row[208usize]) * (node_837)))
                + ((current_base_row[205usize])
                    * ((current_base_row[318usize])
                        * ((next_base_row[33usize]) + (node_545)))))
                + ((current_base_row[214usize]) * (node_888)))
                + ((current_base_row[216usize]) * (node_892)))
                + ((current_base_row[213usize]) * (node_129)))
                + ((current_base_row[220usize]) * (node_892)))
                + ((current_base_row[225usize]) * (node_900)))
                + ((current_base_row[223usize]) * (node_892)))
                + ((current_base_row[224usize]) * (node_212)))
                + ((current_base_row[215usize]) * (node_758)))
                + ((current_base_row[221usize]) * (node_446)))
                + ((current_base_row[240usize]) * (node_411)))
                + ((current_base_row[232usize]) * (node_735)))
                + ((current_base_row[228usize]) * (node_412)))
                + ((current_base_row[229usize]) * (node_892)))
                + ((current_base_row[244usize])
                    * ((next_base_row[27usize]) + (node_243))))
                + ((current_base_row[246usize])
                    * ((next_base_row[37usize]) + (node_203))))
                + ((current_base_row[233usize]) * (node_216)))
                + ((current_base_row[234usize]) * (node_216)))
                + ((current_base_row[235usize]) * (node_892)))
                + ((current_base_row[237usize]) * (node_208)))
                + ((current_base_row[236usize]) * (node_556)))
                + ((current_base_row[238usize]) * (node_220)))
                + ((current_base_row[239usize]) * (node_220)))
                + ((current_base_row[241usize]) * (node_220)))
                + ((current_base_row[242usize]) * (node_896)))
                + ((current_base_row[245usize]) * (node_220)))
                + ((current_base_row[248usize]) * (node_896)))
                + ((current_base_row[251usize]) * (node_896)))
                + ((current_base_row[274usize]) * (node_340)))
                + ((current_base_row[275usize]) * (node_340)))
                + ((current_base_row[277usize]) * (node_908)))
                + ((current_base_row[279usize]) * (node_232)))
                + ((current_base_row[259usize]) * (node_763)))
                + ((current_base_row[263usize]) * (node_451))) * (node_4849))
                + ((node_884) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((current_base_row[202usize])
                * (node_456)) + ((current_base_row[212usize]) * (node_557)))
                + ((current_base_row[203usize]) * (node_768)))
                + ((current_base_row[208usize]) * (node_839)))
                + ((current_base_row[205usize])
                    * ((current_base_row[273usize])
                        * ((next_base_row[34usize]) + (node_545)))))
                + ((current_base_row[214usize]) * (node_892)))
                + ((current_base_row[216usize]) * (node_896)))
                + ((current_base_row[213usize]) * (node_188)))
                + ((current_base_row[220usize]) * (node_896)))
                + ((current_base_row[225usize]) * (node_904)))
                + ((current_base_row[223usize]) * (node_896)))
                + ((current_base_row[224usize]) * (node_216)))
                + (current_ext_row[81usize])) + (current_ext_row[82usize]))
                + ((current_base_row[240usize]) * (node_533)))
                + ((current_base_row[232usize]) * (node_713)))
                + ((current_base_row[228usize]) * (node_413)))
                + ((current_base_row[229usize]) * (node_896)))
                + ((current_base_row[244usize]) * (node_533)))
                + ((current_base_row[246usize]) * (node_533)))
                + ((current_base_row[233usize]) * (node_220)))
                + ((current_base_row[234usize]) * (node_220)))
                + ((current_base_row[235usize]) * (node_896)))
                + ((current_base_row[237usize]) * (node_212)))
                + ((current_base_row[236usize]) * (node_557)))
                + ((current_base_row[238usize]) * (node_224)))
                + ((current_base_row[239usize]) * (node_224)))
                + ((current_base_row[241usize]) * (node_224)))
                + ((current_base_row[242usize]) * (node_900)))
                + ((current_base_row[245usize]) * (node_224)))
                + ((current_base_row[248usize]) * (node_900)))
                + ((current_base_row[251usize]) * (node_900)))
                + ((current_base_row[274usize]) * (node_341)))
                + ((current_base_row[275usize]) * (node_341)))
                + ((current_base_row[277usize]) * (node_912)))
                + ((current_base_row[279usize]) * (node_236)))
                + ((current_base_row[259usize]) * (node_768)))
                + ((current_base_row[263usize]) * (node_456))) * (node_4849))
                + ((node_888) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((current_base_row[202usize])
                * (node_461)) + ((current_base_row[212usize]) * (node_558)))
                + ((current_base_row[203usize]) * (node_773)))
                + ((current_base_row[208usize]) * (node_841)))
                + ((current_base_row[205usize])
                    * ((current_base_row[329usize])
                        * ((next_base_row[35usize]) + (node_545)))))
                + ((current_base_row[214usize]) * (node_896)))
                + ((current_base_row[216usize]) * (node_900)))
                + ((current_base_row[213usize]) * (node_192)))
                + ((current_base_row[220usize]) * (node_900)))
                + ((current_base_row[225usize]) * (node_908)))
                + ((current_base_row[223usize]) * (node_900)))
                + ((current_base_row[224usize]) * (node_220)))
                + ((current_base_row[215usize]) * (node_768)))
                + ((current_base_row[221usize]) * (node_456)))
                + ((current_base_row[240usize]) * (node_537)))
                + ((current_base_row[232usize]) * (node_724)))
                + ((current_base_row[228usize]) * (node_414)))
                + ((current_base_row[229usize]) * (node_900)))
                + ((current_base_row[244usize]) * (node_537)))
                + ((current_base_row[246usize]) * (node_537)))
                + ((current_base_row[233usize]) * (node_224)))
                + ((current_base_row[234usize]) * (node_224)))
                + ((current_base_row[235usize]) * (node_900)))
                + ((current_base_row[237usize]) * (node_216)))
                + ((current_base_row[236usize]) * (node_558)))
                + ((current_base_row[238usize]) * (node_228)))
                + ((current_base_row[239usize]) * (node_228)))
                + ((current_base_row[241usize]) * (node_228)))
                + ((current_base_row[242usize]) * (node_904)))
                + ((current_base_row[245usize]) * (node_228)))
                + ((current_base_row[248usize]) * (node_904)))
                + ((current_base_row[251usize]) * (node_904)))
                + ((current_base_row[274usize]) * (node_317)))
                + ((current_base_row[275usize]) * (node_317)))
                + ((current_base_row[277usize]) * (node_916)))
                + ((current_base_row[279usize]) * (node_240)))
                + ((current_base_row[259usize]) * (node_773)))
                + ((current_base_row[263usize]) * (node_461))) * (node_4849))
                + ((node_892) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((current_base_row[202usize])
                * (node_466)) + ((current_base_row[212usize]) * (node_559)))
                + ((current_base_row[203usize]) * (node_778)))
                + ((current_base_row[208usize]) * (node_843)))
                + ((current_base_row[205usize])
                    * ((current_base_row[330usize])
                        * ((next_base_row[36usize]) + (node_545)))))
                + ((current_base_row[214usize]) * (node_900)))
                + ((current_base_row[216usize]) * (node_904)))
                + ((current_base_row[213usize]) * (node_196)))
                + ((current_base_row[220usize]) * (node_904)))
                + ((current_base_row[225usize]) * (node_912)))
                + ((current_base_row[223usize]) * (node_904)))
                + ((current_base_row[224usize]) * (node_224)))
                + ((current_base_row[215usize]) * (node_773)))
                + ((current_base_row[221usize]) * (node_461)))
                + ((current_base_row[240usize]) * (node_541)))
                + ((current_base_row[232usize]) * (node_121)))
                + ((current_base_row[228usize]) * (node_415)))
                + ((current_base_row[229usize]) * (node_904)))
                + ((current_base_row[244usize]) * (node_541)))
                + ((current_base_row[246usize]) * (node_541)))
                + ((current_base_row[233usize]) * (node_228)))
                + ((current_base_row[234usize]) * (node_228)))
                + ((current_base_row[235usize]) * (node_904)))
                + ((current_base_row[237usize]) * (node_220)))
                + ((current_base_row[236usize]) * (node_559)))
                + ((current_base_row[238usize]) * (node_232)))
                + ((current_base_row[239usize]) * (node_232)))
                + ((current_base_row[241usize]) * (node_232)))
                + ((current_base_row[242usize]) * (node_908)))
                + ((current_base_row[245usize]) * (node_232)))
                + ((current_base_row[248usize]) * (node_908)))
                + ((current_base_row[251usize]) * (node_908)))
                + ((current_base_row[274usize]) * (node_328)))
                + ((current_base_row[275usize]) * (node_328)))
                + ((current_base_row[277usize]) * (node_920)))
                + ((current_base_row[279usize]) * (node_244)))
                + ((current_base_row[259usize]) * (node_778)))
                + ((current_base_row[263usize]) * (node_466))) * (node_4849))
                + ((node_896) * (next_base_row[8usize])),
            (((((((((((((((((((((((((((((((((((((current_base_row[202usize])
                * (node_476)) + ((current_base_row[212usize]) * (node_561)))
                + ((current_base_row[203usize]) * (node_788)))
                + ((current_base_row[208usize]) * (node_132)))
                + ((current_base_row[205usize]) * (node_256)))
                + ((current_base_row[214usize]) * (node_908)))
                + ((current_base_row[216usize]) * (node_912)))
                + ((current_base_row[213usize]) * (node_204)))
                + ((current_base_row[220usize]) * (node_912)))
                + ((current_base_row[225usize]) * (node_920)))
                + ((current_base_row[223usize]) * (node_912)))
                + ((current_base_row[224usize]) * (node_232)))
                + ((current_base_row[215usize]) * (node_783)))
                + ((current_base_row[221usize]) * (node_471)))
                + ((current_base_row[232usize]) * (node_129)))
                + ((current_base_row[228usize]) * (node_417)))
                + ((current_base_row[229usize]) * (node_912)))
                + ((current_base_row[233usize]) * (node_236)))
                + ((current_base_row[234usize]) * (node_236)))
                + ((current_base_row[235usize]) * (node_912)))
                + ((current_base_row[237usize]) * (node_228)))
                + ((current_base_row[236usize]) * (node_561)))
                + ((current_base_row[238usize]) * (node_240)))
                + ((current_base_row[239usize]) * (node_240)))
                + ((current_base_row[241usize]) * (node_240)))
                + ((current_base_row[242usize]) * (node_916)))
                + ((current_base_row[245usize]) * (node_240)))
                + ((current_base_row[248usize]) * (node_916)))
                + ((current_base_row[251usize]) * (node_916)))
                + ((current_base_row[274usize]) * (node_125)))
                + ((current_base_row[275usize]) * (node_125)))
                + ((current_base_row[277usize]) * (node_159)))
                + ((current_base_row[279usize]) * (node_184)))
                + ((current_base_row[259usize]) * (node_788)))
                + ((current_base_row[263usize]) * (node_476))) * (node_4849))
                + ((node_904) * (next_base_row[8usize])),
            (((((((((((((((((((((((((((((((((((((current_base_row[202usize])
                * (node_481)) + ((current_base_row[212usize]) * (node_572)))
                + ((current_base_row[203usize]) * (node_793)))
                + ((current_base_row[208usize]) * (current_base_row[268usize])))
                + ((current_base_row[205usize]) * (node_302)))
                + ((current_base_row[214usize]) * (node_912)))
                + ((current_base_row[216usize]) * (node_916)))
                + ((current_base_row[213usize]) * (node_208)))
                + ((current_base_row[220usize]) * (node_916)))
                + ((current_base_row[225usize]) * (node_924)))
                + ((current_base_row[223usize]) * (node_916)))
                + ((current_base_row[224usize]) * (node_236)))
                + ((current_base_row[215usize]) * (node_788)))
                + ((current_base_row[221usize]) * (node_476)))
                + ((current_base_row[232usize]) * (node_932)))
                + ((current_base_row[228usize]) * (node_418)))
                + ((current_base_row[229usize]) * (node_916)))
                + ((current_base_row[233usize]) * (node_240)))
                + ((current_base_row[234usize]) * (node_240)))
                + ((current_base_row[235usize]) * (node_916)))
                + ((current_base_row[237usize]) * (node_232)))
                + ((current_base_row[236usize]) * (node_572)))
                + ((current_base_row[238usize]) * (node_244)))
                + ((current_base_row[239usize]) * (node_244)))
                + ((current_base_row[241usize]) * (node_244)))
                + ((current_base_row[242usize]) * (node_920)))
                + ((current_base_row[245usize]) * (node_244)))
                + ((current_base_row[248usize]) * (node_920)))
                + ((current_base_row[251usize]) * (node_920)))
                + ((current_base_row[274usize]) * (node_129)))
                + ((current_base_row[275usize]) * (node_129)))
                + ((current_base_row[277usize]) * (node_927)))
                + ((current_base_row[279usize]) * (node_121)))
                + ((current_base_row[259usize]) * (node_793)))
                + ((current_base_row[263usize]) * (node_481))) * (node_4849))
                + ((node_908) * (next_base_row[8usize])),
            (((((((((((((((((((((((((((((((((((((current_base_row[202usize])
                * (node_486)) + ((current_base_row[212usize]) * (node_121)))
                + ((current_base_row[203usize]) * (node_798)))
                + ((current_base_row[208usize]) * (current_base_row[260usize])))
                + ((current_base_row[205usize]) * (node_345)))
                + ((current_base_row[214usize]) * (node_916)))
                + ((current_base_row[216usize]) * (node_920)))
                + ((current_base_row[213usize]) * (node_212)))
                + ((current_base_row[220usize]) * (node_920)))
                + ((current_base_row[225usize]) * (node_159)))
                + ((current_base_row[223usize]) * (node_920)))
                + ((current_base_row[224usize]) * (node_240)))
                + ((current_base_row[215usize]) * (node_793)))
                + ((current_base_row[221usize]) * (node_481)))
                + ((current_base_row[232usize]) * (node_533)))
                + ((current_base_row[228usize]) * (node_419)))
                + ((current_base_row[229usize]) * (node_920)))
                + ((current_base_row[233usize]) * (node_244)))
                + ((current_base_row[234usize]) * (node_244)))
                + ((current_base_row[235usize]) * (node_920)))
                + ((current_base_row[237usize]) * (node_236)))
                + ((current_base_row[236usize]) * (node_121)))
                + ((current_base_row[238usize]) * (node_160)))
                + ((current_base_row[239usize]) * (node_160)))
                + ((current_base_row[241usize]) * (node_160)))
                + ((current_base_row[242usize]) * (node_924)))
                + ((current_base_row[245usize]) * (node_160)))
                + ((current_base_row[248usize]) * (node_924)))
                + ((current_base_row[251usize]) * (node_924)))
                + ((current_base_row[274usize]) * (node_932)))
                + ((current_base_row[275usize]) * (node_932)))
                + ((current_base_row[277usize]) * (node_121)))
                + ((current_base_row[279usize]) * (node_125)))
                + ((current_base_row[259usize]) * (node_798)))
                + ((current_base_row[263usize]) * (node_486))) * (node_4849))
                + ((node_912) * (next_base_row[8usize])),
            (((((((((((((((((((((((((((((((((((((current_base_row[202usize])
                * (node_491)) + ((current_base_row[212usize]) * (node_125)))
                + ((current_base_row[203usize]) * (node_803)))
                + ((current_base_row[208usize]) * (current_base_row[261usize])))
                + ((current_base_row[205usize]) * (node_387)))
                + ((current_base_row[214usize]) * (node_920)))
                + ((current_base_row[216usize]) * (node_924)))
                + ((current_base_row[213usize]) * (node_216)))
                + ((current_base_row[220usize]) * (node_924)))
                + ((current_base_row[225usize]) * (node_927)))
                + ((current_base_row[223usize]) * (node_924)))
                + ((current_base_row[224usize]) * (node_244)))
                + ((current_base_row[215usize]) * (node_798)))
                + ((current_base_row[221usize]) * (node_486)))
                + ((current_base_row[232usize]) * (node_537)))
                + ((current_base_row[228usize]) * (node_420)))
                + ((current_base_row[229usize]) * (node_924)))
                + ((current_base_row[233usize]) * (node_160)))
                + ((current_base_row[234usize]) * (node_160)))
                + ((current_base_row[235usize]) * (node_924)))
                + ((current_base_row[237usize]) * (node_240)))
                + ((current_base_row[236usize]) * (node_125)))
                + ((current_base_row[238usize]) * (node_184)))
                + ((current_base_row[239usize]) * (node_184)))
                + ((current_base_row[241usize]) * (node_184)))
                + ((current_base_row[242usize]) * (node_159)))
                + ((current_base_row[245usize]) * (node_184)))
                + ((current_base_row[248usize]) * (node_159)))
                + ((current_base_row[251usize]) * (node_159)))
                + ((current_base_row[274usize]) * (node_533)))
                + ((current_base_row[275usize]) * (node_533)))
                + ((current_base_row[277usize]) * (node_125)))
                + ((current_base_row[279usize]) * (node_129)))
                + ((current_base_row[259usize]) * (node_803)))
                + ((current_base_row[263usize]) * (node_491))) * (node_4849))
                + ((node_916) * (next_base_row[8usize])),
            (((((((((((((((((((((((((((((((((((((current_base_row[202usize])
                * (node_496)) + ((current_base_row[212usize]) * (node_129)))
                + ((current_base_row[203usize]) * (node_808)))
                + ((current_base_row[208usize]) * (node_155)))
                + ((current_base_row[205usize]) * (current_base_row[292usize])))
                + ((current_base_row[214usize]) * (node_924)))
                + ((current_base_row[216usize]) * (node_159)))
                + ((current_base_row[213usize]) * (node_220)))
                + ((current_base_row[220usize]) * (node_159)))
                + ((current_base_row[225usize]) * (node_533)))
                + ((current_base_row[223usize]) * (node_159)))
                + ((current_base_row[224usize]) * (node_160)))
                + ((current_base_row[215usize]) * (node_803)))
                + ((current_base_row[221usize]) * (node_491)))
                + ((current_base_row[232usize]) * (node_541)))
                + ((current_base_row[228usize]) * (node_421)))
                + ((current_base_row[229usize]) * (node_159)))
                + ((current_base_row[233usize]) * (node_184)))
                + ((current_base_row[234usize]) * (node_184)))
                + ((current_base_row[235usize]) * (node_159)))
                + ((current_base_row[237usize]) * (node_244)))
                + ((current_base_row[236usize]) * (node_129)))
                + ((current_base_row[238usize]) * (node_533)))
                + ((current_base_row[239usize]) * (node_533)))
                + ((current_base_row[241usize]) * (node_533)))
                + ((current_base_row[242usize]) * (node_927)))
                + ((current_base_row[245usize]) * (node_533)))
                + ((current_base_row[248usize]) * (node_927)))
                + ((current_base_row[251usize]) * (node_927)))
                + ((current_base_row[274usize]) * (node_537)))
                + ((current_base_row[275usize]) * (node_537)))
                + ((current_base_row[277usize]) * (node_129)))
                + ((current_base_row[279usize]) * (node_932)))
                + ((current_base_row[259usize]) * (node_808)))
                + ((current_base_row[263usize]) * (node_496))) * (node_4849))
                + ((node_920) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((current_base_row[202usize]) * (node_501))
                + ((current_base_row[212usize]) * (node_117)))
                + ((current_base_row[203usize]) * (node_813)))
                + ((current_base_row[208usize]) * (node_121)))
                + ((current_base_row[205usize]) * (node_827)))
                + ((current_base_row[214usize]) * (node_159)))
                + ((current_base_row[216usize]) * (node_927)))
                + ((current_base_row[213usize]) * (node_224)))
                + ((current_base_row[220usize]) * (node_927)))
                + ((current_base_row[225usize]) * (node_537)))
                + ((current_base_row[223usize]) * (node_927)))
                + ((current_base_row[224usize]) * (node_184)))
                + ((current_base_row[215usize]) * (node_808)))
                + ((current_base_row[221usize]) * (node_496)))
                + ((current_base_row[228usize]) * (node_422)))
                + ((current_base_row[229usize]) * (node_927)))
                + ((current_base_row[233usize]) * (node_533)))
                + ((current_base_row[234usize]) * (node_533)))
                + ((current_base_row[235usize]) * (node_927)))
                + ((current_base_row[237usize]) * (node_160)))
                + ((current_base_row[236usize]) * (node_932)))
                + ((current_base_row[238usize]) * (node_537)))
                + ((current_base_row[239usize]) * (node_537)))
                + ((current_base_row[241usize]) * (node_537)))
                + ((current_base_row[242usize]) * (node_533)))
                + ((current_base_row[245usize]) * (node_537)))
                + ((current_base_row[248usize]) * (node_533)))
                + ((current_base_row[251usize]) * (node_533)))
                + ((current_base_row[274usize]) * (node_541)))
                + ((current_base_row[275usize]) * (node_541)))
                + ((current_base_row[277usize]) * (node_932)))
                + ((current_base_row[279usize]) * (node_533)))
                + ((current_base_row[259usize]) * (node_813)))
                + ((current_base_row[263usize]) * (node_501))) * (node_4849))
                + ((node_924) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((current_base_row[202usize]) * (node_505))
                + ((current_base_row[212usize]) * (node_533)))
                + ((current_base_row[203usize]) * (node_817)))
                + ((current_base_row[208usize]) * (node_125)))
                + ((current_base_row[205usize]) * (node_829)))
                + ((current_base_row[214usize]) * (node_927)))
                + ((current_base_row[216usize]) * (node_533)))
                + ((current_base_row[213usize]) * (node_228)))
                + ((current_base_row[220usize]) * (node_533)))
                + ((current_base_row[225usize]) * (node_541)))
                + ((current_base_row[223usize]) * (node_533)))
                + ((current_base_row[224usize]) * (node_533)))
                + ((current_base_row[215usize]) * (node_813)))
                + ((current_base_row[221usize]) * (node_501)))
                + ((current_base_row[228usize]) * (node_533)))
                + ((current_base_row[229usize]) * (node_533)))
                + ((current_base_row[233usize]) * (node_537)))
                + ((current_base_row[234usize]) * (node_537)))
                + ((current_base_row[235usize]) * (node_533)))
                + ((current_base_row[237usize]) * (node_184)))
                + ((current_base_row[236usize]) * (node_533)))
                + ((current_base_row[238usize]) * (node_541)))
                + ((current_base_row[239usize]) * (node_541)))
                + ((current_base_row[241usize]) * (node_541)))
                + ((current_base_row[242usize]) * (node_537)))
                + ((current_base_row[245usize]) * (node_541)))
                + ((current_base_row[248usize]) * (node_537)))
                + ((current_base_row[251usize]) * (node_537)))
                + ((current_base_row[277usize]) * (node_533)))
                + ((current_base_row[279usize]) * (node_537)))
                + ((current_base_row[259usize])
                    * ((node_817)
                        + (((next_ext_row[3usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((challenges[StandardInputIndeterminate])
                                    * (((challenges[StandardInputIndeterminate])
                                        * (((challenges[StandardInputIndeterminate])
                                            * (((challenges[StandardInputIndeterminate])
                                                * ((node_1372) + (next_base_row[26usize])))
                                                + (next_base_row[25usize]))) + (next_base_row[24usize])))
                                        + (next_base_row[23usize]))) + (next_base_row[22usize]))))
                            * (current_base_row[189usize])))))
                + ((current_base_row[263usize])
                    * ((node_505)
                        + (((next_ext_row[4usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((challenges[StandardOutputIndeterminate]) * (node_1435))
                                    + (current_base_row[26usize]))))
                            * (current_base_row[189usize]))))) * (node_4849))
                + ((node_159) * (next_base_row[8usize])),
            (((((((((((((((((((((((((((((current_base_row[202usize]) * (node_508))
                + ((current_base_row[212usize]) * (node_537)))
                + ((current_base_row[203usize]) * (node_820)))
                + ((current_base_row[208usize]) * (node_129)))
                + ((current_base_row[205usize]) * (node_831)))
                + ((current_base_row[214usize]) * (node_533)))
                + ((current_base_row[216usize]) * (node_537)))
                + ((current_base_row[213usize]) * (node_232)))
                + ((current_base_row[220usize]) * (node_537)))
                + ((current_base_row[223usize]) * (node_537)))
                + ((current_base_row[224usize]) * (node_537)))
                + ((current_base_row[215usize]) * (node_817)))
                + ((current_base_row[221usize]) * (node_505)))
                + ((current_base_row[228usize]) * (node_537)))
                + ((current_base_row[229usize]) * (node_537)))
                + ((current_base_row[233usize]) * (node_541)))
                + ((current_base_row[234usize]) * (node_541)))
                + ((current_base_row[235usize]) * (node_537)))
                + ((current_base_row[237usize]) * (node_533)))
                + ((current_base_row[236usize]) * (node_537)))
                + ((current_base_row[242usize]) * (node_541)))
                + ((current_base_row[248usize]) * (node_541)))
                + ((current_base_row[251usize]) * (node_541)))
                + ((current_base_row[277usize]) * (node_537)))
                + ((current_base_row[279usize]) * (node_541)))
                + ((current_base_row[259usize])
                    * ((node_820)
                        + (((next_ext_row[3usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((challenges[StandardInputIndeterminate])
                                    * (((challenges[StandardInputIndeterminate])
                                        * (((challenges[StandardInputIndeterminate])
                                            * ((node_1372) + (next_base_row[25usize])))
                                            + (next_base_row[24usize]))) + (next_base_row[23usize])))
                                    + (next_base_row[22usize]))))
                            * (current_base_row[188usize])))))
                + ((current_base_row[263usize])
                    * ((node_508)
                        + (((next_ext_row[4usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (node_1435))) * (current_base_row[188usize])))))
                * (node_4849)) + ((node_927) * (next_base_row[8usize])),
            (((((((((((((((((((((((current_base_row[202usize]) * (node_510))
                + ((current_base_row[212usize]) * (node_541)))
                + ((current_base_row[203usize]) * (node_822)))
                + ((current_base_row[208usize]) * (node_117)))
                + ((current_base_row[205usize]) * (node_833)))
                + ((current_base_row[214usize]) * (node_537)))
                + ((current_base_row[216usize]) * (node_541)))
                + ((current_base_row[213usize]) * (node_236)))
                + ((current_base_row[220usize]) * (node_541)))
                + ((current_base_row[223usize]) * (node_541)))
                + ((current_base_row[224usize]) * (node_541)))
                + ((current_base_row[215usize]) * (node_820)))
                + ((current_base_row[221usize]) * (node_508)))
                + ((current_base_row[228usize]) * (node_541)))
                + ((current_base_row[229usize]) * (node_541)))
                + ((current_base_row[235usize]) * (node_541)))
                + ((current_base_row[237usize]) * (node_537)))
                + ((current_base_row[236usize]) * (node_541)))
                + ((current_base_row[277usize]) * (node_541)))
                + ((current_base_row[259usize])
                    * ((node_822)
                        + (((next_ext_row[3usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((challenges[StandardInputIndeterminate])
                                    * (((challenges[StandardInputIndeterminate])
                                        * ((node_1372) + (next_base_row[24usize])))
                                        + (next_base_row[23usize]))) + (next_base_row[22usize]))))
                            * (current_base_row[187usize])))))
                + ((current_base_row[263usize])
                    * ((node_510)
                        + (((next_ext_row[4usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (node_1430))) * (current_base_row[187usize])))))
                * (node_4849)) + ((node_533) * (next_base_row[8usize])),
            (((((((((((((current_base_row[202usize]) * (node_270))
                + ((current_base_row[203usize]) * (node_589)))
                + ((current_base_row[208usize]) * (node_546)))
                + ((current_base_row[205usize]) * (node_835)))
                + ((current_base_row[214usize]) * (node_541)))
                + ((current_base_row[213usize]) * (node_240)))
                + ((current_base_row[215usize]) * (node_822)))
                + ((current_base_row[221usize]) * (node_510)))
                + ((current_base_row[237usize]) * (node_541)))
                + ((current_base_row[259usize])
                    * ((node_589)
                        + (((next_ext_row[3usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((challenges[StandardInputIndeterminate])
                                    * ((node_1372) + (next_base_row[23usize])))
                                    + (next_base_row[22usize]))))
                            * (current_base_row[186usize])))))
                + ((current_base_row[263usize])
                    * ((node_270)
                        + (((next_ext_row[4usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (node_1425))) * (current_base_row[186usize])))))
                * (node_4849)) + ((node_537) * (next_base_row[8usize])),
            (((((((((((current_base_row[202usize]) * (current_base_row[315usize]))
                + ((current_base_row[203usize]) * (current_base_row[315usize])))
                + ((current_base_row[208usize]) * (node_547)))
                + ((current_base_row[205usize]) * (node_837)))
                + ((current_base_row[213usize]) * (node_244)))
                + ((current_base_row[215usize]) * (node_589)))
                + ((current_base_row[221usize]) * (node_270)))
                + ((current_base_row[259usize])
                    * (((next_ext_row[3usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((node_1372) + (next_base_row[22usize]))))
                        * (current_base_row[185usize]))))
                + ((current_base_row[263usize])
                    * (((next_ext_row[4usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_1420))) * (current_base_row[185usize]))))
                * (node_4849)) + ((node_541) * (next_base_row[8usize])),
            ((((((((((current_base_row[202usize]) * (current_base_row[270usize]))
                + ((current_base_row[203usize]) * (current_base_row[270usize])))
                + ((current_base_row[208usize]) * (node_549)))
                + ((current_base_row[205usize]) * (node_841)))
                + ((current_base_row[213usize]) * (node_184)))
                + ((current_base_row[215usize]) * (current_base_row[312usize])))
                + ((current_base_row[221usize]) * (current_base_row[312usize])))
                + ((current_base_row[259usize]) * (current_base_row[312usize])))
                + ((current_base_row[263usize]) * (current_base_row[312usize])))
                * (node_4849),
            ((((((((((current_base_row[202usize]) * (current_base_row[271usize]))
                + ((current_base_row[203usize]) * (current_base_row[271usize])))
                + ((current_base_row[208usize]) * (node_550)))
                + ((current_base_row[205usize]) * (node_843)))
                + ((current_base_row[213usize]) * (node_533)))
                + ((current_base_row[215usize]) * (current_base_row[270usize])))
                + ((current_base_row[221usize]) * (current_base_row[270usize])))
                + ((current_base_row[259usize]) * (current_base_row[270usize])))
                + ((current_base_row[263usize]) * (current_base_row[270usize])))
                * (node_4849),
            ((((((((((current_base_row[202usize]) * (current_base_row[314usize]))
                + ((current_base_row[203usize]) * (current_base_row[314usize])))
                + ((current_base_row[208usize]) * (node_551)))
                + ((current_base_row[205usize]) * (node_845)))
                + ((current_base_row[213usize]) * (node_537)))
                + ((current_base_row[215usize]) * (current_base_row[271usize])))
                + ((current_base_row[221usize]) * (current_base_row[271usize])))
                + ((current_base_row[259usize]) * (current_base_row[271usize])))
                + ((current_base_row[263usize]) * (current_base_row[271usize])))
                * (node_4849),
            ((((((((((current_base_row[202usize]) * (current_base_row[317usize]))
                + ((current_base_row[203usize]) * (current_base_row[317usize])))
                + ((current_base_row[208usize]) * (node_552)))
                + ((current_base_row[205usize])
                    * (((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[185usize]))) * (node_868))))
                + ((current_base_row[213usize]) * (node_541)))
                + ((current_base_row[215usize]) * (current_base_row[314usize])))
                + ((current_base_row[221usize]) * (current_base_row[314usize])))
                + ((current_base_row[259usize]) * (current_base_row[314usize])))
                + ((current_base_row[263usize]) * (current_base_row[314usize])))
                * (node_4849),
            (((((((((current_base_row[202usize]) * (node_533))
                + ((current_base_row[203usize]) * (node_533)))
                + ((current_base_row[208usize]) * (node_558)))
                + ((current_base_row[205usize])
                    * (((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[270usize]))) * (node_892))))
                + ((current_base_row[215usize]) * (current_base_row[331usize])))
                + ((current_base_row[221usize]) * (current_base_row[331usize])))
                + ((current_base_row[259usize]) * (current_base_row[331usize])))
                + ((current_base_row[263usize]) * (current_base_row[331usize])))
                * (node_4849),
            (((((((((current_base_row[202usize]) * (node_537))
                + ((current_base_row[203usize]) * (node_537)))
                + ((current_base_row[208usize]) * (node_559)))
                + ((current_base_row[205usize])
                    * (((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[271usize]))) * (node_896))))
                + ((current_base_row[215usize]) * (node_537)))
                + ((current_base_row[221usize]) * (node_537)))
                + ((current_base_row[259usize]) * (node_533)))
                + ((current_base_row[263usize]) * (node_533))) * (node_4849),
            (((((((((current_base_row[202usize]) * (node_541))
                + ((current_base_row[203usize]) * (node_541)))
                + ((current_base_row[208usize]) * (node_560)))
                + ((current_base_row[205usize])
                    * (((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[314usize]))) * (node_900))))
                + ((current_base_row[215usize]) * (node_541)))
                + ((current_base_row[221usize]) * (node_541)))
                + ((current_base_row[259usize]) * (node_541)))
                + ((current_base_row[263usize]) * (node_537))) * (node_4849),
            (((current_base_row[208usize]) * (node_572))
                + ((current_base_row[205usize])
                    * (((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[318usize]))) * (node_908))))
                * (node_4849),
            (((current_base_row[208usize]) * (node_533))
                + ((current_base_row[205usize])
                    * (((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[273usize]))) * (node_912))))
                * (node_4849),
            (((current_base_row[208usize]) * (node_537))
                + ((current_base_row[205usize])
                    * (((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[329usize]))) * (node_916))))
                * (node_4849),
            (((current_base_row[208usize]) * (node_541))
                + ((current_base_row[205usize])
                    * (((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[330usize]))) * (node_920))))
                * (node_4849),
            ((current_base_row[205usize]) * (node_927)) * (node_4849),
            ((current_base_row[205usize]) * (node_533)) * (node_4849),
            ((current_base_row[205usize]) * (node_537)) * (node_4849),
            ((current_base_row[205usize]) * (node_541)) * (node_4849),
            (((next_ext_row[13usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (current_ext_row[13usize])))
                * ((challenges[ClockJumpDifferenceLookupIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[7usize]))))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (next_base_row[45usize])),
            ((node_4849)
                * (((node_5017)
                    * ((challenges[InstructionLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((((challenges[ProgramAddressWeight])
                                * (next_base_row[9usize]))
                                + ((challenges[ProgramInstructionWeight])
                                    * (next_base_row[10usize])))
                                + ((challenges[ProgramNextInstructionWeight])
                                    * (next_base_row[11usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((next_base_row[8usize]) * (node_5017)),
            (next_ext_row[8usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[8usize])
                        * ((challenges[JumpStackIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((((((challenges[JumpStackClkWeight])
                                    * (next_base_row[7usize]))
                                    + ((challenges[JumpStackCiWeight])
                                        * (next_base_row[10usize])))
                                    + ((challenges[JumpStackJspWeight])
                                        * (next_base_row[19usize])))
                                    + ((challenges[JumpStackJsoWeight])
                                        * (next_base_row[20usize])))
                                    + ((challenges[JumpStackJsdWeight])
                                        * (next_base_row[21usize]))))))),
            (((next_base_row[10usize])
                + (BFieldElement::from_raw_u64(18446743992105173011u64)))
                * ((next_ext_row[9usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_ext_row[9usize]))))
                + ((current_base_row[344usize])
                    * (((next_ext_row[9usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((challenges[HashInputIndeterminate])
                                * (current_ext_row[9usize])))) + (node_5106))),
            (((current_base_row[10usize])
                + (BFieldElement::from_raw_u64(18446743992105173011u64)))
                * ((next_ext_row[10usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_ext_row[10usize]))))
                + ((current_base_row[240usize])
                    * (((next_ext_row[10usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((challenges[HashDigestIndeterminate])
                                * (current_ext_row[10usize]))))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_5089)))),
            (((((((current_base_row[10usize])
                + (BFieldElement::from_raw_u64(18446743897615892521u64)))
                * ((current_base_row[10usize])
                    + (BFieldElement::from_raw_u64(18446743923385696291u64))))
                * ((current_base_row[10usize])
                    + (BFieldElement::from_raw_u64(18446743863256154161u64))))
                * ((next_ext_row[11usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_ext_row[11usize]))))
                + ((current_base_row[229usize]) * (node_5166)))
                + ((current_base_row[244usize])
                    * ((node_5166)
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((((((((challenges[HashStateWeight0])
                                * (current_base_row[22usize]))
                                + ((challenges[HashStateWeight1])
                                    * (current_base_row[23usize])))
                                + ((challenges[HashStateWeight2])
                                    * (current_base_row[24usize])))
                                + ((challenges[HashStateWeight3])
                                    * (current_base_row[25usize])))
                                + ((challenges[HashStateWeight4])
                                    * (current_base_row[26usize])))
                                + ((challenges[HashStateWeight5])
                                    * (current_base_row[27usize])))
                                + ((challenges[HashStateWeight6])
                                    * (current_base_row[28usize])))
                                + ((challenges[HashStateWeight7])
                                    * (current_base_row[29usize])))
                                + ((challenges[HashStateWeight8])
                                    * (current_base_row[30usize])))
                                + ((challenges[HashStateWeight9])
                                    * (current_base_row[31usize])))))))
                + ((current_base_row[246usize]) * ((node_5166) + (node_5106))),
            (((((((((current_base_row[236usize])
                * (((node_5229) * (((node_5186) + (node_5189)) + (node_5193)))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((current_base_row[238usize]) * (node_5233)))
                + ((current_base_row[239usize]) * (node_5233)))
                + ((current_base_row[241usize])
                    * (((node_5229)
                        * (((node_5200)
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((challenges[U32CiWeight])
                                    * (BFieldElement::from_raw_u64(60129542130u64)))))
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((challenges[U32ResultWeight])
                                    * (((current_base_row[22usize])
                                        + (current_base_row[23usize])) + (node_1314)))
                                    * (BFieldElement::from_raw_u64(9223372036854775808u64))))))
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)))))
                + ((current_base_row[245usize]) * (node_5233)))
                + ((current_base_row[242usize]) * (node_5237)))
                + ((current_base_row[248usize])
                    * (((((node_5229) * (node_5223)) * (node_5227))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_5223)))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_5227)))))
                + ((current_base_row[251usize]) * (node_5237)))
                + (((BFieldElement::from_raw_u64(4294967295u64))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[14usize]))) * (node_5229)),
            (((next_ext_row[14usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[14usize])
                        * ((challenges[OpStackIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((((challenges[OpStackClkWeight])
                                    * (next_base_row[46usize]))
                                    + ((challenges[OpStackIb1Weight])
                                        * (next_base_row[47usize])))
                                    + ((challenges[OpStackPointerWeight])
                                        * (next_base_row[48usize])))
                                    + ((challenges[OpStackFirstUnderflowElementWeight])
                                        * (next_base_row[49usize])))))))) * (node_5283))
                + (((next_ext_row[14usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_ext_row[14usize]))) * (node_5300)),
            ((((((node_5309)
                * ((challenges[ClockJumpDifferenceLookupIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((next_base_row[46usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (current_base_row[46usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64))) * (node_5277))
                * (node_5283)) + ((node_5309) * (node_5276)))
                + ((node_5309) * (node_5300)),
            ((node_5352)
                * ((next_ext_row[16usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((current_ext_row[16usize]) * (node_5371)))))
                + ((node_5355) * ((next_ext_row[16usize]) + (node_5376))),
            ((node_5352)
                * (((next_ext_row[17usize]) + (node_5376))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((node_5371) * (current_ext_row[17usize])))))
                + ((node_5355)
                    * ((next_ext_row[17usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_ext_row[17usize])))),
            ((node_5352)
                * (((next_ext_row[18usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((challenges[RamTableBezoutRelationIndeterminate])
                            * (current_ext_row[18usize]))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[55usize]))))
                + ((node_5355)
                    * ((next_ext_row[18usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_ext_row[18usize])))),
            ((node_5352)
                * (((next_ext_row[19usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((challenges[RamTableBezoutRelationIndeterminate])
                            * (current_ext_row[19usize]))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[56usize]))))
                + ((node_5355)
                    * ((next_ext_row[19usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_ext_row[19usize])))),
            (((next_ext_row[20usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[20usize])
                        * ((challenges[RamIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((((next_base_row[50usize]) * (challenges[RamClkWeight]))
                                    + ((next_base_row[52usize])
                                        * (challenges[RamPointerWeight])))
                                    + ((next_base_row[53usize]) * (challenges[RamValueWeight])))
                                    + ((next_base_row[51usize])
                                        * (challenges[RamInstructionTypeWeight]))))))))
                * (node_5346))
                + (((next_ext_row[20usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_ext_row[20usize]))) * (node_5423)),
            (((current_ext_row[80usize]) * (node_5346)) + ((node_5432) * (node_5352)))
                + ((node_5432) * (node_5423)),
            (next_ext_row[22usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[22usize])
                        * ((challenges[JumpStackIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((((((challenges[JumpStackClkWeight])
                                    * (next_base_row[57usize]))
                                    + ((challenges[JumpStackCiWeight])
                                        * (next_base_row[58usize])))
                                    + ((challenges[JumpStackJspWeight])
                                        * (next_base_row[59usize])))
                                    + ((challenges[JumpStackJsoWeight])
                                        * (next_base_row[60usize])))
                                    + ((challenges[JumpStackJsdWeight])
                                        * (next_base_row[61usize]))))))),
            ((node_5461)
                * (((node_5495)
                    * ((challenges[ClockJumpDifferenceLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_5473))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_5460) * (node_5495)),
            (((current_base_row[338usize])
                * (((next_ext_row[24usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((challenges[ProgramAttestationSendChunkIndeterminate])
                            * (current_ext_row[24usize]))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((((((((((((((((((((challenges[ProgramAttestationPrepareChunkIndeterminate])
                            + (node_6264))
                            * (challenges[ProgramAttestationPrepareChunkIndeterminate]))
                            + (node_6275))
                            * (challenges[ProgramAttestationPrepareChunkIndeterminate]))
                            + (node_6286))
                            * (challenges[ProgramAttestationPrepareChunkIndeterminate]))
                            + (node_6297))
                            * (challenges[ProgramAttestationPrepareChunkIndeterminate]))
                            + (next_base_row[97usize]))
                            * (challenges[ProgramAttestationPrepareChunkIndeterminate]))
                            + (next_base_row[98usize]))
                            * (challenges[ProgramAttestationPrepareChunkIndeterminate]))
                            + (next_base_row[99usize]))
                            * (challenges[ProgramAttestationPrepareChunkIndeterminate]))
                            + (next_base_row[100usize]))
                            * (challenges[ProgramAttestationPrepareChunkIndeterminate]))
                            + (next_base_row[101usize]))
                            * (challenges[ProgramAttestationPrepareChunkIndeterminate]))
                            + (next_base_row[102usize])))))
                + ((next_base_row[64usize]) * (node_6615)))
                + ((node_6398) * (node_6615)),
            ((current_base_row[339usize]) * (node_6398))
                * (((((((((((challenges[CompressProgramDigestIndeterminate])
                    + (node_5532)) * (challenges[CompressProgramDigestIndeterminate]))
                    + (node_5543)) * (challenges[CompressProgramDigestIndeterminate]))
                    + (node_5554)) * (challenges[CompressProgramDigestIndeterminate]))
                    + (node_5565)) * (challenges[CompressProgramDigestIndeterminate]))
                    + (current_base_row[97usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (challenges[CompressedProgramDigest]))),
            (current_base_row[352usize])
                * ((((((node_6443) + (node_6444)) + (node_6446)) + (node_6448))
                    + (node_6450)) + (node_6452)),
            (current_base_row[353usize])
                * (((((((((((((((((challenges[HashStateWeight0])
                    * ((node_6264)
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_5532))))
                    + ((challenges[HashStateWeight1])
                        * ((node_6275)
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (node_5543)))))
                    + ((challenges[HashStateWeight2])
                        * ((node_6286)
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (node_5554)))))
                    + ((challenges[HashStateWeight3])
                        * ((node_6297)
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (node_5565)))))
                    + ((challenges[HashStateWeight4])
                        * ((next_base_row[97usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (current_base_row[97usize])))))
                    + ((challenges[HashStateWeight5])
                        * ((next_base_row[98usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (current_base_row[98usize])))))
                    + ((challenges[HashStateWeight6])
                        * ((next_base_row[99usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (current_base_row[99usize])))))
                    + ((challenges[HashStateWeight7])
                        * ((next_base_row[100usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (current_base_row[100usize])))))
                    + ((challenges[HashStateWeight8])
                        * ((next_base_row[101usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (current_base_row[101usize])))))
                    + ((challenges[HashStateWeight9])
                        * ((next_base_row[102usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (current_base_row[102usize]))))) + (node_6443))
                    + (node_6444)) + (node_6446)) + (node_6448)) + (node_6450))
                    + (node_6452)),
            ((((current_base_row[311usize]) * (current_base_row[316usize]))
                * (((next_ext_row[25usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((challenges[HashInputIndeterminate])
                            * (current_ext_row[25usize])))) + (node_6544)))
                + ((next_base_row[64usize]) * (node_6521)))
                + ((node_6404) * (node_6521)),
            ((((node_6564) * (current_base_row[316usize]))
                * (((next_ext_row[26usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((challenges[HashDigestIndeterminate])
                            * (current_ext_row[26usize]))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (node_6530)))) + ((node_6462) * (node_6555)))
                + ((node_6404) * (node_6555)),
            ((((current_base_row[311usize]) * (node_6513))
                * ((((next_ext_row[27usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((challenges[SpongeIndeterminate])
                            * (current_ext_row[27usize]))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((challenges[HashCIWeight]) * (next_base_row[63usize]))))
                    + (node_6544))) + ((next_base_row[64usize]) * (node_6581)))
                + ((((node_6408) * (node_6516)) * (node_6584)) * (node_6581)),
            (((current_base_row[278usize])
                * (((node_6635)
                    * ((challenges[HashCascadeLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[HashCascadeLookInWeight])
                                * (next_base_row[65usize]))
                                + ((challenges[HashCascadeLookOutWeight])
                                    * (next_base_row[81usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_6564) * (node_6635))) + ((node_6643) * (node_6635)),
            (((current_base_row[278usize])
                * (((node_6656)
                    * ((challenges[HashCascadeLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[HashCascadeLookInWeight])
                                * (next_base_row[66usize]))
                                + ((challenges[HashCascadeLookOutWeight])
                                    * (next_base_row[82usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_6564) * (node_6656))) + ((node_6643) * (node_6656)),
            (((current_base_row[278usize])
                * (((node_6673)
                    * ((challenges[HashCascadeLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[HashCascadeLookInWeight])
                                * (next_base_row[67usize]))
                                + ((challenges[HashCascadeLookOutWeight])
                                    * (next_base_row[83usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_6564) * (node_6673))) + ((node_6643) * (node_6673)),
            (((current_base_row[278usize])
                * (((node_6690)
                    * ((challenges[HashCascadeLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[HashCascadeLookInWeight])
                                * (next_base_row[68usize]))
                                + ((challenges[HashCascadeLookOutWeight])
                                    * (next_base_row[84usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_6564) * (node_6690))) + ((node_6643) * (node_6690)),
            (((current_base_row[278usize])
                * (((node_6707)
                    * ((challenges[HashCascadeLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[HashCascadeLookInWeight])
                                * (next_base_row[69usize]))
                                + ((challenges[HashCascadeLookOutWeight])
                                    * (next_base_row[85usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_6564) * (node_6707))) + ((node_6643) * (node_6707)),
            (((current_base_row[278usize])
                * (((node_6724)
                    * ((challenges[HashCascadeLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[HashCascadeLookInWeight])
                                * (next_base_row[70usize]))
                                + ((challenges[HashCascadeLookOutWeight])
                                    * (next_base_row[86usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_6564) * (node_6724))) + ((node_6643) * (node_6724)),
            (((current_base_row[278usize])
                * (((node_6741)
                    * ((challenges[HashCascadeLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[HashCascadeLookInWeight])
                                * (next_base_row[71usize]))
                                + ((challenges[HashCascadeLookOutWeight])
                                    * (next_base_row[87usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_6564) * (node_6741))) + ((node_6643) * (node_6741)),
            (((current_base_row[278usize])
                * (((node_6758)
                    * ((challenges[HashCascadeLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[HashCascadeLookInWeight])
                                * (next_base_row[72usize]))
                                + ((challenges[HashCascadeLookOutWeight])
                                    * (next_base_row[88usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_6564) * (node_6758))) + ((node_6643) * (node_6758)),
            (((current_base_row[278usize])
                * (((node_6775)
                    * ((challenges[HashCascadeLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[HashCascadeLookInWeight])
                                * (next_base_row[73usize]))
                                + ((challenges[HashCascadeLookOutWeight])
                                    * (next_base_row[89usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_6564) * (node_6775))) + ((node_6643) * (node_6775)),
            (((current_base_row[278usize])
                * (((node_6792)
                    * ((challenges[HashCascadeLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[HashCascadeLookInWeight])
                                * (next_base_row[74usize]))
                                + ((challenges[HashCascadeLookOutWeight])
                                    * (next_base_row[90usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_6564) * (node_6792))) + ((node_6643) * (node_6792)),
            (((current_base_row[278usize])
                * (((node_6809)
                    * ((challenges[HashCascadeLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[HashCascadeLookInWeight])
                                * (next_base_row[75usize]))
                                + ((challenges[HashCascadeLookOutWeight])
                                    * (next_base_row[91usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_6564) * (node_6809))) + ((node_6643) * (node_6809)),
            (((current_base_row[278usize])
                * (((node_6826)
                    * ((challenges[HashCascadeLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[HashCascadeLookInWeight])
                                * (next_base_row[76usize]))
                                + ((challenges[HashCascadeLookOutWeight])
                                    * (next_base_row[92usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_6564) * (node_6826))) + ((node_6643) * (node_6826)),
            (((current_base_row[278usize])
                * (((node_6843)
                    * ((challenges[HashCascadeLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[HashCascadeLookInWeight])
                                * (next_base_row[77usize]))
                                + ((challenges[HashCascadeLookOutWeight])
                                    * (next_base_row[93usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_6564) * (node_6843))) + ((node_6643) * (node_6843)),
            (((current_base_row[278usize])
                * (((node_6860)
                    * ((challenges[HashCascadeLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[HashCascadeLookInWeight])
                                * (next_base_row[78usize]))
                                + ((challenges[HashCascadeLookOutWeight])
                                    * (next_base_row[94usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_6564) * (node_6860))) + ((node_6643) * (node_6860)),
            (((current_base_row[278usize])
                * (((node_6877)
                    * ((challenges[HashCascadeLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[HashCascadeLookInWeight])
                                * (next_base_row[79usize]))
                                + ((challenges[HashCascadeLookOutWeight])
                                    * (next_base_row[95usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_6564) * (node_6877))) + ((node_6643) * (node_6877)),
            (((current_base_row[278usize])
                * (((node_6894)
                    * ((challenges[HashCascadeLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[HashCascadeLookInWeight])
                                * (next_base_row[80usize]))
                                + ((challenges[HashCascadeLookOutWeight])
                                    * (next_base_row[96usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_6564) * (node_6894))) + ((node_6643) * (node_6894)),
            ((node_6920)
                * (((node_6930)
                    * ((challenges[HashCascadeLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[HashCascadeLookInWeight])
                                * (((BFieldElement::from_raw_u64(1099511627520u64))
                                    * (next_base_row[130usize])) + (next_base_row[131usize])))
                                + ((challenges[HashCascadeLookOutWeight])
                                    * (((BFieldElement::from_raw_u64(1099511627520u64))
                                        * (next_base_row[132usize]))
                                        + (next_base_row[133usize])))))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[134usize]))))
                + ((next_base_row[129usize]) * (node_6930)),
            ((node_6920)
                * ((((((node_6946)
                    * ((challenges[CascadeLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_6941))))
                    * ((challenges[CascadeLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_6944))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((BFieldElement::from_raw_u64(8589934590u64))
                            * (challenges[CascadeLookupIndeterminate])))) + (node_6941))
                    + (node_6944))) + ((next_base_row[129usize]) * (node_6946)),
            ((node_6972)
                * (((node_6984)
                    * ((challenges[CascadeLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((next_base_row[136usize])
                                * (challenges[LookupTableInputWeight]))
                                + ((next_base_row[137usize])
                                    * (challenges[LookupTableOutputWeight]))))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[138usize]))))
                + ((next_base_row[135usize]) * (node_6984)),
            ((node_6972)
                * (((next_ext_row[47usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((current_ext_row[47usize])
                            * (challenges[LookupTablePublicIndeterminate]))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[137usize]))))
                + ((next_base_row[135usize])
                    * ((next_ext_row[47usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_ext_row[47usize])))),
            (node_7032) * (node_7149),
            (next_base_row[139usize])
                * (((node_7149)
                    * ((challenges[U32Indeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((challenges[U32CiWeight]) * (next_base_row[142usize]))
                                + ((challenges[U32LhsWeight]) * (next_base_row[143usize])))
                                + ((challenges[U32RhsWeight]) * (next_base_row[145usize])))
                                + ((challenges[U32ResultWeight])
                                    * (next_base_row[147usize]))))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[148usize]))),
            (current_ext_row[50usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_178)
                        * ((challenges[OpStackIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_169)
                                    + ((challenges[OpStackPointerWeight])
                                        * ((next_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(4294967295u64)))))
                                    + ((challenges[OpStackFirstUnderflowElementWeight])
                                        * (next_base_row[36usize]))))))),
            (current_ext_row[51usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_568)
                        * ((challenges[OpStackIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_169)
                                    + ((challenges[OpStackPointerWeight])
                                        * ((current_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(4294967295u64)))))
                                    + ((challenges[OpStackFirstUnderflowElementWeight])
                                        * (current_base_row[36usize]))))))),
            (current_ext_row[52usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[50usize])
                        * ((challenges[OpStackIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_169)
                                    + ((challenges[OpStackPointerWeight])
                                        * ((next_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(8589934590u64)))))
                                    + ((challenges[OpStackFirstUnderflowElementWeight])
                                        * (next_base_row[35usize]))))))),
            (current_ext_row[53usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[51usize])
                        * ((challenges[OpStackIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_169)
                                    + ((challenges[OpStackPointerWeight])
                                        * ((current_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(8589934590u64)))))
                                    + ((challenges[OpStackFirstUnderflowElementWeight])
                                        * (current_base_row[35usize]))))))),
            (current_ext_row[54usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[52usize])
                        * ((challenges[OpStackIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_169)
                                    + ((challenges[OpStackPointerWeight])
                                        * ((next_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(12884901885u64)))))
                                    + ((challenges[OpStackFirstUnderflowElementWeight])
                                        * (next_base_row[34usize]))))))),
            (current_ext_row[55usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[53usize])
                        * ((challenges[OpStackIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_169)
                                    + ((challenges[OpStackPointerWeight])
                                        * ((current_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(12884901885u64)))))
                                    + ((challenges[OpStackFirstUnderflowElementWeight])
                                        * (current_base_row[34usize]))))))),
            (current_ext_row[56usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[54usize])
                        * ((challenges[OpStackIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_169)
                                    + ((challenges[OpStackPointerWeight])
                                        * ((next_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(17179869180u64)))))
                                    + ((challenges[OpStackFirstUnderflowElementWeight])
                                        * (next_base_row[33usize]))))))),
            (current_ext_row[57usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_1006)
                        * ((challenges[RamIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_997)
                                    + (((next_base_row[22usize])
                                        + (BFieldElement::from_raw_u64(8589934590u64)))
                                        * (challenges[RamPointerWeight])))
                                    + ((next_base_row[24usize])
                                        * (challenges[RamValueWeight]))))))),
            (current_ext_row[58usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_1088)
                        * ((challenges[RamIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_994)
                                    + (((current_base_row[22usize])
                                        + (BFieldElement::from_raw_u64(4294967295u64)))
                                        * (challenges[RamPointerWeight])))
                                    + ((current_base_row[24usize])
                                        * (challenges[RamValueWeight]))))))),
            (current_ext_row[59usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[6usize]) * (current_ext_row[56usize]))),
            (current_ext_row[60usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[55usize])
                        * ((challenges[OpStackIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_169)
                                    + ((challenges[OpStackPointerWeight])
                                        * ((current_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(17179869180u64)))))
                                    + ((challenges[OpStackFirstUnderflowElementWeight])
                                        * (current_base_row[33usize]))))))),
            (current_ext_row[61usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[57usize])
                        * ((challenges[RamIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_997)
                                    + (((next_base_row[22usize])
                                        + (BFieldElement::from_raw_u64(12884901885u64)))
                                        * (challenges[RamPointerWeight])))
                                    + ((next_base_row[25usize])
                                        * (challenges[RamValueWeight]))))))),
            (current_ext_row[62usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[58usize])
                        * ((challenges[RamIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_994)
                                    + (((current_base_row[22usize])
                                        + (BFieldElement::from_raw_u64(8589934590u64)))
                                        * (challenges[RamPointerWeight])))
                                    + ((current_base_row[25usize])
                                        * (challenges[RamValueWeight]))))))),
            (current_ext_row[63usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[61usize])
                        * ((challenges[RamIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_997)
                                    + (((next_base_row[22usize])
                                        + (BFieldElement::from_raw_u64(17179869180u64)))
                                        * (challenges[RamPointerWeight])))
                                    + ((next_base_row[26usize])
                                        * (challenges[RamValueWeight]))))))),
            (current_ext_row[64usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[62usize])
                        * ((challenges[RamIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_994)
                                    + (((current_base_row[22usize])
                                        + (BFieldElement::from_raw_u64(12884901885u64)))
                                        * (challenges[RamPointerWeight])))
                                    + ((current_base_row[26usize])
                                        * (challenges[RamValueWeight]))))))),
            (current_ext_row[65usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[189usize])
                        * ((next_ext_row[7usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((current_ext_row[7usize])
                                    * ((current_ext_row[63usize])
                                        * ((challenges[RamIndeterminate])
                                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                                * (((node_997)
                                                    + (((next_base_row[22usize])
                                                        + (BFieldElement::from_raw_u64(21474836475u64)))
                                                        * (challenges[RamPointerWeight])))
                                                    + ((next_base_row[27usize])
                                                        * (challenges[RamValueWeight]))))))))))),
            (current_ext_row[66usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[189usize])
                        * ((next_ext_row[7usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((current_ext_row[7usize])
                                    * ((current_ext_row[64usize])
                                        * ((challenges[RamIndeterminate])
                                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                                * (((node_994)
                                                    + (((current_base_row[22usize])
                                                        + (BFieldElement::from_raw_u64(17179869180u64)))
                                                        * (challenges[RamPointerWeight])))
                                                    + ((current_base_row[27usize])
                                                        * (challenges[RamValueWeight]))))))))))),
            (current_ext_row[67usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_ext_row[56usize])
                        * ((challenges[OpStackIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_169)
                                    + ((challenges[OpStackPointerWeight])
                                        * ((next_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(21474836475u64)))))
                                    + ((challenges[OpStackFirstUnderflowElementWeight])
                                        * (next_base_row[32usize]))))))
                        * ((challenges[OpStackIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_169)
                                    + ((challenges[OpStackPointerWeight])
                                        * ((next_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(25769803770u64)))))
                                    + ((challenges[OpStackFirstUnderflowElementWeight])
                                        * (next_base_row[31usize]))))))
                        * ((challenges[OpStackIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_169)
                                    + ((challenges[OpStackPointerWeight])
                                        * ((next_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(30064771065u64)))))
                                    + ((challenges[OpStackFirstUnderflowElementWeight])
                                        * (next_base_row[30usize]))))))),
            (current_ext_row[68usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_ext_row[60usize])
                        * ((challenges[OpStackIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_169)
                                    + ((challenges[OpStackPointerWeight])
                                        * ((current_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(21474836475u64)))))
                                    + ((challenges[OpStackFirstUnderflowElementWeight])
                                        * (current_base_row[32usize]))))))
                        * ((challenges[OpStackIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_169)
                                    + ((challenges[OpStackPointerWeight])
                                        * ((current_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(25769803770u64)))))
                                    + ((challenges[OpStackFirstUnderflowElementWeight])
                                        * (current_base_row[31usize]))))))
                        * ((challenges[OpStackIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_169)
                                    + ((challenges[OpStackPointerWeight])
                                        * ((current_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(30064771065u64)))))
                                    + ((challenges[OpStackFirstUnderflowElementWeight])
                                        * (current_base_row[30usize]))))))),
            (current_ext_row[69usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[6usize])
                        * (((current_ext_row[67usize])
                            * ((challenges[OpStackIndeterminate])
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (((node_169)
                                        + ((challenges[OpStackPointerWeight])
                                            * ((next_base_row[38usize])
                                                + (BFieldElement::from_raw_u64(34359738360u64)))))
                                        + ((challenges[OpStackFirstUnderflowElementWeight])
                                            * (next_base_row[29usize]))))))
                            * ((challenges[OpStackIndeterminate])
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (((node_169)
                                        + ((challenges[OpStackPointerWeight])
                                            * ((next_base_row[38usize])
                                                + (BFieldElement::from_raw_u64(38654705655u64)))))
                                        + ((challenges[OpStackFirstUnderflowElementWeight])
                                            * (next_base_row[28usize])))))))),
            (current_ext_row[70usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[6usize])
                        * (((current_ext_row[68usize])
                            * ((challenges[OpStackIndeterminate])
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (((node_169)
                                        + ((challenges[OpStackPointerWeight])
                                            * ((current_base_row[38usize])
                                                + (BFieldElement::from_raw_u64(34359738360u64)))))
                                        + ((challenges[OpStackFirstUnderflowElementWeight])
                                            * (current_base_row[29usize]))))))
                            * ((challenges[OpStackIndeterminate])
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (((node_169)
                                        + ((challenges[OpStackPointerWeight])
                                            * ((current_base_row[38usize])
                                                + (BFieldElement::from_raw_u64(38654705655u64)))))
                                        + ((challenges[OpStackFirstUnderflowElementWeight])
                                            * (current_base_row[28usize])))))))),
            (current_ext_row[71usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[6usize]) * (current_ext_row[52usize]))),
            (current_ext_row[72usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[6usize]) * (current_ext_row[60usize]))),
            (current_ext_row[73usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[6usize]) * (node_178))),
            (current_ext_row[74usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[186usize])
                        * ((next_ext_row[6usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((current_ext_row[6usize])
                                    * (current_ext_row[50usize])))))),
            (current_ext_row[75usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[188usize])
                        * ((next_ext_row[6usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((current_ext_row[6usize])
                                    * (current_ext_row[54usize])))))),
            (current_ext_row[76usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[185usize]) * (node_572))),
            (current_ext_row[77usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[186usize])
                        * ((next_ext_row[6usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((current_ext_row[6usize])
                                    * (current_ext_row[51usize])))))),
            (current_ext_row[78usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[187usize])
                        * ((next_ext_row[6usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((current_ext_row[6usize])
                                    * (current_ext_row[53usize])))))),
            (current_ext_row[79usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[188usize])
                        * ((next_ext_row[6usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((current_ext_row[6usize])
                                    * (current_ext_row[55usize])))))),
            (current_ext_row[80usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((node_5432)
                        * ((challenges[ClockJumpDifferenceLookupIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((next_base_row[50usize])
                                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                        * (current_base_row[50usize]))))))
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                        * (node_5355))),
            (current_ext_row[81usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[215usize])
                        * ((((((current_base_row[185usize])
                            * ((next_ext_row[7usize])
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * ((current_ext_row[7usize]) * (node_1006)))))
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
                            + ((current_base_row[188usize])
                                * ((next_ext_row[7usize])
                                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                        * ((current_ext_row[7usize])
                                            * (current_ext_row[63usize]))))))
                            + (current_ext_row[65usize])))),
            (current_ext_row[82usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[221usize])
                        * ((((((current_base_row[185usize])
                            * ((next_ext_row[7usize])
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * ((current_ext_row[7usize]) * (node_1088)))))
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
                            + ((current_base_row[188usize])
                                * ((next_ext_row[7usize])
                                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                        * ((current_ext_row[7usize])
                                            * (current_ext_row[64usize]))))))
                            + (current_ext_row[66usize])))),
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
                * (((((((((((challenges[CompressProgramDigestIndeterminate])
                    + ((((((base_row[65usize])
                        * (BFieldElement::from_raw_u64(18446744069414518785u64)))
                        + ((base_row[66usize])
                            * (BFieldElement::from_raw_u64(18446744069414584320u64))))
                        + ((base_row[67usize])
                            * (BFieldElement::from_raw_u64(281474976645120u64))))
                        + (base_row[68usize])) * (BFieldElement::from_raw_u64(1u64))))
                    * (challenges[CompressProgramDigestIndeterminate]))
                    + ((((((base_row[69usize])
                        * (BFieldElement::from_raw_u64(18446744069414518785u64)))
                        + ((base_row[70usize])
                            * (BFieldElement::from_raw_u64(18446744069414584320u64))))
                        + ((base_row[71usize])
                            * (BFieldElement::from_raw_u64(281474976645120u64))))
                        + (base_row[72usize])) * (BFieldElement::from_raw_u64(1u64))))
                    * (challenges[CompressProgramDigestIndeterminate]))
                    + ((((((base_row[73usize])
                        * (BFieldElement::from_raw_u64(18446744069414518785u64)))
                        + ((base_row[74usize])
                            * (BFieldElement::from_raw_u64(18446744069414584320u64))))
                        + ((base_row[75usize])
                            * (BFieldElement::from_raw_u64(281474976645120u64))))
                        + (base_row[76usize])) * (BFieldElement::from_raw_u64(1u64))))
                    * (challenges[CompressProgramDigestIndeterminate]))
                    + ((((((base_row[77usize])
                        * (BFieldElement::from_raw_u64(18446744069414518785u64)))
                        + ((base_row[78usize])
                            * (BFieldElement::from_raw_u64(18446744069414584320u64))))
                        + ((base_row[79usize])
                            * (BFieldElement::from_raw_u64(281474976645120u64))))
                        + (base_row[80usize])) * (BFieldElement::from_raw_u64(1u64))))
                    * (challenges[CompressProgramDigestIndeterminate]))
                    + (base_row[97usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (challenges[CompressedProgramDigest]))),
            (ext_row[47usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (challenges[LookupTablePublicTerminal])),
            (ext_row[2usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[24usize])),
            (challenges[StandardInputTerminal])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[3usize])),
            (ext_row[4usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (challenges[StandardOutputTerminal])),
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
        let node_503 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (base_row[129usize]));
        let node_509 = ((challenges[LookupTableInputWeight]) * (base_row[131usize]))
            + ((challenges[LookupTableOutputWeight]) * (base_row[133usize]));
        let node_512 = ((challenges[LookupTableInputWeight]) * (base_row[130usize]))
            + ((challenges[LookupTableOutputWeight]) * (base_row[132usize]));
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
                    * (challenges[ProgramAttestationPrepareChunkIndeterminate])))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (base_row[1usize])),
            (ext_row[2usize]) + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((((((((((challenges[CompressProgramDigestIndeterminate])
                + (base_row[33usize]))
                * (challenges[CompressProgramDigestIndeterminate]))
                + (base_row[34usize]))
                * (challenges[CompressProgramDigestIndeterminate]))
                + (base_row[35usize]))
                * (challenges[CompressProgramDigestIndeterminate]))
                + (base_row[36usize]))
                * (challenges[CompressProgramDigestIndeterminate]))
                + (base_row[37usize]))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (challenges[CompressedProgramDigest])),
            (ext_row[3usize]) + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[5usize])
                * ((challenges[InstructionLookupIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[ProgramInstructionWeight]) * (base_row[10usize]))
                            + ((challenges[ProgramNextInstructionWeight])
                                * (base_row[11usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            (ext_row[4usize]) + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            (ext_row[6usize]) + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            (ext_row[7usize]) + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            (ext_row[8usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((challenges[JumpStackIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((challenges[JumpStackCiWeight]) * (base_row[10usize]))))),
            ext_row[13usize],
            (((base_row[10usize])
                + (BFieldElement::from_raw_u64(18446743992105173011u64)))
                * ((ext_row[9usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((base_row[150usize])
                    * ((ext_row[9usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (challenges[HashInputIndeterminate])))),
            (ext_row[10usize]) + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            (ext_row[11usize]) + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ext_row[12usize],
            (((ext_row[14usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((challenges[OpStackIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((challenges[OpStackClkWeight]) * (base_row[46usize]))
                                + ((challenges[OpStackIb1Weight]) * (base_row[47usize])))
                                + ((challenges[OpStackPointerWeight])
                                    * (BFieldElement::from_raw_u64(68719476720u64))))
                                + ((challenges[OpStackFirstUnderflowElementWeight])
                                    * (base_row[49usize])))))))
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
                    * (challenges[RamTableBezoutRelationIndeterminate])))
                + (base_row[52usize]),
            (ext_row[17usize]) + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((((ext_row[20usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (challenges[RamIndeterminate])))
                + (((((base_row[50usize]) * (challenges[RamClkWeight]))
                    + ((base_row[51usize]) * (challenges[RamInstructionTypeWeight])))
                    + ((base_row[52usize]) * (challenges[RamPointerWeight])))
                    + ((base_row[53usize]) * (challenges[RamValueWeight]))))
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
                    * ((challenges[JumpStackIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((challenges[JumpStackCiWeight]) * (base_row[58usize]))))),
            ext_row[23usize],
            (ext_row[25usize]) + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            (ext_row[26usize]) + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            (ext_row[27usize]) + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[24usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (challenges[ProgramAttestationSendChunkIndeterminate])))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((((((((((((((((((challenges[ProgramAttestationPrepareChunkIndeterminate])
                        + ((((((base_row[65usize])
                            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
                            + ((base_row[66usize])
                                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
                            + ((base_row[67usize])
                                * (BFieldElement::from_raw_u64(281474976645120u64))))
                            + (base_row[68usize]))
                            * (BFieldElement::from_raw_u64(1u64))))
                        * (challenges[ProgramAttestationPrepareChunkIndeterminate]))
                        + ((((((base_row[69usize])
                            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
                            + ((base_row[70usize])
                                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
                            + ((base_row[71usize])
                                * (BFieldElement::from_raw_u64(281474976645120u64))))
                            + (base_row[72usize]))
                            * (BFieldElement::from_raw_u64(1u64))))
                        * (challenges[ProgramAttestationPrepareChunkIndeterminate]))
                        + ((((((base_row[73usize])
                            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
                            + ((base_row[74usize])
                                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
                            + ((base_row[75usize])
                                * (BFieldElement::from_raw_u64(281474976645120u64))))
                            + (base_row[76usize]))
                            * (BFieldElement::from_raw_u64(1u64))))
                        * (challenges[ProgramAttestationPrepareChunkIndeterminate]))
                        + ((((((base_row[77usize])
                            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
                            + ((base_row[78usize])
                                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
                            + ((base_row[79usize])
                                * (BFieldElement::from_raw_u64(281474976645120u64))))
                            + (base_row[80usize]))
                            * (BFieldElement::from_raw_u64(1u64))))
                        * (challenges[ProgramAttestationPrepareChunkIndeterminate]))
                        + (base_row[97usize]))
                        * (challenges[ProgramAttestationPrepareChunkIndeterminate]))
                        + (base_row[98usize]))
                        * (challenges[ProgramAttestationPrepareChunkIndeterminate]))
                        + (base_row[99usize]))
                        * (challenges[ProgramAttestationPrepareChunkIndeterminate]))
                        + (base_row[100usize]))
                        * (challenges[ProgramAttestationPrepareChunkIndeterminate]))
                        + (base_row[101usize]))
                        * (challenges[ProgramAttestationPrepareChunkIndeterminate]))
                        + (base_row[102usize]))),
            ((ext_row[28usize])
                * ((challenges[HashCascadeLookupIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[HashCascadeLookInWeight]) * (base_row[65usize]))
                            + ((challenges[HashCascadeLookOutWeight])
                                * (base_row[81usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[29usize])
                * ((challenges[HashCascadeLookupIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[HashCascadeLookInWeight]) * (base_row[66usize]))
                            + ((challenges[HashCascadeLookOutWeight])
                                * (base_row[82usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[30usize])
                * ((challenges[HashCascadeLookupIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[HashCascadeLookInWeight]) * (base_row[67usize]))
                            + ((challenges[HashCascadeLookOutWeight])
                                * (base_row[83usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[31usize])
                * ((challenges[HashCascadeLookupIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[HashCascadeLookInWeight]) * (base_row[68usize]))
                            + ((challenges[HashCascadeLookOutWeight])
                                * (base_row[84usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[32usize])
                * ((challenges[HashCascadeLookupIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[HashCascadeLookInWeight]) * (base_row[69usize]))
                            + ((challenges[HashCascadeLookOutWeight])
                                * (base_row[85usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[33usize])
                * ((challenges[HashCascadeLookupIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[HashCascadeLookInWeight]) * (base_row[70usize]))
                            + ((challenges[HashCascadeLookOutWeight])
                                * (base_row[86usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[34usize])
                * ((challenges[HashCascadeLookupIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[HashCascadeLookInWeight]) * (base_row[71usize]))
                            + ((challenges[HashCascadeLookOutWeight])
                                * (base_row[87usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[35usize])
                * ((challenges[HashCascadeLookupIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[HashCascadeLookInWeight]) * (base_row[72usize]))
                            + ((challenges[HashCascadeLookOutWeight])
                                * (base_row[88usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[36usize])
                * ((challenges[HashCascadeLookupIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[HashCascadeLookInWeight]) * (base_row[73usize]))
                            + ((challenges[HashCascadeLookOutWeight])
                                * (base_row[89usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[37usize])
                * ((challenges[HashCascadeLookupIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[HashCascadeLookInWeight]) * (base_row[74usize]))
                            + ((challenges[HashCascadeLookOutWeight])
                                * (base_row[90usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[38usize])
                * ((challenges[HashCascadeLookupIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[HashCascadeLookInWeight]) * (base_row[75usize]))
                            + ((challenges[HashCascadeLookOutWeight])
                                * (base_row[91usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[39usize])
                * ((challenges[HashCascadeLookupIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[HashCascadeLookInWeight]) * (base_row[76usize]))
                            + ((challenges[HashCascadeLookOutWeight])
                                * (base_row[92usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[40usize])
                * ((challenges[HashCascadeLookupIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[HashCascadeLookInWeight]) * (base_row[77usize]))
                            + ((challenges[HashCascadeLookOutWeight])
                                * (base_row[93usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[41usize])
                * ((challenges[HashCascadeLookupIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[HashCascadeLookInWeight]) * (base_row[78usize]))
                            + ((challenges[HashCascadeLookOutWeight])
                                * (base_row[94usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[42usize])
                * ((challenges[HashCascadeLookupIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[HashCascadeLookInWeight]) * (base_row[79usize]))
                            + ((challenges[HashCascadeLookOutWeight])
                                * (base_row[95usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((ext_row[43usize])
                * ((challenges[HashCascadeLookupIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((challenges[HashCascadeLookInWeight]) * (base_row[80usize]))
                            + ((challenges[HashCascadeLookOutWeight])
                                * (base_row[96usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            ((node_503)
                * (((ext_row[44usize])
                    * ((challenges[HashCascadeLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[HashCascadeLookInWeight])
                                * (((BFieldElement::from_raw_u64(1099511627520u64))
                                    * (base_row[130usize])) + (base_row[131usize])))
                                + ((challenges[HashCascadeLookOutWeight])
                                    * (((BFieldElement::from_raw_u64(1099511627520u64))
                                        * (base_row[132usize])) + (base_row[133usize])))))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (base_row[134usize]))))
                + ((base_row[129usize]) * (ext_row[44usize])),
            ((node_503)
                * ((((((ext_row[45usize])
                    * ((challenges[CascadeLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_509))))
                    * ((challenges[CascadeLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_512))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((BFieldElement::from_raw_u64(8589934590u64))
                            * (challenges[CascadeLookupIndeterminate])))) + (node_509))
                    + (node_512))) + ((base_row[129usize]) * (ext_row[45usize])),
            ((ext_row[46usize])
                * ((challenges[CascadeLookupIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((base_row[137usize])
                            * (challenges[LookupTableOutputWeight])))))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (base_row[138usize])),
            ((ext_row[47usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (challenges[LookupTablePublicIndeterminate])))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (base_row[137usize])),
            (((base_row[139usize])
                + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                * (ext_row[48usize]))
                + ((base_row[139usize])
                    * (((ext_row[48usize])
                        * ((challenges[U32Indeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((((challenges[U32LhsWeight]) * (base_row[143usize]))
                                    + ((challenges[U32RhsWeight]) * (base_row[145usize])))
                                    + ((challenges[U32CiWeight]) * (base_row[142usize])))
                                    + ((challenges[U32ResultWeight]) * (base_row[147usize]))))))
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
        let node_107 = (base_row[152usize])
            * ((base_row[64usize])
                + (BFieldElement::from_raw_u64(18446744047939747846u64)));
        let node_227 = (base_row[153usize])
            * ((base_row[64usize])
                + (BFieldElement::from_raw_u64(18446744047939747846u64)));
        let node_244 = ((base_row[154usize])
            * ((base_row[64usize])
                + (BFieldElement::from_raw_u64(18446744052234715141u64))))
            * ((base_row[64usize])
                + (BFieldElement::from_raw_u64(18446744047939747846u64)));
        let node_251 = ((base_row[154usize])
            * ((base_row[64usize])
                + (BFieldElement::from_raw_u64(18446744056529682436u64))))
            * ((base_row[64usize])
                + (BFieldElement::from_raw_u64(18446744047939747846u64)));
        let node_676 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (base_row[157usize]));
        let node_119 = (((base_row[63usize])
            + (BFieldElement::from_raw_u64(18446743992105173011u64)))
            * ((base_row[63usize])
                + (BFieldElement::from_raw_u64(18446743923385696291u64))))
            * ((base_row[63usize])
                + (BFieldElement::from_raw_u64(18446743863256154161u64)));
        let node_121 = (node_107) * (base_row[161usize]);
        let node_681 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (base_row[160usize]));
        let node_106 = (base_row[64usize])
            + (BFieldElement::from_raw_u64(18446744047939747846u64));
        let node_700 = (base_row[142usize])
            + (BFieldElement::from_raw_u64(18446743949155500061u64));
        let node_696 = (base_row[142usize])
            + (BFieldElement::from_raw_u64(18446743940565565471u64));
        let node_99 = (base_row[64usize])
            + (BFieldElement::from_raw_u64(18446744056529682436u64));
        let node_102 = (base_row[64usize])
            + (BFieldElement::from_raw_u64(18446744052234715141u64));
        let node_158 = ((((BFieldElement::from_raw_u64(18446744065119617025u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((base_row[65usize])
                    * (BFieldElement::from_raw_u64(281474976645120u64)))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (base_row[66usize]))) * (base_row[109usize]))
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_160 = ((((BFieldElement::from_raw_u64(18446744065119617025u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((base_row[69usize])
                    * (BFieldElement::from_raw_u64(281474976645120u64)))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (base_row[70usize]))) * (base_row[110usize]))
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_162 = ((((BFieldElement::from_raw_u64(18446744065119617025u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((base_row[73usize])
                    * (BFieldElement::from_raw_u64(281474976645120u64)))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (base_row[74usize]))) * (base_row[111usize]))
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_164 = ((((BFieldElement::from_raw_u64(18446744065119617025u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((base_row[77usize])
                    * (BFieldElement::from_raw_u64(281474976645120u64)))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (base_row[78usize]))) * (base_row[112usize]))
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_702 = (base_row[139usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_95 = (base_row[64usize])
            + (BFieldElement::from_raw_u64(18446744060824649731u64));
        let node_692 = (base_row[142usize])
            + (BFieldElement::from_raw_u64(18446744017874976781u64));
        let node_11 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (((BFieldElement::from_raw_u64(38654705655u64))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (base_row[3usize]))) * (base_row[4usize])));
        let node_8 = (BFieldElement::from_raw_u64(38654705655u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (base_row[3usize]));
        let node_109 = (((base_row[62usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64)))
            * ((base_row[62usize])
                + (BFieldElement::from_raw_u64(18446744060824649731u64))))
            * ((base_row[62usize])
                + (BFieldElement::from_raw_u64(18446744056529682436u64)));
        let node_87 = (base_row[62usize])
            + (BFieldElement::from_raw_u64(18446744060824649731u64));
        let node_74 = (base_row[63usize])
            + (BFieldElement::from_raw_u64(18446743992105173011u64));
        let node_80 = (base_row[63usize])
            + (BFieldElement::from_raw_u64(18446743923385696291u64));
        let node_83 = (base_row[63usize])
            + (BFieldElement::from_raw_u64(18446743863256154161u64));
        let node_131 = ((BFieldElement::from_raw_u64(18446744065119617025u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((base_row[65usize])
                    * (BFieldElement::from_raw_u64(281474976645120u64)))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (base_row[66usize]));
        let node_138 = ((BFieldElement::from_raw_u64(18446744065119617025u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((base_row[69usize])
                    * (BFieldElement::from_raw_u64(281474976645120u64)))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (base_row[70usize]));
        let node_145 = ((BFieldElement::from_raw_u64(18446744065119617025u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((base_row[73usize])
                    * (BFieldElement::from_raw_u64(281474976645120u64)))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (base_row[74usize]));
        let node_152 = ((BFieldElement::from_raw_u64(18446744065119617025u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((base_row[77usize])
                    * (BFieldElement::from_raw_u64(281474976645120u64)))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (base_row[78usize]));
        let node_93 = (base_row[64usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_684 = (base_row[142usize])
            + (BFieldElement::from_raw_u64(18446744052234715141u64));
        let node_688 = (base_row[142usize])
            + (BFieldElement::from_raw_u64(18446744009285042191u64));
        let node_88 = ((base_row[62usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64))) * (node_87);
        let node_84 = (base_row[62usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_108 = (base_row[62usize])
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
            (node_109) * (base_row[62usize]),
            (node_87) * (node_74),
            ((base_row[165usize]) * (node_80)) * (node_83),
            (node_109) * (base_row[64usize]),
            (node_119) * (base_row[64usize]),
            (node_158) * (base_row[109usize]),
            (node_160) * (base_row[110usize]),
            (node_162) * (base_row[111usize]),
            (node_164) * (base_row[112usize]),
            (node_158) * (node_131),
            (node_160) * (node_138),
            (node_162) * (node_145),
            (node_164) * (node_152),
            (node_158)
                * (((base_row[67usize])
                    * (BFieldElement::from_raw_u64(281474976645120u64)))
                    + (base_row[68usize])),
            (node_160)
                * (((base_row[71usize])
                    * (BFieldElement::from_raw_u64(281474976645120u64)))
                    + (base_row[72usize])),
            (node_162)
                * (((base_row[75usize])
                    * (BFieldElement::from_raw_u64(281474976645120u64)))
                    + (base_row[76usize])),
            (node_164)
                * (((base_row[79usize])
                    * (BFieldElement::from_raw_u64(281474976645120u64)))
                    + (base_row[80usize])),
            (node_119) * (base_row[103usize]),
            (node_119) * (base_row[104usize]),
            (node_119) * (base_row[105usize]),
            (node_119) * (base_row[106usize]),
            (node_119) * (base_row[107usize]),
            (node_119) * (base_row[108usize]),
            (node_121)
                * ((base_row[103usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (node_121)
                * ((base_row[104usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (node_121)
                * ((base_row[105usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (node_121)
                * ((base_row[106usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (node_121)
                * ((base_row[107usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (node_121)
                * ((base_row[108usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (((((node_107)
                * ((base_row[113usize])
                    + (BFieldElement::from_raw_u64(11408918724931329738u64))))
                + ((node_227)
                    * ((base_row[113usize])
                        + (BFieldElement::from_raw_u64(16073625066478178581u64)))))
                + ((base_row[155usize])
                    * ((base_row[113usize])
                        + (BFieldElement::from_raw_u64(12231462398569191607u64)))))
                + ((node_244)
                    * ((base_row[113usize])
                        + (BFieldElement::from_raw_u64(9408518518620565480u64)))))
                + ((node_251)
                    * ((base_row[113usize])
                        + (BFieldElement::from_raw_u64(11492978409391175103u64)))),
            (((((node_107)
                * ((base_row[114usize])
                    + (BFieldElement::from_raw_u64(2786462832312611053u64))))
                + ((node_227)
                    * ((base_row[114usize])
                        + (BFieldElement::from_raw_u64(11837051899140380443u64)))))
                + ((base_row[155usize])
                    * ((base_row[114usize])
                        + (BFieldElement::from_raw_u64(11546487907579866869u64)))))
                + ((node_244)
                    * ((base_row[114usize])
                        + (BFieldElement::from_raw_u64(1785884128667671832u64)))))
                + ((node_251)
                    * ((base_row[114usize])
                        + (BFieldElement::from_raw_u64(17615222217495663839u64)))),
            (((((node_107)
                * ((base_row[115usize])
                    + (BFieldElement::from_raw_u64(6782977121958050999u64))))
                + ((node_227)
                    * ((base_row[115usize])
                        + (BFieldElement::from_raw_u64(15625104599191418968u64)))))
                + ((base_row[155usize])
                    * ((base_row[115usize])
                        + (BFieldElement::from_raw_u64(14006427992450931468u64)))))
                + ((node_244)
                    * ((base_row[115usize])
                        + (BFieldElement::from_raw_u64(1188899344229954938u64)))))
                + ((node_251)
                    * ((base_row[115usize])
                        + (BFieldElement::from_raw_u64(5864349944556149748u64)))),
            (((((node_107)
                * ((base_row[116usize])
                    + (BFieldElement::from_raw_u64(8688421733879975670u64))))
                + ((node_227)
                    * ((base_row[116usize])
                        + (BFieldElement::from_raw_u64(12819157612210448391u64)))))
                + ((base_row[155usize])
                    * ((base_row[116usize])
                        + (BFieldElement::from_raw_u64(11770003398407723041u64)))))
                + ((node_244)
                    * ((base_row[116usize])
                        + (BFieldElement::from_raw_u64(14740727267735052728u64)))))
                + ((node_251)
                    * ((base_row[116usize])
                        + (BFieldElement::from_raw_u64(2745609811140253793u64)))),
            (((((node_107)
                * ((base_row[117usize])
                    + (BFieldElement::from_raw_u64(8602724563769480463u64))))
                + ((node_227)
                    * ((base_row[117usize])
                        + (BFieldElement::from_raw_u64(6235256903503367222u64)))))
                + ((base_row[155usize])
                    * ((base_row[117usize])
                        + (BFieldElement::from_raw_u64(15124190001489436038u64)))))
                + ((node_244)
                    * ((base_row[117usize])
                        + (BFieldElement::from_raw_u64(880257844992994007u64)))))
                + ((node_251)
                    * ((base_row[117usize])
                        + (BFieldElement::from_raw_u64(15189664869386394185u64)))),
            (((((node_107)
                * ((base_row[118usize])
                    + (BFieldElement::from_raw_u64(13589155570211330507u64))))
                + ((node_227)
                    * ((base_row[118usize])
                        + (BFieldElement::from_raw_u64(11242082964257948320u64)))))
                + ((base_row[155usize])
                    * ((base_row[118usize])
                        + (BFieldElement::from_raw_u64(14834674155811570980u64)))))
                + ((node_244)
                    * ((base_row[118usize])
                        + (BFieldElement::from_raw_u64(10737952517017171197u64)))))
                + ((node_251)
                    * ((base_row[118usize])
                        + (BFieldElement::from_raw_u64(5192963426821415349u64)))),
            (((((node_107)
                * ((base_row[119usize])
                    + (BFieldElement::from_raw_u64(10263462378312899510u64))))
                + ((node_227)
                    * ((base_row[119usize])
                        + (BFieldElement::from_raw_u64(5820425254787221108u64)))))
                + ((base_row[155usize])
                    * ((base_row[119usize])
                        + (BFieldElement::from_raw_u64(13004675752386552573u64)))))
                + ((node_244)
                    * ((base_row[119usize])
                        + (BFieldElement::from_raw_u64(15757222735741919824u64)))))
                + ((node_251)
                    * ((base_row[119usize])
                        + (BFieldElement::from_raw_u64(11971160388083607515u64)))),
            (((((node_107)
                * ((base_row[120usize])
                    + (BFieldElement::from_raw_u64(3264875873073042616u64))))
                + ((node_227)
                    * ((base_row[120usize])
                        + (BFieldElement::from_raw_u64(12019227591549292608u64)))))
                + ((base_row[155usize])
                    * ((base_row[120usize])
                        + (BFieldElement::from_raw_u64(1475232519215872482u64)))))
                + ((node_244)
                    * ((base_row[120usize])
                        + (BFieldElement::from_raw_u64(14382578632612566479u64)))))
                + ((node_251)
                    * ((base_row[120usize])
                        + (BFieldElement::from_raw_u64(11608544217838050708u64)))),
            (((((node_107)
                * ((base_row[121usize])
                    + (BFieldElement::from_raw_u64(3133435276616064683u64))))
                + ((node_227)
                    * ((base_row[121usize])
                        + (BFieldElement::from_raw_u64(4625353063880731092u64)))))
                + ((base_row[155usize])
                    * ((base_row[121usize])
                        + (BFieldElement::from_raw_u64(4883869161905122316u64)))))
                + ((node_244)
                    * ((base_row[121usize])
                        + (BFieldElement::from_raw_u64(3305272539067787726u64)))))
                + ((node_251)
                    * ((base_row[121usize])
                        + (BFieldElement::from_raw_u64(674972795234232729u64)))),
            (((((node_107)
                * ((base_row[122usize])
                    + (BFieldElement::from_raw_u64(13508500531157332153u64))))
                + ((node_227)
                    * ((base_row[122usize])
                        + (BFieldElement::from_raw_u64(3723900760706330287u64)))))
                + ((base_row[155usize])
                    * ((base_row[122usize])
                        + (BFieldElement::from_raw_u64(12579737103870920763u64)))))
                + ((node_244)
                    * ((base_row[122usize])
                        + (BFieldElement::from_raw_u64(17082569335437832789u64)))))
                + ((node_251)
                    * ((base_row[122usize])
                        + (BFieldElement::from_raw_u64(14165256104883557753u64)))),
            (((((node_107)
                * ((base_row[123usize])
                    + (BFieldElement::from_raw_u64(6968886508437513677u64))))
                + ((node_227)
                    * ((base_row[123usize])
                        + (BFieldElement::from_raw_u64(615596267195055952u64)))))
                + ((base_row[155usize])
                    * ((base_row[123usize])
                        + (BFieldElement::from_raw_u64(10119826060478909841u64)))))
                + ((node_244)
                    * ((base_row[123usize])
                        + (BFieldElement::from_raw_u64(229051680548583225u64)))))
                + ((node_251)
                    * ((base_row[123usize])
                        + (BFieldElement::from_raw_u64(15283356519694111298u64)))),
            (((((node_107)
                * ((base_row[124usize])
                    + (BFieldElement::from_raw_u64(9713264609690967820u64))))
                + ((node_227)
                    * ((base_row[124usize])
                        + (BFieldElement::from_raw_u64(18227830850447556704u64)))))
                + ((base_row[155usize])
                    * ((base_row[124usize])
                        + (BFieldElement::from_raw_u64(1528714547662620921u64)))))
                + ((node_244)
                    * ((base_row[124usize])
                        + (BFieldElement::from_raw_u64(2943254981416254648u64)))))
                + ((node_251)
                    * ((base_row[124usize])
                        + (BFieldElement::from_raw_u64(2306049938060341466u64)))),
            (((((node_107)
                * ((base_row[125usize])
                    + (BFieldElement::from_raw_u64(12482374976099749513u64))))
                + ((node_227)
                    * ((base_row[125usize])
                        + (BFieldElement::from_raw_u64(15609691041895848348u64)))))
                + ((base_row[155usize])
                    * ((base_row[125usize])
                        + (BFieldElement::from_raw_u64(12972275929555275935u64)))))
                + ((node_244)
                    * ((base_row[125usize])
                        + (BFieldElement::from_raw_u64(5767629304344025219u64)))))
                + ((node_251)
                    * ((base_row[125usize])
                        + (BFieldElement::from_raw_u64(11578793764462375094u64)))),
            (((((node_107)
                * ((base_row[126usize])
                    + (BFieldElement::from_raw_u64(13209711277645656680u64))))
                + ((node_227)
                    * ((base_row[126usize])
                        + (BFieldElement::from_raw_u64(15235800289984546486u64)))))
                + ((base_row[155usize])
                    * ((base_row[126usize])
                        + (BFieldElement::from_raw_u64(15992731669612695172u64)))))
                + ((node_244)
                    * ((base_row[126usize])
                        + (BFieldElement::from_raw_u64(16721422493821450473u64)))))
                + ((node_251)
                    * ((base_row[126usize])
                        + (BFieldElement::from_raw_u64(7511767364422267184u64)))),
            (((((node_107)
                * ((base_row[127usize])
                    + (BFieldElement::from_raw_u64(87705059284758253u64))))
                + ((node_227)
                    * ((base_row[127usize])
                        + (BFieldElement::from_raw_u64(11392407538241985753u64)))))
                + ((base_row[155usize])
                    * ((base_row[127usize])
                        + (BFieldElement::from_raw_u64(17877154195438905917u64)))))
                + ((node_244)
                    * ((base_row[127usize])
                        + (BFieldElement::from_raw_u64(5753720429376839714u64)))))
                + ((node_251)
                    * ((base_row[127usize])
                        + (BFieldElement::from_raw_u64(16999805755930336630u64)))),
            (((((node_107)
                * ((base_row[128usize])
                    + (BFieldElement::from_raw_u64(330155256278907084u64))))
                + ((node_227)
                    * ((base_row[128usize])
                        + (BFieldElement::from_raw_u64(11776128816341368822u64)))))
                + ((base_row[155usize])
                    * ((base_row[128usize])
                        + (BFieldElement::from_raw_u64(939319986782105612u64)))))
                + ((node_244)
                    * ((base_row[128usize])
                        + (BFieldElement::from_raw_u64(2063756830275051942u64)))))
                + ((node_251)
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
            (base_row[144usize]) * (node_676),
            (base_row[143usize]) * (node_676),
            (base_row[146usize]) * (node_681),
            (base_row[145usize]) * (node_681),
            (base_row[167usize])
                * ((base_row[147usize])
                    + (BFieldElement::from_raw_u64(18446744060824649731u64))),
            (base_row[168usize]) * (base_row[147usize]),
            (((base_row[163usize]) * (node_676)) * (node_681)) * (base_row[147usize]),
            (((base_row[166usize]) * (node_700)) * (node_681))
                * ((base_row[147usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (((base_row[164usize]) * (node_702)) * (node_676))
                * ((base_row[147usize]) + (BFieldElement::from_raw_u64(4294967295u64))),
            (((base_row[166usize]) * (node_696)) * (node_676)) * (base_row[147usize]),
            ((base_row[164usize]) * (base_row[139usize])) * (node_676),
            (node_702) * (base_row[148usize]),
            (base_row[151usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((base_row[64usize]) * (node_93))),
            (base_row[152usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((node_93) * (node_95)) * (node_99)) * (node_102))),
            (base_row[153usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((base_row[64usize]) * (node_95)) * (node_99)) * (node_102))),
            (base_row[154usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((base_row[151usize]) * (node_95))),
            (base_row[155usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((base_row[151usize]) * (node_99)) * (node_102)) * (node_106))),
            (base_row[156usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_684)
                        * ((base_row[142usize])
                            + (BFieldElement::from_raw_u64(18446744043644780551u64))))),
            (base_row[157usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((base_row[143usize]) * (base_row[144usize]))),
            (base_row[158usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((node_684) * (node_688)) * (node_692)) * (node_696))),
            (base_row[159usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((base_row[156usize]) * (node_688))),
            (base_row[160usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((base_row[145usize]) * (base_row[146usize]))),
            (base_row[161usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_88) * (base_row[62usize]))),
            (base_row[162usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((base_row[158usize]) * (node_700))),
            (base_row[163usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((base_row[156usize]) * (node_692)) * (node_696)) * (node_700))),
            (base_row[164usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((base_row[159usize]) * (node_696)) * (node_700))),
            (base_row[165usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((node_84) * (node_108)) * (base_row[62usize]))
                        * ((base_row[63usize])
                            + (BFieldElement::from_raw_u64(18446743897615892521u64))))),
            (base_row[166usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((base_row[159usize]) * (node_692))),
            (base_row[167usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((base_row[162usize]) * (node_702)) * (node_676)) * (node_681))),
            (base_row[168usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((base_row[162usize]) * (base_row[139usize])) * (node_676))
                        * (node_681))),
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
        let node_4849 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (next_base_row[8usize]));
        let node_121 = (next_base_row[19usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[19usize]));
        let node_537 = (next_ext_row[3usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[3usize]));
        let node_541 = (next_ext_row[4usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[4usize]));
        let node_125 = (next_base_row[20usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[20usize]));
        let node_129 = (next_base_row[21usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[21usize]));
        let node_533 = (next_ext_row[7usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[7usize]));
        let node_1472 = (current_base_row[18usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_932 = ((next_base_row[9usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[9usize])))
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_159 = (next_base_row[38usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[38usize]));
        let node_1470 = (current_base_row[17usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_824 = (next_base_row[22usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[22usize]));
        let node_169 = ((challenges[OpStackClkWeight]) * (current_base_row[7usize]))
            + ((challenges[OpStackIb1Weight]) * (current_base_row[13usize]));
        let node_545 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[22usize]);
        let node_6564 = (current_base_row[276usize])
            * ((next_base_row[64usize])
                + (BFieldElement::from_raw_u64(18446744052234715141u64)));
        let node_6643 = (((next_base_row[63usize])
            + (BFieldElement::from_raw_u64(18446743992105173011u64)))
            * ((next_base_row[63usize])
                + (BFieldElement::from_raw_u64(18446743923385696291u64))))
            * ((next_base_row[63usize])
                + (BFieldElement::from_raw_u64(18446743863256154161u64)));
        let node_5588 = (((((current_base_row[81usize])
            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
            + ((current_base_row[82usize])
                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
            + ((current_base_row[83usize])
                * (BFieldElement::from_raw_u64(281474976645120u64))))
            + (current_base_row[84usize])) * (BFieldElement::from_raw_u64(1u64));
        let node_5599 = (((((current_base_row[85usize])
            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
            + ((current_base_row[86usize])
                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
            + ((current_base_row[87usize])
                * (BFieldElement::from_raw_u64(281474976645120u64))))
            + (current_base_row[88usize])) * (BFieldElement::from_raw_u64(1u64));
        let node_5610 = (((((current_base_row[89usize])
            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
            + ((current_base_row[90usize])
                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
            + ((current_base_row[91usize])
                * (BFieldElement::from_raw_u64(281474976645120u64))))
            + (current_base_row[92usize])) * (BFieldElement::from_raw_u64(1u64));
        let node_5621 = (((((current_base_row[93usize])
            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
            + ((current_base_row[94usize])
                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
            + ((current_base_row[95usize])
                * (BFieldElement::from_raw_u64(281474976645120u64))))
            + (current_base_row[96usize])) * (BFieldElement::from_raw_u64(1u64));
        let node_203 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[27usize]);
        let node_872 = (next_base_row[24usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[24usize]));
        let node_876 = (next_base_row[25usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[25usize]));
        let node_880 = (next_base_row[26usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[26usize]));
        let node_868 = (next_base_row[23usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[23usize]));
        let node_199 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[26usize]);
        let node_298 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[39usize]));
        let node_884 = (next_base_row[27usize]) + (node_203);
        let node_888 = (next_base_row[28usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[28usize]));
        let node_892 = (next_base_row[29usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[29usize]));
        let node_896 = (next_base_row[30usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[30usize]));
        let node_900 = (next_base_row[31usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[31usize]));
        let node_904 = (next_base_row[32usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[32usize]));
        let node_908 = (next_base_row[33usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[33usize]));
        let node_912 = (next_base_row[34usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[34usize]));
        let node_916 = (next_base_row[35usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[35usize]));
        let node_920 = (next_base_row[36usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[36usize]));
        let node_924 = (next_base_row[37usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[37usize]));
        let node_927 = (next_ext_row[6usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[6usize]));
        let node_195 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[25usize]);
        let node_207 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[28usize]);
        let node_223 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[32usize]);
        let node_211 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[29usize]);
        let node_227 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[33usize]);
        let node_215 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[30usize]);
        let node_219 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[31usize]);
        let node_1468 = (current_base_row[16usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_200 = (next_base_row[25usize]) + (node_199);
        let node_204 = (next_base_row[26usize]) + (node_203);
        let node_208 = (next_base_row[27usize]) + (node_207);
        let node_231 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[34usize]);
        let node_212 = (next_base_row[28usize]) + (node_211);
        let node_216 = (next_base_row[29usize]) + (node_215);
        let node_220 = (next_base_row[30usize]) + (node_219);
        let node_160 = (node_159) + (BFieldElement::from_raw_u64(4294967295u64));
        let node_224 = (next_base_row[31usize]) + (node_223);
        let node_184 = (next_ext_row[6usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[73usize]));
        let node_228 = (next_base_row[32usize]) + (node_227);
        let node_232 = (next_base_row[33usize]) + (node_231);
        let node_236 = (next_base_row[34usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[35usize]));
        let node_240 = (next_base_row[35usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[36usize]));
        let node_244 = (next_base_row[36usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[37usize]));
        let node_7032 = (next_base_row[139usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_187 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[23usize]);
        let node_191 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[24usize]);
        let node_117 = ((next_base_row[9usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[9usize])))
            + (BFieldElement::from_raw_u64(18446744060824649731u64));
        let node_192 = (next_base_row[23usize]) + (node_191);
        let node_196 = (next_base_row[24usize]) + (node_195);
        let node_235 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[35usize]);
        let node_5355 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (((next_base_row[52usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[52usize]))) * (current_base_row[54usize])));
        let node_132 = (current_base_row[39usize])
            * ((current_base_row[39usize])
                + (BFieldElement::from_raw_u64(18446744065119617026u64)));
        let node_239 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[36usize]);
        let node_155 = ((((current_base_row[11usize])
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
        let node_243 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[37usize]);
        let node_5352 = (next_base_row[52usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[52usize]));
        let node_6404 = (next_base_row[62usize])
            + (BFieldElement::from_raw_u64(18446744056529682436u64));
        let node_7028 = (current_base_row[145usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((BFieldElement::from_raw_u64(8589934590u64))
                    * (next_base_row[145usize])));
        let node_5229 = (next_ext_row[12usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[12usize]));
        let node_7025 = (current_base_row[143usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((BFieldElement::from_raw_u64(8589934590u64))
                    * (next_base_row[143usize])));
        let node_1466 = (current_base_row[15usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_994 = (current_base_row[7usize]) * (challenges[RamClkWeight]);
        let node_1372 = (challenges[StandardInputIndeterminate])
            * (current_ext_row[3usize]);
        let node_6408 = (next_base_row[63usize])
            + (BFieldElement::from_raw_u64(18446743897615892521u64));
        let node_997 = (node_994) + (challenges[RamInstructionTypeWeight]);
        let node_542 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[11usize]);
        let node_934 = (current_base_row[272usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_114 = (next_base_row[9usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[9usize]));
        let node_547 = (next_base_row[24usize]) + (node_187);
        let node_548 = (next_base_row[25usize]) + (node_191);
        let node_549 = (next_base_row[26usize]) + (node_195);
        let node_550 = (next_base_row[27usize]) + (node_199);
        let node_551 = (next_base_row[28usize]) + (node_203);
        let node_552 = (next_base_row[29usize]) + (node_207);
        let node_553 = (next_base_row[30usize]) + (node_211);
        let node_554 = (next_base_row[31usize]) + (node_215);
        let node_561 = (node_159)
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_555 = (next_base_row[32usize]) + (node_219);
        let node_556 = (next_base_row[33usize]) + (node_223);
        let node_557 = (next_base_row[34usize]) + (node_227);
        let node_558 = (next_base_row[35usize]) + (node_231);
        let node_559 = (next_base_row[36usize]) + (node_235);
        let node_560 = (next_base_row[37usize]) + (node_239);
        let node_572 = (next_ext_row[6usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((current_ext_row[6usize])
                    * ((challenges[OpStackIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((node_169)
                                + ((challenges[OpStackPointerWeight])
                                    * (current_base_row[38usize])))
                                + ((challenges[OpStackFirstUnderflowElementWeight])
                                    * (current_base_row[37usize])))))));
        let node_5461 = ((next_base_row[59usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[59usize])))
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_6264 = (((((next_base_row[65usize])
            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
            + ((next_base_row[66usize])
                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
            + ((next_base_row[67usize])
                * (BFieldElement::from_raw_u64(281474976645120u64))))
            + (next_base_row[68usize])) * (BFieldElement::from_raw_u64(1u64));
        let node_6275 = (((((next_base_row[69usize])
            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
            + ((next_base_row[70usize])
                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
            + ((next_base_row[71usize])
                * (BFieldElement::from_raw_u64(281474976645120u64))))
            + (next_base_row[72usize])) * (BFieldElement::from_raw_u64(1u64));
        let node_6286 = (((((next_base_row[73usize])
            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
            + ((next_base_row[74usize])
                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
            + ((next_base_row[75usize])
                * (BFieldElement::from_raw_u64(281474976645120u64))))
            + (next_base_row[76usize])) * (BFieldElement::from_raw_u64(1u64));
        let node_6297 = (((((next_base_row[77usize])
            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
            + ((next_base_row[78usize])
                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
            + ((next_base_row[79usize])
                * (BFieldElement::from_raw_u64(281474976645120u64))))
            + (next_base_row[80usize])) * (BFieldElement::from_raw_u64(1u64));
        let node_6398 = (next_base_row[62usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_6972 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (next_base_row[135usize]));
        let node_7060 = (next_base_row[142usize])
            + (BFieldElement::from_raw_u64(18446743940565565471u64));
        let node_1464 = (current_base_row[14usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_251 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[40usize]));
        let node_7064 = (next_base_row[142usize])
            + (BFieldElement::from_raw_u64(18446743949155500061u64));
        let node_34 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((current_base_row[4usize])
                    * ((BFieldElement::from_raw_u64(38654705655u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[3usize])))));
        let node_1295 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[337usize]));
        let node_546 = (next_base_row[23usize]) + (node_545);
        let node_256 = (current_base_row[185usize])
            * ((next_base_row[22usize]) + (node_187));
        let node_302 = (current_base_row[186usize])
            * ((next_base_row[22usize]) + (node_191));
        let node_345 = (current_base_row[187usize])
            * ((next_base_row[22usize]) + (node_195));
        let node_332 = (next_base_row[25usize]) + (node_207);
        let node_387 = (current_base_row[188usize])
            * ((next_base_row[22usize]) + (node_199));
        let node_417 = (next_base_row[27usize]) + (node_223);
        let node_333 = (next_base_row[26usize]) + (node_211);
        let node_188 = (next_base_row[22usize]) + (node_187);
        let node_418 = (next_base_row[28usize]) + (node_227);
        let node_334 = (next_base_row[27usize]) + (node_215);
        let node_419 = (next_base_row[29usize]) + (node_231);
        let node_335 = (next_base_row[28usize]) + (node_219);
        let node_420 = (next_base_row[30usize]) + (node_235);
        let node_336 = (next_base_row[29usize]) + (node_223);
        let node_421 = (next_base_row[31usize]) + (node_239);
        let node_337 = (next_base_row[30usize]) + (node_227);
        let node_441 = (((((current_base_row[185usize]) * (node_160))
            + ((current_base_row[186usize])
                * ((node_159) + (BFieldElement::from_raw_u64(8589934590u64)))))
            + ((current_base_row[187usize])
                * ((node_159) + (BFieldElement::from_raw_u64(12884901885u64)))))
            + ((current_base_row[188usize])
                * ((node_159) + (BFieldElement::from_raw_u64(17179869180u64)))))
            + ((current_base_row[189usize])
                * ((node_159) + (BFieldElement::from_raw_u64(21474836475u64))));
        let node_753 = (((((current_base_row[185usize]) * (node_561))
            + ((current_base_row[186usize])
                * ((node_159) + (BFieldElement::from_raw_u64(18446744060824649731u64)))))
            + ((current_base_row[187usize])
                * ((node_159) + (BFieldElement::from_raw_u64(18446744056529682436u64)))))
            + ((current_base_row[188usize])
                * ((node_159) + (BFieldElement::from_raw_u64(18446744052234715141u64)))))
            + ((current_base_row[189usize])
                * ((node_159) + (BFieldElement::from_raw_u64(18446744047939747846u64))));
        let node_422 = (next_base_row[32usize]) + (node_243);
        let node_400 = (node_159) + (BFieldElement::from_raw_u64(21474836475u64));
        let node_338 = (next_base_row[31usize]) + (node_231);
        let node_446 = (((((current_base_row[185usize]) * (node_184))
            + (current_ext_row[74usize]))
            + ((current_base_row[187usize])
                * ((next_ext_row[6usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_ext_row[71usize]))))) + (current_ext_row[75usize]))
            + ((current_base_row[189usize])
                * ((next_ext_row[6usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_ext_row[59usize]))));
        let node_758 = ((((current_ext_row[76usize]) + (current_ext_row[77usize]))
            + (current_ext_row[78usize])) + (current_ext_row[79usize]))
            + ((current_base_row[189usize])
                * ((next_ext_row[6usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_ext_row[72usize]))));
        let node_411 = (next_ext_row[6usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[59usize]));
        let node_339 = (next_base_row[32usize]) + (node_235);
        let node_340 = (next_base_row[33usize]) + (node_239);
        let node_456 = ((((current_base_row[287usize]) + (current_base_row[288usize]))
            + (current_base_row[289usize])) + (current_base_row[290usize]))
            + (current_base_row[291usize]);
        let node_768 = ((((current_base_row[282usize]) + (current_base_row[283usize]))
            + (current_base_row[284usize])) + (current_base_row[285usize]))
            + (current_base_row[286usize]);
        let node_341 = (next_base_row[34usize]) + (node_243);
        let node_461 = ((((current_base_row[319usize]) + (current_base_row[321usize]))
            + (current_base_row[323usize])) + (current_base_row[325usize]))
            + (current_base_row[327usize]);
        let node_773 = (((((current_base_row[185usize]) * (node_548))
            + ((current_base_row[186usize]) * ((next_base_row[26usize]) + (node_191))))
            + ((current_base_row[187usize]) * ((next_base_row[27usize]) + (node_191))))
            + ((current_base_row[188usize]) * ((next_base_row[28usize]) + (node_191))))
            + ((current_base_row[189usize]) * ((next_base_row[29usize]) + (node_191)));
        let node_317 = (node_159) + (BFieldElement::from_raw_u64(12884901885u64));
        let node_466 = ((((current_base_row[320usize]) + (current_base_row[322usize]))
            + (current_base_row[324usize])) + (current_base_row[326usize]))
            + (current_base_row[328usize]);
        let node_778 = (((((current_base_row[185usize]) * (node_549))
            + ((current_base_row[186usize]) * ((next_base_row[27usize]) + (node_195))))
            + ((current_base_row[187usize]) * ((next_base_row[28usize]) + (node_195))))
            + ((current_base_row[188usize]) * ((next_base_row[29usize]) + (node_195))))
            + ((current_base_row[189usize]) * ((next_base_row[30usize]) + (node_195)));
        let node_328 = (next_ext_row[6usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[71usize]));
        let node_471 = (((((current_base_row[185usize]) * (node_204))
            + ((current_base_row[186usize]) * ((next_base_row[26usize]) + (node_207))))
            + ((current_base_row[187usize]) * (node_333)))
            + ((current_base_row[188usize]) * ((next_base_row[26usize]) + (node_215))))
            + ((current_base_row[189usize]) * ((next_base_row[26usize]) + (node_219)));
        let node_783 = (((((current_base_row[185usize]) * (node_550))
            + ((current_base_row[186usize]) * ((next_base_row[28usize]) + (node_199))))
            + ((current_base_row[187usize]) * ((next_base_row[29usize]) + (node_199))))
            + ((current_base_row[188usize]) * ((next_base_row[30usize]) + (node_199))))
            + ((current_base_row[189usize]) * ((next_base_row[31usize]) + (node_199)));
        let node_476 = (((((current_base_row[185usize]) * (node_208))
            + ((current_base_row[186usize]) * ((next_base_row[27usize]) + (node_211))))
            + ((current_base_row[187usize]) * (node_334)))
            + ((current_base_row[188usize]) * ((next_base_row[27usize]) + (node_219))))
            + ((current_base_row[189usize]) * (node_417));
        let node_788 = (((((current_base_row[185usize]) * (node_551))
            + ((current_base_row[186usize]) * ((next_base_row[29usize]) + (node_203))))
            + ((current_base_row[187usize]) * ((next_base_row[30usize]) + (node_203))))
            + ((current_base_row[188usize]) * ((next_base_row[31usize]) + (node_203))))
            + ((current_base_row[189usize]) * ((next_base_row[32usize]) + (node_203)));
        let node_481 = (((((current_base_row[185usize]) * (node_212))
            + ((current_base_row[186usize]) * ((next_base_row[28usize]) + (node_215))))
            + ((current_base_row[187usize]) * (node_335)))
            + ((current_base_row[188usize]) * ((next_base_row[28usize]) + (node_223))))
            + ((current_base_row[189usize]) * (node_418));
        let node_793 = (((((current_base_row[185usize]) * (node_552))
            + ((current_base_row[186usize]) * ((next_base_row[30usize]) + (node_207))))
            + ((current_base_row[187usize]) * ((next_base_row[31usize]) + (node_207))))
            + ((current_base_row[188usize]) * ((next_base_row[32usize]) + (node_207))))
            + ((current_base_row[189usize]) * ((next_base_row[33usize]) + (node_207)));
        let node_486 = (((((current_base_row[185usize]) * (node_216))
            + ((current_base_row[186usize]) * ((next_base_row[29usize]) + (node_219))))
            + ((current_base_row[187usize]) * (node_336)))
            + ((current_base_row[188usize]) * ((next_base_row[29usize]) + (node_227))))
            + ((current_base_row[189usize]) * (node_419));
        let node_798 = (((((current_base_row[185usize]) * (node_553))
            + ((current_base_row[186usize]) * ((next_base_row[31usize]) + (node_211))))
            + ((current_base_row[187usize]) * ((next_base_row[32usize]) + (node_211))))
            + ((current_base_row[188usize]) * ((next_base_row[33usize]) + (node_211))))
            + ((current_base_row[189usize]) * ((next_base_row[34usize]) + (node_211)));
        let node_491 = (((((current_base_row[185usize]) * (node_220))
            + ((current_base_row[186usize]) * ((next_base_row[30usize]) + (node_223))))
            + ((current_base_row[187usize]) * (node_337)))
            + ((current_base_row[188usize]) * ((next_base_row[30usize]) + (node_231))))
            + ((current_base_row[189usize]) * (node_420));
        let node_803 = (((((current_base_row[185usize]) * (node_554))
            + ((current_base_row[186usize]) * ((next_base_row[32usize]) + (node_215))))
            + ((current_base_row[187usize]) * ((next_base_row[33usize]) + (node_215))))
            + ((current_base_row[188usize]) * ((next_base_row[34usize]) + (node_215))))
            + ((current_base_row[189usize]) * ((next_base_row[35usize]) + (node_215)));
        let node_496 = (((((current_base_row[185usize]) * (node_224))
            + ((current_base_row[186usize]) * ((next_base_row[31usize]) + (node_227))))
            + ((current_base_row[187usize]) * (node_338)))
            + ((current_base_row[188usize]) * ((next_base_row[31usize]) + (node_235))))
            + ((current_base_row[189usize]) * (node_421));
        let node_808 = (((((current_base_row[185usize]) * (node_555))
            + ((current_base_row[186usize]) * ((next_base_row[33usize]) + (node_219))))
            + ((current_base_row[187usize]) * ((next_base_row[34usize]) + (node_219))))
            + ((current_base_row[188usize]) * ((next_base_row[35usize]) + (node_219))))
            + ((current_base_row[189usize]) * ((next_base_row[36usize]) + (node_219)));
        let node_501 = (((((current_base_row[185usize]) * (node_228))
            + ((current_base_row[186usize]) * ((next_base_row[32usize]) + (node_231))))
            + ((current_base_row[187usize]) * (node_339)))
            + ((current_base_row[188usize]) * ((next_base_row[32usize]) + (node_239))))
            + ((current_base_row[189usize]) * (node_422));
        let node_813 = (((((current_base_row[185usize]) * (node_556))
            + ((current_base_row[186usize]) * ((next_base_row[34usize]) + (node_223))))
            + ((current_base_row[187usize]) * ((next_base_row[35usize]) + (node_223))))
            + ((current_base_row[188usize]) * ((next_base_row[36usize]) + (node_223))))
            + ((current_base_row[189usize]) * ((next_base_row[37usize]) + (node_223)));
        let node_505 = ((((current_base_row[185usize]) * (node_232))
            + ((current_base_row[186usize]) * ((next_base_row[33usize]) + (node_235))))
            + ((current_base_row[187usize]) * (node_340)))
            + ((current_base_row[188usize]) * ((next_base_row[33usize]) + (node_243)));
        let node_817 = ((((current_base_row[185usize]) * (node_557))
            + ((current_base_row[186usize]) * ((next_base_row[35usize]) + (node_227))))
            + ((current_base_row[187usize]) * ((next_base_row[36usize]) + (node_227))))
            + ((current_base_row[188usize]) * ((next_base_row[37usize]) + (node_227)));
        let node_508 = (((current_base_row[185usize]) * (node_236))
            + ((current_base_row[186usize]) * ((next_base_row[34usize]) + (node_239))))
            + ((current_base_row[187usize]) * (node_341));
        let node_820 = (((current_base_row[185usize]) * (node_558))
            + ((current_base_row[186usize]) * ((next_base_row[36usize]) + (node_231))))
            + ((current_base_row[187usize]) * ((next_base_row[37usize]) + (node_231)));
        let node_510 = ((current_base_row[185usize]) * (node_240))
            + ((current_base_row[186usize]) * ((next_base_row[35usize]) + (node_243)));
        let node_822 = ((current_base_row[185usize]) * (node_559))
            + ((current_base_row[186usize]) * ((next_base_row[37usize]) + (node_235)));
        let node_270 = (current_base_row[185usize]) * (node_244);
        let node_589 = (current_base_row[185usize]) * (node_560);
        let node_5166 = ((next_ext_row[11usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((challenges[SpongeIndeterminate]) * (current_ext_row[11usize]))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((challenges[HashCIWeight]) * (current_base_row[10usize])));
        let node_5193 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * ((challenges[U32CiWeight]) * (current_base_row[10usize]));
        let node_5233 = ((node_5229)
            * (((((challenges[U32Indeterminate])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((challenges[U32LhsWeight]) * (current_base_row[22usize]))))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((challenges[U32RhsWeight]) * (current_base_row[23usize]))))
                + (node_5193))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((challenges[U32ResultWeight]) * (next_base_row[22usize])))))
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_5197 = (challenges[U32Indeterminate])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((challenges[U32LhsWeight]) * (current_base_row[22usize])));
        let node_5277 = ((next_base_row[48usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[48usize])))
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_5276 = (next_base_row[48usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[48usize]));
        let node_5283 = (next_base_row[47usize])
            + (BFieldElement::from_raw_u64(18446744060824649731u64));
        let node_5309 = (next_ext_row[15usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[15usize]));
        let node_5346 = (next_base_row[51usize])
            + (BFieldElement::from_raw_u64(18446744060824649731u64));
        let node_5432 = (next_ext_row[21usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[21usize]));
        let node_5460 = (next_base_row[59usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[59usize]));
        let node_6394 = (current_base_row[62usize])
            + (BFieldElement::from_raw_u64(18446744056529682436u64));
        let node_6462 = (next_base_row[64usize])
            + (BFieldElement::from_raw_u64(18446744047939747846u64));
        let node_6513 = (next_base_row[63usize])
            + (BFieldElement::from_raw_u64(18446743992105173011u64));
        let node_6516 = (next_base_row[63usize])
            + (BFieldElement::from_raw_u64(18446743923385696291u64));
        let node_6635 = (next_ext_row[28usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[28usize]));
        let node_6656 = (next_ext_row[29usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[29usize]));
        let node_6673 = (next_ext_row[30usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[30usize]));
        let node_6690 = (next_ext_row[31usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[31usize]));
        let node_6707 = (next_ext_row[32usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[32usize]));
        let node_6724 = (next_ext_row[33usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[33usize]));
        let node_6741 = (next_ext_row[34usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[34usize]));
        let node_6758 = (next_ext_row[35usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[35usize]));
        let node_6775 = (next_ext_row[36usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[36usize]));
        let node_6792 = (next_ext_row[37usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[37usize]));
        let node_6809 = (next_ext_row[38usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[38usize]));
        let node_6826 = (next_ext_row[39usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[39usize]));
        let node_6843 = (next_ext_row[40usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[40usize]));
        let node_6860 = (next_ext_row[41usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[41usize]));
        let node_6877 = (next_ext_row[42usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[42usize]));
        let node_6894 = (next_ext_row[43usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[43usize]));
        let node_6920 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (next_base_row[129usize]));
        let node_7022 = (current_base_row[142usize])
            + (BFieldElement::from_raw_u64(18446743940565565471u64));
        let node_7049 = (node_7028)
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_7058 = (next_base_row[142usize])
            + (BFieldElement::from_raw_u64(18446744017874976781u64));
        let node_6421 = (next_base_row[62usize])
            + (BFieldElement::from_raw_u64(18446744060824649731u64));
        let node_30 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[3usize]);
        let node_48 = (next_base_row[6usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_52 = (next_ext_row[0usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[0usize]));
        let node_31 = (BFieldElement::from_raw_u64(38654705655u64)) + (node_30);
        let node_75 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (next_base_row[1usize]);
        let node_91 = (BFieldElement::from_raw_u64(38654705655u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (next_base_row[3usize]));
        let node_89 = (next_ext_row[2usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[2usize]));
        let node_931 = (next_base_row[10usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[10usize]));
        let node_128 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[21usize]);
        let node_1289 = (next_base_row[22usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[335usize]));
        let node_1311 = (current_base_row[23usize]) * (next_base_row[23usize]);
        let node_1314 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (next_base_row[22usize]);
        let node_1319 = (current_base_row[22usize]) * (current_base_row[25usize]);
        let node_1320 = (current_base_row[24usize]) * (current_base_row[26usize]);
        let node_1323 = (current_base_row[23usize]) * (current_base_row[27usize]);
        let node_1346 = (current_base_row[24usize]) * (next_base_row[23usize]);
        let node_1349 = (current_base_row[23usize]) * (next_base_row[24usize]);
        let node_575 = (current_base_row[185usize]) * (node_546);
        let node_113 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[9usize]);
        let node_124 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[20usize]);
        let node_1292 = (current_base_row[23usize]) + (node_545);
        let node_1331 = (current_base_row[24usize]) * (current_base_row[27usize]);
        let node_1356 = (current_base_row[24usize]) * (next_base_row[24usize]);
        let node_618 = (current_base_row[186usize])
            * ((next_base_row[24usize]) + (node_545));
        let node_250 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[40usize]);
        let node_145 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * ((BFieldElement::from_raw_u64(34359738360u64))
                * (current_base_row[42usize]));
        let node_725 = (next_base_row[27usize]) + (node_545);
        let node_660 = (current_base_row[187usize])
            * ((next_base_row[25usize]) + (node_545));
        let node_726 = (next_base_row[28usize]) + (node_187);
        let node_700 = (current_base_row[188usize])
            * ((next_base_row[26usize]) + (node_545));
        let node_727 = (next_base_row[29usize]) + (node_191);
        let node_728 = (next_base_row[30usize]) + (node_195);
        let node_827 = (current_base_row[312usize])
            * ((next_base_row[22usize]) + (node_207));
        let node_729 = (next_base_row[31usize]) + (node_199);
        let node_834 = (next_base_row[22usize]) + (node_223);
        let node_854 = (next_base_row[32usize]) + (node_545);
        let node_829 = (current_base_row[270usize])
            * ((next_base_row[22usize]) + (node_211));
        let node_731 = (next_base_row[33usize]) + (node_207);
        let node_831 = (current_base_row[271usize])
            * ((next_base_row[22usize]) + (node_215));
        let node_732 = (next_base_row[34usize]) + (node_211);
        let node_833 = (current_base_row[314usize])
            * ((next_base_row[22usize]) + (node_219));
        let node_733 = (next_base_row[35usize]) + (node_215);
        let node_835 = (current_base_row[317usize]) * (node_834);
        let node_734 = (next_base_row[36usize]) + (node_219);
        let node_451 = ((((node_256) + (node_302)) + (node_345)) + (node_387))
            + (current_base_row[292usize]);
        let node_763 = ((((node_575) + (node_618)) + (node_660)) + (node_700))
            + (current_base_row[293usize]);
        let node_837 = (current_base_row[318usize])
            * ((next_base_row[22usize]) + (node_227));
        let node_735 = (next_base_row[37usize]) + (node_223);
        let node_412 = (next_base_row[22usize]) + (node_203);
        let node_839 = (current_base_row[273usize])
            * ((next_base_row[22usize]) + (node_231));
        let node_713 = (node_159)
            + (BFieldElement::from_raw_u64(18446744047939747846u64));
        let node_413 = (next_base_row[23usize]) + (node_207);
        let node_841 = (current_base_row[329usize])
            * ((next_base_row[22usize]) + (node_235));
        let node_724 = (next_ext_row[6usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[72usize]));
        let node_414 = (next_base_row[24usize]) + (node_211);
        let node_843 = (current_base_row[330usize])
            * ((next_base_row[22usize]) + (node_239));
        let node_415 = (next_base_row[25usize]) + (node_215);
        let node_845 = (current_base_row[331usize])
            * ((next_base_row[22usize]) + (node_243));
        let node_416 = (next_base_row[26usize]) + (node_219);
        let node_1435 = ((challenges[StandardOutputIndeterminate])
            * (((challenges[StandardOutputIndeterminate])
                * (((challenges[StandardOutputIndeterminate])
                    * (((challenges[StandardOutputIndeterminate])
                        * (current_ext_row[4usize])) + (current_base_row[22usize])))
                    + (current_base_row[23usize]))) + (current_base_row[24usize])))
            + (current_base_row[25usize]);
        let node_1430 = ((challenges[StandardOutputIndeterminate])
            * (((challenges[StandardOutputIndeterminate])
                * (((challenges[StandardOutputIndeterminate])
                    * (current_ext_row[4usize])) + (current_base_row[22usize])))
                + (current_base_row[23usize]))) + (current_base_row[24usize]);
        let node_1425 = ((challenges[StandardOutputIndeterminate])
            * (((challenges[StandardOutputIndeterminate]) * (current_ext_row[4usize]))
                + (current_base_row[22usize]))) + (current_base_row[23usize]);
        let node_1420 = ((challenges[StandardOutputIndeterminate])
            * (current_ext_row[4usize])) + (current_base_row[22usize]);
        let node_5017 = (next_ext_row[5usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[5usize]));
        let node_5106 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (((((((((((challenges[HashStateWeight0]) * (next_base_row[22usize]))
                + ((challenges[HashStateWeight1]) * (next_base_row[23usize])))
                + ((challenges[HashStateWeight2]) * (next_base_row[24usize])))
                + ((challenges[HashStateWeight3]) * (next_base_row[25usize])))
                + ((challenges[HashStateWeight4]) * (next_base_row[26usize])))
                + ((challenges[HashStateWeight5]) * (next_base_row[27usize])))
                + ((challenges[HashStateWeight6]) * (next_base_row[28usize])))
                + ((challenges[HashStateWeight7]) * (next_base_row[29usize])))
                + ((challenges[HashStateWeight8]) * (next_base_row[30usize])))
                + ((challenges[HashStateWeight9]) * (next_base_row[31usize])));
        let node_5089 = (((((challenges[HashStateWeight0]) * (next_base_row[22usize]))
            + ((challenges[HashStateWeight1]) * (next_base_row[23usize])))
            + ((challenges[HashStateWeight2]) * (next_base_row[24usize])))
            + ((challenges[HashStateWeight3]) * (next_base_row[25usize])))
            + ((challenges[HashStateWeight4]) * (next_base_row[26usize]));
        let node_5186 = (challenges[U32Indeterminate])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((challenges[U32LhsWeight]) * (next_base_row[22usize])));
        let node_5189 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * ((challenges[U32RhsWeight]) * (next_base_row[23usize]));
        let node_5200 = (node_5197)
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((challenges[U32RhsWeight]) * (current_base_row[23usize])));
        let node_5237 = ((node_5229)
            * (((node_5197) + (node_5193))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((challenges[U32ResultWeight]) * (next_base_row[22usize])))))
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_5223 = (((node_5186)
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((challenges[U32RhsWeight]) * (current_base_row[23usize]))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((challenges[U32CiWeight])
                    * (BFieldElement::from_raw_u64(25769803770u64)))))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (challenges[U32ResultWeight]));
        let node_5227 = ((node_5197) + (node_5189))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * ((challenges[U32CiWeight])
                    * (BFieldElement::from_raw_u64(17179869180u64))));
        let node_5300 = (next_base_row[47usize])
            * ((next_base_row[47usize])
                + (BFieldElement::from_raw_u64(18446744065119617026u64)));
        let node_5371 = (challenges[RamTableBezoutRelationIndeterminate])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (next_base_row[52usize]));
        let node_5376 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_ext_row[16usize]);
        let node_5423 = ((next_base_row[51usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64)))
            * (next_base_row[51usize]);
        let node_5465 = (node_5461)
            * ((current_base_row[58usize])
                + (BFieldElement::from_raw_u64(18446744000695107601u64)));
        let node_5473 = (next_base_row[57usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[57usize]));
        let node_5464 = (current_base_row[58usize])
            + (BFieldElement::from_raw_u64(18446744000695107601u64));
        let node_5495 = (next_ext_row[23usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[23usize]));
        let node_6374 = (current_base_row[63usize])
            + (BFieldElement::from_raw_u64(18446743897615892521u64));
        let node_6376 = (current_base_row[64usize])
            + (BFieldElement::from_raw_u64(18446744047939747846u64));
        let node_6615 = (next_ext_row[24usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[24usize]));
        let node_5532 = (((((current_base_row[65usize])
            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
            + ((current_base_row[66usize])
                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
            + ((current_base_row[67usize])
                * (BFieldElement::from_raw_u64(281474976645120u64))))
            + (current_base_row[68usize])) * (BFieldElement::from_raw_u64(1u64));
        let node_5543 = (((((current_base_row[69usize])
            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
            + ((current_base_row[70usize])
                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
            + ((current_base_row[71usize])
                * (BFieldElement::from_raw_u64(281474976645120u64))))
            + (current_base_row[72usize])) * (BFieldElement::from_raw_u64(1u64));
        let node_5554 = (((((current_base_row[73usize])
            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
            + ((current_base_row[74usize])
                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
            + ((current_base_row[75usize])
                * (BFieldElement::from_raw_u64(281474976645120u64))))
            + (current_base_row[76usize])) * (BFieldElement::from_raw_u64(1u64));
        let node_5565 = (((((current_base_row[77usize])
            * (BFieldElement::from_raw_u64(18446744069414518785u64)))
            + ((current_base_row[78usize])
                * (BFieldElement::from_raw_u64(18446744069414584320u64))))
            + ((current_base_row[79usize])
                * (BFieldElement::from_raw_u64(281474976645120u64))))
            + (current_base_row[80usize])) * (BFieldElement::from_raw_u64(1u64));
        let node_6410 = (node_6376) * (node_6374);
        let node_6425 = ((current_base_row[62usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64)))
            * ((current_base_row[62usize])
                + (BFieldElement::from_raw_u64(18446744060824649731u64)));
        let node_6443 = (challenges[HashStateWeight10])
            * ((next_base_row[103usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (current_base_row[103usize])));
        let node_6444 = (challenges[HashStateWeight11])
            * ((next_base_row[104usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (current_base_row[104usize])));
        let node_6446 = (challenges[HashStateWeight12])
            * ((next_base_row[105usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (current_base_row[105usize])));
        let node_6448 = (challenges[HashStateWeight13])
            * ((next_base_row[106usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (current_base_row[106usize])));
        let node_6450 = (challenges[HashStateWeight14])
            * ((next_base_row[107usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (current_base_row[107usize])));
        let node_6452 = (challenges[HashStateWeight15])
            * ((next_base_row[108usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (current_base_row[108usize])));
        let node_6544 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (((((((((((challenges[HashStateWeight0]) * (node_6264))
                + ((challenges[HashStateWeight1]) * (node_6275)))
                + ((challenges[HashStateWeight2]) * (node_6286)))
                + ((challenges[HashStateWeight3]) * (node_6297)))
                + ((challenges[HashStateWeight4]) * (next_base_row[97usize])))
                + ((challenges[HashStateWeight5]) * (next_base_row[98usize])))
                + ((challenges[HashStateWeight6]) * (next_base_row[99usize])))
                + ((challenges[HashStateWeight7]) * (next_base_row[100usize])))
                + ((challenges[HashStateWeight8]) * (next_base_row[101usize])))
                + ((challenges[HashStateWeight9]) * (next_base_row[102usize])));
        let node_6521 = (next_ext_row[25usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[25usize]));
        let node_6530 = (((((challenges[HashStateWeight0]) * (node_6264))
            + ((challenges[HashStateWeight1]) * (node_6275)))
            + ((challenges[HashStateWeight2]) * (node_6286)))
            + ((challenges[HashStateWeight3]) * (node_6297)))
            + ((challenges[HashStateWeight4]) * (next_base_row[97usize]));
        let node_6555 = (next_ext_row[26usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[26usize]));
        let node_6581 = (next_ext_row[27usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[27usize]));
        let node_6584 = (next_base_row[63usize])
            + (BFieldElement::from_raw_u64(18446743863256154161u64));
        let node_6930 = (next_ext_row[44usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[44usize]));
        let node_6946 = (next_ext_row[45usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[45usize]));
        let node_6941 = ((challenges[LookupTableInputWeight])
            * (next_base_row[131usize]))
            + ((challenges[LookupTableOutputWeight]) * (next_base_row[133usize]));
        let node_6944 = ((challenges[LookupTableInputWeight])
            * (next_base_row[130usize]))
            + ((challenges[LookupTableOutputWeight]) * (next_base_row[132usize]));
        let node_6984 = (next_ext_row[46usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[46usize]));
        let node_7040 = ((next_base_row[140usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[140usize])))
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_7046 = (node_7025)
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_7067 = (next_base_row[147usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_7069 = (next_base_row[147usize])
            + (BFieldElement::from_raw_u64(18446744060824649731u64));
        let node_7072 = (current_base_row[309usize]) * (next_base_row[147usize]);
        let node_7074 = (current_base_row[147usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_7038 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (current_base_row[140usize]);
        let node_7133 = (next_base_row[147usize]) * (next_base_row[147usize]);
        let node_7083 = (BFieldElement::from_raw_u64(18446744065119617026u64))
            * (node_7025);
        let node_7149 = (next_ext_row[48usize])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_ext_row[48usize]));
        let node_246 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[42usize]));
        let node_248 = (BFieldElement::from_raw_u64(4294967295u64))
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (current_base_row[41usize]));
        let node_1493 = (current_base_row[12usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_1474 = (current_base_row[13usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_134 = (current_base_row[40usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_6454 = (next_base_row[64usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_6456 = (next_base_row[64usize])
            + (BFieldElement::from_raw_u64(18446744060824649731u64));
        let node_6458 = (next_base_row[64usize])
            + (BFieldElement::from_raw_u64(18446744056529682436u64));
        let node_7051 = (next_base_row[142usize])
            + (BFieldElement::from_raw_u64(18446744052234715141u64));
        let node_7054 = (next_base_row[142usize])
            + (BFieldElement::from_raw_u64(18446744009285042191u64));
        let node_6460 = (next_base_row[64usize])
            + (BFieldElement::from_raw_u64(18446744052234715141u64));
        let node_6392 = (current_base_row[62usize])
            + (BFieldElement::from_raw_u64(18446744060824649731u64));
        let node_6417 = (current_base_row[62usize])
            + (BFieldElement::from_raw_u64(18446744065119617026u64));
        let node_178 = (challenges[OpStackIndeterminate])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (((node_169)
                    + ((challenges[OpStackPointerWeight]) * (next_base_row[38usize])))
                    + ((challenges[OpStackFirstUnderflowElementWeight])
                        * (next_base_row[37usize]))));
        let node_568 = (challenges[OpStackIndeterminate])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (((node_169)
                    + ((challenges[OpStackPointerWeight]) * (current_base_row[38usize])))
                    + ((challenges[OpStackFirstUnderflowElementWeight])
                        * (current_base_row[37usize]))));
        let node_1006 = (challenges[RamIndeterminate])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (((node_997)
                    + (((next_base_row[22usize])
                        + (BFieldElement::from_raw_u64(4294967295u64)))
                        * (challenges[RamPointerWeight])))
                    + ((next_base_row[23usize]) * (challenges[RamValueWeight]))));
        let node_1088 = (challenges[RamIndeterminate])
            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                * (((node_994)
                    + ((current_base_row[22usize]) * (challenges[RamPointerWeight])))
                    + ((current_base_row[23usize]) * (challenges[RamValueWeight]))));
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
            ((current_base_row[5usize]) * (node_34)) * (node_48),
            ((next_base_row[7usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (current_base_row[7usize])))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)),
            (current_base_row[8usize])
                * ((next_base_row[8usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[8usize]))),
            ((((((((((((((((((((((((((((((((((((((((current_base_row[202usize])
                * (node_121))
                + ((current_base_row[212usize])
                    * ((next_base_row[22usize]) + (node_542))))
                + ((current_base_row[203usize]) * (node_121)))
                + ((current_base_row[208usize])
                    * ((current_base_row[315usize]) * (node_824))))
                + ((current_base_row[205usize]) * (current_base_row[315usize])))
                + ((current_base_row[214usize]) * (node_931)))
                + ((current_base_row[216usize]) * (node_121)))
                + ((current_base_row[213usize])
                    * ((node_934) * (current_base_row[39usize]))))
                + ((current_base_row[220usize])
                    * ((node_121)
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)))))
                + ((current_base_row[225usize])
                    * ((next_base_row[19usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((current_base_row[19usize])
                                + (BFieldElement::from_raw_u64(
                                    18446744065119617026u64,
                                )))))))
                + ((current_base_row[223usize])
                    * ((next_base_row[9usize]) + (node_128))))
                + ((current_base_row[224usize])
                    * ((current_base_row[22usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)))))
                + ((current_base_row[215usize]) * (node_121)))
                + ((current_base_row[221usize]) * (node_121)))
                + ((current_base_row[240usize]) * (node_121)))
                + ((current_base_row[232usize]) * (node_132)))
                + ((current_base_row[228usize])
                    * ((current_base_row[27usize]) + (node_545))))
                + ((current_base_row[229usize]) * (node_121)))
                + ((current_base_row[244usize]) * (node_121)))
                + ((current_base_row[246usize]) * (node_121)))
                + ((current_base_row[233usize]) * ((node_824) + (node_187))))
                + ((current_base_row[234usize]) * (node_1289)))
                + ((current_base_row[235usize])
                    * ((current_base_row[336usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)))))
                + ((current_base_row[237usize])
                    * ((current_base_row[39usize]) * (node_1295))))
                + ((current_base_row[236usize])
                    * ((current_base_row[22usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((BFieldElement::from_raw_u64(18446744069414584320u64))
                                * (next_base_row[23usize])) + (next_base_row[22usize]))))))
                + ((current_base_row[238usize]) * (node_121)))
                + ((current_base_row[239usize]) * (node_121)))
                + ((current_base_row[241usize]) * (node_121)))
                + ((current_base_row[242usize]) * (node_121)))
                + ((current_base_row[245usize]) * (node_121)))
                + ((current_base_row[248usize])
                    * (((current_base_row[22usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_1311))) + (node_1314))))
                + ((current_base_row[251usize]) * (node_121)))
                + ((current_base_row[274usize]) * ((node_824) + (node_195))))
                + ((current_base_row[275usize])
                    * ((next_base_row[22usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((node_1319)
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (node_1320)))
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (node_1323)))))))
                + ((current_base_row[277usize])
                    * ((((current_base_row[336usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_1346)))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_1349)))
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)))))
                + ((current_base_row[279usize]) * (node_1289)))
                + ((current_base_row[259usize]) * (node_121)))
                + ((current_base_row[263usize]) * (node_121))) * (node_4849))
                + ((node_114) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((current_base_row[202usize])
                * (node_125)) + ((current_base_row[212usize]) * (node_546)))
                + ((current_base_row[203usize]) * (node_125)))
                + ((current_base_row[208usize]) * (node_256)))
                + ((current_base_row[205usize]) * (node_575)))
                + ((current_base_row[214usize]) * (node_121)))
                + ((current_base_row[216usize]) * (node_125)))
                + ((current_base_row[213usize])
                    * ((node_934) * (current_base_row[22usize]))))
                + ((current_base_row[220usize])
                    * (((next_base_row[20usize]) + (node_113))
                        + (BFieldElement::from_raw_u64(18446744060824649731u64)))))
                + ((current_base_row[225usize])
                    * ((next_base_row[9usize]) + (node_124))))
                + ((current_base_row[223usize]) * (node_121)))
                + ((current_base_row[224usize]) * (node_121)))
                + ((current_base_row[215usize]) * (node_125)))
                + ((current_base_row[221usize]) * (node_125)))
                + ((current_base_row[240usize]) * (node_125)))
                + ((current_base_row[232usize])
                    * ((((next_base_row[32usize])
                        * (BFieldElement::from_raw_u64(8589934590u64)))
                        + (current_base_row[39usize])) + (node_203))))
                + ((current_base_row[228usize])
                    * ((current_base_row[28usize]) + (node_187))))
                + ((current_base_row[229usize]) * (node_125)))
                + ((current_base_row[244usize]) * (node_125)))
                + ((current_base_row[246usize]) * (node_125)))
                + ((current_base_row[233usize]) * (node_121)))
                + ((current_base_row[234usize]) * (node_121)))
                + ((current_base_row[235usize]) * (node_121)))
                + ((current_base_row[237usize]) * ((node_1292) * (node_1295))))
                + (current_base_row[340usize]))
                + ((current_base_row[238usize]) * (node_125)))
                + ((current_base_row[239usize]) * (node_125)))
                + ((current_base_row[241usize]) * (node_125)))
                + ((current_base_row[242usize]) * (node_125)))
                + ((current_base_row[245usize]) * (node_125)))
                + ((current_base_row[248usize]) * (node_872)))
                + ((current_base_row[251usize]) * (node_125)))
                + ((current_base_row[274usize]) * ((node_868) + (node_199))))
                + ((current_base_row[275usize])
                    * ((next_base_row[23usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((((((current_base_row[23usize])
                                * (current_base_row[25usize]))
                                + ((current_base_row[22usize])
                                    * (current_base_row[26usize])))
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (node_1331))) + (node_1320)) + (node_1323))))))
                + ((current_base_row[277usize])
                    * ((((((current_base_row[23usize]) * (next_base_row[22usize]))
                        + ((current_base_row[22usize]) * (next_base_row[23usize])))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_1356))) + (node_1346)) + (node_1349))))
                + ((current_base_row[279usize])
                    * ((next_base_row[23usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((current_base_row[22usize])
                                * (current_base_row[24usize]))))))
                + ((current_base_row[259usize]) * (node_125)))
                + ((current_base_row[263usize]) * (node_125))) * (node_4849))
                + ((node_931) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((current_base_row[202usize])
                * (node_129)) + ((current_base_row[212usize]) * (node_547)))
                + ((current_base_row[203usize]) * (node_129)))
                + ((current_base_row[208usize]) * (node_302)))
                + ((current_base_row[205usize]) * (node_618)))
                + ((current_base_row[214usize]) * (node_125)))
                + ((current_base_row[216usize]) * (node_129)))
                + ((current_base_row[213usize])
                    * ((((((current_base_row[11usize]) + (node_250))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((BFieldElement::from_raw_u64(8589934590u64))
                                * (current_base_row[41usize])))) + (node_145))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((BFieldElement::from_raw_u64(137438953440u64))
                                * (current_base_row[43usize]))))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((BFieldElement::from_raw_u64(549755813760u64))
                                * (current_base_row[44usize]))))))
                + ((current_base_row[220usize])
                    * ((next_base_row[21usize]) + (node_542))))
                + ((current_base_row[225usize]) * (node_824)))
                + ((current_base_row[223usize]) * (node_125)))
                + ((current_base_row[224usize]) * (node_125)))
                + ((current_base_row[215usize]) * (node_129)))
                + ((current_base_row[221usize]) * (node_129)))
                + ((current_base_row[240usize]) * (node_129)))
                + ((current_base_row[232usize])
                    * (((node_298) * (node_824))
                        + ((current_base_row[39usize]) * (node_725)))))
                + ((current_base_row[228usize])
                    * ((current_base_row[29usize]) + (node_191))))
                + ((current_base_row[229usize]) * (node_129)))
                + ((current_base_row[244usize]) * (node_129)))
                + ((current_base_row[246usize]) * (node_129)))
                + ((current_base_row[233usize]) * (node_125)))
                + ((current_base_row[234usize]) * (node_125)))
                + ((current_base_row[235usize]) * (node_125)))
                + ((current_base_row[237usize])
                    * ((next_base_row[22usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_1295)))))
                + ((current_base_row[236usize]) * (node_547)))
                + ((current_base_row[238usize]) * (node_129)))
                + ((current_base_row[239usize]) * (node_129)))
                + ((current_base_row[241usize]) * (node_129)))
                + ((current_base_row[242usize]) * (node_129)))
                + ((current_base_row[245usize]) * (node_129)))
                + ((current_base_row[248usize]) * (node_121)))
                + ((current_base_row[251usize]) * (node_129)))
                + ((current_base_row[274usize]) * ((node_872) + (node_203))))
                + ((current_base_row[275usize])
                    * ((next_base_row[24usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((current_base_row[24usize])
                                * (current_base_row[25usize]))
                                + ((current_base_row[23usize])
                                    * (current_base_row[26usize])))
                                + ((current_base_row[22usize])
                                    * (current_base_row[27usize]))) + (node_1331))))))
                + ((current_base_row[277usize])
                    * (((((current_base_row[24usize]) * (next_base_row[22usize]))
                        + (node_1311))
                        + ((current_base_row[22usize]) * (next_base_row[24usize])))
                        + (node_1356))))
                + ((current_base_row[279usize])
                    * ((next_base_row[24usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_1319)))))
                + ((current_base_row[259usize]) * (node_129)))
                + ((current_base_row[263usize]) * (node_129))) * (node_4849))
                + (((next_base_row[11usize]) + (node_542)) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((current_base_row[202usize])
                * (node_117)) + ((current_base_row[212usize]) * (node_548)))
                + ((current_base_row[203usize]) * (node_117)))
                + ((current_base_row[208usize]) * (node_345)))
                + ((current_base_row[205usize]) * (node_660)))
                + ((current_base_row[214usize]) * (node_129)))
                + ((current_base_row[216usize]) * (node_932)))
                + (current_base_row[341usize]))
                + ((current_base_row[220usize])
                    * ((next_base_row[9usize]) + (node_542))))
                + ((current_base_row[225usize]) * (node_868)))
                + ((current_base_row[223usize]) * (node_129)))
                + ((current_base_row[224usize]) * (node_129)))
                + ((current_base_row[215usize]) * (node_117)))
                + ((current_base_row[221usize]) * (node_117)))
                + ((current_base_row[240usize]) * (node_932)))
                + ((current_base_row[232usize])
                    * (((node_298) * (node_868))
                        + ((current_base_row[39usize]) * (node_726)))))
                + ((current_base_row[228usize])
                    * ((current_base_row[30usize]) + (node_195))))
                + ((current_base_row[229usize]) * (node_932)))
                + ((current_base_row[244usize]) * (node_932)))
                + ((current_base_row[246usize]) * (node_932)))
                + ((current_base_row[233usize]) * (node_129)))
                + ((current_base_row[234usize]) * (node_129)))
                + ((current_base_row[235usize]) * (node_129)))
                + ((current_base_row[237usize]) * (node_121)))
                + ((current_base_row[236usize]) * (node_548)))
                + ((current_base_row[238usize]) * (node_932)))
                + ((current_base_row[239usize]) * (node_932)))
                + ((current_base_row[241usize]) * (node_932)))
                + ((current_base_row[242usize]) * (node_932)))
                + ((current_base_row[245usize]) * (node_932)))
                + ((current_base_row[248usize]) * (node_125)))
                + ((current_base_row[251usize]) * (node_932)))
                + ((current_base_row[274usize]) * (node_332)))
                + ((current_base_row[275usize]) * (node_332)))
                + ((current_base_row[277usize]) * (node_876)))
                + ((current_base_row[279usize]) * (node_200)))
                + ((current_base_row[259usize]) * (node_117)))
                + ((current_base_row[263usize]) * (node_117))) * (node_4849))
                + ((node_121) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((current_base_row[202usize])
                * (node_132)) + ((current_base_row[212usize]) * (node_549)))
                + ((current_base_row[203usize]) * (node_132)))
                + ((current_base_row[208usize]) * (node_387)))
                + ((current_base_row[205usize]) * (node_700)))
                + ((current_base_row[214usize]) * (node_932)))
                + ((current_base_row[216usize]) * (node_824)))
                + ((current_base_row[213usize]) * (current_base_row[268usize])))
                + ((current_base_row[220usize]) * (node_824)))
                + ((current_base_row[225usize]) * (node_872)))
                + ((current_base_row[223usize]) * (node_824)))
                + ((current_base_row[224usize]) * (node_932)))
                + ((current_base_row[215usize]) * (node_132)))
                + ((current_base_row[221usize]) * (node_132)))
                + ((current_base_row[240usize]) * (node_417)))
                + ((current_base_row[232usize])
                    * (((node_298) * (node_872))
                        + ((current_base_row[39usize]) * (node_727)))))
                + ((current_base_row[228usize])
                    * ((current_base_row[31usize]) + (node_199))))
                + ((current_base_row[229usize]) * (node_824)))
                + ((current_base_row[244usize])
                    * ((node_159) + (BFieldElement::from_raw_u64(42949672950u64)))))
                + ((current_base_row[246usize])
                    * ((node_159)
                        + (BFieldElement::from_raw_u64(18446744026464911371u64)))))
                + ((current_base_row[233usize]) * (node_932)))
                + ((current_base_row[234usize]) * (node_932)))
                + ((current_base_row[235usize]) * (node_932)))
                + ((current_base_row[237usize]) * (node_125)))
                + ((current_base_row[236usize]) * (node_549)))
                + ((current_base_row[238usize]) * (node_192)))
                + ((current_base_row[239usize]) * (node_192)))
                + ((current_base_row[241usize]) * (node_192)))
                + ((current_base_row[242usize]) * (node_868)))
                + ((current_base_row[245usize]) * (node_192)))
                + ((current_base_row[248usize]) * (node_129)))
                + ((current_base_row[251usize]) * (node_868)))
                + ((current_base_row[274usize]) * (node_333)))
                + ((current_base_row[275usize]) * (node_333)))
                + ((current_base_row[277usize]) * (node_880)))
                + ((current_base_row[279usize]) * (node_204)))
                + ((current_base_row[259usize]) * (node_132)))
                + ((current_base_row[263usize]) * (node_132))) * (node_4849))
                + ((node_125) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((current_base_row[202usize])
                * (current_base_row[260usize]))
                + ((current_base_row[212usize]) * (node_551)))
                + ((current_base_row[203usize]) * (current_base_row[260usize])))
                + ((current_base_row[208usize]) * (node_827)))
                + ((current_base_row[205usize])
                    * ((current_base_row[312usize])
                        * ((next_base_row[28usize]) + (node_545)))))
                + ((current_base_row[214usize]) * (node_868)))
                + ((current_base_row[216usize]) * (node_872)))
                + (current_base_row[343usize]))
                + ((current_base_row[220usize]) * (node_872)))
                + ((current_base_row[225usize]) * (node_880)))
                + ((current_base_row[223usize]) * (node_872)))
                + ((current_base_row[224usize]) * (node_192)))
                + ((current_base_row[215usize]) * (current_base_row[260usize])))
                + ((current_base_row[221usize]) * (current_base_row[260usize])))
                + ((current_base_row[240usize]) * (node_419)))
                + ((current_base_row[232usize])
                    * (((node_298) * (node_880))
                        + ((current_base_row[39usize]) * (node_729)))))
                + ((current_base_row[228usize]) * (node_125)))
                + ((current_base_row[229usize]) * (node_872)))
                + ((current_base_row[244usize]) * (node_834)))
                + ((current_base_row[246usize]) * (node_854)))
                + ((current_base_row[233usize]) * (node_196)))
                + ((current_base_row[234usize]) * (node_196)))
                + ((current_base_row[235usize]) * (node_872)))
                + ((current_base_row[237usize]) * (node_932)))
                + ((current_base_row[236usize]) * (node_551)))
                + ((current_base_row[238usize]) * (node_200)))
                + ((current_base_row[239usize]) * (node_200)))
                + ((current_base_row[241usize]) * (node_200)))
                + ((current_base_row[242usize]) * (node_876)))
                + ((current_base_row[245usize]) * (node_200)))
                + ((current_base_row[248usize]) * (node_876)))
                + ((current_base_row[251usize]) * (node_876)))
                + ((current_base_row[274usize]) * (node_335)))
                + ((current_base_row[275usize]) * (node_335)))
                + ((current_base_row[277usize]) * (node_888)))
                + ((current_base_row[279usize]) * (node_212)))
                + ((current_base_row[259usize]) * (current_base_row[260usize])))
                + ((current_base_row[263usize]) * (current_base_row[260usize])))
                * (node_4849)) + ((node_824) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((current_base_row[202usize])
                * (current_base_row[261usize]))
                + ((current_base_row[212usize]) * (node_552)))
                + ((current_base_row[203usize]) * (current_base_row[261usize])))
                + ((current_base_row[208usize]) * (node_829)))
                + ((current_base_row[205usize])
                    * ((current_base_row[270usize])
                        * ((next_base_row[29usize]) + (node_545)))))
                + ((current_base_row[214usize]) * (node_872)))
                + ((current_base_row[216usize]) * (node_876)))
                + ((current_base_row[213usize]) * (current_base_row[280usize])))
                + ((current_base_row[220usize]) * (node_876)))
                + ((current_base_row[225usize]) * (node_884)))
                + ((current_base_row[223usize]) * (node_876)))
                + ((current_base_row[224usize]) * (node_196)))
                + ((current_base_row[215usize]) * (current_base_row[261usize])))
                + ((current_base_row[221usize]) * (current_base_row[261usize])))
                + ((current_base_row[240usize]) * (node_420)))
                + ((current_base_row[232usize]) * (node_731)))
                + ((current_base_row[228usize]) * (node_129)))
                + ((current_base_row[229usize]) * (node_876)))
                + ((current_base_row[244usize])
                    * ((next_base_row[23usize]) + (node_227))))
                + ((current_base_row[246usize])
                    * ((next_base_row[33usize]) + (node_187))))
                + ((current_base_row[233usize]) * (node_200)))
                + ((current_base_row[234usize]) * (node_200)))
                + ((current_base_row[235usize]) * (node_876)))
                + ((current_base_row[237usize]) * (node_192)))
                + ((current_base_row[236usize]) * (node_552)))
                + ((current_base_row[238usize]) * (node_204)))
                + ((current_base_row[239usize]) * (node_204)))
                + ((current_base_row[241usize]) * (node_204)))
                + ((current_base_row[242usize]) * (node_880)))
                + ((current_base_row[245usize]) * (node_204)))
                + ((current_base_row[248usize]) * (node_880)))
                + ((current_base_row[251usize]) * (node_880)))
                + ((current_base_row[274usize]) * (node_336)))
                + ((current_base_row[275usize]) * (node_336)))
                + ((current_base_row[277usize]) * (node_892)))
                + ((current_base_row[279usize]) * (node_216)))
                + ((current_base_row[259usize]) * (current_base_row[261usize])))
                + ((current_base_row[263usize]) * (current_base_row[261usize])))
                * (node_4849)) + ((node_868) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((current_base_row[202usize])
                * (node_155)) + ((current_base_row[212usize]) * (node_553)))
                + ((current_base_row[203usize]) * (node_155)))
                + ((current_base_row[208usize]) * (node_831)))
                + ((current_base_row[205usize])
                    * ((current_base_row[271usize])
                        * ((next_base_row[30usize]) + (node_545)))))
                + ((current_base_row[214usize]) * (node_876)))
                + ((current_base_row[216usize]) * (node_880)))
                + ((current_base_row[213usize]) * (current_base_row[281usize])))
                + ((current_base_row[220usize]) * (node_880)))
                + ((current_base_row[225usize]) * (node_888)))
                + ((current_base_row[223usize]) * (node_880)))
                + ((current_base_row[224usize]) * (node_200)))
                + ((current_base_row[215usize]) * (node_155)))
                + ((current_base_row[221usize]) * (node_155)))
                + ((current_base_row[240usize]) * (node_421)))
                + ((current_base_row[232usize]) * (node_732)))
                + ((current_base_row[228usize]) * (node_932)))
                + ((current_base_row[229usize]) * (node_880)))
                + ((current_base_row[244usize])
                    * ((next_base_row[24usize]) + (node_231))))
                + ((current_base_row[246usize])
                    * ((next_base_row[34usize]) + (node_191))))
                + ((current_base_row[233usize]) * (node_204)))
                + ((current_base_row[234usize]) * (node_204)))
                + ((current_base_row[235usize]) * (node_880)))
                + ((current_base_row[237usize]) * (node_196)))
                + ((current_base_row[236usize]) * (node_553)))
                + ((current_base_row[238usize]) * (node_208)))
                + ((current_base_row[239usize]) * (node_208)))
                + ((current_base_row[241usize]) * (node_208)))
                + ((current_base_row[242usize]) * (node_884)))
                + ((current_base_row[245usize]) * (node_208)))
                + ((current_base_row[248usize]) * (node_884)))
                + ((current_base_row[251usize]) * (node_884)))
                + ((current_base_row[274usize]) * (node_337)))
                + ((current_base_row[275usize]) * (node_337)))
                + ((current_base_row[277usize]) * (node_896)))
                + ((current_base_row[279usize]) * (node_220)))
                + ((current_base_row[259usize]) * (node_155)))
                + ((current_base_row[263usize]) * (node_155))) * (node_4849))
                + ((node_872) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((current_base_row[202usize])
                * (node_441)) + ((current_base_row[212usize]) * (node_554)))
                + ((current_base_row[203usize]) * (node_753)))
                + ((current_base_row[208usize]) * (node_833)))
                + ((current_base_row[205usize])
                    * ((current_base_row[314usize])
                        * ((next_base_row[31usize]) + (node_545)))))
                + ((current_base_row[214usize]) * (node_880)))
                + ((current_base_row[216usize]) * (node_884)))
                + ((current_base_row[213usize]) * (node_121)))
                + ((current_base_row[220usize]) * (node_884)))
                + ((current_base_row[225usize]) * (node_892)))
                + ((current_base_row[223usize]) * (node_884)))
                + ((current_base_row[224usize]) * (node_204)))
                + ((current_base_row[215usize]) * (node_753)))
                + ((current_base_row[221usize]) * (node_441)))
                + ((current_base_row[240usize]) * (node_422)))
                + ((current_base_row[232usize]) * (node_733)))
                + ((current_base_row[228usize]) * (node_400)))
                + ((current_base_row[229usize]) * (node_884)))
                + ((current_base_row[244usize])
                    * ((next_base_row[25usize]) + (node_235))))
                + ((current_base_row[246usize])
                    * ((next_base_row[35usize]) + (node_195))))
                + ((current_base_row[233usize]) * (node_208)))
                + ((current_base_row[234usize]) * (node_208)))
                + ((current_base_row[235usize]) * (node_884)))
                + ((current_base_row[237usize]) * (node_200)))
                + ((current_base_row[236usize]) * (node_554)))
                + ((current_base_row[238usize]) * (node_212)))
                + ((current_base_row[239usize]) * (node_212)))
                + ((current_base_row[241usize]) * (node_212)))
                + ((current_base_row[242usize]) * (node_888)))
                + ((current_base_row[245usize]) * (node_212)))
                + ((current_base_row[248usize]) * (node_888)))
                + ((current_base_row[251usize]) * (node_888)))
                + ((current_base_row[274usize]) * (node_338)))
                + ((current_base_row[275usize]) * (node_338)))
                + ((current_base_row[277usize]) * (node_900)))
                + ((current_base_row[279usize]) * (node_224)))
                + ((current_base_row[259usize]) * (node_753)))
                + ((current_base_row[263usize]) * (node_441))) * (node_4849))
                + ((node_876) * (next_base_row[8usize])),
            (((((((((((((((((((((((((((((((((((((current_base_row[202usize])
                * (node_471)) + ((current_base_row[212usize]) * (node_560)))
                + ((current_base_row[203usize]) * (node_783)))
                + ((current_base_row[208usize]) * (node_845)))
                + ((current_base_row[205usize])
                    * ((current_base_row[331usize])
                        * ((next_base_row[37usize]) + (node_545)))))
                + ((current_base_row[214usize]) * (node_904)))
                + ((current_base_row[216usize]) * (node_908)))
                + ((current_base_row[213usize]) * (node_200)))
                + ((current_base_row[220usize]) * (node_908)))
                + ((current_base_row[225usize]) * (node_916)))
                + ((current_base_row[223usize]) * (node_908)))
                + ((current_base_row[224usize]) * (node_228)))
                + ((current_base_row[215usize]) * (node_778)))
                + ((current_base_row[221usize]) * (node_466)))
                + ((current_base_row[232usize]) * (node_125)))
                + ((current_base_row[228usize]) * (node_416)))
                + ((current_base_row[229usize]) * (node_908)))
                + ((current_base_row[233usize]) * (node_232)))
                + ((current_base_row[234usize]) * (node_232)))
                + ((current_base_row[235usize]) * (node_908)))
                + ((current_base_row[237usize]) * (node_224)))
                + ((current_base_row[236usize]) * (node_560)))
                + ((current_base_row[238usize]) * (node_236)))
                + ((current_base_row[239usize]) * (node_236)))
                + ((current_base_row[241usize]) * (node_236)))
                + ((current_base_row[242usize]) * (node_912)))
                + ((current_base_row[245usize]) * (node_236)))
                + ((current_base_row[248usize]) * (node_912)))
                + ((current_base_row[251usize]) * (node_912)))
                + ((current_base_row[274usize]) * (node_121)))
                + ((current_base_row[275usize]) * (node_121)))
                + ((current_base_row[277usize]) * (node_924)))
                + ((current_base_row[279usize]) * (node_160)))
                + ((current_base_row[259usize]) * (node_783)))
                + ((current_base_row[263usize]) * (node_471))) * (node_4849))
                + ((node_900) * (next_base_row[8usize])),
            ((((((((((current_base_row[202usize]) * (current_base_row[312usize]))
                + ((current_base_row[203usize]) * (current_base_row[312usize])))
                + ((current_base_row[208usize]) * (node_548)))
                + ((current_base_row[205usize]) * (node_839)))
                + ((current_base_row[213usize]) * (node_160)))
                + ((current_base_row[215usize]) * (current_base_row[315usize])))
                + ((current_base_row[221usize]) * (current_base_row[315usize])))
                + ((current_base_row[259usize]) * (current_base_row[315usize])))
                + ((current_base_row[263usize]) * (current_base_row[315usize])))
                * (node_4849),
            (((((((((current_base_row[202usize]) * (current_base_row[318usize]))
                + ((current_base_row[203usize]) * (current_base_row[318usize])))
                + ((current_base_row[208usize]) * (node_553)))
                + ((current_base_row[205usize])
                    * (((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[186usize]))) * (node_872))))
                + ((current_base_row[215usize]) * (current_base_row[317usize])))
                + ((current_base_row[221usize]) * (current_base_row[317usize])))
                + ((current_base_row[259usize]) * (current_base_row[317usize])))
                + ((current_base_row[263usize]) * (current_base_row[317usize])))
                * (node_4849),
            (((((((((current_base_row[202usize]) * (current_base_row[273usize]))
                + ((current_base_row[203usize]) * (current_base_row[273usize])))
                + ((current_base_row[208usize]) * (node_554)))
                + ((current_base_row[205usize])
                    * (((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[187usize]))) * (node_876))))
                + ((current_base_row[215usize]) * (current_base_row[318usize])))
                + ((current_base_row[221usize]) * (current_base_row[318usize])))
                + ((current_base_row[259usize]) * (current_base_row[318usize])))
                + ((current_base_row[263usize]) * (current_base_row[318usize])))
                * (node_4849),
            (((((((((current_base_row[202usize]) * (current_base_row[329usize]))
                + ((current_base_row[203usize]) * (current_base_row[329usize])))
                + ((current_base_row[208usize]) * (node_555)))
                + ((current_base_row[205usize])
                    * (((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[188usize]))) * (node_880))))
                + ((current_base_row[215usize]) * (current_base_row[273usize])))
                + ((current_base_row[221usize]) * (current_base_row[273usize])))
                + ((current_base_row[259usize]) * (current_base_row[273usize])))
                + ((current_base_row[263usize]) * (current_base_row[273usize])))
                * (node_4849),
            (((((((((current_base_row[202usize]) * (current_base_row[330usize]))
                + ((current_base_row[203usize]) * (current_base_row[330usize])))
                + ((current_base_row[208usize]) * (node_556)))
                + ((current_base_row[205usize])
                    * (((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[189usize]))) * (node_884))))
                + ((current_base_row[215usize]) * (current_base_row[329usize])))
                + ((current_base_row[221usize]) * (current_base_row[329usize])))
                + ((current_base_row[259usize]) * (current_base_row[329usize])))
                + ((current_base_row[263usize]) * (current_base_row[329usize])))
                * (node_4849),
            (((((((((current_base_row[202usize]) * (current_base_row[331usize]))
                + ((current_base_row[203usize]) * (current_base_row[331usize])))
                + ((current_base_row[208usize]) * (node_557)))
                + ((current_base_row[205usize])
                    * (((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[312usize]))) * (node_888))))
                + ((current_base_row[215usize]) * (current_base_row[330usize])))
                + ((current_base_row[221usize]) * (current_base_row[330usize])))
                + ((current_base_row[259usize]) * (current_base_row[330usize])))
                + ((current_base_row[263usize]) * (current_base_row[330usize])))
                * (node_4849),
            (((current_base_row[208usize]) * (node_561))
                + ((current_base_row[205usize])
                    * (((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[317usize]))) * (node_904))))
                * (node_4849),
            ((current_base_row[205usize])
                * (((BFieldElement::from_raw_u64(4294967295u64))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[331usize]))) * (node_924))) * (node_4849),
            ((current_base_row[205usize]) * (node_159)) * (node_4849),
            ((current_base_row[205usize]) * (node_132)) * (node_4849),
            ((current_base_row[205usize]) * (current_base_row[268usize])) * (node_4849),
            ((current_base_row[205usize]) * (current_base_row[260usize])) * (node_4849),
            ((current_base_row[205usize]) * (current_base_row[261usize])) * (node_4849),
            ((current_base_row[205usize]) * (node_155)) * (node_4849),
            ((current_base_row[205usize]) * (node_121)) * (node_4849),
            ((current_base_row[205usize]) * (node_125)) * (node_4849),
            ((current_base_row[205usize]) * (node_129)) * (node_4849),
            ((current_base_row[205usize]) * (node_117)) * (node_4849),
            (node_5277) * (node_5276),
            ((node_5277)
                * ((next_base_row[49usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[49usize])))) * (next_base_row[47usize]),
            ((current_base_row[47usize])
                * ((current_base_row[47usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                * (node_5283),
            (((current_base_row[51usize])
                + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                * (current_base_row[51usize])) * (node_5346),
            (current_base_row[54usize]) * (node_5355),
            (node_5352) * (node_5355),
            ((node_5355)
                * ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (next_base_row[51usize])))
                * ((next_base_row[53usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[53usize]))),
            (node_5355)
                * ((next_base_row[55usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[55usize]))),
            (node_5355)
                * ((next_base_row[56usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[56usize]))),
            (node_5461) * (node_5460),
            (node_5465)
                * ((next_base_row[60usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[60usize]))),
            (node_5465)
                * ((next_base_row[61usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[61usize]))),
            (((node_5461)
                * ((node_5473) + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                * ((current_base_row[58usize])
                    + (BFieldElement::from_raw_u64(18446743927680663586u64))))
                * (node_5464),
            ((current_base_row[345usize])
                * ((current_base_row[64usize])
                    + (BFieldElement::from_raw_u64(18446744052234715141u64))))
                * (next_base_row[64usize]),
            (((next_base_row[62usize]) * (node_6374)) * (node_6376))
                * (((next_base_row[64usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[64usize])))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))),
            (current_base_row[350usize]) * (node_6408),
            (node_6410)
                * ((next_base_row[63usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[63usize]))),
            (node_6410)
                * ((next_base_row[62usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[62usize]))),
            ((current_base_row[346usize]) * (node_6404)) * (next_base_row[62usize]),
            (current_base_row[351usize]) * (next_base_row[62usize]),
            ((node_6425) * (node_6394)) * (next_base_row[62usize]),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(263719581847590u64))
                    * (node_5588))
                    + ((BFieldElement::from_raw_u64(76643691379275u64)) * (node_5599)))
                    + ((BFieldElement::from_raw_u64(115096533571410u64)) * (node_5610)))
                    + ((BFieldElement::from_raw_u64(256362302871255u64)) * (node_5621)))
                    + ((BFieldElement::from_raw_u64(51629801853195u64))
                        * (current_base_row[297usize])))
                    + ((BFieldElement::from_raw_u64(175668457332795u64))
                        * (current_base_row[298usize])))
                    + ((BFieldElement::from_raw_u64(177601192615545u64))
                        * (current_base_row[299usize])))
                    + ((BFieldElement::from_raw_u64(118201794925695u64))
                        * (current_base_row[300usize])))
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
                        * (current_base_row[308usize]))) + (current_base_row[113usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (node_6264))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(4758823762860u64))
                    * (node_5588))
                    + ((BFieldElement::from_raw_u64(263719581847590u64)) * (node_5599)))
                    + ((BFieldElement::from_raw_u64(76643691379275u64)) * (node_5610)))
                    + ((BFieldElement::from_raw_u64(115096533571410u64)) * (node_5621)))
                    + ((BFieldElement::from_raw_u64(256362302871255u64))
                        * (current_base_row[297usize])))
                    + ((BFieldElement::from_raw_u64(51629801853195u64))
                        * (current_base_row[298usize])))
                    + ((BFieldElement::from_raw_u64(175668457332795u64))
                        * (current_base_row[299usize])))
                    + ((BFieldElement::from_raw_u64(177601192615545u64))
                        * (current_base_row[300usize])))
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
                        * (current_base_row[308usize]))) + (current_base_row[114usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (node_6275))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(123480309731250u64))
                    * (node_5588))
                    + ((BFieldElement::from_raw_u64(4758823762860u64)) * (node_5599)))
                    + ((BFieldElement::from_raw_u64(263719581847590u64)) * (node_5610)))
                    + ((BFieldElement::from_raw_u64(76643691379275u64)) * (node_5621)))
                    + ((BFieldElement::from_raw_u64(115096533571410u64))
                        * (current_base_row[297usize])))
                    + ((BFieldElement::from_raw_u64(256362302871255u64))
                        * (current_base_row[298usize])))
                    + ((BFieldElement::from_raw_u64(51629801853195u64))
                        * (current_base_row[299usize])))
                    + ((BFieldElement::from_raw_u64(175668457332795u64))
                        * (current_base_row[300usize])))
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
                        * (current_base_row[308usize]))) + (current_base_row[115usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (node_6286))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(145268678818785u64))
                    * (node_5588))
                    + ((BFieldElement::from_raw_u64(123480309731250u64)) * (node_5599)))
                    + ((BFieldElement::from_raw_u64(4758823762860u64)) * (node_5610)))
                    + ((BFieldElement::from_raw_u64(263719581847590u64)) * (node_5621)))
                    + ((BFieldElement::from_raw_u64(76643691379275u64))
                        * (current_base_row[297usize])))
                    + ((BFieldElement::from_raw_u64(115096533571410u64))
                        * (current_base_row[298usize])))
                    + ((BFieldElement::from_raw_u64(256362302871255u64))
                        * (current_base_row[299usize])))
                    + ((BFieldElement::from_raw_u64(51629801853195u64))
                        * (current_base_row[300usize])))
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
                        * (current_base_row[308usize]))) + (current_base_row[116usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (node_6297))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(32014686216930u64))
                    * (node_5588))
                    + ((BFieldElement::from_raw_u64(145268678818785u64)) * (node_5599)))
                    + ((BFieldElement::from_raw_u64(123480309731250u64)) * (node_5610)))
                    + ((BFieldElement::from_raw_u64(4758823762860u64)) * (node_5621)))
                    + ((BFieldElement::from_raw_u64(263719581847590u64))
                        * (current_base_row[297usize])))
                    + ((BFieldElement::from_raw_u64(76643691379275u64))
                        * (current_base_row[298usize])))
                    + ((BFieldElement::from_raw_u64(115096533571410u64))
                        * (current_base_row[299usize])))
                    + ((BFieldElement::from_raw_u64(256362302871255u64))
                        * (current_base_row[300usize])))
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
                        * (current_base_row[308usize]))) + (current_base_row[117usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[97usize]))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(185731565704980u64))
                    * (node_5588))
                    + ((BFieldElement::from_raw_u64(32014686216930u64)) * (node_5599)))
                    + ((BFieldElement::from_raw_u64(145268678818785u64)) * (node_5610)))
                    + ((BFieldElement::from_raw_u64(123480309731250u64)) * (node_5621)))
                    + ((BFieldElement::from_raw_u64(4758823762860u64))
                        * (current_base_row[297usize])))
                    + ((BFieldElement::from_raw_u64(263719581847590u64))
                        * (current_base_row[298usize])))
                    + ((BFieldElement::from_raw_u64(76643691379275u64))
                        * (current_base_row[299usize])))
                    + ((BFieldElement::from_raw_u64(115096533571410u64))
                        * (current_base_row[300usize])))
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
                        * (current_base_row[308usize]))) + (current_base_row[118usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[98usize]))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(231348413345175u64))
                    * (node_5588))
                    + ((BFieldElement::from_raw_u64(185731565704980u64)) * (node_5599)))
                    + ((BFieldElement::from_raw_u64(32014686216930u64)) * (node_5610)))
                    + ((BFieldElement::from_raw_u64(145268678818785u64)) * (node_5621)))
                    + ((BFieldElement::from_raw_u64(123480309731250u64))
                        * (current_base_row[297usize])))
                    + ((BFieldElement::from_raw_u64(4758823762860u64))
                        * (current_base_row[298usize])))
                    + ((BFieldElement::from_raw_u64(263719581847590u64))
                        * (current_base_row[299usize])))
                    + ((BFieldElement::from_raw_u64(76643691379275u64))
                        * (current_base_row[300usize])))
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
                        * (current_base_row[308usize]))) + (current_base_row[119usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[99usize]))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(51685636428030u64))
                    * (node_5588))
                    + ((BFieldElement::from_raw_u64(231348413345175u64)) * (node_5599)))
                    + ((BFieldElement::from_raw_u64(185731565704980u64)) * (node_5610)))
                    + ((BFieldElement::from_raw_u64(32014686216930u64)) * (node_5621)))
                    + ((BFieldElement::from_raw_u64(145268678818785u64))
                        * (current_base_row[297usize])))
                    + ((BFieldElement::from_raw_u64(123480309731250u64))
                        * (current_base_row[298usize])))
                    + ((BFieldElement::from_raw_u64(4758823762860u64))
                        * (current_base_row[299usize])))
                    + ((BFieldElement::from_raw_u64(263719581847590u64))
                        * (current_base_row[300usize])))
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
                        * (current_base_row[308usize]))) + (current_base_row[120usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[100usize]))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(244602682417545u64))
                    * (node_5588))
                    + ((BFieldElement::from_raw_u64(51685636428030u64)) * (node_5599)))
                    + ((BFieldElement::from_raw_u64(231348413345175u64)) * (node_5610)))
                    + ((BFieldElement::from_raw_u64(185731565704980u64)) * (node_5621)))
                    + ((BFieldElement::from_raw_u64(32014686216930u64))
                        * (current_base_row[297usize])))
                    + ((BFieldElement::from_raw_u64(145268678818785u64))
                        * (current_base_row[298usize])))
                    + ((BFieldElement::from_raw_u64(123480309731250u64))
                        * (current_base_row[299usize])))
                    + ((BFieldElement::from_raw_u64(4758823762860u64))
                        * (current_base_row[300usize])))
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
                        * (current_base_row[308usize]))) + (current_base_row[121usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[101usize]))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(118201794925695u64))
                    * (node_5588))
                    + ((BFieldElement::from_raw_u64(244602682417545u64)) * (node_5599)))
                    + ((BFieldElement::from_raw_u64(51685636428030u64)) * (node_5610)))
                    + ((BFieldElement::from_raw_u64(231348413345175u64)) * (node_5621)))
                    + ((BFieldElement::from_raw_u64(185731565704980u64))
                        * (current_base_row[297usize])))
                    + ((BFieldElement::from_raw_u64(32014686216930u64))
                        * (current_base_row[298usize])))
                    + ((BFieldElement::from_raw_u64(145268678818785u64))
                        * (current_base_row[299usize])))
                    + ((BFieldElement::from_raw_u64(123480309731250u64))
                        * (current_base_row[300usize])))
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
                        * (current_base_row[308usize]))) + (current_base_row[122usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[102usize]))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(177601192615545u64))
                    * (node_5588))
                    + ((BFieldElement::from_raw_u64(118201794925695u64)) * (node_5599)))
                    + ((BFieldElement::from_raw_u64(244602682417545u64)) * (node_5610)))
                    + ((BFieldElement::from_raw_u64(51685636428030u64)) * (node_5621)))
                    + ((BFieldElement::from_raw_u64(231348413345175u64))
                        * (current_base_row[297usize])))
                    + ((BFieldElement::from_raw_u64(185731565704980u64))
                        * (current_base_row[298usize])))
                    + ((BFieldElement::from_raw_u64(32014686216930u64))
                        * (current_base_row[299usize])))
                    + ((BFieldElement::from_raw_u64(145268678818785u64))
                        * (current_base_row[300usize])))
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
                        * (current_base_row[308usize]))) + (current_base_row[123usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[103usize]))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(175668457332795u64))
                    * (node_5588))
                    + ((BFieldElement::from_raw_u64(177601192615545u64)) * (node_5599)))
                    + ((BFieldElement::from_raw_u64(118201794925695u64)) * (node_5610)))
                    + ((BFieldElement::from_raw_u64(244602682417545u64)) * (node_5621)))
                    + ((BFieldElement::from_raw_u64(51685636428030u64))
                        * (current_base_row[297usize])))
                    + ((BFieldElement::from_raw_u64(231348413345175u64))
                        * (current_base_row[298usize])))
                    + ((BFieldElement::from_raw_u64(185731565704980u64))
                        * (current_base_row[299usize])))
                    + ((BFieldElement::from_raw_u64(32014686216930u64))
                        * (current_base_row[300usize])))
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
                        * (current_base_row[308usize]))) + (current_base_row[124usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[104usize]))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(51629801853195u64))
                    * (node_5588))
                    + ((BFieldElement::from_raw_u64(175668457332795u64)) * (node_5599)))
                    + ((BFieldElement::from_raw_u64(177601192615545u64)) * (node_5610)))
                    + ((BFieldElement::from_raw_u64(118201794925695u64)) * (node_5621)))
                    + ((BFieldElement::from_raw_u64(244602682417545u64))
                        * (current_base_row[297usize])))
                    + ((BFieldElement::from_raw_u64(51685636428030u64))
                        * (current_base_row[298usize])))
                    + ((BFieldElement::from_raw_u64(231348413345175u64))
                        * (current_base_row[299usize])))
                    + ((BFieldElement::from_raw_u64(185731565704980u64))
                        * (current_base_row[300usize])))
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
                        * (current_base_row[308usize]))) + (current_base_row[125usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[105usize]))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(256362302871255u64))
                    * (node_5588))
                    + ((BFieldElement::from_raw_u64(51629801853195u64)) * (node_5599)))
                    + ((BFieldElement::from_raw_u64(175668457332795u64)) * (node_5610)))
                    + ((BFieldElement::from_raw_u64(177601192615545u64)) * (node_5621)))
                    + ((BFieldElement::from_raw_u64(118201794925695u64))
                        * (current_base_row[297usize])))
                    + ((BFieldElement::from_raw_u64(244602682417545u64))
                        * (current_base_row[298usize])))
                    + ((BFieldElement::from_raw_u64(51685636428030u64))
                        * (current_base_row[299usize])))
                    + ((BFieldElement::from_raw_u64(231348413345175u64))
                        * (current_base_row[300usize])))
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
                        * (current_base_row[308usize]))) + (current_base_row[126usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[106usize]))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(115096533571410u64))
                    * (node_5588))
                    + ((BFieldElement::from_raw_u64(256362302871255u64)) * (node_5599)))
                    + ((BFieldElement::from_raw_u64(51629801853195u64)) * (node_5610)))
                    + ((BFieldElement::from_raw_u64(175668457332795u64)) * (node_5621)))
                    + ((BFieldElement::from_raw_u64(177601192615545u64))
                        * (current_base_row[297usize])))
                    + ((BFieldElement::from_raw_u64(118201794925695u64))
                        * (current_base_row[298usize])))
                    + ((BFieldElement::from_raw_u64(244602682417545u64))
                        * (current_base_row[299usize])))
                    + ((BFieldElement::from_raw_u64(51685636428030u64))
                        * (current_base_row[300usize])))
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
                        * (current_base_row[308usize]))) + (current_base_row[127usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[107usize]))),
            (next_base_row[64usize])
                * (((((((((((((((((((BFieldElement::from_raw_u64(76643691379275u64))
                    * (node_5588))
                    + ((BFieldElement::from_raw_u64(115096533571410u64)) * (node_5599)))
                    + ((BFieldElement::from_raw_u64(256362302871255u64)) * (node_5610)))
                    + ((BFieldElement::from_raw_u64(51629801853195u64)) * (node_5621)))
                    + ((BFieldElement::from_raw_u64(175668457332795u64))
                        * (current_base_row[297usize])))
                    + ((BFieldElement::from_raw_u64(177601192615545u64))
                        * (current_base_row[298usize])))
                    + ((BFieldElement::from_raw_u64(118201794925695u64))
                        * (current_base_row[299usize])))
                    + ((BFieldElement::from_raw_u64(244602682417545u64))
                        * (current_base_row[300usize])))
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
                        * (current_base_row[308usize]))) + (current_base_row[128usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[108usize]))),
            (current_base_row[129usize]) * (node_6920),
            (current_base_row[135usize]) * (node_6972),
            ((next_base_row[135usize]) * (next_base_row[136usize]))
                + ((node_6972)
                    * (((next_base_row[136usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[136usize])))
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)))),
            ((next_base_row[139usize]) * (current_base_row[143usize])) * (node_7022),
            (next_base_row[139usize]) * (current_base_row[145usize]),
            (node_7032)
                * ((next_base_row[142usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[142usize]))),
            (((node_7032) * (current_base_row[143usize])) * (node_7022)) * (node_7040),
            ((node_7032) * (current_base_row[145usize])) * (node_7040),
            (((node_7032) * (node_7022)) * (node_7025)) * (node_7046),
            ((node_7032) * (node_7028)) * (node_7049),
            (((current_base_row[309usize]) * (node_7067)) * (node_7069))
                * (current_base_row[147usize]),
            ((node_7072) * (node_7069)) * (node_7074),
            (((current_base_row[313usize]) * (node_7046)) * (node_7028)) * (node_7074),
            (((current_base_row[313usize]) * (node_7025)) * (node_7049))
                * (current_base_row[147usize]),
            ((current_base_row[348usize])
                * ((current_base_row[139usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                * ((current_base_row[147usize])
                    + (BFieldElement::from_raw_u64(18446744060824649731u64))),
            ((current_base_row[348usize]) * (current_base_row[139usize]))
                * (current_base_row[147usize]),
            ((node_7032) * (current_base_row[347usize]))
                * (((current_base_row[147usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((BFieldElement::from_raw_u64(8589934590u64))
                            * (next_base_row[147usize]))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((node_7025) * (node_7028)))),
            (current_base_row[354usize]) * ((current_base_row[147usize]) + (node_7038)),
            ((current_base_row[334usize]) * (next_base_row[143usize]))
                * ((next_base_row[147usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[147usize]))),
            (current_base_row[349usize])
                * ((next_base_row[143usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[143usize]))),
            ((current_base_row[349usize]) * (node_7049))
                * ((current_base_row[147usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (node_7133))),
            ((current_base_row[349usize]) * (node_7028))
                * ((current_base_row[147usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[355usize]))),
            ((node_7032) * ((current_base_row[332usize]) * (node_7060)))
                * (((current_base_row[147usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[147usize]))) + (node_7083)),
            (current_base_row[169usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_246) * (node_248))),
            (current_base_row[170usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_1493) * (node_1474))),
            (current_base_row[171usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_1493) * (current_base_row[13usize]))),
            (current_base_row[172usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[12usize]) * (node_1474)) * (node_1464))),
            (current_base_row[173usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[170usize]) * (node_1464))),
            (current_base_row[174usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[171usize]) * (node_1464))),
            (current_base_row[175usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[169usize]) * (current_base_row[40usize]))),
            (current_base_row[176usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_246) * (current_base_row[41usize]))),
            (current_base_row[177usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[176usize]) * (node_251))),
            (current_base_row[178usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[172usize]) * (node_1466))),
            (current_base_row[179usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[172usize]) * (current_base_row[15usize]))),
            (current_base_row[180usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[173usize]) * (node_1466))),
            (current_base_row[181usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[173usize]) * (current_base_row[15usize]))),
            (current_base_row[182usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[169usize]) * (node_251))),
            (current_base_row[183usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[174usize]) * (node_1466))),
            (current_base_row[184usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[174usize]) * (current_base_row[15usize]))),
            (current_base_row[185usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[182usize]) * (current_base_row[39usize]))),
            (current_base_row[186usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[175usize]) * (node_298))),
            (current_base_row[187usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[175usize]) * (current_base_row[39usize]))),
            (current_base_row[188usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[177usize]) * (node_298))),
            (current_base_row[189usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[177usize]) * (current_base_row[39usize]))),
            (current_base_row[190usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[12usize]) * (current_base_row[13usize]))
                        * (node_1464))),
            (current_base_row[191usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[179usize]) * (node_1468))),
            (current_base_row[192usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[178usize]) * (node_1468))),
            (current_base_row[193usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[180usize]) * (node_1468))),
            (current_base_row[194usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[183usize]) * (node_1468))),
            (current_base_row[195usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[181usize]) * (node_1468))),
            (current_base_row[196usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[184usize]) * (node_1468))),
            (current_base_row[197usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[171usize]) * (current_base_row[14usize]))),
            (current_base_row[198usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[190usize]) * (node_1466))),
            (current_base_row[199usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[170usize]) * (current_base_row[14usize]))),
            (current_base_row[200usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[178usize]) * (current_base_row[16usize]))),
            (current_base_row[201usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[181usize]) * (current_base_row[16usize]))),
            (current_base_row[202usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[198usize]) * (node_1468)) * (node_1470))
                        * (node_1472))),
            (current_base_row[203usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[191usize]) * (node_1470)) * (node_1472))),
            (current_base_row[204usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[180usize]) * (current_base_row[16usize]))),
            (current_base_row[205usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[179usize]) * (current_base_row[16usize]))
                        * (node_1470)) * (node_1472))),
            (current_base_row[206usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[183usize]) * (current_base_row[16usize]))),
            (current_base_row[207usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[184usize]) * (current_base_row[16usize]))),
            (current_base_row[208usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[200usize]) * (node_1470)) * (node_1472))),
            (current_base_row[209usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[194usize]) * (node_1470))),
            (current_base_row[210usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[193usize]) * (node_1470))),
            (current_base_row[211usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[196usize]) * (node_1470))),
            (current_base_row[212usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[192usize]) * (node_1470)) * (node_1472))),
            (current_base_row[213usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[209usize]) * (node_1472))),
            (current_base_row[214usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[210usize]) * (node_1472))),
            (current_base_row[215usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[191usize]) * (current_base_row[17usize]))
                        * (node_1472))),
            (current_base_row[216usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[195usize]) * (node_1470)) * (node_1472))),
            (current_base_row[217usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[190usize]) * (current_base_row[15usize]))
                        * (node_1468)) * (node_1470))),
            (current_base_row[218usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[197usize]) * (node_1466))),
            (current_base_row[219usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[199usize]) * (node_1466))),
            (current_base_row[220usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[192usize]) * (current_base_row[17usize]))
                        * (node_1472))),
            (current_base_row[221usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[217usize]) * (node_1472))),
            (current_base_row[222usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[197usize]) * (current_base_row[15usize]))),
            (current_base_row[223usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[201usize]) * (node_1470)) * (node_1472))),
            (current_base_row[224usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[211usize]) * (node_1472))),
            (current_base_row[225usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[204usize]) * (node_1470)) * (node_1472))),
            (current_base_row[226usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[199usize]) * (current_base_row[15usize]))),
            (current_base_row[227usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[42usize]) * (node_248))),
            (current_base_row[228usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[207usize]) * (node_1470)) * (node_1472))),
            (current_base_row[229usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[195usize]) * (current_base_row[17usize]))
                        * (node_1472))),
            (current_base_row[230usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[206usize]) * (node_1470))),
            (current_base_row[231usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[42usize]) * (current_base_row[41usize]))),
            (current_base_row[232usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[193usize]) * (current_base_row[17usize]))
                        * (node_1472))),
            (current_base_row[233usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[196usize]) * (current_base_row[17usize]))
                        * (node_1472))),
            (current_base_row[234usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[206usize]) * (current_base_row[17usize]))
                        * (node_1472))),
            (current_base_row[235usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[201usize]) * (current_base_row[17usize]))
                        * (node_1472))),
            (current_base_row[236usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[219usize]) * (node_1468)) * (node_1470))
                        * (node_1472))),
            (current_base_row[237usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[207usize]) * (current_base_row[17usize]))
                        * (node_1472))),
            (current_base_row[238usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[218usize]) * (node_1468)) * (node_1470))
                        * (node_1472))),
            (current_base_row[239usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[222usize]) * (node_1468)) * (node_1470))
                        * (node_1472))),
            (current_base_row[240usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[230usize]) * (node_1472))),
            (current_base_row[241usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[218usize]) * (current_base_row[16usize]))
                        * (node_1470)) * (node_1472))),
            (current_base_row[242usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[226usize]) * (node_1468)) * (node_1470))
                        * (node_1472))),
            (current_base_row[243usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[176usize]) * (current_base_row[40usize]))),
            (current_base_row[244usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[194usize]) * (current_base_row[17usize]))
                        * (node_1472))),
            (current_base_row[245usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[222usize]) * (current_base_row[16usize]))
                        * (node_1470)) * (node_1472))),
            (current_base_row[246usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[204usize]) * (current_base_row[17usize]))
                        * (node_1472))),
            (current_base_row[247usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[227usize]) * (node_251))),
            (current_base_row[248usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[219usize]) * (current_base_row[16usize]))
                        * (node_1470)) * (node_1472))),
            (current_base_row[249usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[97usize]) * (current_base_row[97usize]))
                        * (current_base_row[97usize])) * (current_base_row[97usize]))),
            (current_base_row[250usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[98usize]) * (current_base_row[98usize]))
                        * (current_base_row[98usize])) * (current_base_row[98usize]))),
            (current_base_row[251usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[226usize]) * (current_base_row[16usize]))
                        * (node_1470)) * (node_1472))),
            (current_base_row[252usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[99usize]) * (current_base_row[99usize]))
                        * (current_base_row[99usize])) * (current_base_row[99usize]))),
            (current_base_row[253usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[227usize]) * (current_base_row[40usize]))),
            (current_base_row[254usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[100usize]) * (current_base_row[100usize]))
                        * (current_base_row[100usize])) * (current_base_row[100usize]))),
            (current_base_row[255usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[231usize]) * (node_251))),
            (current_base_row[256usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[231usize]) * (current_base_row[40usize]))),
            (current_base_row[257usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[101usize]) * (current_base_row[101usize]))
                        * (current_base_row[101usize])) * (current_base_row[101usize]))),
            (current_base_row[258usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[102usize]) * (current_base_row[102usize]))
                        * (current_base_row[102usize])) * (current_base_row[102usize]))),
            (current_base_row[259usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[200usize]) * (current_base_row[17usize]))
                        * (node_1472))),
            (current_base_row[260usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[41usize])
                        * ((current_base_row[41usize])
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))))),
            (current_base_row[261usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[42usize])
                        * ((current_base_row[42usize])
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))))),
            (current_base_row[262usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[103usize]) * (current_base_row[103usize]))
                        * (current_base_row[103usize])) * (current_base_row[103usize]))),
            (current_base_row[263usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[198usize]) * (current_base_row[16usize]))
                        * (node_1470)) * (node_1472))),
            (current_base_row[264usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[104usize]) * (current_base_row[104usize]))
                        * (current_base_row[104usize])) * (current_base_row[104usize]))),
            (current_base_row[265usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[105usize]) * (current_base_row[105usize]))
                        * (current_base_row[105usize])) * (current_base_row[105usize]))),
            (current_base_row[266usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[106usize]) * (current_base_row[106usize]))
                        * (current_base_row[106usize])) * (current_base_row[106usize]))),
            (current_base_row[267usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[107usize]) * (current_base_row[107usize]))
                        * (current_base_row[107usize])) * (current_base_row[107usize]))),
            (current_base_row[268usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[40usize]) * (node_134))),
            (current_base_row[269usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[108usize]) * (current_base_row[108usize]))
                        * (current_base_row[108usize])) * (current_base_row[108usize]))),
            (current_base_row[270usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[243usize]) * (current_base_row[39usize]))),
            (current_base_row[271usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[247usize]) * (node_298))),
            (current_base_row[272usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[39usize]) * (current_base_row[22usize]))),
            (current_base_row[273usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[255usize]) * (node_298))),
            (current_base_row[274usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[209usize]) * (current_base_row[18usize]))),
            (current_base_row[275usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[211usize]) * (current_base_row[18usize]))),
            (current_base_row[276usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((next_base_row[64usize]) * (node_6454)) * (node_6456))
                        * (node_6458))),
            (current_base_row[277usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[210usize]) * (current_base_row[18usize]))),
            (current_base_row[278usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((next_base_row[62usize]) * (node_6462)) * (node_6408))),
            (current_base_row[279usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[230usize]) * (current_base_row[18usize]))),
            (current_base_row[280usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[43usize])
                        * ((current_base_row[43usize])
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                        * ((current_base_row[43usize])
                            + (BFieldElement::from_raw_u64(18446744060824649731u64))))
                        * ((current_base_row[43usize])
                            + (BFieldElement::from_raw_u64(18446744056529682436u64))))),
            (current_base_row[281usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[44usize])
                        * ((current_base_row[44usize])
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                        * ((current_base_row[44usize])
                            + (BFieldElement::from_raw_u64(18446744060824649731u64))))
                        * ((current_base_row[44usize])
                            + (BFieldElement::from_raw_u64(18446744056529682436u64))))),
            (current_base_row[282usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[185usize]) * (node_547))),
            (current_base_row[283usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[186usize])
                        * ((next_base_row[25usize]) + (node_187)))),
            (current_base_row[284usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[187usize])
                        * ((next_base_row[26usize]) + (node_187)))),
            (current_base_row[285usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[188usize])
                        * ((next_base_row[27usize]) + (node_187)))),
            (current_base_row[286usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[189usize]) * (node_726))),
            (current_base_row[287usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[185usize]) * (node_192))),
            (current_base_row[288usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[186usize])
                        * ((next_base_row[23usize]) + (node_195)))),
            (current_base_row[289usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[187usize])
                        * ((next_base_row[23usize]) + (node_199)))),
            (current_base_row[290usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[188usize])
                        * ((next_base_row[23usize]) + (node_203)))),
            (current_base_row[291usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[189usize]) * (node_413))),
            (current_base_row[292usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[189usize]) * (node_412))),
            (current_base_row[293usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[189usize]) * (node_725))),
            (current_base_row[294usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((node_7051) * (node_7054)) * (node_7058)) * (node_7060))),
            (current_base_row[295usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_7051)
                        * ((next_base_row[142usize])
                            + (BFieldElement::from_raw_u64(18446744043644780551u64))))),
            (current_base_row[296usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((node_6454) * (node_6456)) * (node_6458)) * (node_6460))),
            (current_base_row[297usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[249usize]) * (current_base_row[97usize]))
                        * (current_base_row[97usize])) * (current_base_row[97usize]))),
            (current_base_row[298usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[250usize]) * (current_base_row[98usize]))
                        * (current_base_row[98usize])) * (current_base_row[98usize]))),
            (current_base_row[299usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[252usize]) * (current_base_row[99usize]))
                        * (current_base_row[99usize])) * (current_base_row[99usize]))),
            (current_base_row[300usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[254usize]) * (current_base_row[100usize]))
                        * (current_base_row[100usize])) * (current_base_row[100usize]))),
            (current_base_row[301usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[257usize]) * (current_base_row[101usize]))
                        * (current_base_row[101usize])) * (current_base_row[101usize]))),
            (current_base_row[302usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[258usize]) * (current_base_row[102usize]))
                        * (current_base_row[102usize])) * (current_base_row[102usize]))),
            (current_base_row[303usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[262usize]) * (current_base_row[103usize]))
                        * (current_base_row[103usize])) * (current_base_row[103usize]))),
            (current_base_row[304usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[264usize]) * (current_base_row[104usize]))
                        * (current_base_row[104usize])) * (current_base_row[104usize]))),
            (current_base_row[305usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[265usize]) * (current_base_row[105usize]))
                        * (current_base_row[105usize])) * (current_base_row[105usize]))),
            (current_base_row[306usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[266usize]) * (current_base_row[106usize]))
                        * (current_base_row[106usize])) * (current_base_row[106usize]))),
            (current_base_row[307usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[267usize]) * (current_base_row[107usize]))
                        * (current_base_row[107usize])) * (current_base_row[107usize]))),
            (current_base_row[308usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[269usize]) * (current_base_row[108usize]))
                        * (current_base_row[108usize])) * (current_base_row[108usize]))),
            (current_base_row[309usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_7032) * ((current_base_row[294usize]) * (node_7064)))),
            (current_base_row[310usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[295usize]) * (node_7054))),
            (current_base_row[311usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[296usize]) * (node_6462))),
            (current_base_row[312usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[243usize]) * (node_298))),
            (current_base_row[313usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_7072) * (node_7067))),
            (current_base_row[314usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[247usize]) * (current_base_row[39usize]))),
            (current_base_row[315usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[182usize]) * (node_298))),
            (current_base_row[316usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((node_6398) * (node_6421)) * (next_base_row[62usize]))),
            (current_base_row[317usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[253usize]) * (node_298))),
            (current_base_row[318usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[253usize]) * (current_base_row[39usize]))),
            (current_base_row[319usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[185usize]) * (node_196))),
            (current_base_row[320usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[185usize]) * (node_200))),
            (current_base_row[321usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[186usize])
                        * ((next_base_row[24usize]) + (node_199)))),
            (current_base_row[322usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[186usize])
                        * ((next_base_row[25usize]) + (node_203)))),
            (current_base_row[323usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[187usize])
                        * ((next_base_row[24usize]) + (node_203)))),
            (current_base_row[324usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[187usize]) * (node_332))),
            (current_base_row[325usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[188usize])
                        * ((next_base_row[24usize]) + (node_207)))),
            (current_base_row[326usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[188usize])
                        * ((next_base_row[25usize]) + (node_211)))),
            (current_base_row[327usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[189usize]) * (node_414))),
            (current_base_row[328usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[189usize]) * (node_415))),
            (current_base_row[329usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[255usize]) * (current_base_row[39usize]))),
            (current_base_row[330usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[256usize]) * (node_298))),
            (current_base_row[331usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[256usize]) * (current_base_row[39usize]))),
            (current_base_row[332usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[310usize]) * (node_7058))),
            (current_base_row[333usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((((next_base_row[12usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                        * (next_base_row[13usize]))
                        * ((next_base_row[14usize])
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                        * ((next_base_row[15usize])
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))))),
            (current_base_row[334usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_7032)
                        * (((current_base_row[310usize]) * (node_7060)) * (node_7064)))),
            (current_base_row[335usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[22usize]) * (current_base_row[23usize]))),
            (current_base_row[336usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((next_base_row[22usize]) * (current_base_row[22usize]))),
            (current_base_row[337usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[39usize]) * (node_1292))),
            (current_base_row[338usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[311usize])
                        * (((node_6421) * (node_6404)) * (next_base_row[62usize])))),
            (current_base_row[339usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((node_6392) * (node_6394)) * (current_base_row[62usize]))),
            (current_base_row[340usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[236usize])
                        * ((next_base_row[22usize])
                            * (((current_base_row[39usize])
                                * ((next_base_row[23usize])
                                    + (BFieldElement::from_raw_u64(4294967296u64))))
                                + (BFieldElement::from_raw_u64(
                                    18446744065119617026u64,
                                )))))),
            (current_base_row[341usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[213usize])
                        * ((((node_932) * (current_base_row[22usize]))
                            + (((node_117) * (node_934)) * (node_134)))
                            + ((((node_114)
                                + (BFieldElement::from_raw_u64(18446744056529682436u64)))
                                * (node_934)) * (current_base_row[40usize]))))),
            (current_base_row[342usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[213usize])
                        * (((current_base_row[260usize])
                            * ((current_base_row[41usize])
                                + (BFieldElement::from_raw_u64(18446744060824649731u64))))
                            * ((current_base_row[41usize])
                                + (BFieldElement::from_raw_u64(
                                    18446744056529682436u64,
                                )))))),
            (current_base_row[343usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[213usize])
                        * (((current_base_row[261usize])
                            * ((current_base_row[42usize])
                                + (BFieldElement::from_raw_u64(18446744060824649731u64))))
                            * ((current_base_row[42usize])
                                + (BFieldElement::from_raw_u64(
                                    18446744056529682436u64,
                                )))))),
            (current_base_row[344usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[333usize]) * (next_base_row[16usize]))
                        * ((next_base_row[17usize])
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                        * ((next_base_row[18usize])
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))))),
            (current_base_row[345usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[64usize])
                        * ((current_base_row[64usize])
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                        * ((current_base_row[64usize])
                            + (BFieldElement::from_raw_u64(18446744060824649731u64))))
                        * ((current_base_row[64usize])
                            + (BFieldElement::from_raw_u64(18446744056529682436u64))))),
            (current_base_row[346usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((node_6417) * (node_6394)) * (current_base_row[62usize]))
                        * (node_6421))),
            (current_base_row[347usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[295usize]) * (node_7058)) * (node_7060))
                        * (node_7064))),
            (current_base_row[348usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[313usize])
                        * ((((BFieldElement::from_raw_u64(4294967295u64)) + (node_7083))
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (node_7028)))
                            + (((BFieldElement::from_raw_u64(8589934590u64))
                                * (node_7025)) * (node_7028))))),
            (current_base_row[349usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_7032) * ((current_base_row[332usize]) * (node_7064)))),
            (current_base_row[350usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[339usize])
                        * (((node_6398) * (node_6404)) * (next_base_row[62usize])))),
            (current_base_row[351usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((node_6425) * (current_base_row[62usize])) * (node_6404))),
            (current_base_row[352usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[311usize]) * (node_6404))
                        * (next_base_row[62usize])) * (node_6408))),
            (current_base_row[353usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[311usize])
                        * (((node_6513) * (node_6408)) * (node_6516)))),
            (current_base_row[354usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((current_base_row[334usize])
                        * ((BFieldElement::from_raw_u64(4294967295u64))
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((next_base_row[143usize]) * (next_base_row[144usize])))))
                        * (current_base_row[143usize]))),
            (current_base_row[355usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_7133) * (current_base_row[143usize]))),
        ];
        let ext_constraints = [
            (((BFieldElement::from_raw_u64(4294967295u64))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (current_base_row[5usize])))
                * (((node_52)
                    * ((challenges[InstructionLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((((challenges[ProgramAddressWeight])
                                * (current_base_row[0usize]))
                                + ((challenges[ProgramInstructionWeight])
                                    * (current_base_row[1usize])))
                                + ((challenges[ProgramNextInstructionWeight])
                                    * (next_base_row[1usize]))))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[2usize]))))
                + ((current_base_row[5usize]) * (node_52)),
            ((node_31)
                * (((next_ext_row[1usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((challenges[ProgramAttestationPrepareChunkIndeterminate])
                            * (current_ext_row[1usize])))) + (node_75)))
                + ((node_34)
                    * (((next_ext_row[1usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (challenges[ProgramAttestationPrepareChunkIndeterminate])))
                        + (node_75))),
            ((((((next_ext_row[2usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((challenges[ProgramAttestationSendChunkIndeterminate])
                        * (current_ext_row[2usize]))))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (next_ext_row[1usize]))) * (node_48))
                * ((BFieldElement::from_raw_u64(4294967295u64))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((next_base_row[4usize]) * (node_91)))))
                + ((node_89) * (next_base_row[6usize]))) + ((node_89) * (node_91)),
            ((((((((((((((((((((((((((((((((((((((((current_base_row[202usize])
                * (current_base_row[268usize]))
                + ((current_base_row[212usize]) * (node_550)))
                + ((current_base_row[203usize]) * (current_base_row[268usize])))
                + ((current_base_row[208usize]) * (current_base_row[292usize])))
                + ((current_base_row[205usize]) * (current_base_row[293usize])))
                + ((current_base_row[214usize]) * (node_824)))
                + ((current_base_row[216usize]) * (node_868)))
                + (current_base_row[342usize]))
                + ((current_base_row[220usize]) * (node_868)))
                + ((current_base_row[225usize]) * (node_876)))
                + ((current_base_row[223usize]) * (node_868)))
                + ((current_base_row[224usize]) * (node_188)))
                + ((current_base_row[215usize]) * (current_base_row[268usize])))
                + ((current_base_row[221usize]) * (current_base_row[268usize])))
                + ((current_base_row[240usize]) * (node_418)))
                + ((current_base_row[232usize])
                    * (((node_298) * (node_876))
                        + ((current_base_row[39usize]) * (node_728)))))
                + ((current_base_row[228usize]) * (node_121)))
                + ((current_base_row[229usize]) * (node_868)))
                + ((current_base_row[244usize])
                    * ((next_ext_row[6usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_ext_row[69usize])))))
                + ((current_base_row[246usize])
                    * ((next_ext_row[6usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_ext_row[70usize])))))
                + ((current_base_row[233usize]) * (node_192)))
                + ((current_base_row[234usize]) * (node_192)))
                + ((current_base_row[235usize]) * (node_868)))
                + ((current_base_row[237usize]) * (node_129)))
                + ((current_base_row[236usize]) * (node_550)))
                + ((current_base_row[238usize]) * (node_196)))
                + ((current_base_row[239usize]) * (node_196)))
                + ((current_base_row[241usize]) * (node_196)))
                + ((current_base_row[242usize]) * (node_872)))
                + ((current_base_row[245usize]) * (node_196)))
                + ((current_base_row[248usize]) * (node_932)))
                + ((current_base_row[251usize]) * (node_872)))
                + ((current_base_row[274usize]) * (node_334)))
                + ((current_base_row[275usize]) * (node_334)))
                + ((current_base_row[277usize]) * (node_884)))
                + ((current_base_row[279usize]) * (node_208)))
                + ((current_base_row[259usize]) * (current_base_row[268usize])))
                + ((current_base_row[263usize]) * (current_base_row[268usize])))
                * (node_4849)) + ((node_129) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((current_base_row[202usize])
                * (node_446)) + ((current_base_row[212usize]) * (node_555)))
                + ((current_base_row[203usize]) * (node_758)))
                + ((current_base_row[208usize]) * (node_835)))
                + ((current_base_row[205usize])
                    * ((current_base_row[317usize]) * (node_854))))
                + ((current_base_row[214usize]) * (node_884)))
                + ((current_base_row[216usize]) * (node_888)))
                + ((current_base_row[213usize]) * (node_125)))
                + ((current_base_row[220usize]) * (node_888)))
                + ((current_base_row[225usize]) * (node_896)))
                + ((current_base_row[223usize]) * (node_888)))
                + ((current_base_row[224usize]) * (node_208)))
                + ((current_base_row[215usize])
                    * ((((((current_base_row[185usize])
                        * ((node_824) + (BFieldElement::from_raw_u64(4294967295u64))))
                        + ((current_base_row[186usize])
                            * ((node_824)
                                + (BFieldElement::from_raw_u64(8589934590u64)))))
                        + ((current_base_row[187usize])
                            * ((node_824)
                                + (BFieldElement::from_raw_u64(12884901885u64)))))
                        + ((current_base_row[188usize])
                            * ((node_824)
                                + (BFieldElement::from_raw_u64(17179869180u64)))))
                        + ((current_base_row[189usize])
                            * ((node_824)
                                + (BFieldElement::from_raw_u64(21474836475u64)))))))
                + ((current_base_row[221usize])
                    * ((((((current_base_row[185usize])
                        * ((node_824)
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                        + ((current_base_row[186usize])
                            * ((node_824)
                                + (BFieldElement::from_raw_u64(18446744060824649731u64)))))
                        + ((current_base_row[187usize])
                            * ((node_824)
                                + (BFieldElement::from_raw_u64(18446744056529682436u64)))))
                        + ((current_base_row[188usize])
                            * ((node_824)
                                + (BFieldElement::from_raw_u64(18446744052234715141u64)))))
                        + ((current_base_row[189usize])
                            * ((node_824)
                                + (BFieldElement::from_raw_u64(
                                    18446744047939747846u64,
                                ))))))) + ((current_base_row[240usize]) * (node_400)))
                + ((current_base_row[232usize]) * (node_734)))
                + ((current_base_row[228usize]) * (node_411)))
                + ((current_base_row[229usize]) * (node_888)))
                + ((current_base_row[244usize])
                    * ((next_base_row[26usize]) + (node_239))))
                + ((current_base_row[246usize])
                    * ((next_base_row[36usize]) + (node_199))))
                + ((current_base_row[233usize]) * (node_212)))
                + ((current_base_row[234usize]) * (node_212)))
                + ((current_base_row[235usize]) * (node_888)))
                + ((current_base_row[237usize]) * (node_204)))
                + ((current_base_row[236usize]) * (node_555)))
                + ((current_base_row[238usize]) * (node_216)))
                + ((current_base_row[239usize]) * (node_216)))
                + ((current_base_row[241usize]) * (node_216)))
                + ((current_base_row[242usize]) * (node_892)))
                + ((current_base_row[245usize]) * (node_216)))
                + ((current_base_row[248usize]) * (node_892)))
                + ((current_base_row[251usize]) * (node_892)))
                + ((current_base_row[274usize]) * (node_339)))
                + ((current_base_row[275usize]) * (node_339)))
                + ((current_base_row[277usize]) * (node_904)))
                + ((current_base_row[279usize]) * (node_228)))
                + ((current_base_row[259usize]) * (node_758)))
                + ((current_base_row[263usize]) * (node_446))) * (node_4849))
                + ((node_880) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((current_base_row[202usize])
                * (node_451)) + ((current_base_row[212usize]) * (node_556)))
                + ((current_base_row[203usize]) * (node_763)))
                + ((current_base_row[208usize]) * (node_837)))
                + ((current_base_row[205usize])
                    * ((current_base_row[318usize])
                        * ((next_base_row[33usize]) + (node_545)))))
                + ((current_base_row[214usize]) * (node_888)))
                + ((current_base_row[216usize]) * (node_892)))
                + ((current_base_row[213usize]) * (node_129)))
                + ((current_base_row[220usize]) * (node_892)))
                + ((current_base_row[225usize]) * (node_900)))
                + ((current_base_row[223usize]) * (node_892)))
                + ((current_base_row[224usize]) * (node_212)))
                + ((current_base_row[215usize]) * (node_758)))
                + ((current_base_row[221usize]) * (node_446)))
                + ((current_base_row[240usize]) * (node_411)))
                + ((current_base_row[232usize]) * (node_735)))
                + ((current_base_row[228usize]) * (node_412)))
                + ((current_base_row[229usize]) * (node_892)))
                + ((current_base_row[244usize])
                    * ((next_base_row[27usize]) + (node_243))))
                + ((current_base_row[246usize])
                    * ((next_base_row[37usize]) + (node_203))))
                + ((current_base_row[233usize]) * (node_216)))
                + ((current_base_row[234usize]) * (node_216)))
                + ((current_base_row[235usize]) * (node_892)))
                + ((current_base_row[237usize]) * (node_208)))
                + ((current_base_row[236usize]) * (node_556)))
                + ((current_base_row[238usize]) * (node_220)))
                + ((current_base_row[239usize]) * (node_220)))
                + ((current_base_row[241usize]) * (node_220)))
                + ((current_base_row[242usize]) * (node_896)))
                + ((current_base_row[245usize]) * (node_220)))
                + ((current_base_row[248usize]) * (node_896)))
                + ((current_base_row[251usize]) * (node_896)))
                + ((current_base_row[274usize]) * (node_340)))
                + ((current_base_row[275usize]) * (node_340)))
                + ((current_base_row[277usize]) * (node_908)))
                + ((current_base_row[279usize]) * (node_232)))
                + ((current_base_row[259usize]) * (node_763)))
                + ((current_base_row[263usize]) * (node_451))) * (node_4849))
                + ((node_884) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((current_base_row[202usize])
                * (node_456)) + ((current_base_row[212usize]) * (node_557)))
                + ((current_base_row[203usize]) * (node_768)))
                + ((current_base_row[208usize]) * (node_839)))
                + ((current_base_row[205usize])
                    * ((current_base_row[273usize])
                        * ((next_base_row[34usize]) + (node_545)))))
                + ((current_base_row[214usize]) * (node_892)))
                + ((current_base_row[216usize]) * (node_896)))
                + ((current_base_row[213usize]) * (node_188)))
                + ((current_base_row[220usize]) * (node_896)))
                + ((current_base_row[225usize]) * (node_904)))
                + ((current_base_row[223usize]) * (node_896)))
                + ((current_base_row[224usize]) * (node_216)))
                + (current_ext_row[81usize])) + (current_ext_row[82usize]))
                + ((current_base_row[240usize]) * (node_533)))
                + ((current_base_row[232usize]) * (node_713)))
                + ((current_base_row[228usize]) * (node_413)))
                + ((current_base_row[229usize]) * (node_896)))
                + ((current_base_row[244usize]) * (node_533)))
                + ((current_base_row[246usize]) * (node_533)))
                + ((current_base_row[233usize]) * (node_220)))
                + ((current_base_row[234usize]) * (node_220)))
                + ((current_base_row[235usize]) * (node_896)))
                + ((current_base_row[237usize]) * (node_212)))
                + ((current_base_row[236usize]) * (node_557)))
                + ((current_base_row[238usize]) * (node_224)))
                + ((current_base_row[239usize]) * (node_224)))
                + ((current_base_row[241usize]) * (node_224)))
                + ((current_base_row[242usize]) * (node_900)))
                + ((current_base_row[245usize]) * (node_224)))
                + ((current_base_row[248usize]) * (node_900)))
                + ((current_base_row[251usize]) * (node_900)))
                + ((current_base_row[274usize]) * (node_341)))
                + ((current_base_row[275usize]) * (node_341)))
                + ((current_base_row[277usize]) * (node_912)))
                + ((current_base_row[279usize]) * (node_236)))
                + ((current_base_row[259usize]) * (node_768)))
                + ((current_base_row[263usize]) * (node_456))) * (node_4849))
                + ((node_888) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((current_base_row[202usize])
                * (node_461)) + ((current_base_row[212usize]) * (node_558)))
                + ((current_base_row[203usize]) * (node_773)))
                + ((current_base_row[208usize]) * (node_841)))
                + ((current_base_row[205usize])
                    * ((current_base_row[329usize])
                        * ((next_base_row[35usize]) + (node_545)))))
                + ((current_base_row[214usize]) * (node_896)))
                + ((current_base_row[216usize]) * (node_900)))
                + ((current_base_row[213usize]) * (node_192)))
                + ((current_base_row[220usize]) * (node_900)))
                + ((current_base_row[225usize]) * (node_908)))
                + ((current_base_row[223usize]) * (node_900)))
                + ((current_base_row[224usize]) * (node_220)))
                + ((current_base_row[215usize]) * (node_768)))
                + ((current_base_row[221usize]) * (node_456)))
                + ((current_base_row[240usize]) * (node_537)))
                + ((current_base_row[232usize]) * (node_724)))
                + ((current_base_row[228usize]) * (node_414)))
                + ((current_base_row[229usize]) * (node_900)))
                + ((current_base_row[244usize]) * (node_537)))
                + ((current_base_row[246usize]) * (node_537)))
                + ((current_base_row[233usize]) * (node_224)))
                + ((current_base_row[234usize]) * (node_224)))
                + ((current_base_row[235usize]) * (node_900)))
                + ((current_base_row[237usize]) * (node_216)))
                + ((current_base_row[236usize]) * (node_558)))
                + ((current_base_row[238usize]) * (node_228)))
                + ((current_base_row[239usize]) * (node_228)))
                + ((current_base_row[241usize]) * (node_228)))
                + ((current_base_row[242usize]) * (node_904)))
                + ((current_base_row[245usize]) * (node_228)))
                + ((current_base_row[248usize]) * (node_904)))
                + ((current_base_row[251usize]) * (node_904)))
                + ((current_base_row[274usize]) * (node_317)))
                + ((current_base_row[275usize]) * (node_317)))
                + ((current_base_row[277usize]) * (node_916)))
                + ((current_base_row[279usize]) * (node_240)))
                + ((current_base_row[259usize]) * (node_773)))
                + ((current_base_row[263usize]) * (node_461))) * (node_4849))
                + ((node_892) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((((((current_base_row[202usize])
                * (node_466)) + ((current_base_row[212usize]) * (node_559)))
                + ((current_base_row[203usize]) * (node_778)))
                + ((current_base_row[208usize]) * (node_843)))
                + ((current_base_row[205usize])
                    * ((current_base_row[330usize])
                        * ((next_base_row[36usize]) + (node_545)))))
                + ((current_base_row[214usize]) * (node_900)))
                + ((current_base_row[216usize]) * (node_904)))
                + ((current_base_row[213usize]) * (node_196)))
                + ((current_base_row[220usize]) * (node_904)))
                + ((current_base_row[225usize]) * (node_912)))
                + ((current_base_row[223usize]) * (node_904)))
                + ((current_base_row[224usize]) * (node_224)))
                + ((current_base_row[215usize]) * (node_773)))
                + ((current_base_row[221usize]) * (node_461)))
                + ((current_base_row[240usize]) * (node_541)))
                + ((current_base_row[232usize]) * (node_121)))
                + ((current_base_row[228usize]) * (node_415)))
                + ((current_base_row[229usize]) * (node_904)))
                + ((current_base_row[244usize]) * (node_541)))
                + ((current_base_row[246usize]) * (node_541)))
                + ((current_base_row[233usize]) * (node_228)))
                + ((current_base_row[234usize]) * (node_228)))
                + ((current_base_row[235usize]) * (node_904)))
                + ((current_base_row[237usize]) * (node_220)))
                + ((current_base_row[236usize]) * (node_559)))
                + ((current_base_row[238usize]) * (node_232)))
                + ((current_base_row[239usize]) * (node_232)))
                + ((current_base_row[241usize]) * (node_232)))
                + ((current_base_row[242usize]) * (node_908)))
                + ((current_base_row[245usize]) * (node_232)))
                + ((current_base_row[248usize]) * (node_908)))
                + ((current_base_row[251usize]) * (node_908)))
                + ((current_base_row[274usize]) * (node_328)))
                + ((current_base_row[275usize]) * (node_328)))
                + ((current_base_row[277usize]) * (node_920)))
                + ((current_base_row[279usize]) * (node_244)))
                + ((current_base_row[259usize]) * (node_778)))
                + ((current_base_row[263usize]) * (node_466))) * (node_4849))
                + ((node_896) * (next_base_row[8usize])),
            (((((((((((((((((((((((((((((((((((((current_base_row[202usize])
                * (node_476)) + ((current_base_row[212usize]) * (node_561)))
                + ((current_base_row[203usize]) * (node_788)))
                + ((current_base_row[208usize]) * (node_132)))
                + ((current_base_row[205usize]) * (node_256)))
                + ((current_base_row[214usize]) * (node_908)))
                + ((current_base_row[216usize]) * (node_912)))
                + ((current_base_row[213usize]) * (node_204)))
                + ((current_base_row[220usize]) * (node_912)))
                + ((current_base_row[225usize]) * (node_920)))
                + ((current_base_row[223usize]) * (node_912)))
                + ((current_base_row[224usize]) * (node_232)))
                + ((current_base_row[215usize]) * (node_783)))
                + ((current_base_row[221usize]) * (node_471)))
                + ((current_base_row[232usize]) * (node_129)))
                + ((current_base_row[228usize]) * (node_417)))
                + ((current_base_row[229usize]) * (node_912)))
                + ((current_base_row[233usize]) * (node_236)))
                + ((current_base_row[234usize]) * (node_236)))
                + ((current_base_row[235usize]) * (node_912)))
                + ((current_base_row[237usize]) * (node_228)))
                + ((current_base_row[236usize]) * (node_561)))
                + ((current_base_row[238usize]) * (node_240)))
                + ((current_base_row[239usize]) * (node_240)))
                + ((current_base_row[241usize]) * (node_240)))
                + ((current_base_row[242usize]) * (node_916)))
                + ((current_base_row[245usize]) * (node_240)))
                + ((current_base_row[248usize]) * (node_916)))
                + ((current_base_row[251usize]) * (node_916)))
                + ((current_base_row[274usize]) * (node_125)))
                + ((current_base_row[275usize]) * (node_125)))
                + ((current_base_row[277usize]) * (node_159)))
                + ((current_base_row[279usize]) * (node_184)))
                + ((current_base_row[259usize]) * (node_788)))
                + ((current_base_row[263usize]) * (node_476))) * (node_4849))
                + ((node_904) * (next_base_row[8usize])),
            (((((((((((((((((((((((((((((((((((((current_base_row[202usize])
                * (node_481)) + ((current_base_row[212usize]) * (node_572)))
                + ((current_base_row[203usize]) * (node_793)))
                + ((current_base_row[208usize]) * (current_base_row[268usize])))
                + ((current_base_row[205usize]) * (node_302)))
                + ((current_base_row[214usize]) * (node_912)))
                + ((current_base_row[216usize]) * (node_916)))
                + ((current_base_row[213usize]) * (node_208)))
                + ((current_base_row[220usize]) * (node_916)))
                + ((current_base_row[225usize]) * (node_924)))
                + ((current_base_row[223usize]) * (node_916)))
                + ((current_base_row[224usize]) * (node_236)))
                + ((current_base_row[215usize]) * (node_788)))
                + ((current_base_row[221usize]) * (node_476)))
                + ((current_base_row[232usize]) * (node_932)))
                + ((current_base_row[228usize]) * (node_418)))
                + ((current_base_row[229usize]) * (node_916)))
                + ((current_base_row[233usize]) * (node_240)))
                + ((current_base_row[234usize]) * (node_240)))
                + ((current_base_row[235usize]) * (node_916)))
                + ((current_base_row[237usize]) * (node_232)))
                + ((current_base_row[236usize]) * (node_572)))
                + ((current_base_row[238usize]) * (node_244)))
                + ((current_base_row[239usize]) * (node_244)))
                + ((current_base_row[241usize]) * (node_244)))
                + ((current_base_row[242usize]) * (node_920)))
                + ((current_base_row[245usize]) * (node_244)))
                + ((current_base_row[248usize]) * (node_920)))
                + ((current_base_row[251usize]) * (node_920)))
                + ((current_base_row[274usize]) * (node_129)))
                + ((current_base_row[275usize]) * (node_129)))
                + ((current_base_row[277usize]) * (node_927)))
                + ((current_base_row[279usize]) * (node_121)))
                + ((current_base_row[259usize]) * (node_793)))
                + ((current_base_row[263usize]) * (node_481))) * (node_4849))
                + ((node_908) * (next_base_row[8usize])),
            (((((((((((((((((((((((((((((((((((((current_base_row[202usize])
                * (node_486)) + ((current_base_row[212usize]) * (node_121)))
                + ((current_base_row[203usize]) * (node_798)))
                + ((current_base_row[208usize]) * (current_base_row[260usize])))
                + ((current_base_row[205usize]) * (node_345)))
                + ((current_base_row[214usize]) * (node_916)))
                + ((current_base_row[216usize]) * (node_920)))
                + ((current_base_row[213usize]) * (node_212)))
                + ((current_base_row[220usize]) * (node_920)))
                + ((current_base_row[225usize]) * (node_159)))
                + ((current_base_row[223usize]) * (node_920)))
                + ((current_base_row[224usize]) * (node_240)))
                + ((current_base_row[215usize]) * (node_793)))
                + ((current_base_row[221usize]) * (node_481)))
                + ((current_base_row[232usize]) * (node_533)))
                + ((current_base_row[228usize]) * (node_419)))
                + ((current_base_row[229usize]) * (node_920)))
                + ((current_base_row[233usize]) * (node_244)))
                + ((current_base_row[234usize]) * (node_244)))
                + ((current_base_row[235usize]) * (node_920)))
                + ((current_base_row[237usize]) * (node_236)))
                + ((current_base_row[236usize]) * (node_121)))
                + ((current_base_row[238usize]) * (node_160)))
                + ((current_base_row[239usize]) * (node_160)))
                + ((current_base_row[241usize]) * (node_160)))
                + ((current_base_row[242usize]) * (node_924)))
                + ((current_base_row[245usize]) * (node_160)))
                + ((current_base_row[248usize]) * (node_924)))
                + ((current_base_row[251usize]) * (node_924)))
                + ((current_base_row[274usize]) * (node_932)))
                + ((current_base_row[275usize]) * (node_932)))
                + ((current_base_row[277usize]) * (node_121)))
                + ((current_base_row[279usize]) * (node_125)))
                + ((current_base_row[259usize]) * (node_798)))
                + ((current_base_row[263usize]) * (node_486))) * (node_4849))
                + ((node_912) * (next_base_row[8usize])),
            (((((((((((((((((((((((((((((((((((((current_base_row[202usize])
                * (node_491)) + ((current_base_row[212usize]) * (node_125)))
                + ((current_base_row[203usize]) * (node_803)))
                + ((current_base_row[208usize]) * (current_base_row[261usize])))
                + ((current_base_row[205usize]) * (node_387)))
                + ((current_base_row[214usize]) * (node_920)))
                + ((current_base_row[216usize]) * (node_924)))
                + ((current_base_row[213usize]) * (node_216)))
                + ((current_base_row[220usize]) * (node_924)))
                + ((current_base_row[225usize]) * (node_927)))
                + ((current_base_row[223usize]) * (node_924)))
                + ((current_base_row[224usize]) * (node_244)))
                + ((current_base_row[215usize]) * (node_798)))
                + ((current_base_row[221usize]) * (node_486)))
                + ((current_base_row[232usize]) * (node_537)))
                + ((current_base_row[228usize]) * (node_420)))
                + ((current_base_row[229usize]) * (node_924)))
                + ((current_base_row[233usize]) * (node_160)))
                + ((current_base_row[234usize]) * (node_160)))
                + ((current_base_row[235usize]) * (node_924)))
                + ((current_base_row[237usize]) * (node_240)))
                + ((current_base_row[236usize]) * (node_125)))
                + ((current_base_row[238usize]) * (node_184)))
                + ((current_base_row[239usize]) * (node_184)))
                + ((current_base_row[241usize]) * (node_184)))
                + ((current_base_row[242usize]) * (node_159)))
                + ((current_base_row[245usize]) * (node_184)))
                + ((current_base_row[248usize]) * (node_159)))
                + ((current_base_row[251usize]) * (node_159)))
                + ((current_base_row[274usize]) * (node_533)))
                + ((current_base_row[275usize]) * (node_533)))
                + ((current_base_row[277usize]) * (node_125)))
                + ((current_base_row[279usize]) * (node_129)))
                + ((current_base_row[259usize]) * (node_803)))
                + ((current_base_row[263usize]) * (node_491))) * (node_4849))
                + ((node_916) * (next_base_row[8usize])),
            (((((((((((((((((((((((((((((((((((((current_base_row[202usize])
                * (node_496)) + ((current_base_row[212usize]) * (node_129)))
                + ((current_base_row[203usize]) * (node_808)))
                + ((current_base_row[208usize]) * (node_155)))
                + ((current_base_row[205usize]) * (current_base_row[292usize])))
                + ((current_base_row[214usize]) * (node_924)))
                + ((current_base_row[216usize]) * (node_159)))
                + ((current_base_row[213usize]) * (node_220)))
                + ((current_base_row[220usize]) * (node_159)))
                + ((current_base_row[225usize]) * (node_533)))
                + ((current_base_row[223usize]) * (node_159)))
                + ((current_base_row[224usize]) * (node_160)))
                + ((current_base_row[215usize]) * (node_803)))
                + ((current_base_row[221usize]) * (node_491)))
                + ((current_base_row[232usize]) * (node_541)))
                + ((current_base_row[228usize]) * (node_421)))
                + ((current_base_row[229usize]) * (node_159)))
                + ((current_base_row[233usize]) * (node_184)))
                + ((current_base_row[234usize]) * (node_184)))
                + ((current_base_row[235usize]) * (node_159)))
                + ((current_base_row[237usize]) * (node_244)))
                + ((current_base_row[236usize]) * (node_129)))
                + ((current_base_row[238usize]) * (node_533)))
                + ((current_base_row[239usize]) * (node_533)))
                + ((current_base_row[241usize]) * (node_533)))
                + ((current_base_row[242usize]) * (node_927)))
                + ((current_base_row[245usize]) * (node_533)))
                + ((current_base_row[248usize]) * (node_927)))
                + ((current_base_row[251usize]) * (node_927)))
                + ((current_base_row[274usize]) * (node_537)))
                + ((current_base_row[275usize]) * (node_537)))
                + ((current_base_row[277usize]) * (node_129)))
                + ((current_base_row[279usize]) * (node_932)))
                + ((current_base_row[259usize]) * (node_808)))
                + ((current_base_row[263usize]) * (node_496))) * (node_4849))
                + ((node_920) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((((current_base_row[202usize]) * (node_501))
                + ((current_base_row[212usize]) * (node_117)))
                + ((current_base_row[203usize]) * (node_813)))
                + ((current_base_row[208usize]) * (node_121)))
                + ((current_base_row[205usize]) * (node_827)))
                + ((current_base_row[214usize]) * (node_159)))
                + ((current_base_row[216usize]) * (node_927)))
                + ((current_base_row[213usize]) * (node_224)))
                + ((current_base_row[220usize]) * (node_927)))
                + ((current_base_row[225usize]) * (node_537)))
                + ((current_base_row[223usize]) * (node_927)))
                + ((current_base_row[224usize]) * (node_184)))
                + ((current_base_row[215usize]) * (node_808)))
                + ((current_base_row[221usize]) * (node_496)))
                + ((current_base_row[228usize]) * (node_422)))
                + ((current_base_row[229usize]) * (node_927)))
                + ((current_base_row[233usize]) * (node_533)))
                + ((current_base_row[234usize]) * (node_533)))
                + ((current_base_row[235usize]) * (node_927)))
                + ((current_base_row[237usize]) * (node_160)))
                + ((current_base_row[236usize]) * (node_932)))
                + ((current_base_row[238usize]) * (node_537)))
                + ((current_base_row[239usize]) * (node_537)))
                + ((current_base_row[241usize]) * (node_537)))
                + ((current_base_row[242usize]) * (node_533)))
                + ((current_base_row[245usize]) * (node_537)))
                + ((current_base_row[248usize]) * (node_533)))
                + ((current_base_row[251usize]) * (node_533)))
                + ((current_base_row[274usize]) * (node_541)))
                + ((current_base_row[275usize]) * (node_541)))
                + ((current_base_row[277usize]) * (node_932)))
                + ((current_base_row[279usize]) * (node_533)))
                + ((current_base_row[259usize]) * (node_813)))
                + ((current_base_row[263usize]) * (node_501))) * (node_4849))
                + ((node_924) * (next_base_row[8usize])),
            ((((((((((((((((((((((((((((((((((current_base_row[202usize]) * (node_505))
                + ((current_base_row[212usize]) * (node_533)))
                + ((current_base_row[203usize]) * (node_817)))
                + ((current_base_row[208usize]) * (node_125)))
                + ((current_base_row[205usize]) * (node_829)))
                + ((current_base_row[214usize]) * (node_927)))
                + ((current_base_row[216usize]) * (node_533)))
                + ((current_base_row[213usize]) * (node_228)))
                + ((current_base_row[220usize]) * (node_533)))
                + ((current_base_row[225usize]) * (node_541)))
                + ((current_base_row[223usize]) * (node_533)))
                + ((current_base_row[224usize]) * (node_533)))
                + ((current_base_row[215usize]) * (node_813)))
                + ((current_base_row[221usize]) * (node_501)))
                + ((current_base_row[228usize]) * (node_533)))
                + ((current_base_row[229usize]) * (node_533)))
                + ((current_base_row[233usize]) * (node_537)))
                + ((current_base_row[234usize]) * (node_537)))
                + ((current_base_row[235usize]) * (node_533)))
                + ((current_base_row[237usize]) * (node_184)))
                + ((current_base_row[236usize]) * (node_533)))
                + ((current_base_row[238usize]) * (node_541)))
                + ((current_base_row[239usize]) * (node_541)))
                + ((current_base_row[241usize]) * (node_541)))
                + ((current_base_row[242usize]) * (node_537)))
                + ((current_base_row[245usize]) * (node_541)))
                + ((current_base_row[248usize]) * (node_537)))
                + ((current_base_row[251usize]) * (node_537)))
                + ((current_base_row[277usize]) * (node_533)))
                + ((current_base_row[279usize]) * (node_537)))
                + ((current_base_row[259usize])
                    * ((node_817)
                        + (((next_ext_row[3usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((challenges[StandardInputIndeterminate])
                                    * (((challenges[StandardInputIndeterminate])
                                        * (((challenges[StandardInputIndeterminate])
                                            * (((challenges[StandardInputIndeterminate])
                                                * ((node_1372) + (next_base_row[26usize])))
                                                + (next_base_row[25usize]))) + (next_base_row[24usize])))
                                        + (next_base_row[23usize]))) + (next_base_row[22usize]))))
                            * (current_base_row[189usize])))))
                + ((current_base_row[263usize])
                    * ((node_505)
                        + (((next_ext_row[4usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((challenges[StandardOutputIndeterminate]) * (node_1435))
                                    + (current_base_row[26usize]))))
                            * (current_base_row[189usize]))))) * (node_4849))
                + ((node_159) * (next_base_row[8usize])),
            (((((((((((((((((((((((((((((current_base_row[202usize]) * (node_508))
                + ((current_base_row[212usize]) * (node_537)))
                + ((current_base_row[203usize]) * (node_820)))
                + ((current_base_row[208usize]) * (node_129)))
                + ((current_base_row[205usize]) * (node_831)))
                + ((current_base_row[214usize]) * (node_533)))
                + ((current_base_row[216usize]) * (node_537)))
                + ((current_base_row[213usize]) * (node_232)))
                + ((current_base_row[220usize]) * (node_537)))
                + ((current_base_row[223usize]) * (node_537)))
                + ((current_base_row[224usize]) * (node_537)))
                + ((current_base_row[215usize]) * (node_817)))
                + ((current_base_row[221usize]) * (node_505)))
                + ((current_base_row[228usize]) * (node_537)))
                + ((current_base_row[229usize]) * (node_537)))
                + ((current_base_row[233usize]) * (node_541)))
                + ((current_base_row[234usize]) * (node_541)))
                + ((current_base_row[235usize]) * (node_537)))
                + ((current_base_row[237usize]) * (node_533)))
                + ((current_base_row[236usize]) * (node_537)))
                + ((current_base_row[242usize]) * (node_541)))
                + ((current_base_row[248usize]) * (node_541)))
                + ((current_base_row[251usize]) * (node_541)))
                + ((current_base_row[277usize]) * (node_537)))
                + ((current_base_row[279usize]) * (node_541)))
                + ((current_base_row[259usize])
                    * ((node_820)
                        + (((next_ext_row[3usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((challenges[StandardInputIndeterminate])
                                    * (((challenges[StandardInputIndeterminate])
                                        * (((challenges[StandardInputIndeterminate])
                                            * ((node_1372) + (next_base_row[25usize])))
                                            + (next_base_row[24usize]))) + (next_base_row[23usize])))
                                    + (next_base_row[22usize]))))
                            * (current_base_row[188usize])))))
                + ((current_base_row[263usize])
                    * ((node_508)
                        + (((next_ext_row[4usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (node_1435))) * (current_base_row[188usize])))))
                * (node_4849)) + ((node_927) * (next_base_row[8usize])),
            (((((((((((((((((((((((current_base_row[202usize]) * (node_510))
                + ((current_base_row[212usize]) * (node_541)))
                + ((current_base_row[203usize]) * (node_822)))
                + ((current_base_row[208usize]) * (node_117)))
                + ((current_base_row[205usize]) * (node_833)))
                + ((current_base_row[214usize]) * (node_537)))
                + ((current_base_row[216usize]) * (node_541)))
                + ((current_base_row[213usize]) * (node_236)))
                + ((current_base_row[220usize]) * (node_541)))
                + ((current_base_row[223usize]) * (node_541)))
                + ((current_base_row[224usize]) * (node_541)))
                + ((current_base_row[215usize]) * (node_820)))
                + ((current_base_row[221usize]) * (node_508)))
                + ((current_base_row[228usize]) * (node_541)))
                + ((current_base_row[229usize]) * (node_541)))
                + ((current_base_row[235usize]) * (node_541)))
                + ((current_base_row[237usize]) * (node_537)))
                + ((current_base_row[236usize]) * (node_541)))
                + ((current_base_row[277usize]) * (node_541)))
                + ((current_base_row[259usize])
                    * ((node_822)
                        + (((next_ext_row[3usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((challenges[StandardInputIndeterminate])
                                    * (((challenges[StandardInputIndeterminate])
                                        * ((node_1372) + (next_base_row[24usize])))
                                        + (next_base_row[23usize]))) + (next_base_row[22usize]))))
                            * (current_base_row[187usize])))))
                + ((current_base_row[263usize])
                    * ((node_510)
                        + (((next_ext_row[4usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (node_1430))) * (current_base_row[187usize])))))
                * (node_4849)) + ((node_533) * (next_base_row[8usize])),
            (((((((((((((current_base_row[202usize]) * (node_270))
                + ((current_base_row[203usize]) * (node_589)))
                + ((current_base_row[208usize]) * (node_546)))
                + ((current_base_row[205usize]) * (node_835)))
                + ((current_base_row[214usize]) * (node_541)))
                + ((current_base_row[213usize]) * (node_240)))
                + ((current_base_row[215usize]) * (node_822)))
                + ((current_base_row[221usize]) * (node_510)))
                + ((current_base_row[237usize]) * (node_541)))
                + ((current_base_row[259usize])
                    * ((node_589)
                        + (((next_ext_row[3usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((challenges[StandardInputIndeterminate])
                                    * ((node_1372) + (next_base_row[23usize])))
                                    + (next_base_row[22usize]))))
                            * (current_base_row[186usize])))))
                + ((current_base_row[263usize])
                    * ((node_270)
                        + (((next_ext_row[4usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (node_1425))) * (current_base_row[186usize])))))
                * (node_4849)) + ((node_537) * (next_base_row[8usize])),
            (((((((((((current_base_row[202usize]) * (current_base_row[315usize]))
                + ((current_base_row[203usize]) * (current_base_row[315usize])))
                + ((current_base_row[208usize]) * (node_547)))
                + ((current_base_row[205usize]) * (node_837)))
                + ((current_base_row[213usize]) * (node_244)))
                + ((current_base_row[215usize]) * (node_589)))
                + ((current_base_row[221usize]) * (node_270)))
                + ((current_base_row[259usize])
                    * (((next_ext_row[3usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((node_1372) + (next_base_row[22usize]))))
                        * (current_base_row[185usize]))))
                + ((current_base_row[263usize])
                    * (((next_ext_row[4usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_1420))) * (current_base_row[185usize]))))
                * (node_4849)) + ((node_541) * (next_base_row[8usize])),
            ((((((((((current_base_row[202usize]) * (current_base_row[270usize]))
                + ((current_base_row[203usize]) * (current_base_row[270usize])))
                + ((current_base_row[208usize]) * (node_549)))
                + ((current_base_row[205usize]) * (node_841)))
                + ((current_base_row[213usize]) * (node_184)))
                + ((current_base_row[215usize]) * (current_base_row[312usize])))
                + ((current_base_row[221usize]) * (current_base_row[312usize])))
                + ((current_base_row[259usize]) * (current_base_row[312usize])))
                + ((current_base_row[263usize]) * (current_base_row[312usize])))
                * (node_4849),
            ((((((((((current_base_row[202usize]) * (current_base_row[271usize]))
                + ((current_base_row[203usize]) * (current_base_row[271usize])))
                + ((current_base_row[208usize]) * (node_550)))
                + ((current_base_row[205usize]) * (node_843)))
                + ((current_base_row[213usize]) * (node_533)))
                + ((current_base_row[215usize]) * (current_base_row[270usize])))
                + ((current_base_row[221usize]) * (current_base_row[270usize])))
                + ((current_base_row[259usize]) * (current_base_row[270usize])))
                + ((current_base_row[263usize]) * (current_base_row[270usize])))
                * (node_4849),
            ((((((((((current_base_row[202usize]) * (current_base_row[314usize]))
                + ((current_base_row[203usize]) * (current_base_row[314usize])))
                + ((current_base_row[208usize]) * (node_551)))
                + ((current_base_row[205usize]) * (node_845)))
                + ((current_base_row[213usize]) * (node_537)))
                + ((current_base_row[215usize]) * (current_base_row[271usize])))
                + ((current_base_row[221usize]) * (current_base_row[271usize])))
                + ((current_base_row[259usize]) * (current_base_row[271usize])))
                + ((current_base_row[263usize]) * (current_base_row[271usize])))
                * (node_4849),
            ((((((((((current_base_row[202usize]) * (current_base_row[317usize]))
                + ((current_base_row[203usize]) * (current_base_row[317usize])))
                + ((current_base_row[208usize]) * (node_552)))
                + ((current_base_row[205usize])
                    * (((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[185usize]))) * (node_868))))
                + ((current_base_row[213usize]) * (node_541)))
                + ((current_base_row[215usize]) * (current_base_row[314usize])))
                + ((current_base_row[221usize]) * (current_base_row[314usize])))
                + ((current_base_row[259usize]) * (current_base_row[314usize])))
                + ((current_base_row[263usize]) * (current_base_row[314usize])))
                * (node_4849),
            (((((((((current_base_row[202usize]) * (node_533))
                + ((current_base_row[203usize]) * (node_533)))
                + ((current_base_row[208usize]) * (node_558)))
                + ((current_base_row[205usize])
                    * (((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[270usize]))) * (node_892))))
                + ((current_base_row[215usize]) * (current_base_row[331usize])))
                + ((current_base_row[221usize]) * (current_base_row[331usize])))
                + ((current_base_row[259usize]) * (current_base_row[331usize])))
                + ((current_base_row[263usize]) * (current_base_row[331usize])))
                * (node_4849),
            (((((((((current_base_row[202usize]) * (node_537))
                + ((current_base_row[203usize]) * (node_537)))
                + ((current_base_row[208usize]) * (node_559)))
                + ((current_base_row[205usize])
                    * (((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[271usize]))) * (node_896))))
                + ((current_base_row[215usize]) * (node_537)))
                + ((current_base_row[221usize]) * (node_537)))
                + ((current_base_row[259usize]) * (node_533)))
                + ((current_base_row[263usize]) * (node_533))) * (node_4849),
            (((((((((current_base_row[202usize]) * (node_541))
                + ((current_base_row[203usize]) * (node_541)))
                + ((current_base_row[208usize]) * (node_560)))
                + ((current_base_row[205usize])
                    * (((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[314usize]))) * (node_900))))
                + ((current_base_row[215usize]) * (node_541)))
                + ((current_base_row[221usize]) * (node_541)))
                + ((current_base_row[259usize]) * (node_541)))
                + ((current_base_row[263usize]) * (node_537))) * (node_4849),
            (((current_base_row[208usize]) * (node_572))
                + ((current_base_row[205usize])
                    * (((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[318usize]))) * (node_908))))
                * (node_4849),
            (((current_base_row[208usize]) * (node_533))
                + ((current_base_row[205usize])
                    * (((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[273usize]))) * (node_912))))
                * (node_4849),
            (((current_base_row[208usize]) * (node_537))
                + ((current_base_row[205usize])
                    * (((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[329usize]))) * (node_916))))
                * (node_4849),
            (((current_base_row[208usize]) * (node_541))
                + ((current_base_row[205usize])
                    * (((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[330usize]))) * (node_920))))
                * (node_4849),
            ((current_base_row[205usize]) * (node_927)) * (node_4849),
            ((current_base_row[205usize]) * (node_533)) * (node_4849),
            ((current_base_row[205usize]) * (node_537)) * (node_4849),
            ((current_base_row[205usize]) * (node_541)) * (node_4849),
            (((next_ext_row[13usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (current_ext_row[13usize])))
                * ((challenges[ClockJumpDifferenceLookupIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[7usize]))))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (next_base_row[45usize])),
            ((node_4849)
                * (((node_5017)
                    * ((challenges[InstructionLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((((challenges[ProgramAddressWeight])
                                * (next_base_row[9usize]))
                                + ((challenges[ProgramInstructionWeight])
                                    * (next_base_row[10usize])))
                                + ((challenges[ProgramNextInstructionWeight])
                                    * (next_base_row[11usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((next_base_row[8usize]) * (node_5017)),
            (next_ext_row[8usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[8usize])
                        * ((challenges[JumpStackIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((((((challenges[JumpStackClkWeight])
                                    * (next_base_row[7usize]))
                                    + ((challenges[JumpStackCiWeight])
                                        * (next_base_row[10usize])))
                                    + ((challenges[JumpStackJspWeight])
                                        * (next_base_row[19usize])))
                                    + ((challenges[JumpStackJsoWeight])
                                        * (next_base_row[20usize])))
                                    + ((challenges[JumpStackJsdWeight])
                                        * (next_base_row[21usize]))))))),
            (((next_base_row[10usize])
                + (BFieldElement::from_raw_u64(18446743992105173011u64)))
                * ((next_ext_row[9usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_ext_row[9usize]))))
                + ((current_base_row[344usize])
                    * (((next_ext_row[9usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((challenges[HashInputIndeterminate])
                                * (current_ext_row[9usize])))) + (node_5106))),
            (((current_base_row[10usize])
                + (BFieldElement::from_raw_u64(18446743992105173011u64)))
                * ((next_ext_row[10usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_ext_row[10usize]))))
                + ((current_base_row[240usize])
                    * (((next_ext_row[10usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((challenges[HashDigestIndeterminate])
                                * (current_ext_row[10usize]))))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_5089)))),
            (((((((current_base_row[10usize])
                + (BFieldElement::from_raw_u64(18446743897615892521u64)))
                * ((current_base_row[10usize])
                    + (BFieldElement::from_raw_u64(18446743923385696291u64))))
                * ((current_base_row[10usize])
                    + (BFieldElement::from_raw_u64(18446743863256154161u64))))
                * ((next_ext_row[11usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_ext_row[11usize]))))
                + ((current_base_row[229usize]) * (node_5166)))
                + ((current_base_row[244usize])
                    * ((node_5166)
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((((((((challenges[HashStateWeight0])
                                * (current_base_row[22usize]))
                                + ((challenges[HashStateWeight1])
                                    * (current_base_row[23usize])))
                                + ((challenges[HashStateWeight2])
                                    * (current_base_row[24usize])))
                                + ((challenges[HashStateWeight3])
                                    * (current_base_row[25usize])))
                                + ((challenges[HashStateWeight4])
                                    * (current_base_row[26usize])))
                                + ((challenges[HashStateWeight5])
                                    * (current_base_row[27usize])))
                                + ((challenges[HashStateWeight6])
                                    * (current_base_row[28usize])))
                                + ((challenges[HashStateWeight7])
                                    * (current_base_row[29usize])))
                                + ((challenges[HashStateWeight8])
                                    * (current_base_row[30usize])))
                                + ((challenges[HashStateWeight9])
                                    * (current_base_row[31usize])))))))
                + ((current_base_row[246usize]) * ((node_5166) + (node_5106))),
            (((((((((current_base_row[236usize])
                * (((node_5229) * (((node_5186) + (node_5189)) + (node_5193)))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((current_base_row[238usize]) * (node_5233)))
                + ((current_base_row[239usize]) * (node_5233)))
                + ((current_base_row[241usize])
                    * (((node_5229)
                        * (((node_5200)
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((challenges[U32CiWeight])
                                    * (BFieldElement::from_raw_u64(60129542130u64)))))
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((challenges[U32ResultWeight])
                                    * (((current_base_row[22usize])
                                        + (current_base_row[23usize])) + (node_1314)))
                                    * (BFieldElement::from_raw_u64(9223372036854775808u64))))))
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)))))
                + ((current_base_row[245usize]) * (node_5233)))
                + ((current_base_row[242usize]) * (node_5237)))
                + ((current_base_row[248usize])
                    * (((((node_5229) * (node_5223)) * (node_5227))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_5223)))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_5227)))))
                + ((current_base_row[251usize]) * (node_5237)))
                + (((BFieldElement::from_raw_u64(4294967295u64))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[14usize]))) * (node_5229)),
            (((next_ext_row[14usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[14usize])
                        * ((challenges[OpStackIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((((challenges[OpStackClkWeight])
                                    * (next_base_row[46usize]))
                                    + ((challenges[OpStackIb1Weight])
                                        * (next_base_row[47usize])))
                                    + ((challenges[OpStackPointerWeight])
                                        * (next_base_row[48usize])))
                                    + ((challenges[OpStackFirstUnderflowElementWeight])
                                        * (next_base_row[49usize])))))))) * (node_5283))
                + (((next_ext_row[14usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_ext_row[14usize]))) * (node_5300)),
            ((((((node_5309)
                * ((challenges[ClockJumpDifferenceLookupIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((next_base_row[46usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (current_base_row[46usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64))) * (node_5277))
                * (node_5283)) + ((node_5309) * (node_5276)))
                + ((node_5309) * (node_5300)),
            ((node_5352)
                * ((next_ext_row[16usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((current_ext_row[16usize]) * (node_5371)))))
                + ((node_5355) * ((next_ext_row[16usize]) + (node_5376))),
            ((node_5352)
                * (((next_ext_row[17usize]) + (node_5376))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((node_5371) * (current_ext_row[17usize])))))
                + ((node_5355)
                    * ((next_ext_row[17usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_ext_row[17usize])))),
            ((node_5352)
                * (((next_ext_row[18usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((challenges[RamTableBezoutRelationIndeterminate])
                            * (current_ext_row[18usize]))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[55usize]))))
                + ((node_5355)
                    * ((next_ext_row[18usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_ext_row[18usize])))),
            ((node_5352)
                * (((next_ext_row[19usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((challenges[RamTableBezoutRelationIndeterminate])
                            * (current_ext_row[19usize]))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[56usize]))))
                + ((node_5355)
                    * ((next_ext_row[19usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_ext_row[19usize])))),
            (((next_ext_row[20usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[20usize])
                        * ((challenges[RamIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((((next_base_row[50usize]) * (challenges[RamClkWeight]))
                                    + ((next_base_row[52usize])
                                        * (challenges[RamPointerWeight])))
                                    + ((next_base_row[53usize]) * (challenges[RamValueWeight])))
                                    + ((next_base_row[51usize])
                                        * (challenges[RamInstructionTypeWeight]))))))))
                * (node_5346))
                + (((next_ext_row[20usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_ext_row[20usize]))) * (node_5423)),
            (((current_ext_row[80usize]) * (node_5346)) + ((node_5432) * (node_5352)))
                + ((node_5432) * (node_5423)),
            (next_ext_row[22usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[22usize])
                        * ((challenges[JumpStackIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((((((challenges[JumpStackClkWeight])
                                    * (next_base_row[57usize]))
                                    + ((challenges[JumpStackCiWeight])
                                        * (next_base_row[58usize])))
                                    + ((challenges[JumpStackJspWeight])
                                        * (next_base_row[59usize])))
                                    + ((challenges[JumpStackJsoWeight])
                                        * (next_base_row[60usize])))
                                    + ((challenges[JumpStackJsdWeight])
                                        * (next_base_row[61usize]))))))),
            ((node_5461)
                * (((node_5495)
                    * ((challenges[ClockJumpDifferenceLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_5473))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_5460) * (node_5495)),
            (((current_base_row[338usize])
                * (((next_ext_row[24usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((challenges[ProgramAttestationSendChunkIndeterminate])
                            * (current_ext_row[24usize]))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((((((((((((((((((((challenges[ProgramAttestationPrepareChunkIndeterminate])
                            + (node_6264))
                            * (challenges[ProgramAttestationPrepareChunkIndeterminate]))
                            + (node_6275))
                            * (challenges[ProgramAttestationPrepareChunkIndeterminate]))
                            + (node_6286))
                            * (challenges[ProgramAttestationPrepareChunkIndeterminate]))
                            + (node_6297))
                            * (challenges[ProgramAttestationPrepareChunkIndeterminate]))
                            + (next_base_row[97usize]))
                            * (challenges[ProgramAttestationPrepareChunkIndeterminate]))
                            + (next_base_row[98usize]))
                            * (challenges[ProgramAttestationPrepareChunkIndeterminate]))
                            + (next_base_row[99usize]))
                            * (challenges[ProgramAttestationPrepareChunkIndeterminate]))
                            + (next_base_row[100usize]))
                            * (challenges[ProgramAttestationPrepareChunkIndeterminate]))
                            + (next_base_row[101usize]))
                            * (challenges[ProgramAttestationPrepareChunkIndeterminate]))
                            + (next_base_row[102usize])))))
                + ((next_base_row[64usize]) * (node_6615)))
                + ((node_6398) * (node_6615)),
            ((current_base_row[339usize]) * (node_6398))
                * (((((((((((challenges[CompressProgramDigestIndeterminate])
                    + (node_5532)) * (challenges[CompressProgramDigestIndeterminate]))
                    + (node_5543)) * (challenges[CompressProgramDigestIndeterminate]))
                    + (node_5554)) * (challenges[CompressProgramDigestIndeterminate]))
                    + (node_5565)) * (challenges[CompressProgramDigestIndeterminate]))
                    + (current_base_row[97usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (challenges[CompressedProgramDigest]))),
            (current_base_row[352usize])
                * ((((((node_6443) + (node_6444)) + (node_6446)) + (node_6448))
                    + (node_6450)) + (node_6452)),
            (current_base_row[353usize])
                * (((((((((((((((((challenges[HashStateWeight0])
                    * ((node_6264)
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_5532))))
                    + ((challenges[HashStateWeight1])
                        * ((node_6275)
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (node_5543)))))
                    + ((challenges[HashStateWeight2])
                        * ((node_6286)
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (node_5554)))))
                    + ((challenges[HashStateWeight3])
                        * ((node_6297)
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (node_5565)))))
                    + ((challenges[HashStateWeight4])
                        * ((next_base_row[97usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (current_base_row[97usize])))))
                    + ((challenges[HashStateWeight5])
                        * ((next_base_row[98usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (current_base_row[98usize])))))
                    + ((challenges[HashStateWeight6])
                        * ((next_base_row[99usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (current_base_row[99usize])))))
                    + ((challenges[HashStateWeight7])
                        * ((next_base_row[100usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (current_base_row[100usize])))))
                    + ((challenges[HashStateWeight8])
                        * ((next_base_row[101usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (current_base_row[101usize])))))
                    + ((challenges[HashStateWeight9])
                        * ((next_base_row[102usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (current_base_row[102usize]))))) + (node_6443))
                    + (node_6444)) + (node_6446)) + (node_6448)) + (node_6450))
                    + (node_6452)),
            ((((current_base_row[311usize]) * (current_base_row[316usize]))
                * (((next_ext_row[25usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((challenges[HashInputIndeterminate])
                            * (current_ext_row[25usize])))) + (node_6544)))
                + ((next_base_row[64usize]) * (node_6521)))
                + ((node_6404) * (node_6521)),
            ((((node_6564) * (current_base_row[316usize]))
                * (((next_ext_row[26usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((challenges[HashDigestIndeterminate])
                            * (current_ext_row[26usize]))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (node_6530)))) + ((node_6462) * (node_6555)))
                + ((node_6404) * (node_6555)),
            ((((current_base_row[311usize]) * (node_6513))
                * ((((next_ext_row[27usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((challenges[SpongeIndeterminate])
                            * (current_ext_row[27usize]))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((challenges[HashCIWeight]) * (next_base_row[63usize]))))
                    + (node_6544))) + ((next_base_row[64usize]) * (node_6581)))
                + ((((node_6408) * (node_6516)) * (node_6584)) * (node_6581)),
            (((current_base_row[278usize])
                * (((node_6635)
                    * ((challenges[HashCascadeLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[HashCascadeLookInWeight])
                                * (next_base_row[65usize]))
                                + ((challenges[HashCascadeLookOutWeight])
                                    * (next_base_row[81usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_6564) * (node_6635))) + ((node_6643) * (node_6635)),
            (((current_base_row[278usize])
                * (((node_6656)
                    * ((challenges[HashCascadeLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[HashCascadeLookInWeight])
                                * (next_base_row[66usize]))
                                + ((challenges[HashCascadeLookOutWeight])
                                    * (next_base_row[82usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_6564) * (node_6656))) + ((node_6643) * (node_6656)),
            (((current_base_row[278usize])
                * (((node_6673)
                    * ((challenges[HashCascadeLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[HashCascadeLookInWeight])
                                * (next_base_row[67usize]))
                                + ((challenges[HashCascadeLookOutWeight])
                                    * (next_base_row[83usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_6564) * (node_6673))) + ((node_6643) * (node_6673)),
            (((current_base_row[278usize])
                * (((node_6690)
                    * ((challenges[HashCascadeLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[HashCascadeLookInWeight])
                                * (next_base_row[68usize]))
                                + ((challenges[HashCascadeLookOutWeight])
                                    * (next_base_row[84usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_6564) * (node_6690))) + ((node_6643) * (node_6690)),
            (((current_base_row[278usize])
                * (((node_6707)
                    * ((challenges[HashCascadeLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[HashCascadeLookInWeight])
                                * (next_base_row[69usize]))
                                + ((challenges[HashCascadeLookOutWeight])
                                    * (next_base_row[85usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_6564) * (node_6707))) + ((node_6643) * (node_6707)),
            (((current_base_row[278usize])
                * (((node_6724)
                    * ((challenges[HashCascadeLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[HashCascadeLookInWeight])
                                * (next_base_row[70usize]))
                                + ((challenges[HashCascadeLookOutWeight])
                                    * (next_base_row[86usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_6564) * (node_6724))) + ((node_6643) * (node_6724)),
            (((current_base_row[278usize])
                * (((node_6741)
                    * ((challenges[HashCascadeLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[HashCascadeLookInWeight])
                                * (next_base_row[71usize]))
                                + ((challenges[HashCascadeLookOutWeight])
                                    * (next_base_row[87usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_6564) * (node_6741))) + ((node_6643) * (node_6741)),
            (((current_base_row[278usize])
                * (((node_6758)
                    * ((challenges[HashCascadeLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[HashCascadeLookInWeight])
                                * (next_base_row[72usize]))
                                + ((challenges[HashCascadeLookOutWeight])
                                    * (next_base_row[88usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_6564) * (node_6758))) + ((node_6643) * (node_6758)),
            (((current_base_row[278usize])
                * (((node_6775)
                    * ((challenges[HashCascadeLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[HashCascadeLookInWeight])
                                * (next_base_row[73usize]))
                                + ((challenges[HashCascadeLookOutWeight])
                                    * (next_base_row[89usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_6564) * (node_6775))) + ((node_6643) * (node_6775)),
            (((current_base_row[278usize])
                * (((node_6792)
                    * ((challenges[HashCascadeLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[HashCascadeLookInWeight])
                                * (next_base_row[74usize]))
                                + ((challenges[HashCascadeLookOutWeight])
                                    * (next_base_row[90usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_6564) * (node_6792))) + ((node_6643) * (node_6792)),
            (((current_base_row[278usize])
                * (((node_6809)
                    * ((challenges[HashCascadeLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[HashCascadeLookInWeight])
                                * (next_base_row[75usize]))
                                + ((challenges[HashCascadeLookOutWeight])
                                    * (next_base_row[91usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_6564) * (node_6809))) + ((node_6643) * (node_6809)),
            (((current_base_row[278usize])
                * (((node_6826)
                    * ((challenges[HashCascadeLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[HashCascadeLookInWeight])
                                * (next_base_row[76usize]))
                                + ((challenges[HashCascadeLookOutWeight])
                                    * (next_base_row[92usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_6564) * (node_6826))) + ((node_6643) * (node_6826)),
            (((current_base_row[278usize])
                * (((node_6843)
                    * ((challenges[HashCascadeLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[HashCascadeLookInWeight])
                                * (next_base_row[77usize]))
                                + ((challenges[HashCascadeLookOutWeight])
                                    * (next_base_row[93usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_6564) * (node_6843))) + ((node_6643) * (node_6843)),
            (((current_base_row[278usize])
                * (((node_6860)
                    * ((challenges[HashCascadeLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[HashCascadeLookInWeight])
                                * (next_base_row[78usize]))
                                + ((challenges[HashCascadeLookOutWeight])
                                    * (next_base_row[94usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_6564) * (node_6860))) + ((node_6643) * (node_6860)),
            (((current_base_row[278usize])
                * (((node_6877)
                    * ((challenges[HashCascadeLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[HashCascadeLookInWeight])
                                * (next_base_row[79usize]))
                                + ((challenges[HashCascadeLookOutWeight])
                                    * (next_base_row[95usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_6564) * (node_6877))) + ((node_6643) * (node_6877)),
            (((current_base_row[278usize])
                * (((node_6894)
                    * ((challenges[HashCascadeLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[HashCascadeLookInWeight])
                                * (next_base_row[80usize]))
                                + ((challenges[HashCascadeLookOutWeight])
                                    * (next_base_row[96usize]))))))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                + ((node_6564) * (node_6894))) + ((node_6643) * (node_6894)),
            ((node_6920)
                * (((node_6930)
                    * ((challenges[HashCascadeLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((challenges[HashCascadeLookInWeight])
                                * (((BFieldElement::from_raw_u64(1099511627520u64))
                                    * (next_base_row[130usize])) + (next_base_row[131usize])))
                                + ((challenges[HashCascadeLookOutWeight])
                                    * (((BFieldElement::from_raw_u64(1099511627520u64))
                                        * (next_base_row[132usize]))
                                        + (next_base_row[133usize])))))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[134usize]))))
                + ((next_base_row[129usize]) * (node_6930)),
            ((node_6920)
                * ((((((node_6946)
                    * ((challenges[CascadeLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_6941))))
                    * ((challenges[CascadeLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (node_6944))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((BFieldElement::from_raw_u64(8589934590u64))
                            * (challenges[CascadeLookupIndeterminate])))) + (node_6941))
                    + (node_6944))) + ((next_base_row[129usize]) * (node_6946)),
            ((node_6972)
                * (((node_6984)
                    * ((challenges[CascadeLookupIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((next_base_row[136usize])
                                * (challenges[LookupTableInputWeight]))
                                + ((next_base_row[137usize])
                                    * (challenges[LookupTableOutputWeight]))))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[138usize]))))
                + ((next_base_row[135usize]) * (node_6984)),
            ((node_6972)
                * (((next_ext_row[47usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((current_ext_row[47usize])
                            * (challenges[LookupTablePublicIndeterminate]))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[137usize]))))
                + ((next_base_row[135usize])
                    * ((next_ext_row[47usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_ext_row[47usize])))),
            (node_7032) * (node_7149),
            (next_base_row[139usize])
                * (((node_7149)
                    * ((challenges[U32Indeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((challenges[U32CiWeight]) * (next_base_row[142usize]))
                                + ((challenges[U32LhsWeight]) * (next_base_row[143usize])))
                                + ((challenges[U32RhsWeight]) * (next_base_row[145usize])))
                                + ((challenges[U32ResultWeight])
                                    * (next_base_row[147usize]))))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (next_base_row[148usize]))),
            (current_ext_row[50usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_178)
                        * ((challenges[OpStackIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_169)
                                    + ((challenges[OpStackPointerWeight])
                                        * ((next_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(4294967295u64)))))
                                    + ((challenges[OpStackFirstUnderflowElementWeight])
                                        * (next_base_row[36usize]))))))),
            (current_ext_row[51usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_568)
                        * ((challenges[OpStackIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_169)
                                    + ((challenges[OpStackPointerWeight])
                                        * ((current_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(4294967295u64)))))
                                    + ((challenges[OpStackFirstUnderflowElementWeight])
                                        * (current_base_row[36usize]))))))),
            (current_ext_row[52usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[50usize])
                        * ((challenges[OpStackIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_169)
                                    + ((challenges[OpStackPointerWeight])
                                        * ((next_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(8589934590u64)))))
                                    + ((challenges[OpStackFirstUnderflowElementWeight])
                                        * (next_base_row[35usize]))))))),
            (current_ext_row[53usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[51usize])
                        * ((challenges[OpStackIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_169)
                                    + ((challenges[OpStackPointerWeight])
                                        * ((current_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(8589934590u64)))))
                                    + ((challenges[OpStackFirstUnderflowElementWeight])
                                        * (current_base_row[35usize]))))))),
            (current_ext_row[54usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[52usize])
                        * ((challenges[OpStackIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_169)
                                    + ((challenges[OpStackPointerWeight])
                                        * ((next_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(12884901885u64)))))
                                    + ((challenges[OpStackFirstUnderflowElementWeight])
                                        * (next_base_row[34usize]))))))),
            (current_ext_row[55usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[53usize])
                        * ((challenges[OpStackIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_169)
                                    + ((challenges[OpStackPointerWeight])
                                        * ((current_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(12884901885u64)))))
                                    + ((challenges[OpStackFirstUnderflowElementWeight])
                                        * (current_base_row[34usize]))))))),
            (current_ext_row[56usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[54usize])
                        * ((challenges[OpStackIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_169)
                                    + ((challenges[OpStackPointerWeight])
                                        * ((next_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(17179869180u64)))))
                                    + ((challenges[OpStackFirstUnderflowElementWeight])
                                        * (next_base_row[33usize]))))))),
            (current_ext_row[57usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_1006)
                        * ((challenges[RamIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_997)
                                    + (((next_base_row[22usize])
                                        + (BFieldElement::from_raw_u64(8589934590u64)))
                                        * (challenges[RamPointerWeight])))
                                    + ((next_base_row[24usize])
                                        * (challenges[RamValueWeight]))))))),
            (current_ext_row[58usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((node_1088)
                        * ((challenges[RamIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_994)
                                    + (((current_base_row[22usize])
                                        + (BFieldElement::from_raw_u64(4294967295u64)))
                                        * (challenges[RamPointerWeight])))
                                    + ((current_base_row[24usize])
                                        * (challenges[RamValueWeight]))))))),
            (current_ext_row[59usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[6usize]) * (current_ext_row[56usize]))),
            (current_ext_row[60usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[55usize])
                        * ((challenges[OpStackIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_169)
                                    + ((challenges[OpStackPointerWeight])
                                        * ((current_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(17179869180u64)))))
                                    + ((challenges[OpStackFirstUnderflowElementWeight])
                                        * (current_base_row[33usize]))))))),
            (current_ext_row[61usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[57usize])
                        * ((challenges[RamIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_997)
                                    + (((next_base_row[22usize])
                                        + (BFieldElement::from_raw_u64(12884901885u64)))
                                        * (challenges[RamPointerWeight])))
                                    + ((next_base_row[25usize])
                                        * (challenges[RamValueWeight]))))))),
            (current_ext_row[62usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[58usize])
                        * ((challenges[RamIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_994)
                                    + (((current_base_row[22usize])
                                        + (BFieldElement::from_raw_u64(8589934590u64)))
                                        * (challenges[RamPointerWeight])))
                                    + ((current_base_row[25usize])
                                        * (challenges[RamValueWeight]))))))),
            (current_ext_row[63usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[61usize])
                        * ((challenges[RamIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_997)
                                    + (((next_base_row[22usize])
                                        + (BFieldElement::from_raw_u64(17179869180u64)))
                                        * (challenges[RamPointerWeight])))
                                    + ((next_base_row[26usize])
                                        * (challenges[RamValueWeight]))))))),
            (current_ext_row[64usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[62usize])
                        * ((challenges[RamIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_994)
                                    + (((current_base_row[22usize])
                                        + (BFieldElement::from_raw_u64(12884901885u64)))
                                        * (challenges[RamPointerWeight])))
                                    + ((current_base_row[26usize])
                                        * (challenges[RamValueWeight]))))))),
            (current_ext_row[65usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[189usize])
                        * ((next_ext_row[7usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((current_ext_row[7usize])
                                    * ((current_ext_row[63usize])
                                        * ((challenges[RamIndeterminate])
                                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                                * (((node_997)
                                                    + (((next_base_row[22usize])
                                                        + (BFieldElement::from_raw_u64(21474836475u64)))
                                                        * (challenges[RamPointerWeight])))
                                                    + ((next_base_row[27usize])
                                                        * (challenges[RamValueWeight]))))))))))),
            (current_ext_row[66usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[189usize])
                        * ((next_ext_row[7usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((current_ext_row[7usize])
                                    * ((current_ext_row[64usize])
                                        * ((challenges[RamIndeterminate])
                                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                                * (((node_994)
                                                    + (((current_base_row[22usize])
                                                        + (BFieldElement::from_raw_u64(17179869180u64)))
                                                        * (challenges[RamPointerWeight])))
                                                    + ((current_base_row[27usize])
                                                        * (challenges[RamValueWeight]))))))))))),
            (current_ext_row[67usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_ext_row[56usize])
                        * ((challenges[OpStackIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_169)
                                    + ((challenges[OpStackPointerWeight])
                                        * ((next_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(21474836475u64)))))
                                    + ((challenges[OpStackFirstUnderflowElementWeight])
                                        * (next_base_row[32usize]))))))
                        * ((challenges[OpStackIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_169)
                                    + ((challenges[OpStackPointerWeight])
                                        * ((next_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(25769803770u64)))))
                                    + ((challenges[OpStackFirstUnderflowElementWeight])
                                        * (next_base_row[31usize]))))))
                        * ((challenges[OpStackIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_169)
                                    + ((challenges[OpStackPointerWeight])
                                        * ((next_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(30064771065u64)))))
                                    + ((challenges[OpStackFirstUnderflowElementWeight])
                                        * (next_base_row[30usize]))))))),
            (current_ext_row[68usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_ext_row[60usize])
                        * ((challenges[OpStackIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_169)
                                    + ((challenges[OpStackPointerWeight])
                                        * ((current_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(21474836475u64)))))
                                    + ((challenges[OpStackFirstUnderflowElementWeight])
                                        * (current_base_row[32usize]))))))
                        * ((challenges[OpStackIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_169)
                                    + ((challenges[OpStackPointerWeight])
                                        * ((current_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(25769803770u64)))))
                                    + ((challenges[OpStackFirstUnderflowElementWeight])
                                        * (current_base_row[31usize]))))))
                        * ((challenges[OpStackIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((node_169)
                                    + ((challenges[OpStackPointerWeight])
                                        * ((current_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(30064771065u64)))))
                                    + ((challenges[OpStackFirstUnderflowElementWeight])
                                        * (current_base_row[30usize]))))))),
            (current_ext_row[69usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[6usize])
                        * (((current_ext_row[67usize])
                            * ((challenges[OpStackIndeterminate])
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (((node_169)
                                        + ((challenges[OpStackPointerWeight])
                                            * ((next_base_row[38usize])
                                                + (BFieldElement::from_raw_u64(34359738360u64)))))
                                        + ((challenges[OpStackFirstUnderflowElementWeight])
                                            * (next_base_row[29usize]))))))
                            * ((challenges[OpStackIndeterminate])
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (((node_169)
                                        + ((challenges[OpStackPointerWeight])
                                            * ((next_base_row[38usize])
                                                + (BFieldElement::from_raw_u64(38654705655u64)))))
                                        + ((challenges[OpStackFirstUnderflowElementWeight])
                                            * (next_base_row[28usize])))))))),
            (current_ext_row[70usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[6usize])
                        * (((current_ext_row[68usize])
                            * ((challenges[OpStackIndeterminate])
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (((node_169)
                                        + ((challenges[OpStackPointerWeight])
                                            * ((current_base_row[38usize])
                                                + (BFieldElement::from_raw_u64(34359738360u64)))))
                                        + ((challenges[OpStackFirstUnderflowElementWeight])
                                            * (current_base_row[29usize]))))))
                            * ((challenges[OpStackIndeterminate])
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (((node_169)
                                        + ((challenges[OpStackPointerWeight])
                                            * ((current_base_row[38usize])
                                                + (BFieldElement::from_raw_u64(38654705655u64)))))
                                        + ((challenges[OpStackFirstUnderflowElementWeight])
                                            * (current_base_row[28usize])))))))),
            (current_ext_row[71usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[6usize]) * (current_ext_row[52usize]))),
            (current_ext_row[72usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[6usize]) * (current_ext_row[60usize]))),
            (current_ext_row[73usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_ext_row[6usize]) * (node_178))),
            (current_ext_row[74usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[186usize])
                        * ((next_ext_row[6usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((current_ext_row[6usize])
                                    * (current_ext_row[50usize])))))),
            (current_ext_row[75usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[188usize])
                        * ((next_ext_row[6usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((current_ext_row[6usize])
                                    * (current_ext_row[54usize])))))),
            (current_ext_row[76usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[185usize]) * (node_572))),
            (current_ext_row[77usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[186usize])
                        * ((next_ext_row[6usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((current_ext_row[6usize])
                                    * (current_ext_row[51usize])))))),
            (current_ext_row[78usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[187usize])
                        * ((next_ext_row[6usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((current_ext_row[6usize])
                                    * (current_ext_row[53usize])))))),
            (current_ext_row[79usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[188usize])
                        * ((next_ext_row[6usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((current_ext_row[6usize])
                                    * (current_ext_row[55usize])))))),
            (current_ext_row[80usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((node_5432)
                        * ((challenges[ClockJumpDifferenceLookupIndeterminate])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((next_base_row[50usize])
                                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                        * (current_base_row[50usize]))))))
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                        * (node_5355))),
            (current_ext_row[81usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[215usize])
                        * ((((((current_base_row[185usize])
                            * ((next_ext_row[7usize])
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * ((current_ext_row[7usize]) * (node_1006)))))
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
                            + ((current_base_row[188usize])
                                * ((next_ext_row[7usize])
                                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                        * ((current_ext_row[7usize])
                                            * (current_ext_row[63usize]))))))
                            + (current_ext_row[65usize])))),
            (current_ext_row[82usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((current_base_row[221usize])
                        * ((((((current_base_row[185usize])
                            * ((next_ext_row[7usize])
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * ((current_ext_row[7usize]) * (node_1088)))))
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
                            + ((current_base_row[188usize])
                                * ((next_ext_row[7usize])
                                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                        * ((current_ext_row[7usize])
                                            * (current_ext_row[64usize]))))))
                            + (current_ext_row[66usize])))),
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
                * (((((((((((challenges[CompressProgramDigestIndeterminate])
                    + ((((((base_row[65usize])
                        * (BFieldElement::from_raw_u64(18446744069414518785u64)))
                        + ((base_row[66usize])
                            * (BFieldElement::from_raw_u64(18446744069414584320u64))))
                        + ((base_row[67usize])
                            * (BFieldElement::from_raw_u64(281474976645120u64))))
                        + (base_row[68usize])) * (BFieldElement::from_raw_u64(1u64))))
                    * (challenges[CompressProgramDigestIndeterminate]))
                    + ((((((base_row[69usize])
                        * (BFieldElement::from_raw_u64(18446744069414518785u64)))
                        + ((base_row[70usize])
                            * (BFieldElement::from_raw_u64(18446744069414584320u64))))
                        + ((base_row[71usize])
                            * (BFieldElement::from_raw_u64(281474976645120u64))))
                        + (base_row[72usize])) * (BFieldElement::from_raw_u64(1u64))))
                    * (challenges[CompressProgramDigestIndeterminate]))
                    + ((((((base_row[73usize])
                        * (BFieldElement::from_raw_u64(18446744069414518785u64)))
                        + ((base_row[74usize])
                            * (BFieldElement::from_raw_u64(18446744069414584320u64))))
                        + ((base_row[75usize])
                            * (BFieldElement::from_raw_u64(281474976645120u64))))
                        + (base_row[76usize])) * (BFieldElement::from_raw_u64(1u64))))
                    * (challenges[CompressProgramDigestIndeterminate]))
                    + ((((((base_row[77usize])
                        * (BFieldElement::from_raw_u64(18446744069414518785u64)))
                        + ((base_row[78usize])
                            * (BFieldElement::from_raw_u64(18446744069414584320u64))))
                        + ((base_row[79usize])
                            * (BFieldElement::from_raw_u64(281474976645120u64))))
                        + (base_row[80usize])) * (BFieldElement::from_raw_u64(1u64))))
                    * (challenges[CompressProgramDigestIndeterminate]))
                    + (base_row[97usize]))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (challenges[CompressedProgramDigest]))),
            (ext_row[47usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (challenges[LookupTablePublicTerminal])),
            (ext_row[2usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[24usize])),
            (challenges[StandardInputTerminal])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (ext_row[3usize])),
            (ext_row[4usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (challenges[StandardOutputTerminal])),
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
    const NUM_TRANSITION_CONSTRAINTS: usize = 398usize;
    const NUM_TERMINAL_CONSTRAINTS: usize = 23usize;
    #[allow(unused_variables)]
    fn initial_quotient_degree_bounds(interpolant_degree: Degree) -> Vec<Degree> {
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
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
        ]
            .to_vec()
    }
    #[allow(unused_variables)]
    fn consistency_quotient_degree_bounds(
        interpolant_degree: Degree,
        padded_height: usize,
    ) -> Vec<Degree> {
        let zerofier_degree = padded_height as Degree;
        [
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
        ]
            .to_vec()
    }
    #[allow(unused_variables)]
    fn transition_quotient_degree_bounds(
        interpolant_degree: Degree,
        padded_height: usize,
    ) -> Vec<Degree> {
        let zerofier_degree = padded_height as Degree - 1;
        [
            interpolant_degree - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
        ]
            .to_vec()
    }
    #[allow(unused_variables)]
    fn terminal_quotient_degree_bounds(interpolant_degree: Degree) -> Vec<Degree> {
        let zerofier_degree = 1;
        [
            interpolant_degree - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree * 3i64 - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree - zerofier_degree,
            interpolant_degree * 2i64 - zerofier_degree,
            interpolant_degree * 4i64 - zerofier_degree,
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
