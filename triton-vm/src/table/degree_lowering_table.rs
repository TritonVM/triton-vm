//! The degree lowering table contains the introduced variables that allow
//! lowering the degree of the AIR. See
//! [`crate::table::master_table::AIR_TARGET_DEGREE`]
//! for additional information.
//!
//! This file has been auto-generated. Any modifications _will_ be lost.
//! To re-generate, execute:
//! `cargo run --bin constraint-evaluation-generator`
use ndarray::s;
use ndarray::ArrayView2;
use ndarray::ArrayViewMut2;
use strum::Display;
use strum::EnumCount;
use strum::EnumIter;
use twenty_first::prelude::BFieldElement;
use twenty_first::prelude::XFieldElement;
use crate::table::challenges::ChallengeId::*;
use crate::table::challenges::Challenges;
use crate::table::master_table::NUM_BASE_COLUMNS;
use crate::table::master_table::NUM_EXT_COLUMNS;
pub const BASE_WIDTH: usize = DegreeLoweringBaseTableColumn::COUNT;
pub const EXT_WIDTH: usize = DegreeLoweringExtTableColumn::COUNT;
pub const FULL_WIDTH: usize = BASE_WIDTH + EXT_WIDTH;
#[repr(usize)]
#[derive(Debug, Display, Copy, Clone, Eq, PartialEq, Hash, EnumCount, EnumIter)]
pub enum DegreeLoweringBaseTableColumn {
    DegreeLoweringBaseCol0,
    DegreeLoweringBaseCol1,
    DegreeLoweringBaseCol2,
    DegreeLoweringBaseCol3,
    DegreeLoweringBaseCol4,
    DegreeLoweringBaseCol5,
    DegreeLoweringBaseCol6,
    DegreeLoweringBaseCol7,
    DegreeLoweringBaseCol8,
    DegreeLoweringBaseCol9,
    DegreeLoweringBaseCol10,
    DegreeLoweringBaseCol11,
    DegreeLoweringBaseCol12,
    DegreeLoweringBaseCol13,
    DegreeLoweringBaseCol14,
    DegreeLoweringBaseCol15,
    DegreeLoweringBaseCol16,
    DegreeLoweringBaseCol17,
    DegreeLoweringBaseCol18,
    DegreeLoweringBaseCol19,
    DegreeLoweringBaseCol20,
    DegreeLoweringBaseCol21,
    DegreeLoweringBaseCol22,
    DegreeLoweringBaseCol23,
    DegreeLoweringBaseCol24,
    DegreeLoweringBaseCol25,
    DegreeLoweringBaseCol26,
    DegreeLoweringBaseCol27,
    DegreeLoweringBaseCol28,
    DegreeLoweringBaseCol29,
    DegreeLoweringBaseCol30,
    DegreeLoweringBaseCol31,
    DegreeLoweringBaseCol32,
    DegreeLoweringBaseCol33,
    DegreeLoweringBaseCol34,
    DegreeLoweringBaseCol35,
    DegreeLoweringBaseCol36,
    DegreeLoweringBaseCol37,
    DegreeLoweringBaseCol38,
    DegreeLoweringBaseCol39,
    DegreeLoweringBaseCol40,
    DegreeLoweringBaseCol41,
    DegreeLoweringBaseCol42,
    DegreeLoweringBaseCol43,
    DegreeLoweringBaseCol44,
    DegreeLoweringBaseCol45,
    DegreeLoweringBaseCol46,
    DegreeLoweringBaseCol47,
    DegreeLoweringBaseCol48,
    DegreeLoweringBaseCol49,
    DegreeLoweringBaseCol50,
    DegreeLoweringBaseCol51,
    DegreeLoweringBaseCol52,
    DegreeLoweringBaseCol53,
    DegreeLoweringBaseCol54,
    DegreeLoweringBaseCol55,
    DegreeLoweringBaseCol56,
    DegreeLoweringBaseCol57,
    DegreeLoweringBaseCol58,
    DegreeLoweringBaseCol59,
    DegreeLoweringBaseCol60,
    DegreeLoweringBaseCol61,
    DegreeLoweringBaseCol62,
    DegreeLoweringBaseCol63,
    DegreeLoweringBaseCol64,
    DegreeLoweringBaseCol65,
    DegreeLoweringBaseCol66,
    DegreeLoweringBaseCol67,
    DegreeLoweringBaseCol68,
    DegreeLoweringBaseCol69,
    DegreeLoweringBaseCol70,
    DegreeLoweringBaseCol71,
    DegreeLoweringBaseCol72,
    DegreeLoweringBaseCol73,
    DegreeLoweringBaseCol74,
    DegreeLoweringBaseCol75,
    DegreeLoweringBaseCol76,
    DegreeLoweringBaseCol77,
    DegreeLoweringBaseCol78,
    DegreeLoweringBaseCol79,
    DegreeLoweringBaseCol80,
    DegreeLoweringBaseCol81,
    DegreeLoweringBaseCol82,
    DegreeLoweringBaseCol83,
    DegreeLoweringBaseCol84,
    DegreeLoweringBaseCol85,
    DegreeLoweringBaseCol86,
    DegreeLoweringBaseCol87,
    DegreeLoweringBaseCol88,
    DegreeLoweringBaseCol89,
    DegreeLoweringBaseCol90,
    DegreeLoweringBaseCol91,
    DegreeLoweringBaseCol92,
    DegreeLoweringBaseCol93,
    DegreeLoweringBaseCol94,
    DegreeLoweringBaseCol95,
    DegreeLoweringBaseCol96,
    DegreeLoweringBaseCol97,
    DegreeLoweringBaseCol98,
    DegreeLoweringBaseCol99,
    DegreeLoweringBaseCol100,
    DegreeLoweringBaseCol101,
    DegreeLoweringBaseCol102,
    DegreeLoweringBaseCol103,
    DegreeLoweringBaseCol104,
    DegreeLoweringBaseCol105,
    DegreeLoweringBaseCol106,
    DegreeLoweringBaseCol107,
    DegreeLoweringBaseCol108,
    DegreeLoweringBaseCol109,
    DegreeLoweringBaseCol110,
    DegreeLoweringBaseCol111,
    DegreeLoweringBaseCol112,
    DegreeLoweringBaseCol113,
    DegreeLoweringBaseCol114,
    DegreeLoweringBaseCol115,
    DegreeLoweringBaseCol116,
    DegreeLoweringBaseCol117,
    DegreeLoweringBaseCol118,
    DegreeLoweringBaseCol119,
    DegreeLoweringBaseCol120,
    DegreeLoweringBaseCol121,
    DegreeLoweringBaseCol122,
    DegreeLoweringBaseCol123,
    DegreeLoweringBaseCol124,
    DegreeLoweringBaseCol125,
    DegreeLoweringBaseCol126,
    DegreeLoweringBaseCol127,
    DegreeLoweringBaseCol128,
    DegreeLoweringBaseCol129,
    DegreeLoweringBaseCol130,
    DegreeLoweringBaseCol131,
    DegreeLoweringBaseCol132,
    DegreeLoweringBaseCol133,
    DegreeLoweringBaseCol134,
    DegreeLoweringBaseCol135,
    DegreeLoweringBaseCol136,
    DegreeLoweringBaseCol137,
    DegreeLoweringBaseCol138,
    DegreeLoweringBaseCol139,
    DegreeLoweringBaseCol140,
    DegreeLoweringBaseCol141,
    DegreeLoweringBaseCol142,
    DegreeLoweringBaseCol143,
    DegreeLoweringBaseCol144,
    DegreeLoweringBaseCol145,
    DegreeLoweringBaseCol146,
    DegreeLoweringBaseCol147,
    DegreeLoweringBaseCol148,
    DegreeLoweringBaseCol149,
    DegreeLoweringBaseCol150,
    DegreeLoweringBaseCol151,
    DegreeLoweringBaseCol152,
    DegreeLoweringBaseCol153,
    DegreeLoweringBaseCol154,
    DegreeLoweringBaseCol155,
    DegreeLoweringBaseCol156,
    DegreeLoweringBaseCol157,
    DegreeLoweringBaseCol158,
    DegreeLoweringBaseCol159,
    DegreeLoweringBaseCol160,
    DegreeLoweringBaseCol161,
    DegreeLoweringBaseCol162,
    DegreeLoweringBaseCol163,
    DegreeLoweringBaseCol164,
    DegreeLoweringBaseCol165,
    DegreeLoweringBaseCol166,
    DegreeLoweringBaseCol167,
    DegreeLoweringBaseCol168,
    DegreeLoweringBaseCol169,
    DegreeLoweringBaseCol170,
    DegreeLoweringBaseCol171,
    DegreeLoweringBaseCol172,
    DegreeLoweringBaseCol173,
    DegreeLoweringBaseCol174,
    DegreeLoweringBaseCol175,
    DegreeLoweringBaseCol176,
    DegreeLoweringBaseCol177,
    DegreeLoweringBaseCol178,
    DegreeLoweringBaseCol179,
    DegreeLoweringBaseCol180,
    DegreeLoweringBaseCol181,
    DegreeLoweringBaseCol182,
    DegreeLoweringBaseCol183,
    DegreeLoweringBaseCol184,
    DegreeLoweringBaseCol185,
    DegreeLoweringBaseCol186,
    DegreeLoweringBaseCol187,
    DegreeLoweringBaseCol188,
    DegreeLoweringBaseCol189,
    DegreeLoweringBaseCol190,
    DegreeLoweringBaseCol191,
    DegreeLoweringBaseCol192,
    DegreeLoweringBaseCol193,
    DegreeLoweringBaseCol194,
    DegreeLoweringBaseCol195,
    DegreeLoweringBaseCol196,
    DegreeLoweringBaseCol197,
    DegreeLoweringBaseCol198,
    DegreeLoweringBaseCol199,
    DegreeLoweringBaseCol200,
    DegreeLoweringBaseCol201,
    DegreeLoweringBaseCol202,
    DegreeLoweringBaseCol203,
    DegreeLoweringBaseCol204,
    DegreeLoweringBaseCol205,
    DegreeLoweringBaseCol206,
}
#[repr(usize)]
#[derive(Debug, Display, Copy, Clone, Eq, PartialEq, Hash, EnumCount, EnumIter)]
pub enum DegreeLoweringExtTableColumn {
    DegreeLoweringExtCol0,
    DegreeLoweringExtCol1,
    DegreeLoweringExtCol2,
    DegreeLoweringExtCol3,
    DegreeLoweringExtCol4,
    DegreeLoweringExtCol5,
    DegreeLoweringExtCol6,
    DegreeLoweringExtCol7,
    DegreeLoweringExtCol8,
    DegreeLoweringExtCol9,
    DegreeLoweringExtCol10,
    DegreeLoweringExtCol11,
    DegreeLoweringExtCol12,
    DegreeLoweringExtCol13,
    DegreeLoweringExtCol14,
    DegreeLoweringExtCol15,
    DegreeLoweringExtCol16,
    DegreeLoweringExtCol17,
    DegreeLoweringExtCol18,
    DegreeLoweringExtCol19,
    DegreeLoweringExtCol20,
    DegreeLoweringExtCol21,
    DegreeLoweringExtCol22,
    DegreeLoweringExtCol23,
    DegreeLoweringExtCol24,
    DegreeLoweringExtCol25,
    DegreeLoweringExtCol26,
    DegreeLoweringExtCol27,
    DegreeLoweringExtCol28,
    DegreeLoweringExtCol29,
    DegreeLoweringExtCol30,
    DegreeLoweringExtCol31,
    DegreeLoweringExtCol32,
}
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct DegreeLoweringTable;
impl DegreeLoweringTable {
    #[allow(unused_variables)]
    pub fn fill_derived_base_columns(
        mut master_base_table: ArrayViewMut2<BFieldElement>,
    ) {
        assert_eq!(NUM_BASE_COLUMNS, master_base_table.ncols());
        master_base_table
            .rows_mut()
            .into_iter()
            .for_each(|mut row| {
                let (base_row, mut det_col) = row
                    .multi_slice_mut((s![..149usize], s![149usize..= 149usize]));
                det_col[0] = ((((base_row[12usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                    * (base_row[13usize]))
                    * ((base_row[14usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                    * ((base_row[15usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                let (base_row, mut det_col) = row
                    .multi_slice_mut((s![..150usize], s![150usize..= 150usize]));
                det_col[0] = (((base_row[149usize]) * (base_row[16usize]))
                    * ((base_row[17usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                    * ((base_row[18usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            });
        master_base_table
            .rows_mut()
            .into_iter()
            .for_each(|mut row| {
                let (base_row, mut det_col) = row
                    .multi_slice_mut((s![..151usize], s![151usize..= 151usize]));
                det_col[0] = (base_row[64usize])
                    * ((base_row[64usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                let (base_row, mut det_col) = row
                    .multi_slice_mut((s![..152usize], s![152usize..= 152usize]));
                det_col[0] = ((((base_row[64usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                    * ((base_row[64usize])
                        + (BFieldElement::from_raw_u64(18446744060824649731u64))))
                    * ((base_row[64usize])
                        + (BFieldElement::from_raw_u64(18446744056529682436u64))))
                    * ((base_row[64usize])
                        + (BFieldElement::from_raw_u64(18446744052234715141u64)));
                let (base_row, mut det_col) = row
                    .multi_slice_mut((s![..153usize], s![153usize..= 153usize]));
                det_col[0] = (((base_row[64usize])
                    * ((base_row[64usize])
                        + (BFieldElement::from_raw_u64(18446744060824649731u64))))
                    * ((base_row[64usize])
                        + (BFieldElement::from_raw_u64(18446744056529682436u64))))
                    * ((base_row[64usize])
                        + (BFieldElement::from_raw_u64(18446744052234715141u64)));
                let (base_row, mut det_col) = row
                    .multi_slice_mut((s![..154usize], s![154usize..= 154usize]));
                det_col[0] = (base_row[151usize])
                    * ((base_row[64usize])
                        + (BFieldElement::from_raw_u64(18446744060824649731u64)));
                let (base_row, mut det_col) = row
                    .multi_slice_mut((s![..155usize], s![155usize..= 155usize]));
                det_col[0] = (((base_row[151usize])
                    * ((base_row[64usize])
                        + (BFieldElement::from_raw_u64(18446744056529682436u64))))
                    * ((base_row[64usize])
                        + (BFieldElement::from_raw_u64(18446744052234715141u64))))
                    * ((base_row[64usize])
                        + (BFieldElement::from_raw_u64(18446744047939747846u64)));
                let (base_row, mut det_col) = row
                    .multi_slice_mut((s![..156usize], s![156usize..= 156usize]));
                det_col[0] = ((base_row[142usize])
                    + (BFieldElement::from_raw_u64(18446744052234715141u64)))
                    * ((base_row[142usize])
                        + (BFieldElement::from_raw_u64(18446744043644780551u64)));
                let (base_row, mut det_col) = row
                    .multi_slice_mut((s![..157usize], s![157usize..= 157usize]));
                det_col[0] = (base_row[143usize]) * (base_row[144usize]);
                let (base_row, mut det_col) = row
                    .multi_slice_mut((s![..158usize], s![158usize..= 158usize]));
                det_col[0] = ((((base_row[142usize])
                    + (BFieldElement::from_raw_u64(18446744052234715141u64)))
                    * ((base_row[142usize])
                        + (BFieldElement::from_raw_u64(18446744009285042191u64))))
                    * ((base_row[142usize])
                        + (BFieldElement::from_raw_u64(18446744017874976781u64))))
                    * ((base_row[142usize])
                        + (BFieldElement::from_raw_u64(18446743940565565471u64)));
                let (base_row, mut det_col) = row
                    .multi_slice_mut((s![..159usize], s![159usize..= 159usize]));
                det_col[0] = (base_row[156usize])
                    * ((base_row[142usize])
                        + (BFieldElement::from_raw_u64(18446744009285042191u64)));
                let (base_row, mut det_col) = row
                    .multi_slice_mut((s![..160usize], s![160usize..= 160usize]));
                det_col[0] = (base_row[145usize]) * (base_row[146usize]);
                let (base_row, mut det_col) = row
                    .multi_slice_mut((s![..161usize], s![161usize..= 161usize]));
                det_col[0] = (((base_row[62usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                    * ((base_row[62usize])
                        + (BFieldElement::from_raw_u64(18446744060824649731u64))))
                    * (base_row[62usize]);
                let (base_row, mut det_col) = row
                    .multi_slice_mut((s![..162usize], s![162usize..= 162usize]));
                det_col[0] = (base_row[158usize])
                    * ((base_row[142usize])
                        + (BFieldElement::from_raw_u64(18446743949155500061u64)));
                let (base_row, mut det_col) = row
                    .multi_slice_mut((s![..163usize], s![163usize..= 163usize]));
                det_col[0] = (((base_row[156usize])
                    * ((base_row[142usize])
                        + (BFieldElement::from_raw_u64(18446744017874976781u64))))
                    * ((base_row[142usize])
                        + (BFieldElement::from_raw_u64(18446743940565565471u64))))
                    * ((base_row[142usize])
                        + (BFieldElement::from_raw_u64(18446743949155500061u64)));
                let (base_row, mut det_col) = row
                    .multi_slice_mut((s![..164usize], s![164usize..= 164usize]));
                det_col[0] = ((base_row[159usize])
                    * ((base_row[142usize])
                        + (BFieldElement::from_raw_u64(18446743940565565471u64))))
                    * ((base_row[142usize])
                        + (BFieldElement::from_raw_u64(18446743949155500061u64)));
                let (base_row, mut det_col) = row
                    .multi_slice_mut((s![..165usize], s![165usize..= 165usize]));
                det_col[0] = ((((base_row[62usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                    * ((base_row[62usize])
                        + (BFieldElement::from_raw_u64(18446744056529682436u64))))
                    * (base_row[62usize]))
                    * ((base_row[63usize])
                        + (BFieldElement::from_raw_u64(18446743897615892521u64)));
                let (base_row, mut det_col) = row
                    .multi_slice_mut((s![..166usize], s![166usize..= 166usize]));
                det_col[0] = (base_row[159usize])
                    * ((base_row[142usize])
                        + (BFieldElement::from_raw_u64(18446744017874976781u64)));
                let (base_row, mut det_col) = row
                    .multi_slice_mut((s![..167usize], s![167usize..= 167usize]));
                det_col[0] = (((base_row[162usize])
                    * ((base_row[139usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                    * ((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (base_row[157usize]))))
                    * ((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (base_row[160usize])));
                let (base_row, mut det_col) = row
                    .multi_slice_mut((s![..168usize], s![168usize..= 168usize]));
                det_col[0] = (((base_row[162usize]) * (base_row[139usize]))
                    * ((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (base_row[157usize]))))
                    * ((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (base_row[160usize])));
            });
        for curr_row_idx in 0..master_base_table.nrows() - 1 {
            let next_row_idx = curr_row_idx + 1;
            let (mut curr_base_row, next_base_row) = master_base_table
                .multi_slice_mut((
                    s![curr_row_idx..= curr_row_idx, ..],
                    s![next_row_idx..= next_row_idx, ..],
                ));
            let mut curr_base_row = curr_base_row.row_mut(0);
            let next_base_row = next_base_row.row(0);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..169usize], s![169usize..= 169usize]));
            det_col[0] = ((BFieldElement::from_raw_u64(4294967295u64))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (current_base_row[42usize])))
                * ((BFieldElement::from_raw_u64(4294967295u64))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[41usize])));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..170usize], s![170usize..= 170usize]));
            det_col[0] = ((current_base_row[12usize])
                + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                * ((current_base_row[13usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..171usize], s![171usize..= 171usize]));
            det_col[0] = ((current_base_row[12usize])
                + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                * (current_base_row[13usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..172usize], s![172usize..= 172usize]));
            det_col[0] = ((current_base_row[12usize])
                * ((current_base_row[13usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                * ((current_base_row[14usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..173usize], s![173usize..= 173usize]));
            det_col[0] = (current_base_row[170usize])
                * ((current_base_row[14usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..174usize], s![174usize..= 174usize]));
            det_col[0] = (current_base_row[171usize])
                * ((current_base_row[14usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..175usize], s![175usize..= 175usize]));
            det_col[0] = (current_base_row[169usize]) * (current_base_row[40usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..176usize], s![176usize..= 176usize]));
            det_col[0] = ((BFieldElement::from_raw_u64(4294967295u64))
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (current_base_row[42usize]))) * (current_base_row[41usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..177usize], s![177usize..= 177usize]));
            det_col[0] = (current_base_row[176usize])
                * ((BFieldElement::from_raw_u64(4294967295u64))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[40usize])));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..178usize], s![178usize..= 178usize]));
            det_col[0] = (current_base_row[172usize])
                * ((current_base_row[15usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..179usize], s![179usize..= 179usize]));
            det_col[0] = (current_base_row[172usize]) * (current_base_row[15usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..180usize], s![180usize..= 180usize]));
            det_col[0] = (current_base_row[173usize])
                * ((current_base_row[15usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..181usize], s![181usize..= 181usize]));
            det_col[0] = (current_base_row[173usize]) * (current_base_row[15usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..182usize], s![182usize..= 182usize]));
            det_col[0] = (current_base_row[169usize])
                * ((BFieldElement::from_raw_u64(4294967295u64))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[40usize])));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..183usize], s![183usize..= 183usize]));
            det_col[0] = (current_base_row[174usize])
                * ((current_base_row[15usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..184usize], s![184usize..= 184usize]));
            det_col[0] = (current_base_row[174usize]) * (current_base_row[15usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..185usize], s![185usize..= 185usize]));
            det_col[0] = (current_base_row[182usize]) * (current_base_row[39usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..186usize], s![186usize..= 186usize]));
            det_col[0] = (current_base_row[175usize])
                * ((BFieldElement::from_raw_u64(4294967295u64))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[39usize])));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..187usize], s![187usize..= 187usize]));
            det_col[0] = (current_base_row[175usize]) * (current_base_row[39usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..188usize], s![188usize..= 188usize]));
            det_col[0] = (current_base_row[177usize])
                * ((BFieldElement::from_raw_u64(4294967295u64))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[39usize])));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..189usize], s![189usize..= 189usize]));
            det_col[0] = (current_base_row[177usize]) * (current_base_row[39usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..190usize], s![190usize..= 190usize]));
            det_col[0] = ((current_base_row[12usize]) * (current_base_row[13usize]))
                * ((current_base_row[14usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..191usize], s![191usize..= 191usize]));
            det_col[0] = (current_base_row[179usize])
                * ((current_base_row[16usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..192usize], s![192usize..= 192usize]));
            det_col[0] = (current_base_row[178usize])
                * ((current_base_row[16usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..193usize], s![193usize..= 193usize]));
            det_col[0] = (current_base_row[180usize])
                * ((current_base_row[16usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..194usize], s![194usize..= 194usize]));
            det_col[0] = (current_base_row[183usize])
                * ((current_base_row[16usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..195usize], s![195usize..= 195usize]));
            det_col[0] = (current_base_row[181usize])
                * ((current_base_row[16usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..196usize], s![196usize..= 196usize]));
            det_col[0] = (current_base_row[184usize])
                * ((current_base_row[16usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..197usize], s![197usize..= 197usize]));
            det_col[0] = (current_base_row[171usize]) * (current_base_row[14usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..198usize], s![198usize..= 198usize]));
            det_col[0] = (current_base_row[190usize])
                * ((current_base_row[15usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..199usize], s![199usize..= 199usize]));
            det_col[0] = (current_base_row[170usize]) * (current_base_row[14usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..200usize], s![200usize..= 200usize]));
            det_col[0] = (current_base_row[178usize]) * (current_base_row[16usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..201usize], s![201usize..= 201usize]));
            det_col[0] = (current_base_row[181usize]) * (current_base_row[16usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..202usize], s![202usize..= 202usize]));
            det_col[0] = (((current_base_row[198usize])
                * ((current_base_row[16usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                * ((current_base_row[17usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                * ((current_base_row[18usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..203usize], s![203usize..= 203usize]));
            det_col[0] = ((current_base_row[191usize])
                * ((current_base_row[17usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                * ((current_base_row[18usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..204usize], s![204usize..= 204usize]));
            det_col[0] = (current_base_row[180usize]) * (current_base_row[16usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..205usize], s![205usize..= 205usize]));
            det_col[0] = (((current_base_row[179usize]) * (current_base_row[16usize]))
                * ((current_base_row[17usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                * ((current_base_row[18usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..206usize], s![206usize..= 206usize]));
            det_col[0] = (current_base_row[183usize]) * (current_base_row[16usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..207usize], s![207usize..= 207usize]));
            det_col[0] = (current_base_row[184usize]) * (current_base_row[16usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..208usize], s![208usize..= 208usize]));
            det_col[0] = ((current_base_row[200usize])
                * ((current_base_row[17usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                * ((current_base_row[18usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..209usize], s![209usize..= 209usize]));
            det_col[0] = (current_base_row[194usize])
                * ((current_base_row[17usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..210usize], s![210usize..= 210usize]));
            det_col[0] = (current_base_row[193usize])
                * ((current_base_row[17usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..211usize], s![211usize..= 211usize]));
            det_col[0] = (current_base_row[196usize])
                * ((current_base_row[17usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..212usize], s![212usize..= 212usize]));
            det_col[0] = ((current_base_row[192usize])
                * ((current_base_row[17usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                * ((current_base_row[18usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..213usize], s![213usize..= 213usize]));
            det_col[0] = (current_base_row[209usize])
                * ((current_base_row[18usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..214usize], s![214usize..= 214usize]));
            det_col[0] = (current_base_row[210usize])
                * ((current_base_row[18usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..215usize], s![215usize..= 215usize]));
            det_col[0] = ((current_base_row[191usize]) * (current_base_row[17usize]))
                * ((current_base_row[18usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..216usize], s![216usize..= 216usize]));
            det_col[0] = ((current_base_row[195usize])
                * ((current_base_row[17usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                * ((current_base_row[18usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..217usize], s![217usize..= 217usize]));
            det_col[0] = (((current_base_row[190usize]) * (current_base_row[15usize]))
                * ((current_base_row[16usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                * ((current_base_row[17usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..218usize], s![218usize..= 218usize]));
            det_col[0] = (current_base_row[197usize])
                * ((current_base_row[15usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..219usize], s![219usize..= 219usize]));
            det_col[0] = (current_base_row[199usize])
                * ((current_base_row[15usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..220usize], s![220usize..= 220usize]));
            det_col[0] = ((current_base_row[192usize]) * (current_base_row[17usize]))
                * ((current_base_row[18usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..221usize], s![221usize..= 221usize]));
            det_col[0] = (current_base_row[217usize])
                * ((current_base_row[18usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..222usize], s![222usize..= 222usize]));
            det_col[0] = (current_base_row[197usize]) * (current_base_row[15usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..223usize], s![223usize..= 223usize]));
            det_col[0] = ((current_base_row[201usize])
                * ((current_base_row[17usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                * ((current_base_row[18usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..224usize], s![224usize..= 224usize]));
            det_col[0] = (current_base_row[211usize])
                * ((current_base_row[18usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..225usize], s![225usize..= 225usize]));
            det_col[0] = ((current_base_row[204usize])
                * ((current_base_row[17usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                * ((current_base_row[18usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..226usize], s![226usize..= 226usize]));
            det_col[0] = (current_base_row[199usize]) * (current_base_row[15usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..227usize], s![227usize..= 227usize]));
            det_col[0] = (current_base_row[42usize])
                * ((BFieldElement::from_raw_u64(4294967295u64))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[41usize])));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..228usize], s![228usize..= 228usize]));
            det_col[0] = ((current_base_row[207usize])
                * ((current_base_row[17usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                * ((current_base_row[18usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..229usize], s![229usize..= 229usize]));
            det_col[0] = ((current_base_row[195usize]) * (current_base_row[17usize]))
                * ((current_base_row[18usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..230usize], s![230usize..= 230usize]));
            det_col[0] = (current_base_row[206usize])
                * ((current_base_row[17usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..231usize], s![231usize..= 231usize]));
            det_col[0] = (current_base_row[42usize]) * (current_base_row[41usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..232usize], s![232usize..= 232usize]));
            det_col[0] = ((current_base_row[193usize]) * (current_base_row[17usize]))
                * ((current_base_row[18usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..233usize], s![233usize..= 233usize]));
            det_col[0] = ((current_base_row[196usize]) * (current_base_row[17usize]))
                * ((current_base_row[18usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..234usize], s![234usize..= 234usize]));
            det_col[0] = ((current_base_row[206usize]) * (current_base_row[17usize]))
                * ((current_base_row[18usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..235usize], s![235usize..= 235usize]));
            det_col[0] = ((current_base_row[201usize]) * (current_base_row[17usize]))
                * ((current_base_row[18usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..236usize], s![236usize..= 236usize]));
            det_col[0] = (((current_base_row[219usize])
                * ((current_base_row[16usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                * ((current_base_row[17usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                * ((current_base_row[18usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..237usize], s![237usize..= 237usize]));
            det_col[0] = ((current_base_row[207usize]) * (current_base_row[17usize]))
                * ((current_base_row[18usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..238usize], s![238usize..= 238usize]));
            det_col[0] = (((current_base_row[218usize])
                * ((current_base_row[16usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                * ((current_base_row[17usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                * ((current_base_row[18usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..239usize], s![239usize..= 239usize]));
            det_col[0] = (((current_base_row[222usize])
                * ((current_base_row[16usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                * ((current_base_row[17usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                * ((current_base_row[18usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..240usize], s![240usize..= 240usize]));
            det_col[0] = (current_base_row[230usize])
                * ((current_base_row[18usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..241usize], s![241usize..= 241usize]));
            det_col[0] = (((current_base_row[218usize]) * (current_base_row[16usize]))
                * ((current_base_row[17usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                * ((current_base_row[18usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..242usize], s![242usize..= 242usize]));
            det_col[0] = (((current_base_row[226usize])
                * ((current_base_row[16usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                * ((current_base_row[17usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                * ((current_base_row[18usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..243usize], s![243usize..= 243usize]));
            det_col[0] = (current_base_row[176usize]) * (current_base_row[40usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..244usize], s![244usize..= 244usize]));
            det_col[0] = ((current_base_row[194usize]) * (current_base_row[17usize]))
                * ((current_base_row[18usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..245usize], s![245usize..= 245usize]));
            det_col[0] = (((current_base_row[222usize]) * (current_base_row[16usize]))
                * ((current_base_row[17usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                * ((current_base_row[18usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..246usize], s![246usize..= 246usize]));
            det_col[0] = ((current_base_row[204usize]) * (current_base_row[17usize]))
                * ((current_base_row[18usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..247usize], s![247usize..= 247usize]));
            det_col[0] = (current_base_row[227usize])
                * ((BFieldElement::from_raw_u64(4294967295u64))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[40usize])));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..248usize], s![248usize..= 248usize]));
            det_col[0] = (((current_base_row[219usize]) * (current_base_row[16usize]))
                * ((current_base_row[17usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                * ((current_base_row[18usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..249usize], s![249usize..= 249usize]));
            det_col[0] = (((current_base_row[97usize]) * (current_base_row[97usize]))
                * (current_base_row[97usize])) * (current_base_row[97usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..250usize], s![250usize..= 250usize]));
            det_col[0] = (((current_base_row[98usize]) * (current_base_row[98usize]))
                * (current_base_row[98usize])) * (current_base_row[98usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..251usize], s![251usize..= 251usize]));
            det_col[0] = (((current_base_row[226usize]) * (current_base_row[16usize]))
                * ((current_base_row[17usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                * ((current_base_row[18usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..252usize], s![252usize..= 252usize]));
            det_col[0] = (((current_base_row[99usize]) * (current_base_row[99usize]))
                * (current_base_row[99usize])) * (current_base_row[99usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..253usize], s![253usize..= 253usize]));
            det_col[0] = (current_base_row[227usize]) * (current_base_row[40usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..254usize], s![254usize..= 254usize]));
            det_col[0] = (((current_base_row[100usize]) * (current_base_row[100usize]))
                * (current_base_row[100usize])) * (current_base_row[100usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..255usize], s![255usize..= 255usize]));
            det_col[0] = (current_base_row[231usize])
                * ((BFieldElement::from_raw_u64(4294967295u64))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[40usize])));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..256usize], s![256usize..= 256usize]));
            det_col[0] = (current_base_row[231usize]) * (current_base_row[40usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..257usize], s![257usize..= 257usize]));
            det_col[0] = (((current_base_row[101usize]) * (current_base_row[101usize]))
                * (current_base_row[101usize])) * (current_base_row[101usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..258usize], s![258usize..= 258usize]));
            det_col[0] = (((current_base_row[102usize]) * (current_base_row[102usize]))
                * (current_base_row[102usize])) * (current_base_row[102usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..259usize], s![259usize..= 259usize]));
            det_col[0] = ((current_base_row[200usize]) * (current_base_row[17usize]))
                * ((current_base_row[18usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..260usize], s![260usize..= 260usize]));
            det_col[0] = (current_base_row[41usize])
                * ((current_base_row[41usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..261usize], s![261usize..= 261usize]));
            det_col[0] = (current_base_row[42usize])
                * ((current_base_row[42usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..262usize], s![262usize..= 262usize]));
            det_col[0] = (((current_base_row[103usize]) * (current_base_row[103usize]))
                * (current_base_row[103usize])) * (current_base_row[103usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..263usize], s![263usize..= 263usize]));
            det_col[0] = (((current_base_row[198usize]) * (current_base_row[16usize]))
                * ((current_base_row[17usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                * ((current_base_row[18usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..264usize], s![264usize..= 264usize]));
            det_col[0] = (((current_base_row[104usize]) * (current_base_row[104usize]))
                * (current_base_row[104usize])) * (current_base_row[104usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..265usize], s![265usize..= 265usize]));
            det_col[0] = (((current_base_row[105usize]) * (current_base_row[105usize]))
                * (current_base_row[105usize])) * (current_base_row[105usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..266usize], s![266usize..= 266usize]));
            det_col[0] = (((current_base_row[106usize]) * (current_base_row[106usize]))
                * (current_base_row[106usize])) * (current_base_row[106usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..267usize], s![267usize..= 267usize]));
            det_col[0] = (((current_base_row[107usize]) * (current_base_row[107usize]))
                * (current_base_row[107usize])) * (current_base_row[107usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..268usize], s![268usize..= 268usize]));
            det_col[0] = (current_base_row[40usize])
                * ((current_base_row[40usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..269usize], s![269usize..= 269usize]));
            det_col[0] = (((current_base_row[108usize]) * (current_base_row[108usize]))
                * (current_base_row[108usize])) * (current_base_row[108usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..270usize], s![270usize..= 270usize]));
            det_col[0] = (current_base_row[243usize]) * (current_base_row[39usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..271usize], s![271usize..= 271usize]));
            det_col[0] = (current_base_row[247usize])
                * ((BFieldElement::from_raw_u64(4294967295u64))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[39usize])));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..272usize], s![272usize..= 272usize]));
            det_col[0] = (current_base_row[39usize]) * (current_base_row[22usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..273usize], s![273usize..= 273usize]));
            det_col[0] = (current_base_row[255usize])
                * ((BFieldElement::from_raw_u64(4294967295u64))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[39usize])));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..274usize], s![274usize..= 274usize]));
            det_col[0] = (current_base_row[209usize]) * (current_base_row[18usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..275usize], s![275usize..= 275usize]));
            det_col[0] = (current_base_row[211usize]) * (current_base_row[18usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..276usize], s![276usize..= 276usize]));
            det_col[0] = (((next_base_row[64usize])
                * ((next_base_row[64usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                * ((next_base_row[64usize])
                    + (BFieldElement::from_raw_u64(18446744060824649731u64))))
                * ((next_base_row[64usize])
                    + (BFieldElement::from_raw_u64(18446744056529682436u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..277usize], s![277usize..= 277usize]));
            det_col[0] = (current_base_row[210usize]) * (current_base_row[18usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..278usize], s![278usize..= 278usize]));
            det_col[0] = ((next_base_row[62usize])
                * ((next_base_row[64usize])
                    + (BFieldElement::from_raw_u64(18446744047939747846u64))))
                * ((next_base_row[63usize])
                    + (BFieldElement::from_raw_u64(18446743897615892521u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..279usize], s![279usize..= 279usize]));
            det_col[0] = (current_base_row[230usize]) * (current_base_row[18usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..280usize], s![280usize..= 280usize]));
            det_col[0] = (((current_base_row[43usize])
                * ((current_base_row[43usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                * ((current_base_row[43usize])
                    + (BFieldElement::from_raw_u64(18446744060824649731u64))))
                * ((current_base_row[43usize])
                    + (BFieldElement::from_raw_u64(18446744056529682436u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..281usize], s![281usize..= 281usize]));
            det_col[0] = (((current_base_row[44usize])
                * ((current_base_row[44usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                * ((current_base_row[44usize])
                    + (BFieldElement::from_raw_u64(18446744060824649731u64))))
                * ((current_base_row[44usize])
                    + (BFieldElement::from_raw_u64(18446744056529682436u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..282usize], s![282usize..= 282usize]));
            det_col[0] = (current_base_row[185usize])
                * ((next_base_row[24usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[23usize])));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..283usize], s![283usize..= 283usize]));
            det_col[0] = (current_base_row[186usize])
                * ((next_base_row[25usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[23usize])));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..284usize], s![284usize..= 284usize]));
            det_col[0] = (current_base_row[187usize])
                * ((next_base_row[26usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[23usize])));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..285usize], s![285usize..= 285usize]));
            det_col[0] = (current_base_row[188usize])
                * ((next_base_row[27usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[23usize])));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..286usize], s![286usize..= 286usize]));
            det_col[0] = (current_base_row[189usize])
                * ((next_base_row[28usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[23usize])));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..287usize], s![287usize..= 287usize]));
            det_col[0] = (current_base_row[185usize])
                * ((next_base_row[23usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[24usize])));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..288usize], s![288usize..= 288usize]));
            det_col[0] = (current_base_row[186usize])
                * ((next_base_row[23usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[25usize])));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..289usize], s![289usize..= 289usize]));
            det_col[0] = (current_base_row[187usize])
                * ((next_base_row[23usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[26usize])));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..290usize], s![290usize..= 290usize]));
            det_col[0] = (current_base_row[188usize])
                * ((next_base_row[23usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[27usize])));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..291usize], s![291usize..= 291usize]));
            det_col[0] = (current_base_row[189usize])
                * ((next_base_row[23usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[28usize])));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..292usize], s![292usize..= 292usize]));
            det_col[0] = (current_base_row[189usize])
                * ((next_base_row[22usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[27usize])));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..293usize], s![293usize..= 293usize]));
            det_col[0] = (current_base_row[189usize])
                * ((next_base_row[27usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[22usize])));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..294usize], s![294usize..= 294usize]));
            det_col[0] = ((((next_base_row[142usize])
                + (BFieldElement::from_raw_u64(18446744052234715141u64)))
                * ((next_base_row[142usize])
                    + (BFieldElement::from_raw_u64(18446744009285042191u64))))
                * ((next_base_row[142usize])
                    + (BFieldElement::from_raw_u64(18446744017874976781u64))))
                * ((next_base_row[142usize])
                    + (BFieldElement::from_raw_u64(18446743940565565471u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..295usize], s![295usize..= 295usize]));
            det_col[0] = ((next_base_row[142usize])
                + (BFieldElement::from_raw_u64(18446744052234715141u64)))
                * ((next_base_row[142usize])
                    + (BFieldElement::from_raw_u64(18446744043644780551u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..296usize], s![296usize..= 296usize]));
            det_col[0] = ((((next_base_row[64usize])
                + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                * ((next_base_row[64usize])
                    + (BFieldElement::from_raw_u64(18446744060824649731u64))))
                * ((next_base_row[64usize])
                    + (BFieldElement::from_raw_u64(18446744056529682436u64))))
                * ((next_base_row[64usize])
                    + (BFieldElement::from_raw_u64(18446744052234715141u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..297usize], s![297usize..= 297usize]));
            det_col[0] = (((current_base_row[249usize]) * (current_base_row[97usize]))
                * (current_base_row[97usize])) * (current_base_row[97usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..298usize], s![298usize..= 298usize]));
            det_col[0] = (((current_base_row[250usize]) * (current_base_row[98usize]))
                * (current_base_row[98usize])) * (current_base_row[98usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..299usize], s![299usize..= 299usize]));
            det_col[0] = (((current_base_row[252usize]) * (current_base_row[99usize]))
                * (current_base_row[99usize])) * (current_base_row[99usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..300usize], s![300usize..= 300usize]));
            det_col[0] = (((current_base_row[254usize]) * (current_base_row[100usize]))
                * (current_base_row[100usize])) * (current_base_row[100usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..301usize], s![301usize..= 301usize]));
            det_col[0] = (((current_base_row[257usize]) * (current_base_row[101usize]))
                * (current_base_row[101usize])) * (current_base_row[101usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..302usize], s![302usize..= 302usize]));
            det_col[0] = (((current_base_row[258usize]) * (current_base_row[102usize]))
                * (current_base_row[102usize])) * (current_base_row[102usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..303usize], s![303usize..= 303usize]));
            det_col[0] = (((current_base_row[262usize]) * (current_base_row[103usize]))
                * (current_base_row[103usize])) * (current_base_row[103usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..304usize], s![304usize..= 304usize]));
            det_col[0] = (((current_base_row[264usize]) * (current_base_row[104usize]))
                * (current_base_row[104usize])) * (current_base_row[104usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..305usize], s![305usize..= 305usize]));
            det_col[0] = (((current_base_row[265usize]) * (current_base_row[105usize]))
                * (current_base_row[105usize])) * (current_base_row[105usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..306usize], s![306usize..= 306usize]));
            det_col[0] = (((current_base_row[266usize]) * (current_base_row[106usize]))
                * (current_base_row[106usize])) * (current_base_row[106usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..307usize], s![307usize..= 307usize]));
            det_col[0] = (((current_base_row[267usize]) * (current_base_row[107usize]))
                * (current_base_row[107usize])) * (current_base_row[107usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..308usize], s![308usize..= 308usize]));
            det_col[0] = (((current_base_row[269usize]) * (current_base_row[108usize]))
                * (current_base_row[108usize])) * (current_base_row[108usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..309usize], s![309usize..= 309usize]));
            det_col[0] = ((next_base_row[139usize])
                + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                * ((current_base_row[294usize])
                    * ((next_base_row[142usize])
                        + (BFieldElement::from_raw_u64(18446743949155500061u64))));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..310usize], s![310usize..= 310usize]));
            det_col[0] = (current_base_row[295usize])
                * ((next_base_row[142usize])
                    + (BFieldElement::from_raw_u64(18446744009285042191u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..311usize], s![311usize..= 311usize]));
            det_col[0] = (current_base_row[296usize])
                * ((next_base_row[64usize])
                    + (BFieldElement::from_raw_u64(18446744047939747846u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..312usize], s![312usize..= 312usize]));
            det_col[0] = (current_base_row[243usize])
                * ((BFieldElement::from_raw_u64(4294967295u64))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[39usize])));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..313usize], s![313usize..= 313usize]));
            det_col[0] = ((current_base_row[309usize]) * (next_base_row[147usize]))
                * ((next_base_row[147usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..314usize], s![314usize..= 314usize]));
            det_col[0] = (current_base_row[247usize]) * (current_base_row[39usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..315usize], s![315usize..= 315usize]));
            det_col[0] = (current_base_row[182usize])
                * ((BFieldElement::from_raw_u64(4294967295u64))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[39usize])));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..316usize], s![316usize..= 316usize]));
            det_col[0] = (((next_base_row[62usize])
                + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                * ((next_base_row[62usize])
                    + (BFieldElement::from_raw_u64(18446744060824649731u64))))
                * (next_base_row[62usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..317usize], s![317usize..= 317usize]));
            det_col[0] = (current_base_row[253usize])
                * ((BFieldElement::from_raw_u64(4294967295u64))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[39usize])));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..318usize], s![318usize..= 318usize]));
            det_col[0] = (current_base_row[253usize]) * (current_base_row[39usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..319usize], s![319usize..= 319usize]));
            det_col[0] = (current_base_row[185usize])
                * ((next_base_row[24usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[25usize])));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..320usize], s![320usize..= 320usize]));
            det_col[0] = (current_base_row[185usize])
                * ((next_base_row[25usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[26usize])));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..321usize], s![321usize..= 321usize]));
            det_col[0] = (current_base_row[186usize])
                * ((next_base_row[24usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[26usize])));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..322usize], s![322usize..= 322usize]));
            det_col[0] = (current_base_row[186usize])
                * ((next_base_row[25usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[27usize])));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..323usize], s![323usize..= 323usize]));
            det_col[0] = (current_base_row[187usize])
                * ((next_base_row[24usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[27usize])));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..324usize], s![324usize..= 324usize]));
            det_col[0] = (current_base_row[187usize])
                * ((next_base_row[25usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[28usize])));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..325usize], s![325usize..= 325usize]));
            det_col[0] = (current_base_row[188usize])
                * ((next_base_row[24usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[28usize])));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..326usize], s![326usize..= 326usize]));
            det_col[0] = (current_base_row[188usize])
                * ((next_base_row[25usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[29usize])));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..327usize], s![327usize..= 327usize]));
            det_col[0] = (current_base_row[189usize])
                * ((next_base_row[24usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[29usize])));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..328usize], s![328usize..= 328usize]));
            det_col[0] = (current_base_row[189usize])
                * ((next_base_row[25usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[30usize])));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..329usize], s![329usize..= 329usize]));
            det_col[0] = (current_base_row[255usize]) * (current_base_row[39usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..330usize], s![330usize..= 330usize]));
            det_col[0] = (current_base_row[256usize])
                * ((BFieldElement::from_raw_u64(4294967295u64))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[39usize])));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..331usize], s![331usize..= 331usize]));
            det_col[0] = (current_base_row[256usize]) * (current_base_row[39usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..332usize], s![332usize..= 332usize]));
            det_col[0] = (current_base_row[310usize])
                * ((next_base_row[142usize])
                    + (BFieldElement::from_raw_u64(18446744017874976781u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..333usize], s![333usize..= 333usize]));
            det_col[0] = ((((next_base_row[12usize])
                + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                * (next_base_row[13usize]))
                * ((next_base_row[14usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                * ((next_base_row[15usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..334usize], s![334usize..= 334usize]));
            det_col[0] = ((next_base_row[139usize])
                + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                * (((current_base_row[310usize])
                    * ((next_base_row[142usize])
                        + (BFieldElement::from_raw_u64(18446743940565565471u64))))
                    * ((next_base_row[142usize])
                        + (BFieldElement::from_raw_u64(18446743949155500061u64))));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..335usize], s![335usize..= 335usize]));
            det_col[0] = (current_base_row[22usize]) * (current_base_row[23usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..336usize], s![336usize..= 336usize]));
            det_col[0] = (next_base_row[22usize]) * (current_base_row[22usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..337usize], s![337usize..= 337usize]));
            det_col[0] = (current_base_row[39usize])
                * ((current_base_row[23usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[22usize])));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..338usize], s![338usize..= 338usize]));
            det_col[0] = (current_base_row[311usize])
                * ((((next_base_row[62usize])
                    + (BFieldElement::from_raw_u64(18446744060824649731u64)))
                    * ((next_base_row[62usize])
                        + (BFieldElement::from_raw_u64(18446744056529682436u64))))
                    * (next_base_row[62usize]));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..339usize], s![339usize..= 339usize]));
            det_col[0] = (((current_base_row[62usize])
                + (BFieldElement::from_raw_u64(18446744060824649731u64)))
                * ((current_base_row[62usize])
                    + (BFieldElement::from_raw_u64(18446744056529682436u64))))
                * (current_base_row[62usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..340usize], s![340usize..= 340usize]));
            det_col[0] = (current_base_row[236usize])
                * ((next_base_row[22usize])
                    * (((current_base_row[39usize])
                        * ((next_base_row[23usize])
                            + (BFieldElement::from_raw_u64(4294967296u64))))
                        + (BFieldElement::from_raw_u64(18446744065119617026u64))));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..341usize], s![341usize..= 341usize]));
            det_col[0] = (current_base_row[213usize])
                * ((((((next_base_row[9usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[9usize])))
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                    * (current_base_row[22usize]))
                    + (((((next_base_row[9usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[9usize])))
                        + (BFieldElement::from_raw_u64(18446744060824649731u64)))
                        * ((current_base_row[272usize])
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                        * ((current_base_row[40usize])
                            + (BFieldElement::from_raw_u64(18446744065119617026u64)))))
                    + (((((next_base_row[9usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[9usize])))
                        + (BFieldElement::from_raw_u64(18446744056529682436u64)))
                        * ((current_base_row[272usize])
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                        * (current_base_row[40usize])));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..342usize], s![342usize..= 342usize]));
            det_col[0] = (current_base_row[213usize])
                * (((current_base_row[260usize])
                    * ((current_base_row[41usize])
                        + (BFieldElement::from_raw_u64(18446744060824649731u64))))
                    * ((current_base_row[41usize])
                        + (BFieldElement::from_raw_u64(18446744056529682436u64))));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..343usize], s![343usize..= 343usize]));
            det_col[0] = (current_base_row[213usize])
                * (((current_base_row[261usize])
                    * ((current_base_row[42usize])
                        + (BFieldElement::from_raw_u64(18446744060824649731u64))))
                    * ((current_base_row[42usize])
                        + (BFieldElement::from_raw_u64(18446744056529682436u64))));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..344usize], s![344usize..= 344usize]));
            det_col[0] = (((current_base_row[333usize]) * (next_base_row[16usize]))
                * ((next_base_row[17usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                * ((next_base_row[18usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..345usize], s![345usize..= 345usize]));
            det_col[0] = (((current_base_row[64usize])
                * ((current_base_row[64usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                * ((current_base_row[64usize])
                    + (BFieldElement::from_raw_u64(18446744060824649731u64))))
                * ((current_base_row[64usize])
                    + (BFieldElement::from_raw_u64(18446744056529682436u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..346usize], s![346usize..= 346usize]));
            det_col[0] = ((((current_base_row[62usize])
                + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                * ((current_base_row[62usize])
                    + (BFieldElement::from_raw_u64(18446744056529682436u64))))
                * (current_base_row[62usize]))
                * ((next_base_row[62usize])
                    + (BFieldElement::from_raw_u64(18446744060824649731u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..347usize], s![347usize..= 347usize]));
            det_col[0] = (((current_base_row[295usize])
                * ((next_base_row[142usize])
                    + (BFieldElement::from_raw_u64(18446744017874976781u64))))
                * ((next_base_row[142usize])
                    + (BFieldElement::from_raw_u64(18446743940565565471u64))))
                * ((next_base_row[142usize])
                    + (BFieldElement::from_raw_u64(18446743949155500061u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..348usize], s![348usize..= 348usize]));
            det_col[0] = (current_base_row[313usize])
                * ((((BFieldElement::from_raw_u64(4294967295u64))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((current_base_row[143usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((BFieldElement::from_raw_u64(8589934590u64))
                                    * (next_base_row[143usize]))))))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((current_base_row[145usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((BFieldElement::from_raw_u64(8589934590u64))
                                    * (next_base_row[145usize]))))))
                    + (((BFieldElement::from_raw_u64(8589934590u64))
                        * ((current_base_row[143usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((BFieldElement::from_raw_u64(8589934590u64))
                                    * (next_base_row[143usize])))))
                        * ((current_base_row[145usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((BFieldElement::from_raw_u64(8589934590u64))
                                    * (next_base_row[145usize]))))));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..349usize], s![349usize..= 349usize]));
            det_col[0] = ((next_base_row[139usize])
                + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                * ((current_base_row[332usize])
                    * ((next_base_row[142usize])
                        + (BFieldElement::from_raw_u64(18446743949155500061u64))));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..350usize], s![350usize..= 350usize]));
            det_col[0] = (current_base_row[339usize])
                * ((((next_base_row[62usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                    * ((next_base_row[62usize])
                        + (BFieldElement::from_raw_u64(18446744056529682436u64))))
                    * (next_base_row[62usize]));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..351usize], s![351usize..= 351usize]));
            det_col[0] = ((((current_base_row[62usize])
                + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                * ((current_base_row[62usize])
                    + (BFieldElement::from_raw_u64(18446744060824649731u64))))
                * (current_base_row[62usize]))
                * ((next_base_row[62usize])
                    + (BFieldElement::from_raw_u64(18446744056529682436u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..352usize], s![352usize..= 352usize]));
            det_col[0] = (((current_base_row[311usize])
                * ((next_base_row[62usize])
                    + (BFieldElement::from_raw_u64(18446744056529682436u64))))
                * (next_base_row[62usize]))
                * ((next_base_row[63usize])
                    + (BFieldElement::from_raw_u64(18446743897615892521u64)));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..353usize], s![353usize..= 353usize]));
            det_col[0] = (current_base_row[311usize])
                * ((((next_base_row[63usize])
                    + (BFieldElement::from_raw_u64(18446743992105173011u64)))
                    * ((next_base_row[63usize])
                        + (BFieldElement::from_raw_u64(18446743897615892521u64))))
                    * ((next_base_row[63usize])
                        + (BFieldElement::from_raw_u64(18446743923385696291u64))));
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..354usize], s![354usize..= 354usize]));
            det_col[0] = ((current_base_row[334usize])
                * ((BFieldElement::from_raw_u64(4294967295u64))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((next_base_row[143usize]) * (next_base_row[144usize])))))
                * (current_base_row[143usize]);
            let (current_base_row, mut det_col) = curr_base_row
                .multi_slice_mut((s![..355usize], s![355usize..= 355usize]));
            det_col[0] = ((next_base_row[147usize]) * (next_base_row[147usize]))
                * (current_base_row[143usize]);
        }
    }
    #[allow(unused_variables)]
    #[allow(unused_mut)]
    pub fn fill_derived_ext_columns(
        master_base_table: ArrayView2<BFieldElement>,
        mut master_ext_table: ArrayViewMut2<XFieldElement>,
        challenges: &Challenges,
    ) {
        assert_eq!(NUM_BASE_COLUMNS, master_base_table.ncols());
        assert_eq!(NUM_EXT_COLUMNS, master_ext_table.ncols());
        assert_eq!(master_base_table.nrows(), master_ext_table.nrows());
        for curr_row_idx in 0..master_base_table.nrows() - 1 {
            let next_row_idx = curr_row_idx + 1;
            let current_base_row = master_base_table.row(curr_row_idx);
            let next_base_row = master_base_table.row(next_row_idx);
            let (mut curr_ext_row, next_ext_row) = master_ext_table
                .multi_slice_mut((
                    s![curr_row_idx..= curr_row_idx, ..],
                    s![next_row_idx..= next_row_idx, ..],
                ));
            let mut curr_ext_row = curr_ext_row.row_mut(0);
            let next_ext_row = next_ext_row.row(0);
            let (current_ext_row, mut det_col) = curr_ext_row
                .multi_slice_mut((s![..50usize], s![50usize..= 50usize]));
            det_col[0] = ((challenges[OpStackIndeterminate])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((((challenges[OpStackClkWeight]) * (current_base_row[7usize]))
                        + ((challenges[OpStackIb1Weight]) * (current_base_row[13usize])))
                        + ((challenges[OpStackPointerWeight])
                            * (next_base_row[38usize])))
                        + ((challenges[OpStackFirstUnderflowElementWeight])
                            * (next_base_row[37usize])))))
                * ((challenges[OpStackIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((((challenges[OpStackClkWeight])
                            * (current_base_row[7usize]))
                            + ((challenges[OpStackIb1Weight])
                                * (current_base_row[13usize])))
                            + ((challenges[OpStackPointerWeight])
                                * ((next_base_row[38usize])
                                    + (BFieldElement::from_raw_u64(4294967295u64)))))
                            + ((challenges[OpStackFirstUnderflowElementWeight])
                                * (next_base_row[36usize])))));
            let (current_ext_row, mut det_col) = curr_ext_row
                .multi_slice_mut((s![..51usize], s![51usize..= 51usize]));
            det_col[0] = ((challenges[OpStackIndeterminate])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((((challenges[OpStackClkWeight]) * (current_base_row[7usize]))
                        + ((challenges[OpStackIb1Weight]) * (current_base_row[13usize])))
                        + ((challenges[OpStackPointerWeight])
                            * (current_base_row[38usize])))
                        + ((challenges[OpStackFirstUnderflowElementWeight])
                            * (current_base_row[37usize])))))
                * ((challenges[OpStackIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((((challenges[OpStackClkWeight])
                            * (current_base_row[7usize]))
                            + ((challenges[OpStackIb1Weight])
                                * (current_base_row[13usize])))
                            + ((challenges[OpStackPointerWeight])
                                * ((current_base_row[38usize])
                                    + (BFieldElement::from_raw_u64(4294967295u64)))))
                            + ((challenges[OpStackFirstUnderflowElementWeight])
                                * (current_base_row[36usize])))));
            let (current_ext_row, mut det_col) = curr_ext_row
                .multi_slice_mut((s![..52usize], s![52usize..= 52usize]));
            det_col[0] = (current_ext_row[50usize])
                * ((challenges[OpStackIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((((challenges[OpStackClkWeight])
                            * (current_base_row[7usize]))
                            + ((challenges[OpStackIb1Weight])
                                * (current_base_row[13usize])))
                            + ((challenges[OpStackPointerWeight])
                                * ((next_base_row[38usize])
                                    + (BFieldElement::from_raw_u64(8589934590u64)))))
                            + ((challenges[OpStackFirstUnderflowElementWeight])
                                * (next_base_row[35usize])))));
            let (current_ext_row, mut det_col) = curr_ext_row
                .multi_slice_mut((s![..53usize], s![53usize..= 53usize]));
            det_col[0] = (current_ext_row[51usize])
                * ((challenges[OpStackIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((((challenges[OpStackClkWeight])
                            * (current_base_row[7usize]))
                            + ((challenges[OpStackIb1Weight])
                                * (current_base_row[13usize])))
                            + ((challenges[OpStackPointerWeight])
                                * ((current_base_row[38usize])
                                    + (BFieldElement::from_raw_u64(8589934590u64)))))
                            + ((challenges[OpStackFirstUnderflowElementWeight])
                                * (current_base_row[35usize])))));
            let (current_ext_row, mut det_col) = curr_ext_row
                .multi_slice_mut((s![..54usize], s![54usize..= 54usize]));
            det_col[0] = (current_ext_row[52usize])
                * ((challenges[OpStackIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((((challenges[OpStackClkWeight])
                            * (current_base_row[7usize]))
                            + ((challenges[OpStackIb1Weight])
                                * (current_base_row[13usize])))
                            + ((challenges[OpStackPointerWeight])
                                * ((next_base_row[38usize])
                                    + (BFieldElement::from_raw_u64(12884901885u64)))))
                            + ((challenges[OpStackFirstUnderflowElementWeight])
                                * (next_base_row[34usize])))));
            let (current_ext_row, mut det_col) = curr_ext_row
                .multi_slice_mut((s![..55usize], s![55usize..= 55usize]));
            det_col[0] = (current_ext_row[53usize])
                * ((challenges[OpStackIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((((challenges[OpStackClkWeight])
                            * (current_base_row[7usize]))
                            + ((challenges[OpStackIb1Weight])
                                * (current_base_row[13usize])))
                            + ((challenges[OpStackPointerWeight])
                                * ((current_base_row[38usize])
                                    + (BFieldElement::from_raw_u64(12884901885u64)))))
                            + ((challenges[OpStackFirstUnderflowElementWeight])
                                * (current_base_row[34usize])))));
            let (current_ext_row, mut det_col) = curr_ext_row
                .multi_slice_mut((s![..56usize], s![56usize..= 56usize]));
            det_col[0] = (current_ext_row[54usize])
                * ((challenges[OpStackIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((((challenges[OpStackClkWeight])
                            * (current_base_row[7usize]))
                            + ((challenges[OpStackIb1Weight])
                                * (current_base_row[13usize])))
                            + ((challenges[OpStackPointerWeight])
                                * ((next_base_row[38usize])
                                    + (BFieldElement::from_raw_u64(17179869180u64)))))
                            + ((challenges[OpStackFirstUnderflowElementWeight])
                                * (next_base_row[33usize])))));
            let (current_ext_row, mut det_col) = curr_ext_row
                .multi_slice_mut((s![..57usize], s![57usize..= 57usize]));
            det_col[0] = ((challenges[RamIndeterminate])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (((((current_base_row[7usize]) * (challenges[RamClkWeight]))
                        + (challenges[RamInstructionTypeWeight]))
                        + (((next_base_row[22usize])
                            + (BFieldElement::from_raw_u64(4294967295u64)))
                            * (challenges[RamPointerWeight])))
                        + ((next_base_row[23usize]) * (challenges[RamValueWeight])))))
                * ((challenges[RamIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((((current_base_row[7usize]) * (challenges[RamClkWeight]))
                            + (challenges[RamInstructionTypeWeight]))
                            + (((next_base_row[22usize])
                                + (BFieldElement::from_raw_u64(8589934590u64)))
                                * (challenges[RamPointerWeight])))
                            + ((next_base_row[24usize])
                                * (challenges[RamValueWeight])))));
            let (current_ext_row, mut det_col) = curr_ext_row
                .multi_slice_mut((s![..58usize], s![58usize..= 58usize]));
            det_col[0] = ((challenges[RamIndeterminate])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * ((((current_base_row[7usize]) * (challenges[RamClkWeight]))
                        + ((current_base_row[22usize]) * (challenges[RamPointerWeight])))
                        + ((current_base_row[23usize]) * (challenges[RamValueWeight])))))
                * ((challenges[RamIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((((current_base_row[7usize]) * (challenges[RamClkWeight]))
                            + (((current_base_row[22usize])
                                + (BFieldElement::from_raw_u64(4294967295u64)))
                                * (challenges[RamPointerWeight])))
                            + ((current_base_row[24usize])
                                * (challenges[RamValueWeight])))));
            let (current_ext_row, mut det_col) = curr_ext_row
                .multi_slice_mut((s![..59usize], s![59usize..= 59usize]));
            det_col[0] = (current_ext_row[6usize]) * (current_ext_row[56usize]);
            let (current_ext_row, mut det_col) = curr_ext_row
                .multi_slice_mut((s![..60usize], s![60usize..= 60usize]));
            det_col[0] = (current_ext_row[55usize])
                * ((challenges[OpStackIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((((challenges[OpStackClkWeight])
                            * (current_base_row[7usize]))
                            + ((challenges[OpStackIb1Weight])
                                * (current_base_row[13usize])))
                            + ((challenges[OpStackPointerWeight])
                                * ((current_base_row[38usize])
                                    + (BFieldElement::from_raw_u64(17179869180u64)))))
                            + ((challenges[OpStackFirstUnderflowElementWeight])
                                * (current_base_row[33usize])))));
            let (current_ext_row, mut det_col) = curr_ext_row
                .multi_slice_mut((s![..61usize], s![61usize..= 61usize]));
            det_col[0] = (current_ext_row[57usize])
                * ((challenges[RamIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((((current_base_row[7usize]) * (challenges[RamClkWeight]))
                            + (challenges[RamInstructionTypeWeight]))
                            + (((next_base_row[22usize])
                                + (BFieldElement::from_raw_u64(12884901885u64)))
                                * (challenges[RamPointerWeight])))
                            + ((next_base_row[25usize])
                                * (challenges[RamValueWeight])))));
            let (current_ext_row, mut det_col) = curr_ext_row
                .multi_slice_mut((s![..62usize], s![62usize..= 62usize]));
            det_col[0] = (current_ext_row[58usize])
                * ((challenges[RamIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((((current_base_row[7usize]) * (challenges[RamClkWeight]))
                            + (((current_base_row[22usize])
                                + (BFieldElement::from_raw_u64(8589934590u64)))
                                * (challenges[RamPointerWeight])))
                            + ((current_base_row[25usize])
                                * (challenges[RamValueWeight])))));
            let (current_ext_row, mut det_col) = curr_ext_row
                .multi_slice_mut((s![..63usize], s![63usize..= 63usize]));
            det_col[0] = (current_ext_row[61usize])
                * ((challenges[RamIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((((current_base_row[7usize]) * (challenges[RamClkWeight]))
                            + (challenges[RamInstructionTypeWeight]))
                            + (((next_base_row[22usize])
                                + (BFieldElement::from_raw_u64(17179869180u64)))
                                * (challenges[RamPointerWeight])))
                            + ((next_base_row[26usize])
                                * (challenges[RamValueWeight])))));
            let (current_ext_row, mut det_col) = curr_ext_row
                .multi_slice_mut((s![..64usize], s![64usize..= 64usize]));
            det_col[0] = (current_ext_row[62usize])
                * ((challenges[RamIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((((current_base_row[7usize]) * (challenges[RamClkWeight]))
                            + (((current_base_row[22usize])
                                + (BFieldElement::from_raw_u64(12884901885u64)))
                                * (challenges[RamPointerWeight])))
                            + ((current_base_row[26usize])
                                * (challenges[RamValueWeight])))));
            let (current_ext_row, mut det_col) = curr_ext_row
                .multi_slice_mut((s![..65usize], s![65usize..= 65usize]));
            det_col[0] = (current_base_row[189usize])
                * ((next_ext_row[7usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((current_ext_row[7usize])
                            * ((current_ext_row[63usize])
                                * ((challenges[RamIndeterminate])
                                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                        * (((((current_base_row[7usize])
                                            * (challenges[RamClkWeight]))
                                            + (challenges[RamInstructionTypeWeight]))
                                            + (((next_base_row[22usize])
                                                + (BFieldElement::from_raw_u64(21474836475u64)))
                                                * (challenges[RamPointerWeight])))
                                            + ((next_base_row[27usize])
                                                * (challenges[RamValueWeight])))))))));
            let (current_ext_row, mut det_col) = curr_ext_row
                .multi_slice_mut((s![..66usize], s![66usize..= 66usize]));
            det_col[0] = (current_base_row[189usize])
                * ((next_ext_row[7usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((current_ext_row[7usize])
                            * ((current_ext_row[64usize])
                                * ((challenges[RamIndeterminate])
                                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                        * ((((current_base_row[7usize])
                                            * (challenges[RamClkWeight]))
                                            + (((current_base_row[22usize])
                                                + (BFieldElement::from_raw_u64(17179869180u64)))
                                                * (challenges[RamPointerWeight])))
                                            + ((current_base_row[27usize])
                                                * (challenges[RamValueWeight])))))))));
            let (current_ext_row, mut det_col) = curr_ext_row
                .multi_slice_mut((s![..67usize], s![67usize..= 67usize]));
            det_col[0] = (((current_ext_row[56usize])
                * ((challenges[OpStackIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((((challenges[OpStackClkWeight])
                            * (current_base_row[7usize]))
                            + ((challenges[OpStackIb1Weight])
                                * (current_base_row[13usize])))
                            + ((challenges[OpStackPointerWeight])
                                * ((next_base_row[38usize])
                                    + (BFieldElement::from_raw_u64(21474836475u64)))))
                            + ((challenges[OpStackFirstUnderflowElementWeight])
                                * (next_base_row[32usize]))))))
                * ((challenges[OpStackIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((((challenges[OpStackClkWeight])
                            * (current_base_row[7usize]))
                            + ((challenges[OpStackIb1Weight])
                                * (current_base_row[13usize])))
                            + ((challenges[OpStackPointerWeight])
                                * ((next_base_row[38usize])
                                    + (BFieldElement::from_raw_u64(25769803770u64)))))
                            + ((challenges[OpStackFirstUnderflowElementWeight])
                                * (next_base_row[31usize]))))))
                * ((challenges[OpStackIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((((challenges[OpStackClkWeight])
                            * (current_base_row[7usize]))
                            + ((challenges[OpStackIb1Weight])
                                * (current_base_row[13usize])))
                            + ((challenges[OpStackPointerWeight])
                                * ((next_base_row[38usize])
                                    + (BFieldElement::from_raw_u64(30064771065u64)))))
                            + ((challenges[OpStackFirstUnderflowElementWeight])
                                * (next_base_row[30usize])))));
            let (current_ext_row, mut det_col) = curr_ext_row
                .multi_slice_mut((s![..68usize], s![68usize..= 68usize]));
            det_col[0] = (((current_ext_row[60usize])
                * ((challenges[OpStackIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((((challenges[OpStackClkWeight])
                            * (current_base_row[7usize]))
                            + ((challenges[OpStackIb1Weight])
                                * (current_base_row[13usize])))
                            + ((challenges[OpStackPointerWeight])
                                * ((current_base_row[38usize])
                                    + (BFieldElement::from_raw_u64(21474836475u64)))))
                            + ((challenges[OpStackFirstUnderflowElementWeight])
                                * (current_base_row[32usize]))))))
                * ((challenges[OpStackIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((((challenges[OpStackClkWeight])
                            * (current_base_row[7usize]))
                            + ((challenges[OpStackIb1Weight])
                                * (current_base_row[13usize])))
                            + ((challenges[OpStackPointerWeight])
                                * ((current_base_row[38usize])
                                    + (BFieldElement::from_raw_u64(25769803770u64)))))
                            + ((challenges[OpStackFirstUnderflowElementWeight])
                                * (current_base_row[31usize]))))))
                * ((challenges[OpStackIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((((challenges[OpStackClkWeight])
                            * (current_base_row[7usize]))
                            + ((challenges[OpStackIb1Weight])
                                * (current_base_row[13usize])))
                            + ((challenges[OpStackPointerWeight])
                                * ((current_base_row[38usize])
                                    + (BFieldElement::from_raw_u64(30064771065u64)))))
                            + ((challenges[OpStackFirstUnderflowElementWeight])
                                * (current_base_row[30usize])))));
            let (current_ext_row, mut det_col) = curr_ext_row
                .multi_slice_mut((s![..69usize], s![69usize..= 69usize]));
            det_col[0] = (current_ext_row[6usize])
                * (((current_ext_row[67usize])
                    * ((challenges[OpStackIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((challenges[OpStackClkWeight])
                                * (current_base_row[7usize]))
                                + ((challenges[OpStackIb1Weight])
                                    * (current_base_row[13usize])))
                                + ((challenges[OpStackPointerWeight])
                                    * ((next_base_row[38usize])
                                        + (BFieldElement::from_raw_u64(34359738360u64)))))
                                + ((challenges[OpStackFirstUnderflowElementWeight])
                                    * (next_base_row[29usize]))))))
                    * ((challenges[OpStackIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((challenges[OpStackClkWeight])
                                * (current_base_row[7usize]))
                                + ((challenges[OpStackIb1Weight])
                                    * (current_base_row[13usize])))
                                + ((challenges[OpStackPointerWeight])
                                    * ((next_base_row[38usize])
                                        + (BFieldElement::from_raw_u64(38654705655u64)))))
                                + ((challenges[OpStackFirstUnderflowElementWeight])
                                    * (next_base_row[28usize]))))));
            let (current_ext_row, mut det_col) = curr_ext_row
                .multi_slice_mut((s![..70usize], s![70usize..= 70usize]));
            det_col[0] = (current_ext_row[6usize])
                * (((current_ext_row[68usize])
                    * ((challenges[OpStackIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((challenges[OpStackClkWeight])
                                * (current_base_row[7usize]))
                                + ((challenges[OpStackIb1Weight])
                                    * (current_base_row[13usize])))
                                + ((challenges[OpStackPointerWeight])
                                    * ((current_base_row[38usize])
                                        + (BFieldElement::from_raw_u64(34359738360u64)))))
                                + ((challenges[OpStackFirstUnderflowElementWeight])
                                    * (current_base_row[29usize]))))))
                    * ((challenges[OpStackIndeterminate])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((challenges[OpStackClkWeight])
                                * (current_base_row[7usize]))
                                + ((challenges[OpStackIb1Weight])
                                    * (current_base_row[13usize])))
                                + ((challenges[OpStackPointerWeight])
                                    * ((current_base_row[38usize])
                                        + (BFieldElement::from_raw_u64(38654705655u64)))))
                                + ((challenges[OpStackFirstUnderflowElementWeight])
                                    * (current_base_row[28usize]))))));
            let (current_ext_row, mut det_col) = curr_ext_row
                .multi_slice_mut((s![..71usize], s![71usize..= 71usize]));
            det_col[0] = (current_ext_row[6usize]) * (current_ext_row[52usize]);
            let (current_ext_row, mut det_col) = curr_ext_row
                .multi_slice_mut((s![..72usize], s![72usize..= 72usize]));
            det_col[0] = (current_ext_row[6usize]) * (current_ext_row[60usize]);
            let (current_ext_row, mut det_col) = curr_ext_row
                .multi_slice_mut((s![..73usize], s![73usize..= 73usize]));
            det_col[0] = (current_ext_row[6usize])
                * ((challenges[OpStackIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((((challenges[OpStackClkWeight])
                            * (current_base_row[7usize]))
                            + ((challenges[OpStackIb1Weight])
                                * (current_base_row[13usize])))
                            + ((challenges[OpStackPointerWeight])
                                * (next_base_row[38usize])))
                            + ((challenges[OpStackFirstUnderflowElementWeight])
                                * (next_base_row[37usize])))));
            let (current_ext_row, mut det_col) = curr_ext_row
                .multi_slice_mut((s![..74usize], s![74usize..= 74usize]));
            det_col[0] = (current_base_row[186usize])
                * ((next_ext_row[6usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((current_ext_row[6usize]) * (current_ext_row[50usize]))));
            let (current_ext_row, mut det_col) = curr_ext_row
                .multi_slice_mut((s![..75usize], s![75usize..= 75usize]));
            det_col[0] = (current_base_row[188usize])
                * ((next_ext_row[6usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((current_ext_row[6usize]) * (current_ext_row[54usize]))));
            let (current_ext_row, mut det_col) = curr_ext_row
                .multi_slice_mut((s![..76usize], s![76usize..= 76usize]));
            det_col[0] = (current_base_row[185usize])
                * ((next_ext_row[6usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((current_ext_row[6usize])
                            * ((challenges[OpStackIndeterminate])
                                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                    * (((((challenges[OpStackClkWeight])
                                        * (current_base_row[7usize]))
                                        + ((challenges[OpStackIb1Weight])
                                            * (current_base_row[13usize])))
                                        + ((challenges[OpStackPointerWeight])
                                            * (current_base_row[38usize])))
                                        + ((challenges[OpStackFirstUnderflowElementWeight])
                                            * (current_base_row[37usize]))))))));
            let (current_ext_row, mut det_col) = curr_ext_row
                .multi_slice_mut((s![..77usize], s![77usize..= 77usize]));
            det_col[0] = (current_base_row[186usize])
                * ((next_ext_row[6usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((current_ext_row[6usize]) * (current_ext_row[51usize]))));
            let (current_ext_row, mut det_col) = curr_ext_row
                .multi_slice_mut((s![..78usize], s![78usize..= 78usize]));
            det_col[0] = (current_base_row[187usize])
                * ((next_ext_row[6usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((current_ext_row[6usize]) * (current_ext_row[53usize]))));
            let (current_ext_row, mut det_col) = curr_ext_row
                .multi_slice_mut((s![..79usize], s![79usize..= 79usize]));
            det_col[0] = (current_base_row[188usize])
                * ((next_ext_row[6usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((current_ext_row[6usize]) * (current_ext_row[55usize]))));
            let (current_ext_row, mut det_col) = curr_ext_row
                .multi_slice_mut((s![..80usize], s![80usize..= 80usize]));
            det_col[0] = ((((next_ext_row[21usize])
                + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                    * (current_ext_row[21usize])))
                * ((challenges[ClockJumpDifferenceLookupIndeterminate])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((next_base_row[50usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (current_base_row[50usize]))))))
                + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                * ((BFieldElement::from_raw_u64(4294967295u64))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((next_base_row[52usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (current_base_row[52usize])))
                            * (current_base_row[54usize]))));
            let (current_ext_row, mut det_col) = curr_ext_row
                .multi_slice_mut((s![..81usize], s![81usize..= 81usize]));
            det_col[0] = (current_base_row[215usize])
                * ((((((current_base_row[185usize])
                    * ((next_ext_row[7usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((current_ext_row[7usize])
                                * ((challenges[RamIndeterminate])
                                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                        * (((((current_base_row[7usize])
                                            * (challenges[RamClkWeight]))
                                            + (challenges[RamInstructionTypeWeight]))
                                            + (((next_base_row[22usize])
                                                + (BFieldElement::from_raw_u64(4294967295u64)))
                                                * (challenges[RamPointerWeight])))
                                            + ((next_base_row[23usize])
                                                * (challenges[RamValueWeight])))))))))
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
                    + (current_ext_row[65usize]));
            let (current_ext_row, mut det_col) = curr_ext_row
                .multi_slice_mut((s![..82usize], s![82usize..= 82usize]));
            det_col[0] = (current_base_row[221usize])
                * ((((((current_base_row[185usize])
                    * ((next_ext_row[7usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((current_ext_row[7usize])
                                * ((challenges[RamIndeterminate])
                                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                        * ((((current_base_row[7usize])
                                            * (challenges[RamClkWeight]))
                                            + ((current_base_row[22usize])
                                                * (challenges[RamPointerWeight])))
                                            + ((current_base_row[23usize])
                                                * (challenges[RamValueWeight])))))))))
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
                    + (current_ext_row[66usize]));
        }
    }
}
