//! The degree lowering table contains the introduced variables that allow
//! lowering the degree of the AIR. See
//! [`crate::table::master_table::AIR_TARGET_DEGREE`]
//! for additional information.
//!
//! This file has been auto-generated. Any modifications _will_ be lost.
//! To re-generate, execute:
//! `cargo run --bin constraint-evaluation-generator`
use ndarray::Array1;
use ndarray::s;
use ndarray::ArrayView2;
use ndarray::ArrayViewMut2;
use ndarray::Axis;
use ndarray::Zip;
use strum::Display;
use strum::EnumCount;
use strum::EnumIter;
use twenty_first::prelude::BFieldElement;
use twenty_first::prelude::XFieldElement;
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
    DegreeLoweringBaseCol207,
    DegreeLoweringBaseCol208,
    DegreeLoweringBaseCol209,
    DegreeLoweringBaseCol210,
    DegreeLoweringBaseCol211,
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
    DegreeLoweringExtCol33,
    DegreeLoweringExtCol34,
    DegreeLoweringExtCol35,
}
#[derive(Debug, Copy, Clone, Eq, PartialEq)]
pub struct DegreeLoweringTable;
impl DegreeLoweringTable {
    #[allow(unused_variables)]
    pub fn fill_derived_base_columns(
        mut master_base_table: ArrayViewMut2<BFieldElement>,
    ) {
        assert_eq!(NUM_BASE_COLUMNS, master_base_table.ncols());
        let (original_part, mut current_section) = master_base_table
            .multi_slice_mut((s![.., 0..149usize], s![.., 149usize..149usize + 2usize]));
        Zip::from(original_part.rows())
            .and(current_section.rows_mut())
            .par_for_each(|original_row, mut section_row| {
                let mut base_row = original_row.to_owned();
                section_row[0usize] = ((((base_row[12usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                    * (base_row[13usize]))
                    * ((base_row[14usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                    * ((base_row[15usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                base_row.push(Axis(0), section_row.slice(s![0usize])).unwrap();
                section_row[1usize] = (((base_row[149usize]) * (base_row[16usize]))
                    * ((base_row[17usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                    * ((base_row[18usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                base_row.push(Axis(0), section_row.slice(s![1usize])).unwrap();
            });
        let (original_part, mut current_section) = master_base_table
            .multi_slice_mut((
                s![.., 0..151usize],
                s![.., 151usize..151usize + 18usize],
            ));
        Zip::from(original_part.rows())
            .and(current_section.rows_mut())
            .par_for_each(|original_row, mut section_row| {
                let mut base_row = original_row.to_owned();
                section_row[0usize] = (base_row[64usize])
                    * ((base_row[64usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                base_row.push(Axis(0), section_row.slice(s![0usize])).unwrap();
                section_row[1usize] = ((((base_row[64usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                    * ((base_row[64usize])
                        + (BFieldElement::from_raw_u64(18446744060824649731u64))))
                    * ((base_row[64usize])
                        + (BFieldElement::from_raw_u64(18446744056529682436u64))))
                    * ((base_row[64usize])
                        + (BFieldElement::from_raw_u64(18446744052234715141u64)));
                base_row.push(Axis(0), section_row.slice(s![1usize])).unwrap();
                section_row[2usize] = (((base_row[64usize])
                    * ((base_row[64usize])
                        + (BFieldElement::from_raw_u64(18446744060824649731u64))))
                    * ((base_row[64usize])
                        + (BFieldElement::from_raw_u64(18446744056529682436u64))))
                    * ((base_row[64usize])
                        + (BFieldElement::from_raw_u64(18446744052234715141u64)));
                base_row.push(Axis(0), section_row.slice(s![2usize])).unwrap();
                section_row[3usize] = (base_row[151usize])
                    * ((base_row[64usize])
                        + (BFieldElement::from_raw_u64(18446744060824649731u64)));
                base_row.push(Axis(0), section_row.slice(s![3usize])).unwrap();
                section_row[4usize] = (((base_row[151usize])
                    * ((base_row[64usize])
                        + (BFieldElement::from_raw_u64(18446744056529682436u64))))
                    * ((base_row[64usize])
                        + (BFieldElement::from_raw_u64(18446744052234715141u64))))
                    * ((base_row[64usize])
                        + (BFieldElement::from_raw_u64(18446744047939747846u64)));
                base_row.push(Axis(0), section_row.slice(s![4usize])).unwrap();
                section_row[5usize] = ((base_row[142usize])
                    + (BFieldElement::from_raw_u64(18446744052234715141u64)))
                    * ((base_row[142usize])
                        + (BFieldElement::from_raw_u64(18446744043644780551u64)));
                base_row.push(Axis(0), section_row.slice(s![5usize])).unwrap();
                section_row[6usize] = (base_row[143usize]) * (base_row[144usize]);
                base_row.push(Axis(0), section_row.slice(s![6usize])).unwrap();
                section_row[7usize] = ((((base_row[142usize])
                    + (BFieldElement::from_raw_u64(18446744052234715141u64)))
                    * ((base_row[142usize])
                        + (BFieldElement::from_raw_u64(18446744009285042191u64))))
                    * ((base_row[142usize])
                        + (BFieldElement::from_raw_u64(18446744017874976781u64))))
                    * ((base_row[142usize])
                        + (BFieldElement::from_raw_u64(18446743940565565471u64)));
                base_row.push(Axis(0), section_row.slice(s![7usize])).unwrap();
                section_row[8usize] = (base_row[156usize])
                    * ((base_row[142usize])
                        + (BFieldElement::from_raw_u64(18446744009285042191u64)));
                base_row.push(Axis(0), section_row.slice(s![8usize])).unwrap();
                section_row[9usize] = (base_row[145usize]) * (base_row[146usize]);
                base_row.push(Axis(0), section_row.slice(s![9usize])).unwrap();
                section_row[10usize] = (((base_row[62usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                    * ((base_row[62usize])
                        + (BFieldElement::from_raw_u64(18446744060824649731u64))))
                    * (base_row[62usize]);
                base_row.push(Axis(0), section_row.slice(s![10usize])).unwrap();
                section_row[11usize] = (base_row[158usize])
                    * ((base_row[142usize])
                        + (BFieldElement::from_raw_u64(18446743949155500061u64)));
                base_row.push(Axis(0), section_row.slice(s![11usize])).unwrap();
                section_row[12usize] = (((base_row[156usize])
                    * ((base_row[142usize])
                        + (BFieldElement::from_raw_u64(18446744017874976781u64))))
                    * ((base_row[142usize])
                        + (BFieldElement::from_raw_u64(18446743940565565471u64))))
                    * ((base_row[142usize])
                        + (BFieldElement::from_raw_u64(18446743949155500061u64)));
                base_row.push(Axis(0), section_row.slice(s![12usize])).unwrap();
                section_row[13usize] = ((base_row[159usize])
                    * ((base_row[142usize])
                        + (BFieldElement::from_raw_u64(18446743940565565471u64))))
                    * ((base_row[142usize])
                        + (BFieldElement::from_raw_u64(18446743949155500061u64)));
                base_row.push(Axis(0), section_row.slice(s![13usize])).unwrap();
                section_row[14usize] = ((((base_row[62usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                    * ((base_row[62usize])
                        + (BFieldElement::from_raw_u64(18446744056529682436u64))))
                    * (base_row[62usize]))
                    * ((base_row[63usize])
                        + (BFieldElement::from_raw_u64(18446743897615892521u64)));
                base_row.push(Axis(0), section_row.slice(s![14usize])).unwrap();
                section_row[15usize] = (base_row[159usize])
                    * ((base_row[142usize])
                        + (BFieldElement::from_raw_u64(18446744017874976781u64)));
                base_row.push(Axis(0), section_row.slice(s![15usize])).unwrap();
                section_row[16usize] = (((base_row[162usize])
                    * ((base_row[139usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                    * ((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (base_row[157usize]))))
                    * ((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (base_row[160usize])));
                base_row.push(Axis(0), section_row.slice(s![16usize])).unwrap();
                section_row[17usize] = (((base_row[162usize]) * (base_row[139usize]))
                    * ((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (base_row[157usize]))))
                    * ((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (base_row[160usize])));
                base_row.push(Axis(0), section_row.slice(s![17usize])).unwrap();
            });
        let num_rows = master_base_table.nrows();
        let (original_part, mut current_section) = master_base_table
            .multi_slice_mut((
                s![.., 0..169usize],
                s![.., 169usize..169usize + 192usize],
            ));
        let row_indices = Array1::from_vec((0..num_rows - 1).collect::<Vec<_>>());
        Zip::from(current_section.slice_mut(s![0..num_rows - 1, ..]).rows_mut())
            .and(row_indices.view())
            .par_for_each(|mut section_row, &current_row_index| {
                let next_row_index = current_row_index + 1;
                let current_base_row_slice = original_part
                    .slice(s![current_row_index..= current_row_index, ..]);
                let next_base_row_slice = original_part
                    .slice(s![next_row_index..= next_row_index, ..]);
                let mut current_base_row = current_base_row_slice.row(0).to_owned();
                let next_base_row = next_base_row_slice.row(0);
                section_row[0usize] = ((current_base_row[12usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                    * ((current_base_row[13usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![0usize])).unwrap();
                section_row[1usize] = ((BFieldElement::from_raw_u64(4294967295u64))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[42usize])))
                    * ((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[41usize])));
                current_base_row.push(Axis(0), section_row.slice(s![1usize])).unwrap();
                section_row[2usize] = ((current_base_row[12usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                    * (current_base_row[13usize]);
                current_base_row.push(Axis(0), section_row.slice(s![2usize])).unwrap();
                section_row[3usize] = (current_base_row[169usize])
                    * ((current_base_row[14usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![3usize])).unwrap();
                section_row[4usize] = ((current_base_row[12usize])
                    * ((current_base_row[13usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                    * ((current_base_row[14usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![4usize])).unwrap();
                section_row[5usize] = (current_base_row[171usize])
                    * ((current_base_row[14usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![5usize])).unwrap();
                section_row[6usize] = (current_base_row[170usize])
                    * (current_base_row[40usize]);
                current_base_row.push(Axis(0), section_row.slice(s![6usize])).unwrap();
                section_row[7usize] = ((BFieldElement::from_raw_u64(4294967295u64))
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_base_row[42usize]))) * (current_base_row[41usize]);
                current_base_row.push(Axis(0), section_row.slice(s![7usize])).unwrap();
                section_row[8usize] = (current_base_row[176usize])
                    * ((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[40usize])));
                current_base_row.push(Axis(0), section_row.slice(s![8usize])).unwrap();
                section_row[9usize] = (current_base_row[172usize])
                    * ((current_base_row[15usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![9usize])).unwrap();
                section_row[10usize] = (current_base_row[173usize])
                    * ((current_base_row[15usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![10usize])).unwrap();
                section_row[11usize] = (current_base_row[172usize])
                    * (current_base_row[15usize]);
                current_base_row.push(Axis(0), section_row.slice(s![11usize])).unwrap();
                section_row[12usize] = (current_base_row[174usize])
                    * ((current_base_row[15usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![12usize])).unwrap();
                section_row[13usize] = (current_base_row[174usize])
                    * (current_base_row[15usize]);
                current_base_row.push(Axis(0), section_row.slice(s![13usize])).unwrap();
                section_row[14usize] = (current_base_row[170usize])
                    * ((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[40usize])));
                current_base_row.push(Axis(0), section_row.slice(s![14usize])).unwrap();
                section_row[15usize] = (current_base_row[173usize])
                    * (current_base_row[15usize]);
                current_base_row.push(Axis(0), section_row.slice(s![15usize])).unwrap();
                section_row[16usize] = (current_base_row[183usize])
                    * (current_base_row[39usize]);
                current_base_row.push(Axis(0), section_row.slice(s![16usize])).unwrap();
                section_row[17usize] = (current_base_row[175usize])
                    * ((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[39usize])));
                current_base_row.push(Axis(0), section_row.slice(s![17usize])).unwrap();
                section_row[18usize] = (current_base_row[175usize])
                    * (current_base_row[39usize]);
                current_base_row.push(Axis(0), section_row.slice(s![18usize])).unwrap();
                section_row[19usize] = (current_base_row[178usize])
                    * ((current_base_row[16usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![19usize])).unwrap();
                section_row[20usize] = (current_base_row[177usize])
                    * ((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[39usize])));
                current_base_row.push(Axis(0), section_row.slice(s![20usize])).unwrap();
                section_row[21usize] = (current_base_row[177usize])
                    * (current_base_row[39usize]);
                current_base_row.push(Axis(0), section_row.slice(s![21usize])).unwrap();
                section_row[22usize] = ((current_base_row[12usize])
                    * (current_base_row[13usize]))
                    * ((current_base_row[14usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![22usize])).unwrap();
                section_row[23usize] = (current_base_row[184usize])
                    * ((current_base_row[16usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![23usize])).unwrap();
                section_row[24usize] = (current_base_row[179usize])
                    * ((current_base_row[16usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![24usize])).unwrap();
                section_row[25usize] = (current_base_row[180usize])
                    * ((current_base_row[16usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![25usize])).unwrap();
                section_row[26usize] = (current_base_row[181usize])
                    * ((current_base_row[16usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![26usize])).unwrap();
                section_row[27usize] = (current_base_row[171usize])
                    * (current_base_row[14usize]);
                current_base_row.push(Axis(0), section_row.slice(s![27usize])).unwrap();
                section_row[28usize] = (current_base_row[182usize])
                    * ((current_base_row[16usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![28usize])).unwrap();
                section_row[29usize] = (current_base_row[169usize])
                    * (current_base_row[14usize]);
                current_base_row.push(Axis(0), section_row.slice(s![29usize])).unwrap();
                section_row[30usize] = (current_base_row[178usize])
                    * (current_base_row[16usize]);
                current_base_row.push(Axis(0), section_row.slice(s![30usize])).unwrap();
                section_row[31usize] = (current_base_row[188usize])
                    * ((current_base_row[17usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![31usize])).unwrap();
                section_row[32usize] = (current_base_row[191usize])
                    * ((current_base_row[15usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![32usize])).unwrap();
                section_row[33usize] = (current_base_row[180usize])
                    * (current_base_row[16usize]);
                current_base_row.push(Axis(0), section_row.slice(s![33usize])).unwrap();
                section_row[34usize] = (current_base_row[179usize])
                    * (current_base_row[16usize]);
                current_base_row.push(Axis(0), section_row.slice(s![34usize])).unwrap();
                section_row[35usize] = (current_base_row[181usize])
                    * (current_base_row[16usize]);
                current_base_row.push(Axis(0), section_row.slice(s![35usize])).unwrap();
                section_row[36usize] = (current_base_row[182usize])
                    * (current_base_row[16usize]);
                current_base_row.push(Axis(0), section_row.slice(s![36usize])).unwrap();
                section_row[37usize] = (current_base_row[195usize])
                    * ((current_base_row[17usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![37usize])).unwrap();
                section_row[38usize] = (((current_base_row[201usize])
                    * ((current_base_row[16usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                    * ((current_base_row[17usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                    * ((current_base_row[18usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![38usize])).unwrap();
                section_row[39usize] = (current_base_row[194usize])
                    * ((current_base_row[17usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![39usize])).unwrap();
                section_row[40usize] = ((current_base_row[192usize])
                    * ((current_base_row[17usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                    * ((current_base_row[18usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![40usize])).unwrap();
                section_row[41usize] = (current_base_row[197usize])
                    * ((current_base_row[17usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![41usize])).unwrap();
                section_row[42usize] = ((current_base_row[203usize])
                    * ((current_base_row[17usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                    * ((current_base_row[18usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![42usize])).unwrap();
                section_row[43usize] = (current_base_row[202usize])
                    * ((current_base_row[17usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![43usize])).unwrap();
                section_row[44usize] = ((current_base_row[193usize])
                    * ((current_base_row[17usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                    * ((current_base_row[18usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![44usize])).unwrap();
                section_row[45usize] = (current_base_row[206usize])
                    * ((current_base_row[18usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![45usize])).unwrap();
                section_row[46usize] = (current_base_row[199usize])
                    * ((current_base_row[17usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![46usize])).unwrap();
                section_row[47usize] = (current_base_row[200usize])
                    * ((current_base_row[18usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![47usize])).unwrap();
                section_row[48usize] = (current_base_row[196usize])
                    * ((current_base_row[15usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![48usize])).unwrap();
                section_row[49usize] = (current_base_row[188usize])
                    * (current_base_row[17usize]);
                current_base_row.push(Axis(0), section_row.slice(s![49usize])).unwrap();
                section_row[50usize] = ((current_base_row[192usize])
                    * (current_base_row[17usize]))
                    * ((current_base_row[18usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![50usize])).unwrap();
                section_row[51usize] = (current_base_row[198usize])
                    * ((current_base_row[15usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![51usize])).unwrap();
                section_row[52usize] = (current_base_row[208usize])
                    * ((current_base_row[18usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![52usize])).unwrap();
                section_row[53usize] = (((current_base_row[191usize])
                    * (current_base_row[15usize]))
                    * ((current_base_row[16usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                    * ((current_base_row[17usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![53usize])).unwrap();
                section_row[54usize] = (current_base_row[196usize])
                    * (current_base_row[15usize]);
                current_base_row.push(Axis(0), section_row.slice(s![54usize])).unwrap();
                section_row[55usize] = ((current_base_row[193usize])
                    * (current_base_row[17usize]))
                    * ((current_base_row[18usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![55usize])).unwrap();
                section_row[56usize] = (current_base_row[198usize])
                    * (current_base_row[15usize]);
                current_base_row.push(Axis(0), section_row.slice(s![56usize])).unwrap();
                section_row[57usize] = (current_base_row[222usize])
                    * ((current_base_row[18usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![57usize])).unwrap();
                section_row[58usize] = (current_base_row[212usize])
                    * ((current_base_row[18usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![58usize])).unwrap();
                section_row[59usize] = (current_base_row[215usize])
                    * ((current_base_row[18usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![59usize])).unwrap();
                section_row[60usize] = (current_base_row[218usize])
                    * ((current_base_row[18usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![60usize])).unwrap();
                section_row[61usize] = (current_base_row[210usize])
                    * ((current_base_row[18usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![61usize])).unwrap();
                section_row[62usize] = (current_base_row[204usize])
                    * ((current_base_row[17usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![62usize])).unwrap();
                section_row[63usize] = (((current_base_row[184usize])
                    * (current_base_row[16usize]))
                    * ((current_base_row[17usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                    * ((current_base_row[18usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![63usize])).unwrap();
                section_row[64usize] = ((current_base_row[205usize])
                    * ((current_base_row[17usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                    * ((current_base_row[18usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![64usize])).unwrap();
                section_row[65usize] = ((current_base_row[194usize])
                    * (current_base_row[17usize]))
                    * ((current_base_row[18usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![65usize])).unwrap();
                section_row[66usize] = (current_base_row[42usize])
                    * ((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[41usize])));
                current_base_row.push(Axis(0), section_row.slice(s![66usize])).unwrap();
                section_row[67usize] = (current_base_row[42usize])
                    * (current_base_row[41usize]);
                current_base_row.push(Axis(0), section_row.slice(s![67usize])).unwrap();
                section_row[68usize] = ((current_base_row[197usize])
                    * (current_base_row[17usize]))
                    * ((current_base_row[18usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![68usize])).unwrap();
                section_row[69usize] = (((current_base_row[220usize])
                    * ((current_base_row[16usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                    * ((current_base_row[17usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                    * ((current_base_row[18usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![69usize])).unwrap();
                section_row[70usize] = ((current_base_row[204usize])
                    * (current_base_row[17usize]))
                    * ((current_base_row[18usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![70usize])).unwrap();
                section_row[71usize] = (current_base_row[200usize])
                    * (current_base_row[18usize]);
                current_base_row.push(Axis(0), section_row.slice(s![71usize])).unwrap();
                section_row[72usize] = ((current_base_row[205usize])
                    * (current_base_row[17usize]))
                    * ((current_base_row[18usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![72usize])).unwrap();
                section_row[73usize] = (((current_base_row[217usize])
                    * ((current_base_row[16usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                    * ((current_base_row[17usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                    * ((current_base_row[18usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![73usize])).unwrap();
                section_row[74usize] = (((current_base_row[223usize])
                    * ((current_base_row[16usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                    * ((current_base_row[17usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                    * ((current_base_row[18usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![74usize])).unwrap();
                section_row[75usize] = ((current_base_row[199usize])
                    * (current_base_row[17usize]))
                    * ((current_base_row[18usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![75usize])).unwrap();
                section_row[76usize] = (((current_base_row[217usize])
                    * (current_base_row[16usize]))
                    * ((current_base_row[17usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                    * ((current_base_row[18usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![76usize])).unwrap();
                section_row[77usize] = (((current_base_row[225usize])
                    * ((current_base_row[16usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                    * ((current_base_row[17usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                    * ((current_base_row[18usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![77usize])).unwrap();
                section_row[78usize] = (((current_base_row[223usize])
                    * (current_base_row[16usize]))
                    * ((current_base_row[17usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                    * ((current_base_row[18usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![78usize])).unwrap();
                section_row[79usize] = (current_base_row[231usize])
                    * ((current_base_row[18usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![79usize])).unwrap();
                section_row[80usize] = (((current_base_row[220usize])
                    * (current_base_row[16usize]))
                    * ((current_base_row[17usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                    * ((current_base_row[18usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![80usize])).unwrap();
                section_row[81usize] = ((current_base_row[195usize])
                    * (current_base_row[17usize]))
                    * ((current_base_row[18usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![81usize])).unwrap();
                section_row[82usize] = (((current_base_row[225usize])
                    * (current_base_row[16usize]))
                    * ((current_base_row[17usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                    * ((current_base_row[18usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![82usize])).unwrap();
                section_row[83usize] = (current_base_row[41usize])
                    * ((current_base_row[41usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![83usize])).unwrap();
                section_row[84usize] = (((current_base_row[97usize])
                    * (current_base_row[97usize])) * (current_base_row[97usize]))
                    * (current_base_row[97usize]);
                current_base_row.push(Axis(0), section_row.slice(s![84usize])).unwrap();
                section_row[85usize] = ((current_base_row[202usize])
                    * (current_base_row[17usize]))
                    * ((current_base_row[18usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![85usize])).unwrap();
                section_row[86usize] = ((current_base_row[203usize])
                    * (current_base_row[17usize]))
                    * ((current_base_row[18usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![86usize])).unwrap();
                section_row[87usize] = (((current_base_row[98usize])
                    * (current_base_row[98usize])) * (current_base_row[98usize]))
                    * (current_base_row[98usize]);
                current_base_row.push(Axis(0), section_row.slice(s![87usize])).unwrap();
                section_row[88usize] = (((current_base_row[99usize])
                    * (current_base_row[99usize])) * (current_base_row[99usize]))
                    * (current_base_row[99usize]);
                current_base_row.push(Axis(0), section_row.slice(s![88usize])).unwrap();
                section_row[89usize] = (((current_base_row[201usize])
                    * (current_base_row[16usize]))
                    * ((current_base_row[17usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                    * ((current_base_row[18usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![89usize])).unwrap();
                section_row[90usize] = (((current_base_row[100usize])
                    * (current_base_row[100usize])) * (current_base_row[100usize]))
                    * (current_base_row[100usize]);
                current_base_row.push(Axis(0), section_row.slice(s![90usize])).unwrap();
                section_row[91usize] = (((current_base_row[101usize])
                    * (current_base_row[101usize])) * (current_base_row[101usize]))
                    * (current_base_row[101usize]);
                current_base_row.push(Axis(0), section_row.slice(s![91usize])).unwrap();
                section_row[92usize] = (current_base_row[42usize])
                    * ((current_base_row[42usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![92usize])).unwrap();
                section_row[93usize] = (((current_base_row[102usize])
                    * (current_base_row[102usize])) * (current_base_row[102usize]))
                    * (current_base_row[102usize]);
                current_base_row.push(Axis(0), section_row.slice(s![93usize])).unwrap();
                section_row[94usize] = (current_base_row[176usize])
                    * (current_base_row[40usize]);
                current_base_row.push(Axis(0), section_row.slice(s![94usize])).unwrap();
                section_row[95usize] = (current_base_row[235usize])
                    * ((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[40usize])));
                current_base_row.push(Axis(0), section_row.slice(s![95usize])).unwrap();
                section_row[96usize] = (current_base_row[235usize])
                    * (current_base_row[40usize]);
                current_base_row.push(Axis(0), section_row.slice(s![96usize])).unwrap();
                section_row[97usize] = (current_base_row[236usize])
                    * ((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[40usize])));
                current_base_row.push(Axis(0), section_row.slice(s![97usize])).unwrap();
                section_row[98usize] = (current_base_row[236usize])
                    * (current_base_row[40usize]);
                current_base_row.push(Axis(0), section_row.slice(s![98usize])).unwrap();
                section_row[99usize] = (((current_base_row[103usize])
                    * (current_base_row[103usize])) * (current_base_row[103usize]))
                    * (current_base_row[103usize]);
                current_base_row.push(Axis(0), section_row.slice(s![99usize])).unwrap();
                section_row[100usize] = (((current_base_row[104usize])
                    * (current_base_row[104usize])) * (current_base_row[104usize]))
                    * (current_base_row[104usize]);
                current_base_row.push(Axis(0), section_row.slice(s![100usize])).unwrap();
                section_row[101usize] = (((current_base_row[105usize])
                    * (current_base_row[105usize])) * (current_base_row[105usize]))
                    * (current_base_row[105usize]);
                current_base_row.push(Axis(0), section_row.slice(s![101usize])).unwrap();
                section_row[102usize] = (current_base_row[40usize])
                    * ((current_base_row[40usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![102usize])).unwrap();
                section_row[103usize] = (((current_base_row[106usize])
                    * (current_base_row[106usize])) * (current_base_row[106usize]))
                    * (current_base_row[106usize]);
                current_base_row.push(Axis(0), section_row.slice(s![103usize])).unwrap();
                section_row[104usize] = (((current_base_row[107usize])
                    * (current_base_row[107usize])) * (current_base_row[107usize]))
                    * (current_base_row[107usize]);
                current_base_row.push(Axis(0), section_row.slice(s![104usize])).unwrap();
                section_row[105usize] = (((current_base_row[108usize])
                    * (current_base_row[108usize])) * (current_base_row[108usize]))
                    * (current_base_row[108usize]);
                current_base_row.push(Axis(0), section_row.slice(s![105usize])).unwrap();
                section_row[106usize] = (current_base_row[39usize])
                    * ((current_base_row[28usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[27usize])));
                current_base_row.push(Axis(0), section_row.slice(s![106usize])).unwrap();
                section_row[107usize] = (current_base_row[39usize])
                    * (current_base_row[22usize]);
                current_base_row.push(Axis(0), section_row.slice(s![107usize])).unwrap();
                section_row[108usize] = (current_base_row[206usize])
                    * (current_base_row[18usize]);
                current_base_row.push(Axis(0), section_row.slice(s![108usize])).unwrap();
                section_row[109usize] = (current_base_row[210usize])
                    * (current_base_row[18usize]);
                current_base_row.push(Axis(0), section_row.slice(s![109usize])).unwrap();
                section_row[110usize] = (current_base_row[208usize])
                    * (current_base_row[18usize]);
                current_base_row.push(Axis(0), section_row.slice(s![110usize])).unwrap();
                section_row[111usize] = (((next_base_row[64usize])
                    * ((next_base_row[64usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                    * ((next_base_row[64usize])
                        + (BFieldElement::from_raw_u64(18446744060824649731u64))))
                    * ((next_base_row[64usize])
                        + (BFieldElement::from_raw_u64(18446744056529682436u64)));
                current_base_row.push(Axis(0), section_row.slice(s![111usize])).unwrap();
                section_row[112usize] = (current_base_row[231usize])
                    * (current_base_row[18usize]);
                current_base_row.push(Axis(0), section_row.slice(s![112usize])).unwrap();
                section_row[113usize] = ((next_base_row[62usize])
                    * ((next_base_row[64usize])
                        + (BFieldElement::from_raw_u64(18446744047939747846u64))))
                    * ((next_base_row[63usize])
                        + (BFieldElement::from_raw_u64(18446743897615892521u64)));
                current_base_row.push(Axis(0), section_row.slice(s![113usize])).unwrap();
                section_row[114usize] = (current_base_row[44usize])
                    * ((current_base_row[44usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![114usize])).unwrap();
                section_row[115usize] = (((current_base_row[43usize])
                    * ((current_base_row[43usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                    * ((current_base_row[43usize])
                        + (BFieldElement::from_raw_u64(18446744060824649731u64))))
                    * ((current_base_row[43usize])
                        + (BFieldElement::from_raw_u64(18446744056529682436u64)));
                current_base_row.push(Axis(0), section_row.slice(s![115usize])).unwrap();
                section_row[116usize] = (current_base_row[185usize])
                    * ((next_base_row[24usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[23usize])));
                current_base_row.push(Axis(0), section_row.slice(s![116usize])).unwrap();
                section_row[117usize] = (current_base_row[186usize])
                    * ((next_base_row[25usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[23usize])));
                current_base_row.push(Axis(0), section_row.slice(s![117usize])).unwrap();
                section_row[118usize] = (current_base_row[187usize])
                    * ((next_base_row[26usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[23usize])));
                current_base_row.push(Axis(0), section_row.slice(s![118usize])).unwrap();
                section_row[119usize] = (current_base_row[189usize])
                    * ((next_base_row[27usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[23usize])));
                current_base_row.push(Axis(0), section_row.slice(s![119usize])).unwrap();
                section_row[120usize] = (current_base_row[190usize])
                    * ((next_base_row[28usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[23usize])));
                current_base_row.push(Axis(0), section_row.slice(s![120usize])).unwrap();
                section_row[121usize] = (current_base_row[185usize])
                    * ((next_base_row[23usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[24usize])));
                current_base_row.push(Axis(0), section_row.slice(s![121usize])).unwrap();
                section_row[122usize] = (current_base_row[186usize])
                    * ((next_base_row[23usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[25usize])));
                current_base_row.push(Axis(0), section_row.slice(s![122usize])).unwrap();
                section_row[123usize] = (current_base_row[187usize])
                    * ((next_base_row[23usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[26usize])));
                current_base_row.push(Axis(0), section_row.slice(s![123usize])).unwrap();
                section_row[124usize] = (current_base_row[189usize])
                    * ((next_base_row[23usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[27usize])));
                current_base_row.push(Axis(0), section_row.slice(s![124usize])).unwrap();
                section_row[125usize] = (current_base_row[190usize])
                    * ((next_base_row[23usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[28usize])));
                current_base_row.push(Axis(0), section_row.slice(s![125usize])).unwrap();
                section_row[126usize] = (current_base_row[215usize])
                    * (current_base_row[18usize]);
                current_base_row.push(Axis(0), section_row.slice(s![126usize])).unwrap();
                section_row[127usize] = (current_base_row[212usize])
                    * (current_base_row[18usize]);
                current_base_row.push(Axis(0), section_row.slice(s![127usize])).unwrap();
                section_row[128usize] = (current_base_row[218usize])
                    * (current_base_row[18usize]);
                current_base_row.push(Axis(0), section_row.slice(s![128usize])).unwrap();
                section_row[129usize] = ((((next_base_row[142usize])
                    + (BFieldElement::from_raw_u64(18446744052234715141u64)))
                    * ((next_base_row[142usize])
                        + (BFieldElement::from_raw_u64(18446744009285042191u64))))
                    * ((next_base_row[142usize])
                        + (BFieldElement::from_raw_u64(18446744017874976781u64))))
                    * ((next_base_row[142usize])
                        + (BFieldElement::from_raw_u64(18446743940565565471u64)));
                current_base_row.push(Axis(0), section_row.slice(s![129usize])).unwrap();
                section_row[130usize] = ((next_base_row[142usize])
                    + (BFieldElement::from_raw_u64(18446744052234715141u64)))
                    * ((next_base_row[142usize])
                        + (BFieldElement::from_raw_u64(18446744043644780551u64)));
                current_base_row.push(Axis(0), section_row.slice(s![130usize])).unwrap();
                section_row[131usize] = ((((next_base_row[64usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                    * ((next_base_row[64usize])
                        + (BFieldElement::from_raw_u64(18446744060824649731u64))))
                    * ((next_base_row[64usize])
                        + (BFieldElement::from_raw_u64(18446744056529682436u64))))
                    * ((next_base_row[64usize])
                        + (BFieldElement::from_raw_u64(18446744052234715141u64)));
                current_base_row.push(Axis(0), section_row.slice(s![131usize])).unwrap();
                section_row[132usize] = (((current_base_row[253usize])
                    * (current_base_row[97usize])) * (current_base_row[97usize]))
                    * (current_base_row[97usize]);
                current_base_row.push(Axis(0), section_row.slice(s![132usize])).unwrap();
                section_row[133usize] = (((current_base_row[256usize])
                    * (current_base_row[98usize])) * (current_base_row[98usize]))
                    * (current_base_row[98usize]);
                current_base_row.push(Axis(0), section_row.slice(s![133usize])).unwrap();
                section_row[134usize] = (((current_base_row[257usize])
                    * (current_base_row[99usize])) * (current_base_row[99usize]))
                    * (current_base_row[99usize]);
                current_base_row.push(Axis(0), section_row.slice(s![134usize])).unwrap();
                section_row[135usize] = (((current_base_row[259usize])
                    * (current_base_row[100usize])) * (current_base_row[100usize]))
                    * (current_base_row[100usize]);
                current_base_row.push(Axis(0), section_row.slice(s![135usize])).unwrap();
                section_row[136usize] = (((current_base_row[260usize])
                    * (current_base_row[101usize])) * (current_base_row[101usize]))
                    * (current_base_row[101usize]);
                current_base_row.push(Axis(0), section_row.slice(s![136usize])).unwrap();
                section_row[137usize] = (((current_base_row[262usize])
                    * (current_base_row[102usize])) * (current_base_row[102usize]))
                    * (current_base_row[102usize]);
                current_base_row.push(Axis(0), section_row.slice(s![137usize])).unwrap();
                section_row[138usize] = (((current_base_row[268usize])
                    * (current_base_row[103usize])) * (current_base_row[103usize]))
                    * (current_base_row[103usize]);
                current_base_row.push(Axis(0), section_row.slice(s![138usize])).unwrap();
                section_row[139usize] = (((current_base_row[269usize])
                    * (current_base_row[104usize])) * (current_base_row[104usize]))
                    * (current_base_row[104usize]);
                current_base_row.push(Axis(0), section_row.slice(s![139usize])).unwrap();
                section_row[140usize] = (((current_base_row[270usize])
                    * (current_base_row[105usize])) * (current_base_row[105usize]))
                    * (current_base_row[105usize]);
                current_base_row.push(Axis(0), section_row.slice(s![140usize])).unwrap();
                section_row[141usize] = (((current_base_row[272usize])
                    * (current_base_row[106usize])) * (current_base_row[106usize]))
                    * (current_base_row[106usize]);
                current_base_row.push(Axis(0), section_row.slice(s![141usize])).unwrap();
                section_row[142usize] = (((current_base_row[273usize])
                    * (current_base_row[107usize])) * (current_base_row[107usize]))
                    * (current_base_row[107usize]);
                current_base_row.push(Axis(0), section_row.slice(s![142usize])).unwrap();
                section_row[143usize] = (((current_base_row[274usize])
                    * (current_base_row[108usize])) * (current_base_row[108usize]))
                    * (current_base_row[108usize]);
                current_base_row.push(Axis(0), section_row.slice(s![143usize])).unwrap();
                section_row[144usize] = ((next_base_row[139usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                    * ((current_base_row[298usize])
                        * ((next_base_row[142usize])
                            + (BFieldElement::from_raw_u64(18446743949155500061u64))));
                current_base_row.push(Axis(0), section_row.slice(s![144usize])).unwrap();
                section_row[145usize] = (current_base_row[299usize])
                    * ((next_base_row[142usize])
                        + (BFieldElement::from_raw_u64(18446744009285042191u64)));
                current_base_row.push(Axis(0), section_row.slice(s![145usize])).unwrap();
                section_row[146usize] = (current_base_row[300usize])
                    * ((next_base_row[64usize])
                        + (BFieldElement::from_raw_u64(18446744047939747846u64)));
                current_base_row.push(Axis(0), section_row.slice(s![146usize])).unwrap();
                section_row[147usize] = ((current_base_row[313usize])
                    * (next_base_row[147usize]))
                    * ((next_base_row[147usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![147usize])).unwrap();
                section_row[148usize] = (current_base_row[39usize])
                    * ((current_base_row[23usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[22usize])));
                current_base_row.push(Axis(0), section_row.slice(s![148usize])).unwrap();
                section_row[149usize] = (((next_base_row[62usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                    * ((next_base_row[62usize])
                        + (BFieldElement::from_raw_u64(18446744060824649731u64))))
                    * (next_base_row[62usize]);
                current_base_row.push(Axis(0), section_row.slice(s![149usize])).unwrap();
                section_row[150usize] = ((((next_base_row[12usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                    * (next_base_row[13usize]))
                    * ((next_base_row[14usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                    * ((next_base_row[15usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![150usize])).unwrap();
                section_row[151usize] = (current_base_row[23usize])
                    * (next_base_row[23usize]);
                current_base_row.push(Axis(0), section_row.slice(s![151usize])).unwrap();
                section_row[152usize] = (current_base_row[22usize])
                    * (current_base_row[25usize]);
                current_base_row.push(Axis(0), section_row.slice(s![152usize])).unwrap();
                section_row[153usize] = (current_base_row[24usize])
                    * (current_base_row[27usize]);
                current_base_row.push(Axis(0), section_row.slice(s![153usize])).unwrap();
                section_row[154usize] = (current_base_row[24usize])
                    * (next_base_row[24usize]);
                current_base_row.push(Axis(0), section_row.slice(s![154usize])).unwrap();
                section_row[155usize] = (current_base_row[314usize])
                    * ((next_base_row[142usize])
                        + (BFieldElement::from_raw_u64(18446744017874976781u64)));
                current_base_row.push(Axis(0), section_row.slice(s![155usize])).unwrap();
                section_row[156usize] = ((((next_base_row[12usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                    * ((next_base_row[13usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                    * ((next_base_row[14usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                    * ((next_base_row[15usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![156usize])).unwrap();
                section_row[157usize] = ((((current_base_row[10usize])
                    + (BFieldElement::from_raw_u64(18446743897615892521u64)))
                    * ((current_base_row[10usize])
                        + (BFieldElement::from_raw_u64(18446743923385696291u64))))
                    * ((current_base_row[10usize])
                        + (BFieldElement::from_raw_u64(18446743863256154161u64))))
                    * ((current_base_row[10usize])
                        + (BFieldElement::from_raw_u64(18446743828896415801u64)));
                current_base_row.push(Axis(0), section_row.slice(s![157usize])).unwrap();
                section_row[158usize] = ((next_base_row[139usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                    * (((current_base_row[314usize])
                        * ((next_base_row[142usize])
                            + (BFieldElement::from_raw_u64(18446743940565565471u64))))
                        * ((next_base_row[142usize])
                            + (BFieldElement::from_raw_u64(18446743949155500061u64))));
                current_base_row.push(Axis(0), section_row.slice(s![158usize])).unwrap();
                section_row[159usize] = (current_base_row[39usize])
                    * ((current_base_row[39usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![159usize])).unwrap();
                section_row[160usize] = (current_base_row[183usize])
                    * ((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[39usize])));
                current_base_row.push(Axis(0), section_row.slice(s![160usize])).unwrap();
                section_row[161usize] = (current_base_row[263usize])
                    * ((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[39usize])));
                current_base_row.push(Axis(0), section_row.slice(s![161usize])).unwrap();
                section_row[162usize] = (current_base_row[263usize])
                    * (current_base_row[39usize]);
                current_base_row.push(Axis(0), section_row.slice(s![162usize])).unwrap();
                section_row[163usize] = (current_base_row[264usize])
                    * ((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[39usize])));
                current_base_row.push(Axis(0), section_row.slice(s![163usize])).unwrap();
                section_row[164usize] = (current_base_row[264usize])
                    * (current_base_row[39usize]);
                current_base_row.push(Axis(0), section_row.slice(s![164usize])).unwrap();
                section_row[165usize] = (current_base_row[265usize])
                    * ((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[39usize])));
                current_base_row.push(Axis(0), section_row.slice(s![165usize])).unwrap();
                section_row[166usize] = (current_base_row[265usize])
                    * (current_base_row[39usize]);
                current_base_row.push(Axis(0), section_row.slice(s![166usize])).unwrap();
                section_row[167usize] = (current_base_row[266usize])
                    * ((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[39usize])));
                current_base_row.push(Axis(0), section_row.slice(s![167usize])).unwrap();
                section_row[168usize] = (current_base_row[266usize])
                    * (current_base_row[39usize]);
                current_base_row.push(Axis(0), section_row.slice(s![168usize])).unwrap();
                section_row[169usize] = (current_base_row[267usize])
                    * ((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[39usize])));
                current_base_row.push(Axis(0), section_row.slice(s![169usize])).unwrap();
                section_row[170usize] = (current_base_row[267usize])
                    * (current_base_row[39usize]);
                current_base_row.push(Axis(0), section_row.slice(s![170usize])).unwrap();
                section_row[171usize] = (current_base_row[39usize])
                    * (current_base_row[42usize]);
                current_base_row.push(Axis(0), section_row.slice(s![171usize])).unwrap();
                section_row[172usize] = (((current_base_row[319usize])
                    * (next_base_row[16usize]))
                    * ((next_base_row[17usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                    * ((next_base_row[18usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)));
                current_base_row.push(Axis(0), section_row.slice(s![172usize])).unwrap();
                section_row[173usize] = (current_base_row[315usize])
                    * ((((next_base_row[62usize])
                        + (BFieldElement::from_raw_u64(18446744060824649731u64)))
                        * ((next_base_row[62usize])
                            + (BFieldElement::from_raw_u64(18446744056529682436u64))))
                        * (next_base_row[62usize]));
                current_base_row.push(Axis(0), section_row.slice(s![173usize])).unwrap();
                section_row[174usize] = (((current_base_row[62usize])
                    + (BFieldElement::from_raw_u64(18446744060824649731u64)))
                    * ((current_base_row[62usize])
                        + (BFieldElement::from_raw_u64(18446744056529682436u64))))
                    * (current_base_row[62usize]);
                current_base_row.push(Axis(0), section_row.slice(s![174usize])).unwrap();
                section_row[175usize] = (current_base_row[238usize])
                    * ((next_base_row[22usize])
                        * (((current_base_row[39usize])
                            * ((next_base_row[23usize])
                                + (BFieldElement::from_raw_u64(4294967296u64))))
                            + (BFieldElement::from_raw_u64(18446744065119617026u64))));
                current_base_row.push(Axis(0), section_row.slice(s![175usize])).unwrap();
                section_row[176usize] = (current_base_row[214usize])
                    * ((((((next_base_row[9usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (current_base_row[9usize])))
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                        * (current_base_row[22usize]))
                        + (((((next_base_row[9usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (current_base_row[9usize])))
                            + (BFieldElement::from_raw_u64(18446744060824649731u64)))
                            * ((current_base_row[276usize])
                                + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                            * ((current_base_row[40usize])
                                + (BFieldElement::from_raw_u64(18446744065119617026u64)))))
                        + (((((next_base_row[9usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (current_base_row[9usize])))
                            + (BFieldElement::from_raw_u64(18446744056529682436u64)))
                            * ((current_base_row[276usize])
                                + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                            * (current_base_row[40usize])));
                current_base_row.push(Axis(0), section_row.slice(s![176usize])).unwrap();
                section_row[177usize] = (current_base_row[214usize])
                    * (((current_base_row[252usize])
                        * ((current_base_row[41usize])
                            + (BFieldElement::from_raw_u64(18446744060824649731u64))))
                        * ((current_base_row[41usize])
                            + (BFieldElement::from_raw_u64(18446744056529682436u64))));
                current_base_row.push(Axis(0), section_row.slice(s![177usize])).unwrap();
                section_row[178usize] = (current_base_row[214usize])
                    * (((current_base_row[261usize])
                        * ((current_base_row[42usize])
                            + (BFieldElement::from_raw_u64(18446744060824649731u64))))
                        * ((current_base_row[42usize])
                            + (BFieldElement::from_raw_u64(18446744056529682436u64))));
                current_base_row.push(Axis(0), section_row.slice(s![178usize])).unwrap();
                section_row[179usize] = (current_base_row[214usize])
                    * (((current_base_row[283usize])
                        * ((current_base_row[44usize])
                            + (BFieldElement::from_raw_u64(18446744060824649731u64))))
                        * ((current_base_row[44usize])
                            + (BFieldElement::from_raw_u64(18446744056529682436u64))));
                current_base_row.push(Axis(0), section_row.slice(s![179usize])).unwrap();
                section_row[180usize] = (((current_base_row[325usize])
                    * (next_base_row[16usize]))
                    * ((next_base_row[17usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                    * (next_base_row[18usize]);
                current_base_row.push(Axis(0), section_row.slice(s![180usize])).unwrap();
                section_row[181usize] = (((current_base_row[64usize])
                    * ((current_base_row[64usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64))))
                    * ((current_base_row[64usize])
                        + (BFieldElement::from_raw_u64(18446744060824649731u64))))
                    * ((current_base_row[64usize])
                        + (BFieldElement::from_raw_u64(18446744056529682436u64)));
                current_base_row.push(Axis(0), section_row.slice(s![181usize])).unwrap();
                section_row[182usize] = ((((current_base_row[62usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                    * ((current_base_row[62usize])
                        + (BFieldElement::from_raw_u64(18446744056529682436u64))))
                    * (current_base_row[62usize]))
                    * ((next_base_row[62usize])
                        + (BFieldElement::from_raw_u64(18446744060824649731u64)));
                current_base_row.push(Axis(0), section_row.slice(s![182usize])).unwrap();
                section_row[183usize] = (((current_base_row[299usize])
                    * ((next_base_row[142usize])
                        + (BFieldElement::from_raw_u64(18446744017874976781u64))))
                    * ((next_base_row[142usize])
                        + (BFieldElement::from_raw_u64(18446743940565565471u64))))
                    * ((next_base_row[142usize])
                        + (BFieldElement::from_raw_u64(18446743949155500061u64)));
                current_base_row.push(Axis(0), section_row.slice(s![183usize])).unwrap();
                section_row[184usize] = (current_base_row[316usize])
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
                current_base_row.push(Axis(0), section_row.slice(s![184usize])).unwrap();
                section_row[185usize] = ((next_base_row[139usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                    * ((current_base_row[324usize])
                        * ((next_base_row[142usize])
                            + (BFieldElement::from_raw_u64(18446743949155500061u64))));
                current_base_row.push(Axis(0), section_row.slice(s![185usize])).unwrap();
                section_row[186usize] = (current_base_row[343usize])
                    * ((((next_base_row[62usize])
                        + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                        * ((next_base_row[62usize])
                            + (BFieldElement::from_raw_u64(18446744056529682436u64))))
                        * (next_base_row[62usize]));
                current_base_row.push(Axis(0), section_row.slice(s![186usize])).unwrap();
                section_row[187usize] = ((((current_base_row[62usize])
                    + (BFieldElement::from_raw_u64(18446744065119617026u64)))
                    * ((current_base_row[62usize])
                        + (BFieldElement::from_raw_u64(18446744060824649731u64))))
                    * (current_base_row[62usize]))
                    * ((next_base_row[62usize])
                        + (BFieldElement::from_raw_u64(18446744056529682436u64)));
                current_base_row.push(Axis(0), section_row.slice(s![187usize])).unwrap();
                section_row[188usize] = (((current_base_row[315usize])
                    * ((next_base_row[62usize])
                        + (BFieldElement::from_raw_u64(18446744056529682436u64))))
                    * (next_base_row[62usize]))
                    * ((next_base_row[63usize])
                        + (BFieldElement::from_raw_u64(18446743897615892521u64)));
                current_base_row.push(Axis(0), section_row.slice(s![188usize])).unwrap();
                section_row[189usize] = (current_base_row[315usize])
                    * ((((next_base_row[63usize])
                        + (BFieldElement::from_raw_u64(18446743992105173011u64)))
                        * ((next_base_row[63usize])
                            + (BFieldElement::from_raw_u64(18446743897615892521u64))))
                        * ((next_base_row[63usize])
                            + (BFieldElement::from_raw_u64(18446743923385696291u64))));
                current_base_row.push(Axis(0), section_row.slice(s![189usize])).unwrap();
                section_row[190usize] = ((current_base_row[327usize])
                    * ((BFieldElement::from_raw_u64(4294967295u64))
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((next_base_row[143usize]) * (next_base_row[144usize])))))
                    * (current_base_row[143usize]);
                current_base_row.push(Axis(0), section_row.slice(s![190usize])).unwrap();
                section_row[191usize] = ((next_base_row[147usize])
                    * (next_base_row[147usize])) * (current_base_row[143usize]);
                current_base_row.push(Axis(0), section_row.slice(s![191usize])).unwrap();
            });
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
        let num_rows = master_base_table.nrows();
        let (original_part, mut current_section) = master_ext_table
            .multi_slice_mut((s![.., 0..50usize], s![.., 50usize..50usize + 36usize]));
        let row_indices = Array1::from_vec((0..num_rows - 1).collect::<Vec<_>>());
        Zip::from(current_section.slice_mut(s![0..num_rows - 1, ..]).rows_mut())
            .and(row_indices.view())
            .par_for_each(|mut section_row, &current_row_index| {
                let next_row_index = current_row_index + 1;
                let current_base_row = master_base_table.row(current_row_index);
                let next_base_row = master_base_table.row(next_row_index);
                let mut current_ext_row = original_part
                    .row(current_row_index)
                    .to_owned();
                let next_ext_row = original_part.row(next_row_index);
                section_row[0usize] = ((challenges[7usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((((challenges[16usize]) * (current_base_row[7usize]))
                            + ((challenges[17usize]) * (current_base_row[13usize])))
                            + ((challenges[18usize]) * (next_base_row[38usize])))
                            + ((challenges[19usize]) * (next_base_row[37usize])))))
                    * ((challenges[7usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((challenges[16usize]) * (current_base_row[7usize]))
                                + ((challenges[17usize]) * (current_base_row[13usize])))
                                + ((challenges[18usize])
                                    * ((next_base_row[38usize])
                                        + (BFieldElement::from_raw_u64(4294967295u64)))))
                                + ((challenges[19usize]) * (next_base_row[36usize])))));
                current_ext_row.push(Axis(0), section_row.slice(s![0usize])).unwrap();
                section_row[1usize] = ((challenges[7usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((((challenges[16usize]) * (current_base_row[7usize]))
                            + ((challenges[17usize]) * (current_base_row[13usize])))
                            + ((challenges[18usize]) * (current_base_row[38usize])))
                            + ((challenges[19usize]) * (current_base_row[37usize])))))
                    * ((challenges[7usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((challenges[16usize]) * (current_base_row[7usize]))
                                + ((challenges[17usize]) * (current_base_row[13usize])))
                                + ((challenges[18usize])
                                    * ((current_base_row[38usize])
                                        + (BFieldElement::from_raw_u64(4294967295u64)))))
                                + ((challenges[19usize]) * (current_base_row[36usize])))));
                current_ext_row.push(Axis(0), section_row.slice(s![1usize])).unwrap();
                section_row[2usize] = (current_ext_row[50usize])
                    * ((challenges[7usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((challenges[16usize]) * (current_base_row[7usize]))
                                + ((challenges[17usize]) * (current_base_row[13usize])))
                                + ((challenges[18usize])
                                    * ((next_base_row[38usize])
                                        + (BFieldElement::from_raw_u64(8589934590u64)))))
                                + ((challenges[19usize]) * (next_base_row[35usize])))));
                current_ext_row.push(Axis(0), section_row.slice(s![2usize])).unwrap();
                section_row[3usize] = (current_ext_row[51usize])
                    * ((challenges[7usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((challenges[16usize]) * (current_base_row[7usize]))
                                + ((challenges[17usize]) * (current_base_row[13usize])))
                                + ((challenges[18usize])
                                    * ((current_base_row[38usize])
                                        + (BFieldElement::from_raw_u64(8589934590u64)))))
                                + ((challenges[19usize]) * (current_base_row[35usize])))));
                current_ext_row.push(Axis(0), section_row.slice(s![3usize])).unwrap();
                section_row[4usize] = (current_ext_row[52usize])
                    * ((challenges[7usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((challenges[16usize]) * (current_base_row[7usize]))
                                + ((challenges[17usize]) * (current_base_row[13usize])))
                                + ((challenges[18usize])
                                    * ((next_base_row[38usize])
                                        + (BFieldElement::from_raw_u64(12884901885u64)))))
                                + ((challenges[19usize]) * (next_base_row[34usize])))));
                current_ext_row.push(Axis(0), section_row.slice(s![4usize])).unwrap();
                section_row[5usize] = (current_ext_row[53usize])
                    * ((challenges[7usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((challenges[16usize]) * (current_base_row[7usize]))
                                + ((challenges[17usize]) * (current_base_row[13usize])))
                                + ((challenges[18usize])
                                    * ((current_base_row[38usize])
                                        + (BFieldElement::from_raw_u64(12884901885u64)))))
                                + ((challenges[19usize]) * (current_base_row[34usize])))));
                current_ext_row.push(Axis(0), section_row.slice(s![5usize])).unwrap();
                section_row[6usize] = (current_ext_row[54usize])
                    * ((challenges[7usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((challenges[16usize]) * (current_base_row[7usize]))
                                + ((challenges[17usize]) * (current_base_row[13usize])))
                                + ((challenges[18usize])
                                    * ((next_base_row[38usize])
                                        + (BFieldElement::from_raw_u64(17179869180u64)))))
                                + ((challenges[19usize]) * (next_base_row[33usize])))));
                current_ext_row.push(Axis(0), section_row.slice(s![6usize])).unwrap();
                section_row[7usize] = ((challenges[8usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((((current_base_row[7usize]) * (challenges[20usize]))
                            + (challenges[23usize]))
                            + (((next_base_row[22usize])
                                + (BFieldElement::from_raw_u64(4294967295u64)))
                                * (challenges[21usize])))
                            + ((next_base_row[23usize]) * (challenges[22usize])))))
                    * ((challenges[8usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((current_base_row[7usize]) * (challenges[20usize]))
                                + (challenges[23usize]))
                                + (((next_base_row[22usize])
                                    + (BFieldElement::from_raw_u64(8589934590u64)))
                                    * (challenges[21usize])))
                                + ((next_base_row[24usize]) * (challenges[22usize])))));
                current_ext_row.push(Axis(0), section_row.slice(s![7usize])).unwrap();
                section_row[8usize] = ((challenges[8usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * ((((current_base_row[7usize]) * (challenges[20usize]))
                            + ((current_base_row[22usize]) * (challenges[21usize])))
                            + ((current_base_row[23usize]) * (challenges[22usize])))))
                    * ((challenges[8usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((((current_base_row[7usize]) * (challenges[20usize]))
                                + (((current_base_row[22usize])
                                    + (BFieldElement::from_raw_u64(4294967295u64)))
                                    * (challenges[21usize])))
                                + ((current_base_row[24usize]) * (challenges[22usize])))));
                current_ext_row.push(Axis(0), section_row.slice(s![8usize])).unwrap();
                section_row[9usize] = (current_ext_row[6usize])
                    * (current_ext_row[56usize]);
                current_ext_row.push(Axis(0), section_row.slice(s![9usize])).unwrap();
                section_row[10usize] = (current_ext_row[55usize])
                    * ((challenges[7usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((challenges[16usize]) * (current_base_row[7usize]))
                                + ((challenges[17usize]) * (current_base_row[13usize])))
                                + ((challenges[18usize])
                                    * ((current_base_row[38usize])
                                        + (BFieldElement::from_raw_u64(17179869180u64)))))
                                + ((challenges[19usize]) * (current_base_row[33usize])))));
                current_ext_row.push(Axis(0), section_row.slice(s![10usize])).unwrap();
                section_row[11usize] = (current_ext_row[57usize])
                    * ((challenges[8usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((current_base_row[7usize]) * (challenges[20usize]))
                                + (challenges[23usize]))
                                + (((next_base_row[22usize])
                                    + (BFieldElement::from_raw_u64(12884901885u64)))
                                    * (challenges[21usize])))
                                + ((next_base_row[25usize]) * (challenges[22usize])))));
                current_ext_row.push(Axis(0), section_row.slice(s![11usize])).unwrap();
                section_row[12usize] = (current_ext_row[58usize])
                    * ((challenges[8usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((((current_base_row[7usize]) * (challenges[20usize]))
                                + (((current_base_row[22usize])
                                    + (BFieldElement::from_raw_u64(8589934590u64)))
                                    * (challenges[21usize])))
                                + ((current_base_row[25usize]) * (challenges[22usize])))));
                current_ext_row.push(Axis(0), section_row.slice(s![12usize])).unwrap();
                section_row[13usize] = (current_ext_row[61usize])
                    * ((challenges[8usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((current_base_row[7usize]) * (challenges[20usize]))
                                + (challenges[23usize]))
                                + (((next_base_row[22usize])
                                    + (BFieldElement::from_raw_u64(17179869180u64)))
                                    * (challenges[21usize])))
                                + ((next_base_row[26usize]) * (challenges[22usize])))));
                current_ext_row.push(Axis(0), section_row.slice(s![13usize])).unwrap();
                section_row[14usize] = (current_ext_row[62usize])
                    * ((challenges[8usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((((current_base_row[7usize]) * (challenges[20usize]))
                                + (((current_base_row[22usize])
                                    + (BFieldElement::from_raw_u64(12884901885u64)))
                                    * (challenges[21usize])))
                                + ((current_base_row[26usize]) * (challenges[22usize])))));
                current_ext_row.push(Axis(0), section_row.slice(s![14usize])).unwrap();
                section_row[15usize] = (current_base_row[190usize])
                    * ((next_ext_row[7usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((current_ext_row[7usize])
                                * ((current_ext_row[63usize])
                                    * ((challenges[8usize])
                                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                            * (((((current_base_row[7usize]) * (challenges[20usize]))
                                                + (challenges[23usize]))
                                                + (((next_base_row[22usize])
                                                    + (BFieldElement::from_raw_u64(21474836475u64)))
                                                    * (challenges[21usize])))
                                                + ((next_base_row[27usize]) * (challenges[22usize])))))))));
                current_ext_row.push(Axis(0), section_row.slice(s![15usize])).unwrap();
                section_row[16usize] = (((current_ext_row[56usize])
                    * ((challenges[7usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((challenges[16usize]) * (current_base_row[7usize]))
                                + ((challenges[17usize]) * (current_base_row[13usize])))
                                + ((challenges[18usize])
                                    * ((next_base_row[38usize])
                                        + (BFieldElement::from_raw_u64(21474836475u64)))))
                                + ((challenges[19usize]) * (next_base_row[32usize]))))))
                    * ((challenges[7usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((challenges[16usize]) * (current_base_row[7usize]))
                                + ((challenges[17usize]) * (current_base_row[13usize])))
                                + ((challenges[18usize])
                                    * ((next_base_row[38usize])
                                        + (BFieldElement::from_raw_u64(25769803770u64)))))
                                + ((challenges[19usize]) * (next_base_row[31usize]))))))
                    * ((challenges[7usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((challenges[16usize]) * (current_base_row[7usize]))
                                + ((challenges[17usize]) * (current_base_row[13usize])))
                                + ((challenges[18usize])
                                    * ((next_base_row[38usize])
                                        + (BFieldElement::from_raw_u64(30064771065u64)))))
                                + ((challenges[19usize]) * (next_base_row[30usize])))));
                current_ext_row.push(Axis(0), section_row.slice(s![16usize])).unwrap();
                section_row[17usize] = (current_base_row[190usize])
                    * ((next_ext_row[7usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((current_ext_row[7usize])
                                * ((current_ext_row[64usize])
                                    * ((challenges[8usize])
                                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                            * ((((current_base_row[7usize]) * (challenges[20usize]))
                                                + (((current_base_row[22usize])
                                                    + (BFieldElement::from_raw_u64(17179869180u64)))
                                                    * (challenges[21usize])))
                                                + ((current_base_row[27usize])
                                                    * (challenges[22usize])))))))));
                current_ext_row.push(Axis(0), section_row.slice(s![17usize])).unwrap();
                section_row[18usize] = (((current_ext_row[60usize])
                    * ((challenges[7usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((challenges[16usize]) * (current_base_row[7usize]))
                                + ((challenges[17usize]) * (current_base_row[13usize])))
                                + ((challenges[18usize])
                                    * ((current_base_row[38usize])
                                        + (BFieldElement::from_raw_u64(21474836475u64)))))
                                + ((challenges[19usize]) * (current_base_row[32usize]))))))
                    * ((challenges[7usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((challenges[16usize]) * (current_base_row[7usize]))
                                + ((challenges[17usize]) * (current_base_row[13usize])))
                                + ((challenges[18usize])
                                    * ((current_base_row[38usize])
                                        + (BFieldElement::from_raw_u64(25769803770u64)))))
                                + ((challenges[19usize]) * (current_base_row[31usize]))))))
                    * ((challenges[7usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((challenges[16usize]) * (current_base_row[7usize]))
                                + ((challenges[17usize]) * (current_base_row[13usize])))
                                + ((challenges[18usize])
                                    * ((current_base_row[38usize])
                                        + (BFieldElement::from_raw_u64(30064771065u64)))))
                                + ((challenges[19usize]) * (current_base_row[30usize])))));
                current_ext_row.push(Axis(0), section_row.slice(s![18usize])).unwrap();
                section_row[19usize] = (current_ext_row[6usize])
                    * (((current_ext_row[66usize])
                        * ((challenges[7usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((((challenges[16usize]) * (current_base_row[7usize]))
                                    + ((challenges[17usize]) * (current_base_row[13usize])))
                                    + ((challenges[18usize])
                                        * ((next_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(34359738360u64)))))
                                    + ((challenges[19usize]) * (next_base_row[29usize]))))))
                        * ((challenges[7usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((((challenges[16usize]) * (current_base_row[7usize]))
                                    + ((challenges[17usize]) * (current_base_row[13usize])))
                                    + ((challenges[18usize])
                                        * ((next_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(38654705655u64)))))
                                    + ((challenges[19usize]) * (next_base_row[28usize]))))));
                current_ext_row.push(Axis(0), section_row.slice(s![19usize])).unwrap();
                section_row[20usize] = (current_ext_row[6usize])
                    * (((current_ext_row[68usize])
                        * ((challenges[7usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((((challenges[16usize]) * (current_base_row[7usize]))
                                    + ((challenges[17usize]) * (current_base_row[13usize])))
                                    + ((challenges[18usize])
                                        * ((current_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(34359738360u64)))))
                                    + ((challenges[19usize]) * (current_base_row[29usize]))))))
                        * ((challenges[7usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((((challenges[16usize]) * (current_base_row[7usize]))
                                    + ((challenges[17usize]) * (current_base_row[13usize])))
                                    + ((challenges[18usize])
                                        * ((current_base_row[38usize])
                                            + (BFieldElement::from_raw_u64(38654705655u64)))))
                                    + ((challenges[19usize]) * (current_base_row[28usize]))))));
                current_ext_row.push(Axis(0), section_row.slice(s![20usize])).unwrap();
                section_row[21usize] = ((((challenges[8usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((((current_base_row[7usize]) * (challenges[20usize]))
                            + (challenges[23usize]))
                            + ((current_base_row[22usize]) * (challenges[21usize])))
                            + ((current_base_row[39usize]) * (challenges[22usize])))))
                    * ((challenges[8usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((current_base_row[7usize]) * (challenges[20usize]))
                                + (challenges[23usize]))
                                + (((current_base_row[22usize])
                                    + (BFieldElement::from_raw_u64(4294967295u64)))
                                    * (challenges[21usize])))
                                + ((current_base_row[40usize]) * (challenges[22usize]))))))
                    * ((challenges[8usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((current_base_row[7usize]) * (challenges[20usize]))
                                + (challenges[23usize]))
                                + (((current_base_row[22usize])
                                    + (BFieldElement::from_raw_u64(8589934590u64)))
                                    * (challenges[21usize])))
                                + ((current_base_row[41usize]) * (challenges[22usize]))))))
                    * ((challenges[8usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((current_base_row[7usize]) * (challenges[20usize]))
                                + (challenges[23usize]))
                                + ((current_base_row[23usize]) * (challenges[21usize])))
                                + ((current_base_row[42usize]) * (challenges[22usize])))));
                current_ext_row.push(Axis(0), section_row.slice(s![21usize])).unwrap();
                section_row[22usize] = ((((challenges[8usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (((((current_base_row[7usize]) * (challenges[20usize]))
                            + (challenges[23usize]))
                            + ((current_base_row[22usize]) * (challenges[21usize])))
                            + ((current_base_row[39usize]) * (challenges[22usize])))))
                    * ((challenges[8usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((current_base_row[7usize]) * (challenges[20usize]))
                                + (challenges[23usize]))
                                + ((current_base_row[23usize]) * (challenges[21usize])))
                                + ((current_base_row[40usize]) * (challenges[22usize]))))))
                    * ((challenges[8usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((current_base_row[7usize]) * (challenges[20usize]))
                                + (challenges[23usize]))
                                + (((current_base_row[23usize])
                                    + (BFieldElement::from_raw_u64(4294967295u64)))
                                    * (challenges[21usize])))
                                + ((current_base_row[41usize]) * (challenges[22usize]))))))
                    * ((challenges[8usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * (((((current_base_row[7usize]) * (challenges[20usize]))
                                + (challenges[23usize]))
                                + (((current_base_row[23usize])
                                    + (BFieldElement::from_raw_u64(8589934590u64)))
                                    * (challenges[21usize])))
                                + ((current_base_row[42usize]) * (challenges[22usize])))));
                current_ext_row.push(Axis(0), section_row.slice(s![22usize])).unwrap();
                section_row[23usize] = (current_base_row[185usize])
                    * ((next_ext_row[6usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((current_ext_row[6usize])
                                * ((challenges[7usize])
                                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                        * (((((challenges[16usize]) * (current_base_row[7usize]))
                                            + ((challenges[17usize]) * (current_base_row[13usize])))
                                            + ((challenges[18usize]) * (next_base_row[38usize])))
                                            + ((challenges[19usize]) * (next_base_row[37usize]))))))));
                current_ext_row.push(Axis(0), section_row.slice(s![23usize])).unwrap();
                section_row[24usize] = (current_base_row[186usize])
                    * ((next_ext_row[6usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((current_ext_row[6usize]) * (current_ext_row[50usize]))));
                current_ext_row.push(Axis(0), section_row.slice(s![24usize])).unwrap();
                section_row[25usize] = (current_base_row[187usize])
                    * ((next_ext_row[6usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((current_ext_row[6usize]) * (current_ext_row[52usize]))));
                current_ext_row.push(Axis(0), section_row.slice(s![25usize])).unwrap();
                section_row[26usize] = (current_base_row[189usize])
                    * ((next_ext_row[6usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((current_ext_row[6usize]) * (current_ext_row[54usize]))));
                current_ext_row.push(Axis(0), section_row.slice(s![26usize])).unwrap();
                section_row[27usize] = (current_base_row[185usize])
                    * ((next_ext_row[6usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((current_ext_row[6usize])
                                * ((challenges[7usize])
                                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                        * (((((challenges[16usize]) * (current_base_row[7usize]))
                                            + ((challenges[17usize]) * (current_base_row[13usize])))
                                            + ((challenges[18usize]) * (current_base_row[38usize])))
                                            + ((challenges[19usize])
                                                * (current_base_row[37usize]))))))));
                current_ext_row.push(Axis(0), section_row.slice(s![27usize])).unwrap();
                section_row[28usize] = (current_base_row[186usize])
                    * ((next_ext_row[6usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((current_ext_row[6usize]) * (current_ext_row[51usize]))));
                current_ext_row.push(Axis(0), section_row.slice(s![28usize])).unwrap();
                section_row[29usize] = (current_base_row[187usize])
                    * ((next_ext_row[6usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((current_ext_row[6usize]) * (current_ext_row[53usize]))));
                current_ext_row.push(Axis(0), section_row.slice(s![29usize])).unwrap();
                section_row[30usize] = (current_base_row[189usize])
                    * ((next_ext_row[6usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((current_ext_row[6usize]) * (current_ext_row[55usize]))));
                current_ext_row.push(Axis(0), section_row.slice(s![30usize])).unwrap();
                section_row[31usize] = (current_base_row[190usize])
                    * ((next_ext_row[6usize])
                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                            * ((current_ext_row[6usize]) * (current_ext_row[60usize]))));
                current_ext_row.push(Axis(0), section_row.slice(s![31usize])).unwrap();
                section_row[32usize] = (current_ext_row[7usize])
                    * (((current_ext_row[71usize])
                        * ((challenges[8usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((((current_base_row[7usize]) * (challenges[20usize]))
                                    + (challenges[23usize]))
                                    + (((current_base_row[23usize])
                                        + (BFieldElement::from_raw_u64(4294967295u64)))
                                        * (challenges[21usize])))
                                    + ((current_base_row[43usize]) * (challenges[22usize]))))))
                        * ((challenges[8usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * (((((current_base_row[7usize]) * (challenges[20usize]))
                                    + (challenges[23usize]))
                                    + (((current_base_row[23usize])
                                        + (BFieldElement::from_raw_u64(8589934590u64)))
                                        * (challenges[21usize])))
                                    + ((current_base_row[44usize]) * (challenges[22usize]))))));
                current_ext_row.push(Axis(0), section_row.slice(s![32usize])).unwrap();
                section_row[33usize] = ((((next_ext_row[21usize])
                    + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                        * (current_ext_row[21usize])))
                    * ((challenges[11usize])
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
                current_ext_row.push(Axis(0), section_row.slice(s![33usize])).unwrap();
                section_row[34usize] = (current_base_row[219usize])
                    * ((((((current_base_row[185usize])
                        * ((next_ext_row[7usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((current_ext_row[7usize])
                                    * ((challenges[8usize])
                                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                            * (((((current_base_row[7usize]) * (challenges[20usize]))
                                                + (challenges[23usize]))
                                                + (((next_base_row[22usize])
                                                    + (BFieldElement::from_raw_u64(4294967295u64)))
                                                    * (challenges[21usize])))
                                                + ((next_base_row[23usize]) * (challenges[22usize])))))))))
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
                        + (current_ext_row[65usize]));
                current_ext_row.push(Axis(0), section_row.slice(s![34usize])).unwrap();
                section_row[35usize] = (current_base_row[226usize])
                    * ((((((current_base_row[185usize])
                        * ((next_ext_row[7usize])
                            + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                * ((current_ext_row[7usize])
                                    * ((challenges[8usize])
                                        + ((BFieldElement::from_raw_u64(18446744065119617026u64))
                                            * ((((current_base_row[7usize]) * (challenges[20usize]))
                                                + ((current_base_row[22usize]) * (challenges[21usize])))
                                                + ((current_base_row[23usize])
                                                    * (challenges[22usize])))))))))
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
                        + (current_ext_row[67usize]));
                current_ext_row.push(Axis(0), section_row.slice(s![35usize])).unwrap();
            });
    }
}
