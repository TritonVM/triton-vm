//! The degree lowering table contains the introduced variables that allow
//! lowering the degree of the AIR. See [`air::TARGET_DEGREE`]
//! for additional information.

include!(concat!(env!("OUT_DIR"), "/degree_lowering_table.rs"));
