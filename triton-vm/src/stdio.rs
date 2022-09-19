use byteorder::{BigEndian, ReadBytesExt};
use std::convert::TryInto;
use std::io::{Error, Write};
use std::io::{Stdin, Stdout};
use twenty_first::shared_math::b_field_element::BFieldElement;

pub trait InputStream {
    fn read_elem(&mut self) -> Result<BFieldElement, Error>;
}

pub trait OutputStream {
    fn write_elem(&mut self, elem: BFieldElement) -> Result<usize, Error>;
}

impl InputStream for Stdin {
    fn read_elem(&mut self) -> Result<BFieldElement, Error> {
        let maybe_elem = self.read_u64::<BigEndian>();
        match maybe_elem {
            Err(e) => {
                println!("Could not read from stdin! <o>");
                Err(e)
            }
            Ok(e) => Ok(BFieldElement::new(e)),
        }
    }
}

impl OutputStream for Stdout {
    fn write_elem(&mut self, elem: BFieldElement) -> Result<usize, Error> {
        let bytes = elem.value().to_be_bytes();
        self.write(&bytes)
    }
}

#[derive(Debug, PartialEq, Eq)]
pub struct VecStream {
    buffer: Vec<BFieldElement>,
    read_index: usize,
}

impl VecStream {
    // pub fn new_bytes(bytes: &[u8]) -> Self {
    //     VecStream {
    //         cursor: Cursor::new(bytes.to_vec()),
    //         read_index: 0,
    //         length: bytes.len(),
    //     }
    // }

    pub fn new(bfes: &[BFieldElement]) -> Self {
        VecStream {
            buffer: bfes.to_owned(),
            read_index: 0,
        }
    }

    pub fn new_from_bytes(bytes: &[u8]) -> Self {
        let mut bfes = vec![];
        for chunk in bytes.chunks(8) {
            if chunk.len() < 8 {
                panic!("Cannot create VecStream object from stream of bytes whose length is not a multiple of 8.");
            }
            bfes.push(BFieldElement::new(u64::from_be_bytes(
                chunk.try_into().unwrap(),
            )));
        }
        Self {
            buffer: bfes,
            read_index: 0,
        }
    }

    pub fn to_bytes_vec(&self) -> Vec<u8> {
        let mut vector = vec![];
        for t in self.buffer.iter() {
            vector.append(&mut t.value().to_be_bytes().to_vec());
        }
        vector
    }

    pub fn to_bword_vec(&self) -> Vec<BFieldElement> {
        self.buffer.clone()
    }
}

impl InputStream for VecStream {
    fn read_elem(&mut self) -> Result<BFieldElement, Error> {
        if self.read_index == self.buffer.len() {
            panic!("Error when reading BFieldElement from VecStream: read index {} exceeds buffer length {}.", self.read_index, self.buffer.len());
        }
        let e = self.buffer[self.read_index];
        self.read_index += 1;
        Ok(e)
    }
}

impl OutputStream for VecStream {
    fn write_elem(&mut self, elem: BFieldElement) -> Result<usize, Error> {
        self.buffer.push(elem.to_owned());
        Ok(1)
    }
}
