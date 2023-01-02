use std::collections::{HashMap, HashSet};

use nom::branch::alt;
use nom::bytes::complete::{tag, take_while, take_while1};
use nom::character::complete::digit1;
use nom::combinator::{eof, fail, opt};
use nom::error::{context, convert_error, ErrorKind, VerboseError, VerboseErrorKind};
use nom::multi::many0;
use nom::{Finish, IResult};

use twenty_first::shared_math::b_field_element::BFieldElement;

use crate::instruction::AnInstruction::{self, *};
use crate::instruction::DivinationHint::Quotient;
use crate::instruction::{token_str, LabelledInstruction};
use crate::ord_n::Ord16::{self, *};

#[derive(Debug, PartialEq)]
pub struct ParseError<'a> {
    pub input: &'a str,
    pub errors: VerboseError<&'a str>,
}

impl<'a> std::fmt::Display for ParseError<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", pretty_print_error(self.input, self.errors.clone()))
    }
}

/// Pretty-print a parse error
///
/// This function wraps `convert_error()`.
///
/// `VerboseError` accumulates each nested contexts in which an error occurs.
///
/// Every `fail()` is wrapped in a `context()`, so by skipping the root `ErrorKind::Fail`s
/// and `ErrorKind::Eof`s, manually triggered custom errors are only shown in the output
/// once with the `context()` message.
pub fn pretty_print_error(s: &str, mut e: VerboseError<&str>) -> String {
    let (_root_s, root_error) = e.errors[0].clone();
    if matches!(root_error, VerboseErrorKind::Nom(ErrorKind::Fail))
        || matches!(root_error, VerboseErrorKind::Nom(ErrorKind::Eof))
    {
        e.errors.remove(0);
    }
    convert_error(s, e)
}

/// Parse a program
pub fn parse(input: &str) -> Result<Vec<LabelledInstruction>, ParseError> {
    let instructions = match program(input).finish() {
        Ok((_s, instructions)) => Ok(instructions),
        Err(errors) => Err(ParseError { input, errors }),
    }?;

    scan_missing_duplicate_labels(input, &instructions)?;

    Ok(instructions)
}

fn scan_missing_duplicate_labels<'a>(
    input: &'a str,
    instructions: &[LabelledInstruction<'a>],
) -> Result<(), ParseError<'a>> {
    let mut seen: HashMap<&str, LabelledInstruction> = HashMap::default();
    let mut duplicates: HashSet<LabelledInstruction> = HashSet::default();
    let mut missings: HashSet<LabelledInstruction> = HashSet::default();

    // Find duplicate labels, including the first occurrence of each duplicate.
    for instruction in instructions.iter() {
        if let LabelledInstruction::Label(label, _token_s) = instruction {
            if let Some(first_label) = seen.get(label.as_str()) {
                duplicates.insert(first_label.to_owned());
                duplicates.insert(instruction.to_owned());
            } else {
                seen.insert(label.as_str(), instruction.to_owned());
            }
        }
    }

    // Find missing labels
    for instruction in instructions.iter() {
        if let LabelledInstruction::Instruction(Call(addr), _token_s) = instruction {
            if !seen.contains_key(addr.as_str()) {
                missings.insert(instruction.to_owned());
            }
        }
    }

    if duplicates.is_empty() && missings.is_empty() {
        return Ok(());
    }

    // Collect duplicate and missing error messages
    let mut errors: Vec<(&str, VerboseErrorKind)> =
        Vec::with_capacity(duplicates.len() + missings.len());

    for duplicate in duplicates {
        let error = (
            token_str(&duplicate),
            VerboseErrorKind::Context("duplicate label"),
        );
        errors.push(error);
    }

    for missing in missings {
        let error = (
            token_str(&missing),
            VerboseErrorKind::Context("missing label"),
        );
        errors.push(error);
    }

    let errors = VerboseError { errors };
    Err(ParseError { input, errors })
}

/// Auxiliary type alias: `IResult` defaults to `nom::error::Error` as concrete
/// error type, but we want `nom::error::VerboseError` as it allows `context()`.
type ParseResult<'input, Out> = IResult<&'input str, Out, VerboseError<&'input str>>;

fn program(s: &str) -> ParseResult<Vec<LabelledInstruction>> {
    let (s, _) = comment_or_whitespace(s)?;
    let (s, instructions) = many0(alt((label, labelled_instruction)))(s)?;
    let (s, _) = context("expecting label, instruction or eof", eof)(s)?;

    Ok((s, instructions))
}

fn labelled_instruction(s_instr: &str) -> ParseResult<LabelledInstruction> {
    let (s, instr) = an_instruction(s_instr)?;
    Ok((s, LabelledInstruction::Instruction(instr, s_instr)))
}

fn label(label_s: &str) -> ParseResult<LabelledInstruction> {
    let (s, addr) = label_addr(label_s)?;
    let (s, _) = token(":")(s)?;

    Ok((s, LabelledInstruction::Label(addr, label_s)))
}

fn an_instruction(s: &str) -> ParseResult<AnInstruction<String>> {
    // OpStack manipulation
    let pop = instruction("pop", Pop);
    let push = push_instruction();
    let divine = instruction("divine", Divine(None));
    let divine_quotient = instruction("divine_quotient", Divine(Some(Quotient)));
    let dup = dup_instruction();
    let swap = swap_instruction();

    let opstack_manipulation = alt((pop, push, dup, swap));

    // Control flow
    let nop = instruction("nop", Nop);
    let skiz = instruction("skiz", Skiz);
    let call = call_instruction();
    let return_ = instruction("return", Return);
    let recurse = instruction("recurse", Recurse);
    let assert_ = instruction("assert", Assert);
    let halt = instruction("halt", Halt);

    let control_flow = alt((nop, skiz, call, return_, recurse, halt));

    // Memory access
    let read_mem = instruction("read_mem", ReadMem);
    let write_mem = instruction("write_mem", WriteMem);

    let memory_access = alt((read_mem, write_mem));

    // Hashing-related instructions
    let hash = instruction("hash", Hash);
    let divine_sibling = instruction("divine_sibling", DivineSibling);
    let assert_vector = instruction("assert_vector", AssertVector);

    let hashing_related = alt((hash, divine_sibling));

    // Arithmetic on stack instructions
    let add = instruction("add", Add);
    let mul = instruction("mul", Mul);
    let invert = instruction("invert", Invert);
    let split = instruction("split", Split);
    let eq = instruction("eq", Eq);
    let lsb = instruction("lsb", Lsb);
    let xxadd = instruction("xxadd", XxAdd);
    let xxmul = instruction("xxmul", XxMul);
    let xinvert = instruction("xinvert", XInvert);
    let xbmul = instruction("xbmul", XbMul);

    let arithmetic_on_stack = alt((
        add, mul, invert, split, eq, lsb, xxadd, xxmul, xinvert, xbmul,
    ));

    // Pseudo-instructions
    // "neg" => vec![Push(BFieldElement::one().neg()), Mul],
    // "sub" => vec![Swap(ST1), Push(BFieldElement::one().neg()), Mul, Add],
    // "lte" => pseudo_instruction_lte(),
    // "lt" => pseudo_instruction_lt(),
    // "and" => pseudo_instruction_and(),
    // "xor" => pseudo_instruction_xor(),
    // "reverse" => pseudo_instruction_reverse(),
    // "div" => pseudo_instruction_div(),
    // "is_u32" => pseudo_instruction_is_u32(),
    // "split_assert" => pseudo_instruction_split_assert(),

    // Read/write
    let read_io = instruction("read_io", ReadIo);
    let write_io = instruction("write_io", WriteIo);

    let read_write = alt((read_io, write_io));

    // Because of common prefixes, the following parsers are sensitive to order:
    //
    // Successfully parsing "assert" before trying "assert_vector" can lead to
    // picking the wrong one. By trying them in the order of longest first, less
    // backtracking is necessary.
    let syntax_ambiguous = alt((assert_vector, assert_, divine_quotient, divine));

    alt((
        opstack_manipulation,
        control_flow,
        memory_access,
        hashing_related,
        arithmetic_on_stack,
        read_write,
        syntax_ambiguous,
    ))(s)
}

fn instruction<'a>(
    name: &'a str,
    instruction: AnInstruction<String>,
) -> impl Fn(&'a str) -> ParseResult<AnInstruction<String>> {
    move |s: &'a str| {
        let (s, _) = token(name)(s)?;
        Ok((s, instruction.clone()))
    }
}

fn push_instruction() -> impl Fn(&str) -> ParseResult<AnInstruction<String>> {
    move |s: &str| {
        let (s, _) = token("push")(s)?;
        let (s, elem) = field_element(s)?;
        Ok((s, Push(elem)))
    }
}

fn dup_instruction() -> impl Fn(&str) -> ParseResult<AnInstruction<String>> {
    move |s: &str| {
        let (s, _) = tag("dup")(s)?;
        let (s, stack_register) = stack_register(s)?;
        Ok((s, Dup(stack_register)))
    }
}

fn swap_instruction() -> impl Fn(&str) -> ParseResult<AnInstruction<String>> {
    move |s: &str| {
        let (s, _) = tag("swap")(s)?;
        let (s, stack_register) = stack_register(s)?;

        if stack_register == ST0 {
            return context("using swap0, which is not a meaningful instruction", fail)(s);
        }

        Ok((s, Swap(stack_register)))
    }
}

fn call_instruction() -> impl Fn(&str) -> ParseResult<AnInstruction<String>> {
    move |s: &str| {
        let (s, _) = token("call")(s)?;
        let (s, addr) = label_addr(s)?;
        let (s, _) = comment_or_whitespace(s)?;
        Ok((s, Call(addr)))
    }
}

fn field_element(s: &str) -> ParseResult<BFieldElement> {
    let (s, negative) = opt(tag("-"))(s)?;
    let (s, n) = digit1(s)?;
    let (s, _) = comment_or_whitespace(s)?;

    let mut n: i128 = match n.parse() {
        Ok(n) => n,
        Err(err) => {
            println!("{}", err);
            return context("out-of-bounds constant", fail)(s);
        }
    };
    if negative.is_some() {
        n *= -1;
    }
    let quotient = BFieldElement::QUOTIENT as i128;
    while n < 0 {
        n += quotient;
    }
    while n >= quotient {
        n -= quotient;
    }

    Ok((s, BFieldElement::new(n as u64)))
}

fn stack_register(s: &str) -> ParseResult<Ord16> {
    let (s, n) = digit1(s)?;
    let (s, _) = comment_or_whitespace(s)?;
    let stack_register = match n {
        "0" => ST0,
        "1" => ST1,
        "2" => ST2,
        "3" => ST3,
        "4" => ST4,
        "5" => ST5,
        "6" => ST6,
        "7" => ST7,
        "8" => ST8,
        "9" => ST9,
        "10" => ST10,
        "11" => ST11,
        "12" => ST12,
        "13" => ST13,
        "14" => ST14,
        "15" => ST15,
        _ => return context("using an out-of-bounds stack register (0-15 exist)", fail)(s),
    };

    Ok((s, stack_register))
}

fn label_addr(s: &str) -> ParseResult<String> {
    let (s, addr) = take_while1(is_label_char)(s)?;
    Ok((s, addr.to_string()))
}

fn comment_or_whitespace(s: &str) -> ParseResult<()> {
    let (s, _) = many0(alt((comment, whitespace)))(s)?;
    Ok((s, ()))
}

fn comment(s: &str) -> ParseResult<()> {
    let (s, _) = tag("//")(s)?;
    let (s, _) = take_while(|c| !is_linebreak(c))(s)?;
    Ok((s, ()))
}

fn whitespace(s: &str) -> ParseResult<()> {
    let (s, _) = take_while1(|c: char| c.is_whitespace())(s)?;
    Ok((s, ()))
}

fn is_linebreak(c: char) -> bool {
    c == '\r' || c == '\n'
}

fn is_label_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || c == '-'
}

fn token<'a>(token: &'a str) -> impl Fn(&'a str) -> ParseResult<()> {
    move |s: &'a str| {
        let (s, _) = tag(token)(s)?;
        let (s, _) = comment_or_whitespace(s)?;
        Ok((s, ()))
    }
}

#[cfg(test)]
mod parser_tests {
    use std::collections::HashSet;

    use itertools::Itertools;
    use rand::distributions::WeightedIndex;
    use rand::prelude::*;
    use rand::Rng;

    use crate::instruction;
    use crate::program::Program;

    use super::*;

    struct TestCase<'a> {
        input: &'a str,
        expected: Program,
        message: &'static str,
    }

    fn parse_program_prop(test_case: TestCase) {
        match parse(test_case.input) {
            Ok(actual) => assert_eq!(
                test_case.expected,
                Program::new(&actual),
                "{}",
                test_case.message
            ),
            Err(parse_err) => panic!("{}:\n{}", test_case.message, parse_err),
        }
    }

    fn whitespace_gen(max_size: usize) -> String {
        let mut rng = rand::thread_rng();
        let spaces = [" ", "\t", "\r\n", "\n", " // comment\n"];
        let weights = [5, 1, 1, 2, 1];
        assert_eq!(spaces.len(), weights.len(), "all generators have weights");
        let dist = WeightedIndex::new(&weights).expect("a weighted distribution of generators");
        let size = rng.gen_range(1..=std::cmp::max(1, max_size));
        (0..size).map(|_| spaces[dist.sample(&mut rng)]).collect()
    }

    fn label_gen(size: usize) -> String {
        let mut rng = rand::thread_rng();
        (0..size).map(|_| rng.gen_range('a'..='z')).collect()
    }

    fn new_label_gen(labels: &mut Vec<String>) -> String {
        let mut rng = rand::thread_rng();
        let count = labels.len() * 3 / 2;
        let index = rng.gen_range(0..=count);

        labels.get(index).cloned().unwrap_or_else(|| {
            let label_size = 4;
            let new_label = label_gen(label_size);
            labels.push(new_label.clone());
            new_label
        })
    }

    fn instruction_gen(labels: &mut Vec<String>) -> Vec<String> {
        let mut rng = rand::thread_rng();

        // not included: push, dup*, swap*, skiz, call
        let simple_instructions = [
            "pop",
            "divine",
            "divine_quotient",
            "nop",
            "return",
            "assert",
            "halt",
            "read_mem",
            "write_mem",
            "hash",
            "divine_sibling",
            "assert_vector",
            "add",
            "mul",
            "invert",
            "split",
            "eq",
            "lsb",
            "xxadd",
            "xxmul",
            "xinvert",
            "xbmul",
            "read_io",
            "write_io",
        ];

        // Test simple instructions, dup* and swap* less frequently.
        let generators = ["simple", "push", "dup", "swap", "skiz", "call"];
        let weights = [simple_instructions.len(), 2, 6, 6, 2, 10];
        assert_eq!(
            generators.len(),
            weights.len(),
            "all generators have weights"
        );
        let dist = WeightedIndex::new(&weights).expect("a weighted distribution of generators");

        match generators[dist.sample(&mut rng)] {
            "simple" => {
                let index: usize = rng.gen_range(0..simple_instructions.len());
                let instruction = simple_instructions[index];
                vec![instruction.to_string()]
            }

            "push" => {
                let max: i128 = BFieldElement::MAX as i128;
                let arg: i128 = rng.gen_range(-max..max);
                vec!["push".to_string(), format!("{}", arg)]
            }

            "dup" => {
                let arg: usize = rng.gen_range(0..15);
                vec![format!("dup{}", arg)]
            }

            "swap" => {
                let arg: usize = rng.gen_range(1..15);
                vec![format!("swap{}", arg)]
            }

            "skiz" => {
                let mut target: Vec<String> = instruction_gen(labels);
                target.insert(0, "skiz".to_string());
                target
            }

            "call" => {
                let some_label: String = new_label_gen(labels);
                vec!["call".to_string(), some_label]
            }

            unknown => panic!("Unknown generator, {}", unknown),
        }
    }

    #[allow(unstable_name_collisions)] // reason = "Switch to standard library intersperse_with() when it's ported"
    fn program_gen(size: usize) -> String {
        let mut labels = vec!["main".to_string()];

        let mut program: Vec<Vec<String>> =
            (0..size).map(|_| instruction_gen(&mut labels)).collect();

        let label_markers = HashSet::<String>::from_iter(labels.into_iter());
        for label in label_markers {
            program.push(vec![format!("{}:", label)]);
        }

        let mut rng = rand::thread_rng();
        program.shuffle(&mut rng);

        program
            .into_iter()
            .flatten()
            .intersperse_with(|| whitespace_gen(5))
            .collect()
    }

    #[test]
    fn parse_program_empty_test() {
        parse_program_prop(TestCase {
            input: "",
            expected: Program::new(&[]),
            message: "empty string should parse as empty program",
        });

        parse_program_prop(TestCase {
            input: "   ",
            expected: Program::new(&[]),
            message: "spaces should parse as empty program",
        });

        parse_program_prop(TestCase {
            input: "\n",
            expected: Program::new(&[]),
            message: "linebreaks should parse as empty program (1)",
        });

        parse_program_prop(TestCase {
            input: "   \n ",
            expected: Program::new(&[]),
            message: "linebreaks should parse as empty program (2)",
        });

        parse_program_prop(TestCase {
            input: "   \n \n",
            expected: Program::new(&[]),
            message: "linebreaks should parse as empty program (3)",
        });

        parse_program_prop(TestCase {
            input: "// empty program",
            expected: Program::new(&[]),
            message: "single comment should parse as empty program",
        });

        parse_program_prop(TestCase {
            input: "// empty program\n",
            expected: Program::new(&[]),
            message: "single comment with linebreak should parse as empty program",
        });

        parse_program_prop(TestCase {
            input: "// multi-line\n// comment",
            expected: Program::new(&[]),
            message: "multiple comments should parse as empty program",
        });

        parse_program_prop(TestCase {
            input: "// multi-line\n// comment\n ",
            expected: Program::new(&[]),
            message: "multiple comments with trailing whitespace should parse as empty program",
        });

        for size in 0..10 {
            let input = whitespace_gen(size);
            parse_program_prop(TestCase {
                input: &input,
                expected: Program::new(&[]),
                message: "arbitrary whitespace should parse as empty program",
            });
        }
    }

    #[test]
    fn parse_program_label_test() {
        use LabelledInstruction::*;

        // FIXME: This should fail.
        // FIXME: Increase coverage.
        parse_program_prop(TestCase {
            input: "pop: call pop",
            expected: Program::new(&[
                Label("pop".to_string(), ""),
                Instruction(Call("pop".to_string()), ""),
            ]),
            message: "label names may not overlap with instruction names",
        });

        // FIXME: Increase coverage of negative test for duplicate labels.
        {
            let input = "foo: pop foo: pop call foo";
            let result = parse(input);
            assert!(result.is_err(), "duplicate labels");

            let error = result.unwrap_err();
            let expected_error_count = 2;
            let actual_error_count = error.errors.errors.len();
            assert_eq!(expected_error_count, actual_error_count);

            let actual_error_message = format!("{}", error);
            let duplicate_label_errors = actual_error_message
                .match_indices("duplicate label")
                .count();
            assert_eq!(expected_error_count, duplicate_label_errors)
        }

        parse_program_prop(TestCase {
            input: "pops: call pops",
            expected: Program::new(&[
                Label("pops".to_string(), ""),
                Instruction(Call("pops".to_string()), ""),
            ]),
            message: "labels that share a common prefix with instruction are labels",
        });

        parse_program_prop(TestCase {
            input: "spop: call spop",
            expected: Program::new(&[
                Label("spop".to_string(), ""),
                Instruction(Call("spop".to_string()), ""),
            ]),
            message: "labels that share a common suffix with instruction are labels",
        });
    }

    #[test]
    fn parse_program_equivalence_test() {
        for size in 0..100 {
            let code = program_gen(size * 10);
            // println!(
            //     "Parsing the following program (size {}):\n---\n{}\n---\n\n",
            //     size, code
            // );

            let old_actual = instruction::parse(&code).map_err(|err| err.to_string());
            let new_actual = super::parse(&code).map_err(|err| err.to_string());

            if let Err(err) = new_actual.clone() {
                panic!("{}", err);
            }

            assert_eq!(old_actual, new_actual);
        }
    }
}
