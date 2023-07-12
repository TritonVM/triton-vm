use std::collections::HashMap;
use std::collections::HashSet;
use std::error::Error;

use nom::branch::alt;
use nom::bytes::complete::tag;
use nom::bytes::complete::take_while;
use nom::bytes::complete::take_while1;
use nom::character::complete::digit1;
use nom::combinator::cut;
use nom::combinator::eof;
use nom::combinator::fail;
use nom::combinator::opt;
use nom::error::context;
use nom::error::convert_error;
use nom::error::ErrorKind;
use nom::error::VerboseError;
use nom::error::VerboseErrorKind;
use nom::multi::many0;
use nom::multi::many1;
use nom::Finish;
use nom::IResult;
use twenty_first::shared_math::b_field_element::BFieldElement;

use crate::instruction::AnInstruction;
use crate::instruction::AnInstruction::*;
use crate::instruction::LabelledInstruction;
use crate::instruction::ALL_INSTRUCTION_NAMES;
use crate::op_stack::OpStackElement;
use crate::op_stack::OpStackElement::*;

#[derive(Debug, PartialEq)]
pub struct ParseError<'a> {
    pub input: &'a str,
    pub errors: VerboseError<&'a str>,
}

/// `InstructionToken` is either an instruction with a label, or a
/// label itself. It is intermediate object used in some middle
/// point of the compilation pipeline. You probably want
/// [`LabelledInstruction`].
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum InstructionToken<'a> {
    Instruction(AnInstruction<String>, &'a str),
    Label(String, &'a str),
}

impl<'a> std::fmt::Display for InstructionToken<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InstructionToken::Instruction(instr, _) => write!(f, "{instr}"),
            InstructionToken::Label(label_name, _) => write!(f, "{label_name}:"),
        }
    }
}

impl<'a> std::fmt::Display for ParseError<'a> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", pretty_print_error(self.input, self.errors.clone()))
    }
}

impl<'a> Error for ParseError<'a> {}

impl<'a> InstructionToken<'a> {
    pub fn token_str(&self) -> &'a str {
        match self {
            InstructionToken::Instruction(_, token_str) => token_str,
            InstructionToken::Label(_, token_str) => token_str,
        }
    }

    pub fn to_labelled_instruction(&self) -> LabelledInstruction {
        use InstructionToken::*;
        match self {
            Instruction(instr, _) => LabelledInstruction::Instruction(instr.to_owned()),
            Label(label, _) => LabelledInstruction::Label(label.to_owned()),
        }
    }
}

pub fn to_labelled_instructions(instructions: &[InstructionToken]) -> Vec<LabelledInstruction> {
    instructions
        .iter()
        .map(|instruction| instruction.to_labelled_instruction())
        .collect()
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
    if matches!(
        root_error,
        VerboseErrorKind::Nom(ErrorKind::Fail) | VerboseErrorKind::Nom(ErrorKind::Eof)
    ) {
        e.errors.remove(0);
    }
    convert_error(s, e)
}

/// Parse a program
pub fn parse(input: &str) -> Result<Vec<InstructionToken>, ParseError> {
    let instructions = match tokenize(input).finish() {
        Ok((_s, instructions)) => Ok(instructions),
        Err(errors) => Err(ParseError { input, errors }),
    }?;

    scan_missing_duplicate_labels(input, &instructions)?;

    Ok(instructions)
}

fn scan_missing_duplicate_labels<'a>(
    input: &'a str,
    instructions: &[InstructionToken<'a>],
) -> Result<(), ParseError<'a>> {
    let mut seen: HashMap<&str, InstructionToken> = HashMap::default();
    let mut duplicates: HashSet<InstructionToken> = HashSet::default();
    let mut missings: HashSet<InstructionToken> = HashSet::default();

    // Find duplicate labels, including the first occurrence of each duplicate.
    for instruction in instructions.iter() {
        if let InstructionToken::Label(label, _token_s) = instruction {
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
        if let InstructionToken::Instruction(Call(addr), _token_s) = instruction {
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
            duplicate.token_str(),
            VerboseErrorKind::Context("duplicate label"),
        );
        errors.push(error);
    }

    for missing in missings {
        let error = (
            missing.token_str(),
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

///
pub fn tokenize(s: &str) -> ParseResult<Vec<InstructionToken>> {
    let (s, _) = comment_or_whitespace0(s)?;
    let (s, instructions) = many0(alt((label, labelled_instruction)))(s)?;
    let (s, _) = context("expecting label, instruction or eof", eof)(s)?;

    Ok((s, instructions))
}

fn labelled_instruction(s_instr: &str) -> ParseResult<InstructionToken> {
    let (s, instr) = an_instruction(s_instr)?;
    Ok((s, InstructionToken::Instruction(instr, s_instr)))
}

fn label(label_s: &str) -> ParseResult<InstructionToken> {
    let (s, addr) = label_addr(label_s)?;
    let (s, _) = token0("")(s)?; // whitespace between label and ':' is allowed
    let (s, _) = token0(":")(s)?; // don't require space after ':'

    // Checking if `<label>:` is an instruction must happen after parsing `:`, since otherwise
    // `cut` will reject the alternative parser of `label`, being `labelled_instruction`, which
    // *is* allowed to contain valid instruction names.
    if is_instruction_name(&addr) {
        return cut(context("label cannot be named after instruction", fail))(label_s);
    }

    Ok((s, InstructionToken::Label(addr, label_s)))
}

fn an_instruction(s: &str) -> ParseResult<AnInstruction<String>> {
    // OpStack manipulation
    let pop = instruction("pop", Pop);
    let push = push_instruction();
    let divine = instruction("divine", Divine);
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
    let absorb_init = instruction("absorb_init", AbsorbInit);
    let absorb = instruction("absorb", Absorb);
    let squeeze = instruction("squeeze", Squeeze);

    let hashing_related = alt((hash, divine_sibling, absorb_init, absorb, squeeze));

    // Arithmetic on stack instructions
    let add = instruction("add", Add);
    let mul = instruction("mul", Mul);
    let invert = instruction("invert", Invert);
    let eq = instruction("eq", Eq);
    let split = instruction("split", Split);
    let lt = instruction("lt", Lt);
    let and = instruction("and", And);
    let xor = instruction("xor", Xor);
    let log_2_floor = instruction("log_2_floor", Log2Floor);
    let pow = instruction("pow", Pow);
    let div = instruction("div", Div);
    let pop_count = instruction("pop_count", PopCount);
    let xxadd = instruction("xxadd", XxAdd);
    let xxmul = instruction("xxmul", XxMul);
    let xinvert = instruction("xinvert", XInvert);
    let xbmul = instruction("xbmul", XbMul);

    let base_field_arithmetic_on_stack = alt((add, mul, invert, eq));
    let bitwise_arithmetic_on_stack = alt((split, lt, and, xor, log_2_floor, pow, div, pop_count));
    let extension_field_arithmetic_on_stack = alt((xxadd, xxmul, xinvert, xbmul));
    let arithmetic_on_stack = alt((
        base_field_arithmetic_on_stack,
        bitwise_arithmetic_on_stack,
        extension_field_arithmetic_on_stack,
    ));

    // Read/write
    let read_io = instruction("read_io", ReadIo);
    let write_io = instruction("write_io", WriteIo);

    let read_write = alt((read_io, write_io));

    // Because of common prefixes, the following parsers are sensitive to order:
    //
    // Successfully parsing "assert" before trying "assert_vector" can lead to
    // picking the wrong one. By trying them in the order of longest first, less
    // backtracking is necessary.
    let syntax_ambiguous = alt((assert_vector, assert_, divine));

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

fn is_instruction_name(s: &str) -> bool {
    ALL_INSTRUCTION_NAMES.contains(&s)
}

fn instruction<'a>(
    name: &'a str,
    instruction: AnInstruction<String>,
) -> impl Fn(&'a str) -> ParseResult<AnInstruction<String>> {
    move |s: &'a str| {
        let (s, _) = token1(name)(s)?; // require space after instruction name
        Ok((s, instruction.clone()))
    }
}

fn push_instruction() -> impl Fn(&str) -> ParseResult<AnInstruction<String>> {
    move |s: &str| {
        let (s, _) = token1("push")(s)?; // require space after instruction name
        let (s, elem) = field_element(s)?;
        let (s, _) = comment_or_whitespace1(s)?; // require space after field element

        Ok((s, Push(elem)))
    }
}

fn dup_instruction() -> impl Fn(&str) -> ParseResult<AnInstruction<String>> {
    move |s: &str| {
        let (s, _) = token1("dup")(s)?; // require space before argument
        let (s, stack_register) = stack_register(s)?;
        let (s, _) = comment_or_whitespace1(s)?;

        Ok((s, Dup(stack_register)))
    }
}

fn swap_instruction() -> impl Fn(&str) -> ParseResult<AnInstruction<String>> {
    move |s: &str| {
        let (s, _) = token1("swap")(s)?; // require space before argument
        let (s, stack_register) = stack_register(s)?;
        let (s, _) = comment_or_whitespace1(s)?;

        if stack_register == ST0 {
            return cut(context("instruction `swap` cannot take argument `0`", fail))(s);
        }

        Ok((s, Swap(stack_register)))
    }
}

fn call_instruction<'a>() -> impl Fn(&'a str) -> ParseResult<AnInstruction<String>> {
    let call_syntax = move |s: &'a str| {
        let (s_label, _) = token1("call")(s)?; // require space before called label
        let (s, addr) = label_addr(s_label)?;
        let (s, _) = comment_or_whitespace1(s)?; // require space after called label

        Ok((s, addr))
    };

    let bracket_syntax = move |s: &'a str| {
        let (s, _) = tag("[")(s)?;
        let (s, addr) = label_addr(s)?;
        let (s, _) = token0("]")(s)?;

        Ok((s, addr))
    };

    move |s: &'a str| {
        let (s, addr) = alt((call_syntax, bracket_syntax))(s)?;

        // This check cannot be moved into `label_addr`, since `label_addr` is shared
        // between the scenarios `<label>:` and `call <label>`; the former requires
        // parsing the `:` before rejecting a possible instruction name in the label.
        if is_instruction_name(&addr) {
            return cut(context("label cannot be named after instruction", fail))(s);
        }

        Ok((s, Call(addr)))
    }
}

fn field_element(s_orig: &str) -> ParseResult<BFieldElement> {
    let (s, negative) = opt(token0("-"))(s_orig)?;
    let (s, n) = digit1(s)?;

    let mut n: i128 = match n.parse() {
        Ok(n) => n,
        Err(_err) => {
            return context("out-of-bounds constant", fail)(s);
        }
    };

    let quotient = BFieldElement::P as i128;
    if n >= quotient {
        return context("out-of-bounds constant", fail)(s_orig);
    }

    if negative.is_some() {
        n *= -1;
        n += quotient;
    }

    Ok((s, BFieldElement::new(n as u64)))
}

fn stack_register(s: &str) -> ParseResult<OpStackElement> {
    let (s, n) = digit1(s)?;
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

/// Parse a label address. This is used in "`<label>:`" and in "`call <label>`".
fn label_addr(s_orig: &str) -> ParseResult<String> {
    let (s, addr_part_0) = take_while1(is_label_start_char)(s_orig)?;
    if addr_part_0.is_empty() {
        // todo: this error is never shown to the user, since the `label` parser is wrapped in an
        //  `alt`. With a custom error type, it is possible to have alt return the error of the
        //  parser that went the farthest in the input data.
        let failure_reason = "label must start with an alphabetic character or underscore";
        return context(failure_reason, fail)(s_orig);
    }
    let (s, addr_part_1) = take_while(is_label_char)(s)?;

    Ok((s, format!("{}{}", addr_part_0, addr_part_1)))
}

fn is_label_start_char(c: char) -> bool {
    c.is_alphabetic() || c == '_'
}

fn is_label_char(c: char) -> bool {
    c.is_alphanumeric() || c == '_' || c == '-'
}

/// Parse 0 or more comments and/or whitespace.
///
/// This is used after places where whitespace is optional (e.g. after ':').
fn comment_or_whitespace0(s: &str) -> ParseResult<&str> {
    let (s, _) = many0(alt((comment1, whitespace1)))(s)?;
    Ok((s, ""))
}

/// Parse at least one comment and/or whitespace, or eof.
///
/// This is used after places where whitespace is mandatory (e.g. after tokens).
fn comment_or_whitespace1<'a>(s: &'a str) -> ParseResult<&'a str> {
    let cws1 = |s: &'a str| -> ParseResult<&str> {
        let (s, _) = many1(alt((comment1, whitespace1)))(s)?;
        Ok((s, ""))
    };
    alt((eof, cws1))(s)
}

/// Parse one comment (not including the linebreak)
fn comment1(s: &str) -> ParseResult<()> {
    let (s, _) = tag("//")(s)?;
    let (s, _) = take_while(|c| !is_linebreak(c))(s)?;
    Ok((s, ()))
}

/// Parse at least one whitespace character
fn whitespace1(s: &str) -> ParseResult<()> {
    let (s, _) = take_while1(|c: char| c.is_whitespace())(s)?;
    Ok((s, ()))
}

fn is_linebreak(c: char) -> bool {
    c == '\r' || c == '\n'
}

/// `token0(tok)` will parse the string `tok` and munch 0 or more comment or whitespace.
fn token0<'a>(token: &'a str) -> impl Fn(&'a str) -> ParseResult<()> {
    move |s: &'a str| {
        let (s, _) = tag(token)(s)?;
        let (s, _) = comment_or_whitespace0(s)?;
        Ok((s, ()))
    }
}

/// `token1(tok)` will parse the string `tok` and munch at least one comment and/or whitespace,
/// or eof.
fn token1<'a>(token: &'a str) -> impl Fn(&'a str) -> ParseResult<()> {
    move |s: &'a str| {
        let (s, _) = tag(token)(s)?;
        let (s, _) = comment_or_whitespace1(s)?;
        Ok((s, ()))
    }
}

#[cfg(test)]
pub mod parser_tests {
    use itertools::Itertools;
    use rand::distributions::WeightedIndex;
    use rand::prelude::*;
    use rand::Rng;

    use LabelledInstruction::*;

    use crate::program::Program;
    use crate::triton_asm;
    use crate::triton_instr;
    use crate::triton_program;

    use super::*;

    struct TestCase<'a> {
        input: &'a str,
        expected: Program,
        message: &'static str,
    }

    struct NegativeTestCase<'a> {
        input: &'a str,
        expected_error: &'static str,
        expected_error_count: usize,
        message: &'static str,
    }

    fn parse_program_prop(test_case: TestCase) {
        match parse(test_case.input) {
            Ok(actual) => assert_eq!(
                test_case.expected,
                Program::new(&to_labelled_instructions(&actual)),
                "{}",
                test_case.message
            ),
            Err(parse_err) => panic!("{}:\n{parse_err}", test_case.message),
        }
    }

    fn parse_program_neg_prop(test_case: NegativeTestCase) {
        let result = parse(test_case.input);
        if result.is_ok() {
            eprintln!("parser input: {}", test_case.input);
            eprintln!("parser output: {:?}", result.unwrap());
            panic!("parser should fail, but didn't: {}", test_case.message);
        }

        let error = result.unwrap_err();
        let actual_error_message = format!("{error}");
        let actual_error_count = actual_error_message
            .match_indices(test_case.expected_error)
            .count();
        if test_case.expected_error_count != actual_error_count {
            eprintln!("Actual error message:");
            eprintln!("{actual_error_message}");
            assert_eq!(
                test_case.expected_error_count, actual_error_count,
                "parser should report '{}' {} times: {}",
                test_case.expected_error, test_case.expected_error_count, test_case.message
            )
        }
    }

    fn whitespace_gen(max_size: usize) -> String {
        let mut rng = rand::thread_rng();
        let spaces = [" ", "\t", "\r", "\r\n", "\n", " // comment\n"];
        let weights = [5, 1, 1, 1, 2, 1];
        assert_eq!(spaces.len(), weights.len(), "all generators have weights");
        let dist = WeightedIndex::new(weights).expect("a weighted distribution of generators");
        let size = rng.gen_range(1..=std::cmp::max(1, max_size));
        (0..size).map(|_| spaces[dist.sample(&mut rng)]).collect()
    }

    fn label_gen(size: usize) -> String {
        let mut rng = rand::thread_rng();
        let mut new_label = || -> String { (0..size).map(|_| rng.gen_range('a'..='z')).collect() };
        let mut label = new_label();
        while is_instruction_name(&label) {
            label = new_label();
        }
        label
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
        let mut rng = thread_rng();

        let difficult_instructions = vec!["push", "dup", "swap", "skiz", "call"];
        let simple_instructions = ALL_INSTRUCTION_NAMES
            .into_iter()
            .filter(|name| !difficult_instructions.contains(name))
            .collect_vec();

        let generators = vec![vec!["simple"], difficult_instructions].concat();
        // Test difficult instructions more frequently.
        let weights = vec![simple_instructions.len(), 2, 6, 6, 2, 10];

        assert_eq!(
            generators.len(),
            weights.len(),
            "all generators must have weights"
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
                vec!["push".to_string(), format!("{arg}")]
            }

            "dup" => {
                let arg: usize = rng.gen_range(0..15);
                vec!["dup".to_string(), format!("{arg}")]
            }

            "swap" => {
                let arg: usize = rng.gen_range(1..15);
                vec!["swap".to_string(), format!("{arg}")]
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

            unknown => panic!("Unknown generator, {unknown}"),
        }
    }

    // FIXME: Apply shrinking.
    #[allow(unstable_name_collisions)]
    // reason = "Switch to standard library intersperse_with() when it's ported"
    pub fn program_gen(size: usize) -> String {
        // Generate random program
        let mut labels = vec![];
        let mut program: Vec<Vec<String>> =
            (0..size).map(|_| instruction_gen(&mut labels)).collect();

        // Embed all used labels randomly
        for label in labels.into_iter().sorted().dedup() {
            program.push(vec![format!("{label}:")]);
        }
        program.shuffle(&mut rand::thread_rng());

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
    fn parse_program_whitespace_test() {
        parse_program_neg_prop(NegativeTestCase {
            input: "poppop",
            expected_error: "n/a",
            expected_error_count: 0,
            message: "whitespace required between instructions (pop)",
        });

        parse_program_neg_prop(NegativeTestCase {
            input: "dup 0dup 0",
            expected_error: "n/a",
            expected_error_count: 0,
            message: "whitespace required between instructions (dup 0)",
        });

        parse_program_neg_prop(NegativeTestCase {
            input: "swap 10swap 10",
            expected_error: "n/a",
            expected_error_count: 0,
            message: "whitespace required between instructions (swap 10)",
        });

        parse_program_neg_prop(NegativeTestCase {
            input: "push10",
            expected_error: "n/a",
            expected_error_count: 0,
            message: "push requires whitespace before its constant",
        });

        parse_program_neg_prop(NegativeTestCase {
            input: "push 10pop",
            expected_error: "n/a",
            expected_error_count: 0,
            message: "push requires whitespace after its constant",
        });

        parse_program_neg_prop(NegativeTestCase {
            input: "hello: callhello",
            expected_error: "n/a",
            expected_error_count: 0,
            message: "call requires whitespace before its label",
        });

        parse_program_neg_prop(NegativeTestCase {
            input: "hello: popcall hello",
            expected_error: "n/a",
            expected_error_count: 0,
            message: "required space between pop and call",
        });
    }

    #[test]
    fn parse_program_label_test() {
        parse_program_prop(TestCase {
            input: "foo: call foo",
            expected: Program::new(&[
                Label("foo".to_string()),
                Instruction(Call("foo".to_string())),
            ]),
            message: "parse labels and calls to labels",
        });

        parse_program_prop(TestCase {
            input: "foo:call foo",
            expected: Program::new(&[
                Label("foo".to_string()),
                Instruction(Call("foo".to_string())),
            ]),
            message: "whitespace is not required after 'label:'",
        });

        // FIXME: Increase coverage of negative tests for duplicate labels.
        parse_program_neg_prop(NegativeTestCase {
            input: "foo: pop foo: pop call foo",
            expected_error: "duplicate label",
            expected_error_count: 2,
            message: "labels cannot occur twice",
        });

        // FIXME: Increase coverage of negative tests for missing labels.
        parse_program_neg_prop(NegativeTestCase {
            input: "foo: pop call herp call derp",
            expected_error: "missing label",
            expected_error_count: 2,
            message: "non-existent labels cannot be called",
        });

        // FIXME: Increase coverage of negative tests for label/keyword overlap.
        parse_program_neg_prop(NegativeTestCase {
            input: "pop: call pop",
            expected_error: "label cannot be named after instruction",
            expected_error_count: 1,
            message: "label names may not overlap with instruction names",
        });

        parse_program_prop(TestCase {
            input: "pops: call pops",
            expected: Program::new(&[
                Label("pops".to_string()),
                Instruction(Call("pops".to_string())),
            ]),
            message: "labels that share a common prefix with instruction are labels",
        });

        parse_program_prop(TestCase {
            input: "_call: call _call",
            expected: Program::new(&[
                Label("_call".to_string()),
                Instruction(Call("_call".to_string())),
            ]),
            message: "labels that share a common suffix with instruction are labels",
        });
    }

    #[test]
    fn parse_program_nonexistent_instructions_test() {
        parse_program_neg_prop(NegativeTestCase {
            input: "swap 0",
            expected_error: "instruction `swap` cannot take argument `0`",
            expected_error_count: 1,
            message: "instruction `swap` cannot take argument `0`",
        });

        parse_program_neg_prop(NegativeTestCase {
            input: "swap 16",
            expected_error: "expecting label, instruction or eof",
            expected_error_count: 1,
            message: "there is no swap 16 instruction",
        });

        parse_program_neg_prop(NegativeTestCase {
            input: "dup 16",
            expected_error: "expecting label, instruction or eof",
            expected_error_count: 1,
            message: "there is no dup 16 instruction",
        });
    }

    #[test]
    fn parse_program_bracket_syntax_test() {
        parse_program_prop(TestCase {
            input: "foo: [foo]",
            expected: Program::new(&[
                Label("foo".to_string()),
                Instruction(Call("foo".to_string())),
            ]),
            message: "Handle brackets as call syntax sugar",
        });

        parse_program_neg_prop(NegativeTestCase {
            input: "foo: [bar]",
            expected_error: "missing label",
            expected_error_count: 1,
            message: "Handle missing labels with bracket syntax",
        })
    }

    #[test]
    fn parse_program_test() {
        for size in 0..100 {
            let code = program_gen(size * 10);

            let new_actual = parse(&code).map_err(|err| err.to_string());

            if new_actual.is_err() {
                println!("The code:\n{code}\n\n");
                panic!("{}", new_actual.unwrap_err());
            }
        }
    }

    #[test]
    fn parse_program_label_must_start_with_alphabetic_character_or_underscore() {
        parse_program_neg_prop(NegativeTestCase {
            input: "1foo: call 1foo",
            expected_error: "expecting label, instruction or eof",
            expected_error_count: 1,
            message: "labels cannot start with a digit",
        });

        parse_program_neg_prop(NegativeTestCase {
            input: "-foo: call -foo",
            expected_error: "expecting label, instruction or eof",
            expected_error_count: 1,
            message: "labels cannot start with a dash",
        });

        parse_program_prop(TestCase {
            input: "_foo: call _foo",
            expected: Program::new(&[
                Label("_foo".to_string()),
                Instruction(Call("_foo".to_string())),
            ]),
            message: "labels can start with an underscore",
        });
    }

    #[test]
    fn triton_asm_macro() {
        let instructions = triton_asm!(write_io push 17 call which_label lt swap 3);
        assert_eq!(Instruction(WriteIo), instructions[0]);
        assert_eq!(Instruction(Push(17_u64.into())), instructions[1]);
        assert_eq!(
            Instruction(Call("which_label".to_string())),
            instructions[2]
        );
        assert_eq!(Instruction(Lt), instructions[3]);
        assert_eq!(Instruction(Swap(ST3)), instructions[4]);
    }

    #[test]
    fn triton_asm_macro_with_a_single_return() {
        let instructions = triton_asm!(return);
        assert_eq!(Instruction(Return), instructions[0]);
    }

    #[test]
    fn triton_asm_macro_with_a_single_assert() {
        let instructions = triton_asm!(assert);
        assert_eq!(Instruction(Assert), instructions[0]);
    }

    #[test]
    fn triton_asm_macro_with_only_assert_and_return() {
        let instructions = triton_asm!(assert return);
        assert_eq!(Instruction(Assert), instructions[0]);
        assert_eq!(Instruction(Return), instructions[1]);
    }

    #[test]
    fn triton_program_macro() {
        let program = triton_program!(
            // main entry point
            call routine_1
            call routine_2 // inline comment
            halt

            // single line content regarding subroutine 1
            routine_1:
                /*
                 * multi-line comment
                 */
                call routine_3
                return

            // subroutine 2 starts here
            routine_2:
                push 18 push 7 add
                push 25 eq assert
                return

            routine_3:
            alternative_label:
                nop nop
                return
        );
        println!("{program}");
    }

    #[test]
    fn triton_program_macro_interpolates_various_types() {
        let push_arg = thread_rng().gen_range(0_u64..BFieldElement::P);
        let instruction_push = Instruction(Push(push_arg.into()));
        let swap_argument = "1";
        triton_program!({instruction_push} push {push_arg} swap {swap_argument} eq assert halt)
            .run(vec![], vec![])
            .unwrap();
    }

    #[test]
    fn triton_instruction_macro() {
        assert_eq!(LabelledInstruction::Instruction(Halt), triton_instr!(halt));
    }
}
