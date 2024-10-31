use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::collections::HashSet;
use std::error::Error;
use std::fmt::Display;
use std::fmt::Formatter;
use std::fmt::Result as FmtResult;

use nom::branch::alt;
use nom::bytes::complete::tag;
use nom::bytes::complete::take_while;
use nom::bytes::complete::take_while1;
use nom::character::complete::digit1;
use nom::combinator::cut;
use nom::combinator::eof;
use nom::combinator::fail;
use nom::combinator::opt;
use nom::error::ErrorKind;
use nom::error::VerboseError;
use nom::error::VerboseErrorKind;
use nom::multi::many0;
use nom::multi::many1;
use nom::Finish;
use nom::IResult;
use twenty_first::bfe;
use twenty_first::prelude::BFieldElement;

use crate::instruction::AnInstruction;
use crate::instruction::AssertionContext;
use crate::instruction::Instruction;
use crate::instruction::LabelledInstruction;
use crate::instruction::TypeHint;
use crate::instruction::ALL_INSTRUCTION_NAMES;
use crate::op_stack::NumberOfWords;
use crate::op_stack::OpStackElement;

const KEYWORDS: [&str; 3] = [
    "hint",
    "error_id",
    "error_message", // reserved for future use
];

#[derive(Debug, PartialEq)]
pub struct ParseError<'a> {
    pub input: &'a str,
    pub errors: VerboseError<&'a str>,
}

/// An intermediate object for the parsing / compilation pipeline. You probably want
/// [`LabelledInstruction`].
#[derive(Debug, Clone, Eq, PartialEq, Hash)]
pub enum InstructionToken<'a> {
    Instruction(AnInstruction<String>, &'a str),
    Label(String, &'a str),
    Breakpoint(&'a str),
    TypeHint(TypeHint, &'a str),
    AssertionContext(AssertionContext, &'a str),
}

impl<'a> Display for ParseError<'a> {
    fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
        let mut errors = self.errors.clone();
        if matches!(
            errors.errors.first(),
            Some((_, VerboseErrorKind::Nom(ErrorKind::Fail | ErrorKind::Eof)))
        ) {
            errors.errors.remove(0);
        }

        write!(f, "{}", nom::error::convert_error(self.input, errors))
    }
}

impl<'a> Error for ParseError<'a> {}

impl<'a> InstructionToken<'a> {
    pub fn token_str(&self) -> &'a str {
        match self {
            Self::Instruction(_, token_str) => token_str,
            Self::Label(_, token_str) => token_str,
            Self::Breakpoint(token_str) => token_str,
            Self::TypeHint(_, token_str) => token_str,
            Self::AssertionContext(_, token_str) => token_str,
        }
    }

    pub fn to_labelled_instruction(&self) -> LabelledInstruction {
        match self {
            Self::Instruction(instr, _) => LabelledInstruction::Instruction(instr.to_owned()),
            Self::Label(label, _) => LabelledInstruction::Label(label.to_owned()),
            Self::Breakpoint(_) => LabelledInstruction::Breakpoint,
            Self::TypeHint(type_hint, _) => LabelledInstruction::TypeHint(type_hint.to_owned()),
            Self::AssertionContext(ctx, _) => LabelledInstruction::AssertionContext(ctx.to_owned()),
        }
    }
}

pub fn to_labelled_instructions(instructions: &[InstructionToken]) -> Vec<LabelledInstruction> {
    instructions
        .iter()
        .map(|instruction| instruction.to_labelled_instruction())
        .collect()
}

/// Parse a program
pub(crate) fn parse(input: &str) -> Result<Vec<InstructionToken>, ParseError> {
    let (_, instructions) = tokenize(input)
        .finish()
        .map_err(|errors| ParseError { input, errors })?;

    ensure_no_missing_or_duplicate_labels(input, &instructions)?;
    ensure_assertion_context_is_matched_with_assertion(input, &instructions)?;

    Ok(instructions)
}

fn ensure_no_missing_or_duplicate_labels<'a>(
    input: &'a str,
    instructions: &[InstructionToken<'a>],
) -> Result<(), ParseError<'a>> {
    // identify all and duplicate labels
    let mut seen_labels = HashMap::<_, InstructionToken>::default();
    let mut duplicate_labels = HashSet::default();
    for instruction in instructions {
        if let InstructionToken::Label(label, _) = instruction {
            if let Some(first_occurrence) = seen_labels.get(label.as_str()) {
                duplicate_labels.insert(first_occurrence.to_owned());
                duplicate_labels.insert(instruction.to_owned());
            } else {
                seen_labels.insert(label.as_str(), instruction.to_owned());
            }
        }
    }

    let missing_labels = identify_missing_labels(instructions, seen_labels);

    if duplicate_labels.is_empty() && missing_labels.is_empty() {
        return Ok(());
    }

    let errors = errors_for_duplicate_and_missing_labels(duplicate_labels, missing_labels);
    Err(ParseError { input, errors })
}

fn identify_missing_labels<'a>(
    instructions: &[InstructionToken<'a>],
    seen_labels: HashMap<&str, InstructionToken>,
) -> HashSet<InstructionToken<'a>> {
    let mut missing_labels = HashSet::default();
    for instruction in instructions {
        if let InstructionToken::Instruction(AnInstruction::Call(label), _) = instruction {
            if !seen_labels.contains_key(label.as_str()) {
                missing_labels.insert(instruction.to_owned());
            }
        }
    }
    missing_labels
}

fn errors_for_duplicate_and_missing_labels<'a>(
    duplicate_labels: HashSet<InstructionToken<'a>>,
    missing_labels: HashSet<InstructionToken<'a>>,
) -> VerboseError<&'a str> {
    let duplicate_label_error_context = VerboseErrorKind::Context("duplicate label");
    let missing_label_error_context = VerboseErrorKind::Context("missing label");

    let duplicate_label_errors =
        errors_for_labels_with_context(duplicate_labels, duplicate_label_error_context);
    let missing_label_errors =
        errors_for_labels_with_context(missing_labels, missing_label_error_context);

    let errors = [duplicate_label_errors, missing_label_errors].concat();
    VerboseError { errors }
}

fn errors_for_labels_with_context(
    labels: HashSet<InstructionToken>,
    context: VerboseErrorKind,
) -> Vec<(&str, VerboseErrorKind)> {
    labels
        .into_iter()
        .map(|label| (label.token_str(), context.clone()))
        .collect()
}

fn ensure_assertion_context_is_matched_with_assertion<'a>(
    input: &'a str,
    instructions: &[InstructionToken<'a>],
) -> Result<(), ParseError<'a>> {
    type IT<'a> = InstructionToken<'a>;

    let mut accept_id = false;
    let mut incorrectly_placed_contexts = HashSet::new();

    for instruction in instructions {
        match instruction {
            IT::Instruction(AnInstruction::Assert | AnInstruction::AssertVector, _) => {
                accept_id = true;
            }
            IT::AssertionContext(AssertionContext::ID(_), _) => {
                if !accept_id {
                    incorrectly_placed_contexts.insert(instruction.clone());
                }
                accept_id = false;
            }

            // any assertion context must immediately follow an `assert` or `assert_vector`
            _ => accept_id = false,
        }
    }

    let parser_context = VerboseErrorKind::Context("incorrectly placed assertion context");
    let errors = errors_for_labels_with_context(incorrectly_placed_contexts, parser_context);

    if errors.is_empty() {
        Ok(())
    } else {
        let errors = VerboseError { errors };
        Err(ParseError { input, errors })
    }
}

/// Auxiliary type alias: `IResult` defaults to `nom::error::Error` as concrete
/// error type, but we want `nom::error::VerboseError` as it allows `context()`.
type ParseResult<'input, Out> = IResult<&'input str, Out, VerboseError<&'input str>>;

pub fn tokenize(s: &str) -> ParseResult<Vec<InstructionToken>> {
    let (s, _) = comment_or_whitespace0(s)?;
    let (s, instructions) = many0(alt((
        label,
        labelled_instruction,
        breakpoint,
        type_hint,
        assertion_context,
    )))(s)?;
    let (s, _) = nom::error::context("expecting label, instruction or eof", eof)(s)?;

    Ok((s, instructions))
}

fn label(label_s: &str) -> ParseResult<InstructionToken> {
    let (s, addr) = label_addr(label_s)?;
    let (s, _) = whitespace0(s)?; // whitespace between label and ':' is allowed
    let (s, _) = token0(":")(s)?; // don't require space after ':'

    // Checking if `<label>:` is an instruction must happen after parsing `:`, since
    // otherwise `cut` will reject the alternative parsers of `label`, like
    // `labelled_instruction`, where valid instructions names *are* allowed.
    if is_illegal_label(&addr) {
        let failure_reason = "label must be neither instruction nor keyword";
        return cut(nom::error::context(failure_reason, fail))(label_s);
    }

    Ok((s, InstructionToken::Label(addr, label_s)))
}

fn breakpoint(breakpoint_s: &str) -> ParseResult<InstructionToken> {
    let (s, _) = token1("break")(breakpoint_s)?;
    Ok((s, InstructionToken::Breakpoint(breakpoint_s)))
}

fn labelled_instruction(s_instr: &str) -> ParseResult<InstructionToken> {
    let (s, instr) = an_instruction(s_instr)?;
    Ok((s, InstructionToken::Instruction(instr, s_instr)))
}

fn an_instruction(s: &str) -> ParseResult<AnInstruction<String>> {
    // OpStack manipulation
    let pop = pop_instruction();
    let push = push_instruction();
    let divine = divine_instruction();
    let pick = pick_instruction();
    let place = place_instruction();
    let dup = dup_instruction();
    let swap = swap_instruction();

    let opstack_manipulation = alt((pop, push, divine, pick, place, dup, swap));

    // Control flow
    let halt = instruction("halt", AnInstruction::Halt);
    let nop = instruction("nop", AnInstruction::Nop);
    let skiz = instruction("skiz", AnInstruction::Skiz);
    let call = call_instruction();
    let return_ = instruction("return", AnInstruction::Return);
    let recurse = instruction("recurse", AnInstruction::Recurse);
    let recurse_or_return = instruction("recurse_or_return", AnInstruction::RecurseOrReturn);
    let assert = instruction("assert", AnInstruction::Assert);

    let control_flow = alt((nop, skiz, call, return_, halt));

    // Memory access
    let read_mem = read_mem_instruction();
    let write_mem = write_mem_instruction();

    let memory_access = alt((read_mem, write_mem));

    // Hashing-related instructions
    let hash = instruction("hash", AnInstruction::Hash);
    let assert_vector = instruction("assert_vector", AnInstruction::AssertVector);
    let sponge_init = instruction("sponge_init", AnInstruction::SpongeInit);
    let sponge_absorb = instruction("sponge_absorb", AnInstruction::SpongeAbsorb);
    let sponge_absorb_mem = instruction("sponge_absorb_mem", AnInstruction::SpongeAbsorbMem);
    let sponge_squeeze = instruction("sponge_squeeze", AnInstruction::SpongeSqueeze);

    let hashing_related = alt((hash, sponge_init, sponge_squeeze));

    // Arithmetic on stack instructions
    let add = instruction("add", AnInstruction::Add);
    let addi = addi_instruction();
    let mul = instruction("mul", AnInstruction::Mul);
    let invert = instruction("invert", AnInstruction::Invert);
    let eq = instruction("eq", AnInstruction::Eq);
    let split = instruction("split", AnInstruction::Split);
    let lt = instruction("lt", AnInstruction::Lt);
    let and = instruction("and", AnInstruction::And);
    let xor = instruction("xor", AnInstruction::Xor);
    let log_2_floor = instruction("log_2_floor", AnInstruction::Log2Floor);
    let pow = instruction("pow", AnInstruction::Pow);
    let div_mod = instruction("div_mod", AnInstruction::DivMod);
    let pop_count = instruction("pop_count", AnInstruction::PopCount);
    let xx_add = instruction("xx_add", AnInstruction::XxAdd);
    let xx_mul = instruction("xx_mul", AnInstruction::XxMul);
    let x_invert = instruction("x_invert", AnInstruction::XInvert);
    let xb_mul = instruction("xb_mul", AnInstruction::XbMul);

    let base_field_arithmetic_on_stack = alt((mul, invert, eq));
    let bitwise_arithmetic_on_stack =
        alt((split, lt, and, xor, log_2_floor, pow, div_mod, pop_count));
    let extension_field_arithmetic_on_stack = alt((xx_add, xx_mul, x_invert, xb_mul));
    let arithmetic_on_stack = alt((
        base_field_arithmetic_on_stack,
        bitwise_arithmetic_on_stack,
        extension_field_arithmetic_on_stack,
    ));

    // Read/write
    let read_io = read_io_instruction();
    let write_io = write_io_instruction();

    let read_write = alt((read_io, write_io));

    // Many-in-One
    let merkle_step = instruction("merkle_step", AnInstruction::MerkleStep);
    let merkle_step_mem = instruction("merkle_step_mem", AnInstruction::MerkleStepMem);
    let xx_dot_step = instruction("xx_dot_step", AnInstruction::XxDotStep);
    let xb_dot_step = instruction("xb_dot_step", AnInstruction::XbDotStep);

    let many_to_one = alt((xx_dot_step, xb_dot_step));

    // Because of common prefixes, the following parsers are sensitive to order.
    // Successfully parsing "assert" before trying "assert_vector" can lead to
    // picking the wrong one. By trying them in the order of longest first, less
    // backtracking is necessary.
    let syntax_ambiguous = alt((
        recurse_or_return,
        recurse,
        assert_vector,
        assert,
        sponge_absorb_mem,
        sponge_absorb,
        addi,
        add,
        merkle_step_mem,
        merkle_step,
    ));

    alt((
        opstack_manipulation,
        control_flow,
        memory_access,
        hashing_related,
        arithmetic_on_stack,
        read_write,
        many_to_one,
        syntax_ambiguous,
    ))(s)
}

fn is_illegal_label(s: &str) -> bool {
    ALL_INSTRUCTION_NAMES.contains(&s) || KEYWORDS.contains(&s)
}

fn instruction<'a>(
    name: &'a str,
    instruction: AnInstruction<String>,
) -> impl Fn(&'a str) -> ParseResult<AnInstruction<String>> {
    move |s: &'a str| {
        let (s, _) = token1(name)(s)?;
        Ok((s, instruction.clone()))
    }
}

fn pop_instruction() -> impl Fn(&str) -> ParseResult<AnInstruction<String>> {
    move |s: &str| {
        let (s, _) = token1("pop")(s)?;
        let (s, arg) = number_of_words(s)?;
        Ok((s, AnInstruction::Pop(arg)))
    }
}

fn push_instruction() -> impl Fn(&str) -> ParseResult<AnInstruction<String>> {
    move |s: &str| {
        let (s, _) = token1("push")(s)?;
        let (s, elem) = field_element(s)?;
        Ok((s, AnInstruction::Push(elem)))
    }
}

fn addi_instruction() -> impl Fn(&str) -> ParseResult<AnInstruction<String>> {
    move |s: &str| {
        let (s, _) = token1("addi")(s)?;
        let (s, elem) = field_element(s)?;
        Ok((s, AnInstruction::AddI(elem)))
    }
}

fn divine_instruction() -> impl Fn(&str) -> ParseResult<AnInstruction<String>> {
    move |s: &str| {
        let (s, _) = token1("divine")(s)?;
        let (s, arg) = number_of_words(s)?;
        Ok((s, AnInstruction::Divine(arg)))
    }
}

fn pick_instruction() -> impl Fn(&str) -> ParseResult<AnInstruction<String>> {
    move |s: &str| {
        let (s, _) = token1("pick")(s)?;
        let (s, arg) = stack_register(s)?;
        Ok((s, AnInstruction::Pick(arg)))
    }
}

fn place_instruction() -> impl Fn(&str) -> ParseResult<AnInstruction<String>> {
    move |s: &str| {
        let (s, _) = token1("place")(s)?;
        let (s, arg) = stack_register(s)?;
        Ok((s, AnInstruction::Place(arg)))
    }
}

fn dup_instruction() -> impl Fn(&str) -> ParseResult<AnInstruction<String>> {
    move |s: &str| {
        let (s, _) = token1("dup")(s)?; // require space before argument
        let (s, stack_register) = stack_register(s)?;
        Ok((s, AnInstruction::Dup(stack_register)))
    }
}

fn swap_instruction() -> impl Fn(&str) -> ParseResult<AnInstruction<String>> {
    move |s: &str| {
        let (s, _) = token1("swap")(s)?;
        let (s, stack_register) = stack_register(s)?;
        Ok((s, AnInstruction::Swap(stack_register)))
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
        let (s, label) = alt((call_syntax, bracket_syntax))(s)?;

        // This check cannot be moved into `label_addr`, since `label_addr` is shared
        // between the scenarios `<label>:` and `call <label>`; the former requires
        // parsing the `:` before rejecting a possible instruction name in the label.
        if is_illegal_label(&label) {
            let failure_reason = "label must be neither instruction nor keyword";
            return cut(nom::error::context(failure_reason, fail))(s);
        }

        Ok((s, AnInstruction::Call(label)))
    }
}

fn read_mem_instruction() -> impl Fn(&str) -> ParseResult<AnInstruction<String>> {
    move |s: &str| {
        let (s, _) = token1("read_mem")(s)?;
        let (s, arg) = number_of_words(s)?;
        Ok((s, AnInstruction::ReadMem(arg)))
    }
}

fn write_mem_instruction() -> impl Fn(&str) -> ParseResult<AnInstruction<String>> {
    move |s: &str| {
        let (s, _) = token1("write_mem")(s)?;
        let (s, arg) = number_of_words(s)?;
        Ok((s, AnInstruction::WriteMem(arg)))
    }
}

fn read_io_instruction() -> impl Fn(&str) -> ParseResult<AnInstruction<String>> {
    move |s: &str| {
        let (s, _) = token1("read_io")(s)?;
        let (s, arg) = number_of_words(s)?;
        Ok((s, AnInstruction::ReadIo(arg)))
    }
}

fn write_io_instruction() -> impl Fn(&str) -> ParseResult<AnInstruction<String>> {
    move |s: &str| {
        let (s, _) = token1("write_io")(s)?;
        let (s, arg) = number_of_words(s)?;
        Ok((s, AnInstruction::WriteIo(arg)))
    }
}

fn field_element(s_orig: &str) -> ParseResult<BFieldElement> {
    let (s, negative) = opt(token0("-"))(s_orig)?;
    let (s, n) = digit1(s)?;
    let (s, _) = comment_or_whitespace1(s)?;

    let Ok(mut n): Result<i128, _> = n.parse() else {
        return nom::error::context("out-of-bounds constant", fail)(s);
    };

    let quotient = i128::from(BFieldElement::P);
    if n >= quotient {
        return nom::error::context("out-of-bounds constant", fail)(s_orig);
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
        "0" => OpStackElement::ST0,
        "1" => OpStackElement::ST1,
        "2" => OpStackElement::ST2,
        "3" => OpStackElement::ST3,
        "4" => OpStackElement::ST4,
        "5" => OpStackElement::ST5,
        "6" => OpStackElement::ST6,
        "7" => OpStackElement::ST7,
        "8" => OpStackElement::ST8,
        "9" => OpStackElement::ST9,
        "10" => OpStackElement::ST10,
        "11" => OpStackElement::ST11,
        "12" => OpStackElement::ST12,
        "13" => OpStackElement::ST13,
        "14" => OpStackElement::ST14,
        "15" => OpStackElement::ST15,
        _ => {
            let failure_reason = "using an out-of-bounds stack register (0-15 exist)";
            return nom::error::context(failure_reason, fail)(s);
        }
    };
    let (s, _) = comment_or_whitespace1(s)?;

    Ok((s, stack_register))
}

fn number_of_words(s: &str) -> ParseResult<NumberOfWords> {
    let (s, n) = digit1(s)?;
    let arg = match n {
        "1" => NumberOfWords::N1,
        "2" => NumberOfWords::N2,
        "3" => NumberOfWords::N3,
        "4" => NumberOfWords::N4,
        "5" => NumberOfWords::N5,
        _ => return nom::error::context("using an out-of-bounds argument (1-5 allowed)", fail)(s),
    };
    let (s, _) = comment_or_whitespace1(s)?; // require space after element

    Ok((s, arg))
}

/// Parse a label address. This is used in "`<label>:`" and in "`call <label>`".
fn label_addr(s_orig: &str) -> ParseResult<String> {
    let (s, addr_part_0) = take_while1(is_label_start_char)(s_orig)?;
    if addr_part_0.is_empty() {
        // todo: this error is never shown to the user, since the `label` parser is wrapped in an
        //  `alt`. With a custom error type, it is possible to have alt return the error of the
        //  parser that went the farthest in the input data.
        let failure_reason = "label must start with an alphabetic character or underscore";
        return nom::error::context(failure_reason, fail)(s_orig);
    }
    let (s, addr_part_1) = take_while(is_label_char)(s)?;

    Ok((s, format!("{addr_part_0}{addr_part_1}")))
}

fn is_label_start_char(c: char) -> bool {
    c.is_alphabetic() || c == '_'
}

fn is_label_char(c: char) -> bool {
    is_label_start_char(c) || c.is_numeric() || c == '-'
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

/// Parse whitespace characters (can be none)
fn whitespace0(s: &str) -> ParseResult<()> {
    let (s, _) = take_while(|c: char| c.is_whitespace())(s)?;
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

/// Parse one type hint.
///
/// Type hints look like this:
///
/// ```text
/// hint <variable_name>[: <type_name>] = stack\[<range_start>[..<range_end>]\]
/// ```
fn type_hint(s_type_hint: &str) -> ParseResult<InstructionToken> {
    let (s, _) = token1("hint")(s_type_hint)?;
    let (s, variable_name) = type_hint_variable_name(s)?;
    let (s, type_name) = type_hint_type_name(s)?;
    let (s, range_start) = type_hint_starting_index(s)?;
    let (s, maybe_range_end) = type_hint_ending_index(s)?;
    let (s, _) = whitespace0(s)?;
    let (s, _) = token0("]")(s)?;

    let length = match maybe_range_end {
        Some(range_end) if range_end <= range_start => {
            let failure_reason = "range end must be greater than range start";
            return cut(nom::error::context(failure_reason, fail))(s);
        }
        Some(range_end) => range_end - range_start,
        None => 1,
    };

    let type_hint = TypeHint {
        starting_index: range_start,
        length,
        type_name,
        variable_name,
    };

    Ok((s, InstructionToken::TypeHint(type_hint, s_type_hint)))
}

fn type_hint_variable_name(s: &str) -> ParseResult<String> {
    let (s, variable_name_start) = take_while1(is_type_hint_variable_name_start_character)(s)?;
    let (s, variable_name_end) = take_while(is_type_hint_variable_name_character)(s)?;
    let (s, _) = whitespace0(s)?;
    let variable_name = format!("{variable_name_start}{variable_name_end}");
    Ok((s, variable_name))
}

fn type_hint_type_name(s: &str) -> ParseResult<Option<String>> {
    let Ok((s, _)) = token0(":")(s) else {
        return Ok((s, None));
    };

    let (s, type_name_start) = take_while1(is_type_hint_type_name_start_character)(s)?;
    let (s, type_name_end) = take_while(is_type_hint_type_name_character)(s)?;
    let (s, _) = whitespace0(s)?;
    let type_name = format!("{type_name_start}{type_name_end}");

    Ok((s, Some(type_name)))
}

fn type_hint_ending_index(s: &str) -> ParseResult<Option<usize>> {
    let Ok((s, _)) = token0("..")(s) else {
        return Ok((s, None));
    };

    let (s, range_end) = take_while(|c: char| c.is_numeric())(s)?;
    let (_, range_end) = parse_str_to_usize(range_end)?;

    Ok((s, Some(range_end)))
}

fn type_hint_starting_index(s: &str) -> ParseResult<usize> {
    let (s, _) = token0("=")(s)?;
    let (s, _) = token0("stack[")(s)?;
    let (s, range_start) = take_while(|c: char| c.is_numeric())(s)?;
    let (_, range_start) = parse_str_to_usize(range_start)?;
    let (s, _) = whitespace0(s)?;
    Ok((s, range_start))
}

fn is_type_hint_variable_name_start_character(c: char) -> bool {
    (c.is_alphabetic() && c.is_lowercase()) || c == '_'
}

fn is_type_hint_variable_name_character(c: char) -> bool {
    is_type_hint_variable_name_start_character(c) || c.is_numeric()
}

fn is_type_hint_type_name_start_character(c: char) -> bool {
    c.is_alphabetic()
}

fn is_type_hint_type_name_character(c: char) -> bool {
    is_type_hint_type_name_start_character(c) || c.is_numeric() || c == '_'
}

fn parse_str_to_usize(s: &str) -> ParseResult<usize> {
    match s.parse::<usize>() {
        Ok(u) => Ok((s, u)),
        Err(_) => cut(nom::error::context("integer conversion failure", fail))(s),
    }
}

fn assertion_context(s_ctx: &str) -> ParseResult<InstructionToken> {
    let (s, assertion_context) = assertion_context_id(s_ctx)?;
    let assertion_context = InstructionToken::AssertionContext(assertion_context, s_ctx);

    Ok((s, assertion_context))
}

fn assertion_context_id(s_ctx: &str) -> ParseResult<AssertionContext> {
    let (s, _) = token1("error_id")(s_ctx)?;
    let (s, id) = nom::character::complete::i128(s)?;
    let (s, _) = comment_or_whitespace1(s)?;

    let assertion_context = AssertionContext::ID(id);
    Ok((s, assertion_context))
}

pub(crate) fn build_label_to_address_map(program: &[LabelledInstruction]) -> HashMap<String, u64> {
    let mut label_map = HashMap::new();
    let mut instruction_pointer = 0;

    for labelled_instruction in program {
        if let LabelledInstruction::Instruction(instruction) = labelled_instruction {
            instruction_pointer += instruction.size() as u64;
            continue;
        }

        let LabelledInstruction::Label(label) = labelled_instruction else {
            continue;
        };
        let Entry::Vacant(new_label_map_entry) = label_map.entry(label.clone()) else {
            panic!("Duplicate label: {label}");
        };
        new_label_map_entry.insert(instruction_pointer);
    }

    label_map
}

pub(crate) fn turn_labels_into_addresses(
    labelled_instructions: &[LabelledInstruction],
    label_to_address: &HashMap<String, u64>,
) -> Vec<Instruction> {
    fn turn_label_to_address_for_instruction(
        labelled_instruction: &LabelledInstruction,
        label_map: &HashMap<String, u64>,
    ) -> Option<Instruction> {
        let LabelledInstruction::Instruction(instruction) = labelled_instruction else {
            return None;
        };

        let instruction_with_absolute_address =
            instruction.map_call_address(|label| address_for_label(label, label_map));
        Some(instruction_with_absolute_address)
    }

    fn address_for_label(label: &str, label_map: &HashMap<String, u64>) -> BFieldElement {
        let maybe_address = label_map.get(label).map(|&a| bfe!(a));
        maybe_address.unwrap_or_else(|| panic!("Label not found: {label}"))
    }

    labelled_instructions
        .iter()
        .filter_map(|inst| turn_label_to_address_for_instruction(inst, label_to_address))
        .flat_map(|inst| vec![inst; inst.size()])
        .collect()
}

#[cfg(test)]
pub(crate) mod tests {
    use assert2::assert;
    use assert2::let_assert;
    use itertools::Itertools;
    use proptest::prelude::*;
    use proptest_arbitrary_interop::arb;
    use rand::prelude::*;
    use rand::Rng;
    use strum::EnumCount;
    use test_strategy::proptest;
    use test_strategy::Arbitrary;
    use twenty_first::bfe;
    use twenty_first::prelude::Digest;

    use crate::triton_asm;
    use crate::triton_instr;
    use crate::triton_program;

    use super::*;

    #[must_use]
    struct TestCase<'a> {
        input: &'a str,
        expected: Vec<Instruction>,
        message: &'static str,
    }

    #[must_use]
    struct NegativeTestCase<'a> {
        input: &'a str,
        expected_error: &'static str,
        expected_error_count: usize,
        message: &'static str,
    }

    impl<'a> TestCase<'a> {
        fn run(&self) {
            let message = self.message;
            let parse_result = parse(self.input).map_err(|err| format!("{message}:\n{err}"));
            let_assert!(Ok(actual) = parse_result);

            let labelled_instructions = to_labelled_instructions(&actual);
            let label_to_address = build_label_to_address_map(&labelled_instructions);
            let instructions =
                turn_labels_into_addresses(&labelled_instructions, &label_to_address);
            assert!(self.expected == instructions, "{message}");
        }
    }

    impl<'a> NegativeTestCase<'a> {
        fn run(self) {
            let Self {
                input,
                expected_error,
                expected_error_count,
                message,
            } = self;

            let parse_result = parse(input);
            if let Ok(instructions) = parse_result {
                eprintln!("parser input: {input}");
                eprintln!("parser output: {instructions:?}");
                panic!("parser should fail, but didn't: {message}");
            }

            let actual_error = parse_result.map_err(|e| e.to_string()).unwrap_err();
            let actual_error_count = actual_error.match_indices(expected_error).count();
            if expected_error_count != actual_error_count {
                eprintln!("Expected error message ({expected_error_count} times):");
                eprintln!("{expected_error}");
                eprintln!("Actual error message (found {actual_error_count} times):");
                eprintln!("{actual_error}");
                panic!("Additional context: {message}");
            }
        }
    }

    #[test]
    fn parse_program_empty() {
        TestCase {
            input: "",
            expected: vec![],
            message: "empty string should parse as empty program",
        }
        .run();

        TestCase {
            input: "   ",
            expected: vec![],
            message: "spaces should parse as empty program",
        }
        .run();

        TestCase {
            input: "\n",
            expected: vec![],
            message: "linebreaks should parse as empty program (1)",
        }
        .run();

        TestCase {
            input: "   \n ",
            expected: vec![],
            message: "linebreaks should parse as empty program (2)",
        }
        .run();

        TestCase {
            input: "   \n \n",
            expected: vec![],
            message: "linebreaks should parse as empty program (3)",
        }
        .run();

        TestCase {
            input: "// empty program",
            expected: vec![],
            message: "single comment should parse as empty program",
        }
        .run();

        TestCase {
            input: "// empty program\n",
            expected: vec![],
            message: "single comment with linebreak should parse as empty program",
        }
        .run();

        TestCase {
            input: "// multi-line\n// comment",
            expected: vec![],
            message: "multiple comments should parse as empty program",
        }
        .run();

        TestCase {
            input: "// multi-line\n// comment\n ",
            expected: vec![],
            message: "multiple comments with trailing whitespace should parse as empty program",
        }
        .run();
    }

    #[proptest]
    fn arbitrary_whitespace_and_comment_sequence_is_empty_program(whitespace: Vec<Whitespace>) {
        let whitespace = whitespace.into_iter().join("");
        TestCase {
            input: &whitespace,
            expected: vec![],
            message: "arbitrary whitespace should parse as empty program",
        }
        .run();
    }

    #[derive(Debug, Copy, Clone, Eq, PartialEq, Hash, EnumCount, Arbitrary)]
    enum Whitespace {
        Space,
        Tab,
        CarriageReturn,
        LineFeed,
        CarriageReturnLineFeed,
        Comment,
    }

    impl Display for Whitespace {
        fn fmt(&self, f: &mut Formatter<'_>) -> FmtResult {
            match self {
                Self::Space => write!(f, " "),
                Self::Tab => write!(f, "\t"),
                Self::CarriageReturn => write!(f, "\r"),
                Self::LineFeed => writeln!(f),
                Self::CarriageReturnLineFeed => writeln!(f, "\r"),
                Self::Comment => writeln!(f, " // comment"),
            }
        }
    }

    #[test]
    fn parse_program_whitespace() {
        NegativeTestCase {
            input: "poppop",
            expected_error: "n/a",
            expected_error_count: 0,
            message: "whitespace required between instructions (pop)",
        }
        .run();

        NegativeTestCase {
            input: "dup 0dup 0",
            expected_error: "n/a",
            expected_error_count: 0,
            message: "whitespace required between instructions (dup 0)",
        }
        .run();

        NegativeTestCase {
            input: "swap 10swap 10",
            expected_error: "n/a",
            expected_error_count: 0,
            message: "whitespace required between instructions (swap 10)",
        }
        .run();

        NegativeTestCase {
            input: "push10",
            expected_error: "n/a",
            expected_error_count: 0,
            message: "push requires whitespace before its constant",
        }
        .run();

        NegativeTestCase {
            input: "push 10pop",
            expected_error: "n/a",
            expected_error_count: 0,
            message: "push requires whitespace after its constant",
        }
        .run();

        NegativeTestCase {
            input: "hello: callhello",
            expected_error: "n/a",
            expected_error_count: 0,
            message: "call requires whitespace before its label",
        }
        .run();

        NegativeTestCase {
            input: "hello: popcall hello",
            expected_error: "n/a",
            expected_error_count: 0,
            message: "required space between pop and call",
        }
        .run();
    }

    #[test]
    fn parse_program_label() {
        TestCase {
            input: "foo: call foo",
            expected: vec![Instruction::Call(bfe!(0)), Instruction::Call(bfe!(0))],
            message: "parse labels and calls to labels",
        }
        .run();

        TestCase {
            input: "foo:call foo",
            expected: vec![Instruction::Call(bfe!(0)), Instruction::Call(bfe!(0))],
            message: "whitespace is not required after 'label:'",
        }
        .run();

        // FIXME: Increase coverage of negative tests for duplicate labels.
        NegativeTestCase {
            input: "foo: pop 1 foo: pop 1 call foo",
            expected_error: "duplicate label",
            expected_error_count: 2,
            message: "labels cannot occur twice",
        }
        .run();

        // FIXME: Increase coverage of negative tests for missing labels.
        NegativeTestCase {
            input: "foo: pop 1 call herp call derp",
            expected_error: "missing label",
            expected_error_count: 2,
            message: "non-existent labels cannot be called",
        }
        .run();

        // FIXME: Increase coverage of negative tests for label/keyword overlap.
        NegativeTestCase {
            input: "pop: call pop",
            expected_error: "label must be neither instruction nor keyword",
            expected_error_count: 1,
            message: "label names may not overlap with instruction names",
        }
        .run();

        TestCase {
            input: "pops: call pops",
            expected: vec![Instruction::Call(bfe!(0)), Instruction::Call(bfe!(0))],
            message: "labels that share a common prefix with instruction are labels",
        }
        .run();

        TestCase {
            input: "_call: call _call",
            expected: vec![Instruction::Call(bfe!(0)), Instruction::Call(bfe!(0))],
            message: "labels that share a common suffix with instruction are labels",
        }
        .run();
    }

    #[test]
    fn parse_program_nonexistent_instructions() {
        NegativeTestCase {
            input: "pop 0",
            expected_error: "expecting label, instruction or eof",
            expected_error_count: 1,
            message: "instruction `pop` cannot take argument `0`",
        }
        .run();

        NegativeTestCase {
            input: "swap 16",
            expected_error: "expecting label, instruction or eof",
            expected_error_count: 1,
            message: "there is no swap 16 instruction",
        }
        .run();

        NegativeTestCase {
            input: "dup 16",
            expected_error: "expecting label, instruction or eof",
            expected_error_count: 1,
            message: "there is no dup 16 instruction",
        }
        .run();
    }

    #[test]
    fn parse_program_bracket_syntax() {
        TestCase {
            input: "foo: [foo]",
            expected: vec![Instruction::Call(bfe!(0)), Instruction::Call(bfe!(0))],
            message: "Handle brackets as call syntax sugar",
        }
        .run();

        NegativeTestCase {
            input: "foo: [bar]",
            expected_error: "missing label",
            expected_error_count: 1,
            message: "Handle missing labels with bracket syntax",
        }
        .run();
    }

    #[test]
    fn parse_program_label_must_start_with_alphabetic_character_or_underscore() {
        NegativeTestCase {
            input: "1foo: call 1foo",
            expected_error: "expecting label, instruction or eof",
            expected_error_count: 1,
            message: "labels cannot start with a digit",
        }
        .run();

        NegativeTestCase {
            input: "-foo: call -foo",
            expected_error: "expecting label, instruction or eof",
            expected_error_count: 1,
            message: "labels cannot start with a dash",
        }
        .run();

        TestCase {
            input: "_foo: call _foo",
            expected: vec![Instruction::Call(bfe!(0)), Instruction::Call(bfe!(0))],
            message: "labels can start with an underscore",
        }
        .run();
    }

    #[test]
    fn parse_type_hint_with_zero_length_range() {
        NegativeTestCase {
            input: "hint foo: Type = stack[0..0]",
            expected_error: "range end must be greater than range start",
            expected_error_count: 1,
            message: "parse type hint with zero-length range",
        }
        .run();
    }

    #[test]
    fn parse_type_hint_with_range_and_offset_and_missing_closing_bracket() {
        NegativeTestCase {
            input: "hint foo: Type = stack[2..5",
            expected_error: "expecting label, instruction or eof",
            expected_error_count: 1,
            message: "parse type hint with range and offset and missing closing bracket",
        }
        .run();
    }

    #[test]
    fn parse_type_hint_with_range_and_offset_and_missing_opening_bracket() {
        NegativeTestCase {
            input: "hint foo: Type = stack2..5]",
            expected_error: "expecting label, instruction or eof",
            expected_error_count: 1,
            message: "parse type hint with range and offset and missing opening bracket",
        }
        .run();
    }

    #[test]
    fn parse_type_hint_with_range_and_offset_and_missing_equals_sign() {
        NegativeTestCase {
            input: "hint foo: Type stack[2..5];",
            expected_error: "expecting label, instruction or eof",
            expected_error_count: 1,
            message: "parse type hint with range and offset and missing equals sign",
        }
        .run();
    }

    #[test]
    fn parse_type_hint_with_range_and_offset_and_missing_type_name() {
        NegativeTestCase {
            input: "hint foo: = stack[2..5]",
            expected_error: "expecting label, instruction or eof",
            expected_error_count: 1,
            message: "parse type hint with range and offset and missing type name",
        }
        .run();
    }

    #[test]
    fn parse_type_hint_with_range_and_offset_and_missing_variable_name() {
        NegativeTestCase {
            input: "hint : Type = stack[2..5]",
            expected_error: "label must be neither instruction nor keyword",
            expected_error_count: 1,
            message: "parse type hint with range and offset and missing variable name",
        }
        .run();
    }

    #[test]
    fn parse_type_hint_with_range_and_offset_and_missing_colon() {
        NegativeTestCase {
            input: "hint foo Type = stack[2..5]",
            expected_error: "expecting label, instruction or eof",
            expected_error_count: 1,
            message: "parse type hint with range and offset and missing colon",
        }
        .run();
    }

    #[test]
    fn parse_type_hint_with_range_and_offset_and_missing_hint() {
        NegativeTestCase {
            input: "foo: Type = stack[2..5];",
            expected_error: "expecting label, instruction or eof",
            expected_error_count: 1,
            message: "parse type hint with range and offset and missing hint",
        }
        .run();
    }

    #[test]
    fn parse_simple_assertion_contexts() {
        TestCase {
            input: "assert",
            expected: vec![Instruction::Assert],
            message: "naked assert",
        }
        .run();
        TestCase {
            input: "assert_vector",
            expected: vec![Instruction::AssertVector],
            message: "naked assert_vector",
        }
        .run();

        TestCase {
            input: "assert error_id 42",
            expected: vec![Instruction::Assert],
            message: "assert, then id",
        }
        .run();
        TestCase {
            input: "assert_vector error_id 42",
            expected: vec![Instruction::AssertVector],
            message: "assert_vector, then id",
        }
        .run();

        TestCase {
            input: "assert error_id -42",
            expected: vec![Instruction::Assert],
            message: "assert, then negative id",
        }
        .run();
        TestCase {
            input: "assert_vector error_id -42",
            expected: vec![Instruction::AssertVector],
            message: "assert_vector, then negative id",
        }
        .run();

        TestCase {
            input: "assert error_id 42 nop",
            expected: vec![Instruction::Assert, Instruction::Nop],
            message: "assert, then id, then nop",
        }
        .run();
        TestCase {
            input: "assert_vector error_id 42 nop",
            expected: vec![Instruction::AssertVector, Instruction::Nop],
            message: "assert_vector, then id, then nop",
        }
        .run();
    }

    #[test]
    fn assertion_context_error_id_can_handle_edge_case_ids() {
        let instructions = [
            ("assert", Instruction::Assert),
            ("assert_vector", Instruction::AssertVector),
        ];
        let ids = [i128::MIN, -1, 0, 1, i128::MAX];

        for ((instruction_str, instruction), id) in instructions.into_iter().cartesian_product(ids)
        {
            TestCase {
                input: &format!("{instruction_str} error_id {id}"),
                expected: vec![instruction],
                message: "assert, then edge case id",
            }
            .run();
        }
    }

    #[test]
    fn assertion_context_error_id_fails_on_too_large_error_ids() {
        NegativeTestCase {
            input: "assert error_id -170141183460469231731687303715884105729",
            expected_error: "expecting label, instruction or eof",
            expected_error_count: 1,
            message: "error id smaller than i128::MIN",
        }
        .run();
        NegativeTestCase {
            input: "assert error_id 170141183460469231731687303715884105728",
            expected_error: "expecting label, instruction or eof",
            expected_error_count: 1,
            message: "error id larger than i128::MAX",
        }
        .run();
    }

    #[proptest]
    fn assertion_context_error_id_can_handle_any_valid_id(id: i128) {
        TestCase {
            input: &format!("assert error_id {id}"),
            expected: vec![Instruction::Assert],
            message: "assert, then random id",
        }
        .run();
    }

    #[test]
    fn parse_erroneous_assertion_contexts_for_assert() {
        NegativeTestCase {
            input: "assert error_id",
            expected_error: "expecting label, instruction or eof",
            expected_error_count: 1,
            message: "missing id after keyword `error_id`",
        }
        .run();
        NegativeTestCase {
            input: "assert error id 42",
            expected_error: "expecting label, instruction or eof",
            expected_error_count: 1,
            message: "incorrect keyword `error id`",
        }
        .run();
        NegativeTestCase {
            input: "error_id 42",
            expected_error: "incorrectly placed assertion context",
            expected_error_count: 1,
            message: "standalone assertion context",
        }
        .run();
        NegativeTestCase {
            input: "error_id 42 error_id 42",
            expected_error: "incorrectly placed assertion context",
            expected_error_count: 2,
            message: "multiple standalone assertion contexts",
        }
        .run();
        NegativeTestCase {
            input: "nop error_id 42",
            expected_error: "incorrectly placed assertion context",
            expected_error_count: 1,
            message: "error id without assertion",
        }
        .run();
        NegativeTestCase {
            input: "assert error_id 42 error_id 1337",
            expected_error: "incorrectly placed assertion context",
            expected_error_count: 1,
            message: "multiple error ids for the same assertion",
        }
        .run();
        NegativeTestCase {
            input: "assert_vector error_id 42 error_id 1337",
            expected_error: "incorrectly placed assertion context",
            expected_error_count: 1,
            message: "multiple error ids for the same vector assertion",
        }
        .run();
        NegativeTestCase {
            input: "error_id 42 assert",
            expected_error: "incorrectly placed assertion context",
            expected_error_count: 1,
            message: "error id precedes assertion",
        }
        .run();
        NegativeTestCase {
            input: "error_id 42 assert error_id 1337",
            expected_error: "incorrectly placed assertion context",
            expected_error_count: 1,
            message: "first error id precedes assertion",
        }
        .run();
    }

    #[proptest]
    fn assertion_context_error_id_fails_for_invalid_id(
        #[strategy(proptest::strategy::Union::new(["assert", "assert_vector"]))]
        instruction: String,
        #[filter(#id.parse::<i128>().is_err())] id: String,
    ) {
        // not using `NegativeTest`: different `id`s trigger different errors
        let input = format!("{instruction} error_id {id}");
        prop_assert!(parse(&input).is_err());
    }

    #[proptest]
    fn type_hint_to_string_to_type_hint_is_identity(#[strategy(arb())] type_hint: TypeHint) {
        let type_hint_string = type_hint.to_string();
        let instruction_tokens =
            parse(&type_hint_string).map_err(|err| TestCaseError::Fail(format!("{err}").into()))?;
        let labelled_instructions = to_labelled_instructions(&instruction_tokens);
        prop_assert_eq!(1, labelled_instructions.len());
        let first_labelled_instruction = labelled_instructions[0].clone();
        let_assert!(LabelledInstruction::TypeHint(parsed_type_hint) = first_labelled_instruction);
        prop_assert_eq!(type_hint, parsed_type_hint);
    }

    #[proptest]
    fn assertion_context_to_string_to_assertion_context_is_identity(
        #[strategy(arb())] context: AssertionContext,
    ) {
        let assert_with_context = format!("assert {context}");
        let instruction_tokens = parse(&assert_with_context)
            .map_err(|err| TestCaseError::Fail(format!("{err}").into()))?;
        let labelled_instructions = to_labelled_instructions(&instruction_tokens);
        prop_assert_eq!(2, labelled_instructions.len());
        let_assert!(LabelledInstruction::AssertionContext(parsed) = &labelled_instructions[1]);
        prop_assert_eq!(&context, parsed);
    }

    #[test]
    fn triton_asm_macro() {
        let instructions = triton_asm!(write_io 3 push 17 call huh lt swap 3);
        assert_eq!(
            LabelledInstruction::Instruction(AnInstruction::WriteIo(NumberOfWords::N3)),
            instructions[0]
        );
        assert_eq!(
            LabelledInstruction::Instruction(AnInstruction::Push(bfe!(17))),
            instructions[1]
        );
        assert_eq!(
            LabelledInstruction::Instruction(AnInstruction::Call("huh".to_string())),
            instructions[2]
        );
        assert_eq!(
            LabelledInstruction::Instruction(AnInstruction::Lt),
            instructions[3]
        );
        assert_eq!(
            LabelledInstruction::Instruction(AnInstruction::Swap(OpStackElement::ST3)),
            instructions[4]
        );
    }

    #[test]
    fn triton_asm_macro_with_a_single_return() {
        let instructions = triton_asm!(return);
        assert_eq!(
            LabelledInstruction::Instruction(AnInstruction::Return),
            instructions[0]
        );
    }

    #[test]
    fn triton_asm_macro_with_a_single_assert() {
        let instructions = triton_asm!(assert);
        assert_eq!(
            LabelledInstruction::Instruction(AnInstruction::Assert),
            instructions[0]
        );
    }

    #[test]
    fn triton_asm_macro_with_only_assert_and_return() {
        let instructions = triton_asm!(assert return);
        assert_eq!(
            LabelledInstruction::Instruction(AnInstruction::Assert),
            instructions[0]
        );
        assert_eq!(
            LabelledInstruction::Instruction(AnInstruction::Return),
            instructions[1]
        );
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
        let instruction_push =
            LabelledInstruction::Instruction(AnInstruction::Push(push_arg.into()));
        let swap_argument = "1";
        triton_program!({instruction_push} push {push_arg} swap {swap_argument} eq assert halt);
    }

    #[test]
    fn triton_instruction_macro_parses_simple_instructions() {
        assert_eq!(
            LabelledInstruction::Instruction(AnInstruction::Halt),
            triton_instr!(halt)
        );
        assert_eq!(
            LabelledInstruction::Instruction(AnInstruction::Add),
            triton_instr!(add)
        );
        assert_eq!(
            LabelledInstruction::Instruction(AnInstruction::Pop(NumberOfWords::N3)),
            triton_instr!(pop 3)
        );
    }

    #[test]
    #[should_panic(expected = "not_an_instruction")]
    fn triton_instruction_macro_fails_when_encountering_unknown_instruction() {
        triton_instr!(not_an_instruction);
    }

    #[test]
    fn triton_instruction_macro_parses_instructions_with_argument() {
        assert_eq!(
            LabelledInstruction::Instruction(AnInstruction::Push(bfe!(7))),
            triton_instr!(push 7)
        );
        assert_eq!(
            LabelledInstruction::Instruction(AnInstruction::Dup(OpStackElement::ST3)),
            triton_instr!(dup 3)
        );
        assert_eq!(
            LabelledInstruction::Instruction(AnInstruction::Swap(OpStackElement::ST5)),
            triton_instr!(swap 5)
        );
        assert_eq!(
            LabelledInstruction::Instruction(AnInstruction::Call("my_label".to_string())),
            triton_instr!(call my_label)
        );
    }

    #[test]
    fn triton_asm_macro_can_repeat_instructions() {
        let instructions = triton_asm![push 42; 3];
        let expected_instructions =
            vec![LabelledInstruction::Instruction(AnInstruction::Push(bfe!(42))); 3];
        assert_eq!(expected_instructions, instructions);

        let instructions = triton_asm![read_io 2; 15];
        let expected_instructions =
            vec![LabelledInstruction::Instruction(AnInstruction::ReadIo(NumberOfWords::N2)); 15];
        assert_eq!(expected_instructions, instructions);

        let instructions = triton_asm![divine 3; Digest::LEN];
        let expected_instructions =
            vec![
                LabelledInstruction::Instruction(AnInstruction::Divine(NumberOfWords::N3));
                Digest::LEN
            ];
        assert_eq!(expected_instructions, instructions);
    }

    #[test]
    fn break_gets_turned_into_labelled_instruction() {
        let instructions = triton_asm![break];
        let expected_instructions = vec![LabelledInstruction::Breakpoint];
        assert_eq!(expected_instructions, instructions);
    }

    #[test]
    fn break_does_not_propagate_to_full_program() {
        let program = triton_program! { break halt break };
        assert_eq!(1, program.len_bwords());
    }
}
