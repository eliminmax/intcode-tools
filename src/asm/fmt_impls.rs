// SPDX-FileCopyrightText: 2025 - 2026 Eli Array Minkoff
//
// SPDX-License-Identifier: 0BSD

use chumsky::span::Spanned;

use super::debug_encode::DebugInfoReadError;
use super::{AssemblyError, BinOperator, Directive, Expr, Instr, Line, Parameter};

use std::error::Error;
use std::fmt::{self, Display};

impl Display for BinOperator {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            BinOperator::Add => write!(f, "+"),
            BinOperator::Sub => write!(f, "-"),
            BinOperator::Mul => write!(f, "*"),
            BinOperator::Div => write!(f, "/"),
        }
    }
}

impl Display for Expr<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Expr::Number(n) => write!(f, "{n}"),
            Expr::Ident(id) => write!(f, "{id}"),
            Expr::BinOp { lhs, op, rhs } => {
                write!(f, "{} {} {}", lhs.inner, op.inner, rhs.inner,)
            }
            Expr::Negate(e) => write!(f, "-{}", e.inner),
            Expr::UnaryAdd(e) => write!(f, "+{}", e.inner),
            Expr::Parenthesized(e) => write!(f, "({})", e.inner),
        }
    }
}

impl Display for Parameter<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "{}{}", self.0, self.1.inner)
    }
}

impl Display for Instr<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Instr::Add(a, b, c) => write!(f, "ADD {a}, {b}, {c}"),
            Instr::Mul(a, b, c) => write!(f, "MUL {a}, {b}, {c}"),
            Instr::In(a) => write!(f, "IN {a}"),
            Instr::Out(a) => write!(f, "OUT {a}"),
            Instr::Jnz(a, b) => write!(f, "JNZ {a}, {b}"),
            Instr::Jz(a, b) => write!(f, "JZ {a}, {b}"),
            Instr::Lt(a, b, c) => write!(f, "SLT {a}, {b}, {c}"),
            Instr::Eq(a, b, c) => write!(f, "SEQ {a}, {b}, {c}"),
            Instr::Rbo(a) => write!(f, "INCB {a}"),
            Instr::Halt => write!(f, "HALT"),
        }
    }
}

impl Display for Directive<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Directive::Data(exprs) => {
                write!(f, "DATA ")?;
                if let Some(expr) = exprs.first() {
                    write!(f, "{}", expr.inner)?;
                }
                for expr in &exprs[1..] {
                    write!(f, ", {}", expr.inner)?;
                }
                Ok(())
            }
            Directive::Ascii(spanned) => {
                write!(f, "ASCII {}", spanned.inner.escape_ascii())
            }
            Directive::Instruction(instr) => write!(f, "{instr}"),
        }
    }
}

impl Display for Line<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        for Spanned { inner, .. } in self.labels.iter() {
            write!(f, "{inner}:\t")?
        }
        if let Some(Spanned { inner, .. }) = &self.inner {
            write!(f, "{inner}")?;
        }
        Ok(())
    }
}

impl Display for AssemblyError<'_> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AssemblyError::UnresolvedLabel { label, .. } => {
                write!(f, "unresolved label: {label:?}")
            }
            AssemblyError::DuplicateLabel { label, .. } => write!(f, "duplicate label: {label:?}"),
            AssemblyError::DirectiveTooLarge { size, .. } => {
                write!(
                    f,
                    "directive too large: size {size} is more than maximum {}",
                    i64::MAX
                )
            }
        }
    }
}

impl Error for AssemblyError<'_> {}

impl Display for DebugInfoReadError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        macro_rules! backwards_span {
            ($span_type: literal, $start: ident, $end: ident) => {{
                write!(
                    f,
                    "backwards {} span from {} to {}",
                    $span_type, $start, $end
                )
            }};
        }
        match self {
            DebugInfoReadError::BadMagic(magic) => write!(
                f,
                "bad magic bytes after decompression: {}",
                magic.escape_ascii()
            ),
            DebugInfoReadError::VersionMismatch(version) => {
                write!(f, "unsupported version: {version}")
            }
            DebugInfoReadError::IoError(error) => Display::fmt(error, f),
            DebugInfoReadError::IntSize(try_from_int_error) => Display::fmt(try_from_int_error, f),
            DebugInfoReadError::BadDirectiveByte(byte) => {
                write!(f, "Bad directive byte: 0x{byte:02x}")
            }
            DebugInfoReadError::BackwardsLabelSpan { start, end } => {
                backwards_span!("label", start, end)
            }
            DebugInfoReadError::BackwardsSrcSpan { start, end } => {
                backwards_span!("source", start, end)
            }
            DebugInfoReadError::BackwardsOutSpan { start, end } => {
                backwards_span!("output", start, end)
            }
            DebugInfoReadError::NonUtf8Label(label) => {
                write!(
                    f,
                    "tried to decode a non-utf8 label: {}",
                    label.escape_ascii()
                )
            }
        }
    }
}

impl Error for DebugInfoReadError {}
