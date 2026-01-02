// SPDX-FileCopyrightText: 2025 - 2026 Eli Array Minkoff
//
// SPDX-License-Identifier: 0BSD

use super::ast_util::*;
use super::*;

#[test]
fn parse_blank_line() {
    assert_eq!(
        parse_line().parse("").unwrap(),
        Line {
            label: None,
            inner: None
        }
    )
}

#[test]
fn parse_data() {
    assert_eq!(
        line_inner().parse("DATA 1, 1, 1").unwrap(),
        Some(span(
            Directive::DataDirective(vec![
                span(expr!(1), 5..6),
                span(expr!(1), 8..9),
                span(expr!(1), 11..12),
            ]),
            0..12
        ))
    );
}

#[test]
fn parse_instrs() {
    let parser = instr();
    macro_rules! p {
        ($t: tt $e: tt, $start: expr, $end: expr) => {
            param!($t <expr!($e);>[$start..$end])
        };
        ($e: tt, $start: expr, $end: expr) => {
            param!(<expr!($e);>[$start..$end])
        };
    }
    macro_rules! i {
        [$i: ident] => { Instr::$i };
        [$i: ident ($($params: expr),+)] => { Instr::$i ($($params),+ ) };
    }
    macro_rules! parse {
        ($text: literal) => {
            parser.parse($text).unwrap()
        };
    }
    assert_eq!(
        parse!("ADD #1, @1, 1"),
        i![Add(p!(#1, 4, 6), p!(@1, 8, 10), p!(1, 12, 13))]
    );
    assert_eq!(
        parse!("MUL 3, @20, e"),
        i![Mul(p!(3, 4, 5), p!(@20, 7, 10), p!(e, 12, 13))]
    );
    assert_eq!(parse!("IN #e"), i![In(p!(#e, 3, 5))]);
    assert_eq!(parse!("OUT #5"), i![Out(p!(#5, 4, 6))]);
    assert_eq!(parse!("JNZ @a, #b"), i![Jnz(p!(@a, 4, 6), p!(#b, 8, 10))]);
    assert_eq!(parse!("JZ @a, #b"), i![Jz(p!(@a, 3, 5), p!(#b, 7, 9))]);
    assert_eq!(
        parse!("SLT 1,@1, #5"),
        i![Lt(p!(1, 4, 5), p!(@1, 6, 8), p!(#5, 10, 12))]
    );
    assert_eq!(
        parse!("LT 1,@1, #5"),
        i![Lt(p!(1, 3, 4), p!(@1, 5, 7), p!(#5, 9, 11))]
    );
    assert_eq!(
        parse!("SEQ @3, 32, 1"),
        i![Eq(p!(@3, 4, 6), p!(32, 8, 10), p!(1, 12, 13))]
    );
    assert_eq!(
        parse!("EQ @3, 32, 1"),
        i![Eq(p!(@3, 3, 5), p!(32, 7, 9), p!(1, 11, 12))]
    );
    assert_eq!(parse!("INCB #hello"), i![Rbo(p!(#hello, 5, 11))]);
    assert_eq!(parse!("HALT"), i![Halt]);
}

#[test]
fn parse_exprs() {
    let expr_parse = expr();

    macro_rules! expr_test {
        ($expr: literal, $expected: expr) => {
            let parsed = expr_parse.parse($expr).unwrap().inner;
            assert_eq!(parsed, $expected, "{{ {} }} != {{ {parsed} }}", $expr);
        };
    }

    expr_test!("1", expr!(1));

    expr_test!(
        "1 + 1",
        expr!(
            expr!(1);[0..1] +[2..3] expr!(1);[4..5]
        )
    );

    expr_test!(
        "1 * 1 + 1",
        expr!(
            expr!(
                expr!(1);[0..1] *[2..3] expr!(1);[4..5]
            );[0..5]
            +[6..7]
            expr!(1);[8..9]
        )
    );

    let expected = expr!( (expr!(+expr!(e);[2..3]);[1..3]) );
    expr_test!("(+e)", expected);

    let expected = expr!(
        expr!((
            expr!(
                expr!(1);[1..2]
                +[3..4]
                expr!(
                    +expr!(
                        - expr!(e);[7..8]
                    );[6..8]
                );[5..8]
            );[1..8]
        ));[0..9]
        -[10..11]
        expr!(1);[12..13]
    );
    expr_test!("(1 + +-e) - 1", expected);
}
