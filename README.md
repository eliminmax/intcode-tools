<!--
SPDX-FileCopyrightText: 2025 - 2026 Eli Array Minkoff

SPDX-License-Identifier: 0BSD
-->

# intcode

A library built around my Intcode module from Advent of Code 2019, cleaned up and reorganized, with additions to support an Intcode assembly language I've dubbed IAL.

**It is in a state of active tweaking with regular breaking changes, and does not yet follow SemVer**

See [ASM.md](./ASM) for documentation of IAL's syntax and semantics.

It's organized as a library which provides the `intcode::Interpreter` struct as a way to interact with intcode, an optional `intcode::asm` module which provides an implementation of the assembly language from the Esolang wiki, and a few small binaries that make use of those: `intcode_as` is an assembler, and `intcode_ascii` provides an interactive interface using [Aft Scaffolding Control and Information Interface](https://adventofcode.com/2019/day/17) *(not to be confused with the [American Standard Code for Information Interchange](https://en.wikipedia.org/wiki/ASCII))*.

The interpreter is fully functional, with all of the [opcodes](https://esolangs.org/wiki/Intcode#Opcodes) and [parameter modes](https://esolangs.org/wiki/Intcode#Parameter_Modes) defined in the completed Intcode computer for [Advent of Code 2019 Day 9](https://adventofcode.com/2019/day/9).

## General Examples

```rust
use intcode::prelude::*;
let mut interpreter = Interpreter::new(vec![104, 1024, 99]);

assert_eq!(
    interpreter.run_through_inputs(std::iter::empty()).unwrap(),
    (vec![1024], State::Halted)
);
```

Additionally, if the `asm` feature is enabled, tools to work with a minimal assembly language
for Intcode are provided in the `asm`.

```rust
use intcode::{prelude::*, asm::assemble};
const ASM: &str = r#"
OUT #1024
HALT
"#;

let assembled = assemble(ASM).unwrap();
assert_eq!(assembled, vec![104, 1024, 99]);

let mut interpreter = Interpreter::new(assembled);
assert_eq!(
    interpreter.run_through_inputs(std::iter::empty()).unwrap(),
    (vec![1024], State::Halted)
);
```

## Code and Documentation Provenance

All code and documentation in this repository is 100% written by me, Eli Array Minkoff, a human being with a cybersecurity degree and an insatiable desire to continuously expand my understanding of computers and computer programming.

I have not used any LLM tools—beyond the widely-discussed ethical and copyright concerns, I program for the joy of learning, and the thrill of seeing my creations come to fruition, and outsourcing the work would undermine that.
