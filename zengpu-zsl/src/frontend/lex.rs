//! ZSL lexer — native ZSL source (`&str`) → tokens.
//!
//! Dependency-free: no `syn`, `quote`, or `proc-macro2`. The proc-macro shell
//! stringifies its input and hands it here; everything downstream (parser,
//! lowering) is a normal, unit-testable Rust library. Per `zen.md`, ZSL has its
//! own syntax, so it has its own lexer rather than borrowing Rust's.

/// A byte range into the source, for diagnostics.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Span {
    pub start: usize,
    pub end: usize,
}

/// A lexical token.
#[derive(Debug, Clone, PartialEq)]
pub enum Tok {
    /// Identifier or keyword (the parser distinguishes keywords).
    Ident(String),
    /// Integer literal value.
    Int(u64),
    /// Float literal value (narrowed to `f32` later).
    Float(f64),

    // Operators
    Plus,
    Minus,
    Star,
    Slash,
    Lt,
    Le,
    Gt,
    Ge,
    EqEq,
    Ne,
    Eq,
    AndAnd,
    OrOr,
    Arrow,  // ->
    DotDot, // ..

    // Punctuation
    Dot,
    Comma,
    Colon,
    Semi,
    At,

    // Delimiters
    LParen,
    RParen,
    LBrace,
    RBrace,
    LBracket,
    RBracket,
}

/// A token plus its source span.
#[derive(Debug, Clone, PartialEq)]
pub struct Token {
    pub tok: Tok,
    pub span: Span,
}

/// A lexing error: a message and the byte offset where it occurred.
#[derive(Debug, Clone, PartialEq)]
pub struct LexError {
    pub msg: String,
    pub at: usize,
}

/// Tokenize ZSL source into a flat token stream.
pub fn lex(src: &str) -> Result<Vec<Token>, LexError> {
    let bytes = src.as_bytes();
    let n = bytes.len();
    let mut i = 0usize;
    let mut out = Vec::new();

    while i < n {
        let c = bytes[i];

        // Whitespace.
        if c == b' ' || c == b'\t' || c == b'\r' || c == b'\n' {
            i += 1;
            continue;
        }

        // Comments.
        if c == b'/' && i + 1 < n && bytes[i + 1] == b'/' {
            i += 2;
            while i < n && bytes[i] != b'\n' {
                i += 1;
            }
            continue;
        }
        if c == b'/' && i + 1 < n && bytes[i + 1] == b'*' {
            let start = i;
            i += 2;
            let mut closed = false;
            while i + 1 < n {
                if bytes[i] == b'*' && bytes[i + 1] == b'/' {
                    i += 2;
                    closed = true;
                    break;
                }
                i += 1;
            }
            if !closed {
                return Err(LexError {
                    msg: "unterminated block comment".into(),
                    at: start,
                });
            }
            continue;
        }

        // Identifiers / keywords.
        if c.is_ascii_alphabetic() || c == b'_' {
            let start = i;
            i += 1;
            while i < n && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                i += 1;
            }
            let text = src[start..i].to_string();
            out.push(Token {
                tok: Tok::Ident(text),
                span: Span { start, end: i },
            });
            continue;
        }

        // Numbers.
        if c.is_ascii_digit() {
            let start = i;
            i += 1;
            while i < n && bytes[i].is_ascii_digit() {
                i += 1;
            }
            // Fractional part — only if `.` is followed by a digit, so `0..n`
            // (range) and `v.x` (field) stay as separate tokens.
            let mut is_float = false;
            if i + 1 < n && bytes[i] == b'.' && bytes[i + 1].is_ascii_digit() {
                is_float = true;
                i += 1; // consume '.'
                while i < n && bytes[i].is_ascii_digit() {
                    i += 1;
                }
            }
            let num_end = i;
            // Optional type suffix (u32/i32/f32/…) — consumed and ignored.
            if i < n && (bytes[i].is_ascii_alphabetic() || bytes[i] == b'_') {
                let suffix_start = i;
                i += 1;
                while i < n && (bytes[i].is_ascii_alphanumeric() || bytes[i] == b'_') {
                    i += 1;
                }
                if src[suffix_start..i].starts_with('f') {
                    is_float = true;
                }
            }
            let text = &src[start..num_end];
            let tok = if is_float {
                Tok::Float(text.parse::<f64>().map_err(|_| LexError {
                    msg: format!("invalid float literal `{text}`"),
                    at: start,
                })?)
            } else {
                Tok::Int(text.parse::<u64>().map_err(|_| LexError {
                    msg: format!("invalid integer literal `{text}`"),
                    at: start,
                })?)
            };
            out.push(Token {
                tok,
                span: Span { start, end: i },
            });
            continue;
        }

        // Operators and punctuation. Try two-char tokens first.
        let two = if i + 1 < n {
            Some((c, bytes[i + 1]))
        } else {
            None
        };
        let (tok, len) = match (c, two) {
            (b'<', Some((_, b'='))) => (Tok::Le, 2),
            (b'>', Some((_, b'='))) => (Tok::Ge, 2),
            (b'=', Some((_, b'='))) => (Tok::EqEq, 2),
            (b'!', Some((_, b'='))) => (Tok::Ne, 2),
            (b'&', Some((_, b'&'))) => (Tok::AndAnd, 2),
            (b'|', Some((_, b'|'))) => (Tok::OrOr, 2),
            (b'-', Some((_, b'>'))) => (Tok::Arrow, 2),
            (b'.', Some((_, b'.'))) => (Tok::DotDot, 2),
            (b'+', _) => (Tok::Plus, 1),
            (b'-', _) => (Tok::Minus, 1),
            (b'*', _) => (Tok::Star, 1),
            (b'/', _) => (Tok::Slash, 1),
            (b'<', _) => (Tok::Lt, 1),
            (b'>', _) => (Tok::Gt, 1),
            (b'=', _) => (Tok::Eq, 1),
            (b'.', _) => (Tok::Dot, 1),
            (b',', _) => (Tok::Comma, 1),
            (b':', _) => (Tok::Colon, 1),
            (b';', _) => (Tok::Semi, 1),
            (b'@', _) => (Tok::At, 1),
            (b'(', _) => (Tok::LParen, 1),
            (b')', _) => (Tok::RParen, 1),
            (b'{', _) => (Tok::LBrace, 1),
            (b'}', _) => (Tok::RBrace, 1),
            (b'[', _) => (Tok::LBracket, 1),
            (b']', _) => (Tok::RBracket, 1),
            _ => {
                return Err(LexError {
                    msg: format!("unexpected character `{}`", c as char),
                    at: i,
                });
            }
        };
        out.push(Token {
            tok,
            span: Span {
                start: i,
                end: i + len,
            },
        });
        i += len;
    }

    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn kinds(src: &str) -> Vec<Tok> {
        lex(src).unwrap().into_iter().map(|t| t.tok).collect()
    }

    #[test]
    fn idents_and_keywords() {
        assert_eq!(
            kinds("kernel saxpy_2"),
            vec![Tok::Ident("kernel".into()), Tok::Ident("saxpy_2".into())]
        );
    }

    #[test]
    fn integer_vs_float() {
        assert_eq!(kinds("256"), vec![Tok::Int(256)]);
        assert_eq!(kinds("1.0"), vec![Tok::Float(1.0)]);
        assert_eq!(kinds("0.5"), vec![Tok::Float(0.5)]);
    }

    #[test]
    fn numeric_suffixes_ignored() {
        assert_eq!(kinds("256u32"), vec![Tok::Int(256)]);
        assert_eq!(kinds("1f32"), vec![Tok::Float(1.0)]);
    }

    #[test]
    fn range_is_not_a_float() {
        // `0..n` must lex as Int, DotDot, Ident — not `0.` float.
        assert_eq!(
            kinds("0..n"),
            vec![Tok::Int(0), Tok::DotDot, Tok::Ident("n".into())]
        );
    }

    #[test]
    fn field_access_is_dot() {
        assert_eq!(
            kinds("id.x"),
            vec![Tok::Ident("id".into()), Tok::Dot, Tok::Ident("x".into())]
        );
    }

    #[test]
    fn operators() {
        assert_eq!(
            kinds("<= >= == != && || -> .. < > ="),
            vec![
                Tok::Le,
                Tok::Ge,
                Tok::EqEq,
                Tok::Ne,
                Tok::AndAnd,
                Tok::OrOr,
                Tok::Arrow,
                Tok::DotDot,
                Tok::Lt,
                Tok::Gt,
                Tok::Eq,
            ]
        );
    }

    #[test]
    fn delimiters_and_punct() {
        assert_eq!(
            kinds("(){}[],:;@"),
            vec![
                Tok::LParen,
                Tok::RParen,
                Tok::LBrace,
                Tok::RBrace,
                Tok::LBracket,
                Tok::RBracket,
                Tok::Comma,
                Tok::Colon,
                Tok::Semi,
                Tok::At,
            ]
        );
    }

    #[test]
    fn comments_skipped() {
        assert_eq!(
            kinds("a // line\n b /* block */ c"),
            vec![
                Tok::Ident("a".into()),
                Tok::Ident("b".into()),
                Tok::Ident("c".into())
            ]
        );
    }

    #[test]
    fn saxpy_signature_lexes() {
        let src = "@workgroup_size(256)\nkernel saxpy(x: device buffer<f32>) { let i = id.x }";
        // Should lex without error and start with the attribute.
        let toks = kinds(src);
        assert_eq!(toks[0], Tok::At);
        assert_eq!(toks[1], Tok::Ident("workgroup_size".into()));
        assert!(toks.contains(&Tok::Ident("kernel".into())));
    }

    #[test]
    fn unterminated_block_comment_errors() {
        assert!(lex("a /* unterminated").is_err());
    }

    #[test]
    fn unexpected_char_errors() {
        let e = lex("a $ b").unwrap_err();
        assert!(e.msg.contains("unexpected character"));
    }

    #[test]
    fn spans_are_tracked() {
        let toks = lex("ab cd").unwrap();
        assert_eq!(toks[0].span, Span { start: 0, end: 2 });
        assert_eq!(toks[1].span, Span { start: 3, end: 5 });
    }
}
