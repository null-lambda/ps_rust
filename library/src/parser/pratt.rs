mod parser {
    use std::{
        iter::Peekable,
        ops::{Add, Range, RangeInclusive, Sub},
    };

    #[derive(Debug, Clone, PartialEq, Eq)]
    enum Token {
        Atom(u8),
        Op(u8),
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum Ast {
        Word(Vec<u8>),
        Quantifier(Box<Ast>, u8),
        Term(Vec<Ast>),
        Choice(Vec<Ast>),
    }

    impl Ast {
        fn cons_infix(self, op: u8, rhs: Ast) -> Self {
            match op {
                b':' => match self {
                    Ast::Word(mut v) => match rhs {
                        Ast::Word(w) => {
                            v.extend(w);
                            Ast::Word(v)
                        }
                        _ => Ast::Term(vec![Ast::Word(v), rhs]),
                    },
                    Ast::Term(v) => {
                        let mut v = v;
                        v.push(rhs);
                        Ast::Term(v)
                    }
                    _ => Ast::Term(vec![self, rhs]),
                },
                b'|' => match self {
                    Ast::Choice(mut v) => {
                        v.push(rhs);
                        Ast::Choice(v)
                    }
                    _ => Ast::Choice(vec![self, rhs]),
                },
                _ => panic!(),
            }
        }

        fn cons_prefix(self, _op: u8) -> Self {
            panic!()
        }

        fn cons_postfix(self, op: u8) -> Self {
            match op {
                b'?' => Ast::Quantifier(Box::new(self), b'?'),
                b'+' => Ast::Quantifier(Box::new(self), b'+'),
                b'*' => Ast::Quantifier(Box::new(self), b'*'),
                _ => panic!(),
            }
        }

        pub fn bbox_size(&self) -> BBox<isize> {
            match self {
                Ast::Word(w) => BBox::new(-1..1, 0..w.len() as isize + 4),
                Ast::Quantifier(e, b'?') => e.bbox_size().pad_down(2).pad_right(6),
                Ast::Quantifier(e, b'+') => e.bbox_size().pad_down(3).pad_right(6),
                Ast::Quantifier(e, b'*') => e.bbox_size().pad_down(5).pad_right(6),
                Ast::Quantifier(..) => panic!(),
                Ast::Term(es) => es
                    .iter()
                    .map(|e| e.bbox_size())
                    .reduce(|a, b| a.pad_right(2).hstack(b))
                    .unwrap(),
                Ast::Choice(es) => es
                    .iter()
                    .map(|e| e.bbox_size())
                    .reduce(|a, b| a.pad_down(1).vstack(b))
                    .unwrap()
                    .pad_right(6),
            }
        }
    }

    impl std::fmt::Display for Ast {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match self {
                Ast::Word(t) => write!(f, "[{}]", std::str::from_utf8(t).unwrap())?,
                Ast::Quantifier(e, op) => write!(f, "{}{}", e, *op as char)?,
                Ast::Term(es) => {
                    write!(f, "{}", es[0])?;
                    for e in &es[1..] {
                        write!(f, "{}", e)?;
                    }
                }
                Ast::Choice(es) => {
                    write!(f, "({}", es[0])?;
                    for e in &es[1..] {
                        write!(f, "|{}", e)?;
                    }
                    write!(f, ")")?;
                }
            }
            Ok(())
        }
    }

    fn tokenize(s: &[u8]) -> Vec<Token> {
        let mut tokens = vec![];
        let mut i = 0;
        while i < s.len() {
            match s[i] {
                b'(' | b')' | b'+' | b'?' | b'*' | b'|' => tokens.push(Token::Op(s[i])),
                b'A'..=b'Z' => tokens.push(Token::Atom(s[i])),
                _ => panic!(),
            }
            i += 1;
        }
        tokens
    }

    // Pratt parser
    // https://github.com/matklad/minipratt
    fn prefix_binding_power(op: u8) -> ((), u8) {
        match op {
            _ => panic!(),
        }
    }
    fn postfix_binding_power(op: u8) -> Option<(u8, ())> {
        let result = match op {
            b'?' | b'+' | b'*' => (9, ()),
            _ => return None,
        };
        Some(result)
    }
    fn infix_binding_power(op: u8) -> Option<(u8, u8)> {
        let result = match op {
            b'|' => (5, 6),
            b':' => (7, 8),
            _ => return None,
        };
        Some(result)
    }

    type TokenStream = Peekable<std::vec::IntoIter<Token>>;
    fn parse_tokens(tokens: &mut TokenStream, min_bp: u8) -> Ast {
        let mut acc = match tokens.next().unwrap() {
            Token::Atom(t) => Ast::Word(vec![t]),
            Token::Op(b'(') => {
                let acc = parse_tokens(tokens, 0);
                assert_eq!(tokens.next(), Some(Token::Op(b')')));
                acc
            }
            Token::Op(op) => {
                let ((), r_bp) = prefix_binding_power(op);
                parse_tokens(tokens, r_bp).cons_prefix(op)
            }
        };

        loop {
            let op = match tokens.peek() {
                Some(Token::Atom(_)) | Some(Token::Op(b'(')) => b':',
                Some(Token::Op(op)) => *op,
                None => break,
            };

            if let Some((l_bp, ())) = postfix_binding_power(op) {
                if l_bp < min_bp {
                    break;
                }
                tokens.next();
                acc = acc.cons_postfix(op);
                continue;
            }

            if let Some((l_bp, r_bp)) = infix_binding_power(op) {
                if l_bp < min_bp {
                    break;
                }
                if op != b':' {
                    tokens.next();
                }
                acc = acc.cons_infix(op, parse_tokens(tokens, r_bp));
                continue;
            }

            break;
        }
        acc
    }

    pub fn parse(s: &[u8]) -> Ast {
        let mut tokens = tokenize(s).into_iter().peekable();
        parse_tokens(&mut tokens, 0)
    }
}
