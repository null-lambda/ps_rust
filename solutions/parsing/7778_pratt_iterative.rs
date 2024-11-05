use std::io::Write;

mod simple_io {
    pub struct InputAtOnce(std::str::SplitAsciiWhitespace<'static>);

    impl InputAtOnce {
        pub fn token(&mut self) -> &str {
            self.0.next().unwrap_or_default()
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().unwrap()
        }
    }

    pub fn stdin_at_once() -> InputAtOnce {
        let buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let buf = Box::leak(buf.into_boxed_str());
        InputAtOnce(buf.split_ascii_whitespace())
    }

    pub fn stdout_buf() -> std::io::BufWriter<std::io::Stdout> {
        std::io::BufWriter::new(std::io::stdout())
    }
}

pub mod ast {
    use std::fmt;

    pub type Link = u32;

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum Expr {
        Var(u8),
        Value(bool),
        Neg(Link),
        And(Link, Link),
        Or(Link, Link),
        Implies(Link, Link),
        Iff(Vec<Link>),
        Parens(Link),
    }

    pub struct ExprPool(Vec<Expr>);

    impl ExprPool {
        fn new() -> Self {
            let vars = (0..12).map(|i| Expr::Var(i)).collect();
            Self(vars)
        }

        fn var(&self, i: u8) -> Link {
            i as Link
        }

        pub fn get(&self, i: Link) -> &Expr {
            &self.0[i as usize]
        }

        unsafe fn get_mut(&mut self, i: Link) -> &mut Expr {
            &mut self.0[i as usize]
        }

        pub fn add(&mut self, e: Expr) -> Link {
            self.0.push(e);
            (self.0.len() - 1) as Link
        }

        pub fn cons(&mut self, op: u8, xs: Vec<Link>) -> Link {
            match op {
                b'0' | b'1' => self.add(Expr::Value(op == b'1')),
                b'A'..=b'L' => self.var(op - b'A'),
                b'(' => xs[0],
                b'=' => match (self.get(xs[0]), self.get(xs[1])) {
                    (Expr::Iff(_), _) => unsafe {
                        let Expr::Iff(ref mut ys) = self.get_mut(xs[0]) else {
                            unreachable!();
                        };
                        ys.push(xs[1]);
                        xs[0]
                    },
                    // (_, Expr::Iff(_)) => unsafe {
                    //     let Expr::Iff(ref mut ys) = self.get_mut(xs[1]) else {
                    //         unreachable!();
                    //     };
                    //     ys.insert(0, ys[0]);
                    //     ys[1]
                    // },
                    (_, _) => self.add(Expr::Iff(vec![xs[0], xs[1]])),
                },
                b'>' => self.add(Expr::Implies(xs[0], xs[1])),
                b'|' => self.add(Expr::Or(xs[0], xs[1])),
                b'&' => self.add(Expr::And(xs[0], xs[1])),
                b'~' => self.add(Expr::Neg(xs[0])),
                _ => panic!(),
            }
        }
    }

    pub struct ExprDisplay<'a>(&'a ExprPool, Link);
    impl fmt::Display for ExprDisplay<'_> {
        fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
            let pool = self.0;
            let expr = pool.get(self.1);
            match expr {
                Expr::Var(x) => write!(f, "{}", ('A' as u8 + *x) as char),
                Expr::Value(b) => write!(f, "{}", *b as u8),
                Expr::Neg(e) => write!(f, "~{}", ExprDisplay(pool, *e)),
                Expr::And(a, b) => {
                    write!(f, "[{}&{}]", ExprDisplay(pool, *a), ExprDisplay(pool, *b))
                }
                Expr::Or(a, b) => {
                    write!(f, "[{}|{}]", ExprDisplay(pool, *a), ExprDisplay(pool, *b))
                }
                Expr::Implies(a, b) => {
                    write!(f, "[{}->{}]", ExprDisplay(pool, *a), ExprDisplay(pool, *b))
                }
                Expr::Iff(xs) => {
                    write!(f, "[")?;
                    for (i, &x) in xs.iter().enumerate() {
                        if i > 0 {
                            write!(f, "<=>")?;
                        }
                        write!(f, "{}", ExprDisplay(pool, x))?;
                    }
                    Ok(())
                }
                Expr::Parens(e) => write!(f, "({})", ExprDisplay(pool, *e)),
            }
        }
    }

    impl ExprPool {
        pub fn disp(&self, e: Link) -> ExprDisplay {
            ExprDisplay(self, e)
        }
    }

    // Shunting-yard algorithm
    // https://matklad.github.io/2020/04/15/from-pratt-to-dijkstra.html
    fn binding_power(op: Option<u8>, prefix: bool) -> Option<(u8, (u8, u8))> {
        let op = op?;
        let res = match op {
            b'0'..=b'1' | b'A'..=b'L' => (99, 100),
            b'(' => (99, 0),
            b')' => (0, 100),
            b'=' => (1, 2),
            b'>' => (4, 3),
            b'|' => (5, 6),
            b'&' => (7, 8),
            b'~' if prefix => (99, 9),
            _ => return None,
        };
        Some((op, res))
    }

    pub fn parse(s: &[u8]) -> (ExprPool, Link) {
        let mut pool = ExprPool::new();
        let mut tokens = s.iter().copied().peekable();

        struct Frame {
            min_bp: u8,
            lhs: Option<Link>,
            token: Option<u8>,
        }
        let mut top = Frame {
            min_bp: 0,
            lhs: None,
            token: None,
        };
        let mut stack = vec![];
        loop {
            let token = tokens.next();
            let (token, r_bp) = loop {
                match binding_power(token, top.lhs.is_none()) {
                    Some((t, (l_bp, r_bp))) if top.min_bp <= l_bp => break (t, r_bp),
                    _ => {
                        let res = top;
                        top = match stack.pop() {
                            Some(it) => it,
                            None => return (pool, res.lhs.unwrap()),
                        };
                        let mut args = vec![];
                        args.extend(top.lhs);
                        args.extend(res.lhs);
                        let token = res.token.unwrap();
                        top.lhs = Some(pool.cons(token, args));
                    }
                };
            };
            if token == b')' {
                assert_eq!(top.token, Some(b'('));
                let res = top;
                top = stack.pop().unwrap();
                top.lhs = res.lhs.map(|x| match pool.get(x) {
                    Expr::Parens(_) => x,
                    _ => pool.add(Expr::Parens(x)),
                });
                continue;
            }
            stack.push(top);
            top = Frame {
                min_bp: r_bp,
                lhs: None,
                token: Some(token),
            };
        }
    }

    type ChunkedBool = u64;
    impl ExprPool {
        pub fn eval(&self, expr: Link, assignment: &[ChunkedBool]) -> ChunkedBool {
            let expr = self.get(expr);
            match expr {
                Expr::Var(x) => assignment[*x as usize],
                Expr::Value(b) => !0 * *b as u64,
                Expr::Neg(e) => !self.eval(*e, assignment),
                Expr::And(a, b) => self.eval(*a, assignment) & self.eval(*b, assignment),
                Expr::Or(a, b) => self.eval(*a, assignment) | self.eval(*b, assignment),
                Expr::Implies(a, b) => !self.eval(*a, assignment) | self.eval(*b, assignment),
                // Expr::Iff(xs) => !(self.eval(*a, assignment) ^ self.eval(*b, assignment)),
                Expr::Iff(xs) => xs.windows(2).fold(!0, |acc, x| {
                    acc & !(self.eval(x[0], assignment) ^ self.eval(x[1], assignment))
                }) as u64,
                Expr::Parens(e) => self.eval(*e, assignment),
            }
        }
    }
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout_buf();

    let s = input.token();
    let s = s.replace("<=>", "=").replace("->", ">");
    let (pool, expr) = ast::parse(s.as_bytes());

    let n_vars = 12;
    let n_chunk_base = 6;
    let assignments = (0..1 << (n_vars - n_chunk_base))
        .flat_map(|i| {
            let i0 = (1 << n_chunk_base) * i;
            let mut chunked_assignment = vec![0; n_vars];
            for j in 0..n_vars {
                for k in 0..1 << n_chunk_base {
                    chunked_assignment[j] |= (((i0 + k) >> j) & 1) << k;
                }
            }
            let res = pool.eval(expr, &chunked_assignment);
            (0..1 << n_chunk_base).map(move |i| (res >> i) & 1 != 0)
        })
        .collect::<Vec<bool>>();

    let n_true = assignments.iter().filter(|&&b| b).count();

    if n_true < assignments.len() / 2 {
        let f = n_true + 1;
        if f == 1 {
            writeln!(output, "2\n1 0\n1 0").unwrap();
            return;
        }
        writeln!(output, "{}", f).unwrap();
        for i in 0..assignments.len() {
            if !assignments[i] {
                continue;
            }

            let neg_sign = |varname| {
                if i & (1 << varname as u8) != 0 {
                    ""
                } else {
                    "~"
                }
            };
            let symbol = |varname| ('A' as u8 + varname as u8) as char;

            write!(output, "{} ", 2 * n_vars - 1).unwrap();
            for _ in 1..n_vars {
                write!(output, "0 ").unwrap();
            }
            for v in 0..n_vars {
                write!(output, "{}{} ", neg_sign(v), symbol(v)).unwrap();
            }
            writeln!(output, "").unwrap();
        }

        write!(output, "{} ", n_true * 2 - 1).unwrap();
        for _ in 1..n_true {
            write!(output, "~0 ").unwrap();
        }
        for i in 0..n_true {
            write!(output, "{} ", i + 1).unwrap();
        }
    } else {
        let n_false = assignments.len() - n_true;
        let f = n_false + 1;

        if f == 1 {
            writeln!(output, "2\n1 0\n1 ~0").unwrap();
            return;
        }

        writeln!(output, "{}", f).unwrap();
        for i in 0..assignments.len() {
            if assignments[i] {
                continue;
            }

            let neg_sign = |varname| {
                if i & (1 << varname as u8) != 0 {
                    ""
                } else {
                    "~"
                }
            };
            let symbol = |varname| ('A' as u8 + varname as u8) as char;

            write!(output, "{} ", 2 * n_vars - 1).unwrap();
            for _ in 1..n_vars {
                write!(output, "0 ").unwrap();
            }
            for v in 0..n_vars {
                write!(output, "{}{} ", neg_sign(v), symbol(v)).unwrap();
            }
            writeln!(output, "").unwrap();
        }

        write!(output, "{} ", n_false * 2 - 1).unwrap();
        for _ in 1..n_false {
            write!(output, "0 ").unwrap();
        }
        for i in 0..n_false {
            write!(output, "~{} ", i + 1).unwrap();
        }
    }
    writeln!(output).unwrap();
    // writeln!(output, "{}", pool.disp(expr)).unwrap();
}
