use std::io::Write;
use std::{collections::HashMap, io::BufRead};

#[allow(dead_code)]
#[macro_use]
pub mod parser {
    pub type ParseResult<S, A> = Option<(A, S)>;
    pub trait Stream<'a>: Sized + Clone + std::fmt::Debug + 'a {
        type Item;

        fn zero() -> Self;
        fn next(self) -> ParseResult<Self, Self::Item>;

        fn take_while(
            self,
            pred: impl Fn(&Self::Item) -> bool,
        ) -> ParseResult<Self, &'a [Self::Item]>;

        fn skip_while(self, pred: impl Fn(&Self::Item) -> bool) -> ParseResult<Self, ()> {
            self.take_while(pred).map(|(_, s)| ((), s))
        }
    }

    pub trait U8Stream<'a>: Stream<'a, Item = u8> {}

    impl<'a> Stream<'a> for &'a [u8] {
        type Item = u8;

        fn zero() -> Self {
            &[]
        }

        fn next(self) -> ParseResult<Self, u8> {
            self.split_first().map(|(b, s)| (*b, s))
        }

        fn take_while(self, pred: impl Fn(&u8) -> bool) -> ParseResult<Self, &'a [u8]> {
            let mut i = 0;
            while i < self.len() && pred(&self[i]) {
                i += 1;
            }
            Some(self.split_at(i))
        }
    }

    impl<'a> U8Stream<'a> for &'a [u8] {}

    pub trait Parser<'a, S: Stream<'a>, A>: Sized {
        fn run(&self, s: S) -> ParseResult<S, A>;

        fn map<B, F>(self, f: F) -> Map<Self, F, A>
        where
            F: Fn(A) -> B,
        {
            Map(self, f, Default::default())
        }

        fn and_then<B, PB, F>(self, p: F) -> AndThen<Self, F, A>
        where
            PB: Parser<'a, S, B>,
            F: Fn(A) -> PB,
        {
            AndThen(self, p, Default::default())
        }

        fn map_option<B, F>(self, f: F) -> MapOption<Self, F, A>
        where
            F: Fn(A) -> Option<B>,
        {
            MapOption(self, f, Default::default())
        }

        fn filter<F>(self, f: F) -> Filter<Self, F>
        where
            F: Fn(&A) -> bool,
        {
            Filter(self, f)
        }

        fn or_else<P>(self, p: P) -> OrElse<Self, P>
        where
            P: Parser<'a, S, A>,
        {
            OrElse(self, p)
        }
    }

    // Can be replaced with impl trait on method return types (rust >= 1.75)
    pub struct Map<PA, F, A>(PA, F, std::marker::PhantomData<A>);
    impl<'a, S, A, B, PA, F> Parser<'a, S, B> for Map<PA, F, A>
    where
        S: Stream<'a>,
        PA: Parser<'a, S, A>,
        F: Fn(A) -> B,
    {
        fn run(&self, s: S) -> ParseResult<S, B> {
            self.0.run(s).map(|(a, s)| (self.1(a), s))
        }
    }

    pub struct AndThen<PA, F, A>(PA, F, std::marker::PhantomData<A>);
    impl<'a, S, A, B, PA, PB, F> Parser<'a, S, B> for AndThen<PA, F, A>
    where
        S: Stream<'a>,
        PA: Parser<'a, S, A>,
        PB: Parser<'a, S, B>,
        F: Fn(A) -> PB,
    {
        fn run(&self, s: S) -> ParseResult<S, B> {
            self.0.run(s).and_then(|(a, s)| self.1(a).run(s))
        }
    }

    pub struct MapOption<PA, F, A>(PA, F, std::marker::PhantomData<A>);
    impl<'a, S, A, B, PA, F> Parser<'a, S, B> for MapOption<PA, F, A>
    where
        S: Stream<'a>,
        PA: Parser<'a, S, A>,
        F: Fn(A) -> Option<B>,
    {
        fn run(&self, s: S) -> ParseResult<S, B> {
            self.0.run(s).and_then(|(a, s)| self.1(a).map(|b| (b, s)))
        }
    }

    pub struct Filter<PA, F>(PA, F);
    impl<'a, S, A, PA, F> Parser<'a, S, A> for Filter<PA, F>
    where
        S: Stream<'a>,
        PA: Parser<'a, S, A>,
        F: Fn(&A) -> bool,
    {
        fn run(&self, s: S) -> ParseResult<S, A> {
            self.0.run(s).filter(|(a, _)| (self.1)(a))
        }
    }

    pub struct OrElse<P, Q>(P, Q);

    impl<'a, S, A, P, Q> Parser<'a, S, A> for OrElse<P, Q>
    where
        S: Stream<'a>,
        P: Parser<'a, S, A>,
        Q: Parser<'a, S, A>,
    {
        fn run(&self, s: S) -> ParseResult<S, A> {
            self.0.run(s.clone()).or_else(|| self.1.run(s))
        }
    }

    impl<'a, S, A, F> Parser<'a, S, A> for F
    where
        S: Stream<'a>,
        F: Fn(S) -> ParseResult<S, A>,
    {
        fn run(&self, s: S) -> ParseResult<S, A> {
            self(s)
        }
    }

    macro_rules! gen_tuple_parser {
        ($($A:tt $PA:tt $a:tt $i:tt),+) => {
            impl<'a,S:Stream<'a>,$($A,)+$($PA,)+> Parser<'a,S,($($A,)+)> for ($($PA,)+)
            where $($PA: Parser<'a,S,$A>,)+
            {
                fn run(&self, s: S) -> ParseResult<S, ($($A,)+)> {
                    $(let ($a, s):($A,S) = self.$i.run(s)?;)+
                    Some((($($a,)+),s))
                }
            }
        }
    }
    gen_tuple_parser!(A0 PA0 a0 0, A1 PA1 a1 1);
    gen_tuple_parser!(A0 PA0 a0 0, A1 PA1 a1 1, A2 PA2 a2 2);
    gen_tuple_parser!(A0 PA0 a0 0, A1 PA1 a1 1, A2 PA2 a2 2, A3 PA3 a3 3);
    gen_tuple_parser!(A0 PA0 a0 0, A1 PA1 a1 1, A2 PA2 a2 2, A3 PA3 a3 3, A4 PA4 a4 4);
    gen_tuple_parser!(A0 PA0 a0 0, A1 PA1 a1 1, A2 PA2 a2 2, A3 PA3 a3 3, A4 PA4 a4 4, A5 PA5 a5 5);

    fn fail<S, A>(_s: S) -> ParseResult<S, A> {
        None
    }

    pub fn eof<'a, S: Stream<'a>>(s: S) -> ParseResult<S, ()> {
        match s.next() {
            Some(_) => None,
            None => Some(((), S::zero())),
        }
    }

    pub fn single<'a, S: Stream<'a>>(s: S) -> ParseResult<S, S::Item> {
        s.next()
    }

    pub fn between<'a, S: Stream<'a>, A, O, E>(
        p_open: impl Parser<'a, S, O>,
        p: impl Parser<'a, S, A>,
        p_close: impl Parser<'a, S, E>,
    ) -> impl Parser<'a, S, A> {
        (p_open, p, p_close).map(|(_, a, _)| a)
    }

    pub fn optional<'a, S: Stream<'a>, A>(
        p: impl Parser<'a, S, A>,
    ) -> impl Parser<'a, S, Option<A>> {
        move |s: S| match p.run(s.clone()) {
            Some((a, s_new)) => Some((Some(a), s_new)),
            None => Some((None, s)),
        }
    }

    pub fn fold<'a, S: Stream<'a>, A, F>(
        init: impl Parser<'a, S, A>,
        p: impl Parser<'a, S, F>,
    ) -> impl Parser<'a, S, A>
    where
        F: Fn(A) -> A,
    {
        move |mut s: S| {
            let (mut acc, s_new) = init.run(s)?;
            s = s_new;
            while let Some((f, s_new)) = p.run(s.clone()) {
                acc = f(acc);
                s = s_new;
            }
            Some((acc, s))
        }
    }

    pub fn many<'a, S: Stream<'a>, A>(p: impl Parser<'a, S, A>) -> impl Parser<'a, S, Vec<A>> {
        move |mut s: S| {
            let mut result = Vec::new();
            while let Some((e, s_new)) = p.run(s.clone()) {
                result.push(e);
                s = s_new;
            }
            Some((result, s))
        }
    }

    pub fn many1<'a, S: Stream<'a>, A>(p: impl Parser<'a, S, A>) -> impl Parser<'a, S, Vec<A>> {
        move |mut s: S| {
            let mut result = Vec::new();
            let (e, s_new) = p.run(s.clone())?;
            result.push(e);
            s = s_new;

            while let Some((e, s_new)) = p.run(s.clone()) {
                result.push(e);
                s = s_new;
            }
            Some((result, s))
        }
    }

    pub fn many_sep1<'a, S: Stream<'a>, A>(
        p: impl Parser<'a, S, A>,
        p_sep: impl Parser<'a, S, ()>,
    ) -> impl Parser<'a, S, Vec<A>> {
        move |mut s: S| -> ParseResult<S, Vec<A>> {
            let mut result = Vec::new();
            let (e, s_new) = p.run(s)?;
            result.push(e);
            s = s_new;

            while let Some((_, s_new)) = p_sep.run(s.clone()) {
                match p.run(s_new) {
                    Some((e, s_new)) => {
                        result.push(e);
                        s = s_new;
                    }
                    None => break,
                }
            }

            Some((result, s))
        }
    }

    // chained unary operators
    pub fn unary_infix<'a, S: Stream<'a>, A, F, PF, P>(p_op: PF, p: P) -> impl Parser<'a, S, A>
    where
        P: Parser<'a, S, A>,
        PF: Parser<'a, S, F>,
        F: Fn(A) -> A + 'a,
    {
        (many(p_op), p)
            .map(move |(cs, e): (Vec<F>, A)| cs.into_iter().rev().fold(e, |acc, c| c(acc)))
    }

    pub fn unary_suffix<'a, S: Stream<'a>, A, F, PF, P>(p_op: PF, p: P) -> impl Parser<'a, S, A>
    where
        P: Parser<'a, S, A>,
        PF: Parser<'a, S, F>,
        F: Fn(A) -> A + 'a,
    {
        (p, many(p_op)).map(move |(e, cs): (A, Vec<F>)| cs.into_iter().fold(e, |acc, c| c(acc)))
    }

    pub fn binary_lassoc<'a, S: Stream<'a>, A, F, PF, P>(p_op: PF, p: P) -> impl Parser<'a, S, A>
    where
        P: Parser<'a, S, A>,
        PF: Parser<'a, S, F>,
        F: Fn(A, A) -> A + 'a,
    {
        move |mut s: S| {
            let (acc, s_new) = p.run(s)?;
            let mut acc = acc;
            s = s_new;

            while let Some((f, s_new)) = p_op.run(s.clone()) {
                let (b, s_new) = p.run(s_new)?;
                acc = f(acc, b);
                s = s_new;
            }

            Some((acc, s))
        }
    }

    pub fn binary_rassoc<'a, S: Stream<'a>, A, F, PF, P>(p_op: PF, p: P) -> impl Parser<'a, S, A>
    where
        P: Parser<'a, S, A>,
        PF: Parser<'a, S, F>,
        F: Fn(A, A) -> A + 'a,
    {
        move |mut s: S| {
            let (x0, s_new) = p.run(s)?;
            s = s_new;
            let mut xs: Vec<A> = vec![x0];
            let mut fs: Vec<F> = vec![];

            while let Some((f, s_new)) = p_op.run(s.clone()) {
                let (b, s_new) = p.run(s_new)?;
                fs.push(f);
                xs.push(b);
                s = s_new;
            }

            let acc = xs.pop().unwrap();
            let acc = xs
                .into_iter()
                .zip(fs.into_iter())
                .rev()
                .fold(acc, |acc, (b, f)| f(b, acc));

            Some((acc, s))
        }
    }

    pub fn debug_print<'a, S, A>(p: impl Parser<'a, S, A>) -> impl Parser<'a, S, A>
    where
        S: Stream<'a>,
        S::Item: std::fmt::Display,
        A: std::fmt::Display,
    {
        move |s| {
            let (a, s) = p.run(s)?;
            Some((a, s))
        }
    }

    pub mod u8 {
        use super::*;

        // macro to quick define a 0-ary parser
        #[macro_export]
        macro_rules! parser_fn {
            (let $name:ident: $T:ty = $body:expr) => {
                fn $name<'a, S: U8Stream<'a>>(s: S) -> ParseResult<S, $T> {
                    $body.run(s)
                }
            };
        }

        pub fn satisfy<'a, S: U8Stream<'a>>(f: impl Fn(&u8) -> bool) -> impl Parser<'a, S, u8> {
            move |s: S| s.next().filter(|(b, _)| f(b))
        }

        pub fn literal_byte<'a, S: U8Stream<'a>>(c: u8) -> impl Parser<'a, S, ()> {
            satisfy(move |b| b == &c).map(|_| ())
        }

        pub fn literal<'a, S: U8Stream<'a>>(keyword: &'static [u8]) -> impl Parser<'a, S, ()> {
            move |mut s: S| {
                for &c in keyword {
                    let (b, s_new) = s.next()?;
                    if b != c {
                        return None;
                    }
                    s = s_new;
                }
                Some(((), s))
            }
        }

        pub fn spaces<'a, S: U8Stream<'a>>(s: S) -> ParseResult<S, ()> {
            s.skip_while(|b| b.is_ascii_whitespace())
        }

        pub fn uint<'a, S: U8Stream<'a>>(s: S) -> ParseResult<S, i32> {
            let mut s = s;
            let (b, s_new) = satisfy(|b| b.is_ascii_digit()).run(s.clone())?;
            let mut result = (b - b'0') as i32;
            s = s_new;

            while let Some((b, s_new)) = satisfy(|b| b.is_ascii_digit()).run(s.clone()) {
                result = result * 10 + (b - b'0') as i32;
                s = s_new;
            }

            Some((result, s))
        }

        pub fn int<'a, S: U8Stream<'a>>(s: S) -> ParseResult<S, i32> {
            let mut s = s;
            let mut sign = 1;
            if let Some((_, s_new)) = satisfy(|&b| b == b'-').run(s.clone()) {
                sign = -1;
                s = s_new;
            }

            // at least one digit is required
            let (b, s_new) = satisfy(|b| b.is_ascii_digit()).run(s.clone())?;
            let mut result = (b - b'0') as i32;
            s = s_new;

            while let Some((b, s_new)) = satisfy(|b| b.is_ascii_digit()).run(s.clone()) {
                result = result * 10 + (b - b'0') as i32;
                s = s_new;
            }

            Some((result * sign, s))
        }
    }
}

#[derive(Debug, Clone, Copy)]
enum UnaryOpCode {
    Neg,
    Not,
}

#[derive(Debug, Clone, Copy)]
enum BinOpCode {
    Mul,
    Div,
    Rem,
    Add,
    Sub,
    Lt,
    Le,
    Gt,
    Ge,
    Eq,
    Ne,
    And,
    Or,
}

#[derive(Debug, Clone)]
enum Expr {
    Var(u8),
    Num(i32),
    UnaryOp(UnaryOpCode, Box<Expr>),
    BinOp(BinOpCode, Box<Expr>, Box<Expr>),
}

#[derive(Debug, Clone)]
enum Stmt {
    Assign(u8, Expr),
    Print(Expr),
    IfThenElse(Expr, Vec<Stmt>, Vec<Stmt>),
    IfThen(Expr, Vec<Stmt>),
    While(Expr, Vec<Stmt>),
}

#[derive(Debug, Clone)]
struct Program(Vec<Stmt>);

struct Context<W: Write> {
    vars: [i32; 256],
    output: W,
}

impl Program {
    fn run<W: Write>(&self, ctx: &mut Context<W>) {
        for stmt in &self.0 {
            stmt.run(ctx);
        }
    }
}

impl Stmt {
    fn run<W: Write>(&self, ctx: &mut Context<W>) {
        match &self {
            Stmt::Assign(v, e) => ctx.vars[*v as usize] = e.eval(ctx),
            Stmt::Print(e) => writeln!(ctx.output, "{}", e.eval(ctx)).unwrap(),
            Stmt::IfThenElse(cond, th, el) => {
                if cond.eval(ctx) != 0 {
                    for stmt in th {
                        stmt.run(ctx);
                    }
                } else {
                    for stmt in el {
                        stmt.run(ctx);
                    }
                }
            }
            Stmt::IfThen(cond, th) => {
                if cond.eval(ctx) != 0 {
                    for stmt in th {
                        stmt.run(ctx);
                    }
                }
            }
            Stmt::While(cond, block) => {
                while cond.eval(ctx) != 0 {
                    for stmt in block {
                        stmt.run(ctx);
                    }
                }
            }
        }
    }
}

impl Expr {
    fn eval<W: Write>(&self, ctx: &Context<W>) -> i32 {
        match &self {
            Expr::Var(v) => ctx.vars[*v as usize],
            Expr::Num(x) => *x,
            Expr::UnaryOp(op, e) => match op {
                UnaryOpCode::Neg => -e.eval(ctx),
                UnaryOpCode::Not => (e.eval(ctx) == 0) as i32,
            },
            Expr::BinOp(op, a, b) => {
                let (a, b) = (a.eval(ctx), b.eval(ctx));
                let (la, lb) = (a != 0, b != 0);
                match op {
                    BinOpCode::Mul => a * b,
                    BinOpCode::Div => a / b,
                    BinOpCode::Rem => a % b,
                    BinOpCode::Add => a + b,
                    BinOpCode::Sub => a - b,
                    BinOpCode::Lt => (a < b) as i32,
                    BinOpCode::Le => (a <= b) as i32,
                    BinOpCode::Gt => (a > b) as i32,
                    BinOpCode::Ge => (a >= b) as i32,
                    BinOpCode::Eq => (a == b) as i32,
                    BinOpCode::Ne => (a != b) as i32,
                    BinOpCode::And => (la && lb) as i32,
                    BinOpCode::Or => (la || lb) as i32,
                }
            }
        }
    }
}

fn parse_program(s: &[u8]) -> Program {
    use parser::u8::*;
    use parser::*;

    fn keyword<'a, S: U8Stream<'a>>(keyword: &'static [u8]) -> impl Parser<'a, S, ()> {
        (literal(keyword), spaces).map(|_| ())
    }

    parser_fn!(let varname: u8 = (satisfy(|c| c.is_ascii_alphabetic()), spaces).map(|(x,_x)| x));
    parser_fn!(let var: Expr = varname.map(Expr::Var));
    parser_fn!(let num: Expr = (uint, spaces).map(|(x, _)| Expr::Num(x)));

    parser_fn!(let parens: Expr = between(keyword(b"("), expr, keyword(b")")));
    parser_fn!(let factor7: Expr = parens.or_else(var).or_else(num));
    parser_fn!(let factor6: Expr = unary_infix(
            keyword(b"-").map(|_| UnaryOpCode::Neg)
                .or_else(keyword(b"!").map(|_| UnaryOpCode::Not))
                .map(|op| move |a| Expr::UnaryOp(op, Box::new(a))), factor7)
    );
    parser_fn!(let factor5: Expr = binary_lassoc(
            keyword(b"*").map(|_| BinOpCode::Mul)
                .or_else(keyword(b"/").map(|_| BinOpCode::Div))
                .or_else(keyword(b"%").map(|_| BinOpCode::Rem))
                .map(|op| move |a, b| Expr::BinOp(op, Box::new(a), Box::new(b))), factor6)
    );
    parser_fn!(let factor4: Expr = binary_lassoc(
            keyword(b"+").map(|_| BinOpCode::Add)
                .or_else(keyword(b"-").map(|_| BinOpCode::Sub))
                .map(|op| move |a, b| Expr::BinOp(op, Box::new(a), Box::new(b))), factor5)
    );
    parser_fn!(let factor3: Expr = binary_lassoc(
            keyword(b"<").map(|_| BinOpCode::Lt)
                .or_else(keyword(b"<=").map(|_| BinOpCode::Le))
                .or_else(keyword(b">").map(|_| BinOpCode::Gt))
                .or_else(keyword(b">=").map(|_| BinOpCode::Ge))
                .map(|op| move |a, b| Expr::BinOp(op, Box::new(a), Box::new(b))), factor4)
    );
    parser_fn!(let factor2: Expr = binary_lassoc(
            keyword(b"==").map(|_| BinOpCode::Eq)
                .or_else(keyword(b"!=").map(|_| BinOpCode::Ne))
                .map(|op| move |a, b| Expr::BinOp(op, Box::new(a), Box::new(b))), factor3)
    );
    parser_fn!(let factor1: Expr = binary_lassoc(
            keyword(b"&&").map(|_| BinOpCode::And)
                .map(|op| move |a, b| Expr::BinOp(op, Box::new(a), Box::new(b))), factor2)
    );
    parser_fn!(let expr: Expr = binary_lassoc(
            keyword(b"||").map(|_| BinOpCode::Or)
                .map(|op| move |a, b| Expr::BinOp(op, Box::new(a), Box::new(b))), factor1)
    );

    parser_fn!(let assign: Stmt = (keyword(b"set"), varname, keyword(b"="), expr)
        .map(|(_, v, _, e)| Stmt::Assign(v, e)));
    parser_fn!(let print_s: Stmt = (keyword(b"print"), expr).map(|(_, e)| Stmt::Print(e)));
    parser_fn!(let ite: Stmt = (
            keyword(b"if"), expr, many(stmt),
            optional((keyword(b"else"), many(stmt))),
            keyword(b"end"), keyword(b"if"))
            .map(|(_, cond, th, el, _, _)| {
                match el {
                    Some((_, el)) => Stmt::IfThenElse(cond, th, el),
                    None => Stmt::IfThen(cond, th),
                }
            })
    );
    parser_fn!(let while_loop: Stmt = (
            keyword(b"while"), expr, many(stmt), keyword(b"end"), keyword(b"while"))
            .map(|(_, cond, body, _, _)| Stmt::While(cond, body))
    );
    parser_fn!(let stmt: Stmt = print_s.or_else(assign).or_else(ite).or_else(while_loop));

    parser_fn!(let program: Program = (spaces, many(stmt)).map(|(_, xs)| Program(xs)));

    program(s).unwrap().0
}

fn main() {
    let input = std::io::BufReader::new(std::io::stdin().lock());
    let mut output = std::io::BufWriter::new(std::io::stdout().lock());
    let mut lines = input.lines().flatten();

    loop {
        let n_lines: usize = lines.next().unwrap().parse().unwrap();
        if n_lines == 0 {
            break;
        }

        let program = (0..n_lines)
            .map(|_| lines.next().unwrap().as_bytes().to_vec())
            .collect::<Vec<_>>()
            .join(&b'\n');

        let ast = parse_program(&program);
        // println!("{:?}", ast);
        ast.run(&mut Context {
            vars: [0; 256],
            output: &mut output,
        });
    }
}
