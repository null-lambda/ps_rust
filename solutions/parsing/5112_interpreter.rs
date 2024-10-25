use std::io::{self, BufRead, Write};

#[allow(dead_code)]
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

#[allow(dead_code)]
#[macro_use]
pub mod parser {
    use std::mem;

    pub type ParseResult<S, A> = Option<(A, S)>;
    pub trait Stream<'a>: Sized + Clone + std::fmt::Debug + 'a {
        type Item;
        type ItemGroup;

        fn empty() -> Self;
        fn next(self) -> ParseResult<Self, Self::Item>;

        fn peek(&self) -> Option<Self::Item> {
            self.clone().next().map(|(b, _)| b)
        }

        fn take_while(
            self,
            pred: impl Fn(&Self::Item) -> bool,
        ) -> ParseResult<Self, Self::ItemGroup>;
    }

    pub trait U8Stream<'a>: Stream<'a, Item = u8, ItemGroup = &'a [u8]> {}

    impl<'a> Stream<'a> for &'a [u8] {
        type Item = u8;
        type ItemGroup = &'a [u8];

        fn empty() -> Self {
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
        fn run(&mut self, s: S) -> ParseResult<S, A>;

        fn run_in_place(&mut self, s: &mut S) -> Option<A> {
            let (a, s_next) = self.run(mem::replace(s, S::empty()))?;
            *s = s_next;
            Some(a)
        }

        fn map<B, F>(self, f: F) -> Map<Self, F, A>
        where
            F: Fn(A) -> B,
        {
            Map(self, f, Default::default())
        }

        fn inspect<F>(self, f: F) -> Inspect<Self, F, A>
        where
            F: Fn(&A, &S),
        {
            Inspect(self, f, Default::default())
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
        fn run(&mut self, s: S) -> ParseResult<S, B> {
            self.0.run(s).map(|(a, s)| (self.1(a), s))
        }
    }

    pub struct Inspect<PA, F, A>(PA, F, std::marker::PhantomData<A>);
    impl<'a, S, A, PA, F> Parser<'a, S, A> for Inspect<PA, F, A>
    where
        S: Stream<'a>,
        PA: Parser<'a, S, A>,
        F: Fn(&A, &S),
    {
        fn run(&mut self, s: S) -> ParseResult<S, A> {
            self.0.run(s).map(|(a, s)| {
                (self.1)(&a, &s);
                (a, s)
            })
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
        fn run(&mut self, s: S) -> ParseResult<S, B> {
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
        fn run(&mut self, s: S) -> ParseResult<S, B> {
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
        fn run(&mut self, s: S) -> ParseResult<S, A> {
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
        fn run(&mut self, s: S) -> ParseResult<S, A> {
            self.0.run(s.clone()).or_else(|| self.1.run(s))
        }
    }

    impl<'a, S, A, F> Parser<'a, S, A> for F
    where
        S: Stream<'a>,
        F: FnMut(S) -> ParseResult<S, A>,
    {
        fn run(&mut self, s: S) -> ParseResult<S, A> {
            self(s)
        }
    }

    macro_rules! gen_tuple_parser {
        ($($A:tt $PA:tt $a:tt $i:tt),+) => {
            impl<'a,S:Stream<'a>,$($A,)+$($PA,)+> Parser<'a,S,($($A,)+)> for ($($PA,)+)
            where $($PA: Parser<'a,S,$A>,)+
            {
                fn run(&mut self, s: S) -> ParseResult<S, ($($A,)+)> {
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

    pub fn fail<S, A>(_s: S) -> ParseResult<S, A> {
        None
    }

    pub fn eof<'a, S: Stream<'a>>(s: S) -> ParseResult<S, ()> {
        match s.next() {
            Some(_) => None,
            None => Some(((), S::empty())),
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
        mut p: impl Parser<'a, S, A>,
    ) -> impl Parser<'a, S, Option<A>> {
        move |s: S| match p.run(s.clone()) {
            Some((a, s_new)) => Some((Some(a), s_new)),
            None => Some((None, s)),
        }
    }

    pub fn fold<'a, S: Stream<'a>, A, F>(
        mut init: impl Parser<'a, S, A>,
        mut p: impl Parser<'a, S, F>,
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

    pub fn many<'a, S: Stream<'a>, A>(mut p: impl Parser<'a, S, A>) -> impl Parser<'a, S, Vec<A>> {
        move |mut s: S| {
            let mut result = Vec::new();
            while let Some((e, s_new)) = p.run(s.clone()) {
                result.push(e);
                s = s_new;
            }
            Some((result, s))
        }
    }

    pub fn many1<'a, S: Stream<'a>, A>(mut p: impl Parser<'a, S, A>) -> impl Parser<'a, S, Vec<A>> {
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
        mut p: impl Parser<'a, S, A>,
        mut p_sep: impl Parser<'a, S, ()>,
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
    pub fn unary_prefix<'a, S: Stream<'a>, A, F, PF, P>(p_op: PF, p: P) -> impl Parser<'a, S, A>
    where
        P: Parser<'a, S, A>,
        PF: Parser<'a, S, F>,
        F: FnMut(A) -> A + 'a,
    {
        (many(p_op), p)
            .map(move |(cs, e): (Vec<F>, A)| cs.into_iter().rev().fold(e, |acc, mut c| c(acc)))
    }

    pub fn unary_postfix<'a, S: Stream<'a>, A, F, PF, P>(p_op: PF, p: P) -> impl Parser<'a, S, A>
    where
        P: Parser<'a, S, A>,
        PF: Parser<'a, S, F>,
        F: FnMut(A) -> A + 'a,
    {
        (p, many(p_op)).map(move |(e, cs): (A, Vec<F>)| cs.into_iter().fold(e, |acc, mut c| c(acc)))
    }

    pub fn binary_lassoc<'a, S: Stream<'a>, A, F, PF, P>(
        mut p_op: PF,
        mut p: P,
    ) -> impl Parser<'a, S, A>
    where
        P: Parser<'a, S, A>,
        PF: Parser<'a, S, F>,
        F: FnMut(A, A) -> A + 'a,
    {
        move |mut s: S| {
            let (acc, s_new) = p.run(s)?;
            let mut acc = acc;
            s = s_new;

            while let Some((mut f, s_new)) = p_op.run(s.clone()) {
                let (b, s_new) = p.run(s_new)?;
                acc = f(acc, b);
                s = s_new;
            }

            Some((acc, s))
        }
    }

    pub fn binary_rassoc<'a, S: Stream<'a>, A, F, PF, P>(
        mut p_op: PF,
        mut p: P,
    ) -> impl Parser<'a, S, A>
    where
        P: Parser<'a, S, A>,
        PF: Parser<'a, S, F>,
        F: FnMut(A, A) -> A + 'a,
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
                .fold(acc, |acc, (b, mut f)| f(b, acc));

            Some((acc, s))
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

        pub fn take_while<'a, S: U8Stream<'a>>(
            pred: impl Fn(&u8) -> bool,
        ) -> impl Parser<'a, S, &'a [u8]> {
            move |s: S| s.take_while(|b| pred(b))
        }

        pub fn skip_while<'a, S: U8Stream<'a>>(
            pred: impl Fn(&S::Item) -> bool,
        ) -> impl Parser<'a, S, ()> {
            take_while(move |b| pred(b)).map(|_| ())
        }

        pub fn satisfy<'a, S: U8Stream<'a>>(f: impl Fn(&u8) -> bool) -> impl Parser<'a, S, u8> {
            move |s: S| s.next().filter(|(b, _)| f(b))
        }

        pub fn lit_byte<'a, S: U8Stream<'a>>(c: u8) -> impl Parser<'a, S, ()> {
            satisfy(move |b| b == &c).map(|_| ())
        }

        pub fn lit<'a, S: U8Stream<'a>>(keyword: &'static [u8]) -> impl Parser<'a, S, ()> {
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
            skip_while(|b: &u8| b.is_ascii_whitespace()).run(s)
        }

        pub fn rspaces<'a, S: U8Stream<'a>, A, P: Parser<'a, S, A>>(p: P) -> impl Parser<'a, S, A> {
            between(spaces, p, spaces)
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

mod ast {
    use core::str;
    use std::{
        cell::UnsafeCell,
        collections::{hash_map::Entry, BTreeSet, HashMap},
        io::Write,
        ptr::{self, NonNull},
    };

    use crate::{
        parser::{ParseResult, Parser, Stream, U8Stream},
        parser_fn,
    };

    type IdentCode = u32;

    #[derive(Debug, Clone)]
    pub enum Arg {
        Var(IdentCode),
        Num(i32),
    }

    #[derive(Debug, Clone, Copy)]
    pub enum BinOpCode {
        Add,
        Sub,
        Mul,
        Div,
        Mod,
    }

    impl BinOpCode {
        fn apply(&self, lhs: i32, rhs: i32) -> i32 {
            match self {
                BinOpCode::Add => lhs + rhs,
                BinOpCode::Sub => lhs - rhs,
                BinOpCode::Mul => lhs * rhs,
                BinOpCode::Div => lhs / rhs,
                BinOpCode::Mod => lhs % rhs,
            }
        }
    }

    #[derive(Debug, Clone)]
    pub enum Expr {
        BinOp(BinOpCode, Box<Expr>, Box<Expr>),
        Call(IdentCode, Box<Expr>),
        Num(i32),
        Var(u32),
    }

    #[derive(Debug, Clone)]
    pub enum Stmt {
        Eval(Expr),
        Assign(IdentCode, Expr),
        Def(IdentCode, Arg, Expr),
        Profile,
        Exit,
    }

    #[derive(Debug, Default)]
    pub struct Func {
        case_num: HashMap<i32, Expr>,
        case_default: Option<(IdentCode, Expr)>,

        n_def: u32,
        case_num_ln: HashMap<i32, u32>,
        case_default_ln: Option<u32>,
        n_calls: Vec<u32>,
    }

    #[derive(Debug, Default)]
    pub struct Ctx<'a> {
        ident_map: HashMap<&'a [u8], IdentCode>,
        ident_map_inv: HashMap<IdentCode, &'a [u8]>,
        vars: HashMap<IdentCode, i32>,
        funcs: HashMap<IdentCode, Func>,
        func_order: BTreeSet<&'a [u8]>,
    }

    #[derive(Debug, Clone)]
    pub struct CtxStream<'a, 'b: 'a> {
        buf: &'a [u8],
        _ctx: *mut Ctx<'b>, // we could have just used Rc<Ctx<'b>>, but that is way to heavy
    }

    impl<'a: 'b, 'b> CtxStream<'a, 'b> {
        pub fn new(buf: &'a [u8], ctx: &mut Ctx<'b>) -> Self {
            Self {
                buf,
                _ctx: ctx as *mut Ctx,
            }
        }

        pub fn ctx(&self) -> &Ctx<'b> {
            unsafe { &*self._ctx }
        }

        pub unsafe fn ctx_mut(&mut self) -> &mut Ctx<'b> {
            &mut *self._ctx
        }
    }

    impl<'a, 'b: 'a> Stream<'a> for CtxStream<'a, 'b> {
        type Item = u8;
        type ItemGroup = &'a [u8];

        fn empty() -> Self {
            Self {
                buf: &[],
                _ctx: ptr::null_mut(),
            }
        }

        fn next(self) -> ParseResult<Self, Self::Item> {
            self.buf.next().map(|(b, s)| {
                (
                    b,
                    Self {
                        buf: s,
                        _ctx: self._ctx,
                    },
                )
            })
        }

        fn take_while(
            self,
            pred: impl Fn(&Self::Item) -> bool,
        ) -> ParseResult<Self, Self::ItemGroup> {
            self.buf.take_while(pred).map(|(s, buf)| {
                (
                    s,
                    Self {
                        buf,
                        _ctx: self._ctx,
                    },
                )
            })
        }
    }

    impl<'a> U8Stream<'a> for CtxStream<'a, '_> {}

    pub fn parse_line<'a: 'b, 'b>(s: &'a [u8], ctx: &mut Ctx<'b>) -> Stmt {
        use crate::parser::{u8::*, *};
        macro_rules! parser_fn {
            ($name:ident: $T:ty = $body:expr) => {
                fn $name<'a: 'b, 'b>(s: CtxStream<'a, 'b>) -> ParseResult<CtxStream<'a, 'b>, $T> {
                    $body.run(s)
                }
            };
        }

        fn ident<'a: 'b, 'b>(s: CtxStream<'a, 'b>) -> ParseResult<CtxStream<'a, 'b>, IdentCode> {
            let (token, mut s) = rspaces(take_while(|b: &u8| b.is_ascii_alphabetic()))
                .filter(|t: &&[u8]| !t.is_empty())
                .run(s)?;
            let code = unsafe {
                let idx = s.ctx().ident_map.len() as u32;
                s.ctx_mut().ident_map_inv.insert(idx, token);
                *s.ctx_mut().ident_map.entry(token).or_insert_with(|| idx)
            };

            Some((code, s))
        }

        fn symbol<'a, S: U8Stream<'a>>(keyword: &'static [u8]) -> impl Parser<'a, S, ()> {
            move |s: S| rspaces(lit(keyword)).run(s)
        }

        parser_fn!(var: IdentCode = ident);
        parser_fn!(num: i32 = rspaces(uint).map(|x| x as i32));

        parser_fn!(pattern_arg: Arg = between(symbol(b"("), var.map(Arg::Var).or_else(num.map(Arg::Num)), symbol(b")")));

        parser_fn!(parens: Expr = between(symbol(b"("), expr, symbol(b")")));
        parser_fn!(call: Expr = (ident, parens).map(|(f, a)| Expr::Call(f, Box::new(a))));
        parser_fn!(factor: Expr = parens.or_else(call).or_else(var.map(Expr::Var)).or_else(num.map(Expr::Num)));

        parser_fn!(term: Expr = binary_lassoc(
            rspaces(
                single
                    .map_option(|op| match op {
                        b'*' => Some(BinOpCode::Mul),
                        b'/' => Some(BinOpCode::Div),
                        b'%' => Some(BinOpCode::Mod),
                        _ => None,
                    })
                    .map(|op| move |a, b| Expr::BinOp(op, Box::new(a), Box::new(b))),
            ),
            factor,
        ));

        parser_fn!(expr: Expr = binary_lassoc(
            rspaces(
                single
                    .map_option(|op| match op {
                        b'+' => Some(BinOpCode::Add),
                        b'-' => Some(BinOpCode::Sub),
                        _ => None,
                    })
                    .map(|op| move |a, b| Expr::BinOp(op, Box::new(a), Box::new(b))),
            ),
            term,
        ));

        parser_fn!(profile: Stmt = symbol(b"profile").map(|_| Stmt::Profile));
        parser_fn!(exit: Stmt = symbol(b"exit").map(|_| Stmt::Exit));
        parser_fn!(assign: Stmt = (symbol(b"set"), ident, symbol(b"="), expr).map(|(_, var, _, e)| Stmt::Assign(var, e)));
        parser_fn!(def: Stmt = (symbol(b"def"), ident, pattern_arg, symbol(b"="), expr).map(|(_, f, a, _, e)| Stmt::Def(f, a, e)));
        parser_fn!(eval: Stmt = expr.map(Stmt::Eval));
        parser_fn!(stmt: Stmt = profile.or_else(exit).or_else(assign).or_else(def).or_else(eval));

        let s = CtxStream::new(s, ctx);
        let (result, _) = stmt(s).expect("parse error");
        result
    }

    impl Expr {
        pub fn eval(&self, ctx: &mut Ctx) -> i32 {
            let Ctx {
                ref mut funcs,
                ref mut vars,
                ..
            } = ctx;
            self.eval_inner(funcs.into(), vars)
        }

        fn eval_inner(
            &self,
            mut funcs: NonNull<HashMap<IdentCode, Func>>,
            vars: &mut HashMap<IdentCode, i32>,
        ) -> i32 {
            let result = match self {
                Expr::BinOp(op, a, b) => {
                    op.apply(a.eval_inner(funcs, vars), b.eval_inner(funcs, vars))
                }
                Expr::Call(vf, a) => {
                    let a = a.eval_inner(funcs, vars);
                    let f = unsafe { funcs.as_mut() }.get_mut(vf).unwrap();
                    if let Some(e) = f.case_num.get(&a) {
                        f.n_calls[f.case_num_ln[&a] as usize] += 1;
                        e.eval_inner(funcs, vars)
                    } else {
                        f.n_calls[f.case_default_ln.unwrap() as usize] += 1;
                        let (bound_var, e) = f.case_default.as_ref().unwrap();
                        let bound_var = *bound_var;
                        let old = vars.insert(bound_var, a);
                        let result = e.eval_inner(funcs, vars);
                        old.map(|old| vars.insert(bound_var, old));
                        result
                    }
                }
                Expr::Num(x) => *x,
                Expr::Var(v) => vars[v],
            };
            result
        }
    }

    impl Stmt {
        pub fn eval(self, ctx: &mut Ctx, mut f: impl Write) {
            match self {
                Stmt::Eval(e) => {
                    let value = e.eval(ctx);
                    writeln!(f, ">> {}", value).unwrap();
                }
                Stmt::Assign(v, e) => {
                    let x = e.eval(ctx);
                    ctx.vars.insert(v, x);
                }
                Stmt::Def(f, arg, e) => {
                    if !ctx.funcs.contains_key(&f) {
                        ctx.func_order.insert(ctx.ident_map_inv[&f]);
                    }
                    let f = ctx.funcs.entry(f).or_default();
                    f.n_calls.push(0);
                    match arg {
                        Arg::Num(x) => {
                            f.case_num.entry(x).or_insert(e);
                            f.case_num_ln.entry(x).or_insert(f.n_def);
                        }
                        Arg::Var(v) => {
                            f.case_default = Some((v, e));
                            if f.case_default_ln == None {
                                f.case_default_ln = Some(f.n_def);
                            }
                        }
                    }
                    f.n_def += 1;
                }
                Stmt::Profile => {
                    for func_name in &ctx.func_order {
                        let func_code = &ctx.ident_map[func_name];
                        let func = ctx.funcs.get_mut(func_code).unwrap();
                        //let profile = e.profile(ctx);

                        write!(f, "{} calls: ", unsafe {
                            str::from_utf8_unchecked(func_name)
                        })
                        .unwrap();
                        for freq in &func.n_calls {
                            write!(f, "{} ", freq).unwrap();
                        }
                        writeln!(f, "=> {}", func.n_calls.iter().sum::<u32>()).unwrap();
                        func.n_calls.fill(0);
                    }
                }
                Stmt::Exit => {}
            }
            //
        }
    }
}

fn main() {
    let input = std::io::read_to_string(std::io::stdin()).unwrap();
    let mut output = simple_io::stdout_buf();

    let mut ctx = ast::Ctx::default();
    for line in input.lines() {
        let line = line.as_bytes();
        let stmt = ast::parse_line(line, &mut ctx);

        stmt.eval(&mut ctx, &mut output);
    }
}
