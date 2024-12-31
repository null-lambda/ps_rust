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
            // between(spaces, p, spaces)
            (p, spaces).map(|(a, _)| a)
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

pub mod ast {
    use std::fmt;
    use std::iter::{Product, Sum};
    use std::ops::{Add, Mul};

    use crate::parser_fn;

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum Expr {
        Var(u8),
        Or(Vec<Expr>),
        And(Vec<Expr>),
        Neg(Box<Expr>),
    }

    impl fmt::Display for Expr {
        fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
            match self {
                Expr::Var(v) => write!(f, "{}", *v as char),
                Expr::Neg(e) => write!(f, "~{}", e),
                Expr::Or(es) => {
                    write!(f, "(+")?;
                    for e in es {
                        write!(f, "{}", e)?;
                    }
                    write!(f, ")")
                }
                Expr::And(es) => {
                    write!(f, "(*")?;
                    for e in es {
                        write!(f, "{}", e)?;
                    }
                    write!(f, ")")
                }
            }
        }
    }

    use crate::parser::u8::*;
    use crate::parser::*;
    fn lit<'a, S: U8Stream<'a>>(t: &'static [u8]) -> impl Parser<'a, S, ()> {
        rspaces(crate::parser::u8::lit(t))
    }

    parser_fn!(let varname: u8 = rspaces(satisfy(|b| b.is_ascii_lowercase())));
    parser_fn!(let var: Expr = varname.map(Expr::Var));
    parser_fn!(let neg: Expr = (lit(b"~"), expr).map(|(_, e)| Expr::Neg(Box::new(e))));
    parser_fn!(let or: Expr = between(lit(b"(+"), many(expr), lit(b")")).map(Expr::Or));
    parser_fn!(let and: Expr = between(lit(b"(*"), many(expr), lit(b")")).map(Expr::And));
    parser_fn!(let expr: Expr = or.or_else(and).or_else(neg).or_else(var));

    pub fn parse_expr(s: &[u8]) -> ParseResult<&[u8], Expr> {
        expr(s)
    }

    pub fn parse_axiom<'a>(s: &'a [u8]) -> Option<(Vec<bool>, &'a [u8])> {
        (lit(b"("), many(varname), lit(b")"))
            .map(|(_, xs, _)| {
                let mut res = vec![false; 26];
                for c in xs {
                    res[(c - b'a') as usize] = true;
                }
                res
            })
            .run(s)
    }

    pub fn parse_num(s: &[u8]) -> Option<(i32, &[u8])> {
        rspaces(int).run(s)
    }

    pub fn push_neg(e: Expr) -> Expr {
        use Expr::*;
        match e {
            Neg(f) => match *f {
                Neg(g) => push_neg(*g),
                And(gs) => Or(gs.into_iter().map(|g| push_neg(Neg(Box::new(g)))).collect()),
                Or(gs) => And(gs.into_iter().map(|g| push_neg(Neg(Box::new(g)))).collect()),
                _ => Neg(Box::new(push_neg(*f))),
            },
            And(fs) => And(fs.into_iter().map(push_neg).collect()),
            Or(fs) => Or(fs.into_iter().map(push_neg).collect()),
            _ => e,
        }
    }

    pub fn push_and(e: Expr) -> Expr {
        use Expr::*;
        match e {
            And(fs) => match fs.iter().position(|f| matches!(f, Or(..))) {
                Some(i) => {
                    let (head, mid) = fs.split_at(i);
                    let (mid, tail) = mid.split_at(1);
                    let Or(gs) = &mid[0] else { unreachable!() };
                    Or(gs
                        .into_iter()
                        .map(|g| {
                            push_and(And((head.iter().cloned())
                                .chain(std::iter::once(g.clone()))
                                .chain(tail.iter().cloned())
                                .collect()))
                        })
                        .collect())
                }
                None => And(fs.into_iter().map(push_and).collect()),
            },
            Or(fs) => Or(fs.into_iter().map(push_and).collect()),
            _ => e,
        }
    }

    pub fn flatten(e: Expr) -> Expr {
        use Expr::*;
        match e {
            And(fs) => {
                let mut result = vec![];
                for f in fs {
                    match f {
                        And(gs) => result.extend(gs.into_iter().map(flatten)),
                        _ => result.push(flatten(f)),
                    }
                }
                if result.len() == 1 {
                    result.pop().unwrap()
                } else {
                    And(result)
                }
            }
            Or(fs) => {
                let mut result = vec![];
                for f in fs {
                    match f {
                        Or(gs) => result.extend(gs.into_iter().map(flatten)),
                        _ => result.push(flatten(f)),
                    }
                }
                if result.len() == 1 {
                    result.pop().unwrap()
                } else {
                    Or(result)
                }
            }
            _ => e,
        }
    }

    pub fn normalize(mut e: Expr) -> Expr {
        loop {
            // The full ACMNF should be not calculated explicitly and be lazily evaluated,
            // since 'push_and' (term rewriting rule 8)
            // expands the formula with exponential space complexity.
            let e_prev = e.clone();
            e = push_neg(e);
            // e = push_and(e);
            e = flatten(e);
            if e == e_prev {
                return e;
            }
        }
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Bounded<const MAX: u64>(pub u64);

    impl<const MAX: u64> Add for Bounded<MAX> {
        type Output = Self;
        fn add(self, rhs: Self) -> Self::Output {
            Self((self.0 + rhs.0) % MAX)
        }
    }

    impl<const MAX: u64> Mul for Bounded<MAX> {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self::Output {
            Self((self.0 * rhs.0) % MAX)
        }
    }

    impl<const MAX: u64> Sum for Bounded<MAX> {
        fn sum<I: Iterator<Item = Self>>(iter: I) -> Self {
            iter.fold(Self(0), |acc, x| acc + x)
        }
    }

    impl<const MAX: u64> Product for Bounded<MAX> {
        fn product<I: Iterator<Item = Self>>(iter: I) -> Self {
            iter.fold(Self(1), |acc, x| acc * x)
        }
    }

    impl From<u64> for Bounded<1_000_000_007> {
        fn from(x: u64) -> Self {
            Self(x % 1_000_000_007)
        }
    }

    type BoundedU32 = Bounded<1_000_000_007>;

    pub fn count_proofs(e: &Expr, axiom: &[bool]) -> BoundedU32 {
        match e {
            Expr::Or(fs) => fs.iter().map(|f| count_proofs(f, axiom)).sum(),
            Expr::And(fs) => fs.iter().map(|f| count_proofs(f, axiom)).product(),
            Expr::Var(v) => (axiom[(*v - b'a') as usize] as u64).into(),
            Expr::Neg(f) => match f.as_ref() {
                Expr::Var(v) => (!axiom[(*v - b'a') as usize] as u64).into(),
                _ => panic!(),
            },
        }
    }

    pub fn nth_proof(e: &Expr, axiom: &[bool], mut n: usize) -> Option<Expr> {
        match e {
            Expr::Or(fs) => {
                for f in fs {
                    let l = count_proofs(f, axiom).0 as usize;
                    if n < l {
                        return nth_proof(f, axiom, n);
                    }

                    n -= l;
                }
                None
            }
            Expr::And(fs) => {
                let mut ls = vec![];
                for f in fs.iter().rev() {
                    let l = count_proofs(f, axiom).0 as usize;
                    if l == 0 {
                        return None;
                    }

                    ls.push(n % l);
                    n /= l;
                }
                let mut res = vec![];
                for (f, l) in fs.iter().zip(ls.iter().rev()) {
                    let Some(proof) = nth_proof(f, axiom, *l) else {
                        return None;
                    };
                    res.push(proof);
                }
                Some(flatten(Expr::And(res)))
            }
            Expr::Var(..) | Expr::Neg(..) => {
                (n == 0 && count_proofs(e, axiom).0 == 1).then(|| e.clone())
            }
        }
    }
}

fn main() {
    use parser::Parser;

    // let mut input = simple_io::stdin_at_once();
    let buf = std::io::read_to_string(std::io::stdin()).unwrap();
    let mut s = buf.trim().as_bytes();
    let mut output = simple_io::stdout_buf();

    loop {
        let Some(mut expr) = ast::parse_expr.run_in_place(&mut s) else {
            break;
        };
        expr = ast::normalize(expr);
        // println!("{}", expr);

        // let axiom = ast::parse_axiom.run_in_place(&mut s).unwrap();
        let axiom = vec![true; 26];

        let m = ast::count_proofs(&expr, &axiom).0 as usize;
        let mut pos = 0;
        loop {
            let k = ast::parse_num.run_in_place(&mut s).unwrap();
            if k == 0 {
                break;
            }

            if m == 0 {
                continue;
            }

            if k > 0 {
                for _ in 0..k {
                    writeln!(output, "{}", ast::nth_proof(&expr, &axiom, pos).unwrap()).unwrap();
                    pos = (pos + 1) % m;
                }
            } else {
                pos = (pos + k.abs() as usize) % m;
            }
        }
    }
}
