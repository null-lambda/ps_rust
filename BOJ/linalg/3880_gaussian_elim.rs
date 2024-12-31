use num::Frac64;
use std::{collections::HashMap, io::Write};

mod simple_io {
    pub struct InputAtOnce<'a> {
        _buf: String,
        iter: std::str::SplitAsciiWhitespace<'a>,
    }

    impl<'a> InputAtOnce<'a> {
        pub fn token(&mut self) -> &'a str {
            self.iter.next().unwrap_or_default()
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().unwrap()
        }
    }

    pub fn stdin_at_once<'a>() -> InputAtOnce<'a> {
        let _buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let iter = _buf.split_ascii_whitespace();
        let iter = unsafe { std::mem::transmute(iter) };
        InputAtOnce { _buf, iter }
    }

    pub fn stdout() -> std::io::BufWriter<std::io::Stdout> {
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
            pred: impl Fn(&u8) -> bool,
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

pub mod num {
    pub fn gcd(mut a: u64, mut b: u64) -> u64 {
        while b != 0 {
            let temp = b;
            b = a % b;
            a = temp;
        }
        a
    }

    pub fn lcm(a: u64, b: u64) -> u64 {
        a / gcd(a, b) * b
    }

    use std::ops::{Add, Div, Mul, Sub};

    #[derive(Debug, Clone, Copy)]
    pub struct Frac64(i64, i64);

    impl Frac64 {
        pub fn new(a: i64, b: i64) -> Self {
            assert!(b > 0);
            Self(a, b).normalize()
        }

        pub fn numer(&self) -> i64 {
            self.0
        }

        pub fn denom(&self) -> i64 {
            self.1
        }

        pub fn inner(&self) -> (i64, i64) {
            (self.0, self.1)
        }

        pub fn normalize(self) -> Self {
            let Self(a, b) = self;
            let d = gcd(a.abs() as u64, b.abs() as u64) as i64;
            Self(a / d, b / d)
        }

        pub fn zero() -> Self {
            Self(0, 1)
        }

        pub fn one() -> Self {
            Self(1, 1)
        }

        pub fn abs(self) -> Self {
            Self(self.0.abs(), self.1)
        }
    }

    impl Add for Frac64 {
        type Output = Self;
        fn add(self, rhs: Self) -> Self {
            Self::new(self.0 * rhs.1 + rhs.0 * self.1, self.1 * rhs.1)
        }
    }

    impl Sub for Frac64 {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self {
            Self::new(self.0 * rhs.1 - rhs.0 * self.1, self.1 * rhs.1)
        }
    }

    impl Mul for Frac64 {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self {
            Self::new(self.0 * rhs.0, self.1 * rhs.1)
        }
    }

    impl Div for Frac64 {
        type Output = Self;
        fn div(self, rhs: Self) -> Self {
            let s = rhs.0.signum();
            Self::new(self.0 * rhs.1 * s, self.1 * rhs.0 * s)
        }
    }

    impl From<i64> for Frac64 {
        fn from(a: i64) -> Self {
            Self::new(a, 1)
        }
    }

    impl PartialEq for Frac64 {
        fn eq(&self, other: &Self) -> bool {
            self.0 * other.1 == other.0 * self.1
        }
    }

    impl Eq for Frac64 {}

    impl PartialOrd for Frac64 {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some((self.0 * other.1).cmp(&(other.0 * self.1)))
        }
    }

    impl Ord for Frac64 {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.partial_cmp(other).unwrap()
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct Matrix<T> {
    n_rows: usize,
    n_cols: usize,
    data: Vec<T>,
}

// gaussian elimination
fn rref(mat: &Matrix<Frac64>) -> (usize, Frac64, Matrix<Frac64>) {
    let Matrix {
        n_rows,
        n_cols,
        ref data,
    } = *mat;

    let mut jagged: Vec<Vec<_>> = data.chunks_exact(n_cols).map(|t| t.to_vec()).collect();
    let mut det = Frac64::one();

    let mut rank = 0;
    for c in 0..n_cols {
        // let Some(pivot) = (rank..n_rows).find(|&j| jagged[j][c] != Frac64::zero()) else {
        //     continue;
        // };
        let Some(pivot) = (rank..n_rows)
            .filter(|&j| jagged[j][c] != Frac64::zero())
            .max_by_key(|&j| jagged[j][c].abs())
        else {
            continue;
        };
        jagged.swap(rank, pivot);

        det = det * jagged[rank][c];
        let inv_x = Frac64::one() / jagged[rank][c];
        for j in 0..n_cols {
            jagged[rank][j] = jagged[rank][j] * inv_x;
        }

        for i in 0..n_rows {
            if i == rank {
                continue;
            }
            let coeff = jagged[i][c];
            for j in 0..n_cols {
                jagged[i][j] = jagged[i][j] - jagged[rank][j] * coeff;
            }
        }
        rank += 1;
    }

    let mat = Matrix {
        n_rows,
        n_cols,
        data: jagged.into_iter().flatten().collect(),
    };
    (rank, det, mat)
}

type Elem = (u8, Option<u8>);
type Molecule = Vec<(Elem, u32)>;
type Eqn = (Vec<Molecule>, Vec<Molecule>);

fn parse_eqn(s: &[u8]) -> Option<Eqn> {
    use parser::u8::*;
    use parser::*;

    parser_fn!(let elem: Elem = (satisfy(|&b| b.is_ascii_uppercase()), optional(satisfy(|&b| b.is_ascii_lowercase()))));
    parser_fn!(let parens: Molecule = between(lit(b"("), molecule, lit(b")")));
    parser_fn!(let unit_group: Molecule = elem.map(|e| vec![(e, 1)]).or_else(parens));
    parser_fn!(let group: Molecule = (unit_group, optional(uint)).map(|(mut g, n)| {
        if let Some(n) = n {
            for (_, ref mut x) in g.iter_mut() {
                *x *= n as u32;
            }
        }
        g
    }));
    parser_fn!(let molecule: Molecule = many1(group).map(|gs| gs.into_iter().flatten().collect()));
    parser_fn!(let seq: Vec<Molecule> = many_sep1(molecule, lit(b"+")));
    parser_fn!(let eqn: Eqn = (seq, lit(b"->"), seq, lit(b".")).map(|(lhs, _, rhs, _)| (lhs, rhs)));

    Some(eqn.run(s)?.0)
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    loop {
        let s = input.token().as_bytes();
        if s == b"." {
            break;
        }

        let eqn = parse_eqn(s).unwrap();

        let mut elem_to_idx: HashMap<Elem, usize> = HashMap::new();
        for seq in [&eqn.0, &eqn.1] {
            for molecule in seq {
                for &(elem, _) in molecule {
                    let next_idx = elem_to_idx.len();
                    elem_to_idx.entry(elem).or_insert(next_idx);
                }
            }
        }

        // Build augmented matrix, and find its kernel basis
        let n_elems = elem_to_idx.len();
        let n_molecules = eqn.0.len() + eqn.1.len();

        let n_rows = n_molecules;
        let n_cols = n_elems + n_molecules;
        let mut data = vec![Frac64::zero(); n_rows * n_cols];
        let mut r = 0;
        for (sign, seq) in [(1, &eqn.0), (-1, &eqn.1)] {
            for molecule in seq {
                for &(elem, count) in molecule {
                    let c = elem_to_idx[&elem];
                    data[r * n_cols + c] = data[r * n_cols + c] + Frac64::from(sign * count as i64);
                }
                data[r * n_cols + n_elems + r] = Frac64::one();
                r += 1;
            }
        }

        let mat = Matrix {
            n_rows,
            n_cols,
            data,
        };
        let (_, _, mat) = rref(&mat);

        let mut kernel_basis: Vec<Vec<Frac64>> = vec![];
        for i in 0..n_rows {
            // if 0..n_elems is alal zero
            if mat.data[i * n_cols..][..n_elems]
                .iter()
                .all(|&x| x == Frac64::zero())
            {
                kernel_basis.push(mat.data[i * n_cols..][n_elems..].to_vec());
            }
        }
        assert_eq!(kernel_basis.len(), 1);
        let coeff = kernel_basis.pop().unwrap();

        // normalize
        let common_denom = coeff[0].numer().signum()
            * coeff
                .iter()
                .map(|x| x.denom())
                .fold(1u64, |acc, x| num::lcm(acc as u64, x.abs() as u64)) as i64;
        let coeff: Vec<i64> = coeff
            .iter()
            .map(|x| x.numer() * common_denom / x.denom())
            .collect();

        for c in coeff {
            write!(output, "{} ", c).unwrap();
        }
        writeln!(output).unwrap();
    }
}
