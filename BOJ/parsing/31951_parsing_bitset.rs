use std::io::Write;

use buffered_io::BufReadExt;

use crate::bitset::BitVec;

mod buffered_io {
    use std::io::{BufRead, BufReader, BufWriter, Stdin, Stdout};
    use std::str::FromStr;

    pub trait BufReadExt: BufRead {
        fn line(&mut self) -> String {
            let mut buf = String::new();
            self.read_line(&mut buf).unwrap();
            buf
        }

        fn skip_line(&mut self) {
            self.line();
        }

        fn token(&mut self) -> String {
            loop {
                let buf = self.fill_buf().unwrap();
                if buf.is_empty() {
                    return String::new();
                }

                let mut i = 0;
                while i < buf.len() && buf[i].is_ascii_whitespace() {
                    i += 1;
                }

                let should_break = i < buf.len();
                self.consume(i);
                if should_break {
                    break;
                }
            }

            let mut res = vec![];
            loop {
                let buf = self.fill_buf().unwrap();
                if buf.is_empty() {
                    break;
                }

                let mut i = 0;
                while i < buf.len() && !buf[i].is_ascii_whitespace() {
                    i += 1;
                }
                res.extend_from_slice(&buf[..i]);

                let should_break = i < buf.len();
                self.consume(i);
                if should_break {
                    break;
                }
            }

            String::from_utf8(res).unwrap()
        }

        fn try_value<T: FromStr>(&mut self) -> Option<T> {
            self.token().parse().ok()
        }

        fn value<T: FromStr>(&mut self) -> T {
            self.try_value().unwrap()
        }
    }

    impl<R: BufRead> BufReadExt for R {}

    pub fn stdin() -> BufReader<Stdin> {
        BufReader::new(std::io::stdin())
    }

    pub fn stdout() -> BufWriter<Stdout> {
        BufWriter::new(std::io::stdout())
    }
}

#[macro_use]
pub mod parser {
    use std::{cell::RefCell, mem, rc::Rc};

    pub type ParseResult<S, A> = Option<(A, S)>;

    pub trait Stream: Sized + Clone + std::fmt::Debug {
        type Item;
        type ItemGroup<'a>
        where
            Self: 'a;

        fn empty() -> Self;
        fn next(self) -> ParseResult<Self, Self::Item>;

        fn peek(&self) -> Option<Self::Item> {
            self.clone().next().map(|(b, _)| b)
        }

        fn take_while<'a>(
            self,
            pred: impl Fn(&Self::Item) -> bool,
        ) -> ParseResult<Self, Self::ItemGroup<'a>>;

        fn skip_while(self, pred: impl Fn(&Self::Item) -> bool) -> ParseResult<Self, ()> {
            let (_, s) = self.take_while(pred)?;
            Some(((), s))
        }
    }

    impl<'a, T: std::fmt::Debug + Clone> Stream for &'a [T] {
        type Item = T;
        type ItemGroup<'b>
            = &'b [T]
        where
            'a: 'b;

        fn empty() -> Self {
            &[]
        }

        fn next(self) -> ParseResult<Self, T> {
            self.split_first().map(|(b, s)| (b.clone(), s))
        }

        fn take_while<'b>(self, pred: impl Fn(&T) -> bool) -> ParseResult<Self, &'b [T]>
        where
            'a: 'b,
        {
            let mut i = 0;
            while i < self.len() && pred(&self[i]) {
                i += 1;
            }
            Some(self.split_at(i))
        }
    }

    pub trait Parser<S: Stream, A>: Sized {
        fn run(&mut self, s: S) -> ParseResult<S, A>;

        fn run_in_place(&mut self, s: &mut S) -> Option<A> {
            let (a, s_next) = self.run(mem::replace(s, S::empty()))?;
            *s = s_next;
            Some(a)
        }

        fn map<B>(mut self, mut f: impl FnMut(A) -> B) -> impl Parser<S, B> {
            move |s| self.run(s).map(|(a, s)| (f(a), s))
        }

        fn inspect(mut self, mut f: impl FnMut(&A, &S)) -> impl Parser<S, A> {
            move |s| {
                self.run(s).map(|(a, s)| {
                    f(&a, &s);
                    (a, s)
                })
            }
        }

        fn and_then<B, PB: Parser<S, B>>(
            mut self,
            mut p: impl FnMut(A) -> PB,
        ) -> impl Parser<S, B> {
            move |s| self.run(s).and_then(|(a, s)| p(a).run(s))
        }

        fn map_option<B>(mut self, mut f: impl FnMut(A) -> Option<B>) -> impl Parser<S, B> {
            move |s| self.run(s).and_then(|(a, s)| f(a).map(|b| (b, s)))
        }

        fn filter(mut self, mut f: impl FnMut(&A) -> bool) -> impl Parser<S, A> {
            move |s| self.run(s).filter(|(a, _)| f(a))
        }

        fn or(mut self, mut p: impl Parser<S, A>) -> impl Parser<S, A> {
            move |s: S| self.run(s.clone()).or_else(|| p.run(s))
        }

        fn optional(mut self) -> impl Parser<S, Option<A>> {
            move |s: S| {
                Some(match self.run(s.clone()) {
                    Some((a, s_new)) => (Some(a), s_new),
                    None => (None, s),
                })
            }
        }

        fn seq(mut self) -> impl Parser<S, Vec<A>> {
            move |mut s: S| {
                let mut result = Vec::new();
                while let Some((e, s_new)) = self.run(s.clone()) {
                    result.push(e);
                    s = s_new;
                }
                Some((result, s))
            }
        }

        fn seq1(mut self) -> impl Parser<S, Vec<A>> {
            move |mut s: S| {
                let mut result = Vec::new();
                let (e, s_new) = self.run(s.clone())?;
                result.push(e);
                s = s_new;

                while let Some((e, s_new)) = self.run(s.clone()) {
                    result.push(e);
                    s = s_new;
                }
                Some((result, s))
            }
        }

        fn seq_sep1(mut self, mut p_sep: impl Parser<S, ()>) -> impl Parser<S, Vec<A>> {
            move |mut s: S| -> ParseResult<S, Vec<A>> {
                let mut result = Vec::new();
                let (e, s_new) = self.run(s)?;
                result.push(e);
                s = s_new;

                while let Some((_, s_new)) = p_sep.run(s.clone()) {
                    match self.run(s_new) {
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

        fn fold_with<F>(mut self, mut p: impl Parser<S, F>) -> impl Parser<S, A>
        where
            F: Fn(A) -> A,
        {
            move |mut s: S| {
                let (mut acc, s_new) = self.run(s)?;
                s = s_new;
                while let Some((f, s_new)) = p.run(s.clone()) {
                    acc = f(acc);
                    s = s_new;
                }
                Some((acc, s))
            }
        }
    }

    impl<S: Stream, A, F: FnMut(S) -> ParseResult<S, A>> Parser<S, A> for F {
        fn run(&mut self, s: S) -> ParseResult<S, A> {
            self(s)
        }
    }

    impl<S: Stream, A> Parser<S, A> for Rc<RefCell<dyn FnMut(S) -> ParseResult<S, A>>> {
        fn run(&mut self, s: S) -> ParseResult<S, A> {
            self.borrow_mut()(s)
        }
    }

    macro_rules! gen_tuple_parser {
        (@tuple $($A:tt $PA:tt $a:tt),+) => {
            impl<S:Stream,$($A,)+$($PA,)+> Parser<S,($($A,)+)> for ($($PA,)+)
            where $($PA: Parser<S,$A>,)+
            {
                fn run(&mut self, mut s: S) -> ParseResult<S, ($($A,)+)> {
                    let ($($a,)+) = self;
                    let a = ($($a.run_in_place(&mut s)?,)+);
                    Some((a, s))
                }
            }
        };

        () => {};
        ($A:tt $PA:tt $a:tt$(, $($rest:tt)+)?) => {
            $(gen_tuple_parser!($($rest)+);)?
            gen_tuple_parser!(@tuple $A $PA $a$(, $($rest)+)?);
        };
    }

    gen_tuple_parser!(A0 PA0 pa0, A1 PA1 pa1, A2 PA2 pa2, A3 PA3 pa3, A4 PA4 pa4, A5 PA5 pa5);

    pub fn fail<S, A>(_s: S) -> ParseResult<S, A> {
        None
    }

    pub fn eof<S: Stream>(s: S) -> ParseResult<S, ()> {
        match s.next() {
            Some(_) => None,
            None => Some(((), S::empty())),
        }
    }

    pub fn single<S: Stream>(s: S) -> ParseResult<S, S::Item> {
        s.next()
    }

    pub fn between<S: Stream, A, O, E>(
        p_open: impl Parser<S, O>,
        p: impl Parser<S, A>,
        p_close: impl Parser<S, E>,
    ) -> impl Parser<S, A> {
        (p_open, p, p_close).map(|(_, a, _)| a)
    }

    // chained unary operators
    pub fn unary_prefix<S: Stream, A, F, PF, P>(p_op: PF, p: P) -> impl Parser<S, A>
    where
        P: Parser<S, A>,
        PF: Parser<S, F>,
        F: FnMut(A) -> A,
    {
        (p_op.seq(), p)
            .map(move |(cs, e): (Vec<F>, A)| cs.into_iter().rev().fold(e, |acc, mut c| c(acc)))
    }

    pub fn unary_postfix<S: Stream, A, F, PF, P>(p_op: PF, p: P) -> impl Parser<S, A>
    where
        P: Parser<S, A>,
        PF: Parser<S, F>,
        F: FnMut(A) -> A,
    {
        (p, p_op.seq()).map(move |(e, cs): (A, Vec<F>)| cs.into_iter().fold(e, |acc, mut c| c(acc)))
    }

    pub fn binary_lassoc<S: Stream, A, F, PF, P>(mut p_op: PF, mut p: P) -> impl Parser<S, A>
    where
        P: Parser<S, A>,
        PF: Parser<S, F>,
        F: FnMut(A, A) -> A,
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

    pub fn binary_rassoc<S: Stream, A, F, PF, P>(mut p_op: PF, mut p: P) -> impl Parser<S, A>
    where
        P: Parser<S, A>,
        PF: Parser<S, F>,
        F: FnMut(A, A) -> A,
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

    pub fn take_while<'a, S: Stream + 'a>(
        pred: impl Fn(&S::Item) -> bool,
    ) -> impl Parser<S, S::ItemGroup<'a>> {
        move |s: S| s.take_while(|b| pred(b))
    }

    pub fn skip_while<S: Stream>(pred: impl Fn(&S::Item) -> bool) -> impl Parser<S, ()> {
        move |s: S| s.skip_while(|b| pred(b))
    }

    pub fn satisfy<S: Stream>(f: impl Fn(&S::Item) -> bool) -> impl Parser<S, S::Item> {
        move |s: S| s.next().filter(|(b, _)| f(b))
    }

    pub fn lit_single<S: Stream>(c: S::Item) -> impl Parser<S, ()>
    where
        S::Item: PartialEq,
    {
        satisfy(move |b| b == &c).map(|_| ())
    }

    pub fn lit<S: Stream>(keyword: &'static [S::Item]) -> impl Parser<S, ()>
    where
        S::Item: PartialEq,
    {
        move |mut s: S| {
            for c in keyword {
                let (b, s_new) = s.next()?;
                if &b != c {
                    return None;
                }
                s = s_new;
            }
            Some(((), s))
        }
    }

    pub mod u8 {
        use super::*;

        // macro to quick define a 0-ary parser
        #[macro_export]
        macro_rules! parser_fn {
            ($name:ident: $T:ty = $body:expr) => {
                fn $name<'a>(s: &'a [u8]) -> ParseResult<&'a [u8], $T> {
                    $body.run(s)
                }
            };
        }

        pub fn spaces<S: Stream<Item = u8>>(s: S) -> ParseResult<S, ()> {
            skip_while(|b: &u8| b.is_ascii_whitespace()).run(s)
        }

        pub fn rspaces<S: Stream<Item = u8>, A, P: Parser<S, A>>(p: P) -> impl Parser<S, A> {
            between(spaces, p, spaces)
        }

        pub fn uint<S: Stream<Item = u8>>(s: S) -> ParseResult<S, i32> {
            let mut s = s;
            let (b, s_new) = satisfy(|b: &u8| b.is_ascii_digit()).run(s.clone())?;
            let mut result = (b - b'0') as i32;
            s = s_new;

            while let Some((b, s_new)) = satisfy(|b: &u8| b.is_ascii_digit()).run(s.clone()) {
                result = result * 10 + (b - b'0') as i32;
                s = s_new;
            }

            Some((result, s))
        }

        pub fn int<S: Stream<Item = u8>>(s: S) -> ParseResult<S, i32> {
            let mut s = s;
            let mut sign = 1;
            if let Some((_, s_new)) = satisfy(|&b| b == b'-').run(s.clone()) {
                sign = -1;
                s = s_new;
            }

            // at least one digit is required
            let (b, s_new) = satisfy(|b: &u8| b.is_ascii_digit()).run(s.clone())?;
            let mut result = (b - b'0') as i32;
            s = s_new;

            while let Some((b, s_new)) = satisfy(|b: &u8| b.is_ascii_digit()).run(s.clone()) {
                result = result * 10 + (b - b'0') as i32;
                s = s_new;
            }

            Some((result * sign, s))
        }
    }
}

pub mod bitset {
    // TODO: avx2
    // TODO: forward
    // TODO: empiricial test

    use std::ops::*;

    pub type B = u64;
    pub const BW: usize = 64;

    #[derive(Clone)]
    pub struct BitVec(pub Vec<B>);

    impl BitVec {
        pub fn zero_bits(n: usize) -> Self {
            Self(vec![0; n.div_ceil(BW)])
        }

        pub fn one_bits(n: usize) -> Self {
            let mut res = Self(vec![!0; n.div_ceil(BW)]);
            if n % BW != 0 {
                res.0[n / BW] = (1 << n % BW) - 1;
            }
            res
        }

        pub fn bitlen(&self) -> usize {
            self.0.len() * BW
        }

        pub fn bit_trunc(&mut self, n: usize) {
            let q = n.div_ceil(BW);
            if q > self.0.len() {
                return;
            }
            self.0.truncate(q);
            if n % BW != 0 {
                self.0[q - 1] &= (1 << n % BW) - 1;
            }
        }

        pub fn get(&self, i: usize) -> bool {
            let (b, s) = (i / BW, i % BW);
            (self.0[b] >> s) & 1 != 0
        }
        #[inline]
        pub fn set(&mut self, i: usize, value: bool) {
            if !value {
                self.0[i / BW] &= !(1 << i % BW);
            } else {
                self.0[i / BW] |= 1 << i % BW;
            }
        }
        #[inline]
        pub fn toggle(&mut self, i: usize) {
            self.0[i / BW] ^= 1 << i % BW;
        }

        pub fn count_ones(&self) -> u32 {
            self.0.iter().map(|&m| m.count_ones()).sum()
        }
    }

    impl Neg for BitVec {
        type Output = Self;
        fn neg(mut self) -> Self::Output {
            for x in &mut self.0 {
                *x = !*x;
            }
            self
        }
    }
    impl BitAndAssign<&'_ Self> for BitVec {
        fn bitand_assign(&mut self, rhs: &'_ Self) {
            assert_eq!(self.0.len(), rhs.0.len());
            for (x, y) in self.0.iter_mut().zip(&rhs.0) {
                x.bitand_assign(y);
            }
        }
    }
    impl BitOrAssign<&'_ Self> for BitVec {
        fn bitor_assign(&mut self, rhs: &'_ Self) {
            assert_eq!(self.0.len(), rhs.0.len());
            for (x, y) in self.0.iter_mut().zip(&rhs.0) {
                x.bitor_assign(y);
            }
        }
    }
    impl BitXorAssign<&'_ Self> for BitVec {
        fn bitxor_assign(&mut self, rhs: &'_ Self) {
            assert_eq!(self.0.len(), rhs.0.len());
            for (x, y) in self.0.iter_mut().zip(&rhs.0) {
                x.bitxor_assign(y);
            }
        }
    }
    impl ShlAssign<usize> for BitVec {
        fn shl_assign(&mut self, shift: usize) {
            if shift == 0 {
                return;
            }

            let n = self.bitlen();
            if shift >= n {
                self.0.fill(0);
                return;
            }

            let q = self.0.len();
            let q_shift = shift / BW;
            let r_shift = shift % BW;

            if r_shift == 0 {
                for n in (q_shift..q).rev() {
                    self.0[n] = self.0[n - q_shift];
                }
            } else {
                let sub_shift = (BW - r_shift) as u32;
                for n in ((q_shift + 1)..q).rev() {
                    self.0[n] =
                        (self.0[n - q_shift] << r_shift) | (self.0[n - q_shift - 1] >> sub_shift);
                }
                self.0[q_shift] = self.0[0] << r_shift;
            }
            self.0[..q_shift].fill(0);
        }
    }
    impl ShrAssign<usize> for BitVec {
        fn shr_assign(&mut self, shift: usize) {
            if shift == 0 {
                return;
            }

            let nbits = self.bitlen();
            if shift >= nbits {
                self.0.fill(0);
                return;
            }

            let q = self.0.len();
            let q_shift = shift / BW;
            let r_shift = shift % BW;

            if r_shift == 0 {
                for n in 0..q - q_shift {
                    self.0[n] = self.0[n + q_shift];
                }
            } else {
                let sub_shift = (BW - r_shift) as u32;
                for n in 0..q - q_shift - 1 {
                    self.0[n] =
                        (self.0[n + q_shift] >> r_shift) | (self.0[n + q_shift + 1] << sub_shift);
                }
                self.0[q - q_shift - 1] = self.0[q - 1] >> r_shift;
            }
            self.0[q.saturating_sub(q_shift)..q].fill(0);
        }
    }

    impl std::fmt::Debug for BitVec {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "BitVec(")?;
            for i in 0..self.0.len() {
                for j in 0..BW {
                    write!(f, "{}", ((self.0[i] >> j) & 1) as u8)?;
                }
            }
            write!(f, ")")?;
            Ok(())
        }
    }
}

fn parse_expr(s: &[u8]) -> Option<u8> {
    use parser::*;

    parser_fn!(atom: u8 = single.map_option(|x| match x {
        b'0' => Some(0),
        b'1' => Some(0b1111),
        b'x' => Some(0b1010),
        b'y' => Some(0b1100),
        _ => None
    }));
    parser_fn!(parens: u8 = between(lit_single(b'('), expr, lit_single(b')')));
    parser_fn!(term0: u8 = atom.or(parens));
    parser_fn!(term1: u8 = unary_prefix(lit_single(b'!').map(|_| |x| x ^ 0b1111), term0));
    parser_fn!(term2: u8 = binary_lassoc(lit_single(b'=').map(|_| |x, y| x ^ y ^ 0b1111), term1));
    parser_fn!(term3: u8 = binary_lassoc(lit_single(b'&').map(|_| |x, y| x & y), term2));
    parser_fn!(term4: u8 = binary_lassoc(lit_single(b'|').map(|_| |x, y| x | y), term3));
    parser_fn!(term5: u8 = binary_lassoc(lit_single(b'^').map(|_| |x, y| x ^ y), term4));
    parser_fn!(expr: u8 = term5);

    Some((expr, eof).run(s)?.0.0)
}

fn main() {
    let mut input = buffered_io::stdin();
    let mut output = buffered_io::stdout();

    let n: usize = input.value();
    let k: usize = input.value();
    const KB: usize = 10;

    let mut marker = vec![[[false; 4]; KB]; KB];
    let mut ops = vec![];
    for _ in 0..n {
        let a = input.value::<usize>() - 1;
        let b = input.value::<usize>() - 1;
        let s = parse_expr(input.token().as_bytes()).unwrap();
        let r = input.token() == "1";
        for i in 0..4 {
            if !marker[a][b][i] && s & (1 << i) != 0 {
                marker[a][b][i] = true;
                ops.push((a, b, i, r));
            }
        }
    }
    let base = input.token() == "1";

    let mut rel = BitVec::zero_bits(1 << 2 * KB);
    if base {
        for u in 0..1 << k {
            for v in 0..1 << k {
                rel.set((u << k) | v, true);
            }
        }
    }
    for (a, b, i, r) in ops.into_iter().rev() {
        let i0 = ((i >> 0) & 1) << a;
        let i1 = ((i >> 1) & 1) << b;
        for u in 0..1 << k {
            if u & (1 << a) != i0 {
                continue;
            }
            for v in 0..1 << k {
                if v & (1 << b) != i1 {
                    continue;
                }
                rel.set((u << k) | v, r);
            }
        }
    }

    let mut trel = BitVec::zero_bits(1 << 2 * KB);
    for u in 0..1 << k {
        for v in 0..1 << k {
            trel.set((u << k) | v, rel.get((v << k) | u));
        }
    }

    let mut loops = 0;
    let mut asym = 0;
    let mut no_trans = 0;
    for u in 0..1 << k {
        if rel.get((u << k) | u) {
            loops += 1;
        }
    }

    for (p, q) in rel.0.iter().zip(&trel.0) {
        asym += (p & q).count_ones();
    }

    for u in 0..1 << k {
        for w in 0..1 << k {
            if rel.get((u << k) | w) {
                continue;
            }
            if k <= 6 {
                for v in 0..1 << k {
                    if rel.get((u << k) | v) && trel.get((w << k) | v) {
                        no_trans += 1;
                    }
                }
            } else {
                for (p, q) in rel.0[u << k - 6..][..1 << k - 6]
                    .iter()
                    .zip(&trel.0[w << k - 6..][..1 << k - 6])
                {
                    no_trans += (p & q).count_ones();
                }
            }
        }
    }

    writeln!(output, "{} {} {}", loops, asym, no_trans).unwrap();
}
