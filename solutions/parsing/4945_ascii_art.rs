use std::{
    cmp::Reverse,
    collections::HashMap,
    fmt::Display,
    io::Write,
    iter::once,
    ops::{Index, IndexMut, RangeBounds, RangeInclusive, RangeToInclusive},
};

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

#[derive(Debug, Clone)]
struct Grid<T> {
    w: usize,
    data: Vec<T>,
}

impl<T> Grid<T> {
    pub fn with_shape(self, w: usize) -> Self {
        debug_assert_eq!(self.data.len() % w, 0);
        Grid { w, data: self.data }
    }
}

impl<T> FromIterator<T> for Grid<T> {
    fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
        Self {
            w: 1,
            data: iter.into_iter().collect(),
        }
    }
}

impl<T: Clone> Grid<T> {
    pub fn sized(fill: T, h: usize, w: usize) -> Self {
        Grid {
            w,
            data: vec![fill; w * h],
        }
    }
}

impl<T: Display> Display for Grid<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        for row in self.data.chunks(self.w) {
            for cell in row {
                cell.fmt(f)?;
                write!(f, " ")?;
            }
            writeln!(f)?;
        }
        writeln!(f)?;
        Ok(())
    }
}

impl<T> Index<(usize, usize)> for Grid<T> {
    type Output = T;
    fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
        debug_assert!(i < self.data.len() / self.w && j < self.w);
        &self.data[i * self.w + j]
    }
}

impl<T> IndexMut<(usize, usize)> for Grid<T> {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
        debug_assert!(i < self.data.len() / self.w && j < self.w);
        &mut self.data[i * self.w + j]
    }
}

struct PrettyColored<'a>(&'a Grid<u8>);

impl Display for PrettyColored<'_> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let colors = (once(37).chain(31..=36))
            .map(|i| (format!("\x1b[{}m", i), format!("\x1b[0m")))
            .collect::<Vec<_>>();

        let mut freq = HashMap::new();
        for c in self.0.data.iter() {
            *freq.entry(c).or_insert(0) += 1;
        }
        let mut freq = freq.into_iter().collect::<Vec<_>>();
        freq.sort_unstable_by_key(|(_, f)| Reverse(*f));

        let mut color_map = HashMap::new();
        let mut idx = 0;
        for (c, _) in freq {
            color_map.insert(c, &colors[idx % colors.len()]);
            idx += 1;
        }

        for row in self.0.data.chunks(self.0.w) {
            for cell in row {
                let (pre, suff) = color_map[&cell];
                write!(f, "{}{}{}", pre, *cell as char, suff)?;
            }
            writeln!(f)?;
        }
        Ok(())
    }
}

impl Grid<u8> {
    fn colored(&self) -> PrettyColored {
        PrettyColored(&self)
    }
}

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
    pub fn unary_prefix<'a, S: Stream<'a>, A, F, PF, P>(p_op: PF, p: P) -> impl Parser<'a, S, A>
    where
        P: Parser<'a, S, A>,
        PF: Parser<'a, S, F>,
        F: Fn(A) -> A + 'a,
    {
        (many(p_op), p)
            .map(move |(cs, e): (Vec<F>, A)| cs.into_iter().rev().fold(e, |acc, c| c(acc)))
    }

    pub fn unary_postfix<'a, S: Stream<'a>, A, F, PF, P>(p_op: PF, p: P) -> impl Parser<'a, S, A>
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

pub mod iter {
    pub fn product<I, J>(i: I, j: J) -> impl Iterator<Item = (I::Item, J::Item)>
    where
        I: IntoIterator,
        I::Item: Clone,
        J: IntoIterator,
        J::IntoIter: Clone,
    {
        let j = j.into_iter();
        i.into_iter()
            .flat_map(move |x| j.clone().map(move |y| (x.clone(), y)))
    }
}

mod ast {
    use super::iter::product;
    use super::Grid;
    use std::fmt::Display;
    use std::ops::Range;

    use crate::parser_fn;

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub enum Tag {
        Word(Vec<u8>),
        Quantifier(Box<Node>, u8),
        Term(Vec<Node>),
        Choice(Vec<Node>),
        Parens(Box<Node>),
    }

    type BBox = (usize, usize);

    fn hstack(lhs: BBox, rhs: BBox) -> BBox {
        (lhs.0.max(rhs.0), lhs.1 + rhs.1)
    }

    fn vstack(lhs: BBox, rhs: BBox) -> BBox {
        (lhs.0 + rhs.0, lhs.1.max(rhs.1))
    }

    fn pad(bbox: BBox, pad_down: usize, pad_right: usize) -> BBox {
        (bbox.0 + pad_down, bbox.1 + pad_right)
    }

    fn rebase(base: BBox, bbox: BBox) -> BBox {
        (base.0 + bbox.0, base.1 + bbox.1)
    }

    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Node {
        tag: Tag,
        bbox: BBox,
    }

    impl Node {
        fn new(tag: Tag) -> Self {
            Self { tag, bbox: (0, 0) }
        }
    }

    impl Display for Node {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            match &self.tag {
                Tag::Word(t) => write!(f, "w({})", std::str::from_utf8(&t).unwrap()),
                Tag::Quantifier(ref n, op) => {
                    write!(f, "[{}]{}", n, *op as char)
                }
                Tag::Term(ref ns) => {
                    for n in ns {
                        write!(f, "{}", n)?;
                    }
                    Ok(())
                }
                Tag::Choice(ref ns) => {
                    write!(f, "[{}", ns[0])?;
                    for n in &ns[1..] {
                        write!(f, "|{}]", n)?;
                    }
                    Ok(())
                }
                Tag::Parens(ref n) => write!(f, "({})", n),
            }
        }
    }

    impl Node {
        pub fn calc_bbox(&mut self) -> BBox {
            pub fn inner(node: &mut Node) -> BBox {
                let result = match &mut node.tag {
                    Tag::Word(t) => (3, t.len() + 4),
                    Tag::Quantifier(n, b'+') => pad(inner(n), 2, 6),
                    Tag::Quantifier(n, b'?') => pad(inner(n), 3, 6),
                    Tag::Quantifier(n, b'*') => pad(inner(n), 5, 6),
                    Tag::Quantifier(..) => panic!(),
                    Tag::Term(ns) => pad(
                        ns.into_iter().fold((0, 0), |acc, n| hstack(acc, inner(n))),
                        0,
                        2 * ns.len() - 2,
                    ),
                    Tag::Choice(ns) => pad(
                        ns.into_iter().fold((0, 0), |acc, n| vstack(acc, inner(n))),
                        ns.len() - 1,
                        6,
                    ),
                    Tag::Parens(n) => inner(n),
                };
                node.bbox = result;
                result
            }

            pad(inner(self), 0, 6)
        }

        pub fn render(&self) -> Grid<u8> {
            fn arrow(grid: &mut Grid<u8>, pos: (usize, usize)) {
                grid[(pos.0 + 1, pos.1)] = b'-';
                grid[(pos.0 + 1, pos.1 + 1)] = b'>';
            }

            fn arrow_to(grid: &mut Grid<u8>, i: usize, j: Range<usize>) {
                for k in j.start..j.end - 1 {
                    grid[(i, k)] = b'-';
                }
                grid[(i, j.end - 1)] = b'>';
            }

            fn larrow_to(grid: &mut Grid<u8>, i: usize, j: Range<usize>) {
                grid[(i, j.start)] = b'<';
                for k in j.start + 1..j.end {
                    grid[(i, k)] = b'-';
                }
            }

            fn inner(grid: &mut Grid<u8>, node: &Node, mut pos: (usize, usize)) {
                match &node.tag {
                    Tag::Word(t) => {
                        for u in product(0..3, [0, t.len() + 3]) {
                            grid[rebase(pos, u)] = b'+';
                        }
                        for u in product([0, 2], 1..t.len() + 3) {
                            grid[rebase(pos, u)] = b'-';
                        }
                        for j in 0..t.len() {
                            grid[rebase(pos, (1, j + 2))] = t[j];
                        }
                    }
                    Tag::Quantifier(child, b'+') => {
                        for u in product(1..node.bbox.0, [0, node.bbox.1 - 1]) {
                            grid[rebase(pos, u)] = b'|';
                        }
                        for u in product([1, node.bbox.0 - 1], [0, node.bbox.1 - 1]) {
                            grid[rebase(pos, u)] = b'+';
                        }
                        inner(grid, child, rebase(pos, (0, 3)));
                        arrow(grid, rebase(pos, (0, 1)));
                        arrow(grid, rebase(pos, (0, 3 + child.bbox.1)));
                        larrow_to(
                            grid,
                            pos.0 + node.bbox.0 - 1,
                            pos.1 + 1..pos.1 + node.bbox.1 - 1,
                        );
                    }
                    Tag::Quantifier(child, b'?') => {
                        for u in product(1..4, [0, node.bbox.1 - 1]) {
                            grid[rebase(pos, u)] = b'|';
                        }
                        for u in product([1, 4], [0, node.bbox.1 - 1]) {
                            grid[rebase(pos, u)] = b'+';
                        }
                        arrow_to(grid, pos.0 + 1, pos.1 + 1..pos.1 + node.bbox.1 - 1);
                        arrow(grid, rebase(pos, (3, 1)));
                        arrow(grid, rebase(pos, (3, 3 + child.bbox.1)));
                        inner(grid, child, rebase(pos, (3, 3)));
                    }
                    Tag::Quantifier(child, b'*') => {
                        // render both of above and below line
                        for u in product(1..node.bbox.0 - 1, [0, node.bbox.1 - 1]) {
                            grid[rebase(pos, u)] = b'|';
                        }
                        for u in product([1, 4, node.bbox.0 - 1], [0, node.bbox.1 - 1]) {
                            grid[rebase(pos, u)] = b'+';
                        }
                        arrow_to(grid, pos.0 + 1, pos.1 + 1..pos.1 + node.bbox.1 - 1);
                        larrow_to(
                            grid,
                            pos.0 + node.bbox.0 - 1,
                            pos.1 + 1..pos.1 + node.bbox.1 - 1,
                        );
                        arrow(grid, rebase(pos, (3, 1)));
                        arrow(grid, rebase(pos, (3, 3 + child.bbox.1)));
                        inner(grid, child, rebase(pos, (3, 3)));
                    }
                    Tag::Quantifier(..) => panic!(),
                    Tag::Term(children) => {
                        for (j, n) in children.iter().enumerate() {
                            inner(grid, n, pos);
                            pos.1 += n.bbox.1;
                            if j < children.len() - 1 {
                                arrow(grid, pos);
                            }
                            pos.1 += 2;
                        }
                    }
                    Tag::Choice(children) => {
                        let i_last = children[..children.len() - 1]
                            .iter()
                            .map(|node| node.bbox.0 + 1)
                            .sum::<usize>()
                            + 2;
                        for u in product(1..i_last, [0, node.bbox.1 - 1]) {
                            grid[rebase(pos, u)] = b'|';
                        }
                        for (i, n) in children.iter().enumerate() {
                            grid[rebase(pos, (1, 0))] = b'+';
                            grid[rebase(pos, (1, node.bbox.1 - 1))] = b'+';
                            arrow(grid, rebase(pos, (0, 1)));
                            arrow_to(
                                grid,
                                pos.0 + 1,
                                pos.1 + n.bbox.1 + 3..pos.1 + node.bbox.1 - 1,
                            );
                            inner(grid, n, rebase(pos, (0, 3)));
                            pos.0 += n.bbox.0;
                            if i < children.len() - 1 {}
                            pos.0 += 1;
                        }
                    }
                    Tag::Parens(child) => inner(grid, child, pos),
                }
            }

            let mut grid = Grid::sized(b' ', self.bbox.0, self.bbox.1 + 6);
            inner(&mut grid, self, (0, 3));
            for j in 0..3 {
                grid[(1, j)] = b"S->"[j];
            }
            for j in 0..3 {
                grid[(1, self.bbox.1 + 3 + j)] = b"->F"[j];
            }

            grid
        }
    }

    pub fn parse(s: &[u8]) -> Node {
        use super::parser::u8::*;
        use super::parser::*;

        parser_fn!(let letter: u8 = satisfy(|b| matches!(b, b'A'..=b'Z')));
        parser_fn!(let parens: Node = (literal_byte(b'('), expr, literal_byte(b')')).map(|(_, n, _)| Tag::Parens(Box::new(n))).map(Node::new));
        parser_fn!(let term1: Node = parens.or_else(letter.map(|x| Tag::Word(vec![x])).map(Node::new)));

        parser_fn!(let op_codes: u8 = satisfy(|b| matches!(b, b'+' | b'*' | b'?')));
        parser_fn!(let op_map: impl Fn(Node) -> Node = op_codes.map(|op| move |n| Node::new(Tag::Quantifier(Box::new(n), op))));
        parser_fn!(let term2: Node = unary_postfix(op_map, term1));

        fn unfold_parens(mut node: Node) -> Node {
            while let Tag::Parens(inner) = node.tag {
                node = *inner;
            }
            node
        }
        parser_fn!(let term3: Node = many1(term2).map(|xs| {
            let pred = |x: &Node| matches!(x.tag, Tag::Word(_));
            let unwrap = |x: Node| match x.tag {
                Tag::Word(w) => w[0],
                _ => unreachable!(),
            };
            let mut i = 0;
            let mut xs_new = vec![];
            while i < xs.len() {
                if pred(&xs[i]) {
                    let mut j = i + 1;
                    while j < xs.len() && pred(&xs[j]) {
                        j += 1;
                    }
                    let word: Vec<u8> = (i..j).map(|i| unwrap(xs[i].clone())).collect();
                    xs_new.push(Node::new(Tag::Word(word)));
                    i = j;
                } else {
                    // dbg!(xs[i].clone(), unfold_parens(xs[i].clone()));
                    // xs_new.push(unfold_parens(xs[i].clone()));
                    xs_new.push(xs[i].clone());
                    i += 1;
                }
            }
            let xs = xs_new;

            if xs.len() == 1{
                xs.into_iter().next().unwrap()
            } else {
                Node::new(Tag::Term(xs))
            }
        }));
        parser_fn!(let expr: Node = many_sep1(term3, literal_byte(b'|')).map(|xs| {
            if xs.len() == 1{
                xs.into_iter().next().unwrap()
            } else {
                Node::new(Tag::Choice(xs))
            }
        }));

        expr.run(s).unwrap().0
    }
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout_buf();

    let pattern = input.token();
    let mut expr = ast::parse(pattern.as_bytes());
    expr.calc_bbox();

    let grid = expr.render();

    let w = grid.w;
    let h = grid.data.len() / w;
    writeln!(output, "{} {}", h, w).unwrap();
    for i in 0..h {
        output.write_all(&grid.data[i * w..][..w]).unwrap();
        writeln!(output).unwrap();
    }
}
