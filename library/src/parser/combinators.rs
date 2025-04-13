#[macro_use]
pub mod parser {
    use std::mem;

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

    impl<'a> Stream for &'a [u8] {
        type Item = u8;
        type ItemGroup<'b>
            = &'b [u8]
        where
            'a: 'b;

        fn empty() -> Self {
            &[]
        }

        fn next(self) -> ParseResult<Self, u8> {
            self.split_first().map(|(b, s)| (*b, s))
        }

        fn take_while<'b>(self, pred: impl Fn(&u8) -> bool) -> ParseResult<Self, &'b [u8]>
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
            PB: Parser<S, B>,
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
            P: Parser<S, A>,
        {
            OrElse(self, p)
        }
    }

    // Can be replaced with impl trait on method return types (rust >= 1.75)
    pub struct Map<PA, F, A>(PA, F, std::marker::PhantomData<A>);
    impl<'a, S, A, B, PA, F> Parser<S, B> for Map<PA, F, A>
    where
        S: Stream,
        PA: Parser<S, A>,
        F: Fn(A) -> B,
    {
        fn run(&mut self, s: S) -> ParseResult<S, B> {
            self.0.run(s).map(|(a, s)| (self.1(a), s))
        }
    }

    pub struct Inspect<PA, F, A>(PA, F, std::marker::PhantomData<A>);
    impl<'a, S, A, PA, F> Parser<S, A> for Inspect<PA, F, A>
    where
        S: Stream,
        PA: Parser<S, A>,
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
    impl<'a, S, A, B, PA, PB, F> Parser<S, B> for AndThen<PA, F, A>
    where
        S: Stream,
        PA: Parser<S, A>,
        PB: Parser<S, B>,
        F: Fn(A) -> PB,
    {
        fn run(&mut self, s: S) -> ParseResult<S, B> {
            self.0.run(s).and_then(|(a, s)| self.1(a).run(s))
        }
    }

    pub struct MapOption<PA, F, A>(PA, F, std::marker::PhantomData<A>);
    impl<'a, S, A, B, PA, F> Parser<S, B> for MapOption<PA, F, A>
    where
        S: Stream,
        PA: Parser<S, A>,
        F: Fn(A) -> Option<B>,
    {
        fn run(&mut self, s: S) -> ParseResult<S, B> {
            self.0.run(s).and_then(|(a, s)| self.1(a).map(|b| (b, s)))
        }
    }

    pub struct Filter<PA, F>(PA, F);
    impl<'a, S, A, PA, F> Parser<S, A> for Filter<PA, F>
    where
        S: Stream,
        PA: Parser<S, A>,
        F: Fn(&A) -> bool,
    {
        fn run(&mut self, s: S) -> ParseResult<S, A> {
            self.0.run(s).filter(|(a, _)| (self.1)(a))
        }
    }

    pub struct OrElse<P, Q>(P, Q);

    impl<'a, S, A, P, Q> Parser<S, A> for OrElse<P, Q>
    where
        S: Stream,
        P: Parser<S, A>,
        Q: Parser<S, A>,
    {
        fn run(&mut self, s: S) -> ParseResult<S, A> {
            self.0.run(s.clone()).or_else(|| self.1.run(s))
        }
    }

    impl<'a, S, A, F> Parser<S, A> for F
    where
        S: Stream,
        F: FnMut(S) -> ParseResult<S, A>,
    {
        fn run(&mut self, s: S) -> ParseResult<S, A> {
            self(s)
        }
    }

    macro_rules! gen_tuple_parser {
        (@tuple $($A:tt $PA:tt $a:tt),+) => {
            impl<'a,S:Stream,$($A,)+$($PA,)+> Parser<S,($($A,)+)> for ($($PA,)+)
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

    pub fn eof<'a, S: Stream>(s: S) -> ParseResult<S, ()> {
        match s.next() {
            Some(_) => None,
            None => Some(((), S::empty())),
        }
    }

    pub fn single<'a, S: Stream>(s: S) -> ParseResult<S, S::Item> {
        s.next()
    }

    pub fn between<'a, S: Stream, A, O, E>(
        p_open: impl Parser<S, O>,
        p: impl Parser<S, A>,
        p_close: impl Parser<S, E>,
    ) -> impl Parser<S, A> {
        (p_open, p, p_close).map(|(_, a, _)| a)
    }

    pub fn optional<'a, S: Stream, A>(mut p: impl Parser<S, A>) -> impl Parser<S, Option<A>> {
        move |s: S| match p.run(s.clone()) {
            Some((a, s_new)) => Some((Some(a), s_new)),
            None => Some((None, s)),
        }
    }

    pub fn fold<'a, S: Stream, A, F>(
        mut init: impl Parser<S, A>,
        mut p: impl Parser<S, F>,
    ) -> impl Parser<S, A>
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

    pub fn many<'a, S: Stream, A>(mut p: impl Parser<S, A>) -> impl Parser<S, Vec<A>> {
        move |mut s: S| {
            let mut result = Vec::new();
            while let Some((e, s_new)) = p.run(s.clone()) {
                result.push(e);
                s = s_new;
            }
            Some((result, s))
        }
    }

    pub fn many1<'a, S: Stream, A>(mut p: impl Parser<S, A>) -> impl Parser<S, Vec<A>> {
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

    pub fn many_sep1<'a, S: Stream, A>(
        mut p: impl Parser<S, A>,
        mut p_sep: impl Parser<S, ()>,
    ) -> impl Parser<S, Vec<A>> {
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
    pub fn unary_prefix<'a, S: Stream, A, F, PF, P>(p_op: PF, p: P) -> impl Parser<S, A>
    where
        P: Parser<S, A>,
        PF: Parser<S, F>,
        F: FnMut(A) -> A + 'a,
    {
        (many(p_op), p)
            .map(move |(cs, e): (Vec<F>, A)| cs.into_iter().rev().fold(e, |acc, mut c| c(acc)))
    }

    pub fn unary_postfix<'a, S: Stream, A, F, PF, P>(p_op: PF, p: P) -> impl Parser<S, A>
    where
        P: Parser<S, A>,
        PF: Parser<S, F>,
        F: FnMut(A) -> A + 'a,
    {
        (p, many(p_op)).map(move |(e, cs): (A, Vec<F>)| cs.into_iter().fold(e, |acc, mut c| c(acc)))
    }

    pub fn binary_lassoc<'a, S: Stream, A, F, PF, P>(mut p_op: PF, mut p: P) -> impl Parser<S, A>
    where
        P: Parser<S, A>,
        PF: Parser<S, F>,
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

    pub fn binary_rassoc<'a, S: Stream, A, F, PF, P>(mut p_op: PF, mut p: P) -> impl Parser<S, A>
    where
        P: Parser<S, A>,
        PF: Parser<S, F>,
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

    pub fn take_while<'a, S: Stream + 'a>(
        pred: impl Fn(&S::Item) -> bool,
    ) -> impl Parser<S, S::ItemGroup<'a>> {
        move |s: S| s.take_while(|b| pred(b))
    }

    pub fn skip_while<'a, S: Stream + 'a>(pred: impl Fn(&S::Item) -> bool) -> impl Parser<S, ()> {
        move |s: S| s.skip_while(|b| pred(b))
    }

    pub fn satisfy<S: Stream>(f: impl Fn(&S::Item) -> bool) -> impl Parser<S, S::Item> {
        move |s: S| s.next().filter(|(b, _)| f(b))
    }

    pub fn lit_single<'a, S: Stream>(c: S::Item) -> impl Parser<S, ()>
    where
        S::Item: PartialEq,
    {
        satisfy(move |b| b == &c).map(|_| ())
    }

    pub fn lit<'a, S: Stream>(keyword: &'static [S::Item]) -> impl Parser<S, ()>
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
            (let $name:ident: $T:ty = $body:expr) => {
                fn $name<'a, S: U8Stream>(s: S) -> ParseResult<S, $T> {
                    $body.run(s)
                }
            };
        }

        pub fn spaces<'a, S: Stream<Item = u8>>(s: S) -> ParseResult<S, ()> {
            skip_while(|b: &u8| b.is_ascii_whitespace()).run(s)
        }

        pub fn rspaces<'a, S: Stream<Item = u8>, A, P: Parser<S, A>>(p: P) -> impl Parser<S, A> {
            between(spaces, p, spaces)
        }

        pub fn uint<'a, S: Stream<Item = u8>>(s: S) -> ParseResult<S, i32> {
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

        pub fn int<'a, S: Stream<Item = u8>>(s: S) -> ParseResult<S, i32> {
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
