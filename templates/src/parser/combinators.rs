// A combinator-based parser, based on the Haskell Parsec library
// https://hackage.haskell.org/package/parsec
#[allow(dead_code)]
#[macro_use]
pub mod parser {
    pub type ParseResult<S, A> = Option<(A, S)>;
    pub trait Stream: Sized + Clone + std::fmt::Debug {
        fn zero() -> Self;
        fn next(self) -> ParseResult<Self, u8>;
        // fn peek(s: Self) -> ParseResult<Self, u8>;
    }

    impl Stream for &[u8] {
        fn zero() -> Self {
            &[]
        }

        fn next(self) -> ParseResult<Self, u8> {
            self.split_first().map(|(b, s)| (*b, s))
        }
    }

    pub trait Parser<S: Stream, A>: Sized {
        // fn run<'b>(&self, s: &'b str) -> ParseResult<'o, A>;
        fn run(&self, s: S) -> ParseResult<S, A>;

        fn map<B, F>(self, f: F) -> Map<Self, F, A>
        where
            F: Fn(A) -> B,
        {
            Map(self, f, Default::default())
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
    impl<S, A, B, PA, F> Parser<S, B> for Map<PA, F, A>
    where
        S: Stream,
        PA: Parser<S, A>,
        F: Fn(A) -> B,
    {
        fn run(&self, s: S) -> ParseResult<S, B> {
            self.0.run(s).map(|(a, s)| (self.1(a), s))
        }
    }

    pub struct AndThen<PA, F, A>(PA, F, std::marker::PhantomData<A>);
    impl<S, A, B, PA, PB, F> Parser<S, B> for AndThen<PA, F, A>
    where
        S: Stream,
        PA: Parser<S, A>,
        PB: Parser<S, B>,
        F: Fn(A) -> PB,
    {
        fn run(&self, s: S) -> ParseResult<S, B> {
            self.0.run(s).and_then(|(a, s)| self.1(a).run(s))
        }
    }

    pub struct MapOption<PA, F, A>(PA, F, std::marker::PhantomData<A>);
    impl<S, A, B, PA, F> Parser<S, B> for MapOption<PA, F, A>
    where
        S: Stream,
        PA: Parser<S, A>,
        F: Fn(A) -> Option<B>,
    {
        fn run(&self, s: S) -> ParseResult<S, B> {
            self.0.run(s).and_then(|(a, s)| self.1(a).map(|b| (b, s)))
        }
    }

    pub struct Filter<PA, F>(PA, F);
    impl<S, A, PA, F> Parser<S, A> for Filter<PA, F>
    where
        S: Stream,
        PA: Parser<S, A>,
        F: Fn(&A) -> bool,
    {
        fn run(&self, s: S) -> ParseResult<S, A> {
            self.0.run(s).filter(|(a, _)| (self.1)(a))
        }
    }

    pub struct OrElse<P, Q>(P, Q);

    impl<S, A, P, Q> Parser<S, A> for OrElse<P, Q>
    where
        S: Stream,
        P: Parser<S, A>,
        Q: Parser<S, A>,
    {
        fn run(&self, s: S) -> ParseResult<S, A> {
            self.0.run(s.clone()).or_else(|| self.1.run(s))
        }
    }

    impl<'a, S, A, F> Parser<S, A> for F
    where
        S: Stream,
        F: Fn(S) -> ParseResult<S, A> + 'a,
    {
        fn run(&self, s: S) -> ParseResult<S, A> {
            self(s)
        }
    }

    macro_rules! gen_tuple_parser {
        ($($A:tt $PA:tt $a:tt $i:tt),+) => {
            impl<S:Stream,$($A,)+$($PA,)+> Parser<S,($($A,)+)> for ($($PA,)+)
            where $($PA: Parser<S,$A>,)+
            {
                fn run(&self, s: S) -> ParseResult<S, ($($A,)+)> {
                    $(let ($a, s):($A,S) = self.$i.run(s)?;)+
                    Some((($($a,)+),s))
                }
            }
        }
    }

    // macro to quick define a 0-ary parser
    #[macro_export]
    macro_rules! parser_fn {
        ($name:ident: $T:ty = $body:expr) => {
            fn $name<S: Stream>(s: S) -> ParseResult<S, $T> {
                $body.run(s)
            }
        };
    }

    gen_tuple_parser!(A0 PA0 a0 0, A1 PA1 a1 1);
    gen_tuple_parser!(A0 PA0 a0 0, A1 PA1 a1 1, A2 PA2 a2 2);
    gen_tuple_parser!(A0 PA0 a0 0, A1 PA1 a1 1, A2 PA2 a2 2, A3 PA3 a3 3);
    gen_tuple_parser!(A0 PA0 a0 0, A1 PA1 a1 1, A2 PA2 a2 2, A3 PA3 a3 3, A4 PA4 a4 4);
    gen_tuple_parser!(A0 PA0 a0 0, A1 PA1 a1 1, A2 PA2 a2 2, A3 PA3 a3 3, A4 PA4 a4 4, A5 PA5 a5 5);

    fn fail<S, A>(_s: S) -> ParseResult<S, A> {
        None
    }

    // macro_rules! choose {
    //     ($p:expr, $($q:expr),+) => {
    //         $p.or_else(choose!($($q),+))
    //     };
    //     ($p:expr) => {
    //         $p
    //     };
    // }

    pub fn satisfy<S: Stream>(f: impl Fn(u8) -> bool) -> impl Parser<S, u8> {
        move |s: S| s.next().filter(|(b, _)| f(*b))
    }

    pub fn eof<S: Stream>(s: S) -> ParseResult<S, ()> {
        match s.next() {
            Some(_) => None,
            None => Some(((), S::zero())),
        }
    }

    pub fn single<S: Stream>(s: S) -> ParseResult<S, u8> {
        s.next()
    }

    pub fn literal_byte<S: Stream>(c: u8) -> impl Parser<S, ()> {
        satisfy(move |b| b == c).map(|_| ())
    }

    pub fn literal<S: Stream>(keyword: &'static [u8]) -> impl Parser<S, ()> {
        move |mut s: S| {
            for &c in keyword {
                let (b, stream_new) = s.next()?;
                if b != c {
                    return None;
                }
                s = stream_new;
            }
            Some(((), s))
        }
    }

    pub fn uint<S: Stream>(s: S) -> ParseResult<S, i32> {
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

    // pub fn int<S: Stream>(s: S) -> ParseResult<S, i32> {
    //     let mut s = s;
    //     let mut sign = 1;
    //     if let Some((_, s_new)) = satisfy(|b| b == b'-').run(s.clone()) {
    //         sign = -1;
    //         s = s_new;
    //     }

    //     // at least one digit is required
    //     let (b, s_new) = satisfy(|b| b.is_ascii_digit()).run(s.clone())?;
    //     let mut result = (b - b'0') as i32;
    //     s = s_new;

    //     while let Some((b, s_new)) = satisfy(|b| b.is_ascii_digit()).run(s.clone()) {
    //         result = result * 10 + (b - b'0') as i32;
    //         s = s_new;
    //     }

    //     Some((result * sign, s))
    // }

    // map A -> ()
    pub fn ignore_value<S: Stream, A>(p: impl Parser<S, A>) -> impl Parser<S, ()> {
        p.map(|_| ())
    }

    pub fn between<S: Stream, A, O, E>(
        p_open: impl Parser<S, O>,
        p: impl Parser<S, A>,
        p_close: impl Parser<S, E>,
    ) -> impl Parser<S, A> {
        (p_open, p, p_close).map(|(_, a, _)| a)
    }

    pub fn optional<S: Stream, A>(p: impl Parser<S, A>) -> impl Parser<S, Option<A>> {
        move |s: S| match p.run(s.clone()) {
            Some((a, s_new)) => Some((Some(a), s_new)),
            None => Some((None, s)),
        }
    }

    pub fn many<S: Stream, A>(p: impl Parser<S, A>) -> impl Parser<S, Vec<A>> {
        move |mut s: S| {
            let mut result = Vec::new();
            while let Some((e, s_new)) = p.run(s.clone()) {
                result.push(e);
                s = s_new;
            }
            Some((result, s))
        }
    }

    pub fn many1<S: Stream, A>(p: impl Parser<S, A>) -> impl Parser<S, Vec<A>> {
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

    pub fn many_sep1<S: Stream, A>(
        p: impl Parser<S, A>,
        p_sep: impl Parser<S, ()>,
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
    pub fn unary_infix<'a, S: Stream, A, F, PF, P>(p_op: PF, p: P) -> impl Parser<S, A>
    where
        P: Parser<S, A>,
        PF: Parser<S, F>,
        F: Fn(A) -> A + 'a,
    {
        (many(p_op), p)
            .map(move |(cs, e): (Vec<F>, A)| cs.into_iter().rev().fold(e, |acc, c| c(acc)))
    }

    pub fn unary_suffix<'a, S: Stream, A, F, PF, P>(p_op: PF, p: P) -> impl Parser<S, A>
    where
        P: Parser<S, A>,
        PF: Parser<S, F>,
        F: Fn(A) -> A + 'a,
    {
        (p, many(p_op)).map(move |(e, cs): (A, Vec<F>)| cs.into_iter().fold(e, |acc, c| c(acc)))
    }

    pub fn binary_lassoc<'a, S: Stream, A, F, PF, P>(p_op: PF, p: P) -> impl Parser<S, A>
    where
        P: Parser<S, A>,
        PF: Parser<S, F>,
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

    pub fn binary_rassoc<'a, S: Stream, A, F, PF, P>(p_op: PF, p: P) -> impl Parser<S, A>
    where
        P: Parser<S, A>,
        PF: Parser<S, F>,
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

    pub fn debug_print<'a, S, A>(p: impl Parser<S, A>) -> impl Parser<S, A>
    where
        S: Stream,
        A: std::fmt::Display,
    {
        move |s| {
            let (a, s) = p.run(s)?;
            Some((a, s))
        }
    }
}
