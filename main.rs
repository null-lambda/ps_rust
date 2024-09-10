#[allow(dead_code)]
mod io {
    use std::fmt::Debug;
    use std::str::*;

    pub trait InputStream {
        fn token(&mut self) -> &[u8];
        fn line(&mut self) -> &[u8];

        fn skip_line(&mut self) {
            self.line();
        }
    }

    #[inline]
    fn is_whitespace(c: u8) -> bool {
        c <= b' '
    }

    fn trim_newline(s: &[u8]) -> &[u8] {
        let mut s = s;
        while s
            .last()
            .map(|&c| match c {
                b'\n' | b'\r' | 0 => true,
                _ => false,
            })
            .unwrap_or_else(|| false)
        {
            s = &s[..s.len() - 1];
        }
        s
    }

    impl InputStream for &[u8] {
        fn token(&mut self) -> &[u8] {
            let idx = self
                .iter()
                .position(|&c| !is_whitespace(c))
                .expect("no available tokens left");
            *self = &self[idx..];
            let idx = self
                .iter()
                .position(|&c| is_whitespace(c))
                .unwrap_or_else(|| self.len());
            let (token, buf_new) = self.split_at(idx);
            *self = buf_new;
            token
        }

        fn line(&mut self) -> &[u8] {
            let idx = self
                .iter()
                .position(|&c| c == b'\n')
                .map(|idx| idx + 1)
                .unwrap_or_else(|| self.len());
            let (line, buf_new) = self.split_at(idx);
            *self = buf_new;
            trim_newline(line)
        }
    }

    pub trait ReadValue<T> {
        fn value(&mut self) -> T;
    }

    impl<T: FromStr, I: InputStream> ReadValue<T> for I
    where
        T::Err: Debug,
    {
        #[inline]
        fn value(&mut self) -> T {
            let token = self.token();
            let token = unsafe { from_utf8_unchecked(token) };
            token.parse::<T>().unwrap()
        }
    }
}

use std::io::{BufReader, Read, Write};

fn stdin() -> Vec<u8> {
    let stdin = std::io::stdin();
    let mut reader = BufReader::new(stdin.lock());
    let mut input_buf: Vec<u8> = vec![];
    reader.read_to_end(&mut input_buf).unwrap();
    input_buf
}

// Abstract syntax tree of logical expresssions 
// atoms: a-z
// operators: ~, &, ^, |, () with optional whitespace
// https://www.acmicpc.net/problem/2769
#[allow(dead_code)]
mod ast {
    use std::collections::HashSet;

    #[derive(Clone, Debug, PartialEq)]
    pub enum Expr {
        Variable(u8),
        Not(Box<Expr>),
        And(Box<Expr>, Box<Expr>),
        Xor(Box<Expr>, Box<Expr>),
        Or(Box<Expr>, Box<Expr>),
        Iff(Box<Expr>, Box<Expr>),
    }

    const MAX_VARIABLES: usize = 26;

    impl std::fmt::Display for Expr {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            use Expr::*;
            match self {
                Variable(v) => write!(f, "{}", v),
                Not(e) => write!(f, "~{}", e),
                And(e1, e2) => write!(f, "({}&{})", e1, e2),
                Xor(e1, e2) => write!(f, "({}^{})", e1, e2),
                Or(e1, e2) => write!(f, "({}|{})", e1, e2),
                Iff(e1, e2) => write!(f, "({}<->{})", e1, e2),
            }
        }
    }

    impl Expr {
        pub fn parse(s: &str) -> Option<Expr> {
            // remove all spaces in s
            let s: String = s.chars().filter(|&c| !c.is_whitespace()).collect();
            let s: &str = s.as_str();
            let (ast, _) = parser_impl::main(s)?;
            Some(ast)
        }

        pub fn get_variables(&self) -> HashSet<u8> {
            fn inner(e: &Expr, vars: &mut HashSet<u8>) {
                use Expr::*;
                match e {
                    Variable(v) => {
                        vars.insert(*v);
                    }
                    Not(e) => inner(e, vars),
                    And(e1, e2) | Xor(e1, e2) | Or(e1, e2) | Iff(e1, e2) => {
                        inner(e1, vars);
                        inner(e2, vars);
                    }
                }
            }

            let mut vars = HashSet::new();
            inner(self, &mut vars);
            vars
        }

        pub fn eval(&self, values: &[bool]) -> bool {
            use Expr::*;
            match self {
                Variable(v) => values[*v as usize],
                Not(e) => !e.eval(values),
                And(e1, e2) => e1.eval(values) && e2.eval(values),
                Xor(e1, e2) => e1.eval(values) ^ e2.eval(values),
                Or(e1, e2) => e1.eval(values) || e2.eval(values),
                Iff(e1, e2) => e1.eval(values) == e2.eval(values),
            }
        } 

        pub fn check_tauto(&self) -> bool {
            let variables: Vec<u8> = self.get_variables().into_iter().collect();

            // assign all possible configurations of variables to eval
            fn inner(e: &Expr, variables: &Vec<u8>, values: &mut [bool], current: usize) -> bool {
                if current == variables.len() {
                    return e.eval(values);
                } 

                let mut result = true;
                values[variables[current] as usize] = true;
                result &= inner(e, variables, values, current + 1);
                values[variables[current] as usize] = false;
                result &= inner(e, variables, values, current + 1);
                result 
            }


            inner(self, &variables, &mut [false; MAX_VARIABLES], 0)
        }
    }

    // A combinator-based parser, based on the Haskell Parsec library
    // https://hackage.haskell.org/package/parsec
    mod parser_impl {
        use super::Expr;

        type ParseResult<S, A> = Option<(A, S)>;
        pub trait Stream : Sized + Clone + std::fmt::Debug { 
            fn zero() -> Self;
            fn next(self) -> ParseResult<Self, u8>;
            // fn peek(s: Self) -> ParseResult<Self, u8>;
        }
        impl Stream for &str {
            fn zero() -> Self {
                ""
            }

            fn next(self) -> ParseResult<Self, u8> {
                self.bytes().next().map(|b| (b, self[1..].into()))
            }

            // fn peek(s: Self) -> ParseResult<Self, u8> {
            //     s.bytes().next().map(|b| (b, s))
            // }
        }

        pub trait Parser<S: Stream, A> : Sized {
            // fn run<'b>(&self, s: &'b str) -> ParseResult<'b, A>;
            fn run(&self, s: S) -> ParseResult<S, A>;

            fn map<B, F>(self, f: F) -> Map<Self, F, A> 
            where 
                F: Fn(A) -> B
            {
                Map(self, f, Default::default())
            }

            fn and_then<B, PB, F>(self, p: F) -> AndThen<Self, F, A>
            where 
                PB: Parser<S, B>,
                F: Fn(A) -> PB
            {
                AndThen(self, p, Default::default())
            }

            fn map_option<B, F>(self, f: F) -> MapOption<Self, F, A>
            where 
                F: Fn(A) -> Option<B>
            {
                MapOption(self, f, Default::default())
            }

            fn or_else<P>(self, p: P) -> OrElse<Self, P> 
            where 
                P: Parser<S, A>
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
            F: Fn(A) -> B
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
            F: Fn(A) -> PB
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
            F: Fn(A) -> Option<B>
        {
            fn run(&self, s: S) -> ParseResult<S, B> {
                self.0.run(s).and_then(|(a, s)| self.1(a).map(|b| (b, s)))
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

        gen_tuple_parser!(A0 PA0 a0 0, A1 PA1 a1 1);
        gen_tuple_parser!(A0 PA0 a0 0, A1 PA1 a1 1, A2 PA2 a2 2);
        gen_tuple_parser!(A0 PA0 a0 0, A1 PA1 a1 1, A2 PA2 a2 2, A3 PA3 a3 3);

        // // the function 'pure' often does not work wel with lifetime semantics>
        // fn pure<A>(a: A) -> impl FnOnce(&str) -> ParseResult<A> {
        //     move |s| Some((a, s))
        // }

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

        fn satisfy<S: Stream>(f: impl Fn(u8) -> bool) -> impl Parser<S, u8> {
            move |s: S| s.next().filter(|(b, _)| f(*b))
        }

        fn eof<S: Stream>(s: S) -> ParseResult<S, ()> {
            match s.next() {
                Some(_) => None,
                None => Some(((), S::zero()))
            }
        }

        fn byte<S: Stream>(c: u8) -> impl Parser<S, u8> {
            satisfy(move |b| b == c) 
        }

        // map A -> ()
        fn symbol<S: Stream, A>(p: impl Parser<S, A>) -> impl Parser<S, ()> {
            p.map(|_| ())
        }

        fn parens<S: Stream>(p: impl Parser<S, Expr>) -> impl Parser<S, Expr> {
            // (char('('), p, char(')')).map(|(_, e, _)| e)
            move |s: S| {
                let (_, s) = byte(b'(').run(s)?;
                let (e, s) = p.run(s)?;
                let (_, s) = byte(b')').run(s)?;
                Some((e, s))
            }
        }

        fn variable<S: Stream>(s: S) -> ParseResult<S, Expr> {
            satisfy(|b| (b'a'..=b'z').contains(&b)).map(|b| Expr::Variable(b - b'a')).run(s)
        }

        fn many<S: Stream, A>(p: impl Parser<S, A>) -> impl Parser<S, Vec<A>> {
            move |mut s: S| {
                let mut result = Vec::new();
                while let Some((e, s_new)) = p.run(s.clone()) {
                    result.push(e);
                    s = s_new;
                }
                Some((result, s))
            }
        }
        
        fn many1<S: Stream, A>(p: impl Parser<S, A>) -> impl Parser<S, Vec<A>> {
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

        fn many_sep1<S: Stream, A>(p: impl Parser<S, A>, p_sep: impl Parser<S, ()>) -> impl Parser<S, Vec<A>> {
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
                        },
                        None => break
                    }
                }

                Some((result, s))
            }
        }

        // chained unary operators 
        fn unary_infix<'a, S: Stream, C>(op: u8, constructor: C, p: impl Parser<S, Expr>) -> impl Parser<S, Expr>
        where 
            C: Fn(Box<Expr>) -> Expr + 'a,
        {
            (many(symbol(byte(op))), p).map(move |(xs, e)| {
                xs.iter().fold(e, |acc, _| constructor(Box::new(acc)))
            })
        }

        fn binary_lassoc<'a, S: Stream, C>(op: u8, constructor: C, p: impl Parser<S, Expr>) -> impl Parser<S, Expr>
        where 
            C: Fn(Box<Expr>, Box<Expr>) -> Expr + 'a,
        {
            let p_sep = symbol(byte(op));
            many_sep1(p, p_sep).map_option(move |es| es.into_iter().reduce(|e1, e2| constructor(Box::new(e1), Box::new(e2)))
            )
        }

        pub fn main<S: Stream>(s: S) -> ParseResult<S, Expr> {
            fn term<S: Stream>(s: S) -> ParseResult<S, Expr> {
                parens(logical_expr).or_else(variable).run(s)
            }

            fn not<S: Stream>(s: S) -> ParseResult<S, Expr> {
                unary_infix(b'~', Expr::Not, term).run(s)
            }

            fn and<S: Stream>(s: S) -> ParseResult<S, Expr> {
                binary_lassoc(b'&', Expr::And, not).run(s)
            }

            fn xor<S: Stream>(s: S) -> ParseResult<S, Expr> {
                binary_lassoc(b'^', Expr::Xor, and).run(s)

            }

            fn or<S: Stream>(s: S) -> ParseResult<S, Expr> {
                binary_lassoc(b'|', Expr::Or, xor).run(s)
            }

            fn logical_expr<S: Stream>(s: S) -> ParseResult<S, Expr> {
                or.run(s)
            // parens(variable).or_else(variable).or_else(or).run(s)
            }

            fn doubled_expr<S: Stream>(s: S) -> ParseResult<S, Expr> {
                (logical_expr, logical_expr).map(|(e1, e2)| Expr::Iff(Box::new(e1), Box::new(e2))).run(s)
            }

            (doubled_expr, eof).map(|(e, _)| e).run(s)
        }

        fn print_parser<'a, S, A>(p: impl Parser<S, A>) -> impl Parser<S, A>
        where
            S: Stream,
            A: std::fmt::Display 
        {
            move |s| {
                let (a, s) = p.run(s)?;
                Some((a, s))
            }
        }
    }
}


fn main() {
    use io::*;
    use std::str;

    let input_buf = stdin();
    let mut input: &[u8] = &input_buf;

    let mut output_buf =  Vec::<u8>::new();

    let t: usize = input.value();
    input.skip_line();

    for i in 1..=t {
        let s  = str::from_utf8(input.line()).unwrap();
        // println!("{:?}", &s);

        let e = ast::Expr::parse(s).unwrap();
        // println!("Ast: {:?}", e);

        if e.check_tauto() {
            writeln!(output_buf, "Data set {}: Equivalent", i).unwrap();
        } else {
            writeln!(output_buf, "Data set {}: Different", i).unwrap();
        }
    }

    std::io::stdout().write(&output_buf[..]).unwrap();
}
