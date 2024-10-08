use std::{
    cmp::Reverse,
    collections::HashMap,
    fmt::Display,
    io::Write,
    iter::{once, Peekable},
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
    pub w: usize,
    pub data: Vec<T>,
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

mod iter {
    pub struct Product<I: Iterator, J: Iterator> {
        i: I,
        j: J,
        i_pos: Option<I::Item>,
        j_orig: J,
    }

    impl<I, J> Iterator for Product<I, J>
    where
        I: Iterator,
        I::Item: Clone,
        J: Iterator + Clone,
    {
        type Item = (I::Item, J::Item);
        fn next(&mut self) -> Option<Self::Item> {
            let j_next = match self.j.next() {
                Some(j_pos) => j_pos,
                None => {
                    self.i_pos = self.i.next();
                    self.j = self.j_orig.clone();
                    self.j.next()?
                }
            };
            Some((self.i_pos.clone()?, j_next))
        }
    }

    pub fn product<I, J>(i: I, j: J) -> Product<I::IntoIter, J::IntoIter>
    where
        I: IntoIterator,
        I::Item: Clone,
        J: IntoIterator,
        J::IntoIter: Clone,
    {
        let mut i = i.into_iter();
        let j_orig = j.into_iter();
        let i_pos = i.next();
        Product {
            i,
            j: j_orig.clone(),
            i_pos,
            j_orig,
        }
    }
}

const P: i32 = 2011;

fn mod_pow(mut base: i32, mut exp: i32) -> i32 {
    let mut result = 1;
    while exp > 0 {
        if exp % 2 == 1 {
            result = result * base % P;
        }
        base = base * base % P;
        exp >>= 1;
    }
    result
}

fn mod_inv(n: i32) -> i32 {
    mod_pow(n, P - 2)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Token {
    Num(i32),
    Op(u8),
}

fn parse_and_eval(
    grid: &Grid<u8>,
    i_range: RangeInclusive<usize>,
    j_range: RangeInclusive<usize>,
) -> i32 {
    use iter::product;

    // println!("{}", grid.colored()); // debug

    let (i0, mut j) = product(j_range.clone(), i_range.clone())
        .map(|(j, i)| (i, j))
        .find(|&u| grid[u] != b'.')
        .unwrap();

    let mut tokens = vec![];
    while j <= *j_range.end() {
        let c = grid[(i0, j)];
        match c {
            b'.' => {
                if i0 > *i_range.start() {
                    let c_upper = grid[(i0 - 1, j)];
                    if matches!(c_upper, b'0'..=b'9') {
                        tokens.push(Token::Op(b'^'));
                        tokens.push(Token::Num((c_upper - b'0') as i32));
                    }
                }
            }
            b'0'..=b'9' => tokens.push(Token::Num((c - b'0') as i32)),
            b'-' if j < *j_range.end() && grid[(i0, j + 1)] == b'-' => {
                // find continuous sequence of j's
                let mut j_end = j;
                while j_end < *j_range.end() - 1 && grid[(i0, j_end + 1)] == b'-' {
                    j_end += 1;
                }
                let top = parse_and_eval(grid, *i_range.start()..=i0 - 1, j..=j_end);
                let bottom = parse_and_eval(grid, i0 + 1..=*i_range.end(), j..=j_end);
                tokens.push(Token::Num(top * mod_inv(bottom) % P));
                j = j_end + 1;
            }
            b'+' | b'-' | b'*' | b'/' | b'(' | b')' => tokens.push(Token::Op(c)),
            _ => panic!(),
        }
        j += 1;
    }

    fn infix_binding_power(op: u8) -> Option<(u8, u8)> {
        let result = match op {
            b'+' | b'-' => (1, 2),
            b'*' => (4, 5),
            b'^' => (7, 6),
            _ => return None,
        };
        Some(result)
    }

    // println!("{:?}", tokens); // debug
    type TokenStream = Peekable<std::vec::IntoIter<Token>>;
    fn parse_tokens(tokens: &mut TokenStream, min_bp: u8) -> i32 {
        let mut acc = match tokens.next().unwrap() {
            Token::Num(n) => n,
            Token::Op(b'(') => {
                let acc = parse_tokens(tokens, 0);
                assert_eq!(tokens.next(), Some(Token::Op(b')')));
                acc
            }
            Token::Op(b'-') => -parse_tokens(tokens, 3),
            _ => panic!(),
        };
        //
        loop {
            let op = match tokens.peek() {
                Some(Token::Op(op)) => *op,
                None => break,
                _ => panic!(),
            };

            if let Some((l_bp, r_bp)) = infix_binding_power(op) {
                if l_bp < min_bp {
                    break;
                }
                tokens.next();

                let rhs = parse_tokens(tokens, r_bp);
                acc = match op {
                    b'+' => acc + rhs,
                    b'-' => acc - rhs,
                    b'*' => acc * rhs % P,
                    b'/' => acc * mod_inv(rhs) % P,
                    b'^' => mod_pow(acc, rhs),
                    _ => panic!(),
                };
                continue;
            }
            break;
        }
        acc
    }

    parse_tokens(&mut tokens.into_iter().peekable(), 0)
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout_buf();

    loop {
        let h: usize = input.value();
        if h == 0 {
            break;
        }
        let rows: Vec<Vec<u8>> = (0..h).map(|_| input.token().bytes().collect()).collect();
        let w = rows[0].len();

        let grid = rows
            .into_iter()
            .flatten()
            .collect::<Grid<u8>>()
            .with_shape(w);
        let mut result = parse_and_eval(&grid, 0..=h - 1, 0..=w - 1);
        result = (result % P + P) % P;
        writeln!(output, "{}", result).unwrap();
    }
}

