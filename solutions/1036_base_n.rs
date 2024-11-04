use std::{cmp::Reverse, io::Write, ops::Add};

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

#[derive(Clone, Debug)]
struct Base36(Vec<u8>);

impl Add for Base36 {
    type Output = Self;
    fn add(self, other: Self) -> Self {
        let mut carry = 0;
        let mut res = vec![];

        let (x, y) = if self.0.len() > other.0.len() {
            (&self.0, &other.0)
        } else {
            (&other.0, &self.0)
        };
        let iter = x.iter().zip(y.iter().chain(std::iter::repeat(&0)));
        for (a, b) in iter {
            let sum = a + b + carry;
            carry = (sum >= 36) as u8;
            res.push(sum % 36);
        }
        res.push(carry);
        while res.last() == Some(&0) {
            res.pop();
        }

        Base36(res)
    }
}

impl PartialEq for Base36 {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0
    }
}

impl Eq for Base36 {}

impl PartialOrd for Base36 {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(
            self.0
                .len()
                .cmp(&other.0.len())
                .then_with(|| self.0.iter().rev().cmp(other.0.iter().rev())),
        )
    }
}

impl Ord for Base36 {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

impl std::str::FromStr for Base36 {
    type Err = ();
    fn from_str(s: &str) -> Result<Self, Self::Err> {
        let mut v = vec![];
        for c in s.chars().rev() {
            if c.is_ascii_digit() {
                v.push(c as u8 - b'0');
            } else {
                v.push(c as u8 - b'A' + 10);
            }
        }
        while v.last() == Some(&0) {
            v.pop();
        }
        Ok(Base36(v))
    }
}

impl std::fmt::Display for Base36 {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        if self.0.is_empty() {
            return write!(f, "0");
        }
        for &d in self.0.iter().rev() {
            if d < 10 {
                write!(f, "{}", d)?;
            } else {
                write!(f, "{}", (b'A' + d - 10) as char)?;
            }
        }
        Ok(())
    }
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let xs: Vec<Base36> = (0..n).map(|_| input.value()).collect();
    let k: usize = input.value();

    let mut sum = xs.iter().fold(Base36(vec![]), |acc, x| acc + x.clone());
    let mut scores: Vec<(Base36, u8)> = (0..36)
        .map(|d| {
            (
                xs.iter()
                    .map(|x| {
                        let mut y: Vec<_> =
                            x.0.iter()
                                .map(|&c| if c == d { 35 - d } else { 0 })
                                .collect();
                        while y.last() == Some(&0) {
                            y.pop();
                        }
                        Base36(y)
                    })
                    .fold(Base36(vec![]), |acc, x| acc + x),
                d,
            )
        })
        .collect();
    scores.sort_unstable_by_key(|(x, _)| Reverse(x.clone()));

    for (s, _d) in &scores[..k] {
        sum = sum + s.clone();
    }

    writeln!(output, "{}", sum).unwrap();
}
