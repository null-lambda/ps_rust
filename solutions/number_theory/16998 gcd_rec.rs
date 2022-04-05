mod io {
    use std::fmt::Debug;
    use std::str::*;

    pub trait InputStream {
        fn token(&mut self) -> &[u8];
        fn line(&mut self) -> &[u8];

        fn skip_line(&mut self) {
            self.line();
        }

        #[inline]
        fn value<T>(&mut self) -> T
        where
            T: FromStr,
            T::Err: Debug,
        {
            let token = self.token();
            let token = unsafe { from_utf8_unchecked(token) };
            token.parse::<T>().unwrap()
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
            .map(|&c| {
                matches! {c, b'\n' | b'\r' | 0}
            })
            .unwrap_or_else(|| false)
        {
            s = &s[..s.len() - 1];
        }
        s
    }

    impl InputStream for &[u8] {
        fn token(&mut self) -> &[u8] {
            let i = self.iter().position(|&c| !is_whitespace(c)).unwrap();
            //.expect("no available tokens left");
            *self = &self[i..];
            let i = self
                .iter()
                .position(|&c| is_whitespace(c))
                .unwrap_or_else(|| self.len());
            let (token, buf_new) = self.split_at(i);
            *self = buf_new;
            token
        }

        fn line(&mut self) -> &[u8] {
            let i = self
                .iter()
                .position(|&c| c == b'\n')
                .map(|i| i + 1)
                .unwrap_or_else(|| self.len());
            let (line, buf_new) = self.split_at(i);
            *self = buf_new;
            trim_newline(line)
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

fn gcd(a: u32, b: u32) -> u32 {
    let mut c = if a > b { (a, b) } else { (b, a) };

    while c.1 > 0 {
        c = (c.1, c.0 % c.1);
    }
    (c.0, x.0, y.0)
}

mod solution {
    use super::gcd;

    // sum 1 <= i <= n, (p * i) mod q
    pub fn get(n: u32, mut p: u32, q: u32) -> i64 {
        // reduce to the case p < q.
        p = p % q;

        // reduce to the case gcd(p, q) = 1.
        let (d, ..) = egcd(p, q);
        d as i64 * solve_1(n as i64, p as i64 / d as i64, q as i64 / d as i64)
    }

    fn solve_1(n: i64, p: i64, q: i64) -> i64 {
        // reduce to the case n < q.
        (n / q) * q * (q - 1) / 2 + solve_2(n % q, p, q)
    }

    fn solve_2(n: i64, p: i64, q: i64) -> i64 {
        if p == 0 {
            return 0;
        }
        let k_max = n * p / q;
        let mut result = p * n * (n + 1) / 2;
        if k_max != 0 {
            // result -= q * (k_max * (n + 1) - (1..=k_max).map(|k| (k * q + p - 1) / p).sum::<i64>());
            // result -= q * (k_max * (n + 1) + (1..=k_max).map(|k| (k * (-q)).div_euclid(p)).sum::<i64>());
            /*
            result -= q
                * (k_max * (n + 1)
                    + (-q * k_max * (k_max + 1) / 2 - solve_1(k_max, (-q).rem_euclid(p), p)) / p);
            */
            result -= q
                * (k_max * (n + 1)
                    + (-(q * k_max * (k_max + 1) + (p - 1) * p) / 2
                        + solve_1(p - 1 - k_max, q % p, p))
                        / p);
        }
        result
    }

    pub fn get_naive(n: u32, p: u32, q: u32) -> i64 {
        (1..=n).map(|i| (i * p) as i64 % q as i64).sum()
    }

    #[test]
    fn test_solution() {
        let bound = 100;
        for p in 1..bound {
            for q in 1..bound {
                for n in 1..bound {
                    assert_eq!(
                        solution::get(n, p, q),
                        solution::get_naive(n, p, q),
                        "{:?}",
                        (n, p, q)
                    );
                }
            }
        }
    }
}

fn main() {
    use io::InputStream;
    let input_buf = stdin();
    let mut input: &[u8] = &input_buf[..];

    let mut output_buf = Vec::<u8>::new();

    let w = input.value();
    for _ in 0..w {
        let p = input.value();
        let q = input.value();
        let n = input.value();
        let result = solution::get(n, p, q);
        writeln!(output_buf, "{:?}", result).unwrap();
    }

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
