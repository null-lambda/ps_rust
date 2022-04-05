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
            .map(|&c| matches!(c, b'\n' | b'\r' | 0))
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

// Extended euclidean algorithm
// find (d, x, y) satisfying d = gcd(a, b) and a * x + b * y = d
fn egcd(a: i64, b: i64) -> (i64, i64, i64) {
    debug_assert!(a >= 0 && b >= 0);
    let (mut c, mut x, mut y) = if a > b {
        ((a, b), (1, 0), (0, 1))
    } else {
        ((b, a), (0, 1), (1, 0))
    };

    while c.1 > 0 {
        let q = c.0 / c.1;
        x = (x.1, (x.0 - q * x.1));
        y = (y.1, (y.0 - q * y.1));
        c = (c.1, c.0 % c.1);
    }
    (c.0, x.0, y.0)
}

fn mod_inv(a: i64, p: i64) -> i64 {
    let (d, x, _) = egcd(a, p);
    debug_assert!(d == 1);
    x
}

use std::iter::once;

fn main() {
    use io::*;

    let input_buf = stdin();
    let mut input: &[u8] = &input_buf;

    let mut output_buf = Vec::<u8>::new();

    const P: u64 = 1_000_000_007;
    let n_bound = 4_000_000;
    let factorial: Vec<u64> = once(1)
        .chain((1..=n_bound).scan(1, |acc, n| {
            *acc = (*acc * n) % P;
            Some(*acc)
        }))
        .collect();

    let n_tests = input.value();
    for _ in 0..n_tests {
        let n: usize = input.value();
        let k: usize = input.value();

        let result = factorial[n]
            * (mod_inv((factorial[k] * factorial[n - k] % P) as i64, P as i64).rem_euclid(P as i64)
                as u64)
            % P;
        writeln!(output_buf, "{}", result).unwrap();
    } 

    std::io::stdout().write(&output_buf[..]).unwrap();
}
