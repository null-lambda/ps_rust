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

// Extended euclidean algorithm
// find (d, x, y) satisfying d = gcd(a, b) and a * x + b * y = d
fn egcd(a: u64, b: u64) -> (u64, i64, i64) {
    let (mut c, mut x, mut y) = if a > b {
        ((a, b), (1, 0), (0, 1))
    } else {
        ((b, a), (0, 1), (1, 0))
    };

    while c.1 > 0 {
        let q = c.0 / c.1;
        x = (x.1, (x.0 - (q as i64) * x.1));
        y = (y.1, (y.0 - (q as i64) * y.1));
        c = (c.1, c.0 % c.1);
    }
    (c.0, x.0, y.0)
}

fn solve(k: u64, c: u64) -> Option<u64> {
    // equation:
    //   1 = k * x + c * (-y)
    // general solution:
    //   x = x0 + c * t, y = y0 + k * t
    //   y > 0 iff y >=0 iff t >= -y0 / k
    let (d, _, y0) = egcd(k, c);
    (d == 1)
        .then(|| {
            let y_lower_bound = ((k + c) / c) as i64;
            let y = (y0 - y_lower_bound).rem_euclid(k as i64) + y_lower_bound;
            (y <= 1_000_000_000).then(|| y as u64)
        })
        .flatten()
}

fn main() {
    use io::*;

    let input_buf = stdin();
    let mut input: &[u8] = &input_buf;

    let mut output_buf = Vec::<u8>::new();

    let t = input.value();
    for _ in 0..t {
        let (k, c) = (input.value(), input.value());
        match solve(k, c) {
            Some(result) => writeln!(output_buf, "{}", result).unwrap(),
            None => writeln!(output_buf, "IMPOSSIBLE").unwrap(),
        };
    }
    std::io::stdout().write(&output_buf[..]).unwrap();
}
