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
            let idx = self.iter().position(|&c| !is_whitespace(c)).unwrap();
            //.expect("no available tokens left");
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
}

use std::io::{BufReader, Read, Write};

fn stdin() -> Vec<u8> {
    let stdin = std::io::stdin();
    let mut reader = BufReader::new(stdin.lock());

    let mut input_buf: Vec<u8> = vec![];
    reader.read_to_end(&mut input_buf).unwrap();
    input_buf
}

fn main() {
    use io::InputStream;
    let input_buf = stdin();
    let mut input: &[u8] = &input_buf[..];

    let mut output_buf = Vec::<u8>::new();

    let (n, m, k): (_, usize, usize) = (input.value(), input.value(), input.value());

    const p: u64 = 1_000_000_007;

    #[derive(Clone, Copy)]
    struct MulInt(u64);

    impl Monoid for MulInt {
        fn id() -> Self {
            Self(1)
        }
        fn op(self, rhs: Self) -> Self {
            Self((self.0 * rhs.0) % p)
        }
        fn op_assign(&mut self, rhs: Self) {
            *self = self.op(rhs);
        }
    }

    let mut tree = SegmentTree::from_sized_iter((0..n).map(|_| MulInt(input.value())));

    for _ in 0..m + k {
        match input.value::<i32>() {
            1 => {
                let (b, c): (usize, u64) = (input.value(), input.value());
                tree.update(b - 1, MulInt(c));
            }
            _ => {
                let (b, c): (usize, usize) = (input.value(), input.value());
                writeln!(output_buf, "{}", tree.query_sum(b - 1..c).0).unwrap();
            }
        }
    }

    //writeln!(output_buf, "{}", result).unwrap();

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
