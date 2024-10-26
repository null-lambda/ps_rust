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

// commutative monoid
trait CommMonoid {
    fn id() -> Self;
    fn op(self, rhs: Self) -> Self;
}

struct SegmentTree<T: CommMonoid> {
    n: usize,
    data: Vec<T>,
}

impl<T: CommMonoid + Copy> SegmentTree<T> {
    fn from_iter<I>(n: usize, iter: I) -> Self
    where
        T: Clone,
        I: Iterator<Item = T>,
    {
        use std::iter::repeat;
        let mut data: Vec<T> = repeat(T::id()).take(n).chain(iter).collect();
        data.resize(2 * n, T::id());
        Self { n, data }
    }

    // sum on interval [left, right)
    fn apply_range(&mut self, mut start: usize, mut end: usize, value: T) {
        debug_assert!(end <= self.n);
        start += self.n;
        end += self.n;

        while start < end {
            if start & 1 != 0 {
                self.data[start] = self.data[start].op(value);
            }
            if end & 1 != 0 {
                self.data[end - 1] = self.data[end - 1].op(value);
            }
            start = (start + 1) >> 1;
            end = end >> 1;
        }
    }

    fn query(&self, mut idx: usize) -> T {
        idx += self.n;
        let mut result = self.data[idx];
        while idx > 1 {
            idx >>= 1;
            result = result.op(self.data[idx]);
        }
        result
    }
}

#[allow(dead_code)]
fn main() {
    use io::InputStream;
    let input_buf = stdin();
    let mut input: &[u8] = &input_buf[..];

    let mut output_buf = Vec::<u8>::new();

    let n: usize = input.value();
    let xs = (0..n).map(|_| input.value());

    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    struct AddGroup(i64);

    impl CommMonoid for AddGroup {
        fn id() -> Self {
            Self(0)
        }
        fn op(self, other: Self) -> Self {
            Self(self.0 + other.0)
        }
    }

    let mut segtree = SegmentTree::from_iter(n, xs.map(|x| AddGroup(x)));

    let m = input.value();
    for _ in 0..m {
        let q = input.value();
        match q {
            1 => {
                let i: usize = input.value();
                let j: usize = input.value();
                let k = input.value();
                segtree.apply_range(i - 1, j, AddGroup(k));
            }
            2 => {
                let x: usize = input.value();
                writeln!(output_buf, "{}", segtree.query(x - 1).0).unwrap();
            }
            _ => unreachable!(),
        }
    }

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
