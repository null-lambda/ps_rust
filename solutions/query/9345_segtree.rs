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

        for i in (1..n).rev() {
            data[i] = data[i << 1].op(data[i << 1 | 1]);
        }
        Self { n, data }
    }

    // sum on interval [left, right)
    fn query_sum(&self, mut start: usize, mut end: usize) -> T {
        debug_assert!(end <= self.n);
        start += self.n;
        end += self.n;

        let mut result = T::id();
        while start < end {
            if start & 1 != 0 {
                result = result.op(self.data[start]);
            }
            if end & 1 != 0 {
                result = result.op(self.data[end - 1]);
            }
            start = (start + 1) >> 1;
            end = end >> 1;
        }
        result
    }

    fn update(&mut self, mut idx: usize, value: T) {
        idx += self.n;
        self.data[idx] = value;
        while idx > 1 {
            idx >>= 1;
            self.data[idx] = self.data[idx << 1].op(self.data[idx << 1 | 1]);
        }
    }

    fn get(&self, idx: usize) -> T {
        self.data[idx + self.n]
    }
}

#[allow(dead_code)]
fn main() {
    use io::InputStream;
    let input_buf = stdin();
    let mut input: &[u8] = &input_buf[..];

    let mut output_buf = Vec::<u8>::new();

    let test_cases = input.value();
    for _ in 0..test_cases {
        let n: usize = input.value();
        let k: usize = input.value();

        #[derive(Copy, Clone, Debug, PartialEq, Eq)]
        struct MinMax<T>(T, T);

        impl CommMonoid for MinMax<usize> {
            fn id() -> Self {
                Self(usize::MAX, 0)
            }
            fn op(self, other: Self) -> Self {
                Self(self.0.min(other.0), self.1.max(other.1))
            }
        }

        let mut segtree = SegmentTree::from_iter(n, (0..n).map(|i| MinMax(i, i)));

        for _ in 0..k {
            let q = input.value();
            let mut a = input.value();
            let mut b = input.value();
            if a > b {
                std::mem::swap(&mut a, &mut b);
            }
            match q {
                0 => {
                    // swap a, b
                    let temp = segtree.get(a);
                    segtree.update(a, segtree.get(b));
                    segtree.update(b, temp);
                }
                1 => {
                    if segtree.query_sum(a, b + 1) == MinMax(a, b) {
                        writeln!(output_buf, "YES").unwrap();
                    } else {
                        writeln!(output_buf, "NO").unwrap();
                    }
                }
                _ => unreachable!(),
            }
        }
    }

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
