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

use std::ops::Range;

// binary operation with identity and associative property
// axioms:
//    op(op(a, b), c) = op(a, op(b, c))
//    op(a, id) = id
//    op(id, a) = a
trait Monoid {
    fn id() -> Self;
    fn op(self, rhs: Self) -> Self;
    fn op_assign(&mut self, rhs: Self);
}

struct SegmentTree<T: Monoid> {
    n: usize,
    data: Vec<T>,
}

impl<T: Monoid + Copy> SegmentTree<T> {
    fn from_sized_iter<I>(iter: I) -> Self
    where
        T: Clone,
        I: IntoIterator<Item = T> + ExactSizeIterator,
    {
        let n = iter.len();
        let mut data = Vec::with_capacity(2 * n);
        data.resize(n, T::id());
        data.extend(iter);

        let mut tree = Self { n, data };
        tree.init();
        tree
    }

    fn init(&mut self) {
        for i in (1..self.n).rev() {
            self.data[i] = T::op(self.data[i << 1], self.data[i << 1 | 1]);
        }
    }

    // sum on interval [left, right)
    fn query_sum(&self, Range { mut start, mut end }: Range<usize>) -> T {
        debug_assert!(end <= self.n);
        start += self.n;
        end += self.n;

        let mut result = T::id();
        while start < end {
            if start & 1 != 0 {
                result.op_assign(self.data[start]);
                start += 1;
            }
            if end & 1 != 0 {
                result.op_assign(self.data[end - 1]);
                end -= 1;
            }
            start >>= 1;
            end >>= 1;
        }
        result
    }

    fn update(&mut self, mut idx: usize, value: T) {
        idx += self.n;
        self.data[idx] = value;
        while idx > 1 {
            self.data[idx >> 1] = T::op(self.data[idx], self.data[idx ^ 1]);
            idx >>= 1;
        }
    }
}

fn main() {
    use io::InputStream;
    let input_buf = stdin();
    let mut input: &[u8] = &input_buf[..];

    let mut output_buf = Vec::<u8>::new();

    // infinity
    const INF: u32 = u32::MAX;

    #[derive(Clone, Copy)]
    struct MinMax {
        min: u32,
        max: u32,
    }

    impl MinMax {
        fn new(value: u32) -> Self {
            Self {
                min: value,
                max: value,
            }
        }
    }

    impl Monoid for MinMax {
        fn id() -> Self {
            Self { min: INF, max: 0 }
        }
        fn op(self, rhs: Self) -> Self {
            Self {
                min: self.min.min(rhs.min),
                max: self.max.max(rhs.max),
            }
        }
        fn op_assign(&mut self, rhs: Self) {
            *self = self.op(rhs);
        }
    }

    let (n, m) = (input.value(), input.value());
    let tree = SegmentTree::from_sized_iter((0..n).map(|_| MinMax::new(input.value())));

    for _ in 0..m {
        let (a, b): (usize, usize) = (input.value(), input.value());
        let MinMax { min, max } = tree.query_sum(a - 1..b);
        writeln!(output_buf, "{} {}", min, max).unwrap();
    }

    //writeln!(output_buf, "{}", result).unwrap();

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
