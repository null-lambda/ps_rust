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

pub mod segtree {
    // monoid, not necesserily commutative
    pub trait Monoid {
        fn id() -> Self;
        fn op(self, rhs: Self) -> Self;
    }

    pub trait PowMonoid: Monoid {
        fn pow(self, n: u32) -> Self;
    }

    // monoid action A -> End(M), where A is a monoid and M is a set.
    // the image of A is a submonoid of End(M)
    pub trait MonoidAction<M>: Monoid {
        fn apply_to_sum(self, x_sum: M, x_count: u32) -> M;
    }

    // monoid action on itself
    impl<M: PowMonoid> MonoidAction<M> for M {
        fn apply_to_sum(self, x_sum: M, x_count: u32) -> M {
            self.pow(x_count).op(x_sum)
        }
    }

    pub struct LazySegTree<T, F> {
        n: usize,
        max_height: u32,
        sum: Vec<T>,
        lazy: Vec<F>,
    }

    impl<T, F> LazySegTree<T, F>
    where
        T: Monoid + Copy + Eq,
        F: MonoidAction<T> + Copy + Eq,
    {
        pub fn from_iter<I>(n: usize, iter: I) -> Self
        where
            T: Clone,
            I: Iterator<Item = T>,
        {
            use std::iter::repeat;
            let n = n.next_power_of_two();
            let mut sum: Vec<T> = repeat(T::id()).take(n).chain(iter).collect();
            sum.resize(2 * n, T::id());
            for i in (0..n).rev() {
                sum[i] = sum[i << 1].op(sum[i << 1 | 1]);
            }
            Self {
                n,
                max_height: usize::BITS - n.leading_zeros(),
                sum,
                lazy: vec![F::id(); n],
            }
        }

        #[inline]
        fn apply(&mut self, node: usize, width: u32, value: F) {
            self.sum[node] = value.apply_to_sum(self.sum[node], width);
            if node < self.n {
                // function application is right associative
                self.lazy[node] = value.op(self.lazy[node]);
            }
        }

        #[inline]
        fn propagate_lazy(&mut self, mut idx: usize) {
            idx += self.n;
            for height in (1..=self.max_height).rev() {
                let node = idx >> height;
                if self.lazy[node] != F::id() {
                    let width: u32 = 1 << (height - 1);
                    self.apply(node << 1, width, self.lazy[node]);
                    self.apply(node << 1 | 1, width, self.lazy[node]);
                    self.lazy[node] = F::id();
                }
            }
        }

        #[inline]
        fn update_sum(&mut self, node: usize, width: u32) {
            self.sum[node] = self.sum[node << 1].op(self.sum[node << 1 | 1]);
            if self.lazy[node] != F::id() {
                self.sum[node] = self.lazy[node].apply_to_sum(self.sum[node], width);
            };
        }

        // sum on interval [left, right)
        pub fn apply_range(&mut self, mut start: usize, mut end: usize, value: F) {
            if value == F::id() || start == end {
                return;
            }
            debug_assert!(end <= self.n);
            self.propagate_lazy(start);
            self.propagate_lazy(end - 1);
            start += self.n;
            end += self.n;
            let mut width: u32 = 1;
            let (mut update_left, mut update_right) = (false, false);
            while start < end {
                if update_left {
                    self.update_sum(start - 1, width);
                }
                if update_right {
                    self.update_sum(end, width);
                }
                if start & 1 != 0 {
                    self.apply(start, width, value);
                    update_left = true;
                }
                if end & 1 != 0 {
                    self.apply(end - 1, width, value);
                    update_right = true;
                }
                start = (start + 1) >> 1;
                end >>= 1;
                width <<= 1;
            }
            start -= 1;
            while end > 0 {
                if update_left {
                    self.update_sum(start, width);
                }
                if update_right && !(update_left && start == end) {
                    self.update_sum(end, width);
                }
                start >>= 1;
                end >>= 1;
                width <<= 1;
            }
        }

        pub fn query_range(&mut self, mut start: usize, mut end: usize) -> T {
            self.propagate_lazy(start);
            self.propagate_lazy(end - 1);
            start += self.n;
            end += self.n;
            let (mut result_left, mut result_right) = (T::id(), T::id());

            while start < end {
                if start & 1 != 0 {
                    result_left = result_left.op(self.sum[start]);
                }
                if end & 1 != 0 {
                    result_right = self.sum[end - 1].op(result_right);
                }
                start = (start + 1) >> 1;
                end >>= 1;
            }

            result_left.op(result_right)
        }
    }
}
fn main() {
    use io::InputStream;
    let input_buf = stdin();
    let mut input: &[u8] = &input_buf[..];

    let mut output_buf = Vec::<u8>::new();

    use segtree::{LazySegTree, Monoid, MonoidAction};

    const p: u64 = 1_000_000_007;

    // prime order field
    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    struct Modular(u64);
    impl Monoid for Modular {
        fn id() -> Self {
            Modular(0)
        }
        fn op(self, other: Self) -> Self {
            Modular((self.0 + other.0) % p)
        }
    }

    // affine transformation x |-> a + b x
    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    struct AffineTrans(u64, u64);
    impl Monoid for AffineTrans {
        fn id() -> Self {
            AffineTrans(0, 1)
        }
        fn op(self, other: Self) -> Self {
            // a comp b = (x |-> a0 + a1 (b0 + b1 x))
            AffineTrans((self.0 + self.1 * other.0) % p, (self.1 * other.1) % p)
        }
    }

    impl MonoidAction<Modular> for AffineTrans {
        fn apply_to_sum(self, x_sum: Modular, x_count: u32) -> Modular {
            // sigma xs (x |-> a0 + a1 x) = x_count * a0 + a1 * sigma xs (x |-> x)
            Modular((self.0 * x_count as u64 + self.1 * x_sum.0) % p)
        }
    }

    let n = input.value();
    let mut segtree = LazySegTree::from_iter(n, (0..n).map(|_| Modular(input.value())));

    let m: usize = input.value();
    for _ in 0..m {
        let q = input.value();
        let x: usize = input.value();
        let y: usize = input.value();
        match q {
            1 => {
                let v = input.value();
                segtree.apply_range(x - 1, y, AffineTrans(v, 1));
            }
            2 => {
                let v = input.value();
                segtree.apply_range(x - 1, y, AffineTrans(0, v));
            }
            3 => {
                let v = input.value();
                segtree.apply_range(x - 1, y, AffineTrans(v, 0));
            }
            4 => {
                writeln!(output_buf, "{}", segtree.query_range(x - 1, y).0).unwrap();
            }
            _ => unreachable!(),
        }
    }

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
