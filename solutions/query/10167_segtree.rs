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

    #[derive(Debug)]
    pub struct SegTree<T> {
        n: usize,
        sum: Vec<T>,
    }

    impl<T> SegTree<T>
    where
        T: Monoid + Copy + Eq,
    {
        pub fn from_iter<I>(n: usize, iter: I) -> Self
        where
            T: Clone,
            I: Iterator<Item = T>,
        {
            use std::iter::repeat;
            // let n = n.next_power_of_two();
            let n = n * 2;
            let mut sum: Vec<T> = repeat(T::id())
                .take(n)
                .chain(iter)
                .chain(repeat(T::id()))
                .take(2 * n)
                .collect();
            for i in (0..n).rev() {
                sum[i] = sum[i << 1].op(sum[i << 1 | 1]);
            }
            Self { n, sum }
        }

        pub fn set(&mut self, mut idx: usize, value: T) {
            debug_assert!(idx < self.n);
            idx += self.n;
            self.sum[idx] = value;
            while idx > 1 {
                idx >>= 1;
                self.sum[idx] = self.sum[idx << 1].op(self.sum[idx << 1 | 1]);
            }
        }

        #[inline]
        pub fn get(&mut self, idx: usize) -> T {
            self.sum[idx + self.n]
        }

        // sum on interval [left, right)
        pub fn query_range(&self, mut start: usize, mut end: usize) -> T {
            debug_assert!(start < self.n && end <= self.n);
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

    // let mut output_buf = Vec::<u8>::new();

    let n = input.value();
    let mut mines: Vec<_> = (0..n)
        .map(|_| {
            let x: u32 = input.value();
            let y: u32 = input.value();
            let w: i64 = input.value();
            (x, y, w)
        })
        .collect();
    mines.sort_unstable_by_key(|&(x, y, ..)| (x, y));

    let mut ys: Vec<_> = mines
        .iter()
        .enumerate()
        .map(|(i, &(_, y, _))| (i as u32, y))
        .collect();
    ys.sort_unstable_by_key(|&(_, y)| y);

    let mut y_count = 0;
    for i in 0..ys.len() {
        mines[ys[i].0 as usize].1 = y_count as u32;
        if i == ys.len() - 1 || ys[i].1 != ys[i + 1].1 {
            y_count += 1;
        }
    }

    use segtree::{Monoid, SegTree};

    impl Monoid for i64 {
        fn id() -> Self {
            0
        }
        fn op(self, other: Self) -> Self {
            self + other
        }
    }

    #[derive(Copy, Clone, PartialEq, Eq, Debug)]
    struct MaximalSubSum<T: Monoid + Ord> {
        sum: T,
        full: T,
        left: T,
        right: T,
    }

    impl Monoid for MaximalSubSum<i64> {
        fn id() -> Self {
            MaximalSubSum {
                sum: 0,
                full: 0,
                left: 0,
                right: 0,
            }
        }
        fn op(self, other: Self) -> Self {
            MaximalSubSum {
                sum: self.sum.max(other.sum).max(self.right.op(other.left)),
                full: self.full.op(other.full),
                left: self.left.max(self.full.op(other.left)),
                right: other.right.max(self.right.op(other.full)),
            }
        }
    }

    let mut result: i64 = 0;
    for i in 0..n {
        if i == 0 || mines[i - 1].0 != mines[i].0 {
            use std::iter::repeat;
            let mut segtree = SegTree::from_iter(n, repeat(MaximalSubSum::id()).take(n));
            for j in 0..mines[i..].len() {
                let (x, y, w) = mines[i..][j];

                let value = segtree.get(y as usize).sum + w;
                segtree.set(
                    y as usize,
                    MaximalSubSum {
                        sum: value,
                        full: value,
                        left: value,
                        right: value,
                    },
                );

                if j == mines[i..].len() - 1 || x != mines[i..][j + 1].0 {
                    let value: i64 = segtree.query_range(0, n).sum;
                    result = result.max(value);
                }
            }
        }
    }

    println!("{:?}", result);

    // std::io::stdout().write_all(&output_buf[..]).unwrap();
}
