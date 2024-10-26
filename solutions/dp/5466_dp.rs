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
        pub fn with_size(n: usize) -> Self {
            Self {
                n,
                sum: vec![T::id(); n * 2],
            }
        }

        pub fn from_iter<I>(n: usize, iter: I) -> Self
        where
            T: Clone,
            I: Iterator<Item = T>,
        {
            use std::iter::repeat;
            // let n = n.next_power_of_two();
            // let n = n * 2;
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
        pub fn get(&self, idx: usize) -> T {
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

    let mut output_buf = Vec::<u8>::new();

    use std::collections::BTreeMap;

    let n: usize = input.value();
    let u: i32 = input.value();
    let d: i32 = input.value();
    let s: i32 = input.value();

    const T_MAX: usize = 500_001;

    let mut events = BTreeMap::<_, Vec<(i32, i32)>>::new();
    (0..n).for_each(|_| {
        let t: usize = input.value();
        let x: i32 = input.value();
        let m: i32 = input.value();
        events.entry(t - 1).or_default().push((x, m));
    });
    events.entry(T_MAX - 1).or_default().push((s, 0));

    use segtree::{Monoid, SegTree};

    #[derive(Copy, Clone, Eq, PartialEq, Debug)]
    struct Max(i32);
    impl Monoid for Max {
        fn id() -> Self {
            Self(i32::MIN / 10)
        }
        fn op(self, other: Self) -> Self {
            Self(self.0.max(other.0))
        }
    }

    const X_MAX: usize = 500_001;
    let mut segtree_left = SegTree::<Max>::with_size(X_MAX + 1);
    let mut segtree_right = SegTree::<Max>::with_size(X_MAX + 1);

    segtree_left.set(s as usize, Max(d * s));
    segtree_right.set(s as usize, Max(-u * s));

    let mut result = 0;
    for events_t in events.values_mut() {
        if events_t.len() >= 2 {
            events_t.sort_unstable_by_key(|&(x, _)| x);
        }

        if events_t.len() >= 2 {
            let profits: Vec<_> = events_t
                .iter()
                .map(|&(x, m)| {
                    m + (segtree_left.query_range(1, x as usize).0 - d * x)
                        .max(segtree_right.query_range(x as usize + 1, X_MAX + 1).0 + u * x)
                })
                .collect();

            let mut dp_left = profits.clone();
            let mut dp_right = profits;
            for i in 1..events_t.len() {
                let (x, m) = events_t[i];
                let (x_prev, _) = events_t[i - 1];
                dp_left[i] = dp_left[i].max(dp_left[i - 1] + m - d * (x - x_prev));
            }
            for i in (0..events_t.len() - 1).rev() {
                let (x, m) = events_t[i];
                let (x_prev, _) = events_t[i + 1];
                dp_right[i] = dp_right[i].max(dp_right[i + 1] + m - u * (x_prev - x));
            }

            let profits = dp_left.into_iter().zip(dp_right).map(|(p1, p2)| p1.max(p2));
            for (profit, &(x, _)) in profits.zip(events_t.iter()) {
                segtree_left.set(x as usize, Max(profit + d * x));
                segtree_right.set(x as usize, Max(profit - u * x));
                if x == s {
                    result = 0.max(profit);
                }
            }
        } else {
            let (x, m) = events_t[0];
            let profit = m
                + (segtree_left.query_range(1, x as usize).0 - d * x)
                    .max(segtree_right.query_range(x as usize + 1, X_MAX + 1).0 + u * x);
            segtree_left.set(x as usize, Max(profit + d * x));
            segtree_right.set(x as usize, Max(profit - u * x));
            if x == s {
                result = 0.max(profit);
            }
        }
    }

    println!("{}", result);

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
