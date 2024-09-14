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

    #[derive(Debug)]
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

    #[derive(Copy, Clone, Debug, PartialEq, Eq)]
    struct Max(i32);
    impl Monoid for Max {
        fn id() -> Self {
            Max(i32::MIN)
        }
        fn op(self, other: Self) -> Self {
            Max(self.0.max(other.0))
        }
    }

    impl Monoid for i32 {
        fn id() -> Self {
            0
        }
        fn op(self, other: Self) -> Self {
            self + other
        }
    }

    impl MonoidAction<Max> for i32 {
        fn apply_to_sum(self, x_sum: Max, _x_count: u32) -> Max {
            Max(self + x_sum.0)
        }
    }

    let n = input.value();
    let mut ys = Vec::new();
    let mut intervals = (0..n)
        .map(|i| {
            let _x1: i32 = input.value();
            let y2: i32 = input.value();
            let _x2: i32 = input.value();
            let y1: i32 = input.value();
            debug_assert!(y2 > y1);
            ys.push((2 * y1, 2 * i));
            ys.push((2 * y2 + 1, 2 * i + 1));
            [y1, y2]
        })
        .collect::<Vec<_>>();

    // coordinate compression
    let y_max = ys.len();
    ys.sort_unstable_by_key(|&(y, _)| y);
    for (order, (_, idx)) in ys.into_iter().enumerate() {
        intervals[idx / 2][idx % 2] = order as i32;
    }

    #[derive(Debug, Copy, Clone)]
    enum Event {
        Start(i32),
        End,
    }

    let mut segtree = LazySegTree::<Max, i32>::from_iter(y_max, (0..y_max).map(|_| Max(0)));
    let mut events = Vec::new();
    for [y1, y2] in intervals {
        segtree.apply_range(y1 as usize, y2 as usize, 1);
        events.push((y1, Event::Start(y2)));
        events.push((y2, Event::End));
    }
    events.sort_unstable_by_key(|&(y, _)| y);

    let mut cross1 = 0;
    let result: i32 = events
        .into_iter()
        .flat_map(|(y1, event_type)| {
            //println!("{:?}", (y1, event_type));
            match event_type {
                Event::Start(y2) => {
                    cross1 += 1;
                    let (y1, y2) = (y1 as usize, y2 as usize);
                    segtree.apply_range(y1, y2, -1);
                    let cross2 = segtree.query_range(y1, y_max).0;
                    Some(cross1 + cross2)
                }
                Event::End => {
                    cross1 -= 1;
                    None
                }
            }
        })
        .max()
        .unwrap();

    println!("{}", result);

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
