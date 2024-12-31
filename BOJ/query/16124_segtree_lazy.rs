use std::io::Write;

use segtree::{LazySegTree, MonoidAction};

mod simple_io {
    pub struct InputAtOnce<'a> {
        _buf: String,
        iter: std::str::SplitAsciiWhitespace<'a>,
    }

    impl<'a> InputAtOnce<'a> {
        pub fn token(&mut self) -> &'a str {
            self.iter.next().unwrap_or_default()
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().unwrap()
        }
    }

    pub fn stdin_at_once<'a>() -> InputAtOnce<'a> {
        let _buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let iter = _buf.split_ascii_whitespace();
        let iter = unsafe { std::mem::transmute(iter) };
        InputAtOnce { _buf, iter }
    }

    pub fn stdout() -> std::io::BufWriter<std::io::Stdout> {
        std::io::BufWriter::new(std::io::stdout())
    }
}

pub mod segtree {
    use std::{iter, ops::Range};

    pub trait MonoidAction {
        type X;
        type F;
        fn id(&self) -> Self::X;
        fn combine(&self, lhs: &Self::X, rhs: &Self::X) -> Self::X;
        fn id_action(&self) -> Self::F;
        fn combine_action(&self, lhs: &Self::F, rhs: &Self::F) -> Self::F;
        fn apply_to_sum(&self, f: &Self::F, x_count: u32, x_sum: &Self::X) -> Self::X;
    }

    pub struct LazySegTree<M: MonoidAction> {
        n: usize,
        max_height: u32,
        pub sum: Vec<M::X>,
        pub lazy: Vec<M::F>,
        pub ma: M,
    }

    impl<M: MonoidAction> LazySegTree<M> {
        pub fn with_size(n: usize, ma: M) -> Self {
            let n = n.next_power_of_two();
            Self {
                n,
                max_height: usize::BITS - n.leading_zeros(),
                sum: iter::repeat_with(|| ma.id()).take(2 * n).collect(),
                lazy: iter::repeat_with(|| ma.id_action()).take(n).collect(),
                ma,
            }
        }

        pub fn from_iter<I>(n: usize, iter: I, ma: M) -> Self
        where
            I: IntoIterator<Item = M::X>,
        {
            let n = n.next_power_of_two();
            let mut sum: Vec<_> = (iter::repeat_with(|| ma.id()).take(n))
                .chain(
                    iter.into_iter()
                        .chain(iter::repeat_with(|| ma.id()))
                        .take(n),
                )
                .collect();
            for i in (0..n).rev() {
                sum[i] = ma.combine(&sum[i << 1], &sum[i << 1 | 1]);
            }
            Self {
                n,
                max_height: usize::BITS - n.leading_zeros(),
                sum,
                lazy: iter::repeat_with(|| ma.id_action()).take(n).collect(),
                ma,
            }
        }

        fn apply(&mut self, idx: usize, width: u32, value: &M::F) {
            self.sum[idx] = self.ma.apply_to_sum(&value, width, &self.sum[idx]);
            if idx < self.n {
                self.lazy[idx] = self.ma.combine_action(&value, &self.lazy[idx]);
            }
        }

        fn push_lazy(&mut self, mut idx: usize) {
            idx += self.n;
            for height in (1..=self.max_height).rev() {
                let node = idx >> height;
                let width: u32 = 1 << (height - 1);
                let value = unsafe { &*(&self.lazy[node] as *const _) };
                self.apply(node << 1, width, value);
                self.apply(node << 1 | 1, width, value);
                self.lazy[node] = self.ma.id_action();
            }
        }

        fn pull_sum(&mut self, node: usize, width: u32) {
            self.sum[node] = (self.ma).combine(&self.sum[node << 1], &self.sum[node << 1 | 1]);
            self.sum[node] = (self.ma).apply_to_sum(&self.lazy[node], width, &self.sum[node]);
        }

        pub fn apply_range(&mut self, range: Range<usize>, value: M::F) {
            let Range { mut start, mut end } = range;
            debug_assert!(start <= end);
            debug_assert!(end <= self.n);
            if start == end {
                return;
            }
            self.push_lazy(start);
            self.push_lazy(end - 1);
            start += self.n;
            end += self.n;
            let mut width: u32 = 1;
            let (mut update_left, mut update_right) = (false, false);
            while start < end {
                if update_left {
                    self.pull_sum(start - 1, width);
                }
                if update_right {
                    self.pull_sum(end, width);
                }
                if start & 1 != 0 {
                    self.apply(start, width, &value);
                    update_left = true;
                }
                if end & 1 != 0 {
                    self.apply(end - 1, width, &value);
                    update_right = true;
                }
                start = (start + 1) >> 1;
                end >>= 1;
                width <<= 1;
            }
            start -= 1;
            while end > 0 {
                if update_left {
                    self.pull_sum(start, width);
                }
                if update_right && !(update_left && start == end) {
                    self.pull_sum(end, width);
                }
                start >>= 1;
                end >>= 1;
                width <<= 1;
            }
        }

        pub fn query_range(&mut self, range: Range<usize>) -> M::X {
            let Range { mut start, mut end } = range;
            self.push_lazy(start);
            self.push_lazy(end - 1);
            start += self.n;
            end += self.n;
            let (mut result_left, mut result_right) = (self.ma.id(), self.ma.id());
            while start < end {
                if start & 1 != 0 {
                    result_left = self.ma.combine(&result_left, &self.sum[start]);
                }
                if end & 1 != 0 {
                    result_right = self.ma.combine(&self.sum[end - 1], &result_right);
                }
                start = (start + 1) >> 1;
                end >>= 1;
            }

            self.ma.combine(&result_left, &result_right)
        }

        pub fn partition_point(&mut self, mut pred: impl FnMut(&M::X, u32) -> bool) -> usize {
            let mut i = 1;
            let mut width = self.n as u32;
            while i < self.n {
                width >>= 1;
                let value = unsafe { &*(&self.lazy[i] as *const _) };
                self.apply(i << 1, width, value);
                self.apply(i << 1 | 1, width, value);
                self.lazy[i] = self.ma.id_action();
                i <<= 1;
                if pred(&self.sum[i], width) {
                    i |= 1;
                }
            }
            i - self.n
        }
    }
}

const P: u64 = 998_244_353;

type DigitSum = ([u64; 10], u64);
type SubsTable = [u8; 10];
struct DigitSumOp;

fn new_digit_sum(b: u8) -> DigitSum {
    let mut sum = [0; 10];
    sum[(b - b'0') as usize] = 1;
    (sum, 10)
}

fn join_digit_sum(x: DigitSum) -> u64 {
    (0..10)
        .reduce(|acc, i| (acc + i * x.0[i as usize]) % P)
        .unwrap()
}

fn subs(from: char, to: char) -> SubsTable {
    let mut res = DigitSumOp.id_action();
    res[(from as u8 - b'0') as usize] = to as u8 - b'0';
    res
}

impl MonoidAction for DigitSumOp {
    type X = DigitSum;
    type F = SubsTable;
    fn id(&self) -> Self::X {
        ([0; 10], 1)
    }
    fn combine(&self, lhs: &Self::X, rhs: &Self::X) -> Self::X {
        let mut sum = [0; 10];
        let pow_10 = lhs.1 * rhs.1 % P;
        for i in 0..10 {
            sum[i] = (lhs.0[i] * rhs.1 + rhs.0[i]) % P;
        }
        (sum, pow_10)
    }
    fn id_action(&self) -> Self::F {
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    }
    fn combine_action(&self, lhs: &Self::F, rhs: &Self::F) -> Self::F {
        let mut res = [0; 10];
        for i in 0..10 {
            res[i] = lhs[rhs[i] as usize];
        }
        res
    }
    fn apply_to_sum(&self, f: &Self::F, _x_count: u32, x_sum: &Self::X) -> Self::X {
        let (orig, pow_10) = x_sum;
        let mut res = [0; 10];
        for i in 0..10 {
            res[f[i] as usize] = (res[f[i] as usize] + orig[i]) % P;
        }
        (res, *pow_10)
    }
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let s = input.token().as_bytes();
    let mut tree = LazySegTree::from_iter(s.len(), s.iter().map(|&b| new_digit_sum(b)), DigitSumOp);
    for _ in 0..input.value() {
        let cmd = input.token();
        let i = input.value::<usize>() - 1;
        let j = input.value::<usize>() - 1;
        match cmd {
            "1" => {
                let from: char = input.value();
                let to: char = input.value();
                tree.apply_range(i..j + 1, subs(from, to));
            }
            "2" => {
                let sum = tree.query_range(i..j + 1);
                writeln!(output, "{}", join_digit_sum(sum)).unwrap();
            }
            _ => panic!(),
        }

        //         for i in 0..s.len() - 3 {
        //             let sum = tree.query_range(i..i + 4);
        //             write!(output, "{} ", join_digit_sum(sum)).unwrap();
        //         }
        //         writeln!(output).unwrap();
    }
}
