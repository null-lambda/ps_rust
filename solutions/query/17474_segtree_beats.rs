use std::io::Write;

use segtree_beats::*;

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

    pub fn stdin<'a>() -> InputAtOnce<'a> {
        let _buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let iter = _buf.split_ascii_whitespace();
        let iter = unsafe { std::mem::transmute(iter) };
        InputAtOnce { _buf, iter }
    }

    pub fn stdout() -> std::io::BufWriter<std::io::Stdout> {
        std::io::BufWriter::new(std::io::stdout())
    }
}

pub mod segtree_beats {
    use std::{iter, ops::Range};

    pub trait MonoidAction {
        type X;
        type F;
        fn id(&self) -> Self::X;
        fn combine(&self, lhs: &Self::X, rhs: &Self::X) -> Self::X;
        fn id_action(&self) -> Self::F;
        fn combine_action(&self, lhs: &Self::F, rhs: &Self::F) -> Self::F;
        fn try_apply_to_sum(&self, f: &Self::F, x_count: u32, x_sum: &Self::X) -> Option<Self::X>;
    }

    pub struct SegTreeBeats<M: MonoidAction> {
        n: usize,
        max_height: u32,
        pub sum: Vec<M::X>,
        pub lazy: Vec<M::F>,
        pub ma: M,
    }

    impl<M: MonoidAction> SegTreeBeats<M> {
        pub fn with_size(n: usize, ma: M) -> Self {
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
        fn push_down(&mut self, node: usize, width: u32) {
            let value = unsafe { &*(&self.lazy[node] as *const _) };
            self.apply(node << 1, width, value);
            self.apply(node << 1 | 1, width, value);
            self.lazy[node] = self.ma.id_action();
        }

        fn pull_up(&mut self, node: usize) {
            self.sum[node] = (self.ma).combine(&self.sum[node << 1], &self.sum[node << 1 | 1]);
        }

        fn apply(&mut self, node: usize, width: u32, value: &M::F) {
            if let Some(applied) = self.ma.try_apply_to_sum(&value, width, &self.sum[node]) {
                self.sum[node] = applied;
                if node < self.n {
                    self.lazy[node] = self.ma.combine_action(&value, &self.lazy[node]);
                }
            } else {
                if node < self.n {
                    self.lazy[node] = self.ma.combine_action(&value, &self.lazy[node]);
                    self.push_down(node, width);
                    self.pull_up(node);
                } else {
                    panic!("try_apply_to_sum should return Some(_) for leaf nodes");
                }
            }
        }

        fn push_range(&mut self, range: Range<usize>) {
            let Range { mut start, mut end } = range;
            start += self.n;
            end += self.n;

            let start_height = 1 + start.trailing_zeros();
            let end_height = 1 + end.trailing_zeros();
            for height in (start_height..=self.max_height).rev() {
                let width = 1 << height - 1;
                self.push_down(start >> height, width);
            }
            for height in (end_height..=self.max_height).rev().skip_while(|&height| {
                height >= start_height && end - 1 >> height == start >> height
            }) {
                let width = 1 << height - 1;
                self.push_down(end - 1 >> height, width);
            }
        }

        pub fn apply_range(&mut self, range: Range<usize>, value: M::F) {
            let Range { mut start, mut end } = range;
            debug_assert!(start <= end && end <= self.n);
            if start == end {
                return;
            }

            self.push_range(range);
            start += self.n;
            end += self.n;
            let mut width: u32 = 1;
            let (mut pull_start, mut pull_end) = (false, false);
            while start < end {
                if pull_start {
                    self.pull_up(start - 1);
                }
                if pull_end {
                    self.pull_up(end);
                }
                if start & 1 != 0 {
                    self.apply(start, width, &value);
                    start += 1;
                    pull_start = true;
                }
                if end & 1 != 0 {
                    self.apply(end - 1, width, &value);
                    pull_end = true;
                }
                start >>= 1;
                end >>= 1;
                width <<= 1;
            }
            start -= 1;
            while end > 0 {
                if pull_start {
                    self.pull_up(start);
                }
                if pull_end && !(pull_start && start == end) {
                    self.pull_up(end);
                }
                start >>= 1;
                end >>= 1;
                width <<= 1;
            }
        }

        pub fn query_range(&mut self, range: Range<usize>) -> M::X {
            let Range { mut start, mut end } = range;

            self.push_range(range);
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

#[derive(Clone)]
struct MinUpdateSumQuery {
    neg_inf: i32,
    inf: i32,
}

#[derive(Clone)]
struct NodeData {
    max: i32,
    max_count: i32,
    sub_max: i32,
    sum: i64,
}

impl MinUpdateSumQuery {
    fn singleton(&self, x: i32) -> NodeData {
        NodeData {
            max: x,
            max_count: 1,
            sub_max: self.neg_inf,
            sum: x as i64,
        }
    }
}

impl MonoidAction for MinUpdateSumQuery {
    type X = NodeData;
    type F = i32;
    fn id(&self) -> Self::X {
        NodeData {
            max: self.neg_inf + 1,
            max_count: 0,
            sub_max: self.neg_inf,
            sum: 0,
        }
    }

    fn combine(&self, lhs: &Self::X, rhs: &Self::X) -> Self::X {
        let (max, max_count, sub_max) = if lhs.max == rhs.max {
            (
                lhs.max,
                lhs.max_count + rhs.max_count,
                lhs.sub_max.max(rhs.sub_max),
            )
        } else if lhs.max > rhs.max {
            (lhs.max, lhs.max_count, lhs.sub_max.max(rhs.max))
        } else {
            (rhs.max, rhs.max_count, lhs.max.max(rhs.sub_max))
        };
        let sum = lhs.sum + rhs.sum;
        NodeData {
            max,
            max_count,
            sub_max,
            sum,
        }
    }

    fn id_action(&self) -> Self::F {
        self.inf
    }

    fn combine_action(&self, lhs: &Self::F, rhs: &Self::F) -> Self::F {
        (*lhs).min(*rhs)
    }

    fn try_apply_to_sum(&self, f: &Self::F, _x_count: u32, x: &Self::X) -> Option<Self::X> {
        if *f >= x.max {
            Some(x.clone())
        } else if *f > x.sub_max {
            Some(NodeData {
                max: *f,
                max_count: x.max_count,
                sub_max: x.sub_max,
                sum: x.sum - x.max_count as i64 * (x.max - *f) as i64,
            })
        } else {
            None
        }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n = input.value();
    let op = MinUpdateSumQuery {
        neg_inf: i32::MIN,
        inf: i32::MAX,
    };
    let xs = (0..n).map(|_| op.singleton(input.value()));
    let mut xs = SegTreeBeats::from_iter(n, xs, op.clone());

    for _ in 0..input.value() {
        let cmd = input.token();
        let l = input.value::<usize>() - 1;
        let r = input.value::<usize>() - 1;
        match cmd {
            "1" => {
                let x: i32 = input.value();
                xs.apply_range(l..r + 1, x);
            }
            "2" => {
                let ans = xs.query_range(l..r + 1).max;
                writeln!(output, "{}", ans).unwrap();
            }
            "3" => {
                let ans = xs.query_range(l..r + 1).sum;
                writeln!(output, "{}", ans).unwrap();
            }
            _ => panic!(),
        }
    }
}
