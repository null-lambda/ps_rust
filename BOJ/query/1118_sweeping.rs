use std::cmp::Ordering;
use std::io::Write;

mod simple_io {
    use std::string::*;

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

fn clamp<T: Ord>(val: T, min: T, max: T) -> T {
    val.max(min).min(max)
}

use std::{collections::HashMap, hash::Hash};

use segtree_lazy::MonoidAction;

fn compress_coord<T: Ord + Clone + Hash>(
    xs: impl IntoIterator<Item = T>,
) -> (Vec<T>, HashMap<T, u32>) {
    let mut x_map: Vec<T> = xs.into_iter().collect();
    x_map.sort_unstable();
    x_map.dedup();

    let x_map_inv = x_map
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, x)| (x, i as u32))
        .collect();

    (x_map, x_map_inv)
}

pub mod segtree_lazy {
    use std::{iter, ops::Range};

    pub trait MonoidAction {
        type X;
        type F;
        const IS_X_COMMUTATIVE: bool = false; // TODO
        fn id(&self) -> Self::X;
        fn combine(&self, lhs: &Self::X, rhs: &Self::X) -> Self::X;
        fn id_action(&self) -> Self::F;
        fn combine_action(&self, lhs: &Self::F, rhs: &Self::F) -> Self::F;
        fn apply_to_sum(&self, f: &Self::F, x_count: u32, x_sum: &mut Self::X);
    }

    pub struct SegTree<M: MonoidAction> {
        n: usize,
        max_height: u32,
        sum: Vec<M::X>,
        lazy: Vec<M::F>,
        ma: M,
    }

    impl<M: MonoidAction> SegTree<M> {
        pub fn with_size(n: usize, ma: M) -> Self {
            Self {
                n,
                max_height: usize::BITS - n.leading_zeros(),
                sum: iter::repeat_with(|| ma.id()).take(2 * n).collect(),
                lazy: iter::repeat_with(|| ma.id_action()).take(n).collect(),
                ma,
            }
        }

        pub fn from_iter<I>(iter: I, ma: M) -> Self
        where
            I: IntoIterator<Item = M::X>,
            I::IntoIter: ExactSizeIterator,
        {
            let iter = iter.into_iter();
            let n = iter.len();
            let mut sum: Vec<_> = (iter::repeat_with(|| ma.id()).take(n))
                .chain(
                    iter.into_iter()
                        .chain(iter::repeat_with(|| ma.id()))
                        .take(n),
                )
                .collect();
            for i in (1..n).rev() {
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
            self.ma.apply_to_sum(&value, width, &mut self.sum[idx]);
            if idx < self.n {
                self.lazy[idx] = self.ma.combine_action(&value, &self.lazy[idx]);
            }
        }

        fn push_down(&mut self, width: u32, node: usize) {
            let value = unsafe { &*(&self.lazy[node] as *const _) };
            self.apply(node << 1, width, value);
            self.apply(node << 1 | 1, width, value);
            self.lazy[node] = self.ma.id_action();
        }

        fn push_range(&mut self, range: Range<usize>) {
            let Range { mut start, mut end } = range;
            start += self.n;
            end += self.n;

            let start_height = 1 + start.trailing_zeros();
            let end_height = 1 + end.trailing_zeros();
            for height in (start_height..=self.max_height).rev() {
                let width = 1 << height - 1;
                self.push_down(width, start >> height);
            }
            for height in (end_height..=self.max_height).rev().skip_while(|&height| {
                height >= start_height && end - 1 >> height == start >> height
            }) {
                let width = 1 << height - 1;
                self.push_down(width, end - 1 >> height);
            }
        }

        fn pull_up(&mut self, node: usize) {
            self.sum[node] = (self.ma).combine(&self.sum[node << 1], &self.sum[node << 1 | 1]);
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
            if M::IS_X_COMMUTATIVE {
                let mut result = self.ma.id();
                while start < end {
                    if start & 1 != 0 {
                        result = self.ma.combine(&result, &self.sum[start]);
                        start += 1;
                    }
                    if end & 1 != 0 {
                        end -= 1;
                        result = self.ma.combine(&result, &self.sum[end]);
                    }
                    start >>= 1;
                    end >>= 1;
                }
                result
            } else {
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
        }

        pub fn query_all(&mut self) -> &M::X {
            assert!(self.n.is_power_of_two());
            self.push_down(self.n as u32, 1);
            &self.sum[1]
        }
    }
}

#[derive(Clone, Debug)]
struct MinCount {
    min: i64,
    count: u32,
}

impl MinCount {
    fn zero(count: u32) -> Self {
        MinCount { min: 0, count }
    }
}

struct MinCountOp;

impl segtree_lazy::MonoidAction for MinCountOp {
    type X = MinCount;
    type F = i64;

    fn id(&self) -> Self::X {
        MinCount {
            min: std::i64::MAX,
            count: 0,
        }
    }

    fn combine(&self, a: &Self::X, b: &Self::X) -> Self::X {
        match a.min.cmp(&b.min) {
            Ordering::Less => a.clone(),
            Ordering::Greater => b.clone(),
            Ordering::Equal => MinCount {
                min: a.min,
                count: a.count + b.count,
            },
        }
    }

    fn id_action(&self) -> Self::F {
        0
    }

    fn combine_action(&self, a: &Self::F, b: &Self::F) -> Self::F {
        a + b
    }

    fn apply_to_sum(&self, f: &Self::F, _x_count: u32, x_sum: &mut Self::X) {
        x_sum.min += f;
    }
}

pub fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let w: i64 = input.value();
    let h: i64 = input.value();
    let k: usize = input.value();

    let mut events = vec![];
    let mut add_rect = |x1, x2, y1, y2| {
        events.push((y1, x1, x2, 1));
        events.push((y2, x1, x2, -1));
    };
    for _ in 0..k {
        let f: i64 = input.value();
        let c: i64 = input.value();

        let x1: i64 = input.value();
        let y1: i64 = input.value();
        let x2: i64 = input.value();
        let y2: i64 = input.value();

        let by = h / (c + 1);
        let cy1 = clamp(y1, 0, by);
        let cy2 = clamp(y2, 0, by);
        {
            let cx1 = clamp(f - x2, 0, f);
            let cx2 = clamp(f - x1, 0, f);
            for i in 0..=c {
                let dy = i * by;
                if i % 2 == 0 {
                    add_rect(cx1, cx2, cy1 + dy, cy2 + dy);
                } else {
                    add_rect(cx1, cx2, by - cy2 + dy, by - cy1 + dy);
                }
            }
        }
        {
            let cx1 = clamp(x1 + f, f, w);
            let cx2 = clamp(x2 + f, f, w);
            for i in 0..=c {
                let dy = i * by;
                if i % 2 == 0 {
                    add_rect(cx1, cx2, cy1 + dy, cy2 + dy);
                } else {
                    add_rect(cx1, cx2, by - cy2 + dy, by - cy1 + dy);
                }
            }
        }
    }

    events.sort_unstable();

    let (x_map, x_inv) = compress_coord(events.iter().flat_map(|&(_, x1, x2, _)| [x1, x2]));
    let x_bound = x_inv.len();

    let mut section = segtree_lazy::SegTree::from_iter(
        (0..(x_bound - 1).next_power_of_two()).map(|i| {
            if i < x_bound - 1 {
                let len = x_map[i + 1] - x_map[i];
                MinCount::zero(len as u32)
            } else {
                MinCountOp.id()
            }
        }),
        MinCountOp,
    );

    let mut area = 0;
    let mut y_prev = 0;
    let n_total = section.query_all().count;
    for (y, x1, x2, sign) in events {
        if y_prev < y {
            let s = section.query_all();
            let n_zeros = if s.min == 0 { s.count } else { 0 };
            let n_ones = n_total - n_zeros;
            area += n_ones as i64 * (y - y_prev) as i64;
            y_prev = y;
        }

        let x1 = x_inv[&x1] as usize;
        let x2 = x_inv[&x2] as usize;
        section.apply_range(x1..x2, sign as i64);
    }
    area = w * h - area;
    writeln!(output, "{}", area).unwrap();
}