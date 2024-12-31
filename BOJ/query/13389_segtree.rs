use std::{collections::BTreeMap, io::Write};

use segtree_lazy::{MonoidAction, SegTree};

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

use std::{collections::HashMap, hash::Hash};

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

// chunk_by in std >= 1.77
fn group_by<T, P, F>(xs: &[T], mut pred: P, mut f: F)
where
    P: FnMut(&T, &T) -> bool,
    F: FnMut(&[T]),
{
    let mut i = 0;
    while i < xs.len() {
        let mut j = i + 1;
        while j < xs.len() && pred(&xs[j - 1], &xs[j]) {
            j += 1;
        }
        f(&xs[i..j]);
        i = j;
    }
}

pub mod segtree_lazy {
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

    pub struct SegTree<M: MonoidAction> {
        n: usize,
        max_height: u32,
        pub sum: Vec<M::X>,
        pub lazy: Vec<M::F>,
        pub ma: M,
    }

    impl<M: MonoidAction> SegTree<M> {
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
            self.sum[idx] = self.ma.apply_to_sum(&value, width, &self.sum[idx]);
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

struct RangeMax {
    neg_inf: i64,
    inf: i64,
}

#[derive(Clone, Copy)]
struct ApplyMin(i64);

impl MonoidAction for RangeMax {
    type X = i64;
    type F = ApplyMin;

    fn id(&self) -> Self::X {
        self.neg_inf
    }

    fn combine(&self, lhs: &Self::X, rhs: &Self::X) -> Self::X {
        (*lhs).max(*rhs)
    }

    fn id_action(&self) -> Self::F {
        ApplyMin(self.inf)
    }

    fn combine_action(&self, lhs: &Self::F, rhs: &Self::F) -> Self::F {
        ApplyMin(lhs.0.min(rhs.0))
    }

    fn apply_to_sum(&self, f: &Self::F, _x_count: u32, x_sum: &Self::X) -> Self::X {
        f.0.min(*x_sum)
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: i64 = input.value();
    let m: usize = input.value();

    let mut queries = vec![];
    for _ in 0..m {
        let l: i64 = input.value();
        let r: i64 = input.value();
        let a: i64 = input.value();
        queries.push((a, l, r));
    }
    queries.sort_unstable();

    let mut queries_joined = vec![];
    group_by(
        &queries,
        |t, s| t.0 == s.0,
        |group| {
            let (a, mut l_min, mut r_min) = group[0];
            let (mut l_max, mut r_max) = (l_min, r_min);
            for (_, l, r) in &group[1..] {
                l_min = l_min.min(*l);
                l_max = l_max.max(*l);
                r_min = r_min.min(*r);
                r_max = r_max.max(*r);
            }
            queries_joined.push((a, l_min, l_max, r_min, r_max));
        },
    );
    let queries = queries_joined;

    let mut xs = vec![];
    xs.push(-1);
    xs.push(0);
    xs.push(1);
    xs.push(2);
    xs.push(n - 1);
    xs.push(n);
    xs.push(n + 1);
    xs.push(n + 2);
    for &(_, l_min, l_max, r_min, r_max) in &queries {
        for p in [l_min, l_max, r_min, r_max] {
            xs.push(p - 1);
            xs.push(p);
            xs.push(p + 1);
        }
    }

    let (x_map, x_inv) = compress_coord(xs);
    let x_bound = x_inv.len();

    let inf = 2_000_000_000;
    let neg_inf = 0;
    let mut seq = SegTree::from_iter(
        x_bound,
        (0..x_bound).map(|_| inf),
        RangeMax { neg_inf, inf },
    );
    let res = 'outer: {
        for (a, l_min, l_max, r_min, r_max) in queries {
            if l_max > r_min {
                break 'outer false;
            }
            if seq.query_range(x_inv[&l_max] as usize..x_inv[&r_min] as usize + 1) != inf {
                break 'outer false;
            }
            seq.apply_range(
                x_inv[&l_min] as usize..x_inv[&r_max] as usize + 1,
                ApplyMin(a),
            );
        }

        let mut counter = BTreeMap::<i64, i64>::new();
        for i in 0..x_bound as usize {
            let a = seq.query_range(i..i + 1);
            if a != inf {
                *counter.entry(a).or_default() += x_map[i + 1] - x_map[i];
            }
        }

        let mut freq_acc = 0;
        for (a, f) in counter {
            freq_acc += f;
            if freq_acc > a {
                break 'outer false;
            }
        }

        true
    };

    writeln!(output, "{}", res as u8).unwrap();
}
