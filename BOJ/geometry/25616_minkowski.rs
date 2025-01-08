use std::io::Write;
use std::{collections::HashMap, hash::Hash};

use segtree_lazy::*;

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

pub mod segtree_lazy {
    use std::{iter, ops::Range};

    pub trait MonoidAction {
        type X;
        type F;
        const IS_X_COMMUTATIVE: bool = false;
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

        // The following two lines are equivalent:
        // - partition_point(0, n, |i| pred(segtree.query_range(0..i+1)));
        // - segtree.partition_point_prefix(|prefix| pred(prefix));
        pub fn partition_point_prefix(&mut self, mut pred: impl FnMut(&M::X) -> bool) -> usize {
            assert!(self.n >= 1 && self.n.is_power_of_two());

            let mut u = 1;
            let mut width = self.n as u32;
            let mut prefix = self.ma.id();

            while u < self.n {
                width >>= 1;
                self.push_down(width, u);

                let new_prefix = self.ma.combine(&prefix, &self.sum[u << 1]);
                u = if pred(&new_prefix) {
                    prefix = new_prefix;
                    u << 1 | 1
                } else {
                    u << 1
                };
            }

            let idx = u - self.n;
            if pred(&self.ma.combine(&prefix, &self.sum[u])) {
                idx + 1
            } else {
                idx
            }
        }
    }
}

// min-plus algebra
struct MinPlus;

impl MonoidAction for MinPlus {
    type X = i32;
    type F = i32;

    fn id(&self) -> Self::X {
        i32::MAX / 4
    }

    fn combine(&self, lhs: &Self::X, rhs: &Self::X) -> Self::X {
        (*lhs).min(*rhs)
    }

    fn id_action(&self) -> Self::F {
        0
    }

    fn combine_action(&self, lhs: &Self::F, rhs: &Self::F) -> Self::F {
        lhs + rhs
    }

    fn apply_to_sum(&self, f: &Self::F, _x_count: u32, x_sum: &mut Self::X) {
        *x_sum += f;
    }
}

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

pub fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let k: i64 = input.value();

    let ps: Vec<[i64; 2]> = (0..n)
        .map(|_| {
            let x: i64 = input.value();
            let y: i64 = input.value();
            [x - y, x + y] // Transform Minkowski to Chebyshev
        })
        .collect();

    let norm = |p: [i64; 2]| p[0].abs().max(p[1].abs());
    let min_norm = ps.iter().map(|&p| norm(p)).min().unwrap();
    let ans = if min_norm >= k {
        min_norm - k
    } else {
        let mut event_groups = [vec![], vec![]];
        for g in 0..2 {
            event_groups[g].push((0, [0, 0], 0));
        }

        let scale = |x: i64| x << 2;
        let inv_scale = |x: i64| (x + 2) >> 2; // Round to nearest integer

        for [x, y] in ps {
            let xs = [scale(x - k) + 1, scale(x + k) - 1];
            let ys = [scale(y - k) + 1, scale(y + k) - 1];

            // Split the cartesian plane along y-axis
            debug_assert!(xs[0] <= xs[1]);
            if xs[1] <= 0 {
                event_groups[0].push((ys[0], [-xs[1], -xs[0]], 1));
                event_groups[0].push((ys[1], [-xs[1], -xs[0]], -1));
            } else if 0 <= xs[0] {
                event_groups[1].push((ys[0], [xs[0], xs[1]], 1));
                event_groups[1].push((ys[1], [xs[0], xs[1]], -1));
            } else {
                event_groups[0].push((ys[0], [0, -xs[0]], 1));
                event_groups[0].push((ys[1], [0, -xs[0]], -1));
                event_groups[1].push((ys[0], [0, xs[1]], 1));
                event_groups[1].push((ys[1], [0, xs[1]], -1));
            }
        }

        let mut ans = i64::MAX;
        for mut events in event_groups {
            const X_MAX: i64 = 1 << 40;
            events.sort_unstable();
            let (x_map, x_inv) =
                compress_coord(events.iter().flat_map(|&(_, xs, _)| xs).chain([0, X_MAX]));
            let x_bound = x_inv.len();

            let mut events = events.into_iter().peekable();
            let mut section =
                SegTree::from_iter((0..(x_bound - 1).next_power_of_two()).map(|_| 0), MinPlus);
            let mut update_ans = |section: &mut SegTree<MinPlus>, y: i64| {
                let i = section.partition_point_prefix(|&prefix| prefix > 0);
                ans = ans.min(inv_scale(y.abs().max(x_map[i])));
            };
            while let Some(&(y, ..)) = events.peek() {
                update_ans(&mut section, y);
                while let Some((_, xs, delta)) = events.next_if(|&(y_next, ..)| y_next == y) {
                    assert!(xs[0] <= xs[1]);
                    section.apply_range(x_inv[&xs[0]] as usize..x_inv[&xs[1]] as usize, delta);
                }
                update_ans(&mut section, y);
            }
        }
        ans
    };
    writeln!(output, "{}", ans).unwrap();
}
