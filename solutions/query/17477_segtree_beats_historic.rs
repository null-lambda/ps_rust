use std::{cmp::Ordering, io::Write};

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

    fn get_many<const N: usize, T>(xs: &mut [T], indices: [usize; N]) -> [&mut T; N] {
        if cfg!(debug_assertions) {
            let mut indices_sorted = indices;
            indices_sorted.sort_unstable();
            debug_assert!(indices_sorted.windows(2).all(|w| w[0] != w[1]));
            debug_assert!(indices_sorted.iter().all(|&i| i < xs.len()));
        }
        let ptr = xs.as_mut_ptr();
        unsafe { indices.map(|i| ptr.add(i).as_mut().unwrap()) }
    }

    pub trait NodeSpec {
        type V: Default;
        fn push_down(&self, parent: &mut Self::V, children: [&mut Self::V; 2], child_size: u32);
        fn pull_up(&self, parent: &mut Self::V, children: [&Self::V; 2]);
    }

    pub trait Action<M: NodeSpec>: Clone {
        fn try_apply_to_sum(&mut self, m: &M, x_count: u32, x_sum: &mut M::V) -> bool;
    }

    pub trait Reducer<M: NodeSpec> {
        type X;
        fn id(m: &M) -> Self::X;
        fn combine(m: &M, lhs: Self::X, rhs: Self::X) -> Self::X;
        fn extract(m: &M, x_sum: &M::V) -> Self::X;
    }

    pub struct SegTree<M: NodeSpec> {
        n: usize,
        max_height: u32,
        pub data: Vec<M::V>,
        pub op: M,
    }

    impl<M: NodeSpec> SegTree<M> {
        pub fn with_size(n: usize, ma: M) -> Self {
            Self {
                n,
                max_height: usize::BITS - n.leading_zeros(),
                data: iter::repeat_with(|| Default::default())
                    .take(2 * n)
                    .collect(),
                op: ma,
            }
        }
    }

    impl<M: NodeSpec> SegTree<M> {
        pub fn from_iter<I>(n: usize, iter: I, ma: M) -> Self
        where
            I: IntoIterator<Item = M::V>,
        {
            let sum: Vec<_> = (iter::repeat_with(|| Default::default()).take(n))
                .chain(
                    iter.into_iter()
                        .chain(iter::repeat_with(|| Default::default()))
                        .take(n),
                )
                .collect();
            let mut this = Self {
                n,
                max_height: usize::BITS - n.leading_zeros(),
                data: sum,
                op: ma,
            };
            for i in (1..n).rev() {
                this.pull_up(i);
            }
            this
        }

        fn apply(&mut self, idx: usize, width: u32, action: impl Action<M>) {
            let mut action = action.clone();
            let complete = action.try_apply_to_sum(&self.op, width, &mut self.data[idx]);
            if !complete {
                assert!(idx < self.n, "try_apply_to_sum should not fail for leaves");
                self.push_down(width, idx);
                self.apply(idx << 1, width >> 1, action.clone());
                self.apply(idx << 1 | 1, width >> 1, action);
                self.pull_up(idx);
            }
        }

        fn push_down(&mut self, width: u32, node: usize) {
            debug_assert!(node != 0);
            let [p, l, r] = get_many(&mut self.data, [node, node << 1, node << 1 | 1]);
            self.op.push_down(p, [l, r], width >> 1);
        }

        fn push_range(&mut self, range: Range<usize>) {
            let Range { mut start, mut end } = range;
            start += self.n;
            end += self.n;

            let start_height = 1 + start.trailing_zeros();
            let end_height = 1 + end.trailing_zeros();
            for height in (start_height..self.max_height).rev() {
                let width = 1 << height;
                self.push_down(width, start >> height);
            }
            for height in (end_height..self.max_height).rev().skip_while(|&height| {
                height >= start_height && end - 1 >> height == start >> height
            }) {
                let width = 1 << height;
                self.push_down(width, end - 1 >> height);
            }
        }

        fn pull_up(&mut self, node: usize) {
            debug_assert!(node != 0);
            let [p, l, r] = get_many(&mut self.data, [node, node << 1, node << 1 | 1]);
            self.op.pull_up(p, [l, r]);
        }

        pub fn apply_range(&mut self, range: Range<usize>, action: impl Action<M>) {
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
                    self.apply(start, width, action.clone());
                    start += 1;
                    pull_start = true;
                }
                if end & 1 != 0 {
                    self.apply(end - 1, width, action.clone());
                    pull_end = true;
                }
                start >>= 1;
                end >>= 1;
                width <<= 1;
            }
            start -= 1;
            while end > 0 {
                if pull_start && start != 0 {
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

        pub fn query_range<R: Reducer<M>>(&mut self, range: Range<usize>) -> R::X {
            let Range { mut start, mut end } = range;

            self.push_range(range);
            start += self.n;
            end += self.n;
            let (mut result_left, mut result_right) = (R::id(&self.op), R::id(&self.op));
            while start < end {
                if start & 1 != 0 {
                    result_left = R::combine(
                        &self.op,
                        result_left,
                        R::extract(&self.op, &self.data[start]),
                    );
                }
                if end & 1 != 0 {
                    result_right = R::combine(
                        &self.op,
                        R::extract(&self.op, &self.data[end - 1]),
                        result_right,
                    );
                    end -= 1;
                }
                start = (start + 1) >> 1;
                end >>= 1;
            }

            R::combine(&self.op, result_left, result_right)
        }
    }
}

const INF: i64 = 1 << 56;
const NEG_INF: i64 = -INF;

#[derive(Debug)]
struct NodeData {
    max: i64, // Simultaneously, a lazy tag
    sub_max: i64,
    max_count: u32,

    min: i64, // Simultaneously, a lazy tag
    sub_min: i64,
    min_count: u32,

    lazy_add: i64,

    n_mod: u64,
    lazy_addop_count: u32,
    lazy_maxop_count: u32,
    lazy_minop_count: u32,
}

impl Default for NodeData {
    fn default() -> Self {
        Self {
            max: NEG_INF + 1,
            sub_max: NEG_INF,
            max_count: 0,

            min: INF - 1,
            sub_min: INF,
            min_count: 0,

            lazy_add: 0,

            n_mod: 0,
            lazy_addop_count: 0,
            lazy_maxop_count: 0,
            lazy_minop_count: 0,
        }
    }
}

impl NodeData {
    fn singleton(value: i32) -> Self {
        Self {
            max: value as i64,
            sub_max: NEG_INF,
            max_count: 1,

            min: value as i64,
            sub_min: INF,
            min_count: 1,

            lazy_add: 0,

            n_mod: 0,
            lazy_addop_count: 0,
            lazy_maxop_count: 0,
            lazy_minop_count: 0,
        }
    }
}

struct Op;

impl NodeSpec for Op {
    type V = NodeData;

    fn push_down(&self, parent: &mut Self::V, children: [&mut Self::V; 2], child_size: u32) {
        for i in 0..2 {
            let res = (parent.lazy_addop_count == 0
                || AddAction(parent.lazy_add, parent.lazy_addop_count).try_apply_to_sum(
                    self,
                    child_size,
                    children[i],
                ))
                && (parent.lazy_minop_count == 0
                    || MinAction(parent.max, parent.lazy_minop_count).try_apply_to_sum(
                        self,
                        child_size,
                        children[i],
                    ))
                && (parent.lazy_maxop_count == 0
                    || MaxAction(parent.min, parent.lazy_maxop_count).try_apply_to_sum(
                        self,
                        child_size,
                        children[i],
                    ));
            debug_assert!(res);
        }
        parent.lazy_add = 0;
        parent.lazy_addop_count = 0;
        parent.lazy_maxop_count = 0;
        parent.lazy_minop_count = 0;
    }

    fn pull_up(&self, parent: &mut Self::V, children: [&Self::V; 2]) {
        debug_assert_eq!(parent.lazy_add, 0);
        debug_assert_eq!(parent.lazy_addop_count, 0);
        debug_assert_eq!(parent.lazy_maxop_count, 0);
        debug_assert_eq!(parent.lazy_minop_count, 0);

        let [left, right] = children;

        match left.max.cmp(&right.max) {
            Ordering::Equal => {
                parent.max = left.max;
                parent.max_count = left.max_count + right.max_count;
                parent.sub_max = left.sub_max.max(right.sub_max);
            }
            Ordering::Less => {
                parent.max = right.max;
                parent.max_count = right.max_count;
                parent.sub_max = left.max.max(right.sub_max);
            }
            Ordering::Greater => {
                parent.max = left.max;
                parent.max_count = left.max_count;
                parent.sub_max = left.sub_max.max(right.max);
            }
        }

        match left.min.cmp(&right.min) {
            Ordering::Equal => {
                parent.min = left.min;
                parent.min_count = left.min_count + right.min_count;
                parent.sub_min = left.sub_min.min(right.sub_min);
            }
            Ordering::Less => {
                parent.min = left.min;
                parent.min_count = left.min_count;
                parent.sub_min = left.sub_min.min(right.min);
            }
            Ordering::Greater => {
                parent.min = right.min;
                parent.min_count = right.min_count;
                parent.sub_min = left.min.min(right.sub_min);
            }
        }

        parent.n_mod = left.n_mod + right.n_mod;
    }
}

#[derive(Clone)]
struct AddAction(i64, u32);

impl Action<Op> for AddAction {
    fn try_apply_to_sum(&mut self, _: &Op, x_count: u32, x_sum: &mut <Op as NodeSpec>::V) -> bool {
        let AddAction(f, mult) = *self;
        x_sum.max += f;
        x_sum.sub_max += f;

        x_sum.min += f;
        x_sum.sub_min += f;

        x_sum.lazy_add += f;

        x_sum.n_mod += x_count as u64 * mult as u64;
        x_sum.lazy_addop_count += mult;

        true
    }
}

#[derive(Clone)]
struct MinAction(i64, u32);

impl Action<Op> for MinAction {
    fn try_apply_to_sum(&mut self, _: &Op, x_count: u32, x_sum: &mut NodeData) -> bool {
        let Self(f, mult) = *self;

        if f >= x_sum.max {
            return true;
        }

        if x_sum.min == x_sum.max {
            // Ensure that accumulated min queries are disjoint with max queries,
            // by converting some of them into add queries.
            let delta = (f - x_sum.min).min(0);
            if delta != 0 {
                // The following operations are semantically equivalent to:
                // AddAction(delta, mult).try_apply_to_sum(m, x_count, x_sum);
                x_sum.max = f;
                x_sum.sub_max = NEG_INF;
                x_sum.max_count = x_count;

                x_sum.min = f;
                x_sum.sub_min = INF;
                x_sum.min_count = x_count;

                x_sum.lazy_add += delta;
                x_sum.lazy_addop_count += mult;
                x_sum.n_mod += x_count as u64 * mult as u64;
            }
            return true;
        }

        if f > x_sum.sub_max {
            x_sum.max = f;

            x_sum.sub_min = x_sum.sub_min.min(f);

            x_sum.n_mod += x_sum.max_count as u64 * mult as u64;
            x_sum.lazy_minop_count += mult;

            return true;
        }

        false
    }
}

#[derive(Clone)]
struct MaxAction(i64, u32);

impl Action<Op> for MaxAction {
    fn try_apply_to_sum(&mut self, _: &Op, x_count: u32, x_sum: &mut NodeData) -> bool {
        let Self(f, mult) = *self;

        if f <= x_sum.min {
            return true;
        }

        if x_sum.min == x_sum.max {
            // Ensure that accumulated min queries are disjoint with max queries,
            // by converting some of them into add queries.
            let delta = (f - x_sum.max).max(0);
            if delta != 0 {
                // The following operations are semantically equivalent to:
                // AddAction(delta, mult).try_apply_to_sum(m, x_count, x_sum);
                x_sum.max = f;
                x_sum.sub_max = NEG_INF;
                x_sum.max_count = x_count;

                x_sum.min = f;
                x_sum.sub_min = INF;
                x_sum.min_count = x_count;

                x_sum.lazy_add += delta;
                x_sum.lazy_addop_count += mult;
                x_sum.n_mod += x_count as u64 * mult as u64;
            }
            return true;
        }

        if f < x_sum.sub_min {
            x_sum.min = f;

            x_sum.sub_max = x_sum.sub_max.max(f);

            x_sum.n_mod += x_sum.min_count as u64 * mult as u64;
            x_sum.lazy_maxop_count += mult;

            return true;
        }

        false
    }
}

struct NModQuery;

impl Reducer<Op> for NModQuery {
    type X = u64;
    fn id(_: &Op) -> Self::X {
        0
    }
    fn combine(_: &Op, lhs: Self::X, rhs: Self::X) -> Self::X {
        lhs + rhs
    }
    fn extract(_: &Op, x_sum: &NodeData) -> Self::X {
        x_sum.n_mod
    }
}

struct MaxQuery;

impl Reducer<Op> for MaxQuery {
    type X = i64;
    fn id(_: &Op) -> Self::X {
        NEG_INF
    }
    fn combine(_: &Op, lhs: Self::X, rhs: Self::X) -> Self::X {
        lhs.max(rhs)
    }
    fn extract(_: &Op, x_sum: &NodeData) -> Self::X {
        x_sum.max
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n = input.value();
    let mut xs = SegTree::from_iter(n, (0..n).map(|_| NodeData::singleton(input.value())), Op);

    let q = input.value();
    for _ in 0..q {
        let cmd = input.token();
        let l = input.value::<usize>() - 1;
        let r = input.value::<usize>() - 1;
        match cmd {
            "1" => {
                let x = input.value();
                if x != 0 {
                    xs.apply_range(l..r + 1, AddAction(x, 1));
                }
            }

            "2" => {
                let y = input.value();
                xs.apply_range(l..r + 1, MaxAction(y, 1));
            }
            "3" => {
                let y = input.value();
                xs.apply_range(l..r + 1, MinAction(y, 1));
            }
            "4" => {
                let ans = xs.query_range::<NModQuery>(l..r + 1);
                writeln!(output, "{}", ans).unwrap();
            }
            _ => panic!(),
        }
    }
}
