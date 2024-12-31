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

fn isqrt(x: i64) -> i64 {
    debug_assert!(0 <= x && x <= 4_503_599_761_588_223i64);
    (x as f64).sqrt() as i64
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
        unsafe { std::array::from_fn(|i| ptr.add(indices[i]).as_mut().unwrap()) }
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

        fn apply(&mut self, idx: usize, width: u32, action: &impl Action<M>) {
            let mut action = action.clone();
            let complete = action.try_apply_to_sum(&self.op, width, &mut self.data[idx]);
            if !complete {
                assert!(idx < self.n, "try_apply_to_sum should not fail for leaves");
                self.push_down(width, idx);
                self.apply(idx << 1, width >> 1, &action);
                self.apply(idx << 1 | 1, width >> 1, &action);
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

        pub fn apply_range(&mut self, range: Range<usize>, action: &impl Action<M>) {
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
                    self.apply(start, width, action);
                    start += 1;
                    pull_start = true;
                }
                if end & 1 != 0 {
                    self.apply(end - 1, width, action);
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

#[derive(Clone, Default)]
struct NodeData {
    min: i64,
    max: i64,
    sum: i64,
    lazy_add: i64,
}

impl NodeData {
    fn singleton(value: i64) -> Self {
        Self {
            min: value,
            max: value,
            sum: value,
            lazy_add: 0,
        }
    }
}

const INF: i64 = 1 << 56;
const NEG_INF: i64 = -INF;

struct Spec;

impl NodeSpec for Spec {
    type V = NodeData;

    fn push_down(&self, parent: &mut Self::V, children: [&mut Self::V; 2], child_size: u32) {
        if parent.lazy_add != 0 {
            for child in children {
                child.min += parent.lazy_add;
                child.max += parent.lazy_add;
                child.sum += parent.lazy_add * child_size as i64;
                child.lazy_add += parent.lazy_add;
            }
            parent.lazy_add = 0;
        }
    }

    fn pull_up(&self, parent: &mut Self::V, children: [&Self::V; 2]) {
        debug_assert_eq!(parent.lazy_add, 0);
        let [left, right] = children;
        parent.min = left.min.min(right.min);
        parent.max = left.max.max(right.max);
        parent.sum = left.sum + right.sum;
    }
}

#[derive(Clone)]
struct AddAction(i64);

impl Action<Spec> for AddAction {
    fn try_apply_to_sum(
        &mut self,
        _: &Spec,
        x_count: u32,
        x_sum: &mut <Spec as NodeSpec>::V,
    ) -> bool {
        x_sum.min += self.0;
        x_sum.max += self.0;
        x_sum.sum += self.0 * x_count as i64;
        x_sum.lazy_add += self.0;
        true
    }
}

#[derive(Clone)]
struct SqrtAction;

impl Action<Spec> for SqrtAction {
    fn try_apply_to_sum(
        &mut self,
        _: &Spec,
        x_count: u32,
        x_sum: &mut <Spec as NodeSpec>::V,
    ) -> bool {
        let s1 = isqrt(x_sum.min);
        let s2 = isqrt(x_sum.max);
        let d1 = s1 - x_sum.min;
        let d2 = s2 - x_sum.max;

        if d1 == d2 {
            x_sum.min = s1;
            x_sum.max = s2;
            x_sum.sum += d1 * x_count as i64;
            x_sum.lazy_add += d1;
            return true;
        }
        false
    }
}

struct SumQuery;

impl Reducer<Spec> for SumQuery {
    type X = i64;
    fn id(_: &Spec) -> Self::X {
        0
    }
    fn combine(_: &Spec, lhs: Self::X, rhs: Self::X) -> Self::X {
        lhs + rhs
    }
    fn extract(_: &Spec, x_sum: &<Spec as NodeSpec>::V) -> Self::X {
        x_sum.sum
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    for _ in 0..input.value() {
        let n = input.value();
        let q = input.value();
        let mut xs =
            SegTree::from_iter(n, (0..n).map(|_| NodeData::singleton(input.value())), Spec);

        for _ in 0..q {
            let cmd = input.token();
            let l = input.value::<usize>() - 1;
            let r = input.value::<usize>() - 1;
            match cmd {
                "3" => {
                    let x: i64 = input.value();
                    xs.apply_range(l..r + 1, &AddAction(x));
                }
                "1" => {
                    xs.apply_range(l..r + 1, &SqrtAction);
                }
                "2" => {
                    let ans = xs.query_range::<SumQuery>(l..r + 1);
                    writeln!(output, "{}", ans).unwrap();
                }
                _ => panic!(),
            }
        }
    }
}
