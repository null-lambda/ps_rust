use std::io::Write;

use segtree::*;

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
        type V;
        fn id(&self) -> Self::V;
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

    pub struct LazySegTree<M: NodeSpec> {
        n: usize,
        max_height: u32,
        pub data: Vec<M::V>,
        pub op: M,
    }

    impl<M: NodeSpec> LazySegTree<M> {
        pub fn with_size(n: usize, ma: M) -> Self {
            Self {
                n,
                max_height: usize::BITS - n.leading_zeros(),
                data: iter::repeat_with(|| ma.id()).take(2 * n).collect(),
                op: ma,
            }
        }
    }

    impl<M: NodeSpec> LazySegTree<M> {
        pub fn from_iter<I>(n: usize, iter: I, ma: M) -> Self
        where
            I: IntoIterator<Item = M::V>,
        {
            let sum: Vec<_> = (iter::repeat_with(|| ma.id()).take(n))
                .chain(
                    iter.into_iter()
                        .chain(iter::repeat_with(|| ma.id()))
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

struct NodeData {
    min: i64,
    max: i64,
    sum: i64,
    lazy_clear: bool,
    lazy_add: i64,
}

impl NodeData {
    fn singleton(value: i64) -> Self {
        Self {
            min: value,
            max: value,
            sum: value,
            lazy_clear: false,
            lazy_add: 0,
        }
    }
}

struct Spec;

impl NodeSpec for Spec {
    type V = NodeData;

    fn id(&self) -> Self::V {
        NodeData {
            min: 0,
            max: 0,
            sum: 0,
            lazy_clear: false,
            lazy_add: 0,
        }
    }

    fn push_down(&self, parent: &mut Self::V, mut children: [&mut Self::V; 2], child_count: u32) {
        debug_assert!(child_count > 0);
        if parent.lazy_clear {
            for child in &mut children {
                child.min = 0;
                child.max = 0;
                child.sum = 0;
                child.lazy_clear = true;
                child.lazy_add = 0;
            }
            parent.lazy_clear = false;
        }

        if parent.lazy_add != 0 {
            for child in &mut children {
                child.sum += parent.lazy_add * child_count as i64;
                child.min += parent.lazy_add;
                child.max += parent.lazy_add;
                child.lazy_add += parent.lazy_add;
            }
            parent.lazy_add = 0;
        }
    }

    fn pull_up(&self, parent: &mut Self::V, [left, right]: [&Self::V; 2]) {
        debug_assert!(parent.lazy_clear == false);
        debug_assert!(parent.lazy_add == 0);
        parent.min = left.min.min(right.min);
        parent.max = left.max.max(right.max);
        parent.sum = left.sum + right.sum;
    }
}

#[derive(Clone)]
struct AddAction(i64);

impl Action<Spec> for AddAction {
    fn try_apply_to_sum(&mut self, _: &Spec, x_count: u32, x_sum: &mut NodeData) -> bool {
        x_sum.min += self.0;
        x_sum.max += self.0;
        x_sum.sum += self.0 * x_count as i64;
        x_sum.lazy_add += self.0;
        true
    }
}

fn div_floor(x: i64, y: i64) -> i64 {
    x / y - ((x ^ y) < 0 && x % y != 0) as i64
}

#[derive(Clone)]
struct DivAction(i64);

impl Action<Spec> for DivAction {
    fn try_apply_to_sum(&mut self, _: &Spec, x_count: u32, x_sum: &mut NodeData) -> bool {
        debug_assert!(self.0 >= 2);
        let f = |x: i64| div_floor(x, self.0);
        let y1 = f(x_sum.min);
        let y2 = f(x_sum.max);
        if y1 == y2 {
            x_sum.min = y1;
            x_sum.max = y1;
            x_sum.sum = y1 * x_count as i64;
            x_sum.lazy_clear = true;
            x_sum.lazy_add = y1;
            true
        } else if x_sum.min + 1 == x_sum.max {
            let delta = y1 - x_sum.min;
            x_sum.min = y1;
            x_sum.max = y2;
            x_sum.sum += delta * x_count as i64;
            x_sum.lazy_add += delta;
            true
        } else {
            false
        }
    }
}

struct MinQuery;

impl Reducer<Spec> for MinQuery {
    type X = i64;
    fn id(_: &Spec) -> i64 {
        i64::MAX
    }
    fn combine(_: &Spec, lhs: i64, rhs: i64) -> i64 {
        lhs.min(rhs)
    }
    fn extract(_: &Spec, x_sum: &NodeData) -> i64 {
        x_sum.min
    }
}

struct SumQuery;

impl Reducer<Spec> for SumQuery {
    type X = i64;
    fn id(_: &Spec) -> i64 {
        0
    }
    fn combine(_: &Spec, lhs: i64, rhs: i64) -> i64 {
        lhs + rhs
    }
    fn extract(_: &Spec, x_sum: &NodeData) -> i64 {
        x_sum.sum
    }
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let q: usize = input.value();
    let mut segtree =
        LazySegTree::from_iter(n, (0..n).map(|_| NodeData::singleton(input.value())), Spec);

    for _ in 0..q {
        let cmd = input.token();
        let s: usize = input.value();
        let t: usize = input.value();
        match cmd {
            "1" => {
                let c = input.value();
                segtree.apply_range(s..t + 1, &AddAction(c));
            }
            "2" => {
                let d = input.value();
                segtree.apply_range(s..t + 1, &DivAction(d));
            }
            "3" => {
                let min = segtree.query_range::<MinQuery>(s..t + 1);
                writeln!(output, "{}", min).unwrap();
            }
            "4" => {
                let sum = segtree.query_range::<SumQuery>(s..t + 1);
                writeln!(output, "{}", sum).unwrap();
            }
            _ => panic!(),
        }
    }
}
