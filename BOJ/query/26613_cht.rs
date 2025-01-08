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

pub mod cht {
    use core::{num::NonZeroU32, ops::RangeInclusive};

    // max Li-Chao tree of lines
    // TODO: add segment insertion
    type V = i64;
    type K = i64;
    const NEG_INF: V = i64::MIN;

    #[derive(Clone)]
    pub struct Line {
        slope: V,
        intercept: V,
    }

    impl Line {
        pub fn new(slope: V, intercept: V) -> Self {
            Self { slope, intercept }
        }

        fn eval(&self, x: V) -> K {
            self.slope * x + self.intercept
        }

        fn bottom() -> Self {
            Self {
                slope: 0,
                intercept: NEG_INF,
            }
        }
    }

    #[derive(Clone)]
    struct NodeRef(NonZeroU32);

    #[derive(Clone)]
    struct Node {
        children: [Option<NodeRef>; 2],
        line: Line,
    }

    impl Node {
        fn new() -> Self {
            Self {
                children: [None, None],
                line: Line::bottom(),
            }
        }
    }

    #[derive(Clone)]
    pub struct LiChaoTree {
        pool: Vec<Node>,
        interval: RangeInclusive<V>,
    }

    impl LiChaoTree {
        pub fn new(interval: RangeInclusive<V>) -> Self {
            Self {
                pool: vec![Node::new()],
                interval,
            }
        }

        fn alloc(&mut self, node: Node) -> NodeRef {
            let index = self.pool.len();
            self.pool.push(node);
            NodeRef(NonZeroU32::new(index as u32).unwrap())
        }

        // pub fn insert_segment(&mut self, interval: (V, V), mut line: Line) {
        //     unimplemented!()
        // }

        pub fn insert(&mut self, mut line: Line) {
            let mut u = 0;
            let (mut x_left, mut x_right) = self.interval.clone().into_inner();
            loop {
                let x_mid = (x_left + x_right) / 2;
                let top = &mut self.pool[u].line;
                if top.eval(x_mid) < line.eval(x_mid) {
                    std::mem::swap(top, &mut line);
                }
                u = if top.eval(x_left) < line.eval(x_left) {
                    x_right = x_mid;
                    match self.pool[u].children[0] {
                        Some(ref c) => c.0.get() as usize,
                        None => {
                            let c = self.alloc(Node::new());
                            self.pool[u].children[0] = Some(c.clone());
                            c.0.get() as usize
                        }
                    }
                } else if top.eval(x_right) < line.eval(x_right) {
                    x_left = x_mid + 1;
                    match self.pool[u].children[1] {
                        Some(ref c) => c.0.get() as usize,
                        None => {
                            let c = self.alloc(Node::new());
                            self.pool[u].children[1] = Some(c.clone());
                            c.0.get() as usize
                        }
                    }
                } else {
                    return;
                };
            }
        }

        pub fn eval(&self, x: V) -> K {
            debug_assert!(self.interval.contains(&x));
            let mut u = 0;
            let mut result = self.pool[u].line.eval(x);
            let (mut x_left, mut x_right) = self.interval.clone().into_inner();
            loop {
                let x_mid = (x_left + x_right) / 2;
                let branch = if x <= x_mid {
                    x_right = x_mid;
                    0
                } else {
                    x_left = x_mid + 1;
                    1
                };
                if let Some(c) = &self.pool[u].children[branch] {
                    u = c.0.get() as usize;
                } else {
                    return result;
                }
                result = result.max(self.pool[u].line.eval(x));
            }
        }
    }
}

pub mod segtree {
    use std::ops::Range;

    pub trait Monoid {
        type X;
        const IS_COMMUTATIVE: bool = false;
        fn id(&self) -> Self::X;
        fn op(&self, a: &Self::X, b: &Self::X) -> Self::X;
    }

    #[derive(Debug)]
    pub struct SegTree<M>
    where
        M: Monoid,
    {
        n: usize,
        sum: Vec<M::X>,
        monoid: M,
    }

    impl<M: Monoid> SegTree<M> {
        pub fn with_size(n: usize, monoid: M) -> Self {
            Self {
                n,
                sum: (0..2 * n).map(|_| monoid.id()).collect(),
                monoid,
            }
        }

        pub fn from_iter<I>(iter: I, monoid: M) -> Self
        where
            I: IntoIterator<Item = M::X>,
            I::IntoIter: ExactSizeIterator<Item = M::X>,
        {
            let iter = iter.into_iter();
            let n = iter.len();
            let mut sum: Vec<_> = (0..n).map(|_| monoid.id()).chain(iter).collect();
            for i in (0..n).rev() {
                sum[i] = monoid.op(&sum[i << 1], &sum[i << 1 | 1]);
            }
            Self { n, sum, monoid }
        }

        pub fn modify(&mut self, mut idx: usize, f: impl FnOnce(&mut M::X)) {
            debug_assert!(idx < self.n);
            idx += self.n;
            f(&mut self.sum[idx]);
            while idx > 1 {
                idx >>= 1;
                self.sum[idx] = self.monoid.op(&self.sum[idx << 1], &self.sum[idx << 1 | 1]);
            }
        }

        pub fn get(&self, idx: usize) -> &M::X {
            &self.sum[idx + self.n]
        }

        pub fn mapped_sum_range<N: Monoid>(
            &self,
            range: Range<usize>,
            codomain: &N,
            morphism: impl Fn(&M::X) -> N::X,
        ) -> N::X {
            let Range { mut start, mut end } = range;
            if start >= end {
                return codomain.id();
            }
            debug_assert!(start < self.n && end <= self.n);
            start += self.n;
            end += self.n;

            if N::IS_COMMUTATIVE {
                let mut result = codomain.id();
                while start < end {
                    if start & 1 != 0 {
                        result = codomain.op(&result, &morphism(&self.sum[start]));
                    }
                    if end & 1 != 0 {
                        result = codomain.op(&morphism(&self.sum[end - 1]), &result);
                    }
                    start = (start + 1) >> 1;
                    end >>= 1;
                }
                result
            } else {
                let (mut result_left, mut result_right) = (codomain.id(), codomain.id());
                while start < end {
                    if start & 1 != 0 {
                        result_left = codomain.op(&result_left, &morphism(&self.sum[start]));
                    }
                    if end & 1 != 0 {
                        result_right = codomain.op(&morphism(&self.sum[end - 1]), &result_right);
                    }
                    start = (start + 1) >> 1;
                    end >>= 1;
                }
                codomain.op(&result_left, &result_right)
            }
        }

        pub fn sum_all(&self) -> &M::X {
            assert!(self.n.is_power_of_two());
            &self.sum[1]
        }
    }

    impl<M: Monoid> SegTree<M>
    where
        M::X: Clone,
    {
        pub fn sum_range(&self, range: Range<usize>) -> M::X {
            self.mapped_sum_range(range, &self.monoid, |x| x.clone())
        }
    }
}

const X_MIN: i64 = 0;
const X_MAX: i64 = 100_000_000;

#[derive(Clone)]
struct NodeData {
    hull: cht::LiChaoTree,
    lines: Vec<cht::Line>,
}

impl NodeData {
    fn singleton(line: cht::Line) -> Self {
        let mut hull = cht::LiChaoTree::new(X_MIN..=X_MAX);
        hull.insert(line.clone());
        Self {
            hull,
            lines: vec![line],
        }
    }
}

struct HullMonoid;

impl Monoid for HullMonoid {
    type X = NodeData;

    fn id(&self) -> Self::X {
        NodeData {
            hull: cht::LiChaoTree::new(X_MIN..=X_MAX),
            lines: vec![],
        }
    }

    fn op(&self, a: &Self::X, b: &Self::X) -> Self::X {
        let mut res = a.clone();
        for line in &b.lines {
            res.hull.insert(line.clone());
            res.lines.push(line.clone());
        }
        res
    }
}

struct MaxOp;

impl Monoid for MaxOp {
    type X = i64;

    fn id(&self) -> Self::X {
        i64::MIN
    }

    fn op(&self, a: &Self::X, b: &Self::X) -> Self::X {
        (*a).max(*b)
    }
}

fn partition_point<P>(mut left: i64, mut right: i64, mut pred: P) -> i64
where
    P: FnMut(i64) -> bool,
{
    while left < right {
        let mid = left + (right - left) / 2;
        if pred(mid) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    left
}

pub fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let q: usize = input.value();
    let mut prefix: Vec<i64> = std::iter::once(0)
        .chain((0..n).map(|_| input.value()))
        .collect();
    for i in 1..n + 1 {
        prefix[i] += prefix[i - 1];
    }

    let line = |i: usize| cht::Line::new(-(i as i64), prefix[i]);
    let flipped = |i: usize| cht::Line::new(i as i64, -prefix[i]);
    let max_hull = SegTree::from_iter((0..n + 1).map(|i| NodeData::singleton(line(i))), HullMonoid);
    let min_hull = SegTree::from_iter(
        (0..n + 1).map(|i| NodeData::singleton(flipped(i))),
        HullMonoid,
    );

    for _ in 0..q {
        let a = input.value::<usize>();
        let b = input.value::<usize>();
        let c = input.value::<usize>();
        let d = input.value::<usize>();

        let satisfiable = |mean: i64| {
            let left = -min_hull.mapped_sum_range(a - 1..b, &MaxOp, |node| node.hull.eval(mean));
            let right = max_hull.mapped_sum_range(c..d + 1, &MaxOp, |node| node.hull.eval(mean));
            left <= right
        };
        let ans = partition_point(X_MIN + 1, X_MAX + 1, |mean| satisfiable(mean)) - 1;
        writeln!(output, "{}", ans).unwrap();
    }
}
