use std::io::Write;

use collections::Jagged;
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

pub mod collections {
    use std::fmt::Debug;
    use std::ops::Index;

    // compressed sparse row format for jagged array
    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
    pub struct Jagged<T> {
        data: Vec<T>,
        head: Vec<u32>,
    }

    impl<T> Debug for Jagged<T>
    where
        T: Debug,
    {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let v: Vec<Vec<&T>> = (0..self.len()).map(|i| self[i].iter().collect()).collect();
            v.fmt(f)
        }
    }

    impl<T, I> FromIterator<I> for Jagged<T>
    where
        I: IntoIterator<Item = T>,
    {
        fn from_iter<J>(iter: J) -> Self
        where
            J: IntoIterator<Item = I>,
        {
            let mut data = vec![];
            let mut head = vec![];
            head.push(0);

            let mut cnt = 0;
            for row in iter {
                data.extend(row.into_iter().inspect(|_| cnt += 1));
                head.push(cnt);
            }
            Jagged { data, head }
        }
    }

    impl<T> Jagged<T> {
        pub fn len(&self) -> usize {
            self.head.len() - 1
        }
    }

    impl<T> Index<usize> for Jagged<T> {
        type Output = [T];
        fn index(&self, index: usize) -> &[T] {
            let start = self.head[index] as usize;
            let end = self.head[index + 1] as usize;
            &self.data[start..end]
        }
    }

    impl<T> Jagged<T> {
        pub fn iter(&self) -> Iter<T> {
            Iter { src: self, pos: 0 }
        }
    }

    impl<'a, T> IntoIterator for &'a Jagged<T> {
        type Item = &'a [T];
        type IntoIter = Iter<'a, T>;
        fn into_iter(self) -> Self::IntoIter {
            self.iter()
        }
    }

    pub struct Iter<'a, T> {
        src: &'a Jagged<T>,
        pos: usize,
    }

    impl<'a, T> Iterator for Iter<'a, T> {
        type Item = &'a [T];
        fn next(&mut self) -> Option<Self::Item> {
            if self.pos < self.src.len() {
                let item = &self.src[self.pos];
                self.pos += 1;
                Some(item)
            } else {
                None
            }
        }
    }
}

pub mod hld {
    use crate::collections::Jagged;

    // Heavy-Light Decomposition
    #[derive(Debug)]
    pub struct HLD {
        pub size: Vec<u32>,
        pub depth: Vec<u32>,
        pub parent: Vec<u32>,
        pub heavy_child: Vec<u32>,
        pub chain_top: Vec<u32>,
        pub euler_idx: Vec<u32>,
    }

    impl HLD {
        pub fn len(&self) -> usize {
            self.parent.len()
        }

        fn dfs_size(&mut self, neighbors: &Jagged<u32>, u: usize) {
            self.size[u] = 1;
            for &v in &neighbors[u] {
                if v == self.parent[u] {
                    continue;
                }
                self.depth[v as usize] = self.depth[u] + 1;
                self.parent[v as usize] = u as u32;
                self.dfs_size(neighbors, v as usize);
                self.size[u] += self.size[v as usize];
            }
            if let Some(h) = neighbors[u]
                .iter()
                .copied()
                .filter(|&v| v != self.parent[u])
                .max_by_key(|&v| self.size[v as usize])
            {
                self.heavy_child[u] = h;
            }
        }

        fn dfs_decompose(&mut self, neighbors: &Jagged<u32>, u: usize, order: &mut u32) {
            self.euler_idx[u] = *order;
            *order += 1;
            if self.heavy_child[u] == u32::MAX {
                return;
            }
            let h = self.heavy_child[u];
            self.chain_top[h as usize] = self.chain_top[u];
            self.dfs_decompose(neighbors, h as usize, order);
            for &v in neighbors[u].iter().filter(|&&v| v != h) {
                if v == self.parent[u] {
                    continue;
                }
                self.chain_top[v as usize] = v;
                self.dfs_decompose(neighbors, v as usize, order);
            }
        }

        pub fn from_graph(neighbors: &Jagged<u32>) -> Self {
            let n = neighbors.len();
            let mut hld = Self {
                size: vec![0; n],
                depth: vec![0; n],
                parent: vec![u32::MAX; n],
                heavy_child: vec![u32::MAX; n],
                chain_top: vec![0; n],
                euler_idx: vec![0; n],
            };
            hld.dfs_size(neighbors, 0);
            hld.dfs_decompose(neighbors, 0, &mut 0);
            hld
        }

        pub fn for_each_path<F>(&self, mut u: usize, mut v: usize, mut visitor: F)
        where
            F: FnMut(usize, usize, bool),
        {
            debug_assert!(u < self.len() && v < self.len());

            while self.chain_top[u] != self.chain_top[v] {
                if self.depth[self.chain_top[u] as usize] < self.depth[self.chain_top[v] as usize] {
                    std::mem::swap(&mut u, &mut v);
                }
                visitor(self.chain_top[u] as usize, u, false);
                u = self.parent[self.chain_top[u] as usize] as usize;
            }
            if self.euler_idx[u] > self.euler_idx[v] {
                std::mem::swap(&mut u, &mut v);
            }
            visitor(u, v, true);
        }

        pub fn for_each_path_splitted<F>(&self, mut u: usize, mut v: usize, mut visit: F)
        where
            F: FnMut(usize, usize, bool, bool),
        {
            debug_assert!(u < self.len() && v < self.len());
            if self.euler_idx[u] > self.euler_idx[v] {
                std::mem::swap(&mut u, &mut v);
            }
            while self.chain_top[u] != self.chain_top[v] {
                if self.depth[self.chain_top[u] as usize] > self.depth[self.chain_top[v] as usize] {
                    visit(self.chain_top[u] as usize, u, true, false);
                    u = self.parent[self.chain_top[u] as usize] as usize;
                } else {
                    visit(self.chain_top[v] as usize, v, false, false);
                    v = self.parent[self.chain_top[v] as usize] as usize;
                }
            }
            if self.depth[u] > self.depth[v] {
                visit(v, u, true, true);
            } else {
                visit(u, v, false, true);
            }
        }

        pub fn lca(&self, mut u: usize, mut v: usize) -> usize {
            debug_assert!(u < self.len() && v < self.len());
            while self.chain_top[u] != self.chain_top[v] {
                if self.depth[self.chain_top[u] as usize] < self.depth[self.chain_top[v] as usize] {
                    std::mem::swap(&mut u, &mut v);
                }
                u = self.parent[self.chain_top[u] as usize] as usize;
            }
            if self.euler_idx[u] > self.euler_idx[v] {
                std::mem::swap(&mut u, &mut v);
            }
            u
        }
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

#[derive(Clone, Debug)]
struct IntervalSum {
    sum: i32,
    max: i32,
    max_left: i32,
    max_right: i32,
}

impl IntervalSum {
    fn new(value: i32) -> Self {
        let max = value.max(0);
        Self {
            sum: value,
            max,
            max_left: max,
            max_right: max,
        }
    }

    fn rev(&self) -> Self {
        Self {
            sum: self.sum,
            max: self.max,
            max_left: self.max_right,
            max_right: self.max_left,
        }
    }
}

struct IntervalSumOp;

impl MonoidAction for IntervalSumOp {
    type X = IntervalSum;
    type F = Option<i32>;
    fn id(&self) -> Self::X {
        IntervalSum::new(0)
    }
    fn combine(&self, lhs: &Self::X, rhs: &Self::X) -> Self::X {
        IntervalSum {
            sum: lhs.sum + rhs.sum,
            max: lhs.max.max(rhs.max).max(lhs.max_right + rhs.max_left),
            max_left: lhs.max_left.max(lhs.sum + rhs.max_left),
            max_right: rhs.max_right.max(rhs.sum + lhs.max_right),
        }
    }
    fn id_action(&self) -> Self::F {
        None
    }
    fn combine_action(&self, lhs: &Self::F, rhs: &Self::F) -> Self::F {
        lhs.or_else(|| *rhs)
    }
    fn apply_to_sum(&self, f: &Self::F, x_count: u32, x_sum: &Self::X) -> Self::X {
        let res = match f {
            None => x_sum.clone(),
            Some(value) => {
                let sum = x_count as i32 * value;
                let max = if *value > 0 {
                    x_count as i32 * *value
                } else {
                    *value
                };
                IntervalSum {
                    sum,
                    max,
                    max_left: max,
                    max_right: max,
                }
            }
        };
        res
    }
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let weights: Vec<i32> = (0..n).map(|_| input.value()).collect();

    let mut neighbors = vec![vec![]; n];
    for _ in 0..n - 1 {
        let u = input.value::<usize>() - 1;
        let v = input.value::<usize>() - 1;
        neighbors[u].push(v as u32);
        neighbors[v].push(u as u32);
    }

    let neighbors: Jagged<_> = neighbors.into_iter().collect();
    let hld = hld::HLD::from_graph(&neighbors);
    let mut euler_idx_inv = vec![0; n];
    for i in 0..n {
        euler_idx_inv[hld.euler_idx[i] as usize] = i;
    }
    let mut weights = LazySegTree::from_iter(
        n,
        (0..n).map(|i| IntervalSum::new(weights[euler_idx_inv[i] as usize])),
        IntervalSumOp,
    );

    let q: usize = input.value();
    for _ in 0..q {
        let cmd = input.token();
        let u = input.value::<usize>() - 1;
        let v = input.value::<usize>() - 1;
        match cmd {
            "1" => {
                let mut res_left = weights.ma.id();
                let mut res_right = weights.ma.id();
                hld.for_each_path_splitted(u, v, |u, v, is_left_path, _has_lca| {
                    let (u, v) = (hld.euler_idx[u] as usize, hld.euler_idx[v] as usize);
                    let x = weights.query_range(u..v + 1);
                    if is_left_path {
                        res_left = weights.ma.combine(&x, &res_left);
                    } else {
                        res_right = weights.ma.combine(&x, &res_right);
                    }
                });
                let res = weights.ma.combine(&res_left.rev(), &res_right);
                writeln!(output, "{}", res.max).unwrap();
            }
            "2" => {
                let w = input.value();
                hld.for_each_path(u, v, |u, v, _is_lca| {
                    let (u, v) = (hld.euler_idx[u] as usize, hld.euler_idx[v] as usize);
                    weights.apply_range(u..v + 1, Some(w));
                });
            }

            _ => panic!(),
        }
    }
}
