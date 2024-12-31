use std::collections::BTreeMap;
use std::io::Write;

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

        fn dfs_size(&mut self, children: &Jagged<u32>, u: usize) {
            self.size[u] = 1;
            for &v in &children[u] {
                self.depth[v as usize] = self.depth[u] + 1;
                self.parent[v as usize] = u as u32;
                self.dfs_size(children, v as usize);
                self.size[u] += self.size[v as usize];
            }
            if let Some(h) = children[u]
                .iter()
                .copied()
                .filter(|&v| v != self.parent[u])
                .max_by_key(|&v| self.size[v as usize])
            {
                self.heavy_child[u] = h;
            }
        }

        fn dfs_decompose(&mut self, children: &Jagged<u32>, u: usize, order: &mut u32) {
            self.euler_idx[u] = *order;
            *order += 1;
            if self.heavy_child[u] == u32::MAX {
                return;
            }
            let h = self.heavy_child[u];
            self.chain_top[h as usize] = self.chain_top[u];

            self.dfs_decompose(children, h as usize, order);
            for &v in children[u].iter().filter(|&&v| v != h) {
                self.chain_top[v as usize] = v;
                self.dfs_decompose(children, v as usize, order);
            }
        }

        pub fn from_graph(children: &Jagged<u32>, root: usize) -> Self {
            let n = children.len();
            let mut hld = Self {
                size: vec![0; n],
                depth: vec![0; n],
                parent: vec![u32::MAX; n],
                heavy_child: vec![u32::MAX; n],
                chain_top: vec![root as u32; n],
                euler_idx: vec![0; n],
            };
            hld.dfs_size(children, root);
            hld.dfs_decompose(children, root, &mut 0);

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

    pub trait Monoid {
        type Elem;
        fn id(&self) -> Self::Elem;
        fn op(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem;
    }

    #[derive(Debug)]
    pub struct SegTree<M>
    where
        M: Monoid,
    {
        n: usize,
        sum: Vec<M::Elem>,
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

        pub fn from_iter<I>(n: usize, iter: I, monoid: M) -> Self
        where
            I: Iterator<Item = M::Elem>,
        {
            let mut sum: Vec<_> = (0..n)
                .map(|_| monoid.id())
                .chain(iter)
                .chain(iter::repeat_with(|| monoid.id()))
                .take(2 * n)
                .collect();
            for i in (0..n).rev() {
                sum[i] = monoid.op(&sum[i << 1], &sum[i << 1 | 1]);
            }
            Self { n, sum, monoid }
        }

        pub fn set(&mut self, mut idx: usize, value: M::Elem) {
            debug_assert!(idx < self.n);
            idx += self.n;
            self.sum[idx] = value;
            while idx > 1 {
                idx >>= 1;
                self.sum[idx] = self.monoid.op(&self.sum[idx << 1], &self.sum[idx << 1 | 1]);
            }
        }

        pub fn get(&self, idx: usize) -> &M::Elem {
            &self.sum[idx + self.n]
        }

        pub fn query_range(&self, range: Range<usize>) -> M::Elem {
            let Range { mut start, mut end } = range;
            debug_assert!(start < self.n && end <= self.n);
            start += self.n;
            end += self.n;
            let (mut result_left, mut result_right) = (self.monoid.id(), self.monoid.id());
            while start < end {
                if start & 1 != 0 {
                    result_left = self.monoid.op(&result_left, &self.sum[start]);
                }
                if end & 1 != 0 {
                    result_right = self.monoid.op(&self.sum[end - 1], &result_right);
                }
                start = (start + 1) >> 1;
                end >>= 1;
            }

            self.monoid.op(&result_left, &result_right)
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
enum Event {
    EnterRect = 0,
    ExitRect = 1,
    TestPoint = 2,
}

use collections::Jagged;
use segtree::Monoid;
use Event::*;

struct BitAndOp;

impl segtree::Monoid for BitAndOp {
    type Elem = u64;
    fn id(&self) -> Self::Elem {
        !0
    }
    fn op(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        a & b
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let mut rects = vec![];
    let mut rect_colors = vec![];

    // Root node
    let bound = 1_000_000_001;
    rects.push([-bound, -bound, bound, bound]);
    rect_colors.push(0);

    for _ in 0..n {
        let x1: i32 = input.value();
        let y1: i32 = input.value();
        let x2: i32 = input.value();
        let y2: i32 = input.value();

        debug_assert!((-bound + 1..bound).contains(&x1));
        debug_assert!((-bound + 1..bound).contains(&y1));
        debug_assert!((-bound + 1..bound).contains(&x2));
        debug_assert!((-bound + 1..bound).contains(&y2));

        rects.push([x1, y1, x2, y2]);

        let c1: u8 = input.value();
        let c2: u8 = input.value();
        let c3: u8 = input.value();
        let c4: u8 = input.value();

        let mask = |c| {
            debug_assert!(c < 6);
            1 << (c - 1) as u8
        };
        let allowed_colors = mask(c1) | mask(c2) | mask(c3) | mask(c4);
        rect_colors.push(allowed_colors);
    }
    let n = n + 1;

    let q: usize = input.value();
    let mut rect_of_point = vec![None; q * 2];
    let mut query_points = vec![];
    for _ in 0..q {
        let x1: i32 = input.value();
        let y1: i32 = input.value();
        let x2: i32 = input.value();
        let y2: i32 = input.value();
        query_points.push([x1, y1]);
        query_points.push([x2, y2]);
    }

    let mut events = vec![];
    for i in 0..n {
        let [_, y1, _, y2] = rects[i];
        events.push((y1, EnterRect, i));
        events.push((y2, ExitRect, i));
    }

    for i in 0..q * 2 {
        let [_, y] = query_points[i];
        events.push((y, TestPoint, i));
    }
    events.sort_unstable();

    let mut children: Vec<Vec<u32>> = vec![vec![]; n];
    let mut parents = vec![None; n];

    let mut active = BTreeMap::new();
    for (_, ty, i) in events {
        let find_parent = |x1: i32, x2: i32| -> Option<usize> {
            let left = active.range(..x1).next_back().map(|(_, &p)| p);
            let right = active.range(x2 + 1..).next().map(|(_, &p)| p);
            if left.is_none() && right.is_none() {
                return None;
            }

            let left = left.unwrap();
            let right = right.unwrap();
            if left == right {
                Some(left)
            } else if parents[left] == Some(right) {
                Some(right)
            } else if parents[right] == Some(left) {
                Some(left)
            } else if parents[left] == parents[right] {
                Some(parents[left].unwrap())
            } else {
                panic!()
            }
        };

        match ty {
            EnterRect => {
                let [x1, _y1, x2, _y2] = rects[i];
                if let Some(parent) = find_parent(x1, x2) {
                    children[parent as usize].push(i as u32);
                    parents[i] = Some(parent);
                }
                active.insert(x1, i);
                active.insert(x2, i);
            }
            ExitRect => {
                let [x1, _, x2, _] = rects[i];
                active.remove(&x1);
                active.remove(&x2);
            }
            TestPoint => {
                let [x, _y] = query_points[i];
                rect_of_point[i] = find_parent(x, x);
            }
        }
    }

    let hld = hld::HLD::from_graph(&Jagged::from_iter(children), 0);
    let mut segtree = segtree::SegTree::with_size(n, BitAndOp);
    for i in 0..n {
        let mut valid_combinations = 0;
        for m in 0..64 {
            if m & rect_colors[i] != 0 {
                valid_combinations |= 1 << m;
            }
        }
        segtree.set(hld.euler_idx[i] as usize, valid_combinations);
    }

    for i in 0..q {
        let s = rect_of_point[i * 2];
        let e = rect_of_point[i * 2 + 1];

        let monoid = BitAndOp;
        let mut res = monoid.id();
        hld.for_each_path(
            s.unwrap() as usize,
            e.unwrap() as usize,
            |u, v, contains_lca| {
                let u = hld.euler_idx[u] as usize;
                let v = hld.euler_idx[v] as usize;
                if contains_lca {
                    res = monoid.op(&res, &segtree.query_range(u + 1..v + 1));
                } else {
                    res = monoid.op(&res, &segtree.query_range(u..v + 1));
                }
            },
        );

        let mut min_count = 6;
        for m in 0..64u8 {
            if res & (1 << m) != 0 {
                min_count = min_count.min(m.count_ones());
            }
        }
        writeln!(output, "{}", min_count).unwrap();
    }
}
