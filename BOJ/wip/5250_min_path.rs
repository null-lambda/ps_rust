use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashSet},
    io::Write,
};

use jagged::Jagged;
use segtree_lazy::{MonoidAction, SegTree};

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

pub mod jagged {
    use std::fmt::Debug;
    use std::iter;
    use std::mem::MaybeUninit;

    // Trait for painless switch between different representations of a jagged array
    pub trait Jagged<'a, T: 'a> {
        type ItemRef: ExactSizeIterator<Item = &'a T>;
        fn len(&self) -> usize;
        fn get(&'a self, u: usize) -> Self::ItemRef;
    }

    impl<'a, T, C> Jagged<'a, T> for C
    where
        C: AsRef<[Vec<T>]> + 'a,
        T: 'a,
    {
        type ItemRef = std::slice::Iter<'a, T>;
        fn len(&self) -> usize {
            <Self as AsRef<[Vec<T>]>>::as_ref(self).len()
        }
        fn get(&'a self, u: usize) -> Self::ItemRef {
            let res = <Self as AsRef<[Vec<T>]>>::as_ref(self)[u].iter();
            res
        }
    }

    // Compressed sparse row format for jagged array
    // Provides good locality for graph traversal, but works only for static ones.
    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
    pub struct CSR<T> {
        data: Vec<T>,
        head: Vec<u32>,
    }

    impl<T> Debug for CSR<T>
    where
        T: Debug,
    {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let v: Vec<Vec<&T>> = (0..self.len()).map(|i| self.get(i).collect()).collect();
            v.fmt(f)
        }
    }

    impl<T, I> FromIterator<I> for CSR<T>
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
            CSR { data, head }
        }
    }

    impl<T: Clone> CSR<T> {
        pub fn from_assoc_list(n: usize, pairs: &[(u32, T)]) -> Self {
            let mut head = vec![0u32; n + 1];

            for &(u, _) in pairs {
                debug_assert!(u < n as u32);
                head[u as usize + 1] += 1;
            }
            for i in 2..n + 1 {
                head[i] += head[i - 1];
            }
            let mut data: Vec<_> = iter::repeat_with(|| MaybeUninit::uninit())
                .take(head[n] as usize)
                .collect();
            let mut pos = head.clone();

            for (u, v) in pairs {
                data[pos[*u as usize] as usize] = MaybeUninit::new(v.clone());
                pos[*u as usize] += 1;
            }

            let data = std::mem::ManuallyDrop::new(data);
            let data = unsafe {
                Vec::from_raw_parts(data.as_ptr() as *mut T, data.len(), data.capacity())
            };

            CSR { data, head }
        }
    }

    impl<'a, T: 'a> Jagged<'a, T> for CSR<T> {
        type ItemRef = std::slice::Iter<'a, T>;

        fn len(&self) -> usize {
            self.head.len() - 1
        }

        fn get(&'a self, u: usize) -> Self::ItemRef {
            self.data[self.head[u] as usize..self.head[u + 1] as usize].iter()
        }
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

struct MinOp;

const INF: i64 = i64::MAX / 4;

impl MonoidAction for MinOp {
    type X = i64;
    type F = i64;
    // const IS_X_COMMUTATIVE: bool = true; // TODO

    fn id(&self) -> Self::X {
        INF
    }

    fn combine(&self, lhs: &Self::X, rhs: &Self::X) -> Self::X {
        (*lhs).min(*rhs)
    }

    fn id_action(&self) -> Self::F {
        i64::MAX
    }

    fn combine_action(&self, lhs: &Self::F, rhs: &Self::F) -> Self::F {
        (*lhs).min(*rhs)
    }

    fn apply_to_sum(&self, f: &Self::F, _x_count: u32, x_sum: &Self::X) -> Self::X {
        (*f).min(*x_sum)
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let src = input.value::<usize>() - 1;
    let dest = input.value::<usize>() - 1;

    let mut edges = vec![];
    for _ in 0..m {
        let u = input.value::<u32>() - 1;
        let v = input.value::<u32>() - 1;
        let w = input.value::<i64>();
        edges.push((u, (v, w)));
        edges.push((v, (u, w)));
    }
    let neighbors = jagged::CSR::from_assoc_list(n, &edges);

    let k: usize = input.value();
    let lucky_path: Vec<u32> = (0..k).map(|_| input.value::<u32>() - 1).collect();
    let lucky_edges = lucky_path
        .windows(2)
        .flat_map(|t| [(t[0], t[1]), (t[1], t[0])])
        .collect::<HashSet<_>>();

    const UNSET: u32 = u32::MAX;
    let mut closest_lucky_node_src = vec![(UNSET, INF); n];
    let mut closest_lucky_node_dest = vec![(UNSET, INF); n];
    for (i, &u) in lucky_path.iter().enumerate() {
        closest_lucky_node_src[u as usize] = (i as u32, 0);
        closest_lucky_node_dest[u as usize] = (i as u32, 0);
    }

    // Dijkstra
    let mut pq = BinaryHeap::new();
    let mut visited = vec![false; n];
    let mut dist_from_src = vec![INF; n];
    pq.push((Reverse(0), src as u32, UNSET, INF));
    while let Some((Reverse(d_u), u, p, w)) = pq.pop() {
        if visited[u as usize] {
            continue;
        }
        visited[u as usize] = true;
        dist_from_src[u as usize] = d_u;

        let (i, d_ui) = &mut closest_lucky_node_src[u as usize];
        if *i == UNSET {
            *i = p;
            *d_ui += w;
        }

        for &(v, w) in neighbors.get(u as usize) {
            if !visited[v as usize] {
                pq.push((Reverse(d_u + w), v, u, w));
            }
        }
    }

    let mut pq = BinaryHeap::new();
    let mut visited = vec![false; n];
    let mut dist_from_dest = vec![INF; n];
    pq.push((Reverse(0), dest as u32, UNSET, INF));
    while let Some((Reverse(d_u), u, p, w)) = pq.pop() {
        if visited[u as usize] {
            continue;
        }
        visited[u as usize] = true;
        dist_from_dest[u as usize] = d_u;

        let (i, d_ui) = &mut closest_lucky_node_src[u as usize];
        if *i == UNSET {
            *i = p;
            *d_ui += w;
        }

        for &(v, w) in neighbors.get(u as usize) {
            if !visited[v as usize] {
                pq.push((Reverse(d_u + w), v, u, w));
            }
        }
    }

    let mut ans = SegTree::with_size(k - 1, MinOp);

    for &(u, (v, w)) in &edges {
        if lucky_edges.contains(&(u, v)) {
            continue;
        }
        let (ps, _) = closest_lucky_node_src[u as usize];
        let (pe, _) = closest_lucky_node_dest[v as usize];
        if ps > pe {
            continue;
        }
        println!("{} {} {} {}", u, v, ps, pe);
        let d = dist_from_src[u as usize] + w + dist_from_dest[v as usize];
        if d < INF {
            ans.apply_range(0..ps as usize, d);
            // ans.apply_range(pe as usize + 1..k - 1, d);
        }
        println!("{} {} {}", u, v, d);
    }

    for i in 0..k - 1 {
        let mut a = ans.query_range(i..i + 1);
        if a == INF {
            a = -1;
        }
        writeln!(output, "{}", a).unwrap();
    }
}
