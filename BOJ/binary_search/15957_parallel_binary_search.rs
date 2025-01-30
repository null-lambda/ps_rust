use std::io::Write;

use fenwick_tree::{FenwickTree, Group};

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
        let buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let iter = buf.split_ascii_whitespace();
        let iter = unsafe { std::mem::transmute(iter) };
        InputAtOnce { _buf: buf, iter }
    }
}

pub mod fenwick_tree {
    pub trait Group {
        type X: Clone;
        fn id(&self) -> Self::X;
        fn add_assign(&self, lhs: &mut Self::X, rhs: Self::X);
        fn sub_assign(&self, lhs: &mut Self::X, rhs: Self::X);
    }

    #[derive(Clone)]
    pub struct FenwickTree<G: Group> {
        n: usize,
        group: G,
        sum: Vec<G::X>,
    }

    impl<G: Group> FenwickTree<G> {
        pub fn new(n: usize, group: G) -> Self {
            let n = n.next_power_of_two(); // Required for binary search
            let sum = (0..n).map(|_| group.id()).collect();
            Self { n, group, sum }
        }

        pub fn from_iter(iter: impl IntoIterator<Item = G::X>, group: G) -> Self {
            let mut sum: Vec<_> = iter.into_iter().collect();
            let n = sum.len();

            let n = n.next_power_of_two(); // Required for binary search
            sum.resize_with(n, || group.id());

            for i in 1..n {
                let prev = sum[i - 1].clone();
                group.add_assign(&mut sum[i], prev);
            }
            for i in (1..n).rev() {
                let j = i & (i + 1);
                if j >= 1 {
                    let prev = sum[j - 1].clone();
                    group.sub_assign(&mut sum[i], prev);
                }
            }

            Self { n, group, sum }
        }

        pub fn add(&mut self, mut idx: usize, value: G::X) {
            debug_assert!(idx < self.n);
            while idx < self.n {
                self.group.add_assign(&mut self.sum[idx], value.clone());
                idx |= idx + 1;
            }
        }

        // Exclusive prefix sum (0..idx)
        pub fn sum_prefix(&self, idx: usize) -> G::X {
            debug_assert!(idx <= self.n);
            let mut res = self.group.id();
            let mut r = idx;
            while r > 0 {
                self.group.add_assign(&mut res, self.sum[r - 1].clone());
                r &= r - 1;
            }
            res
        }

        pub fn sum_range(&self, range: std::ops::Range<usize>) -> G::X {
            debug_assert!(range.start <= range.end && range.end <= self.n);
            let mut res = self.sum_prefix(range.end);
            self.group
                .sub_assign(&mut res, self.sum_prefix(range.start));
            res
        }

        pub fn get(&self, idx: usize) -> G::X {
            self.sum_range(idx..idx + 1)
        }

        // find the first i, such that equiv pred(sum_range(0..=i)) == false
        pub fn partition_point_prefix(&self, mut pred: impl FnMut(&G::X) -> bool) -> usize {
            let p1_log2 = usize::BITS - self.n.leading_zeros();
            let mut idx = 0;
            let mut sum = self.group.id();
            for i in (0..p1_log2).rev() {
                let idx_next = idx | (1 << i);
                if idx_next > self.n {
                    continue;
                }
                let mut sum_next = sum.clone();
                self.group
                    .add_assign(&mut sum_next, self.sum[idx_next - 1].clone());
                if pred(&sum_next) {
                    sum = sum_next;
                    idx = idx_next;
                }
            }
            idx
        }
    }
}

struct Additive<T>(std::marker::PhantomData<T>);

impl<T> Additive<T> {
    pub fn new() -> Self {
        Self(Default::default())
    }
}

impl<T: std::ops::AddAssign + std::ops::SubAssign + Default + Clone> Group for Additive<T> {
    type X = T;

    fn id(&self) -> Self::X {
        T::default()
    }

    fn add_assign(&self, lhs: &mut Self::X, rhs: Self::X) {
        *lhs += rhs;
    }

    fn sub_assign(&self, lhs: &mut Self::X, rhs: Self::X) {
        *lhs -= rhs;
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

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = std::io::BufWriter::new(std::io::stdout().lock());

    let n: usize = input.value();
    let k: usize = input.value();
    let j: i64 = input.value();

    let mut parent = vec![0];
    let mut degree = vec![1u32; n];
    for _ in 1..n as u32 {
        let p = input.value::<u32>() - 1;
        parent.push(p);
        degree[p as usize] += 1;
    }
    degree[0] += 2;

    let mut size = vec![1u32; n];
    let mut topological_order = vec![];
    for mut u in 0..n as u32 {
        while degree[u as usize] == 1 {
            let p = parent[u as usize];
            degree[p as usize] -= 1;
            degree[u as usize] -= 1;
            topological_order.push((u, p));

            size[p as usize] += size[u as usize];

            u = p;
        }
    }

    let mut euler_in = size.clone();
    let mut euler_out = size.clone();
    for &(u, p) in topological_order.iter().rev() {
        let last_idx = euler_in[p as usize];
        euler_in[p as usize] -= euler_in[u as usize];
        euler_in[u as usize] = last_idx;
    }
    for u in 0..n {
        euler_in[u] -= 1;
        euler_out[u] += euler_in[u];
    }

    let mut color = vec![0; n];
    let mut groups = vec![vec![]; n];
    for u in 0..n as u32 {
        let c = input.value::<u32>() - 1;
        color[u as usize] = c;
        groups[c as usize].push(euler_in[u as usize]);
    }

    let mut events = vec![];
    for _ in 0..k as u32 {
        let t: u32 = input.value();
        let u = input.value::<u32>() - 1;
        let w: i32 = input.value();
        let s = size[u as usize];
        events.push((t, u, w / s as i32));
    }
    let (t_map, t_inv) = compress_coord(events.iter().map(|&(t, ..)| t));
    events.iter_mut().for_each(|(t, ..)| *t = t_inv[t]);
    events.sort_unstable();

    // Parallel binary search
    let t_bound = t_inv.len() as u32;
    let mut queries: Vec<_> = (0..n)
        .filter(|&c| !groups[c].is_empty())
        .map(|c| (0, t_bound, c as u32))
        .collect();

    queries.sort_unstable_by_key(|&(l, r, ..)| l + r >> 1);

    let mut ans = vec![-1; n];
    loop {
        let mut events = events.iter().peekable();
        let mut delta_weights = FenwickTree::new(n + 1, Additive::<i64>::new());

        let mut finished = true;
        for (l, r, c) in &mut queries {
            if l == r {
                continue;
            }
            finished = false;

            let mid = *l + *r >> 1;
            while let Some((_, u, w)) = events.next_if(|(t, ..)| *t <= mid) {
                delta_weights.add(euler_in[*u as usize] as usize, *w as i64);
                delta_weights.add(euler_out[*u as usize] as usize, -*w as i64);
            }

            let count = groups[*c as usize].len() as i64;
            let total_weight = groups[*c as usize]
                .iter()
                .map(|&eu| delta_weights.sum_prefix(eu as usize + 1))
                .sum::<i64>();
            if total_weight <= j * count {
                *l = mid + 1;
            } else {
                *r = mid;
            }
            if l == r {
                if *l < t_bound {
                    ans[*c as usize] = t_map[*l as usize] as i32;
                }
            }
        }

        if finished {
            break;
        }

        // Stable sort, for nearly sorted arrays
        queries.sort_by_key(|&(l, r, ..)| l + r >> 1);
    }

    for u in 0..n {
        writeln!(output, "{}", ans[color[u] as usize]).unwrap();
    }
}
