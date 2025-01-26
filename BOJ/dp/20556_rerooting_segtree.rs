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
        Self(std::marker::PhantomData)
    }
}

impl<T: std::ops::AddAssign + std::ops::SubAssign + Clone + Default> Group for Additive<T> {
    type X = T;

    fn id(&self) -> T {
        T::default()
    }

    fn add_assign(&self, lhs: &mut T, rhs: T) {
        *lhs += rhs;
    }

    fn sub_assign(&self, lhs: &mut T, rhs: T) {
        *lhs -= rhs;
    }
}

fn partition_point<P>(mut left: u32, mut right: u32, mut pred: P) -> u32
where
    P: FnMut(u32) -> bool,
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

fn process_query(
    n: usize,
    size_freq: &FenwickTree<Additive<i32>>,
    size_sum: &FenwickTree<Additive<i64>>,
    k: usize,
) -> i64 {
    let total = size_sum.sum_range(0..n);

    let i = size_freq
        .partition_point_prefix(|prefix| (n as i32 - 1 - *prefix >= k as i32))
        .min(n - 1);
    // let i = partition_point(0, n as u32 - 1, |i| {
    //     n as i32 - 1 - size_freq.sum_prefix(i as usize + 1) >= k as i32
    // }) as usize;
    // println!("i: {}", i);
    let i_next = i + 1;

    let sum_largest_k = size_sum.sum_range(i_next..n)
        + (k - size_freq.sum_range(i_next..n) as usize) as i64 * i as i64;

    total * 2 - sum_largest_k
}

fn pull_up(
    size_freq: &mut FenwickTree<Additive<i32>>,
    size_sum: &mut FenwickTree<Additive<i64>>,
    u: usize,
    p: usize,
    size: &mut Vec<u32>,
    inv: bool,
) {
    if !inv {
        size_freq.add(size[u] as usize, 1);
        size_sum.add(size[u] as usize, size[u] as i64);
        size[p] += size[u];
    } else {
        size_freq.add(size[u] as usize, -1);
        size_sum.add(size[u] as usize, -(size[u] as i64));
        size[p] -= size[u];
    }
}

fn reroot_down(
    size_freq: &mut FenwickTree<Additive<i32>>,
    size_sum: &mut FenwickTree<Additive<i64>>,
    u: usize,
    p: usize,
    size: &mut Vec<u32>,
) {
    pull_up(size_freq, size_sum, u, p, size, true);
    pull_up(size_freq, size_sum, p, u, size, false);
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let q: usize = input.value();
    let mut neighbors = vec![vec![]; n];
    let mut degree = vec![0u32; n];
    let mut xor_neighbors = vec![0u32; n];
    for _ in 0..n - 1 {
        let u = input.value::<usize>() - 1;
        let v = input.value::<usize>() - 1;
        neighbors[u].push(v);
        neighbors[v].push(u);
        degree[u] += 1;
        degree[v] += 1;
        xor_neighbors[u] ^= v as u32;
        xor_neighbors[v] ^= u as u32;
    }
    degree[0] += 2;

    let mut queries = vec![vec![]; n];
    for i in 0..q {
        let u = input.value::<usize>() - 1;
        let a: u32 = input.value();
        queries[u].push((i, a));
    }

    let mut size_freq = FenwickTree::new(n.next_power_of_two(), Additive::<i32>::new());
    let mut size_sum = FenwickTree::new(n, Additive::<i64>::new());

    let mut size = vec![1u32; n];
    for mut u in 0..n as u32 {
        while degree[u as usize] == 1 {
            let p = xor_neighbors[u as usize];
            degree[p as usize] -= 1;
            degree[u as usize] -= 1;
            xor_neighbors[p as usize] ^= u;

            size_freq.add(size[u as usize] as usize, 1);
            size_sum.add(size[u as usize] as usize, size[u as usize] as i64);
            size[p as usize] += size[u as usize];

            u = p;
        }
    }

    // dfs, rerooting (rollback on exit)
    let mut stack = vec![(0, 0, 0)]; // (u, p, iv)
    let mut ans = vec![0; q];
    while let Some((u, p, iv)) = stack.pop() {
        if iv == 0 {
            for &(i, a) in &queries[u as usize] {
                ans[i] = process_query(n, &size_freq, &size_sum, a as usize);
            }
        }
        if iv > 0 {
            let v = neighbors[u as usize][iv as usize - 1] as u32;
            if v != p {
                // Rollback u -> v
                reroot_down(
                    &mut size_freq,
                    &mut size_sum,
                    u as usize,
                    v as usize,
                    &mut size,
                );
            }
        }
        if iv < neighbors[u as usize].len() as u32 {
            let v = neighbors[u as usize][iv as usize] as u32;
            stack.push((u, p, iv + 1));
            if v != p {
                stack.push((v, u as u32, 0));
                // Enter
                reroot_down(
                    &mut size_freq,
                    &mut size_sum,
                    v as usize,
                    u as usize,
                    &mut size,
                );
            }
        }
    }

    for &a in &ans {
        writeln!(output, "{}", a).unwrap();
    }
}
