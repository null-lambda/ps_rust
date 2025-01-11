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

struct Additive;

impl Group for Additive {
    type X = i32;
    fn id(&self) -> Self::X {
        0
    }
    fn add_assign(&self, lhs: &mut Self::X, rhs: Self::X) {
        *lhs += rhs;
    }
    fn sub_assign(&self, lhs: &mut Self::X, rhs: Self::X) {
        *lhs -= rhs;
    }
}

#[derive(Debug, Default, Clone)]
struct NodeAgg {
    values: Vec<u32>,
    n_inversions: u64,
}

impl NodeAgg {
    fn leaf(value: u32) -> Self {
        Self {
            values: vec![value],
            n_inversions: 0,
        }
    }

    fn collapse(&self) -> u64 {
        self.n_inversions
    }
}

fn init_rec(
    n_leaves: &mut [u32],
    dp: &mut Vec<Result<NodeAgg, [u32; 2]>>,
    preorder: &mut impl Iterator<Item = u32>,
    timer: &mut u32,
) -> usize {
    let idx = *timer as usize;
    *timer += 1;

    let v = preorder.next().unwrap();
    if v != 0 {
        n_leaves[idx] = 1;
        dp[idx] = Ok(NodeAgg::leaf(v));
    } else {
        let lhs = init_rec(n_leaves, dp, preorder, timer);
        let rhs = init_rec(n_leaves, dp, preorder, timer);
        n_leaves[idx] = n_leaves[lhs] + n_leaves[rhs];
        dp[idx] = Err([lhs as u32, rhs as u32]);
    }
    idx
}

fn solve_rec(
    n_leaves: &[u32],
    dp: &mut [Result<NodeAgg, [u32; 2]>],
    global_counter: &mut FenwickTree<Additive>,
    u: usize,
) -> NodeAgg {
    match dp[u] {
        Ok(ref mut node) => {
            global_counter.add(node.values[0] as usize, 1);
            std::mem::take(node)
        }
        Err([mut lhs, mut rhs]) => {
            if n_leaves[lhs as usize] > n_leaves[rhs as usize] {
                std::mem::swap(&mut lhs, &mut rhs);
            }

            let small = solve_rec(n_leaves, dp, global_counter, lhs as usize);
            for &v in &small.values {
                global_counter.add(v as usize, -1);
            }
            let mut large = solve_rec(n_leaves, dp, global_counter, rhs as usize);

            let mut acc = [0; 2];
            for &v in &small.values {
                let n_lt = global_counter.sum_prefix(v as usize) as u32;
                let n_gt = n_leaves[rhs as usize] - n_lt;
                acc[0] += n_lt as u64;
                acc[1] += n_gt as u64;
            }
            let n_cross_inversions = acc.iter().min().unwrap();
            large.n_inversions += small.n_inversions + n_cross_inversions;

            for &v in &small.values {
                global_counter.add(v as usize, 1);
            }
            large.values.extend(small.values);
            large
        }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let n_nodes = 2 * n - 1;
    let mut preorder = std::iter::repeat_with(|| input.value::<u32>());
    let mut n_leaves = vec![0; n_nodes];
    let mut dp = vec![Err(Default::default()); n_nodes];
    let root = init_rec(&mut n_leaves, &mut dp, &mut preorder, &mut 0);
    let ans = solve_rec(&n_leaves, &mut dp, &mut FenwickTree::new(n, Additive), root).collapse();
    writeln!(output, "{}", ans).unwrap();
}
