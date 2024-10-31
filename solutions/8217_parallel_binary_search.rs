use std::ops::Range;
use std::{io::Write, ops::RangeInclusive};

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

fn partition_in_place<T>(xs: &mut [T], mut pred: impl FnMut(&T) -> bool) -> (&mut [T], &mut [T]) {
    let n = xs.len();
    let mut i = 0;
    for j in 0..n {
        if pred(&xs[j]) {
            xs.swap(i, j);
            i += 1;
        }
    }
    xs.split_at_mut(i)
}

pub mod fenwick_tree {
    pub trait Group {
        type Elem: Clone;
        fn id(&self) -> Self::Elem;
        fn add_assign(&self, lhs: &mut Self::Elem, rhs: Self::Elem);
        fn sub_assign(&self, lhs: &mut Self::Elem, rhs: Self::Elem);
    }

    pub struct FenwickTree<G: Group> {
        n: usize,
        group: G,
        data: Vec<G::Elem>,
    }

    impl<G: Group> FenwickTree<G> {
        pub fn new(n: usize, group: G) -> Self {
            let n_ceil = n.next_power_of_two();
            let data = (0..n_ceil).map(|_| group.id()).collect();
            Self { n, group, data }
        }

        pub fn add(&mut self, mut idx: usize, value: G::Elem) {
            while idx < self.n {
                self.group.add_assign(&mut self.data[idx], value.clone());
                idx |= idx + 1;
            }
        }
        pub fn get(&self, idx: usize) -> G::Elem {
            self.sum_range(idx..idx + 1)
        }

        pub fn sum_prefix(&self, idx: usize) -> G::Elem {
            let mut res = self.group.id();
            let mut r = idx + 1;
            while r > 0 {
                self.group.add_assign(&mut res, self.data[r - 1].clone());
                r &= r - 1;
            }

            res
        }

        pub fn sum_range(&self, range: std::ops::Range<usize>) -> G::Elem {
            let mut res = self.group.id();
            let mut r = range.end;
            while r > 0 {
                self.group.add_assign(&mut res, self.data[r - 1].clone());
                r &= r - 1;
            }

            let mut l = range.start;
            while l > 0 {
                self.group.sub_assign(&mut res, self.data[l - 1].clone());
                l &= l - 1;
            }

            res
        }
    }
}

use fenwick_tree::{FenwickTree, Group};
struct AddGroup;

impl Group for AddGroup {
    type Elem = i64;
    fn id(&self) -> i64 {
        0
    }
    fn add_assign(&self, lhs: &mut i64, rhs: i64) {
        *lhs += rhs;
    }
    fn sub_assign(&self, lhs: &mut i64, rhs: i64) {
        *lhs -= rhs;
    }
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let mut land = vec![vec![]; n];
    for l in 0..m {
        let o = input.value::<usize>() - 1;
        land[o].push(l);
    }
    let goal: Vec<i64> = (0..n).map(|_| input.value()).collect();
    let q: usize = input.value();
    let queries: Vec<(usize, usize, i64)> = (0..q)
        .map(|_| {
            let l: usize = input.value();
            let r: usize = input.value();
            let a: i64 = input.value();
            (l - 1, r, a)
        })
        .collect();

    let mut tree = FenwickTree::new(m, AddGroup);

    // Parallel binary search
    // divide and conquer
    fn dnc(
        land: &[Vec<usize>],
        goal: &[i64],
        queries: &[(usize, usize, i64)],
        tree: &mut FenwickTree<AddGroup>,
        bound: Range<usize>,
        owners: &mut [usize],
        ans: &mut [usize],
    ) {
        let process_queries =
            |tree: &mut FenwickTree<AddGroup>, range: RangeInclusive<usize>, inv: bool| {
                for i in range {
                    let (l, r, mut a) = queries[i];
                    if inv {
                        a = -a;
                    }
                    if l < r {
                        tree.add(l, a);
                        tree.add(r, -a);
                    } else {
                        tree.add(0, a);
                        tree.add(l, a);
                        tree.add(r, -a);
                    }
                }
            };
        let pred = |tree: &FenwickTree<AddGroup>, o: usize| {
            land[o]
                .iter()
                .map(|&l| tree.sum_prefix(l))
                .scan(0, |acc, x| {
                    *acc += x;
                    Some(*acc)
                })
                .any(|sum| sum >= goal[o])
        };

        let Range { start, end } = bound;
        if start == end || owners.is_empty() {
            for &o in owners.iter() {
                ans[o] = start;
            }
            return;
        }

        let mid = (start + end) / 2;

        process_queries(tree, start..=mid, false);
        let (left_owners, right_owners) = partition_in_place(owners, |&o| pred(tree, o));
        dnc(land, goal, queries, tree, mid + 1..end, right_owners, ans);
        process_queries(tree, start..=mid, true);

        dnc(land, goal, queries, tree, start..mid, left_owners, ans);
    }

    let mut ans = vec![0; n];
    dnc(
        &land,
        &goal,
        &queries,
        &mut tree,
        0..q,
        &mut (0..n).collect::<Vec<_>>(),
        &mut ans,
    );

    for a in ans {
        if a < q {
            writeln!(output, "{}", a + 1).unwrap();
        } else {
            writeln!(output, "NIE").unwrap();
        }
    }
}
