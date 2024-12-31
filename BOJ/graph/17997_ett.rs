use std::{
    cmp::{Ordering, Reverse},
    io::Write,
};

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

fn dfs_euler(
    children: &[Vec<usize>],
    euler_interval: &mut Vec<(usize, usize)>,
    order: &mut usize,
    u: usize,
    p: usize,
) {
    euler_interval[u].0 = *order;
    *order += 1;
    for &v in &children[u] {
        if v == p {
            continue;
        }
        dfs_euler(children, euler_interval, order, v, u);
    }
    euler_interval[u].1 = *order;
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

const P: u32 = 11092019;

#[derive(Clone)]
struct MaxCount<T> {
    value: T,
    count: u32,
}

impl<T: Ord> MaxCount<T> {
    fn singleton(value: T) -> Self {
        Self { value, count: 1 }
    }

    fn map(self, f: impl FnOnce(T) -> T) -> Self {
        Self {
            value: f(self.value),
            count: self.count,
        }
    }
}

#[derive(Clone)]
struct MaxCountOp<T> {
    neg_inf: T,
}

impl<T: Ord + Clone> Monoid for MaxCountOp<T> {
    type Elem = MaxCount<T>;

    fn id(&self) -> Self::Elem {
        MaxCount {
            value: self.neg_inf.clone(),
            count: 0,
        }
    }

    fn op(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        match a.value.cmp(&b.value) {
            Ordering::Less => b.clone(),
            Ordering::Greater => a.clone(),
            Ordering::Equal => MaxCount {
                value: a.value.clone(),
                count: (a.count + b.count) % P,
            },
        }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let xs: Vec<u32> = (0..n).map(|_| input.value()).collect();

    let root = 0;
    let mut parent = vec![root; n];
    let mut children = vec![vec![]; n];
    for i in 1..n {
        let p = input.value::<usize>() - 1;
        parent[i] = p;
        children[p].push(i);
    }

    let mut euler_interval = vec![(0, 0); n];
    dfs_euler(&children, &mut euler_interval, &mut 0, root, root);

    let mut order: Vec<_> = (0..n).collect();
    order.sort_unstable_by_key(|&i| (Reverse(xs[i]), Reverse(euler_interval[i].0)));

    let monoid = MaxCountOp { neg_inf: i32::MIN };
    let mut lis_len = SegTree::with_size(n, monoid.clone());

    let mut ans = MaxCount::singleton(0);
    for i in order {
        let old = lis_len.query_range(euler_interval[i].0..euler_interval[i].1);
        let curr = if old.count == 0 {
            MaxCount::singleton(1)
        } else {
            old.map(|x| x + 1)
        };
        ans = monoid.op(&ans, &curr);
        lis_len.set(euler_interval[i].0, curr);
    }

    writeln!(output, "{} {}", ans.value, ans.count).unwrap();
}
