use std::io::Write;
use std::usize;

use segtree::FuncMonoid;

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

pub mod segtree {
    use std::ops::Range;

    pub trait FuncMonoid<F> {
        type A;
        type B;

        fn id(&self) -> F;
        fn op(&self, lhs: &F, rhs: &F) -> F;
        fn eval(&self, f: &F, value: &Self::A) -> Self::B;

        fn id_value(&self) -> Self::B;
        fn op_value(&self, lhs: &Self::B, rhs: &Self::B) -> Self::B;
    }

    #[derive(Debug)]
    pub struct SegTree<F, M> {
        n: usize,
        sum: Vec<F>,
        monoid: M,
    }

    impl<F, M> SegTree<F, M>
    where
        F: Clone + Eq,
        M: FuncMonoid<F>,
    {
        pub fn from_iter<I>(n: usize, monoid: M, iter: I) -> Self
        where
            I: Iterator<Item = F>,
        {
            use std::iter::repeat;
            let mut sum: Vec<F> = repeat(monoid.id())
                .take(n)
                .chain(iter)
                .chain(repeat(monoid.id()))
                .take(2 * n)
                .collect();
            for i in (0..n).rev() {
                sum[i] = monoid.op(&sum[i << 1], &sum[i << 1 | 1]);
            }
            Self { n, sum, monoid }
        }

        pub fn query_range(&self, range: Range<usize>, a: M::A) -> M::B {
            let Range { mut start, mut end } = range;
            debug_assert!(start < self.n && end <= self.n);
            start += self.n;
            end += self.n;
            let mut result = [self.monoid.id_value(), self.monoid.id_value()];
            while start < end {
                if start & 1 != 0 {
                    result[0] = self
                        .monoid
                        .op_value(&result[0], &self.monoid.eval(&self.sum[start], &a));
                }
                if end & 1 != 0 {
                    result[1] = self
                        .monoid
                        .op_value(&self.monoid.eval(&self.sum[end - 1], &a), &result[1]);
                }
                start = (start + 1) >> 1;
                end >>= 1;
            }

            self.monoid.op_value(&result[0], &result[1])
        }
    }
}

struct ArraySorter;

impl<T: Ord + Clone> FuncMonoid<Vec<T>> for ArraySorter {
    type A = T;
    type B = usize;
    fn id(&self) -> Vec<T> {
        vec![]
    }

    fn op(&self, lhs: &Vec<T>, rhs: &Vec<T>) -> Vec<T> {
        let mut res: Vec<T> = lhs.iter().chain(rhs.iter()).cloned().collect();
        res.sort_unstable();
        res
    }

    fn eval(&self, f: &Vec<T>, value: &T) -> usize {
        f.partition_point(|x| x < value)
    }

    fn id_value(&self) -> Self::B {
        0
    }

    fn op_value(&self, lhs: &Self::B, rhs: &Self::B) -> Self::B {
        lhs + rhs
    }
}

fn partition_point(range: std::ops::Range<i32>, mut pred: impl FnMut(&i32) -> bool) -> i32 {
    let mut left = range.start;
    let mut right = range.end;

    while left < right {
        let mid = left + (right - left) / 2;
        if pred(&mid) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    left
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = std::io::BufWriter::new(std::io::stdout());
    let n: usize = input.value();
    let q: usize = input.value();

    let xs = (0..n).map(|_| vec![input.value::<i32>()]);
    let segtree = segtree::SegTree::from_iter(n, ArraySorter, xs);

    for _ in 0..q {
        let i: usize = input.value();
        let j: usize = input.value();
        let k: usize = input.value();
        let result = partition_point(-1_000_000_000..1_000_000_001, |ub| {
            segtree.query_range(i - 1..j, *ub + 1) < k
        });
        writeln!(output, "{}", result).unwrap();
    }
}
