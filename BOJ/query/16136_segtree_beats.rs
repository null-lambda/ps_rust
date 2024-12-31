use std::io::Write;

use segtree_beats::*;

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

pub mod segtree_beats {
    use std::{iter, ops::Range};

    pub trait MonoidAction {
        type X;
        type F;
        fn id(&self) -> Self::X;
        fn combine(&self, lhs: &Self::X, rhs: &Self::X) -> Self::X;
        fn id_action(&self) -> Self::F;
        fn combine_action(&self, lhs: &Self::F, rhs: &Self::F) -> Self::F;
        fn try_apply_to_sum(
            &self,
            f: &Self::F,
            x_count: u32,
            x_sum: &Self::X,
        ) -> Option<(Self::X, Self::F)>;
    }

    pub struct SegTreeBeats<M: MonoidAction> {
        n: usize,
        max_height: u32,
        pub sum: Vec<M::X>,
        pub lazy: Vec<M::F>,
        pub ma: M,
    }

    impl<M: MonoidAction> SegTreeBeats<M> {
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
        fn push_down(&mut self, node: usize, width: u32) {
            let value = unsafe { &*(&self.lazy[node] as *const _) };
            self.apply(node << 1, width, value);
            self.apply(node << 1 | 1, width, value);
            self.lazy[node] = self.ma.id_action();
        }

        fn pull_up(&mut self, node: usize) {
            self.sum[node] = (self.ma).combine(&self.sum[node << 1], &self.sum[node << 1 | 1]);
        }

        fn apply(&mut self, node: usize, width: u32, action: &M::F) {
            if let Some((value, residual_action)) =
                self.ma.try_apply_to_sum(&action, width, &self.sum[node])
            {
                self.sum[node] = value;
                if node < self.n {
                    self.lazy[node] = self.ma.combine_action(&residual_action, &self.lazy[node]);
                }
            } else {
                if node < self.n {
                    self.lazy[node] = self.ma.combine_action(&action, &self.lazy[node]);
                    self.push_down(node, width >> 1);
                    self.pull_up(node);
                } else {
                    panic!("try_apply_to_sum should return Some(_) for leaf nodes");
                }
            }
        }

        fn push_range(&mut self, range: Range<usize>) {
            let Range { mut start, mut end } = range;
            start += self.n;
            end += self.n;

            let start_height = 1 + start.trailing_zeros();
            let end_height = 1 + end.trailing_zeros();
            for height in (start_height..=self.max_height).rev() {
                let width = 1 << height - 1;
                self.push_down(start >> height, width);
            }
            for height in (end_height..=self.max_height).rev().skip_while(|&height| {
                height >= start_height && end - 1 >> height == start >> height
            }) {
                let width = 1 << height - 1;
                self.push_down(end - 1 >> height, width);
            }
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

fn linear_sieve(n_max: u32) -> (Vec<u32>, Vec<u32>) {
    let mut min_prime_factor = vec![0; n_max as usize + 1];
    let mut primes = Vec::new();

    for i in 2..=n_max {
        if min_prime_factor[i as usize] == 0 {
            primes.push(i);
        }
        for &p in primes.iter() {
            if i * p > n_max {
                break;
            }
            min_prime_factor[(i * p) as usize] = p;
            if i % p == 0 {
                break;
            }
        }
    }

    (min_prime_factor, primes)
}

struct DivisorCount {
    n_divisors: Vec<u32>,
}

impl DivisorCount {
    fn new(n_max: u32) -> Self {
        let (mpf, _) = linear_sieve(n_max);
        let mut n_divisors = vec![0; n_max as usize + 1];

        n_divisors[1] = 1;
        for n in 2..=n_max {
            let p = mpf[n as usize];
            if p == 0 {
                n_divisors[n as usize] = 2;
            } else {
                let mut m = n;
                let mut exp = 1;
                loop {
                    m /= p;
                    if m % p != 0 {
                        break;
                    }
                    exp += 1;
                }
                n_divisors[n as usize] = n_divisors[m as usize] * (exp + 1);
            }
        }

        Self { n_divisors }
    }
}

#[derive(Clone)]
struct NodeData {
    sum: u64,
    max: u32,
}

impl NodeData {
    fn singleton(x: u32) -> Self {
        Self {
            sum: x as u64,
            max: x,
        }
    }
}

#[derive(Clone)]
struct TakeDivisorCount;

impl MonoidAction for DivisorCount {
    type X = NodeData;
    type F = Option<TakeDivisorCount>;

    fn id(&self) -> Self::X {
        NodeData { sum: 0, max: 0 }
    }

    fn combine(&self, lhs: &Self::X, rhs: &Self::X) -> Self::X {
        NodeData {
            sum: lhs.sum + rhs.sum,
            max: lhs.max.max(rhs.max),
        }
    }

    fn id_action(&self) -> Self::F {
        None
    }

    fn combine_action(&self, lhs: &Self::F, rhs: &Self::F) -> Self::F {
        debug_assert!(rhs.is_none());
        lhs.clone()
    }

    fn try_apply_to_sum(
        &self,
        f: &Self::F,
        x_count: u32,
        x: &Self::X,
    ) -> Option<(Self::X, Self::F)> {
        if f.is_none() || x.max <= 2 {
            Some((x.clone(), None))
        } else if x_count == 1 {
            let value = self.n_divisors[x.max as usize] as u64;
            Some((NodeData::singleton(value as u32), None))
        } else {
            None
        }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n = input.value();
    let q = input.value();
    let xs = (0..n).map(|_| NodeData::singleton(input.value()));
    let x_max = 1_000_000;
    let mut xs = SegTreeBeats::from_iter(n, xs, DivisorCount::new(x_max));

    for _ in 0..q {
        let cmd = input.token();
        let l = input.value::<usize>() - 1;
        let r = input.value::<usize>() - 1;
        match cmd {
            "1" => {
                xs.apply_range(l..r + 1, Some(TakeDivisorCount));
            }

            "2" => {
                let ans = xs.query_range(l..r + 1).sum;
                writeln!(output, "{}", ans).unwrap();
            }
            _ => panic!(),
        }
    }
}
