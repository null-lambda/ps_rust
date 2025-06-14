use std::{cmp::Reverse, io::Write};

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

fn subset_sums(xs: &[u32], cutoff: u32) -> Vec<u32> {
    let mut res = vec![0];
    for &x in xs {
        if x > cutoff {
            continue;
        }

        for i in 0..res.len() {
            let sum = res[i] + x;
            if sum <= cutoff {
                res.push(sum);
            }
        }
    }

    res
}

pub mod binary_search {
    use std::mem::MaybeUninit;

    pub struct Eytzinger<T> {
        len: usize,
        inner: Vec<MaybeUninit<T>>,
        index_map: Vec<u32>,
    }

    impl<T: Clone> Eytzinger<T> {
        pub fn new(xs: &[T]) -> Self {
            let n = xs.len();
            let mut this = Self {
                len: n,
                inner: (0..n + 1).map(|_| MaybeUninit::uninit()).collect(),
                index_map: vec![n as u32; n + 1],
            };

            this.build_rec(&mut xs.iter().cloned().enumerate(), 1);
            this
        }

        fn build_rec(&mut self, xs: &mut impl Iterator<Item = (usize, T)>, u: usize) {
            if u > self.len {
                return;
            }
            self.build_rec(xs, u << 1);
            let (i, x) = xs.next().unwrap();
            self.inner[u] = MaybeUninit::new(x);
            self.index_map[u] = i as u32;
            self.build_rec(xs, (u << 1) | 1);
        }

        pub fn partition_point(&self, mut pred: impl FnMut(&T) -> bool) -> usize {
            let mut u = 1;
            while u <= self.len {
                let value = unsafe { self.inner[u].assume_init_ref() };
                let b = pred(value);
                u = 2 * u + b as usize;
            }
            u >>= u.trailing_ones() + 1;
            self.index_map[u as usize] as usize
        }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let c: u32 = input.value();

    let mut xs: Vec<u32> = (0..n).map(|_| input.value()).collect();
    xs.sort_unstable_by_key(|&x| Reverse(x));

    let (ls, rs) = xs.split_at(n / 2);

    let l_sums = subset_sums(ls, c);

    let mut r_sums = subset_sums(rs, c);
    r_sums.sort_unstable();
    let r_sums = binary_search::Eytzinger::new(&r_sums);

    let mut ans = 0u64;
    for l in l_sums {
        let i = r_sums.partition_point(|&r| r <= c - l);
        ans += i as u64;
    }
    writeln!(output, "{}", ans).unwrap();
}
