use std::io::Write;

use segtree::{Monoid, SegTree};

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

struct BitFuncOp;

impl BitFuncOp {
    fn singleton(value: u32) -> u64 {
        (0..32)
            .map(|i| (if (value >> i) & 1 != 0 { 0b01 } else { 0b11 }) << 2 * i)
            .fold(0, |acc, i| acc | i)
    }

    fn apply(value: u32, func: u64) -> u32 {
        (0..32)
            .map(|i| {
                let f = (func >> 2 * i) & 0b11;
                (if (value >> i) & 1 != 0 { f >> 1 } else { f & 1 } as u32) << i
            })
            .fold(0, |acc, x| acc | x)
    }
}

impl Monoid for BitFuncOp {
    type Elem = u64;

    fn id(&self) -> Self::Elem {
        0xaaaa_aaaa_aaaa_aaaa
    }

    fn op(&self, a: &Self::Elem, b: &Self::Elem) -> Self::Elem {
        (0..32)
            .map(|i| {
                let shift = 2 * i;
                let fa = ((a >> shift) & 0b11) as u8;
                let fb = ((b >> shift) & 0b11) as u8;
                let c0 = (fb >> ((fa >> 0) & 1)) & 1;
                let c1 = (fb >> ((fa >> 1) & 1)) & 1;
                ((c0 | c1 << 1) as u64) << shift
            })
            .fold(0, |acc, x| acc | x)
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let k: usize = input.value();
    assert!(k <= 31);

    let mut cs: Vec<u32> = (0..n).map(|_| input.value()).collect();
    let mut segtree =
        SegTree::from_iter(n, cs.iter().copied().map(BitFuncOp::singleton), BitFuncOp);

    let mask = (1 << k) - 1;

    for _ in 0..m {
        match input.token() {
            "1" => {
                let x = input.value::<usize>() - 1;
                let y = input.value::<u32>();
                cs[x] = y;
                segtree.set(x, BitFuncOp::singleton(y));
            }
            "2" => {
                let l = input.value::<usize>() - 1;
                let r = input.value::<usize>() - 1;
                let value = cs[l];
                let func = segtree.query_range(l + 1..r + 1);
                let res = BitFuncOp::apply(value, func) & mask;
                writeln!(output, "{}", res).unwrap();
            }
            _ => panic!(),
        }
    }
}
