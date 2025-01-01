use std::io::Write;

use segtree::Monoid;

mod fast_io {
    use std::fs::File;
    use std::io::BufWriter;
    use std::os::unix::io::FromRawFd;

    extern "C" {
        fn mmap(addr: usize, length: usize, prot: i32, flags: i32, fd: i32, offset: i64)
            -> *mut u8;
        fn fstat(fd: i32, stat: *mut usize) -> i32;
    }

    pub struct InputAtOnce {
        buf: &'static [u8],
    }

    impl InputAtOnce {
        fn skip(&mut self) {
            loop {
                match self.buf {
                    &[..=b' ', ..] => self.buf = &self.buf[1..],
                    _ => break,
                }
            }
        }

        fn u32_noskip(&mut self) -> u32 {
            let mut acc = 0;
            loop {
                match self.buf {
                    &[b'0'..=b'9', ..] => acc = acc * 10 + (self.buf[0] - b'0') as u32,
                    _ => break,
                }
                self.buf = &self.buf[1..];
            }
            acc
        }

        pub fn token(&mut self) -> &'static str {
            self.skip();
            let start = self.buf.as_ptr();
            loop {
                match self.buf {
                    &[..=b' ', ..] => break,
                    _ => self.buf = &self.buf[1..],
                }
            }
            let end = self.buf.as_ptr();
            unsafe {
                std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                    start,
                    end.offset_from(start) as usize,
                ))
            }
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().unwrap()
        }

        pub fn u32(&mut self) -> u32 {
            self.skip();
            self.u32_noskip()
        }

        pub fn i32(&mut self) -> i32 {
            self.skip();
            match self.buf {
                &[b'-', ..] => {
                    self.buf = &self.buf[1..];
                    -(self.u32_noskip() as i32)
                }
                _ => self.u32_noskip() as i32,
            }
        }
    }

    pub fn stdin() -> InputAtOnce {
        let mut stat = [0; 18];
        unsafe { fstat(0, (&mut stat).as_mut_ptr()) };
        let buf = unsafe { mmap(0, stat[6], 1, 2, 0, 0) };
        let buf =
            unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(buf, stat[6])) };
        InputAtOnce {
            buf: buf.as_bytes(),
        }
    }

    pub fn stdout() -> BufWriter<File> {
        let stdout = unsafe { File::from_raw_fd(1) };
        BufWriter::with_capacity(1 << 16, stdout)
    }
}

pub mod segtree {
    use std::ops::Range;

    pub trait Monoid {
        type X;
        const IS_COMMUTATIVE: bool = false;
        fn id(&self) -> Self::X;
        fn op(&self, a: &Self::X, b: &Self::X) -> Self::X;
    }

    #[derive(Debug)]
    pub struct SegTree<M>
    where
        M: Monoid,
    {
        n: usize,
        sum: Vec<M::X>,
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

        pub fn from_iter<I>(iter: I, monoid: M) -> Self
        where
            I: IntoIterator<Item = M::X>,
            I::IntoIter: ExactSizeIterator<Item = M::X>,
        {
            let iter = iter.into_iter();
            let n = iter.len();
            let mut sum: Vec<_> = (0..n).map(|_| monoid.id()).chain(iter).collect();
            for i in (0..n).rev() {
                sum[i] = monoid.op(&sum[i << 1], &sum[i << 1 | 1]);
            }
            Self { n, sum, monoid }
        }

        pub fn set(&mut self, mut idx: usize, value: M::X) {
            debug_assert!(idx < self.n);
            idx += self.n;
            self.sum[idx] = value;
            while idx > 1 {
                idx >>= 1;
                self.sum[idx] = self.monoid.op(&self.sum[idx << 1], &self.sum[idx << 1 | 1]);
            }
        }

        pub fn get(&self, idx: usize) -> &M::X {
            &self.sum[idx + self.n]
        }

        pub fn mapped_sum_range<N: Monoid>(
            &self,
            range: Range<usize>,
            codomain: &N,
            morphism: impl Fn(&M::X) -> N::X,
        ) -> N::X {
            let Range { mut start, mut end } = range;
            if start >= end {
                return codomain.id();
            }
            debug_assert!(start < self.n && end <= self.n);
            start += self.n;
            end += self.n;

            if N::IS_COMMUTATIVE {
                let mut result = codomain.id();
                while start < end {
                    if start & 1 != 0 {
                        result = codomain.op(&result, &morphism(&self.sum[start]));
                    }
                    if end & 1 != 0 {
                        result = codomain.op(&morphism(&self.sum[end - 1]), &result);
                    }
                    start = (start + 1) >> 1;
                    end >>= 1;
                }
                result
            } else {
                let (mut result_left, mut result_right) = (codomain.id(), codomain.id());
                while start < end {
                    if start & 1 != 0 {
                        result_left = codomain.op(&result_left, &morphism(&self.sum[start]));
                    }
                    if end & 1 != 0 {
                        result_right = codomain.op(&morphism(&self.sum[end - 1]), &result_right);
                    }
                    start = (start + 1) >> 1;
                    end >>= 1;
                }
                codomain.op(&result_left, &result_right)
            }
        }
    }

    impl<M: Monoid> SegTree<M>
    where
        M::X: Clone,
    {
        pub fn sum_range(&self, range: Range<usize>) -> M::X {
            self.mapped_sum_range(range, &self.monoid, |x| x.clone())
        }
    }
}

const NEG_INF: i64 = -(1 << 56);

#[derive(Clone)]
struct IntervalSum {
    sum: i64,
    max: i64,
    max_left: i64,
    max_right: i64,
}

impl IntervalSum {
    fn singleton(x: i64) -> Self {
        Self {
            sum: x,
            max: x,
            max_left: x,
            max_right: x,
        }
    }
}

struct IntervalSumMonoid;

impl Monoid for IntervalSumMonoid {
    type X = IntervalSum;
    fn id(&self) -> Self::X {
        Self::X {
            sum: 0,
            max: NEG_INF,
            max_left: NEG_INF,
            max_right: NEG_INF,
        }
    }
    fn op(&self, a: &Self::X, b: &Self::X) -> Self::X {
        IntervalSum {
            sum: a.sum + b.sum,
            max: a.max.max(b.max).max(a.max_right + b.max_left),
            max_left: a.max_left.max(a.sum + b.max_left),
            max_right: b.max_right.max(b.sum + a.max_right),
        }
    }
}

// Projections of IntervalSum (unnecessary, only for micro-optimizations) : max_left, max_right and sum
#[derive(Clone)]
struct PrefixSum {
    max_left: i64,
    sum: i64,
}

struct PrefixSumMonoid;

impl Monoid for PrefixSumMonoid {
    type X = PrefixSum;
    fn id(&self) -> Self::X {
        Self::X {
            max_left: NEG_INF,
            sum: 0,
        }
    }
    fn op(&self, a: &Self::X, b: &Self::X) -> Self::X {
        PrefixSum {
            max_left: a.max_left.max(a.sum + b.max_left),
            sum: a.sum + b.sum,
        }
    }
}

#[derive(Clone)]
struct PostfixSum {
    max_right: i64,
    sum: i64,
}

struct PostfixSumMonoid;

impl Monoid for PostfixSumMonoid {
    type X = PostfixSum;
    fn id(&self) -> Self::X {
        Self::X {
            max_right: NEG_INF,
            sum: 0,
        }
    }
    fn op(&self, a: &Self::X, b: &Self::X) -> Self::X {
        PostfixSum {
            max_right: b.max_right.max(b.sum + a.max_right),
            sum: a.sum + b.sum,
        }
    }
}

struct Additive;

impl Monoid for Additive {
    type X = i64;
    const IS_COMMUTATIVE: bool = true;
    fn id(&self) -> Self::X {
        0
    }
    fn op(&self, a: &Self::X, b: &Self::X) -> Self::X {
        a + b
    }
}

impl IntervalSum {
    fn proj_left(&self) -> PrefixSum {
        PrefixSum {
            max_left: self.max_left,
            sum: self.sum,
        }
    }

    fn proj_right(&self) -> PostfixSum {
        PostfixSum {
            max_right: self.max_right,
            sum: self.sum,
        }
    }
}

fn main() {
    let mut input = fast_io::stdin();
    let mut output = fast_io::stdout();

    let n = input.value();
    let xs = (0..n).map(|_| input.i32() as i64);
    let tree = segtree::SegTree::from_iter(xs.map(IntervalSum::singleton), IntervalSumMonoid);

    let m = input.value();
    for _ in 0..m {
        let x1 = input.u32() as usize - 1;
        let y1 = input.u32() as usize - 1;
        let x2 = input.u32() as usize - 1;
        let y2 = input.u32() as usize - 1;
        let ans = if y1 < x2 {
            tree.mapped_sum_range(x1..y1 + 1, &PostfixSumMonoid, IntervalSum::proj_right)
                .max_right
                + tree.mapped_sum_range(y1 + 1..x2, &Additive, |x| x.sum)
                + tree
                    .mapped_sum_range(x2..y2 + 1, &PrefixSumMonoid, IntervalSum::proj_left)
                    .max_left
        } else {
            let left = tree
                .mapped_sum_range(x1..x2, &PostfixSumMonoid, IntervalSum::proj_right)
                .max_right;
            let mid = tree.sum_range(x2..y1 + 1);
            let right = tree
                .mapped_sum_range(y1 + 1..y2 + 1, &PrefixSumMonoid, IntervalSum::proj_left)
                .max_left;
            mid.max
                .max(left + mid.max_left)
                .max(mid.max_right + right)
                .max(left + mid.sum + right)
        };
        writeln!(output, "{}", ans).unwrap();
    }
}
