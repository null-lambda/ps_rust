use std::{io::Write, iter};

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

struct MergeSortTree<T> {
    n: usize,
    max_depth: u32,
    data: Vec<Vec<T>>,
}

use std::iter::*;
use std::ops::Range;

impl<T: Ord + Copy> MergeSortTree<T> {
    fn from_iter<I: IntoIterator<Item = T>>(mut n: usize, iter: I) -> Self {
        n = n.next_power_of_two();
        let max_depth = usize::BITS - n.leading_zeros();
        let mut data: Vec<Vec<T>> = (0..2 * n).map(|_| Vec::new()).collect();

        for (i, x) in iter.into_iter().enumerate() {
            data[n + i].push(x);
        }

        for i in (1..n).rev() {
            pub fn merge<T: Ord + Copy>(xs: &[T], ys: &[T]) -> Vec<T> {
                if xs.len() == 0 {
                    return ys.to_vec();
                } else if ys.len() == 0 {
                    return xs.to_vec();
                }

                let mut result = Vec::with_capacity(xs.len() + ys.len());
                let (mut i, mut j) = (0, 0);
                loop {
                    if xs[i] < ys[j] {
                        result.push(xs[i]);
                        i += 1;
                        if i == xs.len() {
                            result.extend(ys[j..].into_iter());
                            return result;
                        }
                    } else {
                        result.push(ys[j]);
                        j += 1;
                        if j == ys.len() {
                            result.extend(xs[i..].into_iter());
                            return result;
                        }
                    }
                }
            }
            data[i] = merge(&data[i * 2], &data[i * 2 + 1]);
        }

        Self { n, max_depth, data }
    }

    fn count_gt(&self, range: Range<usize>, cutoff: T) -> usize {
        let Range { mut start, mut end } = range;
        start += self.n;
        end += self.n;
        let mut result = 0;
        while start < end {
            if start % 2 == 1 {
                result +=
                    self.data[start].len() - self.data[start].partition_point(|&x| !(x > cutoff));
            }
            if end % 2 == 1 {
                result += self.data[end - 1].len()
                    - self.data[end - 1].partition_point(|&x| !(x > cutoff));
            }
            start = (start + 1) >> 1;
            end >>= 1;
        }
        result
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let y: usize = input.value();
    let n: usize = input.value();

    let xs_orig: Vec<_> = (0..y).map(|_| input.value::<u32>()).collect();
    let xs = MergeSortTree::from_iter(y, xs_orig.iter().copied());

    for _ in 0..n {
        let a = input.value::<usize>() - 1;
        let p: u32 = input.value();
        let f = input.value::<usize>();
        let e = a + f + 1;

        let ans = if p <= xs_orig[a] {
            0
        } else {
            xs.count_gt(a..e, p - 1)
        };
        writeln!(output, "{}", ans).unwrap();
    }
}
