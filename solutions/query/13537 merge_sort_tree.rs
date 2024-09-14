mod io {
    use std::fmt::Debug;
    use std::str::*;

    pub trait InputStream {
        fn token(&mut self) -> &[u8];
        fn line(&mut self) -> &[u8];

        fn skip_line(&mut self) {
            self.line();
        }

        #[inline]
        fn value<T>(&mut self) -> T
        where
            T: FromStr,
            T::Err: Debug,
        {
            let token = self.token();
            let token = unsafe { from_utf8_unchecked(token) };
            token.parse::<T>().unwrap()
        }
    }

    #[inline]
    fn is_whitespace(c: u8) -> bool {
        c <= b' '
    }

    fn trim_newline(s: &[u8]) -> &[u8] {
        let mut s = s;
        while s
            .last()
            .map(|&c| {
                matches! {c, b'\n' | b'\r' | 0}
            })
            .unwrap_or_else(|| false)
        {
            s = &s[..s.len() - 1];
        }
        s
    }

    impl InputStream for &[u8] {
        fn token(&mut self) -> &[u8] {
            let i = self.iter().position(|&c| !is_whitespace(c)).unwrap();
            //.expect("no available tokens left");
            *self = &self[i..];
            let i = self
                .iter()
                .position(|&c| is_whitespace(c))
                .unwrap_or_else(|| self.len());
            let (token, buf_new) = self.split_at(i);
            *self = buf_new;
            token
        }

        fn line(&mut self) -> &[u8] {
            let i = self
                .iter()
                .position(|&c| c == b'\n')
                .map(|i| i + 1)
                .unwrap_or_else(|| self.len());
            let (line, buf_new) = self.split_at(i);
            *self = buf_new;
            trim_newline(line)
        }
    }
}

use std::io::{BufReader, Read, Write};

fn stdin() -> Vec<u8> {
    let stdin = std::io::stdin();
    let mut reader = BufReader::new(stdin.lock());

    let mut input_buf: Vec<u8> = vec![];
    reader.read_to_end(&mut input_buf).unwrap();
    input_buf
}

#[derive(Debug)]
struct MergeSortTree<T> {
    n: usize,
    max_depth: u32,
    data: Vec<Vec<T>>,
}

use std::iter::*;
use std::ops::Range;

impl<T: Ord + Copy + std::fmt::Debug> MergeSortTree<T> {
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
    use io::InputStream;
    let input_buf = stdin();
    let mut input: &[u8] = &input_buf[..];

    let mut output_buf = Vec::<u8>::new();

    let n = input.value();
    let xs = MergeSortTree::from_iter(n, (0..n).map(|_| input.value::<u32>()));

    let n_queries = input.value();
    for _ in 0..n_queries {
        let i: usize = input.value();
        let j: usize = input.value();
        let k: u32 = input.value();
        writeln!(output_buf, "{}", xs.count_gt(i - 1..j, k)).unwrap();
    }

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
