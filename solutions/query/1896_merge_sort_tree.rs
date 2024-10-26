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
            let idx = self.iter().position(|&c| !is_whitespace(c)).unwrap();
            //.expect("no available tokens left");
            *self = &self[idx..];
            let idx = self
                .iter()
                .position(|&c| is_whitespace(c))
                .unwrap_or_else(|| self.len());
            let (token, buf_new) = self.split_at(idx);
            *self = buf_new;
            token
        }

        fn line(&mut self) -> &[u8] {
            let idx = self
                .iter()
                .position(|&c| c == b'\n')
                .map(|idx| idx + 1)
                .unwrap_or_else(|| self.len());
            let (line, buf_new) = self.split_at(idx);
            *self = buf_new;
            trim_newline(line)
        }
    }
}

use std::io::{BufReader, Read, Write};
// use std::io::Write;

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
        let mut result = end - start;
        start += self.n;
        end += self.n;
        while start < end {
            if start % 2 == 1 {
                result -= self.data[start].partition_point(|&x| x <= cutoff);
            }
            if end % 2 == 1 {
                result -= self.data[end - 1].partition_point(|&x| x <= cutoff);
            }
            start = (start + 1) >> 1;
            end >>= 1;
        }
        result
    }
}

fn stdin() -> Vec<u8> {
    let stdin = std::io::stdin();
    let mut reader = BufReader::new(stdin.lock());

    let mut input_buf: Vec<u8> = vec![];
    reader.read_to_end(&mut input_buf).unwrap();
    input_buf
}

fn main() {
    use io::InputStream;
    let input_buf = stdin();
    let mut input: &[u8] = &input_buf[..];

    let mut output_buf = Vec::<u8>::new();

    let n = input.value();
    let mut indices = (0..n as u32)
        .map(|i| (input.value::<u32>(), i))
        .collect::<Vec<_>>();
    indices.sort_unstable();

    const INF: u32 = u32::MAX;
    let mut next = vec![0; n];
    let mut indices = indices.into_iter().peekable();
    while let Some((x, i)) = indices.next() {
        next[i as usize] = match indices.peek() {
            Some(&(x_next, i_next)) if x == x_next => i_next,
            _ => INF,
        };
    }

    let mut tree = MergeSortTree::from_iter(n, next);
    let n_queries = input.value();
    let mut result: i32 = 0;
    for _ in 0..n_queries {
        let left: usize = (result + input.value::<i32>() - 1) as usize;
        let right: usize = input.value::<usize>() - 1;
        result = tree.count_gt(left..right + 1, right as u32) as i32;
        writeln!(output_buf, "{}", result).unwrap();
    }

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
