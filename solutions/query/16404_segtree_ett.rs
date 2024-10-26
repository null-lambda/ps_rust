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

mod segtree {
    pub trait Monoid {
        fn id() -> Self;
        fn op(self, rhs: Self) -> Self;
    }

    pub struct SegTree<T: Monoid> {
        n: usize,
        data: Vec<T>,
    }

    impl<T: Monoid + Copy> SegTree<T> {
        pub fn from_iter<I>(n: usize, iter: I) -> Self
        where
            I: Iterator<Item = T>,
        {
            use std::iter::repeat;
            let mut data: Vec<T> = repeat(T::id()).take(n).chain(iter).collect();
            data.resize(2 * n, T::id());
            Self { n, data }
        }

        pub fn apply_range(&mut self, mut start: usize, mut end: usize, value: T) {
            debug_assert!(end <= self.n);
            start += self.n;
            end += self.n;

            while start < end {
                if start & 1 != 0 {
                    self.data[start] = value.op(self.data[start]);
                }
                if end & 1 != 0 {
                    self.data[end - 1] = self.data[end - 1].op(value);
                }
                start = (start + 1) >> 1;
                end >>= 1;
            }
        }

        pub fn query(&self, mut idx: usize) -> T {
            idx += self.n;
            let mut result = self.data[idx];
            while idx > 1 {
                idx >>= 1;
                result = result.op(self.data[idx]);
            }
            result
        }
    }
}

fn main() {
    use io::InputStream;
    let input_buf = stdin();
    let mut input: &[u8] = &input_buf[..];

    let mut output_buf = Vec::<u8>::new();

    let n: usize = input.value();
    let m: usize = input.value();
    let mut childs: Vec<Vec<usize>> = (0..n).map(|_| Vec::new()).collect();
    for u in 0..n {
        let v: isize = input.value();
        if v != -1 {
            childs[v as usize - 1].push(u);
        }
    }

    // euler tree technique
    let intervals = {
        let mut result = vec![(0, 0); n];
        fn dfs(
            node: usize,
            order: &mut usize,
            childs: &Vec<Vec<usize>>,
            result: &mut Vec<(usize, usize)>,
        ) {
            result[node].0 = *order;
            *order += 1;
            for &v in &childs[node] {
                dfs(v, order, childs, result);
            }
            result[node].1 = *order;
        }
        dfs(0, &mut 0, &childs, &mut result);
        result
    };

    use segtree::{Monoid, SegTree};
    impl Monoid for i32 {
        fn id() -> Self {
            0
        }
        fn op(self, other: Self) -> Self {
            self + other
        }
    }

    let mut segtree = SegTree::from_iter(n, None.into_iter());
    for _ in 0..m {
        let q: u8 = input.value();
        let i: usize = input.value();
        match q {
            1 => {
                let w: i32 = input.value();
                segtree.apply_range(intervals[i - 1].0, intervals[i - 1].1, w);
            }
            2 => {
                writeln!(output_buf, "{}", segtree.query(intervals[i - 1].0)).unwrap();
            }
            _ => unreachable!(),
        }
    }

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
