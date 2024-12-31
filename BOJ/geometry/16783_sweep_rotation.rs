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
            token.parse().unwrap()
        }
    }

    #[inline]
    fn is_whitespace(c: u8) -> bool {
        c <= b' '
    }

    fn trim_newline(s: &[u8]) -> &[u8] {
        let mut s = s;
        while matches!(s.last(), Some(b'\n' | b'\r' | 0)) {
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
                .map_or_else(|| self.len(), |idx| idx + 1);
            let (line, buf_new) = self.split_at(idx);
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

pub mod segtree {
    use std::ops::Range;
    pub trait Monoid {
        fn id() -> Self;
        fn op(self, rhs: Self) -> Self;
    }

    #[derive(Debug)]
    pub struct SegTree<T> {
        n: usize,
        sum: Vec<T>,
    }

    impl<T> SegTree<T>
    where
        T: Monoid + Copy + Eq,
    {
        pub fn from_iter(n: usize, iter: impl IntoIterator<Item = T>) -> Self {
            use std::iter::repeat;
            let n = n.next_power_of_two();
            let mut sum: Vec<T> = repeat(T::id())
                .take(n)
                .chain(iter)
                .chain(repeat(T::id()))
                .take(2 * n)
                .collect();
            for i in (0..n).rev() {
                sum[i] = sum[i << 1].op(sum[i << 1 | 1]);
            }
            Self { n, sum }
        }

        pub fn set(&mut self, mut idx: usize, value: T) {
            debug_assert!(idx < self.n);
            idx += self.n;
            self.sum[idx] = value;
            while idx > 1 {
                idx >>= 1;
                self.sum[idx] = self.sum[idx << 1].op(self.sum[idx << 1 | 1]);
            }
        }

        #[inline]
        pub fn get(&self, idx: usize) -> T {
            self.sum[idx + self.n]
        }

        // sum on interval [left, right)
        pub fn query_range(&self, range: Range<usize>) -> T {
            let Range { mut start, mut end } = range;
            debug_assert!(start < self.n && end <= self.n);
            start += self.n;
            end += self.n;
            let (mut result_left, mut result_right) = (T::id(), T::id());
            while start < end {
                if start & 1 != 0 {
                    result_left = result_left.op(self.sum[start]);
                }
                if end & 1 != 0 {
                    result_right = self.sum[end - 1].op(result_right);
                }
                start = (start + 1) >> 1;
                end >>= 1;
            }

            result_left.op(result_right)
        }

        pub fn query_all(&self) -> T {
            self.sum[1]
        }
    }
}

use segtree::{Monoid, SegTree};

impl Monoid for i64 {
    fn id() -> Self {
        0
    }
    fn op(self, other: Self) -> Self {
        self + other
    }
}

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
struct MaximalConsecutiveSum<T: Monoid + Ord + Copy> {
    sum: T,
    full: T,
    left: T,
    right: T,
}

impl<T: Monoid + Ord + Copy> MaximalConsecutiveSum<T> {
    fn new(value: T) -> Self {
        Self {
            sum: value,
            full: value,
            left: value,
            right: value,
        }
    }
}

impl<T: Monoid + Ord + Copy> Monoid for MaximalConsecutiveSum<T> {
    fn id() -> Self {
        Self::new(T::id())
    }
    fn op(self, other: Self) -> Self {
        Self {
            sum: self.sum.max(other.sum).max(self.right.op(other.left)),
            full: self.full.op(other.full),
            left: self.left.max(self.full.op(other.left)),
            right: other.right.max(self.right.op(other.full)),
        }
    }
}

fn main() {
    use io::InputStream;
    let input_buf = stdin();
    let mut input: &[u8] = &input_buf[..];

    let mut output_buf = Vec::<u8>::new();

    let n: usize = input.value();
    let mut mines: Vec<(i64, i64, i64)> = (0..n)
        .map(|_| (input.value(), input.value(), input.value()))
        .collect();
    mines.sort_unstable_by_key(|&(x, y, _)| (-y, -x));

    let (points, scores): (Vec<_>, Vec<_>) = mines
        .into_iter()
        .inspect(|_| ())
        .map(|(x, y, w)| ((x, y), w))
        .unzip();
    // println!("{}", points.iter().map(|x| format!("{:?}", x)).collect::<Vec<_>>().join("\n"));

    let mut index_map: Vec<usize> = (0..n).collect();
    let mut events: Vec<(usize, usize)> = vec![];
    for (i, &(xi, yi)) in points.iter().enumerate() {
        for (j, &(xj, yj)) in points[..i].iter().enumerate() {
            events.push(if yi < yj || yi == yj && xi < xj {
                (i, j)
            } else {
                (j, i)
            });
        }
    }

    let sub_point = |(x1, y1), (x2, y2)| (x1 - x2, y1 - y2);
    let cross = |(dx1, dy1), (dx2, dy2)| dx1 * dy2 - dx2 * dy1;
    let cmp_angle = |(i1, j1), (i2, j2)| {
        0.cmp(&cross(
            sub_point(points[j1], points[i1]),
            sub_point(points[j2], points[i2]),
        ))
    };

    // sort lines by angle,
    // pretending that the points are slightly perturbed so that no points are colinear.
    events.sort_by(|&e1, &e2| cmp_angle(e1, e2).then_with(|| e1.cmp(&e2)));

    let mut segtree =
        SegTree::from_iter(n, scores.into_iter().map(|w| MaximalConsecutiveSum::new(w)));
    let mut it_events = events.into_iter().peekable();
    let mut result = 0;
    result = result.max(segtree.query_all().sum);

    while let Some(event1) = it_events.next() {
        let (i, j) = event1;
        let (wi, wj) = (segtree.get(index_map[i]), segtree.get(index_map[j]));
        segtree.set(index_map[j], wi);
        segtree.set(index_map[i], wj);
        index_map.swap(i, j);
        if !matches!(it_events.peek(), Some(&event2) if cmp_angle(event1, event2).is_eq()) {
            result = result.max(segtree.query_all().sum);
        }
    }
    println!("{:?}", result);

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
