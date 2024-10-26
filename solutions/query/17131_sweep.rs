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

fn stdin() -> Vec<u8> {
    let stdin = std::io::stdin();
    let mut reader = BufReader::new(stdin.lock());

    let mut input_buf: Vec<u8> = vec![];
    reader.read_to_end(&mut input_buf).unwrap();
    input_buf
}

use std::ops::Range;

// binary operation with identity and associative property
// axioms:
//    op(op(a, b), c) = op(a, op(b, c))
//    op(a, id) = id
//    op(id, a) = a
trait Monoid {
    fn id() -> Self;
    fn op(self, rhs: Self) -> Self;
    fn op_assign(&mut self, rhs: Self);
}

struct SegmentTree<T: Monoid> {
    n: usize,
    data: Vec<T>,
}

impl<T: Monoid + Copy> SegmentTree<T> {
    fn from_sized_iter<I>(iter: I) -> Self
    where
        T: Clone,
        I: IntoIterator<Item = T> + ExactSizeIterator,
    {
        let n = iter.len();
        let mut data = Vec::with_capacity(2 * n);
        data.resize(n, T::id());
        data.extend(iter);

        let mut tree = Self { n, data };
        tree.init();
        tree
    }

    fn init(&mut self) {
        for i in (1..self.n).rev() {
            self.data[i] = T::op(self.data[i << 1], self.data[i << 1 | 1]);
        }
    }

    fn get(&self, idx: usize) -> T {
        self.data[idx + self.n]
    }

    // sum on interval [left, right)
    fn query_sum(&self, Range { mut start, mut end }: Range<usize>) -> T {
        debug_assert!(end <= self.n);
        start += self.n;
        end += self.n;

        let mut result = T::id();
        while start < end {
            if start & 1 != 0 {
                result.op_assign(self.data[start]);
                start += 1;
            }
            if end & 1 != 0 {
                result.op_assign(self.data[end - 1]);
                end -= 1;
            }
            start >>= 1;
            end >>= 1;
        }
        result
    }

    fn update(&mut self, mut idx: usize, value: T) {
        idx += self.n;
        self.data[idx] = value;
        while idx > 1 {
            self.data[idx >> 1] = T::op(self.data[idx], self.data[idx ^ 1]);
            idx >>= 1;
        }
    }
}

fn main() {
    use io::InputStream;
    let input_buf = stdin();
    let mut input: &[u8] = &input_buf[..];

    let mut output_buf = Vec::<u8>::new();

    #[derive(Clone, Copy)]
    struct AddGroup(u64);

    impl Monoid for AddGroup {
        fn id() -> Self {
            Self(0)
        }
        fn op(self, rhs: Self) -> Self {
            Self(self.0 + rhs.0)
        }
        fn op_assign(&mut self, rhs: Self) {
            *self = self.op(rhs);
        }
    }

    use std::cmp;
    let n: usize = input.value();
    let mut points: Vec<(i32, i32)> = (0..n).map(|_| (input.value(), input.value())).collect();

    // compress x coordinates, preserving the entire poset structure
    points.sort_unstable_by_key(|&(x, _)| x);

    let mut prev = points[0].0;
    points[0].0 = 0;
    for i in 1..n {
        let current = points[i].0;
        points[i].0 = if current == prev {
            points[i - 1].0
        } else {
            points[i - 1].0 + 1
        };
        prev = current;
    }
    let x_bound = (points.last().unwrap().0 + 1) as usize;

    points.sort_by_key(|&(_, y)| cmp::Reverse(y));

    let mut seg_tree = SegmentTree::from_sized_iter((0..x_bound).map(|_| AddGroup(0)));

    // iter::group_by?
    let mut total: u64 = 0;
    let mut i_group_start = 0;
    for (i, &(x, _)) in points.iter().enumerate() {
        total +=
            seg_tree.query_sum(0..x as usize).0 * seg_tree.query_sum((x + 1) as usize..x_bound).0;
        total %= 1_000_000_007;

        let group_key = |&(_, y)| y;
        if i >= n - 1 || group_key(&points[i]) != group_key(&points[i + 1]) {
            // group finished
            for &(x, _) in points[i_group_start..=i].iter() {
                seg_tree.update(x as usize, seg_tree.get(x as usize).op(AddGroup(1)));
            }
            i_group_start = i + 1;
            // println!("{:?}", seg_tree.data[n..2 * n].iter().map(|x| x.0).collect::<Vec<_>>());
        }
    }

    writeln!(output_buf, "{:?}", total).unwrap();

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
