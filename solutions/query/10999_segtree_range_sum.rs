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

// commutative monoid
trait CommMonoid {
    fn id() -> Self;
    fn op(self, rhs: Self) -> Self;
    #[inline]
    fn op_assign(&mut self, rhs: Self)
    where
        Self: Sized + Copy,
    {
        *self = self.op(rhs);
    }
    fn pow(self, n: u32) -> Self;
}

struct SegmentTree<T: CommMonoid> {
    n: usize,
    max_height: u32,
    sum: Vec<T>,
    lazy: Vec<T>,
}

impl<T: CommMonoid + Copy + Eq> SegmentTree<T> {
    fn from_iter<I>(n: usize, iter: I) -> Self
    where
        T: Clone,
        I: Iterator<Item = T>,
    {
        use std::iter::repeat;
        let n = n.next_power_of_two();
        let mut sum: Vec<T> = repeat(T::id()).take(n).chain(iter).collect();
        sum.resize(2 * n, T::id());
        for i in (0..n).rev() {
            sum[i] = sum[i << 1].op(sum[i << 1 | 1]);
        }
        Self {
            n,
            max_height: usize::BITS - n.leading_zeros(),
            sum,
            lazy: vec![T::id(); n],
        }
    }

    #[inline]
    fn apply(&mut self, node: usize, width: u32, value: T) {
        self.sum[node].op_assign(value.pow(width));
        if node < self.n {
            self.lazy[node].op_assign(value);
        }
    }

    #[inline]
    fn propagate_lazy(&mut self, mut idx: usize) {
        idx += self.n;
        for height in (1..=self.max_height).rev() {
            let node = idx >> height;
            if self.lazy[node] != T::id() {
                let width: u32 = 1 << (height - 1);
                self.apply(node << 1, width, self.lazy[node]);
                self.apply(node << 1 | 1, width, self.lazy[node]);
                self.lazy[node] = T::id();
            }
        }
    }

    #[inline]
    fn update_sum(&mut self, node: usize, width: u32) {
        self.sum[node] = self.sum[node << 1].op(self.sum[node << 1 | 1]);
        if self.lazy[node] != T::id() {
            self.sum[node].op_assign(self.lazy[node].pow(width));
        };
    }

    // sum on interval [left, right)
    fn apply_range(&mut self, mut start: usize, mut end: usize, value: T) {
        if value == T::id() || start == end {
            return;
        }
        debug_assert!(end <= self.n);
        self.propagate_lazy(start);
        self.propagate_lazy(end - 1);
        start += self.n;
        end += self.n;
        let mut width: u32 = 1;
        let (mut update_left, mut update_right) = (false, false);
        while start < end {
            if update_left {
                self.update_sum(start - 1, width);
            }
            if update_right {
                self.update_sum(end, width);
            }
            if start & 1 != 0 {
                self.apply(start, width, value);
                update_left = true;
            }
            if end & 1 != 0 {
                self.apply(end - 1, width, value);
                update_right = true;
            }
            start = (start + 1) >> 1;
            end >>= 1;
            width <<= 1;
        }
        start -= 1;
        while end > 0 {
            if update_left {
                self.update_sum(start, width);
            }
            if update_right && !(update_left && start == end) {
                self.update_sum(end, width);
            }
            start >>= 1;
            end >>= 1;
            width <<= 1;
        }
    }

    fn query_range(&mut self, mut start: usize, mut end: usize) -> T {
        self.propagate_lazy(start);
        self.propagate_lazy(end - 1);
        start += self.n;
        end += self.n;
        let mut result = T::id();

        while start < end {
            if start & 1 != 0 {
                result.op_assign(self.sum[start]);
            }
            if end & 1 != 0 {
                result.op_assign(self.sum[end - 1]);
            }
            start = (start + 1) >> 1;
            end >>= 1;
        }

        result
    }
}

#[test]
fn test_segtree() {
    let n = 5;
    let mut segtree = SegmentTree::from_iter(n, (0..n).map(|_| 1));
    segtree.apply_range(0, 2, -1);
    // segtree.apply_range(2, 3, -1);
    // segtree.apply_range(3, 5, -1);
    println!("{:?}", segtree.sum);
    println!("{:?}", segtree.lazy);
    for i in 1..=n {
        for j in 0..=n - i {
            println!("{:?}", (j, j + i, segtree.query_range(j, j + i)));
        }
    }
}

fn main() {
    use io::InputStream;
    let input_buf = stdin();
    let mut input: &[u8] = &input_buf[..];

    let mut output_buf = Vec::<u8>::new();

    let n = input.value();
    let m: usize = input.value();
    let k: usize = input.value();
    let xs = (0..n).map(|_| input.value());

    impl CommMonoid for i64 {
        fn id() -> Self {
            0
        }
        fn op(self, other: Self) -> Self {
            self + other
        }
        fn pow(self, n: u32) -> Self {
            self * (n as i64)
        }
    }

    let mut segtree = SegmentTree::from_iter(n, xs);

    for _ in 0..m + k {
        let a = input.value();
        let b: usize = input.value();
        let c: usize = input.value();
        match a {
            1 => {
                let d: i64 = input.value();
                segtree.apply_range(b - 1, c, d);
            }
            2 => {
                writeln!(output_buf, "{}", segtree.query_range(b - 1, c)).unwrap();
            }
            _ => unreachable!(),
        }
    }

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
