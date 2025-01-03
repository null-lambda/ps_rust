use std::{cmp::Ordering, io::Write};

use segtree_lazy::MonoidAction;

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

pub mod segtree_lazy {
    use std::{iter, ops::Range};

    pub trait MonoidAction {
        type X;
        type F;
        const IS_X_COMMUTATIVE: bool = false; // TODO
        fn id(&self) -> Self::X;
        fn combine(&self, lhs: &Self::X, rhs: &Self::X) -> Self::X;
        fn id_action(&self) -> Self::F;
        fn combine_action(&self, lhs: &Self::F, rhs: &Self::F) -> Self::F;
        fn apply_to_sum(&self, f: &Self::F, x_count: u32, x_sum: &mut Self::X);
    }

    pub struct SegTree<M: MonoidAction> {
        n: usize,
        max_height: u32,
        sum: Vec<M::X>,
        lazy: Vec<M::F>,
        ma: M,
    }

    impl<M: MonoidAction> SegTree<M> {
        pub fn with_size(n: usize, ma: M) -> Self {
            Self {
                n,
                max_height: usize::BITS - n.leading_zeros(),
                sum: iter::repeat_with(|| ma.id()).take(2 * n).collect(),
                lazy: iter::repeat_with(|| ma.id_action()).take(n).collect(),
                ma,
            }
        }

        pub fn from_iter<I>(iter: I, ma: M) -> Self
        where
            I: IntoIterator<Item = M::X>,
            I::IntoIter: ExactSizeIterator,
        {
            let iter = iter.into_iter();
            let n = iter.len();
            let mut sum: Vec<_> = (iter::repeat_with(|| ma.id()).take(n))
                .chain(
                    iter.into_iter()
                        .chain(iter::repeat_with(|| ma.id()))
                        .take(n),
                )
                .collect();
            for i in (1..n).rev() {
                sum[i] = ma.combine(&sum[i << 1], &sum[i << 1 | 1]);
            }
            Self {
                n,
                max_height: usize::BITS - n.leading_zeros(),
                sum,
                lazy: iter::repeat_with(|| ma.id_action()).take(n).collect(),
                ma,
            }
        }

        fn apply(&mut self, idx: usize, width: u32, value: &M::F) {
            self.ma.apply_to_sum(&value, width, &mut self.sum[idx]);
            if idx < self.n {
                self.lazy[idx] = self.ma.combine_action(&value, &self.lazy[idx]);
            }
        }

        fn push_down(&mut self, width: u32, node: usize) {
            let value = unsafe { &*(&self.lazy[node] as *const _) };
            self.apply(node << 1, width, value);
            self.apply(node << 1 | 1, width, value);
            self.lazy[node] = self.ma.id_action();
        }

        fn push_range(&mut self, range: Range<usize>) {
            let Range { mut start, mut end } = range;
            start += self.n;
            end += self.n;

            let start_height = 1 + start.trailing_zeros();
            let end_height = 1 + end.trailing_zeros();
            for height in (start_height..=self.max_height).rev() {
                let width = 1 << height - 1;
                self.push_down(width, start >> height);
            }
            for height in (end_height..=self.max_height).rev().skip_while(|&height| {
                height >= start_height && end - 1 >> height == start >> height
            }) {
                let width = 1 << height - 1;
                self.push_down(width, end - 1 >> height);
            }
        }

        fn pull_up(&mut self, node: usize) {
            self.sum[node] = (self.ma).combine(&self.sum[node << 1], &self.sum[node << 1 | 1]);
        }

        pub fn apply_range(&mut self, range: Range<usize>, value: M::F) {
            let Range { mut start, mut end } = range;
            debug_assert!(start <= end && end <= self.n);
            if start == end {
                return;
            }

            self.push_range(range);
            start += self.n;
            end += self.n;
            let mut width: u32 = 1;
            let (mut pull_start, mut pull_end) = (false, false);
            while start < end {
                if pull_start {
                    self.pull_up(start - 1);
                }
                if pull_end {
                    self.pull_up(end);
                }
                if start & 1 != 0 {
                    self.apply(start, width, &value);
                    start += 1;
                    pull_start = true;
                }
                if end & 1 != 0 {
                    self.apply(end - 1, width, &value);
                    pull_end = true;
                }
                start >>= 1;
                end >>= 1;
                width <<= 1;
            }
            start -= 1;
            while end > 0 {
                if pull_start {
                    self.pull_up(start);
                }
                if pull_end && !(pull_start && start == end) {
                    self.pull_up(end);
                }
                start >>= 1;
                end >>= 1;
                width <<= 1;
            }
        }

        pub fn query_range(&mut self, range: Range<usize>) -> M::X {
            let Range { mut start, mut end } = range;

            self.push_range(range);
            start += self.n;
            end += self.n;
            if M::IS_X_COMMUTATIVE {
                let mut result = self.ma.id();
                while start < end {
                    if start & 1 != 0 {
                        result = self.ma.combine(&result, &self.sum[start]);
                        start += 1;
                    }
                    if end & 1 != 0 {
                        end -= 1;
                        result = self.ma.combine(&result, &self.sum[end]);
                    }
                    start >>= 1;
                    end >>= 1;
                }
                result
            } else {
                let (mut result_left, mut result_right) = (self.ma.id(), self.ma.id());
                while start < end {
                    if start & 1 != 0 {
                        result_left = self.ma.combine(&result_left, &self.sum[start]);
                    }
                    if end & 1 != 0 {
                        result_right = self.ma.combine(&self.sum[end - 1], &result_right);
                    }
                    start = (start + 1) >> 1;
                    end >>= 1;
                }
                self.ma.combine(&result_left, &result_right)
            }
        }

        pub fn query_all(&mut self) -> &M::X {
            assert!(self.n.is_power_of_two());
            self.push_down(self.n as u32, 1);
            &self.sum[1]
        }
    }
}

struct MinCount {
    min: i32,
    count: i32,
}

impl MinCount {
    fn zero() -> Self {
        Self { min: 0, count: 1 }
    }
}

struct MinCountOp;

impl MonoidAction for MinCountOp {
    type X = MinCount;
    type F = i32;
    const IS_X_COMMUTATIVE: bool = true;

    fn id(&self) -> Self::X {
        MinCount {
            min: i32::MAX,
            count: 0,
        }
    }

    fn combine(&self, lhs: &Self::X, rhs: &Self::X) -> Self::X {
        let (min, count) = match lhs.min.cmp(&rhs.min) {
            Ordering::Less => (lhs.min, lhs.count),
            Ordering::Equal => (lhs.min, lhs.count + rhs.count),
            Ordering::Greater => (rhs.min, rhs.count),
        };
        MinCount { min, count }
    }

    fn id_action(&self) -> Self::F {
        0
    }

    fn combine_action(&self, lhs: &Self::F, rhs: &Self::F) -> Self::F {
        lhs + rhs
    }

    fn apply_to_sum(&self, f: &Self::F, _x_count: u32, x_sum: &mut Self::X) {
        x_sum.min += f;
    }
}

fn parse_state(s: &str) -> u8 {
    match s {
        "--" => b'-',
        "->" => b'>',
        "<-" => b'<',
        _ => panic!(),
    }
}

fn main() {
    let mut input = fast_io::stdin();
    let mut output = fast_io::stdout();

    let n = input.value();
    let mut degree = vec![0; n];
    let mut xor_neighbors = vec![(0, 0); n];
    let mut base_queries = vec![];
    for _ in 0..n - 1 {
        let u = input.u32() - 1;
        let w = parse_state(input.token());
        let v = input.u32() - 1;
        base_queries.push((u, w, v));
        degree[u as usize] += 1;
        degree[v as usize] += 1;
        xor_neighbors[u as usize].0 ^= v;
        xor_neighbors[v as usize].0 ^= u;
        xor_neighbors[u as usize].1 ^= w;
        xor_neighbors[v as usize].1 ^= w;
    }
    let root = 0;
    degree[root] += 2;

    let mut size = vec![1u32; n];
    let mut topological_order = vec![];
    let mut edge_state = vec![b'-'; n];
    for mut u in 0..n as u32 {
        while degree[u as usize] == 1 {
            let (p, w) = xor_neighbors[u as usize];
            xor_neighbors[p as usize].0 ^= u;
            xor_neighbors[p as usize].1 ^= w;
            degree[p as usize] -= 1;
            degree[u as usize] -= 1;
            topological_order.push((u, p));

            size[p as usize] += size[u as usize];

            u = p;
        }
    }

    let mut euler_in = size.clone(); // 1-based
    for (u, p) in topological_order.into_iter().rev() {
        let last_idx = euler_in[p as usize];
        euler_in[p as usize] -= size[u as usize];
        euler_in[u as usize] = last_idx;
    }

    let euler_in = |u: usize| euler_in[u] as usize - 1; // 0-based
    let euler_out = |u: usize| euler_in(u) + size[u] as usize;

    let n_pad = n.next_power_of_two();
    let mut min_count = segtree_lazy::SegTree::from_iter(
        (0..n_pad).map(|i| {
            if i < n {
                MinCount::zero()
            } else {
                MinCountOp.id()
            }
        }),
        MinCountOp,
    );

    let q = input.u32();
    let queries = base_queries
        .into_iter()
        .chain((0..q).map(|_| (input.u32() - 1, parse_state(input.token()), input.u32() - 1)))
        .map(|(u, w, v)| (u as usize, w, v as usize))
        .map(|(u, w, v)| {
            if euler_in(u) < euler_in(v) {
                (v, w)
            } else {
                let w_flip = match w {
                    b'-' => b'-',
                    b'<' => b'>',
                    b'>' => b'<',
                    _ => unreachable!(),
                };
                (u, w_flip)
            }
        });

    for (i, (v, w)) in queries.enumerate() {
        let w_old = edge_state[v];
        edge_state[v] = w;
        if w_old != w {
            let mut update = |w, delta| match w {
                b'-' => {}
                b'>' => {
                    min_count.apply_range(euler_in(v)..euler_out(v), delta);
                }
                b'<' => {
                    min_count.apply_range(0..n, delta);
                    min_count.apply_range(euler_in(v)..euler_out(v), -delta);
                }
                _ => unreachable!(),
            };

            update(w_old, -1);
            update(w, 1);
        }

        if i >= n - 1 {
            let x = min_count.query_all();
            let ans = if x.min == 0 { x.count } else { 0 };
            writeln!(output, "{}", ans).unwrap();
        }
    }
}
