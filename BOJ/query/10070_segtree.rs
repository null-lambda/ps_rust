use std::io::Write;

mod simple_io {
    use std::string::*;

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

const INF: i32 = 1 << 30;
const NEG_INF: i32 = -1 << 30;

struct BoundsOp;

impl BoundsOp {
    fn singleton(x: i32) -> (i32, i32) {
        (x, x)
    }

    fn apply_min(x: i32) -> (i32, i32) {
        (NEG_INF, x)
    }

    fn apply_max(x: i32) -> (i32, i32) {
        (x, INF)
    }
}

impl segtree_lazy::MonoidAction for BoundsOp {
    type X = (i32, i32);
    type F = (i32, i32);
    const IS_X_COMMUTATIVE: bool = true;
    fn id(&self) -> Self::X {
        (INF, NEG_INF)
    }

    fn combine(&self, lhs: &Self::X, rhs: &Self::X) -> Self::X {
        ((*lhs).0.min((*rhs).0), (*lhs).1.max((*rhs).1))
    }

    fn id_action(&self) -> Self::F {
        (NEG_INF, INF)
    }

    fn combine_action(&self, lhs: &Self::F, rhs: &Self::F) -> Self::F {
        let clip = |x: i32| x.min(lhs.1).max(lhs.0);
        (clip(rhs.0), clip(rhs.1))
    }

    fn apply_to_sum(&self, f: &Self::F, _x_count: u32, x_sum: &mut Self::X) {
        let clip = |x: i32| x.min(f.1).max(f.0);
        *x_sum = (clip(x_sum.0), clip(x_sum.1))
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();

    let mut hs = segtree_lazy::SegTree::from_iter((0..n).map(|_| BoundsOp::singleton(0)), BoundsOp);

    for _ in 0..input.value() {
        let op = input.token();
        let l = input.value::<usize>();
        let r = input.value::<usize>();
        let h: i32 = input.value();

        let action = match op {
            "1" => BoundsOp::apply_max(h),
            "2" => BoundsOp::apply_min(h),
            _ => panic!(),
        };
        hs.apply_range(l..r + 1, action);
    }

    for i in 0..n {
        writeln!(output, "{}", hs.query_range(i..i + 1).0).unwrap();
    }
}
