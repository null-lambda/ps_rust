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

        // The following two lines are equivalent.
        // partition_point(0, n, |i| pred(segtree.query_range(0..i+1)));
        // segtree.partition_point_prefix(|prefix| pred(prefix));
        pub fn partition_point_prefix(&mut self, mut pred: impl FnMut(&M::X) -> bool) -> usize {
            assert!(self.n >= 1 && self.n.is_power_of_two());

            let mut u = 1;
            let mut width = self.n as u32;
            let mut prefix = self.ma.id();

            while u < self.n {
                width >>= 1;
                self.push_down(width, u);

                let new_prefix = self.ma.combine(&prefix, &self.sum[u << 1]);
                u = if pred(&new_prefix) {
                    prefix = new_prefix;
                    u << 1 | 1
                } else {
                    u << 1
                };
            }

            let idx = u - self.n;
            if pred(&self.ma.combine(&prefix, &self.sum[u])) {
                idx + 1
            } else {
                idx
            }
        }
    }
}
