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
        pub fn with_size(n: usize) -> Self {
            Self {
                n,
                sum: vec![T::id(); 2 * n],
            }
        }

        pub fn from_iter<I>(n: usize, iter: I) -> Self
        where
            I: Iterator<Item = T>,
        {
            use std::iter::repeat;
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
    }
}
