pub mod segtree_wide {
    // Cache-friendly segment tree, based on a B-ary tree.
    // https://en.algorithmica.org/hpc/data-structures/segment-trees/#wide-segment-trees

    // const CACHE_LINE_SIZE: usize = 64;

    // const fn adaptive_block_size<T>() -> usize {
    //     assert!(
    //         std::mem::size_of::<T>() > 0,
    //         "Zero-sized types are not supported"
    //     );
    //     let mut res = CACHE_LINE_SIZE / std::mem::size_of::<T>();
    //     if res < 2 {
    //         res = 2;
    //     }
    //     res
    // }

    use std::iter;

    const fn height<const B: usize>(mut node: usize) -> u32 {
        debug_assert!(node > 0);
        let mut res = 1;
        while node > B {
            res += 1;
            node = node.div_ceil(B);
        }
        res
    }

    // yields (h, offset)
    fn offsets<const B: usize>(size: usize) -> impl Iterator<Item = usize> {
        let mut offset = 0;
        let mut n = size;
        iter::once(0).chain((1..).map(move |_| {
            n = n.div_ceil(B);
            offset += n * B;
            offset
        }))
    }

    fn offset<const B: usize>(size: usize, h: u32) -> usize {
        offsets::<B>(size).nth(h as usize).unwrap()
    }

    fn log<const B: usize>() -> u32 {
        usize::BITS - B.leading_zeros() - 1
    }

    fn round<const B: usize>(x: usize) -> usize {
        x & !(B - 1)
    }

    const fn compute_mask<const B: usize>() -> [[X; B]; B] {
        let mut res = [[0; B]; B];
        let mut i = 0;
        while i < B {
            let mut j = 0;
            while j < B {
                res[i][j] = if i < j { !0 } else { 0 };
                j += 1;
            }
            i += 1;
        }
        res
    }

    type X = i64;

    #[derive(Debug, Clone)]
    pub struct SegTree<const B: usize> {
        n: usize,
        sum: Vec<X>,
        mask: [[X; B]; B],
        offsets: Vec<usize>,
    }

    impl<const B: usize> SegTree<B> {
        pub fn with_size(n: usize) -> Self {
            assert!(B >= 2 && B.is_power_of_two());
            let max_height = height::<B>(n);
            Self {
                n,
                sum: vec![0; offset::<B>(n, max_height)],
                mask: compute_mask::<B>(),
                offsets: offsets::<B>(n).take(max_height as usize).collect(),
            }
        }

        #[target_feature(enable = "avx2")] // Required. __mm256 has significant performance benefits over __m128.
        unsafe fn add_avx2(&mut self, mut idx: usize, value: X) {
            debug_assert!(idx < self.n);
            for (_, offset) in self.offsets.iter().enumerate() {
                let block = &mut self.sum[offset + round::<B>(idx)..];
                for (b, m) in block.iter_mut().zip(&self.mask[idx % B]) {
                    *b += value & m;
                }
                idx >>= log::<B>();
            }
        }

        pub fn add(&mut self, idx: usize, value: X) {
            unsafe {
                self.add_avx2(idx, value);
            }
        }

        pub fn sum_prefix(&mut self, idx: usize) -> X {
            debug_assert!(idx <= self.n);
            let mut res = 0;
            for (h, offset) in self.offsets.iter().enumerate() {
                res += self.sum[offset + (idx >> h as u32 * log::<B>())];
            }
            res
        }

        pub fn sum_range(&mut self, range: std::ops::Range<usize>) -> X {
            debug_assert!(range.start <= range.end && range.end <= self.n);
            let r = self.sum_prefix(range.end);
            let l = self.sum_prefix(range.start);
            r - l
        }
    }
}
