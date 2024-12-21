use std::io::Write;

mod fast_io {
    use std::fs::File;
    use std::io::BufWriter;
    use std::os::unix::io::FromRawFd;

    extern "C" {
        fn mmap(addr: usize, length: usize, prot: i32, flags: i32, fd: i32, offset: i64)
            -> *mut u8;
        fn fstat(fd: i32, stat: *mut usize) -> i32;
    }

    pub fn stdout() -> BufWriter<File> {
        let stdout = unsafe { File::from_raw_fd(1) };
        BufWriter::new(stdout)
    }

    pub struct IntScanner {
        buf: &'static [u8],
    }

    impl IntScanner {
        pub fn u32(&mut self) -> u32 {
            loop {
                match self.buf {
                    &[] => panic!(),
                    &[b'0'..=b'9', ..] => break,
                    _ => self.buf = &self.buf[1..],
                }
            }

            let mut acc = 0;
            loop {
                match self.buf {
                    &[] => panic!(),
                    &[b'0'..=b'9', ..] => acc = acc * 10 + (self.buf[0] - b'0') as u32,
                    _ => break,
                }
                self.buf = &self.buf[1..];
            }
            acc
        }
    }

    pub fn stdin_int() -> IntScanner {
        let mut stat = [0; 18];
        unsafe { fstat(0, (&mut stat).as_mut_ptr()) };
        let buf = unsafe { mmap(0, stat[6], 1, 2, 0, 0) };
        let buf =
            unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(buf, stat[6])) };
        IntScanner {
            buf: buf.as_bytes(),
        }
    }
}

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
            node /= B;
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

    type X = u32;

    #[derive(Debug)]
    pub struct SegTree<const B: usize> {
        n: usize,
        sum: Vec<X>,
        mask: [[X; B]; B],
        offsets: Vec<usize>,
    }

    impl<const B: usize> SegTree<B> {
        pub fn new(n: usize) -> Self {
            assert!(B >= 2 && B.is_power_of_two());
            let max_height = height::<B>(n);
            Self {
                n,
                sum: vec![0; offset::<B>(n, max_height)],
                mask: compute_mask::<B>(),
                offsets: offsets::<B>(n).take(max_height as usize).collect(),
            }
        }

        pub fn add(&mut self, mut idx: usize, value: X) {
            debug_assert!(idx < self.n);
            for (_, offset) in self.offsets.iter().enumerate() {
                let block = &mut self.sum[offset + round::<B>(idx)..];
                for (b, m) in block.iter_mut().zip(&self.mask[idx % B]) {
                    *b ^= value & m;
                }
                idx >>= log::<B>();
            }
        }

        pub fn sum_prefix(&mut self, idx: usize) -> X {
            debug_assert!(idx < self.n);
            let mut res = 0;
            for (h, offset) in self.offsets.iter().enumerate() {
                res ^= self.sum[offset + (idx >> h as u32 * log::<B>())];
            }
            res
        }

        pub fn sum_range(&mut self, range: std::ops::Range<usize>) -> X {
            debug_assert!(range.start <= range.end && range.end < self.n);
            let r = self.sum_prefix(range.end);
            let l = self.sum_prefix(range.start);
            r - l
        }
    }
}

fn main() {
    let mut input = fast_io::stdin_int();
    let mut output = fast_io::stdout();

    let n: usize = input.u32() as usize;
    let xs = (0..n).map(|_| input.u32());
    let mut delta = segtree_wide::SegTree::<16>::new(n + 1);

    let mut x_prev = 0;
    for (i, x) in xs.enumerate() {
        delta.add(i, x ^ x_prev);
        x_prev = x;
    }

    for _ in 0..input.u32() {
        match input.u32() {
            1 => {
                let a = input.u32() as usize;
                let b = input.u32() as usize;
                let c: u32 = input.u32();

                delta.add(a, c);
                delta.add(b + 1, c);
            }
            2 => {
                let k = input.u32() as usize;
                writeln!(output, "{}", delta.sum_prefix(k + 1)).unwrap();
            }
            _ => panic!(),
        }
    }
}
