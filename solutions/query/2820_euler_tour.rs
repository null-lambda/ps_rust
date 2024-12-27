use std::io::Write;

mod fast_io {
    use std::fs::File;
    use std::io::BufWriter;
    use std::os::unix::io::FromRawFd;

    pub struct InputAtOnce {
        _buf: &'static str,
        iter: std::str::SplitAsciiWhitespace<'static>,
    }

    impl InputAtOnce {
        pub fn token(&mut self) -> &'static str {
            self.iter.next().unwrap_or_default()
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().unwrap()
        }
    }

    extern "C" {
        fn mmap(addr: usize, length: usize, prot: i32, flags: i32, fd: i32, offset: i64)
            -> *mut u8;
        fn fstat(fd: i32, stat: *mut usize) -> i32;
    }

    pub fn stdin() -> InputAtOnce {
        let mut stat = [0; 18];
        unsafe { fstat(0, (&mut stat).as_mut_ptr()) };
        let _buf = unsafe { mmap(0, stat[6], 1, 2, 0, 0) };
        let _buf =
            unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(_buf, stat[6])) };
        let iter = _buf.split_ascii_whitespace();
        InputAtOnce { _buf, iter }
    }

    pub fn stdout() -> BufWriter<File> {
        let stdout = unsafe { File::from_raw_fd(1) };
        BufWriter::with_capacity(1 << 16, stdout)
    }

    pub struct IntScanner {
        buf: &'static [u8],
    }

    impl IntScanner {
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

pub mod tree {
    pub fn euler_tour<'a>(
        n: usize,
        edges: impl IntoIterator<Item = (u32, u32)>,
        root: usize,
    ) -> (Vec<u32>, Vec<u32>) {
        // Fast tree reconstruction with XOR-linked tree traversal
        // https://codeforces.com/blog/entry/135239
        let mut degree = vec![0u32; n];
        let mut xor_neighbors: Vec<u32> = vec![0u32; n];
        for (u, v) in edges.into_iter().flat_map(|(u, v)| [(u, v), (v, u)]) {
            debug_assert!(u != v);
            degree[u as usize] += 1;
            xor_neighbors[u as usize] ^= v;
        }

        let mut size = vec![1; n];
        degree[root] += 2;
        let mut topological_order = Vec::with_capacity(n);
        for mut u in 0..n {
            while degree[u] == 1 {
                // Topological sort
                let p = xor_neighbors[u];
                topological_order.push(u as u32);
                degree[u] = 0;
                degree[p as usize] -= 1;
                xor_neighbors[p as usize] ^= u as u32;

                // Upward propagation
                size[p as usize] += size[u as usize];
                u = p as usize;
            }
        }
        assert!(topological_order.len() == n - 1, "Invalid tree structure");

        let parent = xor_neighbors;

        // Downward propagation
        let mut euler_in = size.clone();
        for u in topological_order.into_iter().rev() {
            let p = parent[u as usize];
            let final_index = euler_in[p as usize];
            euler_in[p as usize] -= euler_in[u as usize];
            euler_in[u as usize] = final_index;
        }

        let mut euler_out = size;
        for u in 0..n {
            euler_in[u] -= 1;
            euler_out[u] += euler_in[u];
        }

        (euler_in, euler_out)
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

    type X = i32;

    #[derive(Debug, Clone)]
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
                    *b += value & m;
                }
                idx >>= log::<B>();
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

fn main() {
    let mut input = fast_io::stdin();
    let mut output = fast_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();

    let root = 0;
    let mut parent = vec![root as u32; n];
    let mut xs0 = vec![0; n];
    for u in 0..n {
        xs0[u] = input.value::<i32>();
        if u != root {
            parent[u] = input.value::<u32>() - 1;
        }
    }

    let edges = (1..n).map(|u| (parent[u as usize], u as u32));
    let (euler_in, euler_out) = tree::euler_tour(n, edges, root);
    let mut delta = segtree_wide::SegTree::<16>::new(n + 1);

    for _ in 0..m {
        match input.token() {
            "p" => {
                let i = input.value::<usize>() - 1;
                let w = input.value::<i32>();
                delta.add(euler_in[i] as usize + 1, w as i32);
                delta.add(euler_out[i] as usize, -w as i32);
            }
            "u" => {
                let i = input.value::<usize>() - 1;
                let ans = xs0[i] + delta.sum_prefix(euler_in[i] as usize + 1);
                writeln!(output, "{}", ans).unwrap();
            }
            _ => panic!(),
        }
    }
}
