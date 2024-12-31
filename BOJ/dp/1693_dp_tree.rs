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
    // Fast tree reconstruction with XOR-linked tree traversal
    // https://codeforces.com/blog/entry/135239
    pub trait AsBytes<const N: usize> {
        unsafe fn as_bytes(self) -> [u8; N];
        unsafe fn decode(bytes: [u8; N]) -> Self;
    }

    #[macro_export]
    macro_rules! impl_as_bytes {
        ($T:ty, $N: ident) => {
            const $N: usize = std::mem::size_of::<$T>();

            impl crate::tree::AsBytes<$N> for $T {
                unsafe fn as_bytes(self) -> [u8; $N] {
                    std::mem::transmute::<$T, [u8; $N]>(self)
                }

                unsafe fn decode(bytes: [u8; $N]) -> $T {
                    std::mem::transmute::<[u8; $N], $T>(bytes)
                }
            }
        };
    }
    pub use impl_as_bytes;

    impl_as_bytes!((), __N_UNIT);
    impl_as_bytes!(u32, __N_U32);
    impl_as_bytes!(u64, __N_U64);
    impl_as_bytes!(i32, __N_I32);
    impl_as_bytes!(i64, __N_I64);

    #[inline(always)]
    pub unsafe fn assert_unchecked(b: bool) {
        if !b {
            std::hint::unreachable_unchecked();
        }
    }

    #[cold]
    #[inline(always)]
    pub fn cold() {}

    #[inline(always)]
    pub fn likely(b: bool) -> bool {
        if !b {
            cold();
        }
        b
    }

    #[inline(always)]
    pub fn unlikely(b: bool) -> bool {
        if b {
            cold();
        }
        b
    }

    pub fn toposort<'a, const N: usize, E: Clone + Default + AsBytes<N> + 'a>(
        n_verts: usize,
        edges: impl IntoIterator<Item = (u32, u32, E)>,
        root: usize,
    ) -> impl Iterator<Item = (u32, u32, E)> {
        let mut degree = vec![0u32; n_verts];
        let mut xor_neighbors: Vec<(u32, [u8; N])> = vec![(0 as u32, [0u8; N]); n_verts];

        fn xor_assign_bytes<const N: usize>(xs: &mut [u8; N], ys: [u8; N]) {
            for (x, y) in xs.iter_mut().zip(&ys) {
                *x ^= *y;
            }
        }

        for (u, v, w) in edges
            .into_iter()
            .flat_map(|(u, v, w)| [(u, v, w.clone()), (v, u, w)])
        {
            debug_assert!(u != v);
            degree[u as usize] += 1;
            xor_neighbors[u as usize].0 ^= v;
            xor_assign_bytes(&mut xor_neighbors[u as usize].1, unsafe {
                AsBytes::as_bytes(w.clone())
            });
        }

        degree[root] += 2;
        let mut base = 0;
        let mut u = 0;
        std::iter::from_fn(move || {
            if unlikely(degree[u] != 1) {
                u = loop {
                    if unlikely(base >= n_verts) {
                        return None;
                    } else if degree[base] == 1 {
                        break base;
                    }
                    base += 1;
                }
            }
            let (p, w_encoded) = xor_neighbors[u];
            degree[u] = 0;
            degree[p as usize] -= 1;
            xor_neighbors[p as usize].0 ^= u as u32;
            xor_assign_bytes(&mut xor_neighbors[p as usize].1, w_encoded);
            let u_old = u;
            u = p as usize;
            Some((u_old as u32, p, unsafe { AsBytes::decode(w_encoded) }))
        })
    }
}

const C_BOUND: usize = 19;
const INF: u32 = 1 << 30;

#[derive(Clone)]
struct NodeData {
    min_cost: [u32; C_BOUND],
}

impl NodeData {
    fn new() -> Self {
        Self {
            min_cost: std::array::from_fn(|i| 1 + i as u32),
        }
    }

    fn pull_up(&mut self, child_exclusive: &Self) {
        for c in 0..C_BOUND {
            self.min_cost[c] += child_exclusive.min_cost[c];
        }
    }

    fn finalize(&mut self) {
        let mut exclusive = [INF; C_BOUND];
        let mut prefix = INF;
        for c in 0..C_BOUND {
            exclusive[c] = prefix;
            prefix = prefix.min(self.min_cost[c]);
        }

        let mut suffix = INF;
        for c in (0..C_BOUND).rev() {
            exclusive[c] = exclusive[c].min(suffix);
            suffix = suffix.min(self.min_cost[c]);
        }
        self.min_cost = exclusive;
    }

    fn min_cost(&self) -> u32 {
        *self.min_cost.iter().min().unwrap()
    }
}

fn get_two<T>(xs: &mut [T], i: usize, j: usize) -> Option<(&mut T, &mut T)> {
    debug_assert!(i < xs.len() && j < xs.len());
    if i == j {
        return None;
    }
    let ptr = xs.as_mut_ptr();
    Some(unsafe { (&mut *ptr.add(i), &mut *ptr.add(j)) })
}

fn main() {
    let mut input = fast_io::stdin_int();
    let mut output = fast_io::stdout();

    let n: usize = input.u32() as usize;
    let edges = (0..n - 1).map(|_| (input.u32() - 1, input.u32() - 1, ()));

    let root = 0;
    let mut dp = vec![NodeData::new(); n];
    for (u, p, ()) in tree::toposort(n, edges, root) {
        let (dp_u, dp_p) = unsafe { get_two(&mut dp, u as usize, p as usize).unwrap_unchecked() };
        dp_u.finalize();
        dp_p.pull_up(dp_u);
    }
    let ans = dp[root].min_cost();
    writeln!(output, "{}", ans).unwrap();
}
