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

pub mod as_bytes {
    pub trait AsBytes<const N: usize> {
        unsafe fn as_bytes(self) -> [u8; N];
        unsafe fn decode(bytes: [u8; N]) -> Self;
    }

    #[macro_export]
    macro_rules! impl_as_bytes {
        ($T:ty, $N: ident) => {
            const $N: usize = std::mem::size_of::<$T>();

            impl crate::as_bytes::AsBytes<$N> for $T {
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
}

pub mod tree {
    use crate::as_bytes::AsBytes;

    // Fast tree reconstruction with XOR-linked tree traversal
    // https://codeforces.com/blog/entry/135239
    pub fn toposort<'a, const N: usize, E: Clone + Default + 'a>(
        n_verts: usize,
        edges: impl IntoIterator<Item = (u32, u32, E)>,
        root: usize,
    ) -> (Vec<u32>, Vec<(u32, E)>)
    where
        E: AsBytes<N>,
    {
        if n_verts == 1 {
            return (vec![0], vec![(0, E::default())]);
        }

        let mut degree = vec![0u32; n_verts];
        let mut xor: Vec<(u32, [u8; N])> = vec![(0 as u32, [0u8; N]); n_verts];

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
            xor[u as usize].0 ^= v;
            xor_assign_bytes(&mut xor[u as usize].1, unsafe {
                AsBytes::as_bytes(w.clone())
            });
        }

        degree[root] += 2;
        let mut order = vec![];
        for mut u in 0..n_verts {
            while degree[u] == 1 {
                let (v, w_encoded) = xor[u];
                order.push(u as u32);
                degree[u] = 0;
                degree[v as usize] -= 1;
                xor[v as usize].0 ^= u as u32;
                xor_assign_bytes(&mut xor[v as usize].1, w_encoded);
                u = v as usize;
            }
        }
        order.push(root as u32);
        order.reverse();

        // Note: Copying entire vec (from xor to parent) is necessary, since
        // transmuting from (u32, [u8; N]) to (u32, E)
        // or *const (u32, [u8; N]) to *const (u32, E) is UB. (tuples are
        // #[align(rust)] struct and the field ordering is not guaranteed)
        // We may try messing up with custom #[repr(C)] structs, or just trust the compiler.
        let parent: Vec<(u32, E)> = xor
            .into_iter()
            .map(|(v, w)| (v, unsafe { AsBytes::decode(w) }))
            .collect();
        (order, parent)
    }
}

fn main() {
    let mut input = fast_io::stdin_int();
    let mut output = fast_io::stdout();

    let n: usize = input.u32() as usize;
    let xs: Vec<u32> = (0..n).map(|_| input.u32()).collect();

    let root = 0;
    let edges = (0..n - 1).map(|_| (input.u32() - 1, input.u32() - 1, ()));
    let (order, parent) = tree::toposort(n, edges, root);

    let mut dp = vec![[0u32, 0]; n];
    for &u in order.iter().rev() {
        let (p, ()) = parent[u as usize];

        dp[u as usize][1] += xs[u as usize];
        if u as usize != root {
            dp[p as usize][0] += dp[u as usize][0].max(dp[u as usize][1]);
            dp[p as usize][1] += dp[u as usize][0];
        }
    }
    let sum = dp[root][0].max(dp[root][1]);

    let mut visit = vec![false; n];
    visit[root] = dp[root][0] < dp[root][1];

    for &u in &order[1..] {
        let (p, ()) = parent[u as usize];
        if visit[p as usize] {
            visit[u as usize] = false;
        } else {
            visit[u as usize] = dp[u as usize][0] < dp[u as usize][1];
        }
    }

    writeln!(output, "{}", sum).unwrap();
    for u in 0..n {
        if visit[u] {
            write!(output, "{} ", u + 1).unwrap();
        }
    }
}
