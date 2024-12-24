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

pub mod jagged {
    use std::fmt::Debug;
    use std::iter;
    use std::mem::MaybeUninit;

    // Trait for painless switch between different representations of a jagged array
    pub trait Jagged<'a, T: 'a> {
        type ItemRef: ExactSizeIterator<Item = &'a T>;
        fn len(&self) -> usize;
        fn get(&'a self, u: usize) -> Self::ItemRef;
    }

    impl<'a, T, C> Jagged<'a, T> for C
    where
        C: AsRef<[Vec<T>]> + 'a,
        T: 'a,
    {
        type ItemRef = std::slice::Iter<'a, T>;
        fn len(&self) -> usize {
            <Self as AsRef<[Vec<T>]>>::as_ref(self).len()
        }
        fn get(&'a self, u: usize) -> Self::ItemRef {
            let res = <Self as AsRef<[Vec<T>]>>::as_ref(self)[u].iter();
            res
        }
    }

    // Compressed sparse row format for jagged array
    // Provides good locality for graph traversal, but works only for static ones.
    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
    pub struct CSR<T> {
        data: Vec<T>,
        head: Vec<u32>,
    }

    impl<T> Debug for CSR<T>
    where
        T: Debug,
    {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let v: Vec<Vec<&T>> = (0..self.len()).map(|i| self.get(i).collect()).collect();
            v.fmt(f)
        }
    }

    impl<T, I> FromIterator<I> for CSR<T>
    where
        I: IntoIterator<Item = T>,
    {
        fn from_iter<J>(iter: J) -> Self
        where
            J: IntoIterator<Item = I>,
        {
            let mut data = vec![];
            let mut head = vec![];
            head.push(0);

            let mut cnt = 0;
            for row in iter {
                data.extend(row.into_iter().inspect(|_| cnt += 1));
                head.push(cnt);
            }
            CSR { data, head }
        }
    }

    impl<T: Clone> CSR<T> {
        pub fn from_assoc_list(n: usize, pairs: &[(u32, T)]) -> Self {
            let mut head = vec![0u32; n + 1];

            for &(u, _) in pairs {
                debug_assert!(u < n as u32);
                head[u as usize + 1] += 1;
            }
            for i in 2..n + 1 {
                head[i] += head[i - 1];
            }
            let mut data: Vec<_> = iter::repeat_with(|| MaybeUninit::uninit())
                .take(head[n] as usize)
                .collect();
            let mut pos = head.clone();

            for (u, v) in pairs {
                data[pos[*u as usize] as usize] = MaybeUninit::new(v.clone());
                pos[*u as usize] += 1;
            }

            let data = std::mem::ManuallyDrop::new(data);
            let data = unsafe {
                Vec::from_raw_parts(data.as_ptr() as *mut T, data.len(), data.capacity())
            };

            CSR { data, head }
        }
    }

    impl<'a, T: 'a> Jagged<'a, T> for CSR<T> {
        type ItemRef = std::slice::Iter<'a, T>;

        fn len(&self) -> usize {
            self.head.len() - 1
        }

        fn get(&'a self, u: usize) -> Self::ItemRef {
            self.data[self.head[u] as usize..self.head[u + 1] as usize].iter()
        }
    }
}

pub mod reroot {
    pub mod invertible {
        // O(n) rerooting dp for trees, with invertible pulling operation. (group action)
        use crate::jagged::Jagged;

        pub trait AsBytes<const N: usize> {
            unsafe fn as_bytes(self) -> [u8; N];
            unsafe fn decode(bytes: [u8; N]) -> Self;
        }

        #[macro_export]
        macro_rules! impl_as_bytes {
            ($T:ty, $N: ident) => {
                const $N: usize = std::mem::size_of::<$T>();

                impl crate::reroot::invertible::AsBytes<$N> for $T {
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

        pub trait RootData<E> {
            fn pull_from(&mut self, child: &Self, weight: &E, inv: bool);
            fn finalize(&mut self) {}
        }

        fn get_two<T>(xs: &mut [T], i: usize, j: usize) -> Option<(&mut T, &mut T)> {
            debug_assert!(i < xs.len() && j < xs.len());
            if i == j {
                return None;
            }
            let ptr = xs.as_mut_ptr();
            Some(unsafe { (&mut *ptr.add(i), &mut *ptr.add(j)) })
        }

        fn reroot_on_edge<E, R: RootData<E>>(data: &mut [R], u: usize, w: &E, p: usize) {
            let (data_u, data_p) = unsafe { get_two(data, u, p).unwrap_unchecked() };
            data_p.pull_from(data_u, &w, true);
            data_p.finalize();

            data_u.pull_from(data_p, &w, false);
            data_u.finalize();
        }

        fn rec_reroot<'a, E: 'a, R: RootData<E> + Clone>(
            neighbors: &'a impl Jagged<'a, (u32, E)>,
            data: &mut [R],
            yield_root_data: &mut impl FnMut(usize, &R),
            u: usize,
            p: usize,
        ) {
            yield_root_data(u, &data[u]);
            for (v, w) in neighbors.get(u) {
                if *v as usize == p {
                    continue;
                }
                reroot_on_edge(data, *v as usize, w, u);
                rec_reroot(neighbors, data, yield_root_data, *v as usize, u);
                reroot_on_edge(data, u, w, *v as usize);
            }
        }

        fn toposort<'a, const N: usize, E: Clone + Default + 'a>(
            neighbors: &'a impl Jagged<'a, (u32, E)>,
        ) -> (Vec<u32>, Vec<(u32, E)>)
        where
            E: AsBytes<N>,
        {
            // Fast tree reconstruction with XOR-linked tree traversal.
            // Restriction: The graph should be undirected i.e. (u, v, weight) in E <=> (v, u, weight) in E.
            // https://codeforces.com/blog/entry/135239

            let n = neighbors.len();
            if n == 1 {
                return (vec![0], vec![(0, E::default())]);
            }

            let mut degree = vec![0; n];
            let mut xor: Vec<(u32, [u8; N])> = vec![(0u32, [0u8; N]); n];

            fn xor_assign_bytes<const N: usize>(xs: &mut [u8; N], ys: [u8; N]) {
                for (x, y) in xs.iter_mut().zip(&ys) {
                    *x ^= *y;
                }
            }

            for u in 0..n {
                for (v, w) in neighbors.get(u) {
                    degree[u] += 1;
                    xor[u].0 ^= v;
                    xor_assign_bytes(&mut xor[u].1, unsafe { AsBytes::as_bytes(w.clone()) });
                }
            }
            degree[0] += 2;

            let mut toposort = vec![];
            for mut u in 0..n {
                while degree[u] == 1 {
                    let (v, w_encoded) = xor[u];
                    toposort.push(u as u32);
                    degree[u] = 0;
                    degree[v as usize] -= 1;
                    xor[v as usize].0 ^= u as u32;
                    xor_assign_bytes(&mut xor[v as usize].1, w_encoded);
                    u = v as usize;
                }
            }
            toposort.push(0);
            toposort.reverse();

            // Note: Copying entire vec (from xor to parent) is necessary, since
            // transmuting from (u32, [u8; N]) to (u32, E)
            // or *const (u32, [u8; N]) to *const (u32, E) is UB. (tuples are
            // #[align(rust)] struct and the field ordering is not guaranteed)
            // We may try messing up with custom #[repr(C)] structs, or just trust the compiler.
            let parent: Vec<(u32, E)> = xor
                .into_iter()
                .map(|(v, w)| (v, unsafe { AsBytes::decode(w) }))
                .collect();
            (toposort, parent)
        }

        pub fn run<'a, const N: usize, E: Clone + Default + 'a, R: RootData<E> + Clone>(
            neighbors: &'a impl Jagged<'a, (u32, E)>,
            data: &mut [R],
            yield_node_dp: &mut impl FnMut(usize, &R),
        ) where
            E: AsBytes<N>,
        {
            let (order, parent) = toposort(neighbors);
            let root = order[0] as usize;

            // Init tree DP
            for &u in order.iter().rev() {
                data[u as usize].finalize();

                let (p, w) = &parent[u as usize];
                if u as usize != root {
                    let (data_u, data_p) =
                        unsafe { get_two(data, u as usize, *p as usize).unwrap_unchecked() };
                    data_p.pull_from(data_u, &w, false);
                }
            }

            // Reroot
            rec_reroot(neighbors, data, yield_node_dp, root, root);
        }
    }
}

#[derive(Clone)]
struct DistSum {
    sum: u64,
    size: u32,
}

impl DistSum {
    fn new() -> Self {
        Self { sum: 0, size: 1 }
    }
}

impl reroot::invertible::RootData<u32> for DistSum {
    fn pull_from(&mut self, child: &Self, weight: &u32, inv: bool) {
        let delta = child.sum + *weight as u64 * child.size as u64;
        if !inv {
            self.sum += delta;
            self.size += child.size;
        } else {
            self.sum -= delta;
            self.size -= child.size;
        }
    }

    fn finalize(&mut self) {}
}

fn main() {
    let mut input = fast_io::stdin_int();
    let mut output = fast_io::stdout();

    loop {
        let n: usize = input.u32() as usize;
        if n == 0 {
            break;
        }
        let mut edges = vec![];
        for _ in 1..n {
            let u = input.u32();
            let v = input.u32();
            let w = input.u32();
            edges.push((u, (v, w)));
            edges.push((v, (u, w)));
        }
        let neighbors = jagged::CSR::from_assoc_list(n, &edges);

        let mut min_dist_sum = u64::MAX;
        reroot::invertible::run(&neighbors, &mut vec![DistSum::new(); n], &mut |_, root| {
            min_dist_sum = min_dist_sum.min(root.sum);
        });
        writeln!(output, "{}", min_dist_sum).unwrap();
    }
}
