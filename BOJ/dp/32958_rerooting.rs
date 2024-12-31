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
            ($T:ty) => {
                const N: usize = std::mem::size_of::<$T>();

                impl crate::reroot::invertible::AsBytes<N> for $T {
                    unsafe fn as_bytes(self) -> [u8; N] {
                        std::mem::transmute::<$T, [u8; N]>(self)
                    }

                    unsafe fn decode(bytes: [u8; N]) -> $T {
                        std::mem::transmute::<[u8; N], $T>(bytes)
                    }
                }
            };
        }
        pub use impl_as_bytes;

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

        fn construct_parent<'a, const N: usize, E: Clone + Default + 'a>(
            neighbors: &'a impl Jagged<'a, (u32, E)>,
        ) -> (Vec<u32>, Vec<(u32, E)>)
        where
            E: AsBytes<N>,
        {
            // Fast tree reconstruction with XOR-linked tree traversal,
            // which combines toposort with xor encoding.
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
            let (order, parent) = construct_parent(neighbors);
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

const P: u64 = 1_000_000_007;

#[derive(Clone, Copy)]
struct RollingHash {
    digit: u64,
    sum: u64,
    size: u64,
}

impl RollingHash {
    fn singleton(c: u64) -> Self {
        Self {
            digit: c,
            sum: c,
            size: 1,
        }
    }
}

impl reroot::invertible::RootData<()> for RollingHash {
    fn pull_from(&mut self, child: &Self, _weight: &(), inv: bool) {
        let delta = self.digit * child.size + child.sum * 10 % P;
        if !inv {
            self.sum += delta;
            self.size += child.size;
        } else {
            self.sum -= delta;
            self.size -= child.size;
        }
    }
}

reroot::invertible::impl_as_bytes!(());

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let mut data = (0..n)
        .map(|_| RollingHash::singleton(input.value()))
        .collect::<Vec<_>>();

    let mut edges = vec![];
    for _ in 0..n - 1 {
        let u = input.value::<u32>() - 1;
        let v = input.value::<u32>() - 1;
        edges.push((u, (v, ())));
        edges.push((v, (u, ())));
    }
    let neighbors = jagged::CSR::from_assoc_list(n, &edges);

    let mut res = 0u64;
    reroot::invertible::run(&neighbors, &mut data, &mut |_, h| {
        res = (res + h.sum) % P;
    });
    writeln!(output, "{}", res).unwrap();
}
