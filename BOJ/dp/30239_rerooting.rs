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

        pub fn run<'a, E: Clone + Default + 'a, R: RootData<E> + Clone>(
            neighbors: &'a impl Jagged<'a, (u32, E)>,
            data: &mut [R],
            root_init: usize,
            yield_node_dp: &mut impl FnMut(usize, &R),
        ) {
            let mut preorder = vec![]; // Reversed postorder
            let mut parent = vec![(root_init, E::default()); neighbors.len()];
            let mut stack = vec![(root_init, root_init)];
            while let Some((u, p)) = stack.pop() {
                preorder.push(u);
                for (v, w) in neighbors.get(u) {
                    if *v as usize == p {
                        continue;
                    }
                    parent[*v as usize] = (u, w.clone());
                    stack.push((*v as usize, u));
                }
            }

            // Init tree DP
            for &u in preorder.iter().rev() {
                data[u].finalize();

                let (p, w) = &parent[u];
                if u != root_init {
                    let (data_u, data_p) = unsafe { get_two(data, u, *p).unwrap_unchecked() };
                    data_p.pull_from(data_u, &w, false);
                }
            }

            // Reroot
            rec_reroot(neighbors, data, yield_node_dp, root_init, root_init);
        }
    }
}

const P: u64 = 1_000_000_007;

#[derive(Clone, Copy)]
struct NodeDp {
    value: u64,
    size: u32,
    cost: u64,
}

impl NodeDp {
    fn singleton(c: u64) -> Self {
        Self {
            value: c,
            size: 1,
            cost: 0,
        }
    }
}

impl reroot::invertible::RootData<()> for NodeDp {
    fn pull_from(&mut self, child: &Self, _weight: &(), inv: bool) {
        let delta = (self.value ^ child.value) * child.size as u64 + child.cost;
        if !inv {
            self.cost += delta;
            self.size += child.size;
        } else {
            self.cost -= delta;
            self.size -= child.size;
        }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    for _ in 0..input.value() {
        let n: usize = input.value();

        let mut a: Vec<_> = (0..n).map(|_| NodeDp::singleton(input.value())).collect();
        let mut edges = vec![];
        for _ in 0..n - 1 {
            let u = input.value::<u32>() - 1;
            let v = input.value::<u32>() - 1;
            edges.push((u, (v, ())));
            edges.push((v, (u, ())));
        }
        let neighbors = jagged::CSR::from_assoc_list(n, &edges);

        let mut res = vec![0; n];
        reroot::invertible::run(&neighbors, &mut a, 0, &mut |u, dp| {
            res[u] = dp.cost;
        });

        for u in 0..n {
            write!(output, "{} ", res[u]).unwrap();
        }
        writeln!(output).unwrap();
    }
}
