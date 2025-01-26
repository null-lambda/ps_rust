use std::{
    cmp::{Ordering, Reverse},
    io::Write,
};

mod simple_io {
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

mod dset {
    use std::{cell::Cell, mem};

    pub struct DisjointSet {
        // Represents parent if >= 0, size if < 0
        parent_or_size: Vec<Cell<i32>>,
    }

    impl DisjointSet {
        pub fn new(n: usize) -> Self {
            Self {
                parent_or_size: vec![Cell::new(-1); n],
            }
        }

        fn get_parent_or_size(&self, u: usize) -> Result<usize, u32> {
            let x = self.parent_or_size[u].get();
            if x >= 0 {
                Ok(x as usize)
            } else {
                Err((-x) as u32)
            }
        }

        fn set_parent(&self, u: usize, p: usize) {
            self.parent_or_size[u].set(p as i32);
        }

        fn set_size(&self, u: usize, s: u32) {
            self.parent_or_size[u].set(-(s as i32));
        }

        pub fn find_root_with_size(&self, u: usize) -> (usize, u32) {
            match self.get_parent_or_size(u) {
                Ok(p) => {
                    let (root, size) = self.find_root_with_size(p);
                    self.set_parent(u, root);
                    (root, size)
                }
                Err(size) => (u, size),
            }
        }

        pub fn find_root(&self, u: usize) -> usize {
            self.find_root_with_size(u).0
        }

        // Returns true if two sets were previously disjoint
        pub fn merge(&mut self, u: usize, v: usize) -> bool {
            let (mut u, size_u) = self.find_root_with_size(u);
            let (mut v, size_v) = self.find_root_with_size(v);
            if u == v {
                return false;
            }

            if size_u < size_v {
                mem::swap(&mut u, &mut v);
            }
            self.set_parent(v, u);
            self.set_size(u, size_u + size_v);
            true
        }
    }
}

pub mod mst {
    use super::dset::DisjointSet;

    fn partition_in_place<T>(xs: &mut [T], mut pred: impl FnMut(&T) -> bool) -> usize {
        let n = xs.len();
        let mut i = 0;
        for j in 0..n {
            if pred(&xs[j]) {
                xs.swap(i, j);
                i += 1;
            }
        }
        i
    }

    pub fn kruskal<E: Ord + Copy>(
        remained_edges: &mut usize,
        dset: &mut DisjointSet,
        yield_mst_edge: &mut impl FnMut(u32, u32, E),
        edges: &mut [(u32, u32, E)],
    ) {
        if *remained_edges == 0 {
            return;
        }
        edges.sort_unstable_by_key(|&(_, _, w)| w);
        for (u, v, w) in edges.iter().copied() {
            if dset.merge(u as usize, v as usize) {
                yield_mst_edge(u, v, w);
                *remained_edges -= 1;
                if *remained_edges == 0 {
                    break;
                }
            }
        }
    }

    /// # Filter-Kruskal MST
    ///
    /// Time complexity: `O(E + V (log V) (log (E/V)))`
    ///
    /// A quicksort-like divide-and-conquer approach to Kruskal algorithm,
    /// which attempts to reduce sorting overhead by filtering out edges preemptively.
    ///
    /// ## Reference
    ///
    /// Osipov, Vitaly, Peter Sanders, and John Victor Singler.
    /// “The Filter-Kruskal Minimum Spanning Tree Algorithm.”
    /// Workshop on Algorithm Engineering and Experimentation (2009).
    /// [https://dl.acm.org/doi/pdf/10.5555/2791220.2791225]
    pub fn filter_kruskal<E: Ord + Copy>(
        remained_edges: &mut usize,
        dset: &mut DisjointSet,
        yield_mst_edge: &mut impl FnMut(u32, u32, E),
        edges: &mut [(u32, u32, E)],
    ) {
        // A heuristic. should be asymptotically O(V)
        let threshold = *remained_edges * 2;
        if edges.len() <= threshold {
            kruskal(remained_edges, dset, yield_mst_edge, edges);
            return;
        }

        // Take the median as a pivot in O(n).
        let pivot = edges.len() / 2;
        let (lower, mid, upper) = edges.select_nth_unstable_by_key(pivot, |&(_, _, w)| w);
        filter_kruskal(remained_edges, dset, yield_mst_edge, lower);

        {
            // Inlined version of kruskal(.., &mut [*mid]);
            if *remained_edges == 0 {
                return;
            }
            let (u, v, w) = *mid;
            if dset.merge(u as usize, v as usize) {
                yield_mst_edge(u, v, w);
                *remained_edges -= 1;
            }
        }

        let i = partition_in_place(upper, |&(u, v, _)| {
            dset.find_root(u as usize) != dset.find_root(v as usize)
        });
        let filtered = &mut upper[..i];
        filter_kruskal(remained_edges, dset, yield_mst_edge, filtered);
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let mut edges: Vec<_> = (0..m)
        .map(|_| {
            (
                input.value::<u32>() - 1,
                input.value::<u32>() - 1,
                Reverse(input.value::<u32>()),
            )
        })
        .collect();
    let mut dset = dset::DisjointSet::new(n);
    let mut mst_len = 0;
    mst::filter_kruskal(
        &mut (n - 1),
        &mut dset,
        &mut |_, _, Reverse(w)| {
            mst_len += w as u64;
        },
        &mut edges,
    );

    let score_a = (2..=n as i64).map(|x| (x / 2) * (x - x / 2)).sum::<i64>();
    let ans = match score_a.cmp(&(mst_len as i64)) {
        Ordering::Less => "lose",
        Ordering::Equal => "tie",
        Ordering::Greater => "win",
    };
    writeln!(output, "{}", ans).unwrap();
}
