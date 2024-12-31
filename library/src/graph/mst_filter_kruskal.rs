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

mod dset {
    use core::{cell::Cell, mem};

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
    use std::collections::BTreeMap;

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

    fn kruskal_internal<E: Ord + Copy>(
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
        let threshold = *remained_edges * 3 / 2;
        if edges.len() <= threshold {
            kruskal_internal(remained_edges, dset, yield_mst_edge, edges);
            return;
        }

        // Take the median as a pivot in O(n).
        // The authors of Filter-Kruskal paper suggest optimizing via a sqrt N-sized random sample median.
        let pivot = edges.len() / 2;
        let (lower, mid, upper) = edges.select_nth_unstable_by_key(pivot, |&(_, _, w)| w);
        filter_kruskal(remained_edges, dset, yield_mst_edge, lower);

        {
            // Inlined version of filter_kruskal_rec(.., &mut [*mid]);
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

    /// # MST in L^1 / L^inf metrics
    ///
    /// Filter at most 8V/2 edges among V(V-1)/2 potential edges between V points, in O(V log V) time complexity.
    ///
    /// ## Reference
    /// [https://cp-algorithms.com/geometry/manhattan-distance.html#farthest-pair-of-points-in-manhattan-distance]
    pub fn manhattan_mst_candidates(
        ps: impl IntoIterator<Item = (u32, u32)>,
    ) -> Vec<(u32, u32, u32)> {
        let mut ps: Vec<(i32, i32)> = ps.into_iter().map(|(x, y)| (x as i32, y as i32)).collect();
        let mut indices: Vec<_> = (0..ps.len() as u32).collect();
        let mut edges = vec![];

        let dist = |(x1, y1): (i32, i32), (x2, y2): (i32, i32)| ((x1 - x2).abs() + (y1 - y2).abs());

        // Rotate by pi/4
        let u = |(x, y)| x + y;
        let v = |(x, y)| x - y;

        for rot in 0..4 {
            indices.sort_unstable_by_key(|&i| u(ps[i as usize]));

            let mut active: BTreeMap<i32, u32> = BTreeMap::new();
            for &i in &indices {
                let mut to_remove = vec![];
                for (&x, &j) in active.range(..=ps[i as usize].0).rev() {
                    if v(ps[i as usize]) > v(ps[j as usize]) {
                        break;
                    }
                    debug_assert!(
                        ps[i as usize].0 >= ps[j as usize].0
                            && ps[i as usize].1 >= ps[j as usize].1
                    );
                    edges.push((i, j, dist(ps[i as usize], ps[j as usize]) as u32));
                    to_remove.push(x);
                }
                for x in to_remove {
                    active.remove(&x);
                }
                active.insert(ps[i as usize].0, i);
            }

            for p in ps.iter_mut() {
                if rot % 2 == 1 {
                    p.0 = -p.0;
                } else {
                    std::mem::swap(&mut p.0, &mut p.1);
                }
            }
        }

        edges
    }
}

pub fn main() {
    let mut input = fast_io::stdin_int();
    let mut output = fast_io::stdout();

    let _n = input.u32() as usize;
    let k = input.u32() as usize;
    let ps: Vec<(u32, u32)> = (0..k).map(|_| (input.u32(), input.u32())).collect();
    let mut edges = mst::manhattan_mst_candidates(ps.iter().cloned());
    let mut max_edge = 0;
    mst::filter_kruskal(
        &mut (k - 1),
        &mut dset::DisjointSet::new(k),
        &mut |_, _, w| max_edge = w,
        &mut edges,
    );
    let ans = max_edge / 2;
    writeln!(output, "{}", ans).unwrap();
}
