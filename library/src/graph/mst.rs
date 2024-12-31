pub mod mst {
    use super::dset::DisjointSet;
    use std::collections::BTreeMap;
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
        let threshold = (*remained_edges * 2).max(20);
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
        ps: impl IntoIterator<Item = (i32, i32)>,
        mut yield_edge: impl FnMut(u32, u32, i32),
    ) {
        let mut ps: Vec<(i32, i32)> = ps.into_iter().map(|(x, y)| (x as i32, y as i32)).collect();
        let mut indices: Vec<_> = (0..ps.len() as u32).collect();

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
                    yield_edge(i, j, dist(ps[i as usize], ps[j as usize]));
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
    }
}
