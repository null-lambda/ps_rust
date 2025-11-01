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
            if dset.merge(u, v) {
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

        let i = partition_in_place(upper, |&(u, v, _)| dset.root(u) != dset.root(v));
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
        let mut ps: Vec<_> = ps
            .into_iter()
            .enumerate()
            .map(|(i, p)| (p, i as u32))
            .collect();

        let mut buffer = vec![];
        ps.sort_unstable_by_key(|&((x, y), _)| x + y);
        solve_half_quadrant(&mut buffer, &mut yield_edge, ps.iter().copied());
        solve_half_quadrant(
            &mut buffer,
            &mut yield_edge,
            ps.iter().map(|&((x, y), i)| ((y, x), i)),
        );

        ps.sort_unstable_by_key(|&((x, y), _)| y - x);
        solve_half_quadrant(
            &mut buffer,
            &mut yield_edge,
            ps.iter().map(|&((x, y), i)| ((-x, y), i)),
        );
        solve_half_quadrant(
            &mut buffer,
            &mut yield_edge,
            ps.iter().map(|&((x, y), i)| ((y, -x), i)),
        );
    }

    fn solve_half_quadrant(
        buffer: &mut Vec<i32>,
        mut yield_edge: impl FnMut(u32, u32, i32),
        ps: impl IntoIterator<Item = ((i32, i32), u32)>,
    ) {
        let v = |(x, y)| x - y;
        let dist = |(x1, y1): (i32, i32), (x2, y2): (i32, i32)| ((x1 - x2).abs() + (y1 - y2).abs());

        let to_remove = buffer;
        let mut active: BTreeMap<i32, _> = BTreeMap::new();
        for (pi, i) in ps {
            to_remove.extend(
                active
                    .range(..=pi.0)
                    .rev()
                    .take_while(|&(_, &(pj, _))| v(pi) <= v(pj))
                    .inspect(|&(_, &(pj, j))| yield_edge(i, j, dist(pi, pj)))
                    .map(|(&x, _)| x),
            );
            for x in to_remove.drain(..) {
                active.remove(&x);
            }
            active.insert(pi.0, (pi, i));
        }
    }
}
