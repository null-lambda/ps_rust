mod mo {
    // Various ordering strategies for Mo's algorithm.
    //
    // The algorithm has a strong connection to the Traveling Salesman Problem {TSP},
    // as moving from one range query [l, r] to another [l', r'] costs Manhattan distance |l - l'| + |r - r'|.
    // Given a Q random sampled points in a square of side length N,
    // The Beardwood-Halton-Hammersley Theorem establishes an expected minimum path length of O(N sqrt(Q)).
    // Thus, No ordering can escape this asymptotic bound.

    use alloc::{vec, vec::Vec};

    fn isqrt(n: u32) -> u32 {
        if n == 0 {
            return 0;
        }

        let mut x = n;
        loop {
            let next_x = (x + n / x) / 2;
            if next_x >= x {
                return x;
            }
            x = next_x;
        }
    }

    // Simple while efficient. Requires almost no precomputation cost.
    pub fn sort_by_even_odd_order(
        n_lattice: usize,
        n_queries: usize,
        intervals: impl IntoIterator<Item = (u32, u32)>,
    ) -> impl Iterator<Item = (u32, u32, u32)> {
        // Time complexity: T ~ O(Q B + N^2/B)
        // => optimal bucket size: B = N/sqrt(Q), so T ~ O(N sqrt(Q))
        let bucket_size = (n_lattice as u32 / isqrt(n_queries as u32) as u32).max(1);

        let flip_by_parity = |bl: u32, r: u32| if bl & 1 == 0 { r } else { !r };
        let mut res = intervals
            .into_iter()
            .enumerate()
            .map(|(i, (l, r))| {
                let bl = l / bucket_size;
                (l, flip_by_parity(bl, r), bl, i as u32)
            })
            .collect::<Vec<_>>();

        res.sort_unstable_by(move |x, y| {
            let (_, rx, blx, _) = x;
            let (_, ry, bly, _) = y;
            blx.cmp(&bly).then_with(|| rx.cmp(&ry))
        });

        res.into_iter()
            .map(move |(l, r, bl, i)| (l, flip_by_parity(bl, r), i))
    }

    // Space-filling curve of TSP for random sampled points.
    // https://codeforces.com/blog/entry/61203
    // https://codeforces.com/blog/entry/115590
    // Note: use sort_with_cached_key instead of sort_unstable
    //       to avoid recomputing the Hilbert order.
    pub fn hilbert_order(n_lattice: usize) -> impl Fn(u32, u32) -> i64 {
        assert!(n_lattice > 0);
        let log2n_ceil = u32::BITS - 1 - n_lattice.next_power_of_two().leading_zeros();

        move |l, r| {
            debug_assert!(l < n_lattice as u32);
            debug_assert!(r < n_lattice as u32);
            hilbert_rec(l, r, log2n_ceil)
        }
    }

    // Since the set of query points often occupies the upper triangle { (x, y) : 0 <= x <= y < n },
    // the naive Hilbert order may cause large jumps on the boundary (x == y).
    // To mitigate this, we nest Hilbert curves within even-odd square buckets.
    pub fn bucketed_hilbert_order(n_lattice: usize, n_queries: usize) -> impl Fn(u32, u32) -> u64 {
        assert!(n_lattice > 0);

        let mut bucket_size = n_lattice as u32 / isqrt(n_queries as u32).max(1);
        bucket_size = bucket_size.next_power_of_two();
        let log2b_ceil = u32::BITS - 1 - bucket_size.leading_zeros();

        let bucket_area = bucket_size * bucket_size;
        let n_buckets = (n_lattice as u32).div_ceil(bucket_size);

        move |l, r| {
            debug_assert!(l < n_lattice as u32);
            debug_assert!(r < n_lattice as u32);
            let (bl, sl) = (l / bucket_size, l % bucket_size);
            let (br, sr) = (r / bucket_size, r % bucket_size);
            let (x, y, z) = if bl % 2 == 0 {
                (bl, br, hilbert_rec(bucket_size - sl - 1, sr, log2b_ceil))
            } else {
                (
                    bl,
                    n_buckets - br - 1,
                    hilbert_rec(sl, bucket_size - sr - 1, log2b_ceil),
                )
            };
            (x as u64 * n_buckets as u64 + y as u64) * bucket_area as u64 + z as u64
        }
    }

    fn hilbert_rec(mut x: u32, mut y: u32, mut exp: u32) -> i64 {
        let mut res = 0;
        let mut sign = 1;
        let mut rot = 0;

        while exp > 0 {
            let w_half = 1 << exp - 1;
            let quadrant = match (x < w_half, y < w_half) {
                (true, true) => (rot + 0) % 4,
                (false, true) => (rot + 1) % 4,
                (false, false) => (rot + 2) % 4,
                (true, false) => (rot + 3) % 4,
            };
            rot = match quadrant {
                0 => (rot + 3) % 4,
                1 => (rot + 0) % 4,
                2 => (rot + 0) % 4,
                3 => (rot + 1) % 4,
                _ => unsafe { core::hint::unreachable_unchecked() },
            };

            x &= !w_half;
            y &= !w_half;

            let square_area_half = 1 << 2 * exp - 2;
            res += sign * quadrant as i64 * square_area_half;
            if quadrant == 0 || quadrant == 3 {
                res += sign * (square_area_half - 1);
                sign = -sign;
            };

            exp -= 1;
        }
        res
    }

    // Solve a Traveling Salesperson Problem!
    // Exact solution with naive dp in O(N^2 2^N)
    pub fn tsp_naive<const CAP: usize>(
        xs: impl IntoIterator<Item = (u32, u32)>,
        x_start: (u32, u32),
    ) -> Vec<u32> {
        const INF: u32 = u32::MAX / 3;

        let ps: Vec<_> = xs.into_iter().collect();
        let n = ps.len();

        assert!(1 <= n && n <= CAP);
        assert!(CAP <= 16, "N is too large for brute force TSP");

        let metric = |x: &(u32, u32), y: &(u32, u32)| {
            ((x.0 as i32 - y.0 as i32).abs() + (x.1 as i32 - y.1 as i32).abs()) as u32
        };

        // dp[visited][last]
        let n_mask = 1 << n;
        let mut dp = vec![INF; n_mask * CAP];
        for last in 0..n {
            dp[(1 << last) * CAP + last] = metric(&x_start, &ps[last]);
        }

        // let mut cached_metric: [[_; CAP]; CAP] = [[0; CAP]; CAP];
        // for i in 0..n {
        //     for j in 0..n {
        //         cached_metric[i][j] = metric(&xs[i], &xs[j]);
        //     }
        // }
        // let cached_metric = |x: usize, y: usize| cached_metric[x as usize][y as usize];
        let metric = |i: usize, j: usize| metric(&ps[i], &ps[j]);

        for visited in 1..n_mask {
            for last in 0..n {
                if (visited & (1 << last)) == 0 {
                    continue;
                }
                for prev in 0..n {
                    if (visited & (1 << prev)) == 0 || prev == last {
                        continue;
                    }
                    let prev_mask = visited ^ (1 << last);
                    dp[visited * CAP + last] = dp[visited * CAP + last]
                        .min(dp[prev_mask * CAP + prev] + metric(prev, last));
                }
            }
        }

        let mut min_cost = INF;
        let mut last = 0;
        for i in 0..n {
            if dp[(n_mask - 1) * CAP + i] < min_cost {
                min_cost = dp[(n_mask - 1) * CAP + i];
                last = i;
            }
        }

        let mut path = vec![0; n];
        let mut visited = n_mask - 1;
        'outer: for i in (1..n).rev() {
            path[i] = last as u32;
            for prev in 0..n {
                if (visited & (1 << prev)) != 0
                    && dp[visited * CAP + last]
                        == dp[(visited ^ (1 << last)) * CAP + prev] + metric(prev, last)
                {
                    visited ^= 1 << last;
                    last = prev;
                    continue 'outer;
                }
            }
            panic!("TSP path not found");
        }
        path[0] = last as u32;
        path
    }

    // Solve a 2-approximation to Manhattan TSP,
    // by constructing a Manhattan MST and traversing its Euler tour in O(V log V).
    pub fn tsp_two_approx<T: Clone + Default>(
        xs: &mut [T],
        x_start: (u32, u32),
        mut get_point: impl FnMut(&T) -> (u32, u32),
    ) {
        use super::dset::DisjointSet;
        use super::mst;

        let n = xs.len();
        let n_verts = n + 1;
        let root = 0;
        let ps: Vec<_> = core::iter::once(x_start)
            .chain(xs.iter().map(|x| get_point(x)))
            .collect();

        let mut edges = Vec::with_capacity(n_verts * 8 / 2);
        mst::manhattan_mst_candidates(ps.iter().map(|&(x, y)| (x as i32, y as i32)), |u, v, w| {
            edges.push((u, v, w));
        });

        let mut mst_degree = vec![0u32; n_verts];
        let mut mst_xor_neighbors = vec![0u32; n_verts];
        mst_degree[root] += 2;
        mst::filter_kruskal(
            &mut (n_verts - 1),
            &mut DisjointSet::new(n_verts),
            &mut |u, v, _| {
                mst_degree[u as usize] += 1;
                mst_degree[v as usize] += 1;
                mst_xor_neighbors[u as usize] ^= v;
                mst_xor_neighbors[v as usize] ^= u;
            },
            &mut edges,
        );

        let mut size = vec![1; n_verts];
        let mut topological_order = Vec::with_capacity(n_verts - 1);
        for mut u in 0..n_verts as u32 {
            while mst_degree[u as usize] == 1 {
                let p = mst_xor_neighbors[u as usize];
                mst_degree[u as usize] -= 1;
                mst_degree[p as usize] -= 1;
                topological_order.push((u, p));

                mst_xor_neighbors[p as usize] ^= u;
                size[p as usize] += size[u as usize];

                u = p;
            }
        }
        assert_eq!(
            topological_order.len(),
            n_verts - 1,
            "Invalid tree structure"
        );

        let mut euler_in = size; // 1-based
        for (u, p) in topological_order.into_iter().rev() {
            let last_idx = euler_in[p as usize];
            euler_in[p as usize] -= euler_in[u as usize];
            euler_in[u as usize] = last_idx;
        }
        assert_eq!(euler_in[root], 1);

        let mut res = vec![T::default(); n_verts];
        for i in 1..n_verts {
            res[euler_in[i] as usize - 2] = xs[i - 1].clone();
        }
        for (i, r) in res.into_iter().enumerate().take(n) {
            xs[i] = r;
        }
    }
}
