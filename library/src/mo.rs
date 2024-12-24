pub mod mo {
    // Various ordering strategies for Mo's algorithm.
    //
    // The algorithm has a strong connection to the Traveling Salesman Problem (TSP),
    // as moving from one range query [l, r] to another [l', r'] costs Manhattan distance |l - l'| + |r - r'|.
    // Given a Q random sampled points in a square of side length N,
    // The Beardwood-Halton-Hammersley Theorem establishes an expected minimum path length of O(N sqrt(Q)).
    // Thus, No ordering can escape this asymptotic bound.

    pub fn even_odd_order(n_lattice: usize, n_queries: usize) -> impl Fn(u32, u32) -> (u32, i32) {
        assert!(n_lattice > 0);

        // Time complexity: T ~ O(Q B + N^2/B)
        // => optimal bucket size: B = N/sqrt(Q), so T ~ O(N sqrt(Q))
        let bucket_size = (n_lattice as u32 / (n_queries as f64).sqrt().max(1.0) as u32).max(1);
        move |l, r| {
            let bl = l / bucket_size;
            let br = if bl % 2 == 0 { r as i32 } else { -(r as i32) };
            (bl, br)
        }
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

        let mut bucket_size = (n_lattice as u32 / (n_queries as f64).sqrt().max(1.0) as u32).max(1);
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
                _ => unsafe { std::hint::unreachable_unchecked() },
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
}
