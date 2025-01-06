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

mod mo {
    // Various ordering strategies for Mo's algorithm.
    //
    // The algorithm has a strong connection to the Traveling Salesman Problem {TSP},
    // as moving from one range query [l, r] to another [l', r'] costs Manhattan distance |l - l'| + |r - r'|.
    // Given a Q random sampled points in a square of side length N,
    // The Beardwood-Halton-Hammersley Theorem establishes an expected minimum path length of O(N sqrt(Q)).
    // Thus, No ordering can escape this asymptotic bound.

    // Simple while efficient. Requires almost no precomputation cost.
    pub fn sort_by_even_odd_order(
        n_lattice: usize,
        n_queries: usize,
        intervals: impl IntoIterator<Item = (u32, u32)>,
    ) -> impl Iterator<Item = (u32, u32, u32)> + Clone {
        // Time complexity: T ~ O(Q B + N^2/B)
        // => optimal bucket size: B = N/sqrt(Q), so T ~ O(N sqrt(Q))
        let bucket_size = (n_lattice as u32 / (f64::sqrt(n_queries as f64) as u32)).max(1);

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

        let mut bucket_size = (n_lattice as u32 / (f64::sqrt(n_queries as f64) as u32)).max(1);
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
        x: (u32, u32),
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
            dp[(1 << last) * CAP + last] = metric(&x, &ps[last]);
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
}

pub mod bucket_decomp {
    pub trait Group {
        type X: Clone;
        fn id(&self) -> Self::X;
        fn add(&self, lhs: &Self::X, rhs: &Self::X) -> Self::X;
        fn sub(&self, lhs: &Self::X, rhs: &Self::X) -> Self::X;
    }

    #[derive(Clone)]
    pub struct SqrtDecomp<G: Group> {
        n: usize,
        group: G,
        block_size: usize,
        block_prefix: Vec<G::X>,
        inner_prefix: Vec<G::X>,
    }

    impl<G: Group> SqrtDecomp<G> {
        pub fn new(n: usize, group: G) -> Self {
            let block_size = ((n as f64).sqrt() as usize).max(1);
            let block_count = n / block_size + 1;
            let data = vec![group.id(); block_count * block_size];
            let blocks = vec![group.id(); block_count];
            Self {
                n,
                group,
                block_size,
                block_prefix: blocks,
                inner_prefix: data,
            }
        }

        pub fn add(&mut self, idx: usize, value: G::X) {
            debug_assert!(idx < self.n);
            let (block, s) = (idx / self.block_size, idx % self.block_size);
            for i in self.block_size * block + s + 1..self.block_size * (block + 1) {
                self.inner_prefix[i] = self.group.add(&self.inner_prefix[i], &value);
            }
            for b in block + 1..self.block_prefix.len() {
                self.block_prefix[b] = self.group.add(&self.block_prefix[b], &value);
            }
        }

        pub fn sum_prefix(&self, idx: usize) -> G::X {
            debug_assert!(idx <= self.n);
            let block = idx / self.block_size;
            self.group
                .add(&self.block_prefix[block], &self.inner_prefix[idx])
        }

        pub fn get(&self, idx: usize) -> G::X {
            self.inner_prefix[idx].clone()
        }
    }
}

struct PairedSum;

impl bucket_decomp::Group for PairedSum {
    type X = (i32, i64);
    fn id(&self) -> Self::X {
        Default::default()
    }
    fn add(&self, lhs: &Self::X, rhs: &Self::X) -> Self::X {
        (lhs.0 + rhs.0, lhs.1 + rhs.1)
    }
    fn sub(&self, lhs: &Self::X, rhs: &Self::X) -> Self::X {
        (lhs.0 - rhs.0, lhs.1 - rhs.1)
    }
}

fn solve(
    n: usize,
    q: usize,
    xs: Vec<i64>,
    queries: impl IntoIterator<Item = (u32, u32, u32)> + Clone,
    ans: &mut [i64],
) {
    // Sweepline Mo's
    // https://codeforces.com/blog/entry/81716
    // https://infossm.github.io/blog/2023/04/23/sweepline-mo/
    //
    // ## Definitions
    //     p([l,r], k) = [ counts and sums of { y kn [l,r] : y < x[k] } and its complement ]
    //     f([l,r], k) = a transition delta,
    //     which is given by a certain linear combination of p([l,r], k) induced by x[k]
    //         => f[l,r], k) = F(r,k) - F(l-1,k) where F(x, k) := f([1,x], k)
    let (mut start, mut end) = (2, 1);
    let mut grouped_trans = vec![vec![]; n + 1];
    for (l, r, i) in queries.clone() {
        if start > l {
            // delta = f([start, end], start-1) + ... + f([l+1, end], l)
            //       = (F(end, l) + ... + F(end, start-1))    /* sweeping */
            //       - (F(l, l) + ... + F(start-1, start-1))    /* prefix sum or segtree */
            grouped_trans[end as usize].push((l..=start - 1, i, 1));
        }
        if start < l {
            // delta = -f([start+1, end], start) - ... - f([l, end], l-1)
            grouped_trans[end as usize].push((start..=l - 1, i, -1));
        }
        start = l;

        if end < r {
            // delta = f([start, end], end+1) + ... + f([start, r-1], r)
            //       = (F(end, end+1) + ... + F(r-1,r))    /* prefix sum or segtree */
            //       - (F(start-1, end+1) + ... + F(start-1, r))    /* sweeping */
            grouped_trans[start as usize - 1].push((end + 1..=r, i, -1));
        }
        if end > r {
            // delta = -f([start, r], r+1) - ... - f([start, end-1], end)
            grouped_trans[start as usize - 1].push((r + 1..=end, i, 1));
        }
        end = r;
    }

    let mut active = bucket_decomp::SqrtDecomp::new(n + 1, PairedSum);
    let mut diagonal_prefix_sh0 = vec![0; n + 1]; // sum ... + F(i,i)
    let mut diagonal_prefix_sh1 = vec![0; n + 1]; // sum .. + F(i,i+1)
    let mut delta = vec![0; q];
    let (mut count_all, mut sum_all) = (0, 0);
    for i in 1..=n {
        let x = xs[i];
        active.add(x as usize, (1, x));
        count_all += 1;
        sum_all += x;

        let f_prefix_i = |j: usize| {
            if !(1..=n).contains(&j) {
                return 0;
            };
            let x = xs[j];
            let (count_lt, sum_lt) = active.sum_prefix(x as usize);
            let (count_ge, sum_ge) = (count_all - count_lt, sum_all - sum_lt);
            sum_ge - sum_lt - (count_ge - count_lt) as i64 * x
        };

        diagonal_prefix_sh0[i] = diagonal_prefix_sh0[i - 1] + f_prefix_i(i);
        diagonal_prefix_sh1[i] = diagonal_prefix_sh1[i - 1] + f_prefix_i(i + 1);

        for (j_range, i_query, sign) in grouped_trans[i].iter().cloned() {
            for j in j_range {
                delta[i_query as usize] += sign * f_prefix_i(j as usize);
            }
        }
    }

    let (mut start, mut end) = (2, 1);
    let mut acc = 0;
    for (l, r, i) in queries {
        if start != l {
            delta[i as usize] -=
                diagonal_prefix_sh0[start as usize - 1] - diagonal_prefix_sh0[l as usize - 1];
            start = l;
        }
        if end != r {
            delta[i as usize] +=
                diagonal_prefix_sh1[r as usize - 1] - diagonal_prefix_sh1[end as usize - 1];
            end = r;
        }
        acc += delta[i as usize];
        ans[i as usize] = acc;
    }
}

pub fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let q: usize = input.value();
    let xs: Vec<i64> = std::iter::once(0)
        .chain((1..=n).map(|_| input.value()))
        .collect();

    let queries = (0..q).map(|_| (input.value::<u32>(), input.value::<u32>()));
    let mut ans = vec![0; q];
    match q {
        ..=12 => {
            let queries: Vec<_> = queries.collect();
            let path = mo::tsp_naive::<12>(queries.iter().cloned(), (2, 1));
            let queries = path.into_iter().map(|i| {
                let (l, r) = queries[i as usize];
                (l, r, i)
            });
            solve(n, q, xs, queries, &mut ans);
        }
        _ => {
            let queries = mo::sort_by_even_odd_order(n + 1, q, queries);
            solve(n, q, xs, queries, &mut ans);
        }
    }

    for a in ans {
        writeln!(output, "{}", a).unwrap();
    }
}
