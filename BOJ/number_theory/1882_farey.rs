use std::io::Write;

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

pub mod floor_arith {
    pub fn floor_sum_naive(n: i64, p: i64, k: i64, q: i64) -> i64 {
        assert!(n >= 0);
        assert!(q != 0);
        if q < 0 {
            return floor_sum_naive(n, -p, -k, -q);
        }
        (0..n).map(|i| (p * i + k).div_euclid(q)).sum()
    }

    pub fn floor_sum(n: i64, mut p: i64, mut k: i64, mut q: i64) -> i64 {
        assert!(n >= 0);
        assert!(q != 0);
        let mut shift = 0;
        if q < 0 {
            p = -p;
            k = -k;
            q = -q;
        }
        if p < 0 {
            p = -p;
            k -= p * (n - 1);
        }
        if k < 0 {
            let s = k.div_euclid(q);
            shift += s * n;
            k -= s * q;
        }

        shift + floor_sum_u64(n as u64, p as u64, k as u64, q as u64) as i64
    }

    pub fn floor_sum_u64(mut n: u64, mut p: u64, mut k: u64, mut q: u64) -> u64 {
        assert!(q != 0);

        let mut res = 0;
        loop {
            res += n * (n - 1) / 2 * (p / q);
            res += n * (k / q);
            p %= q;
            k %= q;

            let m = p * n + k;
            if m < q {
                break;
            }
            n = m / q;
            k = m % q;
            std::mem::swap(&mut p, &mut q);
        }

        res
    }
}

mod stern_brocot {
    pub type T = u32;

    fn partition_point_nat(right: Option<T>, mut pred: impl FnMut(T) -> bool) -> T {
        todo!("`Replace partition_point_unbounded with partition_point_nat`")
    }

    fn partition_point_unbounded(mut pred: impl FnMut(T) -> bool) -> T {
        // Exponential search
        let mut left = 0;
        let mut right = 0;
        while pred(right) {
            left = right;
            right = right * 2 + 1;
            if right == T::MAX {
                return right;
            }
        }

        // Binary search
        while left < right {
            let mid = left + (right - left) / 2;
            if pred(mid) {
                left = mid + 1;
            } else {
                right = mid;
            }
        }

        right
    }

    // Let
    //    Q_N = { p/q: 1 <= p, 1 <= q <= N },
    //    P = { r in Q_N : pred(r) }.
    //
    // Find max (S U { 0/1 }) and min ((Q_N \ S) U { 1/0 }).
    pub fn partition_point(max_denom: T, mut pred: impl FnMut((T, T)) -> bool) -> [(T, T); 2] {
        assert!(max_denom >= 1);
        let mut l = (0, 1);
        let mut r = (1, 0);
        let mut m = (l.0 + r.0, l.1 + r.1);

        loop {
            let k = partition_point_unbounded(|k| {
                m.1 + l.1 * k <= max_denom && !pred((m.0 + l.0 * k, m.1 + l.1 * k))
            });
            if k > 0 {
                r = (m.0 + l.0 * (k - 1), m.1 + l.1 * (k - 1));
                m = (l.0 + r.0, l.1 + r.1);
                if m.1 > max_denom {
                    break;
                }
            }

            let k = partition_point_unbounded(|k| {
                m.1 + r.1 * (k + 1) <= max_denom && pred((m.0 + r.0 * (k + 1), m.1 + r.1 * (k + 1)))
            });
            l = (m.0 + r.0 * k, m.1 + r.1 * k);
            m = (l.0 + r.0, l.1 + r.1);
            if m.1 > max_denom {
                break;
            }
        }

        [l, r]
    }

    pub fn walk_down(x: (T, T), mut yield_rle_segment: impl FnMut(bool, T)) {
        todo!()
    }
}

pub mod dirichlet_prefix_sum {
    use std::collections::HashMap;

    pub type K = i64;
    pub type T = i64;

    // Let f and g be multiplicative functions.
    // Given O(1) computation oracles for prefix(g * f) and prefix(g),
    // compute prefix(f)(N) in O(N^(3/4)).
    //
    // With a suitable precomputed vector `small` of size O(N^(2/3)),
    // the time complexity reduces to O(N^(2/3)).
    pub struct XudyhSieve<PGF, PG> {
        prefix_g: PG,
        prefix_g_conv_f: PGF,
        g1: T,

        small: Vec<T>,
        large: HashMap<K, T>,
    }

    impl<PG: FnMut(K) -> T, PGF: FnMut(K) -> T> XudyhSieve<PGF, PG> {
        pub fn new(mut prefix_g: PG, prefix_g_conv_f: PGF, small: Vec<T>) -> Self {
            let g1 = prefix_g(1) - prefix_g(0);
            Self {
                prefix_g,
                prefix_g_conv_f,
                g1,
                small,
                large: Default::default(),
            }
        }

        pub fn get(&mut self, n: K) -> T {
            if n < self.small.len() as K {
                return self.small[n as usize];
            }
            if let Some(&res) = self.large.get(&n) {
                return res;
            }

            let mut res = (self.prefix_g_conv_f)(n);
            let mut d = 2;
            while d <= n {
                let t = n / d;
                let d_end = n / t;

                res -= self.get(t) * ((self.prefix_g)(d_end) - (self.prefix_g)(d - 1));

                d = d_end + 1;
            }
            res /= self.g1;

            self.large.insert(n, res);
            res
        }
    }
}

fn linear_sieve(n_max: u32) -> (Vec<u32>, Vec<u32>) {
    let mut min_prime_factor = vec![0; n_max as usize + 1];
    let mut primes = Vec::new();

    for i in 2..=n_max {
        if min_prime_factor[i as usize] == 0 {
            primes.push(i);
        }
        for &p in primes.iter() {
            if i * p > n_max {
                break;
            }
            min_prime_factor[(i * p) as usize] = p;
            if i % p == 0 {
                break;
            }
        }
    }

    (min_prime_factor, primes)
}

fn gen_mobius(min_prime_factor: &[u32]) -> Vec<i8> {
    let n_max = min_prime_factor.len() - 1;
    let mut mu = vec![0; n_max + 1];
    mu[1] = 1;

    for i in 2..=n_max {
        let p = min_prime_factor[i];
        if p == 0 {
            mu[i] = -1;
        } else {
            let m = i as u32 / p;
            mu[i] = if m % p == 0 { 0 } else { -mu[m as usize] };
        }
    }
    mu
}

fn gen_mertens() -> impl FnMut(i64) -> i64 {
    let (mpf, _) = linear_sieve(5e3 as u32);
    let mu = gen_mobius(&mpf);
    let mut mertens = mu.iter().map(|&x| x as i64).collect::<Vec<_>>();
    for i in 1..mertens.len() as usize {
        mertens[i] += mertens[i - 1];
    }

    let mut mertens = dirichlet_prefix_sum::XudyhSieve::new(|n| n, |_| 1, mertens);
    move |i| mertens.get(i)
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let mut mertens = gen_mertens();

    let n: u32 = input.value();
    let k: i64 = input.value();

    let mut count_le = |x: (u32, u32)| {
        // count_le(x)
        // = sum_{p, q in } [gcd(p,q) = 1][p/q <= x]
        // = sum_{d in [1,N]} mu(d) sum_{p, q in [1,N]} [d|(p,q)][p <= qx]
        // = sum_{d in [1,N]} mu(d) sum_{i, j in [1,N/d]} [i <= jx]
        // = sum_{d in [1,N]} mu(d) sum_{j in [1,N/d]} floor(jx)
        // = sum_{dj <= N} mu(d) floor(jx)
        // = sum_{j in [1,N]} mertens(floor(n/j)) floor(jx)

        let mut res = 0;
        let mut j = 1;
        while j <= n {
            let t = n / j;
            let j_end = n / t;

            let df = floor_arith::floor_sum_u64(j_end as u64 + 1, x.0 as u64, 0, x.1 as u64)
                - floor_arith::floor_sum_u64(j as u64, x.0 as u64, 0, x.1 as u64);
            res += mertens(t as i64) * df as i64;
            j = j_end + 1;
        }
        res
    };

    let [l, _r] = stern_brocot::partition_point(n, |x| count_le(x) <= k);
    writeln!(output, "{} {}", l.0, l.1).unwrap();
}
