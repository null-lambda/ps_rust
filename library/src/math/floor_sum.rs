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
