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

fn mertens() -> impl FnMut(i64) -> i64 {
    let (mpf, _) = linear_sieve(2.5e6 as u32);
    let mu = gen_mobius(&mpf);
    let mut mertens = mu.iter().map(|&x| x as i64).collect::<Vec<_>>();
    for i in 1..mertens.len() as usize {
        mertens[i] += mertens[i - 1];
    }

    let mut mertens = XudyhSieve::new(|n| n, |_| 1, mertens);
    move |i| mertens.get(i)
}
