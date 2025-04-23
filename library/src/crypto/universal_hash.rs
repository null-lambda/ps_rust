mod rand {
    // Written in 2015 by Sebastiano Vigna (vigna@acm.org)
    // https://xoshiro.di.unimi.it/splitmix64.c
    use std::ops::Range;

    pub struct SplitMix64(u64);

    impl SplitMix64 {
        pub fn new(seed: u64) -> Self {
            assert_ne!(seed, 0);
            Self(seed)
        }

        // Available on x86-64 and target feature rdrand only.
        #[cfg(target_arch = "x86_64")]
        pub fn from_entropy() -> Option<Self> {
            let mut seed = 0;
            unsafe { (std::arch::x86_64::_rdrand64_step(&mut seed) == 1).then(|| Self(seed)) }
        }
        #[cfg(not(target_arch = "x86_64"))]
        pub fn from_entropy() -> Self {
            use std::time::{SystemTime, UNIX_EPOCH};
            let seed = SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos();
            Self(seed as u64)
        }

        pub fn next_u64(&mut self) -> u64 {
            self.0 = self.0.wrapping_add(0x9e3779b97f4a7c15);
            let mut x = self.0;
            x = (x ^ (x >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
            x = (x ^ (x >> 27)).wrapping_mul(0x94d049bb133111eb);
            x ^ (x >> 31)
        }

        pub fn range_u64(&mut self, range: Range<u64>) -> u64 {
            let Range { start, end } = range;
            debug_assert!(start < end);

            let width = end - start;
            let test = (u64::MAX - width) % width;
            loop {
                let value = self.next_u64();
                if value >= test {
                    return start + value % width;
                }
            }
        }

        pub fn shuffle<T>(&mut self, xs: &mut [T]) {
            let n = xs.len();
            for i in 0..n - 1 {
                let j = self.range_u64(i as u64..n as u64) as usize;
                xs.swap(i, j);
            }
        }
    }
}

pub mod universal_hash {
    use crate::rand;

    const P: u128 = (1u128 << 127) - 1;

    fn mul_128(x: u128, y: u128) -> (u128, u128) {
        let [x0, x1] = [x & ((1u128 << 64) - 1), x >> 64];
        let [y0, y1] = [y & ((1u128 << 64) - 1), y >> 64];
        let (mid, carry1) = (x0 * y1).overflowing_add(x1 * y0);
        let (lower, carry2) = (x0 * y0).overflowing_add(mid << 64);
        let upper = (x1 * y1)
            .wrapping_add(mid >> 64)
            .wrapping_add(carry1 as u128 + carry2 as u128);
        (lower, upper)
    }

    fn mod_p(mut t: u128) -> u128 {
        t = (t & P) + (t >> 127);
        if t >= P {
            t - P
        } else {
            t
        }
    }

    fn mul_mod_p(a: u128, x: u128) -> u128 {
        let (lo, hi) = mul_128(a, x);
        let t = lo.wrapping_add(hi.wrapping_mul(2));
        mod_p(t)
    }

    pub struct UniversalHasher {
        a: u128,
        b: u128,
    }

    impl UniversalHasher {
        pub fn new(rng: &mut rand::SplitMix64) -> Self {
            let mut next_u128 = || {
                let lower = rng.next_u64();
                let upper = rng.next_u64();
                ((lower as u128) << 64) | upper as u128
            };
            let a = next_u128().wrapping_mul(2).wrapping_add(1) % P;
            let b = next_u128() % P;
            Self { a, b }
        }

        pub fn hash(&self, x: u128) -> u128 {
            mod_p(mul_mod_p(self.a, x).wrapping_add(self.b))
        }
    }
}
