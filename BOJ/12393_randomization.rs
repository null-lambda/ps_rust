use std::{
    collections::{hash_map::Entry, HashMap},
    io::Write,
};

mod simple_io {
    pub struct InputAtOnce(std::str::SplitAsciiWhitespace<'static>);

    impl InputAtOnce {
        pub fn token(&mut self) -> &str {
            self.0.next().unwrap_or_default()
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().unwrap()
        }
    }

    pub fn stdin_at_once() -> InputAtOnce {
        let buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let buf = Box::leak(buf.into_boxed_str());
        InputAtOnce(buf.split_ascii_whitespace())
    }

    pub fn stdout_buf() -> std::io::BufWriter<std::io::Stdout> {
        std::io::BufWriter::new(std::io::stdout())
    }
}

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
        pub fn from_entropy() -> Self {
            let mut seed = 0;
            unsafe {
                if std::arch::x86_64::_rdrand64_step(&mut seed) == 1 {
                    Self(seed)
                } else {
                    panic!("Failed to get entropy");
                }
            }
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

pub mod branch {
    #[inline(always)]
    pub unsafe fn assert_unchecked(b: bool) {
        if !b {
            std::hint::unreachable_unchecked();
        }
    }

    #[cold]
    #[inline(always)]
    pub fn cold() {}

    #[inline(always)]
    pub fn likely(b: bool) -> bool {
        if !b {
            cold();
        }
        b
    }

    #[inline(always)]
    pub fn unlikely(b: bool) -> bool {
        if b {
            cold();
        }
        b
    }
}

const CHUNK_SIZE: usize = 4;
fn test_intersection(b1: &[u16], b2: &[u16]) -> bool {
    b1.iter().any(|&x| b2.contains(&x))
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout_buf();

    for i_tc in 1..=input.value() {
        let n: usize = input.value();
        let mut xs: Vec<(u64, u16)> = (0..n).map(|i| (input.value(), i as u16)).collect();
        let xs_orig = xs.clone();
        xs.sort_unstable();

        let mut sums: HashMap<u64, Vec<[u16; CHUNK_SIZE]>> = Default::default();

        let entry_cap = 10;
        let mut hashmap_cap = 1_000_000;

        let mut rng = rand::SplitMix64::from_entropy();
        let ans = 'outer: loop {
            rng.shuffle(&mut xs);
            for block in xs.chunks_exact(CHUNK_SIZE) {
                let sum = block.iter().map(|(x, _)| *x).sum();
                let block_comp = std::array::from_fn(|i| block[i].1);
                let entry = sums.entry(sum);
                if let Entry::Occupied(ref entry) = entry {
                    branch::cold();
                    if let Some(old) = entry
                        .get()
                        .iter()
                        .find(|&old| !test_intersection(old, &block_comp))
                    {
                        break 'outer [
                            old.map(|i| xs_orig[i as usize].0),
                            std::array::from_fn(|i| block[i].0),
                        ];
                    }
                }
                if hashmap_cap > 0 {
                    hashmap_cap -= 1;
                    let entry = entry.or_default();
                    entry.push(block_comp);
                    if entry.len() > entry_cap {
                        branch::cold();
                        let idx = entry.len() - 1;
                        entry.swap(idx, rng.range_u64(0..entry_cap as u64) as usize);
                        entry.pop();
                    }
                }
            }
        };

        writeln!(output, "Case #{}:", i_tc).unwrap();
        for mut row in ans {
            row.sort_unstable();
            for &x in &row {
                write!(output, "{} ", x).unwrap();
            }
            writeln!(output).unwrap();
        }
    }
}
