use std::io::Write;

use dset::DisjointMap;
use universal_hash::UniversalHasher;

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

    pub fn stdin_at_once<'a>() -> InputAtOnce<'a> {
        let _buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let iter = _buf.split_ascii_whitespace();
        let iter = unsafe { std::mem::transmute(iter) };
        InputAtOnce { _buf, iter }
    }

    pub fn stdout() -> std::io::BufWriter<std::io::Stdout> {
        std::io::BufWriter::new(std::io::stdout())
    }
}

mod dset {
    use std::cell::Cell;
    use std::mem::{self, MaybeUninit};

    pub struct DisjointMap<T> {
        // Represents parent if >= 0, size if < 0
        parent_or_size: Vec<Cell<i32>>,
        values: Vec<MaybeUninit<T>>,
    }

    impl<T> DisjointMap<T> {
        pub fn new(values: impl IntoIterator<Item = T>) -> Self {
            let values: Vec<_> = values.into_iter().map(MaybeUninit::new).collect();
            let n = values.len();
            Self {
                parent_or_size: vec![Cell::new(-1); n],
                values,
            }
        }

        fn get_parent_or_size(&self, u: usize) -> Result<usize, u32> {
            let x = self.parent_or_size[u].get();
            if x >= 0 {
                Ok(x as usize)
            } else {
                Err((-x) as u32)
            }
        }

        fn set_parent(&self, u: usize, p: usize) {
            self.parent_or_size[u].set(p as i32);
        }

        fn set_size(&self, u: usize, s: u32) {
            self.parent_or_size[u].set(-(s as i32));
        }

        pub fn find_root_with_size(&self, u: usize) -> (usize, u32) {
            match self.get_parent_or_size(u) {
                Ok(p) => {
                    let (root, size) = self.find_root_with_size(p);
                    self.set_parent(u, root);
                    (root, size)
                }
                Err(size) => (u, size),
            }
        }

        pub fn find_root(&self, u: usize) -> usize {
            self.find_root_with_size(u).0
        }

        pub fn get_size(&self, u: usize) -> u32 {
            self.find_root_with_size(u).1
        }

        pub fn get_mut(&mut self, u: usize) -> &mut T {
            let r = self.find_root(u);
            unsafe { self.values[r].assume_init_mut() }
        }

        // Returns true if two sets were previously disjoint
        pub fn merge(
            &mut self,
            u: usize,
            v: usize,
            mut combine_values: impl FnMut(&mut T, T),
        ) -> bool {
            let (mut u, size_u) = self.find_root_with_size(u);
            let (mut v, size_v) = self.find_root_with_size(v);
            if u == v {
                return false;
            }

            if size_u < size_v {
                mem::swap(&mut u, &mut v);
            }

            let value_v = unsafe {
                std::mem::replace(&mut self.values[v], MaybeUninit::uninit()).assume_init()
            };
            combine_values(unsafe { self.values[u].assume_init_mut() }, value_v);
            self.set_parent(v, u);
            self.set_size(u, size_u + size_v);
            true
        }
    }

    impl<T> Drop for DisjointMap<T> {
        fn drop(&mut self) {
            for u in 0..self.parent_or_size.len() {
                if self.get_parent_or_size(u).is_err() {
                    unsafe {
                        self.values[u].assume_init_drop();
                    }
                }
            }
        }
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

#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
struct ComponentAgg {
    count: u32,
    min_node: u32,
    lock: u32,
}

impl ComponentAgg {
    fn merge(&mut self, other: Self) {
        assert!(self.lock == 0);
        assert!(other.lock == 0);
        self.count += other.count;
        self.min_node = self.min_node.min(other.min_node);
    }

    fn proj_u64(&self) -> u64 {
        if self.lock != 0 {
            0
        } else {
            (self.count as u64) << 32 | (self.min_node as u64)
        }
    }
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let mut rng = rand::SplitMix64::from_entropy().unwrap();
    for tc in 0..input.value() {
        let hasher = UniversalHasher::new(&mut rng);

        let n: usize = input.value();
        let mut edges = vec![];
        let mut neighbors = vec![vec![]; n];
        for _ in 0..n - 1 {
            let u = input.value::<u32>() - 1;
            let v = input.value::<u32>() - 1;
            edges.push([u, v]);
            neighbors[u as usize].push(v);
            neighbors[v as usize].push(u);
        }

        // store (min_node, number of indep node) in each component of dmap
        // track all updates in reverse, and compare delta history offline
        let q: usize = input.value();
        let mut indep = vec![vec![]; 2];
        let mut is_rigid = vec![vec![false; n]; 2];
        for _ in 0..q {
            for g in 0..2 {
                let u = input.value::<u32>() - 1;
                indep[g].push(u);
                is_rigid[g][u as usize] = true;
            }
        }

        let mut history = vec![vec![[0u128; 2]; q]; 2];
        for g in 0..2 {
            assert_eq!(is_rigid[g].iter().filter(|&&b| b).count(), q);

            let mut conn = dset::DisjointMap::new((0..n as u32).map(|u| ComponentAgg {
                count: is_rigid[g][u as usize] as u32,
                min_node: u,
                lock: 0,
            }));
            let mut conn_hash = (0..n)
                .map(|u| hasher.hash(conn.get_mut(u).proj_u64() as u128))
                .fold(0, |acc, x| acc ^ x);
            let on_merge = |conn_hash: &mut u128, x: &mut ComponentAgg, y: ComponentAgg| {
                *conn_hash ^= hasher.hash(x.proj_u64() as u128);
                *conn_hash ^= hasher.hash(y.proj_u64() as u128);
                x.merge(y);
                *conn_hash ^= hasher.hash(x.proj_u64() as u128);
            };

            let mut rigid_hash = (0..n)
                .filter(|&u| is_rigid[g][u])
                .map(|u| hasher.hash(u as u128))
                .fold(0, |acc, x| acc ^ x);

            let mut degree = vec![0u32; n];
            let mut xor_indep = vec![0u32; n];
            for &u in &indep[g] {
                let c = conn.get_mut(u as usize);
                conn_hash ^= hasher.hash(c.proj_u64() as u128);
                c.lock += 1;
                conn_hash ^= hasher.hash(c.proj_u64() as u128);

                for &v in &neighbors[u as usize] {
                    degree[v as usize] += 1;
                    xor_indep[v as usize] ^= u;

                    let c = conn.get_mut(v as usize);
                    conn_hash ^= hasher.hash(c.proj_u64() as u128);
                    c.lock += 1;
                    conn_hash ^= hasher.hash(c.proj_u64() as u128);
                }
            }
            for &[u, v] in &edges {
                if conn.get_mut(u as usize).lock == 0 && conn.get_mut(v as usize).lock == 0 {
                    conn.merge(u as usize, v as usize, |x, y| {
                        on_merge(&mut conn_hash, x, y)
                    });
                }
            }

            let mut queue = vec![];
            let mut timer = 0;
            for u in 0..n as u32 {
                if degree[u as usize] == 1 {
                    let w = xor_indep[u as usize];
                    if is_rigid[g][w as usize] {
                        is_rigid[g][w as usize] = false;
                        rigid_hash ^= hasher.hash(w as u128);
                        queue.push(w);
                    }
                }
            }

            let mut e_stack = vec![];

            let mut unmark = |conn: &mut DisjointMap<ComponentAgg>,
                              conn_hash: &mut u128,
                              queue: &mut Vec<_>,
                              e_stack: &mut Vec<_>,
                              is_rigid: &mut [bool],
                              rigid_hash: &mut u128,
                              u: u32| {
                assert!(!is_rigid[u as usize]);

                let c = conn.get_mut(u as usize);
                *conn_hash ^= hasher.hash(c.proj_u64() as u128);
                c.lock -= 1;
                *conn_hash ^= hasher.hash(c.proj_u64() as u128);
                if c.lock == 0 {
                    e_stack.push(u);
                }

                for &v in &neighbors[u as usize] {
                    degree[v as usize] -= 1;
                    xor_indep[v as usize] ^= u;

                    let w = xor_indep[v as usize];
                    if degree[v as usize] == 1 && is_rigid[w as usize] {
                        is_rigid[w as usize] = false;
                        *rigid_hash ^= hasher.hash(w as u128);
                        queue.push(w);
                    }

                    let c = conn.get_mut(v as usize);
                    *conn_hash ^= hasher.hash(c.proj_u64() as u128);
                    c.lock -= 1;
                    *conn_hash ^= hasher.hash(c.proj_u64() as u128);
                    if c.lock == 0 {
                        e_stack.push(v);
                    }
                }
            };

            let unmark_components = |conn: &mut dset::DisjointMap<ComponentAgg>,
                                     e_stack: &mut Vec<u32>,
                                     conn_hash: &mut u128| {
                for u in e_stack.drain(..) {
                    for &v in &neighbors[u as usize] {
                        if conn.get_mut(u as usize).lock == 0 && conn.get_mut(v as usize).lock == 0
                        {
                            conn.merge(u as usize, v as usize, |x, y| on_merge(conn_hash, x, y));
                        }
                    }
                }
            };

            while let Some(&u) = queue.get(timer) {
                timer += 1;
                unmark(
                    &mut conn,
                    &mut conn_hash,
                    &mut queue,
                    &mut e_stack,
                    &mut is_rigid[g],
                    &mut rigid_hash,
                    u,
                );
            }
            unmark_components(&mut conn, &mut e_stack, &mut conn_hash);

            for i in (0..q).rev() {
                history[g][i] = [conn_hash, rigid_hash];

                let u = indep[g][i];

                let c = conn.get_mut(u as usize);
                conn_hash ^= hasher.hash(c.proj_u64() as u128);
                c.count -= 1;
                conn_hash ^= hasher.hash(c.proj_u64() as u128);

                if is_rigid[g][u as usize] {
                    is_rigid[g][u as usize] = false;
                    rigid_hash ^= hasher.hash(u as u128);

                    queue.push(u);
                    while let Some(&u) = queue.get(timer) {
                        timer += 1;
                        unmark(
                            &mut conn,
                            &mut conn_hash,
                            &mut queue,
                            &mut e_stack,
                            &mut is_rigid[g],
                            &mut rigid_hash,
                            u,
                        );
                    }
                }
                unmark_components(&mut conn, &mut e_stack, &mut conn_hash);
            }

            assert_eq!(
                conn_hash,
                hasher.hash(
                    ComponentAgg {
                        count: 0,
                        min_node: 0,
                        lock: 0,
                    }
                    .proj_u64() as u128
                )
            );
            assert_eq!(rigid_hash, 0);

            assert_eq!(is_rigid[g].iter().filter(|&&b| b).count(), 0);
            assert_eq!(queue.len(), q);
            assert_eq!(conn.get_size(0), n as u32);
        }

        // println!("{:?}", &history[0]);
        // println!("{:?}", &history[1]);

        let ans = (0..q).filter(|&i| history[0][i] == history[1][i]).count();

        writeln!(output, "Case #{}", tc + 1).unwrap();
        writeln!(output, "{}", ans).unwrap();
    }
}
