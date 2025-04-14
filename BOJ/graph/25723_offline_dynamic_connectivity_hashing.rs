use std::{collections::HashMap, io::Write, ops::Range};

use dset::DisjointSet;
use universal_hash::UniversalHasher;

mod fast_io {
    use std::fs::File;
    use std::io::BufWriter;
    use std::os::unix::io::FromRawFd;

    extern "C" {
        fn mmap(addr: usize, length: usize, prot: i32, flags: i32, fd: i32, offset: i64)
            -> *mut u8;
        fn fstat(fd: i32, stat: *mut usize) -> i32;
    }

    pub struct InputAtOnce {
        buf: &'static [u8],
    }

    impl InputAtOnce {
        fn skip(&mut self) {
            loop {
                match self.buf {
                    &[..=b' ', ..] => self.buf = &self.buf[1..],
                    _ => break,
                }
            }
        }

        fn u32_noskip(&mut self) -> u32 {
            let mut acc = 0;
            loop {
                match self.buf {
                    &[b'0'..=b'9', ..] => acc = acc * 10 + (self.buf[0] - b'0') as u32,
                    _ => break,
                }
                self.buf = &self.buf[1..];
            }
            acc
        }

        pub fn token(&mut self) -> &'static str {
            self.skip();
            let start = self.buf.as_ptr();
            loop {
                match self.buf {
                    &[..=b' ', ..] => break,
                    _ => self.buf = &self.buf[1..],
                }
            }
            let end = self.buf.as_ptr();
            unsafe {
                std::str::from_utf8_unchecked(std::slice::from_raw_parts(
                    start,
                    end.offset_from(start) as usize,
                ))
            }
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().unwrap()
        }

        pub fn u32(&mut self) -> u32 {
            self.skip();
            self.u32_noskip()
        }

        pub fn i32(&mut self) -> i32 {
            self.skip();
            match self.buf {
                &[b'-', ..] => {
                    self.buf = &self.buf[1..];
                    -(self.u32_noskip() as i32)
                }
                _ => self.u32_noskip() as i32,
            }
        }
    }

    pub fn stdin() -> InputAtOnce {
        let mut stat = [0; 18];
        unsafe { fstat(0, (&mut stat).as_mut_ptr()) };
        let buf = unsafe { mmap(0, stat[6], 1, 2, 0, 0) };
        let buf =
            unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(buf, stat[6])) };
        InputAtOnce {
            buf: buf.as_bytes(),
        }
    }

    pub fn stdout() -> BufWriter<File> {
        let stdout = unsafe { File::from_raw_fd(1) };
        BufWriter::with_capacity(1 << 16, stdout)
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
            let a = (next_u128() % (P - 1)) + 1;
            let b = next_u128() % P;
            Self { a, b }
        }

        pub fn hash(&self, x: u128) -> u128 {
            mod_p(mul_mod_p(self.a, x).wrapping_add(self.b))
        }
    }
}

mod mem_reserved {
    use std::mem::MaybeUninit;

    pub struct Stack<T> {
        pos: Box<[MaybeUninit<T>]>,
        len: usize,
    }

    impl<T> Stack<T> {
        pub fn with_capacity(capacity: usize) -> Self {
            Self {
                pos: (0..capacity).map(|_| MaybeUninit::uninit()).collect(),
                len: 0,
            }
        }

        #[must_use]
        pub fn push(&mut self, value: T) -> bool {
            if self.len == self.pos.len() {
                return false;
            }
            unsafe { self.push_unchecked(value) };
            return true;
        }

        pub unsafe fn push_unchecked(&mut self, value: T) {
            *self.pos.get_unchecked_mut(self.len) = MaybeUninit::new(value);
            self.len += 1;
        }

        pub fn pop(&mut self) -> Option<T> {
            self.len = self.len.checked_sub(1)?;
            Some(unsafe { self.pos.get_unchecked(self.len).assume_init_read() })
        }
    }
}

mod dset {
    use std::mem;

    use crate::{mem_reserved::Stack, universal_hash::UniversalHasher};

    pub struct DisjointSet {
        parent: Vec<u32>,
        rank: Vec<u32>,

        component_hash: Vec<u128>,
        total_hash: u128,
        hasher: UniversalHasher,

        history: Stack<(u32, u32, u128)>,
    }

    impl DisjointSet {
        pub fn new(n: usize, hasher: UniversalHasher) -> Self {
            let component_hash: Vec<_> = (0..n).map(|u| hasher.hash(u as u128)).collect();
            Self {
                parent: (0..n).map(|x| x as u32).collect(),
                rank: vec![0; n],

                total_hash: component_hash
                    .iter()
                    .fold(0, |acc, &x| acc ^ hasher.hash(x)),
                component_hash,
                hasher,

                history: Stack::with_capacity(n - 1),
            }
        }

        pub fn find_root(&self, u: usize) -> usize {
            if self.parent[u] == u as u32 {
                u
            } else {
                self.find_root(self.parent[u] as usize)
            }
        }

        // Returns true if two sets were previously disjoint
        pub fn merge(&mut self, u: usize, v: usize) -> bool {
            let mut u = self.find_root(u);
            let mut v = self.find_root(v);
            if u == v {
                return false;
            }

            let rank_u = self.rank[u];
            let rank_v = self.rank[v];
            if rank_u < rank_v {
                mem::swap(&mut u, &mut v);
            }

            let old_rank_u = self.rank[u];
            self.parent[v] = u as u32;
            if rank_u == rank_v {
                self.rank[u] += 1;
            }

            let old_total_hash = self.total_hash;
            self.total_hash ^= self.hasher.hash(self.component_hash[u]);
            self.total_hash ^= self.hasher.hash(self.component_hash[v]);
            self.component_hash[u] ^= self.component_hash[v];
            self.total_hash ^= self.hasher.hash(self.component_hash[u]);

            unsafe {
                self.history
                    .push_unchecked((v as u32, old_rank_u as u32, old_total_hash))
            };
            true
        }

        pub fn hash(&self) -> u128 {
            self.total_hash
        }

        pub fn rollback(&mut self) -> bool {
            let Some((v, rank_u, old_total_hash)) = self.history.pop() else {
                return false;
            };

            let u = self.parent[v as usize] as usize;
            self.rank[u] = rank_u;
            self.parent[v as usize] = v;

            self.component_hash[u] ^= self.component_hash[v as usize];
            self.total_hash = old_total_hash;

            true
        }
    }
}

fn partition_in_place<T>(xs: &mut [T], mut pred: impl FnMut(&T) -> bool) -> (&mut [T], &mut [T]) {
    let n = xs.len();
    let mut i = 0;
    for j in 0..n {
        if pred(&xs[j]) {
            xs.swap(i, j);
            i += 1;
        }
    }
    xs.split_at_mut(i)
}

fn offline_dynamic_conn_dnc(
    dset: &mut dset::DisjointSet,
    intervals: &mut [(Range<u32>, [u32; 2])],
    yield_state: &mut impl FnMut(u32, &dset::DisjointSet),
    time_range: Range<u32>,
) {
    debug_assert!(time_range.start < time_range.end);
    let (intervals, _) = partition_in_place(intervals, |(interval, _)| {
        !(interval.end <= time_range.start || time_range.end <= interval.start)
    });
    let (full, partial) = partition_in_place(intervals, |(interval, _)| {
        interval.start <= time_range.start && time_range.end <= interval.end
    });

    let mut full_count = 0;
    for &(_, [x, y]) in full.iter() {
        full_count += dset.merge(x as usize, y as usize) as u32;
    }

    if time_range.start + 1 == time_range.end {
        yield_state(time_range.start, &dset);
    } else {
        let mid = (time_range.start + time_range.end) / 2;
        offline_dynamic_conn_dnc(dset, partial, yield_state, time_range.start..mid);
        offline_dynamic_conn_dnc(dset, partial, yield_state, mid..time_range.end);
    }

    for _ in 0..full_count {
        dset.rollback();
    }
}

fn main() {
    let mut input = fast_io::stdin();
    let mut output = fast_io::stdout();

    for _ in 0..input.value() {
        let k: usize = input.value();
        let n: usize = input.value();
        let m: usize = input.value();

        let mut active_in_prefix = HashMap::new();
        for _ in 0..m {
            let x = input.value::<u32>() - 1;
            let y = input.value::<u32>() - 1;
            active_in_prefix.insert([x, y], 0u32);
        }

        let mut parent = vec![0u32; k];
        let mut weights = vec![(0, !0, !0); k];
        for u in 1..k {
            let p = input.value::<u32>() - 1;
            parent[u] = p;
            let cmd = if input.token() == "add" { 1i8 } else { -1i8 };
            let x = input.value::<u32>() - 1;
            let y = input.value::<u32>() - 1;
            weights[u] = (cmd, x, y);
        }

        let mut size = vec![1u32; k];
        for u in (1..k).rev() {
            let p = parent[u] as usize;
            size[p] += size[u];
        }

        let mut euler_in: Vec<_> = size.iter().map(|&s| s * 2 - 1).collect();
        for u in 1..k {
            let p = parent[u] as usize;
            euler_in[p as usize] -= euler_in[u as usize] + 1;
            euler_in[u as usize] += euler_in[p as usize];
        }

        let mut euler_tour = vec![!0; 2 * k];
        for u in 0..k {
            let euler_in = euler_in[u] as usize - 1;
            let euler_out = euler_in + size[u] as usize * 2 - 1;
            euler_tour[euler_in] = u as i32;
            euler_tour[euler_out] = -(u as i32 + 1);
        }
        while matches!(euler_tour.last(), Some(..=-1)) {
            euler_tour.pop();
        }

        let mut intervals = vec![];
        let t_bound = euler_tour.len();
        for t in 1..t_bound {
            let u = euler_tour[t];
            let (u, flip) = if u >= 0 {
                (u as usize, 1i8)
            } else {
                ((-u - 1) as usize, -1i8)
            };

            let (cmd, x, y) = weights[u];
            match flip * cmd {
                1i8 => {
                    assert!(active_in_prefix.insert([x, y], t as u32).is_none());
                }
                -1i8 => {
                    let t_start = active_in_prefix.remove(&[x, y]).unwrap();
                    intervals.push((t_start..t as u32, [x, y]));
                }
                _ => panic!(),
            }
        }

        for ([x, y], start) in active_in_prefix {
            intervals.push((start..t_bound as u32, [x, y]));
        }

        // let mut rng = rand::SplitMix64::new(42);
        let mut rng = rand::SplitMix64::from_entropy().unwrap();
        let hasher = UniversalHasher::new(&mut rng);
        let mut partitions = HashMap::<_, Vec<u32>>::new();
        offline_dynamic_conn_dnc(
            &mut DisjointSet::new(n, hasher),
            &mut intervals,
            &mut |t, conn| {
                let u = euler_tour[t as usize];
                if u >= 0 {
                    partitions.entry(conn.hash()).or_default().push(u as u32);
                }
            },
            0..t_bound as u32,
        );

        writeln!(output, "{}", partitions.len()).ok();
        for (_, group) in partitions {
            write!(output, "{} ", group.len()).ok();
            for x in group {
                write!(output, "{} ", x + 1).ok();
            }
            writeln!(output).ok();
        }
    }
}
