use std::{cmp::Reverse, collections::HashMap, io::Write};

use fenwick_tree::*;

mod fast_io {
    use std::fs::File;
    use std::io::BufWriter;
    use std::os::unix::io::FromRawFd;

    unsafe extern "C" {
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

pub mod fenwick_tree {
    pub trait Group {
        type X: Clone;
        fn id(&self) -> Self::X;
        fn add_assign(&self, lhs: &mut Self::X, rhs: Self::X);
        fn sub_assign(&self, lhs: &mut Self::X, rhs: Self::X);
    }

    #[derive(Clone)]
    pub struct FenwickTree<G: Group> {
        n: usize,
        group: G,
        sum: Vec<G::X>,
    }

    impl<G: Group> FenwickTree<G> {
        pub fn new(n: usize, group: G) -> Self {
            let n = n.next_power_of_two(); // Required for binary search
            let sum = (0..n).map(|_| group.id()).collect();
            Self { n, group, sum }
        }

        pub fn from_iter(iter: impl IntoIterator<Item = G::X>, group: G) -> Self {
            let mut sum: Vec<_> = iter.into_iter().collect();
            let n = sum.len();

            let n = n.next_power_of_two(); // Required for binary search
            sum.resize_with(n, || group.id());

            for i in 1..n {
                let prev = sum[i - 1].clone();
                group.add_assign(&mut sum[i], prev);
            }
            for i in (1..n).rev() {
                let j = i & (i + 1);
                if j >= 1 {
                    let prev = sum[j - 1].clone();
                    group.sub_assign(&mut sum[i], prev);
                }
            }

            Self { n, group, sum }
        }

        pub fn add(&mut self, mut idx: usize, value: G::X) {
            debug_assert!(idx < self.n);
            while idx < self.n {
                self.group.add_assign(&mut self.sum[idx], value.clone());
                idx |= idx + 1;
            }
        }

        // Exclusive prefix sum (0..idx)
        pub fn sum_prefix(&self, idx: usize) -> G::X {
            debug_assert!(idx <= self.n);
            let mut res = self.group.id();
            let mut r = idx;
            while r > 0 {
                self.group.add_assign(&mut res, self.sum[r - 1].clone());
                r &= r - 1;
            }
            res
        }

        pub fn sum_range(&self, range: std::ops::Range<usize>) -> G::X {
            debug_assert!(range.start <= range.end && range.end <= self.n);
            let mut res = self.sum_prefix(range.end);
            self.group
                .sub_assign(&mut res, self.sum_prefix(range.start));
            res
        }

        pub fn get(&self, idx: usize) -> G::X {
            self.sum_range(idx..idx + 1)
        }

        // find the first i, such that equiv pred(sum_range(0..=i)) == false
        pub fn partition_point_prefix(&self, mut pred: impl FnMut(&G::X) -> bool) -> usize {
            let p1_log2 = usize::BITS - self.n.leading_zeros();
            let mut idx = 0;
            let mut sum = self.group.id();
            for i in (0..p1_log2).rev() {
                let idx_next = idx | (1 << i);
                if idx_next > self.n {
                    continue;
                }
                let mut sum_next = sum.clone();
                self.group
                    .add_assign(&mut sum_next, self.sum[idx_next - 1].clone());
                if pred(&sum_next) {
                    sum = sum_next;
                    idx = idx_next;
                }
            }
            idx
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
            if n == 0 {
                return;
            }

            for i in 0..n - 1 {
                let j = self.range_u64(i as u64..n as u64) as usize;
                xs.swap(i, j);
            }
        }
    }
}

mod mst_incremental {
    // Fast Incremental minimum spanning tree
    // based on a direct implementation of Anti-Monopoly tree
    //
    // ## Reference
    // - Xiangyun Ding, Yan Gu, Yihan Sun.
    // "New Algorithms for Incremental Minimum Spanning Trees and Temporal Graph Applications".
    // [https://arxiv.org/abs/2504.04619]

    const UNSET: u32 = u32::MAX;
    const LINK_BY_STITCH: bool = true;

    #[derive(Clone, Debug)]
    struct Node<T> {
        parent: u32,
        size: i32,
        weight: T,
    }

    // Lazy Anti-Monopoly tree
    #[derive(Clone, Debug)]
    pub struct AMTree<T> {
        nodes: Vec<Node<T>>,
    }

    #[derive(Clone, Debug)]
    pub enum InsertType<T> {
        Connect,
        Replace(T),
    }

    impl<T: Default> InsertType<T> {
        pub fn ok(self) -> Option<T> {
            match self {
                InsertType::Connect => None,
                InsertType::Replace(w) => Some(w),
            }
        }
    }

    impl<T: Ord + Copy + Default> AMTree<T> {
        pub fn new(n_verts: usize) -> Self {
            Self {
                nodes: vec![
                    Node {
                        parent: UNSET,
                        size: 1,
                        weight: T::default() // Dummy
                    };
                    n_verts
                ],
            }
        }

        fn promote(&mut self, u: u32) {
            let p = self.nodes[u as usize].parent;
            let wu = self.nodes[u as usize].weight;
            let g = self.nodes[p as usize].parent;
            let wp = self.nodes[p as usize].weight;

            if wu >= wp && g != UNSET {
                // Shortcut
                self.nodes[u as usize].parent = g;
                self.nodes[p as usize].size -= self.nodes[u as usize].size;
            } else {
                // Rotate
                self.nodes[u as usize].parent = g;
                self.nodes[p as usize].parent = u;
                self.nodes[u as usize].weight = wp;
                self.nodes[p as usize].weight = wu;
                self.nodes[p as usize].size -= self.nodes[u as usize].size;
                self.nodes[u as usize].size += self.nodes[p as usize].size;
            }
        }

        fn perch(&mut self, u: u32) {
            while self.nodes[u as usize].parent != UNSET {
                self.promote(u);
            }
        }

        fn link_by_perch(&mut self, u: u32, v: u32, w: T) -> Option<InsertType<T>> {
            debug_assert!(u != v);
            self.perch(u);
            self.perch(v);
            if self.nodes[u as usize].parent == v {
                let w_old = self.nodes[u as usize].weight;
                if w < w_old {
                    self.nodes[u as usize].weight = w;
                    Some(InsertType::Replace(w_old))
                } else {
                    None
                }
            } else {
                debug_assert!(self.nodes[u as usize].parent == UNSET);
                self.nodes[u as usize].parent = v;
                self.nodes[u as usize].weight = w;
                self.nodes[v as usize].size += self.nodes[u as usize].size;
                Some(InsertType::Connect)
            }
        }

        fn cut_max_path(&mut self, mut u: u32, mut v: u32, w: T) -> Option<InsertType<T>> {
            debug_assert!(u != v);
            let mut w_old = None;
            loop {
                if self.nodes[u as usize].size > self.nodes[v as usize].size {
                    std::mem::swap(&mut u, &mut v);
                }

                let p = self.nodes[u as usize].parent;
                if p == UNSET {
                    // Disconnected
                    return Some(InsertType::Connect);
                }

                w_old = w_old.max(Some((self.nodes[u as usize].weight, u)));
                u = p;
                if u == v {
                    // reached LCA
                    let (w_old, mut t) = w_old.unwrap();
                    if w >= w_old {
                        return None;
                    }

                    // Unlink
                    let p = self.nodes[t as usize].parent;
                    self.nodes[t as usize].parent = UNSET;
                    self.nodes[t as usize].weight = T::default();
                    let delta_size = self.nodes[t as usize].size;

                    t = p;
                    while t != UNSET {
                        self.nodes[t as usize].size -= delta_size;
                        t = self.nodes[t as usize].parent;
                    }

                    return Some(InsertType::Replace(w_old));
                }
            }
        }

        fn link_by_stitch(&mut self, u: u32, v: u32, mut w: T) -> Option<InsertType<T>> {
            debug_assert!(u != v);

            let res = self.cut_max_path(u, v, w);
            if res.is_none() {
                return None;
            }

            let mut u = u;
            let mut v = v;
            let mut delta_size_u = 0i32;
            let mut delta_size_v = 0i32;
            loop {
                while self.nodes[u as usize].parent != UNSET && w >= self.nodes[u as usize].weight {
                    u = self.nodes[u as usize].parent;
                    self.nodes[u as usize].size += delta_size_u;
                }

                while self.nodes[v as usize].parent != UNSET && w >= self.nodes[v as usize].weight {
                    v = self.nodes[v as usize].parent;
                    self.nodes[v as usize].size += delta_size_v;
                }

                if self.nodes[u as usize].size > self.nodes[v as usize].size {
                    std::mem::swap(&mut u, &mut v);
                    std::mem::swap(&mut delta_size_u, &mut delta_size_v);
                }

                let su = self.nodes[u as usize].size;
                delta_size_u -= su;
                delta_size_v += su;
                self.nodes[v as usize].size += su;

                std::mem::swap(&mut self.nodes[u as usize].weight, &mut w);
                u = std::mem::replace(&mut self.nodes[u as usize].parent, v);
                if u == UNSET {
                    loop {
                        v = self.nodes[v as usize].parent;
                        if v == UNSET {
                            return res;
                        }
                        self.nodes[v as usize].size += delta_size_v;
                    }
                }
                self.nodes[u as usize].size += delta_size_u;
            }
        }

        fn upward_calibrate(&mut self, mut u: u32) {
            loop {
                let p = self.nodes[u as usize].parent;
                if p == UNSET {
                    break;
                }

                if self.nodes[u as usize].size * 3 / 2 > self.nodes[p as usize].size {
                    self.promote(u);
                } else {
                    u = p;
                }
            }
        }

        pub fn insert(&mut self, u: u32, v: u32, w: T) -> Option<InsertType<T>> {
            if u == v {
                return None;
            }

            self.upward_calibrate(u);
            self.upward_calibrate(v);

            if LINK_BY_STITCH {
                self.link_by_stitch(u, v, w)
            } else {
                self.link_by_perch(u, v, w)
            }
        }

        pub fn max_path(&mut self, mut u: u32, mut v: u32) -> Option<T> {
            if u == v {
                return None;
            }

            self.upward_calibrate(u);
            self.upward_calibrate(v);

            let mut res = None;
            loop {
                if self.nodes[u as usize].size > self.nodes[v as usize].size {
                    std::mem::swap(&mut u, &mut v);
                }
                res = res.max(Some(self.nodes[u as usize].weight));
                u = self.nodes[u as usize].parent;
                if u == UNSET {
                    return None; // Disconnected
                }
                if u == v {
                    return res;
                }
            }
        }
    }
}

struct Additive;

impl Group for Additive {
    type X = i32;

    fn id(&self) -> Self::X {
        0
    }

    fn add_assign(&self, lhs: &mut Self::X, rhs: Self::X) {
        *lhs += rhs;
    }

    fn sub_assign(&self, lhs: &mut Self::X, rhs: Self::X) {
        *lhs -= rhs;
    }
}

pub fn main() {
    let mut input = fast_io::stdin();
    let mut output = fast_io::stdout();

    let m: usize = input.value();
    let q: usize = input.value();
    let mut edges = vec![[!0; 2]; m];
    for g in 0..2 {
        for e in 0..m {
            edges[e][g] = input.value::<u32>() - 1;
        }
    }

    let mut trans = HashMap::new();
    for e in 0..m {
        for g in 0..2 {
            let l = trans.len() as u32;
            edges[e][g] = *trans.entry(edges[e][g]).or_insert_with(|| l);
        }
    }
    let n = trans.len();

    let mut rng = rand::SplitMix64::from_entropy().unwrap();
    let h_vert = (0..n)
        .map(|_| (rng.next_u64() as u128) << 64 | rng.next_u64() as u128)
        .collect::<Vec<_>>();
    let mut h_edge_prefix = std::iter::once(0)
        .chain(
            edges
                .iter()
                .map(|&[u, v]| h_vert[u as usize] ^ h_vert[v as usize]),
        )
        .collect::<Vec<_>>();
    drop(h_vert);

    for i in 0..m {
        h_edge_prefix[i + 1] ^= h_edge_prefix[i];
    }

    let mut queries = vec![vec![]; m];
    let mut ans = vec![0; q];
    for i in 0..q as i32 {
        let l = input.value::<u32>() - 1;
        let r = input.value::<u32>() - 1;
        if h_edge_prefix[l as usize] != h_edge_prefix[(r + 1) as usize] {
            ans[i as usize] = -1;
            continue;
        }

        queries[r as usize].push((l, i));
    }
    drop(h_edge_prefix);

    const UNSET: u32 = !0;
    let mut t_start = vec![UNSET; n];

    let mut mst = mst_incremental::AMTree::new(n);
    let mut counter = FenwickTree::new(m, Additive);
    for e in 0..m as u32 {
        let [u, v] = edges[e as usize];
        for z in [u, v] {
            let e_old = std::mem::replace(&mut t_start[z as usize], e);
            if e_old != UNSET {
                counter.add(e_old as usize, -1);
            }
            counter.add(e as usize, 1);
        }

        if let Some(mst_incremental::InsertType::Replace(Reverse(e_old))) =
            mst.insert(u, v, Reverse(e))
        {
            counter.add(e_old as usize, 1);
        }
        counter.add(e as usize, -1);

        for (l, i) in std::mem::take(&mut queries[e as usize]) {
            ans[i as usize] = counter.sum_range(l as usize..m)
        }
    }

    for a in ans {
        writeln!(output, "{}", a).unwrap();
    }
}
