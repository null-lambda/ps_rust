use std::io::Write;

use fenwick_tree::FenwickTree;
use jagged::Jagged;
use num_mod::{ModOp, MontgomeryU32};

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

pub mod jagged {
    use std::fmt::Debug;
    use std::iter;
    use std::mem::MaybeUninit;

    // Trait for painless switch between different representations of a jagged array
    pub trait Jagged<'a, T: 'a> {
        type ItemRef: ExactSizeIterator<Item = &'a T>;
        fn len(&self) -> usize;
        fn get(&'a self, u: usize) -> Self::ItemRef;
    }

    impl<'a, T, C> Jagged<'a, T> for C
    where
        C: AsRef<[Vec<T>]> + 'a,
        T: 'a,
    {
        type ItemRef = std::slice::Iter<'a, T>;
        fn len(&self) -> usize {
            <Self as AsRef<[Vec<T>]>>::as_ref(self).len()
        }
        fn get(&'a self, u: usize) -> Self::ItemRef {
            let res = <Self as AsRef<[Vec<T>]>>::as_ref(self)[u].iter();
            res
        }
    }

    // Compressed sparse row format for jagged array
    // Provides good locality for graph traversal, but works only for static ones.
    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
    pub struct CSR<T> {
        data: Vec<T>,
        head: Vec<u32>,
    }

    impl<T> Debug for CSR<T>
    where
        T: Debug,
    {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let v: Vec<Vec<&T>> = (0..self.len()).map(|i| self.get(i).collect()).collect();
            v.fmt(f)
        }
    }

    impl<T, I> FromIterator<I> for CSR<T>
    where
        I: IntoIterator<Item = T>,
    {
        fn from_iter<J>(iter: J) -> Self
        where
            J: IntoIterator<Item = I>,
        {
            let mut data = vec![];
            let mut head = vec![];
            head.push(0);

            let mut cnt = 0;
            for row in iter {
                data.extend(row.into_iter().inspect(|_| cnt += 1));
                head.push(cnt);
            }
            CSR { data, head }
        }
    }

    impl<T: Clone> CSR<T> {
        pub fn from_assoc_list(n: usize, pairs: &[(u32, T)]) -> Self {
            let mut head = vec![0u32; n + 1];

            for &(u, _) in pairs {
                debug_assert!(u < n as u32);
                head[u as usize + 1] += 1;
            }
            for i in 2..n + 1 {
                head[i] += head[i - 1];
            }
            let mut data: Vec<_> = iter::repeat_with(|| MaybeUninit::uninit())
                .take(head[n] as usize)
                .collect();
            let mut pos = head.clone();

            for (u, v) in pairs {
                data[pos[*u as usize] as usize] = MaybeUninit::new(v.clone());
                pos[*u as usize] += 1;
            }

            let data = std::mem::ManuallyDrop::new(data);
            let data = unsafe {
                Vec::from_raw_parts(data.as_ptr() as *mut T, data.len(), data.capacity())
            };

            CSR { data, head }
        }
    }

    impl<'a, T: 'a> Jagged<'a, T> for CSR<T> {
        type ItemRef = std::slice::Iter<'a, T>;

        fn len(&self) -> usize {
            self.head.len() - 1
        }

        fn get(&'a self, u: usize) -> Self::ItemRef {
            self.data[self.head[u] as usize..self.head[u + 1] as usize].iter()
        }
    }
}

pub mod centroid {
    /// Centroid Decomposition
    use crate::jagged::Jagged;

    pub fn init_size<'a, E: 'a>(
        neighbors: &'a impl Jagged<'a, (u32, E)>,
        size: &mut [u32],
        u: usize,
        p: usize,
    ) {
        size[u] = 1;
        for &(v, _) in neighbors.get(u) {
            if v as usize == p {
                continue;
            }
            init_size(neighbors, size, v as usize, u);
            size[u] += size[v as usize];
        }
    }

    fn reroot_to_centroid<'a, _E: 'a>(
        neighbors: &'a impl Jagged<'a, (u32, _E)>,
        size: &mut [u32],
        visited: &[bool],
        mut u: usize,
    ) -> usize {
        let threshold = (size[u] + 1) / 2;
        let mut p = u;
        'outer: loop {
            for &(v, _) in neighbors.get(u) {
                if v as usize == p || visited[v as usize] {
                    continue;
                }
                if size[v as usize] >= threshold {
                    size[u] -= size[v as usize];
                    size[v as usize] += size[u];

                    p = u;
                    u = v as usize;
                    continue 'outer;
                }
            }
            return u;
        }
    }

    pub fn build_centroid_tree<'a, _E: 'a + Clone>(
        neighbors: &'a impl Jagged<'a, (u32, _E)>,
        size: &mut [u32],
        visited: &mut [bool],
        parent_centroid: &mut [u32],
        yield_rooted_tree: &mut impl FnMut(&[u32], &[bool], usize, u32),
        depth: u32,
        init: usize,
    ) -> usize {
        let root = reroot_to_centroid(neighbors, size, visited, init);
        visited[root] = true;
        yield_rooted_tree(size, visited, root, depth);

        for &(v, _) in neighbors.get(root) {
            if visited[v as usize] {
                continue;
            }
            let sub_root = build_centroid_tree(
                neighbors,
                size,
                visited,
                parent_centroid,
                yield_rooted_tree,
                depth + 1,
                v as usize,
            );
            parent_centroid[sub_root] = root as u32;
        }
        root
    }
}

pub mod num_mod_static {
    // TODO: implmeent a static modint (just remove &self args from ModOp)
    use std::ops::*;

    pub trait ModInt:
        Clone
        + Copy
        + Default
        + Add<Output = Self>
        + AddAssign<Self>
        + Sub<Output = Self>
        + SubAssign<Self>
        + Mul<Output = Self>
        + MulAssign<Self>
        + PartialEq
        + Eq
    {
        type BaseT;
        fn zero() -> Self;
        fn one() -> Self;
        fn modulus() -> Self::BaseT;
    }

    pub trait PowBy<E> {
        fn pow(self, exp: E) -> Self;
    }

    macro_rules! impl_powby {
        ($($exp:ty)+) => {
            $(
                impl<M: ModInt> PowBy<$exp> for M {
                    fn pow(self, exp: $exp) -> M {
                        let mut base = self;
                        let mut res = M::one();
                        let mut exp = exp;
                        while exp > 0 {
                            if exp % 2 == 1 {
                                res *= base;
                            }
                            base *= base;
                            exp /= 2;
                        }
                        res
                    }
                }
            )+
        };
    }

    impl_powby!(u32 u64 u128);

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    struct MontgomeryU32<const M: u32>(u32);

    impl<const M: u32> MontgomeryU32<M> {
        const M_INV: u32 = todo!();
        const R2: u32 = todo!();
    }
}

pub mod num_mod {
    use std::ops::*;
    pub trait ModOp<T> {
        fn zero(&self) -> T;
        fn one(&self) -> T;
        fn modulus(&self) -> T;
        fn add(&self, lhs: T, rhs: T) -> T;
        fn sub(&self, lhs: T, rhs: T) -> T;
        fn mul(&self, lhs: T, rhs: T) -> T;
        fn transform(&self, n: T) -> T;
        fn reduce(&self, n: T) -> T;
    }

    pub trait PowBy<T, E> {
        fn pow(&self, base: T, exp: E) -> T;
    }

    pub trait InvOp<T> {
        fn inv(&self, n: T) -> T;
    }

    impl<T: Clone, M: ModOp<T>> PowBy<T, u32> for M {
        fn pow(&self, mut base: T, mut exp: u32) -> T {
            let mut res = self.one();
            while exp > 0 {
                if exp % 2 == 1 {
                    res = self.mul(res, base.clone());
                }
                base = self.mul(base.clone(), base);
                exp >>= 1;
            }
            res
        }
    }

    impl<T: Clone, M: ModOp<T>> PowBy<T, u64> for M {
        fn pow(&self, mut base: T, mut exp: u64) -> T {
            let mut res = self.one();
            while exp > 0 {
                if exp % 2 == 1 {
                    res = self.mul(res, base.clone());
                }
                base = self.mul(base.clone(), base);
                exp >>= 1;
            }
            res
        }
    }

    impl<T: Clone, M: ModOp<T>> PowBy<T, u128> for M {
        fn pow(&self, mut base: T, mut exp: u128) -> T {
            let mut res = self.one();
            while exp > 0 {
                if exp % 2 == 1 {
                    res = self.mul(res, base.clone());
                }
                base = self.mul(base.clone(), base);
                exp >>= 1;
            }
            res
        }
    }

    impl<M: ModOp<u32>> InvOp<u32> for M {
        fn inv(&self, n: u32) -> u32 {
            self.pow(n, self.modulus() - 2)
        }
    }

    impl<M: ModOp<u64>> InvOp<u64> for M {
        fn inv(&self, n: u64) -> u64 {
            self.pow(n, self.modulus() - 2)
        }
    }

    impl<M: ModOp<u128>> InvOp<u128> for M {
        fn inv(&self, n: u128) -> u128 {
            self.pow(n, self.modulus() - 2)
        }
    }

    pub struct NaiveModOp<T> {
        m: T,
    }

    impl<T> NaiveModOp<T> {
        pub fn new(m: T) -> Self {
            Self { m }
        }
    }

    pub trait One {
        fn one() -> Self;
    }

    impl One for u32 {
        fn one() -> Self {
            1
        }
    }

    impl One for u64 {
        fn one() -> Self {
            1
        }
    }

    impl One for u128 {
        fn one() -> Self {
            1
        }
    }

    impl<T> ModOp<T> for NaiveModOp<T>
    where
        T: Copy
            + Default
            + One
            + Add<Output = T>
            + Sub<Output = T>
            + Mul<Output = T>
            + Rem<Output = T>
            + PartialOrd,
    {
        fn zero(&self) -> T {
            T::default()
        }
        fn one(&self) -> T {
            T::one()
        }
        fn modulus(&self) -> T {
            self.m
        }
        fn add(&self, lhs: T, rhs: T) -> T {
            if lhs + rhs >= self.m {
                lhs + rhs - self.m
            } else {
                lhs + rhs
            }
        }
        fn sub(&self, lhs: T, rhs: T) -> T {
            if lhs >= rhs {
                lhs - rhs
            } else {
                lhs + self.m - rhs
            }
        }
        fn mul(&self, lhs: T, rhs: T) -> T {
            (lhs * rhs) % self.m
        }
        fn transform(&self, n: T) -> T {
            n % self.m
        }
        fn reduce(&self, n: T) -> T {
            n % self.m
        }
    }

    macro_rules! impl_montgomery {
        ($type:ident, S = $single:ty, D = $double:ty, EXP = $exp:expr, LOG2_EXP = $log2_exp:expr, ZERO = $zero:expr, ONE = $one:expr) => {
            #[derive(Debug, Clone)]
            pub struct $type {
                m: $single,
                m_inv: $single,
                r2: $single,
            }

            impl $type {
                pub const fn new(m: $single) -> Self {
                    debug_assert!(m % 2 == 1, "modulus must be coprime with 2");
                    let mut m_inv = $one;
                    let two = $one + $one;

                    let mut iter = 0;
                    while iter < $log2_exp {
                        m_inv = m_inv.wrapping_mul(two.wrapping_sub(m.wrapping_mul(m_inv)));
                        iter += 1;
                    }

                    let r = m.wrapping_neg() % m;
                    let r2 = ((r as $double * r as $double % m as $double) as $single);

                    Self { m, m_inv, r2 }
                }

                fn reduce_double(&self, x: $double) -> $single {
                    debug_assert!((x as $double) < (self.m as $double) * (self.m as $double));
                    let q = (x as $single).wrapping_mul(self.m_inv);
                    let a = ((q as $double * self.m as $double) >> $exp) as $single;
                    let mut res = (x >> $exp) as $single + self.m - a;
                    if res >= self.m {
                        res -= self.m;
                    }
                    res
                }
            }

            impl ModOp<$single> for $type {
                fn zero(&self) -> $single {
                    $zero
                }

                fn one(&self) -> $single {
                    self.transform($one)
                }

                fn modulus(&self) -> $single {
                    self.m
                }

                fn mul(&self, x: $single, y: $single) -> $single {
                    debug_assert!(x < self.m);
                    debug_assert!(y < self.m);
                    self.reduce_double(x as $double * y as $double)
                }

                fn add(&self, x: $single, y: $single) -> $single {
                    debug_assert!(x < self.m);
                    debug_assert!(y < self.m);
                    let sum = x + y;
                    if sum >= self.m {
                        sum - self.m
                    } else {
                        sum
                    }
                }

                fn sub(&self, x: $single, y: $single) -> $single {
                    debug_assert!(x < self.m);
                    debug_assert!(y < self.m);
                    if x >= y {
                        x - y
                    } else {
                        x + self.m - y
                    }
                }

                fn reduce(&self, x: $single) -> $single {
                    self.reduce_double(x as $double)
                }

                fn transform(&self, x: $single) -> $single {
                    debug_assert!(x < self.m);
                    self.mul(x, self.r2)
                }
            }
        };
    }

    impl_montgomery! {
        MontgomeryU32,
        S = u32,
        D = u64,
        EXP = 32,
        LOG2_EXP = 5,
        ZERO = 0u32,
        ONE = 1u32
    }

    impl_montgomery! {
        MontgomeryU64,
        S = u64,
        D = u128,
        EXP = 64,
        LOG2_EXP = 6,
        ZERO = 0u64,
        ONE = 1u64
    }
}

// TODO: implement a static modint
const MOD_OP: num_mod::MontgomeryU32 = num_mod::MontgomeryU32::new(1_000_000_007);

pub mod fenwick_tree {
    pub trait Group {
        type X: Clone;
        fn id(&self) -> Self::X;
        fn add_assign(&self, lhs: &mut Self::X, rhs: Self::X);
        fn sub_assign(&self, lhs: &mut Self::X, rhs: Self::X);
    }

    #[derive(Clone, Debug)]
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

        pub fn len(&self) -> usize {
            self.n
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

impl<M: ModOp<u32>> fenwick_tree::Group for M {
    type X = u32;
    fn id(&self) -> Self::X {
        self.zero()
    }
    fn add_assign(&self, lhs: &mut Self::X, rhs: Self::X) {
        *lhs = self.add(*lhs, rhs);
    }
    fn sub_assign(&self, lhs: &mut Self::X, rhs: Self::X) {
        *lhs = self.sub(*lhs, rhs);
    }
}

#[derive(Debug, Clone)]
struct SubtreeAgg {
    sum_v_subtree: FenwickTree<MontgomeryU32>,
    sum_e_subtree: FenwickTree<MontgomeryU32>,
    sum_v_to_root: FenwickTree<MontgomeryU32>,
    sum_e_to_root: FenwickTree<MontgomeryU32>,
    complement_size: u32,
}

impl SubtreeAgg {
    fn empty(s: usize) -> Self {
        Self {
            sum_v_subtree: FenwickTree::new(s as usize, MOD_OP.clone()),
            sum_e_subtree: FenwickTree::new(s as usize, MOD_OP.clone()),
            sum_v_to_root: FenwickTree::new(s as usize + 1, MOD_OP.clone()),
            sum_e_to_root: FenwickTree::new(s as usize + 1, MOD_OP.clone()),
            complement_size: 0,
        }
    }

    fn sum_v(&self) -> u32 {
        self.sum_v_subtree.sum_prefix(self.sum_v_subtree.len())
    }

    fn sum_e(&self) -> u32 {
        self.sum_e_subtree.sum_prefix(self.sum_e_subtree.len())
    }
}

#[derive(Debug, Clone)]
struct CentroidAgg {
    subtree: Vec<SubtreeAgg>,
    sum_v: u32,
    sum_e: u32,
    sum_e_cs: u32,
    sum_dot_cs: u32,
}

impl CentroidAgg {
    fn len(&self) -> usize {
        self.subtree.len()
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let q: usize = input.value();
    let vs_base: Vec<u32> = (0..n).map(|_| input.value()).collect();
    let mut edges = vec![];
    let mut es_base = vec![];
    let mut edges_unweighted = vec![];
    for _ in 0..n - 1 {
        let u = input.value::<u32>() - 1;
        let v = input.value::<u32>() - 1;
        let w: u32 = input.value();
        edges.push((u, v));
        es_base.push(w);
        edges_unweighted.push((u, (v, ())));
        edges_unweighted.push((v, (u, ())));
    }
    let neighbors = jagged::CSR::from_assoc_list(n, &edges_unweighted);

    const UNSET: u32 = u32::MAX;
    let mut parent_centroid = vec![UNSET; n];
    let mut sizes = vec![];
    let mut centroid_depth = vec![0; n];
    let mut subcentroid_children = vec![vec![]; n];
    let mut belongs_to = vec![];
    let mut euler_ins = vec![];
    {
        let base_root = 0;
        let mut visited = vec![false; n];
        let mut size = vec![0; n];
        centroid::init_size(&neighbors, &mut size, base_root, base_root);
        centroid::build_centroid_tree(
            &neighbors,
            &mut size,
            &mut visited,
            &mut parent_centroid,
            &mut |size, visited, root, depth| {
                centroid_depth[root] = depth;
                if sizes.len() <= depth as usize {
                    sizes.resize_with(depth as usize + 1, || vec![0; n]);
                    belongs_to.resize_with(depth as usize + 1, || vec![(UNSET, UNSET); n]);
                    euler_ins.resize_with(depth as usize + 1, || vec![UNSET; n]);
                }

                sizes[depth as usize][root as usize] = size[root as usize];
                belongs_to[depth as usize][root as usize] = (root as u32, UNSET);

                let mut idx_in_subcentroid = 0;
                for &(child, ()) in neighbors.get(root as usize) {
                    if visited[child as usize] {
                        continue;
                    }

                    let mut stack = vec![(child as u32, root as u32)];
                    euler_ins[depth as usize][root as usize] = size[child as usize] - 1;
                    while let Some((u, p)) = stack.pop() {
                        sizes[depth as usize][u as usize] = size[u as usize];

                        let last_idx = euler_ins[depth as usize][p as usize];
                        euler_ins[depth as usize][p as usize] -= size[u as usize];
                        euler_ins[depth as usize][u as usize] = last_idx;

                        belongs_to[depth as usize][u as usize] =
                            (root as u32, idx_in_subcentroid as u32);
                        for &(v, _) in neighbors.get(u as usize) {
                            if visited[v as usize] || v == p {
                                continue;
                            }
                            stack.push((v, u));
                        }
                    }

                    subcentroid_children[root].push(child);

                    idx_in_subcentroid += 1;
                }
                euler_ins[depth as usize][root as usize] = UNSET;
            },
            0,
            base_root,
        );
    }
    let euler_in = |d: usize, u: usize| euler_ins[d][u] as usize;
    let euler_out = |d: usize, u: usize| euler_in(d, u) + sizes[d][u] as usize;
    let depth_bound = sizes.len();

    let mut acc = 0;
    let mut vs = vec![MOD_OP.zero(); n];
    let mut es = vec![vec![0; n]; depth_bound];
    let mut centroid_agg: Vec<_> = (0..n)
        .map(|c| {
            let d = centroid_depth[c] as usize;

            CentroidAgg {
                subtree: subcentroid_children[c]
                    .iter()
                    .map(|&v| SubtreeAgg::empty(sizes[d][v as usize] as usize))
                    .collect(),
                sum_v: 0,
                sum_e: 0,
                sum_e_cs: 0,
                sum_dot_cs: 0,
            }
        })
        .collect();
    for c in 0..n {
        let d = centroid_depth[c] as usize;
        for (iv, &v) in subcentroid_children[c].iter().enumerate() {
            centroid_agg[c].subtree[iv].complement_size =
                MOD_OP.transform(sizes[d][c] - sizes[d][v as usize]);
        }
    }

    let queries = ((0..n).map(|u| (1, u, vs_base[u])))
        .chain((0..n - 1).map(|ie| (2, ie, es_base[ie])))
        .chain((0..q).map(|_| {
            (
                input.value::<u8>(),
                input.value::<usize>() - 1,
                input.value::<u32>(),
            )
        }));

    {
        // println!("parent_centroid: {:?}", parent_centroid);
        // println!("euler_ins {:?}", euler_ins);
        // println!("sizes: {:?}", sizes);
    }

    for (i_query, (cmd, i0, x)) in queries.enumerate() {
        let mut dv = MOD_OP.zero();
        let x = MOD_OP.transform(x);
        let mut c;
        if cmd == 1 {
            dv = MOD_OP.sub(x, vs[i0]);
            vs[i0] = x;

            acc = MOD_OP.add(acc, MOD_OP.mul(centroid_agg[i0].sum_e_cs, dv));
            c = parent_centroid[i0] as usize;
        } else {
            let (q, r) = edges[i0];
            c = (0..depth_bound)
                .rev()
                .find_map(|d| {
                    let (cq, _) = belongs_to[d][q as usize];
                    let (cr, _) = belongs_to[d][r as usize];
                    (cq == cr).then(|| cq)
                })
                .unwrap() as usize;
        }

        while c != UNSET as usize {
            let d = centroid_depth[c] as usize;
            let bottom = if cmd == 1 {
                assert_eq!(c, belongs_to[d][i0].0 as usize);
                i0
            } else {
                let (q, r) = edges[i0];
                assert_eq!(c, belongs_to[d][q as usize].0 as usize);
                assert_eq!(c, belongs_to[d][r as usize].0 as usize);
                if sizes[d][q as usize] < sizes[d][r as usize] {
                    q as usize
                } else {
                    r as usize
                }
            };
            let ib = belongs_to[d][bottom].1 as usize;

            // print!("agg={:?} ", centroid_agg[c]);
            // println!("u={}, c={}, d={}, bottom={}, ib={}", i0, c, d, bottom, ib);

            let agg = &mut centroid_agg[c];
            {
                acc = MOD_OP.sub(acc, agg.sum_dot_cs);
                acc = MOD_OP.sub(acc, MOD_OP.mul(agg.sum_e_cs, vs[c]));
                acc = MOD_OP.sub(acc, MOD_OP.mul(agg.sum_v, agg.sum_e));
                acc = MOD_OP.add(
                    acc,
                    MOD_OP.mul(agg.subtree[ib].sum_e(), agg.subtree[ib].sum_v()),
                );
            }

            let cs = agg.subtree[ib].complement_size;
            let s_bottom = MOD_OP.transform(sizes[d][bottom]);
            if cmd == 1 {
                let dv_dup = MOD_OP.mul(dv, s_bottom);
                // let e = es[d][i0]; // TODO: make this dynamic
                let mut e = MOD_OP.mul(
                    agg.subtree[ib]
                        .sum_e_to_root
                        .sum_prefix(euler_in(d, bottom) + 1),
                    s_bottom,
                );
                e = MOD_OP.add(
                    e,
                    agg.subtree[ib]
                        .sum_e_subtree
                        .sum_range(euler_in(d, bottom) + 1..euler_out(d, bottom)),
                );
                agg.subtree[ib]
                    .sum_v_subtree
                    .add(euler_in(d, bottom), dv_dup);
                agg.subtree[ib].sum_v_to_root.add(euler_in(d, bottom), dv);
                agg.subtree[ib]
                    .sum_v_to_root
                    .add(euler_out(d, bottom), MOD_OP.sub(MOD_OP.zero(), dv));
                agg.sum_v = MOD_OP.add(agg.sum_v, dv_dup);
                agg.sum_dot_cs = MOD_OP.add(agg.sum_dot_cs, MOD_OP.mul(dv_dup, MOD_OP.mul(e, cs)));
            } else {
                let de = MOD_OP.sub(x, es[d][bottom]);
                es[d][bottom] = x;
                let de_dup = MOD_OP.mul(de, s_bottom);
                let mut v = MOD_OP.mul(
                    agg.subtree[ib]
                        .sum_v_to_root
                        .sum_prefix(euler_in(d, bottom) + 1),
                    s_bottom,
                );
                v = MOD_OP.add(
                    v,
                    agg.subtree[ib]
                        .sum_v_subtree
                        .sum_range(euler_in(d, bottom) + 1..euler_out(d, bottom)),
                );
                agg.subtree[ib]
                    .sum_e_subtree
                    .add(euler_in(d, bottom), de_dup);
                agg.subtree[ib].sum_e_to_root.add(euler_in(d, bottom), de);
                agg.subtree[ib]
                    .sum_e_to_root
                    .add(euler_out(d, bottom), MOD_OP.sub(MOD_OP.zero(), de));
                agg.sum_e = MOD_OP.add(agg.sum_e, de_dup);
                agg.sum_e_cs = MOD_OP.add(agg.sum_e_cs, MOD_OP.mul(de_dup, cs));
                agg.sum_dot_cs = MOD_OP.add(agg.sum_dot_cs, MOD_OP.mul(de_dup, MOD_OP.mul(v, cs)));

                {
                    // println!(
                    //     " de={}, de_dup={}, v_to_root={}",
                    //     MOD_OP.reduce(de),
                    //     MOD_OP.reduce(de_dup),
                    //     MOD_OP.reduce(
                    //         agg.subtree[ib]
                    //             .sum_e_to_root
                    //             .sum_prefix(euler_in(d, bottom) + 1)
                    //     )
                    // );
                }
            }

            {
                acc = MOD_OP.add(acc, agg.sum_dot_cs);
                acc = MOD_OP.add(acc, MOD_OP.mul(agg.sum_e_cs, vs[c]));
                acc = MOD_OP.add(acc, MOD_OP.mul(agg.sum_v, agg.sum_e));
                acc = MOD_OP.sub(
                    acc,
                    MOD_OP.mul(agg.subtree[ib].sum_e(), agg.subtree[ib].sum_v()),
                );
            }

            c = parent_centroid[c] as usize;
        }

        {
            // print!("vs: ");
            // for u in 0..n {
            //     print!("{} ", MOD_OP.reduce(vs[u]));
            // }
            // print!("    ");

            // print!("es: ");
            // for d in 0..sizes.len() {
            //     for u in 0..n {
            //         print!("{} ", MOD_OP.reduce(es[d][u]));
            //     }
            //     print!("// ");
            // }

            // print!("agg.sum_v: ");
            // for u in 0..n {
            //     print!("{:?} ", MOD_OP.reduce(centroid_agg[u].sum_v));
            // }

            // print!("agg.sum_dot_cs: ");
            // for u in 0..n {
            //     print!("{:?} ", MOD_OP.reduce(centroid_agg[u].sum_dot_cs));
            // }

            // print!("agg.sum_e_cs: ");
            // for u in 0..n {
            //     print!("{:?} ", MOD_OP.reduce(centroid_agg[u].sum_e_cs));
            // }
            // println!();
        }

        if i_query >= n * 2 - 2 {
            writeln!(output, "{}", MOD_OP.reduce(acc)).unwrap();
        }

        // writeln!(output, "{}", MOD_OP.reduce(acc)).unwrap();
    }
}
