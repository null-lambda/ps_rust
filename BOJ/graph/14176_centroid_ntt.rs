use std::io::Write;

use jagged::Jagged;
use num_mod::{InvOp, ModOp, PowBy};

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

pub mod num_mod {
    use std::ops::*;
    pub trait ModOp<T: Clone> {
        fn zero(&self) -> T;
        fn one(&self) -> T;
        fn modulus(&self) -> T;
        fn add(&self, lhs: T, rhs: T) -> T;
        fn add_assign(&self, lhs: &mut T, rhs: T) {
            *lhs = self.add(lhs.clone(), rhs);
        }
        fn sub(&self, lhs: T, rhs: T) -> T;
        fn sub_assign(&self, lhs: &mut T, rhs: T) {
            *lhs = self.sub(lhs.clone(), rhs);
        }
        fn mul(&self, lhs: T, rhs: T) -> T;
        fn mul_assign(&self, lhs: &mut T, rhs: T) {
            *lhs = self.mul(lhs.clone(), rhs);
        }
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

pub mod ntt {
    use std::iter;

    use crate::num_mod::{ModOp, PowBy};

    fn bit_reversal_perm<T>(xs: &mut [T]) {
        let n = xs.len();
        let n_log2 = u32::BITS - (n as u32).leading_zeros() - 1;

        for i in 0..n as u32 {
            let rev = i.reverse_bits() >> (u32::BITS - n_log2);
            if i < rev {
                xs.swap(i as usize, rev as usize);
            }
        }
    }

    pub fn radix4<T, M>(op: &M, proot: T, xs: &mut [T])
    where
        T: Copy,
        M: ModOp<T> + PowBy<T, u32>,
    {
        let n = xs.len();
        assert!(n.is_power_of_two());
        let n_log2 = u32::BITS - (n as u32).leading_zeros() - 1;
        bit_reversal_perm(xs);

        let base: Vec<_> = (0..n_log2)
            .scan(proot, |acc, _| {
                let prev = *acc;
                *acc = op.mul(*acc, *acc);
                Some(prev)
            })
            .collect();

        let mut proot_pow = vec![op.zero(); n]; // Cache-friendly twiddle factors
        proot_pow[0] = op.one();

        let quartic_root = op.pow(proot, n as u32 / 4);

        let update_proot_pow = |proot_pow: &mut [T], k: u32| {
            let step = 1 << k;
            let base = base[(n_log2 - k - 1) as usize];
            for i in (0..step).rev() {
                proot_pow[i * 2 + 1] = op.mul(proot_pow[i], base);
                proot_pow[i * 2] = proot_pow[i];
            }
        };

        let mut k = 0;
        if n_log2 % 2 == 1 {
            let step = 1 << k;
            // radix-2 butterfly
            update_proot_pow(&mut proot_pow, k);
            for t in xs.chunks_exact_mut(step * 2) {
                let (t0, t1) = t.split_at_mut(step);
                for (a0, a1) in t0.into_iter().zip(t1) {
                    let b0 = *a0;
                    let b1 = *a1;
                    *a0 = op.add(b0, b1);
                    *a1 = op.sub(b0, b1);
                }
            }
            k += 1;
        }
        while k < n_log2 {
            let step = 1 << k;
            // radix-4 butterfly
            update_proot_pow(&mut proot_pow, k);
            update_proot_pow(&mut proot_pow, k + 1);

            for t in xs.chunks_exact_mut(step * 4) {
                let (t0, rest) = t.split_at_mut(step);
                let (t1, rest) = rest.split_at_mut(step);
                let (t2, t3) = rest.split_at_mut(step);

                for ((((a0, a1), a2), a3), &pow1) in
                    t0.into_iter().zip(t1).zip(t2).zip(t3).zip(&proot_pow)
                {
                    let pow2 = op.mul(pow1, pow1);
                    let pow1_shift = op.mul(pow1, quartic_root);

                    let b0 = *a0;
                    let b1 = op.mul(*a1, pow2);
                    let b2 = *a2;
                    let b3 = op.mul(*a3, pow2);

                    let c0 = op.add(b0, b1);
                    let c1 = op.sub(b0, b1);
                    let c2 = op.mul(op.add(b2, b3), pow1);
                    let c3 = op.mul(op.sub(b2, b3), pow1_shift);

                    *a0 = op.add(c0, c2);
                    *a1 = op.add(c1, c3);
                    *a2 = op.sub(c0, c2);
                    *a3 = op.sub(c1, c3);
                }
            }
            k += 2;
        }
    }

    // naive O(n^2)
    pub fn naive<T, M>(op: &M, proot: T, xs: &mut [T])
    where
        T: Copy,
        M: ModOp<T> + PowBy<T, u32>,
    {
        let n = xs.len().next_power_of_two();
        let proot_pow: Vec<T> = iter::successors(Some(op.one()), |&acc| Some(op.mul(acc, proot)))
            .take(n)
            .collect();
        let res: Vec<_> = (0..n)
            .map(|i| {
                (0..n)
                    .map(|j| op.mul(xs[j], proot_pow[(i * j) % n]))
                    .fold(op.zero(), |acc, x| op.add(acc, x))
            })
            .collect();
        xs.copy_from_slice(&res);
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

    pub fn dnc<'a, E: 'a + Clone>(
        neighbors: &'a impl Jagged<'a, (u32, E)>,
        size: &mut [u32],
        visited: &mut [bool],
        yield_rooted_tree: &mut impl FnMut(&[u32], &[bool], usize),
        init: usize,
    ) {
        let root = reroot_to_centroid(neighbors, size, visited, init);
        visited[root] = true;
        yield_rooted_tree(size, visited, root);
        for &(v, _) in neighbors.get(root) {
            if visited[v as usize] {
                continue;
            }
            dnc(neighbors, size, visited, yield_rooted_tree, v as usize)
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

const P: u64 = 9223372036737335297;
const MOD_OP: num_mod::MontgomeryU64 = num_mod::MontgomeryU64::new(P);

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let gen: u64 = MOD_OP.transform(3);

    let n: usize = input.value();
    let edges = (0..n - 1)
        .map(|_| (input.value::<u32>() - 1, input.value::<u32>() - 1))
        .flat_map(|(u, v)| [(u, (v, ())), (v, (u, ()))]);
    let neighbors = jagged::CSR::from_assoc_list(n, &edges.collect::<Vec<_>>());

    let base_root = 0;
    let mut size = vec![0; n];
    let mut visited = vec![false; n];
    let mut acc = vec![MOD_OP.zero(); n];
    let mut depth = vec![0; n];
    centroid::init_size(&neighbors, &mut size, base_root, base_root);
    centroid::dnc(
        &neighbors,
        &mut size,
        &mut visited,
        &mut |size, visited, root| {
            // Convolve smaller ones first
            let mut children: Vec<_> = neighbors
                .get(root)
                .filter(|&&(child, _)| !visited[child as usize])
                .map(|&(child, ())| (child, size[child as usize]))
                .collect();
            children.sort_unstable_by_key(|&(_, size)| size);

            let dist_freq_agg = children
                .into_iter()
                .map(|(child, _)| {
                    let mut bfs_order = vec![(child, root as u32)];
                    let mut timer = 0;
                    depth[child as usize] = 1;
                    let depth_bound = size[child as usize] + 1;
                    let mut dist_freq = vec![0; depth_bound as usize * 2];
                    dist_freq[1] += 1;

                    while let Some(&(u, p)) = bfs_order.get(timer) {
                        for &(v, ()) in neighbors.get(u as usize) {
                            if v == p || visited[v as usize] {
                                continue;
                            }
                            depth[v as usize] = depth[u as usize] + 1;
                            dist_freq[depth[v as usize] as usize] += 1;
                            bfs_order.push((v, u));
                        }
                        timer += 1;
                    }

                    for f in &mut dist_freq {
                        *f = MOD_OP.transform(*f);
                    }
                    dist_freq
                })
                .reduce(|mut xs, mut ys| {
                    // Merge dist freqs
                    let mut sum = vec![MOD_OP.zero(); xs.len().max(ys.len())];
                    for (a, x) in sum.iter_mut().zip(&xs) {
                        *a = MOD_OP.add(*a, *x);
                    }
                    for (a, y) in sum.iter_mut().zip(&ys) {
                        *a = MOD_OP.add(*a, *y);
                    }

                    // Convolution
                    let r = xs.len();
                    let s = ys.len();
                    if r + s <= 256 {
                        debug_assert!(r % 2 == 0);
                        debug_assert!(s % 2 == 0);
                        let mut conv = vec![MOD_OP.zero(); r + s - 2];
                        for i in 0..r / 2 {
                            for j in 0..s / 2 {
                                MOD_OP.add_assign(&mut conv[i + j], MOD_OP.mul(xs[i], ys[j]));
                            }
                        }
                        xs = conv;
                    } else {
                        let t = (r + s - 1).next_power_of_two();
                        xs.resize(t, MOD_OP.zero());
                        ys.resize(t, MOD_OP.zero());

                        assert_eq!((P - 1) % t as u64, 0);
                        let proot = MOD_OP.pow(gen, (P - 1) / t as u64);
                        ntt::radix4(&MOD_OP, proot, &mut xs);
                        ntt::radix4(&MOD_OP, proot, &mut ys);
                        for (x, y) in xs.iter_mut().zip(&ys) {
                            MOD_OP.mul_assign(x, *y);
                        }
                        ntt::radix4(&MOD_OP, MOD_OP.inv(proot), &mut xs);
                        let t_inv = MOD_OP.inv(MOD_OP.transform(t as u64));
                        for x in &mut xs {
                            MOD_OP.mul_assign(x, t_inv);
                        }
                    }

                    for (a, x) in acc.iter_mut().zip(&xs) {
                        MOD_OP.add_assign(a, *x);
                    }

                    sum
                });

            if let Some(xs) = dist_freq_agg {
                for (a, x) in acc.iter_mut().zip(&xs) {
                    MOD_OP.add_assign(a, *x);
                }
            }
        },
        base_root,
    );

    let mut numer = MOD_OP.zero();
    let (min_prime_factor, _) = linear_sieve(acc.len() as u32 - 1);
    for (a, p) in acc.iter().zip(&min_prime_factor).skip(2) {
        if *p == 0 {
            MOD_OP.add_assign(&mut numer, *a);
        }
    }
    numer = MOD_OP.reduce(numer);

    let denom = n as u64 * (n as u64 - 1) / 2;
    let ans = numer as f64 / denom as f64;
    writeln!(output, "{}", ans).unwrap();
}
