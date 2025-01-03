use std::io::Write;

use jagged::Jagged;
use num_mod::ModOp;

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
                pub fn new(m: $single) -> Self {
                    debug_assert!(m % 2 == 1, "modulus must be coprime with 2");
                    let mut m_inv = $one;
                    let two = $one + $one;
                    for _ in 0..$log2_exp {
                        m_inv = m_inv.wrapping_mul(two.wrapping_sub(m.wrapping_mul(m_inv)));
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

    pub fn build_centroid_tree<'a, _E: 'a + Clone>(
        neighbors: &'a impl Jagged<'a, (u32, _E)>,
        size: &mut [u32],
        visited: &mut [bool],
        parent_centroid: &mut [u32],
        init: usize,
    ) -> usize {
        let root = reroot_to_centroid(neighbors, size, visited, init);
        visited[root] = true;

        for &(v, _) in neighbors.get(root) {
            if visited[v as usize] {
                continue;
            }
            let sub_root =
                build_centroid_tree(neighbors, size, visited, parent_centroid, v as usize);
            parent_centroid[sub_root] = root as u32;
        }
        root
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let mod_op = num_mod::MontgomeryU32::new(998_244_353);

    let n: usize = input.value();
    let mut edges = vec![];
    for _ in 0..n - 1 {
        let u = input.value::<u32>() - 1;
        let v = input.value::<u32>() - 1;
        edges.push((u, (v, ())));
        edges.push((v, (u, ())));
    }
    let neighbors = jagged::CSR::from_assoc_list(n, &edges);

    let mut weight = vec![[false; 2]; n];
    for color in 0..2 {
        for _ in 0..input.value() {
            let u = input.value::<usize>() - 1;
            weight[u][color] = true;
        }
    }

    let mut ans = mod_op.zero();
    let mut ans_doubled = mod_op.zero();
    let mut size = vec![0; n];
    let mut visited = vec![false; n];
    let mut depth = vec![0; n];
    centroid::init_size(&neighbors, &mut size, 0, n);
    centroid::dnc(
        &neighbors,
        &mut size,
        &mut visited,
        &mut |_size, visited, root| {
            depth[root] = 0;
            let mut sum = [mod_op.zero(); 2];
            let mut sum_sq = [mod_op.zero(); 2];
            let mut count: [_; 2] =
                std::array::from_fn(|i| mod_op.transform(weight[root][i] as u32));
            for &(child, ()) in neighbors.get(root) {
                if visited[child as usize] {
                    continue;
                }

                let mut timer = 0;
                let mut bfs_order = vec![(child, root as u32)];

                let mut sub_sum = [mod_op.zero(); 2];
                let mut sub_sum_sq = [mod_op.zero(); 2];
                let mut sub_count = [mod_op.zero(); 2];

                while let Some(&(u, p)) = bfs_order.get(timer) {
                    depth[u as usize] = depth[p as usize] + 1;
                    let d = mod_op.transform(depth[u as usize]);
                    for color in 0..2 {
                        if weight[u as usize][color] {
                            sub_sum[color] = mod_op.add(sub_sum[color], d);
                            sub_sum_sq[color] = mod_op.add(sub_sum_sq[color], mod_op.mul(d, d));
                            sub_count[color] = mod_op.add(sub_count[color], mod_op.one());
                        }
                    }

                    timer += 1;
                    for &(v, ()) in neighbors.get(u as usize) {
                        if v == p || visited[v as usize] {
                            continue;
                        }
                        bfs_order.push((v, u));
                    }
                }

                ans_doubled = mod_op.sub(ans_doubled, mod_op.mul(sub_sum[0], sub_sum[1]));
                for color in 0..2 {
                    ans = mod_op.sub(ans, mod_op.mul(sub_sum_sq[color], sub_count[1 - color]));

                    sum[color] = mod_op.add(sum[color], sub_sum[color]);
                    sum_sq[color] = mod_op.add(sum_sq[color], sub_sum_sq[color]);
                    count[color] = mod_op.add(count[color], sub_count[color]);
                }
            }

            ans_doubled = mod_op.add(ans_doubled, mod_op.mul(sum[0], sum[1]));
            for color in 0..2 {
                ans = mod_op.add(ans, mod_op.mul(sum_sq[color], count[1 - color]));
            }
        },
        0,
    );

    ans = mod_op.add(ans, mod_op.add(ans_doubled, ans_doubled));
    ans = mod_op.reduce(ans);

    writeln!(output, "{}", ans).unwrap();
}
