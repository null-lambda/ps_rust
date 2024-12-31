use std::{cmp::Reverse, collections::VecDeque, io::Write, marker::PhantomData};

use num_mod::{InvOp, ModOp, MontgomeryU32};

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

const UNSET: u32 = u32::MAX;

// Build a max cartesian tree from inorder traversal
fn max_cartesian_tree<T>(
    n: usize,
    iter: impl IntoIterator<Item = (usize, T)>,
) -> (Vec<u32>, Vec<[u32; 2]>)
where
    T: Ord,
{
    let mut parent = vec![UNSET; n];
    let mut children = vec![[UNSET; 2]; n];

    // Monotone stack
    let mut stack = vec![];

    for (u, h) in iter {
        let u = u as u32;

        let mut c = None;
        while let Some((prev, _)) = stack.last() {
            if prev > &h {
                break;
            }
            c = stack.pop();
        }
        if let Some(&(_, p)) = stack.last() {
            parent[u as usize] = p;
            children[p as usize][1] = u;
        }
        if let Some((_, c)) = c {
            parent[c as usize] = u;
            children[u as usize][0] = c;
        }
        stack.push((h, u));
    }

    (parent, children)
}

// p_u(x) = P[b_u <= x, T_u],    x in [0, 1]
// f_u(x) = E[S_u | b_u <= x, T_u],    x in [0, 1]
//    where S_u = sum of b_u over subtree
//    T_u = { subtree(u) forms a cartesian tree w.r.t b }
// => p_p(x) = int_[0, x] dy p_c1(y) ... p_ck(y)
//    p_p(x) f_p(x) = int_[0, x] dy ( y + f_c1(y) + ... + f_ck(y) ) p_c1(y) ... p_ck(y)
// => p_i(x) = c_i x ^ s_i,
//    p_i(x) f_i(x) = d_i x ^ (s_i + 1)
//    where s_i = size of a subtree

// Final goal: Compute P(b_root) = f_root(1)
#[derive(Clone)]
struct NodeData<M> {
    size: u32,
    prob: u32,
    expect: u32,
    f: u32,
    _marker: PhantomData<M>,
}

impl<M: ModOp<u32>> NodeData<M> {
    fn leaf(mod_op: &M) -> Self {
        Self {
            size: mod_op.one(),
            prob: mod_op.one(),
            expect: mod_op.one(),
            f: mod_op.one(),
            _marker: PhantomData,
        }
    }

    fn pull_from(&mut self, other: Self, mod_op: &M) {
        self.size = mod_op.add(self.size, other.size);
        self.prob = mod_op.mul(self.prob, other.prob);
        self.expect = mod_op.add(self.expect, other.f);
    }

    fn finalize(&mut self, mod_op: &M) {
        self.expect = mod_op.mul(self.expect, mod_op.inv(mod_op.add(self.size, mod_op.one())));
        self.f = mod_op.mul(self.expect, self.size);
        self.expect = mod_op.mul(self.expect, self.prob);
        self.prob = mod_op.mul(self.prob, mod_op.inv(self.size));
    }

    fn get_exp(&self, mod_op: &M) -> u32 {
        mod_op.reduce(self.expect)
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let mod_op = MontgomeryU32::new(1_000_000_007);

    for _ in 0..input.value() {
        let n: usize = input.value();
        let xs: Vec<u32> = (0..n).map(|_| input.value()).collect();
        let (parent, _) = max_cartesian_tree(n, (0..n).map(|i| (i, (xs[i], Reverse(i)))));

        let mut dp: Vec<_> = (0..n).map(|_| NodeData::leaf(&mod_op)).collect();

        let mut indegree = vec![0; n];
        for u in 0..n {
            if parent[u] != UNSET {
                indegree[parent[u] as usize] += 1;
            }
        }

        let mut queue: VecDeque<usize> = (0..n).filter(|&u| indegree[u] == 0).collect();
        while let Some(u) = queue.pop_front() {
            dp[u].finalize(&mod_op);

            let p = parent[u] as usize;
            if p as u32 == UNSET {
                continue;
            }

            let dp_u = dp[u].clone();
            dp[p].pull_from(dp_u, &mod_op);
            indegree[p] -= 1;
            if indegree[p] == 0 {
                queue.push_back(p);
            }
        }

        let root = (0..n).find(|&u| parent[u] == UNSET).unwrap();
        let ans = dp[root].get_exp(&mod_op);
        writeln!(output, "{}", ans).unwrap();
    }
}
