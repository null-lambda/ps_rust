use std::io::Write;

use algebra::SemiRing;

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

pub mod debug {
    pub fn with(#[allow(unused_variables)] f: impl FnOnce()) {
        #[cfg(debug_assertions)]
        f()
    }
}

pub mod algebra {
    use std::ops::*;
    pub trait SemiRing:
        Add<Output = Self>
        + Sub<Output = Self>
        + Mul<Output = Self>
        + AddAssign
        + SubAssign
        + MulAssign
        + for<'a> Add<&'a Self, Output = Self>
        + for<'a> Sub<&'a Self, Output = Self>
        + for<'a> Mul<&'a Self, Output = Self>
        + for<'a> AddAssign<&'a Self>
        + for<'a> SubAssign<&'a Self>
        + for<'a> MulAssign<&'a Self>
        + Default
        + Clone
    {
        fn zero() -> Self {
            Self::default()
        }
        fn one() -> Self;
    }

    pub trait CommRing: SemiRing + Neg<Output = Self> {}

    pub trait PowBy<E> {
        fn pow(&self, exp: E) -> Self;
    }

    pub trait Field:
        CommRing
        + Div<Output = Self>
        + DivAssign
        + for<'a> Div<&'a Self, Output = Self>
        + for<'a> DivAssign<&'a Self>
    {
        fn inv(&self) -> Self;
    }

    macro_rules! impl_semiring {
        ($($t:ty)+) => {
            $(
                impl SemiRing for $t {
                    fn one() -> Self {
                        1
                    }
                }
            )+
        };
    }

    macro_rules! impl_commring {
        ($($t:ty)+) => {
            $(
                impl CommRing for $t {}
            )+
        };
    }

    impl_semiring!(u8 u16 u32 u64 u128 usize);
    impl_semiring!(i8 i16 i32 i64 i128 isize);
    impl_commring!(i8 i16 i32 i64 i128 isize);

    macro_rules! impl_powby {
        ($(($uexp:ty, $iexp:ty),)+) => {
            $(
                impl<R: CommRing> PowBy<$uexp> for R {
                    fn pow(&self, exp: $uexp) -> R {
                        let mut res = R::one();
                        let mut base = self.clone();
                        let mut exp = exp;
                        while exp > 0 {
                            if exp & 1 == 1 {
                                res *= base.clone();
                            }
                            base *= base.clone();
                            exp >>= 1;
                        }
                        res
                    }
                }

                impl<R: Field> PowBy<$iexp> for R {
                    fn pow(&self, exp: $iexp) -> R {
                        if exp < 0 {
                            self.inv().pow((-exp) as $uexp)
                        } else {
                            self.pow(exp as $uexp)
                        }
                    }
                }
            )+
        };
    }
    impl_powby!(
        (u8, i8),
        (u16, i16),
        (u32, i32),
        (u64, i64),
        (u128, i128),
        (usize, isize),
    );
}

pub mod num_mod {
    use super::algebra::*;
    use std::ops::*;

    pub trait Unsigned:
        Copy
        + Default
        + SemiRing
        + Div<Output = Self>
        + Rem<Output = Self>
        + RemAssign
        + PartialEq
        + PartialOrd
        + From<u8>
    {
        fn zero() -> Self {
            Self::default()
        }
        fn one() -> Self;
    }

    macro_rules! impl_unsigned {
        ($($t:ty)+) => {
            $(
                impl Unsigned for $t {
                    fn one() -> Self {
                        1
                    }
                }
            )+
        };
    }
    impl_unsigned!(u8 u16 u32 u64 u128 usize);

    pub trait ModSpec: Copy {
        type U: Unsigned;
        const MODULUS: Self::U;
    }

    pub trait ByPrime: ModSpec {}

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub struct ModInt<M: ModSpec>(M::U);

    impl<M: ModSpec> ModInt<M> {
        pub fn new(s: M::U) -> Self {
            Self(s % M::MODULUS)
        }
    }

    macro_rules! impl_modspec {
        ($($t:ident $u:ty),+) => {
            $(
                #[derive(Debug, Clone, Copy, PartialEq, Eq)]
                pub struct $t<const M: $u>;

                impl<const MOD: $u> ModSpec for $t<MOD> {
                    type U = $u;
                    const MODULUS: $u = MOD;
                }

            )+
        };
    }
    impl_modspec!(
        ByU32 u32, ByU32Prime u32,
        ByU64 u64, ByU64Prime u64,
        ByU128 u128, ByU128Prime u128
    );

    macro_rules! impl_by_prime {
        ($($t:ident $u:ty),+) => {
            $(
                impl<const MOD: $u> ByPrime for $t<MOD> {}
            )+
        };
    }
    impl_by_prime!(ByU32Prime u32, ByU64Prime u64, ByU128Prime u128);

    impl<M: ModSpec> AddAssign<&'_ Self> for ModInt<M> {
        fn add_assign(&mut self, rhs: &Self) {
            self.0 += rhs.0;
            if self.0 >= M::MODULUS {
                self.0 -= M::MODULUS;
            }
        }
    }

    impl<M: ModSpec> SubAssign<&'_ Self> for ModInt<M> {
        fn sub_assign(&mut self, rhs: &Self) {
            if self.0 < rhs.0 {
                self.0 += M::MODULUS;
            }
            self.0 -= rhs.0;
        }
    }

    impl<M: ModSpec> MulAssign<&'_ Self> for ModInt<M> {
        fn mul_assign(&mut self, rhs: &Self) {
            self.0 *= rhs.0;
            self.0 %= M::MODULUS;
        }
    }

    macro_rules! forward_ref_binop {
        ($($OpAssign:ident $op_assign:ident),+) => {
            $(
                impl<M: ModSpec> $OpAssign for ModInt<M> {
                    fn $op_assign(&mut self, rhs: Self) {
                        self.$op_assign(&rhs);
                    }
                }
            )+
        };
    }
    forward_ref_binop!(AddAssign add_assign, MulAssign mul_assign, SubAssign sub_assign);

    macro_rules! impl_op_by_op_assign {
        ($($Op:ident $op:ident $op_assign:ident),+) => {
            $(
                impl<M: ModSpec> $Op<&'_ Self> for ModInt<M> {
                    type Output = Self;
                    fn $op(mut self, rhs: &Self) -> Self {
                        self.$op_assign(rhs);
                        self
                    }
                }

                impl< M: ModSpec> $Op for ModInt<M> {
                    type Output = ModInt<M>;
                    fn $op(self, rhs: Self) -> Self::Output {
                        self.clone().$op(&rhs)
                    }
                }
            )+
        };
    }
    impl_op_by_op_assign!(Add add add_assign, Mul mul mul_assign, Sub sub sub_assign);

    impl<M: ModSpec> Neg for &'_ ModInt<M> {
        type Output = ModInt<M>;
        fn neg(self) -> ModInt<M> {
            let mut res = M::MODULUS - self.0;
            if res == M::MODULUS {
                res = 0.into();
            }
            ModInt(res)
        }
    }

    impl<M: ModSpec> Neg for ModInt<M> {
        type Output = Self;
        fn neg(self) -> Self::Output {
            (&self).neg()
        }
    }

    impl<M: ModSpec> Default for ModInt<M> {
        fn default() -> Self {
            Self(M::U::default())
        }
    }

    impl<M: ModSpec> SemiRing for ModInt<M> {
        fn one() -> Self {
            Self(1.into())
        }
    }
    impl<M: ModSpec> CommRing for ModInt<M> {}

    impl<M: ByPrime> DivAssign<&'_ Self> for ModInt<M>
    where
        ModInt<M>: PowBy<M::U>,
    {
        fn div_assign(&mut self, rhs: &Self) {
            self.mul_assign(&rhs.inv());
        }
    }

    impl<M: ByPrime> DivAssign for ModInt<M>
    where
        ModInt<M>: PowBy<M::U>,
    {
        fn div_assign(&mut self, rhs: Self) {
            self.div_assign(&rhs);
        }
    }

    impl<M: ByPrime> Div<&'_ Self> for ModInt<M>
    where
        ModInt<M>: PowBy<M::U>,
    {
        type Output = Self;
        fn div(mut self, rhs: &Self) -> Self {
            self.div_assign(rhs);
            self
        }
    }

    impl<M: ByPrime> Div for ModInt<M>
    where
        ModInt<M>: PowBy<M::U>,
    {
        type Output = Self;
        fn div(self, rhs: Self) -> Self {
            self / &rhs
        }
    }

    impl<M: ByPrime> Field for ModInt<M>
    where
        ModInt<M>: PowBy<M::U>,
    {
        fn inv(&self) -> Self {
            self.pow(M::MODULUS - M::U::from(2))
        }
    }

    pub trait CmpUType<Rhs: Unsigned>: Unsigned {
        type MaxT: Unsigned;
        fn upcast(lhs: Self) -> Self::MaxT;
        fn upcast_rhs(rhs: Rhs) -> Self::MaxT;
        fn downcast(max: Self::MaxT) -> Self;
    }

    macro_rules! impl_cmp_utype {
        (
            $( $($lower:ident)* < $target:ident < $($upper:ident)* ),* $(,)?
        ) => {
            $(
                $(
                    impl CmpUType<$lower> for $target {
                        type MaxT = $target;
                        fn upcast(lhs: Self) -> Self::MaxT {
                            lhs as Self::MaxT
                        }
                        fn upcast_rhs(rhs: $lower) -> Self::MaxT {
                            rhs as Self::MaxT
                        }
                        fn downcast(max: Self::MaxT) -> Self {
                            max as Self
                        }
                    }
                )*
                impl CmpUType<$target> for $target {
                    type MaxT = $target;
                    fn upcast(lhs: Self) -> Self::MaxT {
                        lhs as Self::MaxT
                    }
                    fn upcast_rhs(rhs: $target) -> Self::MaxT {
                        rhs as Self::MaxT
                    }
                    fn downcast(max: Self::MaxT) -> Self {
                        max as Self
                    }
                }
                $(
                    impl CmpUType<$upper> for $target {
                        type MaxT = $upper;
                        fn upcast(lhs: Self) -> Self::MaxT {
                            lhs as Self::MaxT
                        }
                        fn upcast_rhs(rhs: $upper) -> Self::MaxT {
                            rhs as Self::MaxT
                        }
                        fn downcast(max: Self::MaxT) -> Self {
                            max as Self
                        }
                    }
                )*
            )*
        };
    }
    impl_cmp_utype!(
        < u8 < u16 u32 u64 u128,
        u8 < u16 < u32 u64 u128,
        u8 u16 < u32 < u64 u128,
        u8 u16 u32 < u64 < u128,
        u8 u16 u32 u64 < u128 <,
    );

    impl<U, S, M> From<S> for ModInt<M>
    where
        U: CmpUType<S>,
        S: Unsigned,
        M: ModSpec<U = U>,
    {
        fn from(s: S) -> Self {
            Self(U::downcast(U::upcast_rhs(s) % U::upcast(M::MODULUS)))
        }
    }

    macro_rules! impl_cast_to_unsigned {
        ($($u:ty)+) => {
            $(
                impl<M: ModSpec<U = $u>> From<ModInt<M>> for $u {
                    fn from(n: ModInt<M>) -> Self {
                        n.0
                    }
                }
            )+
        };
    }
    impl_cast_to_unsigned!(u8 u16 u32 u64 u128);

    impl<U: std::fmt::Debug, M: ModSpec<U = U>> std::fmt::Debug for ModInt<M> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.0.fmt(f)
        }
    }

    impl<U: std::fmt::Display, M: ModSpec<U = U>> std::fmt::Display for ModInt<M> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.0.fmt(f)
        }
    }

    impl<U: std::str::FromStr, M: ModSpec<U = U>> std::str::FromStr for ModInt<M> {
        type Err = <U as std::str::FromStr>::Err;
        fn from_str(s: &str) -> Result<Self, Self::Err> {
            s.parse().map(|x| ModInt::new(x))
        }
    }
}

pub mod jagged {
    use std::fmt::Debug;
    use std::iter;
    use std::mem::MaybeUninit;
    use std::ops::{Index, IndexMut};

    // Trait for painless switch between different representations of a jagged array
    pub trait Jagged<T>: IndexMut<usize, Output = [T]> {
        fn len(&self) -> usize;
    }

    impl<T, C> Jagged<T> for C
    where
        C: AsRef<[Vec<T>]> + IndexMut<usize, Output = [T]>,
    {
        fn len(&self) -> usize {
            <Self as AsRef<[Vec<T>]>>::as_ref(self).len()
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
            let v: Vec<Vec<&T>> = (0..self.len()).map(|i| self[i].iter().collect()).collect();
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
        pub fn from_pairs(n: usize, pairs: &[(u32, T)]) -> Self {
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

    impl<T> Index<usize> for CSR<T> {
        type Output = [T];

        fn index(&self, index: usize) -> &Self::Output {
            &self.data[self.head[index] as usize..self.head[index + 1] as usize]
        }
    }

    impl<T> IndexMut<usize> for CSR<T> {
        fn index_mut(&mut self, index: usize) -> &mut Self::Output {
            &mut self.data[self.head[index] as usize..self.head[index + 1] as usize]
        }
    }

    impl<T> Jagged<T> for CSR<T> {
        fn len(&self) -> usize {
            self.head.len() - 1
        }
    }
}

pub mod bcc {
    /// Biconnected components & 2-edge-connected components
    /// Verified with [Yosupo library checker](https://judge.yosupo.jp/problem/biconnected_components)
    use super::jagged;

    pub const UNSET: u32 = !0;

    pub struct BlockCutForest<'a, E, J> {
        // DFS tree structure
        pub neighbors: &'a J,
        pub parent: Vec<u32>,
        pub euler_in: Vec<u32>,
        pub low: Vec<u32>, // Lowest euler index on a subtree's back edge

        /// Block-cut tree structure,  
        /// represented as a rooted bipartite tree between  
        /// vertex nodes (indices in 0..n) and virtual BCC nodes (indices in n..).  
        /// A vertex node is a cut vertex iff its degree is >= 2,
        /// and the neighbors of a virtual BCC node represents all its belonging vertices.
        pub bct_parent: Vec<u32>,
        pub bct_children: Vec<Vec<u32>>,

        pub bcc_edges: Vec<(u32, u32, E)>,
    }

    impl<'a, E: 'a + Copy, J: jagged::Jagged<(u32, E)>> BlockCutForest<'a, E, J> {
        pub fn from_assoc_list(neighbors: &'a J) -> Self {
            let n = neighbors.len();

            let mut parent = vec![UNSET; n];
            let mut low = vec![0; n];
            let mut euler_in = vec![0; n];
            let mut timer = 1u32;

            let mut bct_parent = vec![UNSET; n];
            let mut bct_children = vec![vec![]; n];

            let mut current_edge = vec![0u32; n];
            let mut stack = vec![];
            for root in 0..n {
                if euler_in[root] != 0 {
                    continue;
                }

                parent[root] = UNSET;
                let mut u = root as u32;
                loop {
                    let p = parent[u as usize];
                    let iv = &mut current_edge[u as usize];
                    if *iv == 0 {
                        // On enter
                        euler_in[u as usize] = timer;
                        low[u as usize] = timer + 1;
                        timer += 1;
                        stack.push(u);
                    }
                    if (*iv as usize) == neighbors[u as usize].len() {
                        // On exit
                        if p == UNSET {
                            break;
                        }

                        low[p as usize] = low[p as usize].min(low[u as usize]);
                        if low[u as usize] >= euler_in[p as usize] {
                            // Found a BCC
                            let bcc_node = bct_parent.len() as u32;

                            bct_parent.push(p);
                            bct_children.push(vec![]);
                            bct_children[p as usize].push(bcc_node);

                            while let Some(c) = stack.pop() {
                                bct_parent[c as usize] = bcc_node;
                                bct_children[bcc_node as usize].push(c);

                                if c == u {
                                    break;
                                }
                            }
                        }

                        u = p;
                        continue;
                    }

                    let (v, _) = neighbors[u as usize][*iv as usize];
                    *iv += 1;
                    if v == p {
                        continue;
                    }

                    if euler_in[v as usize] != 0 {
                        // Back edge
                        low[u as usize] = low[u as usize].min(euler_in[v as usize]);
                        continue;
                    }

                    // Forward edge (a part of DFS spanning tree)
                    parent[v as usize] = u;
                    u = v;
                }
            }

            Self {
                neighbors,
                parent,
                low,
                euler_in,

                bct_parent,
                bct_children,

                bcc_edges: vec![],
            }
        }
    }
}

type ModP = num_mod::ModInt<num_mod::ByU64Prime<993_244_853>>;

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    if n == 1 {
        writeln!(output, "0").unwrap();
        return;
    }

    let mut edges = vec![];
    for _ in 0..m {
        let u = input.value::<u32>() - 1;
        let v = input.value::<u32>() - 1;
        edges.push((u, (v, ())));
        edges.push((v, (u, ())));
    }
    let neighbors = jagged::CSR::from_pairs(n, &edges);

    let bct = bcc::BlockCutForest::from_assoc_list(&neighbors);
    let parent = &bct.bct_parent;
    let children = &bct.bct_children;
    let n_nodes = parent.len();

    let mut degree = vec![1u32; n_nodes];
    for &p in &parent[1..] {
        degree[p as usize] += 1;
    }

    let mut dp = vec![[ModP::one(), ModP::zero()]; n_nodes];
    for mut u in 0..n_nodes {
        while degree[u] == 1 {
            let p = parent[u] as usize;

            let m = children[u].len();
            if u < n {
                let mut prefix0 = vec![ModP::one()];
                for (iv, &v) in children[u].iter().enumerate() {
                    prefix0.push(prefix0[iv] * dp[v as usize][0]);
                }
                dp[u][0] = prefix0[m];

                let mut suffix0 = ModP::one();
                for (iv, &v) in children[u].iter().enumerate().rev() {
                    let mid1 = dp[v as usize][1];
                    dp[u][1] += prefix0[iv] * mid1 * suffix0;
                    suffix0 *= dp[children[u][iv] as usize][0];
                }
            } else {
                debug_assert!(m != 0);
                if m == 1 {
                    let v = children[u][0] as usize;
                    dp[u][1] = -dp[v][0];
                    dp[u][0] = dp[v][1];
                } else {
                    let mut dp_cycle = [ModP::zero(); 4];

                    let v = children[u][0];
                    dp_cycle[0b00] = dp[v as usize][0];
                    dp_cycle[0b01] = dp[v as usize][1];
                    dp_cycle[0b11] = -dp[v as usize][0];
                    for &v in &children[u][1..] {
                        let prev = dp_cycle;
                        for init in [0b00, 0b10] {
                            dp_cycle[init | 0b0] = prev[init | 0b1] * dp[v as usize][0];
                            dp_cycle[init | 0b1] = prev[init | 0b1] * dp[v as usize][1]
                                - prev[init | 0b0] * dp[v as usize][0];
                        }
                    }

                    dp[u][0] = dp_cycle[0b01];
                    dp[u][1] = dp_cycle[0b11] - dp_cycle[0b00];

                    let mut prod0 = children[u]
                        .iter()
                        .map(|&v| dp[v as usize][0])
                        .reduce(|acc, x| acc * ModP::from(x))
                        .unwrap();
                    if m % 2 == 1 {
                        prod0 = -prod0;
                    }
                    dp[u][1] += ModP::from(2u8) * ModP::from(prod0);
                }
            }

            debug::with(|| println!("u={u}, {:?}", dp[u]));

            if p == bcc::UNSET as usize {
                break;
            }
            degree[u] -= 1;
            degree[p] -= 1;
            u = p;
        }
    }

    let ans = dp[0][1];
    writeln!(output, "{}", ans).unwrap();
}
