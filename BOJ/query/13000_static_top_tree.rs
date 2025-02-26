use std::io::Write;

use algebra::SemiRing;
use num_mod::{ByU64, ModInt};
use static_top_tree::rooted::{ClusterCx, StaticTopTree, WeightType};

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

    // Non-commutative algebras are not my business (yet)
    // pub trait Ring: SemiRing + Neg<Output = Self> {}

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

    impl<'a, M: ModSpec> AddAssign<&'a Self> for ModInt<M> {
        fn add_assign(&mut self, rhs: &Self) {
            self.0 += rhs.0;
            if self.0 >= M::MODULUS {
                self.0 -= M::MODULUS;
            }
        }
    }

    impl<'a, M: ModSpec> SubAssign<&'a Self> for ModInt<M> {
        fn sub_assign(&mut self, rhs: &Self) {
            if self.0 < rhs.0 {
                self.0 += M::MODULUS;
            }
            self.0 -= rhs.0;
        }
    }

    impl<'a, M: ModSpec> MulAssign<&'a Self> for ModInt<M> {
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
                impl<'a, M: ModSpec> $Op<&'a Self> for ModInt<M> {
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

    impl<'a, M: ModSpec> Neg for &'a ModInt<M> {
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

    impl<'a, M: ByPrime> DivAssign<&'a Self> for ModInt<M>
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

    impl<'a, M: ByPrime> Div<&'a Self> for ModInt<M>
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
}

pub mod static_top_tree {
    /// https://github.com/null-lambda/ps_rust/tree/main/library/src/tree
    pub mod rooted {
        /// # Static Top Tree
        /// Extend dynamic Divide & Conquer (segment tree) to rooted trees with rebalanced HLD, in O(N log N).
        /// Heavily optimized for performance. (Portability & code golfing is not concerns.)
        ///
        /// Compared to usual edge-based top trees, this one is vertex-based. Each compress
        /// cluster represents a left-open, right-closed path. Simultaneous support for path
        /// reversibility of both vertex and edge weights is not provided.
        ///
        /// Supports subtree queries with lazy propagation. Rerooted queries are also
        /// supported for path-reversible clusters.
        ///
        /// ## Reference:
        /// - [[Tutorial] Theorically Faster HLD and Centroid Decomposition](https://codeforces.com/blog/entry/104997/)
        /// - [ABC 351G Editorial](https://atcoder.jp/contests/abc351/editorial/9899)
        /// - [Self-adjusting top tree](https://renatowerneck.wordpress.com/wp-content/uploads/2016/06/tw05-self-adjusting-top-tree.pdf)
        /// - [[Tutorial] Fully Dynamic Trees Supporting Path/Subtree Aggregates and Lazy Path/Subtree Updates](https://codeforces.com/blog/entry/103726)
        ///
        /// See also:
        /// - [maomao90's static top tree visualisation](https://maomao9-0.github.io/static-top-tree-visualisation/)
        ///
        /// ## TODO
        /// - Path query (Probably, implmenting a dynamic top tree would be much easier)
        /// - Persistence!
        use std::{hint::unreachable_unchecked, num::NonZeroU32};

        pub const UNSET: u32 = !0;

        #[derive(Debug)]
        pub enum Cluster<C: ClusterCx> {
            Compress(C::Compress),
            Rake(C::Rake),
        }

        #[derive(Debug, Clone, Copy)]
        pub enum ClusterType {
            Compress,
            Rake,
        }

        pub enum WeightType {
            Vertex,
            UpwardEdge,
        }

        pub trait ClusterCx: Sized {
            // Vertex weight / weight of an upward edge (u -> parent(u)).
            type V: Default + Clone;

            type Compress: Clone; // Path cluster (aggregate on a subchain)
            type Rake: Clone; // Point cluster (Aggregate of light edges)

            // Compress monoid.
            // Left side is always the top side.
            fn id_compress() -> Self::Compress;
            fn compress(&self, lhs: &Self::Compress, rhs: &Self::Compress) -> Self::Compress;

            // Rake monoid, commutative.
            fn id_rake() -> Self::Rake;
            fn rake(&self, lhs: &Self::Rake, rhs: &Self::Rake) -> Self::Rake;

            // A projection.
            fn collapse_compressed(&self, path: &Self::Compress) -> Self::Rake;
            // Attach a rake cluster to a leaf compress cluster.
            fn collapse_raked(&self, point: &Self::Rake, weight: &Self::V) -> Self::Compress;
            // Make a leaf compress cluster without any rake edge.
            fn make_leaf(&self, weight: &Self::V) -> Self::Compress; // In case of no associated rake edge

            // This is how everything is summed up.
            fn pull_up(
                &self,
                node: &mut Cluster<Self>,
                children: [Option<&mut Cluster<Self>>; 2],
                weight: &Self::V,
            ) {
                use Cluster::*;
                match (node, children) {
                    (Compress(c), [Some(Compress(lhs)), Some(Compress(rhs))]) => {
                        *c = self.compress(lhs, rhs)
                    }
                    (Compress(c), [Some(Rake(top)), None]) => *c = self.collapse_raked(top, weight),
                    (Compress(c), [None, None]) => *c = self.make_leaf(weight),
                    (Rake(r), [Some(Rake(lhs)), Some(Rake(rhs))]) => *r = self.rake(lhs, rhs),
                    (Rake(r), [Some(Compress(top)), None]) => *r = self.collapse_compressed(top),
                    _ => unsafe { unreachable_unchecked() },
                }
            }

            // Lazy propagation (Implement it yourself)
            // Store lazy tags in your own rake/compress clusters.
            // To support both subtree updates and path updates, we need multiple aggregates/lazy tags:
            // one for the path and one for the proper subtree.
            const LAZY: bool; // Should we use type-level boolean?
            fn push_down(
                &self,
                node: &mut Cluster<Self>,
                children: [Option<&mut Cluster<Self>>; 2],
                #[allow(unused_variables)] weight: &mut Self::V,
            ) {
                assert!(!Self::LAZY, "Implement push_down for lazy propagation");
                use Cluster::*;

                #[allow(unused_variables)]
                match (node, children) {
                    (Compress(c), [Some(Compress(lhs)), Some(Compress(rhs))]) => todo!(),
                    (Compress(c), [Some(Rake(top)), None]) => todo!(),
                    (Compress(c), [None, None]) => todo!(),
                    (Rake(r), [Some(Rake(lhs)), Some(Rake(rhs))]) => todo!(),
                    (Rake(r), [Some(Compress(top)), None]) => todo!(),
                    _ => unsafe { unreachable_unchecked() },
                }
            }

            // Required for rerooting operations.
            const REVERSE_TYPE: Option<WeightType> = None;
            fn reverse(&self, _path: &Self::Compress) -> Self::Compress {
                panic!("Implement reverse for rerooting operations");
            }
        }

        #[derive(Debug, Copy, Clone)]
        pub enum ActionRange {
            SubTree,
            Path,
        }

        // Lazy propagation (Implement it yourself, Part II)
        pub trait Action<Cx: ClusterCx> {
            fn apply(&self, cluster: &mut Cluster<Cx>, range: ActionRange);
            fn apply_to_weight(&self, weight: &mut Cx::V);
        }

        impl<Cx: ClusterCx> Clone for Cluster<Cx> {
            fn clone(&self) -> Self {
                match self {
                    Cluster::Compress(c) => Cluster::Compress(c.clone()),
                    Cluster::Rake(r) => Cluster::Rake(r.clone()),
                }
            }
        }

        impl<Cx: ClusterCx> Cluster<Cx> {
            pub fn into_result(self) -> Result<Cx::Compress, Cx::Rake> {
                match self {
                    Cluster::Compress(c) => Ok(c),
                    Cluster::Rake(r) => Err(r),
                }
            }

            pub fn get_compress(&self) -> Option<&Cx::Compress> {
                match self {
                    Cluster::Compress(c) => Some(c),
                    _ => None,
                }
            }

            pub fn get_rake(&self) -> Option<&Cx::Rake> {
                match self {
                    Cluster::Rake(r) => Some(r),
                    _ => None,
                }
            }
        }

        // Heavy-Light Decomposition, prior to top tree construction.
        #[derive(Debug, Default)]
        pub struct HLD {
            // Rooted tree structure
            pub parent: Vec<u32>,
            pub topological_order: Vec<u32>,

            // Chain structure
            pub heavy_child: Vec<u32>,
            pub chain_top: Vec<u32>,

            // Light edges, in linked list
            pub first_light_child: Vec<u32>,
            pub xor_light_siblings: Vec<u32>,
        }

        impl HLD {
            pub fn len(&self) -> usize {
                self.parent.len()
            }

            pub fn from_edges<'a>(
                n_verts: usize,
                edges: impl IntoIterator<Item = (u32, u32)>,
                root: usize,
            ) -> Self {
                assert!(n_verts >= 1);
                let mut degree = vec![0u32; n_verts];
                let mut xor_neighbors: Vec<u32> = vec![0u32; n_verts];
                for (u, v) in edges {
                    debug_assert!(u != v);
                    degree[u as usize] += 1;
                    degree[v as usize] += 1;
                    xor_neighbors[u as usize] ^= v;
                    xor_neighbors[v as usize] ^= u;
                }

                // Upward propagation
                let mut size = vec![1; n_verts];
                let mut heavy_child = vec![UNSET; n_verts];
                degree[root] += 2;
                let mut topological_order = Vec::with_capacity(n_verts);
                for mut u in 0..n_verts {
                    while degree[u] == 1 {
                        let p = xor_neighbors[u];
                        topological_order.push(u as u32);
                        degree[u] = 0;
                        degree[p as usize] -= 1;
                        xor_neighbors[p as usize] ^= u as u32;

                        size[p as usize] += size[u as usize];
                        let h = &mut heavy_child[p as usize];
                        if *h == UNSET || size[*h as usize] < size[u as usize] {
                            *h = u as u32;
                        }

                        u = p as usize;
                    }
                }
                topological_order.push(root as u32);
                assert!(topological_order.len() == n_verts, "Invalid tree structure");
                let mut parent = xor_neighbors;
                parent[root] = UNSET;

                let mut first_light_child = vec![UNSET; n_verts];
                let mut xor_light_siblings = vec![UNSET; n_verts];
                for &u in &topological_order[..n_verts - 1] {
                    let p = parent[u as usize];
                    if u == heavy_child[p as usize] {
                        continue;
                    }

                    let c = first_light_child[p as usize];
                    xor_light_siblings[u as usize] = c ^ UNSET;
                    if c != UNSET {
                        xor_light_siblings[c as usize] ^= u as u32 ^ UNSET;
                    }
                    first_light_child[p as usize] = u;
                }

                // Downward propagation
                let mut chain_top = vec![UNSET; n_verts];
                for u in topological_order.iter().copied().rev() {
                    if chain_top[u as usize] != UNSET {
                        continue;
                    }
                    let mut h = u;
                    loop {
                        chain_top[h as usize] = u;
                        h = heavy_child[h as usize];
                        if h == UNSET {
                            break;
                        }
                    }
                }

                Self {
                    parent,
                    topological_order,

                    heavy_child,
                    chain_top,

                    first_light_child,
                    xor_light_siblings,
                }
            }
        }

        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct NodeRef(NonZeroU32);

        impl NodeRef {
            fn new(idx: u32) -> Self {
                Self(NonZeroU32::new(idx).unwrap())
            }

            pub fn usize(&self) -> usize {
                self.0.get() as usize
            }

            fn get_with_children_in<'a, T>(
                &self,
                children: &'a [[Option<NodeRef>; 2]],
                xs: &'a mut [T],
            ) -> (&'a mut T, [Option<&'a mut T>; 2]) {
                let children = &children[self.usize()];
                let ptr = xs.as_mut_ptr();

                unsafe {
                    (
                        &mut *ptr.add(self.usize()),
                        children.map(|c| c.map(|c| &mut *ptr.add(c.usize()))),
                    )
                }
            }
        }

        pub struct StaticTopTree<Cx: ClusterCx> {
            // Represented tree structure
            pub hld: HLD,
            n_verts: usize,

            // Top tree topology
            root_node: NodeRef,
            size: Vec<u32>,
            children: Vec<[Option<NodeRef>; 2]>,
            parent: Vec<Option<NodeRef>>,
            n_nodes: usize,

            compress_leaf: Vec<NodeRef>, // Leaf node in compress tree (true leaf, or a collapsed rake tree)
            compress_root: Vec<NodeRef>, // Root node in compress tree

            // Maps node indices to their positions in the binary completion of the top tree.
            // Required for fast node locations, path queries and lazy propagations.
            index_in_binary_completion: Vec<u64>,

            // Weights and aggregates
            pub cx: Cx,
            clusters: Vec<Cluster<Cx>>,
            weights: Vec<Cx::V>,
        }

        impl<Cx: ClusterCx> StaticTopTree<Cx> {
            pub fn from_edges(
                n_verts: usize,
                edges: impl IntoIterator<Item = (u32, u32)>,
                root: usize,
                cx: Cx,
            ) -> Self {
                let hld = HLD::from_edges(n_verts, edges, root);
                let dummy = NodeRef::new(!0);
                let nodes_cap = n_verts * 4 + 1;
                let mut this = Self {
                    hld: Default::default(),
                    n_verts,

                    root_node: dummy,
                    size: vec![1; nodes_cap],
                    children: vec![[None; 2]; nodes_cap],
                    parent: vec![None; nodes_cap],
                    n_nodes: 1,

                    compress_leaf: vec![dummy; nodes_cap],
                    compress_root: vec![dummy; nodes_cap],

                    index_in_binary_completion: vec![0; nodes_cap],

                    clusters: vec![Cluster::Compress(Cx::id_compress()); nodes_cap],
                    weights: vec![Default::default(); nodes_cap],

                    cx,
                };

                this.build_topology(&hld);
                this.build_locators();
                this.hld = hld;

                this
            }

            // Build the top tree

            fn alloc(&mut self, children: [Option<NodeRef>; 2]) -> NodeRef {
                let u = NodeRef::new(self.n_nodes as u32);
                self.children[u.usize()] = children;
                for &child in children.iter().flatten() {
                    self.parent[child.usize()] = Some(u);
                    self.size[u.usize()] += self.size[child.usize()];
                }
                self.n_nodes += 1;
                u
            }

            fn build_topology(&mut self, hld: &HLD) {
                for &u in &hld.topological_order {
                    // Build a rake tree
                    let mut light_edges = vec![];
                    let mut l = hld.first_light_child[u as usize];
                    let mut prev = UNSET;
                    while l != UNSET {
                        // Collapse a compress tree
                        light_edges.push(self.alloc([Some(self.compress_root[l as usize]), None]));

                        let next = hld.xor_light_siblings[l as usize] ^ prev;
                        prev = l;
                        l = next;
                    }

                    self.compress_leaf[u as usize] = if light_edges.is_empty() {
                        // Make a leaf cluster
                        self.alloc([None, None])
                    } else {
                        // Collapse a rake tree
                        let rake_root =
                            self.fold_balanced_rec(&light_edges, || Cluster::Rake(Cx::id_rake()));
                        self.alloc([Some(rake_root), None])
                    };

                    if hld.chain_top[u as usize] == u {
                        // Build a compress tree
                        let mut h = u as usize;
                        let mut chain = vec![];
                        loop {
                            chain.push(self.compress_leaf[h]);
                            h = hld.heavy_child[h] as usize;
                            if h == UNSET as usize {
                                break;
                            }
                        }
                        self.compress_root[u as usize] =
                            self.fold_balanced_rec(&chain, || Cluster::Compress(Cx::id_compress()));
                    }
                }
                self.root_node =
                    self.compress_root[*hld.topological_order.last().unwrap() as usize];
            }

            // Make the tree balanced in the global sense.
            fn fold_balanced_rec(
                &mut self,
                nodes: &[NodeRef],
                id_cluster: impl Fn() -> Cluster<Cx> + Copy,
            ) -> NodeRef {
                debug_assert!(!nodes.is_empty());
                if nodes.len() == 1 {
                    self.clusters[nodes[0].usize()] = id_cluster();
                    return nodes[0];
                }

                // Split at the middle. If the split point is not exact, make the tree left-skewed.
                let mut total_size = nodes.iter().map(|u| self.size[u.usize()]).sum::<u32>() as i32;
                let i = nodes
                    .iter()
                    .rposition(|u| {
                        total_size -= self.size[u.usize()] as i32 * 2;
                        total_size <= 0
                    })
                    .unwrap()
                    .max(1);

                let (lhs, rhs) = nodes.split_at(i);
                let lhs = self.fold_balanced_rec(lhs, id_cluster);
                let rhs = self.fold_balanced_rec(rhs, id_cluster);
                let node = self.alloc([Some(lhs), Some(rhs)]);
                self.clusters[node.usize()] = id_cluster();
                node
            }

            fn build_locators(&mut self) {
                self.index_in_binary_completion[self.root_node.usize()] = 1;
                for u in (1..self.n_nodes as u32).rev().map(NodeRef::new) {
                    let i = self.index_in_binary_completion[u.usize()];
                    for branch in 0..2 {
                        if let Some(c) = self.children[u.usize()][branch as usize] {
                            self.index_in_binary_completion[c.usize()] = i << 1 | branch;
                        }
                    }
                }
            }

            fn depth(&self, u: NodeRef) -> u32 {
                let path = self.index_in_binary_completion[u.usize()];
                u64::BITS - 1 - u64::leading_zeros(path)
            }

            pub fn init_weights(&mut self, weights: impl IntoIterator<Item = (usize, Cx::V)>) {
                for (u, w) in weights {
                    debug_assert!(u < self.n_verts);
                    self.weights[self.compress_leaf[u].usize()] = w;
                }

                for u in (1..self.n_nodes as u32).map(NodeRef::new) {
                    self.pull_up(u);
                }
            }

            // A bunch of propagation helpers

            fn push_down(&mut self, u: NodeRef) {
                if !Cx::LAZY {
                    return;
                }

                let (node, children) = u.get_with_children_in(&self.children, &mut self.clusters);
                self.cx
                    .push_down(node, children, &mut self.weights[u.usize()]);
            }

            fn pull_up(&mut self, u: NodeRef) {
                let (node, children) = u.get_with_children_in(&self.children, &mut self.clusters);
                self.cx.pull_up(node, children, &self.weights[u.usize()]);
            }

            fn push_down_from_root(&mut self, u: NodeRef) {
                if !Cx::LAZY {
                    return;
                }

                let mut v = self.root_node;
                self.push_down(v);
                let path = self.index_in_binary_completion[u.usize()];
                for d in (0..self.depth(u)).rev() {
                    let branch = (path >> d) & 1;
                    v = unsafe { self.children[v.usize()][branch as usize].unwrap_unchecked() };
                    self.push_down(v);
                }
            }

            fn pull_up_to_root(&mut self, mut u: NodeRef) {
                self.pull_up(u);
                while let Some(p) = self.parent[u.usize()] {
                    u = p;
                    self.pull_up(u);
                }
            }

            pub fn sum_rerooted(&mut self, u: usize) -> (Cx::Rake, &Cx::V) {
                assert!(
                    Cx::REVERSE_TYPE.is_some(),
                    "Requires reversible compress clusters"
                );
                self.push_down_from_root(self.compress_leaf[u]);

                let u = self.compress_leaf[u];
                let path = self.index_in_binary_completion[u.usize()];

                // Descend from the root. Fold every chain in half, and propagate it down to the lower chain.
                // Do an exclusive sum for the rake tree.
                let mut c_prefix = Cx::id_compress();
                let mut c_suffix = Cx::id_compress();
                let mut r_exclusive = Cx::id_rake();
                let mut rake_pivot = self.root_node;

                match Cx::REVERSE_TYPE {
                    Some(WeightType::Vertex) => {
                        let mut v = self.root_node; // Dummy
                        for branch in (0..self.depth(u))
                            .rev()
                            .map(|d| (path >> d) & 1)
                            .chain(Some(0))
                        {
                            use Cluster::*;
                            match v.get_with_children_in(&self.children, &mut self.clusters) {
                                (Compress(_), [Some(Compress(lhs)), Some(Compress(rhs))]) => {
                                    if branch == 0 {
                                        c_suffix = self.cx.compress(rhs, &c_suffix);
                                    } else {
                                        c_prefix = self.cx.compress(&c_prefix, lhs);
                                    }
                                }
                                (Compress(_), [rake_root, _]) => {
                                    r_exclusive = self.cx.rake(
                                        &self.cx.collapse_compressed(&self.cx.reverse(&c_prefix)),
                                        &self.cx.collapse_compressed(&c_suffix),
                                    );
                                    rake_pivot = v;

                                    if v == u {
                                        if let Some(Rake(rake_root)) = rake_root {
                                            r_exclusive = self.cx.rake(&r_exclusive, rake_root);
                                        }
                                        break;
                                    }
                                }
                                (Rake(_), [Some(Rake(lhs)), Some(Rake(rhs))]) => {
                                    r_exclusive = if branch == 0 {
                                        self.cx.rake(&r_exclusive, rhs)
                                    } else {
                                        self.cx.rake(&r_exclusive, lhs)
                                    };
                                }

                                (Rake(_), [Some(Compress(_)), None]) => {
                                    c_prefix = self.cx.collapse_raked(
                                        &r_exclusive,
                                        &self.weights[rake_pivot.usize()],
                                    );
                                    c_suffix = Cx::id_compress();
                                }
                                _ => unsafe { unreachable_unchecked() },
                            }

                            v = unsafe {
                                self.children[v.usize()][branch as usize].unwrap_unchecked()
                            };
                        }

                        (r_exclusive, &self.weights[u.usize()])
                    }
                    Some(WeightType::UpwardEdge) => {
                        unimplemented!()
                    }
                    _ => unreachable!(),
                }
            }

            pub fn modify(&mut self, u: usize, update_with: impl FnOnce(&mut Cx::V)) {
                assert!(
                    !Cx::LAZY,
                    "Do not mix arbitrary point updates with lazy propagation"
                );
                let u = self.compress_leaf[u];
                update_with(&mut self.weights[u.usize()]);
                self.pull_up_to_root(u);
            }
        }
    }
}

type ModP = ModInt<ByU64<1_000_000_007>>;

// Dynamic connectivity for an induced subgraph of a tree
struct ConnectedSubsets;

#[derive(Debug, Clone)]
struct Compress {
    left: ModP,
    right: ModP,
    full: ModP,
}

impl ClusterCx for ConnectedSubsets {
    type V = u8;
    type Compress = Compress;
    type Rake = ModP;

    fn id_compress() -> Self::Compress {
        Compress {
            left: ModP::zero(),
            right: ModP::zero(),
            full: ModP::one(),
        }
    }
    fn compress(&self, lhs: &Self::Compress, rhs: &Self::Compress) -> Self::Compress {
        Compress {
            left: lhs.left + lhs.full * rhs.left,
            right: rhs.right + rhs.full * lhs.right,
            full: lhs.full * rhs.full,
        }
    }

    fn id_rake() -> Self::Rake {
        ModP::one()
    }
    fn rake(&self, lhs: &Self::Rake, rhs: &Self::Rake) -> Self::Rake {
        *lhs * *rhs
    }

    fn collapse_compressed(&self, path: &Self::Compress) -> Self::Rake {
        path.left + ModP::one()
    }
    fn collapse_raked(&self, point: &Self::Rake, &weight: &Self::V) -> Self::Compress {
        let c = ModP::from(weight) * *point;
        Compress {
            left: c,
            right: c,
            full: c,
        }
    }
    fn make_leaf(&self, &weight: &Self::V) -> Self::Compress {
        let c = ModP::from(weight);
        Compress {
            left: c,
            right: c,
            full: c,
        }
    }

    const LAZY: bool = false;

    const REVERSE_TYPE: Option<WeightType> = Some(WeightType::Vertex);
    fn reverse(&self, path: &Self::Compress) -> Self::Compress {
        let mut path = path.clone();
        std::mem::swap(&mut path.left, &mut path.right);
        path
    }
}

fn main() {
    let mut input = fast_io::stdin();
    let mut output = fast_io::stdout();

    let d: i32 = input.value();
    let n: usize = input.value();

    let mut events: Vec<_> = (0..n).map(|i| (input.u32() as i32 - 1, i as u32)).collect();
    events.sort_unstable();

    let edges = (0..n - 1).map(|_| (input.u32() - 1, input.u32() - 1));
    let mut stt = StaticTopTree::from_edges(n, edges, 0, ConnectedSubsets);
    stt.init_weights((0..n).map(|u| (u, 0)));

    let events_add = events.iter().copied();
    let mut events_remove = events.iter().copied().peekable();

    let mut ans = ModP::zero();

    for (upper, u) in events_add {
        stt.modify(u as usize, |w| *w = 1);
        let lower = upper - d;
        while let Some((_, u)) = events_remove.next_if(|&(x, _)| x < lower) {
            stt.modify(u as usize, |w| *w = 0);
        }

        let (proper_subtree, &_) = stt.sum_rerooted(u as usize);
        ans += proper_subtree;
    }
    writeln!(output, "{}", u64::from(ans)).unwrap();
}
