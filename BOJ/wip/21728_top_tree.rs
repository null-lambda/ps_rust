use std::{cmp::Ordering, io::Write};

use top_tree::ClusterCx;

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

    impl<U: std::fmt::Display, M: ModSpec<U = U>> std::fmt::Display for ModInt<M> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.0.fmt(f)
        }
    }
}

pub mod top_tree {
    /// # Splay Top Tree
    /// Manage dynamic tree dp with link-cut operations in amortized O(N log N).
    ///
    /// Most implementation are derived from the paper of Tarjan & Werneck, with some minor modifications:
    /// - Circular order between rake edges does not preserved, to simplify node structure.
    /// - Each compress node represents an **open** path (without boundary points),
    ///   while a rake node represents a **left-open, right-closed** path.
    ///   Boundary vertices are merged into clusters as late as possible (or not at all) to minimize over-computation.
    /// - We use a separate cluster and node type for compress trees and rake trees.
    ///
    /// ## Reference:
    /// - [Self-adjusting top trees](https://renatowerneck.wordpress.com/wp-content/uploads/2016/06/tw05-self-adjusting-top-tree.pdf)
    /// - [[Tutorial] Fully Dynamic Trees Supporting Path/Subtree Aggregates and Lazy Path/Subtree Updates](https://codeforces.com/blog/entry/103726)
    ///
    /// ## TODO
    ///
    /// ### Optimization
    /// - Remove unnecessary `pull_up`'s.
    /// - Re-implement `modify_edge` without using link/cut operations.
    /// - Simplify control flow in `soft_expose` if possible.
    use std::{
        cmp::Ordering,
        hint::unreachable_unchecked,
        marker::PhantomData,
        num::NonZeroU32,
        ops::{Index, IndexMut},
    };

    use node::BinaryNode;

    pub const UNSET: u32 = !0;

    pub trait ClusterCx: Sized {
        // Vertex weight.
        type V: Default + Clone;

        /// Path cluster (aggregation on a subchain), represented as an **open** interval.
        type C: Clone;

        /// Point cluster (aggregation of light edges), represented as a **left-open, right-closed** interval.
        type R: Clone;

        /// Compress monoid.
        /// Left side is always the top side.
        fn id_compress() -> Self::C;

        fn compress(&self, children: [&Self::C; 2], v: &Self::V, rake: Option<&Self::R>)
            -> Self::C;

        /// Rake monoid, commutative.
        fn id_rake() -> Self::R;

        fn rake(&self, children: [&Self::R; 2]) -> Self::R;

        /// Enclose the right end of a path cluster with a vertex.
        fn collapse_path(&self, c: &Self::C, vr: &Self::V) -> Self::R;

        /// Lazy propagation (implement it yourself)
        #[allow(unused_variables)]
        fn push_down_compress(
            &self,
            node: &mut Self::C,
            children: [&mut Self::C; 2],
            v: &mut Self::V,
            rake: Option<&mut Self::R>,
        ) {
        }

        #[allow(unused_variables)]
        fn push_down_rake(&self, node: &mut Self::R, children: [&mut Self::R; 2]) {}

        #[allow(unused_variables)]
        fn push_down_collapsed(&self, node: &mut Self::R, c: &mut Self::C, vr: &mut Self::V) {}

        #[must_use] // TEMP
        fn reverse(&self, c: &Self::C) -> Self::C;
    }

    #[derive(Debug, Copy, Clone, PartialEq, Eq)]
    pub enum ActionRange {
        Subtree,
        Path,
    }

    /// Lazy propagation (Implement it yourself, Part II)
    pub trait Action<Cx: ClusterCx> {
        fn apply_to_compress(&mut self, compress: &mut Cx::C, range: ActionRange);
        fn apply_to_rake(&mut self, rake: &mut Cx::R);
        fn apply_to_weight(&mut self, weight: &mut Cx::V);
    }

    pub struct NodeRef<T> {
        idx: NonZeroU32,
        _phantom: std::marker::PhantomData<*mut T>,
    }

    #[derive(Debug, Clone)]
    pub struct Pool<T> {
        pub nodes: Vec<T>,
        pub free: Vec<NodeRef<T>>,
    }

    impl<T> Index<NodeRef<T>> for Pool<T> {
        type Output = T;
        fn index(&self, index: NodeRef<T>) -> &Self::Output {
            &self.nodes[index.idx.get() as usize]
        }
    }

    impl<T> IndexMut<NodeRef<T>> for Pool<T> {
        fn index_mut(&mut self, index: NodeRef<T>) -> &mut Self::Output {
            &mut self.nodes[index.idx.get() as usize]
        }
    }

    impl<T> Pool<T> {
        pub unsafe fn many_mut<'a, const N: usize>(
            &'a mut self,
            indices: [NodeRef<T>; N],
        ) -> [&'a mut T; N] {
            let ptr = self.nodes.as_mut_ptr();
            indices.map(|i| &mut *ptr.add(i.idx.get() as usize))
        }
    }

    impl<T> NodeRef<T> {
        pub fn new(idx: u32) -> Self {
            Self {
                idx: NonZeroU32::new(idx).unwrap(),
                _phantom: Default::default(),
            }
        }

        pub unsafe fn dangling() -> Self {
            Self {
                idx: NonZeroU32::new(UNSET).unwrap(),
                _phantom: PhantomData,
            }
        }
    }

    impl<T> Clone for NodeRef<T> {
        fn clone(&self) -> Self {
            Self {
                idx: self.idx,
                _phantom: Default::default(),
            }
        }
    }

    impl<T> Copy for NodeRef<T> {}

    impl<T> PartialEq for NodeRef<T> {
        fn eq(&self, other: &Self) -> bool {
            self.idx == other.idx
        }
    }

    impl<T> Eq for NodeRef<T> {}

    impl<T> std::fmt::Debug for NodeRef<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            write!(f, "{}", self.idx.get())
        }
    }

    pub mod node {
        use std::fmt::Debug;

        use super::*;

        pub enum Parent<Cx: ClusterCx> {
            Compress(NodeRef<Compress<Cx>>),
            Rake(NodeRef<Rake<Cx>>),
        }

        impl<Cx: ClusterCx> Clone for Parent<Cx> {
            fn clone(&self) -> Self {
                match self {
                    Parent::Compress(c) => Parent::Compress(*c),
                    Parent::Rake(r) => Parent::Rake(*r),
                }
            }
        }

        impl<Cx: ClusterCx> Copy for Parent<Cx> {}

        impl<Cx: ClusterCx> PartialEq for Parent<Cx> {
            fn eq(&self, other: &Self) -> bool {
                match (self, other) {
                    (Parent::Compress(c1), Parent::Compress(c2)) if c1 == c2 => true,
                    (Parent::Rake(r1), Parent::Rake(r2)) if r1 == r2 => true,
                    _ => false,
                }
            }
        }

        impl<Cx: ClusterCx> Eq for Parent<Cx> {}

        #[derive(Debug)]
        pub struct CompressPivot<Cx: ClusterCx> {
            pub children: [NodeRef<Compress<Cx>>; 2],
            pub rake_tree: Option<NodeRef<Rake<Cx>>>,
        }

        #[derive(Debug)]
        pub struct Compress<Cx: ClusterCx> {
            // Endpoints of a path cluster, also used as a tag for lazy path reversal.
            pub ends: [u32; 2],

            pub parent: Option<Parent<Cx>>,
            pub pivot: Option<CompressPivot<Cx>>,

            pub sum: Cx::C,
        }

        #[derive(Debug)]
        pub struct Rake<Cx: ClusterCx> {
            pub parent: Parent<Cx>,
            pub children: Result<[NodeRef<Rake<Cx>>; 2], NodeRef<Compress<Cx>>>,

            pub sum: Cx::R,
        }

        impl<Cx: ClusterCx> CompressPivot<Cx> {
            pub unsafe fn uninit() -> Self {
                Self {
                    children: [NodeRef::dangling(); 2],
                    rake_tree: None,
                }
            }
        }

        impl<Cx: ClusterCx> Debug for Parent<Cx> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    Parent::Compress(c) => write!(f, "Compress({c:?})"),
                    Parent::Rake(r) => write!(f, "Rake({r:?})"),
                }
            }
        }

        pub trait BinaryNode: Sized {
            type Parent: Copy;

            unsafe fn uninit() -> Self;

            fn internal_parent(&self) -> Option<NodeRef<Self>>;
            fn parent_mut(&mut self) -> &mut Self::Parent;
            fn is_internal_root(&self) -> bool {
                self.internal_parent().is_none()
            }

            fn children(&self) -> Option<&[NodeRef<Self>; 2]>;
            fn children_mut(&mut self) -> Option<&mut [NodeRef<Self>; 2]>;
        }

        impl<Cx: ClusterCx> BinaryNode for Compress<Cx> {
            type Parent = Option<Parent<Cx>>;

            unsafe fn uninit() -> Self {
                Self {
                    ends: [UNSET; 2],

                    parent: None,
                    pivot: None,

                    sum: Cx::id_compress(),
                }
            }

            fn internal_parent(&self) -> Option<NodeRef<Self>> {
                match self.parent {
                    Some(Parent::Compress(c)) => Some(c),
                    _ => None,
                }
            }

            fn parent_mut(&mut self) -> &mut Self::Parent {
                &mut self.parent
            }

            fn children(&self) -> Option<&[NodeRef<Self>; 2]> {
                Some(&self.pivot.as_ref()?.children)
            }

            fn children_mut(&mut self) -> Option<&mut [NodeRef<Self>; 2]> {
                Some(&mut self.pivot.as_mut()?.children)
            }
        }

        impl<Cx: ClusterCx> Compress<Cx> {
            pub fn reverse(&mut self, cx: &Cx) {
                self.sum = cx.reverse(&self.sum);
                self.ends.swap(0, 1);
                self.children_mut().map(|cs| cs.swap(0, 1));
            }
        }

        impl<Cx: ClusterCx> BinaryNode for Rake<Cx> {
            type Parent = Parent<Cx>;

            unsafe fn uninit() -> Self {
                Self {
                    parent: Parent::Compress(NodeRef::dangling()),
                    children: Err(NodeRef::dangling()),
                    sum: Cx::id_rake(),
                }
            }

            fn internal_parent(&self) -> Option<NodeRef<Self>> {
                match self.parent {
                    Parent::Rake(r) => Some(r),
                    _ => None,
                }
            }

            fn parent_mut(&mut self) -> &mut Self::Parent {
                &mut self.parent
            }

            fn children(&self) -> Option<&[NodeRef<Self>; 2]> {
                self.children.as_ref().ok()
            }

            fn children_mut(&mut self) -> Option<&mut [NodeRef<Self>; 2]> {
                self.children.as_mut().ok()
            }
        }
    }

    impl<T: BinaryNode> Default for Pool<T> {
        fn default() -> Self {
            Self {
                nodes: vec![unsafe { node::BinaryNode::uninit() }],
                free: vec![],
            }
        }
    }

    impl<T: BinaryNode> Pool<T> {
        fn alloc(&mut self, node: T) -> NodeRef<T> {
            let u = if let Some(u) = self.free.pop() {
                self[u] = node;
                u
            } else {
                let idx = self.nodes.len() as u32;
                self.nodes.push(node);
                NodeRef::new(idx)
            };

            u
        }

        unsafe fn mark_free(&mut self, u: NodeRef<T>) {
            self.free.push(u);
        }
    }

    pub struct TopTree<Cx: ClusterCx> {
        pub cx: Cx,

        /// Compress tree
        pub cs: Pool<node::Compress<Cx>>,
        /// Rake tree
        pub rs: Pool<node::Rake<Cx>>,

        /// Vertex info
        pub n_verts: usize,
        pub weights: Vec<Cx::V>,

        /// `handle(v)` is the only compress-node that requires vertex information. There are three cases:
        /// 1. If `degree(v)` = 0 (isolated vertex), then `handle(v)` = null.
        /// 2. If `degree(v)` = 1 (boundary vertex), then `handle(v)` = [topmost root compress-node].
        /// 3. If `degree(v)` ≥ 2 (interior vertex), then `handle(v)` = [compress-node with v as the pivot].

        /// Inversely, each compress-node interacts with at most three vertices:
        /// - Non-root leaf node: 0 vertices.
        /// - Non-root branch node: 1 vertex (compression pivot).
        /// - Collapsed root node: 1 or 2 vertices (right end with an optional compression pivot).
        /// - True root node: 2 or 3 vertices (both ends with an optional compression pivot).
        pub handle: Vec<Option<NodeRef<node::Compress<Cx>>>>,
    }

    /// Splay tree structure for compress and rake trees.
    /// Handles operations where the node topology of compress and rake trees does not swizzle.
    pub trait InternalSplay<T: BinaryNode> {
        fn pool(&mut self) -> &mut Pool<T>;

        fn push_down(&mut self, u: NodeRef<T>);
        fn pull_up(&mut self, u: NodeRef<T>);

        fn branch(&mut self, u: NodeRef<T>) -> Option<(NodeRef<T>, usize)> {
            let pool = self.pool();
            let p = pool[u].internal_parent()?;
            let branch = unsafe { pool[p].children().unwrap_unchecked() }[1] == u;
            Some((p, branch as usize))
        }

        /// Handles interactions between the rake and compress trees after an internal splay.
        ///
        /// The `rotate` operation is internal, applying only within a compress or rake tree.
        /// It fixes the virtual root (i.e., the compress parent of a rake tree root or the rake parent
        /// of a compress tree root), but the virtual root’s child link must be manually updated afterward.
        unsafe fn update_virtual_parent_link(&mut self, u: NodeRef<T>);

        /// Converts `p ->(left child) u ->(right child) c` to `u ->(right child) p ->(left child) c`.
        /// (If `p ->(right child) u`, flip (left child) <-> (right child).)
        ///
        /// ## Constraints
        /// 1. u must be a non-root branch.
        /// 2. push_down and pull_up for (g?), p, and u must be called beforehand.
        ///
        /// ## Diagram
        /// ┌────────┐     ┌───────┐
        /// │      g?│     │    g? │
        /// │     /  │     │   /   │
        /// │    p   │     │  u    │
        /// │   / \  │ ==> │ / \   │
        /// │  u   4 │     │0   p  │
        /// │ / \    │     │   / \ │
        /// │0   c   │     │  c   4│
        /// └────────┘     └───────┘
        unsafe fn rotate(&mut self, u: NodeRef<T>) {
            let (p, bp) = self.branch(u).unwrap_unchecked();
            let c = std::mem::replace(
                &mut self.pool()[u].children_mut().unwrap_unchecked()[bp ^ 1],
                p,
            );
            self.pool()[p].children_mut().unwrap_unchecked()[bp] = c;

            if let Some((g, bg)) = self.branch(p) {
                self.pool()[g].children_mut().unwrap_unchecked()[bg as usize] = u;
            }
            let pp = *self.pool()[p].parent_mut();
            *self.pool()[p].parent_mut() = *self.pool()[c].parent_mut();
            *self.pool()[c].parent_mut() = *self.pool()[u].parent_mut();
            *self.pool()[u].parent_mut() = pp;
        }

        /// Drag `u` up under the guard node. If `u` is a leaf, drag `parent(u)` if it exists.
        /// (An internal tree should be a full binary tree, so the leaves cannot be splayed.)"
        ///
        /// ## Diagram
        /// Step zig-zig:
        /// ┌─────────┐     ┌─────────┐
        /// │      g  │     │  u      │
        /// │     / \ │     │ / \     │
        /// │    p   6│     │0   p    │
        /// │   / \   │ ==> │   / \   │
        /// │  u   4  │     │  2   g  │
        /// │ / \     │     │     / \ │
        /// │0   2    │     │    4   6│
        /// └─────────┘     └─────────┘
        ///
        /// Step zig-zag:
        /// ┌───────┐
        /// │    g  │     ┌───────────┐
        /// │   / \ │     │     u     │
        /// │  p   6│     │   /   \   │
        /// │ / \   │ ==> │  p     g  │
        /// │0   u  │     │ / \   / \ │
        /// │   / \ │     │0   2 4   6│
        /// │   2  4│     └───────────┘
        /// └───────┘
        ///
        /// Step zig:
        /// ┌────────┐     ┌───────┐
        /// │    p   │     │  u    │
        /// │   / \  │     │ / \   │
        /// │  u   4 │ ==> │0   p  │
        /// │ / \    │     │   / \ │
        /// │0   c   │     │  c   4│
        /// └────────┘     └───────┘
        unsafe fn guarded_splay(&mut self, mut u: NodeRef<T>, guard: Option<NodeRef<T>>) {
            if self.pool()[u].children().is_none() {
                if let Some(p) = self.pool()[u].internal_parent() {
                    u = p;
                } else {
                    return;
                }
            }

            unsafe {
                while let Some(p) = self.pool()[u]
                    .internal_parent()
                    .filter(|&p| Some(p) != guard)
                {
                    if let Some(g) = self.pool()[p]
                        .internal_parent()
                        .filter(|&g| Some(g) != guard)
                    {
                        self.push_down(g);
                        self.push_down(p);
                        self.push_down(u);

                        let (_, bp) = self.branch(u).unwrap_unchecked();
                        let (_, bg) = self.branch(p).unwrap_unchecked();
                        if bp == bg {
                            self.rotate(p); // zig-zig
                        } else {
                            self.rotate(u); // zig-zag
                        }
                        self.rotate(u);

                        self.pull_up(g);
                        self.pull_up(p);
                        self.pull_up(u);
                    } else {
                        self.push_down(p);
                        self.push_down(u);

                        self.rotate(u); // zig

                        self.pull_up(p);
                        self.pull_up(u);
                    }
                }

                self.update_virtual_parent_link(u);
            }
        }

        /// Drag `u` up to the root. If `u` is a leaf, drag `parent(u)` if it exists.
        ///
        /// Splaying `handle(u)` should always make it the root in the compressed tree.
        /// However, splaying a leaf rake-node doesn't guarantee this.
        fn splay(&mut self, u: NodeRef<T>) {
            unsafe { self.guarded_splay(u, None) };
        }

        fn walk_down_internal(
            &mut self,
            mut u: NodeRef<T>,
            mut locator: impl FnMut(&mut Self, NodeRef<T>) -> Ordering,
        ) -> NodeRef<T> {
            loop {
                self.push_down(u);
                match locator(self, u) {
                    Ordering::Equal => return u,
                    Ordering::Less => {
                        if let Some(children) = self.pool()[u].children() {
                            u = children[0];
                        } else {
                            return u;
                        }
                    }
                    Ordering::Greater => {
                        if let Some(children) = self.pool()[u].children() {
                            u = children[1];
                        } else {
                            return u;
                        }
                    }
                }
            }
        }

        fn splay_prev(&mut self, root: NodeRef<T>) -> Option<NodeRef<T>> {
            debug_assert!(self.pool()[root].internal_parent().is_none());

            let mut u = root;
            self.push_down(u);
            u = self.pool()[u].children()?[0];
            u = self.walk_down_internal(u, |_, _| Ordering::Greater);

            self.splay(u);
            Some(u)
        }

        fn splay_next(&mut self, root: NodeRef<T>) -> Option<NodeRef<T>> {
            debug_assert!(self.pool()[root].internal_parent().is_none());

            let mut u = root;
            self.push_down(u);
            u = self.pool()[u].children()?[1];
            u = self.walk_down_internal(u, |_, _| Ordering::Less);

            self.splay(u);
            Some(u)
        }

        fn splay_first(&mut self, root: NodeRef<T>) -> NodeRef<T> {
            debug_assert!(self.pool()[root].internal_parent().is_none());

            let u = self.walk_down_internal(root, |_, _| Ordering::Less);
            self.splay(u);
            u
        }

        fn splay_last(&mut self, root: NodeRef<T>) -> NodeRef<T> {
            debug_assert!(self.pool()[root].internal_parent().is_none());

            let u = self.walk_down_internal(root, |_, _| Ordering::Greater);
            self.splay(u);
            u
        }

        fn inorder(&mut self, root: NodeRef<T>, visitor: &mut impl FnMut(&mut Self, NodeRef<T>)) {
            if let Some(children) = self.pool()[root].children().copied() {
                self.inorder(children[0], visitor);
                visitor(self, root);
                self.inorder(children[1], visitor);
            } else {
                visitor(self, root);
            }
        }
    }

    impl<Cx: ClusterCx> InternalSplay<node::Compress<Cx>> for TopTree<Cx> {
        fn pool(&mut self) -> &mut Pool<node::Compress<Cx>> {
            &mut self.cs
        }

        fn push_down(&mut self, u: NodeRef<node::Compress<Cx>>) {
            unsafe {
                if let Some(pivot) = &mut self.cs[u].pivot {
                    let rake_tree = pivot.rake_tree.map(|r| &mut self.rs[r].sum);
                    let [l, r] = pivot.children;

                    let [u, l, r] = self.cs.many_mut([u, l, r]);

                    let rev_lazy = l.ends[1] != r.ends[0];
                    if rev_lazy {
                        l.reverse(&self.cx);
                        r.reverse(&self.cx);
                    }

                    debug_assert_eq!(l.ends[1], r.ends[0]);
                    let vert = l.ends[1];

                    self.cx.push_down_compress(
                        &mut u.sum,
                        [&mut l.sum, &mut r.sum],
                        &mut self.weights[vert as usize],
                        rake_tree,
                    );
                }
            }
        }

        fn pull_up(&mut self, u: NodeRef<node::Compress<Cx>>) {
            unsafe {
                if let Some(pivot) = &self.cs[u].pivot {
                    let rake_tree = pivot.rake_tree;
                    let rake_tree_sum = rake_tree.map(|r| &self.rs[r].sum);

                    let [l, r] = pivot.children;
                    let [u_mut, l, r] = self.cs.many_mut([u, l, r]);

                    debug_assert_eq!(l.ends[1], r.ends[0], "u={u:?} {:?} {:?}", l.ends, r.ends);
                    let vert = l.ends[1];
                    u_mut.ends = [l.ends[0], r.ends[1]];
                    self.handle[vert as usize] = Some(u);

                    u_mut.sum = self.cx.compress(
                        [&l.sum, &r.sum],
                        &mut self.weights[vert as usize],
                        rake_tree_sum,
                    );
                }

                self.update_boundary_handles(u);
            }
        }

        unsafe fn update_virtual_parent_link(&mut self, u: NodeRef<node::Compress<Cx>>) {
            if let Some(node::Parent::Rake(ru)) = self.cs[u].parent {
                self.rs[ru].children = Err(u);
            }
        }
    }

    impl<Cx: ClusterCx> InternalSplay<node::Rake<Cx>> for TopTree<Cx> {
        fn pool(&mut self) -> &mut Pool<node::Rake<Cx>> {
            &mut self.rs
        }

        fn push_down(&mut self, u: NodeRef<node::Rake<Cx>>) {
            unsafe {
                match self.rs[u].children {
                    Ok([l, r]) => {
                        let [u, l, r] = self.rs.many_mut([u, l, r]);
                        self.cx.push_down_rake(&mut u.sum, [&mut l.sum, &mut r.sum]);
                    }
                    Err(compress_tree) => {
                        let compress_tree = &mut self.cs[compress_tree];
                        self.cx.push_down_collapsed(
                            &mut self.rs[u].sum,
                            &mut compress_tree.sum,
                            &mut self.weights[compress_tree.ends[1] as usize],
                        );
                    }
                }
            }
        }

        fn pull_up(&mut self, u: NodeRef<node::Rake<Cx>>) {
            unsafe {
                match self.rs[u].children {
                    Ok([l, r]) => {
                        let [u_mut, l, r] = self.rs.many_mut([u, l, r]);
                        u_mut.sum = self.cx.rake([&l.sum, &r.sum]);
                    }
                    Err(compress_tree) => {
                        self.rs[u].sum = self.cx.collapse_path(
                            &self.cs[compress_tree].sum,
                            &self.weights[self.cs[compress_tree].ends[1] as usize],
                        );
                    }
                }
            }
        }

        unsafe fn update_virtual_parent_link(&mut self, u: NodeRef<node::Rake<Cx>>) {
            if let node::Parent::Compress(cu) = self.rs[u].parent {
                self.cs[cu].pivot.as_mut().unwrap_unchecked().rake_tree = Some(u);
            }
        }
    }

    impl<Cx: ClusterCx> TopTree<Cx> {
        unsafe fn update_boundary_handles(&mut self, u: NodeRef<node::Compress<Cx>>) {
            match self.cs[u].parent {
                Some(node::Parent::Compress(_)) => {}
                Some(node::Parent::Rake(_)) => {
                    self.handle[self.cs[u].ends[1] as usize] = Some(u);
                }
                None => {
                    self.handle[self.cs[u].ends[0] as usize] = Some(u);
                    self.handle[self.cs[u].ends[1] as usize] = Some(u);
                }
            }
        }
    }

    #[derive(Copy, Clone, PartialEq, Eq, Debug)]
    pub enum SoftExposeType {
        Vertex,
        NonEmpty,
    }

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct NoEdgeError;

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct DisconnError;

    impl<Cx: ClusterCx> TopTree<Cx> {
        pub fn new(weights: impl IntoIterator<Item = Cx::V>, cx: Cx) -> Self {
            let weights: Vec<_> = weights.into_iter().collect();
            let n_verts = weights.len();
            Self {
                cx,

                cs: Pool::default(),
                rs: Pool::default(),

                n_verts,
                weights,
                handle: vec![None; n_verts],
            }
        }

        /// Change the partition of the paths.
        ///
        /// ## Diagram
        ///
        /// ### Represented tree
        /// Swap the right-side chain `(c1)` with the raked chain `(cu)`, along the
        /// pivot vertex `v`.
        /// ┌─────────────────┐     ┌─────────────────┐
        /// │v0-(c0)-v-(c1)-v1│     │v0-(c0)-v-(cu)-vu│
        /// │        .        │     │        .        │
        /// │        .        │     │        .        │
        /// │       (cu)      │ ==> │       (c1)      │
        /// │        |        │     │        |        │
        /// │        vu       │     │        v1       │
        /// └─────────────────┘     └─────────────────┘
        ///
        /// ### Internal tree
        /// Splay the collapsed rake-node(`ru`), and then swap the rhs of
        /// the given compress-node(`c1`) with a collapsed rake-node(`cu`).
        /// Since the splay operation handles only branch nodes, we need a multiple
        /// splay pass to bring a leaf closest to the root, with [distance to the rake root] = 0, 1, or 2.
        ///
        /// - Case dist = 1. `parent(ru)` is the root of the rake tree.
        /// ┌───────┐     ┌────────┐
        /// │  cv   │     │  cv    │
        /// │ /| \  │     │ /| \   │
        /// │? rp c1│     │? rp cu │
        /// │ /  \  │     │ /  \   │
        /// │?    ru│ ==> │?    ru │
        /// │     | │     │     |  │
        /// │     cu│     │     c1 │
        /// └───────┘     └────────┘
        ///
        /// ## Constraints
        /// `[cu, ..r_path, cv]` must form an unward path.
        /// Here are the explicit verifications:
        /// ```
        /// debug_assert!(self.cs[cu].parent == Some(node::Parent::Rake(r_path[0])));
        /// debug_assert!(
        ///     (1..N - 1).all(|i| self.rs[r_path[i - 1]].parent == node::Parent::Rake(r_path[i])),
        /// );
        /// debug_assert!(self.rs[r_path[N - 1]].parent == node::Parent::Compress(cv));
        /// ```
        unsafe fn splice<const N: usize>(
            &mut self,
            guard: Option<NodeRef<node::Compress<Cx>>>,
            cu: &mut NodeRef<node::Compress<Cx>>,
            r_path: [NodeRef<node::Rake<Cx>>; N],
            cv: NodeRef<node::Compress<Cx>>,
        ) {
            assert!(1 <= N && N <= 3);
            self.guarded_splay(cv, guard);

            // Flip guard if necessary, ensuring it's not spliced off from the root path.
            if let Some(g) = guard {
                self.push_down(g);
                if self.branch(cv) == Some((g, 0)) {
                    debug_assert!(self.cs[g].is_internal_root());

                    self.cs[g].reverse(&self.cx);
                    self.push_down(g);
                    debug_assert!(self.branch(cv) == Some((g, 1)));
                }
            }

            self.push_down(cv);
            for &r in r_path.iter().rev() {
                self.push_down(r);
            }

            // Swap path
            let ru = r_path[0];
            let c1 = self.cs[cv].pivot.as_ref().unwrap_unchecked().children[1];

            self.rs[ru].children = Err(c1);
            self.cs[cv].pivot.as_mut().unwrap_unchecked().children[1] = *cu;
            self.cs[*cu].parent = Some(node::Parent::Compress(cv));
            self.cs[c1].parent = Some(node::Parent::Rake(ru));

            self.update_boundary_handles(c1);
            for &r in &r_path {
                self.pull_up(r);
            }
            self.pull_up(cv);

            *cu = cv;
        }

        /// Connect all chains between the root and u, then splay handle(u).
        /// This is equivalent to the access operation in a link-cut tree.
        /// If there is a guard (which should be a root internal node), ensure that it is not
        /// spliced off by appropriately flipping the path.
        pub unsafe fn guarded_access(
            &mut self,
            u: usize,
            guard: Option<NodeRef<node::Compress<Cx>>>,
        ) {
            unsafe {
                let Some(mut cu) = self.handle[u] else {
                    // If u is an isolated vertex, do nothing.
                    return;
                };

                self.guarded_splay(cu, guard);

                while let Some(node::Parent::Rake(ru)) = self.cs[cu].parent {
                    let rp_old = self.rs[ru].internal_parent();
                    self.splay(ru);
                    if let Some(rp) = self.rs[ru].internal_parent() {
                        self.guarded_splay(rp, rp_old);
                    }

                    match self.rs[ru].parent {
                        node::Parent::Compress(cv) => self.splice(guard, &mut cu, [ru], cv),
                        node::Parent::Rake(rp1) => match self.rs[rp1].parent {
                            node::Parent::Compress(cv) => {
                                self.splice(guard, &mut cu, [ru, rp1], cv)
                            }
                            node::Parent::Rake(rp2) => match self.rs[rp2].parent {
                                node::Parent::Compress(cv) => {
                                    self.splice(guard, &mut cu, [ru, rp1, rp2], cv)
                                }
                                node::Parent::Rake(_) => unreachable_unchecked(),
                            },
                        },
                    }

                    self.guarded_splay(cu, guard);
                }

                self.guarded_splay(self.handle[u].unwrap_unchecked(), guard);
            }
        }

        pub fn access(&mut self, u: usize) {
            unsafe { self.guarded_access(u, None) };
        }

        /// Set `handle(u)` as the internal root,
        /// ensuring that the root path includes both `u` and `v` (if they are connected).
        /// Flip the root path if necessary to position `u` on the left side of `v`.
        /// - If either `u` or `v` is a boundary vertex, set `handle(u) = handle(v) = [internal root]`.
        /// - If both `u` and `v` are interior vertices, restructure the internal tree as
        ///   `[internal root] = handle(u) ->(right child) handle(v) ->(left child) path(u ~ v)`.
        pub fn soft_expose(&mut self, u: usize, v: usize) -> Result<SoftExposeType, DisconnError> {
            unsafe {
                self.access(u);
                if u == v {
                    return Ok(SoftExposeType::Vertex);
                }

                if self.handle[u].is_none() || self.handle[v].is_none() {
                    self.access(v);
                    return Err(DisconnError);
                };

                let hu = self.handle[u].unwrap_unchecked();
                if u as u32 == self.cs[hu].ends[1] || v as u32 == self.cs[hu].ends[0] {
                    self.cs[hu].reverse(&self.cx);
                }

                if u as u32 == self.cs[hu].ends[0] {
                    self.access(v);

                    let hv = self.handle[v].unwrap_unchecked();
                    if u as u32 != self.cs[hv].ends[0] {
                        return Err(DisconnError);
                    }
                    return Ok(SoftExposeType::NonEmpty);
                }

                self.guarded_access(v, self.handle[u]);

                let hu = self.handle[u].unwrap_unchecked();
                self.push_down(hu);
                self.pull_up(hu);

                let hv = self.handle[v].unwrap_unchecked();
                if hu == hv {
                    return Ok(SoftExposeType::NonEmpty);
                }

                match self.branch(hv) {
                    Some((p, 0)) if p == hu => self.cs[hu].reverse(&self.cx),
                    Some((p, 1)) if p == hu => {}
                    _ => return Err(DisconnError),
                }
                Ok(SoftExposeType::NonEmpty)
            }
        }

        pub fn is_connected(&mut self, u: usize, v: usize) -> bool {
            self.soft_expose(u, v).is_ok()
        }

        unsafe fn link_right_path(&mut self, u: usize, ce: &mut NodeRef<node::Compress<Cx>>) {
            debug_assert!(u == self.cs[*ce].ends[0] as usize);

            if let Some(cu) = self.handle[u] {
                debug_assert!(self.cs[cu].parent == None);

                if u as u32 == self.cs[cu].ends[0] {
                    self.cs[cu].reverse(&self.cx);
                }

                if u as u32 == self.cs[cu].ends[1] {
                    // Case `degree(u)` = 1 (boundary vertex):
                    // Insert the new edge to the right end of the path.
                    let cp = self.cs.alloc(node::Compress {
                        pivot: Some(node::CompressPivot {
                            children: [cu, *ce],
                            ..unsafe { node::CompressPivot::uninit() }
                        }),
                        ..unsafe { node::Compress::uninit() }
                    });

                    self.cs[cu].parent = Some(node::Parent::Compress(cp));
                    self.cs[*ce].parent = Some(node::Parent::Compress(cp));
                    self.pull_up(cp);

                    *ce = cp;
                } else {
                    // Case `degree(u)` >= 2 (interior vertex):
                    // Remove the right path `c1` and replace it with the new edge `ce`,
                    // then insert it into the rake tree.
                    self.push_down(cu);

                    fn lifetime_hint<A, B, F: Fn(&mut A) -> &mut B>(f: F) -> F {
                        f
                    }
                    let cu_pivot = lifetime_hint(|this: &mut Self| {
                        this.cs[cu].pivot.as_mut().unwrap_unchecked()
                    });

                    let c1 = std::mem::replace(&mut cu_pivot(self).children[1], *ce);
                    self.cs[*ce].parent = Some(node::Parent::Compress(cu));

                    let r1 = self.rs.alloc(node::Rake {
                        children: Err(c1),
                        ..unsafe { node::Rake::uninit() }
                    });
                    self.cs[c1].parent = Some(node::Parent::Rake(r1));
                    self.update_boundary_handles(c1);
                    self.pull_up(r1);

                    if let Some(r0) = cu_pivot(self).rake_tree {
                        let rp = self.rs.alloc(node::Rake {
                            children: Ok([r0, r1]),
                            ..unsafe { node::Rake::uninit() }
                        });
                        self.rs[r0].parent = node::Parent::Rake(rp);
                        self.rs[r1].parent = node::Parent::Rake(rp);
                        self.pull_up(rp);

                        cu_pivot(self).rake_tree = Some(rp);
                        self.rs[rp].parent = node::Parent::Compress(cu);
                        self.pull_up(cu);
                    } else {
                        cu_pivot(self).rake_tree = Some(r1);
                        self.rs[r1].parent = node::Parent::Compress(cu);
                        self.pull_up(cu);
                    }

                    *ce = cu;
                }
            } else {
                // Case `degree(u)` = 0 (isolated vertex):
                self.handle[u] = Some(*ce);
            }
        }

        pub fn link(&mut self, u: usize, v: usize, e: Cx::C) -> bool {
            if self.soft_expose(u, v).is_ok() {
                return false;
            }

            let mut ce = self.cs.alloc(node::Compress {
                ends: [u as u32, v as u32],
                sum: e,
                ..unsafe { node::Compress::uninit() }
            });

            unsafe {
                let hu = self.handle[u];
                self.cs[ce].reverse(&self.cx);
                self.link_right_path(v, &mut ce);
                self.cs[ce].reverse(&self.cx);
                self.handle[u] = hu;

                self.link_right_path(u, &mut ce);
            }

            true
        }

        unsafe fn cut_right_path(&mut self, u: usize) -> NodeRef<node::Compress<Cx>> {
            let cu = self.handle[u].unwrap_unchecked();
            debug_assert!(u as u32 != self.cs[cu].ends[1]);

            if u as u32 == self.cs[cu].ends[0] {
                // Case `degree(u)` = 1 (boundary vertex)
                self.handle[u] = None;
                cu
            } else {
                // Case `degree(u)` >= 2 (interior vertex):
                // Split the right path `c1` from `cu`, and attempt to find a replacement path in the rake tree.
                self.push_down(cu);

                fn lifetime_hint<A, B, F: Fn(&mut A) -> &mut B>(f: F) -> F {
                    f
                }
                let cu_pivot =
                    lifetime_hint(|this: &mut Self| this.cs[cu].pivot.as_mut().unwrap_unchecked());

                let [c0, c1] = cu_pivot(self).children;
                self.cs[c1].parent = None;
                self.update_boundary_handles(c1);

                if let Some(r0) = cu_pivot(self).rake_tree {
                    match self.rs[r0].children {
                        Err(cr) => {
                            self.push_down(r0);

                            self.cs[cr].parent = Some(node::Parent::Compress(cu));
                            cu_pivot(self).rake_tree = None;
                            cu_pivot(self).children[1] = cr;

                            self.rs.mark_free(r0);

                            self.pull_up(cu);
                        }
                        Ok(_) => {
                            let rr = self.splay_last(r0);

                            let node::Rake {
                                parent: node::Parent::Rake(rp),
                                children: Err(cr),
                                ..
                            } = self.rs[rr]
                            else {
                                unreachable_unchecked()
                            };
                            let r0 = self.rs[rp].children.unwrap_unchecked()[0];
                            debug_assert!(self.rs[rp].parent == node::Parent::Compress(cu));
                            debug_assert!(self.rs[rp].children.unwrap_unchecked()[1] == rr);

                            self.push_down(rp);
                            self.push_down(rr);

                            self.rs[r0].parent = node::Parent::Compress(cu);
                            self.cs[cr].parent = Some(node::Parent::Compress(cu));
                            cu_pivot(self).rake_tree = Some(r0);
                            cu_pivot(self).children[1] = cr;

                            self.rs.mark_free(rp);
                            self.rs.mark_free(rr);

                            self.pull_up(cu);
                        }
                    };
                } else {
                    // If the rake tree is empty, the vertex `u` becomes a boundary vertex, and `cu` is freed.
                    self.cs[c0].parent = None;
                    self.update_boundary_handles(c0);
                    self.cs.mark_free(cu);
                }

                c1
            }
        }

        pub fn cut(&mut self, u: usize, v: usize) -> Result<Cx::C, NoEdgeError> {
            if self.soft_expose(u, v) != Ok(SoftExposeType::NonEmpty) {
                return Err(NoEdgeError);
            }

            let hu = self.handle[u].unwrap();
            let hv = self.handle[v].unwrap();
            let ends = self.cs[hu].ends;

            unsafe {
                let he = match (ends[0] == u as u32, ends[1] == v as u32) {
                    (true, true) => hu,
                    (true, false) => {
                        self.push_down(hu);
                        self.cs[hu].pivot.as_ref().unwrap_unchecked().children[0]
                    }
                    (false, true) => {
                        self.push_down(hu);
                        self.cs[hu].pivot.as_ref().unwrap_unchecked().children[1]
                    }
                    (false, false) => {
                        self.push_down(hu);
                        self.push_down(hv);
                        self.cs[hv].pivot.as_ref().unwrap_unchecked().children[0]
                    }
                };
                if self.cs[he].pivot.is_some() {
                    return Err(NoEdgeError);
                }

                let h_rest = self.cut_right_path(u);

                let hu = self.handle[u];
                self.cs[h_rest].reverse(&self.cx);
                self.cut_right_path(v);
                self.cs[he].reverse(&self.cx);
                self.handle[u] = hu;

                let c = std::mem::replace(&mut self.cs[he].sum, Cx::id_compress());
                self.pool().mark_free(he);
                Ok(c)
            }
        }

        pub fn get_vertex(&mut self, u: usize) -> &Cx::V {
            self.access(u);
            if let Some(hu) = self.handle[u] {
                self.push_down(hu);
            }
            &self.weights[u]
        }

        pub fn modify_vertex(&mut self, u: usize, update_with: impl FnOnce(&mut Cx::V)) {
            self.access(u);
            if let Some(hu) = self.handle[u] {
                self.push_down(hu);
                update_with(&mut self.weights[u]);
                self.pull_up(hu);
            } else {
                update_with(&mut self.weights[u])
            }
        }

        pub fn modify_edge(
            &mut self,
            u: usize,
            v: usize,
            update_with: impl FnOnce(&mut Cx::C),
        ) -> Result<(), NoEdgeError> {
            let mut w = self.cut(u, v)?;
            update_with(&mut w);
            self.link(u, v, w);
            Ok(())
        }

        pub fn sum_path(
            &mut self,
            u: usize,
            v: usize,
        ) -> Result<(&Cx::V, Option<(&Cx::C, &Cx::V)>), DisconnError> {
            let pseudo_rake = match self.soft_expose(u, v)? {
                SoftExposeType::Vertex => None,
                SoftExposeType::NonEmpty => {
                    let hu = self.handle[u].unwrap();
                    let hv = self.handle[v].unwrap();
                    let ends = self.cs[hu].ends;

                    let h_inner = match (ends[0] as usize == u, ends[1] as usize == v) {
                        (true, true) => hu,
                        (true, false) => {
                            self.push_down(hu);
                            self.cs[hu].pivot.as_ref().unwrap().children[0]
                        }
                        (false, true) => {
                            self.push_down(hu);
                            self.cs[hu].pivot.as_ref().unwrap().children[1]
                        }
                        (false, false) => {
                            self.push_down(hu);
                            self.push_down(hv);
                            self.cs[hv].pivot.as_ref().unwrap().children[0]
                        }
                    };
                    Some((&self.cs[h_inner].sum, &self.weights[v]))
                }
            };
            Ok((&self.weights[u], pseudo_rake))
        }

        pub fn apply_path(
            &mut self,
            u: usize,
            v: usize,
            mut action: impl Action<Cx>,
        ) -> Result<(), DisconnError> {
            match self.soft_expose(u, v)? {
                SoftExposeType::Vertex => {
                    action.apply_to_weight(&mut self.weights[u]);
                }
                SoftExposeType::NonEmpty => {
                    let hu = self.handle[u].unwrap();
                    let hv = self.handle[v].unwrap();
                    let ends = self.cs[hu].ends;

                    let mut update_with = |this: &mut Self, h_inner| {
                        action.apply_to_weight(&mut this.weights[u]);
                        action.apply_to_weight(&mut this.weights[v]);
                        action.apply_to_compress(&mut this.cs[h_inner].sum, ActionRange::Path);
                    };
                    match (ends[0] as usize == u, ends[1] as usize == v) {
                        (true, true) => update_with(self, hu),
                        (true, false) => {
                            self.push_down(hu);
                            update_with(self, self.cs[hu].pivot.as_ref().unwrap().children[0]);
                            self.pull_up(hu);
                        }
                        (false, true) => {
                            self.push_down(hu);
                            update_with(self, self.cs[hu].pivot.as_ref().unwrap().children[1]);
                            self.pull_up(hu);
                        }
                        (false, false) => {
                            self.push_down(hu);
                            self.push_down(hv);
                            update_with(self, self.cs[hv].pivot.as_ref().unwrap().children[0]);
                            self.pull_up(hv);
                            self.pull_up(hu);
                        }
                    };
                }
            };
            Ok(())
        }

        /// A wrapper for `sum_path`. To handle both path and subtree sums,
        /// store both aggregates (and lazy tags) in `Cx::C` and `Cx::R`.
        pub fn sum_subtree(
            &mut self,
            root: usize,
            u: usize,
        ) -> Result<(&Cx::V, Option<Cx::R>), DisconnError> {
            unsafe {
                match self.soft_expose(root, u)? {
                    SoftExposeType::Vertex => {
                        if let Some(hr) = self.handle[root] {
                            self.push_down(hr);

                            let ends = self.cs[hr].ends;
                            let v0 = &self.weights[ends[0] as usize];
                            let v1 = &self.weights[ends[1] as usize];
                            println!("se {:?}", ends);
                            if ends[0] == root as u32 {
                                Ok((v0, Some(self.cx.collapse_path(&self.cs[hr].sum, v1))))
                            } else if ends[1] == root as u32 {
                                let rev_path = self.cx.reverse(&self.cs[hr].sum);
                                Ok((v1, Some(self.cx.collapse_path(&rev_path, v0))))
                            } else {
                                let pivot = self.cs[hr].pivot.as_ref().unwrap_unchecked();
                                let [c0, c1] = pivot.children;
                                let v_pivot = &self.weights[self.cs[c0].ends[1] as usize];

                                let prefix = &self.cs[c0].sum;
                                let suffix = &self.cs[c1].sum;
                                let mut rest = self.cx.rake([
                                    &self.cx.collapse_path(&self.cx.reverse(&prefix), v0),
                                    &self.cx.collapse_path(suffix, v1),
                                ]);
                                if let Some(rake_tree) = pivot.rake_tree {
                                    rest = self.cx.rake([&rest, &self.rs[rake_tree].sum]);
                                }

                                Ok((v_pivot, Some(rest)))
                            }
                        } else {
                            let v0 = &self.weights[root];
                            Ok((v0, None))
                        }
                    }
                    SoftExposeType::NonEmpty => {
                        let hr = self.handle[root].unwrap_unchecked();
                        let hu = self.handle[u].unwrap_unchecked();
                        let ends = self.cs[hr].ends;

                        if ends[1] == u as u32 {
                            let v0 = &self.weights[u];
                            Ok((&v0, None))
                        } else {
                            self.push_down(hr);
                            if ends[0] != root as u32 {
                                self.push_down(hu);
                            }

                            let v0 = &self.weights[u];
                            let v1 = &self.weights[ends[1] as usize];

                            let pivot = self.cs[hu].pivot.as_ref().unwrap_unchecked();
                            let h_suffix = pivot.children[1];
                            let mut rest = self.cx.collapse_path(&self.cs[h_suffix].sum, v1);

                            if let Some(rake_tree) = pivot.rake_tree {
                                rest = self.cx.rake([&rest, &self.rs[rake_tree].sum]);
                            }
                            Ok((v0, Some(rest)))
                        }
                    }
                }
            }
        }

        pub fn sum_rerooted(&mut self, u: usize) -> (&Cx::V, Option<Cx::R>) {
            unsafe { self.sum_subtree(u, u).unwrap_unchecked() }
        }

        /// Find the center edge of a connected component by recursively narrowing the cluster.
        pub fn center_edge(
            &mut self,
            root: usize,
            mut select: impl FnMut(&Cx::V, &Cx::R, &Cx::R) -> bool,
        ) -> Option<[usize; 2]> {
            self.access(root);
            let mut h = self.handle[root]?;

            let v0 = self.weights[self.cs[h].ends[0] as usize].clone();
            let mut v1 = self.weights[self.cs[h].ends[1] as usize].clone();
            let mut s0_rev: Option<(Option<Cx::R>, Cx::C)> = None;
            let mut s1: Option<(Option<Cx::R>, Cx::C)> = None;

            let edge = loop {
                self.push_down(h);
                let Some(pivot) = &self.cs[h].pivot else {
                    break self.cs[h].ends.map(|u| u as usize);
                };

                // Perform a binary/ternary traversal depending on the presence of a rake tree.
                let [c0, c1] = pivot.children;
                let v_pivot = self.weights[self.cs[c0].ends[1] as usize].clone();
                let v00 = &self.weights[self.cs[c0].ends[0] as usize];
                let v11 = &self.weights[self.cs[c1].ends[1] as usize];

                let d0_rev = self.cx.reverse(&self.cs[c0].sum);
                let d1 = &self.cs[c1].sum;

                let s0_rev_next = if let Some(s0_rev) = &s0_rev {
                    self.cx
                        .compress([&d0_rev, &s0_rev.1], &v00, s0_rev.0.as_ref())
                } else {
                    d0_rev
                };
                let s1_next = if let Some(s1) = &s1 {
                    self.cx.compress([d1, &s1.1], &v11, s1.0.as_ref())
                } else {
                    d1.clone()
                };

                let p0 = self.cx.collapse_path(&s0_rev_next, &v0);
                let p1 = self.cx.collapse_path(&s1_next, &v1);

                let branch = select(&v_pivot, &p0, &p1) as usize;
                match pivot.rake_tree {
                    Some(hr) if select(&v_pivot, [&p0, &p1][branch], &self.rs[hr].sum) => {
                        let (h_next, mut r_exclusive) =
                            self.center_edge_in_rake_tree(&v_pivot, hr, &mut select);
                        h = h_next;
                        r_exclusive = self.cx.rake([&r_exclusive, &p1]);

                        v1 = self.weights[self.cs[h].ends[1] as usize].clone();
                        s0_rev = Some((Some(r_exclusive), s0_rev_next));
                        s1 = None;
                    }
                    _ => {
                        h = [c0, c1][branch];

                        let r = pivot.rake_tree.map(|hr| self.rs[hr].sum.clone());
                        if branch == 0 {
                            s1 = Some((r, s1_next));
                        } else {
                            s0_rev = Some((r, s0_rev_next));
                        }
                    }
                };
            };

            self.access(edge[0]);
            Some(edge)
        }

        pub fn center_edge_in_path(
            &mut self,
            u: usize,
            v: usize,
            mut select: impl FnMut(&Cx::V, &Cx::R, &Cx::R) -> bool,
        ) -> Result<Option<[usize; 2]>, DisconnError> {
            self.soft_expose(u, v)?;
            let mut h = match self.handle[u] {
                Some(h) => h,
                None => return Ok(None),
            };

            let v0 = self.weights[self.cs[h].ends[0] as usize].clone();
            let v1 = self.weights[self.cs[h].ends[1] as usize].clone();
            let mut s0_rev: Option<(Option<Cx::R>, Cx::C)> = None;
            let mut s1: Option<(Option<Cx::R>, Cx::C)> = None;

            let edge = loop {
                self.push_down(h);
                let Some(pivot) = &self.cs[h].pivot else {
                    break self.cs[h].ends.map(|u| u as usize);
                };

                // Perform a binary walk.
                let [c0, c1] = pivot.children;
                let v_pivot = self.weights[self.cs[c0].ends[1] as usize].clone();
                let v00 = &self.weights[self.cs[c0].ends[0] as usize];
                let v11 = &self.weights[self.cs[c1].ends[1] as usize];

                let d0_rev = self.cx.reverse(&self.cs[c0].sum);
                let d1 = &self.cs[c1].sum;

                let s0_rev_next = if let Some(s0_rev) = &s0_rev {
                    self.cx
                        .compress([&d0_rev, &s0_rev.1], &v00, s0_rev.0.as_ref())
                } else {
                    d0_rev
                };
                let s1_next = if let Some(s1) = &s1 {
                    self.cx.compress([d1, &s1.1], &v11, s1.0.as_ref())
                } else {
                    d1.clone()
                };

                let p0 = self.cx.collapse_path(&s0_rev_next, &v0);
                let p1 = self.cx.collapse_path(&s1_next, &v1);

                let branch = select(&v_pivot, &p0, &p1) as usize;

                h = [c0, c1][branch];
                let r = pivot.rake_tree.map(|hr| self.rs[hr].sum.clone());
                if branch == 0 {
                    s1 = Some((r, s1_next));
                } else {
                    s0_rev = Some((r, s0_rev_next));
                }
            };

            self.access(edge[0]);
            Ok(Some(edge))
        }

        fn center_edge_in_rake_tree(
            &mut self,
            v_pivot: &Cx::V,
            mut h: NodeRef<node::Rake<Cx>>,
            select: &mut impl FnMut(&Cx::V, &Cx::R, &Cx::R) -> bool,
        ) -> (NodeRef<node::Compress<Cx>>, Cx::R) {
            let mut r_exclusive = Cx::id_rake();
            loop {
                // Perform a binary walk.
                self.push_down(h);
                match self.rs[h].children {
                    Ok(rs) => {
                        let ps = rs.map(|r| &self.rs[r].sum);
                        let branch = select(&v_pivot, &ps[0], &ps[1]) as usize;
                        h = rs[branch];
                        r_exclusive = self.cx.rake([&r_exclusive, &ps[branch ^ 1]]);
                    }
                    Err(hc) => return (hc, r_exclusive),
                }
            }
        }
    }
}

type ModP = num_mod::ModInt<num_mod::ByU64<1_000_000_007>>;
const INV2: u64 = 500_000_004;

const UNSET: i32 = -1;

#[derive(Debug, Clone, Copy, Default)]
struct Path {
    depth: u32,
    terminal: i32,

    c: ModP,
    h: ModP,
}

#[derive(Debug, Clone, Copy, Default)]
struct Compress {
    left: Path,
    right: Path,
    len: u32,
    diameter: (u32, [i32; 2]),
}

#[derive(Debug, Clone, Copy, Default)]
struct Rake {
    left: Path,

    c2: ModP,
    hc: ModP,
    hc2: ModP,

    diameter: (u32, [i32; 2]),
}

impl Path {
    fn prepend_depth(&self, len: u32) -> Self {
        Self {
            depth: self.depth + len,
            ..self.clone()
        }
    }
    fn prepend_h(&self, len: u32) -> Self {
        Self {
            h: self.h + ModP::from(len) * self.c * self.c,
            ..self.clone()
        }
    }

    fn rake_nonnull(&self, other: &Self) -> Self {
        match self.depth.cmp(&other.depth) {
            Ordering::Less => other.clone(),
            Ordering::Greater => self.clone(),
            Ordering::Equal => Self {
                depth: self.depth,
                terminal: self.terminal,
                c: self.c + other.c,
                h: self.h + other.h,
            },
        }
    }

    fn rake(&self, rake: Option<&Rake>) -> Path {
        let mut res = self.clone();
        if let Some(r) = rake {
            res = res.rake_nonnull(&r.left);
        }
        res
    }

    fn unit() -> Self {
        Self {
            depth: 1,
            terminal: UNSET,
            c: 1u8.into(),
            h: 1u8.into(),
        }
    }
}

impl Compress {
    fn unit() -> Self {
        Self {
            left: Path::unit(),
            right: Path::unit(),
            len: 1,
            diameter: (1, [UNSET; 2]),
        }
    }
}

impl Rake {
    fn collapse_at_point(&self) -> ModP {
        let Self {
            left: Path { c, h, .. },
            c2,
            hc,
            hc2,
            ..
        } = self.clone();

        dbg!(c, c2, h, hc, hc2);

        let n_diam = (c * c - c2) * ModP::from(INV2);
        h * c * c - ModP::from(2u8) * hc * c + hc2 + n_diam * n_diam
    }

    fn collapse_at_edge(&self, other: &Self) -> ModP {
        let Path { c: cl, h: hl, .. } = self.left.clone();
        let Path { c: cr, h: hr, .. } = other.left.clone();
        let cl2 = cl * cl;
        let cr2 = cr * cr;

        let c = cl + cr;
        let c2 = cl2 + cr2;
        let h = hl + hr;
        let hc = hl * cl + hr * cr;
        let hc2 = hl * cl2 + hr * cr2;

        let n_diam = (c * c - c2) * ModP::from(INV2);
        h * c * c - ModP::from(2u8) * hc * c + hc2 + ModP::from(2u8) * n_diam * n_diam
    }
}

struct PairDiameter;

impl ClusterCx for PairDiameter {
    type V = i32;
    type C = Compress;
    type R = Rake;

    fn id_compress() -> Self::C {
        Default::default()
    }

    fn compress(&self, cs: [&Self::C; 2], _v: &Self::V, r: Option<&Self::R>) -> Self::C {
        let mut diameter = (cs[0].diameter).max(cs[1].diameter).max((
            cs[0].right.depth + cs[1].left.depth,
            [cs[0].right.terminal, cs[1].left.terminal],
        ));
        if let Some(rake) = r {
            diameter = diameter
                .max(rake.diameter)
                .max((
                    cs[0].right.depth + rake.left.depth,
                    [cs[0].right.terminal, rake.left.terminal],
                ))
                .max((
                    rake.left.depth + cs[1].left.depth,
                    [rake.left.terminal, cs[1].left.terminal],
                ));
        }

        Self::C {
            left: (cs[1].left)
                .rake(r)
                .prepend_depth(cs[0].len)
                .prepend_h(cs[0].len)
                .rake_nonnull(&cs[0].left), // ppr to compress
            right: (cs[0].right)
                .rake(r)
                .prepend_depth(cs[1].len)
                .prepend_h(cs[1].len)
                .rake_nonnull(&cs[1].right),
            len: cs[0].len + cs[1].len,
            diameter,
        }
    }

    fn id_rake() -> Self::R {
        Default::default()
    }

    fn rake(&self, rs: [&Self::R; 2]) -> Self::R {
        let ps = [&rs[0].left, &rs[1].left];
        let diameter = (rs[0].diameter)
            .max(rs[1].diameter)
            .max((ps[0].depth + ps[1].depth, [ps[0].terminal, ps[1].terminal]));
        match ps[0].depth.cmp(&ps[1].depth) {
            Ordering::Less => Self::R {
                diameter,
                ..rs[1].clone()
            },
            Ordering::Greater => Self::R {
                diameter,
                ..rs[0].clone()
            },
            Ordering::Equal => Self::R {
                left: Path {
                    depth: ps[0].depth,
                    terminal: ps[0].terminal,
                    c: ps[0].c + ps[1].c,
                    h: ps[0].h + ps[1].h,
                },
                c2: rs[0].c2 + rs[1].c2,
                hc: rs[0].hc + rs[1].hc,
                hc2: rs[0].hc2 + rs[1].hc2,
                diameter,
            },
        }
    }

    fn collapse_path(&self, compress: &Self::C, &vr: &Self::V) -> Self::R {
        let c = compress.left.c;
        let c2 = c * c;
        let h = compress.left.h;

        let mut diameter = compress.diameter;
        if diameter.1[1] == UNSET {
            diameter.1[1] = vr;
        }

        Self::R {
            left: Path {
                depth: compress.left.depth,
                terminal: if compress.left.terminal == UNSET {
                    vr
                } else {
                    compress.left.terminal
                },
                c,
                h,
            },
            c2,
            hc: h * c,
            hc2: h * c2,

            diameter,
        }
    }

    fn reverse(&self, c: &Self::C) -> Self::C {
        let mut diameter = c.diameter;
        diameter.1.swap(0, 1);
        Self::C {
            left: c.right,
            right: c.left,
            diameter,
            ..c.clone()
        }
    }
}

pub fn main() {
    let mut input = fast_io::stdin();
    let mut output = fast_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let q: usize = input.value();

    let mut tt = top_tree::TopTree::new(0..n as i32, PairDiameter);
    for _ in 0..m {
        let u = input.u32() as usize - 1;
        let v = input.u32() as usize - 1;
        tt.link(u, v, Compress::unit());
    }

    for _ in 0..q {
        match input.token() {
            "1" => {
                let u = input.u32() as usize - 1;
                let v = input.u32() as usize - 1;
                tt.link(u, v, Compress::unit());
            }
            "2" => {
                let u = input.u32() as usize - 1;
                let v = input.u32() as usize - 1;
                tt.cut(u, v).unwrap();
            }
            "3" => {
                let u = input.u32() as usize - 1;
                let (&top, rest) = tt.sum_rerooted(u);
                let mut ans = 1u8.into();

                if let Some(rest) = rest {
                    let (diam, mut ends) = rest.diameter;
                    if ends[0] == UNSET {
                        ends[0] = top;
                    }
                    // println!("{:?} {:?}", top, rest);

                    let e = tt
                        .center_edge_in_path(ends[0] as usize, ends[1] as usize, |_v, r0, r1| {
                            r0.left.depth < r1.left.depth
                        })
                        .unwrap()
                        .unwrap();
                    if diam % 2 == 0 {
                        let r = tt
                            .sum_path(ends[0] as usize, e[1] as usize)
                            .unwrap()
                            .1
                            .map_or(0, |(c, _)| c.right.depth);
                        let v = e[(r * 2 == diam) as usize];
                        println!("{:?}", (ends, e, r, v));
                        ans = tt.sum_rerooted(v).1.unwrap().collapse_at_point();
                    } else {
                        let s1 = tt.sum_subtree(e[0] as usize, e[1] as usize).unwrap().1;
                        let s2 = tt.sum_subtree(e[1] as usize, e[0] as usize).unwrap().1;
                        ans = match (s1, s2) {
                            (Some(s1), Some(s2)) => s1.collapse_at_edge(&s2),
                            (None, None) => 2u8.into(),
                            _ => panic!(),
                        };
                    }
                }
                writeln!(output, "{}", u64::from(ans)).unwrap();
            }
            _ => panic!(),
        }
    }
}
