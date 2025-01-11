use std::io::Write;

use num_mod_static::montgomery::MontgomeryU32;
use segtree_lazy::{MonoidAction, SegTree};

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

pub mod num_mod_static {
    use std::ops::*;

    pub trait Unsigned:
        Copy
        + Default
        + Add<Output = Self>
        + AddAssign
        + Sub<Output = Self>
        + SubAssign
        + Mul<Output = Self>
        + MulAssign
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

    pub trait CommRing:
        Add<Output = Self>
        + AddAssign
        + Sub<Output = Self>
        + SubAssign
        + Neg<Output = Self>
        + Mul<Output = Self>
        + MulAssign
        + Default
        + Clone
    {
        fn zero() -> Self {
            Self::default()
        }
        fn one() -> Self;
    }

    pub trait PowBy<E> {
        fn pow(&self, exp: E) -> Self;
    }

    macro_rules! impl_powby {
        ($($exp:ty)+) => {
            $(
                impl<R: CommRing> PowBy<$exp> for R {
                    fn pow(&self, exp: $exp) -> R {
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
            )+
        };
    }
    impl_powby!(u16 u32 u64 u128);

    pub trait ModSpec: Copy {
        type U: Unsigned;
        const MODULUS: Self::U;
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
    impl_modspec!(ByU16 u16, ByU32 u32, ByU64 u64, ByU128 u128);

    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
    pub struct NaiveModInt<M: ModSpec>(M::U);

    impl<M: ModSpec> AddAssign for NaiveModInt<M> {
        fn add_assign(&mut self, rhs: Self) {
            self.0 += rhs.0;
            if self.0 >= M::MODULUS {
                self.0 -= M::MODULUS;
            }
        }
    }

    impl<M: ModSpec> Add for NaiveModInt<M> {
        type Output = Self;
        fn add(self, rhs: Self) -> Self {
            let mut res = self.clone();
            res += rhs;
            res
        }
    }

    impl<M: ModSpec> SubAssign for NaiveModInt<M> {
        fn sub_assign(&mut self, rhs: Self) {
            if self.0 < rhs.0 {
                self.0 += M::MODULUS;
            }
            self.0 -= rhs.0;
        }
    }

    impl<M: ModSpec> Sub for NaiveModInt<M> {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self {
            let mut res = self.clone();
            res -= rhs;
            res
        }
    }

    impl<M: ModSpec> Neg for NaiveModInt<M> {
        type Output = Self;
        fn neg(self) -> Self {
            let mut res = M::MODULUS - self.0;
            if res == M::MODULUS {
                res = 0.into();
            }
            Self(res)
        }
    }

    impl<M: ModSpec> MulAssign for NaiveModInt<M> {
        fn mul_assign(&mut self, rhs: Self) {
            self.0 = self.0 * rhs.0 % M::MODULUS;
        }
    }

    impl<M: ModSpec> Mul for NaiveModInt<M> {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self {
            let mut res = self.clone();
            res *= rhs;
            res
        }
    }

    impl<M: ModSpec> Default for NaiveModInt<M> {
        fn default() -> Self {
            Self(M::U::default())
        }
    }

    impl<M: ModSpec> CommRing for NaiveModInt<M> {
        fn one() -> Self {
            Self(1.into())
        }
    }

    macro_rules! impl_from {
        ($($u:ty)+) => {
            $(
                impl<M: ModSpec<U = $u>> From<$u> for NaiveModInt<M> {
                    fn from(n: $u) -> Self {
                        Self(n % M::MODULUS)
                    }
                }

                impl<M: ModSpec<U = $u>> From<NaiveModInt<M>> for $u {
                    fn from(n: NaiveModInt<M>) -> Self {
                        n.0
                    }
                }
            )+
        };
    }
    impl_from!(u16 u32 u64 u128);

    pub mod montgomery {
        use super::{CommRing, Unsigned};
        use std::ops::*;

        pub trait ModSpec: Copy {
            type U: Unsigned;
            type D: Unsigned;
            const MODULUS: Self::U;
            const M_INV: Self::U;
            const R2: Self::U;
            fn to_double(u: Self::U) -> Self::D;
            fn reduce_double(value: Self::D) -> Self::U;
        }

        macro_rules! impl_modspec {
            ($($t:ident, $t_impl:ident, U = $single:ty, D = $double:ty, EXP = $exp:expr, LOG2_EXP = $log2_exp: expr);+) => {
                $(
                    #[derive(Debug, Clone, Copy, PartialEq, Eq)]
                    pub struct $t<const M: $single>;

                    impl<const MOD: $single> ModSpec for $t<MOD> {
                        type U = $single;
                        type D = $double;
                        const MODULUS: $single = MOD;
                        const M_INV: $single = $t_impl::eval_m_inv(MOD);
                        const R2: $single = $t_impl::eval_r2(MOD);

                        fn to_double(u: Self::U) -> Self::D {
                            u as Self::D
                        }

                        fn reduce_double(x: Self::D) -> Self::U {
                            debug_assert!(x < (MOD as $double) * (MOD as $double));
                                let q = (x as $single).wrapping_mul(Self::M_INV);
                                let a = ((q as $double * Self::MODULUS as $double) >> $exp) as $single;
                                let mut res = (x >> $exp) as $single + Self::MODULUS - a;
                                if res >= Self::MODULUS {
                                    res -= Self::MODULUS;
                                }
                                res
                        }
                    }

                    mod $t_impl {
                        pub const fn eval_m_inv(m: $single) -> $single {
                            debug_assert!(m % 2 == 1, "modulus must be coprime with 2");
                            let mut m_inv: $single = 1;
                            let two: $single = 2;

                            let mut iter = 0;
                            while iter < $log2_exp {
                                m_inv = m_inv.wrapping_mul(two.wrapping_sub(m.wrapping_mul(m_inv)));
                                iter += 1;
                            }
                            m_inv
                        }

                        pub const fn eval_r2(m: $single) -> $single {
                            let r = m.wrapping_neg() % m;
                            (r as $double * r as $double % m as $double) as $single
                        }
                    }
                )+
            };
        }
        impl_modspec!(
            ByU32, u32_impl, U = u32, D = u64, EXP = 32, LOG2_EXP = 5;
            ByU64, u64_impl, U = u64, D = u128, EXP = 64, LOG2_EXP = 6
        );

        #[derive(Debug, Clone, Copy, PartialEq, Eq)]
        pub struct Montgomery<M: ModSpec>(M::U);

        impl<M: ModSpec> AddAssign for Montgomery<M> {
            fn add_assign(&mut self, rhs: Self) {
                self.0 += rhs.0;
                if self.0 >= M::MODULUS {
                    self.0 -= M::MODULUS;
                }
            }
        }

        impl<M: ModSpec> Add for Montgomery<M> {
            type Output = Self;
            fn add(self, rhs: Self) -> Self {
                let mut res = self.clone();
                res += rhs;
                res
            }
        }

        impl<M: ModSpec> SubAssign for Montgomery<M> {
            fn sub_assign(&mut self, rhs: Self) {
                if self.0 < rhs.0 {
                    self.0 += M::MODULUS;
                }
                self.0 -= rhs.0;
            }
        }

        impl<M: ModSpec> Sub for Montgomery<M> {
            type Output = Self;
            fn sub(self, rhs: Self) -> Self {
                let mut res = self.clone();
                res -= rhs;
                res
            }
        }

        impl<M: ModSpec> Neg for Montgomery<M> {
            type Output = Self;
            fn neg(self) -> Self {
                let mut res = M::MODULUS - self.0;
                if res == M::MODULUS {
                    res = 0.into();
                }
                Self(res)
            }
        }

        impl<M: ModSpec> MulAssign for Montgomery<M> {
            fn mul_assign(&mut self, rhs: Self) {
                self.0 = M::reduce_double(M::to_double(self.0) * M::to_double(rhs.0));
            }
        }

        impl<M: ModSpec> Mul for Montgomery<M> {
            type Output = Self;
            fn mul(self, rhs: Self) -> Self {
                let mut res = self.clone();
                res *= rhs;
                res
            }
        }

        impl<M: ModSpec> Default for Montgomery<M> {
            fn default() -> Self {
                Self(M::U::default())
            }
        }

        impl<M: ModSpec> CommRing for Montgomery<M> {
            fn one() -> Self {
                todo!()
            }
        }

        macro_rules! impl_from {
            ($($u:ty)+) => {
                $(
                    impl<M: ModSpec<U = $u>> From<$u> for Montgomery<M> {
                        fn from(x: $u) -> Self {
                            Self(x) * Self(M::R2)
                        }
                    }

                    impl<M: ModSpec<U = $u>> From<Montgomery<M>> for $u {
                        fn from(x: Montgomery<M>) -> Self {
                            M::reduce_double(M::to_double(x.0))
                        }
                    }
                )+
            };
        }
        impl_from!(u32 u64);

        pub type MontgomeryU32<const M: u32> = Montgomery<ByU32<M>>;
        pub type MontgomeryU64<const M: u64> = Montgomery<ByU64<M>>;
    }
}

pub mod segtree_lazy {
    use std::{iter, ops::Range};

    pub trait MonoidAction {
        type X;
        type F;
        const IS_X_COMMUTATIVE: bool = false; // TODO
        fn id(&self) -> Self::X;
        fn combine(&self, lhs: &Self::X, rhs: &Self::X) -> Self::X;
        fn id_action(&self) -> Self::F;
        fn combine_action(&self, lhs: &Self::F, rhs: &Self::F) -> Self::F;
        fn apply_to_sum(&self, f: &Self::F, x_count: u32, x_sum: &mut Self::X);
    }

    pub struct SegTree<M: MonoidAction> {
        n: usize,
        max_height: u32,
        sum: Vec<M::X>,
        lazy: Vec<M::F>,
        ma: M,
    }

    impl<M: MonoidAction> SegTree<M> {
        pub fn with_size(n: usize, ma: M) -> Self {
            Self {
                n,
                max_height: usize::BITS - n.leading_zeros(),
                sum: iter::repeat_with(|| ma.id()).take(2 * n).collect(),
                lazy: iter::repeat_with(|| ma.id_action()).take(n).collect(),
                ma,
            }
        }

        pub fn from_iter<I>(iter: I, ma: M) -> Self
        where
            I: IntoIterator<Item = M::X>,
            I::IntoIter: ExactSizeIterator,
        {
            let iter = iter.into_iter();
            let n = iter.len();
            let mut sum: Vec<_> = (iter::repeat_with(|| ma.id()).take(n))
                .chain(
                    iter.into_iter()
                        .chain(iter::repeat_with(|| ma.id()))
                        .take(n),
                )
                .collect();
            for i in (1..n).rev() {
                sum[i] = ma.combine(&sum[i << 1], &sum[i << 1 | 1]);
            }
            Self {
                n,
                max_height: usize::BITS - n.leading_zeros(),
                sum,
                lazy: iter::repeat_with(|| ma.id_action()).take(n).collect(),
                ma,
            }
        }

        fn apply(&mut self, idx: usize, width: u32, value: &M::F) {
            self.ma.apply_to_sum(&value, width, &mut self.sum[idx]);
            if idx < self.n {
                self.lazy[idx] = self.ma.combine_action(&value, &self.lazy[idx]);
            }
        }

        fn push_down(&mut self, width: u32, node: usize) {
            let value = unsafe { &*(&self.lazy[node] as *const _) };
            self.apply(node << 1, width, value);
            self.apply(node << 1 | 1, width, value);
            self.lazy[node] = self.ma.id_action();
        }

        fn push_range(&mut self, range: Range<usize>) {
            let Range { mut start, mut end } = range;
            start += self.n;
            end += self.n;

            let start_height = 1 + start.trailing_zeros();
            let end_height = 1 + end.trailing_zeros();
            for height in (start_height..=self.max_height).rev() {
                let width = 1 << height - 1;
                self.push_down(width, start >> height);
            }
            for height in (end_height..=self.max_height).rev().skip_while(|&height| {
                height >= start_height && end - 1 >> height == start >> height
            }) {
                let width = 1 << height - 1;
                self.push_down(width, end - 1 >> height);
            }
        }

        fn pull_up(&mut self, node: usize) {
            self.sum[node] = (self.ma).combine(&self.sum[node << 1], &self.sum[node << 1 | 1]);
        }

        pub fn apply_range(&mut self, range: Range<usize>, value: M::F) {
            let Range { mut start, mut end } = range;
            debug_assert!(start <= end && end <= self.n);
            if start == end {
                return;
            }

            self.push_range(range);
            start += self.n;
            end += self.n;
            let mut width: u32 = 1;
            let (mut pull_start, mut pull_end) = (false, false);
            while start < end {
                if pull_start {
                    self.pull_up(start - 1);
                }
                if pull_end {
                    self.pull_up(end);
                }
                if start & 1 != 0 {
                    self.apply(start, width, &value);
                    start += 1;
                    pull_start = true;
                }
                if end & 1 != 0 {
                    self.apply(end - 1, width, &value);
                    pull_end = true;
                }
                start >>= 1;
                end >>= 1;
                width <<= 1;
            }
            start -= 1;
            while end > 0 {
                if pull_start {
                    self.pull_up(start);
                }
                if pull_end && !(pull_start && start == end) {
                    self.pull_up(end);
                }
                start >>= 1;
                end >>= 1;
                width <<= 1;
            }
        }

        pub fn query_range(&mut self, range: Range<usize>) -> M::X {
            let Range { mut start, mut end } = range;

            self.push_range(range);
            start += self.n;
            end += self.n;
            if M::IS_X_COMMUTATIVE {
                let mut result = self.ma.id();
                while start < end {
                    if start & 1 != 0 {
                        result = self.ma.combine(&result, &self.sum[start]);
                        start += 1;
                    }
                    if end & 1 != 0 {
                        end -= 1;
                        result = self.ma.combine(&result, &self.sum[end]);
                    }
                    start >>= 1;
                    end >>= 1;
                }
                result
            } else {
                let (mut result_left, mut result_right) = (self.ma.id(), self.ma.id());
                while start < end {
                    if start & 1 != 0 {
                        result_left = self.ma.combine(&result_left, &self.sum[start]);
                    }
                    if end & 1 != 0 {
                        result_right = self.ma.combine(&self.sum[end - 1], &result_right);
                    }
                    start = (start + 1) >> 1;
                    end >>= 1;
                }
                self.ma.combine(&result_left, &result_right)
            }
        }

        pub fn query_all(&mut self) -> &M::X {
            assert!(self.n.is_power_of_two());
            self.push_down(self.n as u32, 1);
            &self.sum[1]
        }

        // The following two lines are equivalent.
        // partition_point(0, n, |i| pred(segtree.query_range(0..i+1)));
        // segtree.partition_point_prefix(|prefix| pred(prefix));
        pub fn partition_point_prefix(&mut self, mut pred: impl FnMut(&M::X) -> bool) -> usize {
            assert!(self.n >= 1 && self.n.is_power_of_two());

            let mut u = 1;
            let mut width = self.n as u32;
            let mut prefix = self.ma.id();

            while u < self.n {
                width >>= 1;
                self.push_down(width, u);

                let new_prefix = self.ma.combine(&prefix, &self.sum[u << 1]);
                u = if pred(&new_prefix) {
                    prefix = new_prefix;
                    u << 1 | 1
                } else {
                    u << 1
                };
            }

            let idx = u - self.n;
            if pred(&self.ma.combine(&prefix, &self.sum[u])) {
                idx + 1
            } else {
                idx
            }
        }
    }
}

fn euler_rec(
    children: &[Vec<u32>],
    euler_node: &mut [u32],
    euler_child_start: &mut [u32],
    euler_child_end: &mut [u32],
    euler_subtree_out: &mut [u32],
    timer: &mut u32,
    u: usize,
) {
    euler_child_start[u as usize] = *timer;
    for &v in &children[u] {
        euler_node[v as usize] = *timer;
        *timer += 1;
    }
    euler_child_end[u] = *timer;
    for &v in &children[u] {
        euler_rec(
            children,
            euler_node,
            euler_child_start,
            euler_child_end,
            euler_subtree_out,
            timer,
            v as usize,
        );
    }
    euler_subtree_out[u] = *timer;
}

// type ModP = NaiveModInt<ByU64<1_000_000_007>>;
type ModP = MontgomeryU32<1_000_000_007>;

struct AffineSpace;

impl MonoidAction for AffineSpace {
    type X = ModP;
    type F = (ModP, ModP);
    const IS_X_COMMUTATIVE: bool = true;

    fn id(&self) -> Self::X {
        0.into()
    }

    fn combine(&self, lhs: &Self::X, rhs: &Self::X) -> Self::X {
        *lhs + *rhs
    }

    fn id_action(&self) -> Self::F {
        (0.into(), 1.into())
    }

    fn combine_action(&self, lhs: &Self::F, rhs: &Self::F) -> Self::F {
        (lhs.0 + lhs.1 * rhs.0, lhs.1 * rhs.1)
    }

    fn apply_to_sum(&self, f: &Self::F, x_count: u32, x_sum: &mut Self::X) {
        *x_sum = f.0 * ModP::from(x_count) + f.1 * *x_sum;
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let xs_orig: Vec<u32> = (0..n).map(|_| input.value()).collect();
    let mut parent = vec![0; n];
    let mut children = vec![vec![]; n];
    for u in 1..n as u32 {
        let p = input.value::<u32>() - 1;
        parent[u as usize] = p;
        children[p as usize].push(u);
    }

    let mut euler_idx = vec![0; n];
    let mut euler_child_in = vec![0; n];
    let mut euler_child_out = vec![0; n];
    let mut euler_subtree_out = vec![0; n];
    euler_rec(
        &children,
        &mut euler_idx,
        &mut euler_child_in,
        &mut euler_child_out,
        &mut euler_subtree_out,
        &mut 1,
        0,
    );

    let euler_idx = |u: usize| euler_idx[u] as usize;
    let euler_children = |u: usize| euler_child_in[u] as usize..euler_child_out[u] as usize;
    let euler_proper_subtree = |u: usize| euler_child_in[u] as usize..euler_subtree_out[u] as usize;

    let mut xs = vec![0.into(); n];
    for u in 0..n {
        xs[euler_idx(u)] = ModP::from(xs_orig[u]);
    }
    let mut tree = SegTree::from_iter(xs, AffineSpace);

    for _ in 0..input.value() {
        match input.token() {
            "1" => {
                let v = input.value::<usize>() - 1;
                let a = input.value::<u32>().into();
                let b = input.value::<u32>().into();
                let action = (b, a);

                let p = parent[v] as usize;
                if p != v {
                    tree.apply_range(euler_idx(p)..euler_idx(p) + 1, action);
                }
                tree.apply_range(euler_idx(v)..euler_idx(v) + 1, action);
                tree.apply_range(euler_children(v), action);
            }
            "2" => {
                let v = input.value::<usize>() - 1;
                let a = input.value::<u32>().into();
                let b = input.value::<u32>().into();
                let action = (b, a);

                tree.apply_range(euler_idx(v)..euler_idx(v) + 1, action);
                tree.apply_range(euler_proper_subtree(v), action);
            }
            "3" => {
                let v = input.value::<usize>() - 1;
                let ans = u32::from(tree.query_range(euler_idx(v)..euler_idx(v) + 1));
                writeln!(output, "{}", ans).unwrap();
            }
            _ => panic!(),
        }
    }
}
