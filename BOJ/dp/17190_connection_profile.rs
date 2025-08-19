use std::{cmp::Ordering, collections::HashMap, io::Write};

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

pub mod algebra {
    use std::ops::*;
    pub trait Unsigned:
        Copy
        + Default
        + SemiRing
        + Div<Output = Self>
        + Rem<Output = Self>
        + RemAssign
        + PartialEq
        + Eq
        + PartialOrd
        + Ord
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

        fn pow<U: Unsigned>(&self, exp: U) -> Self {
            let mut res = Self::one();
            let mut base = self.clone();
            let mut exp = exp;
            while exp > U::from(0u8) {
                if exp % U::from(2u8) == U::from(1u8) {
                    res *= base.clone();
                }
                base *= base.clone();
                exp = exp / U::from(2);
            }
            res
        }
    }

    pub trait CommRing: SemiRing + Neg<Output = Self> {}

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
}

pub mod mint {
    use super::algebra::*;
    use std::ops::*;

    pub trait ModSpec: Copy {
        type U: Unsigned;
        const MODULUS: Self::U;
    }

    #[derive(Clone, Copy, PartialEq, Eq)]
    pub struct MInt<M: ModSpec>(M::U);

    impl<M: ModSpec> MInt<M> {
        pub fn new(s: M::U) -> Self {
            Self(s % M::MODULUS)
        }
    }

    macro_rules! impl_modspec {
        ($wrapper:ident $spec:ident $u:ty) => {
            #[derive(Debug, Clone, Copy, PartialEq, Eq)]
            pub struct $spec<const M: $u>;

            impl<const MOD: $u> ModSpec for $spec<MOD> {
                type U = $u;
                const MODULUS: $u = MOD;
            }

            pub type $wrapper<const M: $u> = MInt<$spec<M>>;
        };
    }
    impl_modspec!(M32 __ByU32 u32);
    impl_modspec!(M64 __ByU64 u64);
    impl_modspec!(M128 __ByU128 u128);

    impl<M: ModSpec> AddAssign<&'_ Self> for MInt<M> {
        fn add_assign(&mut self, rhs: &Self) {
            self.0 += rhs.0;
            if self.0 >= M::MODULUS {
                self.0 -= M::MODULUS;
            }
        }
    }

    impl<M: ModSpec> SubAssign<&'_ Self> for MInt<M> {
        fn sub_assign(&mut self, rhs: &Self) {
            if self.0 < rhs.0 {
                self.0 += M::MODULUS;
            }
            self.0 -= rhs.0;
        }
    }

    impl<M: ModSpec> MulAssign<&'_ Self> for MInt<M> {
        fn mul_assign(&mut self, rhs: &Self) {
            self.0 *= rhs.0;
            self.0 %= M::MODULUS;
        }
    }

    impl<M: ModSpec> DivAssign<&'_ Self> for MInt<M> {
        fn div_assign(&mut self, rhs: &Self) {
            self.mul_assign(&rhs.inv());
        }
    }

    macro_rules! forward_binop {
        ($OpAssign:ident $op_assign:ident, $Op:ident $op:ident) => {
            impl<M: ModSpec> $OpAssign for MInt<M> {
                fn $op_assign(&mut self, rhs: Self) {
                    self.$op_assign(&rhs);
                }
            }

            impl<M: ModSpec> $Op<&'_ Self> for MInt<M> {
                type Output = Self;
                fn $op(mut self, rhs: &Self) -> Self {
                    self.$op_assign(rhs);
                    self
                }
            }

            impl<M: ModSpec> $Op for MInt<M> {
                type Output = MInt<M>;
                fn $op(self, rhs: Self) -> Self::Output {
                    self.clone().$op(&rhs)
                }
            }
        };
    }
    forward_binop!(AddAssign add_assign, Add add);
    forward_binop!(SubAssign sub_assign, Sub sub);
    forward_binop!(MulAssign mul_assign, Mul mul);
    forward_binop!(DivAssign div_assign, Div div);

    impl<M: ModSpec> Neg for &'_ MInt<M> {
        type Output = MInt<M>;
        fn neg(self) -> MInt<M> {
            let mut res = M::MODULUS - self.0;
            if res == M::MODULUS {
                res = 0.into();
            }
            MInt(res)
        }
    }

    impl<M: ModSpec> Neg for MInt<M> {
        type Output = Self;
        fn neg(self) -> Self::Output {
            (&self).neg()
        }
    }

    impl<M: ModSpec> Default for MInt<M> {
        fn default() -> Self {
            Self(M::U::default())
        }
    }

    impl<M: ModSpec> SemiRing for MInt<M> {
        fn one() -> Self {
            Self(1.into())
        }
    }
    impl<M: ModSpec> CommRing for MInt<M> {}

    impl<M: ModSpec> Field for MInt<M> {
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
        (@pairwise $lhs:ident $rhs:ident => $wider:ident) => {
            impl CmpUType<$rhs> for $lhs {
                type MaxT = $wider;
                fn upcast(lhs: Self) -> Self::MaxT {
                    lhs as Self::MaxT
                }
                fn upcast_rhs(rhs: $rhs) -> Self::MaxT {
                    rhs as Self::MaxT
                }
                fn downcast(wider: Self::MaxT) -> Self {
                    wider as Self
                }
            }
        };

        (@cascade $target:ident $($upper:ident)*) => {
            $(
                impl_cmp_utype!(@pairwise $target $upper => $upper);
                impl_cmp_utype!(@pairwise $upper $target => $upper);
            )*
            impl_cmp_utype!(@pairwise $target $target => $target);
        };

        ($target:ident $($rest:ident)*) => {
            impl_cmp_utype!(@cascade $target $($rest)*);
            impl_cmp_utype!($($rest)*);
        };

        () => {};
    }
    impl_cmp_utype!(u8 u16 u32 u64 u128);

    impl<U, S, M> From<S> for MInt<M>
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
                impl<M: ModSpec<U = $u>> From<MInt<M>> for $u {
                    fn from(n: MInt<M>) -> Self {
                        n.0
                    }
                }
            )+
        };
    }
    impl_cast_to_unsigned!(u8 u16 u32 u64 u128);

    impl<U: std::fmt::Debug, M: ModSpec<U = U>> std::fmt::Debug for MInt<M> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.0.fmt(f)
        }
    }

    impl<U: std::fmt::Display, M: ModSpec<U = U>> std::fmt::Display for MInt<M> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            self.0.fmt(f)
        }
    }

    impl<U: std::str::FromStr, M: ModSpec<U = U>> std::str::FromStr for MInt<M> {
        type Err = <U as std::str::FromStr>::Err;
        fn from_str(s: &str) -> Result<Self, Self::Err> {
            s.parse().map(|x| MInt::new(x))
        }
    }
}

type M = mint::M64<1000000007>;

#[derive(Clone, Copy, Debug)]
struct MinCount {
    min: u64,
    count: M,
}

impl MinCount {
    fn new(x: u64) -> Self {
        Self {
            min: x,
            count: M::one(),
        }
    }
}

impl MinCount {
    fn min(self, other: Self) -> Self {
        match self.min.cmp(&other.min) {
            Ordering::Less => self,
            Ordering::Greater => other,
            Ordering::Equal => Self {
                min: self.min,
                count: self.count + other.count,
            },
        }
    }

    fn min_assign(&mut self, other: Self) {
        *self = (*self).min(other);
    }

    fn add(self, delta: u64) -> Self {
        Self {
            min: self.min + delta,
            count: self.count,
        }
    }

    fn trunc(self, bound: u64) -> Self {
        Self {
            min: self.min.min(bound),
            ..self
        }
    }
}
const K_MAX: usize = 6;
type Row = [u8; K_MAX];

fn normalize(mut row: Row, k: usize) -> Row {
    let mut color_map = HashMap::new();
    for i in 0..k {
        let c = color_map.len() as u8;
        row[i] = *color_map.entry(row[i]).or_insert_with(|| c);
    }
    row
}

fn push(mut row: Row, k: usize, wall_l: bool, wall_t: bool) -> Option<Row> {
    let top = row[k - 1];
    let left = row[0];

    for i in (1..k).rev() {
        row[i] = row[i - 1];
    }

    if wall_t {
        let mut is_top_unique = true;
        for i in 1..k {
            if row[i] == top {
                is_top_unique = false;
                break;
            }
        }
        if is_top_unique {
            return None;
        }
    }

    match (wall_l, wall_t) {
        (true, true) => row[0] = 42,
        (true, false) => row[0] = top,
        (false, true) => row[0] = left,
        (false, false) => {
            if top != left {
                for x in &mut row {
                    if *x == top {
                        *x = left;
                    }
                }
            }
            row[0] = left;
        }
    }

    row = normalize(row, k);
    Some(row)
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let k: usize = input.value();

    let inf_cost = 1 << 61;
    let mut cost_h = vec![[inf_cost; K_MAX]; n];
    let mut cost_v = vec![[0; K_MAX]; n];
    for i in 0..n {
        for j in 1..k {
            cost_h[i][j] = input.value();
        }
    }
    for j in 0..k {
        for i in 1..n {
            cost_v[i][j] = input.value();
        }
    }

    let mut states = vec![[0; K_MAX]];
    let mut l_prev = states.len();
    loop {
        for i in 0..states.len() {
            let w = states[i];
            states.extend(push(w, k, false, false));
            states.extend(push(w, k, false, true));
            states.extend(push(w, k, true, false));
            states.extend(push(w, k, true, true));
        }

        states.sort_unstable();
        states.dedup();
        if states.len() == l_prev {
            break;
        }
        l_prev = states.len();
    }

    let mut discrete = [0; K_MAX];
    for i in 0..k {
        discrete[i] = i as u8;
    }
    states.push(discrete);
    states.sort_unstable();
    states.dedup();

    let sink = [!0; K_MAX];
    states.push(sink);

    let n_states = states.len();
    let inv = HashMap::<Row, u8>::from_iter(states.iter().enumerate().map(|(u, &s)| (s, u as u8)));

    let mut trans1 = vec![[[inv[&sink]; 2]; 2]; n_states];
    for (iu, &u) in states.iter().enumerate() {
        if u == sink {
            continue;
        }

        for b in 0..2 {
            for c in 0..2 {
                let v = push(u, k, b == 1, c == 1).unwrap_or(sink);
                trans1[iu][b][c] = inv[&v];
            }
        }
    }

    let inf = MinCount::new(inf_cost);
    let mut dp = vec![inf; n_states];

    dp[inv[&normalize(discrete, k)] as usize] = MinCount::new(0);

    for i in 0..n {
        for j in 0..k {
            let prev = dp;
            dp = vec![inf; n_states];

            for s in 0..n_states {
                let p = [prev[s].add(cost_h[i][j]).trunc(inf_cost), prev[s]];
                dp[trans1[s][0][0] as usize].min_assign(p[0].add(cost_v[i][j]));
                dp[trans1[s][0][1] as usize].min_assign(p[0]);
                dp[trans1[s][1][0] as usize].min_assign(p[1].add(cost_v[i][j]));
                dp[trans1[s][1][1] as usize].min_assign(p[1]);
            }
            // println!("{:?}", dp);
        }
        // println!();
    }

    let ans = dp[inv[&[0; K_MAX]] as usize];
    // println!("{:?}", ans);
    writeln!(output, "{}", ans.count).unwrap();
}
