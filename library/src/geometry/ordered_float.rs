pub mod ordered {
    use std::{
        cmp::Ordering,
        ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign},
        str::FromStr,
    };

    #[derive(Clone, Copy, Default)]
    pub struct F64(pub f64);

    impl F64 {
        pub fn new(x: f64) -> Self {
            Self(x)
        }

        pub fn map_inner(self, f: impl FnOnce(f64) -> f64) -> Self {
            Self(f(self.0))
        }
    }

    impl PartialEq for F64 {
        fn eq(&self, other: &Self) -> bool {
            self.0.total_cmp(&other.0).is_eq()
        }
    }

    impl Eq for F64 {}

    impl PartialOrd for F64 {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.0.total_cmp(&other.0))
        }
    }

    impl Ord for F64 {
        fn cmp(&self, other: &Self) -> Ordering {
            self.0.total_cmp(&other.0)
        }
    }

    impl std::hash::Hash for F64 {
        fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
            self.0.to_bits().hash(state);
        }
    }

    impl std::fmt::Debug for F64 {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            self.0.fmt(f)
        }
    }

    macro_rules! impl_binop {
        ($trait:ident, $fn:ident, $trait_assign:ident, $fn_assign:ident) => {
            impl $trait for F64 {
                type Output = Self;
                fn $fn(self, other: Self) -> Self::Output {
                    Self(self.0.$fn(other.0))
                }
            }

            impl $trait_assign for F64 {
                fn $fn_assign(&mut self, other: Self) {
                    self.0.$fn_assign(other.0);
                }
            }
        };
    }

    impl Neg for F64 {
        type Output = Self;
        fn neg(self) -> Self::Output {
            Self(-self.0)
        }
    }

    impl_binop!(Add, add, AddAssign, add_assign);
    impl_binop!(Sub, sub, SubAssign, sub_assign);
    impl_binop!(Mul, mul, MulAssign, mul_assign);
    impl_binop!(Div, div, DivAssign, div_assign);

    impl super::geometry::Scalar for F64 {
        fn one() -> Self {
            Self(1.0)
        }
    }

    impl FromStr for F64 {
        type Err = <f64 as FromStr>::Err;
        fn from_str(s: &str) -> Result<Self, Self::Err> {
            Ok(Self(f64::from_str(s)?))
        }
    }
}
