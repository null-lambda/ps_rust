pub mod frac {
    use std::ops::*;

    type S = i64;
    type U = u64;

    fn gcd(mut a: U, mut b: U) -> U {
        while b != 0 {
            let temp = b;
            b = a % b;
            a = temp;
        }
        a
    }

    #[derive(Debug, Clone, Copy)]
    pub struct Frac(S, S);

    impl Frac {
        pub fn new(n: S, d: S) -> Self {
            assert!(d > 0, "Denominator must be always positive");
            Self(n, d).normalized()
        }

        pub fn numer(&self) -> S {
            self.0
        }

        pub fn denom(&self) -> S {
            self.1
        }

        pub fn inner(&self) -> (S, S) {
            (self.0, self.1)
        }

        pub fn normalized(self) -> Self {
            let Self(n, d) = self;
            let g = gcd(n.abs() as U, d.abs() as U) as S * d.signum();
            Self(n / g, d / g)
        }

        pub fn zero() -> Self {
            Self(0, 1)
        }

        pub fn one() -> Self {
            Self(1, 1)
        }

        pub fn abs(self) -> Self {
            Self(self.0.abs(), self.1)
        }
    }

    impl Add for Frac {
        type Output = Self;
        fn add(self, rhs: Self) -> Self {
            Self::new(self.0 * rhs.1 + rhs.0 * self.1, self.1 * rhs.1)
        }
    }

    impl Sub for Frac {
        type Output = Self;
        fn sub(self, rhs: Self) -> Self {
            Self::new(self.0 * rhs.1 - rhs.0 * self.1, self.1 * rhs.1)
        }
    }

    impl Mul for Frac {
        type Output = Self;
        fn mul(self, rhs: Self) -> Self {
            Self::new(self.0 * rhs.0, self.1 * rhs.1)
        }
    }

    impl Div for Frac {
        type Output = Self;
        fn div(self, rhs: Self) -> Self {
            let s = rhs.0.signum();
            Self::new(self.0 * rhs.1 * s, self.1 * rhs.0 * s)
        }
    }

    macro_rules! forward_binop {
        ($OpAssign:ident $op_assign:ident, $Op:ident $op:ident) => {
            impl $Op<&Frac> for Frac {
                type Output = Frac;
                fn $op(self, rhs: &Frac) -> Self::Output {
                    self.$op(*rhs)
                }
            }

            impl $Op<Frac> for &Frac {
                type Output = Frac;
                fn $op(self, rhs: Frac) -> Self::Output {
                    (*self).$op(rhs)
                }
            }

            impl $Op<&Frac> for &Frac {
                type Output = Frac;
                fn $op(self, rhs: &Frac) -> Self::Output {
                    (*self).$op(*rhs)
                }
            }

            impl $OpAssign<Frac> for Frac {
                fn $op_assign(&mut self, rhs: Frac) {
                    *self = (*self).$op(rhs);
                }
            }

            impl $OpAssign<&Frac> for Frac {
                fn $op_assign(&mut self, rhs: &Frac) {
                    *self = (*self).$op(*rhs);
                }
            }
        };
    }

    forward_binop!(AddAssign add_assign, Add add);
    forward_binop!(SubAssign sub_assign, Sub sub);
    forward_binop!(MulAssign mul_assign, Mul mul);
    forward_binop!(DivAssign div_assign, Div div);

    impl Neg for Frac {
        type Output = Self;
        fn neg(self) -> Self::Output {
            Self(-self.0, self.1)
        }
    }

    impl From<S> for Frac {
        fn from(a: S) -> Self {
            Self::new(a, 1)
        }
    }

    impl From<(S, S)> for Frac {
        fn from((n, d): (S, S)) -> Self {
            Self::new(n, d)
        }
    }

    impl PartialEq for Frac {
        fn eq(&self, other: &Self) -> bool {
            self.0 * other.1 == other.0 * self.1
        }
    }

    impl Eq for Frac {}

    impl PartialOrd for Frac {
        fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
            Some((self.0 * other.1).cmp(&(other.0 * self.1)))
        }
    }

    impl Ord for Frac {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            self.partial_cmp(other).unwrap()
        }
    }
}
