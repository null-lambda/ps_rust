// Squeeze memory with smaller integer types
pub mod num {
    use std::ops::{Add, Div, Mul, Sub};

    macro_rules! impl_base {
        ($t:ident, $size:expr) => {
            #[allow(non_camel_case_types)]
            #[repr(C)]
            #[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
            pub struct $t([u8; $size]);
        };
    }

    macro_rules! impl_from {
        ($t:ident, $padded:ty, $size:expr) => {
            impl From<$padded> for $t {
                fn from(x: $padded) -> Self {
                    let mut res = Self::default();
                    res.0.copy_from_slice(&x.to_le_bytes()[..$size]);
                    res
                }
            }

            impl From<$t> for $padded {
                fn from(x: $t) -> Self {
                    let mut bytes = [0; std::mem::size_of::<$padded>()];
                    bytes[..$size].copy_from_slice(&x.0);
                    <$padded>::from_le_bytes(bytes)
                }
            }
        };
    }

    macro_rules! impl_binop {
        ($t:ty, $padded:ty,  $({$trait:ident, $trait_assign: ident, $method:ident, $method_assign:ident}),*) => {
            $(
                impl std::ops::$trait for $t {
                    type Output = Self;

                    fn $method(self, rhs: Self) -> Self {
                        Self::from(<$padded>::from(self).$method(<$padded>::from(rhs)))
                    }
                }

                impl std::ops::$trait_assign for $t {
                    fn $method_assign(&mut self, rhs: $t) {
                        *self = (*self).$method(rhs);
                    }
                }
            )*
        };
    }

    impl_base!(u24, 3);
    impl_from!(u24, u32, 3);
    impl_binop!(u24, u32,
        {Add, AddAssign, add, add_assign},
        {Sub, SubAssign, sub, sub_assign},
        {Mul, MulAssign, mul, mul_assign},
        {Div, DivAssign, div, div_assign}
    );

    impl_base!(u48, 6);
    impl_from!(u48, u64, 6);
    impl_binop!(u48, u64,
        {Add, AddAssign, add, add_assign},
        {Sub, SubAssign, sub, sub_assign},
        {Mul, MulAssign, mul, mul_assign},
        {Div, DivAssign, div, div_assign}
    );
}
