// monoid, not necesserily commutative
pub trait Monoid {
    fn id() -> Self;
    fn op(self, rhs: Self) -> Self;
}

pub trait PowMonoid: Monoid {
    fn pow(self, n: u32) -> Self;
}

// monoid action A -> End(M), where A is a monoid and M is a set.
// the image of A is a submonoid of End(M)
pub trait MonoidAction<M>: Monoid {
    fn apply_to_sum(self, x_sum: M, x_count: u32) -> M;
}

// monoid action on itself
impl<M: PowMonoid> MonoidAction<M> for M {
    fn apply_to_sum(self, x_sum: M, x_count: u32) -> M {
        self.pow(x_count).op(x_sum)
    }
}

impl Monoid for u32 {
    fn id() -> Self {
        0
    }
    fn op(self, other: Self) -> Self {
        self + other
    }
}

impl PowMonoid for u32 {
    fn pow(self, n: u32) -> Self {
        self * n
    }
}
