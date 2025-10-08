pub mod dp_sos {
    // A pair of subset Zeta and Mobius transforms and their variants.
    use std::ops::{AddAssign, SubAssign};

    pub trait CommGroup: Default + AddAssign + SubAssign + Clone {}
    impl<T: Default + AddAssign + SubAssign + Clone> CommGroup for T {}

    fn unit_bits(pow2: usize) -> impl Iterator<Item = usize> {
        assert!(pow2.is_power_of_two());
        (0..pow2.trailing_zeros()).map(|i| 1 << i)
    }

    fn chunks_exact_mut_paired<T>(
        xs: &mut [T],
        block_size: usize,
    ) -> impl Iterator<Item = (&mut [T], &mut [T])> {
        xs.chunks_exact_mut(block_size * 2)
            .map(move |block| block.split_at_mut(block_size))
    }

    fn kronecker_prod<T: CommGroup>(xs: &mut [T], modifier: impl Fn(&mut T, &mut T)) {
        let n = xs.len();
        for e in unit_bits(n) {
            for (zs, os) in chunks_exact_mut_paired(xs, e) {
                for (z, o) in zs.iter_mut().zip(os) {
                    modifier(z, o);
                }
            }
        }
    }

    pub fn subset_sums<T: CommGroup>(xs: &mut [T]) {
        kronecker_prod(xs, |z, o| *o += z.clone());
    }

    pub fn inv_subset_sums<T: CommGroup>(xs: &mut [T]) {
        kronecker_prod(xs, |z, o| *o -= z.clone());
    }

    pub fn superset_sums<T: CommGroup>(xs: &mut [T]) {
        kronecker_prod(xs, |z, o| *z += o.clone());
    }

    pub fn inv_superset_sums<T: CommGroup>(xs: &mut [T]) {
        kronecker_prod(xs, |z, o| *z -= o.clone());
    }

    // Fast Walsh-Hadamard Transform.
    // For an inverse transform, do fwht and then divide by 2^n == xs.len().
    pub fn fwht<T: CommGroup>(xs: &mut [T]) {
        kronecker_prod(xs, |z, o| {
            let o_old = o.clone();
            *o = z.clone();
            *z += o_old.clone();
            *o -= o_old;
        });
    }
}
