pub mod set_power_series {
    use crate::algebra::CommRing;
    use std::ops::*;

    pub trait CommGroup: Default + AddAssign + SubAssign + Clone {}
    impl<T: Default + AddAssign + SubAssign + Clone> CommGroup for T {}

    fn kronecker_prod<T: CommGroup>(xs: &mut [T], modifier: impl Fn(&mut T, &mut T)) {
        #[target_feature(enable = "avx", enable = "avx2")]
        unsafe fn inner<T: CommGroup>(xs: &mut [T], modifier: impl Fn(&mut T, &mut T)) {
            let n = xs.len().ilog2() as usize;
            for b in 0..n {
                for t in xs.chunks_exact_mut(1 << b + 1) {
                    let (zs, os) = t.split_at_mut(1 << b);
                    zs.iter_mut().zip(os).for_each(|(z, o)| modifier(z, o));
                }
            }
        }
        unsafe { inner(xs, modifier) }
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
    pub fn fwht<T: CommGroup>(xs: &mut [T]) {
        kronecker_prod(xs, |z, o| {
            let o_old = o.clone();
            *o = z.clone();
            *z += o_old.clone();
            *o -= o_old;
        });
    }
    pub fn ifwht<T: CommGroup>(xs: &mut [T]) {
        kronecker_prod(xs, |z, o| {
            let o_old = o.clone();
            *o = z.clone();
            *z += o_old.clone();
            *o -= o_old;
        });
    }

    // Group terms of $\sum_{S \subset [n]} a_S x^S$ by $|S|$
    pub fn chop<T: CommRing>(xs: &[T]) -> Vec<T> {
        assert!(xs.len().is_power_of_two());
        let n = xs.len().ilog2() as usize;
        let mut res = vec![T::zero(); (n + 1) * (1 << n)];
        for (i, x) in xs.iter().cloned().enumerate() {
            res[((i.count_ones() as usize) << n) + i] = x;
        }
        res
    }
    pub fn unchop<T: CommRing>(n: usize, xs: &[T]) -> Vec<T> {
        assert_eq!(xs.len(), n + 1 << n);
        let mut res = vec![T::zero(); 1 << n];
        for i in 0..1 << n {
            res[i] = xs[((i.count_ones() as usize) << n) + i].clone();
        }
        res
    }
    pub fn subset_conv<T: CommRing>(xs: &[T], ys: &[T]) -> Vec<T> {
        #[target_feature(enable = "avx", enable = "avx2")]
        unsafe fn inner<T: CommRing>(xs: &[T], ys: &[T]) -> Vec<T> {
            assert!(xs.len().is_power_of_two());
            let n = xs.len().ilog2() as usize;

            let mut xs = chop(&xs);
            let mut ys = chop(&ys);
            xs.chunks_exact_mut(1 << n).for_each(|bs| subset_sums(bs));
            ys.chunks_exact_mut(1 << n).for_each(|bs| subset_sums(bs));

            let mut zs = vec![T::default(); 1 << n];
            for k in 0..=n {
                let mut zr = vec![T::default(); 1 << n];
                for i in 0..=k {
                    let xr = &xs[i << n..][..1 << n];
                    let yr = &ys[k - i << n..][..1 << n];
                    for ((x, y), z) in xr.iter().zip(yr).zip(&mut zr) {
                        *z += x.clone() * y.clone();
                    }
                }
                inv_subset_sums(&mut zr);
                for (i, z) in zr.into_iter().enumerate() {
                    if i.count_ones() == k as u32 {
                        zs[i] += z;
                    }
                }
            }
            zs
        }
        unsafe { inner(xs, ys) }
    }
}
