pub mod set_power_series {
    use crate::algebra::{CommRing, Field};
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
            assert_eq!(xs.len(), ys.len());
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
    fn subset_inv_inner<T: CommRing>(xs: &[T], inv_xs0: T) -> Vec<T> {
        assert!(xs.len().is_power_of_two());
        let n = xs.len().ilog2() as usize;

        let mut res = vec![T::zero(); 1 << n];
        res[0] = inv_xs0;
        let mut w = 1;
        while w < 1 << n {
            let mut ext = subset_conv(&xs[w..w * 2], &subset_conv(&res[..w], &res[..w]));
            ext.iter_mut().for_each(|x| *x = -x.clone());
            res[w..w * 2].clone_from_slice(&ext);
            w *= 2;
        }
        res
    }
    pub fn subset_inv1<T: CommRing>(xs: &[T]) -> Vec<T> {
        assert!(xs[0] == T::one());
        subset_inv_inner(xs, T::one())
    }
    pub fn subset_inv<T: Field>(xs: &[T]) -> Vec<T> {
        assert!(xs[0] != T::zero());
        subset_inv_inner(xs, xs[0].inv())
    }
    pub fn subset_exp<T: CommRing>(xs: &[T]) -> Vec<T> {
        assert!(xs.len().is_power_of_two());
        assert!(xs[0] == T::zero());
        let n = xs.len().ilog2() as usize;

        let mut res = vec![T::one(); 1 << n];
        let mut w = 1;
        while w < 1 << n {
            let ext = subset_conv(&xs[w..w * 2], &res[..w]);
            res[w..w * 2].clone_from_slice(&ext);
            w *= 2;
        }
        res
    }
    pub fn subset_ln<T: CommRing>(xs: &[T]) -> Vec<T> {
        assert!(xs.len().is_power_of_two());
        assert!(xs[0] == T::one());
        let n = xs.len().ilog2() as usize;

        let mut res = vec![T::zero(); 1 << n];
        let mut w = 1;
        while w < 1 << n {
            let ext = subset_conv(&xs[w..w * 2], &subset_inv1(&xs[..w]));
            res[w..w * 2].clone_from_slice(&ext);
            w *= 2;
        }
        res
    }
    pub fn subset_comp<T: CommRing + From<u32> + std::fmt::Debug>(f: &[T], xs: &[T]) -> Vec<T> {
        assert!(xs.len().is_power_of_two());
        let n = xs.len().ilog2() as usize;

        if f.is_empty() {
            return vec![T::zero(); 1 << n];
        }

        // $(d^k f) \circ xs$
        let mut layers = vec![T::zero(); (n + 1) * 1];
        {
            let mut dk_f = f.to_vec();
            for k in 0..n + 1 {
                if k >= 1 {
                    dk_f = dk_f.into_iter().skip(1).collect();
                    for i in 0..dk_f.len() {
                        dk_f[i] *= T::from(i as u32 + 1);
                    }
                }
                let mut pow = T::one();
                for i in 0..dk_f.len() {
                    layers[k] += dk_f[i].clone() * pow.clone();
                    pow *= &xs[0];
                }
            }
        }
        for b in 1..=n {
            let w = 1 << b - 1;
            let prev = layers;
            layers = vec![T::zero(); (n - b + 1) * (w * 2)];
            for c in 0..=n - b {
                let p0 = &prev[c * w..][..w];
                let p1 = &prev[c * w + w..][..w];
                layers[c * w * 2..][..w].clone_from_slice(p0);
                layers[c * w * 2 + w..][..w].clone_from_slice(&subset_conv(p1, &xs[w..w * 2]));
            }
        }
        layers
    }
    // todo: power projection
}
