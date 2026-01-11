pub mod sps {
    // Variants of R[x_1, ..., x_n]/(q(x_1), ..., q(x_n)),
    // including Set Power Series (q(x) = x^2)

    use crate::algebra::{CommRing, Field};
    use std::ops::*;

    pub trait CommGroup: Default + AddAssign + SubAssign + Clone {}
    impl<T: Default + AddAssign + SubAssign + Clone> CommGroup for T {}

    fn kronecker_prod<T: CommGroup>(f: &mut [T], modifier: impl Fn(&mut T, &mut T)) {
        #[target_feature(enable = "avx", enable = "avx2")]
        unsafe fn inner<T: CommGroup>(f: &mut [T], modifier: impl Fn(&mut T, &mut T)) {
            let n = f.len().ilog2() as usize;
            for b in 0..n {
                for t in f.chunks_exact_mut(1 << b + 1) {
                    let (zs, os) = t.split_at_mut(1 << b);
                    zs.iter_mut().zip(os).for_each(|(z, o)| modifier(z, o));
                }
            }
        }
        unsafe { inner(f, modifier) }
    }
    pub fn subset_sums<T: CommGroup>(f: &mut [T]) {
        kronecker_prod(f, |z, o| *o += z.clone());
    }
    pub fn inv_subset_sums<T: CommGroup>(f: &mut [T]) {
        kronecker_prod(f, |z, o| *o -= z.clone());
    }
    pub fn superset_sums<T: CommGroup>(f: &mut [T]) {
        kronecker_prod(f, |z, o| *z += o.clone());
    }
    pub fn inv_superset_sums<T: CommGroup>(f: &mut [T]) {
        kronecker_prod(f, |z, o| *z -= o.clone());
    }
    pub fn fwht<T: CommGroup>(xs: &mut [T]) {
        kronecker_prod(xs, |z, o| {
            let o_old = o.clone();
            *o = z.clone();
            *z += o_old.clone();
            *o -= o_old;
        });
    }
    pub fn ifwht<T: CommGroup>(f: &mut [T]) {
        kronecker_prod(f, |z, o| {
            let o_old = o.clone();
            *o = z.clone();
            *z += o_old.clone();
            *o -= o_old;
        });
    }

    // Group terms of $\sum_{S \subset [n]} f_S x^S$ by $|S|$
    pub fn chop<T: CommRing>(f: &[T]) -> Vec<T> {
        assert!(f.len().is_power_of_two());
        let n = f.len().ilog2() as usize;
        let mut res = vec![T::zero(); (n + 1) * (1 << n)];
        for (i, x) in f.iter().cloned().enumerate() {
            res[((i.count_ones() as usize) << n) + i] = x;
        }
        res
    }
    pub fn unchop<T: CommRing>(n: usize, f: &[T]) -> Vec<T> {
        assert_eq!(f.len(), n + 1 << n);
        let mut res = vec![T::zero(); 1 << n];
        for i in 0..1 << n {
            res[i] = f[((i.count_ones() as usize) << n) + i].clone();
        }
        res
    }
    pub fn conv<T: CommRing>(f: &[T], g: &[T]) -> Vec<T> {
        #[target_feature(enable = "avx", enable = "avx2")]
        unsafe fn inner<T: CommRing>(f: &[T], g: &[T]) -> Vec<T> {
            assert!(f.len().is_power_of_two());
            assert_eq!(f.len(), g.len());
            let n = f.len().ilog2() as usize;

            let mut xs = chop(&f);
            let mut ys = chop(&g);
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
        unsafe { inner(f, g) }
    }
    fn inv_inner<T: CommRing>(f: &[T], inv_f0: T) -> Vec<T> {
        assert!(f.len().is_power_of_two());
        let n = f.len().ilog2() as usize;

        let mut res = vec![T::zero(); 1 << n];
        res[0] = inv_f0;
        let mut w = 1;
        while w < 1 << n {
            let mut ext = conv(&f[w..w * 2], &conv(&res[..w], &res[..w]));
            ext.iter_mut().for_each(|x| *x = -x.clone());
            res[w..w * 2].clone_from_slice(&ext);
            w *= 2;
        }
        res
    }
    pub fn inv1<T: CommRing>(f: &[T]) -> Vec<T> {
        assert!(f[0] == T::one());
        inv_inner(f, T::one())
    }
    pub fn inv<T: Field>(f: &[T]) -> Vec<T> {
        assert!(f[0] != T::zero());
        inv_inner(f, f[0].inv())
    }
    pub fn exp<T: CommRing>(f: &[T]) -> Vec<T> {
        assert!(f.len().is_power_of_two());
        assert!(f[0] == T::zero());
        let n = f.len().ilog2() as usize;

        let mut res = vec![T::one(); 1 << n];
        let mut w = 1;
        while w < 1 << n {
            let ext = conv(&f[w..w * 2], &res[..w]);
            res[w..w * 2].clone_from_slice(&ext);
            w *= 2;
        }
        res
    }
    pub fn ln<T: CommRing>(f: &[T]) -> Vec<T> {
        assert!(f.len().is_power_of_two());
        assert!(f[0] == T::one());
        let n = f.len().ilog2() as usize;

        let mut res = vec![T::zero(); 1 << n];
        let mut w = 1;
        let f_inv = inv1(&f[..1 << n - 1]);
        while w < 1 << n {
            let ext = conv(&f[w..w * 2], &f_inv[..w]);
            res[w..w * 2].clone_from_slice(&ext);
            w *= 2;
        }
        res
    }
    pub fn comp<T: CommRing + From<u32>>(f: &[T], g: &[T]) -> Vec<T> {
        assert!(g.len().is_power_of_two());
        let n = g.len().ilog2() as usize;

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
                    pow *= &g[0];
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
                layers[c * w * 2 + w..][..w].clone_from_slice(&conv(p1, &g[w..w * 2]));
            }
        }
        layers
    }
    // $[x_i^e] f(x_1, .., x_n)$.
    // In particular, $e=0$ gives $f(x_i = 0)$ and $e=1$ gives $\partial_i f$.
    pub fn extract_axis<T: Clone, const E: usize>(f: &[T], i: usize) -> Vec<T> {
        assert!(f.len().is_power_of_two());
        let n = f.len().ilog2() as usize;
        assert!(i < n);
        (f.chunks_exact(1 << i).skip(E).step_by(2).flatten())
            .cloned()
            .collect()
    }
    // overwrite [x_i^e] f
    pub fn overwrite_axis<T: Clone, const E: usize>(f: &mut [T], sub: &[T], i: usize) {
        assert!(f.len().is_power_of_two());
        assert!(f.len() == sub.len() * 2);
        let n = f.len().ilog2() as usize;
        assert!(i < n);
        (f.chunks_exact_mut(1 << i).skip(E).step_by(2).flatten())
            .zip(sub)
            .for_each(|(x, y)| *x = y.clone());
    }
    // todo: power projection
}
