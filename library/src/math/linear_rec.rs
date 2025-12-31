pub mod linear_rec {
    use super::poly::Poly;
    use crate::{
        algebra::{CommRing, Field},
        conv::Conv,
    };

    pub fn berlekamp_massey<T: CommRing>(_seq: &[T]) -> Vec<T> {
        unimplemented!()
    }

    pub fn next<T: CommRing>(recurrence: &[T], init: &[T]) -> T {
        let l = recurrence.len();
        let n = init.len();
        assert!(n >= l);
        let mut value = recurrence[0].clone() * init[n - 1].clone();
        for i in 1..l {
            value += recurrence[i].clone() * init[n - 1 - i].clone();
        }
        value
    }

    pub fn nth_by_ntt<T: Conv + Field + From<u32>>(recurrence: &[T], init: &[T], n: u64) -> T {
        let l = recurrence.len();
        assert!(l >= 1 && l == init.len());

        let mut q = Vec::with_capacity(l + 1);
        q.push(T::one());
        for c in recurrence.iter().cloned() {
            q.push(-c);
        }
        let q = Poly::new(q);
        let p = (Poly::new(init.to_vec()) * q.clone()).mod_xk(l);

        Poly::nth_of_frac(p, q, n)
    }
}
