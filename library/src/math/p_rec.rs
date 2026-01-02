mod p_rec {
    use super::algebra::*;

    pub mod linalg {
        use crate::algebra::{Field, SemiRing};

        #[derive(Debug, Clone, PartialEq, Eq)]
        pub struct Matrix<T> {
            pub r: usize,
            pub c: usize,
            elem: Vec<T>,
        }

        impl<T> std::ops::Index<[usize; 2]> for Matrix<T> {
            type Output = T;

            #[inline(always)]
            fn index(&self, index: [usize; 2]) -> &Self::Output {
                &self.elem[index[0] * self.c + index[1]]
            }
        }

        impl<T> std::ops::IndexMut<[usize; 2]> for Matrix<T> {
            #[inline(always)]
            fn index_mut(&mut self, index: [usize; 2]) -> &mut Self::Output {
                &mut self.elem[index[0] * self.c + index[1]]
            }
        }

        impl<T> Matrix<T> {
            pub fn new(r: usize, elem: impl IntoIterator<Item = T>) -> Self {
                let elem: Vec<_> = elem.into_iter().collect();
                let c = if r == 0 { 0 } else { elem.len() / r };
                assert_eq!(r * c, elem.len());

                Self { r, c, elem }
            }

            pub fn map<S>(self, f: impl FnMut(T) -> S) -> Matrix<S> {
                Matrix {
                    r: self.r,
                    c: self.c,
                    elem: self.elem.into_iter().map(f).collect(),
                }
            }

            pub fn swap_rows(&mut self, i: usize, j: usize) {
                for k in 0..self.c {
                    self.elem.swap(i * self.c + k, j * self.c + k);
                }
            }
        }

        impl<T: SemiRing + Copy> Matrix<T> {
            pub fn with_size(r: usize, c: usize) -> Self {
                Self {
                    r,
                    c,
                    elem: vec![T::zero(); r * c],
                }
            }

            pub fn apply(&self, rhs: &[T]) -> Vec<T> {
                assert_eq!(self.c, rhs.len());

                let mut res = vec![T::zero(); self.r];
                for i in 0..self.r {
                    for j in 0..self.c {
                        res[i] += self[[i, j]] * rhs[j];
                    }
                }
                res
            }
        }

        // Gaussian elimination
        pub fn rref<T: Field + PartialEq + Copy>(mat: &Matrix<T>) -> (usize, T, Matrix<T>) {
            let (r, c) = (mat.r, mat.c);
            let mut mat = mat.clone();
            let mut det = T::one();

            let mut rank = 0;
            for j0 in 0..c {
                let Some(pivot) = (rank..r).find(|&j| mat[[j, j0]] != T::zero()) else {
                    continue;
                };
                // let Some(pivot) = (rank..n_rows)
                //     .filter(|&j| jagged[j][c] != Frac64::zero())
                //     .max_by_key(|&j| jagged[j][c].abs())
                // else {
                //     continue;
                // };

                if pivot != rank {
                    mat.swap_rows(rank, pivot);
                    det = -det;
                }

                det *= mat[[rank, j0]];
                let inv_x = mat[[rank, j0]].inv();
                for j in 0..c {
                    mat[[rank, j]] *= inv_x;
                }

                for i in 0..r {
                    if i == rank {
                        continue;
                    }

                    let coeff = mat[[i, j0]];
                    for j in 0..c {
                        let f = mat[[rank, j]] * coeff;
                        mat[[i, j]] -= f;
                    }
                }
                rank += 1;
            }

            if rank != mat.r {
                det = T::zero();
            };

            (rank, det, mat)
        }

        pub fn sample_kernel<T: Field + PartialEq + Copy>(mat: &Matrix<T>) -> Option<Vec<T>> {
            let (rank, _det, rmat) = rref(mat);
            if rank == rmat.c {
                return None;
            }

            let mut pivot_cols = vec![];
            let mut is_pivot = vec![false; rmat.c];
            for i in 0..rank {
                let j = (0..rmat.c).find(|&j| rmat[[i, j]] != T::zero()).unwrap();
                pivot_cols.push(j);
                is_pivot[j] = true;
            }

            let j_free = (0..rmat.c).find(|&j| !is_pivot[j]).unwrap();

            let mut res = vec![T::zero(); rmat.c];
            res[j_free] = T::one();
            for (i, &p) in pivot_cols.iter().enumerate() {
                res[p] = -rmat[[i, j_free]];
            }

            Some(res)
        }
    }

    // Infer p-recurrsive relation of the form by gaussian elimination.
    // $$\sum_{a=0...d} \sum_{b=0...r} c_{ab} n^a f_{i-b} = 0$$
    pub fn approx_by_guass_dr<M: SemiRing + Field + PartialEq + Copy + From<u32>>(
        init: &[M],
        d: usize,
        r: usize,
    ) -> Option<Vec<Vec<M>>> {
        let mut elem = vec![];

        let dim = (d + 1) * (r + 1);
        for n in r..init.len() {
            let view = &init[n - r..=n];

            let n = M::from(n as u32);
            let mut n_pow = vec![M::one(); d + 1];
            for i in 1..=d {
                n_pow[i] = n_pow[i - 1] * n;
            }

            for a in 0..=d {
                for b in 0..=r {
                    elem.push(n_pow[a] * view[r - b]);
                }
            }
        }
        if r >= init.len() {
            elem.extend((0..dim).map(|_| M::zero()));
        }

        let mat = linalg::Matrix::new(elem.len() / dim, elem);
        let ker = linalg::sample_kernel(&mat)?;

        let mut coeff = vec![vec![M::zero(); d + 1]; r + 1];
        for a in 0..=d {
            for b in 0..=r {
                coeff[b][a] = ker[a * (r + 1) + b];
            }
        }
        Some(coeff)
    }

    pub fn approx_by_guass<M: SemiRing + Field + Copy>(init: &[M]) -> Option<Vec<Vec<M>>> {
        // exponential search to find one feasible (d,r)
        // binary search to minimize d and r
        todo!()
    }

    pub fn step<M: SemiRing + Field + PartialEq + Copy>(init: &[M], n: M, coeff: &[Vec<M>]) -> M {
        debug_assert!(coeff.len() >= 1);
        let r = coeff.len() - 1;
        let d_p1 = coeff[0].len();

        let mut numer = M::zero();
        let mut denom = M::zero();

        let mut n_pow = vec![M::one(); d_p1];
        for i in 1..d_p1 {
            n_pow[i] = n_pow[i - 1] * n;
        }

        for (b, c) in coeff[0].iter().enumerate() {
            denom += *c * n_pow[b];
        }
        for a in 1..=r {
            for (b, c) in coeff[a].iter().enumerate() {
                numer += *c * n_pow[b] * init[init.len() - a];
            }
        }
        assert!(denom != M::zero());

        -numer / denom
    }
}
