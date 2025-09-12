pub mod linalg {
    use crate::algebra::SemiRing;

    use super::algebra::Field;
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Matrix<T> {
        pub r: usize,
        pub c: usize,
        elem: Vec<T>,
    }

    // impl<T> std::ops::Index<usize> for Matrix<T> {
    //     type Output = [T];

    //     #[inline(always)]
    //     fn index(&self, index: usize) -> &Self::Output {
    //         &self.elem[index * self.c..][..self.c]
    //     }
    // }

    // impl<T> std::ops::IndexMut<usize> for Matrix<T> {
    //     #[inline(always)]
    //     fn index_mut(&mut self, index: usize) -> &mut Self::Output {
    //         &mut self.elem[index * self.c..][..self.c]
    //     }
    // }

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

            let mut res = vec![T::zero(); self.c];
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
}
