pub mod linalg {
    use super::algebra::Field;
    #[derive(Debug, Clone, PartialEq, Eq)]
    pub struct Matrix<T> {
        pub n_rows: usize,
        pub n_cols: usize,
        pub data: Vec<T>,
    }

    // gaussian elimination
    pub fn rref<T: Field + PartialEq + Copy>(mat: &Matrix<T>) -> (usize, T, Matrix<T>) {
        let Matrix {
            n_rows,
            n_cols,
            ref data,
        } = *mat;

        let mut jagged: Vec<Vec<_>> = data.chunks_exact(n_cols).map(Vec::from).collect();
        let mut det = T::one();

        let mut rank = 0;
        for c in 0..n_cols {
            let Some(pivot) = (rank..n_rows).find(|&j| jagged[j][c] != T::zero()) else {
                continue;
            };
            // let Some(pivot) = (rank..n_rows)
            //     .filter(|&j| jagged[j][c] != Frac64::zero())
            //     .max_by_key(|&j| jagged[j][c].abs())
            // else {
            //     continue;
            // };
            if pivot != rank {
                jagged.swap(rank, pivot);
                det = -det;
            }

            det *= jagged[rank][c];
            let inv_x = jagged[rank][c].inv();
            for j in 0..n_cols {
                jagged[rank][j] *= inv_x;
            }

            for i in 0..n_rows {
                if i == rank {
                    continue;
                }

                let coeff = jagged[i][c];
                for j in 0..n_cols {
                    let f = jagged[rank][j] * coeff;
                    jagged[i][j] -= f;
                }
            }
            rank += 1;
        }

        if rank != mat.n_rows {
            det = T::zero();
        };

        let mat = Matrix {
            n_rows,
            n_cols,
            data: jagged.into_iter().flat_map(|row| row.into_iter()).collect(),
        };
        (rank, det, mat)
    }
}
