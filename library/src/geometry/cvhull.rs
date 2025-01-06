mod geometry {
    pub mod cvhull {
        use super::*;

        pub fn dim2<T: Scalar>(points: &mut [Point<T>]) -> Vec<Point<T>> {
            // monotone chain algorithm
            let n = points.len();
            if n <= 1 {
                return points.to_vec();
            }
            assert!(n >= 2);

            points.sort_unstable_by(|&p, &q| p.partial_cmp(&q).unwrap());

            let mut lower = Vec::new();
            let mut upper = Vec::new();
            for &p in points.iter() {
                while matches!(lower.as_slice(), &[.., l1, l2] if signed_area(p, l1, l2) <= T::zero())
                {
                    lower.pop();
                }
                lower.push(p);
            }
            for &p in points.iter().rev() {
                while matches!(upper.as_slice(), &[.., l1, l2] if signed_area(p, l1, l2) <= T::zero())
                {
                    upper.pop();
                }
                upper.push(p);
            }
            lower.pop();
            upper.pop();

            lower.extend(upper);
            lower
        }

        pub fn area<I>(points: I) -> f64
        where
            I: IntoIterator<Item = [f64; 2]>,
            I::IntoIter: Clone,
        {
            let mut area: f64 = 0.0;
            let points = points.into_iter();
            let points_shifted = points.clone().skip(1).chain(points.clone().next());
            for ([x1, y1], [x2, y2]) in points.zip(points_shifted) {
                area += x1 * y2 - x2 * y1;
            }
            area = (area / 2.0).abs();
            area
        }

        pub fn rotating_calipers<T: Scalar>(
            hull: &[Point<T>],
            mut yield_antipodals: impl FnMut(usize, usize, Ordering),
        ) {
            use std::cmp::Ordering::*;
            let n_verts = hull.len();
            let inc = |i| (i + 1) % n_verts;
            let signed_area = |i, j, k| signed_area::<T>(hull[i], hull[j], hull[k]);
            let compare_segments =
                |i, j| signed_area(i, inc(i), inc(j)).cmp(&signed_area(i, inc(i), j));

            let mut i = 0;
            let mut j = (1..n_verts)
                .find(|&j| compare_segments(i, j).is_le())
                .unwrap();
            let i_last = j;
            let j_last = i;
            while (i, j) != (i_last, j_last) {
                let angle_relation = compare_segments(i, j);
                yield_antipodals(i, j, angle_relation);
                match angle_relation {
                    Less => i = inc(i),
                    Greater => j = inc(j),
                    Equal => {
                        yield_antipodals(i, inc(j), Less);
                        yield_antipodals(inc(i), j, Greater);
                        i = inc(i);
                        j = inc(j);
                    }
                }
            }
        }
    }
}
