pub mod geometry {
    pub mod half_plane {
        use std::collections::VecDeque;

        use super::*;

        // A half plane, defined as { x : n dot x < s }
        #[derive(Debug, Clone, PartialEq, Eq)]
        pub struct HalfPlane<T> {
            pub normal_outward: Point<T>,
            pub shift: T,
        }

        impl<T: Scalar> HalfPlane<T> {
            pub fn left_side(s: Point<T>, e: Point<T>) -> Self {
                debug_assert!(s != e);
                let normal_outward = (s - e).rot();
                let shift = normal_outward.dot(s);
                HalfPlane {
                    normal_outward,
                    shift,
                }
            }

            // Avoid division at all costs
            pub fn inter_frac(&self, other: &Self) -> Option<(Point<T>, T)> {
                let det = self.normal_outward.cross(other.normal_outward);
                let x_mul_det =
                    self.shift * other.normal_outward[1] - other.shift * self.normal_outward[1];
                let y_mul_det =
                    self.normal_outward[0] * other.shift - other.normal_outward[0] * self.shift;
                (det != T::zero()).then(|| ([x_mul_det, y_mul_det].into(), det))
            }

            pub fn antiparallel(&self, other: &Self) -> bool {
                self.normal_outward.cross(other.normal_outward) == T::zero()
                    && self.normal_outward.dot(other.normal_outward) < T::zero()
            }

            pub fn contains_frac(&self, (numer, denom): (Point<T>, T)) -> bool {
                self.normal_outward.dot(numer) * denom.signum() < self.shift * denom.abs()
            }
        }

        pub fn bbox<T: Scalar>(
            bottom_left: Point<T>,
            top_right: Point<T>,
        ) -> impl Iterator<Item = HalfPlane<T>> {
            let bottom_right = Point::new(top_right[0], bottom_left[1]);
            let top_left = Point::new(bottom_left[0], top_right[1]);
            [
                HalfPlane::left_side(bottom_left, bottom_right),
                HalfPlane::left_side(bottom_right, top_right),
                HalfPlane::left_side(top_right, top_left),
                HalfPlane::left_side(top_left, bottom_left),
            ]
            .into_iter()
        }

        pub fn intersection<T: Scalar>(
            half_planes: impl IntoIterator<Item = HalfPlane<T>>,
            bottom_left: Point<T>,
            top_right: Point<T>,
        ) -> VecDeque<HalfPlane<T>> {
            let mut half_planes: Vec<_> = half_planes
                .into_iter()
                .chain(bbox(bottom_left, top_right)) // Handling caseworks without a bbox is a huge pain.
                .collect();
            half_planes.sort_unstable_by_key(|h| (Angle(h.normal_outward), h.shift));
            half_planes.dedup_by_key(|h| Angle(h.normal_outward)); // Dedup parallel half planes

            let mut half_planes = half_planes.into_iter();

            let mut inter = VecDeque::new();
            inter.extend(half_planes.next());
            inter.extend(half_planes.next());

            for h in half_planes {
                while inter.len() >= 2 {
                    let [l, m] = [&inter[inter.len() - 2], &inter[inter.len() - 1]];

                    if l.inter_frac(m).map_or(true, |p| h.contains_frac(p)) {
                        break;
                    }
                    inter.pop_back();
                }

                while inter.len() >= 2 {
                    let [l, m] = [&inter[0], &inter[1]];
                    if l.inter_frac(m).map_or(true, |p| h.contains_frac(p)) {
                        break;
                    }
                    inter.pop_front();
                }

                let l = &inter[inter.len() - 1];
                if h.antiparallel(l) {
                    let det = h.shift * (l.normal_outward[0].abs() + l.normal_outward[1].abs())
                        - l.shift
                            * (h.normal_outward[0] * l.normal_outward[0].signum()
                                + h.normal_outward[1] * l.normal_outward[1].signum());
                    if det <= T::zero() {
                        // Exclude boundary
                        return Default::default();
                    }
                    //if det < 0 {
                    //    //Include boundary
                    //    return Default::default();
                    //}
                }

                inter.push_back(h);
            }

            while inter.len() >= 3 {
                let [l, m, h] = [&inter[inter.len() - 2], &inter[inter.len() - 1], &inter[0]];
                if l.inter_frac(m).map_or(true, |p| h.contains_frac(p)) {
                    break;
                }
                inter.pop_back();
            }
            while inter.len() >= 3 {
                let [l, m, h] = [&inter[inter.len() - 1], &inter[0], &inter[1]];
                if l.inter_frac(m).map_or(true, |p| h.contains_frac(p)) {
                    break;
                }
                inter.pop_front();
            }

            if inter.len() < 3 {
                return Default::default();
            }

            inter
        }
    }
}
