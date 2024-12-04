mod cht {
    // Line Container for Convex hull trick
    // adapted from KACTL
    // https://github.com/kth-competitive-programming/kactl/blob/main/content/data-structures/LineContainer.h

    use std::{cell::Cell, cmp::Ordering, collections::BTreeSet};
    type V = i32;
    const NEG_INF: V = V::MIN;
    const INF: V = V::MAX;

    fn div_floor(x: V, y: V) -> V {
        x / y - (((x < 0) ^ (y < 0)) && x % y == 0) as V
    }

    #[derive(Clone, Debug)]
    struct Line {
        slope: V,
        intercept: V,
        right_end: Cell<V>,
        point_query: bool, // Bypass BTreeMap's API with some additional runtime cost
    }

    pub struct LineContainer {
        lines: BTreeSet<Line>,
        stack: Vec<Line>,
    }

    impl Line {
        fn new(slope: V, intercept: V) -> Self {
            Self {
                slope,
                intercept,
                right_end: Cell::new(INF),
                point_query: false,
            }
        }

        fn point_query(x: V) -> Self {
            Self {
                slope: 0,
                intercept: 0,
                right_end: Cell::new(x),
                point_query: true,
            }
        }

        fn inter(&self, other: &Line) -> V {
            if self.slope != other.slope {
                div_floor(self.intercept - other.intercept, other.slope - self.slope)
            } else if self.intercept > other.intercept {
                INF
            } else {
                NEG_INF
            }
        }
    }

    impl PartialEq for Line {
        fn eq(&self, other: &Self) -> bool {
            self.cmp(other) == Ordering::Equal
        }
    }

    impl Eq for Line {}

    impl PartialOrd for Line {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for Line {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            if !self.point_query && !other.point_query {
                self.slope.cmp(&other.slope)
            } else {
                self.right_end.get().cmp(&other.right_end.get())
            }
        }
    }

    impl LineContainer {
        pub fn new() -> Self {
            Self {
                lines: Default::default(),
                stack: Default::default(),
            }
        }

        pub fn insert(&mut self, slope: V, intercept: V) {
            let y = Line::new(slope, intercept);

            let to_remove = &mut self.stack;
            for z in self.lines.range(&y..) {
                y.right_end.set(y.inter(z));
                if y.right_end < z.right_end {
                    break;
                }
                to_remove.push(z.clone());
            }

            let mut r = self.lines.range(..&y).rev();
            if let Some(x) = r.next() {
                let new_x_right_end = x.inter(&y);
                if !(new_x_right_end < y.right_end.get()) {
                    return;
                }
                x.right_end.set(new_x_right_end);

                let mut x_prev = x;
                for x in r {
                    if x.right_end < x_prev.right_end {
                        break;
                    }
                    x.right_end.set(x.inter(&y));
                    to_remove.push(x_prev.clone());

                    x_prev = x;
                }
            }

            for x in to_remove.drain(..) {
                self.lines.remove(&x);
            }
            self.lines.insert(y);
        }

        pub fn query(&self, x: V) -> Option<V> {
            let l = self.lines.range(Line::point_query(x)..).next()?;
            Some(l.slope * x + l.intercept)
        }
    }
}
