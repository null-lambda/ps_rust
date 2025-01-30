pub mod cht {
    // Max-hull of lines with increasing slopes
    type V = i64;

    pub struct Line {
        pub slope: V,
        pub intercept: V,
    }

    impl Line {
        pub fn new(slope: V, intercept: V) -> Self {
            Self { slope, intercept }
        }

        pub fn eval(&self, x: &V) -> V {
            self.slope * x + self.intercept
        }

        fn should_remove(&self, lhs: &Self, rhs: &Self) -> bool {
            debug_assert!(lhs.slope < self.slope && self.slope <= rhs.slope);
            if self.slope == rhs.slope {
                self.intercept <= rhs.intercept
            } else {
                (rhs.slope - self.slope) * (self.intercept - lhs.intercept)
                    <= (self.slope - lhs.slope) * (rhs.intercept - self.intercept)
            }
        }
    }

    pub struct MonotoneStack {
        lines: Vec<Line>,
    }

    impl MonotoneStack {
        pub fn new() -> Self {
            Self { lines: vec![] }
        }

        pub fn insert(&mut self, line: Line) {
            while self.lines.len() >= 2 {
                let n = self.lines.len();
                if self.lines[n - 1].should_remove(&self.lines[n - 2], &line) {
                    self.lines.pop();
                } else {
                    break;
                }
            }
            self.lines.push(line);
        }

        pub fn eval(&self, x: &V) -> V {
            assert!(!self.lines.is_empty());
            let mut left = 0;
            let mut right = self.lines.len() - 1;
            while left < right {
                let mid = left + right >> 1;
                if self.lines[mid].eval(x) >= self.lines[mid + 1].eval(x) {
                    right = mid;
                } else {
                    left = mid + 1;
                }
            }
            self.lines[left].eval(x)
        }
    }
}
