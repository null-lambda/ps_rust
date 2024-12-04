use std::io::Write;
mod simple_io {
    pub struct InputAtOnce<'a> {
        _buf: String,
        iter: std::str::SplitAsciiWhitespace<'a>,
    }

    impl<'a> InputAtOnce<'a> {
        pub fn token(&mut self) -> &'a str {
            self.iter.next().unwrap_or_default()
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().unwrap()
        }
    }

    pub fn stdin<'a>() -> InputAtOnce<'a> {
        let _buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let iter = _buf.split_ascii_whitespace();
        let iter = unsafe { std::mem::transmute(iter) };
        InputAtOnce { _buf, iter }
    }

    pub fn stdout() -> std::io::BufWriter<std::io::Stdout> {
        std::io::BufWriter::new(std::io::stdout())
    }
}

mod cht {
    // Line Container for Convex hull trick
    // adapted from KACTL's LineContainer.h
    // https://github.com/kth-competitive-programming/kactl/blob/main/content/data-structures/LineContainer.h

    // For a further performance improvement, we require a BST that supports:
    // (1) partition_point with custom predicates (query by slope, query by point)
    // (2) Removal-while-iteration

    use std::{cell::Cell, cmp::Ordering, collections::BTreeMap};
    type V = i64;
    const NEG_INF: V = V::MIN;
    const INF: V = V::MAX;

    #[derive(Clone, Debug)]
    struct Key {
        slope: V,
        right_end: Cell<V>,
        point_query: bool, // Bypass BTreeMap's API
    }

    pub struct LineContainer {
        lines: BTreeMap<Key, V>,
    }

    impl Key {
        fn slope(slope: V) -> Self {
            Self {
                slope,
                right_end: Cell::new(INF),
                point_query: false,
            }
        }

        fn point(x: V) -> Self {
            Self {
                slope: 0,
                right_end: Cell::new(x),
                point_query: true,
            }
        }
    }

    impl PartialEq for Key {
        fn eq(&self, other: &Self) -> bool {
            self.cmp(other) == Ordering::Equal
        }
    }

    impl Eq for Key {}

    impl PartialOrd for Key {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.cmp(other))
        }
    }

    impl Ord for Key {
        fn cmp(&self, other: &Self) -> std::cmp::Ordering {
            if !self.point_query && !other.point_query {
                self.slope.cmp(&other.slope)
            } else {
                self.right_end.get().cmp(&other.right_end.get())
            }
        }
    }

    fn div_floor(x: V, y: V) -> V {
        x / y - (x ^ y < 0 && x % y == 0) as V
    }

    fn inter(slope0: V, intercept0: V, slope1: V, intercept1: V) -> V {
        if slope0 != slope1 {
            div_floor(intercept0 - intercept1, slope1 - slope0)
        } else if intercept0 > intercept1 {
            INF
        } else {
            NEG_INF
        }
    }

    fn update_lhs(lhs: (&Key, &V), rhs: (&Key, &V)) {
        lhs.0
            .right_end
            .set(inter(lhs.0.slope, *lhs.1, rhs.0.slope, *rhs.1));
    }

    impl LineContainer {
        pub fn new() -> Self {
            Self {
                lines: Default::default(),
            }
        }

        pub fn insert(&mut self, slope: V, intercept: V) {
            let y = Key::slope(slope);
            let y_intercept = intercept;

            while let Some((z, z_intercept)) = self.lines.range(&y..).next() {
                update_lhs((&y, &y_intercept), (z, z_intercept));
                if y.right_end < z.right_end {
                    break;
                }
                self.lines.remove(&z.clone());
            }

            let mut r = self.lines.range(..&y);
            if let Some((x, x_intercept)) = r.next_back() {
                let old = x.right_end.get();
                update_lhs((x, x_intercept), (&y, &y_intercept));
                if !(x.right_end < y.right_end) {
                    x.right_end.set(old);
                    return;
                }

                loop {
                    let mut r = self.lines.range(..&y);
                    let Some((x, _)) = r.next_back() else {
                        break;
                    };
                    let Some((w, w_intercept)) = r.next_back() else {
                        break;
                    };
                    if w.right_end < x.right_end {
                        break;
                    }
                    update_lhs((w, w_intercept), (&y, &y_intercept));
                    self.lines.remove(&x.clone());
                }
            }

            self.lines.entry(y).or_insert(y_intercept);
        }

        pub fn query(&self, x: V) -> Option<V> {
            let (key, intercept) = self.lines.range(Key::point(x)..).next()?;
            Some(key.slope * x + intercept)
        }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let mut hull = cht::LineContainer::new();
    for _ in 0..input.value() {
        match input.token() {
            "1" => {
                hull.insert(input.value(), input.value());
            }
            "2" => {
                writeln!(output, "{}", hull.query(input.value()).unwrap()).unwrap();
            }
            _ => panic!(),
        }
    }
}
