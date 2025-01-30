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

fn partition_point<P>(mut left: i128, mut right: i128, mut pred: P) -> i128
where
    P: FnMut(i128) -> bool,
{
    while left < right {
        let mid = left + (right - left) / 2;
        if pred(mid) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    left
}

pub mod cht {
    // Max-hull of lines with increasing slopes
    type V = i128;
    type Tag = u32;

    pub struct Line {
        pub slope: V,
        pub intercept: V,
        pub tag: Tag,
    }

    impl Line {
        pub fn new(slope: V, intercept: V, tag: Tag) -> Self {
            Self {
                slope,
                intercept,
                tag,
            }
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

        pub fn eval(&self, x: &V) -> (V, Tag) {
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
            let l = &self.lines[left];
            (l.eval(x), l.tag)
        }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let k: usize = input.value();

    let scale = 2;
    let mut prefix = vec![0; n + 1];
    for i in 1..=n {
        prefix[i] = prefix[i - 1] + scale * input.value::<i64>();
    }

    let min_cost = |slope: i128| {
        let mut neg_hull = cht::MonotoneStack::new();
        neg_hull.insert(cht::Line::new(0, 0, 0));

        let sq = |x: i128| x * x;

        let mut prev = vec![0; n + 1];
        let mut dp_last = 0;
        for i in 1..=n {
            let (neg_dp_j, j) = neg_hull.eval(&(prefix[i] as i128));
            prev[i] = j;
            dp_last = -neg_dp_j + sq(prefix[i] as i128) + slope;

            neg_hull.insert(cht::Line::new(
                2 * prefix[i] as i128,
                -(dp_last + sq(prefix[i] as i128)),
                i as u32,
            ));
        }

        let mut u = n;
        let mut path = vec![];
        while u > 0 {
            path.push(u);
            u = prev[u] as usize;
        }

        dp_last += slope * path.len() as i128;
        (dp_last, path)
    };

    let opt = partition_point(-1, 2.500001e19 as i128, |slope| {
        min_cost(scale as i128 * slope + 1).1.len() > k
    });
    let (_, mut path_lower) = min_cost(scale as i128 * opt + 1);
    let (_, mut path_upper) = min_cost(scale as i128 * opt - 1);
    let k_lower = path_lower.len();
    let _k_upper = path_upper.len();
    assert!(path_lower.len() <= k && k <= path_upper.len());

    path_lower.push(0);
    path_lower.reverse();
    path_lower.pop();
    path_upper.push(0);
    path_upper.reverse();
    path_upper.pop();

    writeln!(output, "Yes").unwrap();
    if k_lower == k {
        for u in &path_lower[1..] {
            write!(output, "{} ", u).unwrap();
        }
    } else {
        let dk = k - k_lower;
        let mut path = vec![];
        for i in 1..k_lower {
            let j = i + dk;
            let ei = [path_lower[i - 1], path_lower[i]];
            let ej = [path_upper[j - 1], path_upper[j]];
            if ei[0] <= ej[0] && ej[1] <= ei[1] {
                let (_pi, si) = path_lower.split_at(i);
                let (pj, _sj) = path_upper.split_at(j);
                path = pj.iter().chain(si).copied().collect::<Vec<_>>();
                break;
            }
        }

        assert!(path.len() == k);
        for u in &path[1..] {
            write!(output, "{} ", u).unwrap();
        }
    }
    writeln!(output).unwrap();

    // println!("{:?}", path_lower);
    // println!("{:?}", path_upper);
}
