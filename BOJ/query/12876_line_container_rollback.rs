use std::{collections::BTreeSet, io::Write, ops::Range};

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
            self.token()
                .parse()
                .map_err(|e| {
                    eprintln!("{:?}", e);
                    e
                })
                .unwrap()
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
    pub mod rollback {
        // Line Container for Convex hull trick
        // adapted from KACTL
        // https://github.com/kth-competitive-programming/kactl/blob/main/content/data-structures/LineContainer.h

        use std::{cmp::Ordering, collections::BTreeSet};
        type V = i64;
        const NEG_INF: V = V::MIN;
        const INF: V = V::MAX;

        fn div_floor(x: V, y: V) -> V {
            x / y - (((x < 0) ^ (y < 0)) && x % y != 0) as V
        }

        #[derive(Clone, Debug)]
        struct Line {
            slope: V,
            intercept: V,
            right_end: V,
            point_query: bool, // Bypass BTreeMap's API with some additional runtime cost
        }

        #[derive(Debug)]
        enum History {
            Init,
            Add(Line),
            Remove(Line),
            Update(Line),
        }

        pub struct LineContainer {
            lines: BTreeSet<Line>,
            history: Vec<History>,
            stack: Vec<Line>,
        }

        impl Line {
            fn new(slope: V, intercept: V) -> Self {
                Self {
                    slope,
                    intercept,
                    right_end: INF,
                    point_query: false,
                }
            }

            fn point_query(x: V) -> Self {
                Self {
                    slope: 0,
                    intercept: 0,
                    right_end: x,
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
                    self.right_end.cmp(&other.right_end)
                }
            }
        }

        impl LineContainer {
            pub fn new() -> Self {
                Self {
                    lines: Default::default(),
                    history: vec![],
                    stack: Default::default(),
                }
            }

            pub fn push(&mut self, slope: V, intercept: V) {
                let mut y = Line::new(slope, intercept);
                self.history.push(History::Init);

                let to_remove = &mut self.stack;
                for z in self.lines.range(&y..) {
                    y.right_end = y.inter(&z);
                    if y.right_end < z.right_end {
                        break;
                    }
                    to_remove.push(z.clone());
                }

                let mut r = self.lines.range(..&y).rev();
                if let Some(x) = r.next() {
                    let x_right_end = x.inter(&y);
                    if !(x_right_end < y.right_end) {
                        return;
                    }

                    let mut prev = x;
                    let mut prev_right_end = x_right_end;
                    for x in r {
                        if x.right_end < prev_right_end {
                            break;
                        }
                        to_remove.push(prev.clone());

                        prev = x;
                        prev_right_end = x.inter(&y);
                    }
                    let prev = prev.clone();
                    assert!(self.lines.remove(&prev));
                    self.history.push(History::Update(prev.clone()));

                    let mut modified = prev;
                    modified.right_end = prev_right_end;
                    assert!(self.lines.insert(modified));
                }
                for x in to_remove.drain(..) {
                    assert!(self.lines.remove(&x));
                    self.history.push(History::Remove(x));
                }

                if let Some(old) = self.lines.get(&y) {
                    self.history.push(History::Update(old.clone()));
                    self.lines.remove(&y);
                } else {
                    self.history.push(History::Add(y.clone()));
                }
                self.lines.insert(y);
            }

            pub fn query(&self, x: V) -> Option<V> {
                let l = self.lines.range(Line::point_query(x)..).next()?;
                Some(l.slope * x + l.intercept)
            }

            pub fn pop(&mut self) -> bool {
                if self.history.is_empty() {
                    return false;
                }
                loop {
                    match self.history.pop().unwrap() {
                        History::Init => {
                            return true;
                        }
                        History::Add(x) => {
                            self.lines.remove(&x);
                        }
                        History::Remove(x) => {
                            self.lines.insert(x);
                        }
                        History::Update(x) => {
                            self.lines.remove(&x);
                            self.lines.insert(x);
                        }
                    }
                }
            }
        }
    }
}

fn partition_in_place<T>(xs: &mut [T], mut pred: impl FnMut(&T) -> bool) -> (&mut [T], &mut [T]) {
    let n = xs.len();
    let mut i = 0;
    for j in 0..n {
        if pred(&xs[j]) {
            xs.swap(i, j);
            i += 1;
        }
    }
    xs.split_at_mut(i)
}

fn dnc(
    hull: &mut cht::rollback::LineContainer,
    intervals: &mut [(Range<u32>, i64, i64)],
    queries: &[Option<i64>],
    ans: &mut [Option<i64>],
    time_range: Range<u32>,
) {
    debug_assert!(time_range.start < time_range.end);
    let (intervals, _) = partition_in_place(intervals, |(interval, _, _)| {
        !(interval.end <= time_range.start || time_range.end <= interval.start)
    });
    let (full, partial) = partition_in_place(intervals, |(interval, _, _)| {
        interval.start <= time_range.start && time_range.end <= interval.end
    });

    for &(_, a, b) in full.iter() {
        hull.push(a, b);
    }

    if time_range.start + 1 == time_range.end {
        assert!(partial.is_empty());
        let i = time_range.start as usize;
        if let Some(x) = queries[i] {
            ans[i] = hull.query(x);
        }
    } else {
        let mid = (time_range.start + time_range.end) / 2;
        dnc(hull, partial, queries, ans, time_range.start..mid);
        dnc(hull, partial, queries, ans, mid..time_range.end);
    }

    for _ in 0..full.len() {
        hull.pop();
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let mut intervals = vec![];
    let mut added_intervals = vec![None; n];
    let mut point_queries = vec![None; n as usize];
    let mut active = BTreeSet::new();
    for time in 0..n as u32 {
        let cmd = input.token();
        match cmd {
            "1" => {
                let a: i64 = input.value();
                let b: i64 = input.value();
                added_intervals[time as usize] = Some((a, b));
                active.insert((time, (a, b)));
            }
            "2" => {
                let start = input.value::<u32>() - 1;
                let (a, b) = added_intervals[start as usize].unwrap();
                intervals.push((start..time, a, b));
                active.remove(&(start, (a, b)));
            }
            "3" => {
                let x: i64 = input.value();
                point_queries[time as usize] = Some(x);
            }
            _ => panic!(),
        }
    }
    for (start, (a, b)) in active {
        intervals.push((start..n as u32, a, b));
    }

    let mut ans = vec![None; n];
    dnc(
        &mut cht::rollback::LineContainer::new(),
        &mut intervals,
        &point_queries,
        &mut ans,
        0..n as u32,
    );

    for i in 0..n {
        if point_queries[i].is_some() {
            if let Some(y) = ans[i] {
                writeln!(output, "{}", y).unwrap();
            } else {
                writeln!(output, "EMPTY").unwrap();
            }
        }
    }
}
