use std::io::Write;

mod fast_io {
    use std::fs::File;
    use std::io::BufWriter;
    use std::os::unix::io::FromRawFd;

    pub struct InputAtOnce {
        _buf: &'static str,
        iter: std::str::SplitAsciiWhitespace<'static>,
    }

    impl InputAtOnce {
        pub fn token(&mut self) -> &'static str {
            self.iter.next().unwrap_or_default()
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().unwrap()
        }
    }

    extern "C" {
        fn mmap(addr: usize, length: usize, prot: i32, flags: i32, fd: i32, offset: i64)
            -> *mut u8;
        fn fstat(fd: i32, stat: *mut usize) -> i32;
    }

    pub fn stdin() -> InputAtOnce {
        let mut stat = [0; 18];
        unsafe { fstat(0, (&mut stat).as_mut_ptr()) };
        let _buf = unsafe { mmap(0, stat[6], 1, 2, 0, 0) };
        let _buf =
            unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(_buf, stat[6])) };
        let iter = _buf.split_ascii_whitespace();
        InputAtOnce { _buf, iter }
    }

    pub fn stdout() -> BufWriter<File> {
        let stdout = unsafe { File::from_raw_fd(1) };
        BufWriter::new(stdout)
    }
}

mod cht {
    // Line Container for Convex hull trick
    // adapted from KACTL
    // https://github.com/kth-competitive-programming/kactl/blob/main/content/data-structures/LineContainer.h

    use std::{cell::Cell, cmp::Ordering, collections::BTreeSet};
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
            for x in to_remove.drain(..) {
                self.lines.remove(&x);
            }

            let mut r = self.lines.range(..&y).rev();
            if let Some(x) = r.next() {
                let x_right_end = x.inter(&y);
                if !(x_right_end < y.right_end.get()) {
                    return;
                }

                let mut prev = x;
                let mut prev_right_end = x_right_end;
                for x in r {
                    if x.right_end.get() < prev_right_end {
                        break;
                    }
                    to_remove.push(prev.clone());

                    prev = x;
                    prev_right_end = x.inter(&y);
                }
                prev.right_end.set(prev_right_end);

                for x in to_remove.drain(..) {
                    self.lines.remove(&x);
                }
            }

            self.lines.insert(y);
        }

        pub fn query(&self, x: V) -> Option<V> {
            let l = self.lines.range(Line::point_query(x)..).next()?;
            Some(l.slope * x + l.intercept)
        }
    }
}

fn main() {
    let mut input = fast_io::stdin();
    let mut output = fast_io::stdout();

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
