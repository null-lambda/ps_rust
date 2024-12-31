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

type Point = (i64, i64);

fn signed_area(p: Point, q: Point, r: Point) -> i64 {
    let dq = (q.0 - p.0, q.1 - p.1);
    let dr = (r.0 - p.0, r.1 - p.1);
    dq.0 * dr.1 - dq.1 * dr.0
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let ps: Vec<Point> = (0..n).map(|_| (input.value(), input.value())).collect();

    let mut ans = 0;
    let mut hull_start = 0;
    for i in 2..n {
        let p = ps[i - 2];
        let q = ps[i - 1];
        let r = ps[i];
        if signed_area(p, q, r) > 0 {
            while hull_start < i - 1
                && !(signed_area(q, r, ps[hull_start]) > 0
                    && signed_area(r, ps[hull_start], ps[hull_start + 1]) > 0)
            {
                hull_start += 1;
            }
        } else {
            hull_start = i - 1;
        }
        ans = ans.max((i - hull_start + 1) as i32);
    }

    if ans < 3 {
        ans = -1;
    }
    writeln!(output, "{}", ans).unwrap();
}
