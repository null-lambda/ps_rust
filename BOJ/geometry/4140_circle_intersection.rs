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

        pub fn try_value<T: std::str::FromStr>(&mut self) -> Option<T>
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().ok()
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

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let mut ps = vec![];
    while let Some(x) = input.try_value::<f64>() {
        let y = input.value();
        ps.push([x, y]);
    }

    let r = 2.5;
    let r_sq = r * r;
    let mut test_points = vec![];
    for i in 0..ps.len() {
        test_points.push(ps[i]);
        for j in i + 1..ps.len() {
            let [px, py] = ps[i];
            let [qx, qy] = ps[j];

            // Calculate intersections of two circles
            let [mx, my] = [(px + qx) / 2.0, (py + qy) / 2.0];
            let [dx, dy] = [qx - px, qy - py];
            let d = dx.hypot(dy);
            if d < 1e-9 {
                continue;
            }
            let [dx, dy] = [dx / d, dy / d];

            let h = (r_sq - d * d / 4.0).sqrt();
            if !h.is_finite() {
                continue;
            }

            test_points.push([mx - dy * h, my + dx * h]);
            test_points.push([mx + dy * h, my - dx * h]);
        }
    }

    let mut ans = 0;
    for [cx, cy] in test_points {
        ans = ans.max(
            ps.iter()
                .filter(|&&[x, y]| (x - cx).hypot(y - cy) <= r + 1e-9)
                .count(),
        );
    }
    writeln!(output, "{}", ans).unwrap();
}
