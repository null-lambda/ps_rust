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

pub fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    loop {
        let n: usize = input.value();
        if n == 0 {
            break;
        }

        let (sx, sy): (f64, f64) = (input.value(), input.value());
        let mut gates = vec![(sx, sx, sy)];
        for _ in 1..=n {
            let y = input.value();
            let x0 = input.value();
            let x1 = input.value();
            gates.push((x0, x1, y));
        }
        let gates = |i: usize, dir: usize| {
            let (x0, x1, y) = gates[i];
            if dir == 0 {
                [x0, y]
            } else {
                [x1, y]
            }
        };

        let mut ans = f64::INFINITY;
        let dist = |p: [f64; 2], q: [f64; 2]| f64::hypot(p[0] - q[0], p[1] - q[1]);
        let mut prefix = vec![[f64::INFINITY; 2]; n + 1];
        prefix[0] = [0.0; 2];
        for i in 1..=n {
            for dir in 0..2 {
                let dest = gates(i, dir);
                let (mut slope0, mut slope1) = (f64::NEG_INFINITY, f64::INFINITY);
                for j in (0..i).rev() {
                    let dx0 = gates(j, 0)[0] - dest[0];
                    let dx1 = gates(j, 1)[0] - dest[0];
                    let dy = gates(j, 0)[1] - dest[1];

                    let slope0_current = dx0 / dy;
                    let slope1_current = dx1 / dy;

                    let thres = 1e-15;

                    if (slope0 - thres..=slope1 + thres).contains(&slope0_current) {
                        prefix[i][dir] = prefix[i][dir].min(prefix[j][0] + dist(dest, gates(j, 0)));
                    }
                    if (slope0 - thres..=slope1 + thres).contains(&slope1_current) {
                        prefix[i][dir] = prefix[i][dir].min(prefix[j][1] + dist(dest, gates(j, 1)));
                    }

                    slope0 = slope0.max(slope0_current);
                    slope1 = slope1.min(slope1_current);
                    if slope0 > slope1 + thres {
                        break;
                    }
                }
            }
        }

        // Draw a straight vertical line, from the ith point to the goal line.
        for i in 0..n {
            let x0_max = (i + 1..n + 1)
                .map(|i| gates(i, 0)[0])
                .max_by(f64::total_cmp)
                .unwrap();
            let x1_min = (i + 1..n + 1)
                .map(|i| gates(i, 1)[0])
                .min_by(f64::total_cmp)
                .unwrap();
            for dir in 0..2 {
                let src = gates(i, dir);
                if (x0_max..=x1_min).contains(&src[0]) {
                    let y_end = gates(n, 0)[1];
                    ans = ans.min(prefix[i][dir] + (src[1] - y_end).abs());
                }
            }
        }
        ans = ans.min(prefix[n][0]).min(prefix[n][1]);
        writeln!(output, "{}", ans).unwrap();
    }
}
