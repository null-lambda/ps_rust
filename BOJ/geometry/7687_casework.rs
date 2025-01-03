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

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    loop {
        let mut l: [i64; 3] = [input.value(), input.value(), input.value()];
        let mut p: [i64; 3] = [input.value(), input.value(), input.value()];
        if l == [0, 0, 0] && p == [0, 0, 0] {
            return;
        }

        if p[1] == 0 || p[1] == l[1] {
            l.swap(0, 1);
            p.swap(0, 1);
        } else if p[2] == 0 || p[2] == l[2] {
            l.swap(0, 2);
            p.swap(0, 2);
        }
        assert!(p[0] == 0 || p[0] == l[0]);

        let hypot = |x, y| x * x + y * y;

        let mut ans = i64::MAX;
        if p[0] == 0 {
            ans = hypot(p[1], p[2]);
        } else {
            for i in [1, 2] {
                let dual = 3 - i;
                ans = ans.min(hypot(p[0] + p[dual], p[i]));

                let dx = l[i] + p[dual];
                let dy = l[0] + l[i] - p[i];
                if dy * l[i] <= dx * l[0] {
                    ans = ans.min(hypot(dx, dy))
                }
            }
        }
        writeln!(output, "{}", ans).unwrap();
    }
}
