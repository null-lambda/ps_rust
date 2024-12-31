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

    let n: usize = input.value();
    let mut common_parity = None;
    let inf = i64::MAX / 100;
    let mut bound = (-inf, inf, -inf, inf);
    for _ in 0..n {
        let a: i64 = input.value();
        let b: i64 = input.value();
        let c: i64 = input.value();

        let parity = (a + b + c) % 2 != 0;
        if let Some(common_parity) = common_parity {
            if common_parity != parity {
                writeln!(output, "NO").unwrap();
                return;
            }
        } else {
            common_parity = Some(parity);
        }
        let p = a + b;
        let q = a - b;
        bound.0 = bound.0.max(p - c);
        bound.1 = bound.1.min(p + c);
        bound.2 = bound.2.max(q - c);
        bound.3 = bound.3.min(q + c);
    }

    let test = |a: i64, b: i64| -> bool {
        let p = a + b;
        let q = a - b;
        (p % 2 != 0) == common_parity.unwrap()
            && (bound.0..=bound.1).contains(&p)
            && (bound.2..=bound.3).contains(&q)
    };

    let (x0, y0) = ((bound.0 + bound.2) / 2, (bound.0 - bound.2) / 2);

    let mut ans = None;
    'outer: for dx in -2..=2 {
        for dy in -2..=2 {
            let (x, y) = (x0 + dx, y0 + dy);
            if test(x, y) {
                ans = Some((x, y));
                break 'outer;
            }
        }
    }

    if let Some((x, y)) = ans {
        writeln!(output, "{} {}", x, y).unwrap();
    } else {
        writeln!(output, "NO").unwrap();
    }
}
