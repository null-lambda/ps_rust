use std::io::Write;

mod simple_io {
    use std::string::*;

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

fn solve(x: u32, y: u32) -> u32 {
    let mut res = 0u32;
    let mut acc = 1;
    for i in 1..=30 {
        let start = (1 << i) - 1;
        let end = (1 << i + 1) - 1;
        acc += i;

        if end <= x {
            continue;
        } else if y <= start {
            break;
        } else if x <= start && end <= y {
            res = res.max(acc);
        } else {
            let x_sub = x.max(start);
            let y_sub = y.min(end);
            res = res.max(solve(x_sub - start, y_sub - start) + i);
        }
    }
    res
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    for _ in 0..input.value() {
        let x: u32 = input.value();
        let y: u32 = input.value();
        writeln!(output, "{}", solve(x, y + 1)).unwrap();
    }
}
