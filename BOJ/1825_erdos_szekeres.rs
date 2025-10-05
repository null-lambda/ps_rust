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

    pub fn stdin_at_once<'a>() -> InputAtOnce<'a> {
        let _buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let iter = _buf.split_ascii_whitespace();
        let iter = unsafe { std::mem::transmute(iter) };
        InputAtOnce { _buf, iter }
    }

    pub fn stdout() -> std::io::BufWriter<std::io::Stdout> {
        std::io::BufWriter::new(std::io::stdout())
    }
}

fn solve_min(n: usize, m: usize, k: usize) -> Vec<u32> {
    let mut res: Vec<_> = (0..n as u32).collect();
    if k == 1 {
        assert_eq!(n, m);
        return res;
    }
    assert!(k >= 2 && m <= n - 1);

    let q = (n - m - 1) / (k - 1);
    let r = (n - m - 1) % (k - 1) + 2;

    let mut i = n;
    for _ in 0..q {
        res[i - k..i].reverse();
        i -= k;
    }
    res[i - r..i].reverse();
    res
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let k: usize = input.value();

    // Erdos-Szekeres theorem
    if m.saturating_mul(k) < n || n < m + k - 1 {
        writeln!(output, "-1").unwrap();
        return;
    }

    for u in solve_min(n, m, k) {
        write!(output, "{} ", u + 1).unwrap();
    }
    writeln!(output).unwrap();

    for u in solve_min(n, k, m) {
        write!(output, "{} ", n as u32 - u).unwrap();
    }
    writeln!(output).unwrap();
}
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

    pub fn stdin_at_once<'a>() -> InputAtOnce<'a> {
        let _buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let iter = _buf.split_ascii_whitespace();
        let iter = unsafe { std::mem::transmute(iter) };
        InputAtOnce { _buf, iter }
    }

    pub fn stdout() -> std::io::BufWriter<std::io::Stdout> {
        std::io::BufWriter::new(std::io::stdout())
    }
}

fn solve_min(n: usize, m: usize, k: usize) -> Vec<u32> {
    let mut res: Vec<_> = (0..n as u32).collect();
    if k == 1 {
        assert_eq!(n, m);
        return res;
    }
    assert!(k >= 2 && m <= n - 1);

    let q = (n - m - 1) / (k - 1);
    let r = (n - m - 1) % (k - 1) + 2;

    let mut i = n;
    for _ in 0..q {
        res[i - k..i].reverse();
        i -= k;
    }
    res[i - r..i].reverse();
    res
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let k: usize = input.value();

    // Erdos-Szekeres theorem
    if m.saturating_mul(k) < n || n < m + k - 1 {
        writeln!(output, "-1").unwrap();
        return;
    }

    for u in solve_min(n, m, k) {
        write!(output, "{} ", u + 1).unwrap();
    }
    writeln!(output).unwrap();

    for u in solve_min(n, k, m) {
        write!(output, "{} ", n as u32 - u).unwrap();
    }
    writeln!(output).unwrap();
}
