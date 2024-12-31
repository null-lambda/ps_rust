use std::{collections::HashMap, io::Write};

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

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let n: u32 = input.value();
    let m: u32 = input.value();
    let k: u32 = input.value();
    if !(m + k - 1 <= n && n <= m * k) {
        writeln!(output, "{}", "-1").unwrap();
        return;
    }

    let mut ans: Vec<u32> = (1..=n).collect();
    let mut group_size = vec![k];
    let mut rest = n - k;
    for s in (1..m).rev() {
        let q = rest / s;
        group_size.push(q);
        rest -= q;
    }
    assert_eq!(rest, 0);

    let mut pos = 0;
    for &size in &group_size {
        ans[pos..(pos + size as usize)].reverse();
        pos += size as usize;
    }
    for a in ans {
        write!(output, "{} ", a).unwrap();
    }
}
