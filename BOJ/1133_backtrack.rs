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

fn postfix_period(s: &[u8], postfix_len: usize) -> usize {
    debug_assert!(postfix_len <= s.len());
    let pattern = &s[s.len() - postfix_len..];
    let res = s
        .rchunks_exact(postfix_len)
        .position(|chunk| chunk != pattern)
        .unwrap_or_else(|| s.len() / postfix_len);
    res
}

fn backtrack(acc: &mut [u8], idx: usize, k: usize, a: usize) -> bool {
    if idx == acc.len() {
        return true;
    }

    (b'A'..b'A' + a as u8).any(|c| {
        acc[idx] = c;
        (1..=idx + 1).all(|l| postfix_period(&acc[..=idx], l) < k) && backtrack(acc, idx + 1, k, a)
    })
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let k: usize = input.value();
    let n: usize = input.value();
    let a: usize = input.value();

    let mut res = vec![0; n];
    if backtrack(&mut res, 0, k, a) {
        writeln!(output, "{}", String::from_utf8(res).unwrap()).unwrap();
    } else {
        writeln!(output, "-1").unwrap();
    }
}
