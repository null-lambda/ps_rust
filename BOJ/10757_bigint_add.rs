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

    let s = input.token();
    let t = input.token();
    let n = s.len().max(t.len());

    let s_pad = s.bytes().rev().chain(std::iter::repeat(b'0'));
    let t_pad = t.bytes().rev().chain(std::iter::repeat(b'0'));
    let mut res = Vec::with_capacity(n + 1);
    let mut carry = 0;
    for (a, b) in s_pad.zip(t_pad).take(n) {
        let a = a - b'0';
        let b = b - b'0';
        let mut c = a + b + carry;
        carry = (c >= 10) as u8;
        c -= carry * 10;
        res.push(c + b'0');
    }
    if carry > 0 {
        res.push(b'1');
    }
    res.reverse();

    writeln!(output, "{}", String::from_utf8(res).unwrap()).unwrap();
}
