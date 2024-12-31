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

    let n: u32 = input.value();
    let m: usize = input.value();
    let mut cs = vec![];
    for _ in 0..m {
        let c: u32 = input.value();
        if c < n {
            cs.push(c);
        }
    }

    cs.sort_unstable();
    let ans = match &cs[..] {
        _ if n <= 3 => 1,
        [] | [_] => 1,
        [min, rest @ ..] => {
            let mut n_components = 1;
            let mut len = *min as i32;
            for &c in rest.into_iter().rev() {
                if len <= 2 {
                    break;
                }
                if len as u32 + c >= n + 2 {
                    n_components += 1;
                    len -= (n - c + 1) as i32;
                }
            }
            n_components
        }
    };
    writeln!(output, "{}", ans).unwrap();
}
