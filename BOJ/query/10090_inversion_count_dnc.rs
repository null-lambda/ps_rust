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
    let mut xs: Vec<_> = (0..n).map(|_| -input.value::<i32>()).collect();

    let mut ans = 0u64;
    let mut w = 1;
    let mut buf = vec![!0; n];
    loop {
        if w >= n {
            break;
        }

        for view in xs.chunks_mut(w * 2) {
            if view.len() <= w {
                break;
            }

            let (lhs, rhs) = view.split_at(w);
            let mut i = 0;
            let mut j = 0;
            let mut cursor = 0;
            while i < lhs.len() && j < rhs.len() {
                if lhs[i] < rhs[j] {
                    buf[cursor] = lhs[i];
                    i += 1;
                } else {
                    buf[cursor] = rhs[j];
                    j += 1;
                    ans += i as u64;
                }
                cursor += 1;
            }
            while i < lhs.len() {
                buf[cursor] = lhs[i];
                i += 1;
                cursor += 1;
            }
            while j < rhs.len() {
                buf[cursor] = rhs[j];
                j += 1;
                ans += i as u64;
                cursor += 1;
            }

            view.copy_from_slice(&buf[..view.len()]);
        }

        w <<= 1;
    }
    writeln!(output, "{}", ans).ok();
}
