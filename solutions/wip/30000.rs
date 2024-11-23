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

        pub fn try_value<T: std::str::FromStr>(&mut self) -> Result<T, T::Err> {
            self.token().parse()
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

    match input.token() {
        "0" => writeln!(output, "BOJ 30000").unwrap(),
        "1" => {
            // Collatz sequence
            let mut x = 989345275647u64;
            for _ in 0..1349 {
                writeln!(output, "{}", x).unwrap();
                x = if x % 2 == 0 { x / 2 } else { 3 * x + 1 };
            }
        }
        "2" => {
            //
        }
        _ => panic!(),
    }
}
