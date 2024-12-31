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
    let k: usize = input.value();
    let q: usize = input.value();

    let mut circular_imos = vec![0; n];
    for src in 0..n {
        let mut fixed = true;
        for _ in 0..k {
            let target = input.value::<usize>() - 1;
            if src < target {
                circular_imos[src] += 1;
                circular_imos[target] -= 1;
                fixed = false;
            } else if src > target {
                circular_imos[src] += 1;
                circular_imos[0] += 1;
                circular_imos[target] -= 1;
                fixed = false;
            }
        }

        if src == 0 && fixed {
            circular_imos[0] += 1;
        }
    }

    let mut commands = vec![0; n];
    let mut acc = 0;
    for i in 0..n {
        acc += circular_imos[i];
        commands[i] = acc;
    }

    let ans = commands.windows(2).all(|t| t[0] == t[1]) && commands[0] <= q;
    writeln!(output, "{}", ans as u8).unwrap();
}
