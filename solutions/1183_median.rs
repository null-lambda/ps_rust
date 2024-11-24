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

fn median<T: Ord + Clone>(xs: &mut [T]) -> Option<(T, Option<T>)> {
    let n = xs.len();
    if n == 0 {
        return None;
    }

    xs.select_nth_unstable((n - 1) / 2);
    let left = xs[(n - 1) / 2].clone();

    let right = (n % 2 == 0).then(|| {
        xs.select_nth_unstable(n / 2);
        xs[n / 2].clone()
    });
    Some((left, right))
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let mut diff: Vec<i64> = (0..n)
        .map(|_| {
            let x = input.value::<i64>();
            let y = input.value::<i64>();
            y - x
        })
        .collect();
    let ans = match median(&mut diff) {
        Some((left, Some(right))) => right - left + 1,
        Some((_, None)) => 1,
        None => 0,
    };
    writeln!(output, "{}", ans).unwrap();
}
