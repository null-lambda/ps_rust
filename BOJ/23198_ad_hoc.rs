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

    let n: usize = input.value();
    let ps: Vec<_> = (0..n)
        .map(|_| {
            let x: i64 = input.value();
            let y: i64 = input.value();
            let z: i64 = input.value();
            [x, y, z, x + y, y + z, x + z, x + y + z]
        })
        .collect();
    let max = ps.iter().fold([i64::MIN; 7], |acc, x| {
        std::array::from_fn(|i| acc[i].max(x[i]))
    });
    let (min_cost, i) = (0..n)
        .map(|i| ((0..7).map(|k| max[k] - ps[i][k]).max().unwrap(), i))
        .min()
        .unwrap();
    writeln!(output, "{} {}", min_cost, i + 1).unwrap();
}
