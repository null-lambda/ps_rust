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

type Point = [i64; 11];

fn dot_parities(p: Point) -> Vec<i64> {
    let mut acc = vec![];
    fn rec(p: Point, i: usize, acc: i64, res: &mut Vec<i64>) {
        if i == 11 {
            res.push(acc);
            return;
        }
        rec(p, i + 1, acc + p[i], res);
        rec(p, i + 1, acc - p[i], res);
    }
    rec(p, 0, 0, &mut acc);

    acc
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let q: usize = input.value();
    let x_sum = (0..n)
        .map(|_| {
            let p = std::array::from_fn(|_| input.value());
            dot_parities(p)
        })
        .reduce(|mut acc, zs| {
            for (x, z) in acc.iter_mut().zip(&zs) {
                *x = (*x).max(*z);
            }
            acc
        })
        .unwrap();

    for _ in 0..q {
        let p = std::array::from_fn(|_| input.value());
        let y_sum = dot_parities(p);

        let ans = x_sum.iter().zip(&y_sum).map(|(x, y)| x - y).max().unwrap();
        writeln!(output, "{}", ans).unwrap();
    }
}
