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

    for _ in 0..input.value() {
        let n: usize = input.value();
        let xs = input.token().bytes().map(|b| (b - b'A') as u32).take(n);

        let mut acc = 0;
        let mut prefix_sum = vec![0];
        for x in xs {
            acc += x;
            prefix_sum.push(acc);
        }
        assert!(prefix_sum.len() == n + 1);

        let l: u32 = input.value();
        let u: u32 = input.value();

        let mut l_acc = 0.0;
        let mut u_acc = 0.0;
        for i in 1..=n {
            let denom = (n - i + 1) as f64;
            let p = prefix_sum[i..].partition_point(|&x| x < prefix_sum[i - 1] + l);
            let q = prefix_sum[i..].partition_point(|&x| x <= prefix_sum[i - 1] + u);
            l_acc += p as f64 / denom;
            u_acc += q as f64 / denom;
        }

        let p0 = (u_acc - l_acc) / n as f64;
        let p1 = l_acc / n as f64;
        let p2 = 1.0 - p0 - p1;
        writeln!(output, "{} {} {}", p0, p1, p2).unwrap();
    }
}
