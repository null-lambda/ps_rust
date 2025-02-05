use std::{io::Write, vec};

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

fn sums_of_subsets(xs: &[i64]) -> Vec<i64> {
    let mut sums = vec![0];
    for &x in xs {
        let prev = std::mem::take(&mut sums);
        let mut s1 = prev.iter().copied().peekable();
        let mut s2 = prev.iter().map(|y| y + x).peekable();
        let mut s3 = prev.iter().map(|y| y - x).peekable();
        loop {
            let y1 = s1.peek().copied().unwrap_or(i64::MAX);
            let y2 = s2.peek().copied().unwrap_or(i64::MAX);
            let y3 = s3.peek().copied().unwrap_or(i64::MAX);
            if y1 <= y2 && y1 <= y3 {
                sums.push(y1);
                s1.next();
            } else if y2 <= y3 {
                sums.push(y2);
                s2.next();
            } else {
                sums.push(y3);
                s3.next();
            }

            if sums.len() == prev.len() * 3 {
                break;
            }
        }
    }
    sums
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let mut xs: Vec<i64> = (0..n).map(|_| input.value()).collect();
    xs.sort_unstable();
    let d: i64 = input.value();

    let (left, right) = xs.split_at(n / 2);
    let left_sums = sums_of_subsets(left);
    let right_sums = sums_of_subsets(right);

    let mut ans = 0u64;
    let (mut i, mut j) = (0, 0);
    for &p in left_sums.iter().rev() {
        let q0 = -d - p;
        let q1 = d - p;
        while i < right_sums.len() && right_sums[i] < q0 {
            i += 1;
        }
        while j < right_sums.len() && right_sums[j] <= q1 {
            j += 1;
        }
        ans += j.saturating_sub(i) as u64;
    }
    writeln!(output, "{}", ans).unwrap();
}
