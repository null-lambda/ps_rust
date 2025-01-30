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

    pub fn stdin_at_once<'a>() -> InputAtOnce<'a> {
        let buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let iter = buf.split_ascii_whitespace();
        let iter = unsafe { std::mem::transmute(iter) };
        InputAtOnce { _buf: buf, iter }
    }
}

fn partition_point<P>(mut left: i64, mut right: i64, mut pred: P) -> i64
where
    P: FnMut(i64) -> bool,
{
    while left < right {
        let mid = left + (right - left) / 2;
        if pred(mid) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    left
}

const NEG_INF: i64 = -(1 << 60);

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = std::io::BufWriter::new(std::io::stdout().lock());

    let n: usize = input.value();
    let k: i64 = input.value();
    let xs: Vec<i32> = (0..n).map(|_| input.value()).collect();

    let score = |slope: i64| {
        let mut dp = [(0, 0), (NEG_INF, 0)];
        for &x in &xs {
            let add = |(x, y), (dx, dy)| (x + dx, y + dy);
            dp = [
                dp[0].max(dp[1]),
                add((x as i64, 0), dp[1].max(add(dp[0], (slope, -1)))),
            ];
        }
        let (s, k_neg) = dp[0].max(dp[1]);
        (s, -k_neg)
    };

    let slope_bound = xs.iter().map(|&x| (x as i64).abs()).sum::<i64>() + 1;
    let opt_slope = partition_point(-slope_bound, 1, |slope| score(slope).1 <= k as i64) - 1;
    let (mut max_score, _k_upper) = score(opt_slope);
    max_score -= opt_slope * k;
    writeln!(output, "{}", max_score).unwrap();

    // for slope in -10..=10 {
    //     println!("{}: {:?}", slope, score(slope));
    //     println!("{:?}", _k_upper);
    // }
}
