use std::io::Write;
use std::ops::Range;

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

fn merge(lhs: &Option<Range<u32>>, rhs: &Option<Range<u32>>) -> Option<Range<u32>> {
    lhs.clone()
        .zip(rhs.clone())
        .and_then(|(lhs, rhs)| {
            assert!(lhs.start <= rhs.end && rhs.start <= lhs.end);
            let start = lhs.start.min(rhs.start);
            let end = lhs.end.max(rhs.end);
            Some(start..end)
        })
        .or(lhs.clone())
        .or(rhs.clone())
}

fn inc(range: &Option<Range<u32>>) -> Option<Range<u32>> {
    let Range { start, end } = range.as_ref()?;
    Some(Range {
        start: start + 1,
        end: end + 1,
    })
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    for _ in 0..input.value() {
        let s = input.token().as_bytes();
        if s.len() % 2 == 1 {
            writeln!(output, "no").unwrap();
            continue;
        }
        let mut dp: [_; 4] = std::array::from_fn(|_| Some(0..1));
        for &b in s {
            let prev = dp;
            dp = std::array::from_fn(|_| None);

            if b == b'0' || b == b'.' {
                dp[0b00] = merge(&dp[0b00], &prev[0b10]);
                dp[0b10] = merge(&dp[0b10], &merge(&prev[0b01], &prev[0b11]));
            }
            if b == b'1' || b == b'.' {
                dp[0b01] = merge(&dp[0b01], &inc(&merge(&prev[0b00], &prev[0b10])));
                dp[0b11] = merge(&dp[0b11], &inc(&prev[0b01]));
            }
        }

        let count_1s = dp
            .iter()
            .fold(None, |acc, x| merge(&acc, x))
            .unwrap_or(u32::MAX..u32::MAX);
        let ans = count_1s.contains(&(s.len() as u32 / 2));
        writeln!(output, "{}", if ans { "yes" } else { "no" }).unwrap();
    }
}
