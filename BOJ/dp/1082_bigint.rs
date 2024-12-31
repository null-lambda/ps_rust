use std::{
    cmp::{Ordering, Reverse},
    io::Write,
};

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
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let n_digits: usize = input.value();
    let cost: Vec<usize> = (0..n_digits).map(|_| input.value()).collect();
    let m_max: usize = input.value();

    type BigInt = Vec<u8>;
    let cmp_bigint = |a: &BigInt, b: &BigInt| a.len().cmp(&b.len()).then_with(|| a.cmp(&b));
    let mut dp: Vec<BigInt> = vec![vec![]; m_max + 1];

    fn insert(x: &BigInt, new_digit: u8) -> BigInt {
        let mut res = x.clone();
        res.push(new_digit);
        res.sort_unstable_by_key(|&x| Reverse(x));
        res
    }

    for m in 0..=m_max {
        for d in 0..n_digits as u8 {
            let m_next = m + cost[d as usize];
            if m_next > m_max {
                continue;
            }

            let dp_next = insert(&dp[m], d);
            if dp_next.iter().all(|&x| x == 0) {
                continue;
            }

            if cmp_bigint(&dp_next, &dp[m_next]) == Ordering::Greater {
                dp[m_next] = dp_next;
            }
        }
    }

    let mut ans = dp.iter().max_by(|a, b| cmp_bigint(a, b)).unwrap().clone();
    if ans.is_empty() {
        ans.push(0);
    }

    writeln!(
        output,
        "{}",
        ans.iter().map(|&x| (x + b'0') as char).collect::<String>()
    )
    .unwrap();
}
