use std::{collections::HashMap, io::Write};

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

    loop {
        let n: usize = input.value();
        if n == 0 {
            break;
        }

        let mut reward = vec![];
        let mut masks = vec![];

        let mut e_max = 0;
        for _ in 0..n {
            let m: usize = input.value();
            let l: u64 = input.value();
            reward.push(l);
            masks.push(
                (0..m)
                    .map(|_| {
                        let s = input.value::<u32>() - 6;
                        let e = input.value::<u32>() - 6;
                        e_max = e_max.max(e);
                        (!0 << s) & !(!0 << e)
                    })
                    .fold(0, |acc, x| acc | x),
            );
        }

        let mut dp = vec![0; 1 << e_max];
        let mut ans = 0;
        for i in 0..n {
            for mask in 0..1 << e_max {
                if mask & masks[i] != masks[i] {
                    continue;
                }
                let mask_prev = mask & !masks[i];
                dp[mask] = dp[mask].max(dp[mask_prev] + reward[i]);
                ans = ans.max(dp[mask]);
            }
        }
        writeln!(output, "{}", ans).unwrap();
    }
}
