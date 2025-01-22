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

    for i_tc in 1..=input.value() {
        let n: usize = input.value();
        let mut items = vec![];
        let mut items_persist = vec![];
        for _ in 0..n {
            let s: i32 = input.value();
            let e: i32 = input.value();
            let l: i32 = input.value();
            if l == 0 {
                items_persist.push((s, e, l));
            } else {
                items.push((s, e, l));
            }
        }
        items.sort_unstable_by(|(s1, _, l1), (s2, _, l2)| (s1 * l2).cmp(&(s2 * l1)));
        items.extend(items_persist);

        let t_bound: i32 = items.iter().map(|&(s, ..)| s).sum();
        let mut dp = vec![0; t_bound as usize + 1];
        for (s, e, l) in items {
            let prev = dp.clone();
            for t in 0..=t_bound - s {
                dp[(t + s) as usize] =
                    dp[(t + s) as usize].max(prev[t as usize] + 0.max(e - l * t));
            }
        }
        let ans = dp.iter().max().unwrap();
        writeln!(output, "Case #{}: {}", i_tc, ans).unwrap();
    }
}
