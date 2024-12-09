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

    let n: usize = input.value();
    let mut ps = vec![];
    for _ in 0..n {
        let b = input.token() == "1";
        let x: u32 = input.value();
        ps.push((b, x));
    }
    ps.sort_unstable_by_key(|&(_, x)| x);

    let mut prefix_sum_groups = HashMap::<i32, Vec<_>>::new();
    let mut acc = 0;
    prefix_sum_groups.entry(0).or_default().push(0);
    for (i, &(p, _)) in ps.iter().enumerate() {
        acc += if p { 1 } else { -1 };
        prefix_sum_groups.entry(acc).or_default().push(i + 1);
    }
    let mut ans = 0;
    for (_, group) in prefix_sum_groups {
        if let [i0, .., i1] = &group[..] {
            let x0 = ps[*i0].1;
            let x1 = ps[*i1 - 1].1;
            ans = ans.max(x1 - x0);
        }
    }
    writeln!(output, "{}", ans).unwrap();
}
