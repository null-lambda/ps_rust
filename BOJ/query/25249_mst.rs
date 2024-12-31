use std::{cmp::Reverse, io::Write};

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
    let m: usize = input.value();
    let mut edges = vec![];
    for _ in 0..m {
        let x: usize = input.value();
        let d: u32 = input.value();
        let c: u32 = input.value();
        edges.push((c, d, x));
    }
    edges.sort_unstable_by_key(|&(c, _, _)| Reverse(c));

    let q: usize = input.value();
    let mut queries = vec![];
    for i in 0..q {
        let c_required: u32 = input.value();
        queries.push((c_required, i));
    }
    queries.sort_unstable_by_key(|&(c, _)| Reverse(c));

    let mut n_disconnected = n - 1;
    let mut dist = vec![None; n];
    let mut total_dist = 0;
    let mut edges = edges.into_iter().peekable();
    let mut ans = vec![None; q];
    for (c_required, i_query) in queries {
        while let Some((_, d, x)) = edges.next_if(|&(c, _, _)| c >= c_required) {
            if let Some(old) = dist[x] {
                if old > d {
                    total_dist = total_dist + d - old;
                    dist[x] = Some(d);
                }
            } else {
                n_disconnected -= 1;
                dist[x] = Some(d);
                total_dist += d;
            }
        }

        if n_disconnected == 0 {
            ans[i_query] = Some(total_dist);
        }
    }

    for a in ans {
        if let Some(a) = a {
            writeln!(output, "{}", a).unwrap();
        } else {
            writeln!(output, "impossible").unwrap();
        }
    }
}
