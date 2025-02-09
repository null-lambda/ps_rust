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

fn maximal_clique_in_split_graph(degree_desc: &[u32]) -> Option<(u32, u32)> {
    let n = degree_desc.len();

    let mut prefix = vec![0; n + 1];
    for i in 1..=n {
        prefix[i] = prefix[i - 1] + degree_desc[i - 1] as u64;
    }
    let d_sum = prefix[n];

    let maximal_size = (0..n).position(|i| degree_desc[i] < i as u32).unwrap_or(n) as u32;
    if 2 * prefix[maximal_size as usize] - d_sum != (maximal_size as u64 - 1) * maximal_size as u64
    {
        return None;
    }

    let mut maximal_count = 1;
    if maximal_size > 0 {
        maximal_count += degree_desc[maximal_size as usize..]
            .iter()
            .filter(|&&d| d == degree_desc[maximal_size as usize - 1])
            .count() as u32;
    }
    Some((maximal_size, maximal_count))
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let mut degree = vec![0u32; n];
    for i in 0..n {
        let d = input.value::<u32>();
        degree[i as usize] = d;
        for _ in 0..d {
            input.token();
        }
    }
    degree.sort_unstable_by_key(|&d| Reverse(d));

    let Some((m, c)) = maximal_clique_in_split_graph(&degree) else {
        writeln!(output, "0").unwrap();
        return;
    };

    let mut degree_complement = degree.iter().map(|&d| n as u32 - 1 - d).collect::<Vec<_>>();
    degree_complement.reverse();
    let (m_inv, c_alt) = maximal_clique_in_split_graph(&degree_complement).unwrap();
    let m_alt = n as u32 - m_inv;

    let mut c_total = 0;
    if 0 < m && m < n as u32 {
        c_total += c;
    }
    if 0 < m_alt && m_alt < n as u32 && m_alt != m {
        c_total += c_alt;
    }
    writeln!(output, "{}", c_total).unwrap();
}
