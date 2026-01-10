use std::{io::Write, iter};

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

fn fact(n: u64) -> u64 {
    (1..=n).product()
}

fn comb(n: u64, m: u64) -> u64 {
    fact(n) / (fact(m) * fact(n - m))
}

fn next_permutation<T: Ord>(arr: &mut [T]) -> bool {
    match arr.windows(2).rposition(|w| w[0] < w[1]) {
        Some(i) => {
            let j = i + arr[i + 1..].partition_point(|x| &arr[i] < x);
            arr.swap(i, j);
            arr[i + 1..].reverse();
            true
        }
        None => {
            arr.reverse();
            false
        }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();

    let s = &input.token().as_bytes()[..n];
    let mut freq = vec![0; m];
    for &b in s {
        freq[(b - b'0') as usize] += 1;
    }

    let mut p = n as u32;
    let mut q = 1u64;
    let mut ans = 0;
    for i in (1..m).rev() {
        for j in 0..freq[i] {
            ans += (i as u64).pow(p - j) * comb(p as u64, j as u64) * q;
        }
        q *= comb(p as u64, freq[i] as u64);
        p -= freq[i];
    }

    let s_rev: Vec<_> = s.iter().rev().copied().collect();
    let mut perm = s_rev.clone();
    perm.sort_unstable();

    loop {
        if perm == s_rev {
            break;
        }
        ans += 1;
        if !next_permutation(&mut perm) {
            panic!();
        }
    }

    writeln!(output, "{}", ans).unwrap();
}
