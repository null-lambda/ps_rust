use std::{
    collections::HashMap,
    io::Write,
    iter::{self, repeat},
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

fn get_period(s: &[u8]) -> usize {
    let mut jump_table = vec![0];
    let mut i_prev = 0;
    for i in 1..s.len() {
        while i_prev > 0 && s[i] != s[i_prev] {
            i_prev = jump_table[i_prev - 1];
        }
        if s[i] == s[i_prev] {
            i_prev += 1;
        }
        jump_table.push(i_prev);
    }
    let res = s.len() - jump_table[s.len() - 1];
    if s.len() % res == 0 {
        res
    } else {
        s.len()
    }
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let words: Vec<&[u8]> = (0..n).map(|_| input.token().as_bytes()).collect();
    let k: usize = input.value();

    let mut indices = (0..n).collect::<Vec<_>>();

    let mut cnt = 0;
    loop {
        let s: Vec<u8> = indices
            .iter()
            .flat_map(|&i| words[i].iter().cloned())
            .collect();
        if s.len() % k == 0 && get_period(&s) == s.len() / k {
            cnt += 1;
        }

        if !next_permutation(&mut indices) {
            break;
        }
    }
    writeln!(output, "{}", cnt).unwrap();
}
