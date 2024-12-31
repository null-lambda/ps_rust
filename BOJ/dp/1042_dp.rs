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

fn char_to_digit(c: u8) -> u8 {
    match c {
        b'A' => 0,
        b'C' => 1,
        b'T' => 2,
        b'G' => 3,
        _ => panic!(),
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let xs: Vec<u8> = input.token().bytes().map(char_to_digit).collect();
    let n = xs.len();

    // Step 1-1. Preprocess set of codons in range xs[start..=end] with bitmasks
    let mut contains_code = vec![vec![0u8; n + 1]; n + 1];
    let mut contains_tuple = vec![vec![0u16; n + 1]; n + 1];
    let mut contains_codon = vec![vec![0u64; n + 1]; n + 1];

    for end in 0..n {
        contains_code[end][end] = 1 << xs[end];
        for start in (0..end).rev() {
            contains_code[start][end] = 1 << xs[start] | contains_code[start + 1][end];
        }
    }

    fn tensor_concat_codes(a: u8, b: u8) -> u16 {
        let mut res = 0;
        for i in 0..4 {
            if b & (1 << i) != 0 {
                res |= (a as u16) << i * 4;
            }
        }
        res
    }

    for tuple in 0..16 {
        let left = tuple % 4;
        let right = tuple / 4;
        for i in 1..n {
            if &xs[i - 1..=i] == &[left, right] {
                contains_tuple[i - 1][i] = 1 << tuple;
            }
        }
    }

    for end in 1..n {
        for start in (0..end).rev() {
            contains_tuple[start][end] = contains_tuple[start + 1][end]
                | tensor_concat_codes(contains_code[start][start], contains_code[start + 1][end]);
        }
    }

    fn tensor_concat_tuple_code(a: u16, b: u8) -> u64 {
        let mut res = 0;
        for i in 0..4 {
            if b & (1 << i) != 0 {
                res |= (a as u64) << i * 16;
            }
        }
        res
    }

    for codon in 0..64 {
        let left = codon % 4;
        let mid = (codon / 4) % 4;
        let right = codon / 16;
        for i in 2..n {
            if &xs[i - 2..=i] == &[left, mid, right] {
                contains_codon[i - 2][i] = 1 << codon;
            }
        }
    }

    for start in 0..n {
        for end in start + 2..n {
            contains_codon[start][end] = contains_codon[start][end - 1]
                | tensor_concat_tuple_code(contains_tuple[start][end - 1], contains_code[end][end]);
        }
    }

    // for left in 0..4 {
    //     for mid in 0..4 {
    //         for right in 0..4 {
    //             for start in 0..n {
    //                 for end in start..n {
    //                     let mut contains_naive = false;
    //                     for i in start..=end {
    //                         for j in i + 1..=end {
    //                             for k in j + 1..=end {
    //                                 if (xs[i], xs[j], xs[k]) == (left, mid, right) {
    //                                     contains_naive = true;
    //                                 }
    //                             }
    //                         }
    //                     }
    //                     assert_eq!(
    //                         contains_naive,
    //                         contains_codon[start][end] & (1 << (left + mid * 4 + right * 16)) != 0
    //                     );
    //                 }
    //             }
    //         }
    //     }
    // }

    // Step 1-2. Build a jump table for the next codon
    // [start..][codon]
    let mut next_codon_end = vec![vec![None; 64]; n];
    for start in 0..n {
        for codon in 0..64 {
            next_codon_end[start][codon] =
                (start + 1..n).find(|&end| contains_codon[start][end] & (1 << codon) != 0);
        }
    }

    // Step 2. Count number of amino acids that has first occurrence in a specified position
    let m: usize = input.value();

    let mut codon_table_rev: HashMap<&str, Vec<u8>> = HashMap::new();
    for _ in 0..m {
        let codon = input
            .token()
            .bytes()
            .take(3)
            .map(char_to_digit)
            .enumerate()
            .fold(0, |acc, (i, x)| acc | x << i * 2);

        let amino = input.token();
        codon_table_rev.entry(amino).or_default().push(codon);
    }
    let codon_groups: Vec<Vec<u8>> = codon_table_rev.into_values().collect();

    const P: u64 = 1_000_000_007;
    let mut ans = 0u64;
    let mut dp = vec![0u64; n + 1]; // Caution: converted index to 1-based

    dp[0] = 1;

    while dp.iter().any(|&x| x > 0) {
        let prev = dp;
        dp = vec![0; n + 1];

        for start in 0..n {
            for codons in &codon_groups {
                let next_pos = codons
                    .iter()
                    .flat_map(|&codon| next_codon_end[start][codon as usize])
                    .min();
                if let Some(end) = next_pos {
                    dp[end + 1] = (dp[end + 1] + prev[start]) % P;
                }
            }
        }

        for x in &dp[1..=n] {
            ans = (ans + x) % P;
        }
    }
    writeln!(output, "{}", ans).unwrap();
}
