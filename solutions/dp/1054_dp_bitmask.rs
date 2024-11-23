use std::{cmp::Ordering, io::Write};

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

fn is_palin<T: Eq>(word: &[T]) -> bool {
    let n = word.len();
    (0..n / 2).all(|i| word[i] == word[n - 1 - i])
}

fn eq_rev<T: Eq>(a: &[T], b: &[T]) -> bool {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter().rev()).all(|(x, y)| x == y)
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Dir {
    Left = 0,
    Right = 1,
}

impl Dir {
    fn rev(self) -> Self {
        match self {
            Left => Right,
            Right => Left,
        }
    }
}

use Dir::*;

// dp[hole_dir][hole_idx][hole_len][used]
type Table = Vec<Vec<Vec<Vec<u64>>>>;
const UNSET: u64 = u64::MAX;

fn solve_naive(words: &[Vec<u8>], available: usize) -> u64 {
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

    let n = words.len();
    let mut indices: Vec<usize> = (0..n).filter(|&u| (available >> u) & 1 != 0).collect();
    let mut res = 0;
    loop {
        let join: Vec<u8> = indices
            .iter()
            .map(|&u| &words[u])
            .flatten()
            .copied()
            .collect();
        res += is_palin(&join) as u64;

        if !next_permutation(&mut indices) {
            break;
        }
    }
    res
}

fn solve_rec(
    memo: &mut Table,
    words: &[Vec<u8>],
    hole_dir: Dir,
    hole_idx: usize,
    hole_len: usize,
    available: usize,
) -> u64 {
    let memo_entry = &memo[hole_dir as usize][hole_idx][hole_len][available];
    if *memo_entry != UNSET {
        return *memo_entry;
    }

    let hole_rev = match hole_dir {
        Left => &words[hole_idx][words[hole_idx].len() - hole_len..],
        Right => &words[hole_idx][..hole_len],
    };

    let available_words = || (0..words.len()).filter(|&u| (available >> u) & 1 == 1);

    let mut res = 0;
    if available == 0 {
        res = is_palin(hole_rev) as u64;
    } else if hole_len == 0 {
        for u in available_words() {
            res += solve_rec(memo, words, Left, u, words[u].len(), available ^ (1 << u));
        }
    } else {
        for u in available_words() {
            match words[u].len().cmp(&hole_len) {
                Ordering::Less => {
                    let hole_trunc = match hole_dir {
                        Left => &hole_rev[..words[u].len()],
                        Right => &hole_rev[hole_len - words[u].len()..],
                    };

                    if eq_rev(&words[u], hole_trunc) {
                        res += solve_rec(
                            memo,
                            words,
                            hole_dir,
                            hole_idx,
                            hole_len - words[u].len(),
                            available ^ (1 << u),
                        );
                    }
                }
                Ordering::Equal => {
                    if eq_rev(&words[u], hole_rev) {
                        res += solve_rec(memo, words, Left, 0, 0, available ^ (1 << u))
                    }
                }
                Ordering::Greater => {
                    let word_trunc = match hole_dir {
                        Left => &words[u][words[u].len() - hole_len..],
                        Right => &words[u][..hole_len],
                    };
                    if eq_rev(word_trunc, hole_rev) {
                        res += solve_rec(
                            memo,
                            words,
                            hole_dir.rev(),
                            u,
                            words[u].len() - hole_len,
                            available ^ (1 << u),
                        );
                    }
                }
            }
        }
    }
    memo[hole_dir as usize][hole_idx][hole_len][available] = res;

    res
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let words: Vec<Vec<u8>> = (0..n).map(|_| input.token().bytes().collect()).collect();

    let max_len = words.iter().map(|w| w.len()).max().unwrap();
    let mut memo: Table = vec![vec![vec![vec![UNSET; 1 << n]; max_len + 1]; n]; 2];

    let mut res = 0;
    for available in 1..1 << n {
        res += solve_rec(&mut memo, &words, Left, 0, 0, available);

        // let left = solve_naive(&words, available);
        // let right = solve_rec(&mut memo, &words, Left, 0, 0, available);
        // if left != right {
        //     println!(
        //         "available: {:?}, naive: {:?}, dp: {:?}",
        //         (0..words.len())
        //             .filter(|&u| (available >> u) & 1 == 1)
        //             .map(|u| std::str::from_utf8(&words[u]).unwrap())
        //             .collect::<Vec<_>>(),
        //         left,
        //         right
        //     );
        // }
    }
    writeln!(output, "{}", res).unwrap();
}
