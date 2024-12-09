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

fn lis_len<T: Clone + Ord>(xs: impl IntoIterator<Item = T>) -> usize {
    let mut dp = vec![];
    for x in xs {
        if dp.last().is_none() || dp.last().unwrap() < &x {
            dp.push(x.clone());
        } else {
            let idx = dp.binary_search(&x).unwrap_or_else(|x| x);
            dp[idx] = x.clone();
        }
    }

    dp.len()
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let mut cards = vec![];
    for _ in 0..n {
        let t = input.token().as_bytes();
        let suit = match t[0] {
            b'S' => 0,
            b'W' => 1,
            b'E' => 2,
            b'R' => 3,
            b'C' => 4,
            _ => panic!(),
        };
        let rank: u32 = unsafe { std::str::from_utf8_unchecked(&t[1..]) }
            .parse()
            .unwrap();
        cards.push((suit, rank));
    }

    let mut suit_map = [0, 1, 2, 3, 4];
    let mut max_lis = 0;
    loop {
        let orders = cards.iter().map(|&(suit, rank)| {
            let suit = suit_map[suit];
            suit * n as u32 * 2 + rank as u32
        });
        max_lis = max_lis.max(lis_len(orders));

        if !next_permutation(&mut suit_map[..4]) {
            break;
        }
    }
    let ans = n as u32 - max_lis as u32;
    writeln!(output, "{}", ans).unwrap();
}
