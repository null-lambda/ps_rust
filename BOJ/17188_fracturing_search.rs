use std::{collections::BinaryHeap, io::Write};

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
    let k: usize = input.value();
    let c: usize = input.value();
    let mut xs = vec![0u32; n * k];
    for u in 0..n {
        for j in 0..k {
            xs[j * n..][u] = input.value();
        }
    }
    let xs = |j: usize| &xs[j * n..];
    assert!(n < 512);

    let init = || {
        let forced = [0u128; 4];
        let banned = [0u128; 4];
        let mut visited = [0u128; 4];
        let mut selected = vec![];
        for j in 0..k {
            let u = (0..n)
                .filter(|&u| (visited[u / 128] >> (u % 128)) & 1 == 0)
                .max_by_key(|&u| xs(j)[u])
                .unwrap();
            selected.push(u as u16);
            visited[u / 128] |= 1 << (u % 128);
        }

        let score = (0..k)
            .map(|j| selected.iter().map(|&u| xs(j)[u as usize]).max().unwrap())
            .sum::<u32>();
        Some((score, selected, forced, banned))
    };

    let mut pq = BinaryHeap::from_iter(init());
    for _ in 0..c - 1 {
        let (_, selected, forced, banned) = pq.pop().unwrap();
        'outer: for j0 in 0..k {
            let mut forced = forced;
            let mut banned = banned;
            for j in 0..j0 {
                let u = selected[j] as usize;
                forced[u / 128] |= 1 << (u % 128);
            }
            {
                let u = selected[j0] as usize;
                if (forced[u / 128] >> (u % 128)) & 1 != 0 {
                    continue 'outer;
                }
                banned[u / 128] |= 1 << (u % 128);
            }

            let mut selected = selected[..j0].to_vec();
            let mut visited: [u128; 4] = std::array::from_fn(|i| forced[i] | banned[i]);
            for j in j0..k {
                let Some(u) = (0..n)
                    .filter(|&u| (visited[u / 128] >> (u % 128)) & 1 == 0)
                    .max_by_key(|&u| xs(j)[u])
                else {
                    continue 'outer;
                };
                selected.push(u as u16);
                visited[u / 128] |= 1 << (u % 128);
            }

            let score = (0..k)
                .map(|j| selected.iter().map(|&u| xs(j)[u as usize]).max().unwrap())
                .sum::<u32>();
            pq.push((score, selected, forced, banned));
        }
    }

    writeln!(output, "{}", pq.pop().unwrap().0).ok();
}
