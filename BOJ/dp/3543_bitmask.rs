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

fn submasks(mask: u32) -> impl Iterator<Item = u32> {
    std::iter::successors(Some(mask), move |&sub| {
        (sub != 0).then(|| ((sub - 1) & mask))
    })
    .chain(std::iter::once(0))
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let parse_node = |b: u8| (b - b'L') as u32;
    let node_to_char = |n: u32| (n as u8 + b'L') as char;

    let n_verts = 15;
    let mut independent = vec![true; 1 << n_verts];

    let n: usize = input.value();
    let mut edges = vec![];
    for _ in 0..n {
        let u = parse_node(input.token().as_bytes()[0]);
        let v = parse_node(input.token().as_bytes()[0]);
        edges.push((u, v));
    }

    for &(u, v) in &edges {
        let edge_mask = 1 << u | 1 << v;
        for mask in 0..1 << n_verts {
            if mask & edge_mask == edge_mask {
                independent[mask] = false;
            }
        }
    }

    let inf = u32::MAX / 3;
    let mut dp = vec![(inf, 0); 1 << n_verts];
    dp[0] = (0, 0);
    for mask in 1..1 << n_verts {
        for sub in submasks(mask) {
            let mask = mask as usize;
            let sub = sub as usize;
            if sub & mask == sub && independent[sub] {
                let new_cost = dp[mask ^ sub].0 + 1;
                if new_cost < dp[mask].0 {
                    dp[mask] = (new_cost, sub);
                }
            }
        }
    }

    let mut current = (1 << n_verts) - 1;
    let (min_cost, mut sub) = dp[current];
    let mut levels = vec![0; n_verts];
    for lv in (0..min_cost).rev() {
        for i in 0..n_verts {
            if (sub >> i) & 1 != 0 {
                levels[i] = lv;
            }
        }

        let (_, next_sub) = dp[current ^ sub];
        current = current ^ sub;
        sub = next_sub;
    }

    writeln!(output, "{}", min_cost - 2).unwrap();
    for &(u, v) in &edges {
        let (u, v) = match levels[u as usize].cmp(&levels[v as usize]) {
            Ordering::Less => (u, v),
            Ordering::Greater => (v, u),
            _ => panic!(),
        };
        writeln!(output, "{} {}", node_to_char(u), node_to_char(v)).unwrap();
    }
}
