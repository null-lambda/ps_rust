use std::{cmp::Reverse, collections::BinaryHeap, io::Write};

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
    let l: usize = input.value();

    const INF: i32 = 1e9 as i32;
    const UNSET: u32 = !0;
    let mut dist = vec![INF; k];
    let mut color = vec![UNSET; k];
    let mut queue = BinaryHeap::new();

    for i in 0..n {
        let u = input.value::<usize>() - 1;
        if dist[u] == INF {
            dist[u] = 0;
            color[u] = i as u32;
            queue.push((Reverse(0), u, color[u]));
        } else {
            writeln!(output, "0").unwrap();
            return;
        }
    }

    let mut neighbors = vec![vec![]; k];
    for _ in 0..l {
        let u = input.value::<usize>() - 1;
        let v = input.value::<usize>() - 1;
        let d = input.value::<i32>();
        neighbors[u].push((v, d));
        neighbors[v].push((u, d));
    }

    let mut ans = i32::MAX;
    while let Some((Reverse(d), u, c)) = queue.pop() {
        if dist[u] < d {
            continue;
        }
        color[u] = c;

        for &(v, w) in &neighbors[u] {
            let dv_new = dist[u] + w;
            if color[v] != UNSET && color[u] != color[v] {
                ans = ans.min(dist[v] + dv_new);
            }
            if dist[v] > dv_new {
                dist[v] = dv_new;
                queue.push((Reverse(dv_new), v, color[u]));
            }
        }
    }
    ans *= 3;
    writeln!(output, "{}", ans).unwrap();
}
