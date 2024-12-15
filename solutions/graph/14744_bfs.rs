use std::{
    collections::{HashSet, VecDeque},
    io::Write,
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

    let n = input.value::<usize>() + 1;
    let m = input.value::<usize>() + 1;

    let n_pad = n + 2;
    let m_pad = m + 2;
    let mut wall_base = vec![true; n_pad * m_pad];

    let mut start = usize::MAX;
    let mut end = usize::MAX;
    for i in 1..=n {
        let row = input.token().as_bytes();
        for j in 1..=m {
            let b = row[j - 1];
            wall_base[i * m_pad + j] = b == b'#';
            match b {
                b'S' => start = i * m_pad + j,
                b'E' => end = i * m_pad + j,
                _ => {}
            }
        }
    }

    let mut obstacles = vec![];
    let mut periods = HashSet::new();
    for _ in 0..input.value() {
        let a: usize = input.value();
        let x: usize = input.value();
        let y: usize = input.value();
        obstacles.push((a, x, y));
        periods.insert(a);
    }

    let periods: Vec<_> = periods.into_iter().collect();
    let period = (1..=60)
        .find(|p| periods.iter().all(|a| p % a == 0))
        .unwrap()
        * 4;

    let mut wall = vec![wall_base; period];
    let mut toward = vec![vec![Vec::with_capacity(4); m_pad * n_pad]; period];
    for (a, x0, y0) in obstacles {
        for t in 0..period {
            let pos = |t| {
                let t = t % (a * 4);
                let (q, r) = (t / a, t % a);
                let (sx, sy) = match q {
                    0 => (0, r),
                    1 => (r, a),
                    2 => (a, a - r),
                    3 => (a - r, 0),
                    _ => panic!(),
                };
                (x0 + sx + 1, y0 + sy + 1)
            };
            let (x, y) = pos(t);
            let (nx, ny) = pos(t + 1);
            wall[t][x * m_pad + y] = true;
            toward[t][x * m_pad + y].push(nx * m_pad + ny);
        }
    }

    for t in 0..period {
        for i in 1..=n {
            for j in 1..=m {
                toward[t][i * m_pad + j].sort_unstable();
                toward[t][i * m_pad + j].dedup();
            }
        }
    }

    //     for t in 0..period {
    //         for row in wall[t].chunks_exact(m_pad) {
    //             for &b in row {
    //                 write!(output, "{}", if b { '#' } else { '.' }).unwrap();
    //             }
    //             writeln!(output).unwrap();
    //         }
    //         writeln!(output).unwrap();
    //     }

    let mut visited = wall.clone();
    let init = (0u32, 0, start);
    let mut queue: VecDeque<_> = [init].into();
    visited[0][start] = true;

    while let Some((d, t, u)) = queue.pop_back() {
        let t_next = (t + 1) % period;
        for v in [u, u - m_pad, u + m_pad, u - 1, u + 1] {
            if u == end {
                writeln!(output, "{}", d).unwrap();
                return;
            }
            if visited[t_next][v] || wall[t][v] && toward[t][v].contains(&u) {
                continue;
            }
            visited[t_next][v] = true;
            queue.push_front((d + 1, t_next, v));
        }
    }
    writeln!(output, "INF").unwrap();
}
