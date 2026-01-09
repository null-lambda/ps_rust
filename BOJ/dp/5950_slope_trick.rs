use std::{collections::BTreeMap, io::Write};

use buffered_io::BufReadExt;

mod buffered_io {
    use std::io::{BufRead, BufReader, BufWriter, Stdin, Stdout};
    use std::str::FromStr;

    pub trait BufReadExt: BufRead {
        fn line(&mut self) -> String {
            let mut buf = String::new();
            self.read_line(&mut buf).unwrap();
            buf
        }

        fn skip_line(&mut self) {
            self.line();
        }

        fn token(&mut self) -> String {
            loop {
                let buf = self.fill_buf().unwrap();
                if buf.is_empty() {
                    return String::new();
                }

                let mut i = 0;
                while i < buf.len() && buf[i].is_ascii_whitespace() {
                    i += 1;
                }

                let should_break = i < buf.len();
                self.consume(i);
                if should_break {
                    break;
                }
            }

            let mut res = vec![];
            loop {
                let buf = self.fill_buf().unwrap();
                if buf.is_empty() {
                    break;
                }

                let mut i = 0;
                while i < buf.len() && !buf[i].is_ascii_whitespace() {
                    i += 1;
                }
                res.extend_from_slice(&buf[..i]);

                let should_break = i < buf.len();
                self.consume(i);
                if should_break {
                    break;
                }
            }

            String::from_utf8(res).unwrap()
        }

        fn try_value<T: FromStr>(&mut self) -> Option<T> {
            self.token().parse().ok()
        }

        fn value<T: FromStr>(&mut self) -> T {
            self.try_value().unwrap()
        }
    }

    impl<R: BufRead> BufReadExt for R {}

    pub fn stdin() -> BufReader<Stdin> {
        BufReader::new(std::io::stdin())
    }

    pub fn stdout() -> BufWriter<Stdout> {
        BufWriter::new(std::io::stdout())
    }
}

const INF: i64 = 1 << 30;

#[derive(Clone, Default, Debug)]
struct NodeAgg {
    y0: i64,
    bps: BTreeMap<i64, i64>,
}

impl NodeAgg {
    fn new(y0: i64) -> Self {
        Self {
            y0,
            bps: Default::default(),
        }
    }

    fn limit(&mut self, cap: i64) {
        if cap == 0 {
            *self = Default::default();
            return;
        }

        let mut x = 0;
        let mut y = std::mem::take(&mut self.y0);
        let mut v = 0;
        self.bps.insert(INF, 0);

        while let Some((&nx, &dv)) = self.bps.first_key_value() {
            if y == 0 && nx == 0 && dv <= cap {
                return;
            }

            let ny = y + v * (nx - x);
            let nv = v + dv;

            if nx != 0 && ny <= cap * nx {
                assert!(cap > v);
                let a = y - v * x;
                let b = cap - v;
                let q = a / b;
                let r = a % b;
                if r == 0 {
                    if q <= INF {
                        *self.bps.entry(q).or_default() += v - cap;
                    }
                } else {
                    let s = (y + v * (q + 1 - x)) - cap * (q + 1);
                    if q + 1 <= INF {
                        *self.bps.entry(q).or_default() += s;
                        *self.bps.entry(q + 1).or_default() += v - cap - s;
                    }
                }
                break;
            }

            self.bps.pop_first();
            y = ny;
            v = nv;
            x = nx;
        }
        *self.bps.entry(0).or_default() += cap;
    }

    fn rake(&mut self, mut other: Self) {
        if self.bps.len() < other.bps.len() {
            std::mem::swap(self, &mut other);
        }
        self.y0 += other.y0;
        for (x, vy) in other.bps {
            *self.bps.entry(x).or_default() += vy;
        }
    }
}

fn main() {
    let mut input = buffered_io::stdin();
    let mut output = buffered_io::stdout();

    let n: usize = input.value();
    let k: usize = input.value();
    let mut deg = vec![1u32; n];
    let mut parent = vec![0u32; n];
    let mut load = vec![0; n];
    let mut cap = vec![0; n];
    for u in 1..n {
        parent[u] = input.value::<u32>() - 1;
        deg[parent[u] as usize] += 1;
        load[u] = input.value::<i64>();
        cap[u] = input.value::<i64>();
    }

    let n = deg.len();
    deg[0 as usize] += 2;
    let mut toposort = vec![];

    let mut dp = load
        .into_iter()
        .map(|y0| NodeAgg::new(y0))
        .collect::<Vec<_>>();
    for mut u in 0..n as u32 {
        while deg[u as usize] == 1 {
            let p = parent[u as usize];
            deg[u as usize] -= 1;
            deg[p as usize] -= 1;

            let mut dp_u = std::mem::take(&mut dp[u as usize]);
            dp_u.limit(cap[u as usize]);
            dp[p as usize].rake(dp_u);

            toposort.push(u);

            u = p;
        }
    }
    toposort.push(0);

    let mut queries = vec![];
    for i in 0..k as u32 {
        let t = input.value::<i64>();
        queries.push((t, i));
    }
    queries.sort_unstable();

    let f = std::mem::take(&mut dp[0]);
    let mut f_bps = f.bps.into_iter().peekable();

    let mut x = 0;
    let mut y = f.y0;
    let mut v = 0;
    let mut ans = vec![!0; k];
    for (t, i) in queries {
        while let Some((nx, dv)) = f_bps.next_if(|&(x, _)| x <= t) {
            y += v * (nx - x);
            v += dv;
            x = nx;
        }

        ans[i as usize] = y + (t - x) * v;

        //
    }

    for y in ans {
        writeln!(output, "{}", y).unwrap();
    }
}
