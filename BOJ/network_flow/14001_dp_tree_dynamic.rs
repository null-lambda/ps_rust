use std::{io::Write, vec};

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
    let m: usize = input.value();
    let parent = |u: usize| u / 2;
    let nth_parent = |u: usize, n: usize| u >> n;
    let children = |u: usize| [u << 1, u << 1 | 1];
    let depth = |u: usize| (usize::BITS - 1 - u.leading_zeros()) as usize;

    let mut sink_cap = vec![0i32; n + 1];
    for i in 1..=n {
        sink_cap[i] = input.value();
    }

    const INF: i32 = 1 << 28;
    const UNSET: u32 = !0;
    let inc = |(d, u): (i32, u32), delta: _| (d + delta, u);
    let mut closest_hole = vec![(INF, UNSET); n + 1];
    for i in (1..=n).rev() {
        if sink_cap[i] > 0 {
            closest_hole[i] = (0, i as u32);
        }
        for c in children(i) {
            if c <= n {
                closest_hole[i] = closest_hole[i].min(inc(closest_hole[c], 1));
            }
        }
    }

    let mut flow_upward = vec![0i32; n + 1];
    let mut min_cost = 0;
    for _ in 0..m {
        let u0: usize = input.value();

        let mut min_path = (INF, UNSET);
        {
            let mut delta = 0;
            let mut u = u0;
            min_path = min_path.min(inc(closest_hole[u], delta));
            while u >= 1 {
                delta += if flow_upward[u] >= 0 { 1 } else { -1 };
                let p = parent(u);

                if sink_cap[p] > 0 {
                    min_path = min_path.min(inc((0, p as u32), delta));
                }
                let c = children(p)[0] ^ children(p)[1] ^ u;
                if c <= n && c != u {
                    min_path = min_path.min(inc(
                        closest_hole[c],
                        delta + if flow_upward[c] <= 0 { 1 } else { -1 },
                    ));
                }

                u = p;
            }
        }

        let (l, v0) = min_path;
        let v0 = v0 as usize;
        assert!(l < INF);
        min_cost += l;
        write!(output, "{} ", min_cost).unwrap();

        let lca = {
            let mut u = u0;
            let mut v = v0;
            if depth(u) < depth(v) {
                std::mem::swap(&mut u, &mut v);
            }
            u = nth_parent(u, depth(u) - depth(v));
            while u != v {
                u = parent(u);
                v = parent(v);
            }
            u
        };

        {
            let mut u = u0;
            while u != lca {
                flow_upward[u] += 1;
                u = parent(u);
            }

            let mut v = v0;
            while v != lca {
                flow_upward[v] -= 1;
                v = parent(v);
            }
        }

        {
            sink_cap[v0] -= 1;
            for (mut u, top) in [(u0, lca), (v0, 1)] {
                while u >= top {
                    closest_hole[u] = (INF, UNSET);
                    if sink_cap[u] > 0 {
                        closest_hole[u] = (0, u as u32);
                    }
                    for c in children(u) {
                        if c <= n {
                            closest_hole[u] = closest_hole[u].min(inc(
                                closest_hole[c],
                                if flow_upward[c] <= 0 { 1 } else { -1 },
                            ));
                        }
                    }
                    u = parent(u);
                }
            }
        }
    }
    writeln!(output).unwrap();
}
