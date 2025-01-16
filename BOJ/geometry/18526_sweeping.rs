use std::{cmp::Ordering, collections::BTreeMap, io::Write};

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

#[derive(Clone, Debug)]
struct HalfArc {
    is_upper: bool,
    x: i32,
    y: i32,
    r: i32,
}

impl HalfArc {
    fn x_bounds(&self) -> (i32, i32) {
        (self.x - self.r, self.x + self.r)
    }

    fn cmp_with_point(&self, [px, py]: [i32; 2]) -> Option<Ordering> {
        use Ordering::*;
        let (s0, e1) = self.x_bounds();
        if !(s0..=e1).contains(&px) {
            return None;
        }
        let (dx, dy) = (px - self.x, py - self.y);
        let sq = |i: i32| i as i64 * i as i64;
        Some(if self.is_upper {
            match (0.cmp(&dy), sq(self.r).cmp(&(sq(dx) + sq(dy)))) {
                (Less, Less) => Less,
                (Less | Equal, Equal) => Equal,
                _ => Greater,
            }
        } else {
            match (0.cmp(&dy), sq(self.r).cmp(&(sq(dx) + sq(dy)))) {
                (Greater, Less) => Greater,
                (Greater | Equal, Equal) => Equal,
                _ => Less,
            }
        })
    }
}

impl PartialEq for HalfArc {
    fn eq(&self, other: &Self) -> bool {
        (self.x, self.y, self.r, self.is_upper) == (other.x, other.y, other.r, other.is_upper)
    }
}

impl Eq for HalfArc {}

impl PartialOrd for HalfArc {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        let (s0, _e0) = self.x_bounds();
        let (s1, _e1) = other.x_bounds();
        if (self.x, self.y, self.r) == (other.x, other.y, other.r) {
            Some(self.is_upper.cmp(&other.is_upper))
        } else if s0 <= s1 {
            self.cmp_with_point([s1, other.y])
        } else {
            other.cmp_with_point([s0, self.y]).map(Ordering::reverse)
        }
    }
}

impl Ord for HalfArc {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.partial_cmp(other).unwrap()
    }
}

#[derive(Debug)]
enum EventType {
    Add,
    Remove,
}

use EventType::*;

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let q: usize = input.value();
    let n_nodes = n + q + 1;
    let root = n_nodes - 1;

    let mut events = vec![];
    for u in 0..n + q {
        let x = input.value();
        let y = input.value();
        let r = input.value();

        let arc_upper = HalfArc {
            is_upper: true,
            x,
            y,
            r,
        };
        events.push((x - r, arc_upper.clone(), Add, u as u32));
        events.push((x + r, arc_upper, Remove, u as u32));
    }
    events.sort_unstable_by_key(|&(x, ..)| x);

    const UNSET: u32 = u32::MAX;
    let mut active: BTreeMap<HalfArc, u32> = Default::default();
    let mut parent = vec![UNSET as u32; n + q + 1];
    parent[root as usize] = root as u32;
    for (_x, arc_upper, ty, u) in events {
        let arc_lower = HalfArc {
            is_upper: false,
            ..arc_upper
        };
        match ty {
            Add => {
                parent[u as usize] = if let Some((arc_v, &v)) = active.range(&arc_upper..).next() {
                    if arc_v.is_upper {
                        v
                    } else {
                        parent[v as usize]
                    }
                } else {
                    root as u32
                };

                active.insert(arc_upper, u as u32);
                active.insert(arc_lower, u as u32);
            }
            Remove => {
                active.remove(&arc_upper);
                active.remove(&arc_lower);
            }
        }
    }

    let mut degree = vec![1; n_nodes];
    degree[root] += 2;
    for &p in &parent {
        degree[p as usize] += 1;
    }

    let mut ans = vec![0u32; q];

    #[derive(Default, Clone)]
    struct NodeAgg {
        score: [u32; 2],
        score_horizontal: [u32; 2],
    }

    impl NodeAgg {
        fn pull_from(&mut self, other: &Self) {
            self.score[0] += other.score[0].max(other.score[1]);
            self.score[1] += other.score[0];
            self.score_horizontal[0] += other.score[0];
            self.score_horizontal[1] += other.score[0].max(other.score[1]);
        }

        fn finalize(&mut self, u: usize, n: usize, ans: &mut [u32]) {
            let query_idx = |u: usize| u.checked_sub(n);
            self.score[1] += 1;
            if let Some(i) = query_idx(u) {
                ans[i] = self.score[0].max(self.score[1]);
                self.score = self.score_horizontal;
            }
        }
    }

    let mut dp = vec![NodeAgg::default(); n_nodes];
    for mut u in 0..n_nodes as u32 {
        while degree[u as usize] == 1 {
            let p = parent[u as usize];
            degree[u as usize] -= 1;
            degree[p as usize] -= 1;

            let mut dp_u = std::mem::take(&mut dp[u as usize]);
            dp_u.finalize(u as usize, n, &mut ans);
            dp[p as usize].pull_from(&dp_u);

            u = p;
        }
    }

    for a in ans {
        writeln!(output, "{}", a).unwrap();
    }
}
