use std::{
    collections::{HashMap, VecDeque},
    io::Write,
    mem::MaybeUninit,
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

struct MinColorDist {
    base: i32,
    deltas: HashMap<u32, i32>,
}

impl MinColorDist {
    fn singleton(color: u32) -> Self {
        Self {
            base: 0,
            deltas: [(color, 0)].into(),
        }
    }

    fn move_up(&mut self) {
        self.base += 1;
    }

    fn union(&mut self, mut other: MinColorDist, ans: &mut i32) {
        if self.deltas.len() < other.deltas.len() {
            std::mem::swap(self, &mut other);
        }

        for (k, d_other) in other.deltas {
            self.deltas
                .entry(k)
                .and_modify(|d_self| {
                    *ans = (*ans).min(*d_self + self.base + d_other + other.base);
                    *d_self = (*d_self).min(d_other + other.base - self.base)
                })
                .or_insert(d_other + other.base - self.base);
        }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let mut colors: Vec<MaybeUninit<MinColorDist>> = (0..n)
        .map(|_| input.value::<u32>() - 1)
        .map(|c| MaybeUninit::new(MinColorDist::singleton(c)))
        .collect();

    let mut neighbors = vec![vec![]; n];
    for _ in 0..n - 1 {
        let u = input.value::<usize>() - 1;
        let v = input.value::<usize>() - 1;
        neighbors[u].push(v);
        neighbors[v].push(u);
    }

    let root = 0;
    let mut parent = vec![root; n];
    let mut stack = vec![root];
    while let Some(u) = stack.pop() {
        for &v in &neighbors[u] {
            if v != parent[u] {
                parent[v] = u;
                stack.push(v);
            }
        }
    }

    let mut indegree = vec![0; n];
    for i in 0..n {
        indegree[parent[i]] += 1;
    }

    let mut ans = i32::MAX;
    let mut queue: VecDeque<usize> = (0..n).filter(|&i| indegree[i] == 0 && i != root).collect();
    while let Some(u) = queue.pop_front() {
        let p = parent[u];

        unsafe {
            let mut colors_u = colors[u].assume_init_read();
            colors_u.move_up();
            colors[p].assume_init_mut().union(colors_u, &mut ans);
        }

        indegree[p] -= 1;
        if indegree[p] == 0 {
            queue.push_back(p);
        }
    }

    writeln!(output, "{}", ans).unwrap();
}
