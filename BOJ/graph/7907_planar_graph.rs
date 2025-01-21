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

const UNSET: u32 = !0;

#[derive(Clone)]
struct SmallVec2([u32; 2]);

impl SmallVec2 {
    fn new() -> Self {
        Self([UNSET; 2])
    }

    fn push(&mut self, value: u32) {
        if self.0[0] == UNSET {
            self.0[0] = value;
        } else {
            debug_assert!(self.0[1] == UNSET);
            self.0[1] = value;
        }
    }

    fn sort_unstable_by_key(&mut self, key: impl Fn(u32) -> i32) {
        if self.0[1] == UNSET {
            return;
        }
        if key(self.0[0] as u32) > key(self.0[1] as u32) {
            self.0.swap(0, 1);
        }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let k: usize = input.value();
    let ps: Vec<[i32; 2]> = (0..n)
        .map(|_| [input.value(), -input.value::<i32>()])
        .collect();
    let mut neighbors = vec![SmallVec2::new(); n];
    let mut indegree = vec![0; n];
    for _ in 0..m {
        let mut u = input.value::<u32>() - 1;
        let mut v = input.value::<u32>() - 1;
        if ps[u as usize] > ps[v as usize] {
            std::mem::swap(&mut u, &mut v);
        }
        neighbors[u as usize].push(v);
        indegree[v as usize] += 1;
    }

    for u in 0..n {
        neighbors[u].sort_unstable_by_key(|v| ps[v as usize][1]);
    }

    let ts1 = {
        let mut indegree = indegree.clone();
        let mut stack = vec![0];
        let mut time = vec![0; n];
        let mut timer = 0u32;
        while let Some(u) = stack.pop() {
            time[u as usize] = timer;
            timer += 1;
            for &v in neighbors[u as usize].0.iter().filter(|&&v| v != UNSET) {
                indegree[v as usize] -= 1;
                if indegree[v as usize] == 0 {
                    stack.push(v);
                }
            }
        }
        time
    };

    let ts2 = {
        let mut indegree = indegree.clone();
        let mut stack = vec![0];
        let mut time = vec![0; n];
        let mut timer = 0u32;
        while let Some(u) = stack.pop() {
            time[u as usize] = timer;
            timer += 1;
            for &v in neighbors[u as usize]
                .0
                .iter()
                .rev()
                .filter(|&&v| v != UNSET)
            {
                indegree[v as usize] -= 1;
                if indegree[v as usize] == 0 {
                    stack.push(v);
                }
            }
        }
        time
    };

    for _ in 0..k {
        let u = input.value::<usize>() - 1;
        let v = input.value::<usize>() - 1;
        if (ts1[u] < ts1[v]) == (ts2[u] < ts2[v]) {
            writeln!(output, "TAK").unwrap();
        } else {
            writeln!(output, "NIE").unwrap();
        }
    }
}
