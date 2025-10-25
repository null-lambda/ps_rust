use std::io::Write;

mod simple_io {
    pub struct InputAtOnce {
        iter: std::str::SplitAsciiWhitespace<'static>,
    }

    impl InputAtOnce {
        pub fn token(&mut self) -> &'static str {
            self.iter.next().unwrap_or_default()
        }

        pub fn try_value<T: std::str::FromStr>(&mut self) -> Option<T> {
            self.token().parse().ok()
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.try_value().unwrap()
        }
    }

    pub fn stdin() -> InputAtOnce {
        let buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let buf = Box::leak(Box::new(buf));
        let iter = buf.split_ascii_whitespace();
        InputAtOnce { iter }
    }

    pub fn stdout() -> std::io::BufWriter<std::io::Stdout> {
        std::io::BufWriter::new(std::io::stdout())
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let ks: Vec<i32> = (0..n).map(|_| input.value()).collect();
    let m = 2520; // lcm(1, ..., 10)

    let mut adj = vec![vec![]; n];
    for u in 0..n {
        for _ in 0..input.value() {
            let v = input.value::<i32>() - 1;
            adj[u].push(v);
        }
    }

    let r = n * m;
    let mut next: Vec<i32> = (0..r as i32).collect();
    for u in 0..n {
        for c in 0..m {
            let p = u * m + c;

            let d = (c as i32 + ks[u]).rem_euclid(m as i32) as usize;
            let v = adj[u][d % adj[u].len()] as usize;
            let q = v * m + d;

            next[p] = q as i32;
        }
    }

    const UNSET: i32 = -1;

    let mut indegree = vec![0u32; r];
    for u in 0..r {
        indegree[next[u] as usize] += 1;
    }

    // Cycle-forest decomposition
    let mut toposort: Vec<_> = (0..r as i32)
        .filter(|&u| indegree[u as usize] == 0)
        .collect();

    // Process a forest
    let mut timer = 0;
    while let Some(&u) = toposort.get(timer) {
        timer += 1;

        let p = next[u as usize];
        indegree[p as usize] -= 1;
        if indegree[p as usize] == 0 {
            toposort.push(p);
        }
    }

    // Process cycles
    let mut cycle_size = vec![UNSET; r];
    let mut active = vec![false; n];
    for mut u in 0..r {
        if indegree[u] == 0 {
            continue;
        }
        indegree[u] = 0;

        let mut cycle = vec![];
        loop {
            indegree[u] = 0;
            toposort.push(u as i32);

            let x = u / m;
            if !active[x] {
                active[x] = true;
                cycle.push(x as i32);
            }

            let p = next[u];
            if indegree[p as usize] == 0 {
                cycle_size[p as usize] = cycle.len() as i32;
                for x in cycle {
                    active[x as usize] = false;
                }
                break;
            }

            u = p as usize;
        }
    }

    for &u in toposort.iter().rev() {
        if cycle_size[u as usize] != UNSET {
            continue;
        }
        let p = next[u as usize];
        cycle_size[u as usize] = cycle_size[p as usize];
    }

    for _ in 0..input.value() {
        let u = input.value::<usize>() - 1;
        let c = input.value::<i32>().rem_euclid(m as i32) as usize;
        let p = u * m + c;
        writeln!(output, "{}", cycle_size[p]).unwrap();
    }
}
