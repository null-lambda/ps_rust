mod simple_io {
    pub struct InputAtOnce(std::str::SplitAsciiWhitespace<'static>);

    impl InputAtOnce {
        pub fn token(&mut self) -> &str {
            self.0.next().unwrap_or_default()
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().unwrap()
        }
    }

    pub fn stdin_at_once() -> InputAtOnce {
        let buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let buf = Box::leak(buf.into_boxed_str());
        let iter = buf.split_ascii_whitespace();
        InputAtOnce(iter)
    }
}

fn main() {
    let mut input = simple_io::stdin_at_once();

    let r: usize = input.value();
    let c: usize = input.value();

    let idx = |s: &[u8]| {
        let mut i = 0;
        let mut j = 0;

        for c in s {
            match c {
                b'A'..=b'Z' => j = j * 26 + (c - b'A') as usize + 1,
                b'0'..=b'9' => i = i * 10 + (c - b'0') as usize,
                _ => panic!(),
            }
        }
        (i - 1) * c + j - 1
    };

    let mut neighbors = vec![vec![]; r * c];
    for u in 0..r * c {
        let cell = input.token();
        if cell == "." {
            continue;
        }
        cell.split(|c| c == '+').for_each(|s| {
            let v = idx(s.as_bytes());
            neighbors[u].push(v);
        });
    }

    // detect cycle
    let has_cycle = || {
        fn dfs(
            u: usize,
            neighbors: &[Vec<usize>],
            visited: &mut [bool],
            path: &mut [bool],
        ) -> bool {
            if visited[u] {
                return false;
            }
            if path[u] {
                return true;
            }
            path[u] = true;
            for &v in &neighbors[u] {
                if dfs(v, neighbors, visited, path) {
                    return true;
                }
            }
            path[u] = false;
            visited[u] = true;
            false
        }

        let mut visited = vec![false; r * c];
        let mut path = vec![false; r * c];
        for start in 0..r * c {
            if dfs(start, &neighbors, &mut visited, &mut path) {
                return true;
            }
        }
        false
    };

    if has_cycle() {
        println!("yes");
    } else {
        println!("no");
    }
}
