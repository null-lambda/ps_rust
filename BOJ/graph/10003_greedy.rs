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

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let _k: usize = input.value();
    let mut neighbors = vec![vec![]; n];
    for _ in 0..m {
        let u = input.value::<usize>() - 1;
        let v = input.value::<usize>() - 1;
        neighbors[u].push(v);
        neighbors[v].push(u);
    }

    let mut visited = vec![false; n];
    let mut visited_as_center = vec![false; n];
    let mut res = vec![];
    for u in 0..n {
        if visited[u] {
            continue;
        }
        res.push(u);
        visited[u] = true;
        visited_as_center[u] = true;
        for &v in &neighbors[u] {
            if visited_as_center[v] {
                continue;
            }
            visited[v] = true;
            visited_as_center[v] = true;
            for &w in &neighbors[v] {
                visited[w] = true;
            }
        }
    }

    writeln!(output, "{}", res.len()).unwrap();
    for u in res {
        write!(output, "{} ", u + 1).unwrap();
    }
}
