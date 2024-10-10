use std::io::Write;

#[allow(dead_code)]
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
        InputAtOnce(buf.split_ascii_whitespace())
    }

    pub fn stdout_buf() -> std::io::BufWriter<std::io::Stdout> {
        std::io::BufWriter::new(std::io::stdout())
    }
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout_buf();

    let n: usize = input.value();

    let mut edges = vec![];
    let mut neighbors = vec![vec![]; n];
    let mut degree = vec![0; n];
    for u in 0..n {
        for v in 0..n {
            let d: i32 = input.value();
            degree[u] += d;
            if d > 0 && u < v {
                let i_edge = edges.len();
                neighbors[u].push(i_edge);
                neighbors[v].push(i_edge);
                edges.push((u, v, d));
            }
        }
    }

    let has_odd = degree.iter().any(|&d| d % 2 == 1);
    if has_odd {
        writeln!(output, "-1").unwrap();
        return;
    }

    // Hierholzer's algorithm
    // Find an euler path in a connected graph (if exists)
    let mut stack = vec![0usize];
    let mut path = vec![];
    while let Some(&u) = stack.last() {
        if let Some(&idx) = neighbors[u].last() {
            let (v, w, d) = &mut edges[idx];
            *d -= 1;
            if *d <= 0 {
                neighbors[u].pop();
                if *d < 0 {
                    // already erased inverted edge
                    continue;
                }
            }
            let v = if *v == u { *w } else { *v };
            stack.push(v);
        } else {
            path.push(u);
            stack.pop();
        }
    }

    for &u in &path {
        write!(output, "{} ", u + 1).unwrap();
    }
}
