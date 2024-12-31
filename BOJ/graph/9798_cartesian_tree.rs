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

    pub fn stdin_at_once<'a>() -> InputAtOnce<'a> {
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
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let mut nodes: Vec<_> = (1..=n as u32)
        .map(|i| (input.value::<i32>(), input.value::<i32>(), i))
        .collect();
    nodes.sort_unstable();

    let mut parent = vec![u32::MAX; n + 1];
    let mut children = vec![[0; 2]; n + 1];

    // Build Cartesian tree from inorder traversal with monotone stack
    let mut stack = vec![(i32::MIN, 0)];

    for &(_, a, u) in &nodes {
        let mut c = None;
        while stack.last().unwrap().0 > a {
            c = stack.pop();
        }
        let (_, p) = *stack.last().unwrap();
        parent[u as usize] = p;
        children[p as usize][1] = u;

        if let Some((_, c)) = c {
            parent[c as usize] = u;
            children[u as usize][0] = c;
        }
        stack.push((a, u));
    }

    writeln!(output, "YES").unwrap();
    for i in 1..=n {
        writeln!(
            output,
            "{} {} {}",
            parent[i], children[i][0], children[i][1]
        )
        .unwrap();
    }
}
