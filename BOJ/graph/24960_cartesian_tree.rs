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
    let hs: Vec<i32> = (0..n).map(|_| input.value()).collect();

    let mut parent = vec![0; n + 1];
    // let mut children = vec![[0; 2]; n + 1];

    // Build Cartesian tree from inorder traversal, with monotone stack
    let mut stack = vec![(i32::MAX, 0)];

    for (u, &h) in hs.iter().enumerate() {
        let u = u as u32 + 1;

        let mut c = None;
        while stack.last().unwrap().0 < h {
            c = stack.pop();
        }
        let (_, p) = *stack.last().unwrap();
        parent[u as usize] = p;
        // children[p as usize][1] = u;

        if let Some((_, c)) = c {
            parent[c as usize] = u;
            // children[u as usize][0] = c;
        }
        stack.push((h, u));
    }

    let mut neighbors = vec![vec![]; n];
    for u in 1..=n as u32 {
        let p = parent[u as usize];
        if p != 0 {
            neighbors[p as usize - 1].push(u - 1);
            neighbors[u as usize - 1].push(p - 1);
        }
    }

    fn dfs_depth(
        neighbors: &[Vec<u32>],
        depth: &mut [u32],
        max_sub_depth: &mut [u32],
        u: u32,
        parent: u32,
    ) {
        max_sub_depth[u as usize] = depth[u as usize] as u32;
        for &v in neighbors[u as usize].iter() {
            if v == parent {
                continue;
            }
            depth[v as usize] = depth[u as usize] + 1;
            dfs_depth(neighbors, depth, max_sub_depth, v, u);
            max_sub_depth[u as usize] = max_sub_depth[u as usize].max(max_sub_depth[v as usize]);
        }
    }

    let mut depth = vec![0; n];
    let mut max_sub_depth = vec![0; n];
    let root = (1..=n as u32).find(|&u| parent[u as usize] == 0).unwrap() - 1;
    dfs_depth(&neighbors, &mut depth, &mut max_sub_depth, root, root);

    let mut ans = 0;
    for u in 0..n {
        let d_join = depth[u];
        let mut sub_depths = neighbors[u]
            .iter()
            .filter(|&&v| v + 1 != parent[u + 1])
            .map(|&v| max_sub_depth[v as usize]);

        ans = ans.max(match (sub_depths.next(), sub_depths.next()) {
            (None, None) => d_join,
            (Some(d_u), None) => d_u,
            (Some(d_u), Some(d_v)) => d_u + d_v - d_join,
            _ => unreachable!(),
        });
    }

    writeln!(output, "{}", ans).unwrap();
}
