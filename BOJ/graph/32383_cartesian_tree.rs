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

    const P: u64 = 1_000_000_007;
    let mut size = vec![0; n];
    fn dfs(
        neighbors: &[Vec<u32>],
        hs: &[i32],
        size: &mut [usize],
        u: u32,
        parent: u32,
        ans: &mut u64,
    ) {
        size[u as usize] = 1;
        for &v in &neighbors[u as usize] {
            if v == parent {
                continue;
            }
            dfs(neighbors, hs, size, v, u, ans);
            size[u as usize] += size[v as usize];
        }
        if u != parent {
            let n = neighbors.len();
            let dh = (hs[u as usize] - hs[parent as usize]).abs() as u64;
            let s1 = size[u as usize] as u64;
            let s2 = n as u64 - s1;
            *ans += dh * dh % P * s1 % P * s2 % P;
            *ans %= P;
        }
    }
    let root = (0..n).max_by_key(|&i| hs[i]).unwrap() as u32;
    let mut ans = 0;
    dfs(&neighbors, &hs, &mut size, root, root, &mut ans);
    writeln!(output, "{}", ans).unwrap();
}
