use std::{cmp::Reverse, io::Write};

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

    let n_edges: usize = input.value();
    let n = n_edges / 2 - 1;
    let mut ps = (0..n_edges)
        .map(|_| (input.value::<i32>(), input.value::<i32>()))
        .skip(1)
        .step_by(2)
        .peekable();

    let mut x_range = vec![[0, 0]];
    let mut ys = vec![0];
    for _ in 0..n {
        let (x, y) = ps.next().unwrap();
        let (x_next, _) = *ps.peek().unwrap();
        x_range.push([x, x_next]);
        ys.push(y);
    }

    let k: usize = input.value();

    const UNSET: u32 = u32::MAX;
    let mut parent = vec![UNSET; n + 1];
    let mut children = vec![[UNSET; 2]; n + 1];

    // Build Cartesian tree from inorder traversal, with monotone stack
    let mut stack = vec![(i32::MIN, 0)];
    for u in 1..=n as u32 {
        let h = ys[u as usize];

        let mut c = None;
        while stack.last().unwrap().0 > h {
            c = stack.pop();
        }
        let (_, p) = *stack.last().unwrap();
        parent[u as usize] = p;
        children[p as usize][1] = u;

        if let Some((_, c)) = c {
            parent[c as usize] = u;
            children[u as usize][0] = c;
        }
        stack.push((h, u));
    }

    fn dnc(
        children: &[[u32; 2]],
        parent: &[u32],
        x_range: &mut [[i32; 2]],
        ys: &[i32],
        u: usize,
        values: &mut Vec<u64>,
    ) -> u64 {
        let mut acc = [0; 2];
        for i in 0..2 {
            if children[u][i] == UNSET {
                continue;
            }
            acc[i] = dnc(
                children,
                parent,
                x_range,
                ys,
                children[u][i] as usize,
                values,
            );
            x_range[u][i] = x_range[children[u][i] as usize][i];
        }
        if acc[0] > acc[1] {
            acc.swap(0, 1);
        }

        let dx = x_range[u][1] - x_range[u][0];
        let dy = ys[u] - ys[parent[u] as usize];
        let value_u = dx as u64 * dy as u64;
        values.push(acc[0]);
        acc[1] + value_u
    }

    let mut values = vec![];
    let root = (1..=n as u32).find(|&u| parent[u as usize] == 0).unwrap() as usize;
    let value_root = dnc(&children, &parent, &mut x_range, &ys, root, &mut values);
    values.push(value_root);

    values.select_nth_unstable_by_key(k, |&x| Reverse(x));
    let ans: u64 = values[..k].iter().sum();

    writeln!(output, "{}", ans).unwrap();
}
