use std::{io::Write, iter, ops::Range};

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
    let level_order: Vec<i64> = (0..n).map(|_| input.value()).collect();
    let mut inorder = vec![];
    fn dfs(level_order: &[i64], inorder: &mut Vec<i64>, i: usize) {
        if i < level_order.len() {
            dfs(level_order, inorder, i * 2 + 1);
            inorder.push(level_order[i]);
            dfs(level_order, inorder, i * 2 + 2);
        }
    }
    dfs(&level_order, &mut inorder, 0);

    let h_max = (n + 1).trailing_zeros() as usize;
    let mut acc = vec![vec![0i64; h_max]; h_max];
    let mut acc_min = vec![vec![0i64; h_max]; h_max];
    let mut ans = i64::MIN;
    for i in 1..=n {
        let x: i64 = inorder[i - 1];
        let h = i.trailing_zeros() as usize;

        for h0 in 0..h_max {
            for h1 in h0..h_max {
                if (h0..=h1).contains(&h) {
                    acc[h0][h1] += x;
                    ans = ans.max(acc[h0][h1] - acc_min[h0][h1]);
                    acc_min[h0][h1] = acc_min[h0][h1].min(acc[h0][h1]);
                }
            }
        }
    }
    writeln!(output, "{}", ans).unwrap();
}
