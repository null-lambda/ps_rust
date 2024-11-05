use std::{collections::HashMap, io::Write, iter::repeat};

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

type BitSet = u16;
fn gen_masks(n: usize) -> Vec<BitSet> {
    assert!(n <= 12);

    let init: Vec<BitSet> = (0..n).map(|i| 1 << i).collect();

    fn fold_paper(xs: Vec<BitSet>, visitor: &mut impl FnMut(&BitSet)) {
        for &x in &*xs {
            visitor(&x);
        }

        for split_pos in 1..xs.len() {
            let (left, right) = xs.split_at(split_pos);

            let len = left.len().max(right.len());
            let join = (left.into_iter().copied().rev().chain(repeat(0)))
                .zip(right.into_iter().copied().chain(repeat(0)))
                .take(len)
                .map(|(a, b)| a | b)
                .collect();
            fold_paper(join, visitor);
        }
    }

    let mut masks = vec![];
    fold_paper(init, &mut |&mask| masks.push(mask));
    masks.sort_unstable();
    masks.dedup();
    masks
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let grid: Vec<Vec<i32>> = (0..n)
        .map(|_| (0..m).map(|_| input.value()).collect())
        .collect();
    let rmasks = gen_masks(n);
    let cmasks = gen_masks(m);

    let mut col_sum = vec![HashMap::new(); n];
    for i in 0..n {
        for &cmask in &cmasks {
            let mut sum = 0;
            for j in 0..m {
                if (cmask >> j) & 1 == 1 {
                    sum += grid[i][j];
                }
            }
            col_sum[i].insert(cmask, sum);
        }
    }

    let mut ans = i32::MIN;
    for &rmask in &rmasks {
        for &cmask in &cmasks {
            let mut sum = 0;
            for i in 0..n {
                if (rmask >> i) & 1 == 1 {
                    sum += col_sum[i].get(&cmask).unwrap();
                }
            }
            ans = ans.max(sum);
        }
    }

    writeln!(output, "{}", ans).unwrap();
}
