use std::{cmp::Reverse, collections::BinaryHeap, io::Write};

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

pub mod debug {
    pub fn with(f: impl FnOnce()) {
        #[cfg(debug_assertions)]
        f()
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let k: usize = input.value();

    let mut pairs = vec![[0i64; 2]; n];
    for g in 0..2 {
        for i in 0..n {
            pairs[i][g] = input.value();
        }
    }
    pairs.sort_unstable_by_key(|&[x, y]| Reverse([y, x]));

    let (cs, ds) = pairs.split_at(k);

    let mut ds_desc_by_x: Vec<_> = (0..ds.len()).map(|i| (ds[i], i as u32)).collect();
    ds_desc_by_x.sort_unstable_by_key(|&([x, _], _)| Reverse(x));

    let mut ds_selected = vec![false; ds.len()];
    let mut selected = BinaryHeap::new();
    for i in 0..k {
        let ([x, _], j) = ds_desc_by_x[i];
        ds_selected[j as usize] = true;
        selected.push((Reverse(x), j));
    }

    let mut acc = -cs.iter().map(|&[_, y]| y as i64).sum::<i64>()
        + ds_desc_by_x[..k]
            .iter()
            .map(|&([x, _], _)| x as i64)
            .sum::<i64>();
    let mut ans = acc;

    let mut p01 = BinaryHeap::new();
    for i in 0..k {
        p01.push(cs[i][0] + cs[i][1]);
    }

    for i in 0..k {
        acc -= ds[i][1] as i64;
        p01.push(ds[i][0] + ds[i][1]);
        acc += p01.pop().unwrap() as i64;

        if ds_selected[i] {
            ds_selected[i] = false;
            acc -= ds[i][0] as i64;
        } else {
            while let Some((Reverse(x), j)) = selected.pop() {
                if ds_selected[j as usize] {
                    ds_selected[j as usize] = false;
                    acc -= x as i64;
                    break;
                }
            }
        }

        ans = ans.max(acc);
    }

    assert!(ds_selected.iter().all(|b| !b));

    writeln!(output, "{}", ans).unwrap();
}
