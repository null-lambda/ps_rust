use std::{collections::BinaryHeap, io::Write};

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

fn partition_point<P>(mut left: u32, mut right: u32, mut pred: P) -> u32
where
    P: FnMut(u32) -> bool,
{
    while left < right {
        let mid = left + (right - left) / 2;
        if pred(mid) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    left
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let t: usize = input.value();
    let mut xs: Vec<u32> = (0..n).map(|_| input.value()).collect();
    let mut ys: Vec<u32> = (0..m).map(|_| input.value()).collect();
    xs.sort_unstable();
    ys.sort_unstable();

    let mut ps: Vec<(u32, u32)> = (0..t).map(|_| (input.value(), input.value())).collect();
    ps.sort_unstable();

    let is_feasible = |t_bound: usize| {
        let mut ps = ps.iter().copied().peekable();
        let mut sizes = BinaryHeap::new();
        for &weight_bound in &xs {
            while let Some(&(weight, size)) = ps.peek() {
                if weight < weight_bound {
                    sizes.push(size);
                    ps.next();
                } else {
                    break;
                }
            }

            if sizes.len() <= t_bound {
                sizes.clear();
            } else {
                for _ in 0..t_bound {
                    unsafe { sizes.pop().unwrap_unchecked() };
                }
            }
        }
        sizes.extend(ps.map(|(_, size)| size));

        for &size_bound in ys.iter().rev() {
            for _ in 0..t_bound {
                let Some(size) = sizes.pop() else {
                    return true;
                };
                if size >= size_bound {
                    return false;
                }
            }
        }
        sizes.is_empty()
    };

    let mut ans = partition_point(1, t as u32 + 1, |t_bound| !is_feasible(t_bound as usize)) as i32;
    if ans == t as i32 + 1 {
        ans = -1;
    }
    writeln!(output, "{}", ans).unwrap();
}
