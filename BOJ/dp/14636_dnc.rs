use std::{io::Write, ops::Range};

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

fn upper_decreasing_hull(mut ps: Vec<(i32, i32)>) -> Vec<(i32, i32)> {
    ps.sort_unstable();
    let mut res: Vec<(i32, i32)> = vec![];
    for (x, y) in ps {
        while res.last().filter(|&&(_, y_prev)| y_prev < y).is_some() {
            res.pop();
        }
        res.push((x, y));
    }
    res
}

// Compute row minimum C(i) = min_j A(i, j)
// where opt(i) = argmin_j A(i, j) is monotonically increasing.
//
// Arguments:
// - naive: Compute opt(i) for a given range of j.
fn dnc_row_min(
    naive: &mut impl FnMut(usize, Range<usize>) -> usize,
    i: Range<usize>,
    j: Range<usize>,
) {
    if i.start >= i.end {
        return;
    }
    let i_mid = i.start + i.end >> 1;
    let j_opt = naive(i_mid, j.clone());
    dnc_row_min(naive, i.start..i_mid, j.start..j_opt + 1);
    dnc_row_min(naive, i_mid + 1..i.end, j_opt..j.end);
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let mut ps: Vec<(i32, i32)> = (0..n).map(|_| (input.value(), input.value())).collect();
    let qs: Vec<(i32, i32)> = (0..m).map(|_| (input.value(), input.value())).collect();

    for (x, y) in &mut ps {
        *x = -*x;
        *y = -*y;
    }

    let mut lower = upper_decreasing_hull(ps);
    let upper = upper_decreasing_hull(qs);
    for (x, y) in &mut lower {
        *x = -*x;
        *y = -*y;
    }
    lower.reverse();

    let mut ans = 0;
    let area =
        |p: (i32, i32), q: (i32, i32)| (q.0 as i64 - p.0 as i64).max(0) * (q.1 as i64 - p.1 as i64);

    let mut naive = |i: usize, j: Range<usize>| {
        let (max, opt_j) = (j.start..j.end)
            .map(|j| (area(lower[i], upper[j]), j))
            .max()
            .unwrap();
        ans = ans.max(max);
        opt_j
    };
    dnc_row_min(&mut naive, 0..lower.len(), 0..upper.len());

    writeln!(output, "{}", ans).unwrap();
}
