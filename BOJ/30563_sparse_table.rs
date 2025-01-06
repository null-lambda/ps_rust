use std::io::Write;

mod simple_io {
    use std::string::*;

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

pub fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let c: i64 = input.value();
    let xs: Vec<i64> = (0..n).map(|_| input.value()).collect();

    if xs.iter().sum::<i64>() <= c {
        for _ in 0..n {
            write!(output, "{} ", 0).unwrap();
        }
        writeln!(output).unwrap();
        return;
    }

    let mut j = 0;
    let mut acc = xs[0];
    let mut jump = vec![0; 2 * n + 1];
    for i in 0..n {
        while acc < c {
            j += 1;
            acc += xs[j % n];
        }
        jump[i] = j + 1;

        acc -= xs[i];
    }

    for i in n..2 * n + 1 {
        jump[i] = jump[i - n] + n;
    }

    let n_log2 = (usize::BITS - jump.len().leading_zeros()) as usize;
    let mut jump_sparse = vec![jump.clone()];
    for i in 1..=n_log2 {
        let prev = &jump_sparse[i - 1];
        let mut next = vec![0; 2 * n + 1];
        for j in 0..2 * n + 1 {
            next[j] = prev.get(prev[j]).copied().unwrap_or(2 * n);
        }
        jump_sparse.push(next);
    }

    // min jumps to reach >= i+n from i
    for mut i in 0..n {
        let target = i + n;
        let mut k = 0;
        for j in (0..=n_log2).rev() {
            if jump_sparse[j][i] <= target {
                i = jump_sparse[j][i];
                k += 1 << j;
            }
        }
        while i < target {
            i = jump_sparse[0][i];
            k += 1;
        }
        write!(output, "{} ", k - 1).unwrap();
    }
    writeln!(output).unwrap();
}
