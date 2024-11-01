use std::{
    collections::{BTreeSet, BinaryHeap},
    io::Write,
    iter,
};

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
    let a_shifted = (0..n as i64).map(|i| input.value::<i64>() - i);

    // Minimize sum |ai - bi| where b[0] <= ... <= b[n-1]
    // => slope trick
    assert!(n >= 1);
    let mut breakpoints: BinaryHeap<i64> = Default::default();

    // let mut y_min = 0i64;
    let mut b = vec![];
    for (i, a) in a_shifted.enumerate() {
        breakpoints.push(a);
        let last = *breakpoints.peek().unwrap();
        b.push(last + i as i64);
        if last > a {
            // y_min += last - a;
            breakpoints.pop();
            breakpoints.push(a);
        }
    }
    // writeln!(output, "{}", y_min).unwrap();

    for i in (0..n - 1).rev() {
        b[i] = b[i].min(b[i + 1] - 1);
    }

    for b_i in b {
        writeln!(output, "{}", b_i).unwrap();
    }
}
