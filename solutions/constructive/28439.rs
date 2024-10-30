use std::{collections::HashMap, io::Write, iter};

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
    let m: usize = input.value();

    let mut mat: Vec<i32> = (0..n * m).map(|_| input.value()).collect();

    let mut col_inst = mat[0..m].to_vec();
    for (j, x) in col_inst.iter().enumerate() {
        for i in 0..n {
            mat[i * m + j] -= x;
        }
    }

    let mut row_inst: Vec<_> = (0..n).map(|i| mat[i * m]).collect();
    if !(0..n).all(|i| mat[i * m..(i + 1) * m].iter().all(|&x| x == row_inst[i])) {
        writeln!(output, "-1").unwrap();
        return;
    }

    let mut inst_freq: HashMap<_, u32> = Default::default();
    for &x in &col_inst {
        *inst_freq.entry(x).or_default() += 1;
    }
    for &x in &row_inst {
        *inst_freq.entry(-x).or_default() += 1;
    }
    let shift = inst_freq.iter().max_by_key(|&(_, &v)| v).unwrap().0;

    let mut cnt = 0;
    for x in &mut col_inst {
        *x -= shift;
        if *x != 0 {
            cnt += 1;
        }
    }

    for x in &mut row_inst {
        *x += shift;
        if *x != 0 {
            cnt += 1;
        }
    }

    writeln!(output, "{}", cnt).unwrap();
    for (j, x) in col_inst.iter().enumerate() {
        if *x != 0 {
            writeln!(output, "2 {} {}", j + 1, x).unwrap();
        }
    }
    for (i, x) in row_inst.iter().enumerate() {
        if *x != 0 {
            writeln!(output, "1 {} {}", i + 1, x).unwrap();
        }
    }
}
