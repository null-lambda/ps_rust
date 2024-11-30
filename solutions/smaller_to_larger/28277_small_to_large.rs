use std::{cell::UnsafeCell, collections::HashSet, io::Write, mem};

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

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let q: usize = input.value();

    let mut dset: Vec<UnsafeCell<HashSet<u32>>> = (0..n)
        .map(|_| {
            let m = input.value();
            UnsafeCell::new((0..m).map(|_| input.value()).collect())
        })
        .collect();
    for _ in 0..q {
        match input.token() {
            "1" => {
                let a = input.value::<usize>() - 1;
                let b = input.value::<usize>() - 1;
                assert!(a != b);

                let dset_a = unsafe { &mut *dset[a].get() };
                let dset_b = unsafe { &mut *dset[b].get() };
                if dset_a.len() < dset_b.len() {
                    dset.swap(a, b);
                }
                dset_a.extend(mem::take(dset_b));
            }
            "2" => {
                let a = input.value::<usize>() - 1;
                let dset_a = dset[a].get_mut();
                writeln!(output, "{}", dset_a.len()).unwrap();
            }
            _ => panic!(),
        }
    }
}
