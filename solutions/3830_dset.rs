mod io {
    use std::fmt::Debug;
    use std::str::*;

    pub trait InputStream {
        fn token(&mut self) -> &[u8];
        fn line(&mut self) -> &[u8];

        fn skip_line(&mut self) {
            self.line();
        }

        #[inline]
        fn value<T>(&mut self) -> T
        where
            T: FromStr,
            T::Err: Debug,
        {
            let token = self.token();
            let token = unsafe { from_utf8_unchecked(token) };
            token.parse::<T>().unwrap()
        }
    }

    #[inline]
    fn is_whitespace(c: u8) -> bool {
        c <= b' '
    }

    fn trim_newline(s: &[u8]) -> &[u8] {
        let mut s = s;
        while s
            .last()
            .map(|&c| matches!(c, b'\n' | b'\r' | 0))
            .unwrap_or_else(|| false)
        {
            s = &s[..s.len() - 1];
        }
        s
    }

    impl InputStream for &[u8] {
        fn token(&mut self) -> &[u8] {
            let i = self.iter().position(|&c| !is_whitespace(c)).unwrap();
            //.expect("no available tokens left");
            *self = &self[i..];
            let i = self
                .iter()
                .position(|&c| is_whitespace(c))
                .unwrap_or_else(|| self.len());
            let (token, buf_new) = self.split_at(i);
            *self = buf_new;
            token
        }

        fn line(&mut self) -> &[u8] {
            let i = self
                .iter()
                .position(|&c| c == b'\n')
                .map(|i| i + 1)
                .unwrap_or_else(|| self.len());
            let (line, buf_new) = self.split_at(i);
            *self = buf_new;
            trim_newline(line)
        }
    }
}

use std::io::{BufReader, Read, Write};

fn stdin() -> Vec<u8> {
    let stdin = std::io::stdin();
    let mut reader = BufReader::new(stdin.lock());

    let mut input_buf: Vec<u8> = vec![];
    reader.read_to_end(&mut input_buf).unwrap();
    input_buf
}

use std::cell::Cell;

#[derive(Debug)]
struct DisjointSet {
    parent: Vec<Cell<usize>>,
    size: Vec<u32>,
    weight: Vec<Cell<i32>>,
}

impl DisjointSet {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).map(|i| Cell::new(i)).collect(),
            size: vec![1; n],
            weight: vec![Cell::new(0); n],
        }
    }

    fn root(&self, u: usize) -> usize {
        if u == self.parent[u].get() {
            u
        } else {
            let ru = self.root(self.parent[u].get());
            self.weight[u].set(self.weight[u].get() + self.weight[self.parent[u].get()].get());
            self.parent[u].set(ru);
            ru
        }
    }

    fn weight_to_root(&self, u: usize) -> i32 {
        if u == self.parent[u].get() {
            0
        } else {
            self.weight[u].get() + self.weight_to_root(self.parent[u].get())
        }
    }

    // returns whether two set were different
    fn merge(&mut self, u: usize, v: usize, w: i32) -> bool {
        let ru = self.root(u);
        let rv = self.root(v);
        if ru == rv {
            return false;
        }
        let wu = self.weight_to_root(u);
        let wv = self.weight_to_root(v);
        if self.size[ru] <= self.size[rv] {
            self.parent[rv].set(ru);
            self.weight[rv].set(wu - wv + w);
            self.size[ru] += self.size[rv];
        } else {
            self.parent[ru].set(rv);
            self.weight[ru].set(wv - wu - w);
            self.size[rv] += self.size[ru];
        }
        true
    }
}

fn main() {
    use io::*;

    let input_buf = stdin();
    let mut input: &[u8] = &input_buf;

    let mut output_buf = Vec::<u8>::new();

    loop {
        let n: usize = input.value();
        let m: usize = input.value();
        if (n, m) == (0, 0) {
            break;
        }

        let mut dset = DisjointSet::new(n);
        for _ in 0..m {
            let q: u8 = input.token()[0];
            match q {
                b'!' => {
                    let a = input.value::<usize>() - 1;
                    let b = input.value::<usize>() - 1;
                    let w: i32 = input.value();
                    dset.merge(a, b, w);
                }
                b'?' => {
                    let a = input.value::<usize>() - 1;
                    let b = input.value::<usize>() - 1;
                    if dset.root(a) == dset.root(b) {
                        let result = dset.weight_to_root(b) - dset.weight_to_root(a);
                        writeln!(output_buf, "{}", result).unwrap();
                    } else {
                        writeln!(output_buf, "UNKNOWN").unwrap();
                    }
                }
                _ => panic!(),
            }

            /*
            for i in 0..n {
                dset.root(i);
                print!("{}: {:?}, ", i, (dset.parent[i].get(), dset.weight[i], dset.weight_to_root(i)));
            }
            println!();
            */
        }
    }

    std::io::stdout().write(&output_buf[..]).unwrap();
}
