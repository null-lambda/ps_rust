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
            .map(|&c| match c {
                b'\n' | b'\r' | 0 => true,
                _ => false,
            })
            .unwrap_or_else(|| false)
        {
            s = &s[..s.len() - 1];
        }
        s
    }

    impl InputStream for &[u8] {
        fn token(&mut self) -> &[u8] {
            let idx = self
                .iter()
                .position(|&c| !is_whitespace(c))
                .expect("no available tokens left");
            *self = &self[idx..];
            let idx = self
                .iter()
                .position(|&c| is_whitespace(c))
                .unwrap_or_else(|| self.len());
            let (token, buf_new) = self.split_at(idx);
            *self = buf_new;
            token
        }

        fn line(&mut self) -> &[u8] {
            let idx = self
                .iter()
                .position(|&c| c == b'\n')
                .map(|idx| idx + 1)
                .unwrap_or_else(|| self.len());
            let (line, buf_new) = self.split_at(idx);
            *self = buf_new;
            trim_newline(line)
        }
    }
}

use std::io::{BufReader, Read, Write};
use std::iter::successors;

fn stdin() -> Vec<u8> {
    let stdin = std::io::stdin();
    let mut reader = BufReader::new(stdin.lock());

    let mut input_buf: Vec<u8> = vec![];
    reader.read_to_end(&mut input_buf).unwrap();
    input_buf
}

fn main() {
    use io::*;

    let input_buf = stdin();
    let mut input: &[u8] = &input_buf;

    let mut output_buf = Vec::<u8>::new();

    let m: usize = input.value();
    let mut next: Vec<u32> = Vec::with_capacity(m + 1);
    next.push(0);
    next.extend((1..=m).map(|_| input.value::<u32>()));

    let n_max = 200_000;
    let log2n_bound = (n_max as f64).log2().ceil() as usize;

    // sparce table
    // stores next 2^j-th node
    let next_table: Vec<Vec<u32>> = successors(Some(next), |prev_row| {
        prev_row
            .iter()
            .map(|&u| Some(prev_row[u as usize]))
            .collect()
    })
    .take(log2n_bound + 1)
    .collect();

    let q = input.value();
    for _ in 0..q {
        let (n, x): (_, u32) = (input.value(), input.value());
        let x = (0..)
            .take_while(|j| (1 << j) <= n)
            .filter(|j| (1 << j) & n != 0)
            .fold(x, |x, j| next_table[j][x as usize]);
        writeln!(output_buf, "{}", x).unwrap();
    }

    std::io::stdout().write(&output_buf[..]).unwrap();
}
