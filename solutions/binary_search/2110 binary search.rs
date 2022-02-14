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

fn stdin() -> Vec<u8> {
    let stdin = std::io::stdin();
    let mut reader = BufReader::new(stdin.lock());

    let mut input_buf: Vec<u8> = vec![];
    reader.read_to_end(&mut input_buf).unwrap();
    input_buf
}

use std::cmp::Ordering;

// adoption from std crate
fn binary_search_by<F>(mut left: u32, mut right: u32, mut f: F) -> Result<u32, u32>
where
    F: FnMut(u32) -> Ordering,
{
    let mut size;
    while left < right {
        size = right - left;
        let mid = left + size / 2;

        let cmp = f(mid);
        match cmp {
            Ordering::Less => {
                left = mid + 1;
            }
            Ordering::Greater => {
                right = mid;
            }
            Ordering::Equal => {
                return Ok(mid);
            }
        }
    }
    Err(left)
}

// adoption from std crate
fn partition_point<P>(left: u32, right: u32, mut pred: P) -> u32
where
    P: FnMut(u32) -> bool,
{
    binary_search_by(left, right, |x| {
        if pred(x) {
            Ordering::Less
        } else {
            Ordering::Greater
        }
    })
    .unwrap_or_else(|i| i)
}

fn main() {
    use io::*;

    let input_buf = stdin();
    let mut input: &[u8] = &input_buf;

    let mut output_buf = Vec::<u8>::new();

    let (n, c): (_, u32) = (input.value(), input.value());
    let mut xs: Vec<u32> = (0..n).map(|_| input.value()).collect();
    xs.sort();

    let result = partition_point(1, (xs[n - 1] - xs[0]) / (c - 1) + 1, |d| {
        let (mut prev, mut current) = (0, 1);
        for _ in 0..(c - 1) {
            while xs[current] - xs[prev] < d {
                current += 1;
                if current == n {
                    return false;
                }
            }
            prev = current;
        }
        true
    }) - 1;
    writeln!(output_buf, "{}", result).unwrap();

    std::io::stdout().write(&output_buf[..]).unwrap();
}
