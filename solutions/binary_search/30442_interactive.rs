use std::{collections::HashMap, io::Write, iter};

use buffered_io::InputStream;

mod buffered_io {
    use std::fmt::Debug;
    use std::str::*;

    pub trait InputStream {
        fn token(&mut self) -> &[u8];
        fn line(&mut self) -> &[u8];

        fn skip_line(&mut self) {
            self.line();
        }

        fn value<T: FromStr>(&mut self) -> T
        where
            <T as FromStr>::Err: Debug,
        {
            let token = self.token();
            let token = unsafe { from_utf8_unchecked(token) };
            token.parse::<T>().unwrap()
        }
    }

    // cheap and unsafe whitespace check
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

    use std::io::{BufRead, BufReader, BufWriter, Read, Stdin, StdinLock, Stdout};

    pub struct LineSyncedInput<R: BufRead> {
        line_buf: Vec<u8>,
        line_cursor: usize,
        inner: R,
    }

    impl<R: BufRead> LineSyncedInput<R> {
        pub fn new(r: R) -> Self {
            Self {
                line_buf: Vec::new(),
                line_cursor: 0,
                inner: r,
            }
        }

        fn take(&mut self, n: usize) -> &[u8] {
            let n = n.min(self.line_buf.len() - self.line_cursor);
            let slice = &self.line_buf[self.line_cursor..self.line_cursor + n];
            self.line_cursor += n;
            slice
        }

        fn eol(&self) -> bool {
            self.line_cursor == self.line_buf.len()
        }

        fn refill_line_buf(&mut self) -> bool {
            self.line_buf.clear();
            self.line_cursor = 0;
            let result = self.inner.read_until(b'\n', &mut self.line_buf).is_ok();
            result
        }
    }

    impl<R: BufRead> InputStream for LineSyncedInput<R> {
        fn token(&mut self) -> &[u8] {
            loop {
                if self.eol() {
                    let b = self.refill_line_buf();
                    if !b {
                        panic!(); // EOF
                    }
                }
                self.take(
                    self.line_buf[self.line_cursor..]
                        .iter()
                        .position(|&c| !is_whitespace(c))
                        .unwrap_or_else(|| self.line_buf.len() - self.line_cursor),
                );

                let idx = self.line_buf[self.line_cursor..]
                    .iter()
                    .position(|&c| is_whitespace(c))
                    .unwrap_or_else(|| self.line_buf.len() - self.line_cursor);
                if idx > 0 {
                    return self.take(idx);
                }
            }
        }

        fn line(&mut self) -> &[u8] {
            if self.eol() {
                self.refill_line_buf();
            }

            self.line_cursor = self.line_buf.len();
            trim_newline(self.line_buf.as_slice())
        }
    }

    pub fn stdin() -> LineSyncedInput<BufReader<Stdin>> {
        LineSyncedInput::new(BufReader::new(std::io::stdin()))
    }

    pub fn stdout() -> BufWriter<Stdout> {
        BufWriter::new(std::io::stdout())
    }
}

fn partition_point<P>(mut left: u64, mut right: u64, mut pred: P) -> u64
where
    P: FnMut(u64) -> bool,
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
    let mut input = buffered_io::stdin();
    let mut output = buffered_io::stdout();

    let mut memo = HashMap::new();
    let mut query = |i: u64| {
        if let Some(&res) = memo.get(&i) {
            return res;
        }
        writeln!(output, "buf[{}]", i).unwrap();
        output.flush().unwrap();

        let res = iter::repeat_with(|| {
            let line = input.line();
            let line = unsafe { std::str::from_utf8_unchecked(line) };
            line.trim().parse::<u8>()
        })
        .flatten()
        .next()
        .unwrap()
            > 0;
        memo.insert(i, res);
        res
    };

    let mut i = 4;
    loop {
        if !query(i - 1) {
            break;
        }
        i *= 2;
    }

    let ans = partition_point((i / 2).max(2), i - 1, |pos| query(pos));
    writeln!(output, "strlen(buf) = {}", ans).unwrap();
}
