#[allow(dead_code)]
mod fast_io {
    use std::fmt::Debug;
    use std::str::*;

    pub trait InputStream {
        fn token(&mut self) -> &[u8];
        fn line(&mut self) -> &[u8];

        fn skip_line(&mut self) {
            self.line();
        }

        fn value<T: FromStr>(&mut self) -> T 
        where <T as FromStr>::Err: Debug {
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

    pub struct InputAtOnce {
        buf: Box<[u8]>,
        cursor: usize,
    }

    impl<'a> InputAtOnce {
        pub fn new(buf: Box<[u8]>) -> Self {
            Self { buf, cursor: 0 }
        }

        fn take(&mut self, n: usize) -> &[u8] {
            let n = n.min(self.buf.len() - self.cursor);
            let slice = &self.buf[self.cursor..self.cursor + n];
            self.cursor += n;
            slice
        }
    }

    impl<'a> InputStream for InputAtOnce {
        fn token(&mut self) -> &[u8] {
            self.take(self.buf[self.cursor..]
                .iter()
                .position(|&c| !is_whitespace(c))
                .expect("no available tokens left")
            );
            self.take(self.buf[self.cursor..]
                .iter()
                .position(|&c| is_whitespace(c))
                .unwrap_or_else(|| self.buf.len() - self.cursor)
            )
        }

        fn line(&mut self) -> &[u8] {
            let line = self.take(self.buf[self.cursor..]
                .iter()
                .position(|&c| c == b'\n')
                .map(|idx| idx + 1)
                .unwrap_or_else(|| self.buf.len() - self.cursor)
            );
            trim_newline(line)
        }
    }

    pub struct LineSyncedInput<R: BufRead> {
        line_buf: Vec<u8>, 
        line_cursor: usize,
        inner: R
    }
    
    impl<R: BufRead> LineSyncedInput<R> {
        pub fn new(r: R) -> Self {
            Self {
                line_buf: Vec::new(),
                line_cursor: 0,
                inner: r 
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
            println!("refill: {}", String::from_utf8(self.line_buf.clone()).unwrap());
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
                self.take(self.line_buf[self.line_cursor..]
                    .iter()
                    .position(|&c| !is_whitespace(c))
                    .unwrap_or_else(|| self.line_buf.len() - self.line_cursor)
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

    pub fn stdin_at_once() -> InputAtOnce {
        let mut reader = BufReader::new(std::io::stdin().lock());
        let mut buf: Vec<u8> = vec![];
        reader.read_to_end(&mut buf).unwrap();
        let buf = buf.into_boxed_slice();
        InputAtOnce::new(buf)
    }

    // pub fn stdin_buf() -> LineSyncedInput<BufReader<StdinLock<'static>>> {
    //     LineSyncedInput::new(BufReader::new(std::io::stdin().lock()))
    // }

    // no lock
    pub fn stdin_buf() -> LineSyncedInput<BufReader<Stdin>> {
        LineSyncedInput::new(BufReader::new(std::io::stdin()))
    }

    pub fn stdout_buf() -> BufWriter<Stdout> {
        BufWriter::new(std::io::stdout())
    }
}