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
            .map(|&c| {
                matches! {c, b'\n' | b'\r' | 0}
            })
            .unwrap_or_else(|| false)
        {
            s = &s[..s.len() - 1];
        }
        s
    }

    impl InputStream for &[u8] {
        fn token(&mut self) -> &[u8] {
            let idx = self.iter().position(|&c| !is_whitespace(c)).unwrap();
            //.expect("no available tokens left");
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

use std::io::Write;
use std::io::{BufReader, Read};

fn stdin() -> Vec<u8> {
    let stdin = std::io::stdin();
    let mut reader = BufReader::new(stdin.lock());

    let mut input_buf: Vec<u8> = vec![];
    reader.read_to_end(&mut input_buf).unwrap();
    input_buf
}

fn main() {
    use io::InputStream;
    let input_buf = stdin();
    let mut input: &[u8] = &input_buf[..];

    let mut output_buf = Vec::<u8>::new();

    let s = input.line().to_vec();
    let pattern = input.line().to_vec();
    assert!(!s.is_empty());
    assert!(!pattern.is_empty());

    // build jump function
    let mut jump_table = vec![0];
    let mut i_prev = 0;
    for i in 1..pattern.len() {
        while i_prev > 0 && pattern[i] != pattern[i_prev]  {
            i_prev = jump_table[i_prev - 1];
        }
        if pattern[i] == pattern[i_prev] {
            i_prev += 1;
        }
        jump_table.push(i_prev);
    }

    // search patterns 
    let mut result = Vec::new();
    let mut j = 0;
    for (i, c) in s.into_iter().enumerate() {
        while j == pattern.len() || j > 0 && pattern[j] != c {
            j = jump_table[j - 1];
        }
        if pattern[j] == c {
            j += 1;
        }
        if j == pattern.len() {
            result.push(1 + i - pattern.len());
        }
    }

    writeln!(output_buf, "{}", result.len()).unwrap();
    for i in result {
        write!(output_buf, "{} ", i + 1).unwrap();
    }

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
