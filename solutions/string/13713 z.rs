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

// z[i]: length of longest common prefix between text[0..] and text[i..]
fn z_array<T: Eq>(text: Vec<T>) -> Vec<usize> {
    let n = text.len();
    assert!(n >= 1);
    let (mut left, mut right) = (0, 0);
    let mut z = vec![n];
    for i in 1..n {    
        z.push(if i <= right && z[i - left] < right + 1 - i {
            z[i - left]
        } else {
            left = i;
            right = right.max(i);
            while right < n && text[right - left] == text[right] {
                right += 1;
            }
            right -= 1;
            right + 1 - left
        });
    }
    z
}

fn main() {
    use io::InputStream;
    let input_buf = stdin();
    let mut input: &[u8] = &input_buf[..];

    let mut output_buf = Vec::<u8>::new();

    let xs: Vec<u8> = input.token().iter().copied().rev().collect();
    let n = xs.len();

    let m = input.value();
    let mut z = z_array(xs);
    for _ in 0..m {
        let i: usize = input.value();
        writeln!(output_buf, "{}", z[n - i]).unwrap();
    }

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
