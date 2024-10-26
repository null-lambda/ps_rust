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

// manachers algorithm
fn max_palindrome_radius(text: Vec<u8>) -> usize {
    let n = text.len();
    let mut i = 0;
    let mut radius = 0;
    let mut radii = vec![];
    while i < n {
        while i >= (radius + 1)
            && i + (radius + 1) < n
            && text[i - (radius + 1)] == text[i + (radius + 1)]
        {
            radius += 1;
        }
        radii.push(radius);

        let mut mirrored_center = i;
        let mut max_mirrored_radius = radius;
        i += 1;
        radius = 0;
        while max_mirrored_radius > 0 {
            mirrored_center -= 1;
            max_mirrored_radius -= 1;
            if radii[mirrored_center] == max_mirrored_radius {
                radius = max_mirrored_radius;
                break;
            }
            radii.push(radii[mirrored_center].min(max_mirrored_radius));
            i += 1;
        }
    }
    radii.into_iter().max().unwrap()
}

fn main() {
    use io::InputStream;
    let input_buf = stdin();
    let mut input: &[u8] = &input_buf[..];

    let mut output_buf = Vec::<u8>::new();

    let mut text = Vec::new();
    for &c in input.line() {
        text.push(0);
        text.push(c);
    }
    text.push(0);
    let result = max_palindrome_radius(text);

    println!("{:?}", result);

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
