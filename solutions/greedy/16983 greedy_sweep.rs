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

fn main() {
    use io::*;

    let input_buf = stdin();
    let mut input: &[u8] = &input_buf;

    let mut output_buf = Vec::<u8>::new();

    let n: usize = input.value();

    let mut result: u64 = 0;
    // gather coins into the rectengular area, allowing to be overlapped.
    let mut count: Vec<[i32; 2]> = vec![[-1; 2]; n];
    for _ in 0..2 * n {
        let x: i32 = input.value::<i32>() - 1;
        let y: i32 = input.value::<i32>() - 1;
        let x2 = x.min(n as i32 - 1).max(0);
        let y2 = y.min(1).max(0);
        result += ((x2 - x).abs() + (y2 - y).abs()) as u64;
        count[x2 as usize][y2 as usize] += 1;
    }

    // treat -1 as negative coin.
    // sweep all coins from left to right, to eliminate all blank or exceeding cells.
    result += count
        .into_iter()
        .scan([0i32; 2], |state, count| {
            let mut moves = state[0].abs() + state[1].abs();
            state[0] += count[0];
            state[1] += count[1];
            if state[0].signum() * state[1].signum() == -1 {
                if state[0].abs() <= state[1].abs() {
                    moves += state[0].abs();
                    *state = [0, state[1] + state[0]];
                } else {
                    moves += state[1].abs();
                    *state = [state[1] + state[0], 0];
                }
            }
            Some(moves as u64)
        })
        .sum::<u64>();

    println!("{}", result);

    std::io::stdout().write(&output_buf[..]).unwrap();
}
