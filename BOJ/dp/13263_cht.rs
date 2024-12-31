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

fn main() {
    use io::InputStream;
    let input_buf = stdin();
    let mut input: &[u8] = &input_buf[..];

    let mut output_buf = Vec::<u8>::new();

    let n = input.value();
    let a_list: Vec<u32> = (0..n).map(|_| input.value()).collect();
    let b_list: Vec<u32> = (0..n).map(|_| input.value()).collect();
    let mut cost = vec![0];

    /*
    // O( n^2 )
    {
        let mut cost = vec![0];
        for i in 1..n {
            let x = (0..i)
                .map(|j| cost[j] + b_list[j]* a_list[i] )
                .min()
                .unwrap();
            cost.push(x);
        }
        // println!("{:?}", cost);
    }
    */

    let mut cvhull: Vec<(f32, (u64, u32))> = vec![];
    for i in 1..n {
        let mut x = 0.0;
        let line = (cost[i - 1], b_list[i - 1]);
        while let Some(&(x_last, last)) = cvhull.last() {
            x = (line.0 - last.0) as f32 / (last.1 - line.1) as f32;
            if x_last < x {
                break;
            }
            cvhull.pop();
        }
        cvhull.push((x, line));

        let (_, line) = cvhull[cvhull.partition_point(|&(x, ..)| x <= a_list[i] as f32) - 1];
        let y = line.0 + a_list[i] as u64 * line.1 as u64;
        cost.push(y);
    }

    println!("{:?}", cost[n - 1]);

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
