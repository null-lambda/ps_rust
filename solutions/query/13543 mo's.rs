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

fn main() {
    use io::*;

    let input_buf = stdin();
    let mut input: &[u8] = &input_buf;

    let mut output_buf = Vec::<u8>::new();

    let n = input.value();
    let xs: Vec<u32> = (0..n)
        .map(|_| input.value())
        .chain(std::iter::once(0))
        .collect();
    let bucket_size = (n as f32).sqrt().round() as usize;

    let m = input.value();
    type Query = (u32, u32, u32);
    let mut queries: Vec<Query> = (0..m as u32)
        .map(|i| {
            let start: u32 = input.value();
            let end: u32 = input.value();
            (start - 1, end - 1, i)
        })
        .collect();
    queries.sort_unstable_by_key(|&(start, end, _)| (end / bucket_size as u32, start));

    const X_MAX: usize = 1_000_000;
    let mut count: Vec<u32> = vec![0; X_MAX + 1];
    let mut count2: Vec<u32> = vec![0; n + 1];
    count2[0] = n as u32;
    let mut count_max: u32 = 0;
    let mut query_result: Vec<u32> = vec![0; m];

    let (mut start_current, mut end_current) = (1, 0);
    for (start, end, i) in queries {
        let (start, end, i) = (start as usize, end as usize, i as usize);
        let mut add_value = |i: usize| {
            count2[count[xs[i] as usize] as usize] -= 1;
            count[xs[i] as usize] += 1;
            count2[count[xs[i] as usize] as usize] += 1;
            count_max = count_max.max(count[xs[i] as usize]);
        };
        while start < start_current {
            start_current -= 1;
            add_value(start_current);
        }
        while end > end_current {
            end_current += 1;
            add_value(end_current);
        }

        let mut sub_value = |i: usize| {
            if count_max == count[xs[i] as usize] && count2[count[xs[i] as usize] as usize] == 1 {
                count_max -= 1;
            }
            count2[count[xs[i] as usize] as usize] -= 1;
            count[xs[i] as usize] -= 1;
            count2[count[xs[i] as usize] as usize] += 1;
        };
        while start > start_current {
            sub_value(start_current);
            start_current += 1;
        }
        while end < end_current {
            sub_value(end_current);
            end_current -= 1;
        }
        query_result[i] = count_max;

        /*
        println!("{:?}", &count[..10]);
        println!("{:?}", count2);
        println!("{:?}", count_max);
        */
    }

    for r in query_result {
        writeln!(output_buf, "{}", r).unwrap();
    }

    std::io::stdout().write(&output_buf[..]).unwrap();
}
