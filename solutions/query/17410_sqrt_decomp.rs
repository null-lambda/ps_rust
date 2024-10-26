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

    use std::iter::*;

    let n = input.value();
    let mut bucket_size = (n as f64).sqrt().round() as usize;
    bucket_size = (bucket_size + 16 - 1) / 16 * 16;

    let mut xs: Vec<u16> = (0..n)
        .map(|_| input.value())
        .chain(repeat(u16::MAX))
        .take(((n + bucket_size - 1) / bucket_size) * bucket_size)
        .collect();
    let n = xs.len();

    let mut xs_buckets: Vec<Box<[u16]>> = xs
        .chunks_exact(bucket_size)
        .map(|chunk| {
            let mut v: Box<[u16]> = chunk.into();
            v.sort_unstable();
            v
        })
        .collect();

    let n_queries = input.value();
    for _ in 0..n_queries {
        let q: u8 = input.value();
        match q {
            1 => {
                let i = input.value::<usize>() - 1;
                let i_bucket = i / bucket_size;
                let v: u16 = input.value();

                let erase_pos = xs_buckets[i_bucket].binary_search(&xs[i]).unwrap();
                let mut insert_pos = xs_buckets[i_bucket].binary_search(&v).unwrap_or_else(|x| x);
                if insert_pos < erase_pos {
                    xs_buckets[i_bucket][insert_pos..=erase_pos].rotate_right(1);
                } else if insert_pos > erase_pos {
                    insert_pos -= 1;
                    xs_buckets[i_bucket][erase_pos..=insert_pos].rotate_left(1);
                }
                xs[i] = v;
                xs_buckets[i_bucket][insert_pos] = v;
            }
            2 => {
                let i = input.value::<usize>() - 1;
                let j = input.value::<usize>() - 1;
                let i_bucket = i / bucket_size;
                let j_bucket = j / bucket_size;
                let k: u16 = input.value();

                let result = if i_bucket == j_bucket {
                    xs[i..=j].iter().filter(|&&x| x > k).count()
                } else if i_bucket < j_bucket {
                    xs_buckets[i_bucket + 1..j_bucket]
                        .iter()
                        .map(|bucket| bucket.len() - bucket.partition_point(|&x| !(x > k)))
                        .sum::<usize>()
                        + xs[i..(i_bucket + 1) * bucket_size]
                            .iter()
                            .filter(|&&x| x > k)
                            .count()
                        + xs[j_bucket * bucket_size..=j]
                            .iter()
                            .filter(|&&x| x > k)
                            .count()
                } else {
                    unreachable!()
                };
                writeln!(output_buf, "{}", result).unwrap();
            }
            _ => unimplemented!(),
        }
    }

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
