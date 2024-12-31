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

    use std::iter::once;

    let n: usize = input.value();
    assert!(n >= 1);
    let mut segments: Vec<_> = (0..n)
        .map(|_| {
            let mut x0: i32 = input.value();
            let mut x1: i32 = input.value();
            if x0 > x1 {
                std::mem::swap(&mut x0, &mut x1);
            }
            let y: i32 = input.value();
            [x0, x1, y, x1 - x0]
        })
        .collect();
    segments.sort_unstable_by_key(|&[x0, ..]| x0);

    let result = segments
        .iter()
        .map(|&[x, _, y, base_score]| {
            let mut events: Vec<_> = segments
                .iter()
                .filter(|&&[_, _, py, _]| y != py)
                .flat_map(|&[px0, px1, py, score]| {
                    let [dx0, dx1, dy] = [px0 - x, px1 - x, py - y];
                    if dy > 0 {
                        once([dx0, dy, score]).chain(once([dx1, dy, -score]))
                    } else {
                        once([-dx1, -dy, score]).chain(once([-dx0, -dy, -score]))
                    }
                })
                .collect();
            let cmp = |&[px, py, p_score]: &[i32; 3], &[qx, qy, q_score]: &[i32; 3]| {
                (px as i64 * qy as i64 - py as i64 * qx as i64)
                    .cmp(&0)
                    .then_with(|| p_score.cmp(&q_score).reverse())
            };
            events.sort_by(cmp);

            base_score
                + events
                    .iter()
                    .map(|&[.., score]| score)
                    .scan(0, |acc, score| {
                        *acc += score;
                        Some(*acc)
                    })
                    .max()
                    .unwrap_or(0)
        })
        .max()
        .unwrap();

    println!("{:?}", result);

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
