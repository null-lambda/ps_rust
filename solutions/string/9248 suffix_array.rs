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

// list of suffix sorted by dictionary order
fn suffix_array(text: &[u8]) -> Vec<u32> {
    use std::iter::successors;
    let n = text.len();
    assert!(n >= 1);
    let mut sa: Vec<u32> = (0..n as u32).collect();
    let mut rank: Vec<u32> = text.iter().map(|&c| c as u32).collect();
    rank.extend((0..n).map(|_| 0));
    let mut temp: Vec<u32> = vec![0; rank.len()];

    for width in successors(Some(1), |width| Some(width << 1)) {
        let key = |i: usize| (rank[i], rank[i + width]);

        sa.sort_unstable_by_key(|&i| key(i as usize));

        if width << 1 >= n {
            break;
        }

        temp[sa[0] as usize] = 1;
        for i in 1..n {
            temp[sa[i] as usize] = temp[sa[i - 1] as usize]
                + u32::from(key(sa[i] as usize) != key(sa[i - 1] as usize));
        }
        std::mem::swap(&mut rank, &mut temp);

        if rank[n - 1] as usize == n {
            break;
        }
    }
    sa
}

fn longest_common_prefix(text: &[u8], suffix_array: &[u32]) -> Vec<u32> {
    let n = text.len();
    let mut rank = vec![0u32; n];
    let mut lcp = vec![0u32; n];
    for i in 0..n as u32 {
        rank[suffix_array[i as usize] as usize] = i;
    }

    let mut k = 0;
    for i in 0..n as u32 {
        if rank[i as usize] == 0 {
            continue;
        }
        let j = suffix_array[(rank[i as usize] - 1) as usize];
        while k < n as u32 - i.max(j) && text[(i + k) as usize] == text[(j + k) as usize] {
            k += 1;
        }
        lcp[rank[i as usize] as usize] = k;
        k = k.saturating_sub(1);
    }
    lcp
}

fn main() {
    use io::InputStream;
    let input_buf = stdin();
    let mut input: &[u8] = &input_buf[..];

    let mut output_buf = Vec::<u8>::new();

    let text = input.token();
    let sa = suffix_array(text);

    let lcp = longest_common_prefix(text, &sa);
    for &i in &sa {
        write!(output_buf, "{} ", i + 1).unwrap();
        // writeln!(output_buf, "{}", std::str::from_utf8(&text[i..]).unwrap()).unwrap();
    }
    writeln!(output_buf).unwrap();
    write!(output_buf, "x ").unwrap();
    for &i in lcp.iter().skip(1) {
        write!(output_buf, "{} ", i).unwrap();
    }

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
