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

mod cmp {
    use std::cmp::Ordering;

    // x <= y iff x = y
    #[derive(Debug, Copy, Clone, Default)]
    pub struct Trivial<T>(pub T);

    impl<T> PartialEq for Trivial<T> {
        #[inline]
        fn eq(&self, _other: &Self) -> bool {
            true
        }
    }
    impl<T> Eq for Trivial<T> {}

    impl<T> PartialOrd for Trivial<T> {
        #[inline]
        fn partial_cmp(&self, _other: &Self) -> Option<Ordering> {
            Some(Ordering::Equal)
        }
    }

    impl<T> Ord for Trivial<T> {
        #[inline]
        fn cmp(&self, _other: &Self) -> Ordering {
            Ordering::Equal
        }
    }
}

fn main() {
    use io::InputStream;
    let input_buf = stdin();
    let mut input: &[u8] = &input_buf[..];

    // let mut output_buf = Vec::<u8>::new();

    let n: usize = input.value();
    let d: usize = input.value();
    let ts: Vec<u32> = (0..n).map(|_| input.value()).collect();
    let vs: Vec<u32> = (0..n).map(|_| input.value()).collect();
    let c = |i: usize, j: usize| (j - i) as u64 * ts[j] as u64 + vs[i] as u64;

    /*
    // O(n^2)
    use std::cmp::Reverse;
    for i in 0..n {
        let mut total = (0, Reverse(0));
        for j in i..(i + d + 1).min(n) {
            total = total.max((c(i, j), Reverse(j)));
        }
        let (score, Reverse(opt)) = total;
        println!("{}: {:?}", i, (score, opt));
    }
    */

    // divide & conquer optimiziation
    // O(n log(n))
    use cmp::Trivial;

    struct DnCState {
        dp: Vec<(u64, Trivial<usize>)>,
        result: u64,
    }

    struct DnCEnv<F>
    where
        F: Fn(usize, usize) -> u64,
    {
        d: usize,
        c: F,
    }

    fn dnc(
        state: &mut DnCState,
        env: &DnCEnv<impl Fn(usize, usize) -> u64>,
        start: usize,
        end: usize,
        j_start: usize,
        j_end: usize,
    ) {
        if start >= end {
            return;
        }
        let mid = (start + end) / 2;
        for j in mid.max(j_start)..(mid + env.d + 1).min(j_end) {
            state.dp[mid] = state.dp[mid].max(((env.c)(mid, j), Trivial(j)));
        }

        let (score, Trivial(opt)) = state.dp[mid];
        let opt = opt as usize;
        state.result = state.result.max(score);

        dnc(state, env, start, mid, j_start, opt + 1);
        dnc(state, env, mid + 1, end, opt, j_end);
    }

    let mut state = DnCState {
        dp: vec![(0, Trivial(0)); n],
        result: 0,
    };
    dnc(&mut state, &DnCEnv { d, c }, 0, n, 0, n);
    println!("{}", state.result);

    // std::io::stdout().write_all(&output_buf[..]).unwrap();
}
