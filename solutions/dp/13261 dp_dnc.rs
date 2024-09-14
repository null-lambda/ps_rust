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

    // let mut output_buf = Vec::<u8>::new();

    use std::iter::once;

    let n: usize = input.value();
    let g_bound: usize = input.value();
    let g_bound = n.min(g_bound);
    let sum: Vec<u64> = once(0)
        .chain((0..n).map(|_| input.value::<u32>()).scan(0, |state, x| {
            *state += x as u64;
            Some(*state)
        }))
        .collect();
    let c = |j: usize, i: usize| (i - j) as u64 * (sum[i] - sum[j]);

    /*
    // O(n^2)
    use std::cmp::Reverse;
    let mut dp = [
        (0..=n).map(|i| (c(0, i), 0)).collect(),
        vec![(0, 0); n + 1],
    ];

    for g in 1..g_bound {
        for i in 1..=n {
            dp[g % 2][i] = (0..i)
                .map(|j| (dp[(g - 1) % 2][j].0 + c(j, i), j))
                .min()
                .unwrap();
        }
        println!("{:?}", dp[g % 2]);
    }
    */

    // divide & conquer optimiziation
    // O(n log(n))

    struct DnCState {
        dp: [Vec<(u64, usize)>; 2],
    }

    struct DnCEnv<F>
    where
        F: Fn(usize, usize) -> u64,
    {
        c: F,
        g: usize,
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
        let i = (start + end) / 2;
        state.dp[env.g % 2][i] = (0.max(j_start)..i.min(j_end))
            .map(|j| (state.dp[(env.g - 1) % 2][j].0 + (env.c)(j, i), j))
            .min()
            .unwrap();

        let (_, opt) = state.dp[env.g % 2][i];
        let opt = opt as usize;

        dnc(state, env, start, i, j_start, opt + 1);
        dnc(state, env, i + 1, end, opt, j_end);
    }

    let mut state = DnCState {
        dp: [(0..=n).map(|i| (c(0, i), 0)).collect(), vec![(0, 0); n + 1]],
    };
    // println!("{:?}", state.dp[0]);
    for g in 1..g_bound {
        dnc(&mut state, &DnCEnv { c, g }, g + 1, n + 1, g, n + 1);
        // println!("{:?}", state.dp[g % 2]);
    }
    let (result, _) = state.dp[(g_bound - 1) % 2][n];
    println!("{}", result);

    // std::io::stdout().write_all(&output_buf[..]).unwrap();
}
