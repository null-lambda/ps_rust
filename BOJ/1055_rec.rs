use std::{
    cmp::{Ordering, Reverse},
    collections::HashMap,
    io::Write,
};

mod simple_io {
    pub struct InputAtOnce<'a> {
        _buf: String,
        iter: std::str::SplitAsciiWhitespace<'a>,
    }

    impl<'a> InputAtOnce<'a> {
        pub fn token(&mut self) -> &'a str {
            self.iter.next().unwrap_or_default()
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().unwrap()
        }
    }

    pub fn stdin_at_once<'a>() -> InputAtOnce<'a> {
        let _buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let iter = _buf.split_ascii_whitespace();
        let iter = unsafe { std::mem::transmute(iter) };
        InputAtOnce { _buf, iter }
    }

    pub fn stdout() -> std::io::BufWriter<std::io::Stdout> {
        std::io::BufWriter::new(std::io::stdout())
    }
}

const MAX_LEN: u64 = 1_000_000_000_000;

fn get_len(dp: &mut HashMap<u64, u64>, s: &[u8], s0: &[u8], depth: u64) -> u64 {
    if let Some(&res) = dp.get(&depth) {
        return res;
    }

    if depth == 0 {
        return s0.len() as u64;
    }

    let sub_len = get_len(dp, s, s0, depth - 1);
    let mut res = 0;
    for &c in s {
        if c == b'$' {
            res += sub_len;
        } else {
            res += 1;
        }
    }

    res = res.min(MAX_LEN);

    dp.insert(depth, res);
    res
}

fn get_nth(
    dp_len: &mut HashMap<u64, u64>,
    s: &[u8],
    s0: &[u8],
    mut pos: u64,
    depth: u64,
) -> Option<u8> {
    if depth == 0 {
        return s0.get(pos as usize).copied();
    }

    for &c in s {
        if c == b'$' {
            let l = get_len(dp_len, s, s0, depth - 1);
            if pos < l {
                return get_nth(dp_len, s, s0, pos, depth - 1);
            }
            pos -= l;
        } else {
            if pos == 0 {
                return Some(c);
            }
            pos -= 1;
        }
    }

    None
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let s0 = input.token().as_bytes();
    let s = input.token().as_bytes();
    let depth = input.value::<u64>();

    let p = s.iter().filter(|&&b| b == b'$').count() as u64;

    let l = input.value::<u64>() - 1;
    let r = input.value::<u64>() - 1;
    let mut dp_len = HashMap::new();
    for pos in l..=r {
        let ans = if p == 0 {
            s.get(pos as usize).copied()
        } else if p == 1 {
            let marker_pos = s.iter().position(|&b| b == b'$').unwrap() as u64;
            let rest_len = s.len() as u64 - marker_pos - 1;
            let len = (s.len() - 1) as u64 * depth + s0.len() as u64;
            if pos < marker_pos * depth {
                Some(s[(pos % marker_pos) as usize])
            } else if pos < marker_pos * depth + s0.len() as u64 {
                Some(s0[(pos - marker_pos * depth) as usize])
            } else if pos < len {
                Some(
                    s[(marker_pos + 1 + (pos - s0.len() as u64 - marker_pos * depth) % rest_len)
                        as usize],
                )
            } else {
                None
            }
        } else {
            let depth = depth.min(1_000_000_000f64.log2().ceil() as u64);
            get_nth(&mut dp_len, s, s0, pos, depth)
        };
        write!(output, "{}", ans.unwrap_or(b'-') as char).unwrap();
    }
}
