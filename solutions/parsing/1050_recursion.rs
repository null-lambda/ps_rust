use std::{
    cmp::Reverse,
    collections::HashMap,
    io::Write,
    iter::{self, repeat},
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

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();

    let trunc = 1_000_000_001;
    let inf = 1e15 as u64;
    let mut cost: HashMap<&[u8], u64> = HashMap::new();
    let mut recipes: Vec<(&[u8], Vec<(u64, &[u8])>)> = vec![];

    for _ in 0..n {
        cost.insert(input.token().as_bytes(), input.value());
    }

    for _ in 0..m {
        let mut line = input.token().as_bytes();

        fn skip_spaces<'a>(s: &mut &[u8]) {
            while let [first, rest @ ..] = s {
                if first.is_ascii_whitespace() {
                    *s = rest;
                } else {
                    return;
                }
            }
        }

        fn ident<'a>(s: &mut &'a [u8]) -> Option<&'a [u8]> {
            let i = s
                .iter()
                .position(|&c| !c.is_ascii_alphabetic())
                .unwrap_or(s.len());
            let (t, rest) = s.split_at(i);
            *s = rest;
            (t.len() > 0).then(|| t)
        }

        fn digit(s: &mut &[u8]) -> Option<u64> {
            match s {
                [c @ b'0'..=b'9', rest @ ..] => {
                    *s = rest;
                    Some((c - b'0') as u64)
                }
                _ => None,
            }
        }

        fn lit_byte(s: &mut &[u8], c: u8) -> Option<()> {
            let (d, rest) = s.split_first()?;
            if *d != c {
                return None;
            }
            *s = rest;
            Some(())
        }

        let dest = ident(&mut line).unwrap();
        skip_spaces(&mut line);
        lit_byte(&mut line, b'=').unwrap();
        skip_spaces(&mut line);

        let mut srcs = vec![];
        loop {
            let coeff = digit(&mut line).unwrap();
            skip_spaces(&mut line);
            let src = ident(&mut line).unwrap();
            skip_spaces(&mut line);
            srcs.push((coeff, src));
            if lit_byte(&mut line, b'+').is_none() {
                break;
            }
            skip_spaces(&mut line);
        }
        recipes.push((dest, srcs));
    }

    for &(dest, _) in &recipes {
        if !cost.contains_key(dest) {
            cost.insert(dest, inf);
        }
    }

    for _ in 0..recipes.len() {
        'outer: for (dest, srcs) in &recipes {
            let mut updated_cost = 0;
            for (coeff, src) in srcs {
                let Some(&c) = cost.get(src) else {
                    continue 'outer;
                };
                if c == inf {
                    continue 'outer;
                }
                updated_cost += coeff * c;
            }
            updated_cost = updated_cost.min(trunc);

            let entry = cost.get_mut(dest).unwrap();
            *entry = (*entry).min(updated_cost);
        }
    }

    let ans = cost.get(&b"LOVE"[..]).copied().unwrap_or(inf);
    if ans == inf {
        writeln!(output, "-1").unwrap();
    } else {
        writeln!(output, "{}", ans.min(trunc)).unwrap();
    }
}
