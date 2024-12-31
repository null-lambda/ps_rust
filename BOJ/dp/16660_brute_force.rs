use std::{collections::HashMap, io::Write};

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

    pub fn stdin<'a>() -> InputAtOnce<'a> {
        let _buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let iter = _buf.split_ascii_whitespace();
        let iter = unsafe { std::mem::transmute(iter) };
        InputAtOnce { _buf, iter }
    }

    pub fn stdout() -> std::io::BufWriter<std::io::Stdout> {
        std::io::BufWriter::new(std::io::stdout())
    }
}

const BOUND: u8 = 200;

type Set = HashMap<(u8, u8), Box<[(u8, u8)]>>;

fn gen_set(win: u8, flip: bool) -> Set {
    let mut res = vec![];
    for a in 0..=win - 2 {
        res.push((win, a));
    }
    for a in win + 1..=BOUND {
        res.push((a, a - 2));
    }

    if flip {
        res = res.into_iter().map(|(a, b)| (b, a)).collect();
    }

    res.into_iter().map(|t| (t, [t].into())).collect()
}

fn merge(mut lhs: Set, rhs: Set) -> Set {
    for (k, v) in rhs {
        lhs.entry(k).or_insert(v);
    }
    lhs
}

fn conv(lhs: Set, rhs: Set) -> Set {
    let mut res = HashMap::new();
    for (k1, v1) in lhs {
        for (k2, v2) in &rhs {
            let k = (k1.0 as u32 + k2.0 as u32, k1.1 as u32 + k2.1 as u32);
            if !(0..=BOUND as u32).contains(&k.0) || !(0..=BOUND as u32).contains(&k.1) {
                continue;
            }

            let k = (k.0 as u8, k.1 as u8);
            res.entry(k)
                .or_insert_with(|| v1.iter().copied().chain(v2.iter().copied()).collect());
        }
    }
    res
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let left_15 = gen_set(15, false);
    let right_15 = gen_set(15, true);
    let left_25 = gen_set(25, false);
    let right_25 = gen_set(25, true);

    let mut dp = HashMap::<_, Set>::new();
    for a in 0..=3 {
        for b in 0..=3 {
            let mut s: Set = Default::default();

            match (a, b) {
                (0, 0) | (3, 3) => continue,
                (1, 0) => s = left_25.clone(),
                (0, 1) => s = right_25.clone(),
                (3, 2) => s = conv(dp[&(2, 2)].clone(), left_15.clone()),
                (2, 3) => s = conv(dp[&(2, 2)].clone(), right_15.clone()),
                (_, _) => {
                    if a > 0 && b != 3 {
                        s = merge(s, conv(dp[&(a - 1, b)].clone(), left_25.clone()));
                    }
                    if b > 0 && a != 3 {
                        s = merge(s, conv(dp[&(a, b - 1)].clone(), right_25.clone()));
                    }
                }
            }

            dp.entry((a, b)).or_insert(s);
        }
    }

    for _ in 0..input.value() {
        let x: u8 = input.value();
        let y: u8 = input.value();

        let mut ans = None;
        'outer: for diff in (-3..=3).rev() {
            for a in 0..=3 {
                let b = a - diff;
                if !(0..=3).contains(&b) {
                    continue;
                }
                if a != 3 && b != 3 {
                    continue;
                }

                if let Some(s) = dp.get(&(a, b)) {
                    if let Some(v) = s.get(&(x, y)) {
                        ans = Some(((a, b), v));
                        break 'outer;
                    }
                }
            }
        }

        if let Some(((a, b), v)) = ans {
            writeln!(output, "{}:{}", a, b).unwrap();
            for (p, q) in v.iter() {
                write!(output, "{}:{} ", p, q).unwrap();
            }
            writeln!(output).unwrap();
        } else {
            writeln!(output, "Impossible").unwrap();
        }
    }
}
