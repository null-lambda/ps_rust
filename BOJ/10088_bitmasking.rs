use std::io::Write;

mod simple_io {
    pub struct InputAtOnce {
        iter: std::str::SplitAsciiWhitespace<'static>,
    }

    impl InputAtOnce {
        pub fn token(&mut self) -> &'static str {
            self.iter.next().unwrap_or_default()
        }

        pub fn try_value<T: std::str::FromStr>(&mut self) -> Option<T> {
            self.token().parse().ok()
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.try_value().unwrap()
        }
    }

    pub fn stdin() -> InputAtOnce {
        let buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let buf = Box::leak(Box::new(buf));
        let iter = buf.split_ascii_whitespace();
        InputAtOnce { iter }
    }

    pub fn stdout() -> std::io::BufWriter<std::io::Stdout> {
        std::io::BufWriter::new(std::io::stdout())
    }
}

fn q_naive(l: u64, r: u64, s: u64) -> u64 {
    #[target_feature(enable = "avx2")]
    unsafe fn inner(l: u64, r: u64, s: u64) -> u64 {
        (l..=r).map(|x| x ^ (x + s)).max().unwrap()
        // s + 2 * (l..=r).map(|x| x & !(x + s)).max().unwrap()
    }
    unsafe { inner(l, r, s) }
}

fn q(l: u64, r: u64, s: u64) -> u64 {
    // q_naive(l, r, s)
    //
    let mut intervals = vec![];
    for b in (0..63).rev() {
        let y = (r >> b) << b;
        if y >= l {
            intervals.push([l.max(y.saturating_sub(s.next_power_of_two())), y]);
        }
    }

    if !intervals.is_empty() {
        let mut merged = vec![intervals[0]];
        for &[s, e] in intervals[1..].iter() {
            let [_, e_prev] = merged.last_mut().unwrap();
            if s > *e_prev {
                merged.push([s, e]);
            } else {
                (*e_prev) = (*e_prev).max(e);
            }
        }
        intervals = merged;
    }

    let mut ans = 0;
    for [l, r] in intervals {
        ans = ans.max(q_naive(l, r, s));
    }
    ans
}

fn p(l: u64, r: u64, s: u64) -> u64 {
    assert!(s <= l && l <= r);
    assert!(s <= 3e7 as u64);
    q(l - s, r - s, s)
}

fn f(a: u64) -> u64 {
    match a % 4 {
        0 => a,
        1 => 1,
        2 => a + 1,
        _ => 0,
    }
}

fn g_naive(a: u64, b: u64, n: u64) -> u64 {
    #[target_feature(enable = "avx2")]
    unsafe fn inner(a: u64, b: u64, n: u64) -> u64 {
        (a + n - 1..=b).map(|x| f(x) ^ f(x - n)).max().unwrap()
    }
    unsafe { inner(a, b, n) }
}

fn g_n0(a: u64, b: u64, n: u64) -> u64 {
    assert!(n % 4 == 0);
    let l = a + n - 1;
    let r = b;
    p((l + 1) / 4, r / 4, n / 4) * 4
}

fn g_n2(a: u64, b: u64, n: u64) -> u64 {
    assert!(n % 4 == 2);
    let l = a + n - 1;
    let r = b;
    p((l + 3) / 4, r / 4, (n + 2) / 4).max(p((l + 1) / 4, (r - 2) / 4, (n - 2) / 4)) * 4 + 3
}

fn g(a: u64, b: u64, n: u64) -> u64 {
    unsafe fn inner(a: u64, b: u64, n: u64) -> u64 {
        if b - a + 1 - n < 12 {
            return g_naive(a, b, n);
        }

        let mut base = g_naive(b - (n - 1) - 8, b, n);
        match n % 4 {
            0 => base = base.max(g_n0(a, b, n)),
            2 => base = base.max(g_n2(a, b, n)),
            _ => {}
        }
        base
    }
    unsafe { inner(a, b, n) }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    {
        // let r = 150;
        // for n in 1..=r {
        //     for a in 1..=r {
        //         for b in a + n - 1..=a + n - 1 + r {
        //             assert_eq!(g(a, b, n), g_naive(a, b, n));
        //         }
        //     }
        // }
    }

    let a: u64 = input.value();
    let b: u64 = input.value();
    let n: u64 = input.value();

    let ans = g(a, b, n);
    // let ans = g_naive(a, b, n);
    writeln!(output, "{}", ans).unwrap();
}
