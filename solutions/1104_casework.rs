use std::io::Write;

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

    let s1 = input.token().as_bytes();
    let s2 = input.token().as_bytes();
    let c: u128 = input.value();

    let n1 = s1.len();
    let n2 = s2.len();

    let s1_rep = 1_000_000u128;

    let s1_all_zero = s1.iter().all(|&c| c == b'0');
    let s2_all_zero = s2.iter().all(|&c| c == b'0');

    let s1_double: Vec<u8> = s1.iter().chain(s1).copied().collect();
    let s2_double: Vec<u8> = s2.iter().chain(s2).copied().collect();
    let inf = 10u128.pow(16) + 1;
    let mut ans = inf;

    let s1_zero_left = s1.iter().take_while(|&&c| c == b'0').count();
    let s1_zero_right = s1.iter().rev().take_while(|&&c| c == b'0').count();

    let s2_zero_left = s2.iter().take_while(|&&c| c == b'0').count();
    let s2_zero_right = s2.iter().rev().take_while(|&&c| c == b'0').count();

    if c <= n1 as u128 * 2 {
        for i in 0..=n1 * 2 - c as usize {
            if s1_double[i..i + c as usize].iter().all(|&c| c == b'0') {
                ans = ans.min(i as u128);
            }
        }
    }
    if c <= n2 as u128 {
        for i in 0..=n2 - c as usize {
            if s2[i..i + c as usize].iter().all(|&c| c == b'0') {
                ans = ans.min(n1 as u128 * s1_rep + i as u128);
            }
        }
    }
    if c <= n2 as u128 * 2 {
        for i in 0..=n2 * 2 - c as usize {
            if s2_double[i..i + c as usize].iter().all(|&c| c == b'0') {
                ans = ans.min(n1 as u128 * s1_rep * 2 + n2 as u128 + i as u128);
            }
        }
    }

    if s1_all_zero && s2_all_zero {
        ans = ans.min(0);
    } else if s1_all_zero {
        if c <= s1_rep * n1 as u128 {
            ans = ans.min(0);
        } else if c <= s1_rep * n1 as u128 + s2_zero_left as u128 {
            ans = ans.min(0);
        } else if c <= s1_rep * n1 as u128 + s2_zero_left as u128 + s2_zero_right as u128 {
            ans = ans.min(s1_rep * n1 as u128 + n2 as u128 - s2_zero_right as u128);
        }
    } else if s2_all_zero {
        let ex = (s1_zero_left + s1_zero_right) as u128;
        let k = (c as u128 - ex as u128 + n2 as u128 - 1) / n2 as u128;
        assert!(c <= ex + k * n2 as u128);
        if k >= 1 {
            ans = ans.min(
                s1_rep * n1 as u128 * k + k * (k - 1) / 2 * n2 as u128 - s1_zero_right as u128,
            );
        }
        // if c <= (s1_zero_right
    } else {
        if c <= (s1_zero_right + s2_zero_left) as u128 {
            ans = ans.min(s1_rep * n1 as u128 - s1_zero_right as u128);
        }
        if c <= (s2_zero_right + s1_zero_left) as u128 {
            ans = ans.min(s1_rep * n1 as u128 + n2 as u128 - s2_zero_right as u128);
        }
    }

    if ans >= inf || ans + c - 1 >= inf || c >= inf {
        writeln!(output, "-1").unwrap();
    } else {
        writeln!(output, "{}", ans).unwrap();
    }
}
