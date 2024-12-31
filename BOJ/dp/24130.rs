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

const P: u32 = 10_000_000;

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let w: usize = input.value();
    let h: usize = input.value();

    let h_pad = h + 1;
    let w_pad = w + 1;

    let n: usize = input.value();
    let inst = &input.token().as_bytes()[..n];

    let mut dir;
    let mut dirs_x = vec![];
    let mut dirs_y = vec![];
    if inst[0] == b'L' {
        dir = 0;
        dirs_x.push(true)
    } else {
        dir = 1;
        dirs_y.push(true)
    };
    for &cmd in inst {
        dir = (dir + 4 + 2 * (cmd == b'L') as u8 - 1) % 4;
        if dir % 2 == 0 {
            dirs_x.push(dir == 0);
        } else {
            dirs_y.push(dir == 1);
        }
    }

    let solve_1d = |dirs: &[bool], m: usize| {
        let mut dp = vec![0; m];
        dp[0] = 1;
        for &d in dirs {
            let mut dp_new = vec![0; m];
            if d {
                for i in 1..m {
                    dp_new[i] = (dp_new[i - 1] + dp[i - 1]) % P;
                }
            } else {
                for i in (0..m - 1).rev() {
                    dp_new[i] = (dp_new[i + 1] + dp[i + 1]) % P;
                }
            }
            dp = dp_new;
        }
        dp[m - 1] as u64
    };

    let ans = solve_1d(&dirs_x, w_pad) * solve_1d(&dirs_y, h_pad) % P as u64;
    writeln!(output, "{}", ans).unwrap();
}
