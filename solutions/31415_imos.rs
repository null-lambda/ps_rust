use std::{cmp::Reverse, collections::HashMap, io::Write};

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

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let q: usize = input.value();
    let t: i64 = input.value();

    let min_bound = 0;
    let max_bound = 100_000i32;
    let clamp_ext = |x: i64| x.max(min_bound as i64 - 1).min(max_bound as i64 + 1);
    let in_bound = |x: i32| (min_bound..=max_bound).contains(&x);

    let mut xs_imos = HashMap::<(i32, i32), Vec<(i32, i32)>>::new();
    let mut ys_imos = HashMap::<(i32, i32), Vec<(i32, i32)>>::new();
    for _ in 0..n {
        let x: i32 = input.value();
        let y: i32 = input.value();
        let dx: i32 = input.value();
        let dy: i32 = input.value();

        let x_mod = if dx == 0 { 0 } else { x % dx.abs() };
        let y_mod = if dy == 0 { 0 } else { y % dy.abs() };
        assert!(x_mod >= 0);
        assert!(y_mod >= 0);

        let x_term = clamp_ext(x as i64 + dx as i64 * t as i64) as i32;
        let y_term = clamp_ext(y as i64 + dy as i64 * t as i64) as i32;
        xs_imos.entry((dx, x_mod)).or_default().push((x, 1));
        xs_imos.entry((dx, x_mod)).or_default().push((x_term, -1));

        ys_imos.entry((dy, y_mod)).or_default().push((y, 1));
        ys_imos.entry((dy, y_mod)).or_default().push((y_term, -1));
    }

    let solve_1d = |xs_imos: HashMap<(i32, i32), Vec<(i32, i32)>>| {
        let mut xs_count = vec![0i32; max_bound as usize + 1];
        for ((dx, x_mod), mut imos) in xs_imos {
            if dx == 0 {
                for (x, v) in imos {
                    if v == 1 {
                        xs_count[x as usize] += 1;
                    }
                }
                continue;
            }

            if dx > 0 {
                imos.sort_unstable_by_key(|(x, _)| *x);
            } else {
                imos.sort_unstable_by_key(|(x, _)| Reverse(*x));
            }
            let mut imos = imos.into_iter().peekable();

            let mut x_curr = if dx > 0 {
                x_mod
            } else {
                (max_bound - x_mod) / dx.abs() * dx.abs() + x_mod
            };

            let mut count = 0;
            while in_bound(x_curr) {
                while let Some((_, dc)) = imos.next_if(|(x, _)| *x == x_curr) {
                    count += dc;
                }

                xs_count[x_curr as usize] += count;

                x_curr += dx;
            }
        }
        xs_count
    };

    let xs_count = solve_1d(xs_imos);
    let ys_count = solve_1d(ys_imos);

    for _ in 0..q {
        let ans = match input.token() {
            "1" => ys_count[input.value::<usize>()],
            "2" => xs_count[input.value::<usize>()],
            _ => panic!(),
        };
        writeln!(output, "{}", ans).unwrap();
    }
}
