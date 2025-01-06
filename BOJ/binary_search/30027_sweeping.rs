use std::io::Write;

mod simple_io {
    use std::string::*;

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

fn partition_point<P>(mut left: i32, mut right: i32, mut pred: P) -> i32
where
    P: FnMut(i32) -> bool,
{
    while left < right {
        let mid = left + (right - left) / 2;
        if pred(mid) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    left
}

pub fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: i32 = input.value();
    let m: i32 = input.value();
    let k: usize = input.value();
    let mut ps: Vec<(i32, i32)> = (0..k)
        .map(|_| (input.value::<i32>() - 1, input.value::<i32>() - 1))
        .collect();
    assert!(k >= 1);

    let ans = (0..n)
        .map(|y_base| {
            // stable sort, since the ordering changes infrequently
            ps.sort_by_key(|&(y, x)| x + (y - y_base).abs());
            let satisfiable = |day: i32| {
                let interval = |j: usize| {
                    let (y, x) = ps[j];
                    let dy = (y - y_base).abs();
                    let start = x + dy - day;
                    let len = 0.max(1 + 2 * (day - dy));
                    (start, start + len)
                };

                if interval(0).0 > 0 {
                    return false;
                }
                let mut current_end = interval(0).1;

                for i in 1..k {
                    let (start, end) = interval(i);
                    if start >= end {
                        continue;
                    }
                    if current_end < start {
                        return false;
                    }
                    current_end = current_end.max(end);
                }

                if current_end < m {
                    return false;
                }
                true
            };
            partition_point(0, n + m, |day| !satisfiable(day))
        })
        // .inspect(|&x| eprintln!("x = {}", x))
        .max()
        .unwrap();
    writeln!(output, "{}", ans).unwrap();
}
