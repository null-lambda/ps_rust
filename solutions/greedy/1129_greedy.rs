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

fn partition_point<P>(mut left: u32, mut right: u32, mut pred: P) -> u32
where
    P: FnMut(u32) -> bool,
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

fn test_step(xs_sorted: &[i32], max_step: i32) -> bool {
    debug_assert!(xs_sorted.windows(2).all(|w| w[0] <= w[1]));
    let [mut a, mut b, rest @ ..] = xs_sorted else {
        unimplemented!()
    };

    for &x in rest {
        if a <= b {
            if x - a <= max_step {
                a = x;
            } else if x - b <= max_step {
                b = x;
            } else {
                return false;
            }
        } else {
            if x - b <= max_step {
                b = x;
            } else if x - a <= max_step {
                a = x;
            } else {
                return false;
            }
        }
    }

    (a - b).abs() <= max_step
}

fn construct_lex_min(xs_sorted: &[i32], max_step: i32) -> Vec<i32> {
    debug_assert!(xs_sorted.len() >= 2);
    debug_assert!(xs_sorted.windows(2).all(|w| w[0] <= w[1]));

    let mut left = vec![];
    let mut right = vec![];

    let mut i = 0;
    let mut right_base = xs_sorted[0];
    debug_assert!(xs_sorted[1] - right_base <= max_step);
    while i < xs_sorted.len() {
        let mut i_next = i + 1;
        if i_next == xs_sorted.len() {
            left.push(xs_sorted[i]);
            break;
        }
        while i_next + 1 < xs_sorted.len() && xs_sorted[i_next + 1] - right_base <= max_step {
            i_next += 1;
        }

        for j in i..i_next {
            left.push(xs_sorted[j]);
        }

        if i_next == xs_sorted.len() - 1 {
            left.push(xs_sorted[i_next]);
        } else {
            right.push(xs_sorted[i_next]);
            right_base = xs_sorted[i_next];
        }
        i = i_next + 1;
    }

    left.into_iter().chain(right.into_iter().rev()).collect()
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let mut xs: Vec<i32> = (0..n).map(|_| input.value()).collect();
    xs.sort_unstable();

    let optimal_step = partition_point(0, 1001, |step| !test_step(&xs, step as i32));
    for x in construct_lex_min(&xs, optimal_step as i32) {
        write!(output, "{} ", x).unwrap();
    }
}
