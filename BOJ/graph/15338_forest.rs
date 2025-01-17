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

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: u32 = input.value();
    let a: usize = input.value();
    let b: usize = input.value();
    let q: usize = input.value();
    let positions: Vec<_> = (0..a)
        .map(|_| (input.value::<u32>(), input.token().as_bytes()[0]))
        .collect();
    let parent: Vec<_> = (0..b)
        .map(|_| (input.value::<u32>(), input.value::<u32>()))
        .chain(Some((n + 1, n + 1)))
        .collect();
    let ranged_parent: Vec<_> = parent
        .windows(2)
        .map(|p| {
            let ((y, h), (y_next, _)) = (p[0], p[1]);
            (y, y_next, h)
        })
        .filter(|&(.., h)| h != 0)
        .collect();
    let ascend_to_root = |mut u: u32| {
        for &(y, y_next, h) in ranged_parent.iter().rev() {
            if u < y {
                continue;
            }
            if u >= y_next {
                break;
            }
            u = h + (u - h) % (y - h);
        }
        u
    };

    let mut known: Vec<(u32, u8)> = positions
        .iter()
        .map(|&(u, b)| (ascend_to_root(u), b))
        .collect();
    known.sort_unstable_by_key(|&(u, _)| u);

    for _ in 0..q {
        let u = ascend_to_root(input.value());
        if let Ok(i) = known.binary_search_by_key(&u, |&(u, _)| u) {
            let (_, b) = known[i];
            write!(output, "{}", b as char).unwrap();
        } else {
            write!(output, "?").unwrap();
        }
    }
    writeln!(output).unwrap();
}
