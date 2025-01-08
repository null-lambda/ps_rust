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

fn solve_naive(m: usize, perm: &[u32]) -> u64 {
    let n = perm.len();
    assert_eq!(n, 1 << m);
    let top = (1 << m) - 1;

    let mut acc = 0;
    for i in 0..n {
        'outer: for j in i..n {
            let xor = perm[i..=j]
                .iter()
                .copied()
                .reduce(|acc, x| acc ^ x)
                .unwrap();

            let len = j - i + 1;
            if (len >= 2 || len <= n - 2) && xor ^ top == 0 {
                acc += 1;
                continue;
            }
            for p in 0..n {
                for q in 0..n {
                    if (i..=j).contains(&p) && !(i..=j).contains(&q) {
                        if xor ^ perm[p] ^ perm[q] ^ top == 0 {
                            acc += 1;
                            continue 'outer;
                        }
                    }
                }
            }

            println!("{}..={} xor={:b}", i, j, xor);
        }
    }
    acc
}

pub fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let m: usize = input.value();
    let perm: Vec<u32> = (0..1 << m).map(|_| input.value()).collect();

    let ans = match m {
        0 => panic!(),
        1..=4 => solve_naive(m, &perm),
        // 2..=4 => solve_naive(m, &perm),
        _ => {
            let n = 1u64 << m;
            let mut ans = n * (n + 1) / 2 - 1;
            // perm = (a1, .., ar) // rest
            // where xor(rest) = a1^...^ar =: s = (s^t)^t
            // => for some i, swap ai and s^t^ai, where s^t^ai != aj for all j
            // => exists i. forall j. s^ai^aj != t
            ans
        }
    };
    writeln!(output, "{}", ans).unwrap();
}
