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

fn gcd(mut a: u64, mut b: u64) -> u64 {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

// Extended euclidean algorithm
// Find (d, x, y) satisfying d = gcd(a, b) and a * x + b * y = d
fn egcd(a: u64, b: u64) -> (u64, i64, i64) {
    let (mut c, mut x, mut y) = if a > b {
        ((a, b), (1, 0), (0, 1))
    } else {
        ((b, a), (0, 1), (1, 0))
    };

    while c.1 > 0 {
        let q = c.0 / c.1;
        x = (x.1, (x.0 - (q as i64) * x.1));
        y = (y.1, (y.0 - (q as i64) * y.1));
        c = (c.1, c.0 % c.1);
    }
    (c.0, x.0, y.0)
}

// Find (d, x, y) satisfying d = gcd(abs(a), abs(b)) and a * x + b * y = d
fn egcd_i64(a: i64, b: i64) -> (i64, i64, i64) {
    let (d, x, y) = egcd(a.abs() as u64, b.abs() as u64);
    (d as i64, x as i64 * a.signum(), y as i64 * b.signum())
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let p0: [i64; 2] = std::array::from_fn(|_| input.value());
    let gens: Vec<[i64; 2]> = (0..n - 1)
        .map(|_| std::array::from_fn(|i| input.value::<i64>() - p0[i]))
        .collect();
    let cross = |b: [i64; 2], c: [i64; 2]| b[0] * c[1] - b[1] * c[0];

    let mut basis = vec![];
    for mut g in gens {
        assert!(g != [0; 2]);
        match &mut basis[..] {
            [] => basis.push(g),
            [b] => {
                if cross(*b, g) == 0 {
                    let r = if b[0] != 0 {
                        b[0] / gcd(b[0].abs() as u64, g[0].abs() as u64) as i64
                    } else {
                        b[1] / gcd(b[1].abs() as u64, g[1].abs() as u64) as i64
                    };
                    *b = b.map(|x| x / r);
                } else {
                    basis.push(g);
                }
            }
            [b, c] => {
                let reduce = |r: &mut [i64; 2], s: &mut [i64; 2]| {
                    if r[0] == 0 && s[0] == 0 {
                        let g = gcd(r[1].abs() as u64, s[1].abs() as u64) as i64;
                        *r = [0, g];
                        *s = [0, 0];
                        return;
                    }

                    if s[0] != 0 {
                        let (g, u, v) = egcd_i64(r[0], s[0]);
                        let r_next = [g, u * r[1] + v * s[1]];
                        let s_next = [0, cross(*r, *s) / g];
                        *r = r_next;
                        *s = s_next;
                    }

                    if s[1] != 0 {
                        r[1] %= s[1]; // Prevent overflow
                    }
                };

                reduce(&mut g, b);
                reduce(&mut g, c);
                reduce(c, b);
                assert_eq!(b, &[0; 2]);

                if c[1] != 0 {
                    g[1] %= c[1]; // Prevent overflow
                }
                *b = g;
            }
            _ => panic!(),
        }
    }

    let ans = match basis[..] {
        [] | [_] => -1,
        [b, c] => cross(b, c).abs(),
        _ => panic!(),
    };

    writeln!(output, "{}", ans).unwrap();
}
