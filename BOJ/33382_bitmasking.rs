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

fn f_naive(base: i64, l: i64, r: i64) -> i64 {
    assert!(l <= r);
    let mask = (1 << base) - 1;

    (l..=r)
        .map(|d| base + ((d & mask).count_ones() as i64) + (d >> base))
        .min()
        .unwrap()
}

fn f(base: i64, l: i64, r: i64) -> i64 {
    assert!(l <= r);
    let mask = (1 << base) - 1;
    let ql = l >> base;
    let qr = r >> base;

    let bl = l & mask;
    let br = r & mask;

    base + ql
        + if ql == qr {
            min_bit_count(bl, br) as i64
        } else {
            bl.count_ones().min(1) as i64
        }
}

fn min_bit_count(mut l: i64, mut r: i64) -> u32 {
    let mut ans = 0;
    while l > 0 {
        ans += 1;

        let msb = 1 << (i64::BITS - 1 - i64::leading_zeros(r));
        if l < msb {
            break;
        }
        l -= msb;
        r -= msb;
    }
    ans
}

#[test]
fn test_f() {
    for base in 0..30 {
        for r in 0..300 {
            for l in 0..=r {
                assert_eq!(f_naive(base, l, r), f(base, l, r), "{:?}", (base, l, r));
            }
        }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let xs: Vec<i64> = (0..n).map(|_| input.value()).collect();
    let mut ys: Vec<i64> = (0..n).map(|_| input.value()).collect();

    const INF: i64 = 1 << 62;

    let mut ans = INF;
    for base in 0..30 {
        let space = 1 << base;
        let mut d = [0, INF];
        for (&x, &y) in xs.iter().zip(&ys) {
            d[0] = d[0].max(y - x);
            d[1] = d[1].min(y - x + space - 1);
        }

        if d[0] <= d[1] {
            ans = ans.min(f(base, d[0], d[1]));
        }

        for y in &mut ys {
            *y *= 2;
        }
    }

    if ans >= INF {
        ans = -1;
    }
    writeln!(output, "{}", ans).unwrap();
}
