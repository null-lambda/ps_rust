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

fn div_rem(x: u64, y: u64) -> (u64, u64) {
    (x / y, x % y)
}

fn div_rem_ceil(x: u64, y: u64) -> (u64, u64) {
    let q = x.div_ceil(y);
    (q, x + y - q * y)
}

type Pieces = [u64; 6];

fn combine_pieces(lhs: Pieces, rhs: Pieces) -> Pieces {
    let mut res = [0; 6];
    for i in 0..6 {
        res[i] = lhs[i] + rhs[i];
    }
    res
}

fn mul_scalar(lhs: Pieces, rhs: u64) -> Pieces {
    let mut res = [0; 6];
    for i in 0..6 {
        res[i] = lhs[i] * rhs;
    }
    res
}

fn single(idx: usize, count: u64) -> Pieces {
    if count == 0 {
        return zero();
    }
    assert_ne!(idx, 0);
    let mut res = [0; 6];
    res[idx] = count;
    res
}

fn zero() -> Pieces {
    [0; 6]
}

fn f(x1: u64, y1: u64, x2: u64, y2: u64) -> Pieces {
    assert!(x1 <= x2 && y1 <= y2);
    let (qx1, rx1) = div_rem(x1, 5);
    let (qy1, ry1) = div_rem(y1, 5);
    let (cqx2, crx2) = div_rem_ceil(x2, 5);
    let (cqy2, cry2) = div_rem_ceil(y2, 5);

    let dqx = cqx2 - qx1;
    let dqy = cqy2 - qy1;

    if x1 == x2 || y1 == y2 {
        return zero();
    }
    if (rx1, crx2 % 5, ry1, cry2 % 5) == (0, 0, 0, 0) {
        return single(5, dqx * dqy * 5);
    }

    let x_split = vec![x1, (qx1 + 1) * 5, (cqx2 - 1) * 5, x2];
    let y_split = vec![y1, (qy1 + 1) * 5, (cqy2 - 1) * 5, y2];

    let mut res = zero();
    match (dqx, dqy) {
        (..=1, ..=1) => {
            if (qx1 + qy1) % 2 == 0 {
                res = single((crx2 - rx1) as usize, cry2 - ry1);
            } else {
                res = single((cry2 - ry1) as usize, crx2 - rx1);
            }
        }

        (..=1, 2..) => {
            let b1 = f(x1, y_split[1], x2, y_split[1] + 5);
            let b2 = f(x1, y_split[1] + 5, x2, y_split[1] + 10);
            res = combine_pieces(res, f(x1, y1, x2, y_split[1]));
            res = combine_pieces(res, mul_scalar(b1, (dqy - 1) / 2));
            res = combine_pieces(res, mul_scalar(b2, (dqy - 2) / 2));
            res = combine_pieces(res, f(x1, y_split[2], x2, y2));
        }
        (2.., ..=1) => {
            let b1 = f(x_split[1], y1, x_split[1] + 5, y2);
            let b2 = f(x_split[1] + 5, y1, x_split[1] + 10, y2);
            res = combine_pieces(res, f(x1, y1, x_split[1], y2));
            res = combine_pieces(res, mul_scalar(b1, (dqx - 1) / 2));
            res = combine_pieces(res, mul_scalar(b2, (dqx - 2) / 2));
            res = combine_pieces(res, f(x_split[2], y1, x2, y2));
        }
        (2.., 2..) => {
            for tx in x_split.windows(2) {
                for ty in y_split.windows(2) {
                    res = combine_pieces(res, f(tx[0], ty[0], tx[1], ty[1]));
                }
            }
        }
    }
    res
}

fn group_pieces(mut pieces: Pieces) -> u64 {
    let m = pieces[4].min(pieces[1]);
    pieces[4] -= m;
    pieces[1] -= m;
    pieces[5] += m;

    let m = pieces[3].min(pieces[2]);
    pieces[3] -= m;
    pieces[2] -= m;
    pieces[5] += m;

    let m = pieces[3].min(pieces[1] / 2);
    pieces[1] -= 2 * m;
    pieces[3] -= m;
    pieces[5] += m;

    let m = (pieces[2] / 2).min(pieces[1]);
    pieces[2] -= 2 * m;
    pieces[1] -= m;
    pieces[5] += m;

    let m = pieces[2].min(pieces[1] / 3);
    pieces[2] -= m;
    pieces[1] -= 3 * m;
    pieces[5] += m;

    let m = pieces[1].min(pieces[1] / 5);
    pieces[1] -= 5 * m;
    pieces[5] += m;

    let m = pieces[3].min(pieces[1]);
    pieces[3] -= m;
    pieces[1] -= m;
    pieces[4] += m;

    let m = pieces[2] / 2;
    pieces[2] -= 2 * m;
    pieces[4] += m;

    let m = pieces[2].min(pieces[1] / 2);
    pieces[2] -= m;
    pieces[1] -= 2 * m;
    pieces[4] += m;

    let m = pieces[1] / 4;
    pieces[1] -= 4 * m;
    pieces[4] += m;

    let m = pieces[2].min(pieces[1]);
    pieces[2] -= m;
    pieces[1] -= m;
    pieces[3] += m;

    let m = pieces[1] / 3;
    pieces[1] -= 3 * m;
    pieces[3] += m;

    let m = pieces[1] / 2;
    pieces[1] -= 2 * m;
    pieces[2] += m;

    for i in 1..=4 {
        pieces[5] += pieces[i];
        pieces[i] = 0;
    }

    return pieces[5];
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let pieces = f(input.value(), input.value(), input.value(), input.value());
    let ans = group_pieces(pieces);
    writeln!(output, "{:?}", ans).unwrap();
}
