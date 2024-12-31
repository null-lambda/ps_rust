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

fn signed_area(p: [f64; 2], q: [f64; 2], r: [f64; 2]) -> f64 {
    (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])
}

fn partition_point_u32<P>(mut left: u32, mut right: u32, mut pred: P) -> u32
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

fn partition_point_f64<P>(
    mut left: f64,
    mut right: f64,
    eps: f64,
    mut max_iter: u32,
    mut pred: P,
) -> f64
where
    P: FnMut(f64) -> bool,
{
    while right - left > eps && max_iter > 0 {
        let mid = left + (right - left) / 2.0;
        if pred(mid) {
            left = mid;
        } else {
            right = mid;
        }
        max_iter -= 1;
    }
    left
}

type BoundaryPoint = (u32, f64);

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let ps: Vec<[f64; 2]> = (0..n).map(|_| [input.value(), input.value()]).collect();

    let mut perimeter_prefix = vec![0.0; n + 1];
    for i in 1..n + 1 {
        let p = ps[i - 1];
        let q = ps[i % n];
        perimeter_prefix[i] = perimeter_prefix[i - 1] + (p[0] - q[0]).hypot(p[1] - q[1]);
    }
    let total_perimeter = perimeter_prefix[n];
    let perimeter_half = total_perimeter / 2.0;
    let eval_perimeter_prefix = |(j, alpha): BoundaryPoint| -> f64 {
        let s = perimeter_prefix[j as usize];
        let t = perimeter_prefix[j as usize + 1];
        s + (t - s) * alpha
    };
    let eval_boundary_point = |(j, alpha): BoundaryPoint| -> [f64; 2] {
        let p = ps[j as usize];
        let q = ps[(j as usize + 1) % n];
        [p[0] + alpha * (q[0] - p[0]), p[1] + alpha * (q[1] - p[1])]
    };
    let dual = |(j, alpha): BoundaryPoint| -> BoundaryPoint {
        let s = eval_perimeter_prefix((j, alpha)) + perimeter_half;
        let k = partition_point_u32(0, n as u32 - 1, |k| eval_perimeter_prefix((k, 1.0)) < s);
        let beta = partition_point_f64(0.0, 1.0, 1e-9, 100, |beta| {
            eval_perimeter_prefix((k, beta)) < s
        });

        (k, beta)
    };

    let eval_left_area = |(j, alpha): BoundaryPoint| -> f64 {
        let (k, beta) = dual((j, alpha));

        let p0 = eval_boundary_point((j, alpha));
        let p_end = eval_boundary_point((k, beta));
        let mut rest = ps[j as usize + 1..=k as usize]
            .iter()
            .cloned()
            .chain(Some(p_end))
            .peekable();

        let mut res = 0.0;
        let mut q0 = rest.next().unwrap();
        for q1 in rest {
            res += signed_area(p0, q0, q1);

            q0 = q1;
        }
        res
    };

    let total_area = {
        let mut res = 0.0;
        let p0 = ps[0];
        for i in 2..n {
            let p1 = ps[i - 1];
            let p2 = ps[i];
            res += signed_area(p0, p1, p2);
        }
        res
    };
    let area_half = total_area * 0.5;

    let b0 = (0, 0.0);
    let b1 = dual(b0);

    let base = eval_left_area(b0) - area_half;
    let (j, alpha) = if base.abs() < 1e-9 {
        b0
    } else if base > 0.0 {
        let j = partition_point_u32(0, b1.0, |j| eval_left_area((j, 1.0)) > area_half);
        let alpha = partition_point_f64(0.0, 1.0, 1e-9, 100, |alpha| {
            eval_left_area((j, alpha)) > area_half
        });
        (j, alpha)
    } else {
        let j = partition_point_u32(0, b1.0, |j| eval_left_area((j, 1.0)) < area_half);
        let alpha = partition_point_f64(0.0, 1.0, 1e-9, 100, |alpha| {
            eval_left_area((j, alpha)) < area_half
        });
        (j, alpha)
    };

    let (k, beta) = dual((j, alpha));
    writeln!(output, "YES").unwrap();
    writeln!(output, "{} {}", j + 1, alpha).unwrap();
    writeln!(output, "{} {}", k + 1, beta).unwrap();
}
