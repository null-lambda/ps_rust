mod io {
    use std::fmt::Debug;
    use std::str::*;

    pub trait InputStream {
        fn token(&mut self) -> &[u8];
        fn line(&mut self) -> &[u8];

        fn skip_line(&mut self) {
            self.line();
        }

        #[inline]
        fn value<T>(&mut self) -> T
        where
            T: FromStr,
            T::Err: Debug,
        {
            let token = self.token();
            let token = unsafe { from_utf8_unchecked(token) };
            token.parse::<T>().unwrap()
        }
    }

    #[inline]
    fn is_whitespace(c: u8) -> bool {
        c <= b' '
    }

    fn trim_newline(s: &[u8]) -> &[u8] {
        let mut s = s;
        while s
            .last()
            .map(|&c| {
                matches! {c, b'\n' | b'\r' | 0}
            })
            .unwrap_or_else(|| false)
        {
            s = &s[..s.len() - 1];
        }
        s
    }

    impl InputStream for &[u8] {
        fn token(&mut self) -> &[u8] {
            let idx = self.iter().position(|&c| !is_whitespace(c)).unwrap();
            //.expect("no available tokens left");
            *self = &self[idx..];
            let idx = self
                .iter()
                .position(|&c| is_whitespace(c))
                .unwrap_or_else(|| self.len());
            let (token, buf_new) = self.split_at(idx);
            *self = buf_new;
            token
        }

        fn line(&mut self) -> &[u8] {
            let idx = self
                .iter()
                .position(|&c| c == b'\n')
                .map(|idx| idx + 1)
                .unwrap_or_else(|| self.len());
            let (line, buf_new) = self.split_at(idx);
            *self = buf_new;
            trim_newline(line)
        }
    }
}

use std::io::{BufReader, Read, Write};

fn stdin() -> Vec<u8> {
    let stdin = std::io::stdin();
    let mut reader = BufReader::new(stdin.lock());

    let mut input_buf: Vec<u8> = vec![];
    reader.read_to_end(&mut input_buf).unwrap();
    input_buf
}

#[derive(Clone, Copy, Debug)]
struct Vec2(f64, f64);

use std::ops::*;
impl Add<Self> for Vec2 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self::Output {
        Vec2(self.0 + rhs.0, self.1 + rhs.1)
    }
}

impl Mul<f64> for Vec2 {
    type Output = Vec2;
    fn mul(self, rhs: f64) -> Self::Output {
        Vec2(self.0 * rhs, self.1 * rhs)
    }
}

// runge-kutta method
fn rk4<S, F>(f: F, initial_value: S, [x0, x1]: [f64; 2], n_steps: usize) -> Vec<(f64, S)>
where
    S: Copy + Clone + Add<S, Output = S> + Mul<f64, Output = S>,
    F: Fn(&f64, &S) -> S,
{
    let h: f64 = (x1 - x0) / (n_steps as f64);
    let h_half: f64 = h * 0.5;
    let mut result = Vec::new();

    let mut x = x0;
    let mut y = initial_value;
    result.push((x, y));
    for _ in 0..n_steps {
        let k1 = f(&x, &y) * h;
        let k2 = f(&(x + h_half), &(y + k1 * 0.5)) * h;
        let k3 = f(&(x + h_half), &(y + k2 * 0.5)) * h;
        let k4 = f(&(x + h), &(y + k3)) * h;

        x = x + h;
        y = y + (k1 + (k2 + k3) * 2.0 + k4) * (1.0 / 6.0);
        result.push((x, y));
    }
    result
}

fn integrate_simpson([x0, x1]: [f64; 2], ys: &[f64]) -> f64 {
    let n = ys.len() - 1;
    assert!(n >= 3);
    let n_trunc = if n % 2 == 1 { n } else { n - 1 };

    let t1: f64 = ys[2..n_trunc - 2].iter().step_by(2).sum();
    let t2: f64 = ys[1..n_trunc - 1].iter().step_by(2).sum();
    let mut total: f64 = (t1 * 2.0 + t2 * 4.0 + ys[0] + ys[n_trunc - 1]) * (1.0 / 3.0);
    if n_trunc < n {
        // trapzoidal approximation at remains
        total += ys[n - 2] + ys[n - 1];
    }
    (x1 - x0) * total / (n as f64)
}

#[derive(Debug)]
enum BisectionError {
    OutOfBound,
    Timeout,
    None,
}

// find zero by bisection method
fn bisection<F>(
    f: F,
    [mut x0, mut x1]: [f64; 2],
    f_threshold: f64,
    f_bound: f64,
    max_iter: usize,
) -> Result<f64, BisectionError>
where
    F: Fn(&f64) -> f64,
{
    assert!(f_threshold > 0.0);
    if x0 > x1 {
        std::mem::swap(&mut x0, &mut x1);
        // panic!()
    }

    let (f_left, f_right) = (f(&x0), f(&x1));
    if f_left.signum() * f_right.signum() == 1.0 {
        return Err(BisectionError::None);
    }

    for _ in 0..max_iter {
        let x_mid = (x0 + x1) * 0.5;

        let f_mid = f(&x_mid);
        if f_mid.abs() < f_threshold {
            return Ok(x_mid);
        } else if f_mid.abs() > f_bound {
            return Err(BisectionError::OutOfBound);
        }
        if f_left.signum() == f_mid.signum() {
            x0 = x_mid;
        } else {
            x1 = x_mid;
        }
    }
    Err(BisectionError::Timeout)
}

fn minimal_radiation_1(a: f64, b: f64, c: f64) -> f64 {
    let [x0, x1] = [-10.0, 10.0];
    let [y0, y1] = [a - c, b - c];
    let n_steps = 50;

    let path = |&dydx0: &f64| {
        rk4(
            |&x, &Vec2(y, dydx)| {
                let r2 = x * x + y * y;
                let dydx_2 = 2.0 * (x * dydx - y) * (1.0 + dydx * dydx) / (r2 * (1.0 + r2));
                Vec2(dydx, dydx_2)
            },
            Vec2(y0, dydx0),
            [x0, x1],
            n_steps,
        )
    };

    let error_func = |&dydx0: &f64| {
        let path = path(&dydx0);
        let (_, Vec2(y1_est, _)) = path.last().unwrap();
        let y_error = y1_est - y1;
        y_error
    };

    // solve bvp with shooting method
    let mut dfs_stack = vec![[-20.0, y0 / x0 - 1e-5], [y0 / x0 + 1e-5, 20.0]];
    let mut dydx0_candidates = Vec::new();
    while let Some(dydx0_interval) = dfs_stack.pop() {
        match bisection(error_func, dydx0_interval, 1e-5, 1e+20, 100) {
            Ok(dydx0) if dydx0.is_finite() => {
                dydx0_candidates.push(dydx0);
            }
            Err(BisectionError::OutOfBound) => {
                let [a, b] = dydx0_interval;
                let mid = (a + b) * 0.5;
                dfs_stack.push([a, mid]);
                dfs_stack.push([mid, b]);
            }
            _ => {}
        }
    }
    dydx0_candidates.sort_unstable_by(|&p, &q| p.partial_cmp(&q).unwrap());

    // dbg!(dydx0_candidates.clone());
    dydx0_candidates
        .into_iter()
        .map(|dydx0| {
            let d_radiation: Vec<f64> = path(&dydx0)
                .into_iter()
                .map(|(x, Vec2(y, dydx))| {
                    (1.0 + dydx * dydx).sqrt() * (1.0 + 1.0 / (x * x + y * y))
                })
                .collect();
            let radiation = integrate_simpson([x0, x1], &d_radiation);

            // dbg!([[x0, y0], [x1, y1]], [dydx0, radiation]);
            radiation
        })
        .min_by(|&p, &q| p.partial_cmp(&q).unwrap())
        .unwrap()
}

fn minimal_radiation_2(a: f64, b: f64, c1: f64, c2: f64) -> f64 {
    let (c, d) = ((c1 + c2) * 0.5, (c1 - c2).abs() * 0.5);

    let [x0, x1] = [-10.0, 10.0];
    let [y0, y1] = [a - c, b - c];
    let n_steps = 50;

    let path = |&dydx0: &f64| {
        rk4(
            |&x, &Vec2(y, dydx)| {
                let (x2, d2, y2) = (x * x, d * d, y * y);
                let (x4, d4, y4) = (x2 * x2, d2 * d2, y2 * y2);
                let x2_d2 = x2 + d2;
                let r2 = x2 + (y - d) * (y - d);
                let s2 = x2 + (y + d) * (y + d);
                let mut numer = 4.0 * (1.0 + dydx * dydx);
                numer *= y * (3.0 * d4 + 2.0 * d2 * x2 - x4 - 2.0 * x2_d2 * y2 - y4)
                    + x * dydx * (x2_d2 * x2_d2 + 2.0 * y2 * (3.0 * d2 + x2) + y4);
                let denom = r2 * s2 * (y4 + 2.0 * y2 * (1.0 - d2 + x2) + x2_d2 * (x2_d2 + 2.0));
                let dydx_2 = numer / denom;
                Vec2(dydx, dydx_2)
            },
            Vec2(y0, dydx0),
            [x0, x1],
            n_steps,
        )
    };

    let error_func = |&dydx0: &f64| {
        let path = path(&dydx0);
        let (_, Vec2(y1_est, _)) = path.last().unwrap();
        let y_error = y1_est - y1;
        y_error
    };

    let dydx0_straight = [(y0 + d) / x0, (y0 - d) / x0];
    let mut shoots = vec![-40.0, 40.0];
    for center in dydx0_straight {
        let n_subdivisions = 5;
        let delta = 1.0;
        shoots.extend(
            (-n_subdivisions..n_subdivisions)
                .map(|i| i as f64 * (delta / n_subdivisions as f64))
                .map(|x| x + center),
        );
    }
    shoots.sort_unstable_by(|&p, &q| p.partial_cmp(&q).unwrap());
    shoots.dedup_by(|&mut p, &mut q| (p - q).abs() < 1e-6);
    let mut dfs_stack: Vec<[f64; 2]> = shoots.windows(2).map(|t| [t[0], t[1]]).collect();

    let mut dydx0_candidates = Vec::new();
    while let Some(dydx0_interval) = dfs_stack.pop() {
        //println!("{:?}", dydx0_interval);
        match bisection(error_func, dydx0_interval, 1e-7, 1e+25, 1000) {
            Ok(dydx0) if dydx0.is_finite() => {
                dydx0_candidates.push(dydx0);
            }
            Err(BisectionError::OutOfBound) => {
                let [a, b] = dydx0_interval;
                if (a - b).abs() < 1e-7 {
                    continue;
                }
                let mid = (a + b) * 0.5;
                dfs_stack.push([a, mid]);
                dfs_stack.push([mid, b]);
                // println!("oob, {:?}", dydx0_interval);
            }
            _ => {} // Err(BisectionError::None) => {}
                    // e => {println!("{:?} {:?}",dydx0_interval, e);}
        }
    }
    // dydx0_candidates.sort_unstable_by(|&p, &q| p.partial_cmp(&q).unwrap());
    // dbg!(dydx0_candidates.clone());

    dydx0_candidates
        .into_iter()
        .map(|dydx0| {
            let d_radiation: Vec<f64> = path(&dydx0)
                .into_iter()
                .map(|(x, Vec2(y, dydx))| {
                    let r2 = x * x + (y - d) * (y - d);
                    let s2 = x * x + (y + d) * (y + d);
                    (1.0 + dydx * dydx).sqrt() * (1.0 + (r2 + s2) / (r2 * s2))
                })
                .collect();
            let radiation = integrate_simpson([x0, x1], &d_radiation);

            // dbg!([[x0, y0], [x1, y1]], [dydx0, radiation]);
            radiation
        })
        .min_by(|&p, &q| p.partial_cmp(&q).unwrap())
        .unwrap()
}

#[test]
fn test_rk4() {
    use std::f64::consts::PI;
    let path = rk4(
        |&x, &Vec2(y, dydx)| Vec2(dydx, -y),
        Vec2(1.0, -0.5),
        [0.2, 0.2 + PI],
        1000,
    );
    let (_, Vec2(y1, dydx1)) = path.last().unwrap();
    assert!((y1 - (-1.0)).abs() < 1e-9);
    assert!((dydx1 - 0.5).abs() < 1e-9);
}

#[test]
fn test_simpson() {
    let n = 1000;
    let xs = (0..=n)
        .map(|i| i as f64 / n as f64)
        .map(|x| x * x)
        .collect::<Vec<_>>();
    let result = integrate_simpson([0.0, 1.0], &xs);
    assert!((result - 1.0 / 3.0).abs() < 1e-5);
}

#[test]
fn test_radiation_2() {
    for i in -10..=10 {
        for j in 0..=10 {
            for k in 0..10 {
                let a = i as f64 / 10.0 * 20.0;
                let b = a + j as f64 / 10.0 * 20.0;
                let d = k as f64 / 10.0 * 10.0;
                let result = minimal_radiation_2(a, b, d, -d);
                assert!(
                    result.abs() <= 35.0,
                    "{:?}", (a, b, d, result)
                );
            }
        }
    }
}

fn main() {
    use io::InputStream;
    let input_buf = stdin();
    let mut input: &[u8] = &input_buf[..];

    // let mut output_buf = Vec::<u8>::new();

    let test_cases = input.value();
    for i in 1..=test_cases {
        let n: usize = input.value();
        let a = input.value();
        let b = input.value();
        let result = match n {
            1 => {
                let c = input.value();
                minimal_radiation_1(a, b, c)
            }
            2 => {
                let c1 = input.value();
                let c2 = input.value();
                minimal_radiation_2(a, b, c1, c2)
            }
            _ => unreachable!(),
        };
        println!("Case #{}: {:.2}", i, result);
    }

    // std::io::stdout().write_all(&output_buf[..]).unwrap();
}
