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

// Extended euclidean algorithm
// find (d, x, y) satisfying d = gcd(a, b) and a * x + b * y = d
fn egcd(a: u128, b: u128) -> (u128, i128, i128) {
    let (mut c, mut x, mut y) = if a > b {
        ((a, b), (1, 0), (0, 1))
    } else {
        ((b, a), (0, 1), (1, 0))
    };

    while c.1 > 0 {
        let q = c.0 / c.1;
        x = (x.1, (x.0 - (q as i128) * x.1));
        y = (y.1, (y.0 - (q as i128) * y.1));
        c = (c.1, c.0 % c.1);
    }
    (c.0, x.0, y.0)
}
fn crt(a1: u128, m1: u128, a2: u128, m2: u128) -> Option<(u128, u128)> {
    let (d, x, _y) = egcd(m1, m2);
    let m = m1 / d * m2;
    let da = ((a2 as i128 - a1 as i128) % m as i128 + m as i128) as u128 % m;
    if da % d != 0 {
        return None;
    }
    let mut x = ((x % m as i128) + m as i128) as u128 % m;
    x = (da / d % m) * x % m;
    let a = (a1 + m1 * x) % m;

    Some((a, m))
}

const INF: u32 = 1 << 28;

#[derive(Debug, Clone, Copy)]
struct AP {
    init: u128,
    period: u128,
}

impl AP {
    fn contains(&self, t: u128) -> bool {
        t >= self.init && (t - self.init) % self.period == 0
    }

    fn isect(&self, other: &Self) -> Option<Self> {
        if self.period == 0 {
            other.contains(self.init).then(|| *self)
        } else if other.period == 0 {
            self.contains(other.init).then(|| *other)
        } else {
            crt(
                (self.init % self.period) as u128,
                self.period as u128,
                (other.init % other.period) as u128,
                other.period as u128,
            )
            .map(|(mut init, period)| {
                init = init % period;
                let thres = self.init.max(other.init) as u128;
                if init < thres {
                    init += period * (thres - init).div_ceil(period);
                }

                // println!(
                //     "self {:?}, other {:?} -> {:?}",
                //     self,
                //     other,
                //     AP {
                //         init: init as u128,
                //         period: period as u128,
                //     }
                // );

                AP {
                    init: init as u128,
                    period: period as u128,
                }
            })
        }
    }
}

fn for_each_product(dims: Vec<usize>, mut visitor: impl FnMut(&[usize])) {
    let mut indices = vec![0; dims.len()];
    if dims.iter().all(|&d| d == 0) {
        return;
    }
    loop {
        visitor(&indices);
        let mut i = 0;
        while i < dims.len() {
            indices[i] += 1;
            if indices[i] == dims[i] {
                indices[i] = 0;
                i += 1;
            } else {
                break;
            }
        }
        if i == dims.len() {
            break;
        }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let k: usize = input.value();

    let n_pad = n + 2;
    let m_pad = m + 2;
    let n_cells = n_pad * m_pad;
    let trap = input.value::<usize>() * m_pad + input.value::<usize>();
    let mut seq = vec![];
    let mut ps = vec![];
    let mut delta_dir = vec![];
    for v in 0..k {
        let i = input.value::<usize>();
        let j = input.value::<usize>();
        let u = i * m_pad + j;
        let dir = match input.token() {
            "U" => 0,
            "R" => 1,
            "D" => 2,
            "L" => 3,
            _ => panic!(),
        };
        ps.push((u, dir));
        delta_dir.push(vec![!0; n_cells]);
        for i in 1..=n {
            let row = input.token().as_bytes();
            for j in 1..=m {
                delta_dir[v][i * m_pad + j] = (row[j - 1] - b'0') as i32;
            }
        }
    }

    for v in 0..k {
        let mut seq_options = vec![];
        for target_dir in 0..4 {
            let (mut u, mut dir) = ps[v];
            let mut entrance_time = vec![INF; n_cells * 4];
            let mut init = INF;
            let mut timer = 1;
            let mut period = INF;
            while (init == INF || period == INF) && timer < n_cells as u32 * 8 {
                if (u, dir) == (trap, target_dir) {
                    if init == INF {
                        init = timer;
                    } else {
                        period = timer - init;
                    }
                }
                entrance_time[u * 4 + dir] = timer;
                timer += 1;

                dir = (dir + delta_dir[v][u] as usize) % 4;
                let mut next = match dir {
                    0 => u - m_pad,
                    1 => u + 1,
                    2 => u + m_pad,
                    3 => u - 1,
                    _ => panic!(),
                };
                if delta_dir[v][next] == !0 {
                    dir = (dir + 2) % 4;
                    next = match dir {
                        0 => u - m_pad,
                        1 => u + 1,
                        2 => u + m_pad,
                        3 => u - 1,
                        _ => panic!(),
                    };
                }
                u = next;
            }

            if init != INF {
                seq_options.push(AP {
                    init: init as u128,
                    period: if timer >= n_cells as u32 * 8 {
                        0
                    } else {
                        period as u128
                    },
                });
            }
        }
        seq.push(seq_options);
    }

    let dims: Vec<usize> = seq.iter().map(|seq| seq.len()).collect();
    const INF_U128: u128 = 1 << 62;
    let mut ans = INF_U128;
    for_each_product(dims, |indices| {
        if let Some(ap) = indices
            .iter()
            .enumerate()
            .map(|(i, &j)| seq[i][j])
            // .inspect(|ap| println!("{:?}", ap))
            .fold(Some(AP { init: 0, period: 1 }), |acc, ap| {
                acc.and_then(|acc| acc.isect(&ap))
            })
        {
            ans = ans.min(ap.init);
        }
    });
    if ans == INF_U128 {
        writeln!(output, "-1").unwrap();
    } else {
        writeln!(output, "{}", ans).unwrap();
    }
    // println!("{:?}", seq);
}
