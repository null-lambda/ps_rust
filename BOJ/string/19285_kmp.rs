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
// find (d, x, y) satisfying d = gcd(a, b) and a * x + b * y = d
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

fn crt(a1: u64, m1: u64, a2: u64, m2: u64) -> Option<(u64, u64)> {
    let (d, x, _y) = egcd(m1, m2);
    let m = m1 / d * m2;
    let da = ((a2 as i64 - a1 as i64) % m as i64 + m as i64) as u64 % m;
    if da % d != 0 {
        return None;
    }
    let mut x = ((x % m as i64) + m as i64) as u64 % m;
    x = (da / d % m) * x % m;
    let a = (a1 + m1 * x) % m;

    Some((a, m))
}

fn kmp<'c, 'a: 'c, 'b: 'c>(
    s: impl IntoIterator<Item = u8> + 'a,
    pattern: &'b [u8],
) -> impl Iterator<Item = usize> + 'c {
    // build jump function
    let mut jump_table = vec![0];
    let mut i_prev = 0;
    for i in 1..pattern.len() {
        while i_prev > 0 && pattern[i] != pattern[i_prev] {
            i_prev = jump_table[i_prev - 1];
        }
        if pattern[i] == pattern[i_prev] {
            i_prev += 1;
        }
        jump_table.push(i_prev);
    }

    // search patterns
    let mut j = 0;
    s.into_iter().enumerate().filter_map(move |(i, c)| {
        while j == pattern.len() || j > 0 && pattern[j] != c {
            j = jump_table[j - 1];
        }
        if pattern[j] == c {
            j += 1;
        }
        (j == pattern.len()).then(|| i + 1 - pattern.len())
    })
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let s = input.token().as_bytes();
    let t = input.token().as_bytes();
    let n = s.len();
    assert_eq!(t.len(), n);

    let perm: Vec<_> = (0..n as u32)
        .step_by(2)
        .chain((1..n as u32).step_by(2))
        .collect();

    let mut visited = vec![false; n];
    let mut cycles = vec![];
    for mut i in 0..n {
        if visited[i] {
            continue;
        }

        let mut cycle = vec![];
        loop {
            cycle.push(i);
            visited[i] = true;
            i = perm[i] as usize;
            if visited[i] {
                break;
            }
        }
        cycles.push(cycle);
    }

    let mut constraints = vec![];
    for cycle in cycles {
        let m = cycle.len();
        let src = (cycle.iter().map(|&u| s[u])).chain(cycle[..m - 1].iter().map(|&u| s[u]));
        let pattern = cycle.iter().map(|&i| t[i]).collect::<Vec<_>>();

        let xs = kmp(src, &pattern).map(|x| x as u64).collect::<Vec<_>>();
        if xs.is_empty() {
            writeln!(output, "-1").unwrap();
            return;
        };
        constraints.push((xs, m as u64));
    }
    constraints.sort_unstable_by_key(|(_, m)| *m);
    let ans = constraints
        .into_iter()
        .reduce(|(a1, m1), (a2, m2)| {
            let m = m1 / gcd(m1, m2) * m2;
            let mut valid = vec![0; m as usize];
            for i in 0..(m / m1) as usize {
                for &a in a1.iter() {
                    valid[a as usize + i * m1 as usize] += 1;
                }
            }
            for i in 0..(m / m2) as usize {
                for &a in a2.iter() {
                    valid[a as usize + i * m2 as usize] += 1;
                }
            }
            let a = valid
                .iter()
                .enumerate()
                .filter_map(|(i, &c)| (c == 2).then(|| i as u64))
                .collect();
            (a, m)
        })
        .unwrap();
    let ans = ans.0.first();

    if let Some(a) = ans {
        writeln!(output, "{}", a).unwrap();
    } else {
        writeln!(output, "-1").unwrap();
    }
}
