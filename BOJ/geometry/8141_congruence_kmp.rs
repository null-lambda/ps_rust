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

pub mod debug {
    pub fn with(#[allow(unused_variables)] f: impl FnOnce()) {
        #[cfg(debug_assertions)]
        f()
    }
}

fn encode<'a>(
    ps: &'a [[i32; 2]],
) -> impl 'a + Iterator<Item = impl PartialEq + Copy + std::fmt::Debug> {
    let n = ps.len();
    (0..n).map(move |i| {
        let trunc = |i| if i >= n { i - n } else { i };
        let ext = |p: [_; 2]| p.map(|x| x as i64);
        let sub = |p: [_; 2], q: [_; 2]| [p[0] - q[0], p[1] - q[1]];
        let dot = |p: [_; 2], q: [_; 2]| p[0] * q[0] + p[1] * q[1];
        let cross = |p: [_; 2], q: [_; 2]| p[0] * q[1] - p[1] * q[0];

        let p = ps[i];
        let q = ps[trunc(i + 1)];
        let r = ps[trunc(i + 2)];

        let s = ext(sub(q, p));
        let t = ext(sub(q, r));
        (dot(s, s), (dot(s, t), cross(s, t)))
    })
}

fn kmp<'a: 'c, 'b: 'c, 'c, T: PartialEq>(
    s: impl IntoIterator<Item = T> + 'a,
    pattern: &'b [T],
) -> impl Iterator<Item = usize> + 'c {
    // Build a jump table
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

    // Search patterns
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

    for _ in 0..input.value() {
        let n: usize = input.value();
        let mut ps: Vec<[i32; 2]> = (0..n)
            .map(|_| std::array::from_fn(|_| input.value()))
            .collect();

        let pattern: Vec<_> = encode(&ps).collect();
        ps.reverse();
        ps.iter_mut().for_each(|p| p[0] *= -1);

        let s = encode(&ps).collect::<Vec<_>>();

        let s = s.iter().chain(&s).take(2 * n - 1).copied();

        let ans = kmp(s, &pattern).count();
        writeln!(output, "{}", ans).unwrap();
    }
}
