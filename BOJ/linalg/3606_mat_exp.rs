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

fn mul_mat_cyclic(a: &[u64], b: &[u64], n: usize, m: u64) -> Vec<u64> {
    assert!(a.len() == n && b.len() == n);
    let mut c = vec![0u64; n];
    let b_ext: Vec<_> = b.iter().chain(b).copied().collect();

    for j in 0..n {
        for i in 0..n {
            c[i] += a[j] * b_ext[j..][i];
        }
    }
    for i in 0..n {
        c[i] %= m;
    }

    c
}

fn apply_pow_mat_symm(xs: &[u64], a: &[u64], n: usize, mut exp: u32, m: u64) -> Vec<u64> {
    let mut a = a.to_vec();
    let mut res = xs.to_vec();
    while exp > 0 {
        if exp & 1 == 1 {
            res = mul_mat_cyclic(&a, &res, n, m);
        }
        a = mul_mat_cyclic(&a, &a, n, m);
        exp >>= 1;
    }
    res
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: u64 = input.value();
    let d: usize = input.value();
    let k: u32 = input.value();
    let init: Vec<u64> = (0..n).map(|_| input.value()).collect();

    let mut base = vec![0u64; n];
    {
        let i = 0;
        for j in 0..n {
            let d_linear = (i as isize - j as isize).abs() as usize;
            let d_circ = d_linear.min(n - d_linear);
            base[i * n + j] = (d_circ <= d) as u64 % m;
        }
    }

    let ans = apply_pow_mat_symm(&init, &base, n, k, m);
    for i in 0..n {
        write!(output, "{} ", ans[i]).unwrap();
    }
}
