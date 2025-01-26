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

const N_BITS: usize = 20;

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let k: usize = input.value();
    let mut freq = vec![0u64; 1 << N_BITS];
    for _ in 0..n {
        let x: usize = input.value();
        freq[x] += 1;
    }

    let mut xor_sum = 0;
    {
        for mask in 0..1 << N_BITS {
            xor_sum += freq[mask] * freq[mask ^ k];
        }
        if k == 0 {
            xor_sum -= n as u64;
        }
        xor_sum /= 2;
    }

    let and_sum = {
        let mut freq_shifted = vec![0u64; 1 << N_BITS];
        for mask in 0..1 << N_BITS {
            if mask & k == k {
                freq_shifted[mask ^ k] += freq[mask];
            }
        }

        let mut sos = freq_shifted.clone();
        for i in 0..N_BITS {
            let bit = 1 << i;
            for mask in 0..1 << N_BITS {
                if mask & bit != 0 {
                    sos[mask] += sos[mask ^ bit];
                }
            }
        }

        let mut res = 0;
        for mask in 0..1 << N_BITS {
            res += sos[((1 << N_BITS) - 1) ^ mask] * freq_shifted[mask];
        }
        res -= freq[k];
        res /= 2;
        res
    };

    let or_sum = {
        let mut freq_shifted = vec![0u64; 1 << N_BITS];
        for mask in 0..1 << N_BITS {
            if mask | k == k {
                freq_shifted[mask ^ k] += freq[mask];
            }
        }

        let mut sos = freq_shifted.clone();
        for i in 0..N_BITS {
            let bit = 1 << i;
            for mask in 0..1 << N_BITS {
                if mask & bit != 0 {
                    sos[mask] += sos[mask ^ bit];
                }
            }
        }

        let mut res = 0;
        for mask in 0..1 << N_BITS {
            res += sos[((1 << N_BITS) - 1) ^ mask] * freq_shifted[mask];
        }
        res -= freq[k];
        res /= 2;
        res
    };

    writeln!(output, "{} {} {} ", and_sum, or_sum, xor_sum).unwrap();
}
