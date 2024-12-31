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
            .map(|&c| matches!(c, b'\n' | b'\r' | 0))
            .unwrap_or_else(|| false)
        {
            s = &s[..s.len() - 1];
        }
        s
    }

    impl InputStream for &[u8] {
        fn token(&mut self) -> &[u8] {
            let i = self.iter().position(|&c| !is_whitespace(c)).unwrap();
            //.expect("no available tokens left");
            *self = &self[i..];
            let i = self
                .iter()
                .position(|&c| is_whitespace(c))
                .unwrap_or_else(|| self.len());
            let (token, buf_new) = self.split_at(i);
            *self = buf_new;
            token
        }

        fn line(&mut self) -> &[u8] {
            let i = self
                .iter()
                .position(|&c| c == b'\n')
                .map(|i| i + 1)
                .unwrap_or_else(|| self.len());
            let (line, buf_new) = self.split_at(i);
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

const P: u64 = 1_000_000_007;

fn id_matrix(n: usize) -> Vec<Vec<u64>> {
    (0..n)
        .map(|i| (0..n).map(|j| if i == j { 1 } else { 0 }).collect())
        .collect()
}

fn clone_matrix(n: usize, a: &Vec<Vec<u64>>, out: &mut Vec<Vec<u64>>) {
    for i in 0..n {
        for j in 0..n {
            out[i][j] = a[i][j];
        }
    }
}

fn mul_matrix(n: usize, a: &Vec<Vec<u64>>, b: &Vec<Vec<u64>>, out: &mut Vec<Vec<u64>>) {
    for i in 0..n {
        for j in 0..n {
            out[i][j] = (0..n).map(|l| (a[i][l] * b[l][j]) % P).sum::<u64>() % P;
        }
    }
}

fn pow_matrix(n: usize, a: &Vec<Vec<u64>>, mut exponent: u64, out: &mut Vec<Vec<u64>>) {
    *out = id_matrix(n);
    let mut a: Vec<Vec<_>> = a.clone();
    let mut temp: Vec<Vec<_>> = a.clone();

    while exponent > 0 {
        if exponent % 2 == 0 {
            mul_matrix(n, &a, &a, &mut temp);
            clone_matrix(n, &temp, &mut a);
            exponent /= 2;
        } else {
            mul_matrix(n, &a, out, &mut temp);
            clone_matrix(n, &temp, out);
            exponent -= 1;
        }
    }
}

fn main() {
    use io::*;

    let input_buf = stdin();
    let mut input: &[u8] = &input_buf;

    let mut output_buf = Vec::<u8>::new();

    let period: u32 = input.value();
    let n_verts: usize = input.value();
    let elapsed: u64 = input.value();

    let mut adj_matrices: Vec<Vec<Vec<u64>>> = (0..period)
        .map(|_| (0..n_verts).map(|_| vec![0; n_verts]).collect())
        .collect();
    for t in 0..period as usize {
        let m = input.value();
        for _ in 0..m {
            let u = input.value::<usize>() - 1;
            let v = input.value::<usize>() - 1;
            let count: u64 = input.value();
            adj_matrices[t][u][v] = count;
        }
    }

    let (n_periods, t_remains) = (elapsed / period as u64, (elapsed % period as u64) as usize);
    let mat_remains = adj_matrices[0..t_remains]
        .iter()
        .fold(id_matrix(n_verts), |acc, mat| {
            let mut buffer = id_matrix(n_verts);
            mul_matrix(n_verts, &acc, &mat, &mut buffer);
            buffer
        });
    let mat_period = adj_matrices[t_remains..].iter().fold(
        mat_remains.iter().map(|row| row.clone()).collect(),
        |acc, mat| {
            let mut buffer = id_matrix(n_verts);
            mul_matrix(n_verts, &acc, &mat, &mut buffer);
            buffer
        },
    );

    let mut mat_pow = id_matrix(n_verts);
    pow_matrix(n_verts, &mat_period, n_periods, &mut mat_pow);

    let mut result = id_matrix(n_verts);
    mul_matrix(n_verts, &mat_pow, &mat_remains, &mut result);

    for i in 0..n_verts {
        for j in 0..n_verts {
            write!(output_buf, "{} ", result[i][j]).unwrap();
        }
        writeln!(output_buf).unwrap();
    }
    /*
    for a in 1..10 {
        println!("{} {:?}", a, solve(a) - solve(a - 1));
    }
    */

    std::io::stdout().write(&output_buf[..]).unwrap();
}
