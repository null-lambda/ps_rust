mod io {
    use std::fmt::Debug;
    use std::str::*;

    pub trait InputStream {
        fn token(&mut self) -> &[u8];
        fn line(&mut self) -> &[u8];

        fn skip_line(&mut self) {
            self.line();
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
            .map(|&c| match c {
                b'\n' | b'\r' | 0 => true,
                _ => false,
            })
            .unwrap_or_else(|| false)
        {
            s = &s[..s.len() - 1];
        }
        s
    }

    impl InputStream for &[u8] {
        fn token(&mut self) -> &[u8] {
            let idx = self
                .iter()
                .position(|&c| !is_whitespace(c))
                .expect("no available tokens left");
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

    pub trait ReadValue<T> {
        fn value(&mut self) -> T;
    }

    impl<T: FromStr, I: InputStream> ReadValue<T> for I
    where
        T::Err: Debug,
    {
        #[inline]
        fn value(&mut self) -> T {
            let token = self.token();
            let token = unsafe { from_utf8_unchecked(token) };
            token.parse::<T>().unwrap()
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

fn count_quad_nodes(grid: &Vec<Vec<u32>>, width: usize) -> (u32, u32) {
    // Returns (count of quad nodes, value)
    fn dfs(grid: &Vec<Vec<u32>>, width: usize, x: usize, y: usize) -> (u32, u32) {
        if width == 1 {
            match grid[x][y] {
                0 => (1, 0),
                _ => (0, 1),
            }
        } else {
            let hw = width / 2;
            let update = |(a0, a1), (x0, x1)| (a0 + x0, a1 + x1);
            let mut result = dfs(grid, hw, x, y);
            result = update(result, dfs(grid, hw, x + hw, y));
            result = update(result, dfs(grid, hw, x, y + hw));
            result = update(result, dfs(grid, hw, x + hw, y + hw));
            match result {
                (_, 0) => (1, 0),
                (0, _) => (0, 1),
                x => x
            }
        }
    }

    dfs(grid, width, 0, 0)
}

fn main() {
    use io::*;

    let input_buf = stdin();
    let mut input: &[u8] = &input_buf;

    let mut output_buf = Vec::<u8>::new();

    let n = input.value();
    let grid: Vec<Vec<_>> = (0..n)
        .map(|_| (0..n).map(|_| input.value()).collect())
        .collect();

    let result = count_quad_nodes(&grid, n);
    writeln!(output_buf, "{}\n{}", result.0, result.1).unwrap();

    std::io::stdout().write(&output_buf[..]).unwrap();
}
