mod io {
    use std::fmt::debug;
    use std::str::*;

    pub trait inputstream {
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

    impl inputstream for &[u8] {
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

    pub trait readvalue<t> {
        fn value(&mut self) -> t;
    }

    impl<t: fromstr, i: inputstream> readvalue<t> for i
    where
        t::err: debug,
    {
        #[inline]
        fn value(&mut self) -> t {
            let token = self.token();
            let token = unsafe { from_utf8_unchecked(token) };
            token.parse::<t>().unwrap()
        }
    }
}

use std::io::{bufreader, read, write};

fn stdin() -> vec<u8> {
    let stdin = std::io::stdin();
    let mut reader = bufreader::new(stdin.lock());
    let mut input_buf: vec<u8> = vec![];
    reader.read_to_end(&mut input_buf).unwrap();
    input_buf
}

fn count_quad_nodes(output_buf: &mut vec<u8>, grid: &vec<vec<u8>>, width: usize) {
    fn dfs(
        output_buf: &mut vec<u8>,
        grid: &vec<vec<u8>>,
        width: usize,
        x: usize,
        y: usize,
    ) -> option<u8> {
        if width == 1 {
            output_buf.push(grid[y][x]);
            some(grid[y][x])
        } else {
            let hw = width / 2;
            let update = |acc, x| match (acc, x) {
                (some(_), _) if acc == x => acc,
                _ => none,
            };

            output_buf.push(b'(');
            let mut result = dfs(output_buf, grid, hw, x, y);
            result = update(result, dfs(output_buf, grid, hw, x + hw, y));
            result = update(result, dfs(output_buf, grid, hw, x, y + hw));
            result = update(result, dfs(output_buf, grid, hw, x + hw, y + hw));
            match result {
                Some(v) => {
                    output_buf.truncate(output_buf.len() - 5);
                    output_buf.push(v);
                }
                None => {
                    output_buf.push(b')');
                }
            }
            result
        }
    }

    dfs(output_buf, grid, width, 0, 0);
}

fn main() {
    use io::*;

    let input_buf = stdin();
    let mut input: &[u8] = &input_buf;

    let mut output_buf = Vec::<u8>::new();

    let n = input.value();
    input.skip_line();
    let grid: Vec<Vec<_>> = (0..n)
        .map(|_| input.line()[0..n].iter().map(|&x| x as u8).collect())
        .collect();

    count_quad_nodes(&mut output_buf, &grid, n);

    std::io::stdout().write(&output_buf[..]).unwrap();
}
