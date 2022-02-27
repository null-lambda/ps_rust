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

    // pub struct LineBufStream{

    // }

    // impl<'a> InputStream for Bytes<'a> {
    //     fn token(&mut self) -> &[u8] {
    //         self.skip_while(|&c| !is_whitespace(c)).take_while(|&c| is_whitespace(c)).collect();
    //         todo!()
    //     }
    //     fn line(&mut self) -> &[u8] {
    //         todo!()
    //     }
    // }

    pub trait ReadValue<T> {
        fn value(&mut self) -> T;
        fn vec(&mut self, n: usize) -> Vec<T> {
            (0..n).map(|_| self.value()).collect()
        }
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

    pub trait ReadTuple<T> {
        fn tuple(&mut self) -> T;
    }

    macro_rules! impl_tuple {
($($T:ident )+) => {
    impl<$($T,)+ I> ReadTuple<($($T,)+)> for I
    where
        I: $(ReadValue<$T> + )+ InputStream
    {
        #[inline]
        fn tuple(&mut self) -> ($($T,)+) {
            ($(<I as ReadValue<$T>>::value(self),)+)
        }
    }
};
}

    macro_rules! impl_tuples {
() => {};
($T1:ident $($T:ident)*) => {
    impl_tuples! {$($T )*}
    impl_tuple! {$T1 $($T )*}
};
}

    impl_tuples! {T0 T1 T2 T3 T4 T5 T6 T7 T8 T9 T10 T11}

    #[test]
    fn test_stream() {
        let source = " 2 3 4 5 \r\n 2 4 \n\n\n-19235 3\na나䫂defg -0.12e+5\n123\r\n";
        fn test_sub(mut input: impl InputStream) {
            assert_eq!(input.line(), " 2 3 4 5 ".as_bytes());
            let (n, k): (usize, usize) = input.tuple();
            let v: Vec<i32> = input.vec(2);
            assert_eq!((n, k), (2, 4));
            assert_eq!(v, [-19235, 3]);
            let s: String = input.value();
            assert_eq!(s, "a나䫂defg");
            assert_eq!((|| -> f64 { input.value() })(), -0.12e+5);
            assert_eq!(input.line(), "".as_bytes());
            assert_eq!(input.line(), "123".as_bytes());
        }
        test_sub(source.as_bytes());
    }
}

fn main() {
    use io::*;

    // requires owning_ref crates for abstraction
    let input_buf = {
        use std::io::Read;
        let mut v: Vec<u8> = vec![];
        let stdin = std::io::stdin();
        let mut stdin = std::io::BufReader::new(stdin.lock());
        stdin.read_to_end(&mut v).unwrap();
        v
    };
    let mut input: &[u8] = &input_buf;

    // main algorithm
    let n = input.value();
    let s: Vec<i32> = input.vec(n * n);
    let idx = |row: usize, col: usize| row * n + col;
    let split_idx = |idx: usize| (idx / n, idx % n);

    let a: Vec<i32> = (0..n * n)
        .map(|idx| split_idx(idx))
        .map(|(i, j)| s[idx(i, j)] + s[idx(j, i)])
        .collect();
    let a_row: Vec<i32> = (0..n).map(|i| (0..n).map(|j| a[idx(i, j)]).sum()).collect();
    let a_total: i32 = a_row.iter().sum();
    let mut visited = vec![false; 20];

    struct DfsEnv {
        n: usize,
        a: Vec<i32>,
        a_row: Vec<i32>,
    };
    struct DfsState<F: FnMut(i32)> {
        visited: Vec<bool>,
        update: F,
    };

    fn dfs<F: FnMut(i32)>(
        i: isize,
        depth: usize,
        value: i32,
        env: &DfsEnv,
        state: &mut DfsState<F>,
    ) {
        if depth == env.n / 2 {
            (state.update)(value.abs());
            return;
        }

        for j in (i + 1) as usize..env.n {
            state.visited[j] = true;
            dfs(j as isize, depth + 1, value - env.a_row[j], env, state);
            state.visited[j] = false;
        }
    }

    let mut d_min = 100_000_000;
    let mut update_min = |x| d_min = d_min.min(x);
    let env = DfsEnv { n, a, a_row };
    let mut state = DfsState {
        visited,
        update: |x| {
            d_min = d_min.min(x);
        },
    };
    dfs(-1, 0, a_total / 2, &env, &mut state);
    println!("{}", d_min);
}
