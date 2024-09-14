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

fn write_num(w: &mut Vec<u8>, c: u32) {
    if c / 10 > 0 {
        write_num(w, c / 10);
    }
    w.push((c % 10) as u8 + b'0');
}

extern "C" {
    fn mmap(addr: usize, length: usize, prot: i32, flags: i32, fd: i32, offset: i64) -> *mut u8;
    fn fstat(fd: i32, stat: *mut usize) -> i32;
    fn write(fd: i32, buf: *const u8, count: usize) -> usize;
}

fn stdin() -> &'static str {
    let mut stat = [0; 18];
    unsafe { fstat(0, (&mut stat).as_mut_ptr()) };
    let buffer = unsafe { mmap(0, stat[6], 1, 2, 0, 0) };
    unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(buffer, stat[6])) }
}

fn main() {
    use io::*;
    use std::io::{BufReader, BufWriter, Read, Write};

    /*
    let stdin = std::io::stdin();
    let stdin = stdin.lock();
    let stdout = std::io::stdout();
    let stdout = stdout.lock();
    */
    /*
    let mut reader = BufReader::with_capacity(5000000, stdin);
    let mut input_buf: Vec<u8> = vec![];
    reader.read_to_end(&mut input_buf).unwrap();
    let mut input: &[u8] = &input_buf;
    */
    let mut input = stdin().as_bytes();

    let stdout = std::io::stdout();
    let stdout = stdout.lock();
    // let mut writer = BufWriter::with_capacity(5000000,stdout);
    let mut output_buf: Vec<u8> = vec![];

    // main algorithm
    let t: usize = input.value();
    for _ in 0..t {
        let (a, b): (u32, u32) = input.tuple();
        write_num(&mut output_buf, a + b);
        output_buf.push(b'\n');
    }
    unsafe { write(1, &output_buf[0], output_buf.len()) };
    //writer.write(&output_buf).unwrap();

    //let _ = reader.into_inner().into_raw_fd();
    //let _ = writer.into_inner().unwrap().into_raw_fd();
}
