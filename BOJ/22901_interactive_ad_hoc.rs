use buffered_io::BufReadExt;

mod buffered_io {
    use std::io::{BufRead, BufReader, BufWriter, Stdin, Stdout};
    use std::str::FromStr;

    pub trait BufReadExt: BufRead {
        fn line(&mut self) -> String {
            let mut buf = String::new();
            self.read_line(&mut buf).unwrap();
            buf
        }

        fn skip_line(&mut self) {
            self.line();
        }

        fn token(&mut self) -> String {
            loop {
                let buf = self.fill_buf().unwrap();
                if buf.is_empty() {
                    return String::new();
                }

                let mut i = 0;
                while i < buf.len() && buf[i].is_ascii_whitespace() {
                    i += 1;
                }

                let should_break = i < buf.len();
                self.consume(i);
                if should_break {
                    break;
                }
            }

            let mut res = vec![];
            loop {
                let buf = self.fill_buf().unwrap();
                if buf.is_empty() {
                    break;
                }

                let mut i = 0;
                while i < buf.len() && !buf[i].is_ascii_whitespace() {
                    i += 1;
                }
                res.extend_from_slice(&buf[..i]);

                let should_break = i < buf.len();
                self.consume(i);
                if should_break {
                    break;
                }
            }

            String::from_utf8(res).unwrap()
        }

        fn try_value<T: FromStr>(&mut self) -> Option<T> {
            self.token().parse().ok()
        }

        fn value<T: FromStr>(&mut self) -> T {
            self.try_value().unwrap()
        }
    }

    impl<R: BufRead> BufReadExt for R {}

    pub fn stdin() -> BufReader<Stdin> {
        BufReader::new(std::io::stdin())
    }

    pub fn stdout() -> BufWriter<Stdout> {
        BufWriter::new(std::io::stdout())
    }
}

mod emulated {
    pub struct Interactor {
        y: u32,
        rem_to_error: i32,
        rem_query_calls: u32,
    }

    impl Interactor {
        pub fn new(target: u32, rem_to_error: u32) -> Self {
            Self {
                y: target,
                rem_to_error: rem_to_error as i32 + 1,
                rem_query_calls: 13,
            }
        }

        pub fn query(&mut self, y: u32) -> u32 {
            if self.rem_query_calls == 0 {
                panic!()
            }
            self.rem_to_error -= 1;
            self.rem_query_calls -= 1;
            if self.rem_to_error == 0 {
                1
            } else {
                (self.y >= y) as u32
            }
        }

        pub fn terminate(self, x: u32) {
            assert!(x == self.y);
        }
    }
}

mod stdio {
    pub struct Interactor<'a, R, W> {
        input: &'a mut R,
        output: &'a mut W,
    }

    impl<'a, R: super::buffered_io::BufReadExt, W: std::io::Write> Interactor<'a, R, W> {
        pub fn new(input: &'a mut R, output: &'a mut W) -> Self {
            Self { input, output }
        }

        pub fn query(&mut self, y: u32) -> u32 {
            writeln!(self.output, "? {y}").unwrap();
            self.output.flush().unwrap();

            self.input.value()
        }

        pub fn terminate(self, x: u32) {
            writeln!(self.output, "! {x}").unwrap();
            self.output.flush().unwrap();
        }
    }
}

#[test]
fn brute() {
    for x in 2100..2400 {
        for rem in 0..14 {
            let mut itr = emulated::Interactor::new(x, rem);

            // If query result is 0, proceed left
            // If query result is 1, repeat one more time and proceed appropriately

            // Biasing factor: (1 - alpha)^k = alpha^2k
            let alpha = (618, 1000);

            let mut left = 2101;
            let mut right = 2400;
            let mut failed_once = false;
            while left < right {
                let mut mid = ((alpha.1 - alpha.0) * left + alpha.0 * right) / alpha.1;
                mid = mid.max(left).min(right - 1);

                let mut res = itr.query(mid);
                if !failed_once && res == 1 {
                    let res_alt = itr.query(mid);
                    failed_once = res != res_alt;
                    res = res_alt;
                }

                if res == 1 {
                    left = mid + 1;
                } else {
                    right = mid;
                }
            }
            itr.terminate(left - 1);
        }
    }
}

fn main() {
    let mut input = buffered_io::stdin();
    let mut output = buffered_io::stdout();

    let q: usize = input.value();

    for _ in 0..q {
        let mut itr = stdio::Interactor::new(&mut input, &mut output);

        // If query result is 0, proceed left
        // If query result is 1, repeat one more time and proceed appropriately

        // Biasing factor: (1 - alpha)^k = alpha^2k
        let alpha = (618, 1000);

        let mut left = 2101;
        let mut right = 2400;
        let mut failed_once = false;
        while left < right {
            let mut mid = ((alpha.1 - alpha.0) * left + alpha.0 * right) / alpha.1;
            mid = mid.max(left).min(right - 1);

            let mut res = itr.query(mid);
            if !failed_once && res == 1 {
                let res_alt = itr.query(mid);
                failed_once = res != res_alt;
                res = res_alt;
            }

            if res == 1 {
                left = mid + 1;
            } else {
                right = mid;
            }
        }
        itr.terminate(left - 1);
    }
}
