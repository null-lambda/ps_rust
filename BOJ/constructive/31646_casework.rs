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

fn construct(mut s: Vec<u8>, mut t: Vec<u8>, key: u8) -> Vec<&'static str> {
    let mut res = vec![];
    let mut storage_occupied = false;
    loop {
        match (s.len(), t.len()) {
            (0, 2) => {
                res.push("2 1");
                s.push(t.pop().unwrap());
                break;
            }
            (2, 0) => {
                res.push("1 2");
                t.push(s.pop().unwrap());
                break;
            }
            (1, 1) if s == t => {
                res.push("1 2");
                s.pop();
                break;
            }
            (0 | 1, 0 | 1) => {
                break;
            }
            (1, 2) if s == t => {
                res.push("2 1");
                t.pop();
                break;
            }
            (2, 1) if s == t => {
                res.push("1 2");
                s.pop();
                break;
            }
            _ => {}
        }

        if s.last() == Some(&key) {
            res.push("1 3");
            storage_occupied = true;
            s.pop();
        } else if t.last() == Some(&key) {
            res.push("2 3");
            storage_occupied = true;
            t.pop();
        } else if s.is_empty() {
            res.push("2 1");
            debug_assert!(t.len() >= 3);
            s.push(t.pop().unwrap());
        } else if t.is_empty() {
            res.push("1 2");
            debug_assert!(s.len() >= 3);
            t.push(s.pop().unwrap());
        } else if s.len() >= t.len() {
            s.pop();
            res.push("1 2");
        } else {
            t.pop();
            res.push("2 1");
        }
    }

    debug_assert!(s.len() <= 1 && t.len() <= 1);

    if storage_occupied {
        if s.first() == Some(&key) || s.is_empty() && t.first() != Some(&key) {
            res.push("3 1");
        } else if t.first() == Some(&key) || t.is_empty() && s.first() != Some(&key) {
            res.push("3 2");
        } else {
            panic!()
        }
    }

    res
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    for _ in 0..input.value() {
        let n: usize = input.value();
        let p = input.token();

        let mut s: Vec<u8> = input.token().as_bytes().to_vec();
        let mut t: Vec<u8> = input.token().as_bytes().to_vec();
        assert!(s.len() == n && t.len() == n);

        s.dedup();
        t.dedup();

        let r1 = construct(s.clone(), t.clone(), b'1');
        let r2 = construct(s.clone(), t.clone(), b'2');
        let ans = [r1, r2].into_iter().min_by_key(|v| v.len()).unwrap();

        writeln!(output, "{}", ans.len()).unwrap();
        if p != "1" {
            for v in ans {
                writeln!(output, "{}", v).unwrap();
            }
        }
    }
}
