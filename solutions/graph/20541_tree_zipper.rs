use std::collections::VecDeque;
use std::collections::{btree_map::Entry, BTreeMap, BTreeSet};
use std::io::Write;

#[allow(dead_code)]
mod simple_io {
    pub struct InputAtOnce(std::str::SplitAsciiWhitespace<'static>);

    impl InputAtOnce {
        pub fn token(&mut self) -> &str {
            self.0.next().unwrap_or_default()
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().unwrap()
        }
    }

    pub fn stdin_at_once() -> InputAtOnce {
        let buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let buf = Box::leak(buf.into_boxed_str());
        InputAtOnce(buf.split_ascii_whitespace())
    }

    pub fn stdout_buf() -> std::io::BufWriter<std::io::StdoutLock<'static>> {
        std::io::BufWriter::new(std::io::stdout().lock())
    }
}

struct Queueueue<T> {
    hq: VecDeque<T>,
    vq: VecDeque<T>,
}

impl<T: Clone> Queueueue<T> {
    fn new() -> Self {
        Self {
            hq: Default::default(),
            vq: Default::default(),
        }
    }

    fn is_empty(&self) -> bool {
        self.hq.is_empty() && self.vq.is_empty()
    }

    fn update_vq(&mut self) {
        let nv = self.vq.len() / 2;
        let nh = self.hq.len() / 2;
        self.vq[nv] = self.hq[nh].clone();
    }

    fn update_hq(&mut self) {
        let nv = self.vq.len() / 2;
        let nh = self.hq.len() / 2;
        self.hq[nh] = self.vq[nv].clone();
    }

    fn hpush(&mut self, x: T) {
        self.hq.push_back(x);
        if self.vq.len() >= 1 && self.hq.len() % 2 == 0 {
            let nv = self.vq.len() / 2;
            let nh = self.hq.len() / 2;
            self.update_vq();
        }
    }

    fn vpush(&mut self, x: T) {
        self.vq.push_back(x);
        if self.hq.len() >= 1 && self.vq.len() % 2 == 0 {
            let nv = self.vq.len() / 2;
            let nh = self.hq.len() / 2;
            self.update_hq();
        }
    }
}

fn main() {
    let buf = std::io::read_to_string(std::io::stdin()).unwrap();
    let mut lines = buf.lines();
    let mut output = simple_io::stdout_buf();

    let mut q = Queueueue::<&str>::new();

    let n = lines.next().unwrap().parse::<usize>().unwrap();
    for _ in 0..n {
        let mut tokens = lines.next().unwrap().split_ascii_whitespace();
        let (cmd, arg) = (tokens.next().unwrap(), || tokens.next().unwrap());

        match cmd {
            "empty" => {
                writeln!(output, "{}", q.is_empty() as u8).unwrap();
            }
            _ => panic!(),
        }
    }
}
