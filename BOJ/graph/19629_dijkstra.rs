use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap},
    io::Write,
};

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

fn truncate_prefix<'a>(s: &'a [u8], prefix: &[u8]) -> Option<&'a [u8]> {
    (s.len() >= prefix.len() && &s[..prefix.len()] == prefix).then(|| &s[prefix.len()..])
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let _m: usize = input.value();
    let words: Vec<&[u8]> = (0..n).map(|_| input.token().as_bytes()).collect();

    type State<'a> = (&'a [u8], u32);
    let mut dist: HashMap<State, u32> = Default::default();
    let mut pq: BinaryHeap<(Reverse<u32>, State)> = Default::default();
    for (i, s) in words.iter().enumerate() {
        let d = s.len() as u32;
        dist.entry((s, i as u32))
            .and_modify(|e| *e = (*e).min(d))
            .or_insert(d);
        pq.push((Reverse(d), (s, i as u32)));
    }

    let unset = u32::MAX;
    while let Some((Reverse(d), (s, i))) = pq.pop() {
        if s == b"" {
            writeln!(output, "{}", d).unwrap();
            return;
        }
        if d > *dist.get(&(s, i)).unwrap() {
            continue;
        }
        for j in 0..n {
            if i == j as u32 {
                continue;
            }
            let w = words[j];
            if let Some(t) = truncate_prefix(s, w) {
                let t = (t, unset);
                if dist.get(&t).map_or(true, |&d_old| d_old > d) {
                    dist.insert(t, d);
                    pq.push((Reverse(d), t));
                }
            }
            if let Some(t) = truncate_prefix(w, s) {
                let d_new = d + t.len() as u32;
                let t = (t, unset);
                if dist.get(&t).map_or(true, |&d_old| d_old > d_new) {
                    dist.insert(t, d_new);
                    pq.push((Reverse(d_new), t));
                }
            }
        }
    }
    writeln!(output, "-1").unwrap();
}
