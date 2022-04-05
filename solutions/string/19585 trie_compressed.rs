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

use std::io::{BufRead, BufReader, BufWriter, Write};

use std::collections::hash_map::Entry;
use std::collections::HashMap;
use std::hash::Hash;
use std::mem;

// compressed trie
#[derive(Clone, Debug)]
struct Trie<T> {
    children: HashMap<T, (Vec<T>, Trie<T>)>,
    is_terminal: bool,
}

impl<T: Ord + Clone + Hash> Trie<T> {
    fn new() -> Self {
        Self {
            children: Default::default(),
            is_terminal: false,
        }
    }

    fn insert<I: IntoIterator<Item = T>>(&mut self, path: I) {
        let mut path = path.into_iter().peekable();
        let mut current = self;
        while let Some(c) = path.next() {
            match current.children.entry(c) {
                Entry::Occupied(entry) => {
                    let (label, next) = entry.into_mut();
                    for (i, d) in label.into_iter().enumerate() {
                        if matches!(path.peek(), Some(c) if c == d) {
                            path.next();
                        } else {
                            let label_bottom = label[i + 1..].to_vec();
                            label.truncate(i + 1);
                            let d = label.pop().unwrap();
                            debug_assert!(label.len() == i);

                            let next_bottom = mem::replace(next, Trie::new());
                            next.children.insert(d, (label_bottom, next_bottom));

                            if let Some(c) = path.next() {
                                let (_, next) = next
                                    .children
                                    .entry(c)
                                    .or_insert((path.collect(), Trie::new()));
                                next.is_terminal = true;
                            } else {
                                next.is_terminal = true;
                            }
                            return;
                        }
                    }
                    current = next;
                }
                Entry::Vacant(entry) => {
                    let (_, next) = entry.insert((path.collect(), Trie::new()));
                    next.is_terminal = true;
                    return;
                }
            }
        }
        current.is_terminal = true;
    }

    fn contains_prefix<I: IntoIterator<Item = T>>(&self, path: I, result: &mut [bool])
    where
        T: std::fmt::Debug,
    {
        let mut current = self;
        let mut path = path.into_iter();
        let mut i = 0;
        while let Some((label, next)) = path.next().and_then(|c| current.children.get(&c)) {
            for d in label {
                if !matches!(path.next(), Some(c) if &c == d) {
                    return;
                }
            }
            i += label.len() + 1;
            current = next;
            result[i - 1] = current.is_terminal;
        }
    }
}

fn main() {
    use io::*;

    let stdin = std::io::stdin();
    let reader = BufReader::with_capacity(10000, stdin.lock());
    let mut input = reader.lines();

    let stdout = std::io::stdout();
    let mut output = BufWriter::new(stdout.lock());

    let line_buf = input.next().unwrap().unwrap();
    let mut line = line_buf.as_bytes();
    let n_colors = line.value();
    let n_teams = line.value();

    let mut trie_color = Trie::new();
    let mut trie_team_rev = Trie::new();
    (0..n_colors).for_each(|_| trie_color.insert(input.next().unwrap().unwrap().bytes()));
    (0..n_teams).for_each(|_| trie_team_rev.insert(input.next().unwrap().unwrap().bytes().rev()));

    // println!("{:#?}", trie_color);
    // println!("{:#?}", trie_team_rev);

    let n_queries = input.next().unwrap().unwrap().as_bytes().value();
    let mut pred1 = vec![false; 2001];
    let mut pred2 = vec![false; 2001];
    for _ in 0..n_queries {
        let word = input.next().unwrap().unwrap().into_bytes();
        let n = word.len();
        pred1[..n - 1].fill(false);
        pred2[..n - 1].fill(false);
        trie_color.contains_prefix(word[..n - 1].iter().copied(), &mut pred1[..n - 1]);
        trie_team_rev.contains_prefix(word[1..].iter().rev().copied(), &mut pred2[..n - 1]);
        // println!("{:?}", (&word[..n - 1], &pred1[1..n]));
        // println!("{:?}", (&word[1..], &pred2[1..n]));

        let result = (0..n - 1).any(|i| pred1[i] && pred2[n - 2 - i]);
        writeln!(output, "{}", if result { "Yes" } else { "No" }).unwrap();
    }
}
