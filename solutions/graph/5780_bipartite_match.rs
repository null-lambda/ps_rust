use std::io::Write;

mod fast_io {
    use std::fs::File;
    use std::io::BufWriter;
    use std::os::unix::io::FromRawFd;

    extern "C" {
        fn mmap(addr: usize, length: usize, prot: i32, flags: i32, fd: i32, offset: i64)
            -> *mut u8;
        fn fstat(fd: i32, stat: *mut usize) -> i32;
    }

    pub fn stdin() -> &'static str {
        let mut stat = [0; 18];
        unsafe { fstat(0, (&mut stat).as_mut_ptr()) };
        let buffer = unsafe { mmap(0, stat[6], 1, 2, 0, 0) };
        unsafe { std::str::from_utf8_unchecked(std::slice::from_raw_parts(buffer, stat[6])) }
    }

    pub fn stdout() -> BufWriter<File> {
        let stdout = unsafe { File::from_raw_fd(1) };
        BufWriter::new(stdout)
    }
}

use std::io::{BufRead, BufReader, BufWriter, Write};

use std::collections::btree_map::Entry;
use std::collections::BTreeMap;
use std::mem;

// compressed trie
#[derive(Clone, Debug)]
struct Trie<T> {
    children: BTreeMap<T, (Vec<T>, Trie<T>)>,
    is_terminal: bool,
}

impl<T: Eq + Ord + Clone> Trie<T> {
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
    let mut line = line_buf.split_ascii_whitespace();
    let n_colors = line.next().unwrap().parse().unwrap();
    let n_teams = line.next().unwrap().parse().unwrap();

    let mut trie_color = Trie::new();
    let mut trie_team_rev = Trie::new();
    (0..n_colors).for_each(|_| trie_color.insert(input.next().unwrap().unwrap().bytes()));
    (0..n_teams).for_each(|_| trie_team_rev.insert(input.next().unwrap().unwrap().bytes().rev()));

    // println!("{:#?}", trie_color);
    // println!("{:#?}", trie_team_rev);

    let n_queries = input.next().unwrap().unwrap().parse().unwrap();
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
