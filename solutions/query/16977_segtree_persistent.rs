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
            .map(|&c| {
                matches! {c, b'\n' | b'\r' | 0}
            })
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

use std::io::{BufReader, Read, Write};

fn stdin() -> Vec<u8> {
    let stdin = std::io::stdin();
    let mut reader = BufReader::new(stdin.lock());

    let mut input_buf: Vec<u8> = vec![];
    reader.read_to_end(&mut input_buf).unwrap();
    input_buf
}

pub mod segtree {
    use std::ops::Range;

    // monoid, not necesserily commutative
    pub trait Monoid {
        fn id() -> Self;
        fn op(self, rhs: Self) -> Self;
    }

    #[derive(Debug)]
    struct Node<T> {
        left: usize,
        right: usize,
        value: T,
    }

    #[derive(Debug)]
    pub struct PersistentSegTree<T> {
        bound: Range<usize>,
        nodes: Vec<Node<T>>,
        last_root: usize,
    }

    impl<T> PersistentSegTree<T>
    where
        T: Monoid + Copy + Eq,
    {
        pub fn new(bound: Range<usize>) -> Self {
            let mut tree = Self {
                bound,
                nodes: vec![],
                last_root: 0,
            };
            tree.build(tree.bound.clone());
            tree
        }

        fn build(&mut self, range: Range<usize>) -> usize {
            let node = if range.start + 1 == range.end {
                Node {
                    left: 0,
                    right: 0,
                    value: T::id(),
                }
            } else {
                let mid = (range.start + range.end) / 2;
                Node {
                    left: self.build(range.start..mid),
                    right: self.build(mid..range.end),
                    value: T::id(),
                }
            };
            self.nodes.push(node);
            self.nodes.len() - 1
        }

        // return: new root idx
        pub fn insert(&mut self, idx: usize, value: T) -> usize {
            self.last_root = self.insert_rec(idx, value, self.last_root, self.bound.clone());
            self.last_root
        }

        fn insert_rec(&mut self, idx: usize, value: T, u: usize, range: Range<usize>) -> usize {
            if idx < range.start || range.end <= idx {
                return u;
            }
            let node = if range.start + 1 == range.end {
                Node {
                    left: 0,
                    right: 0,
                    value,
                }
            } else {
                let mid = (range.start + range.end) / 2;
                let left = self.insert_rec(idx, value, self.nodes[u].left, range.start..mid);
                let right = self.insert_rec(idx, value, self.nodes[u].right, mid..range.end);
                Node {
                    left,
                    right,
                    value: self.nodes[left].value.op(self.nodes[right].value),
                }
            };
            self.nodes.push(node);
            self.nodes.len() - 1
        }

        // sum on interval [left, right)
        pub fn query(&self, root_idx: usize, bound: Range<usize>) -> T {
            self.query_rec(root_idx, self.bound.clone(), bound)
        }

        fn query_rec(&self, u: usize, range: Range<usize>, bound: Range<usize>) -> T {
            if bound.end <= range.start || range.end <= bound.start {
                T::id()
            } else if bound.start <= range.start && range.end <= bound.end {
                self.nodes[u].value
            } else {
                let mid = (range.start + range.end) / 2;
                self.query_rec(self.nodes[u].left, range.start..mid, bound.clone())
                    .op(self.query_rec(self.nodes[u].right, mid..range.end, bound.clone()))
            }
        }
    }
}

use segtree::{Monoid, PersistentSegTree};

#[derive(Copy, Clone, PartialEq, Eq, Debug)]
struct MaximalConsecutiveLen {
    sum: u32,
    full: u32,
    left: u32,
    right: u32,
}

impl Monoid for MaximalConsecutiveLen {
    fn id() -> Self {
        MaximalConsecutiveLen {
            sum: 0,
            full: 1,
            left: 0,
            right: 0,
        }
    }
    fn op(self, other: Self) -> Self {
        MaximalConsecutiveLen {
            sum: self.sum.max(other.sum).max(self.right + other.left),
            full: self.full + other.full,
            left: if self.sum == self.full {
                self.full + other.left
            } else {
                self.left
            },
            right: if other.sum == other.full {
                self.right + other.full
            } else {
                other.right
            },
        }
    }
}

fn main() {
    use io::InputStream;
    let input_buf = stdin();
    let mut input: &[u8] = &input_buf[..];

    let mut output_buf = Vec::<u8>::new();

    use std::cmp::Reverse;
    let n: usize = input.value();
    let mut heights: Vec<(u32, u32)> = (0..n as u32).map(|i| (input.value(), i)).collect();
    heights.sort_unstable_by_key(|&(x, _idx)| Reverse(x));

    let mut segtree = PersistentSegTree::new(0..n);
    let mut roots = vec![];

    let mut it = heights.iter().peekable();
    while let Some(&(h, idx)) = it.next() {
        let root = segtree.insert(
            idx as usize,
            MaximalConsecutiveLen {
                sum: 1,
                full: 1,
                left: 1,
                right: 1,
            },
        );

        if !matches!(it.peek(), Some(&&(h_next, _)) if h == h_next) {
            roots.push((root, h));
        }
    }

    let n_queries: usize = input.value();
    /*
    let queries: Vec<(u32, u32, u32, u32)> = (0..n_queries as u32)
        .map(|i| (input.value(), input.value(), input.value(), i))
        .collect();
    */
    for _ in 0..n_queries {
        let left: usize = input.value();
        let right: usize = input.value();
        let width: u32 = input.value();

        let i = roots.partition_point(|&(root, _h)| {
            // println!("{}: {:?}", _h, segtree.query(root, left - 1..right).sum);
            segtree.query(root, left - 1..right).sum < width
        });
        writeln!(output_buf, "{:?}", roots[i].1).unwrap();
    }

    std::io::stdout().write_all(&output_buf[..]).unwrap();
}
