use std::io::Write;

use linked_list::MultiList;

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

mod linked_list {
    use std::{
        marker::PhantomData,
        num::NonZeroU32,
        ops::{Index, IndexMut},
    };

    #[derive(Debug)]
    pub struct Cursor<T> {
        idx: NonZeroU32,
        _marker: PhantomData<*const T>,
    }

    // Arena-based pool of doubly linked lists
    #[derive(Clone, Debug)]
    pub struct MultiList<T> {
        links: Vec<[Option<Cursor<T>>; 2]>,
        values: Vec<T>,
    }

    impl<T> Clone for Cursor<T> {
        fn clone(&self) -> Self {
            Self::new(self.idx.get() as usize)
        }
    }

    impl<T> Copy for Cursor<T> {}

    impl<T> PartialEq for Cursor<T> {
        fn eq(&self, other: &Self) -> bool {
            self.idx == other.idx
        }
    }

    impl<T> Eq for Cursor<T> {}

    impl<T> Cursor<T> {
        fn new(idx: usize) -> Self {
            Self {
                idx: NonZeroU32::new(idx as u32).unwrap(),
                _marker: PhantomData,
            }
        }

        fn usize(&self) -> usize {
            self.idx.get() as usize
        }
    }

    impl<T> Index<Cursor<T>> for MultiList<T> {
        type Output = T;
        fn index(&self, index: Cursor<T>) -> &Self::Output {
            &self.values[index.usize()]
        }
    }

    impl<T> IndexMut<Cursor<T>> for MultiList<T> {
        fn index_mut(&mut self, index: Cursor<T>) -> &mut Self::Output {
            &mut self.values[index.usize()]
        }
    }

    impl<T: Default> MultiList<T> {
        pub fn new() -> Self {
            Self {
                links: vec![[None; 2]],
                values: vec![Default::default()],
            }
        }

        pub fn next(&self, i: Cursor<T>) -> Option<Cursor<T>> {
            self.links[i.usize()][1]
        }

        pub fn prev(&self, i: Cursor<T>) -> Option<Cursor<T>> {
            self.links[i.usize()][0]
        }

        pub fn first(&self, mut i: Cursor<T>) -> Cursor<T> {
            while let Some(i_next) = self.prev(i) {
                i = i_next;
            }
            i
        }

        pub fn singleton(&mut self, value: T) -> Cursor<T> {
            let idx = self.links.len();
            self.links.push([None; 2]);
            self.values.push(value);
            Cursor::new(idx)
        }

        fn link(&mut self, u: Cursor<T>, v: Cursor<T>) {
            self.links[u.usize()][1] = Some(v);
            self.links[v.usize()][0] = Some(u);
        }

        pub fn insert_left(&mut self, i: Cursor<T>, value: T) -> Cursor<T> {
            let v = self.singleton(value);
            if let Some(j) = self.prev(i) {
                self.link(j, v);
            }
            self.link(v, i);
            v
        }

        pub fn erase_left(&mut self, i: Cursor<T>) -> bool {
            if let Some(j) = self.prev(i) {
                if let Some(k) = self.prev(j) {
                    self.links[j.usize()] = [None; 2];
                    self.link(k, i);
                } else {
                    self.links[j.usize()][1] = None;
                    self.links[i.usize()][0] = None;
                }
                return true;
            }
            false
        }

        pub fn swap(&mut self, i: Cursor<T>, j: Cursor<T>) {
            self.values.swap(i.usize(), j.usize());
        }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let mut ll = MultiList::<u8>::new();
    let mut cursor = ll.singleton(b'$');
    let tail = cursor;
    for b in input.token().bytes().rev() {
        cursor = ll.insert_left(cursor, b);
    }
    cursor = tail;

    for _ in 0..input.value() {
        match input.token() {
            "L" => {
                if let Some(next) = ll.prev(cursor) {
                    cursor = next;
                }
            }
            "D" => {
                if let Some(next) = ll.next(cursor) {
                    cursor = next;
                }
            }
            "B" => {
                ll.erase_left(cursor);
            }
            "P" => {
                let v = ll.insert_left(cursor, input.token().as_bytes()[0]);
                cursor = ll.next(v).unwrap();
            }
            _ => panic!(),
        }
    }

    let mut cursor = Some(ll.first(cursor));
    let mut res = vec![];
    while let Some(c) = cursor {
        res.push(ll[c]);
        cursor = ll.next(c);
    }

    write!(output, "{}", unsafe {
        std::str::from_utf8_unchecked(&res[..res.len() - 1])
    })
    .ok();
}
