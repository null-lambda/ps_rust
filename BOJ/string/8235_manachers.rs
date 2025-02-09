use std::io::Write;

use heap::RemovableHeap;

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

pub mod heap {
    use std::collections::BinaryHeap;

    #[derive(Clone)]
    pub struct RemovableHeap<T> {
        items: BinaryHeap<T>,
        to_remove: BinaryHeap<T>,
    }

    impl<T: Ord> RemovableHeap<T> {
        pub fn new() -> Self {
            Self {
                items: BinaryHeap::new().into(),
                to_remove: BinaryHeap::new().into(),
            }
        }

        pub fn push(&mut self, item: T) {
            self.items.push(item);
        }

        pub fn remove(&mut self, item: T) {
            self.to_remove.push(item);
        }

        fn clean_top(&mut self) {
            while let Some((r, x)) = self.to_remove.peek().zip(self.items.peek()) {
                if r != x {
                    break;
                }
                self.to_remove.pop();
                self.items.pop();
            }
        }

        pub fn peek(&mut self) -> Option<&T> {
            self.clean_top();
            self.items.peek()
        }

        pub fn pop(&mut self) -> Option<T> {
            self.clean_top();
            self.items.pop()
        }
    }
}

// Manacher's algorithm
fn palindrome_radius<T: Eq>(s: &[T]) -> Vec<usize> {
    let n = s.len();
    let mut i = 0;
    let mut radius = 0;
    let mut rs = vec![];
    while i < n {
        while i >= (radius + 1)
            && i + (radius + 1) < n
            && s[i - (radius + 1)] == s[i + (radius + 1)]
        {
            radius += 1;
        }
        rs.push(radius);

        let mut mirrored_center = i;
        let mut max_mirrored_radius = radius;
        i += 1;
        radius = 0;
        while max_mirrored_radius > 0 {
            mirrored_center -= 1;
            max_mirrored_radius -= 1;
            if rs[mirrored_center] == max_mirrored_radius {
                radius = max_mirrored_radius;
                break;
            }
            rs.push(rs[mirrored_center].min(max_mirrored_radius));
            i += 1;
        }
    }
    rs
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let s = input.token().as_bytes();

    let mut t = vec![];
    let sep = 0;
    t.push(sep);
    let (lower, rest) = s.split_at(n / 2);
    let (_, upper) = rest.split_at(n % 2);
    for (&b1, &b2) in lower.iter().zip(upper.iter().rev()) {
        t.push(b1);
        t.push(sep);
        t.push(b2);
        t.push(sep);
    }

    let r_max = palindrome_radius(&t);

    let mut r_max_from_start = vec![0; t.len() + 2];
    let mut events = vec![vec![]; t.len() + 2];
    for i in (0..t.len()).step_by(2) {
        let r = r_max[i];
        events[i - r].push((i, true));
        events[i].push((i, false));
    }

    let mut active = RemovableHeap::new();
    for i in 0..t.len() {
        for &(c, insert) in &events[i] {
            if insert {
                active.push(c);
            } else {
                active.remove(c);
            }
        }
        if let Some(c) = active.peek() {
            r_max_from_start[i] = c - i;
        }
    }

    let mut ans = 0;
    for i in (0..t.len()).step_by(2) {
        let r = r_max[i];
        let start = i - r;
        if start != 0 {
            continue;
        }
        let end = r + i;

        debug_assert!(r % 2 == 0);
        debug_assert!(r_max_from_start[end] % 2 == 0);
        ans = ans.max(r / 2 + r_max_from_start[end] / 2);
    }
    debug_assert!(ans <= n / 2);
    writeln!(output, "{}", ans).unwrap();
}
