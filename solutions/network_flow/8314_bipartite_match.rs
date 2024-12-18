use std::{cell::UnsafeCell, collections::VecDeque, io::Write};

use collections::Jagged;

mod simple_io {
    use std::string::*;

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

pub mod collections {
    use std::fmt::Debug;
    use std::iter;
    use std::mem::MaybeUninit;
    use std::ops::Index;

    // Compressed sparse row format for jagged array
    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
    pub struct Jagged<T> {
        data: Vec<T>,
        head: Vec<u32>,
    }

    impl<T> Debug for Jagged<T>
    where
        T: Debug,
    {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let v: Vec<Vec<&T>> = (0..self.len()).map(|i| self[i].iter().collect()).collect();
            v.fmt(f)
        }
    }

    impl<T, I> FromIterator<I> for Jagged<T>
    where
        I: IntoIterator<Item = T>,
    {
        fn from_iter<J>(iter: J) -> Self
        where
            J: IntoIterator<Item = I>,
        {
            let mut data = vec![];
            let mut head = vec![];
            head.push(0);

            let mut cnt = 0;
            for row in iter {
                data.extend(row.into_iter().inspect(|_| cnt += 1));
                head.push(cnt);
            }
            Jagged { data, head }
        }
    }

    impl<T: Clone> Jagged<T> {
        pub fn from_assoc_list(n: usize, pairs: &[(u32, T)]) -> Self {
            let mut head = vec![0u32; n + 1];

            for &(u, _) in pairs {
                head[u as usize + 1] += 1;
            }
            for i in 2..n + 1 {
                head[i] += head[i - 1];
            }
            let mut data: Vec<_> = iter::repeat_with(|| MaybeUninit::uninit())
                .take(head[n] as usize)
                .collect();
            let mut pos = head.clone();

            for (u, v) in pairs {
                data[pos[*u as usize] as usize] = MaybeUninit::new(v.clone());
                pos[*u as usize] += 1;
            }

            let data = std::mem::ManuallyDrop::new(data);
            let data = unsafe {
                Vec::from_raw_parts(data.as_ptr() as *mut T, data.len(), data.capacity())
            };

            Jagged { data, head }
        }
    }

    impl<T> Jagged<T> {
        pub fn len(&self) -> usize {
            self.head.len() - 1
        }
    }

    impl<T> Index<usize> for Jagged<T> {
        type Output = [T];
        fn index(&self, index: usize) -> &[T] {
            let start = self.head[index] as usize;
            let end = self.head[index + 1] as usize;
            &self.data[start..end]
        }
    }

    impl<T> Jagged<T> {
        pub fn iter(&self) -> Iter<T> {
            Iter { src: self, pos: 0 }
        }
    }

    impl<'a, T> IntoIterator for &'a Jagged<T> {
        type Item = &'a [T];
        type IntoIter = Iter<'a, T>;
        fn into_iter(self) -> Self::IntoIter {
            self.iter()
        }
    }

    pub struct Iter<'a, T> {
        src: &'a Jagged<T>,
        pos: usize,
    }

    impl<'a, T> Iterator for Iter<'a, T> {
        type Item = &'a [T];
        fn next(&mut self) -> Option<Self::Item> {
            if self.pos < self.src.len() {
                let item = &self.src[self.pos];
                self.pos += 1;
                Some(item)
            } else {
                None
            }
        }
    }
}

// Y combinator for recursive closures
trait FnLike<A, B> {
    fn call(&self, x: A) -> B;
}

impl<A, B, F: Fn(A) -> B> FnLike<A, B> for F {
    fn call(&self, x: A) -> B {
        self(x)
    }
}

fn fix<A, B, F: Fn(&dyn FnLike<A, B>, A) -> B>(f: F) -> impl Fn(A) -> B {
    struct FixImpl<A, B, F: Fn(&dyn FnLike<A, B>, A) -> B>(F, std::marker::PhantomData<(A, B)>);

    impl<A, B, F: Fn(&dyn FnLike<A, B>, A) -> B> FnLike<A, B> for FixImpl<A, B, F> {
        fn call(&self, x: A) -> B {
            (self.0)(self, x)
        }
    }

    let fix = FixImpl(f, std::marker::PhantomData);
    move |x| fix.call(x)
}

const UNSET: u32 = u32::MAX;
fn bipartite_match(n: usize, m: usize, neighbors: &Jagged<u32>) -> [Vec<u32>; 2] {
    // Hopcroft-Karp
    const INF: u32 = u32::MAX / 2;

    let mut assignment = [vec![UNSET; n], vec![UNSET; m]];
    let mut left_level = vec![INF; n];
    let mut queue = VecDeque::new();
    loop {
        left_level.fill(INF);
        queue.clear();
        for u in 0..n {
            if assignment[0][u] == UNSET {
                queue.push_back(u as u32);
                left_level[u] = 0;
            }
        }

        while let Some(u) = queue.pop_front() {
            for &v in &neighbors[u as usize] {
                let w = assignment[1][v as usize];
                if w == UNSET || left_level[w as usize] != INF {
                    continue;
                }
                left_level[w as usize] = left_level[u as usize] + 1;
                queue.push_back(w);
            }
        }
        let mut found_augmenting_path = false;
        for u in 0..n {
            if assignment[0][u] != UNSET {
                continue;
            }

            let assignment_cell = UnsafeCell::new(assignment);
            {
                let dfs = fix(|dfs: &dyn FnLike<u32, bool>, u: u32| -> bool {
                    unsafe {
                        let assignment = &mut *assignment_cell.get();
                        for &v in &neighbors[u as usize] {
                            let w = assignment[1][v as usize];
                            if w == UNSET
                                || left_level[w as usize] == left_level[u as usize] + 1
                                    && dfs.call(w)
                            {
                                assignment[0][u as usize] = v;
                                assignment[1][v as usize] = u;
                                return true;
                            }
                        }
                    }
                    false
                });
                if dfs(u as u32) {
                    found_augmenting_path = true;
                }
            }
            assignment = assignment_cell.into_inner();
        }

        if !found_augmenting_path {
            break;
        }
    }

    assignment
}

fn partition_point<P>(mut left: u32, mut right: u32, mut pred: P) -> u32
where
    P: FnMut(u32) -> bool,
{
    while left < right {
        let mid = left + (right - left) / 2;
        if pred(mid) {
            left = mid + 1;
        } else {
            right = mid;
        }
    }
    left
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    for _ in 0..input.value() {
        let n: usize = input.value();
        let t_max = 100;
        let mut neighbors = vec![vec![]; t_max + 1];

        for _ in 0..input.value() {
            let t1: usize = input.value();
            let t2: usize = input.value();
            for _ in 0..input.value() {
                let q: u32 = input.value();
                for t in t1..t2 {
                    neighbors[t].push(q - 1);
                }
            }
        }

        let neighbors = Jagged::from_iter(neighbors);
        let test = |t_bound: usize| {
            let assignment = bipartite_match(t_bound, n, &neighbors);
            !assignment[1].iter().all(|&x| x != UNSET)
        };
        let ans = partition_point(1, t_max as u32 + 1, |t| test(t as usize));

        if ans != t_max as u32 + 1 {
            writeln!(output, "{}", ans).unwrap();
        } else {
            writeln!(output, "-1").unwrap();
        }
    }
}
