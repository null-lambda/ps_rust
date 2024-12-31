use std::{
    cmp::Reverse,
    collections::{BTreeMap, HashSet},
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

// chunk_by in std >= 1.77.0
fn group_by<T, P, F>(xs: &[T], mut pred: P, mut f: F)
where
    P: FnMut(&T, &T) -> bool,
    F: FnMut(&[T]),
{
    let mut i = 0;
    while i < xs.len() {
        let mut j = i + 1;
        while j < xs.len() && pred(&xs[j - 1], &xs[j]) {
            j += 1;
        }
        f(&xs[i..j]);
        i = j;
    }
}

#[derive(Debug)]
enum Event {
    EnterLongSection,
    MoveToHalfSection,
    Exit,
}
use Event::*;

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    loop {
        let n: usize = input.value();
        let a: u32 = input.value::<u32>() * 2; // scale by 2
        let b: u32 = input.value::<u32>() * 2;
        if (n, a, b) == (0, 0, 0) {
            break;
        }

        let mut ps: Vec<(u32, u32)> = (0..n)
            .map(|_| (input.value::<u32>() * 2, input.value::<u32>() * 2))
            .collect();
        ps.push((0, 0));

        let test = |l: u32| {
            let mut events = vec![];
            let x_max = a - l * 2;
            let y_max = b - l * 2;

            for &(x, y) in &ps {
                events.push((x, y, EnterLongSection));
                if x >= l {
                    events.push((x - l, y, MoveToHalfSection));
                }
                if x >= 2 * l {
                    events.push((x - 2 * l, y, Exit));
                }
            }
            events.sort_unstable_by_key(|(x, ..)| Reverse(*x));

            let mut long_section = BTreeMap::<u32, u32>::new();
            let mut half_section = BTreeMap::<u32, u32>::new();

            let mut satisfiable = false;
            let mut query_points = HashSet::new();
            group_by(
                &events,
                |(x1, ..), (x2, ..)| x1 == x2,
                |group| {
                    if satisfiable {
                        return;
                    }

                    let x = group[0].0;

                    let mut removed_points = HashSet::new();
                    for (_, y, event) in group {
                        match event {
                            EnterLongSection => {}
                            MoveToHalfSection => {
                                *long_section.entry(*y).or_default() -= 1;
                                if long_section[&y] == 0 {
                                    long_section.remove(&y);
                                    removed_points.insert(*y);
                                }

                                *half_section.entry(*y).or_default() += 1;
                                query_points.insert(*y);
                            }
                            Exit => {
                                *half_section.entry(*y).or_default() -= 1;
                                if half_section[&y] == 0 {
                                    half_section.remove(&y);
                                    removed_points.insert(*y);
                                }
                            }
                        }
                    }

                    query_points.insert(0);
                    for y in removed_points {
                        query_points.extend(long_section.range(..y).next_back().map(|(&y, _)| y));
                        query_points.extend(half_section.range(..y).next_back().map(|(&y, _)| y));
                    }

                    if x <= x_max {
                        for y in query_points.drain() {
                            if y > y_max {
                                continue;
                            }
                            if long_section.range(y + 1..y + 2 * l).next().is_none()
                                && half_section.range(y + 1..y + l).next().is_none()
                            {
                                satisfiable = true;
                                return;
                            }
                        }
                    }

                    for (_, y, event) in group {
                        match event {
                            EnterLongSection => {
                                *long_section.entry(*y).or_default() += 1;
                                query_points.insert(*y);
                            }
                            MoveToHalfSection => {}
                            Exit => {}
                        }
                    }
                },
            );
            satisfiable
        };
        let l_max = partition_point(1, a.min(b) / 2 + 1, test) - 1;
        let area = (l_max as u64).pow(2) * 100 * 3 / 4;
        let area = area as f64 / 100.0;
        writeln!(output, "{:.2}", area).unwrap();
    }
}
