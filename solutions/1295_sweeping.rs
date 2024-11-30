use std::{
    collections::{BTreeMap, HashMap},
    io::Write,
};

use collections::DisjointSet;

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

mod collections {
    use std::cell::Cell;

    pub struct DisjointSet {
        // represents parent if >= 0, size if < 0
        parent_or_size: Vec<Cell<i32>>,
    }

    impl DisjointSet {
        pub fn empty() -> Self {
            Self {
                parent_or_size: vec![],
            }
        }

        pub fn insert_node(&mut self) -> usize {
            let idx = self.parent_or_size.len();
            self.parent_or_size.push(Cell::new(-1));
            idx
        }

        pub fn get_size(&self, u: usize) -> u32 {
            debug_assert!(u < self.parent_or_size.len());
            -self.parent_or_size[self.find_root(u)].get() as u32
        }

        pub fn find_root(&self, u: usize) -> usize {
            debug_assert!(u < self.parent_or_size.len());
            if self.parent_or_size[u].get() < 0 {
                u
            } else {
                let root = self.find_root(self.parent_or_size[u].get() as usize);
                self.parent_or_size[u].set(root as i32);
                root
            }
        }
        // returns whether two set were different
        pub fn merge(&mut self, mut u: usize, mut v: usize) -> bool {
            debug_assert!(u < self.parent_or_size.len());
            debug_assert!(v < self.parent_or_size.len());
            u = self.find_root(u);
            v = self.find_root(v);
            if u == v {
                return false;
            }
            let size_u = -self.parent_or_size[u].get() as i32;
            let size_v = -self.parent_or_size[v].get() as i32;
            if size_u < size_v {
                std::mem::swap(&mut u, &mut v);
            }
            self.parent_or_size[v].set(u as i32);
            self.parent_or_size[u].set(-(size_u + size_v));
            true
        }
    }
}

#[derive(Debug, Clone)]
struct Strip {
    component: usize,
    y0: u32,
}

impl Strip {
    fn new(dset: &mut DisjointSet, y0: u32) -> Self {
        Self {
            component: dset.insert_node(),
            y0,
        }
    }
}

#[derive(Debug)]
enum EventType {
    Enter,
    Exit,
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let mut events = vec![];
    for _ in 0..n {
        let x1 = input.value::<u32>() + 1;
        let y1 = input.value::<u32>() + 1;
        let x2 = input.value::<u32>() + 1;
        let y2 = input.value::<u32>() + 1;

        debug_assert!(x1 < x2);
        debug_assert!(y1 > y2);
        events.push((y2, EventType::Enter, x1, x2));
        events.push((y1, EventType::Exit, x1, x2));
    }
    events.sort_unstable_by_key(|&(y, ..)| y);

    let mut active = BTreeMap::<_, Strip>::new();
    let mut dset = DisjointSet::empty();
    let mut area_acc = HashMap::<usize, u64>::new();

    let x_max = 1_000_000_001;
    let external_strip = Strip::new(&mut dset, 0);
    active.insert(0, external_strip.clone());
    active.insert(x_max + 1, external_strip.clone());

    for (y, event_type, x1, x2) in events {
        debug_assert!(!active.is_empty());
        let px_left = *active.range(..x1).next_back().unwrap().0;
        let xs: Vec<u32> = active.range(x1..=x2).map(|(&x, _)| x).collect();
        let px_right = *active.range(x2 + 1..).next().unwrap().0;

        let mut update_area = |strip: &Strip, dx: u32, y: u32| {
            let dy = y - strip.y0;
            *area_acc.entry(strip.component).or_default() += dx as u64 * dy as u64;
        };

        match event_type {
            EventType::Enter => {
                let m = xs.len();
                if m == 0 {
                    let p = active.get_mut(&px_right).unwrap();
                    update_area(p, px_right - px_left, y);
                    p.y0 = y;

                    let p_split = p.clone();
                    active.insert(x1, p_split);
                    active.insert(x2, Strip::new(&mut dset, y));
                } else {
                    let mut p_left = active.remove(&xs[0]).unwrap();
                    update_area(&p_left, xs[0] - px_left, y);
                    p_left.y0 = y;
                    active.insert(x1, p_left);

                    for (i, &x) in xs.iter().chain(&Some(x2)).enumerate() {
                        let old = active.insert(x, Strip::new(&mut dset, y));
                        if 0 < i && i < m {
                            update_area(&old.unwrap(), x - xs[i - 1], y);
                        }
                    }

                    let p_right = active.get_mut(&px_right).unwrap();
                    update_area(p_right, px_right - xs[m - 1], y);
                    p_right.y0 = y;
                }
            }
            EventType::Exit => {
                debug_assert!(xs.len() >= 2 && xs[0] == x1 && *xs.last().unwrap() == x2);
                let m = xs.len();
                if m == 2 {
                    let p_left = active.remove(&x1).unwrap();
                    update_area(&p_left, x1 - px_left, y);

                    let mid = active.remove(&x2).unwrap();
                    update_area(&mid, x2 - x1, y);

                    let p_right = active.get_mut(&px_right).unwrap();
                    update_area(p_right, px_right - x2, y);
                    p_right.y0 = y;

                    dset.merge(p_left.component, p_right.component);
                } else {
                    for i in 1..m {
                        let old = if 1 < i && i < m - 1 {
                            active.insert(xs[i], Strip::new(&mut dset, y))
                        } else {
                            active.remove(&xs[i])
                        }
                        .unwrap();
                        update_area(&old, xs[i] - xs[i - 1], y);
                    }

                    let mut p_left = active.remove(&x1).unwrap();
                    update_area(&p_left, x1 - px_left, y);
                    p_left.y0 = y;
                    active.insert(xs[1], p_left);

                    let p_right = active.get_mut(&px_right).unwrap();
                    update_area(p_right, px_right - x2, y);
                    p_right.y0 = y;
                }
            }
        }
    }

    assert!(dset.find_root(external_strip.component) == external_strip.component);
    area_acc.remove(&external_strip.component);
    let mut area_merged = HashMap::<usize, u64>::new();
    for (component, area) in area_acc {
        *area_merged.entry(dset.find_root(component)).or_default() += area;
    }

    let area_count = area_merged.len();
    let max_area = *area_merged.values().max().unwrap();
    writeln!(output, "{} {}", area_count, max_area).unwrap();
}
