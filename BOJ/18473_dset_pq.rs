use std::{cell::RefCell, cmp::Reverse, collections::BinaryHeap, io::Write};

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

mod dset {
    use std::cell::Cell;
    use std::mem::{self};

    pub struct DisjointMap<T> {
        // Represents parent if >= 0, size if < 0
        parent_or_size: Vec<Cell<i32>>,
        values: Vec<T>,
    }

    impl<T: Default /* WIP */> DisjointMap<T> {
        pub fn new(values: impl IntoIterator<Item = T>) -> Self {
            let values: Vec<_> = values.into_iter().collect();
            let n = values.len();
            Self {
                parent_or_size: vec![Cell::new(-1); n],
                values,
            }
        }

        fn get_parent_or_size(&self, u: usize) -> Result<usize, u32> {
            let x = self.parent_or_size[u].get();
            if x >= 0 {
                Ok(x as usize)
            } else {
                Err((-x) as u32)
            }
        }

        fn set_parent(&self, u: usize, p: usize) {
            self.parent_or_size[u].set(p as i32);
        }

        fn set_size(&self, u: usize, s: u32) {
            self.parent_or_size[u].set(-(s as i32));
        }

        pub fn find_root_with_size(&self, u: usize) -> (usize, u32) {
            match self.get_parent_or_size(u) {
                Ok(p) => {
                    let (root, size) = self.find_root_with_size(p);
                    self.set_parent(u, root);
                    (root, size)
                }
                Err(size) => (u, size),
            }
        }

        pub fn find_root(&self, u: usize) -> usize {
            self.find_root_with_size(u).0
        }

        pub fn get_size(&self, u: usize) -> u32 {
            self.find_root_with_size(u).1
        }

        pub fn get(&self, u: usize) -> &T {
            let r = self.find_root(u);
            &self.values[r]
        }

        pub fn get_mut(&mut self, u: usize) -> &mut T {
            let r = self.find_root(u);
            &mut self.values[r]
        }

        // Returns true if two sets were previously disjoint
        pub fn merge(
            &mut self,
            u: usize,
            v: usize,
            mut combine_values: impl FnMut(&mut T, T),
        ) -> bool {
            let (mut u, size_u) = self.find_root_with_size(u);
            let (mut v, size_v) = self.find_root_with_size(v);
            if u == v {
                return false;
            }

            if size_u < size_v {
                mem::swap(&mut u, &mut v);
            }

            let value_v = std::mem::take(&mut self.values[v]);
            combine_values(&mut self.values[u], value_v);
            self.set_parent(v, u);
            self.set_size(u, size_u + size_v);
            true
        }
    }
}

type Criterion = (u32, u32, u64);

#[derive(Default, Debug)]
struct RootAgg {
    weight: u64,
    criterion_queue: BinaryHeap<(Reverse<u64>, u32)>,
}

impl RootAgg {
    fn merge_with(&mut self, mut filter_by: impl FnMut(u32) -> bool, mut other: Self) {
        if self.criterion_queue.len() < other.criterion_queue.len() {
            std::mem::swap(self, &mut other);
        }
        self.weight += other.weight;
        self.criterion_queue.extend(
            other
                .criterion_queue
                .into_iter()
                .filter(|&(_, j)| filter_by(j)),
        );
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let weights: Vec<u64> = (0..n).map(|_| input.value()).collect();
    let criterions: Vec<Criterion> = (0..m)
        .map(|_| {
            (
                input.value::<u32>() - 1,
                input.value::<u32>() - 1,
                input.value(),
            )
        })
        .collect();

    let mut criterion_queues = vec![BinaryHeap::new(); n];
    let mut satisfied = BinaryHeap::new();
    for (i, &(u, v, s)) in criterions.iter().enumerate() {
        let wu = weights[u as usize];
        let wv = weights[v as usize];
        if wu + wv >= s {
            satisfied.push(Reverse(i as u32));
        } else {
            let mid = (s - wu - wv).div_ceil(2);
            criterion_queues[u as usize].push((Reverse(wu + mid), i as u32));
            criterion_queues[v as usize].push((Reverse(wv + mid), i as u32));
        }
    }

    let mut dsu = dset::DisjointMap::new(weights.into_iter().zip(criterion_queues).map(
        |(weight, criterion_queue)| {
            RefCell::new(RootAgg {
                weight,
                criterion_queue,
            })
        },
    ));
    let mut fast_spanning_tree = vec![];
    while let Some(Reverse(i)) = satisfied.pop() {
        let (u, v, _) = criterions[i as usize];

        let mut agg_v = Default::default();
        if !dsu.merge(u as usize, v as usize, |_u, v| agg_v = v) {
            continue;
        }
        let agg_u = dsu.get(u as usize);
        agg_u.borrow_mut().merge_with(
            |j| {
                let (a, b, _) = criterions[j as usize];
                dsu.find_root(a as usize) != dsu.find_root(b as usize)
            },
            agg_v.into_inner(),
        );

        fast_spanning_tree.push(i);
        let wc = dsu.get(u as usize).borrow().weight;

        loop {
            let Some(&(Reverse(s_lower), j)) = agg_u.borrow().criterion_queue.peek() else {
                // Assure borrowed ref is early-dropped
                break;
            };

            if wc < s_lower {
                break;
            }

            let (a, b, s) = criterions[j as usize];

            agg_u.borrow_mut().criterion_queue.pop();
            if dsu.find_root(a as usize) == dsu.find_root(b as usize) {
                continue;
            }

            let wa = dsu.get(a as usize).borrow().weight;
            let wb = dsu.get(b as usize).borrow().weight;
            if wa + wb >= s {
                satisfied.push(Reverse(j));
            } else {
                let mid = (s - wa - wb).div_ceil(2);
                dsu.get(a as usize)
                    .borrow_mut()
                    .criterion_queue
                    .push((Reverse(wa + mid), j));
                dsu.get(b as usize)
                    .borrow_mut()
                    .criterion_queue
                    .push((Reverse(wb + mid), j));
            }
        }
    }

    writeln!(output, "{}", fast_spanning_tree.len()).unwrap();
    for i in fast_spanning_tree {
        write!(output, "{} ", i + 1).unwrap();
    }
    writeln!(output).unwrap();
}
