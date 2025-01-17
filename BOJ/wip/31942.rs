use std::{cmp::Reverse, io::Write};

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
pub mod dset {

    pub mod potential {
        pub trait Group: Clone {
            fn id() -> Self;
            fn add_assign(&mut self, b: &Self);
            fn sub_assign(&mut self, b: &Self);
        }

        #[derive(Clone, Copy)]
        struct Link(i32); // Represents parent if >= 0, size if < 0

        impl Link {
            fn node(p: u32) -> Self {
                Self(p as i32)
            }

            fn size(s: u32) -> Self {
                Self(-(s as i32))
            }

            fn get(&self) -> Result<u32, u32> {
                if self.0 >= 0 {
                    Ok(self.0 as u32)
                } else {
                    Err((-self.0) as u32)
                }
            }
        }

        pub struct DisjointSet<E> {
            links: Vec<(Link, E)>,
        }

        impl<E: Group> DisjointSet<E> {
            pub fn with_size(n: usize) -> Self {
                Self {
                    links: (0..n).map(|_| (Link::size(1), E::id())).collect(),
                }
            }

            pub fn find_root_with_size(&mut self, u: usize) -> (usize, E, u32) {
                let (l, w) = &self.links[u];
                match l.get() {
                    Ok(p) => {
                        let mut w_acc = w.clone();
                        let (root, w_to_root, size) = self.find_root_with_size(p as usize);
                        w_acc.add_assign(&w_to_root);
                        self.links[u] = (Link::node(root as u32), w_acc.clone());
                        (root, w_acc, size)
                    }
                    Err(size) => (u, w.clone(), size),
                }
            }

            pub fn find_root(&mut self, u: usize) -> usize {
                self.find_root_with_size(u).0
            }

            pub fn get_size(&mut self, u: usize) -> u32 {
                self.find_root_with_size(u).2
            }

            // Returns true if two sets were previously disjoint
            pub fn merge(&mut self, u: usize, v: usize, mut weight_uv: E) -> bool {
                let (mut u, mut weight_u, mut size_u) = self.find_root_with_size(u);
                let (mut v, mut weight_v, mut size_v) = self.find_root_with_size(v);
                if u == v {
                    return false;
                }

                if size_u < size_v {
                    std::mem::swap(&mut u, &mut v);
                    std::mem::swap(&mut weight_u, &mut weight_v);
                    std::mem::swap(&mut size_u, &mut size_v);

                    let mut neg = E::id();
                    neg.sub_assign(&weight_uv);
                    weight_uv = neg;
                }

                weight_u.add_assign(&weight_uv);
                weight_v.sub_assign(&weight_u);
                self.links[v] = (Link::node(u as u32), weight_v);
                self.links[u] = (Link::size(size_u + size_v), E::id());
                true
            }

            pub fn delta_potential(&mut self, u: usize, v: usize) -> Option<E> {
                let (u, weight_u, _) = self.find_root_with_size(u);
                let (v, weight_v, _) = self.find_root_with_size(v);
                (u == v).then(|| {
                    let mut delta = weight_u.clone();
                    delta.sub_assign(&weight_v);
                    delta
                })
            }
        }
    }
}

impl<T: std::ops::AddAssign + std::ops::SubAssign + Default + Clone> dset::potential::Group for T {
    fn id() -> Self {
        Self::default()
    }
    fn add_assign(&mut self, b: &Self) {
        *self += b.clone();
    }
    fn sub_assign(&mut self, b: &Self) {
        *self -= b.clone();
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
struct Z2(bool);

impl dset::potential::Group for Z2 {
    fn id() -> Self {
        Z2(false)
    }
    fn add_assign(&mut self, b: &Self) {
        self.0 ^= b.0;
    }
    fn sub_assign(&mut self, b: &Self) {
        self.0 ^= b.0;
    }
}

#[derive(Debug, Clone, Default)]
struct RootAgg {
    is_root_blue: bool,
    n_color: u32,
    min_idx: u32,
}

impl RootAgg {
    fn singleton(color: bool, idx: u32) -> Self {
        Self {
            is_root_blue: color,
            n_color: 1,
            min_idx: idx as u32,
        }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let q: usize = input.value();

    let mut counts = vec![heap::RemovableHeap::new(); 2];
    let mut agg = (0..n)
        .map(|u| {
            let x = RootAgg::singleton(input.token() == "1", u as u32);
            counts[x.is_root_blue as usize].push((1, Reverse(x.min_idx)));
            counts[!x.is_root_blue as usize].push((0, Reverse(x.min_idx)));
            x
        })
        .collect::<Vec<_>>();
    let mut dsu = dset::potential::DisjointSet::<Z2>::with_size(n);

    for _ in 0..q {
        match input.token() {
            "1" => {
                let u = input.value::<usize>() - 1;
                let v = input.value::<usize>() - 1;
                // println!("{} {}", u, v);

                let (mut ru, _, mut su) = dsu.find_root_with_size(u);
                let (mut rv, _, mut sv) = dsu.find_root_with_size(v);
                if !dsu.merge(u, v, Z2(true)) {
                    if dsu.delta_potential(u, v).unwrap() == Z2(false) {
                        counts[agg[ru].is_root_blue as usize]
                            .remove((agg[ru].n_color, Reverse(agg[ru].min_idx)));
                        counts[!agg[ru].is_root_blue as usize]
                            .remove((su - agg[ru].n_color, Reverse(agg[ru].min_idx)));
                    }
                    continue;
                }
                let (rc, _, sc) = dsu.find_root_with_size(u);

                // println!("    merge {:?} {:?} {:?}", ru, rv, rc);

                counts[agg[ru].is_root_blue as usize]
                    .remove((agg[ru].n_color, Reverse(agg[ru].min_idx)));
                counts[!agg[ru].is_root_blue as usize]
                    .remove((su - agg[ru].n_color, Reverse(agg[ru].min_idx)));
                counts[agg[rv].is_root_blue as usize]
                    .remove((agg[rv].n_color, Reverse(agg[rv].min_idx)));
                counts[!agg[rv].is_root_blue as usize]
                    .remove((sv - agg[rv].n_color, Reverse(agg[rv].min_idx)));

                if su < sv {
                    std::mem::swap(&mut ru, &mut rv);
                    std::mem::swap(&mut su, &mut sv);
                }
                if agg[ru].is_root_blue ^ dsu.delta_potential(ru, u).unwrap().0
                    == agg[rv].is_root_blue ^ dsu.delta_potential(rv, v).unwrap().0
                {
                    assert!(su > sv);
                    agg[rv].is_root_blue ^= true;
                }
                if agg[ru].is_root_blue == agg[rv].is_root_blue {
                    agg[ru].n_color += agg[rv].n_color;
                } else {
                    agg[ru].n_color += sv - agg[rv].n_color;
                }
                // println!(
                //     "        dp({}, {}) = {:?}    {:?} {:?}",
                //     u,
                //     v,
                //     dsu.delta_potential(u, v).unwrap().0,
                //     agg[ru],
                //     agg[rv],
                // );

                agg[rc].min_idx = agg[ru].min_idx.min(agg[rv].min_idx);
                if dsu.delta_potential(ru, rc).unwrap().0 == false {
                    agg[rc].is_root_blue = agg[ru].is_root_blue;
                    agg[rc].n_color = agg[ru].n_color;
                } else {
                    agg[rc].is_root_blue = agg[ru].is_root_blue ^ true;
                    agg[rc].n_color = sc - agg[ru].n_color;
                }

                counts[agg[rc].is_root_blue as usize]
                    .push((agg[rc].n_color, Reverse(agg[rc].min_idx)));
                counts[!agg[rc].is_root_blue as usize]
                    .push((sc - agg[rc].n_color, Reverse(agg[rc].min_idx)));
            }
            "2" => {
                let u = input.value::<usize>() - 1;
                let (ru, _, su) = dsu.find_root_with_size(u);

                counts[agg[ru].is_root_blue as usize]
                    .remove((agg[ru].n_color, Reverse(agg[ru].min_idx)));
                counts[!agg[ru].is_root_blue as usize]
                    .remove((su - agg[ru].n_color, Reverse(agg[ru].min_idx)));

                agg[ru].is_root_blue ^= true;

                counts[agg[ru].is_root_blue as usize]
                    .push((agg[ru].n_color, Reverse(agg[ru].min_idx)));
                counts[!agg[ru].is_root_blue as usize]
                    .push((su - agg[ru].n_color, Reverse(agg[ru].min_idx)));
            }
            "3" => {
                let c = input.token();
                println!(
                    "{:?} {:?}",
                    counts[0].peek().cloned(),
                    counts[1].peek().cloned()
                );
                let (_m, Reverse(u)) = counts[(c == "1") as usize].peek().unwrap();
                writeln!(output, "{}", u + 1).unwrap();
            }
            _ => panic!(),
        }

        // for u in 0..n {
        //     if dsu.find_root(u) == u {
        //         let su = dsu.get_size(u);
        //         let (r, b) = if agg[u].is_root_blue {
        //             (su - agg[u].n_color, agg[u].n_color)
        //         } else {
        //             (agg[u].n_color, su - agg[u].n_color)
        //         };
        //         print!("u={u}, r{} {:?}    ", agg[u].is_root_blue as u32, (r, b));
        //     }
        // }
        // println!();
    }
}
