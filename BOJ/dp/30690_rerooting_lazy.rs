use std::io::Write;

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

pub mod jagged {
    use std::fmt::Debug;
    use std::iter;
    use std::mem::MaybeUninit;

    // Trait for painless switch between different representations of a jagged array
    pub trait Jagged<'a, T: 'a> {
        type ItemRef: ExactSizeIterator<Item = &'a T>;
        fn len(&self) -> usize;
        fn get(&'a self, u: usize) -> Self::ItemRef;
    }

    impl<'a, T, C> Jagged<'a, T> for C
    where
        C: AsRef<[Vec<T>]> + 'a,
        T: 'a,
    {
        type ItemRef = std::slice::Iter<'a, T>;
        fn len(&self) -> usize {
            <Self as AsRef<[Vec<T>]>>::as_ref(self).len()
        }
        fn get(&'a self, u: usize) -> Self::ItemRef {
            let res = <Self as AsRef<[Vec<T>]>>::as_ref(self)[u].iter();
            res
        }
    }

    // Compressed sparse row format for jagged array
    // Provides good locality for graph traversal, but works only for static ones.
    #[derive(Clone, PartialEq, Eq, PartialOrd, Ord)]
    pub struct CSR<T> {
        data: Vec<T>,
        head: Vec<u32>,
    }

    impl<T> Debug for CSR<T>
    where
        T: Debug,
    {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let v: Vec<Vec<&T>> = (0..self.len()).map(|i| self.get(i).collect()).collect();
            v.fmt(f)
        }
    }

    impl<T, I> FromIterator<I> for CSR<T>
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
            CSR { data, head }
        }
    }

    impl<T: Clone> CSR<T> {
        pub fn from_assoc_list(n: usize, pairs: &[(u32, T)]) -> Self {
            let mut head = vec![0u32; n + 1];

            for &(u, _) in pairs {
                debug_assert!(u < n as u32);
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

            CSR { data, head }
        }
    }

    impl<'a, T: 'a> Jagged<'a, T> for CSR<T> {
        type ItemRef = std::slice::Iter<'a, T>;

        fn len(&self) -> usize {
            self.head.len() - 1
        }

        fn get(&'a self, u: usize) -> Self::ItemRef {
            self.data[self.head[u] as usize..self.head[u + 1] as usize].iter()
        }
    }
}

pub mod reroot {
    pub mod lazy {
        // O(n) rerooting dp for trees, with combinable pulling operation. (Monoid action)
        // https://codeforces.com/blog/entry/124286
        // https://github.com/koosaga/olympiad/blob/master/Library/codes/data_structures/all_direction_tree_dp.cpp
        use crate::jagged::Jagged;

        pub trait RootData {
            type E: Clone; // edge weight
            type F: Clone; // pulling operation (edge dp)
            fn lift_to_action(&self, weight: &Self::E) -> Self::F;
            fn id_action() -> Self::F;
            fn combine_action(&self, lhs: &Self::F, rhs: &Self::F) -> Self::F;
            fn apply(&mut self, action: &Self::F);
            fn finalize(&self) -> Self;
        }

        fn get_two<T>(xs: &mut [T], i: usize, j: usize) -> Option<(&mut T, &mut T)> {
            debug_assert!(i < xs.len() && j < xs.len());
            if i == j {
                return None;
            }
            let ptr = xs.as_mut_ptr();
            Some(unsafe { (&mut *ptr.add(i), &mut *ptr.add(j)) })
        }

        pub fn run<'a, E: Clone + Default + 'a, R: RootData<E = E> + Clone>(
            neighbors: &'a impl Jagged<'a, (u32, E)>,
            data: &[R],
            root: usize,
            yield_node_dp: &mut impl FnMut(usize, R),
            yield_edge_dp: &mut impl FnMut(usize, &R::F, &R::F, &E),
        ) -> Vec<(u32, E)> {
            let n = neighbors.len();
            let mut preorder = vec![];
            let mut parent = vec![(root as u32, E::default()); n];
            let mut stack = vec![(root as u32, root as u32)];
            while let Some((u, p)) = stack.pop() {
                preorder.push(u);
                for (v, w) in neighbors.get(u as usize) {
                    if *v == p {
                        continue;
                    }
                    parent[*v as usize] = (u, w.clone());
                    stack.push((*v, u));
                }
            }

            // Init tree DP
            let mut sum_upward = data.to_owned();
            let mut action_upward = vec![R::id_action(); n];
            for &u in preorder[1..].iter().rev() {
                let (p, w) = &parent[u as usize];
                let (data_u, data_p) =
                    unsafe { get_two(&mut sum_upward, u as usize, *p as usize).unwrap_unchecked() };
                action_upward[u as usize] = data_u.finalize().lift_to_action(w);
                data_p.apply(&action_upward[u as usize]);
            }

            // Reroot
            let mut action_from_parent = vec![R::id_action(); n];
            for &u in &preorder {
                let (p, w) = &parent[u as usize];

                let mut sum_u = sum_upward[u as usize].clone();
                sum_u.apply(&action_from_parent[u as usize]);
                yield_node_dp(u as usize, sum_u.finalize());
                if u as usize != root {
                    yield_edge_dp(
                        u as usize,
                        &action_upward[u as usize],
                        &action_from_parent[u as usize],
                        w,
                    );
                }

                let n_child = neighbors.get(u as usize).len() - (u as usize != root) as usize;
                match n_child {
                    0 => {}
                    1 => {
                        for (v, w) in neighbors.get(u as usize) {
                            if *v == *p {
                                continue;
                            }
                            let exclusive = action_from_parent[u as usize].clone();
                            let mut sum_exclusive = data[u as usize].clone();
                            sum_exclusive.apply(&exclusive);
                            action_from_parent[*v as usize] =
                                sum_exclusive.finalize().lift_to_action(&w);
                        }
                    }
                    _ => {
                        let mut prefix: Vec<R::F> = neighbors
                            .get(u as usize)
                            .map(|(v, _)| {
                                if *v == *p {
                                    action_from_parent[u as usize].clone()
                                } else {
                                    action_upward[*v as usize].clone()
                                }
                            })
                            .collect();
                        let mut postfix = prefix.clone();
                        for i in (1..neighbors.get(u as usize).len()).rev() {
                            postfix[i - 1] =
                                data[u as usize].combine_action(&postfix[i - 1], &postfix[i]);
                        }
                        for i in 1..neighbors.get(u as usize).len() {
                            prefix[i] = data[u as usize].combine_action(&prefix[i - 1], &prefix[i]);
                        }

                        for (i, (v, w)) in neighbors.get(u as usize).enumerate() {
                            if *v == *p {
                                continue;
                            }
                            let exclusive = if i == 0 {
                                postfix[1].clone()
                            } else if i == neighbors.get(u as usize).len() - 1 {
                                prefix[neighbors.get(u as usize).len() - 2].clone()
                            } else {
                                data[u as usize].combine_action(&prefix[i - 1], &postfix[i + 1])
                            };

                            let mut sum_exclusive = data[u as usize].clone();
                            sum_exclusive.apply(&exclusive);
                            action_from_parent[*v as usize] =
                                sum_exclusive.finalize().lift_to_action(&w);
                        }
                    }
                }
            }
            parent
        }
    }
}

#[derive(Clone, Default, Debug)]
struct NodeDp {
    diam: u64,
    depth: u64,
}

impl reroot::lazy::RootData for NodeDp {
    type E = ();
    type F = NodeDp;

    fn lift_to_action(&self, (): &()) -> NodeDp {
        NodeDp {
            diam: self.diam,
            depth: self.depth + 1,
        }
    }

    fn id_action() -> NodeDp {
        NodeDp { diam: 0, depth: 0 }
    }

    fn combine_action(&self, lhs: &NodeDp, rhs: &NodeDp) -> NodeDp {
        let mut res = lhs.clone();
        res.apply(rhs);
        res
    }

    fn apply(&mut self, action: &NodeDp) {
        self.diam = self.diam.max(action.diam).max(self.depth + action.depth);
        self.depth = self.depth.max(action.depth);
    }

    fn finalize(&self) -> NodeDp {
        Self {
            diam: self.diam,
            depth: self.depth,
        }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let q: usize = input.value();
    let mut edges = vec![];
    for _ in 0..n - 1 {
        let a = input.value::<u32>() - 1;
        let b = input.value::<u32>() - 1;
        edges.push((a, (b, ())));
        edges.push((b, (a, ())));
    }
    let neighbors = jagged::CSR::from_assoc_list(n, &edges);

    let mut ans = vec![0; n];
    let parent = reroot::lazy::run(
        &neighbors,
        &vec![NodeDp::default(); n],
        0,
        &mut |_, _| {},
        &mut |u, e1, e2, ()| {
            ans[u] = e1.diam + e2.diam + 1;
        },
    );

    for _ in 0..q {
        let u = input.value::<usize>() - 1;
        let v = input.value::<usize>() - 1;

        let x = if parent[u].0 == v as u32 { u } else { v };
        writeln!(output, "{}", ans[x]).unwrap();
    }
}
