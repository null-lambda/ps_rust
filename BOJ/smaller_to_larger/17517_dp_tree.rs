use std::{cmp::Reverse, collections::BinaryHeap, io::Write, mem::MaybeUninit};

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

const INF: u32 = 1 << 31;

#[derive(Default, Clone)]
struct NodeData {
    deltas: BinaryHeap<u64>,
}

impl NodeData {
    fn pull_from(&mut self, mut child: Self, weight: u64) {
        child.deltas.push(weight);

        if self.deltas.len() < child.deltas.len() {
            std::mem::swap(&mut self.deltas, &mut child.deltas);
        }

        let mut sums = vec![];
        while let Some(d1) = child.deltas.pop() {
            let d2 = unsafe { self.deltas.pop().unwrap_unchecked() };
            sums.push(d1 + d2);
        }
        self.deltas.extend(sums);
    }

    fn collapse(&mut self, n: usize, yield_ans: &mut impl FnMut(u64)) {
        let mut acc = 0;
        let mut i = 0;
        while let Some(d) = self.deltas.pop() {
            if i == n {
                break;
            }
            i += 1;
            acc += d;
            yield_ans(acc);
        }

        while i < n {
            yield_ans(acc);
            i += 1;
        }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let mut events = vec![];
    for _ in 0..n {
        let s = input.value::<u32>() - 1;
        let e = input.value::<u32>() - 1;
        let w: u64 = input.value();
        events.push((s, e, w));
        events.push((e, INF, 0));
    }
    events.sort_unstable_by_key(|&(s, e, _)| (s, Reverse(e)));

    let mut parent = vec![(0, 0)];
    let mut stack = vec![0];
    for (_s, e, w) in events {
        if e != INF {
            let u = parent.len() as u32;
            parent.push((*stack.last().unwrap(), w));
            stack.push(u);
        } else {
            stack.pop();
        }
    }

    let n_nodes = parent.len();
    let mut dp: Vec<_> = (0..n_nodes)
        .map(|_| MaybeUninit::new(NodeData::default()))
        .collect();

    for u in (1..n_nodes).rev() {
        let (p, w) = parent[u];
        let dp_u = unsafe { std::mem::replace(&mut dp[u], MaybeUninit::uninit()).assume_init() };
        unsafe { dp[p as usize].assume_init_mut() }.pull_from(dp_u, w);
    }

    unsafe { dp[0].assume_init_mut() }
        .collapse(n as usize, &mut |ans| write!(output, "{} ", ans).unwrap());
    writeln!(output).unwrap();
}
