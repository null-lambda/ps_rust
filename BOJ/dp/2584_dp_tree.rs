use std::io::Write;

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

const INF: u32 = 1 << 30;
const UNSET: u32 = u32::MAX;

#[derive(Clone, Default)]
struct Tag {
    parity_rel_parent: bool,
    k_prev: u32,
}

#[derive(Clone, Default)]
struct NodeData {
    cost: Vec<u32>, // Upward prop
    tag: Vec<Tag>,
    opt_parity: bool, // Downward prop
    opt_k: u32,       // Downward prop
}

impl NodeData {
    fn empty() -> Self {
        Self {
            cost: vec![INF, 0],
            tag: vec![],
            opt_k: UNSET,
            opt_parity: false,
        }
    }

    fn size(&self) -> usize {
        self.cost.len() - 1
    }

    // Upward propagation, eval optimal cost
    fn pull_up(&mut self, child: &mut NodeData, weight: u32) {
        let n = self.size();
        let m = child.size();
        let mut conv = vec![INF; n + m + 1];
        child.tag = vec![Tag::default(); n + m + 1];

        for j in 1..=m {
            for i in 1..=n {
                let joint_cost = self.cost[i] + child.cost[j] + weight;
                if joint_cost < conv[i + j] {
                    conv[i + j] = joint_cost;
                    child.tag[i + j] = Tag {
                        parity_rel_parent: false,
                        k_prev: i as u32,
                    };
                }
            }
        }

        for j in 1..=m {
            for i in 1..=n {
                let joint_cost = self.cost[i] + child.cost[j];
                if joint_cost < conv[i + m - j] {
                    conv[i + m - j] = joint_cost;
                    child.tag[i + m - j] = Tag {
                        parity_rel_parent: true,
                        k_prev: i as u32,
                    };
                }
            }
        }

        self.cost = conv;
    }

    // Downward propagation, reconstruct opt cost and partiy
    fn push_down(&mut self, child: &mut NodeData) {
        let parity_rel_parent = child.tag[self.opt_k as usize].parity_rel_parent;
        child.opt_parity = self.opt_parity ^ parity_rel_parent;
        child.opt_k = if !parity_rel_parent {
            self.opt_k - child.tag[self.opt_k as usize].k_prev
        } else {
            child.size() as u32 + child.tag[self.opt_k as usize].k_prev - self.opt_k
        };
        self.opt_k = child.tag[self.opt_k as usize].k_prev;
    }
}

fn get_two<T>(xs: &mut [T], i: usize, j: usize) -> Option<(&mut T, &mut T)> {
    debug_assert!(i < xs.len() && j < xs.len());
    if i == j {
        return None;
    }
    let ptr = xs.as_mut_ptr();
    Some(unsafe { (&mut *ptr.add(i), &mut *ptr.add(j)) })
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let k: usize = input.value();

    let mut degree = vec![0; n];
    let mut xor_neighbors = vec![(0, 0); n];
    for _ in 0..n - 1 {
        let u = input.value::<u32>() - 1;
        let v = input.value::<u32>() - 1;
        let w = input.value::<u32>();
        degree[u as usize] += 1;
        degree[v as usize] += 1;
        xor_neighbors[u as usize].0 ^= v;
        xor_neighbors[v as usize].0 ^= u;
        xor_neighbors[u as usize].1 ^= w;
        xor_neighbors[v as usize].1 ^= w;
    }
    let root = 0;
    degree[root] += 2;

    let mut dp = vec![NodeData::empty(); n];
    let mut topological_order = vec![];
    for mut u in 0..n as u32 {
        while degree[u as usize] == 1 {
            let (p, w) = xor_neighbors[u as usize];
            degree[u as usize] -= 1;
            degree[p as usize] -= 1;
            xor_neighbors[p as usize].0 ^= u;
            xor_neighbors[p as usize].1 ^= w;
            topological_order.push((u, p));

            let (dp_u, dp_p) = get_two(&mut dp, u as usize, p as usize).unwrap();
            dp_p.pull_up(dp_u, w);

            u = p;
        }
    }
    let root_cost = [dp[root].cost[k], dp[root].cost[n - k]];
    let min_cost;
    if root_cost[0] < root_cost[1] {
        min_cost = root_cost[0];
        dp[root].opt_k = k as u32;
        dp[root].opt_parity = false;
    } else {
        min_cost = root_cost[1];
        dp[root].opt_k = (n - k) as u32;
        dp[root].opt_parity = true;
    }

    for (u, p) in topological_order.into_iter().rev() {
        let (dp_u, dp_p) = get_two(&mut dp, u as usize, p as usize).unwrap();
        dp_p.push_down(dp_u);
    }

    writeln!(output, "{}", min_cost).unwrap();
    for u in 0..n {
        if !dp[u].opt_parity {
            write!(output, "{} ", u + 1).unwrap();
        }
    }
    writeln!(output).unwrap();
}
