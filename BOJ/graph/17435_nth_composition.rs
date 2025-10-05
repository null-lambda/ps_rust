use std::io::Write;

mod simple_io {
    pub struct InputAtOnce {
        iter: std::str::SplitAsciiWhitespace<'static>,
    }

    impl InputAtOnce {
        pub fn token(&mut self) -> &'static str {
            self.iter.next().unwrap_or_default()
        }

        pub fn try_value<T: std::str::FromStr>(&mut self) -> Option<T> {
            self.token().parse().ok()
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.try_value().unwrap()
        }
    }

    pub fn stdin() -> InputAtOnce {
        let buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let buf = Box::leak(Box::new(buf));
        let iter = buf.split_ascii_whitespace();
        InputAtOnce { iter }
    }

    pub fn stdout() -> std::io::BufWriter<std::io::Stdout> {
        std::io::BufWriter::new(std::io::stdout())
    }
}

pub mod endomorphic_sparse_table {
    // I hate sparse tables

    pub const UNSET: u32 = !0;

    fn inv_perm(perm: &[u32]) -> Vec<u32> {
        let mut res = vec![UNSET; perm.len()];
        for u in 0..perm.len() as u32 {
            res[perm[u as usize] as usize] = u;
        }
        res
    }

    pub fn gen_nth_composition(next: &[u32]) -> impl Fn(u32, u32) -> u32 {
        let n = next.len();

        let mut indegree = vec![0u32; n];
        for u in 0..n {
            indegree[next[u] as usize] += 1;
        }

        // Cycle-forest decomposition + HLD with one cycle edge removed per component.
        let mut toposort: Vec<_> = (0..n as u32)
            .filter(|&u| indegree[u as usize] == 0)
            .collect();

        // Process a forest
        let mut timer = 0;
        let mut size = vec![1u32; n];
        let mut heavy_child = vec![UNSET; n];
        while let Some(&u) = toposort.get(timer) {
            timer += 1;

            let p = next[u as usize];
            indegree[p as usize] -= 1;
            if indegree[p as usize] == 0 {
                toposort.push(p);
            }

            size[p as usize] += size[u as usize];
            let h = &mut heavy_child[p as usize];
            if *h == UNSET || (size[*h as usize] < size[u as usize]) {
                *h = u as u32;
            }
        }

        // Process cycles
        let mut cycle_size = vec![UNSET; n];
        for mut u in 0..n {
            if indegree[u] == 0 {
                continue;
            }
            indegree[u] = 0;

            let mut s = 0;
            loop {
                indegree[u] = 0;
                toposort.push(u as u32);
                s += 1;

                let p = next[u];
                if indegree[p as usize] == 0 {
                    cycle_size[u] = s;
                    break;
                }

                // Ensure that every cycle lies on a single chain.
                heavy_child[p as usize] = u as u32;

                u = p as usize;
            }
        }

        // A numbering, continuous in any chain
        let mut sid = vec![UNSET; n];
        let mut chain_top = vec![UNSET; n];
        let mut timer = 0;
        for mut u in toposort.into_iter().rev() {
            if sid[u as usize] != UNSET {
                continue;
            }

            let u0 = u;
            loop {
                chain_top[u as usize] = u0;
                sid[u as usize] = timer;
                timer += 1;
                u = heavy_child[u as usize];
                if u == UNSET {
                    break;
                }
            }
        }
        let sid_inv = inv_perm(&sid);

        let nth_parent = move |mut u: u32, mut k: u32| -> u32 {
            loop {
                let t = chain_top[u as usize];
                let d = sid[u as usize] - sid[t as usize];
                if k <= d {
                    return sid_inv[(sid[u as usize] - k) as usize];
                }
                k -= d + 1;

                if cycle_size[t as usize] != UNSET {
                    k %= cycle_size[t as usize];
                    u = next[t as usize];
                    let d = sid[u as usize] - sid[t as usize];
                    assert!(k <= d);

                    return sid_inv[(sid[u as usize] - k) as usize];
                }
                u = next[t as usize];
            }
        };
        nth_parent
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let next: Vec<u32> = (0..n).map(|_| input.value::<u32>() - 1).collect();
    let nth_composition = endomorphic_sparse_table::gen_nth_composition(&next);

    for _ in 0..input.value() {
        let k: u32 = input.value();
        let u = input.value::<u32>() - 1;
        writeln!(output, "{}", nth_composition(u, k) + 1).unwrap();
    }
}
