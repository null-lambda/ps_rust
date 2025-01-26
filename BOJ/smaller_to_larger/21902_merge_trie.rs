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

fn parse_byte(b: u8) -> u8 {
    b - b'a'
}

const N_ALPHABETS: usize = 26;

const UNSET: u32 = !0;
type TransitionMap = [u32; N_ALPHABETS];

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let mut degree = vec![0u32; n];
    let mut xor_neighbors = vec![(0u32, 0u8); n];
    for _ in 0..n - 1 {
        let u = input.value::<u32>() - 1;
        let v = input.value::<u32>() - 1;
        let c = parse_byte(input.token().as_bytes()[0]);
        degree[u as usize] += 1;
        degree[v as usize] += 1;
        xor_neighbors[u as usize].0 ^= v;
        xor_neighbors[u as usize].1 ^= c;
        xor_neighbors[v as usize].0 ^= u;
        xor_neighbors[v as usize].1 ^= c;
    }
    degree[0] += 2;

    let mut sizes = vec![1u32; n];
    let mut neighbors = vec![[UNSET; N_ALPHABETS]; n];
    let mut depth = vec![0u32; n];
    let mut topological_order = vec![];
    for mut u in 0..n as u32 {
        while degree[u as usize] == 1 {
            let (p, c) = xor_neighbors[u as usize];
            degree[u as usize] -= 1;
            degree[p as usize] -= 1;
            xor_neighbors[p as usize].0 ^= u;
            xor_neighbors[p as usize].1 ^= c;
            topological_order.push((u, p));

            sizes[p as usize] += sizes[u as usize];
            neighbors[p as usize][c as usize] = u;

            u = p;
        }
    }

    for (u, p) in topological_order.into_iter().rev() {
        depth[u as usize] = depth[p as usize] + 1;
    }
    let max_depth = *depth.iter().max().unwrap();

    let mut score = vec![n as u32; max_depth as usize + 1];
    for r in 0..n {
        let dr = depth[r as usize];
        let Some(&large) = neighbors[r]
            .iter()
            .filter(|&&v| v != UNSET)
            .max_by_key(|&&v| sizes[v as usize])
        else {
            continue;
        };

        score[dr as usize + 1] -= 1;
        let mut history = vec![];
        for i_small in 0..N_ALPHABETS {
            let small = neighbors[r][i_small];
            if small == UNSET || small == large {
                continue;
            }

            // Merge two tries
            let mut stack = vec![(large, small, 0u32)];
            while let Some((u, v, c)) = stack.pop() {
                if c == 0 {
                    score[dr as usize + 1] -= 1;
                }

                if (c as usize) < N_ALPHABETS {
                    stack.push((u, v, c + 1));

                    let nu = neighbors[u as usize][c as usize];
                    let nv = neighbors[v as usize][c as usize];
                    if nv != UNSET {
                        if nu != UNSET {
                            stack.push((nu, nv, 0));
                        } else {
                            // println!("merge {} {} {}", u, v, c);
                            neighbors[u as usize][c as usize] = nv;
                            history.push((u, c));
                        }
                    }
                }
            }

            // println!("{:?}", (r, small, large));
        }

        // Rollback
        for (u, c) in history.drain(..) {
            neighbors[u as usize][c as usize] = UNSET;
        }
    }

    // println!("{:?}", score);

    let (score, d_opt) = (1..=max_depth)
        .map(|d| (score[d as usize], d))
        .min()
        .unwrap();
    writeln!(output, "{}", score).unwrap();
    writeln!(output, "{}", d_opt).unwrap();
}
