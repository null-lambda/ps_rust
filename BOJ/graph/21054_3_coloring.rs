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

fn mex5(xs: impl IntoIterator<Item = u8>) -> u8 {
    let mut used = [false; 5];
    for x in xs {
        used[x as usize] = true;
    }
    (0..5).find(|&x| !used[x]).unwrap() as u8
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let mut stacked_hiring = vec![];
    let mut edges = vec![];
    for day in 0..n as u32 {
        let mut fire: u64 = input.value();
        let hire: u64 = input.value();

        while fire > 0 {
            let (x, past) = stacked_hiring.last_mut().unwrap();
            let delta = fire.min(*x);
            fire -= delta;
            *x -= delta;

            if delta > 0 {
                edges.push((*past, day));
            }

            if *x == 0 {
                stacked_hiring.pop();
            }
        }

        if hire > 0 {
            stacked_hiring.push((hire, day));
        }
    }

    let mut neighbors = vec![vec![]; n];
    for &(u, v) in &edges {
        neighbors[u as usize].push(v);
        neighbors[v as usize].push(u);
    }

    for u in 0..n {
        neighbors[u].sort_unstable_by_key(|&v| if v > u as u32 { -(v as i32) } else { v as i32 });
    }

    let mut color = vec![4; n];
    for u in 0..n {
        if color[u] != 4 {
            continue;
        }

        let mut stack = vec![(u as u32, 0)];
        while let Some((u, iv)) = stack.pop() {
            if iv == 0 {
                if color[u as usize] != 4 {
                    continue;
                }
                color[u as usize] = mex5(neighbors[u as usize].iter().map(|&v| color[v as usize]));
            }
            if iv < neighbors[u as usize].len() as u32 {
                let v = neighbors[u as usize][iv as usize];
                stack.push((u, iv + 1));
                if color[v as usize] == 4 {
                    stack.push((v, 0));
                }
            }
        }
    }

    let n_colors = color.iter().max().unwrap() + 1;
    writeln!(output, "{}", n_colors).unwrap();
    for i in 0..n {
        write!(output, "{}", color[i] + 1).unwrap();
        if i + 1 < n {
            write!(output, " ").unwrap();
        }
    }
    writeln!(output).unwrap();
}
