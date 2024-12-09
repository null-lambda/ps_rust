use std::{collections::HashMap, io::Write};

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

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let mut left_names = vec![];
    let mut right_names = vec![];
    let mut left_index_map = HashMap::new();
    let mut right_index_map = HashMap::new();
    for i in 0..n {
        let t = input.token();
        left_names.push(t);
        left_index_map.insert(t, i);
    }
    for i in 0..n {
        let t = input.token();
        right_names.push(t);
        right_index_map.insert(t, i);
    }

    let mut left_list = vec![vec![]; n];
    for _ in 0..n {
        let i = left_index_map[&input.token()];
        for _ in 0..n {
            left_list[i].push(right_index_map[&input.token()]);
        }
    }

    let mut right_priority = vec![vec![0; n]; n];
    for _ in 0..n {
        let i = right_index_map[&input.token()];
        for p in 0..n {
            right_priority[i][left_index_map[&input.token()]] = p;
        }
    }

    // Gale-Shapley Algorithm for Stable matching problem
    let mut assignment = [vec![None; n], vec![None; n]];
    for _ in 0..n {
        for i in 0..n {
            if assignment[0][i].is_some() {
                continue;
            }
            for &j in &left_list[i] {
                if let Some(k) = assignment[1][j] {
                    if right_priority[j][i] < right_priority[j][k] {
                        assignment[0][i] = Some(j);
                        assignment[1][j] = Some(i);
                        assignment[0][k] = None;
                        break;
                    }
                } else {
                    assignment[0][i] = Some(j);
                    assignment[1][j] = Some(i);
                    break;
                }
            }
        }
    }

    for i in 0..n {
        let left_name = left_names[i];
        let right_name = right_names[assignment[0][i].unwrap()];
        writeln!(output, "{} {}", left_name, right_name).unwrap();
    }
}
