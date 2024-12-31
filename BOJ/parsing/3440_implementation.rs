mod graph {
    // topological sort, with lazy evaluation
    pub fn toposort<'a>(neighbors: &'a Vec<Vec<usize>>) -> impl Iterator<Item = usize> + 'a {
        let n = neighbors.len();
        let mut indegree: Vec<u32> = vec![0; n];
        for u in 0..n {
            for &v in &neighbors[u] {
                indegree[v] += 1;
            }
        }

        // intialize queue with zero indegree nodes
        let mut queue: std::collections::VecDeque<usize> = indegree
            .iter()
            .enumerate()
            .filter_map(|(i, &d)| (d == 0).then_some(i))
            .collect();

        // topological sort
        std::iter::from_fn(move || {
            queue.pop_front().map(|u| {
                for &v in &neighbors[u] {
                    indegree[v] -= 1;
                    if indegree[v] == 0 {
                        queue.push_back(v);
                    }
                }
                u
            })
        })
    }
}

fn main() {
    use std::collections::HashSet;
    use std::io::BufRead;
    use std::io::Write;
    use std::iter::*;

    // let mut input_buf = std::io::read_to_string(std::io::stdin()).unwrap();
    let mut output = std::io::BufWriter::new(std::io::stdout());
    // let mut lines = input_buf.lines();

    let input = std::io::BufReader::new(std::io::stdin().lock());
    let mut grid_lines = vec![];
    for line in input.lines().flatten() {
        let line = line.as_bytes();
        if line.get(0).copied() != Some(b'*') {
            grid_lines.push(line.to_vec());
            continue;
        }
        if grid_lines.is_empty() {
            return;
        }

        // Step 1. build grid with empty border
        let n: usize = grid_lines.len();
        let m: usize = grid_lines.iter().map(|row| row.len()).max().unwrap();

        let n_pad = n + 2;
        let m_pad = m + 2;

        let mut grid = vec![b' '; n_pad * m_pad];
        for (i, s) in (1..=n).zip(grid_lines) {
            grid[i * m_pad..][1..=s.len()].copy_from_slice(&s);
        }
        grid_lines = vec![];

        #[derive(Debug, Clone)]
        struct Rect {
            x0: usize,
            x1: usize,
            y0: usize,
            y1: usize,
        }

        #[derive(Debug, Clone)]
        pub enum Node {
            Gate(Rect, u8),
            Src(usize, u8),
            Dest(usize, u8),
        }

        type Point = (usize, usize);

        // Step 2-1. find all gates, sources and dests
        const UNSET: usize = usize::MAX;
        let mut nodes = vec![];
        let mut node_indices: Vec<usize> = vec![UNSET; grid.len()];

        for i in 1..=n {
            for j in 1..=m {
                let idx = i * m_pad + j;
                if node_indices[idx] != UNSET {
                    continue;
                }
                match grid[idx] {
                    b'#' => {
                        // find rect
                        let (y0, x0) = (i, j);
                        let (mut x1, mut y1) = (x0, y0);
                        while grid[(y1 + 1) * m_pad + x0] == b'#' {
                            y1 += 1;
                        }
                        while grid[y0 * m_pad + (x1 + 1)] == b'#' {
                            x1 += 1;
                        }
                        let rect = Rect { x0, x1, y0, y1 };

                        // mark visited cells
                        for k in y0..=y1 {
                            for l in x0..=x1 {
                                node_indices[k * m_pad + l] = nodes.len();
                            }
                        }

                        let op = (y0 + 1..y1)
                            .flat_map(|i| (x0 + 1..x1).map(move |j| i * m_pad + j))
                            .map(|idx| grid[idx])
                            .find(|&c| c != b' ')
                            .unwrap();
                        nodes.push(Node::Gate(rect, op));
                    }
                    b'0' | b'1' => {
                        node_indices[idx] = nodes.len();
                        nodes.push(Node::Src(idx, grid[idx]));
                    }
                    b'A'..=b'Z' => {
                        node_indices[idx] = nodes.len();
                        nodes.push(Node::Dest(idx, grid[idx]));
                    }
                    _ => {}
                }
            }
        }

        // Step 2-2. find all connections from the source, and build graph structure
        let mut parents: Vec<Vec<(usize, bool)>> = vec![vec![]; nodes.len()];
        let mut visited = HashSet::new();
        for (u, node) in nodes.iter().enumerate() {
            // dfs
            match node {
                Node::Gate(rect, _) => {
                    let &Rect { x1, y0, y1, .. } = rect;
                    for i in y0..=y1 {
                        let j = x1 + 1;
                        if grid[i * m_pad + j] == b'=' {
                            dfs(
                                &grid,
                                m_pad,
                                &node_indices,
                                &mut visited,
                                &mut |v| parents[v].push((u, false)),
                                i * m_pad + j + 1,
                                i * m_pad + j,
                            );
                            visited.clear();
                        } else if grid[i * m_pad + j] == b'o' && grid[i * m_pad + j + 1] == b'=' {
                            dfs(
                                &grid,
                                m_pad,
                                &node_indices,
                                &mut visited,
                                &mut |v| parents[v].push((u, true)),
                                i * m_pad + j + 2,
                                i * m_pad + j + 1,
                            );
                            visited.clear();
                        }
                    }
                }
                Node::Src(idx, _) => {
                    let (i, j) = (idx / m_pad, idx % m_pad);
                    assert_eq!(grid[i * m_pad + j + 1], b'=');
                    dfs(
                        &grid,
                        m_pad,
                        &node_indices,
                        &mut visited,
                        &mut |v| parents[v].push((u, false)),
                        i * m_pad + j + 2,
                        i * m_pad + j + 1,
                    );
                    visited.clear();
                }
                Node::Dest(_, _) => {}
            }
        }
        fn dfs(
            grid: &[u8],
            m_pad: usize,
            node_indices: &[usize],
            visited: &mut HashSet<usize>,
            yield_node: &mut impl FnMut(usize),
            u: usize,
            u_prev: usize,
        ) {
            if visited.contains(&u) {
                return;
            }
            if node_indices[u] != UNSET {
                if u == u_prev + 1 {
                    yield_node(node_indices[u]);
                }
                return;
            }

            visited.insert(u);
            match grid[u] {
                b'-' | b'|' | b'x' | b'=' => {
                    let u_next = 2 * u - u_prev;
                    dfs(grid, m_pad, node_indices, visited, yield_node, u_next, u);
                }
                b'+' => {
                    let neighbors =
                        empty()
                            .chain([u + 1, u - 1].into_iter().filter(|&u_next| {
                                matches!(grid[u_next], b'-' | b'=' | b'x' | b'+')
                            }))
                            .chain(
                                [u + m_pad, u - m_pad]
                                    .into_iter()
                                    .filter(|&u_next| matches!(grid[u_next], b'|' | b'x' | b'+')),
                            );
                    for u_next in neighbors {
                        dfs(grid, m_pad, node_indices, visited, yield_node, u_next, u);
                    }
                }
                _ => panic!(),
            }
        }

        let mut children: Vec<Vec<usize>> = vec![vec![]; nodes.len()];
        for u in 0..nodes.len() {
            for &(v, _) in &parents[u] {
                children[v].push(u);
            }
        }

        let mut values = vec![false; nodes.len()];
        let mut dest_values = vec![];

        // Step 3. Evaluate
        for u in graph::toposort(&children) {
            match &nodes[u] {
                Node::Src(_, value) => values[u] = *value == b'1',
                Node::Gate(_, op) => {
                    values[u] = match op {
                        b'1' => parents[u].iter().any(|&(v, inv)| values[v] != inv),
                        b'&' => parents[u].iter().all(|&(v, inv)| values[v] != inv),
                        b'=' => parents[u]
                            .iter()
                            .fold(false, |acc, &(v, inv)| acc ^ (values[v] != inv)),
                        _ => panic!(),
                    }
                }
                Node::Dest(_, varname) => {
                    let (v, inv) = parents[u][0];
                    dest_values.push((*varname, values[v] != inv));
                }
            }
        }
        dest_values.sort_unstable();

        for &(varname, value) in &dest_values {
            let value = (value as u8 + b'0') as char;
            writeln!(output, "{}={}", varname as char, value).unwrap();
        }
        writeln!(output).unwrap();
    }
}
