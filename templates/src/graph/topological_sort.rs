mod graph {
    // topological sort, with lazy evaluation
    fn toposort<'a>(neighbors: &'a Vec<Vec<usize>>) -> impl Iterator<Item = usize> + 'a {
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
