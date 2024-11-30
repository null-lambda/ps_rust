mod graph {
    use std::{collections::VecDeque, iter};

    // Topological sort, with lazy evaluation
    pub fn toposort<'a>(children: &'a [Vec<usize>]) -> impl Iterator<Item = usize> + 'a {
        let n = children.len();
        let mut indegree: Vec<u32> = vec![0; n];
        for u in 0..n {
            for &v in &children[u] {
                indegree[v] += 1;
            }
        }

        let mut queue: VecDeque<usize> = (0..n).filter(|&i| indegree[i] == 0).collect();
        iter::from_fn(move || {
            let u = queue.pop_front()?;
            for &v in &children[u] {
                indegree[v] -= 1;
                if indegree[v] == 0 {
                    queue.push_back(v);
                }
            }
            Some(u)
        })
    }
}
