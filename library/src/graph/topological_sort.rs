mod graph {
    use std::{collections::VecDeque, iter};

    pub fn toposort<'a, C, F>(n: usize, mut children: F) -> impl Iterator<Item = usize> + 'a
    where
        F: 'a + FnMut(usize) -> C,
        C: IntoIterator<Item = usize>,
    {
        let mut indegree = vec![0u32; n];
        for node in 0..n {
            for child in children(node) {
                indegree[child] += 1;
            }
        }

        let mut queue: VecDeque<usize> = (0..n).filter(|&node| indegree[node] == 0).collect();
        iter::from_fn(move || {
            let current = queue.pop_front()?;
            for child in children(current) {
                indegree[child] -= 1;
                if indegree[child] == 0 {
                    queue.push_back(child);
                }
            }
            Some(current)
        })
    }
}
