mod tree {
    use std::iter;

    pub fn preorder_edge_lazy<'a, T: Copy>(
        neighbors: &'a [Vec<(usize, T)>],
        node: usize,
        parent: usize,
    ) -> impl Iterator<Item = (usize, usize, T)> + 'a {
        let mut stack = vec![(node, parent, neighbors[node].iter())];
        iter::from_fn(move || {
            stack.pop().map(|(node, parent, mut iter_child)| {
                let (child, weight) = *iter_child.next()?;
                stack.push((node, parent, iter_child));
                if child == parent {
                    return None;
                }
                stack.push((child, node, neighbors[child].iter()));
                Some((child, node, weight))
            })
        })
        .flatten()
    }
}
