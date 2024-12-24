const UNSET: u32 = u32::MAX;

// Build a max cartesian tree from inorder traversal
fn max_cartesian_tree<T>(
    n: usize,
    iter: impl IntoIterator<Item = (usize, T)>,
) -> (Vec<u32>, Vec<[u32; 2]>)
where
    T: Ord,
{
    let mut parent = vec![UNSET; n];
    let mut children = vec![[UNSET; 2]; n];

    // Monotone stack
    let mut stack = vec![];

    for (u, h) in iter {
        let u = u as u32;

        let mut c = None;
        while let Some((prev, _)) = stack.last() {
            if prev > &h {
                break;
            }
            c = stack.pop();
        }
        if let Some(&(_, p)) = stack.last() {
            parent[u as usize] = p;
            children[p as usize][1] = u;
        }
        if let Some((_, c)) = c {
            parent[c as usize] = u;
            children[u as usize][0] = c;
        }
        stack.push((h, u));
    }

    (parent, children)
}
