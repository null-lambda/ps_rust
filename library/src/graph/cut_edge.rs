fn cut_edges<'a>(
    n: usize,
    neighbors: &'a impl Jagged<'a, (u32, ())>,
    init: usize,
) -> Vec<(u32, u32)> {
    let mut dfs_order = vec![0; n];
    let mut low = vec![0; n];
    let mut cut_edges = vec![];

    fn dfs<'a>(
        neighbors: &'a impl Jagged<'a, (u32, ())>,
        dfs_order: &mut Vec<u32>,
        cut_edges: &mut Vec<(u32, u32)>,
        low: &mut Vec<u32>,
        timer: &mut u32,
        u: u32,
        p: u32,
    ) {
        if dfs_order[u as usize] != 0 {
            low[p as usize] = low[p as usize].min(dfs_order[u as usize]);
            return;
        }

        *timer += 1;
        dfs_order[u as usize] = *timer;
        low[u as usize] = *timer;
        for &(v, ()) in neighbors.get(u as usize) {
            if p == v {
                continue;
            }
            dfs(neighbors, dfs_order, cut_edges, low, timer, v, u);
        }

        if low[u as usize] > dfs_order[p as usize] {
            cut_edges.push((p, u));
        }
        low[p as usize] = low[p as usize].min(low[u as usize]);
    }

    dfs(
        neighbors,
        &mut dfs_order,
        &mut cut_edges,
        &mut low,
        &mut 0,
        init as u32,
        init as u32,
    );
    cut_edges
}
