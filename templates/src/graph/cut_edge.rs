fn cut_edges(n: usize, neighbors: Vec<Vec<u32>>) -> Vec<(u32, u32)> {
    let mut dfs_order = vec![0; n];
    let mut cut_edges = vec![];
    let mut order = 1;
    fn dfs(
        u: u32,
        parent: u32,
        neighbors: &Vec<Vec<u32>>,
        dfs_order: &mut Vec<u32>,
        cut_edges: &mut Vec<(u32, u32)>,
        order: &mut u32,
    ) -> u32 {
        dfs_order[u as usize] = *order;
        *order += 1;
        let mut low_u = *order;
        for &v in &neighbors[u as usize] {
            if parent == v {
                continue;
            }
            if dfs_order[v as usize] != 0 {
                low_u = low_u.min(dfs_order[v as usize]);
            } else {
                let low_v = dfs(v, u, neighbors, dfs_order, cut_edges, order);
                if low_v > dfs_order[u as usize] {
                    cut_edges.push((u.min(v), u.max(v)));
                }
                low_u = low_u.min(low_v);
            }
        }
        low_u
    }

    const UNDEFINED: u32 = i32::MAX as u32;
    dfs(
        0,
        UNDEFINED,
        &neighbors,
        &mut dfs_order,
        &mut cut_edges,
        &mut order,
    );
    cut_edges
}

unsafe fn cut_edges_unsafe(
    n: usize,
    neighbors: Vec<Vec<u32>>,
    mut visit_edge: impl FnMut(u32, u32),
) {
    // Messed up thread safety
    static mut DFS_ORDER: Vec<u32> = vec![];
    static mut ORDER: u32 = 0;
    static mut NEIGHBORS: *const Vec<Vec<u32>> = &vec![];

    unsafe fn dfs(u: u32, parent: u32, visit_edge: &mut impl FnMut(u32, u32)) -> u32 {
        DFS_ORDER[u as usize] = ORDER;
        ORDER += 1;
        let mut low_u = ORDER;
        for &v in &NEIGHBORS.as_ref().unwrap()[u as usize] {
            if parent == v {
                continue;
            }
            if DFS_ORDER[v as usize] != 0 {
                low_u = low_u.min(DFS_ORDER[v as usize]);
            } else {
                let low_v = dfs(v, u, visit_edge);
                if low_v > DFS_ORDER[u as usize] {
                    visit_edge(u, v);
                }
                low_u = low_u.min(low_v);
            }
        }
        low_u
    }

    const UNDEFINED: u32 = i32::MAX as u32;
    DFS_ORDER = vec![0; n];
    ORDER = 1;
    NEIGHBORS = &neighbors;
    dfs(0, UNDEFINED, &mut visit_edge);
}
