fn cut_edges<'a, _E: 'a, F>(neighbors: &'a impl Jagged<'a, (u32, _E)>, mut yield_edge: F)
where
    F: FnMut(u32, u32),
{
    const UNSET: u32 = u32::MAX;
    let n = neighbors.len();

    fn rec<'a, _E: 'a>(
        order: &mut [u32],
        timer: &mut u32,
        neighbors: &'a impl Jagged<'a, (u32, _E)>,
        visit_edge: &mut impl FnMut(u32, u32),
        u: u32,
        p: u32,
    ) -> u32 {
        order[u as usize] = *timer;
        *timer += 1;
        let mut low_u = *timer;
        for (v, _) in neighbors.get(u as usize) {
            let v = *v;
            if p == v {
                continue;
            }
            if order[v as usize] != 0 {
                low_u = low_u.min(order[v as usize]);
            } else {
                let low_v = rec(order, timer, neighbors, visit_edge, v, u);
                if low_v > order[u as usize] {
                    visit_edge(u, v);
                }
                low_u = low_u.min(low_v);
            }
        }
        low_u
    }

    rec(
        &mut vec![0; n],
        &mut 1,
        neighbors,
        &mut yield_edge,
        0,
        UNSET,
    );
}
