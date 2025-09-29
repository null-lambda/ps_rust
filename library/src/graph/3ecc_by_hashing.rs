pub mod three_ecc {
    use super::dset;
    use super::jagged::CSR;
    use super::rand;
    use std::collections::HashMap;

    const UNSET: u32 = !0;

    pub fn get_by_hashing(n: usize, edges: &[[u32; 2]]) -> CSR<u32> {
        let m = edges.len();
        let neighbors = CSR::from_pairs(
            n,
            edges
                .iter()
                .enumerate()
                .flat_map(|(e, &[u, v])| [(u, (v, e as u32)), (v, (u, e as u32))]),
        );

        let mut rng = rand::SplitMix64::from_entropy().unwrap();
        let zobrist: Vec<_> = (0..m)
            .map(|_| (rng.next_u64() as u128) << 64 | rng.next_u64() as u128)
            .collect();

        let mut lowpt = vec![0u32; n];
        let mut lowe = vec![UNSET; n];
        let mut n_cover = vec![0i32; n];
        let mut xor_cover = vec![0; n];

        let mut t_in = vec![UNSET; n];
        let mut parent: Vec<_> = (0..n as u32).collect();
        let mut parent_edge = vec![UNSET; n];
        let mut timer = 0;

        let mut associated_cut = vec![0u8; m]; // Alternative name: `is_bridge`
        let mut groups = HashMap::<_, Vec<u32>>::new();

        let mut current_edge: Vec<_> = (0..n)
            .map(|u| neighbors.edge_range(u).start as u32)
            .collect();
        for root in 0..n {
            if t_in[root] != UNSET {
                continue;
            }
            t_in[root] = timer;
            timer += 1;

            let mut u = root as u32;
            loop {
                let p = parent[u as usize];
                let ie = current_edge[u as usize];
                current_edge[u as usize] += 1;
                if ie == neighbors.edge_range(u as usize).start as u32 {
                    // On enter
                    t_in[u as usize] = timer;
                    lowpt[u as usize] = timer;
                    timer += 1;
                }
                if ie == neighbors.edge_range(u as usize).end as u32 {
                    // On exit
                    if p == u {
                        break;
                    }

                    match n_cover[u as usize] {
                        0 => {
                            associated_cut[parent_edge[u as usize] as usize] = 1;
                        }
                        1 => {
                            associated_cut[parent_edge[u as usize] as usize] = 2;
                            associated_cut[lowe[u as usize] as usize] = 2;
                        }
                        _ => {
                            groups.entry(xor_cover[u as usize]).or_default().push(u);
                        }
                    }

                    if lowpt[u as usize] < lowpt[p as usize] {
                        lowpt[p as usize] = lowpt[u as usize];
                        lowe[p as usize] = lowe[u as usize];
                    }

                    n_cover[p as usize] += n_cover[u as usize];
                    xor_cover[p as usize] ^= xor_cover[u as usize];

                    u = p;
                    continue;
                }

                let (v, e) = neighbors.links[ie as usize];
                if e == parent_edge[u as usize] {
                    continue;
                }

                if t_in[v as usize] == UNSET {
                    // Front edge
                    parent[v as usize] = u;
                    parent_edge[v as usize] = e;

                    u = v;
                } else if t_in[v as usize] < t_in[u as usize] {
                    // Back edge
                    if t_in[v as usize] < lowpt[u as usize] {
                        lowpt[u as usize] = t_in[v as usize];
                        lowe[u as usize] = e;
                    }

                    n_cover[u as usize] += 1;
                    n_cover[v as usize] -= 1;
                    xor_cover[u as usize] ^= zobrist[e as usize];
                    xor_cover[v as usize] ^= zobrist[e as usize];
                }
            }
        }

        let mut additional_edges = vec![];
        for path in groups.into_values() {
            if path.len() <= 1 {
                continue;
            }

            for &u in &path {
                associated_cut[parent_edge[u as usize] as usize] = 2;
            }

            let tail = path[0];
            let head = parent[*path.last().unwrap() as usize];
            additional_edges.push([tail, head]);
        }

        let mut conn_3ec = dset::DisjointSet::new(n);
        for e in 0..m {
            if associated_cut[e] != 0 {
                continue;
            }
            let [u, v] = edges[e];
            conn_3ec.merge(u as usize, v as usize);
        }
        for [u, v] in additional_edges {
            conn_3ec.merge(u as usize, v as usize);
        }

        let comps_3ec =
            CSR::from_pairs(n, (0..n).map(|u| (conn_3ec.find_root(u) as u32, u as u32)));
        comps_3ec
    }
}
