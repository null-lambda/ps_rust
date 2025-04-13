const UNSET: u32 = !0;
fn bipartite_match(n: usize, m: usize, neighbors: &impl jagged::Jagged<u32>) -> [Vec<u32>; 2] {
    // Hopcroft-Karp
    const INF: u32 = u32::MAX / 2;

    let mut assignment = [vec![UNSET; n], vec![UNSET; m]];
    let mut left_level = vec![INF; n];
    let mut queue = std::collections::VecDeque::new();
    loop {
        left_level.fill(INF);
        queue.clear();
        for u in 0..n {
            if assignment[0][u] == UNSET {
                queue.push_back(u as u32);
                left_level[u] = 0;
            }
        }

        while let Some(u) = queue.pop_front() {
            for &v in &neighbors[u as usize] {
                let w = assignment[1][v as usize];
                if w == UNSET || left_level[w as usize] != INF {
                    continue;
                }
                left_level[w as usize] = left_level[u as usize] + 1;
                queue.push_back(w);
            }
        }

        fn dfs(
            u: u32,
            neighbors: &impl jagged::Jagged<u32>,
            assignment: &mut [Vec<u32>; 2],
            left_level: &Vec<u32>,
        ) -> bool {
            for &v in &neighbors[u as usize] {
                let w = assignment[1][v as usize];
                if w == UNSET
                    || left_level[w as usize] == left_level[u as usize] + 1
                        && dfs(w, neighbors, assignment, left_level)
                {
                    assignment[0][u as usize] = v as u32;
                    assignment[1][v as usize] = u;
                    return true;
                }
            }
            false
        }

        let mut found_augmenting_path = false;
        for u in 0..n {
            if assignment[0][u] == UNSET
                && dfs(u as u32, neighbors, &mut assignment, &mut left_level)
            {
                found_augmenting_path = true;
            }
        }

        if !found_augmenting_path {
            break;
        }
    }

    assignment
}
