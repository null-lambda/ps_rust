mod kd_tree {
    // 2D static KD-tree, for nearest neighbor query
    // - Robust (no sensitive geometric predicates, except the metric)
    // - Best only for nearly-uniform distribution. Worst case O(N)

    const UNSET: u32 = !0;

    type T = i32;
    type D = i64;
    const INF_DIST_SQ: D = 1e18 as D;
    const POINT_AT_INFINITY: [T; 2] = [-1e9 as T, -1e9 as T];

    // L2 norm
    fn dist_sq(p: [T; 2], q: [T; 2]) -> D {
        let dr: [_; 2] = std::array::from_fn(|i| (p[i] - q[i]) as D);
        dr[0] * dr[0] + dr[1] * dr[1]
    }

    // No child pointers (same indexing scheme as static segment trees)
    #[derive(Clone, Debug)]
    pub struct Node {
        pub point: [T; 2],
        // pub idx: u32,
    }

    pub struct KDTree {
        pub nodes: Vec<Node>,
    }

    impl KDTree {
        pub fn from_iter(points: impl IntoIterator<Item = [T; 2]>) -> Self {
            let mut points = points
                .into_iter()
                .enumerate()
                .map(|(idx, point)| Node {
                    point,
                    // idx: idx as u32,
                })
                .collect::<Vec<_>>();
            let n = points.len() + 1;

            let dummy = Node {
                point: POINT_AT_INFINITY,
                // idx: UNSET,
            };
            points.resize(n - 1, dummy.clone());

            let mut this = Self {
                nodes: vec![dummy; n],
            };
            this.build_rec(1, &mut points, 0);
            this
        }

        fn build_rec(&mut self, u: usize, ps: &mut [Node], depth: usize) {
            if ps.is_empty() {
                return;
            }

            let left_size = {
                let n = ps.len();
                let p = (n + 1).next_power_of_two() / 2;
                (n - p / 2).min(p - 1)
            };
            let (left, pivot, right) = ps.select_nth_unstable_by_key(left_size, |p| p.point[depth]);
            self.nodes[u] = pivot.clone();

            self.build_rec(u << 1, left, depth ^ 1);
            self.build_rec(u << 1 | 1, right, depth ^ 1);
        }

        pub fn find_nn_except_self(&self, p: [T; 2]) -> (D, &Node) {
            let mut res = (INF_DIST_SQ, 0);
            self.find_nn_except_self_rec(p, &mut res, 1, 0);
            (res.0, &self.nodes[res.1 as usize])
        }

        fn find_nn_except_self_rec(&self, p: [T; 2], opt: &mut (D, u32), u: usize, depth: usize) {
            if u >= self.nodes.len() {
                return;
            }

            let d_sq = dist_sq(self.nodes[u].point, p);
            if d_sq != 0 && d_sq < opt.0 {
                *opt = (d_sq, u as u32);
            }

            let d_ax = p[depth] - self.nodes[u].point[depth];
            let branch = (d_ax > 0) as usize;

            self.find_nn_except_self_rec(p, opt, u << 1 | branch, depth ^ 1);
            if (d_ax as D * d_ax as D) < opt.0 {
                self.find_nn_except_self_rec(p, opt, u << 1 | branch ^ 1, depth ^ 1);
            }
        }
    }
}
