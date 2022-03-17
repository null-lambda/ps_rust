struct DisjointSet {
    parent: Vec<usize>,
    size: Vec<usize>,
}

impl DisjointSet {
    fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            size: std::iter::repeat(1).take(n).collect(),
        }
    }

    fn find_root(&mut self, u: usize) -> usize {
        if u == self.parent[u] {
            u
        } else {
            self.parent[u] = self.find_root(self.parent[u]);
            self.parent[u]
        }
    }

    fn merge(&mut self, mut u: usize, mut v: usize) {
        u = self.find_root(u);
        v = self.find_root(v);
        if u != v {
            if self.size[u] > self.size[v] {
                std::mem::swap(&mut u, &mut v);
            }
            self.parent[v] = u;
            self.size[u] += self.size[v];
        }
    }
}
