mod linked_list {
    // Arena-allocated pool of cyclic doubly linked lists,
    // represented by a directed permutation graph.
    #[derive(Clone, Debug)]
    pub struct CyclicListPool {
        links: Vec<[u32; 2]>,
    }

    impl CyclicListPool {
        pub fn new(n_nodes: usize) -> Self {
            Self {
                links: (0..n_nodes as u32).map(|u| [u, u]).collect(),
            }
        }

        pub fn links(&self, u: u32) -> [u32; 2] {
            self.links[u as usize]
        }

        pub fn next(&self, u: u32) -> u32 {
            self.links[u as usize][1]
        }

        pub fn is_isolated(&self, u: u32) -> bool {
            self.links[u as usize][0] == u
        }

        pub fn isolate(&mut self, u: u32) {
            let [a, b] = self.links[u as usize];
            if a == u {
                return;
            }

            self.links[a as usize][1] = b;
            self.links[b as usize][0] = a;
            self.links[u as usize] = [u, u];
        }

        pub fn is_connected(&self, u: u32, v: u32) -> bool {
            let mut c = u;
            while c != v {
                c = self.links[c as usize][1];
                if c == u {
                    return false;
                }
            }
            true
        }

        pub fn insert_left(&mut self, pivot: u32, u: u32) {
            debug_assert!(self.is_isolated(u));
            let [a, _] = self.links[pivot as usize];
            self.links[a as usize][1] = u;
            self.links[u as usize] = [a, pivot];
            self.links[pivot as usize][0] = u;
        }

        pub fn insert_right(&mut self, pivot: u32, u: u32) {
            debug_assert!(self.is_isolated(u));
            let [_, a] = self.links[pivot as usize];
            self.links[pivot as usize][1] = u;
            self.links[u as usize] = [pivot, a];
            self.links[a as usize][0] = u;
        }

        pub fn split_slice_out(&mut self, head: u32, tail: u32) {
            debug_assert!(self.is_connected(head, tail));

            let [a, _] = self.links[head as usize];
            let [_, b] = self.links[tail as usize];
            if a == tail {
                return;
            }

            self.links[a as usize][1] = b;
            self.links[b as usize][0] = a;
            self.links[tail as usize][1] = head;
            self.links[head as usize][0] = tail;
        }

        pub fn pop_left(&mut self, u: u32) -> Option<u32> {
            if self.is_isolated(u) {
                return None;
            }

            let [a, _] = self.links[u as usize];
            let [b, _] = self.links[a as usize];

            self.links[b as usize][1] = u;
            self.links[a as usize] = [a; 2];
            self.links[u as usize][0] = b;

            Some(a)
        }

        pub fn pop_right(&mut self, u: u32) -> Option<u32> {
            if self.is_isolated(u) {
                return None;
            }

            let [_, a] = self.links[u as usize];
            let [_, b] = self.links[a as usize];

            self.links[u as usize][1] = b;
            self.links[a as usize] = [a; 2];
            self.links[b as usize][0] = u;

            Some(a)
        }
    }
}
