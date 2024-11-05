use std::iter::once;

mod simple_io {
    pub struct InputAtOnce<'a> {
        _buf: String,
        iter: std::str::SplitAsciiWhitespace<'a>,
    }

    impl<'a> InputAtOnce<'a> {
        pub fn token(&mut self) -> &'a str {
            self.iter.next().unwrap_or_default()
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().unwrap()
        }
    }

    pub fn stdin_at_once<'a>() -> InputAtOnce<'a> {
        let buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let iter = buf.split_ascii_whitespace();
        let iter = unsafe { std::mem::transmute(iter) };
        InputAtOnce { _buf: buf, iter }
    }
}

mod dlx {
    type Link = [usize; 2];

    #[derive(Debug)]
    struct CircLinkedList {
        links: Vec<Link>,
    }

    impl CircLinkedList {
        fn link(&mut self, idx: usize) {
            let [left, right] = self.links[idx];
            self.links[left][1] = idx;
            self.links[right][0] = idx;
        }

        fn unlink(&mut self, idx: usize) {
            let [left, right] = self.links[idx];
            self.links[left][1] = right;
            self.links[right][0] = left;
        }
    }

    #[derive(Debug)]
    struct Cursor {
        head: usize,
        current: usize,
    }

    impl Cursor {
        fn new(head: usize) -> Self {
            Cursor {
                head,
                current: head,
            }
        }

        fn next(&mut self, list: &CircLinkedList) -> Option<usize> {
            self.current = list.links[self.current][1];
            (self.current != self.head).then(|| self.current)
        }

        fn prev(&mut self, list: &CircLinkedList) -> Option<usize> {
            self.current = list.links[self.current][0];
            (self.current != self.head).then(|| self.current)
        }
    }

    #[derive(Debug)]
    struct ColHead {
        node: usize,
        size: usize,
    }

    pub struct Dlx {
        xs: CircLinkedList,
        ys: CircLinkedList,
        col_heads: Vec<ColHead>,
        col_ids: Vec<usize>,
        row_ids: Vec<usize>,
        last_in_row: Vec<usize>,
    }

    impl Dlx {
        pub fn new(n_cols: usize) -> Self {
            let mut x_links = vec![];
            let mut y_links = vec![];
            let mut col_heads = Vec::with_capacity(n_cols);
            for i in 0..n_cols + 1 {
                let (i_prev, i_next) = ((i + n_cols) % (n_cols + 1), (i + 1) % (n_cols + 1));
                x_links.push([i_prev, i_next]);
                y_links.push([i, i]);
                col_heads.push(ColHead { node: i, size: 0 });
            }
            Dlx {
                xs: CircLinkedList { links: x_links },
                ys: CircLinkedList { links: y_links },
                col_heads,
                col_ids: (0..n_cols + 1).collect(),
                row_ids: vec![0; n_cols + 1],
                last_in_row: vec![0],
            }
        }

        pub fn n_nodes(&self) -> usize {
            self.xs.links.len()
        }

        pub fn n_cols(&self) -> usize {
            self.col_heads.len() - 1
        }

        pub fn n_rows(&self) -> usize {
            self.last_in_row.len()
        }

        pub fn active_cols(&self) -> usize {
            let mut result = 0;
            let mut cursor = Cursor::new(0);
            while let Some(_) = cursor.next(&self.xs) {
                result += 1;
            }
            result
        }

        pub fn push_row<I>(&mut self, row: I)
        where
            I: IntoIterator<Item = usize>,
            I::IntoIter: Clone,
        {
            let row = row.into_iter();

            let row_id = self.n_rows();
            let n_new_nodes = row.clone().count();
            let i_base = self.n_nodes();
            for (i, mut col_id) in row.enumerate() {
                col_id += 1;
                debug_assert!((1 <= col_id) && (col_id <= self.n_cols()));

                let u = self.n_nodes();
                let (i_prev, i_next) = ((i + n_new_nodes - 1) % n_new_nodes, (i + 1) % n_new_nodes);
                let head = self.col_heads[col_id].node;
                self.xs.links.push([i_prev + i_base, i_next + i_base]);
                self.ys.links.push([self.ys.links[head][0], head]);
                self.ys.link(u);
                self.col_heads[col_id].size += 1;
                self.row_ids.push(row_id);
                self.col_ids.push(col_id);
            }
            self.last_in_row.push(self.n_nodes() - 1);

            debug_assert_eq!(self.row_ids.len(), self.n_nodes());
            debug_assert_eq!(self.col_ids.len(), self.n_nodes());
            debug_assert_eq!(self.xs.links.len(), self.n_nodes());
            debug_assert_eq!(self.ys.links.len(), self.n_nodes());
        }

        pub fn pop_row(&mut self) {
            assert!(self.n_rows() >= 2);
            let r = self.n_rows() - 1;
            let indices = self.last_in_row[r - 1] + 1..=self.last_in_row[r];

            for u in indices.clone() {
                self.col_heads[self.col_ids[u]].size -= 1;
                self.ys.unlink(u);
            }
            for _ in indices {
                self.xs.links.pop();
                self.ys.links.pop();
                self.row_ids.pop();
                self.col_ids.pop();
            }
            self.last_in_row.pop();

            debug_assert_eq!(self.row_ids.len(), self.n_nodes());
            debug_assert_eq!(self.col_ids.len(), self.n_nodes());
            debug_assert_eq!(self.xs.links.len(), self.n_nodes());
            debug_assert_eq!(self.ys.links.len(), self.n_nodes());
        }

        pub fn cover(&mut self, col_id: usize) {
            debug_assert!((1..=self.n_cols()).contains(&col_id));

            let col_head = self.col_heads[col_id].node;
            self.xs.unlink(col_head);

            let mut cursor = Cursor::new(col_head);
            while let Some(u) = cursor.next(&self.ys) {
                let mut v_cursor = Cursor::new(u);
                while let Some(v) = v_cursor.next(&self.xs) {
                    self.col_heads[self.col_ids[v]].size -= 1;
                    self.ys.unlink(v);
                }
            }
        }

        pub fn uncover(&mut self, col_id: usize) {
            debug_assert!((1..=self.n_cols()).contains(&col_id));

            let col_head = self.col_heads[col_id].node;
            self.xs.link(col_head);
            let mut cursor = Cursor::new(col_head);
            while let Some(u) = cursor.prev(&self.ys) {
                let mut v_cursor = Cursor::new(u);
                while let Some(v) = v_cursor.prev(&self.xs) {
                    self.col_heads[self.col_ids[v]].size += 1;
                    self.ys.link(v);
                }
            }
        }

        pub fn search(&mut self, solution: &mut Vec<usize>) -> bool {
            if self.xs.links[0][1] == 0 {
                return true;
            }

            let mut min_col = (usize::MAX, 0);
            let mut cursor = Cursor::new(0);
            while let Some(col_id) = cursor.next(&self.xs) {
                debug_assert!((1..=self.n_cols()).contains(&col_id));

                debug_assert_eq!(self.col_heads[col_id].node, col_id);
                debug_assert_eq!(col_id, col_id);
                if self.col_heads[col_id].size < min_col.0 {
                    min_col = (self.col_heads[col_id].size, col_id);
                }
            }
            let (_size, col_id) = min_col;

            self.cover(self.col_ids[col_id]);

            let mut cursor = Cursor::new(col_id);
            while let Some(u) = cursor.next(&self.ys) {
                solution.push(self.row_ids[u] - 1);
                let mut v_cursor = Cursor::new(u);
                while let Some(v) = v_cursor.next(&self.xs) {
                    self.cover(self.col_ids[v]);
                }

                if self.search(solution) {
                    return true;
                }

                solution.pop();
                while let Some(v) = v_cursor.prev(&self.xs) {
                    self.uncover(self.col_ids[v]);
                }
            }

            self.uncover(col_id);
            false
        }
    }

    impl std::fmt::Debug for Dlx {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let h = self.n_rows();
            let w = self.active_cols();

            let mut grid = vec![None; h * w];
            let mut i = 0;
            let mut cursor = Cursor::new(0);
            while let Some(u) = cursor.next(&self.xs) {
                let mut current = u;
                loop {
                    let row_id = self.row_ids[current];
                    grid[row_id * w + i] = Some(current);
                    current = self.ys.links[current][1];
                    if current == u {
                        break;
                    }
                }
                i += 1;
            }

            for i in 0..h * 3 {
                for j in 0..w {
                    if let Some(idx) = grid[i / 3 * w + j] {
                        let [left, right] = self.xs.links[idx];
                        let [up, down] = self.ys.links[idx];
                        match i % 3 {
                            0 => write!(f, "    {:2}    ", up)?,
                            1 => write!(f, " {:2}|{:2}|{:<2} ", left, idx, right)?,
                            2 => write!(f, "    {:2}    ", down)?,
                            _ => unreachable!(),
                        }
                    } else {
                        write!(f, "          ")?;
                    }
                }
                writeln!(f)?;
            }
            Ok(())
        }
    }
}

fn main() {
    let mut input = simple_io::stdin_at_once();

    let n = 3;
    let nsq = n * n;
    let n4 = nsq * nsq;

    // let grid: Vec<u8> = (0..nsq)
    //     .flat_map(|_| input.token().as_bytes()[..nsq].to_vec())
    //     .collect();
    let grid: Vec<u8> = (0..nsq * nsq)
        .map(|_| input.token().as_bytes()[0])
        .collect();

    let mut dlx = dlx::Dlx::new(n4 * 4);
    let mut row_labels = vec![];
    for i in 0..nsq {
        for j in 0..nsq {
            let c = grid[i * nsq + j];
            let mut rule = |i, j, value| {
                let value = value as usize;

                row_labels.push((i, j, value));
                dlx.push_row(
                    once(n4 * 0 + i * nsq + j)
                        .chain(once(n4 * 1 + i * nsq + value))
                        .chain(once(n4 * 2 + j * nsq + value))
                        .chain(once(n4 * 3 + (i / n * n + j / n) * nsq + value)),
                );
            };
            match c {
                b'1'..=b'9' => rule(i, j, c - b'1'),
                _ => {
                    for value in 0..nsq as u8 {
                        rule(i, j, value);
                    }
                }
            }
        }
    }

    let mut solution = vec![];
    dlx.search(&mut solution);

    let mut grid = vec![b'-'; n4];
    for &u in &solution {
        let (i, j, value) = row_labels[u];
        grid[i * nsq + j] = value as u8 + b'1';
    }

    for i in 0..nsq {
        for j in 0..nsq {
            print!("{} ", grid[i * nsq + j] as char);
        }
        println!();
    }
}
