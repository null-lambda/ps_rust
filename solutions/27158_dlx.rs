use std::{collections::HashMap, iter::once};

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

#[allow(dead_code)]
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

#[derive(Debug, Eq, PartialEq, Ord, PartialOrd, Clone)]
struct Piece {
    name: &'static str,
    h: usize,
    w: usize,
    dpos: Vec<(usize, usize)>,
}

impl Piece {
    fn new(name: &'static str, pattern: &str) -> Self {
        let rows = || pattern.split('/');
        let h = rows().count();
        let w = rows().map(|t| t.len()).max().unwrap();

        let mut dpos = vec![];
        for (i, row) in rows().enumerate() {
            for (j, c) in row.chars().enumerate() {
                if c == '#' {
                    dpos.push((i, j));
                }
            }
        }
        Self { h, w, dpos, name }
    }

    fn rotate(&self) -> Self {
        let mut dpos = vec![];
        for (i, j) in &self.dpos {
            dpos.push((self.w - 1 - j, *i));
        }
        dpos.sort_unstable();
        Self {
            h: self.w,
            w: self.h,
            dpos,
            name: self.name,
        }
    }

    fn flip(&self) -> Self {
        let mut dpos = vec![];
        for (i, j) in &self.dpos {
            dpos.push((self.h - 1 - i, *j));
        }
        dpos.sort_unstable();
        Self {
            h: self.h,
            w: self.w,
            dpos,
            name: self.name,
        }
    }
}

fn polynominoes() -> Vec<Piece> {
    let original = [
        ("F", ".##/##/.#"),
        ("I", "#####"),
        ("L", "####/#"),
        ("N", "###/..##"),
        ("P", "###/##"),
        ("T", "###/.#/.#"),
        ("U", "###/#.#"),
        ("V", "###/#/#"),
        ("W", "#/##/.##"),
        ("X", ".#/###/.#"),
        ("Y", "####/.#"),
        ("Z", "#/###/..#"),
    ]
    .into_iter()
    .map(|(name, pattern)| Piece::new(name, pattern))
    .collect::<Vec<_>>();

    let mut transformed = vec![];
    for mut piece in original {
        for _ in 0..4 {
            transformed.push(piece.clone());
            transformed.push(piece.clone().flip());
            piece = piece.rotate();
        }
    }
    transformed.sort_unstable();
    transformed.dedup();
    transformed
}

fn main() {
    let mut input = simple_io::stdin_at_once();

    let pieces = polynominoes();

    let n: usize = input.value();
    let m: usize = input.value();
    let grid: Vec<bool> = (0..n * m).map(|_| input.token() == "1").collect();

    let filled: Vec<usize> = (0..n * m).filter(|&i| grid[i]).collect();
    let mut name_to_idx = HashMap::new();
    let mut coord_to_col = HashMap::new();

    for i in 0..n {
        for j in 0..m {
            if grid[i * m + j] {
                continue;
            }
            let idx = coord_to_col.len();
            coord_to_col.insert((i, j), 12 + idx);
        }
    }

    let mut dlx = dlx::Dlx::new(12 + coord_to_col.len());
    let mut labels = vec![];
    for (label, piece) in pieces.iter().enumerate() {
        let idx = if let Some(&idx) = name_to_idx.get(&piece.name) {
            idx
        } else {
            let idx = name_to_idx.len();
            name_to_idx.insert(piece.name, idx);
            idx
        };

        for i in piece.h - 1..n {
            for j in piece.w - 1..m {
                let collision = piece
                    .dpos
                    .iter()
                    .any(|(di, dj)| grid[(i - di) * m + j - dj]);
                if collision {
                    continue;
                }
                dlx.push_row(
                    once(idx).chain(
                        piece
                            .dpos
                            .iter()
                            .map(|(di, dj)| coord_to_col[&(i - di, j - dj)]),
                    ),
                );
                labels.push((label, i, j));
            }
        }
    }

    let mut solution = vec![];
    dlx.search(&mut solution);

    let mut canvas = vec![vec![b'1'; m]; n];
    for &idx in &solution {
        let (label, i, j) = labels[idx];
        let piece = &pieces[label];
        for &(di, dj) in &piece.dpos {
            canvas[i - di][j - dj] = piece.name.bytes().next().unwrap();
        }
    }

    for row in canvas {
        for c in row {
            print!("{} ", c as char);
        }
        println!();
    }
    println!();
}
