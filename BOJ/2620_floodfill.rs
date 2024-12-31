use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap, VecDeque},
    hash::Hash,
    io::Write,
    usize,
};

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
        let _buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let iter = _buf.split_ascii_whitespace();
        let iter = unsafe { std::mem::transmute(iter) };
        InputAtOnce { _buf, iter }
    }

    pub fn stdout() -> std::io::BufWriter<std::io::Stdout> {
        std::io::BufWriter::new(std::io::stdout())
    }
}

fn compress_coord<T: Ord + Clone + Hash>(
    xs: impl IntoIterator<Item = T>,
) -> (Vec<T>, HashMap<T, u32>) {
    let mut x_map: Vec<T> = xs.into_iter().collect();
    x_map.sort_unstable();
    x_map.dedup();

    let x_map_inv = x_map
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, x)| (x, i as u32))
        .collect();

    (x_map, x_map_inv)
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let ps: Vec<(i64, i64)> = (0..n).map(|_| (input.value(), input.value())).collect();

    let (x_map, x_map_inv) = compress_coord(ps.iter().map(|&(x, _)| x));
    let (y_map, y_map_inv) = compress_coord(ps.iter().map(|&(_, y)| y));

    let ps_compressed: Vec<(u32, u32)> = ps
        .iter()
        .map(|&(x, y)| (x_map_inv[&x] + 1, y_map_inv[&y] + 1))
        .collect();
    let h = y_map_inv.len() + 2;
    let w = x_map_inv.len() + 2;

    let mut wall_h = vec![false; h * w];
    let mut wall_v = vec![false; h * w];

    // Create boundary walls
    for i in 0..h {
        wall_v[i * w] = true;
        wall_v[i * w + w - 1] = true;
    }
    for j in 0..w {
        wall_h[j] = true;
        wall_h[(h - 1) * w + j] = true;
    }

    // Create internal walls
    let edges = ps_compressed
        .iter()
        .zip(ps_compressed.iter().cycle().skip(1));
    for (&(x1, y1), &(x2, y2)) in edges {
        if x1 == x2 {
            for y in y1.min(y2)..y1.max(y2) {
                wall_v[y as usize * w + x1 as usize] = true;
            }
        } else {
            for x in x1.min(x2)..x1.max(x2) {
                wall_h[y1 as usize * w + x as usize] = true;
            }
        }
    }

    // flood fill and measure area of all regions
    let mut visited = vec![false; h * w];
    fn flood_fill(
        h: usize,
        w: usize,
        x_map: &[i64],
        y_map: &[i64],
        wall_h: &[bool],
        wall_v: &[bool],
        visited: &mut [bool],
        measure_area: bool,
        start: (usize, usize),
    ) -> i64 {
        let start = start.0 * w + start.1;
        let mut queue: VecDeque<usize> = [start].into();

        let mut area = 0;
        while let Some(u) = queue.pop_front() {
            if visited[u] {
                continue;
            }

            let (y, x) = (u / w, u % w);
            if measure_area {
                area += (x_map[x] - x_map[x - 1]) * (y_map[y] - y_map[y - 1]);
            }

            visited[u] = true;
            if !wall_h[u] && !visited[u - w] {
                queue.push_back(u - w);
            }
            if !wall_h[u + w] && !visited[u + w] {
                queue.push_back(u + w);
            }
            if !wall_v[u] && !visited[u - 1] {
                queue.push_back(u - 1);
            }
            if !wall_v[u + 1] && !visited[u + 1] {
                queue.push_back(u + 1);
            }
        }
        area
    }

    let mut ans = 0;
    flood_fill(
        h,
        w,
        &x_map,
        &y_map,
        &wall_h,
        &wall_v,
        &mut visited,
        false,
        (0, 0),
    );
    for i in 1..h - 1 {
        for j in 1..w - 1 {
            if visited[i * w + j] {
                continue;
            }
            ans = ans.max(flood_fill(
                h,
                w,
                &x_map,
                &y_map,
                &wall_h,
                &wall_v,
                &mut visited,
                true,
                (i, j),
            ));
        }
    }

    writeln!(output, "{}", ans).unwrap();
}
