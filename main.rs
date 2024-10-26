use std::io::Write;

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

fn dijkstra(nieghbors: &Vec<Vec<(u32, u32)>>, start: u32) -> Vec<u32> {
    let mut dist = vec![i32::MAX as u32; nieghbors.len()];
    let mut pq: BinaryHeap<_> = [Reverse((0, start))].into();
    dist[start as usize] = 0;

    while let Some(Reverse((d, d_u))) = pq.pop() {
        if dist[d_u as usize] < d {
            continue;
        }
        for &(v, d_uv) in &nieghbors[d_u as usize] {
            let d_v_new = d + d_uv;
            if dist[v as usize] > d_v_new {
                dist[v as usize] = d_v_new;
                pq.push(Reverse((d_v_new, v)));
            }
        }
    }
    dist
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
}
