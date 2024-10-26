mod io {
    use std::fmt::Debug;
    use std::str::*;

    pub trait InputStream {
        fn token(&mut self) -> &[u8];
        fn line(&mut self) -> &[u8];

        fn skip_line(&mut self) {
            self.line();
        }

        #[inline]
        fn value<T>(&mut self) -> T
        where
            T: FromStr,
            T::Err: Debug,
        {
            let token = self.token();
            let token = unsafe { from_utf8_unchecked(token) };
            token.parse::<T>().unwrap()
        }
    }

    #[inline]
    fn is_whitespace(c: u8) -> bool {
        c <= b' '
    }

    fn trim_newline(s: &[u8]) -> &[u8] {
        let mut s = s;
        while s
            .last()
            .map(|&c| {
                matches! {c, b'\n' | b'\r' | 0}
            })
            .unwrap_or_else(|| false)
        {
            s = &s[..s.len() - 1];
        }
        s
    }

    impl InputStream for &[u8] {
        fn token(&mut self) -> &[u8] {
            let idx = self.iter().position(|&c| !is_whitespace(c)).unwrap();
            //.expect("no available tokens left");
            *self = &self[idx..];
            let idx = self
                .iter()
                .position(|&c| is_whitespace(c))
                .unwrap_or_else(|| self.len());
            let (token, buf_new) = self.split_at(idx);
            *self = buf_new;
            token
        }

        fn line(&mut self) -> &[u8] {
            let idx = self
                .iter()
                .position(|&c| c == b'\n')
                .map(|idx| idx + 1)
                .unwrap_or_else(|| self.len());
            let (line, buf_new) = self.split_at(idx);
            *self = buf_new;
            trim_newline(line)
        }
    }
}

use std::io::{BufReader, Read, Write};

fn stdin() -> Vec<u8> {
    let stdin = std::io::stdin();
    let mut reader = BufReader::new(stdin.lock());

    let mut input_buf: Vec<u8> = vec![];
    reader.read_to_end(&mut input_buf).unwrap();
    input_buf
}

struct SegmentTree {
    n: usize,
    n_segs: Vec<i32>,
    sum: Vec<u32>,
    seg_len: Vec<u32>,
}

impl SegmentTree {
    fn new(n: usize, xs: Vec<u32>) -> Self {
        use std::iter::repeat;
        let n_segs = vec![0; 2 * n];
        let sum = vec![0; 2 * n];
        let mut seg_len: Vec<u32> = repeat(0)
            .take(n)
            .chain(xs.windows(2).map(|t| t[1] - t[0]))
            .collect();
        seg_len.resize(2 * n, 0);

        for i in (1..n).rev() {
            seg_len[i] = seg_len[i << 1] + seg_len[i << 1 | 1];
        }
        Self {
            n,
            n_segs,
            sum,
            seg_len,
        }
    }

    fn apply(&mut self, i: usize, value: i32) {
        self.n_segs[i] += value;
    }

    fn update_sum(&mut self, i: usize) {
        self.sum[i] = if self.n_segs[i] > 0 {
            self.seg_len[i]
        } else if i < self.n {
            self.sum[i << 1] + self.sum[i << 1 | 1usize]
        } else {
            0
        };
    }

    // add on interval [left, right)
    fn add_range(&mut self, [left, right]: [usize; 2], value: i32) {
        debug_assert!(right <= self.n);

        let (mut start, mut end) = (left + self.n, right + self.n);
        let (mut update_left, mut update_right) = (false, false);
        while start < end {
            if update_left {
                self.update_sum(start - 1);
            }
            if update_right {
                self.update_sum(end);
            }
            if start & 1 != 0 {
                self.apply(start, value);
                self.update_sum(start);
                update_left = true;
            }
            if end & 1 != 0 {
                self.apply(end - 1, value);
                self.update_sum(end - 1);
                update_right = true;
            }

            start = (start + 1) >> 1;
            end = end >> 1;
        }

        start -= 1;
        while end > 0 {
            if update_left {
                self.update_sum(start);
            }
            if update_right && !(update_left && start == end) {
                self.update_sum(end);
            }
            start >>= 1;
            end >>= 1;
        }
    }

    fn sum(&self) -> u32 {
        self.sum[1]
    }
}

#[allow(dead_code)]
fn main() {
    use io::InputStream;
    let input_buf = stdin();
    let mut input: &[u8] = &input_buf[..];

    // let mut output_buf = Vec::<u8>::new();

    let n = input.value();
    let mut xs = Vec::new();
    let mut segments: Vec<[[u32; 2]; 2]> = (0..n)
        .map(|i| {
            let x1 = input.value();
            let x2 = input.value();
            let y1 = input.value();
            let y2 = input.value();
            xs.push((x1, 2 * i));
            xs.push((x2, 2 * i + 1));
            [[x1, x2], [y1, y2]]
        })
        .collect();

    // x coordinate compression
    xs.sort_unstable();

    let mut xs2 = Vec::new();
    for (order, (x, i)) in xs.into_iter().enumerate() {
        segments[i / 2][0][i % 2] = order as u32;
        xs2.push(x);
    }
    let xs = xs2;
    let nx = xs.len();

    // sweeping by y coordinate
    #[derive(Debug, Copy, Clone)]
    enum EventType {
        Start = 1,
        End = 0,
    }
    use EventType::*;

    let mut events = Vec::new();
    for [[x1, x2], [y1, y2]] in segments.into_iter() {
        events.push((Start, y1, [x1, x2]));
        events.push((End, y2, [x1, x2]));
    }
    events.sort_unstable_by_key(|&(event_type, y, _)| (y, event_type as usize));

    let mut segtree = SegmentTree::new(nx, xs);
    let mut total_area: u64 = 0;
    let mut y_prev = 0;
    for (event_type, y, [x1, x2]) in events.into_iter() {
        if y > y_prev {
            let cross_section = segtree.sum();
            total_area += (y - y_prev) as u64 * cross_section as u64;
        }
        match event_type {
            Start => {
                segtree.add_range([x1 as usize, x2 as usize], 1);
            }
            End => {
                segtree.add_range([x1 as usize, x2 as usize], -1);
            }
        }

        y_prev = y;
    }

    println!("{}", total_area);

    // std::io::stdout().write_all(&output_buf[..]).unwrap();
}
