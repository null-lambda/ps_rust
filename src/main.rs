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

fn main() {
    use io::InputStream;
    let input_buf = stdin();
    let mut input: &[u8] = &input_buf[..];

    // let mut output_buf = Vec::<u8>::new();

    // goal: https://www.acmicpc.net/problem/1046?
    let (n, m) = {
        let mut line = input.line();
        (line.value(), line.value())
    };
    let grid: Vec<Vec<u8>> = (0..n).map(|_| input.line()[0..m].to_vec()).collect();

    // upscale grid coordinates by 2,
    // and set origin to the light source
    let (source_x, source_y) = (0..n)
        .flat_map(|j| (0..m).map(move |i| (i, j)))
        .find(|&(i, j)| grid[j][i] == b'*')
        .map(|(i, j)| (2 * i as i32 + 1, 2 * j as i32 + 1))
        .unwrap();

    let coord_transform = |x, y| (2 * (x as i32) - source_x, 2 * (y as i32) - source_y);
    let (bx0, by0) = coord_transform(0, 0);
    let (bx1, by1) = coord_transform(m, n);

    #[derive(Clone, Copy, Debug, PartialEq, Eq)]
    struct Point {
        x: i32,
        y: i32,
        ordinal: u32,
    }

    impl Point {
        fn new(x: i32, y: i32) -> Self {
            Self {
                x,
                y,
                ordinal: {
                    let angle = (y as f32).atan2(x as f32);
                    // rotate -45 deg, to ensure that no segment cross the coordinate axis.
                    let angle_scaled = (angle * (2.0 * PI).recip() + 0.25 + 1e-7).rem_euclid(1.0);
                    // integer representation for efficient comparsion
                    // (float type does not implement Ord)
                    (angle_scaled * 50_000.0) as u32
                },
            }
        }
    }

    impl From<(i32, i32)> for Point {
        fn from((x, y): (i32, i32)) -> Point {
            Point::new(x, y)
        }
    }

    type Segment = (Point, Point);
    let mut segments: Vec<Segment> = vec![];

    let mut add_rect = |(x1, y1): (i32, i32), (x2, y2): (i32, i32)| {
        use std::cmp::Ordering::*;
        segments.extend(
            [
                ((x1, y1), (x1, y2)),
                ((x2, y1), (x2, y2)),
                ((x1, y1), (x2, y1)),
                ((x1, y2), (x2, y2)),
            ]
            .iter()
            .flat_map(|&(s, e)| match (s.0 * e.1 - s.1 * e.0).cmp(&0) {
                Greater => Some((s, e)),
                Less => Some((e, s)),
                _ => unsafe { std::hint::unreachable_unchecked() },
            })
            .map(|(s, e)| (s.into(), e.into())),
        );
    };

    // convert walls to segments, with ccw ordering
    for j in 0..n {
        for i in 0..m {
            if grid[j][i] == b'#' {
                let (x, y) = coord_transform(i, j);
                add_rect((x, y), (x + 2, y + 2));
            }
        }
    }

    // add boundary
    add_rect((bx0, by0), (bx1, by1));

    drop(add_rect);
    drop(grid);

    use std::cmp::{Ordering, Reverse};
    use std::f32::consts::PI;

    use std::collections::BinaryHeap;

    type TruncatedSegment<'a> = (&'a Segment, Option<Box<Point>>);

    #[derive(Clone, PartialEq, Eq, Debug)]
    struct Idx<'a>(u32, TruncatedSegment<'a>);

    impl<'a> PartialOrd for Idx<'a> {
        fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
            Some(self.0.cmp(&other.0))
        }
    }

    impl<'a> Ord for Idx<'a> {
        fn cmp(&self, other: &Self) -> Ordering {
            self.partial_cmp(other).unwrap()
        }
    }

    let mut segments_queue: BinaryHeap<_> = segments
        .iter()
        .map(|seg| Reverse(Idx(seg.0.ordinal, (seg, None))))
        .collect();
    println!(
        "{:?}",
        segments_queue
            .iter()
            .map(|n| n.0 .1.clone())
            .collect::<Vec<_>>()
    );

    assert!(segments_queue.len() >= 4);
    let (mut prev, mut prev_trunc) = segments_queue.pop().unwrap().0 .1;
    while let Some((seg, trunc)) = segments_queue.pop().map(|x| x.0 .1) {
        use std::cmp::Ordering::*;
        let (s, e) = seg;

        let prev_finished = prev.1.ordinal
            <= trunc
                .as_ref()
                .map_or_else(|| seg.0.ordinal, |trunc| trunc.ordinal);
        if prev_finished {
            println!("finished {:?}", (prev, trunc.as_ref().clone()));
            prev = seg;
            prev_trunc = trunc;
            continue;
        }

        let at_front = if prev.0.y == prev.1.y {
            (seg.0.y + seg.1.y).abs() < prev.0.y.abs() * 2
        } else {
            (seg.0.x + seg.1.x).abs() < prev.0.x.abs() * 2
        };

        let overlap_state = prev.1.ordinal.cmp(&seg.1.ordinal);
        match (overlap_state, at_front) {
            (Less, false) => {
                println!("back half {:?} {:?}", (prev.0, prev.1), seg);
                segments_queue.push(Reverse(Idx(prev.1.ordinal, (seg, Some(Box::new(prev.1))))));
            }
            (Equal, false) | (Greater, false) => {}
            (Less, true) | (Equal, true) => {
                println!(
                    "front half {:?} with end trunc {:?}",
                    (prev.0, prev.1),
                    seg.0
                );
                prev = seg;
                prev_trunc = trunc;
            }
            (Greater, true) => {
                println!(
                    "front half {:?} with end trunc {:?}",
                    (prev.0, prev.1),
                    seg.0
                );
                segments_queue.push(Reverse(Idx(seg.1.ordinal, (prev, Some(Box::new(seg.1))))));
                prev = seg;
                prev_trunc = trunc;
            }
        }
    }

    // println!("{}", result);

    // std::io::stdout().write_all(&output_buf[..]).unwrap();
}
