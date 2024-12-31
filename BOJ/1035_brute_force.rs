use std::{collections::HashSet, io::Write, iter};

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

fn next_permutation<T: Ord>(arr: &mut [T]) -> bool {
    match arr.windows(2).rposition(|w| w[0] < w[1]) {
        Some(i) => {
            let j = i + arr[i + 1..].partition_point(|x| &arr[i] < x);
            arr.swap(i, j);
            arr[i + 1..].reverse();
            true
        }
        None => {
            arr.reverse();
            false
        }
    }
}

type BitSet = u32;
fn gen_configurations(n_pieces: usize) -> Vec<BitSet> {
    // place 5 pices on 5x5 grid
    assert!(n_pieces >= 1);

    let mut current: HashSet<BitSet> = (0..25).map(|u| 1 << u).collect();
    let neighbors = |u: usize| {
        let (i, j) = (u / 5, u % 5);
        iter::empty()
            .chain((i > 0).then(|| u - 5))
            .chain((i < 5 - 1).then(|| u + 5))
            .chain((j > 0).then(|| u - 1))
            .chain((j < 5 - 1).then(|| u + 1))
    };

    for _ in 1..n_pieces {
        current = current
            .into_iter()
            .flat_map(|grid| {
                (0..25)
                    .filter(move |&u| {
                        (grid >> u) & 1 == 0 && neighbors(u).any(|v| (grid >> v) & 1 == 1)
                    })
                    .map(move |u| grid | (1 << u))
            })
            .collect();
    }

    current.into_iter().collect()
}

type Point = (u8, u8);
fn bitset_to_points(grid: BitSet) -> Vec<Point> {
    (0..25)
        .filter(|u| (grid >> u) & 1 == 1)
        .map(|u| (u as u8 / 5, u as u8 % 5))
        .collect()
}

fn dist_l1(p: Point, q: Point) -> u8 {
    let (x1, y1) = p;
    let (x2, y2) = q;
    ((x1 as i8 - x2 as i8).abs() + (y1 as i8 - y2 as i8).abs()) as u8
}

fn dist(u: &Vec<Point>, v: &Vec<Point>) -> u8 {
    debug_assert_eq!(u.len(), v.len());
    let n = u.len();
    let mut perm: Vec<_> = (0..n).collect();

    let mut ans = u8::MAX;
    loop {
        ans = ans.min((0..n).map(|i| dist_l1(u[i], v[perm[i]])).sum());
        if !next_permutation(&mut perm) {
            break;
        }
    }

    ans
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let grid = (0..5)
        .flat_map(|_| input.token().bytes().take(5))
        .fold(0, |acc, c| acc << 1 | (c == b'*') as BitSet);
    let n_pieces = (0..25).filter(|u| (grid >> u) & 1 == 1).count();

    let grid = bitset_to_points(grid);
    let configurations = gen_configurations(n_pieces);

    let ans = configurations
        .iter()
        .map(|c| dist(&grid, &bitset_to_points(*c)))
        .min()
        .unwrap();

    writeln!(output, "{}", ans).unwrap();
}
