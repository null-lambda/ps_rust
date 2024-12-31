use std::{io::Write, iter, ops::Range};

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

struct FenwickTreeND {
    n: usize,
    k: usize,
    data: Vec<i64>,
}

type Point = Vec<usize>;
type BBox = Vec<Range<usize>>;

impl FenwickTreeND {
    fn with_size(n: usize, k: usize, iter: impl IntoIterator<Item = i64>) -> Self {
        let data = vec![0; n.next_power_of_two().pow(k as u32)];
        let mut res = FenwickTreeND { n, k, data };

        let mut iter = iter.into_iter();
        // recur
        fn rec(
            res: &mut FenwickTreeND,
            n: usize,
            k: usize,
            iter: &mut impl Iterator<Item = i64>,
            i: &mut Point,
        ) {
            if k == 0 {
                res.add(i, iter.next().unwrap());
                return;
            }
            for x in 0..n {
                i.push(x);
                rec(res, n, k - 1, iter, i);
                i.pop();
            }
        }
        rec(&mut res, n, k, &mut iter, &mut vec![]);
        res
    }

    fn add(&mut self, i: &Point, value: i64) {
        self.add_rec(&i, value, 0, 0);
    }

    fn add_rec(&mut self, i: &Point, value: i64, axis: usize, coord: usize) {
        if axis == self.k {
            self.data[coord] += value;
            return;
        }

        let mut j = i[axis];
        while j < self.n {
            self.add_rec(i, value, axis + 1, coord * self.n + j);
            j |= j + 1;
        }
    }

    fn get(&self, i: &Point) -> i64 {
        self.sum_range(&i.iter().cloned().map(|x| x..x + 1).collect())
    }

    fn sum_range(&self, range: &BBox) -> i64 {
        self.sum_range_rec(&range, 0, 0)
    }

    fn sum_range_rec(&self, range: &BBox, axis: usize, coord: usize) -> i64 {
        if axis == self.k {
            return self.data[coord];
        }
        let mut r = range[axis].end;
        let mut res = 0;
        while r > 0 {
            res += self.sum_range_rec(range, axis + 1, coord * self.n + r - 1);
            r = r & (r - 1);
        }
        let mut l = range[axis].start;
        while l > 0 {
            res -= self.sum_range_rec(range, axis + 1, coord * self.n + l - 1);
            l = l & (l - 1);
        }
        res
    }
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let k: usize = input.value();
    let q: usize = input.value();

    let mut tree = FenwickTreeND::with_size(n, k, (0..n.pow(k as u32)).map(|_| input.value()));

    for _ in 0..q {
        match input.token() {
            "1" => {
                let mut range = vec![];
                for _ in 0..k {
                    let s: usize = input.value();
                    let e: usize = input.value();
                    range.push(s - 1..e);
                }
                writeln!(output, "{}", tree.sum_range(&range)).unwrap();
            }
            "2" => {
                let point = (0..k).map(|_| input.value::<usize>() - 1).collect();
                let old = tree.get(&point);
                let new: i64 = input.value();
                tree.add(&point, new - old);
            }
            _ => panic!(),
        }
    }
}
