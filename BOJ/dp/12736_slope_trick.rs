use std::{collections::BinaryHeap, io::Write, mem::MaybeUninit};

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

    pub fn stdin<'a>() -> InputAtOnce<'a> {
        let _buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let iter = _buf.split_ascii_whitespace();
        let iter = unsafe { std::mem::transmute(iter) };
        InputAtOnce { _buf, iter }
    }

    pub fn stdout() -> std::io::BufWriter<std::io::Stdout> {
        std::io::BufWriter::new(std::io::stdout())
    }
}

// Piecewise linear, convex function on [0, infty)
// with slope in {..., -2, -1, 0, 1}
#[derive(Debug)]
struct Func {
    breakpoints: BinaryHeap<u64>,
    intercept: u64,
    rightmost_slope: u64,
}

impl Func {
    fn empty() -> Self {
        Self {
            breakpoints: [].into(),
            intercept: 0,
            rightmost_slope: 1,
        }
    }

    // A convolution, g(x) <- min_{y>=0, z>=0, y+z<=x} (f(y) + |z-c|)
    fn move_up(&mut self, c: u64) {
        assert!(self.rightmost_slope == 1);
        self.intercept += c;
        if self.breakpoints.is_empty() {
            self.breakpoints.push(c);
            self.breakpoints.push(c);
        } else {
            let mut x1 = unsafe { self.breakpoints.pop().unwrap_unchecked() };
            let mut x2 = unsafe { self.breakpoints.peek_mut().unwrap_unchecked() };
            *x2 += c;
            drop(x2);
            x1 += c;
            self.breakpoints.push(x1);
        }
    }

    // Pointwise addition, g(x) <- f1(x) + f2(x),
    // with slope truncation at right end, h(0) <- g(0), h'(x) <- min (g'(x), 1)
    fn union(&mut self, mut other: Self) {
        if self.breakpoints.len() < other.breakpoints.len() {
            std::mem::swap(self, &mut other);
        }
        // print!("{:?} <| {:?} ", self, other);
        self.intercept += other.intercept;
        if !other.breakpoints.is_empty() {
            self.breakpoints.extend(other.breakpoints);
            self.rightmost_slope += other.rightmost_slope;
        }
        // println!("=> {:?}", self);
    }

    fn truncate_slope(&mut self) {
        while self.rightmost_slope > 1 {
            self.breakpoints.pop().unwrap();
            self.rightmost_slope -= 1;
        }
    }

    fn eval_min(&mut self) -> u64 {
        assert!(self.rightmost_slope == 1);
        if self.breakpoints.len() < 2 {
            return self.intercept;
        }

        self.breakpoints.pop();
        let mut acc = self.intercept;
        let mut neg_slope = 1;
        let mut x_prev = self.breakpoints.pop().unwrap();
        while let Some(x) = self.breakpoints.pop() {
            let dx = x_prev - x;
            acc -= dx * neg_slope;
            neg_slope += 1;
            x_prev = x;
        }
        acc -= x_prev * neg_slope;
        acc
    }
}

mod tree {
    pub fn postorder<'a, T: Clone, F>(neighbors: &[Vec<(usize, T)>], root: usize, mut visitor: F)
    where
        F: FnMut(usize, Option<(usize, T)>),
    {
        fn rec<'a, T: Clone, F>(neighbors: &[Vec<(usize, T)>], u: usize, p: usize, visitor: &mut F)
        where
            F: FnMut(usize, Option<(usize, T)>),
        {
            for (v, w) in &neighbors[u] {
                if *v == p {
                    continue;
                }
                rec(neighbors, *v, u, visitor);
                visitor(*v, Some((u, w.clone())));
            }
        }
        rec(neighbors, root, root, &mut visitor);
        visitor(root, None);
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let v = n + m;

    let root = 0;
    let mut neighbors = vec![vec![]; v];
    for i in 1..v {
        let p = input.value::<usize>() - 1;
        let c: u64 = input.value();
        neighbors[p].push((i, c));
        neighbors[i].push((p, c));
    }

    let mut fs: Vec<_> = (0..v).map(|_| MaybeUninit::new(Func::empty())).collect();
    tree::postorder(&neighbors, root, |u, p| unsafe {
        let f_u = fs[u].assume_init_mut();
        f_u.truncate_slope();
        if let Some((p, c)) = p {
            let mut f_u = fs[u].assume_init_read();
            f_u.move_up(c);
            fs[p].assume_init_mut().union(f_u);
        }
    });

    let ans = unsafe { fs[0].assume_init_read().eval_min() };
    writeln!(output, "{}", ans).unwrap();
}
