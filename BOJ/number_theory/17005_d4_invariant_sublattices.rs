// - Lemma (Sublattice clasification): Any sublattice of Z^2
//   spanned from the finite union of sets S_{a,b} = { (a,+-b), (b, +-a) }    (1 <= a, b)
//   falls into one of two types:
//   - Sublattice type 0: span { [g, 0], [0, g] }
//   - Sublattice type 1: span { [2g, 0], [g, g] }
// (proof outline):
//   Calculate hSublattice classification of Z^2 under D4 symmetry + Dijkstraermite normal form of the matrix with S_{a,b} as rows to obtain two types.
//   Then show that these types are closed under binary operation (S, S') |-> span S U S',
//   so that we can apply standard induction on the number of S_{a,b} used.
//
//
// => Encode sublattice by L |-> [2 * g(L) + type(L)]

use std::{
    cmp::Reverse,
    collections::{BinaryHeap, HashMap, HashSet},
    io::Write,
};

mod simple_io {
    pub struct InputAtOnce {
        iter: std::str::SplitAsciiWhitespace<'static>,
    }

    impl InputAtOnce {
        pub fn token(&mut self) -> &'static str {
            self.iter.next().unwrap_or_default()
        }

        pub fn try_value<T: std::str::FromStr>(&mut self) -> Option<T> {
            self.token().parse().ok()
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.try_value().unwrap()
        }
    }

    pub fn stdin() -> InputAtOnce {
        let buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let buf = Box::leak(Box::new(buf));
        let iter = buf.split_ascii_whitespace();
        InputAtOnce { iter }
    }

    pub fn stdout() -> std::io::BufWriter<std::io::Stdout> {
        std::io::BufWriter::new(std::io::stdout())
    }
}

fn gcd(mut a: i32, mut b: i32) -> i32 {
    while b != 0 {
        let temp = b;
        b = a % b;
        a = temp;
    }
    a
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let mut items = vec![];
    for _ in 0..n {
        let r: i32 = input.value();
        let c: i32 = input.value();
        let a: i32 = input.value();
        let b: i32 = input.value();
        let p: i64 = input.value();

        let g = gcd(a, b);
        let x = a / g;
        let y = b / g;
        let s = if (x ^ y) & 1 == 1 { g * 2 } else { g * 2 + 1 };

        items.push((r, c, s, p));
    }

    let (r0, c0, ..) = items[0];
    if (r0, c0) == (0, 0) {
        writeln!(output, "0").unwrap();
        return;
    }

    let mut pq = BinaryHeap::new();
    let mut dist = HashMap::new();
    let mut visited = HashSet::new();
    for (r, c, s, p) in &mut items {
        (*r, *c) = (*r - r0, *c - c0);

        if (*r, *c) == (0, 0) {
            pq.push((Reverse(*p), *s));
            dist.insert(*s, *p);
        }
    }

    let reachable = |g, g_ty, r, c| {
        if g_ty == 0 {
            r % g == 0 && c % g == 0
        } else {
            r % g == 0 && c % g == 0 && (r + c) % (2 * g) == 0
        }
    };

    while let Some((Reverse(d), s)) = pq.pop() {
        if visited.contains(&s) {
            continue;
        }
        visited.insert(s);

        let (g, g_ty) = (s / 2, s % 2);
        if reachable(g, g_ty, r0, c0) {
            writeln!(output, "{}", d).unwrap();
            return;
        }

        for (r, c, t, p) in &items {
            if !reachable(g, g_ty, *r, *c) {
                continue;
            }

            let (h, h_ty) = (t / 2, t % 2);
            let (k, k_ty) = match (g_ty, h_ty) {
                (0, 0) | (1, 1) => (gcd(g, h), g_ty),
                (0, 1) => {
                    let k = gcd(g, 2 * h);
                    if h % k == 0 {
                        (k, 0)
                    } else {
                        (k / 2, 1)
                    }
                }
                (1, 0) => {
                    let k = gcd(2 * g, h);
                    if g % k == 0 {
                        (k, 0)
                    } else {
                        (k / 2, 1)
                    }
                }
                _ => unreachable!(),
            };

            let u = k * 2 + k_ty;
            let d_new = dist[&s] + p;
            if dist.get(&u).map_or(true, |&d_old| d_new < d_old) {
                dist.insert(u, d_new);
                pq.push((Reverse(d_new), u));
            }
        }
    }

    writeln!(output, "-1").unwrap();
}
