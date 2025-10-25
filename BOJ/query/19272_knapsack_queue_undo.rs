use std::io::Write;

use generalized_undo::*;

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

struct Crypto {
    sm: i64,
    cnt: u32,
    data: [u8; 256],
    i: usize,
    j: usize,
}

impl Crypto {
    fn new() -> Self {
        let mut this = Self {
            sm: 0,
            cnt: 0,
            data: [0; 256],
            i: 0,
            j: 0,
        };
        this.seed();
        this
    }

    fn decode(&mut self, mut z: u32) -> u32 {
        z ^= self.next();
        z ^= self.next() << 8;
        z ^= self.next() << 16;
        z ^= self.next() << 22;
        z
    }

    fn query(&mut self, z: i64) {
        const B: i64 = 425481007;
        const MD: i64 = 1000000007;
        self.cnt += 1;
        self.sm = ((self.sm * B % MD + z) % MD + MD) % MD;
        self.seed();
    }

    fn seed(&mut self) {
        let mut key = [0u8; 8];
        for k in 0..4 {
            key[k] = (self.sm >> (k * 8)) as u8;
        }
        for k in 0..4 {
            key[k + 4] = (self.cnt >> (k * 8)) as u8;
        }

        self.data = std::array::from_fn(|i| i as u8);
        self.i = 0;
        self.j = 0;

        let mut q = 0u8;
        for p in 0..256 {
            q = q.wrapping_add(self.data[p]).wrapping_add(key[p % 8]);
            self.data.swap(p, q as usize);
        }
    }

    fn next(&mut self) -> u32 {
        self.i = (self.i + 1) % 256;
        self.j = (self.j + self.data[self.i] as usize) % 256;
        self.data.swap(self.i, self.j);
        self.data[(self.data[self.i].wrapping_add(self.data[self.j])) as usize] as u32
    }
}

// class Crypto {
// public:
//     Crypto() {
//         sm = cnt = 0;
//         seed();
//     }
//
//     int decode(int z) {
//         z ^= next();
//         z ^= (next() << 8);
//         z ^= (next() << 16);
//         z ^= (next() << 22);
//         return z;
//     }
//
//     void query(long long z) {
//         const long long B = 425481007;
//         const long long MD = 1000000007;
//         cnt++;
//         sm = ((sm * B % MD + z) % MD + MD) % MD;
//         seed();
//     }
// private:
//     long long sm;
//     int cnt;
//
//     uint8_t data[256];
//     int I, J;
//
//     void swap_data(int i, int j) {
//         uint8_t tmp = data[i];
//         data[i] = data[j];
//         data[j] = tmp;
//     }
//
//     void seed() {
//         uint8_t key[8];
//         for (int i = 0; i < 4; i++) {
//             key[i] = (sm >> (i * 8));
//         }
//         for (int i = 0; i < 4; i++) {
//             key[i+4] = (cnt >> (i * 8));
//         }
//
//         for (int i = 0; i < 256; i++) {
//             data[i] = i;
//         }
//         I = J = 0;
//
//         int j = 0;
//         for (int i = 0; i < 256; i++) {
//             j = (j + data[i] + key[i%8]) % 256;
//             swap_data(i, j);
//         }
//     }
//
//     uint8_t next() {
//         I = (I+1) % 256;
//         J = (J + data[I]) % 256;
//         swap_data(I, J);
//         return data[(data[I] + data[J]) % 256];
//     }
// };

pub mod generalized_undo {
    use std::collections::BTreeSet;

    const INF: u32 = u32::MAX;

    pub trait StackUndo {
        type Item;
        fn push(&mut self, value: Self::Item);
        fn pop(&mut self) -> Option<Self::Item>;
    }

    // Given a black-box data structure that supports O(T) stack-like push and pop,
    // `MultiPriorityUndo` is an **online** data structure that supports following ops:
    // - Push, O(TD log Q).
    //   Accepts a D-dimensional priority vector for each pushed item.
    // - Pop item with maximum ith priority, amortized O(TD log Q).
    //
    // ## Reference
    // [[Tutorial] Supporting Queue-like Undoing on DS](https://codeforces.com/blog/entry/83467)
    // [[Tutorial] Supporting Priority-Queue-like Undoing on DS](https://codeforces.com/blog/entry/111117)
    #[derive(Clone)]
    pub struct MultiPriorityUndo<S: StackUndo, const D: usize, K: Ord> {
        pub inner: S,

        n_id: u32,
        idx_stack: Vec<u32>,

        priorities: Vec<[K; D]>,
        pos_in_stack: Vec<u32>,
        sets: [BTreeSet<(K, u32)>; D],
    }

    const fn inv_alpha<const D: usize>() -> usize {
        assert!(D >= 1);
        if D == 1 {
            2
        } else {
            D + (D * (D + 1)).isqrt()
        }
    }

    impl<S: StackUndo, const D: usize, K: Ord> From<S> for MultiPriorityUndo<S, D, K> {
        fn from(stack: S) -> Self {
            Self {
                inner: stack,

                n_id: 0,
                idx_stack: vec![],

                priorities: vec![],
                pos_in_stack: vec![],
                sets: std::array::from_fn(|_| Default::default()),
            }
        }
    }

    impl<S: StackUndo + Default, const D: usize, K: Ord> Default for MultiPriorityUndo<S, D, K> {
        fn default() -> Self {
            S::default().into()
        }
    }

    impl<S: StackUndo, const D: usize, K: Ord + Copy> MultiPriorityUndo<S, D, K> {
        pub fn inner(&self) -> &S {
            &self.inner
        }

        pub fn push(&mut self, x: S::Item, priority: [K; D]) {
            self.pos_in_stack.push(self.idx_stack.len() as u32);

            self.idx_stack.push(self.n_id);
            self.inner.push(x);

            self.priorities.push(priority);
            for ax in 0..D {
                self.sets[ax].insert((priority[ax], self.n_id));
            }

            self.n_id += 1;
        }

        pub fn pop<const AXIS: usize>(&mut self) -> Option<(S::Item, [K; D])> {
            assert!(AXIS < D);
            let l = self.idx_stack.len();
            if l == 0 {
                return None;
            }

            // Pop the target item.
            let (_, i_target) = *self.sets[AXIS].last().unwrap();
            for ax in 0..D {
                self.sets[ax].remove(&(self.priorities[i_target as usize][ax], i_target));
            }
            let target_pos = self.pos_in_stack[i_target as usize] as usize;
            self.pos_in_stack[i_target as usize] = INF;

            // Temporarily mark items with the highest priorities, up to the threshold.
            let mut iter: [_; D] = std::array::from_fn(|ax| self.sets[ax].iter().rev());
            let mut top = vec![];
            let mut min_pos = target_pos;
            for j in 0..l {
                if inv_alpha::<D>().saturating_mul(j + 1) >= l - min_pos {
                    break;
                }

                for ax in 0..D {
                    let (_, i) = *iter[ax].next().unwrap();
                    if self.pos_in_stack[i as usize] != INF {
                        min_pos = min_pos.min(self.pos_in_stack[i as usize] as usize);
                        top.push(self.pos_in_stack[i as usize]);
                        self.pos_in_stack[i as usize] = INF;
                    }
                }
            }

            // Reorder the stack up to the marked items. The marked items goes to the top, sorted.
            let mut to_reorder = vec![];
            for _ in (min_pos..l).rev() {
                let x = self.inner.pop().unwrap();
                let i = self.idx_stack.pop().unwrap();
                to_reorder.push(Some((x, i)));
            }

            for e in to_reorder.iter_mut().rev() {
                let (_, i) = *e.as_ref().unwrap();
                if self.pos_in_stack[i as usize] != INF {
                    let (x, i) = e.take().unwrap();
                    self.pos_in_stack[i as usize] = self.idx_stack.len() as u32;
                    self.inner.push(x);
                    self.idx_stack.push(i);
                }
            }

            for p in top.into_iter().rev() {
                let (x, i) = to_reorder[l - 1 - p as usize].take().unwrap();
                self.pos_in_stack[i as usize] = self.idx_stack.len() as u32;
                self.inner.push(x);
                self.idx_stack.push(i);
            }

            let (x, _) = to_reorder[l - 1 - target_pos as usize].take().unwrap();
            return Some((x, self.priorities[i_target as usize]));
        }
    }
}

const BOUND: usize = 500;
const INF: i64 = 1 << 60;

struct Knapsack {
    m: usize,
    items: Vec<(u32, u32)>,
    dp: Vec<[i64; BOUND]>,
}

impl Knapsack {
    fn new(m: usize) -> Self {
        let mut base = [-INF; BOUND];
        base[0] = 0;
        Self {
            m,
            items: vec![],
            dp: vec![base],
        }
    }

    fn query(&self, l: usize, r: usize) -> i64 {
        let row = self.dp.last().unwrap();
        row[l..=r].iter().copied().max().unwrap().max(-1)
    }
}

impl StackUndo for Knapsack {
    type Item = (u32, u32);

    fn push(&mut self, (w, v): Self::Item) {
        self.items.push((w, v));

        let prev = self.dp.last().unwrap();
        let mut row = prev.clone();
        let w = w as usize % self.m;
        for i in 0..self.m {
            row[(i + w) % self.m] = row[(i + w) % self.m].max(prev[i] + v as i64);
        }
        self.dp.push(row);
    }

    fn pop(&mut self) -> Option<Self::Item> {
        if let Some(item) = self.items.pop() {
            self.dp.pop();
            Some(item)
        } else {
            None
        }
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let m: usize = input.value();
    let q: usize = input.value();
    let mut c = Crypto::new();

    let mut ks = MultiPriorityUndo::from(Knapsack::new(m));

    for _ in 0..q {
        let t = c.decode(input.value());
        let w = c.decode(input.value());
        let v = c.decode(input.value());
        let l = c.decode(input.value());
        let r = c.decode(input.value());
        if t == 1 {
            ks.push((w, v), [std::cmp::Reverse(w)]);
        } else {
            let _popped = ks.pop::<0>();
        }

        let ans = ks.inner().query(l as usize, r as usize);
        c.query(ans);
        writeln!(output, "{}", ans).unwrap();
    }
}
