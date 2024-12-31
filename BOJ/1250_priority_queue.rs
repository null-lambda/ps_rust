use std::io::{BufRead, Read, Write};

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
#[repr(C)]
struct U24([u8; 3]);

impl From<u32> for U24 {
    fn from(value: u32) -> Self {
        let [b3, b2, b1, b0] = value.to_be_bytes();
        debug_assert_eq!(b3, 0);
        U24([b2, b1, b0])
    }
}

// impl TryFrom<u32> for U24 {
//     type Error = &'static str;
//     fn try_from(value: u32) -> Result<Self, Self::Error> {
//         let [b3, b2, b1, b0] = value.to_be_bytes();
//         if b3 != 0 {
//             Err("Value out of range")
//         } else {
//             Ok(U24([b2, b1, b0]))
//         }
//     }
// }

impl Into<u32> for U24 {
    fn into(self) -> u32 {
        let U24([b2, b1, b0]) = self;
        u32::from_be_bytes([0, b2, b1, b0])
    }
}

impl Into<usize> for U24 {
    fn into(self) -> usize {
        let x: u32 = self.into();
        x as usize
    }
}

impl std::ops::Add<U24> for U24 {
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output {
        let lhs: u32 = self.into();
        let rhs: u32 = rhs.into();
        (lhs + rhs).into()
    }
}

#[derive(Debug, Clone, Copy)]
struct NodeRef {
    idx: u32,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct ByteGroup {
    count: U24,
    key: u8,
}

// A bitset with nearest nonzero neighbor query
// Each block stores bit data with nearest nonempty bucket indices (cyclic doubly linked list)
// Memory consumption: 1 bit * (32 + 32 + 32)/32 = 3/8 byte per node
type Bucket = u32;
struct LinkedBitset {
    size: usize,
    buckets: Vec<(u32, u32, Bucket)>,
}

impl LinkedBitset {
    fn full(size: usize) -> Self {
        let n_buckets = size.div_ceil(32);
        let buckets: Vec<_> = (0..n_buckets)
            .map(|i| {
                let left = if i > 0 { i - 1 } else { n_buckets - 1 };
                let right = if i < n_buckets - 1 { i + 1 } else { 0 };
                let mask = if i < n_buckets - 1 || size % 32 == 0 {
                    !0
                } else {
                    !(!0 << size % 32)
                };
                (left as u32, right as u32, mask)
            })
            .collect();

        Self { size, buckets }
    }

    #[inline(always)]
    fn get(&self, idx: usize) -> bool {
        debug_assert!(idx < self.size);
        let (b, i) = (idx / 32, idx % 32);
        let (_, _, mask) = self.buckets[b];
        mask & (1 << i) != 0
    }

    #[inline(always)]
    fn remove(&mut self, idx: usize) -> bool {
        debug_assert!(idx < self.size);
        let (b, i) = (idx / 32, idx % 32);
        let (b_left, b_right, mask) = &mut self.buckets[b];
        if *mask & (1 << i) == 0 {
            return false;
        }
        *mask &= !(1 << i);
        if *mask == 0 {
            let b_left = *b_left;
            let b_right = *b_right;
            self.buckets[b_left as usize].1 = b_right;
            self.buckets[b_right as usize].0 = b_left;

            // Neighboring nodes must nonempty, or equal to self
            debug_assert!(
                b_left as usize == b && b_right as usize == b
                    || self.buckets[b_left as usize].2 != 0
                        && self.buckets[b_right as usize].2 != 0
            );
        }
        true
    }

    #[inline(always)]
    fn left(&self, idx: usize) -> Option<usize> {
        debug_assert!(idx < self.size);
        let (b, i) = (idx / 32, idx % 32);
        let (b_left, _, mask) = self.buckets[b];

        let left_mask = mask & ((1 << i) - 1);
        if left_mask != 0 {
            let j = 32 - 1 - left_mask.leading_zeros() as usize;
            return Some(b * 32 + j);
        }

        if b_left as usize >= b {
            return None; // First node
        }

        let (_, _, mask) = self.buckets[b_left as usize];
        debug_assert!(mask != 0);
        let j = 32 - 1 - mask.leading_zeros() as usize;
        Some(b_left as usize * 32 + j)
    }

    #[inline(always)]
    fn right(&self, idx: usize) -> Option<usize> {
        debug_assert!(idx < self.size);
        let (b, i) = (idx / 32, idx % 32);
        let (_, b_right, mask) = self.buckets[b];

        let mask: u32 = mask >> i >> 1 << i << 1;
        if mask != 0 {
            let j = mask.trailing_zeros() as usize;
            return Some(b * 32 + j);
        }

        if b_right as usize <= b {
            return None; // Last node
        }

        let (_, _, mask) = self.buckets[b_right as usize];
        debug_assert!(mask != 0);
        let j = mask.trailing_zeros() as usize;
        Some(b_right as usize * 32 + j)
    }
}

pub mod heap {
    use std::cmp::Ordering;

    #[derive(Clone)]
    pub struct BinaryHeap<T, C> {
        data: Vec<T>,
        pub cmp: C,
    }

    impl<T, C: FnMut(&T, &T) -> Ordering> BinaryHeap<T, C> {
        pub fn with_capacity(size: usize, cmp: C) -> Self {
            Self {
                data: Vec::with_capacity(size),
                cmp,
            }
        }

        pub fn push(&mut self, value: T) {
            self.data.push(value);
            self.heapify_up(self.data.len() - 1);
        }

        pub fn pop(&mut self) -> Option<T> {
            if self.data.is_empty() {
                return None;
            }
            let n = self.data.len();
            self.data.swap(0, n - 1);
            let value = self.data.pop();
            self.heapify_down(0);
            value
        }

        pub fn peek(&self) -> Option<&T> {
            self.data.get(0)
        }

        pub fn update_top(&mut self) {
            if !self.data.is_empty() {
                self.heapify_down(0);
            }
        }

        fn heapify_up(&mut self, mut idx: usize) {
            while idx > 0 {
                let parent = (idx - 1) / 2;
                if (self.cmp)(&self.data[idx], &self.data[parent]) == Ordering::Greater {
                    self.data.swap(idx, parent);
                    idx = parent;
                } else {
                    break;
                }
            }
        }

        fn heapify_down(&mut self, mut idx: usize) {
            let n = self.data.len();
            loop {
                let left = 2 * idx + 1;
                let right = 2 * idx + 2;

                let mut max_idx = idx;
                if left < n
                    && (self.cmp)(&self.data[left], &self.data[max_idx]) == Ordering::Greater
                {
                    max_idx = left;
                }
                if right < n
                    && (self.cmp)(&self.data[right], &self.data[max_idx]) == Ordering::Greater
                {
                    max_idx = right;
                }
                if max_idx == idx {
                    break;
                }
                self.data.swap(idx, max_idx);
                idx = max_idx;
            }
        }
    }
}

use crate::heap::BinaryHeap;

fn main() {
    // Memory Limit: 128 MB
    // Constraint: n <= 1e7

    // mem 1. Default memory consumption = 14 MB
    // (might reduce to ~3MB with no_std and c interops)
    // Buffered io's memory consumption is insignificant
    let mut input = std::io::BufReader::with_capacity(1 << 16, std::io::stdin().lock());
    let mut output = std::io::BufWriter::with_capacity(1 << 16, std::io::stdout());

    let mut line = String::new();
    input.read_line(&mut line).unwrap();
    let mut tokens = line.split_whitespace();
    let n: usize = tokens.next().unwrap().parse().unwrap();
    let k: usize = tokens.next().unwrap().parse::<usize>().unwrap() - 1;

    // mem 2. Rle data: 4 byte * 1e7 = 39 MB
    let mut rle = Vec::with_capacity(n);

    let mut current_key = 0;
    let mut current_count = 0;
    for c in input
        .bytes()
        .flatten()
        .filter(|&c| c.is_ascii_alphabetic())
        .take(n)
    {
        if c != current_key && current_count > 0 {
            rle.push(ByteGroup {
                count: U24::from(current_count),
                key: current_key,
            });
            current_count = 0;
        }
        current_count += 1;
        current_key = c;
    }
    if current_count > 0 {
        rle.push(ByteGroup {
            count: U24::from(current_count),
            key: current_key,
        });
    }
    let n = rle.len();

    let mut k_group = 0;
    let mut len_acc = 0;
    for (i, group) in rle.iter().enumerate() {
        len_acc += Into::<usize>::into(group.count);
        if k < len_acc {
            k_group = i;
            break;
        }
    }

    // mem 3. Priority queue: 4 byte * 1e7 = 39 MB
    let rle_ptr = rle.as_ptr();
    let mut pq = BinaryHeap::<NodeRef, _>::with_capacity(n, |i, j| {
        let ci = unsafe { &*rle_ptr.wrapping_offset(i.idx as isize) };
        let cj = unsafe { &*rle_ptr.wrapping_offset(j.idx as isize) };
        ci.count
            .cmp(&cj.count)
            .then_with(|| i.idx.cmp(&j.idx).reverse())
    });
    for idx in 0..n as u32 {
        pq.push(NodeRef { idx });
    }

    // mem 4. Linked bitset: 3/8 byte * 1e7 = 4 MB
    let mut active = LinkedBitset::full(n);

    let mut count = 0;
    while let Some(NodeRef { idx }) = pq.peek().copied() {
        if !active.get(idx as usize) {
            pq.pop();
            continue;
        }

        let idx: usize = idx as usize;
        count += 1;

        if idx == k_group {
            writeln!(output, "{}", count).unwrap();
            return;
        }

        if let (Some(left), Some(right)) = (active.left(idx), active.right(idx)) {
            if rle[left].key == rle[right].key {
                // Merge a pair of groups with same key,
                // into the black cell of the removed group.
                // (Nodes holded by pq should be immutable)
                active.remove(left);
                active.remove(right);
                rle[idx] = ByteGroup {
                    count: rle[left].count + rle[right].count,
                    key: rle[left].key,
                };
                pq.update_top();
                if k_group == left || k_group == right {
                    k_group = idx;
                }
                continue;
            }
        }

        pq.pop();
        active.remove(idx);
    }

    panic!()

    // Total memory consumption (expected): 96 MB
}
