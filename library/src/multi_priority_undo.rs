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

        pub fn pop(&mut self, axis: usize) -> Option<(S::Item, [K; D])> {
            debug_assert!(axis < D);
            let l = self.idx_stack.len();
            if l == 0 {
                return None;
            }

            // Pop the target item.
            let (_, i_target) = *self.sets[axis].last().unwrap();
            for b in 0..D {
                self.sets[b].remove(&(self.priorities[i_target as usize][b], i_target));
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

                for b in 0..D {
                    let (_, i) = *iter[b].next().unwrap();
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
