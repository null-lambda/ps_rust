pub mod bag {
    use std::ops::{Index, IndexMut};
    pub const UNSET: u32 = u32::MAX;

    // Small-sized array-vec with none-like value
    #[derive(Clone, Copy)]
    pub struct Bag<const N: usize>(pub [u32; N]);

    impl<const N: usize> Default for Bag<N> {
        fn default() -> Self {
            Self([UNSET; N])
        }
    }

    impl<const N: usize> Bag<N> {
        pub fn get_raw(&self, idx: usize) -> &u32 {
            &self.0[idx]
        }

        pub fn get_raw_mut(&mut self, idx: usize) -> &mut u32 {
            &mut self.0[idx]
        }

        pub fn get(&self, idx: usize) -> Option<&u32> {
            let res = self.get_raw(idx);
            (res != &UNSET).then(|| res)
        }

        pub fn get_mut(&mut self, idx: usize) -> Option<&mut u32> {
            let res = self.get_raw_mut(idx);
            (res != &UNSET).then(|| res)
        }

        pub fn len(&self) -> usize {
            self.iter().count()
        }

        pub fn iter(&self) -> Iter<'_, N> {
            Iter {
                inner: self.0.iter(),
            }
        }

        pub fn iter_mut(&mut self) -> IterMut<'_, N> {
            IterMut {
                inner: self.0.iter_mut(),
            }
        }

        pub fn try_push(&mut self, value: u32) -> Result<usize, ()> {
            for i in 0..N {
                if self.0[i] == UNSET {
                    self.0[i] = value;
                    return Ok(i);
                }
            }
            Err(())
        }

        pub fn push(&mut self, value: u32) {
            self.try_push(value).unwrap();
        }

        pub fn sort(&mut self) {
            self.0.sort_unstable();
        }

        pub fn dedup(&mut self) {
            let mut end = 0;
            let mut prev = UNSET;
            for i in 0..N {
                let x = self.0[i];
                if x == UNSET {
                    break;
                }
                if x != prev {
                    prev = x;
                    self.0[end] = x;
                    end += 1;
                }
            }

            for x in &mut self.0[end..N] {
                *x = UNSET;
            }
        }
    }

    impl<const N: usize> Index<usize> for Bag<N> {
        type Output = u32;

        fn index(&self, index: usize) -> &Self::Output {
            self.get(index).unwrap()
        }
    }

    impl<const N: usize> IndexMut<usize> for Bag<N> {
        fn index_mut(&mut self, index: usize) -> &mut Self::Output {
            self.get_mut(index).unwrap()
        }
    }

    impl<const N: usize, const M: usize> From<[u32; M]> for Bag<N> {
        fn from(value: [u32; M]) -> Self {
            assert!(M <= N);
            let mut this = Self::default();
            for i in 0..M {
                this.0[i] = value[i];
            }
            this
        }
    }

    pub struct IntoIter<const N: usize> {
        inner: std::array::IntoIter<u32, N>,
    }

    pub struct Iter<'a, const N: usize> {
        inner: std::slice::Iter<'a, u32>,
    }

    pub struct IterMut<'a, const N: usize> {
        inner: std::slice::IterMut<'a, u32>,
    }

    impl<const N: usize> Iterator for IntoIter<N> {
        type Item = u32;
        fn next(&mut self) -> Option<Self::Item> {
            self.inner.next().filter(|&u| u != UNSET)
        }
    }

    impl<'a, const N: usize> Iterator for Iter<'a, N> {
        type Item = &'a u32;

        fn next(&mut self) -> Option<Self::Item> {
            self.inner.next().filter(|&&u| u != UNSET)
        }
    }

    impl<'a, const N: usize> Iterator for IterMut<'a, N> {
        type Item = &'a mut u32;

        fn next(&mut self) -> Option<Self::Item> {
            self.inner.next().filter(|&&mut u| u != UNSET)
        }
    }

    impl<const N: usize> IntoIterator for Bag<N> {
        type Item = u32;
        type IntoIter = IntoIter<N>;
        fn into_iter(self) -> Self::IntoIter {
            IntoIter {
                inner: self.0.into_iter(),
            }
        }
    }

    impl<'a, const N: usize> IntoIterator for &'a Bag<N> {
        type Item = &'a u32;
        type IntoIter = Iter<'a, N>;
        fn into_iter(self) -> Self::IntoIter {
            self.iter()
        }
    }
    impl<'a, const N: usize> IntoIterator for &'a mut Bag<N> {
        type Item = &'a mut u32;
        type IntoIter = IterMut<'a, N>;
        fn into_iter(self) -> Self::IntoIter {
            self.iter_mut()
        }
    }
}
