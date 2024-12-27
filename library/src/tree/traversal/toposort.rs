pub mod tree {
    // Fast tree reconstruction with XOR-linked tree traversal
    // https://codeforces.com/blog/entry/135239
    pub trait AsBytes<const N: usize> {
        unsafe fn as_bytes(self) -> [u8; N];
        unsafe fn decode(bytes: [u8; N]) -> Self;
    }

    #[macro_export]
    macro_rules! impl_as_bytes {
        ($T:ty, $N: ident) => {
            const $N: usize = std::mem::size_of::<$T>();

            impl crate::tree::AsBytes<$N> for $T {
                unsafe fn as_bytes(self) -> [u8; $N] {
                    std::mem::transmute::<$T, [u8; $N]>(self)
                }

                unsafe fn decode(bytes: [u8; $N]) -> $T {
                    std::mem::transmute::<[u8; $N], $T>(bytes)
                }
            }
        };
    }
    pub use impl_as_bytes;

    impl_as_bytes!((), __N_UNIT);
    impl_as_bytes!(u32, __N_U32);
    impl_as_bytes!(u64, __N_U64);
    impl_as_bytes!(i32, __N_I32);
    impl_as_bytes!(i64, __N_I64);

    #[inline(always)]
    pub unsafe fn assert_unchecked(b: bool) {
        if !b {
            std::hint::unreachable_unchecked();
        }
    }

    #[cold]
    #[inline(always)]
    pub fn cold() {}

    #[inline(always)]
    pub fn likely(b: bool) -> bool {
        if !b {
            cold();
        }
        b
    }

    #[inline(always)]
    pub fn unlikely(b: bool) -> bool {
        if b {
            cold();
        }
        b
    }

    pub fn toposort<'a, const N: usize, E: Clone + Default + AsBytes<N> + 'a>(
        n_verts: usize,
        edges: impl IntoIterator<Item = (u32, u32, E)>,
        root: usize,
    ) -> impl Iterator<Item = (u32, u32, E)> {
        let mut degree = vec![0u32; n_verts];
        let mut xor_neighbors: Vec<(u32, [u8; N])> = vec![(0 as u32, [0u8; N]); n_verts];

        fn xor_assign_bytes<const N: usize>(xs: &mut [u8; N], ys: [u8; N]) {
            for (x, y) in xs.iter_mut().zip(&ys) {
                *x ^= *y;
            }
        }

        for (u, v, w) in edges
            .into_iter()
            .flat_map(|(u, v, w)| [(u, v, w.clone()), (v, u, w)])
        {
            debug_assert!(u != v);
            degree[u as usize] += 1;
            xor_neighbors[u as usize].0 ^= v;
            xor_assign_bytes(&mut xor_neighbors[u as usize].1, unsafe {
                AsBytes::as_bytes(w.clone())
            });
        }

        degree[root] += 2;
        let mut base = 0;
        let mut u = 0;
        std::iter::from_fn(move || {
            if unlikely(degree[u] != 1) {
                u = loop {
                    if unlikely(base >= n_verts) {
                        return None;
                    } else if degree[base] == 1 {
                        break base;
                    }
                    base += 1;
                }
            }
            let (p, w_encoded) = xor_neighbors[u];
            degree[u] = 0;
            degree[p as usize] -= 1;
            xor_neighbors[p as usize].0 ^= u as u32;
            xor_assign_bytes(&mut xor_neighbors[p as usize].1, w_encoded);
            let u_old = u;
            u = p as usize;
            Some((u_old as u32, p, unsafe { AsBytes::decode(w_encoded) }))
        })
    }
}
