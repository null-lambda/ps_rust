#[allow(dead_code)]
mod collections {
    use std::{
        cmp::Reverse,
        collections::HashMap,
        fmt::Display,
        iter::once,
        ops::{Index, IndexMut},
    };

    #[derive(Debug, Clone)]
    pub struct Grid<T> {
        pub w: usize,
        pub data: Vec<T>,
    }

    impl<T> Grid<T> {
        pub fn with_shape(self, w: usize) -> Self {
            debug_assert_eq!(self.data.len() % w, 0);
            Grid { w, data: self.data }
        }
    }

    impl<T> FromIterator<T> for Grid<T> {
        fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
            Self {
                w: 1,
                data: iter.into_iter().collect(),
            }
        }
    }

    impl<T: Clone> Grid<T> {
        pub fn sized(fill: T, h: usize, w: usize) -> Self {
            Grid {
                w,
                data: vec![fill; w * h],
            }
        }
    }

    impl<T: Display> Display for Grid<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            for row in self.data.chunks(self.w) {
                for cell in row {
                    cell.fmt(f)?;
                    write!(f, " ")?;
                }
                writeln!(f)?;
            }
            writeln!(f)?;
            Ok(())
        }
    }

    impl<T> Index<(usize, usize)> for Grid<T> {
        type Output = T;
        fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
            debug_assert!(i < self.data.len() / self.w && j < self.w);
            &self.data[i * self.w + j]
        }
    }

    impl<T> IndexMut<(usize, usize)> for Grid<T> {
        fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
            debug_assert!(i < self.data.len() / self.w && j < self.w);
            &mut self.data[i * self.w + j]
        }
    }

    struct PrettyColored<'a>(&'a Grid<u8>);

    impl Display for PrettyColored<'_> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let colors = (once(37).chain(31..=36))
                .map(|i| (format!("\x1b[{}m", i), format!("\x1b[0m")))
                .collect::<Vec<_>>();

            let mut freq = HashMap::new();
            for c in self.0.data.iter() {
                *freq.entry(c).or_insert(0) += 1;
            }
            let mut freq = freq.into_iter().collect::<Vec<_>>();
            freq.sort_unstable_by_key(|(_, f)| Reverse(*f));

            let mut color_map = HashMap::new();
            let mut idx = 0;
            for (c, _) in freq {
                color_map.insert(c, &colors[idx % colors.len()]);
                idx += 1;
            }

            for row in self.0.data.chunks(self.0.w) {
                for cell in row {
                    let (pre, suff) = color_map[&cell];
                    write!(f, "{}{}{}", pre, *cell as char, suff)?;
                }
                writeln!(f)?;
            }
            Ok(())
        }
    }

    impl Grid<u8> {
        fn colored(&self) -> PrettyColored {
            PrettyColored(&self)
        }
    }
}
