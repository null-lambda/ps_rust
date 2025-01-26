pub mod jagged {
    // Trait for painless switch between different representations of a jagged array
    pub trait Jagged<'a, T: 'a> {
        fn len(&self) -> usize;
        fn get(&'a self, u: usize) -> &'a [T];
    }

    impl<'a, T, C> Jagged<'a, T> for C
    where
        C: AsRef<[Vec<T>]> + 'a,
        T: 'a,
    {
        fn len(&self) -> usize {
            self.as_ref().len()
        }

        fn get(&'a self, u: usize) -> &'a [T] {
            &self.as_ref()[u]
        }
    }
}
