trait IteratorExt: Iterator {
    fn intersp_with(self, mut sep: impl FnMut() -> Self::Item) -> impl Iterator<Item = Self::Item>
    where
        Self: Sized,
    {
        let mut this = self.peekable();
        let mut yield_sep = true;
        std::iter::from_fn(move || {
            yield_sep ^= true;
            if this.peek().is_none() {
                None
            } else if yield_sep {
                Some(sep())
            } else {
                this.next()
            }
        })
    }

    fn intersp(self, sep: Self::Item) -> impl Iterator<Item = Self::Item>
    where
        Self: Sized,
        Self::Item: Clone,
    {
        self.intersp_with(move || sep.clone())
    }

    fn join(self) -> String
    where
        Self: Sized,
        Self::Item: std::fmt::Display,
    {
        use std::fmt::Write;
        let mut iter = self.into_iter();
        let mut res = String::new();
        if let Some(x) = iter.next() {
            write!(&mut res, "{x}").unwrap();
            for x in iter {
                write!(&mut res, " {x}").unwrap();
            }
        }
        res
    }

    fn cprod<Other>(self, other: Other) -> impl Iterator<Item = (Self::Item, Other::Item)>
    where
        Self: Sized,
        Self::Item: Clone,
        Other: IntoIterator,
        Other::IntoIter: Clone,
    {
        let other = other.into_iter();
        self.flat_map(move |x| other.clone().map(move |y| (x.clone(), y)))
    }

    // fn cpow(self, exp: usize) -> impl Iterator<Item = Vec<Self::Item>>
    // where
    //     Self: Sized,
    //     Self::Item: Clone,
    // {
    //     todo!()
    // }
}
impl<I: Iterator> IteratorExt for I {}

pub trait TuplePush<S> {
    type Output;
    fn push(self, x: S) -> Self::Output;
}
macro_rules! impl_tuple_ops {
(@cascade $($t:ident)*) => {
    impl<$($t,)* S> TuplePush<S> for ($($t,)*) {
        type Output = ($($t,)* S,);
        fn push(self, x: S) -> Self::Output {
            #[allow(non_snake_case)]
            let ($($t,)*) = self;
            ($($t,)* x,)
        }
    }
};
($t:ident $($rest:ident)*) => {
    impl_tuple_ops!(@cascade $t $($rest)*);
    impl_tuple_ops!($($rest)*);
};
() => {
    impl_tuple_ops!(@cascade);
};
}
impl_tuple_ops!(T0 T1 T2 T3 T4 T5 T6 T7 T8 T9);

pub trait TupleFlatten {
    type Output;
    fn flatten(self) -> Self::Output;
}
impl<T> TupleFlatten for (T,) {
    type Output = (T,);
    fn flatten(self) -> Self::Output {
        self
    }
}
impl<T, S> TupleFlatten for (T, S)
where
    T: TuplePush<S>,
{
    type Output = <T as TuplePush<S>>::Output;
    fn flatten(self) -> Self::Output {
        self.0.push(self.1)
    }
}
macro_rules! cprods {
    ($t:expr) => {
        $t.map(|t| (t,))
    };
    ($t:expr, $($rest:expr),+) => {
        ($t.map(|t| (t,)) $(.cprod($rest))*).map(TupleFlatten::flatten)
    };
}
