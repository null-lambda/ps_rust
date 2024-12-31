use std::io::{BufRead, Write};

use num_mod::*;

mod simple_io {
    pub struct InputAtOnce<'a> {
        _buf: String,
        iter: std::str::SplitAsciiWhitespace<'a>,
    }

    impl<'a> InputAtOnce<'a> {
        pub fn token(&mut self) -> &'a str {
            self.iter.next().unwrap_or_default()
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> Option<T>
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().ok()
        }
    }

    pub fn stdin_at_once<'a>() -> InputAtOnce<'a> {
        let _buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let iter = _buf.split_ascii_whitespace();
        let iter = unsafe { std::mem::transmute(iter) };
        InputAtOnce { _buf, iter }
    }

    pub fn stdout() -> std::io::BufWriter<std::io::Stdout> {
        std::io::BufWriter::new(std::io::stdout())
    }
}

pub mod num_mod {
    use std::ops::*;

    pub trait One {
        fn one() -> Self;
    }

    impl One for u32 {
        fn one() -> Self {
            1
        }
    }

    impl One for u64 {
        fn one() -> Self {
            1
        }
    }

    impl One for u128 {
        fn one() -> Self {
            1
        }
    }

    pub trait ModOp<T> {
        fn zero(&self) -> T;
        fn one(&self) -> T;
        fn modulus(&self) -> T;
        fn add(&self, lhs: T, rhs: T) -> T;
        fn sub(&self, lhs: T, rhs: T) -> T;
        fn mul(&self, lhs: T, rhs: T) -> T;
        fn transform(&self, x: T) -> T {
            x
        }
        fn reduce(&self, x: T) -> T {
            x
        }
    }

    pub trait PowBy<T, E> {
        fn pow(&self, base: T, exp: E) -> T;
    }

    pub trait InvOp<T> {
        fn inv(&self, n: T) -> T;
    }

    impl<T: Clone, M: ModOp<T>> PowBy<T, u32> for M {
        fn pow(&self, mut base: T, mut exp: u32) -> T {
            let mut res = self.one();
            while exp > 0 {
                if exp % 2 == 1 {
                    res = self.mul(res, base.clone());
                }
                base = self.mul(base.clone(), base);
                exp >>= 1;
            }
            res
        }
    }

    impl<T: Clone, M: ModOp<T>> PowBy<T, u64> for M {
        fn pow(&self, mut base: T, mut exp: u64) -> T {
            let mut res = self.one();
            while exp > 0 {
                if exp % 2 == 1 {
                    res = self.mul(res, base.clone());
                }
                base = self.mul(base.clone(), base);
                exp >>= 1;
            }
            res
        }
    }

    impl<T: Clone, M: ModOp<T>> PowBy<T, u128> for M {
        fn pow(&self, mut base: T, mut exp: u128) -> T {
            let mut res = self.one();
            while exp > 0 {
                if exp % 2 == 1 {
                    res = self.mul(res, base.clone());
                }
                base = self.mul(base.clone(), base);
                exp >>= 1;
            }
            res
        }
    }

    impl<M: ModOp<u32>> InvOp<u32> for M {
        fn inv(&self, n: u32) -> u32 {
            self.pow(n, self.modulus() - 2)
        }
    }

    impl<M: ModOp<u64>> InvOp<u64> for M {
        fn inv(&self, n: u64) -> u64 {
            self.pow(n, self.modulus() - 2)
        }
    }

    impl<M: ModOp<u128>> InvOp<u128> for M {
        fn inv(&self, n: u128) -> u128 {
            self.pow(n, self.modulus() - 2)
        }
    }

    pub struct NaiveModOp<T> {
        m: T,
    }

    impl<T> NaiveModOp<T> {
        pub fn new(m: T) -> Self {
            Self { m }
        }
    }

    impl<T> ModOp<T> for NaiveModOp<T>
    where
        T: Copy
            + Default
            + One
            + Add<Output = T>
            + Sub<Output = T>
            + Mul<Output = T>
            + Rem<Output = T>
            + PartialOrd,
    {
        fn zero(&self) -> T {
            T::default()
        }
        fn one(&self) -> T {
            T::one()
        }
        fn modulus(&self) -> T {
            self.m
        }
        fn add(&self, lhs: T, rhs: T) -> T {
            let res = lhs + rhs;
            if res >= self.m {
                res - self.m
            } else {
                res
            }
        }
        fn sub(&self, lhs: T, rhs: T) -> T {
            if lhs >= rhs {
                lhs - rhs
            } else {
                lhs + self.m - rhs
            }
        }
        fn mul(&self, lhs: T, rhs: T) -> T {
            (lhs * rhs) % self.m
        }
        fn transform(&self, x: T) -> T {
            x % self.m
        }
        fn reduce(&self, x: T) -> T {
            x % self.m
        }
    }

    // Montgomery reduction
    #[derive(Debug, Clone)]
    pub struct Montgomery<T> {
        m: T,
        m_inv: T,
        r2: T,
    }

    impl Montgomery<u32> {
        pub fn new(m: u32) -> Self {
            debug_assert!(m % 2 == 1, "modulus must be coprime with 2");
            let mut m_inv = 1u32;
            for _ in 0..5 {
                m_inv = m_inv.wrapping_mul(2u32.wrapping_sub(m.wrapping_mul(m_inv)));
            }
            let r = m.wrapping_neg() % m;
            let r2 = (r as u64 * r as u64 % m as u64) as u32;

            Self { m, m_inv, r2 }
        }

        fn reduce_double(&self, x: u64) -> u32 {
            debug_assert!((x as u64) < (self.m as u64) * (self.m as u64));
            let q = (x as u32).wrapping_mul(self.m_inv);
            let a = ((q as u64 * self.m as u64) >> 32) as u32;
            let mut res = (x >> 32) as u32 + self.m - a;
            if res >= self.m {
                res -= self.m;
            }
            res as u32
        }
    }

    impl ModOp<u32> for Montgomery<u32> {
        fn zero(&self) -> u32 {
            0
        }
        fn one(&self) -> u32 {
            self.transform(1)
        }
        fn modulus(&self) -> u32 {
            self.m
        }
        fn mul(&self, x: u32, y: u32) -> u32 {
            debug_assert!(x < self.m);
            debug_assert!(y < self.m);
            self.reduce_double(x as u64 * y as u64)
        }

        fn add(&self, x: u32, y: u32) -> u32 {
            debug_assert!(x < self.m);
            debug_assert!(y < self.m);
            let sum = x + y;
            if sum >= self.m {
                sum - self.m
            } else {
                sum
            }
        }

        fn sub(&self, x: u32, y: u32) -> u32 {
            debug_assert!(x < self.m);
            debug_assert!(y < self.m);
            if x >= y {
                x - y
            } else {
                x + self.m - y
            }
        }

        fn reduce(&self, x: u32) -> u32 {
            self.reduce_double(x as u64)
        }

        fn transform(&self, x: u32) -> u32 {
            debug_assert!(x < self.m);
            self.mul(x, self.r2)
        }
    }

    impl Montgomery<u64> {
        pub fn new(m: u64) -> Self {
            debug_assert!(m % 2 == 1, "modulus must be coprime with 2");
            let mut m_inv = 1u64;
            for _ in 0..6 {
                // More iterations may be needed for u64 precision
                m_inv = m_inv.wrapping_mul(2u64.wrapping_sub(m.wrapping_mul(m_inv)));
            }
            let r = m.wrapping_neg() % m;
            let r2 = (r as u128 * r as u128 % m as u128) as u64;

            Self { m, m_inv, r2 }
        }

        pub fn reduce_double(&self, x: u128) -> u64 {
            debug_assert!((x as u128) < (self.m as u128) * (self.m as u128));
            let q = (x as u64).wrapping_mul(self.m_inv);
            let a = ((q as u128 * self.m as u128) >> 64) as u64;
            let mut res = (x >> 64) as u64 + self.m - a;
            if res >= self.m {
                res -= self.m;
            }
            res as u64
        }
    }

    impl ModOp<u64> for Montgomery<u64> {
        fn zero(&self) -> u64 {
            0
        }

        fn one(&self) -> u64 {
            self.transform(1)
        }

        fn modulus(&self) -> u64 {
            self.m
        }

        fn mul(&self, x: u64, y: u64) -> u64 {
            debug_assert!(x < self.m);
            debug_assert!(y < self.m);
            self.reduce_double(x as u128 * y as u128)
        }

        fn add(&self, x: u64, y: u64) -> u64 {
            debug_assert!(x < self.m);
            debug_assert!(y < self.m);
            let sum = x + y;
            if sum >= self.m {
                sum - self.m
            } else {
                sum
            }
        }

        fn sub(&self, x: u64, y: u64) -> u64 {
            debug_assert!(x < self.m);
            debug_assert!(y < self.m);
            if x >= y {
                x - y
            } else {
                x + self.m - y
            }
        }

        fn reduce(&self, x: u64) -> u64 {
            self.reduce_double(x as u128)
        }

        fn transform(&self, x: u64) -> u64 {
            debug_assert!(x < self.m);
            self.mul(x, self.r2)
        }
    }
}

// p must be prime of the form 4k + 3
fn discrete_sqrt_u32(a: u32, mod_op: &impl ModOp<u32>) -> Option<u32> {
    debug_assert_eq!(mod_op.modulus() % 4, 3);
    let p = mod_op.modulus();

    if a == 0 {
        return Some(0);
    }
    if mod_op.pow(a, (p - 1) / 2) != mod_op.one() {
        return None;
    }

    let r1 = mod_op.pow(a, (p + 1) / 4);
    let r2 = mod_op.sub(0, r1);
    Some(if mod_op.reduce(r1) < mod_op.reduce(r2) {
        r1
    } else {
        r2
    })
}

// root of x^4=a^2 (mod p)
fn root(a: u32, mod_op: &impl ModOp<u32>) -> u32 {
    match (
        discrete_sqrt_u32(a, mod_op),
        discrete_sqrt_u32(mod_op.sub(0, a), mod_op),
    ) {
        (Some(r1), Some(r2)) => {
            if mod_op.reduce(r1) < mod_op.reduce(r2) {
                r1
            } else {
                r2
            }
        }
        (Some(r1), None) => r1,
        (None, Some(r2)) => r2,
        _ => panic!(),
    }
}

mod parser {
    use crate::{InvOp, ModOp};

    type Grid<T> = Vec<Vec<T>>;

    pub struct Context<M: ModOp<u32>> {
        pub grid: Grid<u8>,
        pub mod_op: M,
    }

    fn eval_binop(ctx: &Context<impl ModOp<u32>>, op: u8, lhs: u32, rhs: u32) -> u32 {
        match op {
            b'+' => ctx.mod_op.add(lhs, rhs),
            b'-' => ctx.mod_op.sub(lhs, rhs),
            b'*' => ctx.mod_op.mul(lhs, rhs),
            b'/' if rhs == 0 => ctx.mod_op.transform(19981204),
            b'/' => ctx.mod_op.mul(lhs, ctx.mod_op.inv(rhs)),
            _ => unreachable!("Unsupported operation byte: {}", op),
        }
    }

    fn is_eof(view: &[usize; 4]) -> bool {
        let [_, _, c0, c1] = view;
        c0 > c1
    }

    fn op(ctx: &Context<impl ModOp<u32>>, view: &mut [usize; 4]) -> Option<u8> {
        let [r0, r1, c0, c1] = *view;
        let res = (r0..=r1)
            .map(|i| ctx.grid[i][c0])
            .find(|&op| matches!(op, b'+' | b'-' | b'*' | b'/'))?;
        *view = [r0, r1, c0 + 1, c1];
        Some(res)
    }

    fn space_col(ctx: &Context<impl ModOp<u32>>, view: &mut [usize; 4]) -> Option<()> {
        let [r0, r1, c0, c1] = *view;
        if !(r0..=r1).all(|i| ctx.grid[i][c0] == b' ') {
            return None;
        }
        *view = [r0, r1, c0 + 1, c1];
        Some(())
    }

    fn num(ctx: &Context<impl ModOp<u32>>, view: &mut [usize; 4]) -> Option<u32> {
        let [r0, r1, c0, c1] = *view;
        let i0 = (r0..=r1).find(|&i| ctx.grid[i][c0].is_ascii_digit())?;
        let mut c_end = c0;
        while c_end + 1 <= c1 && ctx.grid[i0][c_end + 1].is_ascii_digit() {
            c_end += 1;
        }

        let x = (c0..=c_end)
            .map(|j| ctx.mod_op.transform((ctx.grid[i0][j] - b'0') as u32))
            .fold(0, |acc, d| {
                ctx.mod_op
                    .add(ctx.mod_op.mul(acc, ctx.mod_op.transform(10)), d)
            });
        *view = [r0, r1, c_end + 1, c1];
        Some(x)
    }

    fn parens(ctx: &Context<impl ModOp<u32>>, view: &mut [usize; 4]) -> Option<u32> {
        let [r0, r1, c0, c1] = *view;
        if r1 - r0 == 0 {
            if !(ctx.grid[r0][c0] == b'(') {
                return None;
            }

            *view = [r0, r1, c0 + 1, c1];
            let x = expr(ctx, view);

            let [r0, r1, c0, c1] = *view;
            assert_eq!(ctx.grid[r0][c0], b')');
            *view = [r0, r1, c0 + 1, c1];
            Some(x)
        } else {
            if !(ctx.grid[r0][c0] == b'/'
                && ctx.grid[r1][c0] == b'\\'
                && (r0 + 1..r1).all(|i| ctx.grid[i][c0] == b'|'))
            {
                return None;
            }

            *view = [r0, r1, c0 + 1, c1];
            let x = expr(ctx, view);

            let [r0, r1, c0, c1] = *view;
            assert!(
                ctx.grid[r0][c0] == b'\\'
                    && ctx.grid[r1][c0] == b'/'
                    && (r0 + 1..r1).all(|i| ctx.grid[i][c0] == b'|')
            );
            *view = [r0, r1, c0 + 1, c1];
            Some(x)
        }
    }

    fn root(ctx: &Context<impl ModOp<u32>>, view: &mut [usize; 4]) -> Option<u32> {
        let [r0, r1, c0, c1] = *view;
        if r1 - r0 < 1 || c1 - c0 < 3 {
            return None;
        }

        if !(&ctx.grid[r0][c0..=c0 + 2] == b"  +"
            && &ctx.grid[r1][c0..=c0 + 2] == b"/\\|"
            && (r0 + 1..r1).all(|i| ctx.grid[i][c0 + 2] == b'|'))
        {
            return None;
        }

        let mut c_end = c0 + 3;
        assert_eq!(ctx.grid[r0][c_end], b'-');
        while c_end + 1 <= c1 && ctx.grid[r0][c_end + 1] == b'-' {
            c_end += 1;
        }

        let x = expr(ctx, &mut [r0 + 1, r1, c0 + 3, c_end]);
        *view = [r0, r1, c_end + 1, c1];
        Some(crate::root(x, &ctx.mod_op))
    }

    fn frac(ctx: &Context<impl ModOp<u32>>, view: &mut [usize; 4]) -> Option<u32> {
        let [r0, r1, c0, c1] = *view;
        if r1 - r0 < 2 || c1 - c0 < 2 {
            return None;
        }

        let i0 = (r0 + 1..r1).find(|&i| ctx.grid[i][c0] == b'-')?;
        let mut c_end = c0;
        while c_end + 1 <= c1 && ctx.grid[i0][c_end + 1] == b'-' {
            c_end += 1;
        }

        let lhs = expr(ctx, &mut [r0, i0 - 1, c0 + 1, c_end - 1]);
        let op = b'/';
        let rhs = expr(ctx, &mut [i0 + 1, r1, c0 + 1, c_end - 1]);
        *view = [r0, r1, c_end + 1, c1];
        Some(eval_binop(ctx, op, lhs, rhs))
    }

    fn factor(ctx: &Context<impl ModOp<u32>>, view: &mut [usize; 4]) -> u32 {
        num(ctx, view)
            .or_else(|| parens(ctx, view))
            .or_else(|| root(ctx, view))
            .or_else(|| frac(ctx, view))
            .unwrap()
    }

    fn expr(ctx: &Context<impl ModOp<u32>>, view: &mut [usize; 4]) -> u32 {
        // 3-Pass evaluation

        // 1. Evaluate all factors
        let lhs = factor(ctx, view);
        let mut rest = vec![];
        while !is_eof(view) && space_col(ctx, view).is_some() {
            let op = op(ctx, view).unwrap();
            space_col(ctx, view).unwrap();
            let rhs = factor(ctx, view);
            rest.push((op, rhs));
        }

        // 2. Evaluate */
        let mut terms = vec![];
        let mut ops = vec![];

        let mut acc = lhs;
        for (op, rhs) in rest {
            if op == b'*' || op == b'/' {
                acc = eval_binop(ctx, op, acc, rhs);
            } else {
                terms.push(acc);
                ops.push(op);
                acc = rhs;
            }
        }
        terms.push(acc);

        // 3. Evaluate +-
        let mut res = terms[0];
        for (&op, &term) in ops.iter().zip(&terms[1..]) {
            res = eval_binop(ctx, op, res, term);
        }
        res
    }

    pub fn parse_and_eval(ctx: &Context<impl ModOp<u32>>) -> u32 {
        let r = ctx.grid.len();
        let c = ctx.grid[0].len();
        let mut view = [0, r - 1, 0, c - 1];
        let res = expr(ctx, &mut view);
        assert!(is_eof(&view));
        ctx.mod_op.reduce(res)
    }
}
fn main() {
    // let mut input = simple_io::stdin_at_once();
    let mut input = std::io::BufReader::new(std::io::stdin()).lines().flatten();
    let mut output = simple_io::stdout();

    let line = input.next().unwrap();
    let mut tokens = line.split_ascii_whitespace();
    let r: usize = tokens.next().unwrap().parse().unwrap();
    let c: usize = tokens.next().unwrap().parse().unwrap();

    let grid: Vec<Vec<u8>> = (0..r)
        .map(|_| input.next().unwrap().as_bytes()[..c].to_vec())
        .collect();

    let p = 1_000_000_007;
    let mod_op = Montgomery::<u32>::new(p);
    // let mod_op = NaiveModOp::new(p);

    let ctx = parser::Context { grid, mod_op };

    let ans = parser::parse_and_eval(&ctx) % p;
    writeln!(output, "{}", ans).unwrap();
}
