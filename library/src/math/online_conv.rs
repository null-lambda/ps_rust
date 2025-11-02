pub mod online_conv {
    // # Reference
    // - [https://codeforces.com/blog/entry/111399]
    // - [https://infossm.github.io/blog/2025/03/23/online-fft/]
    use crate::algebra::Field;
    use crate::ntt::NTTSpec;
    use crate::poly::Poly;

    #[derive(Default, Clone)]
    pub struct Mul<M> {
        pub ls: Vec<M>,
        pub rs: Vec<M>,
        pub res: Vec<M>,
    }

    impl<M: NTTSpec + PartialEq + Field> Mul<M> {
        pub fn push(&mut self, l: M, r: M) -> M {
            let k = self.ls.len();
            self.ls.push(l);
            self.rs.push(r);

            let mut touch = |i: usize, j: usize, s: usize| {
                if self.res.len() < k + s * 2 - 1 {
                    self.res.resize(k + s * 2 - 1, M::zero());
                }

                let ls = Poly::new(self.ls[i..][..s].to_vec());
                let rs = Poly::new(self.rs[j..][..s].to_vec());
                let zs = ls * rs;
                for l in 0..s * 2 - 1 {
                    self.res[k..][l] += zs.coeff(l);
                }
            };

            let p = (k + 2).next_power_of_two() != k + 2;
            let t = (k + 2).trailing_zeros();

            for l in 0..t + p as u32 * 2 - 1 {
                let i = (1 << l) - 1;
                touch(i, k - i, 1 << l);
            }
            for l in (0..t + p as u32).rev() {
                let i = k + 1 - (1 << l);
                touch(i, k - i, 1 << l);
            }

            self.res[k].clone()
        }
    }

    // TODO: exp, unlabeled list, log_1p, ... or generic newton method
    // do I really need them?
}
