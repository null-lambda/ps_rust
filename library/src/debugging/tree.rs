pub mod debug {
    pub mod tree {
        #[derive(Clone)]
        pub struct Pretty(pub String, pub Vec<Pretty>);

        impl Pretty {
            fn fmt_rec(
                &self,
                f: &mut std::fmt::Formatter,
                prefix: &str,
                first: bool,
                last: bool,
            ) -> std::fmt::Result {
                let space = format!("{}   ", prefix);
                let bar = format!("{}|  ", prefix);
                let sep = if first && last {
                    "*--"
                } else if first {
                    "┌--"
                } else if last {
                    "└--"
                } else {
                    "+--"
                };

                let m = self.1.len();
                for i in 0..m / 2 {
                    let c = &self.1[i];
                    let prefix_ext = if first && i == 0 { &space } else { &bar };
                    c.fmt_rec(f, &prefix_ext, i == 0, i == m - 1)?;
                }

                writeln!(f, "{}{}{}", prefix, sep, self.0)?;

                for i in m / 2..m {
                    let c = &self.1[i];
                    let prefix_ext = if last && i == m - 1 { &space } else { &bar };
                    c.fmt_rec(f, &prefix_ext, i == 0, i == m - 1)?;
                }

                Ok(())
            }
        }

        impl<'a> std::fmt::Debug for Pretty {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                writeln!(f)?;
                self.fmt_rec(f, "", true, true)
            }
        }
    }
}
