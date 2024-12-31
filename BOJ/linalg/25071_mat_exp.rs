use std::io::Write;

mod simple_io {
    pub struct InputAtOnce<'a> {
        _buf: String,
        iter: std::str::SplitAsciiWhitespace<'a>,
    }

    impl<'a> InputAtOnce<'a> {
        pub fn token(&mut self) -> &'a str {
            self.iter.next().unwrap_or_default()
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().unwrap()
        }
    }

    pub fn stdin<'a>() -> InputAtOnce<'a> {
        let _buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let iter = _buf.split_ascii_whitespace();
        let iter = unsafe { std::mem::transmute(iter) };
        InputAtOnce { _buf, iter }
    }

    pub fn stdout() -> std::io::BufWriter<std::io::Stdout> {
        std::io::BufWriter::new(std::io::stdout())
    }
}

const NEG_INF: i64 = i64::MIN / 10000;

type Mat = Vec<Vec<i64>>;

fn mul_mat(lhs: &Mat, rhs: &Mat) -> Mat {
    let n = lhs.len();
    let mut res = vec![vec![NEG_INF; n]; n];
    for i in 0..n {
        for j in 0..n {
            for k in 0..n {
                res[i][j] = res[i][j].max(lhs[i][k] + rhs[k][j]);
            }
        }
    }
    res
}

fn apply_mat(state: &Vec<i64>, mat: &Mat) -> Vec<i64> {
    let n = state.len();
    let mut res = vec![NEG_INF; n];
    for i in 0..n {
        for j in 0..n {
            res[j] = res[j].max(state[i] + mat[i][j]);
        }
    }
    res
}

struct PowQuery(Vec<Mat>);

impl PowQuery {
    fn new(base: Mat, max_exp: u32) -> Self {
        let mut queries = vec![];
        let mut base = base.clone();
        for _ in 0..max_exp {
            queries.push(base.clone());
            base = mul_mat(&base, &base);
        }
        Self(queries)
    }

    fn step(&self, initial: &Vec<i64>, exp: u32) -> Vec<i64> {
        let mut state = initial.clone();
        for i in 0..32 {
            if exp == 0 {
                break;
            }
            if exp & (1 << i) != 0 {
                state = apply_mat(&state, &self.0[i]);
            }
        }
        state
    }
}

fn main() {
    let mut input = simple_io::stdin();
    let mut output = simple_io::stdout();

    let n: usize = input.value();
    let m: usize = input.value();
    let t: u64 = input.value();
    let k: usize = input.value();

    let cs: Vec<i64> = (0..n).map(|_| input.value()).collect();
    let n_verts = n * 5;

    let mut base_mat = vec![vec![NEG_INF; n_verts]; n_verts];
    for i in 0..n {
        for t in 1..5 {
            base_mat[i * 5 + t][i * 5 + t - 1] = 0;
        }
    }

    for _ in 0..m {
        let u = input.value::<usize>() - 1;
        let v = input.value::<usize>() - 1;
        let w = input.value::<usize>();
        base_mat[u * 5 + 0][v * 5 + (w - 1)] = 0;
    }

    for src in 0..n * 5 {
        for dest in 0..n {
            if base_mat[src][dest * 5 + 0] != NEG_INF {
                base_mat[src][dest * 5 + 0] += cs[dest];
            }
        }
    }

    let mut festivals = vec![];
    for _ in 0..k {
        let t = input.value::<u32>();
        let dest = input.value::<usize>() - 1;
        let y = input.value::<i64>();

        festivals.push((t, dest, y));
    }
    festivals.sort_unstable_by_key(|(t, ..)| *t);

    let mut state = vec![NEG_INF; n_verts];
    state[0] = 0;

    let pow_query = PowQuery::new(base_mat.clone(), 32);

    let mut t_prev = 0;
    for (t, dest, y) in festivals {
        state = pow_query.step(&state, t - t_prev);
        state[dest * 5 + 0] += y;

        t_prev = t;
    }
    state = pow_query.step(&state, t as u32 - t_prev);

    let mut ans = state[0] + cs[0];
    if ans < 0 {
        ans = -1;
    }
    writeln!(output, "{}", ans).unwrap();
}
