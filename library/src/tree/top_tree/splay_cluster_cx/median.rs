type X = u64;

#[derive(Clone, Copy, Default)]
struct Compress {
    count: u32,
    len: X,
    left: X,
    right: X,
}

#[derive(Clone, Copy, Default)]
struct Rake {
    count: u32,
    left: X,
}

struct WeightedMedian;

impl Compress {
    fn singleton(x: X) -> Self {
        Self {
            count: 0,
            len: x,
            left: 0,
            right: 0,
        }
    }
}

impl ClusterCx for WeightedMedian {
    type V = bool;

    type C = Compress;
    type R = Rake;

    fn id_compress() -> Self::C {
        Default::default()
    }

    fn compress(&self, children: [&Self::C; 2], v: &Self::V, rake: Option<&Self::R>) -> Self::C {
        let rake = rake.copied().unwrap_or(Self::id_rake());
        let count = children[0].count + *v as u32 + children[1].count + rake.count;
        Compress {
            len: children[0].len + children[1].len,
            count,
            left: children[0].left
                + children[1].left
                + rake.left
                + children[0].len * (count - children[0].count) as X,
            right: children[1].right
                + children[0].right
                + rake.left
                + children[1].len * (count - children[1].count) as X,
        }
    }

    fn id_rake() -> Self::R {
        Default::default()
    }

    fn rake(&self, children: [&Self::R; 2]) -> Self::R {
        Rake {
            count: children[0].count + children[1].count,
            left: children[0].left + children[1].left,
        }
    }

    fn collapse_path(&self, c: &Self::C, vr: &Self::V) -> Self::R {
        Rake {
            count: c.count + *vr as u32,
            left: c.left + c.len * *vr as X,
        }
    }

    fn reverse(&self, c: &Self::C) -> Self::C {
        Compress {
            left: c.right,
            right: c.left,
            ..c.clone()
        }
    }
}

fn weighted_moment(tt: &mut TopTree<WeightedMedian>, u: usize) -> X {
    let moment = |tt: &mut TopTree<_>, u: usize| {
        let (_v, rest) = tt.sum_rerooted(u);
        rest.map_or(0, |r: Rake| r.left)
    };
    tt.center_edge(u, |_v_pivot, r0, r1| r0.count < r1.count)
        .map_or(0, |e| moment(tt, e[0]).min(moment(tt, e[1])))
}
