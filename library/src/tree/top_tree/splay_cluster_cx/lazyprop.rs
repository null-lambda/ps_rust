type X = i32;
const INF: X = X::MAX;

// Tropical semirings
#[derive(Debug, Clone, Copy)]
struct Tropical {
    sum: X,
    min: X,
    max: X,
}

#[derive(Default, Debug, Clone, Copy)]
struct ResetAdd {
    reset: bool,
    delta: X,
}

#[derive(Debug, Default, Clone, Copy)]
struct LazyNode {
    x: Tropical,
    count: u32,
    lazy: ResetAdd,
}

#[derive(Debug, Default, Clone, Copy)]
struct Compress {
    path: LazyNode,
    subtree: LazyNode,
}

struct TropicalOp;

impl Default for Tropical {
    fn default() -> Self {
        Self {
            sum: 0,
            min: INF,
            max: -INF,
        }
    }
}

impl From<X> for Tropical {
    fn from(x: X) -> Self {
        Self {
            sum: x,
            min: x,
            max: x,
        }
    }
}

impl Tropical {
    fn combine(&self, other: &Self) -> Self {
        Self {
            sum: self.sum + other.sum,
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    fn accept(&mut self, action: ResetAdd, count: u32) {
        if count == 0 {
            return;
        }
        if action.reset {
            self.sum = action.delta * count as X;
            self.min = action.delta;
            self.max = action.delta;
        } else {
            self.sum += action.delta * count as X;
            self.min += action.delta;
            self.max += action.delta;
        }
    }
}

impl From<Tropical> for LazyNode {
    fn from(x: Tropical) -> Self {
        Self {
            x,
            count: 1,
            lazy: Default::default(),
        }
    }
}

impl ResetAdd {
    fn overwrite(x: i32) -> Self {
        Self {
            reset: true,
            delta: x as X,
        }
    }

    fn add(x: i32) -> Self {
        Self {
            reset: false,
            delta: x as X,
        }
    }

    fn combine(&self, other: &Self) -> Self {
        if self.reset {
            *self
        } else {
            Self {
                reset: other.reset,
                delta: self.delta + other.delta,
            }
        }
    }
}

impl LazyNode {
    fn accept(&mut self, action: ResetAdd) {
        self.x.accept(action, self.count);
        self.lazy = action.combine(&self.lazy);
    }
}

impl ClusterCx for TropicalOp {
    type V = X;

    type C = Compress;
    type R = LazyNode;

    fn id_compress() -> Self::C {
        Default::default()
    }

    fn compress(&self, children: [&Self::C; 2], v: &Self::V, rake: Option<&Self::R>) -> Self::C {
        let v = LazyNode::from(Tropical::from(*v));
        let mut r = Self::id_rake();
        if let Some(rake) = rake {
            r = self.rake([&r, rake]);
        }

        Self::C {
            path: self.rake([&self.rake([&children[0].path, &children[1].path]), &v]),
            subtree: self.rake([&self.rake([&children[0].subtree, &children[1].subtree]), &r]),
        }
    }

    fn id_rake() -> Self::R {
        Default::default()
    }

    fn rake(&self, children: [&Self::R; 2]) -> Self::R {
        Self::R {
            x: children[0].x.combine(&children[1].x),
            count: children[0].count + children[1].count,
            ..Default::default()
        }
    }

    fn collapse_path(&self, c: &Self::C, vr: &Self::V) -> Self::R {
        self.rake([
            &self.rake([&c.path, &c.subtree]),
            &LazyNode::from(Tropical::from(*vr)),
        ])
    }

    fn reverse(&self, c: &Self::C) -> Self::C {
        *c
    }

    fn push_down_compress(
        &self,
        node: &mut Self::C,
        children: [&mut Self::C; 2],
        v: &mut Self::V,
        rake: Option<&mut Self::R>,
    ) {
        children[0].path.accept(node.path.lazy);
        children[1].path.accept(node.path.lazy);
        node.path.lazy.apply_to_weight(v);
        node.path.lazy = Default::default();

        children[0].subtree.accept(node.subtree.lazy);
        children[1].subtree.accept(node.subtree.lazy);
        if let Some(rake) = rake {
            rake.accept(node.subtree.lazy);
        }
        node.subtree.lazy = Default::default();
    }

    fn push_down_rake(&self, node: &mut Self::R, children: [&mut Self::R; 2]) {
        children[0].accept(node.lazy);
        children[1].accept(node.lazy);
        node.lazy = Default::default();
    }

    fn push_down_collapsed(&self, node: &mut Self::R, c: &mut Self::C, vr: &mut Self::V) {
        c.path.accept(node.lazy);
        c.subtree.accept(node.lazy);
        node.lazy.apply_to_weight(vr);
        node.lazy = Default::default();
    }
}

impl Action<TropicalOp> for ResetAdd {
    fn apply_to_compress(
        &mut self,
        compress: &mut <TropicalOp as ClusterCx>::C,
        range: ActionRange,
    ) {
        match range {
            ActionRange::Subtree => {
                compress.path.accept(*self);
                compress.subtree.accept(*self);
            }
            ActionRange::Path => {
                compress.path.accept(*self);
            }
        }
    }

    fn apply_to_rake(&mut self, rake: &mut <TropicalOp as ClusterCx>::R) {
        rake.accept(*self);
    }

    fn apply_to_weight(&mut self, weight: &mut <TropicalOp as ClusterCx>::V) {
        if self.reset {
            *weight = self.delta;
        } else {
            *weight += self.delta;
        }
    }
}
