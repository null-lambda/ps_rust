struct RootedForest;
type RootIdx = u32;

#[derive(Clone, Copy, Default)]
pub struct Cluster {
    lazy_root: Option<u32>,
}

impl ClusterCx for RootedForest {
    type V = RootIdx;

    type C = Cluster;
    type R = Cluster;

    fn id_compress() -> Self::C {
        Default::default()
    }
    fn compress(&self, _children: [&Self::C; 2], _v: &Self::V, _rake: Option<&Self::R>) -> Self::C {
        Default::default()
    }

    fn id_rake() -> Self::R {
        Default::default()
    }
    fn rake(&self, _children: [&Self::R; 2]) -> Self::R {
        Default::default()
    }

    fn collapse_path(&self, _c: &Self::C, _v: &Self::V) -> Self::R {
        Default::default()
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
        if let Some(r) = node.lazy_root.take() {
            children[0].lazy_root = Some(r);
            children[1].lazy_root = Some(r);
            *v = r;
            if let Some(rake) = rake {
                rake.lazy_root = Some(r);
            }
        }
    }

    fn push_down_rake(&self, node: &mut Self::R, children: [&mut Self::R; 2]) {
        if let Some(r) = node.lazy_root.take() {
            children[0].lazy_root = Some(r);
            children[1].lazy_root = Some(r);
        }
    }

    #[allow(unused_variables)]
    fn push_down_collapsed(&self, node: &mut Self::R, c: &mut Self::C, vr: &mut Self::V) {
        if let Some(r) = node.lazy_root.take() {
            c.lazy_root = Some(r);
            *vr = r;
        }
    }
}

impl Action<RootedForest> for RootIdx {
    fn apply_to_compress(
        &mut self,
        compress: &mut <RootedForest as ClusterCx>::C,
        range: ActionRange,
    ) {
        debug_assert!(range == ActionRange::Subtree);
        compress.lazy_root = Some(*self);
    }

    fn apply_to_rake(&mut self, rake: &mut <RootedForest as ClusterCx>::R) {
        rake.lazy_root = Some(*self);
    }

    fn apply_to_weight(&mut self, weight: &mut <RootedForest as ClusterCx>::V) {
        *weight = *self;
    }
}
