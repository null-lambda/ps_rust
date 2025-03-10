impl ClusterCx for () {
    type V = ();

    type C = ();
    type R = ();

    fn id_compress() -> Self::C {}
    fn compress(&self, _: [&Self::C; 2], _: &Self::V, _: Option<&Self::R>) -> Self::C {}

    fn id_rake() -> Self::R {}
    fn rake(&self, _: [&Self::R; 2]) -> Self::R {}

    fn collapse_path(&self, _: &Self::C, _: &Self::V) -> Self::R {}

    fn reverse(&self, _: &Self::C) -> Self::C {}
}
