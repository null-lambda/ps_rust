use std::{collections::HashMap, hash::Hash};

fn compress_coord<T: Ord + Clone + Hash>(
    xs: impl IntoIterator<Item = T>,
) -> (Vec<T>, HashMap<T, u32>) {
    let mut x_map: Vec<T> = xs.into_iter().collect();
    x_map.sort_unstable();
    x_map.dedup();

    let x_map_inv = x_map
        .iter()
        .cloned()
        .enumerate()
        .map(|(i, x)| (x, i as u32))
        .collect();

    (x_map, x_map_inv)
}

use std::{collections::HashMap, hash::Hash};

fn gen_index_mapper<T: Eq + Hash>() -> impl FnMut(T) -> u32 {
    let mut map = HashMap::new();
    move |x| {
        let idx = map.len() as u32;
        *map.entry(x).or_insert_with(|| idx)
    }
}
