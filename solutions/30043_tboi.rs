use std::collections::{HashSet, VecDeque};
use std::iter::repeat;

use collections::{neighbors_e_ws_n, neighbors_eswn, Grid, PrettyColored};
use iter::product;

mod simple_io {
    pub struct InputAtOnce(std::str::SplitAsciiWhitespace<'static>);

    impl InputAtOnce {
        pub fn token(&mut self) -> &str {
            self.0.next().unwrap_or_default()
        }

        pub fn value<T: std::str::FromStr>(&mut self) -> T
        where
            T::Err: std::fmt::Debug,
        {
            self.token().parse().unwrap()
        }
    }

    pub fn stdin_at_once() -> InputAtOnce {
        let buf = std::io::read_to_string(std::io::stdin()).unwrap();
        let buf = Box::leak(buf.into_boxed_str());
        let iter = buf.split_ascii_whitespace();
        InputAtOnce(iter)
    }
}

#[derive(Debug, Clone)]
struct Rng(u64, u64);

impl Rng {
    const A: u64 = 1_103_515_245;
    const C: u64 = 12345;
    const M: u64 = 1 << 31;

    fn new(seed: u64) -> Self {
        Self(seed % Self::M, 0)
    }

    fn count_calls(&self) -> u64 {
        self.1
    }

    fn next(&mut self) -> u64 {
        let result = self.0;
        self.0 = (Self::A * self.0 + Self::C) % Self::M;
        self.1 += 1;
        result
    }

    fn next_u64(&mut self, l: u64, r: u64) -> u64 {
        l + self.next() % (r - l + 1)
    }

    fn chance(&mut self, p: u64) -> bool {
        self.next_u64(1, 100) <= p
    }

    fn choice<'a, T: Clone>(&mut self, xs: &'a [T]) -> T {
        xs[self.next_u64(0, xs.len() as u64 - 1) as usize].clone()
    }
}

#[allow(dead_code)]
pub mod iter {
    pub fn product<I, J>(i: I, j: J) -> impl Iterator<Item = (I::Item, J::Item)>
    where
        I: IntoIterator,
        I::Item: Clone,
        J: IntoIterator,
        J::IntoIter: Clone,
    {
        let j = j.into_iter();
        i.into_iter()
            .flat_map(move |x| j.clone().map(move |y| (x.clone(), y)))
    }
}

#[allow(dead_code)]
mod collections {
    use std::{
        cmp::Reverse,
        collections::HashMap,
        fmt::Display,
        iter::{empty, once},
        ops::{Index, IndexMut},
    };

    #[derive(Debug, Clone)]
    pub struct Grid<T> {
        pub w: usize,
        pub data: Vec<T>,
    }

    impl<T> Grid<T> {
        pub fn with_shape(self, w: usize) -> Self {
            debug_assert_eq!(self.data.len() % w, 0);
            Grid { w, data: self.data }
        }
    }

    impl<T> FromIterator<T> for Grid<T> {
        fn from_iter<I: IntoIterator<Item = T>>(iter: I) -> Self {
            Self {
                w: 1,
                data: iter.into_iter().collect(),
            }
        }
    }

    impl<T: Clone> Grid<T> {
        pub fn sized(fill: T, h: usize, w: usize) -> Self {
            Grid {
                w,
                data: vec![fill; w * h],
            }
        }
    }

    impl<T: Display> Display for Grid<T> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            for row in self.data.chunks(self.w) {
                for cell in row {
                    cell.fmt(f)?;
                    write!(f, " ")?;
                }
                writeln!(f)?;
            }
            writeln!(f)?;
            Ok(())
        }
    }

    impl<T> Index<(usize, usize)> for Grid<T> {
        type Output = T;
        fn index(&self, (i, j): (usize, usize)) -> &Self::Output {
            debug_assert!(i < self.data.len() / self.w && j < self.w);
            &self.data[i * self.w + j]
        }
    }

    impl<T> IndexMut<(usize, usize)> for Grid<T> {
        fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut Self::Output {
            debug_assert!(i < self.data.len() / self.w && j < self.w);
            &mut self.data[i * self.w + j]
        }
    }

    pub struct PrettyColored<'a>(&'a Grid<u8>);

    impl Display for PrettyColored<'_> {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            let colors = (once(37).chain(31..=36))
                .map(|i| (format!("\x1b[{}m", i), format!("\x1b[0m")))
                .collect::<Vec<_>>();

            let mut freq = HashMap::new();
            for c in self.0.data.iter() {
                *freq.entry(c).or_insert(0) += 1;
            }
            let mut freq = freq.into_iter().collect::<Vec<_>>();
            freq.sort_unstable_by_key(|(_, f)| Reverse(*f));

            let mut color_map = HashMap::new();
            let mut idx = 0;
            for (c, _) in freq {
                color_map.insert(c, &colors[idx % colors.len()]);
                idx += 1;
            }

            for row in self.0.data.chunks(self.0.w) {
                for cell in row {
                    let (pre, suff) = color_map[&cell];
                    write!(f, "{}{}{}", pre, *cell as char, suff)?;
                }
                writeln!(f)?;
            }
            Ok(())
        }
    }

    impl Grid<u8> {
        pub fn colored(&self) -> PrettyColored {
            PrettyColored(&self)
        }
    }

    pub fn neighbors_eswn(
        u: (usize, usize),
        h: usize,
        w: usize,
    ) -> impl Iterator<Item = (usize, usize)> {
        let (i, j) = u;
        empty()
            .chain((j + 1 < w).then(|| (i, j + 1)))
            .chain((i + 1 < h).then(|| (i + 1, j)))
            .chain((j > 0).then(|| (i, j - 1)))
            .chain((i > 0).then(|| (i - 1, j)))
    }

    pub fn neighbors_e_ws_n(
        u: (usize, usize),
        h: usize,
        w: usize,
    ) -> impl Iterator<Item = (usize, usize)> {
        let (i, j) = u;
        empty()
            .chain((j + 1 < w).then(|| (i, j + 1)))
            .chain((j > 0).then(|| (i, j - 1)))
            .chain((i + 1 < h).then(|| (i + 1, j)))
            .chain((i > 0).then(|| (i - 1, j)))
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum RoomType {
    Init,
    Normal,
    Boss,
    Secret,
    Treasure,
    Store,
    Angel,
    Devil,
    Sacrificial,
    Cursed,
}

impl RoomType {
    fn disconnected(&self) -> bool {
        matches!(self, Self::Angel | Self::Devil | Self::Secret)
    }

    fn inaccessible(&self) -> bool {
        matches!(self, Self::Angel | Self::Devil)
    }
}

#[derive(Debug, Clone)]
struct Room {
    ty: RoomType,
    neighbors: Vec<usize>,
    required_atk: i8,
}

impl Room {
    fn as_byte(&self) -> u8 {
        match self.ty {
            RoomType::Init => b'R',
            RoomType::Normal => b'0' + self.required_atk as u8,
            RoomType::Boss => b'B',
            RoomType::Secret => b'X',
            RoomType::Treasure => b'T',
            RoomType::Store => b'M',
            RoomType::Angel => b'A',
            RoomType::Devil => b'D',
            RoomType::Sacrificial => b'S',
            RoomType::Cursed => b'C',
        }
    }
}

impl Room {
    fn new(ty: RoomType) -> Self {
        Self {
            ty,
            neighbors: vec![],
            required_atk: 0,
        }
    }

    fn normal(atk: i8) -> Self {
        Self {
            ty: RoomType::Normal,
            neighbors: vec![],
            required_atk: atk,
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash)]
struct BitMask(u32);

impl BitMask {
    fn fill(n: usize) -> Self {
        Self((1 << n) - 1)
    }

    fn get(&self, v: usize) -> bool {
        self.0 & (1 << v) != 0
    }

    fn set(self, v: usize) -> Self {
        Self(self.0 | (1 << v))
    }
}

fn iter_bits(bitmask: BitMask, n: usize) -> impl Iterator<Item = usize> {
    (0..n).filter(move |&i| bitmask.get(i))
}

impl std::fmt::Debug for BitMask {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:0>32b}", self.0)
    }
}

#[derive(Debug, Clone)]
struct DungeonTopology {
    rooms: Vec<Room>,
    next_rooms: Vec<BitMask>,
    shop_pos: usize,
}

impl DungeonTopology {
    fn new(rooms: Vec<Room>) -> Self {
        let n = rooms.len();

        let shop_pos = rooms.iter().position(|room| room.ty == RoomType::Store);

        let unset = BitMask(1 << (n + 1));
        let mut next_rooms = vec![unset; 1 << n];
        Self::build_next_rooms(&rooms, &mut next_rooms, shop_pos, BitMask(0).set(0));

        Self {
            rooms,
            next_rooms,
            shop_pos: shop_pos.unwrap_or(n),
        }
    }

    fn build_next_rooms(
        rooms: &Vec<Room>,
        next_rooms: &mut Vec<BitMask>,
        shop_pos: Option<usize>,
        visited: BitMask,
    ) {
        next_rooms[visited.0 as usize] = BitMask(0);
        for u in iter_bits(visited, rooms.len()) {
            for &v in &rooms[u].neighbors {
                if !visited.get(v) {
                    next_rooms[visited.0 as usize] = next_rooms[visited.0 as usize].set(v);
                }
            }
            for &v in &rooms[u].neighbors {
                let n = rooms.len();
                if next_rooms[visited.set(v).0 as usize].0 == 1 << (n + 1) {
                    Self::build_next_rooms(rooms, next_rooms, shop_pos, visited.set(v));
                }
            }
        }
        if let Some(shop_pos) = shop_pos {
            if visited.get(shop_pos) {
                next_rooms[visited.0 as usize] = next_rooms[visited.0 as usize].set(shop_pos);
            }
        }
    }

    fn solve(&mut self) -> bool {
        let n = self.rooms.len();

        let init = GameState {
            hp: 6,
            atk: 1,
            coin: 0,
            bomb: 3,
            visited: BitMask(0).set(0),
        };
        let mut queue: VecDeque<GameState> = vec![init].into();
        let mut visited_states: HashSet<GameState> = HashSet::new();

        while let Some(state) = queue.pop_back() {
            // println!("{:?}", state);
            if visited_states.contains(&state) {
                continue;
            }
            visited_states.insert(state.clone());
            let GameState {
                hp,
                atk,
                coin,
                bomb,
                ..
            } = state;
            for v in iter_bits(self.next_rooms[state.visited.0 as usize], n) {
                let next = GameState {
                    visited: state.visited.set(v),
                    ..state
                };

                match self.rooms[v].ty {
                    RoomType::Init => panic!(),
                    RoomType::Normal => {
                        if atk >= self.rooms[v].required_atk {
                            queue.push_back(GameState {
                                coin: coin + 1,
                                ..next
                            });
                        } else if hp >= 2 {
                            queue.push_back(GameState {
                                hp: hp - 1,
                                coin: coin + 1,
                                ..next
                            });
                        }
                        if bomb >= 1 {
                            queue.push_back(GameState {
                                bomb: bomb - 1,
                                coin: coin + 1,
                                ..next
                            });
                        }
                    }
                    RoomType::Boss => {
                        if next.atk >= 10 {
                            return true;
                        }
                    }
                    RoomType::Secret => {
                        if next.bomb >= 1 {
                            queue.push_back(GameState {
                                hp: hp + 2,
                                atk: atk + 2,
                                coin: coin + 2,
                                bomb: bomb - 1,
                                ..next
                            });
                        }
                    }
                    RoomType::Treasure => {
                        queue.push_back(GameState {
                            atk: atk + 1,
                            ..next
                        });
                    }
                    RoomType::Store => {
                        if coin >= 2 {
                            queue.push_back(GameState {
                                coin: coin - 2,
                                hp: hp + 1,
                                ..next
                            });
                            queue.push_back(GameState {
                                coin: coin - 2,
                                atk: atk + 1,
                                ..next
                            });
                        }
                    }
                    RoomType::Angel => panic!(),
                    RoomType::Devil => panic!(),
                    RoomType::Sacrificial => {
                        if hp >= 3 {
                            queue.push_back(GameState {
                                hp: hp - 2,
                                atk: atk + 3,
                                ..next
                            });
                        }
                    }
                    RoomType::Cursed => {
                        queue.push_back(GameState {
                            atk: atk - 2,
                            coin: coin + 3,
                            bomb: bomb + 1,
                            ..next
                        });
                    }
                }
            }
        }

        false
    }

    fn print_graph(&self) {
        for (i, room) in self.rooms.iter().enumerate() {
            // for v in iter_bits(self.next_rooms[1 | (1 << i)], self.rooms.len()) {
            //     println!("{} -> {}", i, v);
            // }
            // println!();
            println!("{}: {:?}", i, room);
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct GameState {
    hp: i8,
    atk: i8,
    coin: u8,
    bomb: u8,
    visited: BitMask,
}

#[derive(Debug)]
struct DungeonGenerator {
    rng: Rng,
    n_rooms: usize,
    rooms: Vec<Room>,
    has_room: Grid<bool>,
    room_idx: Grid<Option<usize>>,
    room_pos: Vec<(usize, usize)>,
    init_pos: (usize, usize),
}

impl DungeonGenerator {
    fn new(seed: u64) -> Self {
        let rng = Rng::new(seed);
        Self {
            rng,
            n_rooms: 0,
            rooms: vec![],
            has_room: Grid::sized(false, 9, 9),
            room_idx: Grid::sized(None, 9, 9),
            room_pos: vec![],
            init_pos: (4, 4),
        }
    }

    fn run(&mut self) -> DungeonTopology {
        self.generate_map();
        self.assign_rooms();
        self.build_topology();
        DungeonTopology::new(self.rooms.clone())
    }

    fn generate_map(&mut self) {
        let Self {
            ref mut rng,
            ref mut has_room,
            ref mut room_pos,
            ref mut n_rooms,
            ..
        } = self;
        let init_pos = self.init_pos;

        // Step 1
        *n_rooms = rng.next_u64(10, 20) as usize;
        assert_eq!(rng.count_calls(), 1);

        // Step 2
        room_pos.push(init_pos);
        has_room[init_pos] = true;
        while room_pos.len() < *n_rooms + 1 {
            let mut queue: VecDeque<(usize, usize)> = [rng.choice(&room_pos)].into();
            while let Some(u) = queue.pop_front() {
                for v in neighbors_eswn(u, 9, 9) {
                    if room_pos.len() == *n_rooms + 1 {
                        break;
                    }
                    if has_room[v] {
                        continue;
                    }
                    let cnt = neighbors_eswn(v, 9, 9).filter(|&w| has_room[w]).count();
                    if cnt >= 2 {
                        continue;
                    }
                    if !rng.chance(50) {
                        continue;
                    }
                    room_pos.push(v);
                    queue.push_back(v);
                    has_room[v] = true;
                }
            }
        }
    }

    fn assign_rooms(&mut self) {
        let Self {
            ref mut rng,
            ref room_pos,
            ref mut rooms,
            ref mut room_idx,
            ref n_rooms,
            init_pos,
            ..
        } = self;
        let init_pos = *init_pos;

        // Step 3
        let mut normal = vec![];
        let mut special = vec![];
        for &u in room_pos {
            if u == init_pos {
                continue;
            }
            let cnt = neighbors_eswn(u, 9, 9)
                .filter(|&v| self.has_room[v])
                .count();
            if cnt == 1 {
                special.push(u);
            } else {
                normal.push(u);
            }
        }

        let require: Vec<i32> = (0..=9)
            .flat_map(|i| repeat(i).take(10 - i as usize))
            .collect();

        let mut add_room = |u, room| {
            room_idx[u] = Some(rooms.len());
            rooms.push(room);
        };
        add_room(init_pos, Room::new(RoomType::Init));

        for u in normal {
            add_room(u, Room::normal(rng.choice(&require) as i8));
        }

        let boss_candidates: Vec<_> = special
            .iter()
            .copied()
            .filter(|&u| neighbors_eswn(u, 9, 9).all(|v| v != init_pos))
            .collect();
        let boss_pos = rng.choice(&boss_candidates);
        add_room(boss_pos, Room::new(RoomType::Boss));
        special.retain(|&u| u != boss_pos);

        if !special.is_empty() {
            let secret_pos = rng.choice(&special);
            add_room(secret_pos, Room::new(RoomType::Secret));
            special.retain(|&u| u != secret_pos);
        }

        if !special.is_empty() {
            let treasure_pos = rng.choice(&special);
            add_room(treasure_pos, Room::new(RoomType::Treasure));
            special.retain(|&u| u != treasure_pos);

            if !special.is_empty() {
                if *n_rooms >= 15 && rng.chance(25) {
                    let treasure_pos = rng.choice(&special);
                    add_room(treasure_pos, Room::new(RoomType::Treasure));
                    special.retain(|&u| u != treasure_pos);
                }
            }
        }

        if !special.is_empty() {
            if *n_rooms <= 15 || rng.chance(66) {
                let store_pos = rng.choice(&special);
                add_room(store_pos, Room::new(RoomType::Store));
                special.retain(|&u| u != store_pos);
            }
        }

        let mut has_devil = false;
        let mut has_angel = false;
        if rng.chance(20) {
            let bbox = room_pos.iter().fold((8, 8, 0, 0), |bbox, pos| {
                let (y0, x0, y1, x1) = bbox;
                (y0.min(pos.0), x0.min(pos.1), y1.max(pos.0), x1.max(pos.1))
            });
            let (y0, x0, y1, x1) = bbox;
            let reward: Vec<(usize, usize)> = neighbors_e_ws_n(boss_pos, 9, 9)
                .filter(|&u| !self.has_room[u])
                .filter(|&u| x0 <= u.1 && u.1 <= x1 && y0 <= u.0 && u.0 <= y1)
                .collect();

            if !reward.is_empty() {
                let new_pos = rng.choice(&reward);
                if rng.chance(50) {
                    add_room(new_pos, Room::new(RoomType::Devil));
                    has_devil = true;
                } else {
                    add_room(new_pos, Room::new(RoomType::Angel));
                    has_angel = true;
                }
                self.has_room[new_pos] = true;
            }
        }

        if !special.is_empty() {
            if has_angel || rng.chance(14) {
                let sacrificial_pos = rng.choice(&special);
                add_room(sacrificial_pos, Room::new(RoomType::Sacrificial));
                special.retain(|&u| u != sacrificial_pos);
            }
        }

        if !special.is_empty() {
            if has_devil && rng.chance(50) {
                let cursed_pos = rng.choice(&special);
                add_room(cursed_pos, Room::new(RoomType::Cursed));
                special.retain(|&u| u != cursed_pos);
            }
        }

        for u in special {
            add_room(u, Room::normal(rng.choice(&require) as i8));
        }
    }

    fn build_topology(&mut self) {
        for u in product(0..9, 0..9) {
            if !self.has_room[u] {
                continue;
            }
            let u_idx = self.room_idx[u].unwrap();
            for v in neighbors_eswn(u, 9, 9) {
                if !self.has_room[v] {
                    continue;
                }
                let v_idx = self.room_idx[v].unwrap();
                if self.rooms[u_idx].ty.inaccessible() || self.rooms[v_idx].ty.inaccessible() {
                    continue;
                }
                self.rooms[u_idx].neighbors.push(v_idx);
            }
        }
    }

    fn render(&self) {
        let bbox = self.room_pos.iter().fold((8, 8, 0, 0), |bbox, pos| {
            let (y0, x0, y1, x1) = bbox;
            (y0.min(pos.0), x0.min(pos.1), y1.max(pos.0), x1.max(pos.1))
        });
        let (y0, x0, y1, x1) = bbox;

        let h = y1 - y0 + 1;
        let w = x1 - x0 + 1;
        let h_pad = 6 * h + 3;
        let w_pad = 6 * w + 3;

        let add = |u: (usize, usize), v: (usize, usize)| (u.0 + v.0, u.1 + v.1);
        let transform = |u: (usize, usize), v: (usize, usize)| {
            (6 * (u.0 - y0) + 2 + v.0, 6 * (u.1 - x0) + v.1 + 2)
        };

        let mut canvas = Grid::sized(b' ', h_pad, w_pad);

        // draw_boundary walls
        for u in product([0, h_pad - 1], 0..w_pad).chain(product(0..h_pad, [0, w_pad - 1])) {
            canvas[u] = b'#';
        }

        for u in product(0..9, 0..9) {
            if self.room_idx[u].is_none() {
                continue;
            }
            let u_idx = self.room_idx[u].unwrap();
            let room = &self.rooms[u_idx];

            canvas[transform(u, (2, 2))] = room.as_byte();

            let wall_symbols = match room.ty {
                RoomType::Angel | RoomType::Devil => br#"^\>/v\</"#,
                RoomType::Init | RoomType::Boss => b"@@@@@@@@",
                _ => b"-+|+-+|+",
            };

            let (i0, j0) = transform(u, (0, 0));
            let (i1, j1) = transform(u, (4, 4));

            for u in product(i0..=i0, j0 + 1..j1) {
                canvas[u] = wall_symbols[0];
            }
            for u in product(i0 + 1..i1, j1..=j1) {
                canvas[u] = wall_symbols[2];
            }
            for u in product(i1..=i1, j0 + 1..j1) {
                canvas[u] = wall_symbols[4];
            }
            for u in product(i0 + 1..i1, j0..=j0) {
                canvas[u] = wall_symbols[6];
            }

            canvas[(i0, j1)] = wall_symbols[1];
            canvas[(i1, j1)] = wall_symbols[3];
            canvas[(i1, j0)] = wall_symbols[5];
            canvas[(i0, j0)] = wall_symbols[7];
        }

        // draw bridges
        for u in product(0..9, 0..9) {
            let wall_symbol = |u: (usize, usize)| match self.rooms[self.room_idx[u].unwrap()].ty {
                RoomType::Init | RoomType::Boss => b'@',
                _ => b'+',
            };
            if self.room_idx[u].is_none() || self.rooms[self.room_idx[u].unwrap()].ty.disconnected()
            {
                continue;
            }

            let east = add(u, (0, 1));
            if east.1 < 9
                && self.room_idx[east].is_some()
                && !self.rooms[self.room_idx[east].unwrap()].ty.disconnected()
            {
                canvas[transform(u, (2, 4))] = b' ';
                canvas[transform(u, (2, 6))] = b' ';

                canvas[transform(u, (1, 4))] = wall_symbol(u);
                canvas[transform(u, (3, 4))] = wall_symbol(u);
                canvas[transform(u, (1, 5))] = b'-';
                canvas[transform(u, (3, 5))] = b'-';
                canvas[transform(u, (1, 6))] = wall_symbol(east);
                canvas[transform(u, (3, 6))] = wall_symbol(east);
            }

            let south = add(u, (1, 0));
            if south.0 < 9
                && self.room_idx[south].is_some()
                && !self.rooms[self.room_idx[south].unwrap()].ty.disconnected()
            {
                canvas[transform(u, (4, 2))] = b' ';
                canvas[transform(u, (6, 2))] = b' ';

                canvas[transform(u, (4, 1))] = wall_symbol(u);
                canvas[transform(u, (4, 3))] = wall_symbol(u);
                canvas[transform(u, (5, 1))] = b'|';
                canvas[transform(u, (5, 3))] = b'|';
                canvas[transform(u, (6, 1))] = wall_symbol(south);
                canvas[transform(u, (6, 3))] = wall_symbol(south);
            }
        }

        // emit
        for row in canvas.data.chunks(w_pad) {
            for cell in row {
                print!("{}", *cell as char);
            }
            println!();
        }

        // println!("{}", canvas.colored());
    }
}

fn main() {
    let mut input = simple_io::stdin_at_once();
    let seed = input.token().as_bytes();
    let seed: u64 = seed[..4]
        .into_iter()
        .chain(&seed[5..9])
        .map(|&x| match x {
            b'0'..=b'9' => x - b'0',
            b'A'..=b'Z' => x - b'A' + 10,
            _ => panic!(),
        } as u64)
        .fold(0, |acc, x| acc * 36 + x);

    let mut builder = DungeonGenerator::new(seed);
    let mut dungeon: DungeonTopology = builder.run();

    let result = dungeon.solve();
    if result {
        println!("CLEAR");
    } else {
        println!("GAME OVER");
    }

    builder.render();

    // dungeon.print_graph();

    // for seed in 1..10000000 {
    //     let mut builder = DungeonGenerator::new(seed);
    //     builder.run();
    //     if seed % 100 == 0 {
    //         println!("{}", seed);
    //         // builder.render_large();
    //     }
    // }
}
