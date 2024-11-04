// Z85 Encoding/Decoding

// # python3
// z85table = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-:+=^!/*?&<>()[]{}@%$#"
// z85table_inv = {c: i for i, c in enumerate(z85table)}

// pow85 = [85**i for i in range(5)]

// def z85encode(s: List[int]) -> str:
//     assert len(s) % 4 == 0

//     res = []
//     for i in range(0, len(s), 4):
//         acc = 0
//         for c in s[i : i + 4]:
//             assert 0 <= c < 256
//             acc = acc * 256 + c
//         for _ in range(5):
//             acc, r = divmod(acc, 85)
//             res.append(z85table[r])

//     return "".join(res)

// def z85decode(s: str) -> List[int]:
//     assert len(s) % 5 == 0

//     res = []
//     for i in range(0, len(s), 5):
//         acc = 0
//         for p in pow85:
//             acc = acc + p * z85table_inv[s[i]]
//             i += 1
//         for i in reversed(range(4)):
//             res.append((acc >> (8 * i)) & 255)
//     return res

fn z85decode<'a>(s: &'a [u8]) -> impl Iterator<Item = u8> + 'a {
    assert_eq!(s.len() % 5, 0);
    const POW85: [u32; 5] = [1, 85u32.pow(1), 85u32.pow(2), 85u32.pow(3), 85u32.pow(4)];
    const TABLE: &[u8] =
        b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-:+=^!/*?&<>()[]{}@%$#";
    let table_inv: [u8; 128] = {
        let mut table = [0; 128];
        for (i, &c) in TABLE.iter().enumerate() {
            table[c as usize] = i as u8;
        }
        table
    };

    s.chunks(5).flat_map(move |chunk| {
        let mut n = 0;
        for (&b, &p) in chunk.iter().zip(POW85.iter()) {
            n += table_inv[b as usize] as u32 * p;
        }
        (0..4).rev().map(move |i| ((n >> (i * 8)) & 255) as u8)
    })
}
