pub mod z85 {
    const POW85: [u32; 5] = [1, 85u32.pow(1), 85u32.pow(2), 85u32.pow(3), 85u32.pow(4)];
    const TABLE: &[u8] =
        b"0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ.-:+=^!/*?&<>()[]{}@%$#";

    pub fn encode<'a>(s: &'a [u8]) -> impl Iterator<Item = u8> + 'a {
        assert_eq!(s.len() % 4, 0);
        s.chunks(4).flat_map(move |chunk| {
            let mut n = 0;
            for &b in chunk {
                n = n * 256 + b as u32;
            }
            (0..5).map(move |i| TABLE[((n / 85u32.pow(i)) % 85) as usize])
        })
    }

    pub fn decode<'a>(s: &'a [u8]) -> impl Iterator<Item = u8> + 'a {
        assert_eq!(s.len() % 5, 0);

        // Something like lazy_static! would be better, if external crates are allowed.
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
}
