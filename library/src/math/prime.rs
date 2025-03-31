fn linear_sieve(n_max: u32) -> (Vec<u32>, Vec<u32>) {
    let mut min_prime_factor = vec![0; n_max as usize + 1];
    let mut primes = Vec::new();

    for i in 2..=n_max {
        if min_prime_factor[i as usize] == 0 {
            primes.push(i);
        }
        for &p in primes.iter() {
            if i * p > n_max {
                break;
            }
            min_prime_factor[(i * p) as usize] = p;
            if i % p == 0 {
                break;
            }
        }
    }

    (min_prime_factor, primes)
}

fn gen_euler_phi(min_prime_factor: &[u32]) -> Vec<u32> {
    let n_bound = min_prime_factor.len();
    let mut phi = vec![0; n_bound];
    phi[1] = 1;
    for i in 2..n_bound {
        let p = min_prime_factor[i as usize];
        phi[i] = if p == 0 {
            i as u32 - 1
        } else {
            let m = i as u32 / p;
            phi[m as usize] * if m % p == 0 { p } else { p - 1 }
        };
    }
    phi
}

fn gen_mobius(min_prime_factor: &[u32]) -> Vec<i32> {
    let n_max = min_prime_factor.len() - 1;
    let mut mu = vec![0; n_max + 1];
    mu[1] = 1;

    for i in 2..=n_max {
        let p = min_prime_factor[i];
        if p == 0 {
            mu[i] = -1;
        } else {
            let m = i as u32 / p;
            mu[i] = if m % p == 0 { 0 } else { -mu[m as usize] };
        }
    }
    mu
}

fn factorize(n: u32, min_prime_factor: &[u32]) -> Vec<(u32, u8)> {
    let mut factors = Vec::new();
    let mut x = n;
    while x > 1 {
        let p = min_prime_factor[x as usize];
        if p == 0 {
            factors.push((x as u32, 1));
            break;
        }
        let mut exp = 0;
        while x % p == 0 {
            exp += 1;
            x /= p;
        }
        factors.push((p, exp));
    }

    factors
}

fn for_each_divisor(factors: &[(u32, u8)], mut visitor: impl FnMut(u32)) {
    let mut stack = vec![(1, 0u32)];
    while let Some((mut d, i)) = stack.pop() {
        if i as usize == factors.len() {
            visitor(d);
        } else {
            let (p, exp) = factors[i as usize];
            for _ in 0..=exp {
                stack.push((d, i + 1));
                d *= p;
            }
        }
    }
}
