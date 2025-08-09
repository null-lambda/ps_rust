
# Notes on hashes

## Birthday paradox
Suppose we have n unique items that we want to test for equality using a hash.
Assuming each value of hash function is uniformly distributed over a domain 1..=h,
the probability that a hash collision occurs is 
p = 1 - exp(binom(n,2)/h) approx n^2/2h.
Thus, we need h >= n^2/2p to ensure the ratio of false positives is less than p.

For example: 
  - If we have n ~ 2e5 items to test, with a confidence level p ~ 1e-8, 
    we need a 64-bit hash as h ~ 2e18.

