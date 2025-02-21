import random

def generate_test_case(N, M, T, T_SCALE):
    print(N, M, T * T_SCALE)

    # BS = int(T ** 0.5)
    BS = int(T ** 0.5)
        
    # B = [random.randint(1, M) for _ in range(T+1)]
    B = [random.randint(1, BS) for _ in range(T * T_SCALE +1)]
    print(*B)

    KS = [1000000] * BS + [0] * (M - BS)
    # KS = [random.randint(1, 1000000) for _ in range(M)]
    while sum(KS) > 200000:
        for i in range(M):
            if KS[i] > 1:
                KS[i] = KS[i] * 4 // 5

    
    for i in range(M):
        K = KS[i]
        A_i = [K]
        for _ in range(K):
            lake1 = random.randint(1, N)
            lake2 = random.randint(1, N)
            while lake1 == lake2:  # Ensure different lakes
                lake2 = random.randint(1, N)
            A_i.extend([lake1, lake2])
        
        print(*A_i)

# N, M, T = 10, 3, 10
# N, M, T = 100, 100, 100
N, M, T = 100000, 100000, 200000
# N, M, T = 100000, 100000, 1000000
generate_test_case(N, M, T, 10)
