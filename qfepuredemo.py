import random
from charm.toolbox.pairinggroup import PairingGroup, G1, G2, GT
from qfehelpers import (
    inner_product_mod,
    transpose_matrix,
    random_int_matrix,
    random_vector,
    generate_matrix_Lk,
    matrix_multiply_mod,
    vector_matrix_multiply_mod,
    vector_multiply_mod,
    matrix_vector_multiply,
    vector_transposed_mul_matrix_mul_vector,
    dot_product,
    scalar_multiply,
    MPK,
    MSK,
    SKF,
    CTXY,
)


######################################## PARAMETERS ###################################################################

# Initialize a set of parameters from a string
# check the PBC documentation (http://crypto.stanford.edu/pbc/manual/) for more information
#
## Type f pairing
# TODO: Q: How to find parameters for thes pairing ? Are these parameters a dependency of choosing k ? Or F, n, m ?
# TODO: BM: Benchmark maybe over different curves ?
# TODO: Find curve of Type III -> https://arxiv.org/pdf/1908.05366
# TODO: Find such a curve: https://www.cryptojedi.org/papers/pfcpo.pdf
# TODO: Different sizes of ciphertexts
# TODO: 5 different types of experiments


print("PARAMETERS")
# Initialize the pairing
group = PairingGroup("BN254")
print("Group order: ", group.order())

# TODO: Find p > mnB_xB_yB_f where M: {0,...,B_x}^n x {0,...B_y}^m and K:={0,..,B_f}^nxm to efficiently compute dlog P8,9 QFE Paper
# TODO: BM: Maybe do benchmarks over different sizes of p
# TODO: Outsource parameters in files
# p = group.order() # prime for group Z_p
p = group.order()
k = 9  # parameter for generation of D-k matrices
m = k
n = k - 1
print("p: ", p)
print("k: ", k)
print("m: ", m)
print("n: ", n)
print("\n")

######################################## ALGORITHM ####################################################################

# 1. Add to Mult
# 2. Search curves
# 3. Search other implementations

x = random_vector(1, 3, n)
y = random_vector(1, 2, m)
F = random_int_matrix(1, 2, n, m)  # F <- Z_p^(n x m)


def setup(p, k):
    # Generate random elements (each element of a group is also a generator)
    g1 = group.random(G1)
    g2 = group.random(G2)
    gt = group.pair_prod(g1, g2)  # compute generator gT = e(g1, g2)

    # Generate D-k matrices A, B <- Z_p^(k+1) x k
    # A,B <- Z_p^(k+1) x k
    # a,b <- Z_p^(k+1)
    A, a = generate_matrix_Lk(p, k)
    B, b = generate_matrix_Lk(p, k)

    # Generate random elements r, s <- Z_p^k
    # r_i, s_j <- Z_p^k
    r = random_int_matrix(1, p, n, k)
    s = random_int_matrix(1, p, m, k)

    # master public key (TODO: is this always gt^1 ?)
    mpk = MPK(g1, g2, gt, inner_product_mod(b, a, p))
    # master secret key
    msk = MSK(A, a, B, b, r, s)
    return mpk, msk


def keygen(p, mpk, msk):
    # Generate random element u <- Z_p
    u = random.randint(0, p - 1)  # u <- Z_p
    # Generate random matrix F <- Z_p^(n x m)

    A = msk.A
    B = msk.B
    a = msk.a
    b = msk.b
    r = msk.r
    s = msk.s
    g1 = mpk.g1
    g2 = mpk.g2

    sum = 0
    ATB = matrix_multiply_mod(transpose_matrix(A), B, p)
    for i in range(n):
        riT_AT_B = vector_matrix_multiply_mod(r[i], ATB, p)
        for j in range(m):
            riT_AT_B_sj = vector_multiply_mod(riT_AT_B, s[j], p)
            sum += (F[i][j] * riT_AT_B_sj) % p

    # Compute K and K_tilde
    K = g1 ** int(sum - u)
    K_tilde = g2 ** int(u)

    skF = SKF(K, K_tilde)  # secret key for F
    return skF


def encrypt(msk, x, y):
    A = msk.A
    B = msk.B
    a = msk.a
    b = msk.b
    r = msk.r
    s = msk.s

    # Compute c and c_tilde
    c = [
        matrix_vector_multiply(A, r[i]) + scalar_multiply(b, x[i])
        for i in range(len(x))
    ]
    c_tilde = [
        matrix_vector_multiply(B, s[j]) + scalar_multiply(a, y[j])
        for j in range(len(y))
    ]

    CT_xy = CTXY(c, c_tilde)
    return CT_xy


def decrypt(p, mpk, skF, CT_xy, n, m):
    c = CT_xy.c
    c_tilde = CT_xy.c_tilde
    K = skF.K
    K_tilde = skF.K_tilde
    g2 = mpk.g2
    g1 = mpk.g1
    gt = mpk.gt

    D = group.random(GT)
    exp = 0
    for i in range(n):
        for j in range(m):
            # TODO: Question: If we use CT_xy c and c_tilde are already in G1, G2 -> How to compute the dot product ?
            exp += int(F[i][j] * int(dot_product(c[i], c_tilde[j])))

    D = gt**exp
    D *= -(group.pair_prod(K, g2))
    D *= -(group.pair_prod(g1, K_tilde))

    # Find v such that [v * (b.T)*a]_T = D
    v = 0
    res = group.random(GT)
    inner = mpk.baT
    while D != res and v < p:
        v += 1
        res = gt ** int(v * inner)

    return v


def qfe(p, x, F, y, k):
    mpk, msk = setup(p, k)
    skF = keygen(p, mpk, msk)
    CT_xy = encrypt(msk, x, y)
    v = decrypt(p, mpk, skF, CT_xy, n, m)

    expected = vector_transposed_mul_matrix_mul_vector(x, F, y, p)
    print("expected result: ", expected)
    print("calculated result: ", v)


qfe(p, x, F, y, k)
