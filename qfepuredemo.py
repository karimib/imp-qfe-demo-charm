import random
from charm.toolbox.pairinggroup import PairingGroup, G1, G2, GT
from qfehelpers import (
    matrix_vector_dot,
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
    dot_product
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


print("PARAMETERS")
# Initialize the pairing
group = PairingGroup("BN254")
print("Group order: ", group.order())

# TODO: Find p > mnB_xB_yB_f where M: {0,...,B_x}^n x {0,...B_y}^m and K:={0,..,B_f}^nxm to efficiently compute dlog P8,9 QFE Paper
# TODO: BM: Maybe do benchmarks over different sizes of p
# TODO: Outsource parameters in files
# p = group.order() # prime for group Z_p
p = group.order()
k = 3  # parameter for generation of D-k matrices
m = 3
n = 2
print("p: ", p)
print("k: ", k)
print("m: ", m)
print("n: ", n)
print("\n")

######################################## ALGORITHM ####################################################################

# 1. Add to Mult
# 2. Search curves
# 3. Search other implementations


def qfe(p, k):

    ## SETUP
    print("SETUP")

    # Generate random elements (each element of a group is also a generator)
    g1 = group.random(G1)
    g2 = group.random(G2)
    gt = group.pair_prod(g1, g2)  # compute generator gT = e(g1, g2)

    print("g1:", g1)
    print("g2:", g2)
    print("gt:", gt)

    # Generate D-k matrices A, B <- Z_p^(k+1) x k
    # A,B <- Z_p^(k+1) x k
    # a,b <- Z_p^(k+1)
    A, a = generate_matrix_Lk(p, k)
    B, b = generate_matrix_Lk(p, k)

    print("A.T*a", matrix_vector_dot(transpose_matrix(A), a, p))
    print("B.T*b", matrix_vector_dot(transpose_matrix(B), b, p))
    print("b.T*a", inner_product_mod(b, a, p))

    # Generate random elements r, s <- Z_p^k
    # r_i, s_j <- Z_p^k
    r = random_int_matrix(1, p, n, k)
    s = random_int_matrix(1, p, m, k)
    print("r: ", r)
    print("s: ", s)

    # master public key (TODO: is this always gt^1 ?)
    mpk = gt ** int(inner_product_mod(b, a, p))
    # master secret key
    msk = (A, a, B, b, r, s)

    print("A: ", A)
    print("a: ", a)
    print("B: ", B)
    print("b: ", b)
    print("r: ", r)
    print("s: ", s)
    print("mpk: ", mpk)
    print("\n")
    
    ## KEYGEN
    print("KEYGEN")
    # We assume multiplicative groups
    # Generate random element u <- Z_p
    u = random.randint(0, p - 1)  # u <- Z_p
    # Generate random matrix F <- Z_p^(n x m)
    F = random_int_matrix(1, 2, n, m)  # F <- Z_p^(n x m)

    print("u: ", u)
    print("F: ", F)

    sum = 0
    ATB = matrix_multiply_mod(transpose_matrix(A), B, p)
    for i in range(n):
        riT_AT_B = vector_matrix_multiply_mod(r[i], ATB, p)
        for j in range(m):
            riT_AT_B_sj = vector_multiply_mod(riT_AT_B, s[j], p)
            sum += ((F[i][j] * riT_AT_B_sj) % p)

    print("sum: ", sum)
    print("u: ", u)

    # Compute K and K_tilde
    K = g1 ** int(sum - u)
    K_tilde = g2 ** int(u)

    skF = (K, K_tilde)  # secret key for F

    print("K: ", K)
    print("K_tilde: ", K_tilde)
    print("\n")
    
    ## ENCRYPT
    print("ENCRYPT")
    # Input vectors (x,y) <- Z_p^n x Z_p^m
    # TODO: BM: Benchmark over different sizes of x and y
    x = random_vector(1, 2, n)
    y = random_vector(1, 2, m)
    
    print("x :", x)
    print("y :", y)

    # Compute c and c_tilde
    c = [matrix_vector_multiply(A, r[i]) + (b * x[i]) for i in range(n)]
    c_tilde = [matrix_vector_multiply(B, s[j]) + (a * y[j]) for j in range(m)]

    print("c: ", c)
    print("c_tilde: ", c_tilde)
    print("\n")
    
    ## DECRYPT
    print("DECRYPT")
    D = group.random(GT)
    exp = 0
    for i in range(n):
        for j in range(m):
            exp += int(F[i][j] * int(dot_product(c[i], c_tilde[j])))
    

    D = gt ** exp
    D *= -(group.pair_prod(K, g2))
    D *= -(group.pair_prod(g1, K_tilde))
    
    print("D: ", D)

    # Find v such that [v * (b.T)*a]_T = D
    v = 0
    res = group.random(GT)
    inner = dot_product(b, a)
    while D != res and v < p:
        v += 1
        res = gt ** int(v * inner)


 
    expected = vector_transposed_mul_matrix_mul_vector(x, F, y, p)
    print("expected result: ", expected)
    print("calculated result: ", v)


qfe(p, k)