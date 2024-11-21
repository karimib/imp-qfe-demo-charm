import random
import math
import numpy as np
from charm.toolbox.pairinggroup import PairingGroup,ZR,G1,G2,GT
from charm.core.math.pairing import pair
from qfehelpers import matrix_vector_dot, inner_product_mod, transpose_matrix, random_int_matrix, random_vector, generate_matrix_Lk, vector_matrix_dot_mod, transpose_vector, compute_rT_AT_for_row



######################################## HELPERS ######################################################################

# def extended_gcd(a, b):
#     if a == 0:
#         return b, 0, 1
#     gcd, x1, y1 = extended_gcd(b % a, a)
#     x = y1 - (b // a) * x1
#     y = x1
#     return gcd, x, y

# def modular_inverse(a, m):
#     gcd, x, _ = extended_gcd(a, m)
#     if gcd != 1:
#         raise ValueError("Inverse does not exist")
#     else:
#         return x % m



######################################## PARAMETERS ###################################################################

# Initialize a set of parameters from a string 
# check the PBC documentation (http://crypto.stanford.edu/pbc/manual/) for more information
# 
## Type f pairing 
# TODO: Q: How to find parameters for thes pairing ? Are these parameters a dependency of choosing k ? Or F, n, m ?
# TODO; 
# TODO: BM: Benchmark maybe over different curves ?
# TODO: Find curve of Type III -> https://arxiv.org/pdf/1908.05366
# TODO: Find such a curve: https://www.cryptojedi.org/papers/pfcpo.pdf


 # Initialize the pairing
#pairing = Pairing(params)
#print(pairing.is_symmetric())   
group = PairingGroup('BN254')
print("Group order: ", group.order())

# TODO: Find p > mnB_xB_yB_f where M: {0,...,B_x}^n x {0,...B_y}^m and K:={0,..,B_f}^nxm to efficiently compute dlog P8,9 QFE Paper
# TODO: BM: Maybe do benchmarks over different sizes of p
# TODO: Outsource parameters in files
#p = group.order() # prime for group Z_p
p = group.order()
k = 3 # parameter for generation of D-k matrices
m = 3
n = 2


######################################## ALGORITHM ####################################################################

# 1. Add to Mult
# 2. Search curves
# 3. Search other implementations

def qfe(p, k):
    
    ## SETUP
    # F is Functionality were we get n and m
    print("SETUP")
    np.random.seed(42)
    
    # Generate random elements (each element of a group is also a generator)
    
    g1 = group.random(G1)
    g2 = group.random(G2)
    gt = group.pair_prod(g1,g2) # compute generator gT = e(g1, g2)

    print("g1:", g1)
    print("g2:", g2)
    print("gt:", gt)

    
    # TODO: Add mod artihmetic
    # TODO: Add random samling of matrices -> Use this paper, P11, https://eprint.iacr.org/2015/409.pdf
    # A,B <- Z_p^(k+1) x k
    # a,b <- Z_p^(k+1)
    A, a = generate_matrix_Lk(p, k)
    B, b  = generate_matrix_Lk(p, k)
    print("A.T*a", matrix_vector_dot(transpose_matrix(A), a, p))
    print("B.T*b", matrix_vector_dot(transpose_matrix(B), b, p))
    print("b.T*a", inner_product_mod(b, a, p))
    
    # r_i, s_j <- Z_p^k
    # TODO: Add random sampling of r and s
    #r = np.random.randint(0, 2, size=(n, k)) 
    #s = np.random.randint(0, 2, size=(m, k)) 
    r = random_int_matrix(1, 2, n, k)
    s = random_int_matrix(1, 2, m, k)


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
    u = random.randint(1, 1)
    F = random_int_matrix(1, 2, n, m) # F <- Z_p^(n x m) TODO: parameterize

    print("u: ", u)
    print("F: ", F)
    
    u = 1 # TODO: Remove this line
    sum = 0
    for i in range(n):
        print("r[i]: ", r[i])
        A = np.array(A)
        r = np.array(r)
        riT_AT = np.dot(r[i].T, A.T)
        for j in range(m):
            riT_AT_B = np.dot(riT_AT, B)
            riT_AT_B_sj = np.dot(riT_AT_B, s[j])
            sum += (F[i][j] * riT_AT_B_sj)
    

    print("sum: ", sum)
    print("u: ", u)

    K = g1 ** int(sum - u)
    K_tilde = g2 ** int(u)
            
    skF = (K, K_tilde) # secret key for F
    
    print("K: ", K)
    print("K_tilde: ", K_tilde)
    print("\n")
    ## ENCRYPT
    print("ENCRYPT")
    # Input vectors (x,y) <- Z_p^n x Z_p^m
    # TODO: Parameterize
    #x = np.random.randint(1, p, size=n) # x <- Z_p^n
    #y = np.random.randint(1, p, size=m) # y <- Z_p^m
    x = random_vector(1, 2, n)
    y = random_vector(1, 2, m)
    print("x :", x)
    print("y :", y)
    
    # Compute c and c_tilde
    c = [np.dot(A, r[i]) + np.dot(b, x[i]) for i in range(n)]
    c_tilde = [np.dot(B, s[j]) + np.dot(a, y[j]) for j in range(m)]
    
    #c = np.vectorize(lambda x: g1 ** int(x))(c)
    #c_tilde = np.vectorize(lambda x: g2 ** int(x))(c_tilde)
    #Ct_xy = (np.vectorize(lambda x: g1 ** int(x))(c), np.vectorize(lambda x: g2 ** int(x))(c_tilde))

    print("c: ", c)
    print("c_tilde: ", c_tilde)
    #print("Ct_xy: ", Ct_xy)
    print("\n")
    ## DECRYPT
    print("DECRYPT")
    D = group.random(GT)
    for i in range(n):
        for j in range(m):
            # TODO: Q: Is this computation correct ? e(c_i, c_tilde_j) = gt ** <c_i,c_tilde_j>
            print("c[i] :", c[i])
            print("c[j] :", c_tilde[j])
            print("dot :", np.dot(c[i], c_tilde[j]))
            D *= gt ** (int(F[i][j] * int(np.dot(c[i], c_tilde[j])))) # D *= e(c_i, c_tilde_j)^F_ij

    # []_T - e(K, [1]_2) - e([1]_2, K_tilde)
    D *= -(group.pair_prod(K, g2))
    D *= -(group.pair_prod(g1, K_tilde))
    # Find v such that [v * (b.T)*a]_T = D
    res = group.random(GT)
    v = 0
    b = np.array(b)
    a = np.array(a)
    while (D != res and v < 7):
        res = gt ** int(v * (np.dot(b.T,a)))
        v += 1

    print("D: ", D)
    print("res: ", res)
    x = np.array(x)
    y = np.array(y)
    F = np.array(F)
    print("expected result: ", np.dot(np.dot(x.T,F), y))
    print("calculated result: ", v)
    




qfe(p, k)



#######################################################################################################################
# Example of a D-2 Matrices in Z_7 (L_k)
#A = np.array([[4,0],[0,5],[1,1]])
#a = np.array([2,3,-1])
#B = np.array([[2,0],[0,4],[1,1]])
#b = np.array([4,2,-1])

#print(np.matmul(A.T, a) % 7)
#print(np.matmul(B.T, b) % 7)
#print(np.matmul(b.T, a) % 7)
