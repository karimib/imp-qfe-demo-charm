import random
import math
import numpy as np
from charm.toolbox.pairinggroup import PairingGroup,ZR,G1,G2,GT


######################################## HELPERS ######################################################################

def extended_gcd(a, b):
    if a == 0:
        return b, 0, 1
    gcd, x1, y1 = extended_gcd(b % a, a)
    x = y1 - (b // a) * x1
    y = x1
    return gcd, x, y

def modular_inverse(a, m):
    gcd, x, _ = extended_gcd(a, m)
    if gcd != 1:
        raise ValueError("Inverse does not exist")
    else:
        return x % m




## Generates a Matrix A <- Z_p^(k+1)xk and a vector a <_ Z_p^k such that A^T*a = 0
def generate_matrix_Lk(p, k):
    matrix = np.zeros((k+1, k))
    vector = np.zeros(k+1)
    
    for i in range(k):
        val = np.random.randint(1, p)
        matrix[i][i] = val
        vector[i] = modular_inverse(val, p)
    
    matrix[k] = np.ones(k)
    vector[k] = -1
    
    return np.array(matrix), np.array(vector)

## Generates Matrices A, B <- Z_p^(k+1)xk and vertices a,b <- Z_p^(k+1) such that A^T*a=B^T*b = 0 and b^T*a=1
def generate_matrix_Lk_AB(p, k):
    A, a = generate_matrix_Lk(p, k)
    B = np.zeros((k+1, k))
    b = np.zeros(k+1)

    while((b.T @ a) % p != 1):
        B,b = generate_matrix_Lk(p,k)
        
    
    return np.array(A), np.array(a), np.array(B), np.array(b)

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

# TODO: Find p > mnB_xB_yB_f where M: {0,...,B_x}^n x {0,...B_y}^m and K:={0,..,B_f}^nxm to efficiently compute dlog P8,9 QFE Paper
# TODO: BM: Maybe do benchmarks over different sizes of p
# TODO: Outsource parameters in files
p = 797 # prime for group Z_p
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
    #g1 = Element.random(pairing, G1) # choose random element from G1 => Generator g1
    #g2 = Element.random(pairing, G2) # choose random element from G2 => Generator g2
    gt = group.pair_prod([g1],[g2]) # compute generator gT = e(g1, g2)

    print("g1 :", g1)
    print("g2 :", g2)
    print("gt:", gt)

    
    # TODO: Add mod artihmetic
    # TODO: Add random samling of matrices -> Use this paper, P11, https://eprint.iacr.org/2015/409.pdf
    # A,B <- Z_p^(k+1) x k
    # a,b <- Z_p^(k+1)
    A, a, B, b  = generate_matrix_Lk_AB(p, k)
    print(np.dot(A.T, a) % p)
    print(np.dot(B.T, b) % p)
    print(np.dot(b.T, a) % p)
    
    # r_i, s_j <- Z_p^k
    r = np.random.randint(0, p, size=(n, k)) 
    s = np.random.randint(0, p, size=(m, k)) 

    # master public key (TODO: is this always gt^1 ?)
    mpk = gt ** int(np.dot(b.T, a))
    # master secret key
    msk = (A, a, B, b, r, s) 
        
    print("A: ", A)
    print("a: ", a) 
    print("B: ", B)
    print("b: ", b)
    print("r: ", r)
    print("s: ", s)
    print("mpk: ", mpk)
    
    ## KEYGEN
    print("KEYGEN")
    # We assume multiplicative groups
    u = np.random.randint(1, p) 
    F = np.random.randint(1, 2, size=(n, m)) # F <- Z_p^(n x m) TODO: parameterize

    print("u: ", u)
    print("F: ", F)
    
    sum = 0
    for i in range(n):
        for j in range(m):
            sum += (F[i][j] * np.dot(np.dot(np.dot(r[i].T, A.T), B), s[j]))
    
    sum %= p
    sum = group.init(ZR, sum)
    u = group.init(ZR, u)
    print("sum: ", sum)
    K = group.pair_prod(g1 ** (sum - u), g2)
    K_tilde = group.pair_prod(g1,g2 ** u)
    
    skF = (K, K_tilde) # secret key for F
    
    print("K: ", K)
    print("K_tilde: ", K_tilde)
      
    ## ENCRYPT
    print("ENCRYPT")
    # Input vectors (x,y) <- Z_p^n x Z_p^m
    #x = np.random.randint(1, p, size=n) # x <- Z_p^n
    #y = np.random.randint(1, p, size=m) # y <- Z_p^m
    x = np.random.randint(1, 2, size=n)
    y = np.random.randint(1, 2, size=m)
    print("x :", x)
    print("y :", y)
    
    # Compute c and c_tilde
    c = [np.mod(np.dot(A, r[i]) + np.dot(b, x[i]),p)for i in range(n)]
    c_tilde = [np.mod(np.dot(B, s[j]) + np.dot(a, y[j]),p) for j in range(m)]
    
    #c = np.vectorize(lambda x: g1 ** int(x))(c)
    #c_tilde = np.vectorize(lambda x: g2 ** int(x))(c_tilde)
    #Ct_xy = (np.vectorize(lambda x: g1 ** int(x))(c), np.vectorize(lambda x: g2 ** int(x))(c_tilde))

    print("c: ", c)
    print("c_tilde: ", c_tilde)
    #print("Ct_xy: ", Ct_xy)

    ## DECRYPT
    print("DECRYPT")
    D = res = group.random(GT)
    for i in range(n):
        for j in range(m):
            # TODO: Q: Is this computation correct ? e(c_i, c_tilde_j) = gt ** <c_i,c_tilde_j>
            print("c[i] :", c[i])
            print("c[j] :", c_tilde[j])
            print("dot :", np.dot(c[i], c_tilde[j]))
            D+= group.init(GT,(int(F[i][j] + int(np.dot(c[i], c_tilde[j])))))

            
    D -= group.pair_prod(K, g2 ** int(1))
    D -= group.pair_prod(g1 ** int(1), K_tilde)
    print("D: ", D)

    # Find v such that [v * (b.T)*a]_T = D
    res = group.random(GT)
    v = 0
    while (D != res and v < p):
        res = gt ** int(v * (np.dot(b.T,a)))
        v += 1

    print("v:", v)
    print("res: ", res)
    print("expected result: ", np.dot(np.dot(x.T,F), y))
    print("calculated result: ", v)
    print("g")
    




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