
from charm.toolbox.pairinggroup import PairingGroup, GT, G1, G2
import time
import csv
from qfehelpers import (
    random_vector,
    random_int_matrix,
)
from qfepuredemo import QFE

k = 9  # parameter for generation of D-k matrices
m = k
n = k - 1
x = random_vector(1, 3, n)
y = random_vector(1, 2, m)
F = random_int_matrix(1, 2, n, m)
#p = get_p_order()

# def qfe(p, x, F, y, k):
#     mpk, msk = setup(p, k)
#     skF = keygen(p, mpk, msk, F)
#     CT_xy = encrypt(msk, x, y)
#     v = decrypt(p, mpk, skF, CT_xy, n, m, F)

#     expected = get_expected_result(p, x, F, y)
#     print("expected result: ", expected)
#     print("calculated result: ", v)


group = PairingGroup("BN254")
p_order = group.order()
g1 = group.random(G1)
g1.initPP()
g2 = group.random(G2)
g2.initPP()
gt = group.pair_prod(g1, g2)
G = QFE(group, p_order, g1, g2, gt)

results = []

for k in range(3, 11):
    m = k
    n = k - 1
    x = random_vector(1, 3, n)
    y = random_vector(1, 2, m)
    F = random_int_matrix(1, 2, n, m)
    p = p_order

    start_time = time.time()
    mpk, msk = G.setup(p, k)
    setup_time = time.time() - start_time

    start_time = time.time()
    skF = G.keygen(p, mpk, msk, F)
    keygen_time = time.time() - start_time


    start_time = time.time()
    CT_xy = G.encrypt(msk, x, y)
    encrypt_time = time.time() - start_time

    start_time = time.time()
    v = G.decrypt(p, mpk, skF, CT_xy, n, m, F)
    decrypt_time = time.time() - start_time

    setup_time *= 1_000_000_000
    keygen_time *= 1_000_000_000
    encrypt_time *= 1_000_000_000
    decrypt_time *= 1_000_000_000

    expected = G.get_expected_result(p, x, F, y)
    print("expected result: ", expected)
    print("calculated result: ", v)

    results.append([k, n, m, setup_time, keygen_time, encrypt_time, decrypt_time])

with open('data/qfe_timings.csv', 'w', newline='') as csvfile:
    csvwriter = csv.writer(csvfile)
    csvwriter.writerow(['k', 'n', 'm', 'time setup', 'time keygen', 'time encrypt', 'time decrypt'])
    csvwriter.writerows(results)

