
from charm.toolbox.pairinggroup import PairingGroup, GT, G1, G2
import time
import csv
from qfehelpers import (
    random_vector,
    random_int_matrix,
)
from qfebounded import QFE

k = 9  # parameter for generation of D-k matrices
m = k # mxn matric
n = k - 1




group = PairingGroup("BN254")
p_order = group.order()
g1 = group.random(G1)
g1.initPP()
g2 = group.random(G2)
g2.initPP()
gt = group.pair_prod(g1, g2)
G = QFE(group, p_order, g1, g2, gt)



# Inputs, Ciphertexts size, How expensive are the single steps
# Presentation
def simulation():
    results = []
    x_max = 3
    y_max = 2
    F_max = 2
    for k in range(3, 12):
        m = k
        n = k - 1
        x = random_vector(1, x_max, n)
        y = random_vector(1, y_max, m)
        F = random_int_matrix(1, F_max, n, m)
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
        total_time = setup_time + keygen_time + encrypt_time + decrypt_time

        expected = G.get_expected_result(p, x, F, y)
        print("expected result: ", expected)
        print("calculated result: ", v)

        results.append([k, m, n, x_max, y_max, F_max, setup_time, keygen_time, encrypt_time, decrypt_time, total_time])

    with open('data/qfe_timings.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['k', 'm', 'n', 'x_max', 'y_max', 'F_max' ,'time setup', 'time keygen', 'time encrypt', 'time decrypt', 'time total'])
        csvwriter.writerows(results)


simulation()