from charm.toolbox.pairinggroup import PairingGroup, GT, G1, G2
import time
import csv
from qfehelpers import (
    random_vector,
    random_int_matrix,
    size_in_kilobits,
)
from qfebounded import QFE



# We use the BN254 pairing-friendly elliptic curve
# To benchmark we initialize the group, the generators and the scheme
group = PairingGroup("BN254")
p_order = group.order()
g1 = group.random(G1)
g1.initPP()
g2 = group.random(G2)
g2.initPP()
gt = group.pair_prod(g1, g2)
gt.initPP()
G = QFE(group, p_order, g1, g2, gt)


def implementation_check():
    """
    Check if the implementation is correct by running a simple test.
    """
    p = p_order
    k = 3
    m = k
    n = k - 1
    x = [1, 2]
    y = [1, 2, 3]
    F = [[1, 2, 3], [1, 2, 3]]

    mpk, msk = G.setup(p, k)
    skF = G.keygen(p, mpk, msk, F)
    CT_xy = G.encrypt(msk, x, y)
    v = G.decrypt(p, mpk, skF, CT_xy, n, m, F)

    expected = G.get_expected_result(p, x, F, y)
    print("expected result: ", expected)
    print("calculated result: ", v)


# Simulation with fixed vectors of length m and n where the values of the vectors are between 1 and 3 and k is varied
def simulation_fixed_vectors():
    results = []
    x_max = 3
    y_max = 2
    F_max = 2
    for k in range(3, 100):
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

        s_msk = size_in_kilobits(msk)
        s_mpk = size_in_kilobits(mpk)
        s_ct = size_in_kilobits(CT_xy)
        s_sk = size_in_kilobits(skF)

        expected = G.get_expected_result(p, x, F, y)
        print("expected result: ", expected)
        print("calculated result: ", v)

        results.append(
            [
                k,
                m,
                n,
                x_max,
                y_max,
                F_max,
                s_msk,
                s_mpk,
                s_ct,
                s_sk,
                setup_time,
                keygen_time,
                encrypt_time,
                decrypt_time,
                total_time,
            ]
        )

    with open("data/qfe_benchmark_fixed_vectors_sizes.csv", "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(
            [
                "k",
                "m",
                "n",
                "x_max",
                "y_max",
                "F_max",
                "size msk",
                "size mpk",
                "size ct",
                "size sk",
                "time setup",
                "time keygen",
                "time encrypt",
                "time decrypt",
                "time total",
            ]
        )
        csvwriter.writerows(results)


# Simulation with random vectors of length m and n where the values of the vectors are between 1 and p and k is fixed
def simulation_fixed_k():
    results = []
    x_max = p_order
    y_max = p_order
    F_max = p_order
    for k in range(3, 4):
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

        s_msk = size_in_kilobits(msk)
        s_mpk = size_in_kilobits(mpk)
        s_ct = size_in_kilobits(CT_xy)
        s_sk = size_in_kilobits(skF)

        expected = G.get_expected_result(p, x, F, y)
        print("expected result: ", expected)
        print("calculated result: ", v)

        results.append(
            [
                k,
                m,
                n,
                x_max,
                y_max,
                F_max,
                s_msk,
                s_mpk,
                s_ct,
                s_sk,
                setup_time,
                keygen_time,
                encrypt_time,
                decrypt_time,
                total_time,
            ]
        )

    with open("data/qfe_benchmark_fixed_k.csv", "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(
            [
                "k",
                "m",
                "n",
                "x_max",
                "y_max",
                "F_max",
                "size msk",
                "size mpk",
                "size ct",
                "size sk",
                "time setup",
                "time keygen",
                "time encrypt",
                "time decrypt",
                "time total",
            ]
        )
        csvwriter.writerows(results)


#simulation_fixed_vectors()
#simulation_fixed_k()
implementation_check()
