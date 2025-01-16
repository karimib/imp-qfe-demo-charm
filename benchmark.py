from charm.toolbox.pairinggroup import PairingGroup, GT, G1, G2
import time
import sys
import csv
from qfehelpers import (
    random_vector,
    random_int_matrix,
)
from qfebounded import QFE

def size_in_kilobits(parameter):
    """
    Calculate the size of a parameter in kilobits.
    
    Args:
    parameter: The parameter whose size is to be calculated
    
    Returns:
    float: Size of the parameter in kilobits
    """
    size_in_bytes = sys.getsizeof(parameter)
    size_in_bits = size_in_bytes * 8
    size_in_kilobits = size_in_bits / 1024
    return size_in_kilobits


group = PairingGroup("BN254")
p_order = group.order()
g1 = group.random(G1)
g1.initPP()
g2 = group.random(G2)
g2.initPP()
gt = group.pair_prod(g1, g2)
gt.initPP()
G = QFE(group, p_order, g1, g2, gt)


def simulation(input):
    x_max = 3
    y_max = 2
    F_max = 2
    k = input
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

    print("Setup time: ", setup_time)
    print("Keygen time: ", keygen_time)
    print("Encrypt time: ", encrypt_time)
    print("Decrypt time: ", decrypt_time)
    print("Total time: ", total_time)


# Inputs, Ciphertexts size, How expensive are the single steps
# Presentation
def simulation_fixed_vectors():
    results = []
    x_max = 3
    y_max = 2
    F_max = 2
    for k in range(3, 65):
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


def simulation_p_vectors():
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
                setup_time,
                keygen_time,
                encrypt_time,
                decrypt_time,
                total_time,
            ]
        )

    with open("data/qfe_benchmark_p_vectors.csv", "w", newline="") as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(
            [
                "k",
                "m",
                "n",
                "x_max",
                "y_max",
                "F_max",
                "time setup",
                "time keygen",
                "time encrypt",
                "time decrypt",
                "time total",
            ]
        )
        csvwriter.writerows(results)


simulation_fixed_vectors()
# simulation_p_vectors()
#simulation(9)
