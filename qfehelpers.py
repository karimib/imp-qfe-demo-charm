import random
import math


def modular_inverse(a, p):
    """
    Computes the modular inverse of a with respect to p using the extended Euclidean algorithm.
    Args:
        a (int): The number to invert.
        p (int): The modulus.
    Returns:
        int: The modular inverse of a modulo p.
    """
    t, new_t = 0, 1
    r, new_r = p, a

    while new_r != 0:
        quotient = r // new_r
        t, new_t = new_t, t - quotient * new_t
        r, new_r = new_r, r - quotient * new_r

    if r > 1:
        raise ValueError(f"{a} has no modular inverse modulo {p}")
    if t < 0:
        t += p

    return t


def generate_matrix_Lk(p, k):
    """
    Generates a matrix and a vector for the given p and k with arbitrary long integers.

    Args:
        p (int): The modulus for the modular arithmetic.
        k (int): The size of the matrix and vector.

    Returns:
        tuple: A tuple (matrix, vector), where:
            - matrix is a (k+1) x k matrix filled with random values and ones on the last row.
            - vector is a (k+1) x 1 vector with modular inverses and -1 as the last value.
    """
    # Initialize matrix and vector
    matrix = [[0 for _ in range(k)] for _ in range(k + 1)]
    vector = [0 for _ in range(k + 1)]

    for i in range(k):
        val = random.randint(1, p - 1)  # Random integer in the range [1, p-1]
        matrix[i][i] = val
        vector[i] = modular_inverse(val, p)

    # Fill the last row of the matrix with ones
    matrix[k] = [1 for _ in range(k)]
    vector[k] = -1

    return matrix, vector


def generate_matrix_Lk_AB(p, k):
    """
    Generates a matrix and a vector for the given p and k with arbitrary long integers.

    Args:
        p (int): The modulus for the modular arithmetic.
        k (int): The size of the matrix and vector.

    Returns:
        tuple: A tuple (matrix, vector), where:
            - matrix is a (k+1) x k matrix filled with random values and ones on the last row.
            - vector is a (k+1) x 1 vector with modular inverses and -1 as the last value.
    """
    # Initialize matrix and vector
    A, a = generate_matrix_Lk(p, k)
    B, b = generate_matrix_Lk(p, k)

    while (inner_product_mod(b, a, p) != 1):
        B, b = generate_matrix_Lk(p, k)

    return A, a, B, b


def matrix_vector_dot(matrix, vector, p):
    """
    Computes the dot product of a matrix and a vector, reducing results modulo p.
    """
    if len(matrix[0]) != len(vector):
        raise ValueError("Number of columns in the matrix must match the length of the vector")

    # Compute the dot product row-wise, reducing modulo p
    result = [sum((row[i] * vector[i]) for i in range(len(vector))) % p for row in matrix]
    return result


def vector_matrix_dot_mod(vector, matrix, p):
    """
    Computes the dot product of a vector and a matrix modulo p.

    Args:
        vector (list[int]): The input vector (1D list).
        matrix (list[list[int]]): The input matrix (2D list).
        p (int): The modulus.

    Returns:
        list[int]: Resultant vector after the dot product, reduced modulo p.

    Raises:
        ValueError: If the number of elements in the vector does not match the number of rows in the matrix.
    """
    # Ensure vector and matrix dimensions match
    if len(vector) != len(matrix[0]):
        raise ValueError("Number of elements in the vector must match the number of rows in the matrix.")

    # Compute the dot product modulo p
    result = [sum(vector[j] * row[j] for j in range(len(vector))) % p for row in matrix]
    return result


def inner_product_mod(vector1, vector2, p):
    """
    Computes the inner product (dot product) of two vectors modulo p.

    Args:
        vector1 (list[int or float]): The first vector.
        vector2 (list[int or float]): The second vector.
        p (int): The modulus.

    Returns:
        int: The inner product of the two vectors modulo p.

    Raises:
        ValueError: If the vectors are not of the same length.
    """
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must have the same length")

    # Compute the inner product modulo p
    return sum((x * y) % p for x, y in zip(vector1, vector2)) % p


def transpose_matrix(matrix):
    """
    Transposes a given matrix.

    Args:
        matrix (list[list[int or float]]): A 2D list representing the matrix.

    Returns:
        list[list[int or float]]: The transposed matrix.
    """
    # Ensure the matrix is not empty
    if not matrix or not matrix[0]:
        raise ValueError("Matrix cannot be empty")

    # Transpose the matrix
    transposed = [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]
    return transposed


def transpose_vector(vector):
    """
    Transposes a vector (1D list to 2D column vector).

    Args:
        vector (list): A 1D list representing the vector.

    Returns:
        list[list]: A 2D list representing the transposed vector (column vector).
    """
    return [[element] for element in vector]


def random_int_matrix(low, high, n, k):
    """
    Generates a matrix of random integers in the range [low, high) with dimensions (n, k).

    Args:
        low (int): The lower bound (inclusive).
        high (int): The upper bound (exclusive).
        n (int): Number of rows in the matrix.
        k (int): Number of columns in the matrix.

    Returns:
        list[list[int]]: A 2D list (matrix) of random integers.
    """
    return [[random.randint(low, high - 1) for _ in range(k)] for _ in range(n)]


def random_vector(low, high, n):
    """
    Generates a random vector with elements from range [a, b].

    Args:
        a (int): The lower bound (inclusive).
        b (int): The upper bound (inclusive).
        n (int): The size of the vector.

    Returns:
        list[int]: A vector (list) of random integers.
    """
    return [random.randint(low, high-1) for _ in range(n)]


def transpose(matrix):
    """
    Transposes a given matrix.

    Args:
        matrix (list[list[float]]): The matrix to transpose.

    Returns:
        list[list[float]]: The transposed matrix.
    """
    return [[matrix[j][i] for j in range(len(matrix))] for i in range(len(matrix[0]))]

def dot_product(vector1, vector2):
    """
    Computes the dot product of two vectors.

    Args:
        vector1 (list[float]): The first vector.
        vector2 (list[float]): The second vector.

    Returns:
        float: The dot product of the two vectors.
    """
    if len(vector1) != len(vector2):
        raise ValueError("Vectors must have the same length.")
    return sum(x * y for x, y in zip(vector1, vector2))

def compute_rT_AT_for_row(r_i, A):
    """
    Computes r_i^T * A^T for a single row r_i and the given matrix A.

    Args:
        r_i (list[float]): A single row vector.
        A (list[list[float]]): A 2D list representing the matrix.

    Returns:
        list[float]: Resulting vector after computing r_i^T * A^T.
    """
    # Transpose A
    A_T = transpose(A)

    # Compute dot product of r_i with each row of A^T
    result = [dot_product(r_i, row) for row in A_T]
    return result