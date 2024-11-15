import random
from mock_group import MockGroup  # Stellen Sie sicher, dass MockGroup in mock_group.py gespeichert ist
import logging
import traceback

# Konfigurieren des Loggers
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')

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
        raise ValueError("Inverse existiert nicht")
    else:
        return x % m

## Generates a Matrix A <- Z_p^(k+1)xk and a vector a <_ Z_p^k such that A^T*a = 0
def generate_matrix_Lk(p, k, group):
    matrix = [[0 for _ in range(k)] for _ in range(k+1)]
    vector = [0 for _ in range(k+1)]
    
    for i in range(k):
        val = random.randint(1, p-1)
        matrix[i][i] = val
        try:
            inv_val = modular_inverse(val, p)
            vector[i] = inv_val
        except ValueError:
            logging.error(f"Inverse existiert nicht fuer val={val} und p={p}")
            raise
    
    for i in range(k):
        matrix[k][i] = 1
    vector[k] = (-1) % p  # Sicherstellen, dass -1 modulo p korrekt ist

    assert len(matrix) == k + 1, "Matrix hat nicht die erwartete Anzahl von Zeilen."
    assert len(matrix[0]) == k, "Matrix hat nicht die erwartete Anzahl von Spalten."
    assert len(vector) == k + 1, "Vektor hat nicht die erwartete Laenge."
    assert vector[i] > 0, f"Inverse fuer Matrix[{i}][{i}] konnte nicht berechnet werden."
    for col in range(k):
        assert sum(matrix[row][col] * vector[row] for row in range(k + 1)) % p == 0, \
            f"A^T * a ist nicht 0 fuer Spalte {col}."


    logging.debug(f"Generierte Matrix A: {matrix}")
    logging.debug(f"Generierter Vektor a: {vector}")
    
    return matrix, vector


## Generates Matrices A, B <- Z_p^(k+1)xk and vectors a, b <- Z_p^(k+1) such that A^T*a=B^T*b = 0 and b^T*a=1
def generate_matrix_Lk_AB(p, k, group):
    A, a = generate_matrix_Lk(p, k, group)
    B, b = [], []
    
    attempts = 0
    while True:
        A_new, a_new = generate_matrix_Lk(p, k, group)
        B_new, b_new = generate_matrix_Lk(p, k, group)
        dot_product = sum([b_new[i] * a_new[i] for i in range(k+1)]) % p
        if dot_product == 1:
            A, a = A_new, a_new
            B, b = B_new, b_new
            break
        attempts += 1
        if attempts > 1000000:
            raise ValueError("Zu viele Versuche, Matrizen A und B zu generieren, sodass b^T * a = 1 mod p")
    
    assert len(A) == k + 1 and len(A[0]) == k, "Matrix A hat falsche Dimensionen."
    assert len(a) == k + 1, "Vektor a hat falsche Laenge."
    assert len(B) == k + 1 and len(B[0]) == k, "Matrix B hat falsche Dimensionen."
    assert len(b) == k + 1, "Vektor b hat falsche Laenge."
    assert dot_product == 1, "Dot-Produkt von a und b ist nicht 1."

    logging.debug(f"Generierte Matrix A: {A}")
    logging.debug(f"Generierter Vektor a: {a}")
    logging.debug(f"Generierte Matrix B: {B}")
    logging.debug(f"Generierter Vektor b: {b}")
    
    return A, a, B, b

######################################## PARAMETERS ###################################################################


# Initialisieren der Mock-Gruppe
group = MockGroup(p=13) 

# Set p to die tatsaechliche Gruppenordnung
p = group.p
logging.info(f"Gruppenordnung (p): {101}")


# Parameters
k = 12    # Parameter zur Generierung von D-k Matrizen
m = 3
n = 2

def qfe(p, k, group):
    try:
        ## SETUP
        logging.info("SETUP")
        random.seed(42)
        
        # Generate random elements (each element of a group is also a generator)
        logging.info("Generiere zufaellige Gruppenelemente g1 und g2")
        g1 = group.random(group.G1)
        g2 = group.random(group.G2)
        assert group.is_valid(g1), "g1 ist kein gueltiges Gruppenelement."
        assert group.is_valid(g2), "g2 ist kein gueltiges Gruppenelement."
        logging.info(f"g1: {g1}")
        logging.info(f"g2: {g2}")
        logging.info(f"Typ von g1: {type(g1)}")
        logging.info(f"Typ von g2: {type(g2)}")
        
        # Korrekte Paarung
        pairing_result = group.pair_prod([g1], [g2])  # Paarungsergebnis als separates Element
        assert group.is_valid(pairing_result), "Pairing-Result ist kein gueltiges Gruppenelement."
        logging.info(f"pairing_result (e(g1, g2)): {pairing_result}")
        logging.info(f"Typ von pairing_result: {type(pairing_result)}")
        
        ## Generieren von A, B, a, b
        A, a, B, b = generate_matrix_Lk_AB(p, k, group)
        # Berechnung von A^T * a mod p
        AT_a = [sum([A[row][col] * a[row] for row in range(k+1)]) % p for col in range(k)]
        BT_b = [sum([B[row][col] * b[row] for row in range(k+1)]) % p for col in range(k)]
        bT_a = sum([b[i] * a[i] for i in range(k+1)]) % p
        assert all(value == 0 for value in AT_a), "A^T * a ist nicht ueberall 0."
        assert all(value == 0 for value in BT_b), "B^T * b ist nicht ueberall 0."
        assert bT_a == 1, "b^T * a ist nicht 1."
        logging.info(f"A^T * a mod p: {AT_a}")
        logging.info(f"B^T * b mod p: {BT_b}")
        logging.info(f"b^T * a mod p: {bT_a}")
        
        # Generieren von r und s
        r = [[random.randint(0, p-1) for _ in range(k)] for _ in range(n)]
        s = [[random.randint(0, p-1) for _ in range(k)] for _ in range(m)]
        assert len(r) == n and all(len(row) == k for row in r), "Matrix r hat falsche Dimensionen."
        assert len(s) == m and all(len(row) == k for row in s), "Matrix s hat falsche Dimensionen."
        logging.info(f"r: {r}")
        logging.info(f"s: {s}")
        logging.info(f"Form von r: {len(r)}x{len(r[0])}")
        logging.info(f"Form von s: {len(s)}x{len(s[0])}")
        
        # Berechnung des Master Public Key (mpk)
        exponent_mpk = sum([b[i] * a[i] for i in range(k+1)]) % p
        exponent_mpk_zr = group.init(group.ZR, exponent_mpk)
        mpk = group.gt ** exponent_mpk_zr  # Verwenden Sie group.gt, nicht pairing_result
        logging.info(f"mpk: {mpk}")
        logging.info(f"Typ von mpk: {type(mpk)}")
        
        # Master Secret Key
        msk = (A, a, B, b, r, s)
        logging.info(f"Master Secret Key (msk): {msk}")
        
        ## KEYGEN
        logging.info("KEYGEN")
        u = random.randint(1, p-1)
        # F ist eine Matrix in Z_p^(n x m). Hier ein Beispiel mit zufaelligen Werten:
        F = [[random.randint(0, p-1) for _ in range(m)] for _ in range(n)]
        assert len(F) == n and all(len(row) == m for row in F), "Matrix F hat falsche Dimensionen."
        logging.info(f"u: {u}")
        logging.info(f"F: {F}")
        logging.info(f"Form von F: {len(F)}x{len(F[0])}")
        
        sum_val = 0
        for i in range(n):
            for j in range(m):
                # Berechnung des Summanden
                # Schritt 1: Berechnung von r[i] * A^T
                r_dot_A = 0
                for row in range(k+1):
                    for col in range(k):
                        r_dot_A += A[row][col] * r[i][col]
                r_dot_A %= p
                
                # Schritt 2: Berechnung von B * s[j]
                B_dot_s = 0
                for row in range(k+1):
                    for col in range(k):
                        B_dot_s += B[row][col] * s[j][col]
                B_dot_s %= p
                
                # Schritt 3: Berechnung des Dot-Produkts
                dot_product = (r_dot_A * B_dot_s) % p
                sum_val = (sum_val + (F[i][j] * dot_product)) % p
                logging.debug(f"i={i}, j={j}, F[{i}][{j}]={F[i][j]}, dot_product={dot_product}, sum_val={sum_val}")
        
        sum_zr = group.init(group.ZR, sum_val)
        logging.info(f"sum_val: {sum_val}, type: {type(sum_val)}")
        logging.info(f"sum_zr: {sum_zr}, Typ von sum_zr: {type(sum_zr)}")
        
        u_zr = group.init(group.ZR, u)
        logging.info(f"u_zr: {u_zr}, Typ von u_zr: {type(u_zr)}")
        
        # Berechnung von K und K_tilde
        try:
            exponent_K = (sum_zr.value - u_zr.value) % p
            logging.info(f"exponent_K: {exponent_K}")
            K = group.pair_prod([g1 ** exponent_K], [g2])       # Paarungsergebnis fuer K
            K_tilde = group.pair_prod([g1], [g2 ** u_zr.value])  # Paarungsergebnis fuer K_tilde
            logging.info(f"K: {K.value}")
            logging.info(f"K_tilde: {K_tilde.value}")
            logging.info(f"Typ von K: {type(K)}")
            logging.info(f"Typ von K_tilde: {type(K_tilde)}")
        except ValueError as ve:
            logging.error(f"Fehler bei der Berechnung von K oder K_tilde: {ve}")
            traceback.print_exc()
            raise ve
        
        skF = (K, K_tilde)  # Secret Key fuer F
        logging.info(f"Secret Key fuer F (skF): {skF}")
          
        ## ENCRYPT
        logging.info("ENCRYPT")
        # Input vectors (x, y) <- Z_p^n x Z_p^m
        x = [random.randint(1, p-1) for _ in range(n)]
        y = [random.randint(1, p-1) for _ in range(m)]
        logging.info(f"x: {x}")
        logging.info(f"y: {y}")
        logging.info(f"Form von x: {len(x)}")
        logging.info(f"Form von y: {len(y)}")
        
        # Compute c and c_tilde with integer values
        try:
            # Berechnung von c und c_tilde
            # c = [ (A * r[i] + b * x[i]) mod p for i in range(n) ]
            # c_tilde = [ (B * s[j] + a * y[j]) mod p for j in range(m) ]
            c = []
            for i in range(n):
                sum_A_r = 0
                for row in range(k+1):
                    for col in range(k):
                        sum_A_r += A[row][col] * r[i][col]
                sum_A_r %= p
                sum_b_x = (b[i] * x[i]) % p
                c_val = (sum_A_r + sum_b_x) % p
                c.append(c_val)
            
            c_tilde = []
            for j in range(m):
                sum_B_s = 0
                for row in range(k+1):
                    for col in range(k):
                        sum_B_s += B[row][col] * s[j][col]
                sum_B_s %= p
                sum_a_y = (a[j] * y[j]) % p
                c_tilde_val = (sum_B_s + sum_a_y) % p
                c_tilde.append(c_tilde_val)
            assert len(c) == n, "Vektor c hat falsche Laenge."
            assert len(c_tilde) == m, "Vektor c_tilde hat falsche Laenge."
            logging.info(f"c: {c}")
            logging.info(f"c_tilde: {c_tilde}")
        except Exception as e:
            logging.error(f"Fehler bei der Berechnung von c oder c_tilde: {e}")
            traceback.print_exc()
            raise e
        
        ## DECRYPT
        logging.info("DECRYPT")
        D = group.init(group.GT, 1)  # Initialisierung von D als Identitaetselement in GT
        assert group.is_valid(D), "D ist kein gueltiges Gruppenelement."
        logging.info(f"Initialisiertes D: {D.value}")
        logging.info(f"Typ von D: {type(D)}")
        
        for i in range(n):
            for j in range(m):
                # Berechnung des Exponenten: F[i][j] + c[i] * c_tilde[j]
                exponent = (F[i][j] + (c[i] * c_tilde[j])) % p  # Verwenden Sie eine einfache Multiplikation
                logging.info(f"Exponent fuer D *= gt^{exponent}: {exponent}")
                
                # Konvertieren des Exponenten zu einem ZR-Element
                exponent_zr = group.init(group.ZR, exponent)
                logging.info(f"Exponent ZR: {exponent_zr.value}, Typ von exponent_zr: {type(exponent_zr)}")
                
                # Multiplizieren von D mit gt^exponent_zr
                try:
                    logging.info(f"Multipliziere D mit group.gt^{exponent_zr.value}...")
                    D_before = D.value
                    D = D * (group.gt ** exponent_zr)  # Verwenden Sie group.gt, nicht pairing_result
                    logging.info(f"D vor Multiplikation: {D_before}")
                    logging.info(f"D nach Multiplikation: {D.value}")
                    logging.info(f"Typ von D nach Multiplikation: {type(D)}")
                except Exception as e:
                    logging.error(f"Fehler bei der Multiplikation: {e}")
                    traceback.print_exc()
                    raise e  # Optional: Weiterwerfen, um den Haupt-try-except-Block zu aktivieren
        
        try:
            # Adjust D by multiplying with the inverses of K and K_tilde
            logging.info("Passe D durch Inversen von K und K_tilde an...")
            
            # Berechnung der Inversen von K und K_tilde direkt in GT
            inverse_K = K ** (-1)
            inverse_K_tilde = K_tilde ** (-1)
            logging.info(f"inverse_K: {inverse_K.value}")
            logging.info(f"inverse_K_tilde: {inverse_K_tilde.value}")
            
            # Multiplikation mit den Inversen
            D_before = D.value
            D = D * inverse_K
            assert group.is_valid(D), "D ist kein gueltiges Gruppenelement."
            logging.info(f"D vor Multiplikation mit inverse_K: {D_before}")
            logging.info(f"D nach Multiplikation mit inverse_K: {D.value}")
            
            D_before = D.value
            D = D * inverse_K_tilde
            logging.info(f"D vor Multiplikation mit inverse_K_tilde: {D_before}")
            logging.info(f"D nach Multiplikation mit inverse_K_tilde: {D.value}")
            
            logging.info(f"Typ von D nach Multiplikation mit Inversen: {type(D)}")
            logging.info(f"D nach Anpassung: {D.value}")
        except Exception as e:
            logging.error(f"Fehler bei der Anpassung von D: {e}")
            traceback.print_exc()
            raise e
        
        # Suche nach v, sodass [v * (b^T * a)]_T = D
        res = group.init(group.GT, 1)  # Start mit dem Identitaetselement
        v = 0
        exponent_factor = sum([b[i] * a[i] for i in range(k+1)]) % p
        logging.info(f"Exponentenfaktor: {exponent_factor}")
        
        while (D != res and v < 100000):  # Begrenzung auf 100.000 Versuche
            assert group.is_valid(D), "Ungueltiges Gruppenelement D"
            try:
                # Berechnung des Exponenten: v * (b^T * a)
                current_exponent = (v * exponent_factor) % p  # Beide sind Integers
                current_exponent_zr = group.init(group.ZR, current_exponent)
                logging.info(f"Iteration v={v}: current_exponent_zr={current_exponent_zr.value}, Typ: {type(current_exponent_zr)}")
                
                # Berechnung von group.gt^current_exponent_zr
                res_before = res.value
                res = group.gt ** current_exponent_zr
                assert group.is_valid(res), f"Zwischenergebnis res ist kein gueltiges Gruppenelement bei v={v}."
                logging.info(f"res vor Multiplikation: {res_before}")
                logging.info(f"res nach Exponentiation: {res.value}")
                logging.info(f"Typ von res: {type(res)}")
                v += 1
            except Exception as e:
                logging.error(f"Fehler bei der Berechnung von res fuer v={v}: {e}")
                traceback.print_exc()
                raise e
        
        if D == res:
            logging.info(f"v: {v}")
            logging.info(f"res: {res.value}")
            # Beispielhafte Berechnung des erwarteten Ergebnisses
            expected_result = sum([x[i] * F[i][j] * y[j] for i in range(n) for j in range(m)]) % p
            logging.info(f"expected result: {expected_result}")
            logging.info(f"calculated result: {v}")
        else:
            logging.warning("Failed to find a valid v within the range.")

    except Exception as e:
        logging.error(f"Fehler bei QFE: {e}")
        traceback.print_exc()


if __name__ == "__main__":
    try:
        qfe(p, k, group)

    except Exception as e:
        logging.error(f"Fehler bei QFE: {e}")
        traceback.print_exc()
