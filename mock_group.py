import random
import logging

class MockElement:
    def __init__(self, group, value, group_type):
        self.group = group
        self.value = value % group.p
        self.type = group_type

    def __pow__(self, exponent):
        if isinstance(exponent, MockElement):
            if exponent.type != 'ZR':
                raise TypeError("Exponent must be of type ZR")
            exp = exponent.value
        elif isinstance(exponent, int):
            exp = exponent
        else:
            raise TypeError("Unsupported exponent type")

        # Stellen Sie sicher, dass der Exponent positiv ist
        exp = exp % self.group.p

        result = pow(self.value, exp, self.group.p)
        return MockElement(self.group, result, self.type)

    def __mul__(self, other):
        if isinstance(other, MockElement):
            if self.type != other.type:
                raise TypeError("Cannot multiply elements of different types")
            result = (self.value * other.value) % self.group.p
            return MockElement(self.group, result, self.type)
        elif isinstance(other, int):
            result = (self.value * other) % self.group.p
            return MockElement(self.group, result, self.type)
        else:
            raise TypeError("Unsupported multiplication type")

    def __truediv__(self, other):
        if isinstance(other, MockElement):
            if other.type != 'ZR':
                raise TypeError("Can only divide by elements of type ZR")
            inv = self.group.mod_inverse(other.value, self.group.p)
            result = (self.value * inv) % self.group.p
            return MockElement(self.group, result, self.type)
        elif isinstance(other, int):
            inv = self.group.mod_inverse(other, self.group.p)
            result = (self.value * inv) % self.group.p
            return MockElement(self.group, result, self.type)
        else:
            raise TypeError("Unsupported division type")

    def __eq__(self, other):
        if not isinstance(other, MockElement):
            return False
        return self.value == other.value and self.type == other.type and self.group.p == other.group.p

    def __repr__(self):
        return f"{self.value}"

class MockGroup:
    def __init__(self, name='MockGroup', p=101):
        self.name = name
        self.p = p  # Eine kleine Primzahl für die Mock-Gruppe
        self.ZR = 'ZR'
        self.G1 = 'G1'
        self.G2 = 'G2'
        self.GT = 'GT'
        self.gt = MockElement(self, 2, self.GT)  # Festes Paarungsergebnis auf 2 gesetzt

    def random(self, group_type):
        if group_type not in [self.G1, self.G2, self.GT]:
            raise ValueError(f"Unsupported group type: {group_type}")
        value = random.randint(1, self.p - 1)
        return MockElement(self, value, group_type)

    def init(self, group_type, value):
        if group_type not in [self.ZR, self.G1, self.G2, self.GT]:
            raise ValueError(f"Unsupported group type: {group_type}")
        return MockElement(self, value, group_type)

    def pair_prod(self, list1, list2):
        if len(list1) != len(list2):
            raise ValueError("Lists must have the same length for pairing")
        if len(list1) == 0:
            raise ValueError("Lists cannot be empty for pairing")
        # Paarung e(G1, G2) = gt^{a * b} mod p
        product = 1
        for a, b in zip(list1, list2):
            if not isinstance(a, MockElement) or not isinstance(b, MockElement):
                raise TypeError("Elements must be of type MockElement")
            if a.type != self.G1 or b.type != self.G2:
                raise TypeError("Pairing requires elements of type G1 and G2")
            exponent = (a.value * b.value) % self.p
            pairing = pow(self.gt.value, exponent, self.p)
            logging.debug(f"Pairing e({a.value}, {b.value}) = {pairing} (exponent={exponent})")
            product = (product * pairing) % self.p
        logging.debug(f"Gesamte Paarung: {product}")
        return MockElement(self, product, self.GT)

    def mod_inverse(self, a, p):
        # Berechnet die modulare Inverse von a modulo p
        # Wir verwenden den erweiterten euklidischen Algorithmus
        def extended_gcd(a, b):
            if a == 0:
                return b, 0, 1
            gcd, x1, y1 = extended_gcd(b % a, a)
            x = y1 - (b // a) * x1
            y = x1
            return gcd, x, y

        gcd, x, _ = extended_gcd(a, p)
        if gcd != 1:
            raise ValueError(f"Inverse existiert nicht für {a} und {p}")
        else:
            return x % p

    def pair_prod_multiple(self, list1, list2):
        # Unterstützt Paarungen mit mehreren Elementen
        return self.pair_prod(list1, list2)
