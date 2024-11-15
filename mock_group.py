import random
import logging

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s")

class MockElement:
    def __init__(self, group, value, group_type):
        self.group = group
        self.value = value % group.p
        self.type = group_type

    def validate(self):
        if not (0 <= self.value < self.group.p):
            raise ValueError(f"Wert {self.value} liegt nicht in der Gruppe modulo {self.group.p}.")

    def __pow__(self, exponent):
        self.validate()
        if isinstance(exponent, MockElement):
            if exponent.type != 'ZR':
                raise TypeError("Exponent must be of type ZR")
            exp = exponent.value
        elif isinstance(exponent, int):
            exp = exponent
        else:
            raise TypeError("Unsupported exponent type")
        
        exp %= self.group.p
        result = pow(self.value, exp, self.group.p)
        return MockElement(self.group, result, self.type)

    def __mul__(self, other):
        self.validate()
        if isinstance(other, MockElement):
            other.validate()
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
        self.validate()
        if isinstance(other, MockElement):
            other.validate()
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
        if not self.is_prime(p):
            raise ValueError(f"{p} ist keine gültige Primzahl.")
        self.name = name
        self.p = p
        self.ZR = 'ZR'
        self.G1 = 'G1'
        self.G2 = 'G2'
        self.GT = 'GT'
        self.gt = MockElement(self, 12, self.GT)

    @staticmethod
    def is_prime(num):
        if num <= 1:
            return False
        for i in range(2, int(num**0.5) + 1):
            if num % i == 0:
                return False
        return True

    def is_valid(self, element):
        """
        Überprüft, ob ein Element ein gültiges Gruppenelement ist.
        """
        if not isinstance(element, MockElement):
            logging.error(f"Element {element} ist kein gültiges MockElement.")
            return False
        if element.value < 0 or element.value >= self.p:
            logging.error(f"Wert {element.value} liegt außerhalb des gültigen Bereichs (0 bis {self.p - 1}).")
            return False
        if element.type not in [self.G1, self.G2, self.GT, self.ZR]:
            logging.error(f"Ungültiger Typ {element.type} für Gruppenelement.")
            return False
        return True

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
        try:
            if len(list1) != len(list2):
                raise ValueError("Lists must have the same length for pairing")
            if len(list1) == 0:
                raise ValueError("Lists cannot be empty for pairing")
            
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
        except Exception as e:
            logging.error(f"Fehler bei der Paarungsoperation: {e}")
            logging.info("Suche nach passenden Parametern für gültige Paarungen...")
            self.find_valid_pairing(list1, list2)
            raise

    def find_valid_pairing(self, list1, list2):
        for a in range(1, self.p):
            for b in range(1, self.p):
                if (a * b) % self.p != 0:
                    logging.info(f"Gefundene gültige Werte: G1={a}, G2={b}")
                    return MockElement(self, a, self.G1), MockElement(self, b, self.G2)

    def mod_inverse(self, a, p):
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
        return x % p

    def pair_prod_multiple(self, list1, list2):
        return self.pair_prod(list1, list2)
