from charm.toolbox.pairinggroup import PairingGroup, G1, G2, GT, ZR
import logging
import traceback

# Konfigurieren des Loggers
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def test_charm():
    try:
        logging.info("Initialisiere PairingGroup 'BN254'")
        group = PairingGroup('BN254')
        logging.info("PairingGroup erfolgreich initialisiert.")
        
        logging.info("Generiere zufällige Elemente g1 und g2")
        g1 = group.random(G1)
        g2 = group.random(G2)
        logging.info(f"g1: {g1}")
        logging.info(f"g2: {g2}")
        
        logging.info("Berechne das Paarungsprodukt gt = e(g1, g2)")
        gt = group.pair_prod(g1, g2)  # Verwendung der korrekten Methode 'pair_prod'
        logging.info(f"gt: {gt}")
        logging.info(f"Typ von gt: {type(gt)}")
        
        logging.info("Exponentiere gt mit einem ZR-Element")
        exponent = group.init(ZR, 5)
        result = gt ** exponent
        logging.info(f"gt^5: {result}")
        
        logging.info("Test erfolgreich abgeschlossen.")
        
    except Exception as e:
        logging.error(f"Ein Fehler ist aufgetreten: {e}")
        traceback.print_exc()

test_charm()
