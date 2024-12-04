# tests/test_utils.py

import unittest
import numpy as np
from modules.utils import classify_damage, calculate_pci

class TestUtils(unittest.TestCase):

    def setUp(self):
        # Configuración inicial para las pruebas
        self.damages_dict = {
            1: {'nombre': 'Grieta longitudinal', 'unidad': 'm2'},
            2: {'nombre': 'Grieta transversal', 'unidad': 'm2'}
        }
        self.deduct_values = {
            '1': {'B': {0: 0, 1: 10}, 'M': {0: 0, 1: 20}, 'A': {0: 0, 1: 30}},
            '2': {'B': {0: 0, 1: 15}, 'M': {0: 0, 1: 25}, 'A': {0: 0, 1: 35}}
        }

    def test_classify_damage(self):
        mask = np.array([
            [0, 1, 1, 0],
            [0, 1, 1, 0],
            [0, 2, 2, 0],
            [0, 2, 2, 0]
        ])
        damage_info = classify_damage(mask, self.damages_dict)
        self.assertIn(1, damage_info)
        self.assertIn(2, damage_info)
        self.assertEqual(damage_info[1]['nombre'], 'Grieta longitudinal')
        self.assertEqual(damage_info[2]['nombre'], 'Grieta transversal')

    def test_calculate_pci(self):
        damage_info = {
            1: {'severity': 'B', 'density': 1.0, 'deduct_value': 10},
            2: {'severity': 'M', 'density': 2.0, 'deduct_value': 25}
        }
        pci = calculate_pci(damage_info, self.deduct_values)
        self.assertAlmostEqual(pci, 65.0, places=1)

if __name__ == '__main__':
    unittest.main()