import unittest
from gmpy2 import mpz

import lib.vector as vect

class TestVector(unittest.TestCase):

    def test_vectorize(self):
        self.assertEqual(vect.vectorize([0, 0, 0]), mpz(8))

    def test_devectorize(self):
        self.assertEqual(vect.devectorize(mpz(8)), [0, 0, 0])

if __name__ == '__main__':
    unittest.main()
