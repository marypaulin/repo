import unittest
from lib.vector import Vector

class TestVector(unittest.TestCase):

    def test_constructor(self):
        v = Vector('1001110000')
        self.assertEqual(str(v), '1001110000')
        self.assertEqual(Vector.ones(5), Vector('11111'))
        self.assertEqual(Vector.zeros(5), Vector('00000'))
        self.assertEqual(Vector([0,1,0,0,1]), Vector('01001'))
        self.assertEqual(Vector.repeat(1, 4), Vector('1111'))

    def test_index(self):
        v = Vector('1001110000')
        self.assertEqual(v[0], 1)
        self.assertEqual(v[1], 0)
        self.assertEqual(v[2], 0)
        self.assertEqual(v[3], 1)
        self.assertEqual(v[4], 1)
        self.assertEqual(v[5], 1)
        self.assertEqual(v[6], 0)
        self.assertEqual(v[7], 0)
        self.assertEqual(v[8], 0)
        self.assertEqual(v[9], 0)

    def test_count(self):
        self.assertEqual(Vector('0000').count(), 0)
        self.assertEqual(Vector('0110').count(), 2)
        self.assertEqual(Vector('1111').count(), 4)

    def test_inversion(self):
        v = ~Vector('1001110000')
        self.assertEqual(v[0], 0)
        self.assertEqual(v[1], 1)
        self.assertEqual(v[2], 1)
        self.assertEqual(v[3], 0)
        self.assertEqual(v[4], 0)
        self.assertEqual(v[5], 0)
        self.assertEqual(v[6], 1)
        self.assertEqual(v[7], 1)
        self.assertEqual(v[8], 1)
        self.assertEqual(v[9], 1)
        self.assertEqual(v, Vector('0110001111'))
        self.assertEqual(~v, Vector('1001110000'))

    def test_and(self):
        a = Vector('1001110000')
        b = Vector('0000111001')
        c = Vector('0000110000')
        d = a & b
        self.assertEqual(d[0], 0)
        self.assertEqual(d[1], 0)
        self.assertEqual(d[2], 0)
        self.assertEqual(d[3], 0)
        self.assertEqual(d[4], 1)
        self.assertEqual(d[5], 1)
        self.assertEqual(d[6], 0)
        self.assertEqual(d[7], 0)
        self.assertEqual(d[8], 0)
        self.assertEqual(d[9], 0)
        self.assertEqual(d, c)

    def test_or(self):
        a = Vector('1001110000')
        b = Vector('0000111001')
        c = Vector('1001111001')
        d = a | b
        self.assertEqual(d[0], 1)
        self.assertEqual(d[1], 0)
        self.assertEqual(d[2], 0)
        self.assertEqual(d[3], 1)
        self.assertEqual(d[4], 1)
        self.assertEqual(d[5], 1)
        self.assertEqual(d[6], 1)
        self.assertEqual(d[7], 0)
        self.assertEqual(d[8], 0)
        self.assertEqual(d[9], 1)
        self.assertEqual(d, c)

    def test_xor(self):
        a = Vector('1001110000')
        b = Vector('0000111001')
        c = Vector('1001001001')
        d = a ^ b
        self.assertEqual(d[0], 1)
        self.assertEqual(d[1], 0)
        self.assertEqual(d[2], 0)
        self.assertEqual(d[3], 1)
        self.assertEqual(d[4], 0)
        self.assertEqual(d[5], 0)
        self.assertEqual(d[6], 1)
        self.assertEqual(d[7], 0)
        self.assertEqual(d[8], 0)
        self.assertEqual(d[9], 1)
        self.assertEqual(d, c)

    def test_mul(self):
        a = Vector('1001110000')
        b = Vector('0000111001')
        c = 2
        d = a * b
        self.assertEqual(d, c)

    def test_len(self):
        a = Vector('1001110000')
        self.assertEqual(len(a), 10)

    def test_str(self):
        a = Vector('1001110000')
        self.assertEqual(str(a), '1001110000')

    def test_iter(self):
        a = Vector('1001110000')
        b = [ e for e in a ]
        c = [1, 0, 0, 1, 1, 1, 0, 0, 0, 0]
        self.assertEqual(b, c)

if __name__ == '__main__':
    unittest.main()
