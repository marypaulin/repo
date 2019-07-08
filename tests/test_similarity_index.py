import unittest
from lib.similarity_index import SimilarityIndex
from lib.vector import Vector

class TestSimilarityIndex(unittest.TestCase):

    def test_add(self):
        index = SimilarityIndex(distance=1, dimensions=5, tables=5)
        a = Vector('11111')
        b = Vector('01111')
        c = Vector('10111')
        d = Vector('11011')
        e = Vector('11101')
        f = Vector('11110')
        g = Vector('00000')

        index.add(b)
        index.add(c)
        index.add(d)
        index.add(e)
        index.add(f)
        index.add(g)

        self.assertFalse(a in index, False)
        self.assertEqual(b in index, True)
        self.assertEqual(c in index, True)
        self.assertEqual(d in index, True)
        self.assertEqual(e in index, True)
        self.assertEqual(f in index, True)
        self.assertEqual(g in index, True)

    def test_neigbours(self):
        index = SimilarityIndex(distance=1, dimensions=5, tables=5)
        a = Vector('11111')
        b = Vector('01111')
        c = Vector('10111')
        d = Vector('11011')
        e = Vector('11101')
        f = Vector('11110')
        g = Vector('00000')

        index.add(b)
        index.add(c)
        index.add(d)
        index.add(e)
        index.add(f)
        index.add(g)

        neighbours = index.neighbours(a)
        self.assertEqual(a in neighbours, False)
        self.assertEqual(b in neighbours, True)
        self.assertEqual(c in neighbours, True)
        self.assertEqual(d in neighbours, True)
        self.assertEqual(e in neighbours, True)
        self.assertEqual(f in neighbours, True)
        self.assertEqual(g in neighbours, False)

        index.remove(f)
        neighbours = index.neighbours(a)
        self.assertEqual(f in neighbours, False)

    def test_remove(self):
        index = SimilarityIndex(distance=1, dimensions=5, tables=5)
        a = Vector('11111')
        b = Vector('01111')
        c = Vector('10111')
        d = Vector('11011')
        e = Vector('11101')
        f = Vector('11110')
        g = Vector('00000')

        index.add(b)
        index.add(c)
        index.add(d)
        index.add(e)
        index.add(f)
        index.add(g)

        index.remove(g)
        self.assertEqual(g in index, False)

if __name__ == '__main__':
    unittest.main()
