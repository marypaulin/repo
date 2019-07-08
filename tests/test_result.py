import unittest
from lib.similarity_index import SimilarityIndex
from lib.result import Result
from lib.interval import Interval

class TestResult(unittest.TestCase):

    def test_overwrite(self):
        a = None
        b = Result(optimum=None)
        c = Result(optimum=Interval(1, 10))
        d = Result(optimum=Interval(4, 5))
        e = Result(optimum=Interval(4.5))

        self.assertFalse(b.overwrites(c))
        self.assertFalse(c.overwrites(d))
        self.assertFalse(d.overwrites(e))

        self.assertTrue(b.overwrites(a))
        self.assertTrue(c.overwrites(b))
        self.assertTrue(d.overwrites(c))
        self.assertTrue(e.overwrites(d))

if __name__ == '__main__':
    unittest.main()
