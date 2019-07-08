import unittest
from lib.interval import Interval

class TestInterval(unittest.TestCase):

    def test_main(self):
        reference = Interval(-10, 10)
        subset = Interval(0)
        superset = Interval(-20, 20)
        le = Interval(-20, 10)
        lt = Interval(-20, -15)
        ge = Interval(10, 20)
        gt = Interval(15, 20)

        self.assertEqual(Interval(0), Interval(0))
        self.assertNotEqual(Interval(0), Interval(-10, 10))
        self.assertEqual(reference.value(), (-10, 10))
        self.assertEqual(subset.value(), 0)

        self.assertTrue(reference | superset, superset)
        self.assertTrue(superset | reference, superset)
        self.assertTrue(reference + superset, superset)
        self.assertTrue(superset + reference, superset)
        self.assertTrue(reference & superset, reference)
        self.assertTrue(superset & reference, reference)

        self.assertTrue(subset.subset(reference))
        self.assertTrue(superset.superset(reference))
        
        self.assertTrue(le <= reference)
        self.assertTrue(lt < reference)
        self.assertTrue(ge >= reference)
        self.assertTrue(gt > reference)

if __name__ == '__main__':
    unittest.main()
