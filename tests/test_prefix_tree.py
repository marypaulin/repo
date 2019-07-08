import unittest
from lib.prefix_tree import PrefixTree

class TestPrefixTree(unittest.TestCase):

    def test_set_prefix_queries(self):
        prefixes = PrefixTree()
        a = 'foo'
        b = 'foobar'
        c = 'foobarbaz'
        d = 'barbaz'
        e = 'baz'

        prefixes.add(a)
        prefixes.add(b)
        prefixes.add(c)
        prefixes.add(d)

        self.assertEqual(c[:prefixes.shortest_prefix(c)], a)
        self.assertEqual(c[:prefixes.longest_prefix(c)], c)
        self.assertEqual(a[:prefixes.shortest_prefix(a)], a)
        self.assertEqual(a[:prefixes.longest_prefix(a)], a)
        self.assertEqual(e[:prefixes.shortest_prefix(e)], '')
        self.assertEqual(e[:prefixes.longest_prefix(e)], '')

    def test_set_interface(self):
        prefixes = PrefixTree()
        a = 'foo'
        b = 'foobar'

        self.assertFalse(a in prefixes)
        self.assertFalse(b in prefixes)

        prefixes.add(a)

        self.assertTrue(a in prefixes)
        self.assertFalse(b in prefixes)

        prefixes.add(b)

        self.assertTrue(a in prefixes)
        self.assertTrue(b in prefixes)

        prefixes.remove(a)

        self.assertFalse(a in prefixes)
        self.assertTrue(b in prefixes)

        prefixes.remove(b)

        self.assertFalse(a in prefixes)
        self.assertFalse(b in prefixes)

    def test_dictonary_interface(self):
        prefixes = PrefixTree()
        a = 'foo'
        b = 'foobar'

        self.assertFalse(a in prefixes)
        self.assertFalse(b in prefixes)
        self.assertEqual(prefixes[a], None)
        self.assertEqual(prefixes[b], None)

        prefixes[a] = 41

        self.assertTrue(a in prefixes)
        self.assertFalse(b in prefixes)
        self.assertEqual(prefixes[a], 41)
        self.assertEqual(prefixes[b], None)

        prefixes[b] = 42

        self.assertTrue(a in prefixes)
        self.assertTrue(b in prefixes)
        self.assertEqual(prefixes[a], 41)
        self.assertEqual(prefixes[b], 42)

        prefixes[b] = 43

        self.assertTrue(a in prefixes)
        self.assertTrue(b in prefixes)
        self.assertEqual(prefixes[a], 41)
        self.assertEqual(prefixes[b], 43)

        del prefixes[a]

        self.assertFalse(a in prefixes)
        self.assertTrue(b in prefixes)
        self.assertEqual(prefixes[a], None)
        self.assertEqual(prefixes[b], 43)

        del prefixes[b]

        self.assertFalse(a in prefixes)
        self.assertFalse(b in prefixes)
        self.assertEqual(prefixes[a], None)
        self.assertEqual(prefixes[b], None)


if __name__ == '__main__':
    unittest.main()
