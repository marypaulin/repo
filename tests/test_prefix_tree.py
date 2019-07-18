import unittest
from lib.data_structures.prefix_tree import PrefixTree

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

    def test_set_bad_queries(self):
        prefixes = PrefixTree()
        keys = [
            (1,), (5,), (6,), ('False', 2), ('False', 8), ('False', 11), ('False',9), 
            ('False', 'False', 0), ('False', 10), ('False', 3, 'False', 0)
        ]
        for key in keys:
            prefixes[key] = True
        self.assertEqual(len(prefixes), 10)
        self.assertEqual(prefixes.shortest_prefix(('True', 7, 'False', 0)), 0)

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

    def test_items(self):
        prefixes = PrefixTree()
        a = 'foo'
        b = 'foobar'

        self.assertEqual(prefixes.items(), {})

        prefixes[a] = True

        self.assertEqual(prefixes.items(), { ('f','o','o'): True })

        prefixes[b] = True

        self.assertEqual(prefixes.items(), { ('f','o','o'): True , ('f','o','o','b','a','r'): True })

        del prefixes[a]

        self.assertEqual(prefixes.items(), { ('f','o','o','b','a','r'): True })

        del prefixes[b]

        self.assertEqual(prefixes.items(), {})


if __name__ == '__main__':
    unittest.main()
