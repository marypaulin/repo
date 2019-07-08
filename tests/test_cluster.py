import unittest

from lib.fibonacci import Fibonacci

class TestCluster(unittest.TestCase):
    # Fibonacci effectively performs a small functional test on cluster.py
    # That was the purpose of creating it from the start
    def test_cluster_on_fibonacci(self):
        problem = Fibonacci(100)
        solution = problem.solve(clients=4, servers=1)
        self.assertEqual(solution, 354224848179261915075)

if __name__ == '__main__':
    unittest.main()
