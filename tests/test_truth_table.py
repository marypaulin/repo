import unittest
from os import kill
from signal import SIGINT
from multiprocessing import Process

from lib.vector import Vector
from lib.prefix_tree import PrefixTree
from lib.parallel.truth_table import TruthTable

class TestTruthTable(unittest.TestCase):
    # Test the basic function of the truth table
    def test_value_propagation(self):
        def serve(services):
            while True:
                try:
                    for service in services:
                        service.serve()
                except KeyboardInterrupt:
                    break

        def peer_task(table):
            table.identify(0)
            input = [0, -3, -1, -9, -7, -4, -2, -8, -5, -6]
            for e in input:
                table.put(e, e * 2)

        # Declare that there are 2 clients sharing this table
        table = TruthTable(degree=2)

        peer = Process(target=peer_task, args=(table,))
        peer.start()

        server = Process(target=serve, args=([table],))
        server.start()


        table.identify(0)
        for i in range(10):
            self.assertEqual(table.get(-i), -i*2)

        peer.join()
        kill(server.pid, SIGINT)

    # Test that the truth table integrates with an alternative dictionary implementation
    def test_prefix_tree_integration(self):
        input = [Vector('00'), Vector('01'), Vector('10'), Vector('11')]

        def serve(services):
            while True:
                try:
                    for service in services:
                        service.serve()
                except KeyboardInterrupt:
                    break

        def peer_task(table):
            table.identify(0)
            for e in input:
                table.put(e, ~e)

        # Declare that there are 2 clients sharing this table
        table = TruthTable(table=PrefixTree(), degree=2)

        peer = Process(target=peer_task, args=(table,))
        peer.start()

        server = Process(target=serve, args=([table],))
        server.start()

        table.identify(0)
        for e in input:
            self.assertEqual(table.get(e), ~e)

        peer.join()
        kill(server.pid, SIGINT)


if __name__ == '__main__':
    unittest.main()
