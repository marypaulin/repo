import unittest

from lib.parallel.truth_table import TruthTable
from lib.parallel.actor import Server, Client, LocalClient

class TestTruthTable(unittest.TestCase):
    # Test the basic function of the truth table
    def test_value_propagation(self):

        def client_task(id, services, termination):
            (table,) = services
            input = [0, -3, -1, -9, -7, -4, -2, -8, -5, -6]
            for e in input:
                table.put(e, e * 2)

        def local_client_task(id, services, termination):
            (table,) = services
            for i in range(10):
                self.assertEqual(table.get(-i), -i*2)


        # Declare that there are 2 clients sharing this table
        (server_table, client_tables) = TruthTable(degree=2)

        client = Client(1, (client_tables[1],), client_task)
        client.start()

        server = Server(0, (server_table,))
        server.start()

        local_client = LocalClient(0, (client_tables[0],), local_client_task)
        local_client.start()

        client.stop()
        server.stop()

if __name__ == '__main__':
    unittest.main()
