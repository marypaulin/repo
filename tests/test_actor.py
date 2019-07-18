import unittest

from lib.parallel.actor import LocalClient, Client, Server
from lib.parallel.truth_table import TruthTable

class TestActor(unittest.TestCase):
    # Test the basic function of the truth table
    def test_truth_table_integration_on_process_actors(self):
        def server_task(services):
            for service in services:
                service.identify(0, 'server')
            while True:
                try:
                    for service in services:
                        service.serve()
                except KeyboardInterrupt:
                    break
            for service in services:
                service.close()

        def client_task(client_id, services, terminate):
            (table,) = services
            input = [0, -3, -1, -9, -7, -4, -2, -8, -5, -6]
            for e in input:
                table.put(e, e * 2)
            table.close()

        def local_client_task(client_id, services, terminate):
            (table,) = services
            for i in range(10):
                self.assertEqual(table.get(-i), -i*2)
            table.close()

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

    def test_truth_table_integration_on_thread_actors(self):
        def server_task(services):
            for service in services:
                service.identify(0, 'server')
            while True:
                try:
                    for service in services:
                        service.serve()
                except KeyboardInterrupt:
                    break
            for service in services:
                service.close()

        def client_task(client_id, services, terminate):
            (table,) = services
            input = [0, -3, -1, -9, -7, -4, -2, -8, -5, -6]
            for e in input:
                table.put(e, e * 2)
            table.close()

        def local_client_task(client_id, services, terminate):
            (table,) = services
            for i in range(10):
                self.assertEqual(table.get(-i), -i*2)
            table.close()

        # Declare that there are 2 clients sharing this table
        (server_table, client_tables) = TruthTable(degree=2)

        client = Client(1, (client_tables[1],), client_task, client_type='thread')
        client.start()

        server = Server(0, (server_table,), server_type='thread')
        server.start()

        local_client = LocalClient(0, (client_tables[0],), local_client_task)
        local_client.start()

        client.stop()
        server.stop()

if __name__ == '__main__':
    unittest.main()
