import unittest
from os import kill
from signal import SIGINT
from multiprocessing import Process
from time import sleep

from lib.parallel.client import Client
from lib.parallel.truth_table import TruthTable


class TestClient(unittest.TestCase):
    # Test the basic function of the truth table
    def test_truth_table_integration(self):
        def serve(services):
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

        def peer_task(client_id, services):
            try:
                (table,) = services
                table.identify(0, 'client')
                input = [0, -3, -1, -9, -7, -4, -2, -8, -5, -6]
                for e in input:
                    table.put(e, e * 2)
                table.close()
            except KeyboardInterrupt:
                pass

        # Declare that there are 2 clients sharing this table
        table = TruthTable(degree=2)

        peer = Client(1, (table,), peer_task)
        peer.start()

        server = Process(target=serve, args=([table],))
        server.start()

        table.identify(1, 'client')
        for i in range(10):
            self.assertEqual(table.get(-i), -i*2)
        table.close()

        peer.stop()
        kill(server.pid, SIGINT)

if __name__ == '__main__':
    unittest.main()
