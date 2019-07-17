import unittest
from os import kill
from signal import SIGINT
from multiprocessing import Process

from lib.parallel.priority_queue import PriorityQueue

class TestPriortyQueue(unittest.TestCase):

    def test_priority(self):
        def serve(services):
            while True:
                try:
                    for service in services:
                        service.serve()
                except KeyboardInterrupt:
                    break
            for service in services:
                service.close()
        
        (queue_server, queue_clients) = PriorityQueue(degree=1)
        server = Process(target=serve, args=([queue_server],))

        queue_client = queue_clients[0]
        input = [0, -3, -1, -9, -7, -4, -2, -8, -5, -6]
        for e in input:
            queue_client.push(e)

        server.start()
        output = [ queue_client.pop(block=True) for _i in range(10) ]
        kill(server.pid, SIGINT)
        self.assertEqual(output, [-9, -8, -7, -6, -5, -4, -3, -2, -1, 0])

if __name__ == '__main__':
    unittest.main()
