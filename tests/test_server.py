import unittest
from os import kill
from signal import SIGINT
from multiprocessing import Process

from lib.parallel.server import Server
from lib.parallel.priority_queue import PriorityQueue

class TestServer(unittest.TestCase):

    def test_priority_queue_integration(self):
        queue = PriorityQueue()
        services = (queue,)
        server_id = 21
        server = Server(server_id, services)

        self.assertFalse(server.is_alive())
        self.assertEqual(server.id, server_id)

        queue.identify(0, 'client')
        input = [0, -3, -1, -9, -7, -4, -2, -8, -5, -6]
        for e in input:
            queue.push(e)
        
        server.start()

        self.assertTrue(server.is_alive())

        output = [queue.pop() for _i in range(10)]
        server.stop()

        self.assertFalse(server.is_alive())
        self.assertEqual(output, [-9, -8, -7, -6, -5, -4, -3, -2, -1, 0])

if __name__ == '__main__':
    unittest.main()
