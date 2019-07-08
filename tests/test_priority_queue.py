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
        
        queue = PriorityQueue()
        server = Process(target=serve, args=([queue],))
        input = [0, -3, -1, -9, -7, -4, -2, -8, -5, -6]
        for e in input:
            queue.push(e)

        server.start()
        output = [ queue.pop() for _i in range(10) ]
        kill(server.pid, SIGINT)
        self.assertEqual(output, [-9, -8, -7, -6, -5, -4, -3, -2, -1, 0])

if __name__ == '__main__':
    unittest.main()
