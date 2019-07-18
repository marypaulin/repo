import unittest

from lib.parallel.priority_queue import PriorityQueue
from lib.parallel.actor import Server, Client, LocalClient
class TestPriortyQueue(unittest.TestCase):

    def test_priority(self):
        def client_task(id, services, termination):
            (queue,) = services
            input = [0, -3, -1, -9, -7, -4, -2, -8, -5, -6]
            for e in input:
                queue.push(e)

        def local_client_task(id, services, termination):
            (queue,) = services
            output = [queue.pop(block=True) for _i in range(10)]
            self.assertEqual(output, [-9, -8, -7, -6, -5, -4, -3, -2, -1, 0])


        (queue_server, queue_clients) = PriorityQueue(degree=2)
        server = Server(0, (queue_server,))

        client = Client(0, (queue_clients[0],), client_task)
        client.start()

        server.start()

        local_client = LocalClient(1, (queue_clients[1],), local_client_task)
        local_client.start()

        server.stop()
        client.stop()

if __name__ == '__main__':
    unittest.main()
