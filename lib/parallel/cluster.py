from multiprocessing import Process
from time import sleep

from lib.parallel.priority_queue import PriorityQueue
from lib.parallel.truth_table import TruthTable
from lib.parallel.client import Client
from lib.parallel.server import Server

class Cluster:
    def __init__(self, task, terminate, queue, table, clients=1, servers=1):
        self.queue = queue
        self.table = table
        self.task = task
        self.clients = clients
        self.servers = servers
        self.terminate = terminate

    def compute(self):
        clients = [Process(target=Client, args=(i+1, self.task, self.queue, self.table)) for i in range(self.clients-1)]
        servers = [Process(target=Server, args=(i, self.queue, self.table)) for i in range(self.servers)]

        for node in (servers + clients):
            node.start()

        # Run self as client 0
        self.table.identify(0)
        self.queue.identify(0)
        Client(0, self.task, self.queue, self.table)

        for client in (clients):
            client.join()

        while any(server.is_alive() for server in servers):
            self.table.put('__terminate__', True, prefilter=False)  # Signal to servers to terminate
            sleep(0.1)

        return self.queue, self.table
