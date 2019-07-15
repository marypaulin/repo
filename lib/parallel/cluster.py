from multiprocessing import Process
from time import sleep

from lib.parallel.priority_queue import PriorityQueue
from lib.parallel.truth_table import TruthTable
from lib.parallel.client import Client
from lib.parallel.server import Server

class Cluster:
    def __init__(self, task, services, clients=1, servers=1):
        self.services = services
        self.task = task
        self.clients = clients
        self.servers = servers

    def compute(self):
        clients = tuple(Client(i + 1, self.services, self.task) for i in range(self.clients-1))
        servers = tuple(Server(i, self.services) for i in range(self.servers))

        for node in (servers + clients):
            node.start(block=False)

        # Run self as client 0
        for service in self.services:
            service.identify(0)
        # Kind of hacky but kind of works
        Client.__run__(self, 0, self.services, self.task)

        for client in (clients):
            client.join()
        for server in servers:
            server.stop()

        return self.services