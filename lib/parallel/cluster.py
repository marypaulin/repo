from time import time

from lib.parallel.actor import Client, LocalClient, Server, LocalServer
from lib.parallel.channel import Channel
class Cluster:
    def __init__(self, task, services, size=1):
        self.task = task
        self.size = size

        # Each service is structured as:
        # tuple( server_interface, tuple( client_1_interface_1, client_2_interface, ... ) )

        # Extract a bundle of server interfaces out of each service
        self.server_bundle = tuple( service[0] for service in services)

        # Extract bundles of client interfaces out of each service
        self.client_bundles = tuple( tuple( service[1][client_id] for service in services ) for client_id in range(self.size))
        self.client_bundle = self.client_bundles[0]

    def compute(self, max_time):
        (output_consumer, output_producer) = Channel(write_lock=True, channel_type='queue')
        clients = tuple(Client(i, self.client_bundles[i], self.task, output_channel=output_consumer) for i in range(0, self.size))
        server = Server(self.size, self.server_bundle, output_channel=output_consumer)


        start_time = time()

        server.start()
        for client in clients:
            client.start()

        # Permit GC on local service resources now that they have been transferred to their respective subprocesses
        self.server_bundle = None
        self.client_bundles = None

        result = None
        while time() - start_time < max_time:
            result = output_producer.pop(block=False)
            if result != None:
                break

        # Possibly terminate daemon actors?
        server.actor.terminate()
        for client in clients:
            client.actor.terminate()

        return result
