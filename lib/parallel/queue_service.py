from lib.parallel.channel import Channel, EndPoint
from lib.data_structures.heap_queue import HeapQueue

def QueueService(queue=None, degree=1, buffer_limit=None):
    if queue == None:
        queue = HeapQueue()

    # Degree > 1 will introduce a lock on the client producer
    server_consumer, client_producer = Channel(read_lock=True, channel_type='pipe')

    clients = []
    server_producers = []
    for i in range(degree):
        client_consumer, server_producer = Channel(channel_type='pipe')

        client_endpoint = EndPoint(client_consumer, client_producer)
        client = __QueueClient__(queue, client_endpoint)
        clients.append(client)

        server_producers.append(server_producer)

    server = __QueueServer__(queue, tuple(server_producers), server_consumer)

    return (server, tuple(clients))

class __QueueServer__:
    def __init__(self, queue, producers, consumer):
        self.queue = queue
        self.producers = producers # Multiple endpoints to read from
        self.consumer = consumer # Single endpoint to write to
        self.online = True

    def serve(self):
        '''
        Call periodiclly to transfer elements along 3-stage pipeline
        Stage 1: inbound
          messages aggregated from processes in FIFO order
        Stage 2: priority
          messages sorted by priority using heapq module
        Stage 3: outbound
          sorted messages ready for distribution
        '''
        modified = False
        if self.online:
            seen = set()
            # Transfer from inbound queue to priority queue
            for producer in self.producers:
                while not self.queue.full():
                    element = producer.pop(block=False)
                    if element == None:
                        break
                    key = element if not type(element) in {tuple, list} else element[1:]
                    if not key in seen:
                        seen.add(key)
                        self.queue.push(element)

            modified = len(self.queue) > 0

            # Transfer from priorty queue to outbound queue
            while not self.queue.empty() and not self.consumer.full():
                element = self.queue.pop()
                if not self.consumer.push(element, block=False):
                    self.queue.push(element)
                    break
        return modified

    def flush(self):
        self.serve()

class __QueueClient__:
    def __init__(self, queue, endpoint):
        self.endpoint = endpoint
        self.online = True

    def push(self, element, block=False):
        '''
        Pushes object into pipeline
        Returns True if successful
        Returns False if unsuccessful
        '''
        if not self.online:
            raise Exception("QueueServiceError: Operation unavailable when offline")
        return self.endpoint.push(element, block=block)

    def pop(self, block=False):
        '''
        Pops object from pipeline
        Returns (priority, element) if successful
        Returns (None, None) if unsuccessful
        '''
        if not self.online:
            raise Exception("QueueServiceError: Operation unavailable when offline")
        element = self.endpoint.pop(block=block)
        return element

    def flush(self):
        pass