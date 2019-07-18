from lib.parallel.channel import Channel, EndPoint
from heapq import heappush, heappop

def PriorityQueue(queue=None, degree=1, buffer_limit=None):
    if queue == None:
        queue = []

    # Degree > 1 will introduce a lock on the client producer
    server_consumer, client_producer = Channel(read_lock=True, buffer_limit=degree, channel_type='queue')

    clients = []
    server_producers = []
    for i in range(degree):
        client_consumer, server_producer = Channel(channel_type='pipe')

        client_endpoint = EndPoint(client_consumer, client_producer)
        client = __PriorityQueueClient__(queue, client_endpoint)
        clients.append(client)

        server_producers.append(server_producer)

    server = __PriorityQueueServer__(queue, tuple(server_producers), server_consumer)

    ##################

    # server_consumer, client_producer = Channel(read_lock=True)
    # client_consumer, server_producer = Channel(write_lock=True)

    # clients = tuple(__PriorityQueueClient__(queue, EndPoint(client_consumer, client_producer)) for _i in range(degree))

    # server = __PriorityQueueServer__(queue, (server_producer,), server_consumer)

    return (server, tuple(clients))
class __PriorityQueueServer__:
    def __init__(self, queue, producers, consumer):
        self.priority_queue = queue
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
                while True:
                    element = producer.pop(block=False)
                    if element == None:
                        break
                    if not type(element) in {tuple, list}:
                        key = element
                    else:
                        key = element[1:]
                    if not key in seen:
                        seen.add(key)
                        heappush(self.priority_queue, element)

            modified = len(self.priority_queue) > 0

            # Transfer from priorty queue to outbound queue
            while self.priority_queue and not self.consumer.full():
                element = heappop(self.priority_queue)
                if not self.consumer.push(element, block=False):
                    heappush(self.priority_queue, element)
                    break
        return modified

    def close(self, block=True):
        self.online = False
        self.consumer.close()
        for producer in self.producers:
            producer.close()
        self.consumer = None
        self.producers = None

class __PriorityQueueClient__:
    def __init__(self, queue, endpoint):
        self.endpoint = endpoint
        self.online = True

    def push(self, element, block=True):
        '''
        Pushes object into pipeline
        Returns True if successful
        Returns False if unsuccessful
        '''
        if not self.online:
            raise Exception("PriorityQueueError: Operation unavailable when offline")
        self.endpoint.push(element, block=block)

    def pop(self, block=True):
        '''
        Pops object from pipeline
        Returns (priority, element) if successful
        Returns (None, None) if unsuccessful
        '''
        if not self.online:
            raise Exception("PriorityQueueError: Operation unavailable when offline")
        element = self.endpoint.pop(block=block)
        return element

    def close(self, block=True):
        self.online = False
        self.endpoint.close()
        self.endpoint = None
