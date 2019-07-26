from itertools import cycle
from random import shuffle
from multiprocessing import Value
from math import ceil, floor
from time import time

from lib.parallel.channel import Channel, EndPoint
from lib.data_structures.heap_queue import HeapQueue

def QueueService(queue=None, degree=1, synchronization_cooldown=0):
    if queue == None:
        queue = HeapQueue()

    server_consumer, client_producer = Channel(read_lock=True, channel_type='queue')
    global_length = Value('i', len(queue))
    clients = []
    server_producers = []
    for i in range(degree):
        # client_endpoint, server_endpoint = Channel(duplex=True, channel_type='pipe')
        client_consumer, server_producer = Channel(channel_type='pipe')

        client_endpoint = EndPoint(client_consumer, client_producer)
        client = __QueueClient__(queue.new(), client_endpoint, global_length, 
            degree=degree, synchronization_cooldown=synchronization_cooldown)
        clients.append(client)

        server_producers.append(server_producer)

    server = __QueueServer__(queue, server_consumer, server_producers, global_length)

    return (server, tuple(clients))

class __QueueServer__:
    def __init__(self, queue, consumer, producers, global_length):
        self.queue = queue
        self.consumer = consumer
        self.producers = producers
        self.global_length = global_length
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
            # shuffle(self.endpoints)
            filtered = 0
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
                    else:
                        filtered += 1
                    
            with self.global_length.get_lock():
                self.global_length.value -= filtered

            # print("Server queue has {} items".format(len(self.queue)))

            modified = len(self.queue) > 0
            while not self.queue.empty() and not self.consumer.full():
                element = self.queue.pop()
                if not self.consumer.push(element, block=False):
                    self.queue.push(element)
                    break

        return modified

    def flush(self):
        self.serve()

class __QueueClient__:
    def __init__(self, queue, endpoint, global_length, degree=1, synchronization_cooldown=0):
        self.queue = queue
        self.endpoint = endpoint
        self.global_length = global_length
        self.synchronization_cooldown = synchronization_cooldown

        self.degree = degree
        self.delta = 0
        self.last_synchronization = 0
        self.online = True
        self.visualizer = None

    def synchronize(self):
        if not self.online:
            return
        if time() > self.last_synchronization + self.synchronization_cooldown:
            self.last_synchronization = time()
        else:
            return

        # Run distribution check
        if self.delta != 0:
            with self.global_length.get_lock():
                self.global_length.value += self.delta
            self.delta = 0

        target = self.global_length.value / self.degree
        tolerance = floor(self.global_length.value * 0.1)
        lower_target = floor(target) - tolerance
        upper_target = ceil(target) + tolerance

        if len(self.queue) > upper_target and len(self.queue) > 1:
            while len(self.queue) > target and len(self.queue) > 1:
                element = self.queue.pop()
                self.endpoint.push(element, block=False)
                if self.visualizer != None:
                    self.visualizer.send(('queue', 'pop', element))
        elif len(self.queue) < lower_target or len(self.queue) < 1:
            while len(self.queue) < target or len(self.queue) < 1:
                element = self.endpoint.pop(block=False)
                if element == None:
                    break
                self.queue.push(element)
                if self.visualizer != None:
                    self.visualizer.send(('queue', 'push', element))

    def push(self, element, block=False):
        '''
        Pushes object into pipeline
        Returns True if successful
        Returns False if unsuccessful
        '''
        self.synchronize()
        self.queue.push(element)
        self.delta += 1
        if self.visualizer != None:
            self.visualizer.send(('queue', 'push', element))
        return

    def pop(self, block=False):
        '''
        Pops object from pipeline
        Returns (priority, element) if successful
        Returns (None, None) if unsuccessful
        '''
        self.synchronize()
        element = self.queue.pop()
        if element != None:
            self.delta -= 1
            if self.visualizer != None:
                self.visualizer.send(('queue', 'pop', element))
        return element

    def length(self, local=True):
        if local:
            return len(self.queue)
        else:
            return self.global_length.value

    def flush(self):
        self.synchronize()
