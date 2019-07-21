from itertools import cycle
from random import shuffle
from multiprocessing import Value
from math import ceil
from time import time

from lib.parallel.channel import Channel, EndPoint
from lib.data_structures.heap_queue import HeapQueue

def QueueService(queue=None, degree=1, synchronization_cooldown=0):
    if queue == None:
        queue = HeapQueue()
    global_length = Value('d', len(queue))
    clients = []
    server_endpoints = []
    for i in range(degree):
        client_endpoint, server_endpoint = Channel(duplex=True, channel_type='pipe')

        client = __QueueClient__(queue, client_endpoint, global_length, 
            degree=degree, synchronization_cooldown=synchronization_cooldown)
        clients.append(client)

        server_endpoints.append(server_endpoint)

    server = __QueueServer__(queue, server_endpoints, global_length)

    return (server, tuple(clients))

class __QueueServer__:
    def __init__(self, queue, endpoints, global_length):
        self.queue = queue
        self.endpoints = endpoints
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

            shuffle(self.endpoints)

            filtered = 0
            seen = set()
            # Transfer from inbound queue to priority queue
            for endpoint in self.endpoints:
                while not self.queue.full():
                    element = endpoint.pop(block=False)
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

            modified = len(self.queue) > 0
           

            for endpoint in cycle(self.endpoints):
                if self.queue.empty():
                    break
                element = self.queue.pop()
                if not endpoint.push(element, block=False):
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

        target = ceil(self.global_length.value / self.degree)

        if abs(target - len(self.queue)) / max(self.global_length.value, 1) < 0.5:
            return

        # Push elements if over the average
        while len(self.queue) > target and not self.queue.empty():
            element = self.queue.pop()
            self.endpoint.push(element, block=False)

        # Pop element if under the average
        while len(self.queue) < target:
            element = self.endpoint.pop(block=False)
            if element == None:
                break
            self.queue.push(element)

    def push(self, element, block=False):
        '''
        Pushes object into pipeline
        Returns True if successful
        Returns False if unsuccessful
        '''
        self.synchronize()
        success = self.queue.push(element)
        self.delta += 1
        return success

    def pop(self, block=False):
        '''
        Pops object from pipeline
        Returns (priority, element) if successful
        Returns (None, None) if unsuccessful
        '''
        element = self.queue.pop()
        if element == None:
            self.synchronize()
        else:
            self.delta -= 1
        return element

    def flush(self):
        pass
