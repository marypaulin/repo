from queue import Empty as QueueEmpty, Full as QueueFull
from multiprocessing import Queue
from collections import deque
from time import sleep
from random import random

import lib.vector as vect
from lib.result import Result
from lib.similarity_index import SimilarityIndex

class TruthTable:
    def __init__(self, table, degree=1):
        self.id = None
        self.degree = degree
        self.local_table = table
        self.inbound_queue = Queue()
        self.outbound_queues = {}
        for i in range(self.degree):
            # key = outbound message queue
            # value = overflow buffer
            self.outbound_queues[i] = (Queue(), deque([]))

    # Service routine called by server
    def service(self):
        '''
        Call periodically to transfer elements along 3-stage pipeline
        Stage 1: inbound
            entries aggregated from processes in FIFO order
            repeated entries are discarded, new entries advance to stage 2
        Stage 2: buffers
            new entries stored in buffers (1 replica per subscriber)
            buffered elements transfer to outbound queue when possible
        Stage 3: outbound
            outbound entries ready for consumption
        '''

        # Transfer from inbound queue to broadcast buffers (if the entry is new)
        while True:
            try:
                (key, value) = self.inbound_queue.get(False)
            except (QueueEmpty):
                break
            else:
                previous_value = self.local_table.get(key)
                if type(previous_value) != Result or value.overwrites(previous_value):
                    # if key != '__terminate__':
                    #     print("TruthTable Update table[{}] = from {} to {}".format(vect.__str__(key), previous_value, value))
                    self.local_table[key] = value
                    for i in range(self.degree):
                        (queue, buffer) = self.outbound_queues[i]
                        buffer.append((key, value))
                else:
                    # print("Rejected TruthTable Update table[{}] = from {} to {}".format(vect.__str__(key), previous_value, value))
                    pass

        # Transfer from broadcast buffers to outbound_queues
        for i in range(self.degree):
            (queue, buffer) = self.outbound_queues[i]
            while len(buffer) > 0:
                element = buffer.popleft()
                try:
                    queue.put(element, False)
                except (QueueFull):
                    buffer.appendleft(element)
                    break

    def __refresh__(self):
        '''
        Receives broadcasted entries from pipeline into local cache
        '''
        if self.id == None:
            return
        (queue, _buffer) = self.outbound_queues[self.id]
        while True:
            try:
                (key, value) = queue.get(False)
            except (QueueEmpty):
                break
            else:
                self.local_table[key] = value

    # API called by workers
    def identify(self, id):
        '''
        Modifies the local instance's id so that broadcasts can be pulled from the correct queue
        '''
        self.id = id

    # API called by workers
    def has(self, key):
        '''
        Queries local cache for value
        Triggers a refresh upon miss
        Returns True upon hit (after refresh)
        Returns False upon miss (after refresh)
        '''
        self.__refresh__()
        return key in self.local_table

    # API called by workers
    def get(self, key):
        '''
        Queries local cache for value
        Triggers a refresh upon miss
        Returns value upon hit (after refresh)
        Returns None upon miss (after refresh)
        '''
        self.__refresh__()
        return self.local_table.get(key)

    # API called by workers
    def put(self, key, value, block=True, prefilter=True):
        '''
        Stores key-value into local cache and sends entry into pipeline
        Returns True if successfully sent into pipeline
        Returns False if unsuccessful in sending to pipeline
        key-value is always written to local cache

        Blocking Semantics:
        '''
        previous_value = self.local_table.get(key)
        if prefilter and type(previous_value) == Result and not value.overwrites(previous_value):
            return False

        self.local_table[key] = value
        while True:
            try:
                self.inbound_queue.put((key, value), False)
            except (QueueFull):
                if block:
                    sleep(random() * 0.1)
                else:
                    return False
            else:
                return True
    
    def neighbours(self, key):
        return self.similarity_index.neighbours(key) if self.similarity_index != None else set()

    def __str__(self):
        return str(self.local_table)
