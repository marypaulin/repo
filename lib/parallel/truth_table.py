from queue import Empty as QueueEmpty, Full as QueueFull
from multiprocessing import Queue
from collections import deque
from time import sleep, time
from random import random

import lib.vector as vect
from lib.result import Result
from lib.interval import Interval
from lib.similarity_index import SimilarityIndex
from lib.prefix_tree import PrefixTree
class TruthTable:
    def __init__(self, table=None, propagator=None, refresh_cooldown=0, degree=1):
        self.id = None
        self.degree = degree
        self.client_table = table if table != None else {}
        self.server_table = {}

        self.inbound_queue = Queue()
        self.outbound_queues = {}
        self.resolved = 0
        self.pending = 0
        self.last_refresh = 0
        self.refresh_cooldown = refresh_cooldown
        self.propagator = propagator
        for i in range(self.degree):
            self.outbound_queues[i] = (Queue(), deque([]))

    # Service routine called by server
    def serve(self):
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
                previous_value = self.server_table.get(key)
                if type(previous_value) != Result or value.overwrites(previous_value):
                    # print("TruthTable Update table[{}] = from {} to {}".format(str(key), str(previous_value), str(value)))
                    self.server_table[key] = value
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
            raise Exception("TruthTableException: Client API invoked without client identification")
        (queue, _buffer) = self.outbound_queues[self.id]
        if time() > self.last_refresh + self.refresh_cooldown:
            self.last_refresh = time()
        else:
            return
        while True:
            try:
                (key, value) = queue.get(False)
            except (QueueEmpty):
                break
            else:
                self.client_table[key] = value
                # Perform information propagation
                if self.propagator != None and type(value) == Result:
                    result = value
                    if result.optimizer == None and not self.propagator.tracking(key):
                        self.propagator.track(key)
                    if result.optimizer != None and self.propagator.tracking(key):
                        self.propagator.untrack(key)
                    
                    updates = self.propagator.propagate(key, result, self.client_table)
                    for update_key, update_value in updates.items():
                        self.put(update_key, update_value)


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
        return key in self.client_table

    # API called by workers
    def get(self, key, block=True):
        '''
        Queries local cache for value
        Triggers a refresh upon miss
        Returns value upon hit (after refresh)
        Returns None upon miss (after refresh)
        '''
        
        self.__refresh__()
        if block:
            while not key in self.client_table:
                self.__refresh__()
                sleep(random() * 0.01)
        return self.client_table.get(key)

    # API called by workers
    def put(self, key, value, block=True, prefilter=True):
        '''
        Stores key-value into local cache and sends entry into pipeline
        Returns True if successfully sent into pipeline
        Returns False if unsuccessful in sending to pipeline
        key-value is always written to local cache

        Blocking Semantics:
        '''
        previous_value = self.client_table.get(key)
        if prefilter and type(previous_value) == Result and not value.overwrites(previous_value):
            return False

        self.client_table[key] = value
        while True:
            try:
                self.inbound_queue.put((key, value), False)
            except (QueueFull):
                if block:
                    sleep(random() * 0.01)
                else:
                    return False
            else:
                return True
    
    def shortest_prefix(self, key):
        if type(self.client_table) != PrefixTree:
            raise Exception("TruthTableError: TruthTable of internal type {} does no support prefix queries".format(type(self.client_table)))
        self.__refresh__()
        return self.client_table.shortest_prefix(key)

    def longest_prefix(self, key):
        if type(self.client_table) != PrefixTree:
            raise Exception("TruthTableError: TruthTable of internal type {} does no support prefix queries".format(type(self.client_table)))
        self.__refresh__()
        return self.client_table.longest_prefix(key)

    
    def __contains__(self, key):
        self.__refresh__()
        return key in self.client_table

    def __str__(self):
        return str(self.client_table)
