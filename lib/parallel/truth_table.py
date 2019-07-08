from queue import Empty as QueueEmpty, Full as QueueFull
from multiprocessing import Queue
from collections import deque
from time import sleep
from random import random

import lib.vector as vect
from lib.result import Result
from lib.interval import Interval
from lib.similarity_index import SimilarityIndex
class TruthTable:
    def __init__(self, table=None, degree=1):
        self.id = None
        self.degree = degree
        self.local_table = table if table != None else {}
        self.inbound_queue = Queue()
        self.outbound_queues = {}
        for i in range(self.degree):
            self.outbound_queues[i] = (Queue(), deque([]))

        # self.dataset = dataset
        # self.lamb = lamb
        # self.similarity_tolerance = 1
        # self.unresolved = SimilarityIndex(self.dataset.height, self.similarity_tolerance, self.dataset.height)

    # def __count_leaves__(self, capture):
    #     result = self.local_table.get(capture)
    #     if result.optimizer == None: # Return an upperbound on leaf count instead
    #         return vect.count(capture)
    #     (split, prediction) = self.local_table.get(capture).optimizer
    #     if split == None: # This node is a leaf
    #         return 1
    #     else: # This node splits into subtrees
    #         (left_capture, right_capture) = self.dataset.split(capture, split)
    #         return self.__count_leaves__(left_capture) + self.__count_leaves__(right_capture)

    # def __similarity_propagation__(self, key, result):
    #     if result.optimizer != None: # The subproblem has a solved optimal subtree
    #         if key in self.unresolved: # Resolved problems don't need further bounding
    #             self.unresolved.remove(key)


    #         _total, zeros, ones, minority, _majority = self.dataset.count(key)
    #         # lowerbound_base = result.optimum.lowerbound + self.lamb * (self.similarity_tolerance-self.__count_leaves__(key))
    #         lowerbound_base = minority / self.dataset.sample_size + self.lamb * (self.similarity_tolerance-self.__count_leaves__(key))

    #         for neighbour_key in self.unresolved.neighbours(key):
                
    #             self_difference = key & vect.negate(neighbour_key)
    #             neighbour_difference = neighbour_key & vect.negate(key)
    #             if False and self.local_table.get(self_difference) != None:
    #                 drop = self.local_table.get(self_difference).optimum.upperbound
    #             else:
    #                 _total, zeros, ones, minority, _majority = self.dataset.count(self_difference)
    #                 drop = min(zeros, ones) / self.dataset.sample_size
    #             if False and self.local_table.get(neighbour_difference) != None:
    #                 rise = self.local_table.get(neighbour_difference).optimum.lowerbound
    #             else:
    #                 _total, zeros, ones, minority, _majority = self.dataset.count(neighbour_difference)
    #                 rise = minority / self.dataset.sample_size
    #             lowerbound = lowerbound_base - drop + rise

    #             interval = self.local_table.get(neighbour_key).optimum
    #             if lowerbound > interval.upperbound:
    #                 print("Similarity Bound {} appears to be overbounding".format(lowerbound))
    #             elif lowerbound > interval.lowerbound:
    #                 print("Similarity Narrowed {} to {}".format((interval.lowerbound, interval.upperbound), (lowerbound, interval.upperbound)))
    #                 new_neighbour = Result(optimizer=None, optimum=Interval(lowerbound, interval.upperbound))
    #                 self.local_table[neighbour_key] = new_neighbour
    #                 for i in range(self.degree):
    #                     (queue, buffer) = self.outbound_queues[i]
    #                     buffer.append((neighbour_key, new_neighbour))

    #                 self.__similarity_propagation__(neighbour_key, new_neighbour)
    #             else:
    #                 print("Similarity Bound too weak")
    #     else:
    #         # Store all unresolved problem keys in the similarity index
    #         if not key in self.unresolved:
    #             self.unresolved.add(key)

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
                previous_value = self.local_table.get(key)
                if type(previous_value) != Result or value.overwrites(previous_value):
                    # print("TruthTable Update table[{}] = from {} to {}".format(str(key), str(previous_value), str(value)))
                    self.local_table[key] = value
                    for i in range(self.degree):
                        (queue, buffer) = self.outbound_queues[i]
                        buffer.append((key, value))

                    # if type(value) == Result:
                    #     self.__similarity_propagation__(key, value)
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
    def get(self, key, block=True):
        '''
        Queries local cache for value
        Triggers a refresh upon miss
        Returns value upon hit (after refresh)
        Returns None upon miss (after refresh)
        '''
        
        self.__refresh__()
        if block:
            while not key in self.local_table:
                self.__refresh__()
                sleep(random() * 0.1)
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
    
    def __contains__(self, key):
        self.__refresh__()
        return key in self.local_table

    def __str__(self):
        return str(self.local_table)
