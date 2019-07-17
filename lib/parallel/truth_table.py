from time import time, sleep
from random import random

from lib.result import Result
from lib.interval import Interval
from lib.prefix_tree import PrefixTree
from lib.parallel.channel import Channel

class TruthTable:
    def __init__(self, table=None, propagator=None, refresh_cooldown=0, degree=1):
        self.id = None
        self.role = 'unidentified'
        self.degree = degree

        self.client_table = table if table != None else {}
        self.server_table = {}

        self.inbound_channels = dict( (i, Channel(multiproducer=False, multiconsumer=False)) for i in range(self.degree) )
        self.outbound_channels = dict( (i, Channel(multiproducer=False, multiconsumer=False)) for i in range(self.degree) )

        self.last_refresh = 0
        self.refresh_cooldown = refresh_cooldown
        self.propagator = propagator
        self.online = True

    def identify(self, id, role, block=True):
        '''
        Modifies the local instance's id so that broadcasts can be pulled from the correct queue
        '''
        self.id = id
        self.role = role
        if self.role == 'client':
            for channel in self.inbound_channels.values():
                channel.identify('producer')
            for channel in self.outbound_channels.values():
                channel.identify('consumer')
        elif self.role == 'server':
            for channel in self.inbound_channels.values():
                channel.identify('consumer')
            for channel in self.outbound_channels.values():
                channel.identify('producer')
        else:
            raise Exception('TruthTableError: Invalid role {}'.format(self.role))

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
        if self.role != 'server':
            raise Exception('TruthTableError: Unauthorized access to server API from role {}'.format(self.role))

        updates = {}
        # Transfer from inbound queue to broadcast buffers (if the entry is new)
        for channel in self.inbound_channels.values():
            while True:
                element = channel.pop(block=False)
                if element == None:
                    break
                (key, value) = element
                previous_value = self.server_table.get(key)
                if type(previous_value) != Result or value.overwrites(previous_value):
                    # print("TruthTable Update table[{}] = from {} to {}".format(str(key), str(previous_value), str(value)))
                    updates[key] = value
                else:
                    # print("Rejected TruthTable Update table[{}] = from {} to {}".format(vect.__str__(key), previous_value, value))
                    pass

        self.server_table.update(updates)

        for key, value in updates.items():
            for channel in self.outbound_channels.values():
                channel.push((key, value), block=False)

    def __refresh__(self):
        '''
        Receives broadcasted entries from pipeline into local cache
        '''
        if self.role != 'client':
            raise Exception("TruthTableException: Unauthorized access to client API from role {}".format(self.role))
        if not self.online:
            return
        if time() > self.last_refresh + self.refresh_cooldown:
            self.last_refresh = time()
        else:
            return

        channel = self.outbound_channels[self.id]
        
        while True:

            element = channel.pop(block=False)
            if element == None:
                break
            (key, value) = element
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
    def put(self, key, value, prefilter=True):
        '''
        Stores key-value into local cache and sends entry into pipeline
        Returns True if successfully sent into pipeline
        Returns False if unsuccessful in sending to pipeline
        key-value is always written to local cache
        '''
        previous_value = self.client_table.get(key)
        if prefilter and type(previous_value) == Result and not value.overwrites(previous_value):
            return False

        self.client_table[key] = value
        # print("Worker {} Starting TruthTable#put".format(self.id))
        self.inbound_channels[self.id].push((key, value), block=False)
        # print("Worker {} Finishing TruthTable#put".format(self.id))

    
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

    def close(self, block=True):
        # if self.role == 'server':
        #     for channel in self.inbound_channels.values():
        #         while channel.pop(block=False) != None:
        #             pass
        #     for channel in self.outbound_channels.values():
        #         while channel.pop(block=False) != None:
        #             pass
        self.online = False
        for channel in self.inbound_channels.values():
            channel.close()
        self.inbound_channels = {}
        for channel in self.outbound_channels.values():
            channel.close()
        self.outbound_channels = {}

        
