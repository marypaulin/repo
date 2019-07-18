from time import time
from random import random

from lib.result import Result
from lib.interval import Interval
from lib.prefix_tree import PrefixTree
from lib.parallel.channel import Channel

def TruthTable(table=None, propagator=None, refresh_cooldown=0, degree=1):
    if table == None:
        table = {}
    clients = []
    server_endpoints = []
    for i in range(degree):
        client_endpoint, server_endpoint = Channel(duplex=True)
        client = __TruthTableClient__(table, client_endpoint, propagator=propagator, refresh_cooldown=refresh_cooldown)
        clients.append(client)
        server_endpoints.append(server_endpoint)

    server = __TruthTableServer__({}, tuple(server_endpoints))

    return (server, tuple(clients))

class __TruthTableServer__:
    def __init__(self, table, endpoints):
        self.table = {}
        self.endpoints = endpoints
        self.online = True

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
        modified = False
        if self.online:
            updates = {}
            # Transfer from inbound queue to broadcast buffers (if the entry is new)
            for endpoint in self.endpoints:
                while True:
                    element = endpoint.pop(block=False)
                    if element == None:
                        break
                    (key, value) = element
                    previous_value = self.table.get(key)
                    if type(previous_value) != Result or value.overwrites(previous_value):
                        # print("TruthTable Update table[{}] = from {} to {}".format(str(key), str(previous_value), str(value)))
                        updates[key] = value
                    else:
                        # print("Rejected TruthTable Update table[{}] = from {} to {}".format(vect.__str__(key), previous_value, value))
                        pass
            
            self.table.update(updates)

            modified = len(updates) > 0

            for key, value in updates.items():
                for endpoint in self.endpoints:
                    endpoint.push((key, value), block=False)
        return modified

    def close(self, block=True):
        self.online = False
        for endpoint in self.endpoints:
            endpoint.close(block=block)
        self.endpoints = None

class __TruthTableClient__:
    def __init__(self, table, endpoint, propagator=None, refresh_cooldown=0):
        self.table = table
        self.endpoint = endpoint
        self.refresh_cooldown = refresh_cooldown
        self.propagator = propagator
        self.last_refresh = 0
        self.online = True

    def __refresh__(self):
        '''
        Receives broadcasted entries from pipeline into local cache
        '''

        if not self.online:
            return

        if time() > self.last_refresh + self.refresh_cooldown:
            self.last_refresh = time()
        else:
            return
        
        while True:

            element = self.endpoint.pop(block=False)
            if element == None:
                break
            (key, value) = element
            self.table[key] = value
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

    def has(self, key):
        '''
        Queries local cache for value
        Triggers a refresh upon miss
        Returns True upon hit (after refresh)
        Returns False upon miss (after refresh)
        '''
        self.__refresh__()
        return key in self.table

    def get(self, key, block=True):
        '''
        Queries local cache for value
        Triggers a refresh upon miss
        Returns value upon hit (after refresh)
        Returns None upon miss (after refresh)
        '''
        self.__refresh__()
        if block:
            while not key in self.table:
                self.__refresh__()
        return self.table.get(key)

    def put(self, key, value, block=True, prefilter=True):
        '''
        Stores key-value into local cache and sends entry into pipeline
        Returns True if successfully sent into pipeline
        Returns False if unsuccessful in sending to pipeline
        key-value is always written to local cache
        '''
        previous_value = self.table.get(key)
        if prefilter and type(previous_value) == Result and not value.overwrites(previous_value):
            return False

        self.table[key] = value
        self.endpoint.push((key, value), block=block)
    
    def shortest_prefix(self, key):
        if type(self.table) != PrefixTree:
            raise Exception("TruthTableError: TruthTable of internal type {} does no support prefix queries".format(type(self.table)))
        self.__refresh__()
        return self.table.shortest_prefix(key)

    def longest_prefix(self, key):
        if type(self.table) != PrefixTree:
            raise Exception("TruthTableError: TruthTable of internal type {} does no support prefix queries".format(type(self.table)))
        self.__refresh__()
        return self.table.longest_prefix(key)

    def __contains__(self, key):
        self.__refresh__()
        return key in self.table

    def __str__(self):
        return str(self.table)

    def close(self, block=True):
        self.online = False
        self.endpoint.close(block=block)
        self.endpoint = None
