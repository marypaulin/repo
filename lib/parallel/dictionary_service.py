from time import time
from random import shuffle

from lib.data_structures.result import Result
from lib.data_structures.interval import Interval
from lib.data_structures.prefix_tree import PrefixTree
from lib.parallel.channel import Channel

def DictionaryService(table=None, propagator=None, synchronization_cooldown=0, degree=1):
    if table == None:
        table = {}
    clients = []
    server_endpoints = []
    for i in range(degree):
        client_endpoint, server_endpoint = Channel(duplex=True, channel_type='pipe')
        client = __DictionaryClient__(table, client_endpoint, propagator=propagator, synchronization_cooldown=synchronization_cooldown)
        clients.append(client)
        server_endpoints.append(server_endpoint)

    server = __DictionaryServer__({}, server_endpoints)

    if degree <= 1:
        server.online = False
        for client in clients:
            client.online = False

    return (server, tuple(clients))


class __DictionaryServer__:
    def __init__(self, table, endpoints):
        self.table = {}
        self.updates = {}
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

            shuffle(self.endpoints)

            self.updates = {}
            # Transfer from inbound queue to broadcast buffers (if the entry is new)
            for endpoint in self.endpoints:
                while True:
                    element = endpoint.pop(block=False)
                    if element == None:
                        break
                    (key, value) = element
                    previous_value = self.table.get(key)
                    if type(previous_value) != Result or value.overwrites(previous_value):
                        # print("DictionaryTable Update table[{}] from {} to {}".format(str(key), str(previous_value), str(value)))
                        self.updates[key] = value
                    else:
                        # print("Rejected DictionaryTable Update table[{}] = from {} to {}".format(str(key), str(previous_value), str(value)))
                        pass
            
            self.table.update(self.updates)
            modified = len(self.updates) > 0

            for key, value in self.updates.items():
                for endpoint in self.endpoints:
                    endpoint.push((key, value), block=False)
        return modified

    def flush(self):
        self.serve()

class __DictionaryClient__:
    def __init__(self, table, endpoint, propagator=None, synchronization_cooldown=0):
        self.table = table
        self.endpoint = endpoint
        self.synchronization_cooldown = synchronization_cooldown
        self.propagator = propagator
        self.last_synchronization = 0
        self.online = True

    def synchronize(self):
        '''
        Receives broadcasted entries from pipeline into local cache
        '''

        if not self.online:
            return

        if time() > self.last_synchronization + self.synchronization_cooldown:
            self.last_synchronization = time()
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
                
                updates = self.propagator.propagate(key, result, self.table)
                for update_key, update_value in updates.items():
                    self.put(update_key, update_value)

    def has(self, key):
        '''
        Queries local cache for value
        Triggers a refresh upon miss
        Returns True upon hit (after refresh)
        Returns False upon miss (after refresh)
        '''
        self.synchronize()
        return key in self.table

    def get(self, key, block=False):
        '''
        Queries local cache for value
        Triggers a refresh upon miss
        Returns value upon hit (after refresh)
        Returns None upon miss (after refresh)
        '''
        self.synchronize()
        if block:
            while not key in self.table:
                self.synchronize()
        return self.table.get(key)

    def put(self, key, value, block=False, prefilter=True):
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
        return self.endpoint.push((key, value), block=block)
    
    def shortest_prefix(self, key):
        if type(self.table) != PrefixTree:
            raise Exception("DictionaryServiceError: DictionaryTable of internal type {} does no support prefix queries".format(type(self.table)))
        self.synchronize()
        return self.table.shortest_prefix(key)

    def longest_prefix(self, key):
        if type(self.table) != PrefixTree:
            raise Exception("DictionaryTableError: DictionaryTable of internal type {} does no support prefix queries".format(type(self.table)))
        self.synchronize()
        return self.table.longest_prefix(key)

    def __getitem__(self, key):
        return self.get(key, block=False)

    def __setitem__(self, key, value):
        return self.put(key, value, block=False)

    def __contains__(self, key):
        return self.has(key)

    def __str__(self):
        return str(self.table)

    def flush(self):
        self.synchronize()
