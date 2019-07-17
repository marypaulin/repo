from lib.parallel.channel import Channel
from heapq import heappush, heappop

class PriorityQueue:
    def __init__(self, queue=None, degree=1):
        self.id = None
        self.role = 'unidentified'
        self.degree = degree

        self.priority_queue = queue if queue != None else []

        self.inbound_channels = dict( (i, Channel(multiproducer=False, multiconsumer=False)) for i in range(self.degree) )
        self.outbound_channel = Channel(multiproducer=False, multiconsumer=True)

    def identify(self, id, role):
        self.id = id
        self.role = role
        if self.role == 'client':
            for channel in self.inbound_channels.values():
                channel.identify('producer')
            self.outbound_channel.identify('consumer')
        elif self.role == 'server':
            for channel in self.inbound_channels.values():
                channel.identify('consumer')
            self.outbound_channel.identify('producer')
        else:
            raise Exception('PriorityQueueError: Invalid role {}'.format(self.role))

    # Service routine called by server
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
        if self.role != 'server':
            raise Exception("PriorityQueueException: Unauthorized access to server API from role {}".format(self.role))
        seen = set()
        # Transfer from inbound queue to priority queue
        for channel in self.inbound_channels.values():
            while True:
                element = channel.pop(block=False)
                if element == None:
                    break
                if not type(element) in {tuple, list}:
                    key = element
                else:
                    key = element[1:]
                if not key in seen:
                    seen.add(key)
                    heappush(self.priority_queue, element)

        # Transfer from priorty queue to outbound queue
        while self.priority_queue:
            element = heappop(self.priority_queue)
            self.outbound_channel.push(element, block=False)

    # API called by workers
    def push(self, element, block=False):
        '''
        Pushes object into pipeline
        Returns True if successful
        Returns False if unsuccessful
        '''
        if self.role != 'client':
            raise Exception("PriorityQueueException: Unauthorized access to client API from role {}".format(self.role))
        self.inbound_channels[self.id].push(element, block=block)


    # API called by workers
    def pop(self, block=True):
        '''
        Pops object from pipeline
        Returns (priority, element) if successful
        Returns (None, None) if unsuccessful
        '''
        if self.role != 'client':
            raise Exception("PriorityQueueException: Unauthorized access to client API from role {}".format(self.role))
        return self.outbound_channel.pop(block=block)

    def close(self, block=True):
        for channel in self.inbound_channels.values():
            channel.close()
        self.inbound_channels = {}
        self.outbound_channel.close()
        self.outbound_channel = None
