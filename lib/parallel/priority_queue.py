from queue import Empty as QueueEmpty, Full as QueueFull
from multiprocessing import Queue
from heapq import heappush, heappop
from time import sleep

from lib.prefix_tree import PrefixTree
class PriorityQueue:
    def __init__(self, queue=None, sigma=1):
        self.id = None
        self.priority_queue = queue if queue != None else []
        self.inbound_queue = Queue()
        self.outbound_queue = Queue(sigma)

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
        # seen = set()
        # Transfer from inbound queue to priority queue
        while True:
            try:
                element = self.inbound_queue.get(False)
            except (QueueEmpty):
                break
            else:
                heappush(self.priority_queue, element)

        # Transfer from priorty queue to outbound queue
        while self.priority_queue:
            element = heappop(self.priority_queue)
            try:
                self.outbound_queue.put(element, False)
            except (QueueFull):
                # Re-insert element into priority queue
                heappush(self.priority_queue, element)
                break

    def identify(self, id):
        self.id = id

    # API called by workers
    def push(self, element, block=True):
        '''
        Pushes object into pipeline
        Returns True if successful
        Returns False if unsuccessful
        '''
        while True:
            try:
                self.inbound_queue.put(element, block)
            except (QueueFull):
                if block:
                    sleep(random() * 0.1)
                else:
                    raise QueueFull
            else:
                return True

    # API called by workers
    def pop(self, block=True):
        '''
        Pops object from pipeline
        Returns (priority, element) if successful
        Returns (None, None) if unsuccessful
        '''
        while True:
            try:
                element = self.outbound_queue.get(block)
            except (QueueEmpty):
                if block:
                    sleep(random() * 0.1)
                else:
                    raise QueueEmpty
            else:
                return element
