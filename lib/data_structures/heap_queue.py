from heapq import heappush, heappop

class HeapQueue:
    def __init__(self, queue=None, limit=None):
        self.queue = queue if queue != None else []
        self.length = len(self.queue)
        self.limit = limit

    def push(self, element):
        if not self.full():
            self.length += 1
            heappush(self.queue, element)
            return True
        else:
            return False

    def pop(self):
        if not self.empty():
            self.length -= 1
            return heappop(self.queue)
        else:
            return None

    def empty(self):
        return self.length == 0

    def full(self):
        return self.limit != None and self.length >= self.limit

    def clear(self):
        while self.pop() != None:
            pass

    def __len__(self):
        return self.length

    def __str__(self):
        return str(self.queue)
