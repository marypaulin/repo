import heapq

class Queue:
    def __init__(self, init=[]):
        self.state = []
        self.count = len(self.state)
        self.size = len(self.state)

    def push(self, element):
        self.count += 1
        self.size += 1
        heapq.heappush(self.state, (element.priority(), element))

    def pop(self):
        self.size -= 1
        return heapq.heappop(self.state)

    def size(self):
        return self.size

    def count(self):
        return self.count
