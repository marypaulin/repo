class CaptureCluster:
    def __init__(self):
        self.cluster = None
        self.size = 0

    def add(self, task):
        if self.cluster == None:
            self.cluster = list(task.capture)
        else:
            for i, e in enumerate(task.capture):
                self.cluster[i] += e
        self.size += 1

    def remove(self, task):
        if self.size > 0:
            self.cluster
            for i, e in enumerate(task.capture):
                self.cluster[i] += e
            self.size -= 1

    def proximity(self, task):
        # 0 = No match
        # 1 = Perfect match
        if self.cluster == None:
            distance = task.capture.count()
        else:
            distance = 0
            for i, e in enumerate(task.capture):
                if e != 0 and self.cluster[i] == 0:
                    distance += 1

        return 1 - (distance / max(len(task.capture), 1))

    def __len__(self):
        return self.size
