from lib.data_structures.prefix_tree import PrefixTree

class PathCluster:
    def __init__(self):
        self.cluster = PrefixTree()

    def add(self, task):
        self.cluster.add(task.path)

    def remove(self, task):
        self.cluster.remove(task.path)

    def proximity(self, task):
        # 0 = No match
        # 1 = Perfect match
        return self.cluster.shortest_prefix(task.path) / max(len(task.path), 1)

    def __len__(self):
        return len(self.cluster)
