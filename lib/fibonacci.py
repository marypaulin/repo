from queue import Empty as QueueEmpty, Full as QueueFull

from lib.parallel.cluster import Cluster
from lib.parallel.priority_queue import PriorityQueue
from lib.parallel.truth_table import TruthTable

class Fibonacci:
    def __init__(self, n):
        # Set all global variables, these are statically available to all workers
        self.n = n

    # Task method that gets run by all worker nodes (clients)
    def task(self, worker_id, services):
        (table, queue) = services
        try:
            while not self.terminate(table):
                message = queue.pop(block=False)
                if message == None:
                    continue
                (priority, n) = message
                # Check status
                result = table.get(n, block=False)
                if result == None:  # New problem
                    if n <= 2:  # Base Case
                        output = 1
                        table.put(n, output) # Memoize resolved problem
                        # print("Fib({}) = {}".format(n, output))
                    else: # Recursive Case
                        # print("Fib({}) = ?".format(n))
                        dependencies = (n-1, n-2)
                        table.put(n, (n-1, n-2))  # Memoize pending problem
                        if not table.has(n-1):
                            queue.push((priority-1, n-1)) # Enqueue subproblem
                        if not table.has(n-2):
                            queue.push((priority-2, n-2)) # Enqueue subproblem
                        queue.push((priority+0.5, n)) # re-enqueue problem
                else:  # Revisited problem
                    if type(result) == int: # Problem solved (No work needed)
                        pass
                    elif all(type(table.get(dependency, block=False)) == int for dependency in result): # Dependencies resolved, resolve self
                        output = sum(table.get(dependency, block=False) for dependency in result) # Compute output from subproblems' outputs
                        table.put(n, output) # Re-memoize as resolved problem
                        # print("Fib({}) = {}".format(n, output))
                    else: # Dependencies not resolved, re-enqueue problem
                        queue.push((priority+0.5, n))  # re-enqueue problem
        except KeyboardInterrupt:
            pass

    # Method run by worker nodes to decide when to terminate
    def terminate(self, table):
        return table.get(self.n, block=False) != None and type(table.get(self.n, block=False)) == int

    # Method for extracting the output 
    def output(self, table):
        return table.get(self.n)

    def solve(self, clients=1, servers=1):
        # Shared Data structures that get serviced by servers
        table = TruthTable(degree=clients)
        queue = PriorityQueue(queue=[(self.n, self.n)], degree=clients)
        services = (table, queue)

        # Initialize and run the multi-node client-server cluster
        cluster = Cluster(self.task, services, clients=clients, servers=servers)
        cluster.compute()

        solution = self.output(table)
        return solution
