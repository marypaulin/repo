from lib.parallel.cluster import Cluster
from lib.parallel.priority_queue import PriorityQueue
from lib.parallel.truth_table import TruthTable
from lib.parallel.problem import Problem

class Fibonacci:
    def __init__(self, n):
        # Set all global variables, these are statically available to all workers
        self.n = n

    # Task method that gets run by all worker nodes (clients)
    def task(self, worker_id, queue, table):
        while not self.terminate(queue, table):
            (priority, n) = queue.pop()
            if priority != None:
                # Check status
                problem = table.get(n)
                if problem == None:  # New problem
                    if n <= 2:  # Base Case
                        output = 1
                        table.put(n, Problem(output=output)) # Memoize resolved problem
                    else: # Recursive Case
                        dependencies = (n-1, n-2)
                        table.put(n, Problem(dependencies=dependencies))  # Memoize pending problem
                        if not table.has(n-1):
                            queue.push((priority-1, n-1)) # Enqueue subproblem
                        if not table.has(n-2):
                            queue.push((priority-2, n-2)) # Enqueue subproblem
                        queue.push((priority+0.5, n)) # re-enqueue problem
                else:  # Revisited problem
                    if problem.output != None: # Problem solved (No work needed)
                        pass
                    elif not problem.pending(table): # Dependencies resolved, resolve self
                        output = sum(subproblem.output for subproblem in problem.subproblems(table)) # Compute output from subproblems' outputs
                        table.put(n, Problem(output=output)) # Re-memoize as resolved problem
                    else: # Dependencies not resolved, re-enqueue problem
                        queue.push((priority+0.5, n))  # re-enqueue problem
            else: # (None, None) dequeued, which indicates queue is empty
                pass # Cannot terminate yet so the worker idles

    # Method run by worker nodes to decide when to terminate
    def terminate(self, queue, table):
        return table.get(self.n) != None and table.get(self.n).output != None

    # Method for extracting the output 
    def output(self, queue, table):
        return table.get(self.n).output

    def solve(self, clients=1, servers=1):
        # Shared Data structures that get serviced by servers
        table = TruthTable({}, degree=clients)
        queue = PriorityQueue([(self.n, self.n)])

        # Initialize and run the multi-node client-server cluster
        cluster = Cluster(self.task, self.terminate, queue, table, clients=clients, servers=servers)
        cluster.compute()

        solution = self.output(queue, table)
        print("Solution is {}".format(solution))
        return solution
