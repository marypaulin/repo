from queue import Empty as QueueEmpty, Full as QueueFull

from lib.parallel.cluster import Cluster
from lib.parallel.priority_queue import PriorityQueue
from lib.parallel.truth_table import TruthTable
from lib.prefix_tree import PrefixTree
from lib.result import Result
from lib.interval import Interval
from lib.dataset import DataSet
from lib.vector import Vector
from lib.tree import Tree
from lib.logger import Logger

# Theorems Applied

# Notes:
# - In this recursive problem formulation, an optimal subtree can be either a leaf or a tree
# - The nodes outside of a subtree doesn't affect the risk contribution of the subtree by any sort of interaction
# - Therefore, a lot of bounds we make on leafs actually generalize to subtrees in this scenario
#   since the nodes outside of a subtree treat the subtree as a black box implementation of a leaf

# Theorem 3 Lower Bound on Leaf Support (Generalized to Subtree Support)
#  - This lowerbound generalizes to the children of all internal nodes, not just leaves
#    If a subtree has a support set of size of k < 2 * lambda,
#    if follows that no pair of leaves of this subtree has support adding up to k >= 2 * lambda
#  - During the downward recursive traversal avoid subproblem pairs if their total support < 2 * lambda

#  The core intuition is that each split increases regularization cost by 1 * lambda
#  To justify the cost, the two subtrees must have a joint miclassification 1 * lambda lower than the parent
#  A subtree is lowerbound by their EQ Minority
#  We know the minority before the split which acts as a baseline
#  
#  
#  - Further, we can create an upperbound on leaf support k <= (parent support) - lambda

# Corollary 6 Leaf Permutation Bound
#  - This generalizes to all subtree combination
#  - In this framework, leaf-sets don't exist but subtrees' subcapture set pairs do
#  - When recursing, we remove all duplicate subcapture sets to avoid unnecessary access to the memoization table
#  - If duplicate subcaptures are generated from different parent (i.e. they don't share the same recursive call)
#    then the memoization table will remove this duplication

# Proposition 8 Equivalent Points Bound
#  - The algorithm uses equivalent points to summarize capture vectors leading to smaller memory usage
#  - Equivalent Points place a lower bound on misclassification for any capture set
#    this lowerbound allows us to potentially prune subproblems without fully computing
#    the true minimal classification with regularization in mind

# TODO:
# Theorem 1 Hierarchical Objective Lowerbound
#  - During the downward recursive traversal,
#    lowerbound the subtree by the inaccuracy of an optimal leaf over the same capture set since it qualifies
#    as a child of such a leaf

# Theorem 2 One Step Lookahead Bound
#  This one is trickier in the new execution framework
#  This would be equivalent to the following:
#   - During the recursive traversal we consider the optimal leaf and all possible splits over the same capture set
#   - Each one is a candidate for the optimal subtree over the capture set C
#   - If we find any one of them has a lowerbound (plus lambda if not a leaf) greater than the current best candidate
#     then that candidate can be ignored.
#   - This means that this problem no longer depends on evaluation of those subproblems but doesn't guarantee that
#     those subproblems are not dependencies for other problems (which means we can't prune them)
#   The best we can do is one of the following
#    - only evaluate one split at a time for stricter comparison, i.e. don't enqueue all subproblems making this more
#      of a DFS like the previous implementation
#    - evaluate all lowerbounds against the optimal leaf (the leaf evaluation is constant) and don't enqueue subproblems
#      that perform worse. This coincidentally is the same as applying Theorem 4 during downward traversal.

# Theorem 4 Lower bound on Incremental Leaf Accuracy (Generalized to Incremental Subtree Accuracy)
#  - This lowerbound generalizes to the children of all internal nodes, not just leaves
#    if we consider an exchange from a leaf to a subtree containing 2 leaves, the risk trade-off is as follows:
#     - The misclassification term decreases by the incremental accuracy from exchanging the leaf to the subtree of 2 leaves (3.10)
#     - The regularization term increases by 1 * lambda
#    However, we can generalize the risk trade-off for exchanging a leaf with a subtree containing k leaves where k >= 2:
#     - The misclassification term decreases by the incremental accuracy from exchanging the leaf to the subtree of k leaves
#     - The regularization term increases by (k-1) * lambda
#     - Note k >= 2, since k = 1 is just an optimal leaf and there's no point exchanging the optimal leaf with itself
#  - For consistency, consider an equivalent upperbound on incemental inaccuracy:
#    (incremental inaccuracy) = -(incremental accuracy)
#    Constraint: incremental inaccuracy <= - (k-1) * lambda
#  - During the downward recursive traversal avoid subproblem pairs if their total lowerbound - (inaccuracy before split) > - lambda
#    lowerbound incremental inaccuracy = (lowerbound after split) - (inaccuracy before split)
#    lowerbound incremental inaccuracy <= incremental inaccuracy <= -lambda
#    (we don't know the number of leaves the optimal subtree will yield so just assume k = 2)
#  - During the upward recursive traversal, ignore subresults pairs if their total inaccuracy > -(k-1) * lambda
#    (we know the optimal subtree in the upward call, so a specific k can be applied)

# Theorem 5 Lowerbound on Leaf Accuracy (Generalized to Lowerbound on Subtree Accuracy)
#  - This lowerbound generalizes to the children of all internal nodes, not just leaves
#    Since optimal trees have a set of k-leaves satisfying the leaf accuracy lowerbound of lambda
#    we can assume an optimal subtree containing k-leaves have an accuracy lowerbound of k * lambda
#  - For consistency, consider an equivalent upperbound on inaccuracy:
    #  subtree accuracy = correct / n
    #  subtree inaccuracy = support - accuracy
#    (subtree inaccuracy) = support - (subtree accuracy)
    #  accuracy = support - inaccuracy
    #  support - inaccuracy < lambda
#    Constraint: subtree inaccuracy <= (1 - k * lambda)
#  - During the downward recursive traversal, avoid subproblems if their lowerbound > (1 - lambda)
#    lowerbound < inaccuracy <= (1 - lambda)
#  - During the upward recursive traversal, ignore subresults inaccuracy > (1 - k * lambda)

# Theorem 7 Similar Support Bound
#  - During any update to the truth table, if problem B exist within neighbourhood of A with within distance w = |CA xor CB|
#     - Their upperbounds may not differ more than w + #(EQ groups) in CA xor CB
#     - Their lowerbounds may not differ more than w
#    More Strictly
#     - Their upperbounds may not differ more than w
#     - Their lowerbounds may not differ more than w

#  - For any two optimal trees A, B with respective capture sets CA, CB and optimal risks RA, RB 
#    |RA - RB| <= hamming_distance(CA, CB) = |CA xor CB|
#  - During the upward recursive traversal, we are informed with an optimal subtree for CA
#    For all subproblems CB satisfying hamming_distance(CA, CB) <= D
#    Apply |RA - RB| <= D to possibly increase the lowerbound of CB by claiming RA - D <= RB
#    If the bound is ineffective the attempt a stricter bound RA - |CA-CB| <= RA - D <= RB
# Changelog
# - stricter base case
class OSDT:
    def __init__(self,
        X, y, lamb, priority_metric='curiosity',
        bounds=None,
        max_depth=float('Inf'), max_width=float('Inf'), max_time=float('Inf'),
        verbose=True, log=True):
        # Set all global variables, these are statically available to all workers
        self.dataset = DataSet(X, y)
        self.lamb = lamb
        self.priority_metric = priority_metric

        self.bounds = bounds if bounds != None else self.__default_bounds__()
        self.max_width = min(self.dataset.height, 2 ** self.dataset.width, max_width)
        self.max_depth = min(self.max_width, max_depth)
        self.max_time = max_time
        self.verbose = verbose
        self.log = log

    # Task method that gets run by all worker nodes (clients)
    def task(self, worker_id, services):
        (queue, table) = services
        self.logger = Logger(path='logs/worker_{}.log'.format(worker_id)) if self.log else None
        while not self.terminate(table):
            try:
                task = queue.pop(block=False)
                (priority, capture, path) = task
                result = table.get(capture, block=False)
                if result == None:  # New problem
                    lowerbound, upperbound, base_prediction = self.compute_bounds(capture)
                    # if 0.5 * normalized_support < self.lamb: # Previous base condition
                    if upperbound - lowerbound <= self.lamb:
                        # Compute the optimal subtree knowing the subtree must have 1 leaf
                        table.put(capture, Result(optimizer=(None, base_prediction), optimum=Interval(upperbound))) # Associate the optimizer with the optimum in a result
                    else: # Recursive Case
                        dependencies = self.recurse(priority, capture, base_prediction, upperbound, queue, table)
                        dependency_path = (*path, capture)
                        if len(dependencies) > 0:
                            for priority, dependency in dependencies:
                                queue.push((priority, dependency, dependency_path))  # Enqueue subproblem
                else: # Revisited problem
                    if result.optimizer != None: # Problem solved (No work needed)
                        pass
                    else: # Also recursive case
                        _lowerbound, upperbound, base_prediction = self.compute_bounds(capture)
                        dependencies = self.recurse(priority, capture, base_prediction, upperbound, queue, table)
                        if len(dependencies) > 0:
                            for priority, dependency in dependencies:
                                queue.push((priority, dependency, dependency_path))  # Enqueue subproblem
            except QueueEmpty: # Idle path
                pass

    def recurse(self, priority, capture, base_prediction, base_objective, queue, table):
        split_bounds, minimum_lowerbound, minimum_upperbound, minimum_split = self.compute_split_bounds(capture, base_objective, table)
        # Select only splits whose lowerbound is less than or equal to the split with the lowest upper bound
        possible_splits = [j for j in self.dataset.gini_index if split_bounds[j][0] <= minimum_upperbound]
        if base_objective <= minimum_upperbound:
            possible_splits.append(None)
        if minimum_lowerbound < minimum_upperbound:
            # Optimum not yet determined, but we can update our bounds
            optimizer = None # Optimizer still unknown
            optimum = Interval(minimum_lowerbound, minimum_upperbound) # uncertainty interval with possible optimal objective values
            result = Result(optimizer, optimum) # Associate the optimizer with the optimum in a result
            table.put(capture, result) # Memoize this result in the global truth table

            dependencies = set()
            for j in possible_splits:
                if j == None:
                    continue
                (left_capture, right_capture) = self.dataset.split(j, capture=capture)
                # Re-enqueue subproblems (This duplicate enqueueing could allow us to perform cache evictions without dependency graphs)
                if not left_capture in table:
                    dependencies.add((priority, left_capture))
                if not right_capture in table:
                    dependencies.add((priority, right_capture))
            # print('Traversal: Downward, Problem: {}, Dependencies: {}'.format(str(capture), tuple(str(d) for _p, d in dependencies)))
            dependencies.add((priority + 0.1, capture))
            return tuple(dependencies)
        elif minimum_lowerbound == minimum_upperbound:
            # Compute the optimal subtree knowing minimum split is the optimal split (which might be no-spit)
            split = minimum_split # Choose the first among possibly multiple equally optimal subtrees
            prediction = base_prediction if split == None else None # Set the prediction if the optimal split is no-split
            table.put(capture, Result(optimizer=(split, prediction), optimum=Interval(minimum_upperbound))) # Associate the optimizer with the optimum in a result
            # print('Traversal: Upward, Problem: {}, Optimal Result: {}'.format(str(capture), str(Result(optimizer=(split, prediction), optimum=Interval(minimum_upperbound)))))
            return tuple()
        else:
            raise Exception("OSDTError: Invalid hierarchical bounds {} for problem {}".format((minimum_lowerbound, minimum_upperbound), capture))

    def compute_bounds(self, capture):
        _total, zeros, ones, minority, _majority = self.dataset.label_distribution(capture)
        lowerbound = minority / self.dataset.sample_size + self.lamb * 1
        upperbound = min(zeros, ones) / self.dataset.sample_size + self.lamb * 1
        prediction = 0 if zeros >= ones else 1
        if lowerbound > upperbound:
            raise Exception("OSDTError: Invalid bounds {} for problem {}".format(
                (lowerbound, upperbound), capture))
        else:
            return lowerbound, upperbound, prediction

    def compute_split_bounds(self, capture, base_objective, table):
        # Track the lowest upperbound and lowest lowerbound of all splits
        intervals = [ None for _j in range(self.dataset.width) ]
        minimum_lowerbound = base_objective
        minimum_upperbound = base_objective
        minimum_split = None

        for j, left_capture, right_capture in self.dataset.splits(capture):
            if left_capture in table:
                left_result = table.get(left_capture)
                left_lowerbound = left_result.optimum.lowerbound
                left_upperbound = left_result.optimum.upperbound
            else:
                left_lowerbound, left_upperbound, prediction = self.compute_bounds(left_capture)
                if left_lowerbound == left_upperbound: # Prematurely discovered optimal subtree as a leaf
                    table.put(left_capture, Result(optimizer=(None, prediction), optimum=Interval(left_upperbound)))

            if right_capture in table:
                right_result = table.get(right_capture)
                right_lowerbound = right_result.optimum.lowerbound
                right_upperbound = right_result.optimum.upperbound
            else:
                right_lowerbound, right_upperbound, prediction = self.compute_bounds(right_capture)
                if right_lowerbound == right_upperbound: # Prematurely discovered optimal subtree as a leaf
                    table.put(right_capture, Result(optimizer=(None, prediction), optimum=Interval(right_upperbound)))

            split_lowerbound = left_lowerbound + right_lowerbound # Initially weak lowerbound, to be narrowed over time
            split_upperbound = left_upperbound + right_upperbound # Should be less than or equal to base upperbound

            if split_lowerbound > split_upperbound:
                raise Exception("OSDTError: Invalid split bound {}".format((split_lowerbound, split_upperbound)))
            
            intervals[j] = (split_lowerbound, split_upperbound)

            minimum_lowerbound = min(minimum_lowerbound, split_lowerbound)
            if split_upperbound < minimum_upperbound:
                minimum_upperbound = split_upperbound
                minimum_split = j
            
        if minimum_lowerbound > minimum_upperbound:
            raise Exception("OSDTError: Invalid hierarchical bound {}".format((minimum_lowerbound, minimum_upperbound)))

        return tuple(intervals), minimum_lowerbound, minimum_upperbound, minimum_split

    # Method run by worker nodes to decide when to terminate
    def terminate(self, table):
        # Termination condition
        return table.get(self.root, block=False) != None and table.get(self.root, block=False).optimizer != None

    # Method for extracting the output
    def output(self, queue, table):
        return Tree(self.root, table, self.dataset)

    def solve(self, clients=1, servers=1, visualize=False):
        # Root capture
        self.root = Vector.ones(self.dataset.height)

        # Shared Data structures that get serviced by servers
        table = TruthTable(degree=clients)
        queue = PriorityQueue([ ( 0, self.root, () ) ])
        services = (queue, table)

        # Initialize and run the multi-node client-server cluster
        cluster = Cluster(self.task, services, clients=clients, servers=servers)
        cluster.compute()

        solution = self.output(queue, table)
        if visualize:
            solution.visualize(self.dataset)
        return solution

    def __print__(self, message):
        # Internal print method to deal with verbosity and logging
        if self.verbose:
            print(message)
        if self.logger != None:
            self.logger.log([time.time(), message])

    def __default_bounds__(self):
        return {
            'incremental_support': True,
            'minimum_support': True,
            'minimum_accuracy': True,
            'equivalent_points': True,
            'look-ahead': True,
            'length': True,
        }
