from lib.data_structures.interval import Interval

# Structure for holding the current set of knowledge about a particular optimization problem
# The purpose of this class is to standardize the information format for dynamically computing a global optimum
# The data structure is meant to store incremental results using bounding intervals that progressively narrow onto a single value
class Result:
    # Constructor Arguments:
    #  - optimizer = None or any data structure containing enough information to compute f(optimizer) = min(f) or max(f)
    #  - optimum = A interval which contains a lowerbound and upperbound on the optimum min(f) or max(f)
    def __init__(self, optimizer=None, optimum=None, running=False):
        self.optimum = optimum if optimum != None else Interval()
        if type(self.optimum) != Interval:
            raise Exception("ResultError: optimum class {} must be {}".format(type(self.optimum), Interval))
        self.optimizer = optimizer
        self.running = running

    # This defines the precedence of results over the same problem
    # If the current result is None then any result can overwrite it
    # If a result already exists, then the overwriting result must have a bounding interval that is a strict subset
    # This ensures that the precision of our optimum estimate increases monotonically
    # This also ensures we don't accidentally lose information by overwriting with a less precise estimate
    def overwrites(self, result):
        if result == None:
            return True
        if self.optimum.uncertainty == 0 or self.optimum.subset(result.optimum) or self.running == True and result.running == False:
            return True
        return False

    # Overrides the str(result) operator to get a nice readable string
    def __str__(self):
        return "Result(optimizer={}, optimum={})".format(str(self.optimizer), str(self.optimum))
