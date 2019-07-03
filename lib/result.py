from lib.interval import Interval

# Defines generic behaviour of an optimization result that has an optimal value known within a certain interval
class Result:
    def __init__(self, optimizer=None, optimum=None, stats=None):
        self.optimum = optimum if optimum != None else Interval()
        self.optimizer = optimizer

    # This defines the precedence of problem state
    #  - A result is always able to overwrite None
    #  - A result is always able to overwrite another result if its interval is a subset of the other interval
    def overwrites(self, result):
        if result == None:
            return True
        if self.optimum.subset(result.optimum):
            return True
        return False

    def resolved(self):
        return self.optimum.uncertainty == 0
    
    def pending(self):
        return self.optimumt.uncertainty != 0

    def __str__(self):
        return "Result(optimizer={}, optimum={})".format(str(self.optimizer), str(self.optimum))
