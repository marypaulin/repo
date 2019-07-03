# Defines the behaviour of a floating point interval
# Used to represent the possible values of the optimal objective to a particular problem

class Interval:
    def __init__(self, lowerbound=-float('Inf'), upperbound=float('Inf'), value=None):
        if value != None:
            lowerbound = value
            upperbound = value
        if lowerbound > upperbound:
            Exception("Invalid Interval Bounds [{}, {}]".format(lowerbound, upperbound))
        self.lowerbound = lowerbound
        self.upperbound = upperbound
        self.uncertainty = upperbound - lowerbound
    
    def union(self, interval):
        return Interval(min(self.lowerbound, interval.lowerbound), max(self.upperbound, interval.upperbound))

    def intersection(self, interval):
        return Interval(max(self.lowerbound, interval.lowerbound), min(self.upperbound, interval.upperbound))

    def subset(self, interval):
        return self.lowerbound >= interval.lowerbound and self.upperbound <= interval.upperbound and self.uncertainty < interval.uncertainty

    def superset(self, inteval):
        return self.lowerbound <= interval.lowerbound and self.upperbound >= interval.upperbound and self.uncertainty > interval.uncertainty

    def less_than(self, interval):
        return self.lowerbound <= interval.lowerbound and self.upperbound <= interval.upperbound

    def greater_than(self, interval):
        return self.lowerbound >= interval.lowerbound and self.upperbound >= interval.upperbound

    def value(self):
        if self.uncertainty == 0:
            return self.lowerbound
        else:
            return (self.lowerbound, self.upperbound)

    def __eq__(self, interval):
        if interval == None:
            return False
        return self.lowerbound == interval.lowerbound and self.upperbound == interval.upperbound

    def __str__(self):
        return str(self.value())
