import numpy as np
from functools import reduce

import lib.vector as vect

class DataSet:
    def __init__(self, X, y, compress=True):
        self.compress = compress
        self.X = X # Deprecated
        self.y = vect.vectorize(y) # Deprecated

        z, rows, columns = self.__summarize__(X, y)
        self.rows = rows
        self.columns = columns
        self.z = z
        self.sample_size = X.shape[0]
        self.equivalent_set_count = len(rows)
        self.height = len(rows)
        self.width = len(columns)

        self.gini_index = self.__gini_reduction_index__(rows, columns, y, z)

    def split(self, capture, j):
        return (vect.negate(self.columns[j]) & capture, self.columns[j] & capture)

    def splits(self, capture):
        return ( (j, self.split(capture, j)) for j in self.gini_index)

    # Count various frequencies of the y-labels over a capture set
    def count(self, capture):
        (zeros, ones, minority, majority) = reduce(
            lambda x, y: tuple(sum(z) for z in zip(x, y)),
            (self.z[i] + (min(self.z[i]), max(self.z[i])) for i in range(self.height) if vect.test(capture, i)),
            (0, 0, 0, 0))
        return (ones + zeros, zeros, ones, minority, majority)

    def __summarize__(self, X, y):
        """
        Summarize the training dataset via equivalent points
        This simultaneously computes:
         - equivalent points bound data
         - majority labels for leaf construction,
        also reduce the row count which makes column vectors smaller

        Return values:
        z is a dictionary in which:
         - keys are unique combinations of features which represent an equivalent point set
         - values are tuples representing the frequencies of labels in that equivalent point set

        rows is the bit-vector rows filtered down to only unique ones
        columns is the bit-vector columns shortened to only the unique rows
        """
        (n, m) = X.shape
        # Vectorize features and labels for bitvector operations
        columns = tuple(vect.vectorize(X[:, j]) for j in range(m))
        rows = tuple(vect.vectorize(X[i, :]) for i in range(n))

        z = {}
        for i in range(n):
            row = rows[i]
            if z.get(row) == None:
                # z stores a tuple for each unique row (equivalent point set)
                # the tuple stores (in order) the frequency of labels 0 and 1
                z[row] = [0, 0]

            # Increment the corresponding label frequency in the equivalent point set
            z[row][y[i]] += 1

        reduced_rows = list(z.keys())

        compression_rate = len(rows) / len(reduced_rows)
        print("Row Compression Factor: {}".format(round(compression_rate, 3)))

        # This causes column vectors to have lots of leading zeros (except the first one)
        # it may help later for doing more compression
        list.sort(reduced_rows)
        list.reverse(reduced_rows)

        # Convert to tuples for higher performance
        z = tuple(tuple(z[row]) for row in reduced_rows)
        reduced_columns = tuple(vect.vectorize(vect.read(row, j) for row in reduced_rows) for j in range(m))
        reduced_rows = tuple(reduced_rows)

        if self.compress:
            return z, reduced_rows, reduced_columns
        else:
            return z, rows, columns

    def __reduction__(self, captures, x_j, y, z):
        """
        computes the weighted sum of Gini coefficients of bisection subsets by feature j
        reference: https://en.wikipedia.org/wiki/Gini_coefficient
        """
        if self.compress:
            (negative_total, _zeros, ones, _minority, _majority) = self.count(captures & vect.negate(x_j))
            p_1 = ones / negative_total if negative_total > 0 else 0

            (positive_total, _zeros, ones, _minority, _majority) = self.count(captures & x_j)
            p_2 = ones / positive_total if positive_total > 0 else 0
        else:
            negative = captures & vect.negate(x_j)
            negative_total = vect.count(negative)
            # negative correlation of feature j with labels
            p_1 = vect.count(negative & y) / \
                negative_total if negative_total > 0 else 0

            positive = captures & x_j
            positive_total = vect.count(positive)
            # positive correlation of feature j with labels
            p_2 = vect.count(positive & y) / \
                positive_total if positive_total > 0 else 0

        # Degree of inequality of labels in samples of negative feature
        gini_1 = 2 * p_1 * (1 - p_1)
        # Degree of inequality of labels in samples of positive feature
        gini_2 = 2 * p_2 * (1 - p_2)
        # Base inequality minus inequality
        reduction = negative_total * gini_1 + positive_total * gini_2
        return reduction

    def __gini_reduction_index__(self, rows, columns, y, z, captures=None):
        """
        calculate the gini reduction by each feature
        return the rank of by descending
        """
        (n, m) = (len(rows), len(columns))
        captures = vect.ones(n) if captures == None else captures
        # Sets the subset we compute probability over
        if self.compress:
            (total, _zeros, ones, _minority, _majority) = self.count(captures)
            p_0=ones / total if total > 0 else 0
        else:
            total = vect.count(captures)
            p_0 = vect.count(y & captures) / \
                total if total > 0 else 0

        gini_0 = 2 * p_0 * (1 - p_0)

        # The change in degree of inequality when splitting the capture set by each feature j
        # In general, positive values are similar to information gain while negative values are similar to information loss
        # All values are negated so that the list can then be sorted by descending information gain
        reductions = np.array([-(gini_0 - self.__reduction__(captures,
                                                             column, y, z) / total) for column in columns])
        order_index = reductions.argsort()

        print("Negative Gini Reductions: {}".format(tuple(reductions)))
        print("Negative Gini Reductions Index: {}".format(tuple(order_index)))

        return tuple(order_index)
